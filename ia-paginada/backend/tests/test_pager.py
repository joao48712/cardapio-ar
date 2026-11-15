"""Testes: o modelo paginado deve dar o mesmo resultado que o modelo normal.

Rode com:  python -m pytest backend/tests -q   (a partir de ia-paginada/)
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paged.loaders import load_diffusion, load_text  # noqa: E402
from paged.pager import CheckpointIndex, PagedModel  # noqa: E402


def _tiny_llama(tmp_path):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    model.save_pretrained(tmp_path)
    return model


def test_text_paged_matches_full(tmp_path):
    full = _tiny_llama(tmp_path)
    events = []
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    with init_empty_weights(include_buffers=False):
        empty = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(tmp_path)).eval()
    pager = PagedModel(empty, CheckpointIndex(str(tmp_path)), "cpu", torch.float32,
                       on_event=events.append)
    assert pager.stats["pages"] == 4
    ids = torch.randint(0, 256, (1, 12))
    with torch.inference_mode():
        a = full(ids).logits
        b = empty(ids).logits
    assert torch.allclose(a, b, atol=1e-5)
    # depois do forward nenhuma camada fica na memória
    assert all(p.device.type == "meta" for p in empty.model.layers.parameters())
    assert sum(e["type"] == "page" for e in events) == 4

    with torch.inference_mode():
        g1 = full.generate(ids, max_new_tokens=8, do_sample=False)
        g2 = empty.generate(ids, max_new_tokens=8, do_sample=False)
    assert torch.equal(g1, g2)


def test_ram_cache_and_no_prefetch(tmp_path):
    full = _tiny_llama(tmp_path)
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    with init_empty_weights(include_buffers=False):
        empty = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(tmp_path)).eval()
    pager = PagedModel(empty, CheckpointIndex(str(tmp_path)), "cpu", torch.float32,
                       prefetch=False, ram_cache_gb=1)
    ids = torch.randint(0, 256, (1, 6))
    with torch.inference_mode():
        for _ in range(2):
            assert torch.allclose(full(ids).logits, empty(ids).logits, atol=1e-5)
    assert len(pager.ram_cache) == 4


def test_load_text_helper(tmp_path):
    _tiny_llama(tmp_path)
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {chr(97 + i): i for i in range(26)} | {"<unk>": 26}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<unk>").save_pretrained(tmp_path)
    m = load_text(str(tmp_path), torch.device("cpu"), torch.float32, None)
    out = []
    m.generate("abc", out.append, max_new_tokens=5, temperature=0)
    assert m.pagers[0].stats["loads"] >= 4


def test_diffusion_paged_matches_full(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    torch.manual_seed(0)
    unet = diffusers.UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=1, sample_size=16, in_channels=4,
        out_channels=4, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32,
        attention_head_dim=8, norm_num_groups=8)
    vae = diffusers.AutoencoderKL(block_out_channels=(32, 64), in_channels=3, out_channels=3,
                                  down_block_types=("DownEncoderBlock2D",) * 2,
                                  up_block_types=("UpDecoderBlock2D",) * 2, latent_channels=4,
                                  norm_num_groups=8)
    te = CLIPTextModel(CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size=32,
                                      intermediate_size=37, num_attention_heads=4,
                                      num_hidden_layers=3, vocab_size=1000))
    tok = CLIPTokenizer.from_pretrained(os.environ.get("TINY_CLIP_TOKENIZER",
                                                       "hf-internal-testing/tiny-random-clip"))
    pipe = diffusers.StableDiffusionPipeline(
        unet=unet, vae=vae, text_encoder=te, tokenizer=tok,
        scheduler=diffusers.DDIMScheduler(), safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    pipe.save_pretrained(tmp_path)
    kw = dict(prompt="um gato", num_inference_steps=2, height=32, width=32, output_type="np")
    ref = pipe(generator=torch.Generator().manual_seed(1), **kw).images

    m = load_diffusion(str(tmp_path), torch.device("cpu"), torch.float32, None,
                       resident_below_mb=0)
    assert all(p.stats["pages"] >= 2 for p in m.pagers)
    got = m.pipe(generator=torch.Generator().manual_seed(1), **kw).images
    assert abs(ref - got).max() < 1e-4
    out = m.generate(str(tmp_path / "out"), "um gato", lambda s, t: None, steps=2,
                     width=32, height=32, seed=3)
    assert os.path.exists(out)
