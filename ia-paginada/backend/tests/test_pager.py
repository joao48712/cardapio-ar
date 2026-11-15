"""Testes: o modelo paginado deve dar o mesmo resultado que o modelo normal.

Rode com:  npm run test:python   (ou python -m pytest backend/tests -q)
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paged.loaders import load_diffusion, load_text  # noqa: E402
from paged.pager import CheckpointIndex, PackedStore, PagedModel  # noqa: E402


def _tiny_llama(tmp_path, dtype=torch.float32, tie=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=tie)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    model.to(dtype).save_pretrained(tmp_path)
    return model.float()


def _empty_llama(path):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    with init_empty_weights(include_buffers=False):
        return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path)).eval()


def test_text_paged_matches_full(tmp_path):
    full = _tiny_llama(tmp_path)
    events = []
    empty = _empty_llama(tmp_path)
    pager = PagedModel(empty, CheckpointIndex(str(tmp_path)), "cpu", torch.float32,
                       on_event=events.append)
    assert pager.stats["pages"] == 4
    assert pager.stats["zero_copy"] and pager.stats["store"] == "direto"
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


def test_every_piece_paged_and_order_learned(tmp_path):
    """Divide tudo: embedding e cabeça de saída também viram páginas."""
    full = _tiny_llama(tmp_path, tie=False)
    empty = _empty_llama(tmp_path)
    pager = PagedModel(empty, CheckpointIndex(str(tmp_path)), "cpu", torch.float32,
                       min_page_mb=0, prefetch=3)
    names = pager.page_prefixes
    assert "model.embed_tokens" in names and "lm_head" in names
    ids = torch.randint(0, 256, (1, 7))
    with torch.inference_mode():
        for _ in range(2):
            assert torch.allclose(full(ids).logits, empty(ids).logits, atol=1e-5)
    # aprendeu a ordem real: embedding -> camada 0 -> ... -> lm_head
    emb, l0 = names.index("model.embed_tokens"), names.index("model.layers.0")
    assert pager._next[emb] == l0
    assert all(p.device.type == "meta" for p in empty.lm_head.parameters())


def test_packed_store_fp16_to_fp32(tmp_path):
    """Checkpoint em fp16, cálculo em fp32: encaderna uma vez e lê sem cópia."""
    _tiny_llama(tmp_path, dtype=torch.float16)
    ref = _tiny_llama(tmp_path / "ref", dtype=torch.float16)  # mesmos pesos arredondados
    ref = type(ref).from_pretrained(tmp_path, dtype=torch.float32).eval()
    empty = _empty_llama(tmp_path)
    progress = []
    pager = PagedModel(empty, CheckpointIndex(str(tmp_path)), "cpu", torch.float32,
                       on_event=lambda e: progress.append(e) if e["type"] == "pack" else None)
    assert pager.stats["store"] == "encadernado" and pager.stats["zero_copy"]
    assert progress and progress[-1]["progress"] == pytest.approx(1.0)
    bin_p, _ = PackedStore.paths(CheckpointIndex(str(tmp_path)), torch.float32)
    assert os.path.exists(bin_p)
    ids = torch.randint(0, 256, (1, 9))
    with torch.inference_mode():
        assert torch.allclose(ref(ids).logits, empty(ids).logits, atol=1e-5)
    # segunda vez reaproveita o arquivo encadernado
    mtime = os.path.getmtime(bin_p)
    PagedModel(_empty_llama(tmp_path), CheckpointIndex(str(tmp_path)), "cpu", torch.float32)
    assert os.path.getmtime(bin_p) == mtime

    # modo sem encadernação: converte a cada leitura, mesmo resultado
    e2 = _empty_llama(tmp_path)
    p2 = PagedModel(e2, CheckpointIndex(str(tmp_path)), "cpu", torch.float32, pack="off", prefetch=0)
    assert not p2.stats["zero_copy"]
    with torch.inference_mode():
        assert torch.allclose(ref(ids).logits, e2(ids).logits, atol=1e-5)


def test_load_text_helper(tmp_path):
    _tiny_llama(tmp_path)
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {chr(97 + i): i for i in range(26)} | {"<unk>": 26}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<unk>").save_pretrained(tmp_path)
    m = load_text(str(tmp_path), torch.device("cpu"), torch.float32, None)
    assert m.model.device.type == "cpu"  # não "meta"
    out = []
    m.generate("abc", out.append, max_new_tokens=5, temperature=0)
    assert m.pagers[0].stats["loads"] >= 4


@pytest.mark.parametrize("min_page_mb", [1.0, 0.0])
def test_diffusion_paged_matches_full(tmp_path, min_page_mb):
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
                       min_page_mb=min_page_mb)
    assert all(p.stats["pages"] >= 2 for p in m.pagers)
    got = m.pipe(generator=torch.Generator().manual_seed(1), **kw).images
    assert abs(ref - got).max() < 1e-4
    out = m.generate(str(tmp_path / "out"), "um gato", lambda s, t: None, steps=2,
                     width=32, height=32, seed=3)
    assert os.path.exists(out)


def test_tied_weights_t5_stay_consistent(tmp_path):
    """T5: shared e encoder.embed_tokens são o mesmo peso; não pode paginar só um lado."""
    from accelerate import init_empty_weights
    from transformers import T5Config, T5EncoderModel
    cfg = T5Config(vocab_size=128, d_model=32, d_kv=8, d_ff=64, num_layers=3, num_heads=4)
    torch.manual_seed(0)
    full = T5EncoderModel(cfg).eval()
    full.save_pretrained(tmp_path)
    with init_empty_weights(include_buffers=False):
        empty = T5EncoderModel(cfg).eval()
    PagedModel(empty, CheckpointIndex(str(tmp_path)), "cpu", torch.float32, min_page_mb=0)
    ids = torch.randint(0, 128, (1, 10))
    with torch.inference_mode():
        a = full(ids).last_hidden_state
        b = empty(ids).last_hidden_state
    assert torch.allclose(a, b, atol=1e-5)


def test_legacy_vae_attention_names(tmp_path):
    """VAEs antigos salvam query/key/value/proj_attn; o modelo atual usa to_q/.../to_out.0."""
    diffusers = pytest.importorskip("diffusers")
    from accelerate import init_empty_weights
    from safetensors.torch import load_file, save_file
    torch.manual_seed(0)
    kw = dict(block_out_channels=(32, 64), in_channels=3, out_channels=3, latent_channels=4,
              down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
              norm_num_groups=8)
    full = diffusers.AutoencoderKL(**kw).eval()
    full.save_pretrained(tmp_path)
    f = tmp_path / "diffusion_pytorch_model.safetensors"
    ren = {".to_q.": ".query.", ".to_k.": ".key.", ".to_v.": ".value.", ".to_out.0.": ".proj_attn."}
    sd = {}
    for k, v in load_file(f).items():
        for a, b in ren.items():
            k = k.replace(a, b)
        sd[k] = v
    save_file(sd, f)
    with init_empty_weights(include_buffers=False):
        empty = diffusers.AutoencoderKL(**kw).eval()
    PagedModel(empty, CheckpointIndex(str(tmp_path)), "cpu", torch.float32, min_page_mb=0)
    z = torch.randn(1, 4, 8, 8)
    with torch.inference_mode():
        assert torch.allclose(full.decode(z).sample, empty.decode(z).sample, atol=1e-5)
