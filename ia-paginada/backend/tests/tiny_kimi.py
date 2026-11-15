"""Gera um Kimi K3 em miniatura, no MESMO formato do checkpoint real.

Usa o código oficial do modelo (baixado do Hugging Face, não incluído aqui) e uma
configuração encolhida. Os especialistas são gravados comprimidos em MXFP4
(``weight_packed`` + ``weight_scale``), como no arquivo de 1,4 TB, e a
``quantization_config`` real é mantida.

Uso:  python tiny_kimi.py <pasta_com_codigo_do_K3> <pasta_saida>
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CODE_FILES = ("configuration_kimi_k3.py", "modeling_kimi_k3.py", "modeling_kimi_linear.py",
              "tokenization_kimi.py", "tiktoken.model", "tokenizer_config.json",
              "encoding_k3.py", "kimi_k3_processor.py", "kimi_k3_vision_processing.py",
              "media_utils.py", "preprocessor_config.json", "generation_config.json")


def tiny_config(real: dict) -> dict:
    cfg = copy.deepcopy(real)
    t = cfg["text_config"]
    t.update(hidden_size=64, intermediate_size=128, moe_intermediate_size=64,
             routed_expert_hidden_size=32, num_experts=16, num_experts_per_token=4,
             num_shared_experts=1, num_hidden_layers=6, num_attention_heads=4,
             num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16,
             qk_rope_head_dim=8, v_head_dim=16, attn_res_block_size=2, max_position_embeddings=4096)
    # vocabulário real (163.840) para o tokenizer oficial funcionar
    t["linear_attn_config"].update(head_dim=16, num_heads=4, kda_layers=[1, 2, 3, 5],
                                   full_attn_layers=[4, 6])
    v = cfg["vision_config"]
    v.update(vt_hidden_size=32, vt_intermediate_size=64, vt_num_attention_heads=2,
             vt_num_hidden_layers=2, qkv_hidden_size=32, mm_hidden_size=32, text_hidden_size=64,
             init_pos_emb_height=8, init_pos_emb_width=8, init_pos_emb_time=2)
    return cfg


def build(code_dir: str, out_dir: str, seed: int = 0):
    """Cria a pasta do modelo e devolve o modelo de referência (pesos densos, fp32)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors.torch import save_file

    from paged import fla_cpu
    from paged.loaders import _force_sdpa, _prepare_remote_code
    from paged.mxfp4 import dequantize, quantize

    _prepare_remote_code(torch.device("cpu"))
    fla_cpu.install()
    os.makedirs(out_dir, exist_ok=True)
    for f in CODE_FILES:
        if os.path.exists(os.path.join(code_dir, f)):
            shutil.copy(os.path.join(code_dir, f), out_dir)
    with open(os.path.join(code_dir, "config.json"), encoding="utf-8") as fh:
        cfg = tiny_config(json.load(fh))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)

    # referência: mesma arquitetura, sem quantização, fp32
    ref_cfg = copy.deepcopy(cfg)
    ref_cfg["text_config"].pop("quantization_config", None)
    ref_cfg.pop("quantization_config", None)
    ref_dir = os.path.join(out_dir, ".ref-config")
    os.makedirs(ref_dir, exist_ok=True)
    for f in ("configuration_kimi_k3.py", "modeling_kimi_k3.py", "modeling_kimi_linear.py"):
        shutil.copy(os.path.join(code_dir, f), ref_dir)
    with open(os.path.join(ref_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(ref_cfg, fh)
    config = AutoConfig.from_pretrained(ref_dir, trust_remote_code=True)
    torch.manual_seed(seed)
    from paged.loaders import build_remote_model
    ref = build_remote_model(config, ref_dir).float().eval()
    _force_sdpa(ref)

    # pesos aleatórios bem comportados
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16)))
            elif name.endswith(("dt_bias", "e_score_correction_bias")):
                p.normal_(0, 0.1)
            elif "norm" in name and p.dim() == 1:
                p.fill_(1.0)
            else:
                p.normal_(0, 1.0 / math.sqrt(max(1, p.shape[-1])))

    # especialistas -> MXFP4 (e a referência passa a usar os valores representáveis)
    sd = {}
    with torch.no_grad():
        for name, p in ref.state_dict().items():
            if ".experts." in name and name.endswith(".weight"):
                packed, scale = quantize(p)
                p.copy_(dequantize(packed, scale, torch.float32))
                base = name[: -len("weight")]
                sd[base + "weight_packed"] = packed.contiguous()
                sd[base + "weight_scale"] = scale.contiguous()
            else:
                sd[name] = p.detach().clone().contiguous()

    # dois arquivos + índice, como um checkpoint fatiado real
    keys = sorted(sd)
    half = len(keys) // 2
    shards = {"model-00001-of-00002.safetensors": keys[:half],
              "model-00002-of-00002.safetensors": keys[half:]}
    wm = {}
    for fname, ks in shards.items():
        save_file({k: sd[k] for k in ks}, os.path.join(out_dir, fname))
        wm.update({k: fname for k in ks})
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": {}, "weight_map": wm}, fh)
    return ref


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
    print("ok", sys.argv[2])
