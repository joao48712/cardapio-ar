"""Mede a velocidade da troca de páginas na CPU.

    python bench.py [--layers 16] [--hidden 1024]

Compara o modelo inteiro na RAM com os modos paginados: conversão a cada
leitura (como a versão antiga), encadernado sem cópia e bfloat16.
"""
import argparse
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from accelerate import init_empty_weights  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM  # noqa: E402

from paged.loaders import cpu_has_bf16, tune_cpu  # noqa: E402
from paged.pager import CheckpointIndex, PagedModel  # noqa: E402


def timed(model, ids, reps):
    with torch.inference_mode():
        model(ids)  # aquece (e encaderna/aprende a ordem)
        t = time.perf_counter()
        for _ in range(reps):
            model(ids)
    return (time.perf_counter() - t) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    tune_cpu(torch.device("cpu"))
    d = tempfile.mkdtemp(prefix="iapag-bench-")
    cfg = LlamaConfig(vocab_size=32000, hidden_size=a.hidden, intermediate_size=a.hidden * 4,
                      num_hidden_layers=a.layers, num_attention_heads=16, num_key_value_heads=4,
                      tie_word_embeddings=False)
    LlamaForCausalLM(cfg).half().save_pretrained(d)  # checkpoint em fp16, como no Hub
    size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 2**30
    ids = torch.randint(0, 32000, (1, a.tokens))
    print(f"modelo: {a.layers} camadas, {size:.2f} GB em fp16, {torch.get_num_threads()} threads\n")

    rows = []
    full = LlamaForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    rows.append(("modelo inteiro na RAM (fp32)", timed(full, ids, a.reps), None))
    del full

    modes = [("paginado, converte a cada leitura (antigo)", torch.float32, "off", 1),
             ("paginado, encadernado sem cópia (fp32)", torch.float32, "auto", 2)]
    if cpu_has_bf16():
        modes.append(("paginado, sem cópia em bfloat16 (AMX)", torch.bfloat16, "auto", 2))
    for label, dtype, pack, pf in modes:
        with init_empty_weights(include_buffers=False):
            m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(d)).eval()
        p = PagedModel(m, CheckpointIndex(d), "cpu", dtype, pack=pack, prefetch=pf, min_page_mb=1)
        ms = timed(m, ids, a.reps)
        rows.append((label, ms, p.stats["avg_ms"]))
        p.close()

    base = rows[1][1]
    print(f"{'modo':48} {'ms/passada':>11} {'troca média':>12} {'vs antigo':>10}")
    for label, ms, sw in rows:
        swt = f"{sw:.3f} ms" if sw is not None else "-"
        print(f"{label:48} {ms:11.1f} {swt:>12} {base / ms:9.2f}x")


if __name__ == "__main__":
    main()
