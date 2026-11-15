"""Testes do suporte a MoE em MXFP4 (Kimi K3 e afins).

O teste com o Kimi K3 de verdade (arquitetura oficial, em miniatura) roda quando
``IAPAG_KIMI_CODE`` aponta para uma pasta com o código baixado do Hugging Face
(configuration_kimi_k3.py, modeling_kimi_k3.py, modeling_kimi_linear.py,
tokenization_kimi.py, tiktoken.model, tokenizer_config.json, config.json).
"""
import glob
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paged import fla_cpu  # noqa: E402
from paged.mxfp4 import dequantize, quantize  # noqa: E402
from paged.pager import CheckpointIndex, PagedModel  # noqa: E402


# --------------------------------------------------------------- fla na CPU
def _fla_naive():
    try:
        import importlib.metadata as md
        dist = md.distribution("fla-core")
    except Exception:  # noqa: BLE001
        return None
    path = os.path.join(str(dist.locate_file("")), "fla", "ops", "kda", "naive.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("kda_naive", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # só torch + einops
    return mod


def test_kda_cpu_matches_fla_reference():
    nv = _fla_naive()
    if nv is None:
        pytest.skip("fla-core não instalada (referência)")
    torch.manual_seed(0)
    B, T, H, K, V = 2, 32, 3, 16, 16
    q, k, v = torch.randn(B, T, H, K), torch.randn(B, T, H, K), torch.randn(B, T, H, V)
    graw, braw, A, dt = torch.randn(B, T, H, K), torch.randn(B, T, H), torch.randn(H), torch.randn(H * K)
    # fórmula de fla/ops/kda/gate.py::naive_kda_lowerbound_gate
    g = -5.0 * torch.sigmoid(A.view(H, 1).exp() * (graw + dt.view(H, K)))
    ref, Sref = nv.naive_recurrent_kda(F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, g,
                                       braw.sigmoid(), output_final_state=True)
    kw = dict(A_log=A, dt_bias=dt, output_final_state=True, use_qk_l2norm_in_kernel=True,
              use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, lower_bound=-5.0,
              transpose_state_layout=True)
    o, S = fla_cpu.chunk_kda(q, k, v, graw, braw, **kw)
    assert torch.allclose(o, ref, atol=1e-5)
    assert torch.allclose(S.transpose(-1, -2), Sref, atol=1e-5)
    # prompt + decodificação token a token = sequência inteira
    o1, S1 = fla_cpu.chunk_kda(q[:, :20], k[:, :20], v[:, :20], graw[:, :20], braw[:, :20], **kw)
    outs = [o1]
    for t in range(20, T):
        ot, S1 = fla_cpu.fused_recurrent_kda(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
                                             graw[:, t:t + 1], braw[:, t:t + 1],
                                             initial_state=S1, **kw)
        outs.append(ot)
    assert torch.allclose(torch.cat(outs, 1), o, atol=1e-5)


def test_short_conv_matches_conv1d_and_cache():
    c = fla_cpu.ShortConvolution(8, 4)
    x = torch.randn(1, 10, 8)
    ref = F.silu(F.conv1d(x.transpose(1, 2), c.weight, None, padding=3, groups=8)[..., :10]).transpose(1, 2)
    y, _ = c(x, output_final_state=True)
    y1, st = c(x[:, :7], output_final_state=True)
    y2, _ = c(x[:, 7:], cache=st, output_final_state=True)
    assert torch.allclose(y, ref, atol=1e-6)
    assert torch.allclose(torch.cat([y1, y2], 1), y, atol=1e-6)


# ------------------------------------------------------------------- MXFP4
def test_mxfp4_matches_compressed_tensors():
    helpers = pytest.importorskip("compressed_tensors.compressors.nvfp4.helpers")
    mx = pytest.importorskip("compressed_tensors.compressors.mx_utils")
    w = torch.randn(64, 128) * 0.02
    p, s = quantize(w)
    theirs = helpers.unpack_fp4_from_uint8(p, 64, 128, torch.float32) * \
        mx.decompress_mx_scale(s).float().repeat_interleave(32, 1)
    assert torch.equal(dequantize(p, s, torch.float32), theirs)


def test_kernel_matches_dequant():
    from paged.mxfp4_kernel import HAVE_NUMBA, mxfp4_linear
    if not HAVE_NUMBA:
        pytest.skip("numba não instalado")
    w = torch.randn(96, 128) * 0.05
    p, s = quantize(w)
    for T in (1, 3, 40):  # 40 tokens usa o caminho de descompressão
        x = torch.randn(T, 128)
        assert torch.allclose(mxfp4_linear(x, p, s), x @ dequantize(p, s, torch.float32).T, atol=1e-4)


# ------------------------------------------------- MoE genérico paginado
class Expert(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w1, self.w3, self.w2 = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, d, e, k):
        super().__init__()
        self.weight, self.k = nn.Parameter(torch.empty(e, d)), k

    def forward(self, x):
        scores = F.linear(x, self.weight).softmax(-1)
        w, idx = scores.topk(self.k, -1)
        return idx, w


class Moe(nn.Module):
    def __init__(self, d, f, e, k):
        super().__init__()
        self.gate = Gate(d, e, k)
        self.experts = nn.ModuleList(Expert(d, f) for _ in range(e))

    def forward(self, x):
        idx, w = self.gate(x)
        out = torch.zeros_like(x)
        for e in torch.unique(idx).tolist():  # só os escolhidos rodam
            rows, slot = (idx == e).nonzero(as_tuple=True)
            out[rows] += w[rows, slot, None] * self.experts[e](x[rows])
        return out


class Block(nn.Module):
    def __init__(self, d, f, e, k):
        super().__init__()
        self.norm, self.moe = nn.LayerNorm(d), Moe(d, f, e, k)

    def forward(self, h):
        return h + self.moe(self.norm(h))


class TinyMoeLM(nn.Module):
    def __init__(self, vocab=100, d=64, f=64, e=16, k=4, layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(Block(d, f, e, k) for _ in range(layers))
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        h = self.embed(ids).view(-1, self.embed.embedding_dim)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def _save_packed(model, folder):
    from safetensors.torch import save_file
    sd = {}
    with torch.no_grad():
        for name, p in model.state_dict().items():
            if ".experts." in name:
                pk, sc = quantize(p)
                p.copy_(dequantize(pk, sc, torch.float32))
                sd[name[:-len("weight")] + "weight_packed"] = pk
                sd[name[:-len("weight")] + "weight_scale"] = sc
            else:
                sd[name] = p.clone()
    save_file(sd, os.path.join(folder, "model.safetensors"))


@pytest.mark.parametrize("direct", [True, False])
def test_generic_moe_pages_each_expert(tmp_path, direct):
    from accelerate import init_empty_weights
    torch.manual_seed(0)
    ref = TinyMoeLM().eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.2)
    _save_packed(ref, tmp_path)
    with init_empty_weights(include_buffers=False):
        empty = TinyMoeLM().eval()
    pager = PagedModel(empty, CheckpointIndex(str(tmp_path)), "cpu", torch.float32,
                       mxfp4_direct=direct, expert_cache_gb=1 if not direct else 0)
    from paged.mxfp4_kernel import HAVE_NUMBA
    assert pager.stats["experts"] == 3 * 16
    assert pager.mxfp4_modules == (3 * 16 * 3 if direct and HAVE_NUMBA else 0)
    ids = torch.randint(0, 100, (1, 6))
    with torch.inference_mode():
        a, b = ref(ids), empty(ids)
    assert torch.allclose(a, b, atol=1e-4)
    st = pager.stats
    assert 0 < st["expert_loads"] <= 3 * 16  # só os escolhidos pelo roteador
    assert st["expert_prefetched"] == st["expert_loads"]  # todos pedidos pelo roteador
    assert all(p.device.type == "meta" for n, p in empty.named_parameters() if ".experts." in n)


# --------------------------------------------------- Kimi K3 (miniatura)
KIMI_CODE = os.environ.get("IAPAG_KIMI_CODE", "")


@pytest.mark.skipif(not (KIMI_CODE and glob.glob(os.path.join(KIMI_CODE, "modeling_kimi_k3.py"))),
                    reason="defina IAPAG_KIMI_CODE com o código oficial do Kimi K3")
def test_kimi_k3_tiny_paged_matches_reference(tmp_path):
    sys.path.insert(0, os.path.dirname(__file__))
    import tiny_kimi
    from paged.loaders import RemoteCodeRequired, load_text, manual_generate
    ref = tiny_kimi.build(KIMI_CODE, str(tmp_path / "k3"))
    torch.manual_seed(1)
    ids = torch.randint(4, 512, (1, 11))
    gref = manual_generate(ref, ids, 8)
    with pytest.raises(RemoteCodeRequired):
        load_text(str(tmp_path / "k3"), torch.device("cpu"), torch.float32, None)
    m = load_text(str(tmp_path / "k3"), torch.device("cpu"), torch.float32, None, trust_remote_code=True)
    p = m.pagers[0]
    assert p.stats["experts"] == 5 * 16  # 5 camadas MoE x 16 especialistas
    with torch.inference_mode():
        assert torch.allclose(ref(input_ids=ids).logits, m.model(input_ids=ids).logits, atol=1e-4)
    assert torch.equal(manual_generate(m.model, ids, 8), gref)
    assert p.stats["expert_prefetched"] == p.stats["expert_loads"] > 0
