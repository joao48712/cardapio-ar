"""Multiplicação direta sobre pesos MXFP4, sem descomprimir (estilo llama.cpp).

``y = x @ W.T`` onde W está em ``weight_packed`` (2 valores FP4 por byte) e
``weight_scale`` (expoente E8M0 por grupo de 32). O kernel lê só os bytes
comprimidos (4x menos memória que bf16) e não grava nenhuma matriz temporária.
Compilado com Numba (JIT, sem precisar de compilador C instalado); se o Numba
não existir, cai na descompressão em PyTorch.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    HAVE_NUMBA = False

_FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
_SCALE = np.minimum(2.0 ** (np.arange(256, dtype=np.float64) - 127),
                    np.finfo(np.float32).max).astype(np.float32)

if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _gemm_rows(x, packed, scale, fp4, sc, out):  # pragma: no cover - compilado
        T, n = x.shape
        m, half = packed.shape
        groups = scale.shape[1]
        gbytes = half // groups  # 16 bytes = 32 valores por grupo
        for i in prange(m):
            for t in range(T):
                acc = np.float32(0.0)
                for g in range(groups):
                    part = np.float32(0.0)
                    base = g * gbytes
                    for b in range(gbytes):
                        byte = packed[i, base + b]
                        j = (base + b) * 2
                        part += fp4[byte & 15] * x[t, j] + fp4[byte >> 4] * x[t, j + 1]
                    acc += part * sc[scale[i, g]]
                out[t, i] = acc


def mxfp4_linear(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor | None = None) -> torch.Tensor:
    """x: [..., n]; packed: [m, n/2] uint8; scale: [m, n/32] uint8 -> [..., m]."""
    shape = x.shape
    m = packed.shape[0]
    x2 = x.reshape(-1, shape[-1])
    if HAVE_NUMBA and x2.shape[0] <= 16 and x2.device.type == "cpu":
        xf = x2.float().contiguous().numpy()
        out = np.empty((xf.shape[0], m), dtype=np.float32)
        _gemm_rows(xf, packed.numpy(), scale.numpy(), _FP4, _SCALE, out)
        y = torch.from_numpy(out).to(x.dtype)
    else:  # muitos tokens (prompt longo) ou sem Numba: descomprime e usa matmul
        from .mxfp4 import dequantize
        w = dequantize(packed, scale, x.dtype if x.is_floating_point() else torch.float32)
        y = x2 @ w.T
    if bias is not None:
        y = y + bias
    return y.reshape(*shape[:-1], m)


class MXFP4Linear(torch.nn.Module):
    """Substitui nn.Linear: guarda os bytes comprimidos e multiplica direto neles."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_packed = torch.nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.uint8, device="meta"),
            requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            torch.empty(out_features, in_features // 32, dtype=torch.uint8, device="meta"),
            requires_grad=False)
        self.bias = None if not bias else torch.nn.Parameter(torch.empty(out_features, device="meta"))

    @property
    def weight(self):  # alguns códigos só olham dtype/device
        return self.weight_packed

    def forward(self, x):
        return mxfp4_linear(x, self.weight_packed, self.weight_scale, self.bias)


def warmup() -> None:
    """Compila o kernel (a primeira chamada leva alguns segundos)."""
    if HAVE_NUMBA:
        x = torch.zeros(1, 64)
        mxfp4_linear(x, torch.zeros(2, 32, dtype=torch.uint8), torch.full((2, 2), 127, dtype=torch.uint8))
