"""Descompressão MXFP4 (formato ``mxfp4-pack-quantized`` do compressed-tensors).

* ``weight_packed`` (uint8, [saídas, entradas/2]): dois valores FP4 E2M1 por byte,
  o nibble baixo é o elemento par; bit 0x8 = sinal, bits 0x7 = índice na tabela
  {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
* ``weight_scale`` (uint8, [saídas, entradas/32]): expoente E8M0, escala = 2^(s-127),
  um por grupo de 32 valores.

Peso = valor_fp4 × escala_do_grupo.
"""
from __future__ import annotations

from typing import Dict

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GROUP = 32
_LUT: Dict[torch.dtype, torch.Tensor] = {}
_SCALE: Dict[torch.dtype, torch.Tensor] = {}


def _fp4_values() -> torch.Tensor:
    mag = torch.tensor(E2M1, dtype=torch.float32)
    return torch.cat([mag, -mag])  # índices 0..15 (bit 3 = sinal)


def byte_lut(dtype: torch.dtype) -> torch.Tensor:
    """[256, 2]: os dois valores (baixo, alto) de cada byte."""
    t = _LUT.get(dtype)
    if t is None:
        v = _fp4_values()
        b = torch.arange(256)
        t = _LUT[dtype] = torch.stack([v[b & 0xF], v[b >> 4]], dim=1).to(dtype)
    return t


def scale_lut(dtype: torch.dtype) -> torch.Tensor:
    t = _SCALE.get(dtype)
    if t is None:
        t = _SCALE[dtype] = torch.exp2(torch.arange(256, dtype=torch.float32) - 127).to(dtype)
    return t


def dequantize(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Devolve o peso denso [saídas, entradas] em ``dtype``."""
    m, half = packed.shape
    n = half * 2
    vals = byte_lut(dtype)[packed.to(torch.int32)].view(m, n // GROUP, GROUP)
    s = scale_lut(dtype)[scale.to(torch.int32)].view(m, n // GROUP, 1)
    return (vals * s).view(m, n)


def quantize(w: torch.Tensor):
    """Compressão MXFP4 (para os testes): devolve (packed uint8, scale uint8)."""
    m, n = w.shape
    assert n % GROUP == 0
    wf = w.float().view(m, n // GROUP, GROUP)
    amax = wf.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    exp = torch.floor(torch.log2(amax)) - 2  # maior magnitude E2M1 = 6 = 1.5 * 2^2
    exp = exp.clamp(-127, 127)
    scaled = wf / torch.exp2(exp)
    mag = torch.tensor(E2M1)
    idx = (scaled.abs().unsqueeze(-1) - mag).abs().argmin(-1)
    code = idx | ((scaled < 0) & (idx > 0)).to(torch.int64) << 3
    code = code.view(m, n)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    scale = (exp.view(m, n // GROUP) + 127).to(torch.uint8)
    return packed, scale
