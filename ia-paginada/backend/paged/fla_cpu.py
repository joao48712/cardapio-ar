"""Versão em PyTorch puro (CPU) do pedaço da biblioteca ``fla`` usado pelo Kimi.

A ``fla`` (flash-linear-attention) só tem kernels Triton para GPU. O código do
Kimi K3 importa dela a atenção KDA (Kimi Delta Attention), a convolução curta e a
norma com gate. Aqui está a mesma matemática em torch, para rodar na CPU.
As fórmulas seguem as referências "naive" da própria fla (ops/kda/naive.py e
ops/kda/gate.py); os testes comparam com elas.

``install()`` registra estes módulos como ``fla.*`` em ``sys.modules`` antes de
o código do modelo ser importado.
"""
from __future__ import annotations

import functools
import sys
import types
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# Utilidades (fla.utils / fla.ops.utils.index)
# --------------------------------------------------------------------------- #
def tensor_cache(fn):
    """Na fla é um cache do último resultado; aqui basta repassar."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(-1, dtype=torch.int32)


def prepare_cu_seqlens_from_mask(mask: torch.Tensor, dtype=torch.int32) -> torch.Tensor:
    return F.pad(prepare_lens_from_mask(mask).cumsum(0, dtype=dtype), (1, 0))


# --------------------------------------------------------------------------- #
# KDA
# --------------------------------------------------------------------------- #
def _kda_gate(g, A_log, dt_bias, lower_bound):
    H = g.shape[-2]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.float().view(H, -1)
    if lower_bound is not None:
        if A_log is not None:
            g = A_log.float().view(H, 1).exp() * g
        return lower_bound * torch.sigmoid(g)
    return -A_log.float().view(H, 1).exp() * F.softplus(g)


def _kda_segment(q, k, v, g, beta, scale, state):
    """Recorrência KDA token a token. q,k: [B,T,H,K]; v: [B,T,HV,V]; g: [B,T,HV,K]."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = q.new_zeros(B, HV, K, V) if state is None else state.float().clone()
    o = torch.empty(B, T, HV, V, dtype=torch.float32, device=q.device)
    for t in range(T):
        S = S * g[:, t].exp().unsqueeze(-1)
        kt = k[:, t]
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", beta[:, t].unsqueeze(-1) * kt, v[:, t] - pred)
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
    return o, S


def _kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, initial_state=None,
         output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
         use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, lower_bound=None,
         state_v_first=False, cu_seqlens=None, **kwargs):
    if "transpose_state_layout" in kwargs:
        state_v_first = bool(kwargs.pop("transpose_state_layout"))
    dtype = v.dtype
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
    g = _kda_gate(g, A_log, dt_bias, lower_bound) if use_gate_in_kernel else g.float()
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)

    def to_kv(s):  # estado guardado como [.., V, K] quando state_v_first
        return None if s is None else (s.transpose(-1, -2) if state_v_first else s)

    def from_kv(s):
        return s.transpose(-1, -2).contiguous() if state_v_first else s

    if cu_seqlens is None:
        o, S = _kda_segment(q, k, v, g, beta, scale, to_kv(initial_state))
        return o.to(dtype), (from_kv(S) if output_final_state else None)

    outs, states = [], []
    bounds = cu_seqlens.tolist()
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        h0 = to_kv(initial_state[i:i + 1]) if initial_state is not None else None
        o_i, S_i = _kda_segment(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], scale, h0)
        outs.append(o_i)
        states.append(S_i)
    o = torch.cat(outs, dim=1)
    return o.to(dtype), (from_kv(torch.cat(states, 0)) if output_final_state else None)


def chunk_kda(q, k, v, g, beta, *args, safe_gate=False, chunk_size=64, disable_recompute=False,
              return_intermediate_states=False, **kwargs):
    # Na CPU, a forma recorrente dá o mesmo resultado que a forma em blocos.
    return _kda(q, k, v, g, beta, *args, **kwargs)


def fused_recurrent_kda(q, k, v, g, beta, *args, **kwargs):
    return _kda(q, k, v, g, beta, *args, **kwargs)


# --------------------------------------------------------------------------- #
# Módulos (fla.modules)
# --------------------------------------------------------------------------- #
class ShortConvolution(nn.Conv1d):
    """Convolução causal por canal com estado [N, D, W] (últimas W entradas)."""

    def __init__(self, hidden_size: int, kernel_size: int, bias: bool = False,
                 activation: Optional[str] = "silu", backend: Optional[str] = None,
                 device=None, dtype=None, **kwargs):
        super().__init__(hidden_size, hidden_size, kernel_size, padding=kernel_size - 1,
                         groups=hidden_size, bias=bias, device=device, dtype=dtype)
        self.hidden_size = hidden_size
        self.activation = activation

    def _conv(self, x: torch.Tensor, cache: Optional[torch.Tensor]):
        B, T, D = x.shape
        W = self.kernel_size[0]
        xt = x.transpose(1, 2)  # [B, D, T]
        prev = cache.to(x.dtype) if cache is not None else x.new_zeros(B, D, W)
        seq = torch.cat([prev, xt], dim=-1)  # [B, D, W+T]
        w = self.weight.to(x.dtype)  # [D, 1, W]
        y = F.conv1d(seq[..., 1:], w, self.bias, groups=D)  # janelas que terminam em cada t
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        return y.transpose(1, 2), seq[..., -W:].contiguous()

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                cu_seqlens=None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        if cu_seqlens is None:
            y, state = self._conv(x, cache)
        else:
            ys, sts = [], []
            bounds = cu_seqlens.tolist()
            for i in range(len(bounds) - 1):
                c = cache[i:i + 1] if cache is not None else None
                y_i, s_i = self._conv(x[:, bounds[i]:bounds[i + 1]], c)
                ys.append(y_i)
                sts.append(s_i)
            y, state = torch.cat(ys, dim=1), torch.cat(sts, 0)
        if residual is not None:
            y = y + residual
        return y, (state if output_final_state else None)

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]


class FusedRMSNormGated(nn.Module):
    """y = rmsnorm(x) * weight * act(g)."""

    def __init__(self, hidden_size: int, elementwise_affine: bool = True, eps: float = 1e-5,
                 activation: str = "swish", device=None, dtype=None, **kwargs):
        super().__init__()
        self.hidden_size, self.eps, self.activation = hidden_size, eps, activation
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        dtype = x.dtype
        xf = x.float()
        if residual is not None:
            xf = xf + residual.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        gf = g.float()
        y = y * (torch.sigmoid(gf) if self.activation == "sigmoid" else F.silu(gf))
        return y.to(dtype)


# --------------------------------------------------------------------------- #
def install() -> None:
    """Registra os módulos acima como ``fla``, ``fla.modules``, ``fla.ops.kda``..."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        m.__path__ = []  # vira "pacote" para imports aninhados
        sys.modules[name] = m
        return m

    mod("fla", __version__="cpu-shim")
    mod("fla.modules", FusedRMSNormGated=FusedRMSNormGated, ShortConvolution=ShortConvolution)
    mod("fla.ops")
    mod("fla.ops.kda", chunk_kda=chunk_kda, fused_recurrent_kda=fused_recurrent_kda)
    mod("fla.ops.utils")
    mod("fla.ops.utils.index", prepare_cu_seqlens_from_mask=prepare_cu_seqlens_from_mask,
        prepare_lens_from_mask=prepare_lens_from_mask)
    mod("fla.utils", tensor_cache=tensor_cache)
