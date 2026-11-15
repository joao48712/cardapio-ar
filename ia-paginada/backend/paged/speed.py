"""Variável "velocidade": quantas vezes as páginas pesadas são realmente percorridas.

Na difusão (imagem/vídeo), o modelo principal (UNet/transformer) roda uma vez
por passo, e é ele que consome quase todo o tempo. A troca de página em si já
custa < 1 ms; o que pesa é o cálculo dentro de cada página.

Com velocidade ``v``, o modelo principal só é recalculado em ~1 de cada ``v``
passos. Nos demais, a saída do último cálculo é reaproveitada e NENHUMA página
é carregada. Os primeiros passos (onde a composição nasce) e o último são sempre
calculados. ``v = 1`` é idêntico ao normal; não há limite superior, mas quanto
maior, mais a qualidade cai.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional

import torch


def plan_steps(total: int, speed: float, warmup: int = 2) -> List[bool]:
    """Lista de passos: True = calcula de verdade, False = reaproveita."""
    total = max(0, int(total))
    if speed <= 1 or total <= warmup + 1:
        return [True] * total
    plan = [False] * total
    acc = 0.0
    for s in range(total):
        if s < warmup or s == total - 1:
            plan[s] = True
            continue
        acc += 1.0 / speed
        if acc >= 1.0 - 1e-9:
            plan[s] = True
            acc -= 1.0
    return plan


def _clone(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone(o) for o in obj)
    if isinstance(obj, list):
        return [_clone(o) for o in obj]
    if isinstance(obj, dict):  # inclui BaseOutput do diffusers (OrderedDict)
        out = type(obj)(**{k: _clone(v) for k, v in obj.items()}) if is_dataclass(obj) else \
            type(obj)((k, _clone(v)) for k, v in obj.items())
        return out
    if is_dataclass(obj):
        return type(obj)(**{f.name: _clone(getattr(obj, f.name)) for f in fields(obj)})
    return obj


def _signature(args, kwargs) -> tuple:
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, torch.Tensor):
            return tuple(v.shape)
    return ()


class StepCache:
    """Envolve o ``forward`` do modelo principal para pular passos conforme o plano."""

    def __init__(self, module: torch.nn.Module, name: str):
        self.module = module
        self.name = name
        self._orig = module.forward
        module.forward = self  # o __call__ do nn.Module chama isto
        self.plan: Optional[List[bool]] = None
        self.step = 0
        self.call = 0
        self.cache: Dict[int, tuple] = {}
        self.computed = 0
        self.reused = 0

    def start(self, plan: Optional[List[bool]]) -> None:
        self.plan, self.step, self.call = plan, 0, 0
        self.cache.clear()
        self.computed = self.reused = 0

    def next_step(self) -> None:
        self.step += 1
        self.call = 0

    def __call__(self, *args, **kwargs):
        i = self.call
        self.call += 1
        sig = _signature(args, kwargs)
        plan = self.plan
        compute = (plan is None or self.step >= len(plan) or plan[self.step]
                   or i not in self.cache or self.cache[i][0] != sig)
        if not compute:
            self.reused += 1
            return _clone(self.cache[i][1])
        out = self._orig(*args, **kwargs)
        self.computed += 1
        if plan is not None:
            self.cache[i] = (sig, _clone(out))
        return out

    def remove(self) -> None:
        self.module.forward = self._orig
        self.cache.clear()


DENOISERS = ("transformer", "transformer_2", "unet", "prior")


def install(pipe) -> List[StepCache]:
    caches = []
    for name in DENOISERS:
        mod = getattr(pipe, name, None)
        if isinstance(mod, torch.nn.Module):
            caches.append(StepCache(mod, name))
    return caches


# --------------------------------------------------------------------------- #
# Velocidade sem teto: depois dos passos, o excedente vai para resolução e quadros
# --------------------------------------------------------------------------- #
MIN_SIDE = 64      # menor lado calculado internamente (px)
MIN_FRAMES = 5     # menos quadros que isso o modelo de vídeo não aguenta


def _round_side(x: float, mult: int = 16) -> int:
    return max(MIN_SIDE, int(x) // mult * mult)


def _round_frames(n: float, default: int) -> int:
    n = max(MIN_FRAMES, int(n))
    for mod in (8, 4):  # LTX usa 8k+1, Wan usa 4k+1
        if default and default % mod == 1:
            return max(MIN_FRAMES, (n - 1) // mod * mod + 1)
    return n


def split_speed(speed: float, steps: int, width: int, height: int, frames: int,
                default_frames: int = 0, can_skip_steps: bool = True) -> dict:
    """Distribui a velocidade pedida entre passos, resolução e quadros.

    Retorna o plano interno e o ganho estimado (passos x área x quadros).
    """
    speed = max(1.0, float(speed))
    plan = plan_steps(steps, speed) if can_skip_steps else [True] * int(steps)
    computed = max(1, sum(plan))
    step_gain = steps / computed
    rest = speed / step_gain
    w, h, f = width, height, frames
    area_gain = frame_gain = 1.0
    if rest > 1.05 and width and height:
        s = rest ** 0.5
        w, h = _round_side(width / s), _round_side(height / s)
        area_gain = (width * height) / (w * h)
        rest /= area_gain
    if rest > 1.05 and frames:
        f = _round_frames(frames / rest, default_frames)
        frame_gain = frames / f
    return {"plan": plan, "steps_computed": computed, "width": w, "height": h, "frames": f,
            "estimated_gain": round(step_gain * area_gain * frame_gain, 2),
            "at_floor": speed > step_gain * area_gain * frame_gain * 1.05}


def upscale_image(img, width: int, height: int):
    from PIL import Image
    return img.resize((width, height), Image.LANCZOS) if img.size != (width, height) else img


def upscale_video(frames, width: int, height: int, n_out: int):
    """Amplia cada quadro e interpola para chegar a n_out quadros."""
    import numpy as np
    from PIL import Image
    arr = []
    for fr in frames:
        a = np.asarray(fr)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
        im = Image.fromarray(a)
        if width and height and im.size != (width, height):
            im = im.resize((width, height), Image.LANCZOS)
        arr.append(np.asarray(im, dtype=np.float32) / 255.0)
    n_in = len(arr)
    if n_out and n_out != n_in and n_in > 1:
        out = []
        for k in range(n_out):
            x = k * (n_in - 1) / (n_out - 1)
            i = min(int(x), n_in - 2)
            t = x - i
            out.append(arr[i] * (1 - t) + arr[i + 1] * t)
        arr = out
    return arr
