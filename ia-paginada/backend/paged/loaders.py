"""Monta modelos de texto, imagem e vídeo sobre o paginador."""
from __future__ import annotations

import importlib
import inspect
import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional

import torch
from accelerate import init_empty_weights

from .pager import CheckpointIndex, PagedModel, default_dtype

VIDEO_HINTS = ("video", "i2v", "t2v", "wan", "ltx", "mochi", "cogvideo", "hunyuanvideo", "allegro", "animate")


def pick_device(pref: str = "auto") -> torch.device:
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(pref: str, device: torch.device) -> torch.dtype:
    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if pref in table:
        return table[pref]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    # CPU: bfloat16 só se o processador calcula bf16 em hardware (AMX/AVX512-BF16);
    # metade dos bytes para mover e sem conversão quando o checkpoint já é bf16.
    return torch.bfloat16 if cpu_has_bf16() else torch.float32


def cpu_has_bf16() -> bool:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            flags = fh.read()
        return "amx_bf16" in flags or "avx512_bf16" in flags
    except OSError:
        return False


def tune_cpu(device: torch.device) -> None:
    """Usa todos os núcleos no cálculo quando a "impressora" é a CPU."""
    if device.type == "cpu":
        n = os.cpu_count() or 1
        torch.set_num_threads(n)


def pin_device(model: torch.nn.Module, device: torch.device) -> None:
    """Pesos fora da memória ficam em 'meta'; o modelo deve dizer o dispositivo real."""
    base = type(model)
    model.__class__ = type(base.__name__, (base,), {"device": property(lambda self: device)})


def detect_kind(path: str) -> Optional[str]:
    """'text', 'image' ou 'video' (None se não reconhecido)."""
    mi = os.path.join(path, "model_index.json")
    if os.path.exists(mi):
        with open(mi, encoding="utf-8") as fh:
            name = json.load(fh).get("_class_name", "").lower()
        return "video" if any(h in name for h in VIDEO_HINTS) else "image"
    cfg = os.path.join(path, "config.json")
    if os.path.exists(cfg):
        return "text"
    return None


def _folder_has_weights(folder: str) -> bool:
    return os.path.isdir(folder) and any(f.endswith(".safetensors") for f in os.listdir(folder))


# --------------------------------------------------------------------------- #
class LoadedModel:
    kind = "?"

    def __init__(self, path: str, pagers: List[PagedModel], device, dtype):
        self.path = path
        self.pagers = pagers
        self.device = device
        self.dtype = dtype
        self.lock = threading.Lock()

    def stats(self) -> List[dict]:
        return [dict(p.stats) for p in self.pagers]

    def reset(self) -> None:
        for p in self.pagers:
            p.reset()

    def close(self) -> None:
        for p in self.pagers:
            p.close()


class TextModel(LoadedModel):
    kind = "text"

    def __init__(self, path, model, tokenizer, pager, device, dtype):
        super().__init__(path, [pager], device, dtype)
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, on_text: Callable[[str], None], max_new_tokens=256,
                 temperature=0.7, top_p=0.95, system: str = "", stopping_criteria=None, **_):
        from transformers import TextIteratorStreamer

        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = tok(text, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(**inputs, max_new_tokens=int(max_new_tokens), streamer=streamer,
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if stopping_criteria is not None:
            kw["stopping_criteria"] = stopping_criteria
        if float(temperature) > 0:
            kw.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
        else:
            kw.update(do_sample=False)
        err: List[BaseException] = []

        def run():
            try:
                with torch.inference_mode():
                    self.model.generate(**kw)
            except BaseException as e:  # noqa: BLE001
                err.append(e)
                streamer.end()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        full = ""
        for piece in streamer:
            full += piece
            on_text(full)
        th.join()
        if err:
            raise err[0]
        return full


class DiffusionModel(LoadedModel):
    def __init__(self, path, pipe, pagers, device, dtype, kind):
        super().__init__(path, pagers, device, dtype)
        self.pipe = pipe
        self.kind = kind
        self.call_params = set(inspect.signature(pipe.__call__).parameters)

    def accepts_image(self) -> bool:
        return "image" in self.call_params

    def generate(self, out_path: str, prompt: str, on_step: Callable[[int, int], None],
                 negative_prompt: str = "", steps: int = 25, guidance: float = 5.0,
                 width: int = 0, height: int = 0, frames: int = 0, fps: int = 16,
                 seed: int = -1, image=None, **_):
        p = self.call_params
        kw: Dict[str, Any] = {"prompt": prompt}
        if negative_prompt and "negative_prompt" in p:
            kw["negative_prompt"] = negative_prompt
        if "num_inference_steps" in p:
            kw["num_inference_steps"] = int(steps)
        if "guidance_scale" in p:
            kw["guidance_scale"] = float(guidance)
        if width and "width" in p:
            kw["width"] = int(width)
        if height and "height" in p:
            kw["height"] = int(height)
        if frames and "num_frames" in p:
            kw["num_frames"] = int(frames)
        if image is not None and "image" in p:
            kw["image"] = image
        gen_device = "cpu" if self.device.type == "mps" else self.device
        if int(seed) >= 0:
            kw["generator"] = torch.Generator(gen_device).manual_seed(int(seed))
        total = int(steps)
        if "callback_on_step_end" in p:
            def cb(_pipe, step, _t, cb_kwargs):
                on_step(step + 1, total)
                return cb_kwargs
            kw["callback_on_step_end"] = cb
        elif "callback" in p:  # pipelines antigos (ex.: ModelScope text-to-video)
            kw["callback"] = lambda step, _t, _lat: on_step(step + 1, total)
            if "callback_steps" in p:
                kw["callback_steps"] = 1
        with torch.inference_mode():
            result = self.pipe(**kw)
        if getattr(result, "frames", None) is not None:
            from diffusers.utils import export_to_video
            out_path = os.path.splitext(out_path)[0] + ".mp4"
            export_to_video(result.frames[0], out_path, fps=int(fps))
        else:
            out_path = os.path.splitext(out_path)[0] + ".png"
            result.images[0].save(out_path)
        return out_path


# --------------------------------------------------------------------------- #
def _build_empty(cls, folder: str, dtype: torch.dtype):
    from diffusers import ModelMixin
    with init_empty_weights(include_buffers=False), default_dtype(dtype):
        if issubclass(cls, ModelMixin):
            model = cls.from_config(cls.load_config(folder))
        else:  # transformers PreTrainedModel
            from transformers import AutoConfig
            cfg_cls = getattr(cls, "config_class", None)
            cfg = cfg_cls.from_pretrained(folder) if cfg_cls else AutoConfig.from_pretrained(folder)
            model = cls(cfg)
    return model.eval()


def load_text(path: str, device, dtype, on_event, prefetch=2, pack="auto",
              min_page_mb=1.0) -> TextModel:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(path)
    with init_empty_weights(include_buffers=False), default_dtype(dtype):
        model = AutoModelForCausalLM.from_config(config)
    model.eval()
    pager = PagedModel(model, CheckpointIndex(path), device, dtype, prefetch=prefetch,
                       pack=pack, min_page_mb=min_page_mb, on_event=on_event, label="texto")
    pin_device(model, torch.device(device))
    tok = AutoTokenizer.from_pretrained(path)
    return TextModel(path, model, tok, pager, device, dtype)


def load_diffusion(path: str, device, dtype, on_event, prefetch=2, pack="auto",
                   min_page_mb=1.0, resident_below_mb: float = 0.0) -> DiffusionModel:
    from diffusers import DiffusionPipeline

    with open(os.path.join(path, "model_index.json"), encoding="utf-8") as fh:
        index = json.load(fh)
    components: Dict[str, Any] = {}
    pagers: List[PagedModel] = []
    for name, spec in index.items():
        if name.startswith("_") or not isinstance(spec, list) or spec[0] is None:
            continue
        folder = os.path.join(path, name)
        if not _folder_has_weights(folder):
            continue  # tokenizer, scheduler... o pipeline carrega sozinho
        lib, cls_name = spec
        cls = getattr(importlib.import_module(lib), cls_name)
        if not issubclass(cls, torch.nn.Module):
            continue
        model = _build_empty(cls, folder, dtype)
        ckpt = CheckpointIndex(folder)
        small = ckpt.total_bytes < resident_below_mb * 2**20
        pagers.append(PagedModel(model, ckpt, device, dtype, prefetch=prefetch, pack=pack,
                                 paged=not small, min_page_mb=min_page_mb,
                                 on_event=on_event, label=name))
        pin_device(model, torch.device(device))
        components[name] = model

    import diffusers.utils.deprecation_utils as du
    dtype_kw = "dtype" if hasattr(du, "_TORCH_DTYPE_DEPRECATION_MESSAGE") else "torch_dtype"
    pipe = DiffusionPipeline.from_pretrained(path, **{dtype_kw: dtype}, **components)
    # O pipeline acha que o dispositivo é "meta" (pesos vazios): corrigimos.
    base = type(pipe)
    dev = torch.device(device)
    patched = type(base.__name__, (base,), {
        "_execution_device": property(lambda self: dev),
        "device": property(lambda self: dev),
    })
    pipe.__class__ = patched
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return DiffusionModel(path, pipe, pagers, dev, dtype, detect_kind(path) or "image")


def load_any(path: str, device_pref="auto", dtype_pref="auto", on_event=None,
             prefetch=2, pack="auto", min_page_mb=1.0, resident_below_mb=0.0) -> LoadedModel:
    kind = detect_kind(path)
    device = pick_device(device_pref)
    dtype = pick_dtype(dtype_pref, device)
    tune_cpu(device)
    prefetch = int(prefetch)
    if kind == "text":
        return load_text(path, device, dtype, on_event, prefetch, pack, min_page_mb)
    if kind in ("image", "video"):
        return load_diffusion(path, device, dtype, on_event, prefetch, pack, min_page_mb,
                              resident_below_mb)
    raise ValueError(f"Formato de modelo não reconhecido em {path}")
