"""Monta modelos de texto, imagem e vídeo sobre o paginador."""
from __future__ import annotations

import contextlib
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


def manual_generate(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.0,
                    top_p: float = 1.0, eos_ids=(), stop=None, on_token=None) -> torch.Tensor:
    """Geração token a token com cache, sem depender do ``generate()`` do transformers
    (código de modelo de outra versão pode não ter, ou não ser compatível)."""
    ids = input_ids
    cur = input_ids
    past = None
    pos = 0
    eos = set(int(e) for e in eos_ids if e is not None)
    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            q = cur.shape[1]
            out = model(input_ids=cur, past_key_values=past, use_cache=True,
                        **({"cache_position": torch.arange(pos, pos + q)} if _accepts(model, "cache_position") else {}))
            pos += q
            past = out.past_key_values
            logits = out.logits[:, -1].float()
            if temperature and temperature > 0:
                probs = torch.softmax(logits / float(temperature), -1)
                if top_p < 1.0:
                    sp, si = probs.sort(-1, descending=True)
                    keep = sp.cumsum(-1) - sp < float(top_p)
                    probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            cur = nxt
            if on_token:
                on_token(ids)
            if int(nxt[0, 0]) in eos or (stop is not None and stop(ids)):
                break
    return ids


def _accepts(model, arg: str) -> bool:
    try:
        return arg in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


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

        if not hasattr(self.model, "generate"):
            return self._generate_manual(inputs["input_ids"], on_text, max_new_tokens,
                                         temperature, top_p, stopping_criteria)
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


    def _generate_manual(self, input_ids, on_text, max_new_tokens, temperature, top_p, stopping):
        tok = self.tokenizer
        n0 = input_ids.shape[1]
        eos = [tok.eos_token_id, getattr(self.model.config, "eos_token_id", None)]
        eos = [e for x in eos for e in (x if isinstance(x, (list, tuple)) else [x])]
        stop = (lambda ids: bool(torch.as_tensor(stopping(ids, None)).any())) if stopping is not None else None

        def emit(ids):
            on_text(tok.decode(ids[0, n0:], skip_special_tokens=True))

        ids = manual_generate(self.model, input_ids, max_new_tokens, float(temperature),
                              float(top_p), eos, stop, emit)
        return tok.decode(ids[0, n0:], skip_special_tokens=True)


class DiffusionModel(LoadedModel):
    def __init__(self, path, pipe, pagers, device, dtype, kind):
        super().__init__(path, pagers, device, dtype)
        self.pipe = pipe
        self.kind = kind
        self.call_params = set(inspect.signature(pipe.__call__).parameters)
        from .speed import install
        self.step_caches = install(pipe)
        self.last_speed_info: Dict[str, Any] = {}

    def default_size(self):
        pipe = self.pipe
        h = getattr(pipe, "default_height", None)
        w = getattr(pipe, "default_width", None)
        if h and w:
            return int(w), int(h)
        den = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
        cfg = getattr(den, "config", None)
        size = getattr(cfg, "sample_size", None) if cfg is not None else None
        vsf = getattr(pipe, "vae_scale_factor", None) or getattr(pipe, "vae_scale_factor_spatial", 8)
        if isinstance(size, (list, tuple)):
            size = size[0]
        if isinstance(size, int) and isinstance(vsf, int):
            return size * vsf, size * vsf
        return 0, 0

    def default_frames(self) -> int:
        prm = inspect.signature(self.pipe.__call__).parameters.get("num_frames")
        d = prm.default if prm is not None else None
        return int(d) if isinstance(d, int) else 0

    def supports_speed(self) -> bool:
        return bool(self.step_caches) and bool({"callback_on_step_end", "callback"} & self.call_params)

    def accepts_image(self) -> bool:
        return "image" in self.call_params

    def generate(self, out_path: str, prompt: str, on_step: Callable[[int, int], None],
                 negative_prompt: str = "", steps: int = 25, guidance: float = 5.0,
                 width: int = 0, height: int = 0, frames: int = 0, fps: int = 16,
                 seed: int = -1, image=None, speed: float = 1.0, **_):
        from .speed import split_speed, upscale_image, upscale_video
        p = self.call_params
        speed = max(1.0, float(speed or 1))
        want_w, want_h = int(width or 0), int(height or 0)
        want_f = int(frames or 0)
        if speed > 1:  # tamanhos padrão do pipeline, para poder reduzir por dentro
            dw, dh = self.default_size()
            want_w, want_h = want_w or dw, want_h or dh
            if self.kind == "video":
                want_f = want_f or self.default_frames()
        sp = split_speed(speed, int(steps), want_w, want_h, want_f, self.default_frames(),
                         can_skip_steps=self.supports_speed())
        if speed > 1:
            width, height, frames = sp["width"], sp["height"], sp["frames"]
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
        plan = sp["plan"] if self.supports_speed() and speed > 1 else None
        for c in self.step_caches:
            c.start(plan)

        def step_done(step):
            for c in self.step_caches:
                c.next_step()
            on_step(step + 1, total)

        if "callback_on_step_end" in p:
            def cb(_pipe, step, _t, cb_kwargs):
                step_done(step)
                return cb_kwargs
            kw["callback_on_step_end"] = cb
        elif "callback" in p:  # pipelines antigos (ex.: ModelScope text-to-video)
            kw["callback"] = lambda step, _t, _lat: step_done(step)
            if "callback_steps" in p:
                kw["callback_steps"] = 1
        try:
            with torch.inference_mode():
                result = self.pipe(**kw)
        finally:
            self.last_speed_info = {
                "speed": speed,
                "estimated_gain": sp["estimated_gain"] if speed > 1 else 1.0,
                "at_floor": bool(sp["at_floor"]) if speed > 1 else False,
                "internal": f"{width or '?'}x{height or '?'}" + (f", {frames} quadros" if frames else ""),
                "computed": sum(c.computed for c in self.step_caches),
                "reused": sum(c.reused for c in self.step_caches),
            }
            for c in self.step_caches:
                c.start(None)
        if getattr(result, "frames", None) is not None:
            from diffusers.utils import export_to_video
            out_path = os.path.splitext(out_path)[0] + ".mp4"
            vid = result.frames[0]
            if speed > 1 and ((want_w, want_h) != (width, height) or (want_f and want_f != frames)):
                vid = upscale_video(vid, want_w, want_h, want_f)
            export_to_video(vid, out_path, fps=int(fps))
        else:
            out_path = os.path.splitext(out_path)[0] + ".png"
            img = result.images[0]
            if speed > 1 and want_w and want_h:
                img = upscale_image(img, want_w, want_h)
            img.save(out_path)
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


class RemoteCodeRequired(RuntimeError):
    pass


def needs_remote_code(path: str) -> bool:
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as fh:
            return bool(json.load(fh).get("auto_map"))
    except (OSError, ValueError):
        return False


def _transformers_compat() -> None:
    """Códigos de modelo escritos para o transformers 4.x importam nomes que o 5.x
    mudou de lugar; recoloca-os onde o código antigo procura."""
    import transformers.utils.generic as gen
    if not hasattr(gen, "OutputRecorder"):
        with contextlib.suppress(Exception):
            from transformers.utils.output_capturing import OutputRecorder
            gen.OutputRecorder = OutputRecorder


def simple_causal_mask(config=None, input_embeds=None, attention_mask=None, cache_position=None,
                       past_key_values=None, position_ids=None, inputs_embeds=None, **_):
    """Máscara causal [B, 1, Q, KV] (True = pode olhar), independente da versão do
    transformers. KV = tokens já no cache + tokens novos."""
    emb = inputs_embeds if inputs_embeds is not None else input_embeds
    B, Q = emb.shape[:2]
    dev = emb.device
    if cache_position is None:
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past, past + Q, device=dev)
    kv = int(cache_position[-1].item()) + 1
    if attention_mask is not None and attention_mask.dim() == 2:
        kv = max(kv, attention_mask.shape[-1])
    if Q == 1 and (attention_mask is None or bool(attention_mask.all())):
        return None  # 1 token novo vê tudo o que veio antes
    mask = torch.arange(kv, device=dev)[None, :] <= cache_position[:, None]  # [Q, KV]
    mask = mask[None, None].expand(B, 1, Q, kv)
    if attention_mask is not None and attention_mask.dim() == 2:
        mask = mask & attention_mask[:, None, None, :kv].bool()
    return mask


def _patch_remote_classes(cls) -> None:
    """Ajusta métodos de código antigo às chamadas do transformers 5.x."""
    import sys as _sys
    pkg = cls.__module__.rpartition(".")[0]
    mods = [m for n, m in list(_sys.modules.items()) if m is not None and (n == pkg or n.startswith(pkg + "."))]
    renamed = {"input_embeds": "inputs_embeds"}  # argumentos renomeados no transformers 5

    def kw_compat(fn):
        if getattr(fn, "_iapag_compat", False):
            return fn
        try:
            sig = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return fn
        accepted = set(sig)
        open_kw = any(p.kind == p.VAR_KEYWORD for p in sig.values())

        def call(*a, **kw):
            for old, new in renamed.items():
                if old in kw and old not in accepted and new in accepted:
                    kw[new] = kw.pop(old)
            if not open_kw:  # argumentos que deixaram de existir (ex.: cache_position)
                kw = {k: v for k, v in kw.items() if k in accepted}
            return fn(*a, **kw)
        call._iapag_compat = True
        return call

    for mod in mods:
        if callable(getattr(mod, "create_causal_mask", None)):
            mod.create_causal_mask = simple_causal_mask
        for name in ("create_sliding_window_causal_mask",):
            if callable(getattr(mod, name, None)):
                setattr(mod, name, kw_compat(getattr(mod, name)))
        for obj in list(vars(mod).values()):
            if not isinstance(obj, type) or "tie_weights" not in vars(obj):
                continue
            fn = vars(obj)["tie_weights"]
            if getattr(fn, "_iapag_compat", False):
                continue
            params = inspect.signature(fn).parameters
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                continue

            def wrapped(self, *a, _fn=fn, **kw):
                return _fn(self, *a)
            wrapped._iapag_compat = True
            obj.tie_weights = wrapped


def build_remote_model(config, path: str):
    """Instancia o modelo de código próprio já com os ajustes de compatibilidade."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    ref = config.auto_map.get("AutoModelForCausalLM") or config.auto_map.get("AutoModel")
    cls = get_class_from_dynamic_module(ref, path)
    _patch_remote_classes(cls)
    return cls._from_config(config) if hasattr(cls, "_from_config") else cls(config)


def _prepare_remote_code(device: torch.device) -> None:
    """Código próprio do modelo (ex.: Kimi) pode importar a biblioteca `fla`, que só
    roda em GPU. Na CPU (ou sem fla funcional) usamos a versão em PyTorch puro."""
    _transformers_compat()
    use_shim = device.type != "cuda"
    if not use_shim:
        try:
            import fla.ops.kda  # noqa: F401
        except Exception:  # noqa: BLE001
            use_shim = True
    if use_shim:
        from . import fla_cpu
        fla_cpu.install()


def _force_sdpa(model: torch.nn.Module) -> None:
    """Alguns códigos fixam flash_attention_2 (só GPU); usa a atenção do PyTorch."""
    from transformers import PretrainedConfig
    seen = set()
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if isinstance(cfg, PretrainedConfig) and id(cfg) not in seen:
            seen.add(id(cfg))
            for sub in [cfg] + [v for v in vars(cfg).values() if isinstance(v, PretrainedConfig)]:
                with contextlib.suppress(Exception):
                    sub._attn_implementation = "sdpa"
        if hasattr(m, "_use_flash_attention_2"):
            m._use_flash_attention_2 = False


def load_text(path: str, device, dtype, on_event, prefetch=2, pack="auto",
              min_page_mb=1.0, trust_remote_code=False, expert_cache_gb=0.0) -> TextModel:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device)
    remote = needs_remote_code(path)
    if remote and not trust_remote_code:
        raise RemoteCodeRequired(
            "Este modelo traz código Python próprio (ex.: Kimi K3). Marque “Confiar no código "
            "do modelo” para carregá-lo — ele será executado no seu computador.")
    if remote:
        _prepare_remote_code(device)
    config = AutoConfig.from_pretrained(path, trust_remote_code=remote)
    with init_empty_weights(include_buffers=False), default_dtype(dtype):
        if remote:
            model = build_remote_model(config, path)
        else:
            model = AutoModelForCausalLM.from_config(config)
    model.eval()
    if remote or device.type != "cuda":
        _force_sdpa(model)
    pager = PagedModel(model, CheckpointIndex(path), device, dtype, prefetch=prefetch,
                       pack=pack, min_page_mb=min_page_mb, on_event=on_event, label="texto",
                       expert_cache_gb=expert_cache_gb)
    pin_device(model, device)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=remote)
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
             prefetch=2, pack="auto", min_page_mb=1.0, resident_below_mb=0.0,
             trust_remote_code=False, expert_cache_gb=0.0) -> LoadedModel:
    kind = detect_kind(path)
    device = pick_device(device_pref)
    dtype = pick_dtype(dtype_pref, device)
    tune_cpu(device)
    prefetch = int(prefetch)
    if kind == "text":
        return load_text(path, device, dtype, on_event, prefetch, pack, min_page_mb,
                         trust_remote_code, expert_cache_gb)
    if kind in ("image", "video"):
        return load_diffusion(path, device, dtype, on_event, prefetch, pack, min_page_mb,
                              resident_below_mb)
    raise ValueError(f"Formato de modelo não reconhecido em {path}")
