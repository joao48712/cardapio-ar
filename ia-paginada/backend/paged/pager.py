"""Carregamento paginado de pesos: "imprimir o livro uma página por vez".

Em vez de colocar o modelo inteiro na memória, o modelo é construído vazio
(no dispositivo ``meta``, sem ocupar RAM/VRAM). Cada bloco grande (uma camada
de transformer, um bloco da UNet, ...) vira uma "página". Logo antes de a página
ser usada, um hook lê os pesos dela do arquivo ``.safetensors`` e os coloca no
dispositivo; logo depois, os pesos são jogados fora e o próximo bloco é lido.
Enquanto uma página "imprime", a próxima já é lida do disco em segundo plano.

O pico de memória fica perto de: pesos residentes (embeddings, normas, cabeça)
+ 1 ou 2 páginas + ativações, em vez do modelo inteiro.
"""
from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional

import torch
from safetensors import safe_open
from torch import nn

EventFn = Callable[[dict], None]


# --------------------------------------------------------------------------- #
# Índice dos arquivos .safetensors
# --------------------------------------------------------------------------- #
def _pick_safetensor_files(folder: str) -> List[str]:
    """Escolhe um conjunto coerente de arquivos (sem duplicar variantes)."""
    names = os.listdir(folder)
    indexes = sorted(n for n in names if n.endswith(".safetensors.index.json"))
    plain_idx = [n for n in indexes if n.count(".") == 3]  # model.safetensors.index.json
    for idx in plain_idx + indexes:
        with open(os.path.join(folder, idx), encoding="utf-8") as fh:
            files = sorted(set(json.load(fh)["weight_map"].values()))
        if all(os.path.exists(os.path.join(folder, f)) for f in files):
            return files
    files = sorted(n for n in names if n.endswith(".safetensors"))
    plain = [f for f in files if f.count(".") == 1]  # sem ".fp16." etc.
    return plain or files


class CheckpointIndex:
    """Sabe em qual arquivo está cada tensor e lê só os tensores pedidos."""

    def __init__(self, folder: str):
        self.folder = folder
        self.files = _pick_safetensor_files(folder)
        if not self.files:
            raise FileNotFoundError(f"Nenhum .safetensors em {folder}")
        self.key_to_file: Dict[str, str] = {}
        self.key_bytes: Dict[str, int] = {}
        for f in self.files:
            with safe_open(os.path.join(folder, f), framework="pt") as h:
                for k in h.keys():
                    self.key_to_file[k] = f
                    sl = h.get_slice(k)
                    numel = 1
                    for d in sl.get_shape():
                        numel *= d
                    self.key_bytes[k] = numel * _dtype_size(sl.get_dtype())
        self._local = threading.local()
        self._suffix: Optional[Dict[str, List[str]]] = None

    @property
    def by_suffix(self) -> Dict[str, List[str]]:
        """sufixo -> chaves completas (ex.: "embeddings.x" -> ["text_model.embeddings.x"])."""
        if self._suffix is None:
            self._suffix = {}
            for k in self.key_to_file:
                parts = k.split(".")
                for i in range(1, len(parts) - 1):
                    self._suffix.setdefault(".".join(parts[i:]), []).append(k)
        return self._suffix

    @property
    def total_bytes(self) -> int:
        return sum(self.key_bytes.values())

    def __contains__(self, key: str) -> bool:
        return key in self.key_to_file

    def _handle(self, fname: str):
        cache = getattr(self._local, "handles", None)
        if cache is None:
            cache = self._local.handles = {}
        h = cache.get(fname)
        if h is None:
            h = cache[fname] = safe_open(os.path.join(self.folder, fname), framework="pt")
        return h

    def load(self, keys: Iterable[str]) -> Dict[str, torch.Tensor]:
        out = {}
        for k in keys:
            out[k] = self._handle(self.key_to_file[k]).get_tensor(k)
        return out


def _dtype_size(st_dtype: str) -> int:
    return {
        "F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4,
        "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    }.get(st_dtype, 1)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def default_dtype(dtype: torch.dtype):
    old = torch.get_default_dtype()
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _set_tensor(root: nn.Module, name: str, value: torch.Tensor, is_param: bool) -> None:
    mod_name, _, leaf = name.rpartition(".")
    mod = root.get_submodule(mod_name) if mod_name else root
    if is_param:
        mod._parameters[leaf] = nn.Parameter(value, requires_grad=False)
    else:
        mod._buffers[leaf] = value


def _find_page_prefixes(model: nn.Module) -> List[str]:
    """Páginas = filhos das ModuleList mais externas (camadas/blocos)."""
    lists = [n for n, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) >= 2]
    outer = [n for n in lists if not any(n != o and n.startswith(o + ".") for o in lists)]
    pages: List[str] = []
    for n in outer:
        for i, _ in enumerate(model.get_submodule(n)):
            pages.append(f"{n}.{i}")
    if not pages:  # modelo sem ModuleList: usa os filhos de primeiro nível
        pages = [n for n, _ in model.named_children()]
    return pages


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


# --------------------------------------------------------------------------- #
# O paginador
# --------------------------------------------------------------------------- #
class PagedModel:
    """Liga hooks a um modelo construído em ``meta`` para paginar seus pesos.

    Args:
        model: modelo construído com ``accelerate.init_empty_weights``.
        index: índice dos pesos no disco.
        device: onde a página é "impressa" (cuda, mps ou cpu).
        dtype: tipo dos pesos de ponto flutuante.
        prefetch: lê a próxima página em segundo plano.
        ram_cache_gb: mantém até N GB de páginas já lidas na RAM (0 = sempre
            relê do disco). Útil quando o modelo cabe na RAM mas não na VRAM.
        paged: False carrega tudo de uma vez (para componentes pequenos).
    """

    def __init__(
        self,
        model: nn.Module,
        index: CheckpointIndex,
        device: torch.device,
        dtype: torch.dtype,
        prefetch: bool = True,
        ram_cache_gb: float = 0.0,
        paged: bool = True,
        on_event: Optional[EventFn] = None,
        label: str = "modelo",
    ):
        self.model = model
        self.index = index
        self.device = torch.device(device)
        self.dtype = dtype
        self.prefetch = prefetch
        self.on_event = on_event
        self.label = label
        self.ram_budget = int(ram_cache_gb * 1024**3)
        self.ram_cache: "OrderedDict[int, Dict[str, torch.Tensor]]" = OrderedDict()
        self.ram_used = 0
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        self._futures: Dict[int, Future] = {}
        self.stats = {"label": label, "pages": 0, "loads": 0, "bytes_read": 0,
                      "current": None, "last_ms": 0.0, "resident_bytes": 0}

        prefixes = [getattr(model, "base_model_prefix", "") or ""]
        params = dict(model.named_parameters(remove_duplicate=False))
        persistent = {n for n, _ in _named_persistent_buffers(model)}
        buffers = dict(model.named_buffers(remove_duplicate=False))

        # nome no modelo -> chave no checkpoint
        self.key_of: Dict[str, str] = {}
        for name in list(params) + [b for b in buffers if b in persistent]:
            key = _resolve_key(name, index, prefixes)
            if key is not None:
                self.key_of[name] = key

        self.page_prefixes = _find_page_prefixes(model) if paged else []
        self.page_names: List[List[str]] = [[] for _ in self.page_prefixes]
        self.is_param: Dict[str, bool] = {n: True for n in params}
        self.is_param.update({n: False for n in buffers if n not in params})
        resident: List[str] = []
        for name in self.key_of:
            for i, p in enumerate(self.page_prefixes):
                if _under(name, p):
                    self.page_names[i].append(name)
                    break
            else:
                resident.append(name)

        missing = [n for n in params if n not in self.key_of]
        missing_in_pages = [n for n in missing if any(_under(n, p) for p in self.page_prefixes)]
        if missing_in_pages:
            raise KeyError(
                f"{label}: {len(missing_in_pages)} pesos de páginas não encontrados no "
                f"checkpoint, ex.: {missing_in_pages[:5]}"
            )

        # 1) buffers que não estão no checkpoint (ex.: rotary inv_freq) vão já
        for name, buf in buffers.items():
            if name not in self.key_of and buf.device.type != "meta":
                _set_tensor(model, name, buf.to(self.device), False)

        # 2) pesos residentes (fora das páginas) ficam sempre carregados
        tensors = index.load(self.key_of[n] for n in resident)
        for name in resident:
            t = self._cast(tensors[self.key_of[name]])
            self.stats["resident_bytes"] += t.numel() * t.element_size()
            _set_tensor(model, name, t, self.is_param[name])
        del tensors

        # 3) pesos amarrados (ex.: lm_head = embed_tokens) que faltam no arquivo
        if missing and hasattr(model, "tie_weights"):
            with contextlib.suppress(Exception):
                model.tie_weights()
        still_meta = [n for n, p in model.named_parameters()
                      if p.device.type == "meta" and not any(_under(n, q) for q in self.page_prefixes)]
        if still_meta:
            raise KeyError(f"{label}: pesos ausentes no checkpoint: {still_meta[:5]}")

        # 4) hooks nas páginas
        self._depth = [0] * len(self.page_prefixes)
        self._loaded = [False] * len(self.page_prefixes)
        self._handles = []
        for i, p in enumerate(self.page_prefixes):
            mod = model.get_submodule(p)
            self._handles.append(mod.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(mod.register_forward_hook(self._make_post(i)))
        self.stats["pages"] = len(self.page_prefixes)
        self._emit({"type": "ready", "pages": len(self.page_prefixes),
                    "resident_mb": round(self.stats["resident_bytes"] / 2**20, 1)})

    # ----------------------------------------------------------------- hooks
    def _make_pre(self, i: int):
        def pre(_module, _args):
            with self._lock:
                self._depth[i] += 1
                if not self._loaded[i]:
                    self._load_page(i)
        return pre

    def _make_post(self, i: int):
        def post(_module, _args, _out):
            with self._lock:
                self._depth[i] = max(0, self._depth[i] - 1)
                if self._depth[i] == 0 and self._loaded[i]:
                    self._unload_page(i)
        return post

    # ------------------------------------------------------------ leitura
    def _cast(self, t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point() and t.dtype != self.dtype:
            t = t.to(self.dtype)
        return t.to(self.device, non_blocking=True)

    def _read_cpu(self, i: int) -> Dict[str, torch.Tensor]:
        cached = self.ram_cache.get(i)
        if cached is not None:
            self.ram_cache.move_to_end(i)
            return cached
        names = self.page_names[i]
        raw = self.index.load(self.key_of[n] for n in names)
        out = {}
        for n in names:
            t = raw[self.key_of[n]]
            if t.is_floating_point() and t.dtype != self.dtype:
                t = t.to(self.dtype)
            if self.device.type == "cuda":
                t = t.pin_memory()
            out[n] = t
        if self.ram_budget > 0:
            size = sum(t.numel() * t.element_size() for t in out.values())
            if size <= self.ram_budget:
                while self.ram_used + size > self.ram_budget and self.ram_cache:
                    _, old = self.ram_cache.popitem(last=False)
                    self.ram_used -= sum(t.numel() * t.element_size() for t in old.values())
                self.ram_cache[i] = out
                self.ram_used += size
        return out

    def _load_page(self, i: int) -> None:
        t0 = time.perf_counter()
        fut = self._futures.pop(i, None)
        for j in list(self._futures):  # leitura antecipada errada: descarta
            self._futures.pop(j).cancel()
        tensors = fut.result() if fut is not None else self._read_cpu(i)
        nbytes = 0
        for name, t in tensors.items():
            t = t.to(self.device, non_blocking=True)
            nbytes += t.numel() * t.element_size()
            _set_tensor(self.model, name, t, self.is_param[name])
        self._loaded[i] = True
        if self.prefetch and len(self.page_prefixes) > 1:
            nxt = (i + 1) % len(self.page_prefixes)
            if nxt not in self._futures and not self._loaded[nxt]:
                self._futures[nxt] = self._pool.submit(self._read_cpu, nxt)
        ms = (time.perf_counter() - t0) * 1000
        st = self.stats
        st["loads"] += 1
        st["bytes_read"] += nbytes
        st["current"] = i
        st["last_ms"] = round(ms, 1)
        self._emit({"type": "page", "action": "load", "index": i, "total": len(self.page_prefixes),
                    "name": self.page_prefixes[i], "ms": round(ms, 1), "mb": round(nbytes / 2**20, 1)})

    def _unload_page(self, i: int) -> None:
        for name in self.page_names[i]:
            mod_name, _, leaf = name.rpartition(".")
            mod = self.model.get_submodule(mod_name) if mod_name else self.model
            store = mod._parameters if self.is_param[name] else mod._buffers
            old = store[leaf]
            empty = torch.empty(old.shape, dtype=old.dtype, device="meta")
            store[leaf] = nn.Parameter(empty, requires_grad=False) if self.is_param[name] else empty
        self._loaded[i] = False

    # ------------------------------------------------------------- público
    def reset(self) -> None:
        """Descarta todas as páginas (ex.: após um erro no meio do forward)."""
        with self._lock:
            for i, loaded in enumerate(self._loaded):
                self._depth[i] = 0
                if loaded:
                    self._unload_page(i)
            for f in self._futures.values():
                f.cancel()
            self._futures.clear()

    def close(self) -> None:
        self.reset()
        for h in self._handles:
            h.remove()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.ram_cache.clear()

    def _emit(self, ev: dict) -> None:
        if self.on_event:
            ev["label"] = self.label
            with contextlib.suppress(Exception):
                self.on_event(ev)


def _named_persistent_buffers(model: nn.Module):
    for mod_name, mod in model.named_modules():
        skip = getattr(mod, "_non_persistent_buffers_set", set())
        for b_name, buf in mod._buffers.items():
            if buf is not None and b_name not in skip:
                yield (f"{mod_name}.{b_name}" if mod_name else b_name), buf


def _resolve_key(name: str, index: CheckpointIndex, prefixes: List[str]) -> Optional[str]:
    if name in index:
        return name
    for p in prefixes:
        if not p:
            continue
        if name.startswith(p + ".") and name[len(p) + 1:] in index:
            return name[len(p) + 1:]
        if f"{p}.{name}" in index:
            return f"{p}.{name}"
    # O checkpoint tem um prefixo a mais (ex.: "text_model.") ou a menos: casa
    # pelo sufixo, desde que a correspondência seja única.
    hits = index.by_suffix.get(name, [])
    if len(hits) == 1:
        return hits[0]
    parts = name.split(".")
    for i in range(1, len(parts) - 1):
        tail = ".".join(parts[i:])
        if tail in index:
            return tail
    return None
