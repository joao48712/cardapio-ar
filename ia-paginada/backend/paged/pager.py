"""Carregamento paginado de pesos: "imprimir o livro uma página por vez".

O modelo é construído vazio (dispositivo ``meta``, sem ocupar memória). Cada
bloco vira uma "página": logo antes de a página rodar, um hook coloca os pesos
dela no lugar; logo depois, eles são jogados fora e a próxima página entra.

Para a troca de página ser rápida (principalmente só na CPU):

* **Sem cópia (mmap):** os pesos são *janelas* direto no arquivo mapeado em
  memória (``torch.frombuffer``). "Carregar" uma página na CPU custa microssegundos;
  o sistema operacional lê do disco só o que for tocado e guarda no cache dele.
* **Encadernação:** se o checkpoint está num tipo diferente do usado no cálculo
  (ex.: fp16 no disco, fp32 na CPU), converte-se UMA vez para um arquivo na ordem
  das páginas, alinhado, já no tipo final. Depois disso, nunca mais há conversão.
* **Leitura antecipada que aprende a ordem:** o paginador registra a ordem real
  em que as páginas rodam e pede ao sistema (``madvise WILLNEED``) as próximas N
  páginas enquanto a atual calcula.
* **Tudo vira página:** além das camadas, peças grandes como embeddings e a
  cabeça de saída também são paginadas.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import mmap
import os
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

EventFn = Callable[[dict], None]
PACK_VERSION = 1
ALIGN = 64

ST_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
for _n, _a in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
    if hasattr(torch, _a):
        ST_DTYPES[_n] = getattr(torch, _a)
DTYPE_NAMES = {v: k for k, v in ST_DTYPES.items()}


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# --------------------------------------------------------------------------- #
# Arquivo mapeado em memória
# --------------------------------------------------------------------------- #
class MmapFile:
    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        size = os.fstat(self._fh.fileno()).st_size
        # ACCESS_COPY = mapeamento privado e "gravável" (exigido pelo
        # torch.frombuffer); nada é copiado enquanto ninguém escrever.
        self.mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_COPY) if size else None

    def view(self, offset: int, dtype: torch.dtype, shape) -> torch.Tensor:
        n = _numel(shape)
        if n == 0:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(self.mm, dtype=dtype, count=n, offset=offset).view(shape)

    def willneed(self, offset: int, nbytes: int) -> None:
        if self.mm is None or nbytes <= 0 or not hasattr(self.mm, "madvise"):
            return
        start = offset - offset % mmap.PAGESIZE
        with contextlib.suppress(Exception):
            self.mm.madvise(mmap.MADV_WILLNEED, start, min(nbytes + offset - start, len(self.mm) - start))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self.mm is not None:
                self.mm.close()
        self._fh.close()


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


def _read_st_header(path: str) -> Dict[str, Tuple[int, int, str, list]]:
    """chave -> (offset absoluto, bytes, dtype safetensors, shape)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    base = 8 + n
    return {k: (base + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0],
                v["dtype"], v["shape"]) for k, v in hdr.items()}


class CheckpointIndex:
    """Sabe onde está cada tensor e entrega *janelas* sobre o arquivo (sem cópia)."""

    def __init__(self, folder: str):
        self.folder = folder
        self.files = _pick_safetensor_files(folder)
        if not self.files:
            raise FileNotFoundError(f"Nenhum .safetensors em {folder}")
        self.entries: Dict[str, Tuple[str, int, int, str, list]] = {}
        for f in self.files:
            for k, (off, nb, dt, shape) in _read_st_header(os.path.join(folder, f)).items():
                self.entries[k] = (f, off, nb, dt, shape)
        self._maps: Dict[str, MmapFile] = {}
        self._suffix: Optional[Dict[str, List[str]]] = None

    # -- informações
    @property
    def key_bytes(self) -> Dict[str, int]:
        return {k: e[2] for k, e in self.entries.items()}

    @property
    def total_bytes(self) -> int:
        return sum(e[2] for e in self.entries.values())

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def dtype_of(self, key: str) -> torch.dtype:
        return ST_DTYPES[self.entries[key][3]]

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for f in self.files:
            st = os.stat(os.path.join(self.folder, f))
            h.update(f"{f}:{st.st_size}:{int(st.st_mtime)};".encode())
        return h.hexdigest()

    @property
    def by_suffix(self) -> Dict[str, List[str]]:
        """sufixo -> chaves completas (ex.: "embeddings.x" -> ["text_model.embeddings.x"])."""
        if self._suffix is None:
            self._suffix = {}
            for k in self.entries:
                parts = k.split(".")
                for i in range(1, len(parts) - 1):
                    self._suffix.setdefault(".".join(parts[i:]), []).append(k)
        return self._suffix

    # -- acesso aos dados
    def _map(self, fname: str) -> MmapFile:
        m = self._maps.get(fname)
        if m is None:
            m = self._maps[fname] = MmapFile(os.path.join(self.folder, fname))
        return m

    def aligned(self, key: str) -> bool:
        _, off, _, dt, _ = self.entries[key]
        return off % ST_DTYPES[dt].itemsize == 0

    def view(self, key: str) -> torch.Tensor:
        f, off, _, dt, shape = self.entries[key]
        if not self.aligned(key):  # raro: offset desalinhado -> cópia
            raw = self._map(f).view(off, torch.uint8, [self.entries[key][2]]).clone()
            return raw.view(ST_DTYPES[dt]).view(shape)
        return self._map(f).view(off, ST_DTYPES[dt], shape)

    def willneed(self, key: str) -> None:
        f, off, nb, _, _ = self.entries[key]
        self._map(f).willneed(off, nb)

    def load(self, keys: Iterable[str]) -> Dict[str, torch.Tensor]:
        return {k: self.view(k) for k in keys}

    def close(self) -> None:
        for m in self._maps.values():
            m.close()
        self._maps.clear()


# --------------------------------------------------------------------------- #
# Fontes de pesos já no tipo final
# --------------------------------------------------------------------------- #
class DirectStore:
    """Lê do .safetensors original. Sem cópia quando o tipo já é o de cálculo."""

    kind = "direto"

    def __init__(self, index: CheckpointIndex, dtype: torch.dtype):
        self.index = index
        self.dtype = dtype
        self.zero_copy = not needs_conversion(index, dtype)
        if not self.zero_copy:
            self.kind = "conversão a cada leitura"

    def get(self, key: str) -> torch.Tensor:
        t = self.index.view(key)
        if t.is_floating_point() and t.dtype != self.dtype:
            t = t.to(self.dtype)
        return t

    def willneed(self, keys: Iterable[str]) -> None:
        for k in keys:
            self.index.willneed(k)

    def close(self) -> None:
        pass


class PackedStore:
    """Arquivo "encadernado": tensores já convertidos, alinhados e na ordem das páginas."""

    kind = "encadernado"
    zero_copy = True

    def __init__(self, path: str, meta: dict):
        self.path = path
        self.meta = meta["tensors"]
        self.file = MmapFile(path)

    def get(self, key: str) -> torch.Tensor:
        off, _, dt, shape = self.meta[key]
        return self.file.view(off, ST_DTYPES[dt], shape)

    def willneed(self, keys: Iterable[str]) -> None:
        for k in keys:
            off, nb, _, _ = self.meta[k]
            self.file.willneed(off, nb)

    def close(self) -> None:
        self.file.close()

    @staticmethod
    def paths(index: CheckpointIndex, dtype: torch.dtype) -> Tuple[str, str]:
        d = os.path.join(index.folder, ".ia-paginada-pack")
        name = str(dtype).replace("torch.", "")
        return os.path.join(d, f"{name}.bin"), os.path.join(d, f"{name}.json")

    @classmethod
    def open_or_build(cls, index: CheckpointIndex, dtype: torch.dtype, order: List[str],
                      on_progress: Optional[Callable[[float], None]] = None) -> "PackedStore":
        bin_p, meta_p = cls.paths(index, dtype)
        fp = index.fingerprint()
        if os.path.exists(bin_p) and os.path.exists(meta_p):
            with open(meta_p, encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("version") == PACK_VERSION and meta.get("source") == fp:
                return cls(bin_p, meta)
        os.makedirs(os.path.dirname(bin_p), exist_ok=True)
        keys = list(dict.fromkeys(list(order) + sorted(index.entries)))  # ordem das páginas
        total = max(1, index.total_bytes)
        done = 0
        last = -1.0
        tensors = {}
        off = 0
        tmp = bin_p + ".tmp"
        with open(tmp, "wb") as fh:
            for k in keys:
                t = index.view(k)
                if t.is_floating_point() and t.dtype != dtype:
                    t = t.to(dtype)
                t = t.contiguous()
                pad = (-off) % ALIGN
                if pad:
                    fh.write(b"\0" * pad)
                    off += pad
                raw = t.reshape(-1).view(torch.uint8).numpy() if t.numel() else b""
                fh.write(memoryview(raw) if t.numel() else b"")
                nb = t.numel() * t.element_size()
                tensors[k] = [off, nb, DTYPE_NAMES[t.dtype], list(t.shape)]
                off += nb
                done += index.entries[k][2]
                frac = done / total
                if on_progress and frac - last >= 0.01:
                    last = frac
                    on_progress(frac)
        meta = {"version": PACK_VERSION, "source": fp, "dtype": str(dtype), "tensors": tensors}
        os.replace(tmp, bin_p)
        with open(meta_p, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        if on_progress:
            on_progress(1.0)
        return cls(bin_p, meta)


def needs_conversion(index: CheckpointIndex, dtype: torch.dtype) -> bool:
    for k, (_, _, _, dt, _) in index.entries.items():
        t = ST_DTYPES[dt]
        if (t.is_floating_point and t != dtype) or not index.aligned(k):
            return True
    return False


def make_store(index: CheckpointIndex, dtype: torch.dtype, pack: str, order: List[str],
               on_progress=None):
    """pack: 'auto' (encaderna se precisar converter), 'off' (converte a cada leitura)."""
    if pack == "off" or not needs_conversion(index, dtype):
        return DirectStore(index, dtype)
    return PackedStore.open_or_build(index, dtype, order, on_progress)


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
    """Páginas principais = filhos das ModuleList mais externas (camadas/blocos)."""
    lists = [n for n, m in model.named_modules() if isinstance(m, nn.ModuleList) and len(m) >= 2]
    outer = [n for n in lists if not any(n != o and n.startswith(o + ".") for o in lists)]
    pages: List[str] = []
    for n in outer:
        for i, _ in enumerate(model.get_submodule(n)):
            pages.append(f"{n}.{i}")
    return pages


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _page_of(name: str, prefixes: List[str]) -> int:
    for i, p in enumerate(prefixes):
        if _under(name, p):
            return i
    return -1


# --------------------------------------------------------------------------- #
# O paginador
# --------------------------------------------------------------------------- #
class PagedModel:
    """Liga hooks a um modelo construído em ``meta`` para paginar seus pesos.

    Args:
        model: modelo construído com ``accelerate.init_empty_weights``.
        index: índice dos pesos no disco.
        device: onde a página é "impressa" (cuda, mps ou cpu).
        dtype: tipo dos pesos de ponto flutuante no cálculo.
        prefetch: quantas páginas ler adiante (0 = nenhuma).
        pack: 'auto' encaderna (converte uma vez) quando o tipo do disco difere.
        paged: False deixa o componente inteiro residente.
        min_page_mb: peças fora das camadas (embeddings, cabeça...) maiores que
            isso também viram páginas; as menores ficam residentes.
    """

    def __init__(
        self,
        model: nn.Module,
        index: CheckpointIndex,
        device,
        dtype: torch.dtype,
        prefetch: int = 2,
        pack: str = "auto",
        paged: bool = True,
        min_page_mb: float = 1.0,
        on_event: Optional[EventFn] = None,
        label: str = "modelo",
    ):
        self.model = model
        self.index = index
        self.device = torch.device(device)
        self.dtype = dtype
        self.prefetch = int(prefetch)
        self.on_event = on_event
        self.label = label
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prefetch")
        self._futures: Dict[int, Future] = {}
        self._next: Dict[int, int] = {}  # ordem aprendida: página -> próxima
        self._last: Optional[int] = None
        self.stats = {"label": label, "pages": 0, "loads": 0, "bytes_read": 0, "current": None,
                      "last_ms": 0.0, "avg_ms": 0.0, "resident_bytes": 0, "store": "",
                      "zero_copy": False, "learned": 0}
        self._ms_total = 0.0

        prefixes = [getattr(model, "base_model_prefix", "") or ""]
        # amarra pesos compartilhados (ex.: T5 shared = embed_tokens) ANTES de
        # decidir as páginas, para que o compartilhamento seja visível
        if hasattr(model, "tie_weights"):
            with contextlib.suppress(Exception):
                model.tie_weights()
        params = dict(model.named_parameters(remove_duplicate=False))
        persistent = {n for n, _ in _named_persistent_buffers(model)}
        buffers = dict(model.named_buffers(remove_duplicate=False))
        self.is_param: Dict[str, bool] = {n: True for n in params}
        self.is_param.update({n: False for n in buffers if n not in params})

        # nome no modelo -> chave no checkpoint
        self.key_of: Dict[str, str] = {}
        for name in list(params) + [b for b in buffers if b in persistent]:
            key = _resolve_key(name, index, prefixes)
            if key is not None:
                self.key_of[name] = key

        # --- páginas: camadas + peças grandes fora delas
        page_prefixes = _find_page_prefixes(model) if paged else []
        if paged:
            page_prefixes += self._extra_pages(model, params, page_prefixes, min_page_mb)
        self.page_prefixes = page_prefixes
        self.page_names: List[List[str]] = [[] for _ in page_prefixes]
        resident: List[str] = []
        for name in self.key_of:
            i = _page_of(name, page_prefixes)
            (self.page_names[i] if i >= 0 else resident).append(name)

        missing = [n for n in params if n not in self.key_of]
        bad = [n for n in missing if _page_of(n, page_prefixes) >= 0]
        if bad:
            raise KeyError(f"{label}: {len(bad)} pesos de páginas não encontrados no "
                           f"checkpoint, ex.: {bad[:5]}")

        # --- fonte dos pesos (direta sem cópia, ou encadernada)
        order = [self.key_of[n] for n in resident] + [self.key_of[n] for names in self.page_names for n in names]

        def pack_progress(frac):
            self._emit({"type": "pack", "progress": round(frac, 3)})

        self.store = make_store(index, dtype, pack, order, pack_progress)
        self.stats["store"] = self.store.kind
        self.stats["zero_copy"] = bool(self.store.zero_copy) and self.device.type == "cpu"

        # 1) buffers que não estão no checkpoint (ex.: rotary inv_freq) vão já
        for name, buf in buffers.items():
            if name not in self.key_of and buf.device.type != "meta":
                _set_tensor(model, name, buf.to(self.device), False)

        # 2) pesos residentes
        for name in resident:
            t = self._to_device(self.store.get(self.key_of[name]))
            self.stats["resident_bytes"] += t.numel() * t.element_size()
            _set_tensor(model, name, t, self.is_param[name])

        # 3) pesos amarrados (ex.: lm_head = embed_tokens) que faltam no arquivo
        if missing and hasattr(model, "tie_weights"):
            with contextlib.suppress(Exception):
                model.tie_weights()
        still_meta = [n for n, p in model.named_parameters()
                      if p.device.type == "meta" and _page_of(n, page_prefixes) < 0]
        if still_meta:
            raise KeyError(f"{label}: pesos ausentes no checkpoint: {still_meta[:5]}")

        # 4) hooks nas páginas
        self._depth = [0] * len(page_prefixes)
        self._loaded = [False] * len(page_prefixes)
        self._handles = []
        for i, p in enumerate(page_prefixes):
            mod = model.get_submodule(p)
            self._handles.append(mod.register_forward_pre_hook(self._make_pre(i)))
            self._handles.append(mod.register_forward_hook(self._make_post(i)))
        self.stats["pages"] = len(page_prefixes)
        self._emit({"type": "ready", "pages": len(page_prefixes), "store": self.store.kind,
                    "resident_mb": round(self.stats["resident_bytes"] / 2**20, 1)})

    def _extra_pages(self, model, params, layer_pages, min_page_mb) -> List[str]:
        """Peças grandes fora das camadas (embedding, cabeça, convs) viram páginas.

        Só módulos-folha (Linear, Embedding, Conv...) cujo peso pertence a eles e
        não é compartilhado: assim o hook dispara sempre que o peso é usado.
        """
        ids: Dict[int, int] = {}
        for p in params.values():
            ids[id(p)] = ids.get(id(p), 0) + 1
        size: Dict[str, int] = {}
        for name, p in params.items():
            if _page_of(name, layer_pages) >= 0:
                continue
            owner = name.rpartition(".")[0]
            mod = model.get_submodule(owner) if owner else model
            if (not owner or ids[id(p)] > 1 or name not in self.key_of
                    or any(True for _ in mod.children())):
                size[owner] = -(10**18)  # não paginável (compartilhado/ausente/não-folha)
                continue
            size[owner] = size.get(owner, 0) + self.index.entries[self.key_of[name]][2]
        lim = min_page_mb * 2**20
        return [o for o, s in size.items() if s > 0 and s >= lim]

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
    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cpu":
            return t
        return t.to(self.device, non_blocking=self.device.type == "cuda" and t.is_pinned())

    def _materialize(self, i: int) -> Dict[str, torch.Tensor]:
        """Lê a página para a RAM (usado quando há cópia: GPU ou conversão)."""
        out = {}
        for n in self.page_names[i]:
            t = self.store.get(self.key_of[n])
            if self.device.type == "cuda":
                t = t.pin_memory()
            elif not self.store.zero_copy:
                t = t.contiguous()
            out[n] = t
        return out

    def _predict(self, i: int) -> List[int]:
        n = len(self.page_prefixes)
        out, j = [], i
        for _ in range(min(self.prefetch, n - 1)):
            j = self._next.get(j, (j + 1) % n)
            if j == i or j in out:
                break
            out.append(j)
        return out

    def _load_page(self, i: int) -> None:
        t0 = time.perf_counter()
        if self._last is not None and self._last != i:
            if self._next.get(self._last) != i:
                self._next[self._last] = i
                self.stats["learned"] = len(self._next)
        self._last = i
        fast = self.device.type == "cpu" and self.store.zero_copy
        if fast:  # janelas sobre o arquivo: nada a copiar
            tensors = {n: self.store.get(self.key_of[n]) for n in self.page_names[i]}
        else:
            fut = self._futures.pop(i, None)
            tensors = fut.result() if fut is not None else self._materialize(i)
        nbytes = 0
        for name, t in tensors.items():
            t = self._to_device(t)
            nbytes += t.numel() * t.element_size()
            _set_tensor(self.model, name, t, self.is_param[name])
        self._loaded[i] = True

        # leitura antecipada das próximas páginas (na ordem aprendida)
        nxt = self._predict(i)
        if fast:
            for j in nxt:
                self.store.willneed(self.key_of[n] for n in self.page_names[j])
        else:
            for j in list(self._futures):
                if j not in nxt:
                    self._futures.pop(j).cancel()
            for j in nxt:
                if j not in self._futures and not self._loaded[j]:
                    self._futures[j] = self._pool.submit(self._materialize, j)

        ms = (time.perf_counter() - t0) * 1000
        st = self.stats
        st["loads"] += 1
        st["bytes_read"] += nbytes
        st["current"] = i
        st["last_ms"] = round(ms, 3)
        self._ms_total += ms
        st["avg_ms"] = round(self._ms_total / st["loads"], 3)
        self._emit({"type": "page", "action": "load", "index": i, "total": len(self.page_prefixes),
                    "name": self.page_prefixes[i], "ms": round(ms, 2), "mb": round(nbytes / 2**20, 1)})

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
        self.store.close()

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
