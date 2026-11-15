"""Busca e download de modelos do Hugging Face (Hub e Spaces)."""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Dict, List, Optional

from huggingface_hub import HfApi, snapshot_download

TASKS = {
    "text": "text-generation",
    "image": "text-to-image",
    "image2image": "image-to-image",
    "video": "text-to-video",
    "image2video": "image-to-video",
}
SKIP_EXT = (".bin", ".pt", ".pth", ".ckpt", ".onnx", ".onnx_data", ".msgpack", ".h5",
            ".gguf", ".ot", ".tflite", ".mlmodel", ".npz", ".pb")
KEEP_EXT = (".json", ".txt", ".model", ".safetensors", ".tiktoken", ".jinja", ".py")


def _api(token: Optional[str]) -> HfApi:
    return HfApi(token=token or None)


def search(query: str = "", kind: str = "", limit: int = 30, token: Optional[str] = None) -> List[dict]:
    tag = TASKS.get(kind) or None
    out = []
    for m in _api(token).list_models(search=query or None, pipeline_tag=tag, sort="downloads",
                                     limit=limit, full=False):
        out.append({
            "id": m.id, "downloads": getattr(m, "downloads", 0) or 0,
            "likes": getattr(m, "likes", 0) or 0, "task": getattr(m, "pipeline_tag", None),
            "library": getattr(m, "library_name", None),
            "gated": bool(getattr(m, "gated", False)),
        })
    return out


def models_from_space(space_id: str, token: Optional[str] = None) -> List[str]:
    """Descobre quais modelos um Space do Hugging Face usa (para baixá-los)."""
    api = _api(token)
    info = api.space_info(space_id)
    found = list(getattr(info, "models", None) or [])
    if not found:
        files = [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".py")][:10]
        pat = re.compile(r"""["']([A-Za-z0-9][\w.-]*/[\w.-]+)["']""")
        cands = set()
        for f in files:
            try:
                p = api.hf_hub_download(space_id, f, repo_type="space")
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    cands.update(pat.findall(fh.read()))
            except Exception:  # noqa: BLE001
                continue
        for c in sorted(cands):
            try:
                api.model_info(c)
                found.append(c)
            except Exception:  # noqa: BLE001
                pass
    return found


def plan(repo_id: str, revision: Optional[str] = None, prefer_fp16: bool = True,
         token: Optional[str] = None) -> dict:
    """Lista os arquivos que serão baixados (só safetensors + configs)."""
    info = _api(token).model_info(repo_id, revision=revision, files_metadata=True)
    sib = {s.rfilename: (s.size or 0) for s in info.siblings or []}
    names = list(sib)
    is_diffusers = "model_index.json" in sib
    has_st = any(n.endswith(".safetensors") for n in names)
    if not has_st:
        raise ValueError("Este repositório não tem arquivos .safetensors (formato exigido "
                         "para a leitura página por página).")
    chosen = []
    by_dir: Dict[str, List[str]] = {}
    for n in names:
        if n.endswith(SKIP_EXT) or not n.endswith(KEEP_EXT):
            continue
        if n.endswith(".py") and not is_diffusers:
            continue
        d = os.path.dirname(n)
        if is_diffusers and d == "" and n.endswith(".safetensors"):
            continue  # checkpoint "single file" duplicado na raiz
        by_dir.setdefault(d, []).append(n)
    for d, files in by_dir.items():
        st = [f for f in files if f.endswith(".safetensors") or f.endswith(".safetensors.index.json")]
        plain = [f for f in st if "." not in os.path.basename(f).split(".safetensors")[0]]
        fp16 = [f for f in st if ".fp16." in f]
        keep_st = fp16 if prefer_fp16 and fp16 else (plain or st)
        for f in files:
            if f in st and f not in keep_st:
                continue
            chosen.append(f)
    total = sum(sib[f] for f in chosen)
    kind = None
    if is_diffusers:
        kind = "video" if "video" in (info.pipeline_tag or "") else "image"
    elif "config.json" in sib and (info.library_name in (None, "transformers")):
        kind = "text"
    if kind is None:
        raise ValueError("Formato não suportado: escolha um repositório no formato Diffusers "
                         "(imagem/vídeo, procure o sufixo “-Diffusers”) ou Transformers (texto).")
    return {"repo_id": repo_id, "files": sorted(chosen), "bytes": total, "kind": kind,
            "task": info.pipeline_tag, "gated": bool(getattr(info, "gated", False))}


def local_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def download(repo_id: str, models_dir: str, files: List[str], revision: Optional[str] = None,
             token: Optional[str] = None) -> str:
    dest = os.path.join(models_dir, local_name(repo_id))
    os.makedirs(dest, exist_ok=True)
    snapshot_download(repo_id, revision=revision, local_dir=dest, allow_patterns=files,
                      token=token or None)
    with open(os.path.join(dest, ".ia-paginada.json"), "w", encoding="utf-8") as fh:
        json.dump({"repo_id": repo_id, "revision": revision}, fh)
    return dest


def list_local(models_dir: str) -> List[dict]:
    from .loaders import detect_kind
    out = []
    if not os.path.isdir(models_dir):
        return out
    for name in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, name)
        meta_p = os.path.join(path, ".ia-paginada.json")
        if not os.path.isdir(path):
            continue
        meta = {}
        if os.path.exists(meta_p):
            with open(meta_p, encoding="utf-8") as fh:
                meta = json.load(fh)
        out.append({"name": name, "repo_id": meta.get("repo_id", name), "kind": detect_kind(path),
                    "bytes": dir_size(path), "complete": bool(meta)})
    return out


def delete_local(models_dir: str, name: str) -> None:
    path = os.path.realpath(os.path.join(models_dir, name))
    if os.path.dirname(path) != os.path.realpath(models_dir):
        raise ValueError("nome inválido")
    shutil.rmtree(path)
