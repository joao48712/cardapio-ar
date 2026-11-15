"""Servidor local da IA Paginada (iniciado pelo Electron).

    python server.py --port 8765 --home ~/IA-Paginada
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from paged import hub  # noqa: E402

HOME = os.path.expanduser(os.environ.get("IAPAG_HOME", "~/IA-Paginada"))
SECRET = os.environ.get("IAPAG_SECRET", "")


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- #
# Tarefas em segundo plano
# --------------------------------------------------------------------------- #
class Job:
    def __init__(self, kind: str, title: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.status = "running"  # running | done | error | cancelled
        self.progress = 0.0
        self.message = ""
        self.text = ""
        self.result: Any = None
        self.error: Optional[str] = None
        self.log: deque = deque(maxlen=300)
        self.cancel = False
        self.started = time.time()
        self.finished: Optional[float] = None

    def add_log(self, msg: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")

    def to_dict(self, full: bool = True) -> dict:
        d = {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
             "progress": round(self.progress, 4), "message": self.message,
             "error": self.error, "result": self.result, "started": self.started,
             "finished": self.finished}
        if full:
            d["text"] = self.text
            d["log"] = list(self.log)
        return d


JOBS: Dict[str, Job] = {}
STATE: Dict[str, Any] = {"model": None, "name": None, "job": None, "loading": False}
GPU_LOCK = threading.Lock()


def start_job(kind: str, title: str, fn) -> Job:
    job = Job(kind, title)
    JOBS[job.id] = job

    def run():
        try:
            job.result = fn(job)
            job.status = "cancelled" if job.cancel else "done"
            job.progress = 1.0 if job.status == "done" else job.progress
        except Cancelled:
            job.status = "cancelled"
        except Exception as e:  # noqa: BLE001
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.add_log(traceback.format_exc(limit=6))
        finally:
            job.finished = time.time()

    threading.Thread(target=run, daemon=True, name=f"job-{kind}").start()
    return job


def on_page_event(ev: dict) -> None:
    job: Optional[Job] = STATE.get("job")
    if job is None:
        return
    if ev["type"] == "page":
        job.message = (f"📄 {ev['label']}: página {ev['index'] + 1}/{ev['total']} "
                       f"({ev['mb']} MB, {ev['ms']} ms)")
        # registra só a primeira passada (e páginas lentas) para não inundar o log
        seen = job.__dict__.setdefault("pages_seen", {})
        seen[ev["label"]] = seen.get(ev["label"], 0) + 1
        if seen[ev["label"]] <= ev["total"] or ev["ms"] > 2000:
            job.add_log(job.message)
    elif ev["type"] == "ready":
        job.add_log(f"✅ {ev['label']}: {ev['pages']} páginas, {ev['resident_mb']} MB residentes")


# --------------------------------------------------------------------------- #
app = FastAPI(title="IA Paginada")
app.add_middleware(CORSMiddleware, allow_origins=["null", "http://localhost", "http://127.0.0.1"],
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def check_secret(request: Request, call_next):
    p = request.url.path
    if (SECRET and p.startswith("/api/") and p != "/api/health" and request.method != "OPTIONS"
            and request.headers.get("x-iapag-key") != SECRET):
        return JSONResponse({"detail": "chave inválida"}, status_code=401)
    return await call_next(request)


def models_dir() -> str:
    d = os.path.join(HOME, "models")
    os.makedirs(d, exist_ok=True)
    return d


def outputs_dir() -> str:
    d = os.path.join(HOME, "outputs")
    os.makedirs(d, exist_ok=True)
    return d


def token_of(t: Optional[str]) -> Optional[str]:
    return t or os.environ.get("HF_TOKEN") or None


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/system")
def system():
    import platform

    import psutil
    import torch
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.cuda.is_available(), "gpus": [], "home": HOME,
            "ram_total": psutil.virtual_memory().total,
            "ram_available": psutil.virtual_memory().available,
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "disk_free": psutil.disk_usage(HOME if os.path.isdir(HOME) else os.path.expanduser("~")).free}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            info["gpus"].append({"name": torch.cuda.get_device_name(i), "total": total, "free": free})
    return info


# ------------------------------------------------------------------ Hub
@app.get("/api/hub/search")
def hub_search(q: str = "", kind: str = "", token: str = ""):
    try:
        return hub.search(q, kind, token=token_of(token))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Falha ao buscar no Hugging Face: {e}")


@app.get("/api/hub/space")
def hub_space(id: str, token: str = ""):
    try:
        return {"space": id, "models": hub.models_from_space(id, token=token_of(token))}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Não consegui ler o Space: {e}")


@app.get("/api/hub/plan")
def hub_plan(repo: str, fp16: bool = True, token: str = ""):
    try:
        return hub.plan(repo, prefer_fp16=fp16, token=token_of(token))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))


class DownloadReq(BaseModel):
    repo_id: str
    fp16: bool = True
    token: Optional[str] = None


@app.post("/api/download")
def download(req: DownloadReq):
    tok = token_of(req.token)

    def work(job: Job):
        p = hub.plan(req.repo_id, prefer_fp16=req.fp16, token=tok)
        job.add_log(f"{len(p['files'])} arquivos, {p['bytes'] / 2**30:.2f} GB")
        dest = os.path.join(models_dir(), hub.local_name(req.repo_id))
        done = threading.Event()

        def watch():
            while not done.wait(1.0):
                size = hub.dir_size(dest) if os.path.isdir(dest) else 0
                job.progress = min(0.999, size / max(1, p["bytes"]))
                job.message = f"{size / 2**30:.2f} / {p['bytes'] / 2**30:.2f} GB"

        threading.Thread(target=watch, daemon=True).start()
        try:
            hub.download(req.repo_id, models_dir(), p["files"], token=tok)
        finally:
            done.set()
        job.add_log("Download concluído")
        return {"name": hub.local_name(req.repo_id)}

    return start_job("download", req.repo_id, work).to_dict(False)


# ------------------------------------------------------------- Modelos
@app.get("/api/models")
def list_models():
    return {"models": hub.list_local(models_dir()), "loaded": STATE["name"]}


@app.delete("/api/models/{name}")
def delete_model(name: str):
    if STATE["name"] == name:
        unload()
    try:
        hub.delete_local(models_dir(), name)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))
    return {"ok": True}


class LoadReq(BaseModel):
    name: str
    device: str = "auto"
    dtype: str = "auto"
    prefetch: bool = True
    ram_cache_gb: float = 0.0
    resident_below_mb: float = 300.0  # componentes menores que isso não são paginados


@app.post("/api/load")
def load(req: LoadReq):
    path = os.path.join(models_dir(), req.name)
    if not os.path.isdir(path):
        raise HTTPException(404, "modelo não encontrado")
    if STATE["loading"]:
        raise HTTPException(409, "já existe um modelo carregando")

    def work(job: Job):
        from paged.loaders import load_any
        with GPU_LOCK:
            STATE["loading"] = True
            STATE["job"] = job
            try:
                _unload_locked()
                job.add_log(f"Montando {req.name} vazio (dispositivo meta)...")
                m = load_any(path, req.device, req.dtype, on_page_event, req.prefetch, req.ram_cache_gb,
                             req.resident_below_mb)
                STATE["model"], STATE["name"] = m, req.name
                job.add_log(f"Pronto: {m.kind} em {m.device} ({str(m.dtype).replace('torch.', '')})")
                return loaded_info()
            finally:
                STATE["loading"] = False
                STATE["job"] = None

    return start_job("load", req.name, work).to_dict(False)


def _unload_locked():
    m = STATE.get("model")
    if m is not None:
        m.close()
    STATE["model"], STATE["name"] = None, None
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.post("/api/unload")
def unload():
    with GPU_LOCK:
        _unload_locked()
    return {"ok": True}


def loaded_info():
    m = STATE.get("model")
    if m is None:
        return None
    return {"name": STATE["name"], "kind": m.kind, "device": str(m.device),
            "dtype": str(m.dtype).replace("torch.", ""),
            "accepts_image": bool(getattr(m, "accepts_image", lambda: False)()),
            "pagers": m.stats()}


@app.get("/api/loaded")
def loaded():
    return loaded_info()


# ------------------------------------------------------------- Geração
class GenReq(BaseModel):
    prompt: str
    system: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    negative_prompt: str = ""
    steps: int = 25
    guidance: float = 5.0
    width: int = 0
    height: int = 0
    frames: int = 0
    fps: int = 16
    seed: int = -1
    image: Optional[str] = None  # data URL


@app.post("/api/generate")
def generate(req: GenReq):
    m = STATE.get("model")
    if m is None:
        raise HTTPException(400, "carregue um modelo primeiro")

    def work(job: Job):
        import torch
        with GPU_LOCK:
            STATE["job"] = job
            t0 = time.time()
            try:
                if m.kind == "text":
                    from transformers import StoppingCriteria, StoppingCriteriaList

                    class Stop(StoppingCriteria):
                        def __call__(self, *_a, **_k):
                            return job.cancel

                    def on_text(txt):
                        job.text = txt

                    out = m.generate(req.prompt, on_text, max_new_tokens=req.max_new_tokens,
                                     temperature=req.temperature, top_p=req.top_p,
                                     system=req.system,
                                     stopping_criteria=StoppingCriteriaList([Stop()]))
                    job.add_log(f"Texto gerado em {time.time() - t0:.1f}s")
                    return {"text": out}
                image = _decode_image(req.image) if req.image else None

                def on_step(step, total):
                    if job.cancel:
                        raise Cancelled()
                    job.progress = step / max(1, total)
                    job.add_log(f"Passo {step}/{total}")

                fname = time.strftime("%Y%m%d-%H%M%S-") + job.id
                out = m.generate(os.path.join(outputs_dir(), fname), on_step=on_step,
                                 image=image, **req.model_dump(exclude={"image"}))
                job.add_log(f"Arquivo salvo em {time.time() - t0:.1f}s")
                return {"file": os.path.basename(out), "type": "video" if out.endswith(".mp4") else "image"}
            except BaseException:
                m.reset()  # descarta páginas que ficaram pela metade
                raise
            finally:
                STATE["job"] = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return start_job("generate", req.prompt[:60], work).to_dict(False)


def _decode_image(data_url: str):
    from PIL import Image
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


# --------------------------------------------------------------- Tarefas
@app.get("/api/jobs")
def jobs():
    return [j.to_dict(False) for j in sorted(JOBS.values(), key=lambda j: -j.started)][:50]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = JOBS.get(job_id)
    if j is None:
        raise HTTPException(404, "tarefa não encontrada")
    d = j.to_dict()
    if STATE.get("model") is not None and j.kind in ("generate", "load"):
        d["pagers"] = STATE["model"].stats()
    return d


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    j = JOBS.get(job_id)
    if j is None:
        raise HTTPException(404, "tarefa não encontrada")
    j.cancel = True
    return {"ok": True}


def main():
    global HOME
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--home", default=HOME)
    a = ap.parse_args()
    HOME = os.path.expanduser(a.home)
    os.makedirs(HOME, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=outputs_dir()), name="outputs")
    print(f"IA Paginada: http://127.0.0.1:{a.port} (pasta: {HOME})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
