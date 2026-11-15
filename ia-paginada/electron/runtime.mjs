// Onde está o Python/back-end e a instalação da primeira abertura.
//
// * Em desenvolvimento: backend/.venv (npm run setup) ou o Python do sistema.
// * Instalado pelo .exe: Python embutido em resources/python; na primeira vez
//   instala pip, PyTorch (CUDA se houver placa NVIDIA, senão CPU) e o resto.
import { spawn, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { findPython, root } from "../scripts/python-path.mjs";

const CUDA_INDEX = process.env.IAPAG_CUDA_INDEX || "https://download.pytorch.org/whl/cu126";
const CPU_INDEX = "https://download.pytorch.org/whl/cpu";

export function resolveRuntime(app) {
  const forced = process.env.IAPAG_BUNDLED_PYTHON; // para testar o fluxo do instalador
  if (app.isPackaged || forced) {
    const res = app.isPackaged ? process.resourcesPath : root;
    const pyDir = forced ? path.dirname(path.dirname(forced)) : path.join(res, "python");
    const python = forced || (process.platform === "win32"
      ? path.join(pyDir, "python.exe")
      : path.join(pyDir, "bin", "python3"));
    const backend = path.join(res, "backend");
    return { packaged: true, python, pyDir, backend, marker: path.join(pyDir, ".iapag-setup.json") };
  }
  return { packaged: false, python: findPython(), backend: path.join(root, "backend") };
}

function requirementsHash(rt) {
  const req = fs.readFileSync(path.join(rt.backend, "requirements.txt"));
  return crypto.createHash("sha1").update(req).digest("hex").slice(0, 12);
}

export function needsSetup(rt) {
  if (!rt.packaged) return false;
  try {
    const m = JSON.parse(fs.readFileSync(rt.marker, "utf8"));
    return m.requirements !== requirementsHash(rt);
  } catch {
    return true;
  }
}

function hasNvidia() {
  try {
    const r = spawnSync("nvidia-smi", ["-L"], { timeout: 8000, encoding: "utf8" });
    return r.status === 0 && /GPU/.test(r.stdout || "");
  } catch {
    return false;
  }
}

function run(cmd, args, log) {
  return new Promise((resolve, reject) => {
    log(`> ${path.basename(cmd)} ${args.join(" ")}`);
    const p = spawn(cmd, args, { env: { ...process.env, PYTHONIOENCODING: "utf-8", PIP_DISABLE_PIP_VERSION_CHECK: "1" } });
    const onData = (b) => b.toString().split(/\r?\n/).filter(Boolean).forEach((l) => log(l));
    p.stdout.on("data", onData);
    p.stderr.on("data", onData);
    p.on("error", reject);
    p.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${path.basename(cmd)} saiu com código ${code}`))));
  });
}

/** Instala pip, PyTorch e dependências no Python embutido. `log(texto)`, `step(n, total, titulo)`. */
export async function runSetup(rt, log, step) {
  const py = rt.python;
  const pip = (...a) => run(py, ["-m", "pip", ...a], log);
  const total = 4;
  step(1, total, "Preparando o instalador de pacotes (pip)");
  const hasPip = spawnSync(py, ["-m", "pip", "--version"]).status === 0;
  if (!hasPip) await run(py, [path.join(rt.pyDir, "get-pip.py"), "--no-warn-script-location"], log);

  const gpu = hasNvidia();
  step(2, total, gpu ? "Instalando PyTorch para GPU NVIDIA (CUDA) — pode levar vários minutos"
    : "Instalando PyTorch (CPU)");
  try {
    await pip("install", "--no-warn-script-location", "torch", "--index-url", gpu ? CUDA_INDEX : CPU_INDEX);
  } catch (e) {
    if (!gpu) throw e;
    log("Falhou a versão CUDA; instalando a versão para CPU.");
    await pip("install", "--no-warn-script-location", "torch", "--index-url", CPU_INDEX);
  }

  step(3, total, "Instalando transformers, diffusers e o restante");
  await pip("install", "--no-warn-script-location", "-r", path.join(rt.backend, "requirements.txt"));

  step(4, total, "Conferindo a instalação");
  await run(py, ["-c", "import torch, transformers, diffusers, fastapi; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"], log);
  fs.writeFileSync(rt.marker, JSON.stringify({ requirements: requirementsHash(rt), gpu, date: new Date().toISOString() }));
}
