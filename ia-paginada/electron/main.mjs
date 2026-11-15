// Processo principal do Electron: sobe o backend Python e abre a interface.
import { app, BrowserWindow, dialog, shell } from "electron";
import { spawn } from "node:child_process";
import crypto from "node:crypto";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { root } from "../scripts/python-path.mjs";
import { needsSetup, resolveRuntime, runSetup } from "./runtime.mjs";

const HOME = process.env.IAPAG_HOME || path.join(os.homedir(), "IA-Paginada");
const SECRET = crypto.randomBytes(24).toString("hex");
let backend = null;
let backendLog = "";
let win = null;
let rt = null;

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function startBackend(port) {
  const python = rt.python;
  const script = path.join(rt.backend, "server.py");
  backend = spawn(python, ["-u", script, "--port", String(port), "--home", HOME], {
    cwd: rt.backend,
    env: { ...process.env, IAPAG_SECRET: SECRET, IAPAG_HOME: HOME, PYTHONIOENCODING: "utf-8" },
  });
  const keep = (buf) => {
    const s = buf.toString();
    process.stdout.write(`[python] ${s}`);
    backendLog = (backendLog + s).slice(-8000);
  };
  backend.stdout.on("data", keep);
  backend.stderr.on("data", keep);
  backend.on("error", (err) => {
    backendLog += `\nNão consegui iniciar "${python}": ${err.message}`;
  });
  backend.on("exit", (code) => {
    backend = null;
    if (!app.isQuitting && win && !win.isDestroyed()) {
      dialog.showErrorBox("Backend Python parou", `Código ${code}.\n\n${backendLog.slice(-3000)}`);
    }
  });
}

async function waitHealthy(port, ms = 90_000) {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    if (!backend) break;
    try {
      const r = await fetch(`http://127.0.0.1:${port}/api/health`);
      if (r.ok) return true;
    } catch {
      /* ainda subindo */
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  return false;
}

async function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    title: "IA Paginada",
    backgroundColor: "#0f1115",
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("did-navigate", (_e, url) => {
    if (url.includes("retry=1")) bootstrap();
  });
  await win.loadFile(path.join(root, "renderer", "loading.html"));
  await bootstrap();
}

let booting = false;
async function bootstrap() {
  if (booting) return;
  booting = true;
  try {
    await boot();
  } finally {
    booting = false;
  }
}

async function boot() {
  rt = resolveRuntime(app);
  if (needsSetup(rt)) {
    // primeira abertura depois de instalar: baixa PyTorch e o restante
    const js = (fn, ...a) => win && !win.isDestroyed() &&
      win.webContents.executeJavaScript(`window.${fn} && window.${fn}(...${JSON.stringify(a)})`).catch(() => {});
    js("setupStart");
    try {
      await runSetup(rt, (line) => js("setupLog", line), (n, total, title) => js("setupStep", n, total, title));
    } catch (e) {
      js("setupFail", String(e.message || e));
      return;
    }
    js("setupDone");
  }

  const port = await freePort();
  startBackend(port);
  const ok = await waitHealthy(port);
  if (!ok) {
    await win.loadFile(path.join(root, "renderer", "loading.html"), {
      query: { error: backendLog.slice(-4000) || "O backend não respondeu." },
    });
    return;
  }
  await win.loadFile(path.join(root, "renderer", "index.html"), {
    query: { port: String(port), key: SECRET },
  });
}

app.whenReady().then(createWindow);
app.on("before-quit", () => {
  app.isQuitting = true;
  if (backend) backend.kill();
});
app.on("window-all-closed", () => app.quit());
