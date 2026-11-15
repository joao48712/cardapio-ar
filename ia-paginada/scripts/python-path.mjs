// Descobre qual Python usar: IAPAG_PYTHON, o venv do projeto ou o do sistema.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const venvDir = path.join(root, "backend", ".venv");

export function venvPython() {
  return process.platform === "win32"
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");
}

export function findPython() {
  if (process.env.IAPAG_PYTHON) return process.env.IAPAG_PYTHON;
  const v = venvPython();
  if (fs.existsSync(v)) return v;
  return process.platform === "win32" ? "python" : "python3";
}

export { root };
