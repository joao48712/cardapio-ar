// Cria backend/.venv e instala as dependências Python.
//   npm run setup              -> torch padrão do PyPI
//   npm run setup -- --cpu     -> torch só CPU (download menor)
//   npm run setup -- --cuda cu128
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { root, venvDir, venvPython } from "./python-path.mjs";

const args = process.argv.slice(2);
const sysPython = process.env.IAPAG_PYTHON || (process.platform === "win32" ? "python" : "python3");

function run(cmd, cmdArgs) {
  console.log(`> ${cmd} ${cmdArgs.join(" ")}`);
  const r = spawnSync(cmd, cmdArgs, { stdio: "inherit", cwd: root });
  if (r.status !== 0) process.exit(r.status ?? 1);
}

if (!fs.existsSync(venvPython())) run(sysPython, ["-m", "venv", venvDir]);
const py = venvPython();
run(py, ["-m", "pip", "install", "--upgrade", "pip"]);

const torchArgs = ["-m", "pip", "install", "torch"];
const cudaIdx = args.indexOf("--cuda");
if (args.includes("--cpu")) torchArgs.push("--index-url", "https://download.pytorch.org/whl/cpu");
else if (cudaIdx >= 0 && args[cudaIdx + 1])
  torchArgs.push("--index-url", `https://download.pytorch.org/whl/${args[cudaIdx + 1]}`);
run(py, torchArgs);
run(py, ["-m", "pip", "install", "-r", path.join("backend", "requirements.txt"), "pytest"]);
console.log("\nPronto! Agora rode: npm start");
