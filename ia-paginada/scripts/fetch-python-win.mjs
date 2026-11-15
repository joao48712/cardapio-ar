// Baixa o Python oficial "embutível" do Windows (python.org) para build/python-win,
// liga o site-packages e inclui o get-pip.py. Vai dentro do instalador .exe;
// as bibliotecas (torch etc.) são instaladas na primeira vez que o app abre.
import fs from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import extract from "extract-zip";
import { root } from "./python-path.mjs";

const VERSION = process.env.IAPAG_PY_VERSION || "3.12.10";
const out = path.join(root, "build", "python-win");
const zipUrl = `https://www.python.org/ftp/python/${VERSION}/python-${VERSION}-embed-amd64.zip`;
const getPipUrl = "https://bootstrap.pypa.io/get-pip.py";

async function download(url, dest) {
  console.log(`baixando ${url}`);
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  await pipeline(r.body, fs.createWriteStream(dest));
}

if (fs.existsSync(path.join(out, "python.exe")) && fs.existsSync(path.join(out, ".ok"))) {
  console.log(`Python ${VERSION} já preparado em ${out}`);
  process.exit(0);
}
fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });
const zip = path.join(root, "build", `python-${VERSION}-embed-amd64.zip`);
if (!fs.existsSync(zip)) await download(zipUrl, zip);
await extract(zip, { dir: out });
await download(getPipUrl, path.join(out, "get-pip.py"));

// O ._pth do Python embutível desliga o "import site"; ligamos para o pip funcionar.
const pth = fs.readdirSync(out).find((f) => f.endsWith("._pth"));
const pthPath = path.join(out, pth);
let text = fs.readFileSync(pthPath, "utf8").replace(/^#\s*import site/m, "import site");
if (!/Lib\\site-packages/.test(text)) text += "\nLib\\site-packages\n";
fs.writeFileSync(pthPath, text);
fs.mkdirSync(path.join(out, "Lib", "site-packages"), { recursive: true });
fs.writeFileSync(path.join(out, ".ok"), VERSION);
console.log(`Python ${VERSION} pronto em ${out}`);
