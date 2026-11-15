// Executa o Python do projeto com os argumentos dados.
import { spawnSync } from "node:child_process";
import { findPython, root } from "./python-path.mjs";

const r = spawnSync(findPython(), process.argv.slice(2), { stdio: "inherit", cwd: root });
process.exit(r.status ?? 1);
