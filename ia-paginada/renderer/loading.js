const err = new URLSearchParams(location.search).get("error");
if (err) {
  document.getElementById("msg").textContent =
    "O motor Python não iniciou. Rode “npm run setup” na pasta ia-paginada e tente de novo.";
  const pre = document.getElementById("err");
  pre.hidden = false;
  pre.textContent = err;
  document.querySelector(".book").hidden = true;
}

// ---- instalação da primeira abertura (chamado pelo processo principal) ----
const lines = [];
window.setupStart = () => {
  document.getElementById("setup").hidden = false;
  document.getElementById("msg").textContent = "Instalando o motor de IA…";
};
window.setupStep = (n, total, title) => {
  document.getElementById("msg").textContent = `Passo ${n} de ${total}: ${title}`;
  document.getElementById("setupBar").style.width = `${Math.round(((n - 1) / total) * 100)}%`;
};
window.setupLog = (line) => {
  lines.push(line);
  if (lines.length > 400) lines.shift();
  const pre = document.getElementById("setupLog");
  pre.textContent = lines.join("\n");
  pre.scrollTop = pre.scrollHeight;
};
window.setupFail = (err) => {
  document.getElementById("msg").textContent = `A instalação falhou: ${err}`;
  document.querySelector(".book").hidden = true;
  const b = document.getElementById("retry");
  b.hidden = false;
  b.onclick = () => {
    location.search = "?retry=1"; // o processo principal detecta e recomeça
  };
};
window.setupDone = () => {
  document.getElementById("setupBar").style.width = "100%";
  document.getElementById("msg").textContent = "Pronto! Iniciando…";
};
