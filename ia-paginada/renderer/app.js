// Interface da IA Paginada (sem frameworks).
const qs = new URLSearchParams(location.search);
const API = `http://127.0.0.1:${qs.get("port")}`;
const KEY = qs.get("key") || "";
const $ = (id) => document.getElementById(id);

const state = { loaded: null, genJob: null, downloads: {} };

// ------------------------------------------------------------------ util
async function api(path, opts = {}) {
  const r = await fetch(API + path, {
    ...opts,
    headers: { "Content-Type": "application/json", "X-IAPAG-Key": KEY, ...(opts.headers || {}) },
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}
const post = (p, body) => api(p, { method: "POST", body: JSON.stringify(body || {}) });

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid);
  return e;
}
const gb = (b) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(2)} GB` : `${(b / 2 ** 20).toFixed(1)} MB`);
const num = (n) => Intl.NumberFormat("pt-BR", { notation: "compact" }).format(n);
const kindLabel = { text: "📝 Texto", image: "🖼️ Imagem", video: "🎬 Vídeo" };
const token = () => $("hfToken").value.trim();

function toast(msg, bad = false) {
  const t = el("div", { class: `toast ${bad ? "bad" : ""}` }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), 5000);
}

async function pollJob(id, onUpdate) {
  for (;;) {
    const j = await api(`/api/jobs/${id}`);
    onUpdate(j);
    if (j.status !== "running") return j;
    await new Promise((r) => setTimeout(r, 600));
  }
}

// ------------------------------------------------------------------ abas
document.querySelectorAll("nav button").forEach((b) =>
  b.addEventListener("click", () => showTab(b.dataset.tab)),
);
function showTab(name) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  if (name === "models") refreshModels();
  if (name === "system") refreshSystem();
}

// ------------------------------------------------------------------- hub
$("searchForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const box = $("results");
  box.replaceChildren(el("p", { class: "muted" }, "Buscando…"));
  try {
    const params = new URLSearchParams({ q: $("q").value, kind: $("kind").value, token: token() });
    const list = await api(`/api/hub/search?${params}`);
    box.replaceChildren(...(list.length ? list.map(resultCard) : [el("p", { class: "muted" }, "Nada encontrado.")]));
  } catch (e) {
    box.replaceChildren(el("p", { class: "error" }, e.message));
  }
});

function resultCard(m) {
  return el("div", { class: "card result" },
    el("b", {}, m.id),
    el("div", { class: "meta" },
      el("span", {}, m.task || "—"), el("span", {}, `⬇ ${num(m.downloads)}`), el("span", {}, `♥ ${num(m.likes)}`),
      m.gated ? el("span", { class: "warn" }, "🔒 restrito") : null),
    el("div", { class: "row small-gap" },
      el("button", { onclick: () => showPlan(m.id) }, "Ver arquivos e baixar"),
      el("a", { href: `https://huggingface.co/${m.id}`, target: "_blank" }, "abrir no site")),
  );
}

$("repoDirectBtn").addEventListener("click", () => {
  const id = $("repoDirect").value.trim();
  if (id) showPlan(id);
});

$("spaceBtn").addEventListener("click", async () => {
  const id = $("spaceId").value.trim().replace(/^https?:\/\/huggingface\.co\/spaces\//, "");
  if (!id) return;
  const box = $("results");
  box.replaceChildren(el("p", { class: "muted" }, `Lendo o Space ${id}…`));
  try {
    const r = await api(`/api/hub/space?${new URLSearchParams({ id, token: token() })}`);
    if (!r.models.length) {
      box.replaceChildren(el("p", { class: "muted" }, "Esse Space não declara nenhum modelo do Hub."));
      return;
    }
    box.replaceChildren(...r.models.map((m) => resultCard({ id: m, downloads: 0, likes: 0, task: `usado por ${id}` })));
  } catch (e) {
    box.replaceChildren(el("p", { class: "error" }, e.message));
  }
});

async function showPlan(repo) {
  const box = $("plan");
  box.hidden = false;
  box.replaceChildren(el("p", { class: "muted" }, `Lendo arquivos de ${repo}…`));
  box.scrollIntoView({ behavior: "smooth" });
  try {
    const p = await api(`/api/hub/plan?${new URLSearchParams({ repo, fp16: $("fp16").checked, token: token() })}`);
    const files = el("ul", { class: "files" }, ...p.files.slice(0, 60).map((f) => el("li", {}, f)));
    if (p.files.length > 60) files.append(el("li", { class: "muted" }, `… e mais ${p.files.length - 60}`));
    box.replaceChildren(
      el("div", { class: "row space" },
        el("div", {}, el("h3", {}, repo),
          el("p", { class: "muted" }, `${kindLabel[p.kind] || "tipo desconhecido"} · ${p.files.length} arquivos · ${gb(p.bytes)}`)),
        el("div", { class: "row small-gap" },
          el("button", { class: "primary", onclick: () => startDownload(repo) }, `Baixar ${gb(p.bytes)}`),
          el("button", { onclick: () => (box.hidden = true) }, "Fechar"))),
      p.gated ? el("p", { class: "warn" }, "Modelo restrito: aceite os termos no site e informe seu token.") : null,
      p.remote_code ? el("p", { class: "warn" },
        "Este modelo traz código Python próprio: para carregar, marque “Confiar no código do modelo” em Meus modelos.") : null,
      p.quantization === "mxfp4" ? el("p", { class: "muted" },
        "Pesos em MXFP4 (4 bits): o app multiplica direto nos bytes comprimidos, especialista por especialista.") : null,
      state.diskFree && p.bytes > state.diskFree ? el("p", { class: "error" },
        `Não cabe no disco: precisa de ${gb(p.bytes)} e há ${gb(state.diskFree)} livres em ${state.home}.`) : null,
      el("details", {}, el("summary", {}, "Arquivos"), files),
    );
  } catch (e) {
    box.replaceChildren(el("p", { class: "error" }, e.message), el("button", { onclick: () => (box.hidden = true) }, "Fechar"));
  }
}

async function startDownload(repo) {
  try {
    const job = await post("/api/download", { repo_id: repo, fp16: $("fp16").checked, token: token() || null });
    state.downloads[job.id] = job;
    renderDownloads();
    toast(`Baixando ${repo}…`);
    const done = await pollJob(job.id, (j) => {
      state.downloads[job.id] = j;
      renderDownloads();
    });
    if (done.status === "done") toast(`✅ ${repo} baixado`);
    else toast(`Falha ao baixar ${repo}: ${done.error || done.status}`, true);
    setTimeout(() => {
      delete state.downloads[job.id];
      renderDownloads();
    }, 8000);
    refreshModels();
  } catch (e) {
    toast(e.message, true);
  }
}

function renderDownloads() {
  $("dlMini").replaceChildren(...Object.values(state.downloads).map((j) =>
    el("div", { class: "dl" },
      el("small", {}, j.title), progressBar(j.progress),
      el("small", { class: "muted" }, j.status === "running" ? j.message || "…" : j.status))));
}

function progressBar(frac) {
  const inner = el("div");
  inner.style.width = `${Math.round((frac || 0) * 100)}%`;
  return el("div", { class: "bar" }, inner);
}

// ---------------------------------------------------------------- modelos
async function refreshModels() {
  try {
    const { models, loaded } = await api("/api/models");
    const list = $("modelList");
    if (!models.length) {
      list.replaceChildren(el("p", { class: "muted" }, "Nenhum modelo baixado ainda. Use “Baixar modelos”."));
      return;
    }
    list.replaceChildren(...models.map((m) =>
      el("div", { class: `card model ${m.name === loaded ? "is-loaded" : ""}` },
        el("div", {}, el("b", {}, m.repo_id),
          el("div", { class: "meta" }, el("span", {}, kindLabel[m.kind] || "❓ desconhecido"), el("span", {}, gb(m.bytes)),
            m.complete ? null : el("span", { class: "warn" }, "download incompleto"))),
        el("div", { class: "row small-gap" },
          m.name === loaded
            ? el("button", { onclick: unloadModel }, "Descarregar")
            : el("button", { class: "primary", disabled: !m.kind, onclick: () => loadModel(m.name) }, "Carregar"),
          el("button", { class: "danger", onclick: () => deleteModel(m) }, "Apagar")))));
  } catch (e) {
    toast(e.message, true);
  }
}

async function loadModel(name) {
  try {
    const job = await post("/api/load", {
      name, device: $("device").value, dtype: $("dtype").value,
      prefetch: Number($("prefetch").value) || 0, pack: $("pack").value,
      min_page_mb: Number($("minPageMb").value) || 0,
      trust_remote_code: $("trustCode").checked,
      expert_cache_gb: Number($("expertCache").value) || 0,
      resident_below_mb: Number($("residentMb").value) || 0,
    });
    showTab("run");
    $("runInfo").textContent = `Carregando ${name}…`;
    $("output").replaceChildren();
    const done = await pollJob(job.id, (j) => renderJobOnPress(j, true));
    if (done.status === "done") {
      setLoaded(done.result);
      toast(`✅ ${name} pronto`);
    } else {
      $("runInfo").textContent = `Erro ao carregar: ${done.error}`;
      toast(done.error || "erro", true);
    }
  } catch (e) {
    toast(e.message, true);
  }
}

async function unloadModel() {
  await post("/api/unload");
  setLoaded(null);
  refreshModels();
}

async function deleteModel(m) {
  if (!confirm(`Apagar ${m.repo_id} (${gb(m.bytes)}) do disco?`)) return;
  try {
    await api(`/api/models/${encodeURIComponent(m.name)}`, { method: "DELETE" });
    if (state.loaded?.name === m.name) setLoaded(null);
    refreshModels();
  } catch (e) {
    toast(e.message, true);
  }
}

function setLoaded(info) {
  state.loaded = info;
  $("loadedBadge").textContent = info ? `${kindLabel[info.kind]} · ${info.name}` : "Nenhum modelo carregado";
  $("loadedBadge").classList.toggle("on", !!info);
  $("runInfo").textContent = info
    ? `${info.name} — ${kindLabel[info.kind]} em ${info.device} (${info.dtype})`
    : "Carregue um modelo em “Meus modelos”.";
  const kind = info?.kind;
  document.querySelectorAll("[data-for]").forEach((n) => {
    n.hidden = !kind || !n.dataset.for.split(" ").includes(kind);
  });
  const img = document.querySelector("[data-image]");
  if (img) img.hidden = !info?.accepts_image;
  const sp = document.querySelector("[data-speed]");
  if (sp) sp.hidden = !info?.supports_speed;
  speedHint();
  $("genBtn").disabled = !info;
  renderPress(info?.pagers || []);
}

// ------------------------------------------------------------- impressora
function renderPress(pagers, msg = "") {
  $("pressMsg").textContent = msg;
  $("press").replaceChildren(...pagers.filter((p) => p.pages > 0 || p.resident_bytes > 0).map((p) => {
    const strip = el("div", { class: "pages" });
    for (let i = 0; i < p.pages; i++) {
      strip.append(el("i", { class: i === p.current ? "now" : "", title: `página ${i + 1}` }));
    }
    if (!p.pages) strip.append(el("span", { class: "muted" }, "inteiro na memória (pequeno)"));
    return el("div", { class: "pager" },
      el("div", { class: "pager-head" }, el("b", {}, p.label),
        el("small", { class: "muted" },
          `${p.pages} páginas · ${p.store}${p.zero_copy ? " (sem cópia)" : ""} · ${p.loads} trocas · ` +
          `${gb(p.bytes_read)} passados · residente ${gb(p.resident_bytes)}` +
          (p.loads ? ` · troca média ${p.avg_ms} ms` : ""))),
      strip);
  }));
}

function renderJobOnPress(j, loading = false) {
  if (j.pagers) renderPress(j.pagers, j.message || "");
  else $("pressMsg").textContent = j.message || "";
  $("genBar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
  $("genLog").textContent = (j.log || []).join("\n");
  if (loading) $("runInfo").textContent = j.log?.at(-1) || "Carregando…";
}

// ------------------------------------------------------------------ gerar
async function readImage() {
  const f = $("inputImage").files?.[0];
  if (!f || $("inputImage").closest("[data-image]").hidden) return null;
  return new Promise((res) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.readAsDataURL(f);
  });
}

$("genForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!state.loaded) return toast("Carregue um modelo primeiro", true);
  const prompt = $("prompt").value.trim();
  if (!prompt) return toast("Escreva um prompt", true);
  const body = {
    prompt, system: $("system").value, max_new_tokens: +$("maxTokens").value,
    temperature: +$("temperature").value, top_p: +$("topP").value,
    negative_prompt: $("negative").value, steps: +$("steps").value, guidance: +$("guidance").value,
    width: +$("width").value, height: +$("height").value, frames: +$("frames").value,
    fps: +$("fps").value, seed: +$("seed").value, image: await readImage(),
    speed: Math.max(1, +$("speed").value || 1),
  };
  $("genBtn").disabled = true;
  $("cancelBtn").disabled = false;
  const out = $("output");
  const textBox = el("div", { class: "text-out" });
  out.replaceChildren(state.loaded.kind === "text" ? textBox : el("p", { class: "muted" }, "Gerando…"));
  try {
    const job = await post("/api/generate", body);
    state.genJob = job.id;
    const done = await pollJob(job.id, (j) => {
      renderJobOnPress(j);
      if (state.loaded?.kind === "text") textBox.textContent = j.text || "…";
    });
    if (done.status === "done" && done.result?.file) {
      const url = `${API}/outputs/${encodeURIComponent(done.result.file)}`;
      const media = done.result.type === "video"
        ? el("video", { src: url, controls: true, autoplay: true, loop: true })
        : el("img", { src: url, alt: prompt });
      const r = done.result;
      const extra = r.reused ? ` · velocidade ${r.speed}× (${r.computed} calculadas, ${r.reused} reaproveitadas)` : "";
      out.replaceChildren(media, el("p", { class: "muted" }, `Salvo em outputs/${r.file} · ${r.seconds}s${extra}`));
    } else if (done.status === "done") {
      textBox.textContent = done.result?.text ?? done.text;
    } else if (done.status === "cancelled") {
      out.append(el("p", { class: "warn" }, "Cancelado."));
    } else {
      out.append(el("p", { class: "error" }, done.error || "erro"));
    }
  } catch (e) {
    out.replaceChildren(el("p", { class: "error" }, e.message));
  } finally {
    state.genJob = null;
    $("genBtn").disabled = false;
    $("cancelBtn").disabled = true;
    const info = await api("/api/loaded").catch(() => null);
    if (info) renderPress(info.pagers);
  }
});

// ------------------------------------------------------------- velocidade
function speedHint() {
  // Mesmo cálculo do backend (paged/speed.py: split_speed)
  const v = Math.max(1, +$("speed").value || 1);
  const steps = Math.max(1, +$("steps").value || 1);
  const L = state.loaded || {};
  const [dw, dh] = L.default_size || [0, 0];
  const W = +$("width").value || dw, H = +$("height").value || dh;
  const F = L.kind === "video" ? (+$("frames").value || L.default_frames || 0) : 0;
  let computed = steps;
  if (L.can_skip_steps !== false && v > 1 && steps > 3) {
    computed = 0; let acc = 0;
    for (let s = 0; s < steps; s++) {
      if (s < 2 || s === steps - 1) { computed++; continue; }
      acc += 1 / v;
      if (acc >= 1 - 1e-9) { computed++; acc -= 1; }
    }
  }
  const stepGain = steps / computed;
  let rest = v / stepGain, w = W, h = H, f = F, area = 1, fg = 1;
  const side = (x) => Math.max(64, Math.floor(x / 16) * 16);
  if (rest > 1.05 && W && H) {
    const s = Math.sqrt(rest); w = side(W / s); h = side(H / s);
    area = (W * H) / (w * h); rest /= area;
  }
  if (rest > 1.05 && F) { f = Math.max(5, Math.floor(F / rest)); fg = F / f; }
  const gain = stepGain * area * fg;
  $("speedVal").textContent = `${v}×`;
  if (v <= 1) { $("speedHint").textContent = "Qualidade máxima: todas as páginas em todos os passos."; return; }
  const parts = [`${computed}/${steps} passos calculados`];
  if (area > 1) parts.push(`calcula em ${w}×${h} e amplia para ${W}×${H}`);
  if (fg > 1) parts.push(`${f} de ${F} quadros calculados (resto interpolado)`);
  let txt = `≈ ${gain.toFixed(1)}× mais rápido: ${parts.join(" · ")}.`;
  if (v > gain * 1.05) txt += " Piso físico atingido: acima disso não fica mais rápido.";
  else if (v > 4) txt += " Qualidade cai bastante.";
  $("speedHint").textContent = txt;
}
for (const id of ["steps", "width", "height", "frames"]) $(id).addEventListener("input", () => speedHint());
$("speed").addEventListener("input", () => {
  $("speedRange").value = Math.min(10, +$("speed").value || 1);
  speedHint();
});
$("speedRange").addEventListener("input", () => {
  $("speed").value = $("speedRange").value;
  speedHint();
});
speedHint();

$("cancelBtn").addEventListener("click", async () => {
  if (state.genJob) await post(`/api/jobs/${state.genJob}/cancel`);
});

// ---------------------------------------------------------------- sistema
async function refreshSystem() {
  try {
    const s = await api("/api/system");
    const rows = [
      ["Python", s.python], ["PyTorch", s.torch],
      ["GPU CUDA", s.cuda ? s.gpus.map((g) => `${g.name} (${gb(g.free)} livres de ${gb(g.total)})`).join(", ") : "não"],
      ["Apple MPS", s.mps ? "sim" : "não"],
      ["RAM", `${gb(s.ram_available)} livres de ${gb(s.ram_total)}`],
      ["Disco livre", gb(s.disk_free)], ["Pasta de dados", s.home],
    ];
    $("sysInfo").replaceChildren(el("table", {}, ...rows.map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, v)))));
    $("homeInfo").textContent = `Modelos ficam em ${s.home}/models`;
    state.diskFree = s.disk_free;
    state.home = s.home;
    const jobs = await api("/api/jobs");
    $("jobList").replaceChildren(...(jobs.length ? jobs.map((j) =>
      el("div", { class: "job" }, el("span", { class: `st ${j.status}` }, j.status), el("b", {}, j.kind), el("span", {}, j.title),
        j.error ? el("small", { class: "error" }, j.error) : null)) : [el("p", { class: "muted" }, "Nenhuma tarefa ainda.")]));
  } catch (e) {
    toast(e.message, true);
  }
}

// ------------------------------------------------------------------ início
(async () => {
  setLoaded(await api("/api/loaded").catch(() => null));
  refreshModels();
  refreshSystem();
})();
