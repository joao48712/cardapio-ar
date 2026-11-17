"use client";

// Página de suporte: mostra o que o celular aceita de realidade aumentada e registra o que
// acontece em cada tentativa. Serve para descobrir por que o AR não abre em um aparelho.
import { useEffect, useRef, useState } from "react";

type MV = HTMLElement & { canActivateAR: boolean; activateAR: () => Promise<void> };

const MODES = ["scene-viewer", "webxr", "quick-look"] as const;

function intentUrl(file: string, pkg: string) {
  const back = `${location.origin}${location.pathname}#voltou-sem-abrir`;
  return (
    `intent://arvr.google.com/scene-viewer/1.2?mode=ar_preferred&resizable=false&file=${encodeURIComponent(file)}` +
    `#Intent;scheme=https;package=${pkg};action=android.intent.action.VIEW;` +
    `S.browser_fallback_url=${encodeURIComponent(back)};end;`
  );
}

export function View() {
  const [ready, setReady] = useState(false);
  const [info, setInfo] = useState<string[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [file, setFile] = useState("");
  const refs = useRef<Record<string, MV | null>>({});
  const t0 = useRef(0);

  const add = (line: string) =>
    setLog((l) => [...l, `${((performance.now() - t0.current) / 1000).toFixed(1)}s  ${line}`]);

  useEffect(() => {
    t0.current = performance.now();
    const model = new URLSearchParams(location.search).get("modelo") ?? `${location.origin}/seed/sobremesa.glb`;
    const onVis = () =>
      add(`página ${document.visibilityState === "hidden" ? "coberta por outro app" : "visível de novo"}`);
    const onHash = () => add(`endereço mudou para ${location.hash || "(sem marcador)"}`);
    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("hashchange", onHash);

    (async () => {
      setFile(model);
      const lines = [`Navegador: ${navigator.userAgent}`, `Modelo: ${model}`];
      const xr = (navigator as Navigator & { xr?: { isSessionSupported(m: string): Promise<boolean> } }).xr;
      try {
        const ok = xr ? await xr.isSessionSupported("immersive-ar") : null;
        lines.push(`WebXR (AR pelo navegador): ${ok === null ? "não existe" : ok ? "sim" : "não"}`);
      } catch (e) {
        lines.push(`WebXR: erro ${(e as Error).message}`);
      }
      let flag = "?";
      try {
        flag = localStorage.getItem("ar-sem-scene-viewer") ?? "não";
      } catch {}
      lines.push(`Marcado como "nativo falhou": ${flag}`);
      setInfo(lines);
      await import("@google/model-viewer");
      setReady(true);
    })();

    return () => {
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("hashchange", onHash);
    };
  }, []);

  async function tryMode(m: string) {
    const el = refs.current[m];
    if (!el) return;
    add(`[${m}] toque. Pode abrir: ${el.canActivateAR ? "sim" : "não"}`);
    try {
      await el.activateAR();
      add(`[${m}] chamada terminou sem erro`);
    } catch (e) {
      add(`[${m}] erro: ${(e as Error).message}`);
    }
  }

  const text = [...info, "", ...log].join("\n");

  function copy() {
    navigator.clipboard?.writeText(text).then(
      () => add("resultado copiado"),
      () => add("não deu para copiar; tire um print"),
    );
  }

  return (
    <main className="mx-auto w-full max-w-xl px-4 py-8">
      <h1 className="font-display text-3xl">Diagnóstico de AR</h1>
      <p className="mt-2 text-sm text-muted">
        Toque em cada botão, um por vez, e veja se a câmera abre. Depois toque em <b>Copiar resultado</b> e envie o
        texto.
      </p>

      <h2 className="mt-6 text-xs font-semibold tracking-[0.18em] text-muted uppercase">Pelo site</h2>
      <div className="mt-2 flex flex-col gap-2">
        {MODES.map((m) => (
          <button key={m} type="button" className="btn btn-ghost" disabled={!ready} onClick={() => tryMode(m)}>
            Abrir com {m}
          </button>
        ))}
      </div>

      <h2 className="mt-6 text-xs font-semibold tracking-[0.18em] text-muted uppercase">
        Link direto para o visualizador do Google
      </h2>
      {file && (
        <div className="mt-2 flex flex-col gap-2">
          <a
            className="btn btn-ghost"
            href={intentUrl(file, "com.google.android.googlequicksearchbox")}
            onClick={() => add("[link app Google] toque")}
          >
            Link direto (app Google)
          </a>
          <a className="btn btn-ghost" href={intentUrl(file, "com.google.ar.core")} onClick={() => add("[link ARCore] toque")}>
            Link direto (ARCore)
          </a>
        </div>
      )}

      <button type="button" className="btn btn-primary mt-6 w-full" onClick={copy}>
        Copiar resultado
      </button>
      <pre className="mt-4 overflow-x-auto rounded-lg bg-surface p-3 text-xs leading-relaxed whitespace-pre-wrap">{text}</pre>

      {ready &&
        file &&
        MODES.map((m) => (
          <model-viewer
            key={m}
            ref={(el: HTMLElement | null) => {
              refs.current[m] = el as MV | null;
            }}
            src={file}
            alt=""
            ar
            ar-modes={m}
            ar-scale="fixed"
            loading="eager"
            aria-hidden="true"
            style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none" }}
          >
            <span slot="ar-button" hidden />
          </model-viewer>
        ))}
    </main>
  );
}
