"use client";

import { useEffect, useRef, useState } from "react";
import { PlateIcon } from "./PlateIcon";

type MV = HTMLElement & {
  canActivateAR: boolean;
  activateAR: () => Promise<void>;
  toBlob: (o?: { mimeType?: string; qualityArgument?: number; idealAspect?: boolean }) => Promise<Blob>;
};

type Props = {
  src: string;
  poster?: string | null;
  alt: string;
  /** Carrega o 3D só quando o cliente pede (padrão). */
  deferred?: boolean;
  className?: string;
  onReady?: (el: MV) => void;
};

let loader: Promise<unknown> | null = null;
function loadModelViewer() {
  loader ??= import("@google/model-viewer");
  return loader;
}

export function ModelViewer({ src, poster, alt, deferred = true, className = "", onReady }: Props) {
  const [active, setActive] = useState(!deferred);
  const [ready, setReady] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [canAR, setCanAR] = useState(false);
  const [error, setError] = useState(false);
  const ref = useRef<MV | null>(null);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    loadModelViewer()
      .then(() => !cancelled && setReady(true))
      .catch(() => !cancelled && setError(true));
    return () => {
      cancelled = true;
    };
  }, [active]);

  useEffect(() => {
    const el = ref.current;
    if (!el || !ready) return;
    const onLoad = () => {
      setLoaded(true);
      setCanAR(Boolean(el.canActivateAR));
      onReady?.(el);
    };
    const onError = () => setError(true);
    el.addEventListener("load", onLoad);
    el.addEventListener("error", onError);
    return () => {
      el.removeEventListener("load", onLoad);
      el.removeEventListener("error", onError);
    };
  }, [ready, onReady]);

  return (
    <div className={`relative overflow-hidden rounded-[var(--radius)] bg-surface ${className}`}>
      {ready ? (
        <model-viewer
          ref={ref as React.Ref<HTMLElement>}
          src={src}
          poster={poster ?? undefined}
          alt={alt}
          ar
          ar-modes="webxr scene-viewer quick-look"
          ar-scale="fixed"
          ar-placement="floor"
          camera-controls
          auto-rotate
          auto-rotate-delay={1500}
          rotation-per-second="18deg"
          camera-orbit="30deg 65deg auto"
          max-camera-orbit="auto 88deg auto"
          shadow-intensity="1"
          shadow-softness="0.8"
          exposure="1.05"
          tone-mapping="neutral"
          interaction-prompt="none"
          style={{ width: "100%", height: "100%", background: "transparent", ["--poster-color" as string]: "transparent" }}
        >
          {/* Esconde o botão de AR padrão; usamos o nosso abaixo */}
          <span slot="ar-button" hidden />
        </model-viewer>
      ) : poster ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={poster} alt={alt} className="h-full w-full object-cover" />
      ) : (
        <div className="flex h-full w-full items-center justify-center">
          <PlateIcon />
        </div>
      )}

      {!active && (
        <div className="absolute inset-x-0 bottom-0 flex justify-center p-4">
          <button type="button" className="btn btn-primary" onClick={() => setActive(true)}>
            <CubeIcon /> Ver em 3D
          </button>
        </div>
      )}

      {active && !loaded && !error && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center" aria-live="polite">
          <span className="rounded-full bg-bg/70 px-4 py-2 text-sm text-muted backdrop-blur">Carregando 3D…</span>
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-sm text-muted">
          Não deu para abrir o 3D neste aparelho.
        </div>
      )}

      {loaded && (
        <div className="absolute inset-x-0 bottom-0 flex justify-center gap-2 p-4">
          {canAR ? (
            <button type="button" className="btn btn-primary" onClick={() => ref.current?.activateAR()}>
              <ARIcon /> Ver na sua mesa
            </button>
          ) : (
            <span className="rounded-full bg-bg/70 px-4 py-2 text-xs text-muted backdrop-blur">
              Arraste para girar. AR abre no celular.
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function CubeIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
      <path d="M12 2 3 7v10l9 5 9-5V7l-9-5Z" />
      <path d="m3 7 9 5 9-5M12 12v10" />
    </svg>
  );
}

function ARIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
      <path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" />
      <path d="m12 7-4 2.3v4.6l4 2.3 4-2.3V9.3L12 7Z" />
    </svg>
  );
}

export type { MV as ModelViewerElement };
