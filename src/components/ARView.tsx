"use client";

import Image from "next/image";
import QRCode from "qrcode";
import { useEffect, useRef, useState } from "react";
import { PlateIcon } from "./PlateIcon";

type MV = HTMLElement & { canActivateAR: boolean; loaded: boolean; activateAR: () => Promise<void> };

type Props = {
  src: string;
  photo: string | null;
  alt: string;
  className?: string;
};

type Mode = "loading" | "ready" | "in-app" | "desktop" | "error";

let loader: Promise<unknown> | null = null;
function loadModelViewer() {
  loader ??= import("@google/model-viewer");
  return loader;
}

function isMobile() {
  return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && /Macintosh/.test(navigator.userAgent));
}

function isInAppBrowser() {
  return /Instagram|FBAN|FBAV|FB_IAB|Line\/|TikTok|Snapchat|WhatsApp|LinkedInApp|Twitter/i.test(navigator.userAgent);
}

/**
 * Foto do prato + botão "Ver na sua mesa" que abre o AR em tamanho real.
 * O modelo começa a baixar quando a página abre, para o AR abrir na hora do toque
 * (o iPhone só abre o AR se for direto no toque).
 */
export function ARView({ src, photo, alt, className = "" }: Props) {
  const [ready, setReady] = useState(false);
  const [mode, setMode] = useState<Mode>("loading");
  const [qr, setQr] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  const ref = useRef<MV | null>(null);

  // Carrega o model-viewer logo que a página abre
  useEffect(() => {
    let cancelled = false;
    loadModelViewer()
      .then(() => !cancelled && setReady(true))
      .catch(() => !cancelled && setMode("error"));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || !ready) return;
    const decide = () => {
      if (el.canActivateAR) setMode("ready");
      else if (isMobile()) setMode(isInAppBrowser() ? "in-app" : "error");
      else setMode("desktop");
    };
    const onError = () => setMode("error");
    el.addEventListener("load", decide);
    el.addEventListener("error", onError);
    if (el.loaded) decide();
    return () => {
      el.removeEventListener("load", decide);
      el.removeEventListener("error", onError);
    };
  }, [ready]);

  useEffect(() => {
    if (mode !== "desktop") return;
    QRCode.toString(window.location.href, { type: "svg", margin: 1, color: { dark: "#12100e", light: "#ffffff" } }).then(setQr);
  }, [mode]);

  async function open() {
    if (!ref.current) return;
    setOpening(true);
    try {
      await ref.current.activateAR();
    } finally {
      setTimeout(() => setOpening(false), 1500);
    }
  }

  return (
    <div className={className}>
      <div className="relative aspect-square w-full overflow-hidden rounded-[var(--radius)] bg-surface sm:aspect-[4/3]">
        {/* Modelo fica atrás da foto: só existe para o AR */}
        {ready && (
          <model-viewer
            ref={ref as React.Ref<HTMLElement>}
            src={src}
            alt={alt}
            ar
            ar-modes="webxr scene-viewer quick-look"
            ar-scale="fixed"
            ar-placement="floor"
            loading="eager"
            aria-hidden="true"
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", opacity: 0, pointerEvents: "none" }}
          >
            <span slot="ar-button" hidden />
            <span slot="progress-bar" hidden />
          </model-viewer>
        )}
        {photo ? (
          <Image src={photo} alt={alt} fill priority sizes="(max-width: 768px) 100vw, 768px" className="object-cover" />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <PlateIcon size={72} />
          </div>
        )}
      </div>

      <div className="mt-4" aria-live="polite">
        {(mode === "loading" || mode === "ready") && (
          <button type="button" className="btn btn-primary w-full !min-h-14 text-base" disabled={mode === "loading" || opening} onClick={open}>
            <ARIcon />
            {mode === "loading" ? "Preparando o prato…" : opening ? "Abrindo a câmera…" : "Ver na sua mesa"}
          </button>
        )}

        {mode === "ready" && (
          <p className="mt-2 text-center text-xs text-muted">Aponte a câmera para a mesa. O prato aparece no tamanho real.</p>
        )}

        {mode === "in-app" && (
          <Notice>
            Para ver o prato na mesa, abra esta página no navegador do celular: toque em <b>⋯</b> e escolha{" "}
            <b>Abrir no navegador</b> (Safari ou Chrome).
          </Notice>
        )}

        {mode === "error" && (
          <Notice>Este celular não abre a realidade aumentada. Peça ao garçom para ver o prato.</Notice>
        )}

        {mode === "desktop" && (
          <div className="flex items-center gap-4 rounded-[var(--radius)] border border-line p-4">
            {qr && <div className="w-28 shrink-0 overflow-hidden rounded-lg bg-white p-1" dangerouslySetInnerHTML={{ __html: qr }} />}
            <p className="text-sm leading-relaxed text-muted">
              Aponte a câmera do celular para o código e toque em <b className="text-ink">Ver na sua mesa</b> para ver o prato no tamanho real.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function Notice({ children }: { children: React.ReactNode }) {
  return <p className="rounded-[var(--radius)] border border-line p-4 text-sm leading-relaxed text-muted">{children}</p>;
}

function ARIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
      <path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" />
      <path d="m12 7-4 2.3v4.6l4 2.3 4-2.3V9.3L12 7Z" />
    </svg>
  );
}
