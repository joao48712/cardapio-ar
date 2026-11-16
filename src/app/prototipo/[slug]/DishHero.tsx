"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { ARView } from "@/components/ARView";

/**
 * Topo do prato: a foto chega (vinda do card) e, assim que o modelo carrega,
 * vira o 3D girando. O ARView segue intacto: o botão "Ver na sua mesa" é o dele.
 */
export function DishHero({ model, photo, name }: { model: string; photo: string | null; name: string }) {
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = root.current;
    // Sem foto o ARView já abre no 3D; com movimento reduzido, fica na foto
    if (!el || !photo || matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    let timer = 0;
    const started = performance.now();
    const mo = new MutationObserver(tryOpen);

    function tryOpen() {
      const btn = el!.querySelector<HTMLButtonElement>('button[aria-pressed="false"]');
      if (!btn) return;
      mo.disconnect();
      // Espera a transição da foto terminar antes de trocar pelo 3D
      const wait = Math.max(0, 700 - (performance.now() - started));
      timer = window.setTimeout(() => btn.isConnected && btn.getAttribute("aria-pressed") === "false" && btn.click(), wait);
    }

    // Se a pessoa já tocou na área, a escolha é dela
    const cancel = () => {
      mo.disconnect();
      clearTimeout(timer);
    };

    mo.observe(el, { childList: true, subtree: true });
    tryOpen();
    el.addEventListener("pointerdown", cancel, { once: true });
    return () => {
      cancel();
      el.removeEventListener("pointerdown", cancel);
    };
  }, [photo]);

  return (
    <div ref={root}>
      <ARView src={model} photo={photo} alt={name} />
    </div>
  );
}

/** Voltar "de app": se a pessoa veio do cardápio, volta no histórico (mantém a rolagem) */
export function BackButton() {
  const router = useRouter();
  return (
    <Link
      href="/prototipo"
      aria-label="Voltar"
      onClick={(e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
        try {
          if (sessionStorage.getItem("proto-menu") === "1" && window.history.length > 1) {
            e.preventDefault();
            router.back();
          }
        } catch {}
      }}
      className="press flex size-11 items-center justify-center rounded-full border border-line bg-surface/70 text-ink hover:border-muted"
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
        <path d="m15 18-6-6 6-6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </Link>
  );
}
