"use client";

import Image from "next/image";
import Link from "next/link";
import { ViewTransition, useEffect, useRef, useState } from "react";
import { formatPrice } from "@/lib/format";
import type { ProtoDish } from "./data";

type MV = HTMLElement & { cameraOrbit: string; loaded: boolean };

let loader: Promise<unknown> | null = null;
const loadModelViewer = () => (loader ??= import("@google/model-viewer"));

/**
 * Destaques do chef. Só o card do centro carrega o 3D (um modelo por vez: leve no Android),
 * e o prato gira conforme a pessoa rola, de lado ou para baixo.
 */
export function Featured({ dishes }: { dishes: ProtoDish[] }) {
  const track = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);
  const [mvReady, setMvReady] = useState(false);

  useEffect(() => {
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    // Espera a página assentar antes de baixar o 3D
    const id = setTimeout(() => loadModelViewer().then(() => setMvReady(true)), 600);
    return () => clearTimeout(id);
  }, []);

  // Card ativo = o mais perto do centro do carrossel
  useEffect(() => {
    const el = track.current;
    if (!el) return;
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const center = el.scrollLeft + el.clientWidth / 2;
        let best = 0;
        let dist = Infinity;
        Array.from(el.children).forEach((c, i) => {
          const card = c as HTMLElement;
          const d = Math.abs(card.offsetLeft + card.clientWidth / 2 - center);
          if (d < dist) {
            dist = d;
            best = i;
          }
        });
        setActive(best);
      });
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <section aria-labelledby="h-destaques" className="pt-6">
      <div className="mx-auto max-w-3xl px-4">
        <p className="ornament text-xs tracking-[0.3em] uppercase">
          <span id="h-destaques">Destaques do chef</span>
        </p>
      </div>
      <div
        ref={track}
        className="no-scrollbar mt-4 flex snap-x snap-mandatory gap-4 overflow-x-auto scroll-smooth px-[11vw] pb-2 sm:px-[calc(50vw-170px)]"
      >
        {dishes.map((d, i) => (
          <FeaturedCard key={d.slug} dish={d} active={i === active} mvReady={mvReady} track={track} />
        ))}
      </div>
      <div className="mt-3 flex justify-center gap-1.5" aria-hidden>
        {dishes.map((d, i) => (
          <span
            key={d.slug}
            className="h-1 rounded-full bg-accent transition-[width,opacity] duration-300"
            style={{ width: i === active ? 18 : 6, opacity: i === active ? 1 : 0.35 }}
          />
        ))}
      </div>
    </section>
  );
}

function FeaturedCard({
  dish,
  active,
  mvReady,
  track,
}: {
  dish: ProtoDish;
  active: boolean;
  mvReady: boolean;
  track: React.RefObject<HTMLDivElement | null>;
}) {
  const mv = useRef<MV | null>(null);
  const [loaded, setLoaded] = useState(false);
  const show3d = active && mvReady && Boolean(dish.model);

  // Gira com o scroll (lateral e vertical) + giro lento parado
  useEffect(() => {
    if (!show3d) return;
    let raf = 0;
    let idle = 0;
    let last = performance.now();
    const loop = (t: number) => {
      idle += (t - last) * 0.012; // ~12°/s parado
      last = t;
      const el = mv.current;
      if (el) {
        const scroll = (track.current?.scrollLeft ?? 0) * 0.35 + window.scrollY * 0.45;
        el.cameraOrbit = `${(scroll + idle) % 360}deg 68deg auto`;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [show3d, track]);

  useEffect(() => {
    const el = mv.current;
    if (!el) return;
    const onLoad = () => setLoaded(true);
    el.addEventListener("load", onLoad);
    return () => el.removeEventListener("load", onLoad);
  }, [show3d]);

  return (
    <Link
      href={`/prototipo/${dish.slug}`}
      className="press group relative w-[78vw] max-w-[340px] shrink-0 snap-center overflow-hidden rounded-[22px] bg-surface"
      style={{ transition: "transform 160ms var(--ease-out), opacity 300ms ease", opacity: active ? 1 : 0.6 }}
    >
      <div className="relative aspect-[4/5] w-full">
        <ViewTransition name={`dish-${dish.slug}`}>
          <div className="absolute inset-0">
            {dish.photo && (
              <Image src={dish.photo} alt={dish.name} fill sizes="340px" className="object-cover" />
            )}
          </div>
        </ViewTransition>

        {show3d && (
          <model-viewer
            ref={mv as React.Ref<HTMLElement>}
            src={dish.model!}
            alt=""
            aria-hidden="true"
            loading="eager"
            interaction-prompt="none"
            shadow-intensity="1"
            exposure="1.05"
            tone-mapping="neutral"
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
              background: "radial-gradient(circle at 50% 60%, #4a352a 0%, #32211a 70%)",
              opacity: loaded ? 1 : 0,
              transition: "opacity 500ms ease",
            }}
          >
            <span slot="progress-bar" hidden />
          </model-viewer>
        )}

        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/35 to-transparent p-4 pt-16">
          <div className="flex items-end justify-between gap-3">
            <div className="text-left">
              <h3 className="font-display text-2xl leading-tight font-semibold text-[#f6ede0]">{dish.name}</h3>
              <p className="mt-0.5 text-sm font-semibold text-accent tabular-nums">{formatPrice(dish.price)}</p>
            </div>
            {dish.model && (
              <span className="rounded-full border border-accent/50 bg-black/40 px-2.5 py-1 text-[11px] font-semibold tracking-wide text-accent backdrop-blur">
                AR
              </span>
            )}
          </div>
        </div>
      </div>
    </Link>
  );
}
