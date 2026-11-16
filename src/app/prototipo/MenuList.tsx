"use client";

import Image from "next/image";
import Link from "next/link";
import { ViewTransition, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { PlateIcon } from "@/components/PlateIcon";
import { formatPrice } from "@/lib/format";
import type { ProtoCategory, ProtoDish } from "./data";

const INDICATOR_TRANSITION = "transform 280ms var(--ease-out), width 280ms var(--ease-out), opacity 200ms ease";

/**
 * Cardápio por categoria. A barra de categorias gruda no topo e um traço dourado
 * desliza até a categoria que está na tela (ou a que a pessoa tocou).
 */
export function MenuList({ dishes, categories }: { dishes: ProtoDish[]; categories: ProtoCategory[] }) {
  const groups = useMemo(
    () =>
      categories
        .map((c) => ({ ...c, dishes: dishes.filter((d) => d.category === c.slug) }))
        .filter((g) => g.dishes.length > 0),
    [categories, dishes],
  );

  const [active, setActive] = useState(groups[0]?.slug ?? "");
  const track = useRef<HTMLDivElement>(null);
  const indicator = useRef<HTMLSpanElement>(null);
  const placed = useRef(false);
  const end = useRef<HTMLDivElement>(null);
  // Depois de um toque, o scroll-spy espera a rolagem terminar (senão o traço passeia por todas)
  const lockUntil = useRef(0);

  // Move o indicador direto no DOM: só transform e width, sem re-render
  const place = useCallback(() => {
    const t = track.current;
    const ind = indicator.current;
    const btn = t?.querySelector<HTMLElement>(`[data-cat="${active}"]`);
    if (!t || !ind || !btn) return;
    if (!placed.current) ind.style.transition = "none";
    ind.style.width = `${btn.offsetWidth}px`;
    ind.style.transform = `translate3d(${btn.offsetLeft}px, 0, 0)`;
    ind.style.opacity = "1";
    if (!placed.current) {
      placed.current = true;
      requestAnimationFrame(() => {
        ind.style.transition = INDICATOR_TRANSITION;
      });
    }
    return btn;
  }, [active]);

  useLayoutEffect(() => {
    const t = track.current;
    const first = !placed.current;
    const btn = place();
    if (!t || !btn || first) return;
    // Mantém a categoria ativa à vista na barra (rola só a barra, não a página)
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    t.scrollTo({
      left: btn.offsetLeft - (t.clientWidth - btn.offsetWidth) / 2,
      behavior: reduce ? "auto" : "smooth",
    });
  }, [place]);

  // Fonte carregou ou a tela girou: mede de novo
  useEffect(() => {
    const t = track.current;
    if (!t) return;
    const ro = new ResizeObserver(() => place());
    ro.observe(t);
    document.fonts?.ready.then(() => place());
    return () => ro.disconnect();
  }, [place]);

  // Scroll-spy: a seção que cruza uma linha a 25% da tela é a ativa
  useEffect(() => {
    const visible = new Set<string>();
    let atEnd = false;
    const pick = () => {
      if (performance.now() < lockUntil.current) return;
      const last = groups[groups.length - 1];
      const hit = atEnd && last ? last : groups.find((g) => visible.has(g.slug));
      if (hit) setActive(hit.slug);
    };
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          const slug = (e.target as HTMLElement).dataset.section!;
          if (e.isIntersecting) visible.add(slug);
          else visible.delete(slug);
        }
        pick();
      },
      { rootMargin: "-25% 0px -75% 0px" },
    );
    document.querySelectorAll<HTMLElement>("[data-section]").forEach((s) => io.observe(s));
    // Última seção curta nunca chega à linha: vale quando o fim da lista aparece
    const ioEnd = new IntersectionObserver(
      ([e]) => {
        atEnd = e.isIntersecting;
        pick();
      },
      { rootMargin: "0px 0px -96px 0px" },
    );
    if (end.current) ioEnd.observe(end.current);
    return () => {
      io.disconnect();
      ioEnd.disconnect();
    };
  }, [groups]);

  // A página do prato usa isto para saber que "voltar" pode ser o voltar do navegador
  useEffect(() => {
    try {
      sessionStorage.setItem("proto-menu", "1");
    } catch {}
  }, []);

  function go(slug: string, at: number) {
    lockUntil.current = at + 900; // at = timeStamp do toque (mesmo relógio do performance.now)
    setActive(slug);
    document.getElementById(`cat-${slug}`)?.scrollIntoView({ block: "start" });
  }

  if (groups.length === 0) return null;

  return (
    <div className="mt-8">
      <nav
        aria-label="Categorias"
        className="sticky top-0 z-30 bg-bg/85 pt-[env(safe-area-inset-top)] backdrop-blur-md"
      >
        <div
          ref={track}
          className="no-scrollbar relative mx-auto flex max-w-3xl gap-1 overflow-x-auto overscroll-x-contain px-3 py-2"
        >
          <span
            ref={indicator}
            aria-hidden
            className="pointer-events-none absolute top-1/2 left-0 -mt-[18px] h-9 rounded-full border border-accent/55 bg-accent/[0.13] opacity-0"
            style={{ width: 0, willChange: "transform, width" }}
          />
          {groups.map((g) => {
            const on = g.slug === active;
            return (
              <button
                key={g.slug}
                type="button"
                data-cat={g.slug}
                aria-current={on ? "true" : undefined}
                onClick={(e) => go(g.slug, e.timeStamp)}
                className={`relative flex min-h-11 shrink-0 items-center rounded-full px-4 text-sm font-semibold whitespace-nowrap transition-colors duration-200 select-none ${
                  on ? "text-accent" : "text-muted hover:text-ink"
                }`}
              >
                {g.name}
              </button>
            );
          })}
        </div>
        <div
          aria-hidden
          className="h-px"
          style={{
            background:
              "linear-gradient(90deg, transparent, color-mix(in oklab, var(--accent) 45%, transparent), transparent)",
          }}
        />
      </nav>

      <div className="mx-auto max-w-3xl px-4">
        {groups.map((g) => (
          <section key={g.slug} id={`cat-${g.slug}`} data-section={g.slug} aria-labelledby={`h-${g.slug}`} className="pt-10">
            <h2 id={`h-${g.slug}`} className="ornament">
              <span className="font-display text-[30px] leading-none font-medium tracking-wide">{g.name}</span>
            </h2>
            <p className="mt-1.5 text-center text-[11px] tracking-[0.28em] text-muted uppercase">
              {g.dishes.length === 1 ? "1 opção" : `${g.dishes.length} opções`}
            </p>
            <ul className="mt-5 grid gap-3 sm:grid-cols-2">
              {g.dishes.map((d) => (
                <li key={d.slug}>
                  <DishCard dish={d} />
                </li>
              ))}
            </ul>
          </section>
        ))}
        <div ref={end} aria-hidden className="h-px" />
      </div>
    </div>
  );
}

function DishCard({ dish }: { dish: ProtoDish }) {
  // Os destaques já usam `dish-${slug}` no carrossel; o nome não pode repetir na mesma página
  const vtName = dish.featured ? `dish-list-${dish.slug}` : `dish-${dish.slug}`;

  return (
    <Link
      href={`/prototipo/${dish.slug}`}
      className="press card-link grid h-full grid-cols-[112px_1fr] gap-3.5 rounded-[20px] border border-line/60 bg-surface/70 p-2.5 min-[400px]:grid-cols-[128px_1fr]"
    >
      <div className="relative aspect-[4/5] self-start">
        <ViewTransition name={vtName}>
          <div className="zoom-frame absolute inset-0 rounded-[14px] bg-surface-2">
            {dish.photo ? (
              <Image src={dish.photo} alt="" fill sizes="128px" className="object-cover" />
            ) : (
              <div className="flex h-full items-center justify-center">
                <PlateIcon size={40} />
              </div>
            )}
          </div>
        </ViewTransition>
        {dish.model && (
          <span className="absolute top-1.5 left-1.5 rounded-full border border-accent/50 bg-black/45 px-2 py-0.5 text-[10px] font-semibold tracking-wider text-accent backdrop-blur-sm">
            AR
          </span>
        )}
      </div>

      <div className="flex min-w-0 flex-col py-1 pr-1">
        <h3 className="font-display text-[22px] leading-[1.08] font-semibold text-ink">{dish.name}</h3>
        <p className="mt-1.5 line-clamp-2 text-[13.5px] leading-snug text-muted">{dish.description}</p>
        {dish.tags.length > 0 && (
          <ul className="mt-2 flex flex-wrap gap-1.5" aria-label="Características">
            {dish.tags.map((t) => (
              <li key={t} className="rounded-full border border-line px-2 py-px text-[11px] tracking-wide text-muted">
                {t}
              </li>
            ))}
          </ul>
        )}
        <div className="mt-auto flex items-end justify-between gap-2 pt-2.5">
          <span className="text-[15px] font-semibold text-accent tabular-nums">{formatPrice(dish.price)}</span>
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            className="text-accent/70"
            aria-hidden
          >
            <path d="M5 12h14M13 6l6 6-6 6" />
          </svg>
        </div>
      </div>
    </Link>
  );
}
