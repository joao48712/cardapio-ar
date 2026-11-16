"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { flushSync } from "react-dom";
import { PlateIcon } from "@/components/PlateIcon";
import { formatPrice } from "@/lib/format";
import { ORDER_URL, type ProtoCategory, type ProtoDish } from "./data";

const HAIRLINE =
  "linear-gradient(90deg, transparent, color-mix(in oklab, var(--accent) 70%, transparent), transparent)";

/** Tira acentos e caixa: "sanduiche" acha "Sanduíche" */
function norm(text: string) {
  return text
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase();
}

/** Mesa do QR (`?mesa=12`): guarda na sessão para valer nas próximas telas */
function readMesa(): string | null {
  try {
    const q = new URLSearchParams(window.location.search).get("mesa");
    if (q && /^\d{1,3}$/.test(q)) {
      if (sessionStorage.getItem("mesa") !== q) sessionStorage.setItem("mesa", q);
      return q;
    }
    return sessionStorage.getItem("mesa");
  } catch {
    return null;
  }
}
const noSubscribe = () => () => {};

export function BottomBar({ dishes, categories }: { dishes: ProtoDish[]; categories: ProtoCategory[] }) {
  const mesa = useSyncExternalStore(noSubscribe, readMesa, () => null);
  const [open, setOpen] = useState(false);
  const searchBtn = useRef<HTMLButtonElement>(null);
  const input = useRef<HTMLInputElement>(null);

  function openSearch() {
    // Abre e foca no mesmo toque (o iPhone só sobe o teclado assim)
    flushSync(() => setOpen(true));
    input.current?.focus({ preventScroll: true });
  }

  function closeSearch() {
    setOpen(false);
    searchBtn.current?.focus({ preventScroll: true });
  }

  return (
    <>
      {/* Reserva o espaço da barra no fim da página */}
      <div aria-hidden style={{ height: "calc(84px + env(safe-area-inset-bottom))" }} />

      <nav
        aria-label="Ações do cardápio"
        className="fixed inset-x-0 bottom-0 z-40 bg-bg/80 backdrop-blur-xl"
        style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
      >
        <div aria-hidden className="absolute inset-x-0 top-0 h-px" style={{ background: HAIRLINE }} />
        <div className="mx-auto flex max-w-3xl items-center gap-1.5 px-3 py-2">
          <button
            ref={searchBtn}
            type="button"
            onClick={openSearch}
            aria-haspopup="dialog"
            aria-expanded={open}
            className="press flex min-h-12 w-[72px] shrink-0 flex-col items-center justify-center gap-0.5 rounded-2xl text-[11px] font-semibold tracking-wide text-muted hover:text-ink"
          >
            <SearchIcon />
            Buscar
          </button>
          <button
            type="button"
            onClick={() => window.scrollTo({ top: 0 })}
            className="press flex min-h-12 w-[72px] shrink-0 flex-col items-center justify-center gap-0.5 rounded-2xl text-[11px] font-semibold tracking-wide text-muted hover:text-ink"
          >
            <MenuIcon />
            Cardápio
          </button>

          {mesa ? (
            <p className="ml-1.5 flex min-h-12 min-w-0 flex-1 items-center justify-center gap-1.5 rounded-full border border-accent/50 bg-accent/10 px-4 text-[13px] font-semibold whitespace-nowrap text-accent">
              Mesa {mesa}
              <span className="truncate font-medium text-muted">· peça ao garçom</span>
            </p>
          ) : (
            <a
              href={ORDER_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary ml-1.5 flex-1"
            >
              <BagIcon />
              Pedir
              <span className="sr-only">(abre o delivery em nova aba)</span>
            </a>
          )}
        </div>
      </nav>

      <SearchSheet open={open} onClose={closeSearch} inputRef={input} dishes={dishes} categories={categories} />
    </>
  );
}

function SearchSheet({
  open,
  onClose,
  inputRef,
  dishes,
  categories,
}: {
  open: boolean;
  onClose: () => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  dishes: ProtoDish[];
  categories: ProtoCategory[];
}) {
  const titleId = useId();
  const dialog = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [category, setCategory] = useState<string | null>(null);

  const allTags = useMemo(
    () => Array.from(new Set(dishes.flatMap((d) => d.tags))).sort((a, b) => a.localeCompare(b, "pt-BR")),
    [dishes],
  );
  const usedCategories = useMemo(
    () => categories.filter((c) => dishes.some((d) => d.category === c.slug)),
    [categories, dishes],
  );
  const categoryName = useMemo(() => new Map(categories.map((c) => [c.slug, c.name])), [categories]);

  const results = useMemo(() => {
    const q = norm(query.trim());
    return dishes.filter(
      (d) =>
        (!category || d.category === category) &&
        tags.every((t) => d.tags.includes(t)) &&
        (!q || norm(`${d.name} ${d.description}`).includes(q)),
    );
  }, [dishes, query, tags, category]);

  const filtering = query.trim() !== "" || tags.length > 0 || category !== null;

  // Trava a rolagem da página enquanto a busca está aberta
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.stopPropagation();
      onClose();
      return;
    }
    if (e.key !== "Tab" || !dialog.current) return;
    // Foco não sai da busca
    const items = dialog.current.querySelectorAll<HTMLElement>("a[href], button:not([disabled]), input");
    if (items.length === 0) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  function clear() {
    setQuery("");
    setTags([]);
    setCategory(null);
    inputRef.current?.focus({ preventScroll: true });
  }

  return (
    <div className="fixed inset-0 z-50" inert={!open} style={{ pointerEvents: open ? "auto" : "none" }} onKeyDown={onKeyDown}>
      <div
        aria-hidden
        onClick={onClose}
        className="absolute inset-0 bg-black/60"
        style={{ opacity: open ? 1 : 0, transition: `opacity ${open ? 300 : 220}ms ease` }}
      />

      <div
        ref={dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="absolute inset-x-0 bottom-0 mx-auto flex h-[88dvh] max-w-3xl flex-col overflow-hidden rounded-t-[26px] bg-bg shadow-[0_-18px_50px_rgb(0_0_0/0.45)]"
        style={{
          transform: open ? "translateY(0)" : "translateY(100%)",
          // Entra em 400ms; sai mais rápido para não segurar a pessoa
          transition: `transform ${open ? 400 : 280}ms var(--ease-drawer)`,
          willChange: "transform",
        }}
      >
        <div aria-hidden className="h-px shrink-0" style={{ background: HAIRLINE }} />

        <div className="shrink-0 px-4 pt-4">
          <div className="flex items-center justify-between gap-3">
            <h2 id={titleId} className="font-display text-[26px] leading-none font-medium text-accent">
              Buscar no cardápio
            </h2>
            <button
              type="button"
              onClick={onClose}
              aria-label="Fechar busca"
              className="press -mr-1.5 flex size-11 shrink-0 items-center justify-center rounded-full border border-line text-ink hover:border-muted"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
                <path d="M6 6l12 12M18 6 6 18" />
              </svg>
            </button>
          </div>

          <div className="relative mt-3.5">
            <span className="pointer-events-none absolute top-1/2 left-4 -translate-y-1/2 text-muted">
              <SearchIcon />
            </span>
            <input
              ref={inputRef}
              type="search"
              inputMode="search"
              enterKeyHint="search"
              autoComplete="off"
              autoCorrect="off"
              spellCheck={false}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Prato, ingrediente…"
              aria-label="Buscar por nome ou ingrediente"
              className="h-12 w-full appearance-none rounded-full border border-line bg-surface-2 pr-4 pl-12 text-[16px] text-ink outline-none placeholder:text-muted/80 focus:border-accent [&::-webkit-search-cancel-button]:hidden"
            />
          </div>
        </div>

        <div className="shrink-0 pt-3">
          <ChipRow label="Categorias">
            {usedCategories.map((c) => (
              <Chip key={c.slug} on={category === c.slug} onClick={() => setCategory(category === c.slug ? null : c.slug)}>
                {c.name}
              </Chip>
            ))}
          </ChipRow>
          {allTags.length > 0 && (
            <ChipRow label="Preferências">
              {allTags.map((t) => (
                <Chip
                  key={t}
                  on={tags.includes(t)}
                  onClick={() => setTags(tags.includes(t) ? tags.filter((x) => x !== t) : [...tags, t])}
                >
                  {t}
                </Chip>
              ))}
            </ChipRow>
          )}
        </div>

        <div className="mt-1 flex shrink-0 items-center justify-between border-t border-line/60 px-4 py-1">
          <p aria-live="polite" className="text-[11px] tracking-[0.22em] text-muted uppercase">
            {results.length === 1 ? "1 prato" : `${results.length} pratos`}
          </p>
          <button
            type="button"
            onClick={clear}
            disabled={!filtering}
            className="min-h-11 px-1 text-[13px] font-semibold text-accent transition-opacity duration-150 disabled:opacity-0"
          >
            Limpar filtros
          </button>
        </div>

        <div
          className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-2"
          style={{ paddingBottom: "calc(16px + env(safe-area-inset-bottom))" }}
        >
          {results.length === 0 ? (
            <div className="px-6 pt-10 text-center">
              <p className="font-display text-2xl text-ink">Nada por aqui</p>
              <p className="mx-auto mt-2 max-w-[28ch] text-sm leading-relaxed text-muted">
                Nenhum prato combina com essa busca. Tente outra palavra ou tire algum filtro.
              </p>
            </div>
          ) : (
            <ul>
              {results.map((d) => (
                <li key={d.slug}>
                  <Link
                    href={`/prototipo/${d.slug}`}
                    className="press flex min-h-[72px] items-center gap-3 rounded-2xl p-2 hover:bg-surface"
                  >
                    <span className="relative size-14 shrink-0 overflow-hidden rounded-xl bg-surface-2">
                      {d.photo ? (
                        <Image src={d.photo} alt="" fill sizes="56px" className="object-cover" />
                      ) : (
                        <span className="flex h-full items-center justify-center">
                          <PlateIcon size={24} />
                        </span>
                      )}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-display text-[19px] leading-tight font-semibold text-ink">
                        {d.name}
                      </span>
                      <span className="mt-0.5 block truncate text-xs text-muted">
                        {[categoryName.get(d.category), ...d.tags].filter(Boolean).join(" · ")}
                      </span>
                    </span>
                    <span className="shrink-0 pr-1 text-sm font-semibold text-accent tabular-nums">
                      {formatPrice(d.price)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function ChipRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div role="group" aria-label={label} className="no-scrollbar flex gap-2 overflow-x-auto overscroll-x-contain px-4 py-1">
      {children}
    </div>
  );
}

function Chip({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      // Visual de 36px, área de toque de 44px
      className={`press relative flex h-9 shrink-0 items-center rounded-full border px-3.5 text-[13px] font-semibold whitespace-nowrap select-none after:absolute after:-inset-y-1 after:inset-x-0 ${
        on ? "border-accent bg-accent text-accent-ink" : "border-line text-muted hover:border-muted"
      }`}
    >
      {children}
    </button>
  );
}

function SearchIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden>
      <circle cx="11" cy="11" r="6.5" />
      <path d="m16 16 4.5 4.5" strokeLinecap="round" />
    </svg>
  );
}

function MenuIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden>
      <path d="M12 6.5C10 5 7.5 4.5 4 4.5v13c3.5 0 6 .5 8 2 2-1.5 4.5-2 8-2v-13c-3.5 0-6 .5-8 2Zm0 0v13" strokeLinejoin="round" />
    </svg>
  );
}

function BagIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
      <path d="M5 8h14l-1 12H6L5 8Z" strokeLinejoin="round" />
      <path d="M9 8V6.5a3 3 0 0 1 6 0V8" />
    </svg>
  );
}
