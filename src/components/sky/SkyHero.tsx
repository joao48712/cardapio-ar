"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { skyColors, skyPhase, sunAltitude, sunsetTime } from "@/lib/sun";

type Props = {
  children: ReactNode;
  /** Protótipo: mostra o controle de hora para ver o céu em qualquer momento. */
  controls?: boolean;
};

const fmt = new Intl.DateTimeFormat("pt-BR", { hour: "2-digit", minute: "2-digit", timeZone: "America/Sao_Paulo" });

/** Topo do cardápio com o céu de Dracena na hora real. */
export function SkyHero({ children, controls = false }: Props) {
  const [now, setNow] = useState<Date | null>(null);
  const [override, setOverride] = useState<number | null>(null);

  // Só depois de montar: o servidor não sabe a hora do cliente
  useEffect(() => {
    const tick = () => setNow(new Date());
    const first = setTimeout(tick, 0);
    const id = setInterval(tick, 60_000);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  }, []);

  const date = useMemo(() => {
    if (!now) return null;
    if (override === null) return now;
    const d = new Date(now);
    d.setHours(Math.floor(override), Math.round((override % 1) * 60), 0, 0);
    return d;
  }, [now, override]);

  const altitude = date ? sunAltitude(date) : 2; // antes de montar: hora dourada (bonito e neutro)
  const sky = skyColors(altitude);
  const phase = skyPhase(altitude);
  const sunset = date ? sunsetTime(date) : null;

  // Sol desce do alto-direita até o horizonte; some abaixo de -6°
  const sunY = Math.min(92, Math.max(18, 86 - altitude * 2.2));

  return (
    <section
      aria-label="Topo do cardápio"
      className="sky relative isolate flex min-h-[64svh] flex-col items-center justify-end overflow-hidden px-4 pb-8 text-center"
      style={{
        background: `linear-gradient(180deg, ${sky.top} 0%, ${sky.mid} 46%, ${sky.low} 74%, var(--bg) 100%)`,
      }}
    >
      {/* Estrelas: visíveis só no crepúsculo e à noite */}
      <div aria-hidden className="sky-stars pointer-events-none absolute inset-0 -z-10" style={{ opacity: sky.stars }} />

      {/* Brilho do sol perto do horizonte */}
      <div
        aria-hidden
        className="pointer-events-none absolute -z-10 h-[70vmin] w-[70vmin] -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          left: "72%",
          top: `${sunY}%`,
          opacity: sky.glow,
          background: "radial-gradient(circle, rgb(255 214 150 / 0.85) 0%, rgb(255 170 100 / 0.35) 22%, transparent 62%)",
        }}
      />

      {/* Silhueta da cidade vista do terraço */}
      <Skyline />

      <div className="relative flex flex-col items-center">
        <div className="gold-shimmer">{children}</div>
        <p className="mt-1 font-display text-xl text-accent italic drop-shadow-[0_1px_8px_rgb(0_0_0/0.45)]">
          O sabor que acompanha o espetáculo do céu
        </p>
        {sunset && <SunsetLine now={date!} sunset={sunset} phase={phase} />}
      </div>

      {controls && (
        <div className="absolute top-3 right-3 left-3 flex items-center gap-2 rounded-full bg-black/35 px-3 py-1.5 text-xs text-white backdrop-blur">
          <span className="shrink-0 tabular-nums">{date ? fmt.format(date) : "--:--"}</span>
          <input
            aria-label="Hora do céu (protótipo)"
            type="range"
            min={5}
            max={23.75}
            step={0.25}
            value={override ?? (date ? date.getHours() + date.getMinutes() / 60 : 18)}
            onChange={(e) => setOverride(parseFloat(e.target.value))}
            className="w-full accent-[var(--accent)]"
          />
          <button className="shrink-0 rounded-full px-2 py-1 hover:bg-white/15" onClick={() => setOverride(null)}>
            Agora
          </button>
        </div>
      )}
    </section>
  );
}

function SunsetLine({ now, sunset, phase }: { now: Date; sunset: Date; phase: string }) {
  const diff = Math.round((sunset.valueOf() - now.valueOf()) / 60_000);
  let text: string;
  if (diff > 90) text = `Pôr do sol hoje às ${fmt.format(sunset)}. Garanta seu lugar no terraço.`;
  else if (diff > 0) text = `Pôr do sol em ${diff >= 60 ? `${Math.floor(diff / 60)}h${String(diff % 60).padStart(2, "0")}` : `${diff} min`}. O terraço já está dourado.`;
  else if (phase === "sunset" || phase === "twilight") text = "O céu está no auge agora. Olhe para o oeste.";
  else text = `O sol se pôs às ${fmt.format(sunset)}. A noite no terraço é sua.`;

  return (
    <p className="mt-3 inline-flex items-center gap-2 rounded-full border border-white/15 bg-black/25 px-4 py-1.5 text-[13px] text-[#f6ede0] backdrop-blur">
      <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-accent" />
      {text}
    </p>
  );
}

function Skyline() {
  // Prédios e árvores baixos de cidade do interior, desenhados à mão em SVG.
  return (
    <svg
      aria-hidden
      viewBox="0 0 400 60"
      preserveAspectRatio="none"
      className="pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-[16svh] w-full"
    >
      <path
        fill="var(--bg)"
        d="M0 44 L14 44 L14 36 L26 36 L26 41 L40 41 L40 30 L48 30 L48 26 L56 26 L56 38 L70 38 C74 33 80 31 86 35 C90 31 96 31 100 37 L112 37 L112 22 L120 22 L120 18 L126 18 L126 34 L140 34 L140 40 L156 40 C160 34 168 33 172 38 L186 38 L186 28 L198 28 L198 33 L214 33 L214 24 L222 24 L222 14 L228 14 L228 24 L236 24 L236 39 L252 39 C256 35 262 34 266 39 L280 39 L280 31 L294 31 L294 36 L306 36 L306 27 L318 27 L318 41 L334 41 C338 36 346 35 350 40 L362 40 L362 33 L376 33 L376 42 L400 42 L400 60 L0 60 Z"
      />
    </svg>
  );
}
