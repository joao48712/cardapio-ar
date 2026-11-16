// Posição do sol sem dependências (fórmulas astronômicas simplificadas, precisão de ~1 minuto).
// Usado pelo "céu vivo": a cor do topo do cardápio segue o céu real de Dracena.

export const DRACENA = { lat: -21.4826, lon: -51.5326 };

const RAD = Math.PI / 180;
const DAY_MS = 86_400_000;
const J1970 = 2440588;
const J2000 = 2451545;
const OBLIQUITY = RAD * 23.4397;

function toDays(date: Date) {
  return date.valueOf() / DAY_MS - 0.5 + J1970 - J2000;
}

/** Altura do sol em graus (negativa = abaixo do horizonte). */
export function sunAltitude(date: Date, lat = DRACENA.lat, lon = DRACENA.lon): number {
  const d = toDays(date);
  const M = RAD * (357.5291 + 0.98560028 * d);
  const C = RAD * (1.9148 * Math.sin(M) + 0.02 * Math.sin(2 * M) + 0.0003 * Math.sin(3 * M));
  const L = M + C + RAD * 102.9372 + Math.PI;
  const dec = Math.asin(Math.sin(L) * Math.sin(OBLIQUITY));
  const ra = Math.atan2(Math.sin(L) * Math.cos(OBLIQUITY), Math.cos(L));
  const sidereal = RAD * (280.16 + 360.9856235 * d) + RAD * lon;
  const H = sidereal - ra;
  const phi = RAD * lat;
  return Math.asin(Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.cos(H)) / RAD;
}

/** Horário do pôr do sol no dia de `date` (procura minuto a minuto; é barato). */
export function sunsetTime(date: Date, lat = DRACENA.lat, lon = DRACENA.lon): Date | null {
  const start = new Date(date);
  start.setHours(12, 0, 0, 0);
  let prev = sunAltitude(start, lat, lon);
  for (let m = 1; m <= 12 * 60; m++) {
    const t = new Date(start.valueOf() + m * 60_000);
    const alt = sunAltitude(t, lat, lon);
    if (prev >= -0.833 && alt < -0.833) return t;
    prev = alt;
  }
  return null;
}

export type SkyPhase = "day" | "golden" | "sunset" | "twilight" | "night";

export function skyPhase(altitude: number): SkyPhase {
  if (altitude > 10) return "day";
  if (altitude > 2) return "golden";
  if (altitude > -4) return "sunset";
  if (altitude > -12) return "twilight";
  return "night";
}

type Stop = { alt: number; top: string; mid: string; low: string; glow: number; stars: number };

// Cores do céu por altura do sol. Entre dois pontos, as cores são interpoladas.
const STOPS: Stop[] = [
  { alt: -18, top: "#07091a", mid: "#141432", low: "#2b1d2a", glow: 0, stars: 1 },
  { alt: -10, top: "#0f1430", mid: "#2b2350", low: "#5a2f45", glow: 0.15, stars: 0.75 },
  { alt: -4, top: "#1d2452", mid: "#6a3a63", low: "#d0644a", glow: 0.55, stars: 0.25 },
  { alt: 0, top: "#2e3c78", mid: "#b8586a", low: "#f0904f", glow: 1, stars: 0 },
  { alt: 4, top: "#3f5d9e", mid: "#d98a6a", low: "#f6b766", glow: 0.85, stars: 0 },
  { alt: 12, top: "#4f86c6", mid: "#9cc0e0", low: "#f3d9a8", glow: 0.35, stars: 0 },
  { alt: 40, top: "#3d7ec4", mid: "#8fbde6", low: "#dcebf5", glow: 0.1, stars: 0 },
];

function mix(a: string, b: string, t: number) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(" ")})`;
}

export type SkyColors = { top: string; mid: string; low: string; glow: number; stars: number };

export function skyColors(altitude: number): SkyColors {
  const alt = Math.max(STOPS[0].alt, Math.min(STOPS[STOPS.length - 1].alt, altitude));
  let i = 0;
  while (i < STOPS.length - 2 && alt > STOPS[i + 1].alt) i++;
  const a = STOPS[i];
  const b = STOPS[i + 1];
  const t = (alt - a.alt) / (b.alt - a.alt);
  return {
    top: mix(a.top, b.top, t),
    mid: mix(a.mid, b.mid, t),
    low: mix(a.low, b.low, t),
    glow: a.glow + (b.glow - a.glow) * t,
    stars: a.stars + (b.stars - a.stars) * t,
  };
}
