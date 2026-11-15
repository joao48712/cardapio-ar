import Image from "next/image";
import Link from "next/link";
import { RESTAURANT } from "@/lib/restaurant";

export function MenuHeader({ compact = false }: { compact?: boolean }) {
  const size = compact ? 52 : 120;
  return (
    <Link href="/" className="inline-flex min-h-11 items-center gap-3" aria-label={`${RESTAURANT.name}, voltar ao cardápio`}>
      {RESTAURANT.logo ? (
        // O logo já traz o nome. A máscara suaviza as bordas da textura de fundo.
        <Image
          src={RESTAURANT.logo}
          alt={RESTAURANT.name}
          width={size}
          height={size}
          priority={!compact}
          className="logo-fade"
          style={{ width: size, height: size }}
        />
      ) : (
        <span className={`font-display font-semibold tracking-tight ${compact ? "text-xl" : "text-2xl"}`}>
          {RESTAURANT.name}
        </span>
      )}
    </Link>
  );
}
