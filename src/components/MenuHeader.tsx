import Image from "next/image";
import Link from "next/link";
import { RESTAURANT } from "@/lib/restaurant";

export function MenuHeader({ compact = false }: { compact?: boolean }) {
  return (
    <Link href="/" className="inline-flex min-h-11 items-center gap-3" aria-label={`${RESTAURANT.name}, voltar ao cardápio`}>
      {RESTAURANT.logo ? (
        <Image src={RESTAURANT.logo} alt="" width={compact ? 32 : 44} height={compact ? 32 : 44} className="h-auto" />
      ) : null}
      <span className={`font-display font-semibold tracking-tight ${compact ? "text-xl" : "text-2xl"}`}>
        {RESTAURANT.name}
      </span>
    </Link>
  );
}
