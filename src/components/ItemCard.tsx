import Image from "next/image";
import Link from "next/link";
import { formatPrice } from "@/lib/format";
import { mediaUrl } from "@/lib/supabase";
import type { Item } from "@/lib/types";
import { PlateIcon } from "./PlateIcon";

export function ItemCard({ item, priority = false }: { item: Item; priority?: boolean }) {
  const photo = mediaUrl("photos", item.photo_path) ?? mediaUrl("photos", item.poster_path);
  return (
    <Link
      href={`/prato/${item.slug}`}
      className="card-link -mx-2 flex gap-4 rounded-[var(--radius)] p-2 hover:bg-surface"
    >
      <div className="zoom-frame relative aspect-square w-28 shrink-0 rounded-[calc(var(--radius)-4px)] bg-surface sm:w-32">
        {photo ? (
          <Image src={photo} alt={item.name} fill sizes="128px" className="object-cover" priority={priority} />
        ) : (
          <div className="flex h-full items-center justify-center">
            <PlateIcon size={44} />
          </div>
        )}
        {item.model_path && (
          <span className="absolute top-1.5 left-1.5 rounded-full bg-bg/80 px-2 py-0.5 text-[11px] font-semibold tracking-wide text-accent backdrop-blur">
            AR
          </span>
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col py-1">
        <div className="flex items-baseline justify-between gap-3">
          <h3 className="font-display text-xl leading-tight font-semibold">{item.name}</h3>
          <span className="shrink-0 text-sm font-semibold text-accent tabular-nums">{formatPrice(item.price_cents)}</span>
        </div>
        <p className="mt-1 line-clamp-2 text-sm leading-relaxed text-muted">{item.description}</p>
        {item.tags.length > 0 && (
          <div className="mt-auto flex flex-wrap gap-1.5 pt-2">
            {item.tags.map((t) => (
              <span key={t} className="rounded-full border border-line px-2 py-0.5 text-[11px] text-muted">
                {t}
              </span>
            ))}
          </div>
        )}
      </div>
    </Link>
  );
}
