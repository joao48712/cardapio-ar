import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Suspense } from "react";
import { MenuHeader } from "@/components/MenuHeader";
import { ARView } from "@/components/ARView";
import { PlateIcon } from "@/components/PlateIcon";
import { formatPrice } from "@/lib/format";
import { getItem, getSlugs } from "@/lib/menu";
import { RESTAURANT } from "@/lib/restaurant";
import { mediaUrl } from "@/lib/supabase";

export async function generateStaticParams() {
  const slugs = await getSlugs();
  // Cache Components exige ao menos um resultado; com cardápio vazio, usa um slug que dá 404.
  if (slugs.length === 0) return [{ slug: "_vazio" }];
  return slugs.map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: PageProps<"/prato/[slug]">): Promise<Metadata> {
  const { slug } = await params;
  const item = await getItem(slug);
  if (!item) return {};
  const photo = mediaUrl("photos", item.photo_path);
  return {
    title: `${item.name} | ${RESTAURANT.name}`,
    description: item.description,
    openGraph: photo ? { images: [photo] } : undefined,
  };
}

export default function ItemPage({ params }: PageProps<"/prato/[slug]">) {
  return (
    <Suspense fallback={<ItemSkeleton />}>
      <ItemContent params={params} />
    </Suspense>
  );
}

function ItemSkeleton() {
  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 pt-20 pb-16" aria-busy="true">
      <div className="aspect-square w-full animate-pulse rounded-[var(--radius)] bg-surface sm:aspect-[4/3]" />
      <div className="mt-6 h-10 w-2/3 animate-pulse rounded bg-surface" />
    </main>
  );
}

async function ItemContent({ params }: { params: PageProps<"/prato/[slug]">["params"] }) {
  const { slug } = await params;
  const item = await getItem(slug);
  if (!item) notFound();

  const photo = mediaUrl("photos", item.photo_path);
  const poster = photo ?? mediaUrl("photos", item.poster_path);
  const model = mediaUrl("models", item.model_path);

  return (
    <>
      <header className="mx-auto flex w-full max-w-3xl items-center justify-between gap-4 px-4 py-4">
        <Link href={`/#${item.category.slug}`} className="btn btn-ghost">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
            <path d="m15 18-6-6 6-6" />
          </svg>
          Cardápio
        </Link>
        <MenuHeader compact />
      </header>

      <main className="mx-auto w-full max-w-3xl flex-1 px-4 pb-16">
        {model ? (
          <ARView src={model} photo={poster} alt={item.name} />
        ) : photo ? (
          <div className="relative aspect-square w-full overflow-hidden rounded-[var(--radius)] bg-surface sm:aspect-[4/3]">
            <Image src={photo} alt={item.name} fill priority sizes="(max-width: 768px) 100vw, 768px" className="object-cover" />
          </div>
        ) : (
          <div className="flex aspect-[4/3] w-full items-center justify-center rounded-[var(--radius)] bg-surface">
            <PlateIcon size={72} />
          </div>
        )}

        <p className="mt-6 text-xs font-semibold tracking-[0.18em] text-muted uppercase">{item.category.name}</p>
        <div className="mt-2 flex items-baseline justify-between gap-4">
          <h1 className="font-display text-4xl leading-[1.05] font-medium">{item.name}</h1>
          <span className="shrink-0 text-lg font-semibold text-accent tabular-nums">{formatPrice(item.price_cents)}</span>
        </div>
        <p className="mt-4 text-[16px] leading-relaxed text-muted">{item.description}</p>
        {item.tags.length > 0 && (
          <div className="mt-5 flex flex-wrap gap-2">
            {item.tags.map((t) => (
              <span key={t} className="rounded-full border border-line px-3 py-1 text-xs text-muted">
                {t}
              </span>
            ))}
          </div>
        )}
        <p className="mt-10 border-t border-line pt-5 text-sm text-muted">
          Peça ao garçom. O pagamento é feito no caixa.
        </p>
      </main>
    </>
  );
}
