import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { notFound } from "next/navigation";
import { Suspense, ViewTransition } from "react";
import { PlateIcon } from "@/components/PlateIcon";
import { formatPrice } from "@/lib/format";
import { CATEGORIES, type ProtoDish } from "../data";
import { getProtoMenu } from "../menu.server";
import { BackButton, DishHero } from "./DishHero";

type Params = Promise<{ slug: string }>;

export async function generateStaticParams() {
  const menu = await getProtoMenu();
  return menu.map((d) => ({ slug: d.slug }));
}

export async function generateMetadata({ params }: { params: Params }): Promise<Metadata> {
  const { slug } = await params;
  const dish = (await getProtoMenu()).find((d) => d.slug === slug);
  return dish ? { title: `${dish.name} | Restaurante 153`, description: dish.description } : {};
}

export default function ProtoDishPage({ params }: { params: Params }) {
  return (
    <Suspense fallback={<DishSkeleton />}>
      <DishContent params={params} />
    </Suspense>
  );
}

function DishSkeleton() {
  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 pt-[76px] pb-16" aria-busy="true">
      <div className="aspect-square w-full rounded-[var(--radius)] bg-surface sm:aspect-[4/3]" />
      <div className="mt-8 h-3 w-24 rounded bg-surface" />
      <div className="mt-4 h-11 w-2/3 rounded bg-surface" />
    </main>
  );
}

async function DishContent({ params }: { params: Params }) {
  const { slug } = await params;
  const menu = await getProtoMenu();
  const dish = menu.find((d) => d.slug === slug);
  if (!dish) notFound();

  const category = CATEGORIES.find((c) => c.slug === dish.category);
  const pairings = dish.pairings
    .map((s) => menu.find((d) => d.slug === s))
    .filter((d): d is ProtoDish => d !== undefined && d.slug !== dish.slug);

  return (
    <>
      <header
        className="mx-auto grid w-full max-w-3xl grid-cols-[44px_1fr_44px] items-center px-4 pb-3"
        style={{ paddingTop: "calc(12px + env(safe-area-inset-top))" }}
      >
        <BackButton />
        <p className="text-center text-[11px] tracking-[0.32em] text-muted uppercase">Restaurante 153</p>
      </header>

      <main
        className="mx-auto w-full max-w-3xl flex-1 px-4"
        style={{ paddingBottom: "calc(56px + env(safe-area-inset-bottom))" }}
      >
        <ViewTransition name={`dish-${dish.slug}`}>
          <div>
            {dish.model ? (
              <DishHero model={dish.model} photo={dish.photo} name={dish.name} />
            ) : (
              <div className="relative flex aspect-square w-full items-center justify-center overflow-hidden rounded-[var(--radius)] bg-surface sm:aspect-[4/3]">
                {dish.photo ? (
                  <Image
                    src={dish.photo}
                    alt={dish.name}
                    fill
                    priority
                    sizes="(max-width: 768px) 100vw, 768px"
                    className="object-cover"
                  />
                ) : (
                  <PlateIcon size={72} />
                )}
              </div>
            )}
          </div>
        </ViewTransition>

        <article className="mt-9">
          {category && (
            <p className="text-xs font-semibold tracking-[0.3em] text-accent uppercase">{category.name}</p>
          )}
          <h1 className="mt-3 font-display text-[44px] leading-[0.98] font-medium text-balance sm:text-6xl">
            {dish.name}
          </h1>
          <p className="mt-4 text-[22px] font-semibold text-accent tabular-nums">{formatPrice(dish.price)}</p>
          <p className="mt-5 max-w-[58ch] text-[17px] leading-relaxed text-muted">{dish.description}</p>

          {dish.tags.length > 0 && (
            <ul className="mt-6 flex flex-wrap gap-2" aria-label="Características">
              {dish.tags.map((t) => (
                <li key={t} className="rounded-full border border-line px-3 py-1 text-xs tracking-wide text-muted">
                  {t}
                </li>
              ))}
            </ul>
          )}
        </article>

        <div className="ornament mt-11" aria-hidden>
          <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
            <path d="M5 0 6.4 3.6 10 5 6.4 6.4 5 10 3.6 6.4 0 5l3.6-1.4L5 0Z" />
          </svg>
        </div>

        {pairings.length > 0 && (
          <section aria-labelledby="h-combina" className="mt-9">
            <h2 id="h-combina" className="font-display text-[28px] leading-none font-medium">
              Combina com
            </h2>
            <ul className="no-scrollbar -mx-4 mt-4 flex snap-x gap-3 overflow-x-auto overscroll-x-contain px-4 pb-1">
              {pairings.map((p) => (
                <li key={p.slug} className="w-[188px] shrink-0 snap-start">
                  <Link
                    href={`/prototipo/${p.slug}`}
                    className="press card-link block overflow-hidden rounded-[18px] border border-line/60 bg-surface"
                  >
                    <ViewTransition name={`dish-${p.slug}`}>
                      <div className="zoom-frame relative aspect-[4/3] bg-surface-2">
                        {p.photo ? (
                          <Image src={p.photo} alt="" fill sizes="188px" className="object-cover" />
                        ) : (
                          <div className="flex h-full items-center justify-center">
                            <PlateIcon size={36} />
                          </div>
                        )}
                      </div>
                    </ViewTransition>
                    <div className="p-3">
                      <h3 className="font-display text-[19px] leading-tight font-semibold text-ink">{p.name}</h3>
                      <p className="mt-1 text-sm font-semibold text-accent tabular-nums">{formatPrice(p.price)}</p>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        )}

        <div className="mt-10">
          <Link href="/prototipo" className="btn btn-ghost w-full">
            Ver o cardápio completo
          </Link>
        </div>
      </main>
    </>
  );
}
