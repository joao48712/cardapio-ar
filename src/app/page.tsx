import Link from "next/link";
import { CategoryNav } from "@/components/CategoryNav";
import { ItemCard } from "@/components/ItemCard";
import { MenuHeader } from "@/components/MenuHeader";
import { getMenu, isDemoMenu } from "@/lib/menu";
import { RESTAURANT } from "@/lib/restaurant";

export default async function MenuPage() {
  const [menu, demo] = await Promise.all([getMenu(), isDemoMenu()]);
  const firstIds = new Set(menu.flatMap((c) => c.items).slice(0, 4).map((i) => i.id));

  return (
    <>
      <header className="mx-auto flex w-full max-w-3xl flex-col items-center px-4 pt-8 pb-6 text-center sm:pt-12">
        <MenuHeader />
        <h1 className="mt-4 font-display text-4xl leading-[1.05] font-medium sm:text-5xl">Cardápio</h1>
        <p className="mt-2 font-display text-lg text-accent italic">{RESTAURANT.tagline}</p>
        <p className="mt-4 max-w-md text-[15px] leading-relaxed text-muted">
          Pratos com o selo <span className="font-semibold text-accent">AR</span> aparecem na sua mesa, no tamanho real,
          pela câmera do celular.
        </p>
        {demo && (
          <p className="mt-5 rounded-[var(--radius)] border border-line px-4 py-3 text-left text-sm leading-relaxed text-muted">
            <span className="font-semibold text-ink">Cardápio de demonstração.</span> Estes pratos são exemplos e somem
            quando o primeiro prato for cadastrado no{" "}
            <Link href="/admin" className="text-accent underline">
              painel
            </Link>
            .
          </p>
        )}
      </header>

      {menu.length > 0 && <CategoryNav categories={menu.map((c) => ({ slug: c.slug, name: c.name }))} />}

      <main className="mx-auto w-full max-w-3xl flex-1 px-4 pb-16">
        {menu.length === 0 && <p className="py-16 text-center text-muted">Cardápio em preparo.</p>}
        {menu.map((cat) => (
          <section key={cat.id} id={cat.slug} aria-labelledby={`h-${cat.slug}`} className="pt-8">
            <h2 id={`h-${cat.slug}`} className="mb-3 text-xs font-semibold tracking-[0.18em] text-muted uppercase">
              {cat.name}
            </h2>
            <ul className="flex flex-col gap-3">
              {cat.items.map((item) => (
                <li key={item.id}>
                  <ItemCard item={item} priority={firstIds.has(item.id)} />
                </li>
              ))}
            </ul>
          </section>
        ))}
      </main>

      <footer className="border-t border-line py-8 text-center text-xs text-muted">
        {RESTAURANT.name} · {RESTAURANT.city} · Pagamento no caixa
      </footer>
    </>
  );
}
