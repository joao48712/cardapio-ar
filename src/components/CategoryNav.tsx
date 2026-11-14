"use client";

import { useEffect, useRef, useState } from "react";

type Cat = { slug: string; name: string };

export function CategoryNav({ categories }: { categories: Cat[] }) {
  const [current, setCurrent] = useState(categories[0]?.slug);
  const bar = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sections = categories
      .map((c) => document.getElementById(c.slug))
      .filter((el): el is HTMLElement => Boolean(el));
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setCurrent(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -60% 0px" },
    );
    sections.forEach((s) => io.observe(s));
    return () => io.disconnect();
  }, [categories]);

  useEffect(() => {
    const wrap = bar.current;
    const el = wrap?.querySelector<HTMLElement>(`[data-slug="${current}"]`);
    if (wrap && el) {
      wrap.scrollTo({ left: el.offsetLeft - wrap.clientWidth / 2 + el.clientWidth / 2, behavior: "smooth" });
    }
  }, [current]);

  return (
    <nav aria-label="Categorias" className="sticky top-0 z-20 border-b border-line bg-bg/90 backdrop-blur">
      <div ref={bar} className="no-scrollbar mx-auto flex max-w-3xl gap-1 overflow-x-auto px-4 py-2">
        {categories.map((c) => (
          <a
            key={c.slug}
            href={`#${c.slug}`}
            data-slug={c.slug}
            aria-current={current === c.slug ? "true" : undefined}
            className={`flex min-h-11 shrink-0 items-center rounded-full px-4 text-sm font-medium transition-colors ${
              current === c.slug ? "bg-accent text-accent-ink" : "text-muted hover:text-ink"
            }`}
          >
            {c.name}
          </a>
        ))}
      </div>
    </nav>
  );
}
