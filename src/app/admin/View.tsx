"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { formatPrice } from "@/lib/format";
import type { Category, Item } from "@/lib/types";
import { useAdmin } from "./AdminContext";

export function View() {
  const { db, publish } = useAdmin();
  const [cats, setCats] = useState<Category[]>([]);
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    Promise.all([
      db.from("categories").select("*").order("sort"),
      db.from("items").select("*").order("sort"),
    ]).then(([c, i]) => {
      if (c.error || i.error) setError((c.error ?? i.error)!.message);
      setCats((c.data ?? []) as Category[]);
      setItems((i.data ?? []) as Item[]);
      setLoading(false);
      });
  }, [db]);

  useEffect(() => {
    load();
  }, [load]);

  async function toggle(item: Item) {
    setItems((list) => list.map((x) => (x.id === item.id ? { ...x, active: !x.active } : x)));
    const { error } = await db.from("items").update({ active: !item.active }).eq("id", item.id);
    if (error) {
      setError(error.message);
      load();
    } else publish();
  }

  async function move(item: Item, dir: -1 | 1) {
    const list = items.filter((x) => x.category_id === item.category_id);
    const idx = list.findIndex((x) => x.id === item.id);
    const other = list[idx + dir];
    if (!other) return;
    const reordered = [...list];
    reordered[idx] = other;
    reordered[idx + dir] = item;
    const updates = reordered.map((x, sort) => ({ id: x.id, sort }));
    setItems((all) => all.map((x) => ({ ...x, sort: updates.find((u) => u.id === x.id)?.sort ?? x.sort })).sort((a, b) => a.sort - b.sort));
    const results = await Promise.all(updates.map((u) => db.from("items").update({ sort: u.sort }).eq("id", u.id)));
    const failed = results.find((r) => r.error);
    if (failed?.error) {
      setError(failed.error.message);
      load();
    } else publish();
  }

  const total = items.length;
  const with3d = items.filter((i) => i.model_path).length;

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl">Pratos</h1>
          <p className="mt-1 text-sm text-muted">
            {total} pratos · {with3d} com 3D
          </p>
        </div>
        <Link href="/admin/pratos/novo" className="btn btn-primary">Novo prato</Link>
      </div>

      {error && <p className="mt-4 text-sm text-danger" role="alert">{error}</p>}
      {loading && <p className="mt-8 text-muted">Carregando…</p>}
      {!loading && cats.length === 0 && (
        <p className="mt-8 text-muted">
          Crie primeiro uma <Link className="text-accent underline" href="/admin/categorias">categoria</Link>.
        </p>
      )}

      {cats.map((cat) => {
        const list = items.filter((i) => i.category_id === cat.id);
        return (
          <section key={cat.id} className="mt-8">
            <h2 className="mb-2 text-xs font-semibold tracking-[0.18em] text-muted uppercase">
              {cat.name} {!cat.active && "(oculta)"}
            </h2>
            {list.length === 0 && <p className="text-sm text-muted">Nenhum prato.</p>}
            <ul className="divide-y divide-line overflow-hidden rounded-[var(--radius)] border border-line">
              {list.map((item, i) => (
                <li key={item.id} className={`flex items-center gap-3 bg-surface px-3 py-2 ${item.active ? "" : "opacity-50"}`}>
                  <div className="flex flex-col">
                    <button aria-label={`Subir ${item.name}`} className="h-6 w-8 text-muted hover:text-ink disabled:opacity-20" disabled={i === 0} onClick={() => move(item, -1)}>▲</button>
                    <button aria-label={`Descer ${item.name}`} className="h-6 w-8 text-muted hover:text-ink disabled:opacity-20" disabled={i === list.length - 1} onClick={() => move(item, 1)}>▼</button>
                  </div>
                  <Link href={`/admin/pratos/${item.id}`} className="min-w-0 flex-1 py-2">
                    <span className="block truncate font-medium">{item.name}</span>
                    <span className="text-xs text-muted">
                      {formatPrice(item.price_cents)} · {item.model_path ? "3D ok" : "sem 3D"} · {item.photo_path ? "foto ok" : "sem foto"}
                    </span>
                  </Link>
                  <label className="flex min-h-11 items-center gap-2 text-xs text-muted">
                    <input type="checkbox" checked={item.active} onChange={() => toggle(item)} className="h-4 w-4 accent-[var(--accent)]" />
                    Visível
                  </label>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
