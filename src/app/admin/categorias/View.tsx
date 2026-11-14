"use client";

import { useCallback, useEffect, useState, type FormEvent } from "react";
import { slugify } from "@/lib/format";
import type { Category } from "@/lib/types";
import { useAdmin } from "../AdminContext";

export function View() {
  const { db, publish } = useAdmin();
  const [cats, setCats] = useState<Category[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    db.from("categories").select("*").order("sort").then(({ data, error }) => {
      if (error) setError(error.message);
      setCats((data ?? []) as Category[]);
    });
  }, [db]);

  useEffect(() => {
    load();
  }, [load]);

  async function run(p: PromiseLike<{ error: { message: string } | null }>) {
    const { error } = await p;
    if (error) setError(error.message.includes("violates foreign key") ? "Esta categoria ainda tem pratos." : error.message);
    else {
      setError(null);
      publish();
    }
    load();
  }

  async function add(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    await run(db.from("categories").insert({ name: name.trim(), slug: slugify(name), sort: cats.length }));
    setName("");
  }

  async function move(idx: number, dir: -1 | 1) {
    const a = cats[idx], b = cats[idx + dir];
    if (!a || !b) return;
    await Promise.all([
      db.from("categories").update({ sort: idx + dir }).eq("id", a.id),
      db.from("categories").update({ sort: idx }).eq("id", b.id),
    ]);
    publish();
    load();
  }

  return (
    <div className="max-w-2xl">
      <h1 className="font-display text-3xl">Categorias</h1>
      <form onSubmit={add} className="mt-6 flex gap-2">
        <label className="sr-only" htmlFor="new-cat">Nova categoria</label>
        <input id="new-cat" className="field flex-1" placeholder="Ex: Entradas" value={name} onChange={(e) => setName(e.target.value)} />
        <button className="btn btn-primary">Adicionar</button>
      </form>
      {error && <p className="mt-3 text-sm text-danger" role="alert">{error}</p>}

      <ul className="mt-6 divide-y divide-line overflow-hidden rounded-[var(--radius)] border border-line">
        {cats.map((c, i) => (
          <li key={c.id} className="flex items-center gap-3 bg-surface px-3 py-2">
            <div className="flex flex-col">
              <button aria-label={`Subir ${c.name}`} className="h-6 w-8 text-muted hover:text-ink disabled:opacity-20" disabled={i === 0} onClick={() => move(i, -1)}>▲</button>
              <button aria-label={`Descer ${c.name}`} className="h-6 w-8 text-muted hover:text-ink disabled:opacity-20" disabled={i === cats.length - 1} onClick={() => move(i, 1)}>▼</button>
            </div>
            <input
              aria-label="Nome da categoria"
              className="field flex-1"
              defaultValue={c.name}
              onBlur={(e) => {
                const v = e.target.value.trim();
                if (v && v !== c.name) run(db.from("categories").update({ name: v, slug: slugify(v) }).eq("id", c.id));
              }}
            />
            <label className="flex min-h-11 items-center gap-2 text-xs text-muted">
              <input type="checkbox" checked={c.active} className="h-4 w-4 accent-[var(--accent)]" onChange={() => run(db.from("categories").update({ active: !c.active }).eq("id", c.id))} />
              Visível
            </label>
            <button
              className="min-h-11 px-2 text-xs text-muted hover:text-danger"
              onClick={() => confirm(`Apagar a categoria "${c.name}"?`) && run(db.from("categories").delete().eq("id", c.id))}
            >
              Apagar
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
