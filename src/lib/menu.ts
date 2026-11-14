import { cacheLife, cacheTag } from "next/cache";
import { publicClient } from "./supabase";
import { seedCategories, seedItems } from "./seed";
import type { Category, Item, MenuSection } from "./types";

export const MENU_TAG = "menu";

type Loaded = { categories: Category[]; items: Item[]; demo: boolean };

const DEMO: Loaded = { categories: seedCategories, items: seedItems, demo: true };

async function load(): Promise<Loaded> {
  const db = publicClient();
  if (!db) return DEMO;

  const [cats, items] = await Promise.all([
    db.from("categories").select("*").eq("active", true).order("sort"),
    db.from("items").select("*").eq("active", true).order("sort"),
  ]);
  if (cats.error) throw cats.error;
  if (items.error) throw items.error;

  const categories = cats.data as Category[];
  const list = items.data as Item[];
  // Banco conectado mas ainda sem nenhum prato visível: mostra os pratos de exemplo
  // em vez de uma página vazia. Some sozinho quando o primeiro prato é publicado no painel.
  const hasDish = list.some((i) => categories.some((c) => c.id === i.category_id));
  if (!hasDish) return DEMO;
  return { categories, items: list, demo: false };
}

/** true enquanto o site mostra os pratos de exemplo (sem Supabase ou sem pratos cadastrados). */
export async function isDemoMenu(): Promise<boolean> {
  "use cache";
  cacheTag(MENU_TAG);
  cacheLife("hours");
  return (await load()).demo;
}

export async function getMenu(): Promise<MenuSection[]> {
  "use cache";
  cacheTag(MENU_TAG);
  cacheLife("hours");
  const { categories, items } = await load();
  return categories
    .map((c) => ({ ...c, items: items.filter((i) => i.category_id === c.id) }))
    .filter((c) => c.items.length > 0);
}

export async function getItem(slug: string): Promise<(Item & { category: Category }) | null> {
  "use cache";
  cacheTag(MENU_TAG);
  cacheLife("hours");
  const { categories, items } = await load();
  const item = items.find((i) => i.slug === slug);
  if (!item) return null;
  const category = categories.find((c) => c.id === item.category_id);
  if (!category) return null;
  return { ...item, category };
}

export async function getSlugs(): Promise<string[]> {
  "use cache";
  cacheTag(MENU_TAG);
  cacheLife("hours");
  const { items } = await load();
  return items.map((i) => i.slug);
}
