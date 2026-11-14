import { cacheLife, cacheTag } from "next/cache";
import { publicClient } from "./supabase";
import { seedCategories, seedItems } from "./seed";
import type { Category, Item, MenuSection } from "./types";

export const MENU_TAG = "menu";

async function load(): Promise<{ categories: Category[]; items: Item[] }> {
  const db = publicClient();
  if (!db) return { categories: seedCategories, items: seedItems };

  const [cats, items] = await Promise.all([
    db.from("categories").select("*").eq("active", true).order("sort"),
    db.from("items").select("*").eq("active", true).order("sort"),
  ]);
  if (cats.error) throw cats.error;
  if (items.error) throw items.error;
  return { categories: cats.data as Category[], items: items.data as Item[] };
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
