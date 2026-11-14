import type { Category, Item } from "./types";

// Dados de exemplo, usados enquanto o Supabase não está configurado.
export const seedCategories: Category[] = [
  { id: "c1", name: "Entradas", slug: "entradas", sort: 0, active: true },
  { id: "c2", name: "Principais", slug: "principais", sort: 1, active: true },
  { id: "c3", name: "Sobremesas", slug: "sobremesas", sort: 2, active: true },
];

const base = { photo_path: null, poster_path: null, active: true } as const;

export const seedItems: Item[] = [
  {
    ...base, id: "i1", category_id: "c1", sort: 0, name: "Burrata com tomates assados", slug: "burrata-com-tomates-assados",
    description: "Burrata fresca, tomates confitados no forno, manjericão e azeite extravirgem.",
    price_cents: 6400, model_path: "/seed/sobremesa.glb", model_size_bytes: null, tags: ["vegetariano"],
  },
  {
    ...base, id: "i2", category_id: "c1", sort: 1, name: "Carpaccio de wagyu", slug: "carpaccio-de-wagyu",
    description: "Lâminas finas de wagyu, alcaparras, parmesão e rúcula.",
    price_cents: 7800, model_path: null, model_size_bytes: null, tags: [],
  },
  {
    ...base, id: "i3", category_id: "c2", sort: 0, name: "Risoto de cogumelos", slug: "risoto-de-cogumelos",
    description: "Arbóreo, shiitake, shimeji e cogumelo-de-paris, finalizado com manteiga trufada.",
    price_cents: 9200, model_path: null, model_size_bytes: null, tags: ["vegetariano", "sem glúten"],
  },
  {
    ...base, id: "i4", category_id: "c3", sort: 0, name: "Petit gâteau", slug: "petit-gateau",
    description: "Bolo quente de chocolate 70% com sorvete de baunilha.",
    price_cents: 4200, model_path: "/seed/sobremesa.glb", model_size_bytes: null, tags: [],
  },
];
