export type Category = {
  id: string;
  name: string;
  slug: string;
  sort: number;
  active: boolean;
};

export type Item = {
  id: string;
  category_id: string;
  name: string;
  slug: string;
  description: string;
  price_cents: number;
  photo_path: string | null;
  model_path: string | null;
  poster_path: string | null;
  model_size_bytes: number | null;
  tags: string[];
  sort: number;
  active: boolean;
};

export type MenuSection = Category & { items: Item[] };
