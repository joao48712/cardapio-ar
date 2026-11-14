-- Cardápio 3D/AR — schema inicial
-- Rodar no Supabase: SQL Editor > colar > Run

create extension if not exists "pgcrypto";

-- Quem pode editar o cardápio
create table if not exists public.admins (
  email text primary key
);

create or replace function public.is_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.admins
    where lower(email) = lower(coalesce(auth.jwt() ->> 'email', ''))
  );
$$;

create table if not exists public.categories (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  slug text not null unique,
  sort int not null default 0,
  active boolean not null default true,
  created_at timestamptz not null default now()
);

create table if not exists public.items (
  id uuid primary key default gen_random_uuid(),
  category_id uuid not null references public.categories(id) on delete restrict,
  name text not null,
  slug text not null unique,
  description text not null default '',
  price_cents int not null check (price_cents >= 0),
  photo_path text,
  model_path text,
  poster_path text,
  model_size_bytes int,
  tags text[] not null default '{}',
  sort int not null default 0,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists items_category_idx on public.items(category_id, sort);

create or replace function public.touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists items_touch on public.items;
create trigger items_touch before update on public.items
for each row execute function public.touch_updated_at();

-- RLS
alter table public.admins enable row level security;
alter table public.categories enable row level security;
alter table public.items enable row level security;

drop policy if exists "admins read self" on public.admins;
create policy "admins read self" on public.admins
  for select using (public.is_admin());

drop policy if exists "public read categories" on public.categories;
create policy "public read categories" on public.categories
  for select using (active or public.is_admin());

drop policy if exists "admin write categories" on public.categories;
create policy "admin write categories" on public.categories
  for all using (public.is_admin()) with check (public.is_admin());

drop policy if exists "public read items" on public.items;
create policy "public read items" on public.items
  for select using (active or public.is_admin());

drop policy if exists "admin write items" on public.items;
create policy "admin write items" on public.items
  for all using (public.is_admin()) with check (public.is_admin());

-- Storage: fotos e modelos (leitura pública, escrita só admin)
insert into storage.buckets (id, name, public)
values ('photos', 'photos', true), ('models', 'models', true)
on conflict (id) do nothing;

drop policy if exists "public read media" on storage.objects;
create policy "public read media" on storage.objects
  for select using (bucket_id in ('photos', 'models'));

drop policy if exists "admin insert media" on storage.objects;
create policy "admin insert media" on storage.objects
  for insert with check (bucket_id in ('photos', 'models') and public.is_admin());

drop policy if exists "admin update media" on storage.objects;
create policy "admin update media" on storage.objects
  for update using (bucket_id in ('photos', 'models') and public.is_admin());

drop policy if exists "admin delete media" on storage.objects;
create policy "admin delete media" on storage.objects
  for delete using (bucket_id in ('photos', 'models') and public.is_admin());

-- Depois de criar seu usuário em Authentication > Users, rode:
-- insert into public.admins (email) values ('seu-email@exemplo.com');
