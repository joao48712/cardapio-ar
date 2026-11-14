import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

export const hasSupabase = Boolean(url && anonKey);

/** Cliente sem sessão, para ler o cardápio público (servidor). */
export function publicClient(): SupabaseClient | null {
  if (!url || !anonKey) return null;
  return createClient(url, anonKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

let browser: SupabaseClient | null = null;

/** Cliente do navegador com sessão, usado no painel admin. */
export function browserClient(): SupabaseClient {
  if (!url || !anonKey) {
    throw new Error("Supabase não configurado: defina NEXT_PUBLIC_SUPABASE_URL e NEXT_PUBLIC_SUPABASE_ANON_KEY.");
  }
  browser ??= createClient(url, anonKey);
  return browser;
}

/** URL pública de um arquivo. Caminhos que começam com "/" são arquivos locais (seed). */
export function mediaUrl(bucket: "photos" | "models", path: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("/") || path.startsWith("http")) return path;
  if (!url) return null;
  return `${url}/storage/v1/object/public/${bucket}/${path}`;
}
