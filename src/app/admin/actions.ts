"use server";

import { createClient } from "@supabase/supabase-js";
import { updateTag } from "next/cache";
import { MENU_TAG } from "@/lib/menu";

/**
 * Atualiza o cardápio público depois de uma edição no painel.
 * Recebe o token da sessão e confere se a pessoa é admin antes de agir.
 */
export async function refreshMenu(accessToken: string): Promise<{ ok: boolean; error?: string }> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anon) return { ok: false, error: "Supabase não configurado." };

  const db = createClient(url, anon, {
    auth: { persistSession: false, autoRefreshToken: false },
    global: { headers: { Authorization: `Bearer ${accessToken}` } },
  });

  const { data: user, error: userError } = await db.auth.getUser(accessToken);
  if (userError || !user.user) return { ok: false, error: "Sessão inválida." };

  const { data: isAdmin, error } = await db.rpc("is_admin");
  if (error || !isAdmin) return { ok: false, error: "Sem permissão." };

  updateTag(MENU_TAG);
  return { ok: true };
}
