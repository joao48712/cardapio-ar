"use client";

import type { Session, SupabaseClient } from "@supabase/supabase-js";
import { createContext, useCallback, useContext } from "react";
import { refreshMenu } from "./actions";

type Ctx = {
  db: SupabaseClient;
  session: Session;
};

const AdminCtx = createContext<Ctx | null>(null);

export const AdminProvider = AdminCtx.Provider;

export function useAdmin() {
  const ctx = useContext(AdminCtx);
  if (!ctx) throw new Error("useAdmin fora do painel");
  const { db, session } = ctx;

  /** Chame depois de salvar algo, para o cardápio público mostrar a mudança. */
  const publish = useCallback(async () => {
    const { data } = await db.auth.getSession();
    const token = data.session?.access_token ?? session.access_token;
    return refreshMenu(token);
  }, [db, session]);

  return { db, session, publish };
}
