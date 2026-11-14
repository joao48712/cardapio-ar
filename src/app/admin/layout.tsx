import type { Metadata } from "next";
import { Suspense, type ReactNode } from "react";
import { AdminShell } from "./AdminShell";

// Painel roda todo no navegador (login Supabase), sem navegação instantânea.
export const instant = false;

export const metadata: Metadata = {
  title: "Painel | Cardápio",
  robots: { index: false, follow: false },
};

export default function AdminLayout({ children }: { children: ReactNode }) {
  return (
    <Suspense fallback={<main className="mx-auto w-full max-w-xl px-4 py-16 text-muted">Carregando…</main>}>
      <AdminShell>{children}</AdminShell>
    </Suspense>
  );
}
