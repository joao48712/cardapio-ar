"use client";

import type { Session } from "@supabase/supabase-js";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { browserClient, hasSupabase } from "@/lib/supabase";
import { AdminProvider } from "./AdminContext";

const NAV = [
  { href: "/admin", label: "Pratos" },
  { href: "/admin/categorias", label: "Categorias" },
  { href: "/admin/qr", label: "QR codes" },
  { href: "/admin/guia-captura", label: "Guia de captura" },
];

export function AdminShell({ children }: { children: ReactNode }) {
  if (!hasSupabase) return <Setup />;
  return <Gate>{children}</Gate>;
}

function Gate({ children }: { children: ReactNode }) {
  const db = browserClient();
  const pathname = usePathname();
  const [session, setSession] = useState<Session | null | undefined>(undefined);
  const [isAdmin, setIsAdmin] = useState<boolean | undefined>(undefined);

  useEffect(() => {
    db.auth.getSession().then(({ data }) => setSession(data.session));
    const { data } = db.auth.onAuthStateChange((_e, s) => setSession(s));
    return () => data.subscription.unsubscribe();
  }, [db]);

  useEffect(() => {
    if (!session) return;
    db.rpc("is_admin").then(({ data }) => setIsAdmin(Boolean(data)));
  }, [db, session]);

  if (session === undefined || (session && isAdmin === undefined)) {
    return <Shell><p className="text-muted">Carregando…</p></Shell>;
  }
  if (!session) return <Login />;
  if (!isAdmin) {
    return (
      <Shell>
        <h1 className="font-display text-3xl">Sem acesso</h1>
        <p className="mt-3 text-muted">
          O e-mail {session.user.email} não está na lista de admins. Rode no SQL do Supabase:
        </p>
        <pre className="mt-3 overflow-x-auto rounded-lg bg-surface p-3 text-sm">
          {`insert into public.admins (email) values ('${session.user.email}');`}
        </pre>
        <button className="btn btn-ghost mt-6" onClick={() => db.auth.signOut()}>Sair</button>
      </Shell>
    );
  }

  return (
    <AdminProvider value={{ db, session }}>
      <div className="min-h-full">
        <header className="sticky top-0 z-30 border-b border-line bg-bg/95 backdrop-blur">
          <div className="mx-auto flex max-w-5xl items-center gap-2 px-4 py-2">
            <span className="mr-2 font-display text-xl font-semibold">Painel</span>
            <nav className="no-scrollbar flex flex-1 gap-1 overflow-x-auto" aria-label="Painel">
              {NAV.map((n) => {
                const active = n.href === "/admin" ? pathname === "/admin" || pathname.startsWith("/admin/pratos") : pathname.startsWith(n.href);
                return (
                  <Link
                    key={n.href}
                    href={n.href}
                    aria-current={active ? "page" : undefined}
                    className={`flex min-h-11 shrink-0 items-center rounded-full px-3 text-sm ${active ? "bg-surface-2 text-ink" : "text-muted hover:text-ink"}`}
                  >
                    {n.label}
                  </Link>
                );
              })}
            </nav>
            <Link href="/" target="_blank" className="hidden min-h-11 items-center px-2 text-sm text-muted hover:text-ink sm:flex">
              Ver cardápio ↗
            </Link>
            <button className="min-h-11 px-2 text-sm text-muted hover:text-ink" onClick={() => db.auth.signOut()}>
              Sair
            </button>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-4 py-8">{children}</main>
      </div>
    </AdminProvider>
  );
}

function Login() {
  const db = browserClient();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const { error } = await db.auth.signInWithPassword({ email, password });
    if (error) setError("E-mail ou senha incorretos.");
    setBusy(false);
  }

  return (
    <Shell>
      <h1 className="font-display text-3xl">Entrar no painel</h1>
      <form onSubmit={submit} className="mt-6 flex max-w-sm flex-col gap-4">
        <label className="flex flex-col gap-1.5 text-sm">
          E-mail
          <input className="field" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label className="flex flex-col gap-1.5 text-sm">
          Senha
          <input className="field" type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} />
        </label>
        {error && <p className="text-sm text-danger" role="alert">{error}</p>}
        <button className="btn btn-primary" disabled={busy}>{busy ? "Entrando…" : "Entrar"}</button>
      </form>
    </Shell>
  );
}

function Setup() {
  return (
    <Shell>
      <h1 className="font-display text-3xl">Falta conectar o Supabase</h1>
      <ol className="mt-4 list-decimal space-y-2 pl-5 text-muted">
        <li>Crie um projeto em supabase.com.</li>
        <li>No SQL Editor, rode o arquivo <code>supabase/migrations/0001_init.sql</code>.</li>
        <li>Em Authentication &gt; Users, crie seu usuário (e-mail e senha).</li>
        <li>Adicione seu e-mail em <code>public.admins</code> (comando no fim do SQL).</li>
        <li>
          Copie a URL e a chave anon para <code>.env.local</code> (veja <code>.env.example</code>) e reinicie o servidor.
        </li>
      </ol>
    </Shell>
  );
}

function Shell({ children }: { children: ReactNode }) {
  return <main className="mx-auto w-full max-w-xl px-4 py-16">{children}</main>;
}
