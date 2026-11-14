import Link from "next/link";

export default function NotFound() {
  return (
    <main className="mx-auto flex w-full max-w-xl flex-1 flex-col items-start justify-center px-4 py-24">
      <p className="text-xs font-semibold tracking-[0.18em] text-muted uppercase">Não encontrado</p>
      <h1 className="mt-3 font-display text-4xl font-medium">Este prato saiu do cardápio</h1>
      <p className="mt-3 text-muted">O link pode estar antigo. Veja o que está sendo servido hoje.</p>
      <Link href="/" className="btn btn-primary mt-8">Ver o cardápio</Link>
    </main>
  );
}
