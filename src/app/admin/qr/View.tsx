"use client";

import QRCode from "qrcode";
import { useEffect, useState } from "react";
import { RESTAURANT } from "@/lib/restaurant";

export function View() {
  const [base, setBase] = useState(process.env.NEXT_PUBLIC_SITE_URL ?? "");
  const [tables, setTables] = useState(10);
  const [codes, setCodes] = useState<{ n: number; svg: string }[]>([]);

  useEffect(() => {
    const origin = base || window.location.origin;
    const n = Math.max(1, Math.min(200, tables || 1));
    Promise.all(
      Array.from({ length: n }, (_, i) =>
        // ?mesa=N já fica no QR para o pedido pelo celular (fase 2) sem reimprimir
        QRCode.toString(`${origin}/?mesa=${i + 1}`, { type: "svg", margin: 1, errorCorrectionLevel: "M", color: { dark: "#2a1a12", light: "#ffffff" } })
          .then((svg) => ({ n: i + 1, svg })),
      ),
    ).then(setCodes);
  }, [base, tables]);

  return (
    <div>
      <div className="print:hidden">
        <h1 className="font-display text-3xl">QR codes das mesas</h1>
        <div className="mt-6 flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1.5 text-sm">
            Endereço do site
            <input className="field w-80 max-w-full" placeholder="https://seu-cardapio.vercel.app" value={base} onChange={(e) => setBase(e.target.value.replace(/\/$/, ""))} />
          </label>
          <label className="flex flex-col gap-1.5 text-sm">
            Mesas
            <input className="field w-24" type="number" min={1} max={200} value={tables} onChange={(e) => setTables(parseInt(e.target.value, 10))} />
          </label>
          <button className="btn btn-primary" onClick={() => window.print()}>Imprimir</button>
        </div>
        <p className="mt-3 text-sm text-muted">
          Use o endereço público (o da Vercel), não o localhost. Cada QR leva o número da mesa.
        </p>
      </div>

      <div className="mt-8 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 print:mt-0 print:grid-cols-3">
        {codes.map(({ n, svg }) => (
          <div
            key={n}
            className="flex break-inside-avoid flex-col items-center rounded-xl bg-[#32211a] p-4 text-center text-[#f6ede0] [print-color-adjust:exact]"
          >
            {RESTAURANT.logo ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={RESTAURANT.logo} alt={RESTAURANT.name} className="logo-fade h-20 w-20" />
            ) : (
              <p className="font-display text-lg font-semibold">{RESTAURANT.name}</p>
            )}
            <div className="mt-1 w-full max-w-40 rounded-lg bg-white p-1.5" dangerouslySetInnerHTML={{ __html: svg }} />
            <p className="mt-3 text-xs leading-snug text-[#c9b79f]">Aponte a câmera e veja o prato na sua mesa</p>
            <p className="mt-1 font-display text-lg text-[#dcbc7c]">Mesa {n}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
