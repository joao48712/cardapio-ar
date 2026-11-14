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
        QRCode.toString(`${origin}/?mesa=${i + 1}`, { type: "svg", margin: 1, errorCorrectionLevel: "M", color: { dark: "#12100e", light: "#ffffff" } })
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
          <div key={n} className="flex break-inside-avoid flex-col items-center rounded-xl bg-white p-4 text-center text-[#12100e]">
            <p className="font-display text-lg font-semibold">{RESTAURANT.name}</p>
            <div className="mt-2 w-full max-w-40" dangerouslySetInnerHTML={{ __html: svg }} />
            <p className="mt-2 text-xs">Aponte a câmera e veja os pratos em 3D</p>
            <p className="mt-1 text-sm font-semibold">Mesa {n}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
