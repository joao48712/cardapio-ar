"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { ModelViewer, type ModelViewerElement } from "@/components/ModelViewer";
import { formatBytes, slugify } from "@/lib/format";
import { optimizeGlb, type OptimizeResult } from "@/lib/optimize-glb";
import { mediaUrl } from "@/lib/supabase";
import type { Category, Item } from "@/lib/types";
import { useAdmin } from "../../AdminContext";

const MAX_MODEL_BYTES = 5 * 1024 * 1024;

type Form = {
  name: string;
  description: string;
  price: string;
  category_id: string;
  tags: string;
  active: boolean;
};

export function View() {
  const { id } = useParams<{ id: string }>();
  const isNew = id === "novo";
  const router = useRouter();
  const { db, publish } = useAdmin();

  const [itemId] = useState(() => (isNew ? crypto.randomUUID() : id));
  const [cats, setCats] = useState<Category[]>([]);
  const [item, setItem] = useState<Item | null>(null);
  const [form, setForm] = useState<Form>({ name: "", description: "", price: "", category_id: "", tags: "", active: true });
  const [photo, setPhoto] = useState<Blob | null>(null);
  const [model, setModel] = useState<OptimizeResult | null>(null);
  const [modelPreview, setModelPreview] = useState<string | null>(null);
  const [mv, setMv] = useState<ModelViewerElement | null>(null);
  const [realWidth, setRealWidth] = useState("");
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    db.from("categories").select("*").order("sort").then(({ data }) => {
      const list = (data ?? []) as Category[];
      setCats(list);
      if (isNew && list[0]) setForm((f) => ({ ...f, category_id: f.category_id || list[0].id }));
    });
    if (!isNew) {
      db.from("items").select("*").eq("id", id).single().then(({ data, error }) => {
        if (error) return setError(error.message);
        const it = data as Item;
        setItem(it);
        setForm({
          name: it.name,
          description: it.description,
          price: (it.price_cents / 100).toFixed(2).replace(".", ","),
          category_id: it.category_id,
          tags: it.tags.join(", "),
          active: it.active,
        });
      });
    }
  }, [db, id, isNew]);

  useEffect(() => () => { if (modelPreview) URL.revokeObjectURL(modelPreview); }, [modelPreview]);

  const processModel = useCallback(async (file: File, widthCm: number | null) => {
    setError(null);
    setStatus("Otimizando o modelo 3D… pode levar alguns segundos.");
    try {
      const result = await optimizeGlb(file, { realWidthCm: widthCm });
      setModel(result);
      setModelPreview(URL.createObjectURL(result.blob));
      setMv(null);
      if (!widthCm) setRealWidth(String(Math.max(result.after.sizeCm[0], result.after.sizeCm[2])));
      setStatus(null);
    } catch (e) {
      setStatus(null);
      setError(`Não deu para ler o arquivo 3D: ${(e as Error).message}`);
    }
  }, []);

  async function onPhoto(file: File | undefined) {
    if (!file) return;
    setPhoto(await resizeImage(file, 1600));
  }

  async function onModel(file: File | undefined) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".glb")) {
      setError("Envie um arquivo .glb (exporte em GLB no RealityScan ou Polycam).");
      return;
    }
    setModelFile(file);
    await processModel(file, null);
  }

  async function upload(bucket: "photos" | "models", path: string, blob: Blob, contentType: string) {
    const { error } = await db.storage.from(bucket).upload(path, blob, { contentType, cacheControl: "31536000", upsert: false });
    if (error) throw new Error(`Upload falhou (${bucket}): ${error.message}`);
    return path;
  }

  async function save(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const price = Math.round(parseFloat(form.price.replace(/\./g, "").replace(",", ".")) * 100);
    if (!form.name.trim() || !form.category_id || Number.isNaN(price)) {
      setError("Preencha nome, categoria e preço.");
      return;
    }
    if (model && model.after.bytes > MAX_MODEL_BYTES * 2) {
      setError(`Modelo ainda muito pesado (${formatBytes(model.after.bytes)}). Exporte com menos detalhe no app.`);
      return;
    }

    setBusy(true);
    try {
      const ts = Date.now();
      const patch: Partial<Item> & Record<string, unknown> = {
        name: form.name.trim(),
        slug: item?.slug ?? `${slugify(form.name)}-${itemId.slice(0, 4)}`,
        description: form.description.trim(),
        price_cents: price,
        category_id: form.category_id,
        tags: form.tags.split(",").map((t) => t.trim().toLowerCase()).filter(Boolean),
        active: form.active,
      };

      if (photo) {
        setStatus("Enviando foto…");
        patch.photo_path = await upload("photos", `${itemId}/foto-${ts}.jpg`, photo, "image/jpeg");
      }
      if (model) {
        setStatus("Enviando modelo 3D…");
        patch.model_path = await upload("models", `${itemId}/modelo-${ts}.glb`, model.blob, "model/gltf-binary");
        patch.model_size_bytes = model.after.bytes;
        if (mv) {
          const poster = await mv.toBlob({ mimeType: "image/webp", qualityArgument: 0.85, idealAspect: false });
          patch.poster_path = await upload("photos", `${itemId}/poster-${ts}.webp`, poster, "image/webp");
        }
      }

      setStatus("Salvando…");
      const { error } = isNew
        ? await db.from("items").insert({ id: itemId, sort: 999, ...patch })
        : await db.from("items").update(patch).eq("id", itemId);
      if (error) throw new Error(error.message);

      await publish();
      router.push("/admin");
    } catch (err) {
      setError((err as Error).message);
      setStatus(null);
      setBusy(false);
    }
  }

  async function remove() {
    if (!item || !confirm(`Apagar "${item.name}"? Não dá para desfazer.`)) return;
    setBusy(true);
    const { error } = await db.from("items").delete().eq("id", item.id);
    if (error) {
      setError(error.message);
      setBusy(false);
      return;
    }
    await db.storage.from("photos").remove([item.photo_path, item.poster_path].filter(Boolean) as string[]);
    if (item.model_path) await db.storage.from("models").remove([item.model_path]);
    await publish();
    router.push("/admin");
  }

  const currentModel = modelPreview ?? mediaUrl("models", item?.model_path ?? null);
  const photoPreview = useMemo(() => (photo ? URL.createObjectURL(photo) : null), [photo]);
  useEffect(() => () => { if (photoPreview) URL.revokeObjectURL(photoPreview); }, [photoPreview]);
  const currentPhoto = photoPreview ?? mediaUrl("photos", item?.photo_path ?? null);

  return (
    <form onSubmit={save} className="grid gap-8 lg:grid-cols-[1fr_1fr]">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <Link href="/admin" className="text-sm text-muted hover:text-ink">← Pratos</Link>
        </div>
        <h1 className="font-display text-3xl">{isNew ? "Novo prato" : form.name || "Editar prato"}</h1>

        <label className="flex flex-col gap-1.5 text-sm">
          Nome
          <input className="field" required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
        </label>
        <label className="flex flex-col gap-1.5 text-sm">
          Descrição
          <textarea className="field min-h-24" rows={3} value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
        </label>
        <div className="grid grid-cols-2 gap-3">
          <label className="flex flex-col gap-1.5 text-sm">
            Preço (R$)
            <input className="field" inputMode="decimal" placeholder="49,90" required value={form.price} onChange={(e) => setForm({ ...form, price: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1.5 text-sm">
            Categoria
            <select className="field" required value={form.category_id} onChange={(e) => setForm({ ...form, category_id: e.target.value })}>
              {cats.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </label>
        </div>
        <label className="flex flex-col gap-1.5 text-sm">
          Selos (separados por vírgula)
          <input className="field" placeholder="vegano, sem glúten" value={form.tags} onChange={(e) => setForm({ ...form, tags: e.target.value })} />
        </label>
        <label className="flex min-h-11 items-center gap-2 text-sm">
          <input type="checkbox" className="h-4 w-4 accent-[var(--accent)]" checked={form.active} onChange={(e) => setForm({ ...form, active: e.target.checked })} />
          Visível no cardápio
        </label>

        <fieldset className="rounded-[var(--radius)] border border-line p-4">
          <legend className="px-1 text-sm">Foto</legend>
          {currentPhoto && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={currentPhoto} alt="" className="mb-3 aspect-square w-32 rounded-lg object-cover" />
          )}
          <input type="file" accept="image/*" onChange={(e) => onPhoto(e.target.files?.[0])} className="text-sm" />
          <p className="mt-2 text-xs text-muted">A foto é reduzida para 1600 px antes de enviar.</p>
        </fieldset>

        {error && <p className="text-sm text-danger" role="alert">{error}</p>}
        {status && <p className="text-sm text-muted" aria-live="polite">{status}</p>}

        <div className="flex flex-wrap gap-3">
          <button className="btn btn-primary" disabled={busy}>{busy ? "Salvando…" : "Salvar"}</button>
          {!isNew && <button type="button" className="btn btn-ghost" disabled={busy} onClick={remove}>Apagar prato</button>}
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <fieldset className="rounded-[var(--radius)] border border-line p-4">
          <legend className="px-1 text-sm">Modelo 3D (.glb)</legend>
          <input type="file" accept=".glb,model/gltf-binary" onChange={(e) => onModel(e.target.files?.[0])} className="text-sm" />
          <p className="mt-2 text-xs text-muted">
            Exporte em GLB no RealityScan ou Polycam. O painel reduz o arquivo e põe a base do prato no chão.{" "}
            <Link href="/admin/guia-captura" className="text-accent underline">Guia de captura</Link>
          </p>

          {model && (
            <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
              <Stat label="Antes" info={model.before} />
              <Stat label="Depois" info={model.after} warn={model.after.bytes > MAX_MODEL_BYTES} />
            </div>
          )}

          {model && modelFile && (
            <div className="mt-4 flex items-end gap-2">
              <label className="flex flex-1 flex-col gap-1.5 text-sm">
                Largura real do prato (cm)
                <input className="field" inputMode="decimal" value={realWidth} onChange={(e) => setRealWidth(e.target.value)} />
              </label>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => processModel(modelFile, parseFloat(realWidth.replace(",", ".")) || null)}
              >
                Ajustar escala
              </button>
            </div>
          )}
        </fieldset>

        {currentModel ? (
          <ModelViewer
            key={currentModel}
            src={currentModel}
            alt="Prévia do modelo"
            deferred={false}
            onReady={setMv}
            className="aspect-square w-full"
          />
        ) : (
          <div className="flex aspect-square w-full items-center justify-center rounded-[var(--radius)] border border-dashed border-line text-sm text-muted">
            Sem modelo 3D ainda
          </div>
        )}
        {model && <p className="text-xs text-muted">A imagem de capa do 3D é tirada desta prévia ao salvar. Gire o prato para o melhor ângulo antes.</p>}
      </div>
    </form>
  );
}

function Stat({ label, info, warn }: { label: string; info: OptimizeResult["before"]; warn?: boolean }) {
  return (
    <div className={`rounded-lg bg-surface-2 p-3 ${warn ? "ring-1 ring-danger" : ""}`}>
      <p className="font-semibold">{label}</p>
      <p className="mt-1 text-muted">{formatBytes(info.bytes)}</p>
      <p className="text-muted">{info.triangles.toLocaleString("pt-BR")} triângulos</p>
      <p className="text-muted">{info.sizeCm[0]} × {info.sizeCm[2]} × {info.sizeCm[1]} cm</p>
      {warn && <p className="mt-1 text-danger">Pesado para 4G. Exporte com menos detalhe.</p>}
    </div>
  );
}

async function resizeImage(file: File, max: number): Promise<Blob> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, max / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Falha ao converter a foto"))), "image/jpeg", 0.85),
  );
}
