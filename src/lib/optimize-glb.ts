// Otimiza um GLB de fotogrametria no navegador, antes do upload.
// Lê arquivos com compressão de geometria (Draco/meshopt), mas sempre grava sem ela,
// para manter compatível com Scene Viewer e Quick Look.
import { WebIO, type Document } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import {
  center,
  dedup,
  getBounds,
  getSceneVertexCount,
  prune,
  simplify,
  textureCompress,
  VertexCountMethod,
  weld,
} from "@gltf-transform/functions";
import { MeshoptDecoder, MeshoptSimplifier } from "meshoptimizer";

export type OptimizeOptions = {
  /** Triângulos máximos depois de simplificar. */
  maxTriangles?: number;
  /** Lado máximo das texturas, em pixels. */
  maxTexture?: number;
  /** Largura real do prato em cm. Se vier, o modelo é escalado para esse tamanho. */
  realWidthCm?: number | null;
};

export type ModelInfo = {
  bytes: number;
  triangles: number;
  /** Largura, altura e profundidade em cm. */
  sizeCm: [number, number, number];
};

export type OptimizeResult = {
  blob: Blob;
  before: ModelInfo;
  after: ModelInfo;
};

const DRACO = "KHR_draco_mesh_compression";
const MESHOPT = "EXT_meshopt_compression";

let draco: Promise<unknown> | null = null;

/** Decodificador Draco (arquivos em public/draco). Só é baixado quando o GLB precisa. */
function loadDraco(): Promise<unknown> {
  draco ??= (async () => {
    type Factory = (o: { wasmBinary: ArrayBuffer }) => Promise<unknown>;
    const w = window as unknown as { DracoDecoderModule?: Factory };
    if (!w.DracoDecoderModule) {
      await new Promise<void>((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "/draco/draco_wasm_wrapper.js";
        s.onload = () => resolve();
        s.onerror = () => reject(new Error("não foi possível baixar o decodificador Draco"));
        document.head.appendChild(s);
      });
    }
    const res = await fetch("/draco/draco_decoder.wasm");
    if (!res.ok || !w.DracoDecoderModule) throw new Error("não foi possível baixar o decodificador Draco");
    return w.DracoDecoderModule({ wasmBinary: await res.arrayBuffer() });
  })().catch((e) => {
    draco = null;
    throw e;
  });
  return draco;
}

/** Leitor com os decodificadores que este arquivo pede. */
async function io(bytes: Uint8Array) {
  const reader = new WebIO().registerExtensions(ALL_EXTENSIONS);
  const used = (await reader.binaryToJSON(bytes)).json.extensionsUsed ?? [];
  if (used.includes(DRACO)) reader.registerDependencies({ "draco3d.decoder": await loadDraco() });
  if (used.includes(MESHOPT)) {
    await MeshoptDecoder.ready;
    reader.registerDependencies({ "meshopt.decoder": MeshoptDecoder });
  }
  return reader;
}

function info(doc: Document, bytes: number): ModelInfo {
  const scene = doc.getRoot().getDefaultScene() ?? doc.getRoot().listScenes()[0];
  const triangles = scene ? Math.round(getSceneVertexCount(scene, VertexCountMethod.RENDER) / 3) : 0;
  const { min, max } = scene ? getBounds(scene) : { min: [0, 0, 0], max: [0, 0, 0] };
  const sizeCm = [0, 1, 2].map((i) => Math.round((max[i] - min[i]) * 1000) / 10) as [number, number, number];
  return { bytes, triangles, sizeCm };
}

export async function inspectGlb(file: Blob): Promise<ModelInfo> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const doc = await (await io(bytes)).readBinary(bytes);
  return info(doc, file.size);
}

export async function optimizeGlb(file: Blob, opts: OptimizeOptions = {}): Promise<OptimizeResult> {
  const { maxTriangles = 60_000, maxTexture = 2048, realWidthCm = null } = opts;
  await MeshoptSimplifier.ready;

  const bytes = new Uint8Array(await file.arrayBuffer());
  const reader = await io(bytes);
  const doc = await reader.readBinary(bytes);
  const before = info(doc, file.size);

  // A geometria já foi descomprimida na leitura; tira a marca de compressão para gravar sem ela
  for (const ext of doc.getRoot().listExtensionsUsed()) {
    if (ext.extensionName === DRACO || ext.extensionName === MESHOPT) ext.dispose();
  }

  await doc.transform(dedup(), weld());

  if (before.triangles > maxTriangles) {
    const ratio = Math.max(0.01, maxTriangles / before.triangles);
    await doc.transform(simplify({ simplifier: MeshoptSimplifier, ratio, error: 0.002 }));
  }

  await doc.transform(
    textureCompress({ targetFormat: "jpeg", resize: [maxTexture, maxTexture], quality: 85 }),
    prune(),
  );

  // Escala para o tamanho real, se pedido (usa a maior medida horizontal)
  if (realWidthCm && realWidthCm > 0) {
    const current = Math.max(before.sizeCm[0], before.sizeCm[2]);
    if (current > 0) {
      const k = realWidthCm / current;
      for (const scene of doc.getRoot().listScenes()) {
        for (const node of scene.listChildren()) {
          const [x, y, z] = node.getScale();
          const [tx, ty, tz] = node.getTranslation();
          node.setScale([x * k, y * k, z * k]).setTranslation([tx * k, ty * k, tz * k]);
        }
      }
    }
  }

  // Origem no centro da base: o prato pousa certinho na mesa no AR
  await doc.transform(center({ pivot: "below" }));

  const out = await reader.writeBinary(doc);
  const blob = new Blob([out as BlobPart], { type: "model/gltf-binary" });
  return { blob, before, after: info(doc, blob.size) };
}
