// Otimiza um GLB de fotogrametria no navegador, antes do upload.
// Lê arquivos com compressão de geometria (Draco/meshopt), mas sempre grava sem ela,
// para manter compatível com Scene Viewer e Quick Look.
import { GLB_BUFFER, WebIO, type Document, type JSONDocument } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import {
  center,
  dedup,
  dequantize,
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
  /** Lado máximo da textura de cor, em pixels. */
  maxTexture?: number;
  /** Lado máximo das texturas de detalhe (relevo, rugosidade, oclusão). */
  maxDetailTexture?: number;
  /** Largura real do prato em cm. Se vier, o modelo é escalado para esse tamanho. */
  realWidthCm?: number | null;
  /** Imagens de textura exportadas ao lado do .glb (quando o app não embute as texturas). */
  textures?: File[];
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

const baseName = (path: string) => path.split(/[\/]/).pop()!.toLowerCase();

/**
 * Abre o contêiner GLB. Feito à mão porque o leitor da biblioteca recusa GLB com
 * texturas em arquivos separados (caso do RealityScan sem "embutir texturas").
 */
async function parseGlb(bytes: Uint8Array<ArrayBuffer>,textures: File[]): Promise<JSONDocument> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67) throw new Error("o arquivo não é um GLB válido.");
  if (view.getUint32(4, true) !== 2) throw new Error("versão de GLB não suportada (precisa ser glTF 2.0).");

  let json: JSONDocument["json"] | null = null;
  const resources: JSONDocument["resources"] = {};
  for (let offset = 12; offset + 8 <= bytes.byteLength; ) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const chunk = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(chunk));
    else if (type === 0x004e4942) resources[GLB_BUFFER] = chunk;
    offset += 8 + length;
  }
  if (!json) throw new Error("o arquivo não é um GLB válido.");

  if (json.buffers?.some((b) => b.uri && !b.uri.startsWith("data:"))) {
    throw new Error("a geometria está em um arquivo .bin separado. Exporte de novo como um único .glb.");
  }

  const missing: string[] = [];
  for (const image of json.images ?? []) {
    if (!image.uri || image.uri.startsWith("data:")) continue;
    let name = image.uri;
    try {
      name = decodeURIComponent(image.uri);
    } catch {}
    const file = textures.find((t) => baseName(t.name) === baseName(name));
    if (file) resources[image.uri] = new Uint8Array(await file.arrayBuffer());
    else missing.push(baseName(name));
  }
  if (missing.length > 0) {
    const list = [...new Set(missing)].join(", ");
    throw new Error(
      `este .glb guarda as texturas em arquivos separados (${list}). Selecione o .glb junto com essas imagens (Ctrl+clique), ou exporte de novo com as texturas embutidas.`,
    );
  }
  return { json, resources };
}

/** Lê o GLB com os decodificadores que ele pede. */
async function read(file: Blob, textures: File[] = []) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const jsonDoc = await parseGlb(bytes, textures);
  const reader = new WebIO().registerExtensions(ALL_EXTENSIONS);
  const used = jsonDoc.json.extensionsUsed ?? [];
  if (used.includes(DRACO)) reader.registerDependencies({ "draco3d.decoder": await loadDraco() });
  if (used.includes(MESHOPT)) {
    await MeshoptDecoder.ready;
    reader.registerDependencies({ "meshopt.decoder": MeshoptDecoder });
  }
  return { reader, doc: await reader.readJSON(jsonDoc) };
}

function info(doc: Document, bytes: number): ModelInfo {
  const scene = doc.getRoot().getDefaultScene() ?? doc.getRoot().listScenes()[0];
  const triangles = scene ? Math.round(getSceneVertexCount(scene, VertexCountMethod.RENDER) / 3) : 0;
  const { min, max } = scene ? getBounds(scene) : { min: [0, 0, 0], max: [0, 0, 0] };
  const sizeCm = [0, 1, 2].map((i) => Math.round((max[i] - min[i]) * 1000) / 10) as [number, number, number];
  return { bytes, triangles, sizeCm };
}

export async function inspectGlb(file: Blob): Promise<ModelInfo> {
  const { doc } = await read(file);
  return info(doc, file.size);
}

export async function optimizeGlb(file: Blob, opts: OptimizeOptions = {}): Promise<OptimizeResult> {
  // 40k triângulos e relevo em 1024 px: roda liso em Android intermediário (testado no AR)
  const {
    maxTriangles = 40_000,
    maxTexture = 2048,
    maxDetailTexture = 1024,
    realWidthCm = null,
    textures = [],
  } = opts;
  await MeshoptSimplifier.ready;

  const { reader, doc } = await read(file, textures);
  const before = info(doc, file.size + textures.reduce((sum, t) => sum + t.size, 0));

  // A geometria já foi descomprimida na leitura; tira a marca de compressão para gravar sem ela
  for (const ext of doc.getRoot().listExtensionsUsed()) {
    if (ext.extensionName === DRACO || ext.extensionName === MESHOPT) ext.dispose();
  }

  // dequantize: desfaz a quantização que acompanha o meshopt, para o arquivo final não exigir extensões
  await doc.transform(dequantize(), dedup(), weld());

  if (before.triangles > maxTriangles) {
    const ratio = Math.max(0.01, maxTriangles / before.triangles);
    await doc.transform(simplify({ simplifier: MeshoptSimplifier, ratio, error: 0.002 }));
  }

  await doc.transform(
    // Cor do prato: mantém a nitidez
    textureCompress({ slots: /^baseColorTexture$/, targetFormat: "jpeg", resize: [maxTexture, maxTexture], quality: 85 }),
    // Relevo, rugosidade e oclusão: quase não aparecem na mesa, então podem ser menores
    textureCompress({ slots: /^(?!baseColorTexture$).*$/, targetFormat: "jpeg", resize: [maxDetailTexture, maxDetailTexture], quality: 85 }),
    prune(),
  );

  // Fotogrametria fecha a superfície: desenhar o lado de dentro só gasta GPU
  for (const material of doc.getRoot().listMaterials()) material.setDoubleSided(false);

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
