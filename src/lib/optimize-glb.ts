// Otimiza um GLB de fotogrametria no navegador, antes do upload.
// Sem compressão de geometria (Draco/meshopt) para manter compatível com Scene Viewer e Quick Look.
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
import { MeshoptSimplifier } from "meshoptimizer";

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

function io() {
  return new WebIO().registerExtensions(ALL_EXTENSIONS);
}

function info(doc: Document, bytes: number): ModelInfo {
  const scene = doc.getRoot().getDefaultScene() ?? doc.getRoot().listScenes()[0];
  const triangles = scene ? Math.round(getSceneVertexCount(scene, VertexCountMethod.RENDER) / 3) : 0;
  const { min, max } = scene ? getBounds(scene) : { min: [0, 0, 0], max: [0, 0, 0] };
  const sizeCm = [0, 1, 2].map((i) => Math.round((max[i] - min[i]) * 1000) / 10) as [number, number, number];
  return { bytes, triangles, sizeCm };
}

export async function inspectGlb(file: Blob): Promise<ModelInfo> {
  const doc = await io().readBinary(new Uint8Array(await file.arrayBuffer()));
  return info(doc, file.size);
}

export async function optimizeGlb(file: Blob, opts: OptimizeOptions = {}): Promise<OptimizeResult> {
  const { maxTriangles = 60_000, maxTexture = 2048, realWidthCm = null } = opts;
  await MeshoptSimplifier.ready;

  const reader = io();
  const doc = await reader.readBinary(new Uint8Array(await file.arrayBuffer()));
  const before = info(doc, file.size);

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
