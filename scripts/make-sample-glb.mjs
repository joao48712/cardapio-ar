// Gera public/seed/sobremesa.glb: prato + bolo + bola de sorvete, em escala real (metros).
// Uso: node scripts/make-sample-glb.mjs
import { Document, NodeIO } from "@gltf-transform/core";
import { mkdirSync } from "node:fs";

const doc = new Document();
const buffer = doc.createBuffer();
const scene = doc.createScene("Prato");

function material(name, rgb, roughness = 0.6, metallic = 0) {
  return doc.createMaterial(name).setBaseColorFactor([...rgb, 1]).setRoughnessFactor(roughness).setMetallicFactor(metallic).setDoubleSided(true);
}

function addMesh(name, { positions, normals, indices }, mat, y = 0) {
  const prim = doc
    .createPrimitive()
    .setAttribute("POSITION", doc.createAccessor().setType("VEC3").setArray(new Float32Array(positions)).setBuffer(buffer))
    .setAttribute("NORMAL", doc.createAccessor().setType("VEC3").setArray(new Float32Array(normals)).setBuffer(buffer))
    .setIndices(doc.createAccessor().setType("SCALAR").setArray(new Uint16Array(indices)).setBuffer(buffer))
    .setMaterial(mat);
  const mesh = doc.createMesh(name).addPrimitive(prim);
  scene.addChild(doc.createNode(name).setMesh(mesh).setTranslation([0, y, 0]));
}

// Superfície de revolução a partir de um perfil [raio, altura]
function lathe(profile, segments = 64) {
  const positions = [], normals = [], indices = [];
  const rings = profile.length;
  for (let i = 0; i < rings; i++) {
    const [r, h] = profile[i];
    const prev = profile[Math.max(i - 1, 0)], next = profile[Math.min(i + 1, rings - 1)];
    const dr = next[0] - prev[0], dh = next[1] - prev[1];
    let nr = dh, nh = -dr; // normal 2D do perfil
    const len = Math.hypot(nr, nh) || 1; nr /= len; nh /= len;
    for (let s = 0; s <= segments; s++) {
      const a = (s / segments) * Math.PI * 2;
      const c = Math.cos(a), sn = Math.sin(a);
      positions.push(r * c, h, r * sn);
      normals.push(nr * c, nh, nr * sn);
    }
  }
  const row = segments + 1;
  for (let i = 0; i < rings - 1; i++) {
    for (let s = 0; s < segments; s++) {
      const a = i * row + s, b = a + row;
      indices.push(a, a + 1, b, b, a + 1, b + 1);
    }
  }
  return { positions, normals, indices };
}

// Prato: 26 cm de diâmetro, com aba levemente subida
const plate = lathe([
  [0, 0.004], [0.08, 0.004], [0.095, 0.006], [0.12, 0.014], [0.13, 0.016],
  [0.13, 0.012], [0.118, 0.009], [0.09, 0.0], [0.0, 0.0],
].reverse());
addMesh("Prato", plate, material("Porcelana", [0.93, 0.92, 0.89], 0.25));

// Bolo de chocolate: 8 cm de diâmetro, 4,5 cm de altura
const cake = lathe([
  [0, 0], [0.04, 0], [0.041, 0.004], [0.04, 0.04], [0.036, 0.045], [0, 0.046],
]);
addMesh("Bolo", cake, material("Chocolate", [0.22, 0.11, 0.06], 0.55), 0.005);

// Bola de sorvete de baunilha ao lado
const scoopProfile = [];
for (let i = 0; i <= 16; i++) {
  const t = (i / 16) * Math.PI;
  scoopProfile.push([Math.sin(t) * 0.025, -Math.cos(t) * 0.025 + 0.025]);
}
const scoop = lathe(scoopProfile, 48);
scoop.positions = scoop.positions.map((v, i) => (i % 3 === 0 ? v + 0.075 : v));
addMesh("Sorvete", scoop, material("Baunilha", [0.96, 0.9, 0.74], 0.7), 0.003);

mkdirSync("public/seed", { recursive: true });
await new NodeIO().write("public/seed/sobremesa.glb", doc);
console.log("ok: public/seed/sobremesa.glb");
