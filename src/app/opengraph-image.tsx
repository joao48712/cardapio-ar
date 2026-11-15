import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";
import { RESTAURANT } from "@/lib/restaurant";

// Prévia do link no WhatsApp, Instagram e Facebook
export const alt = `${RESTAURANT.name}: cardápio com realidade aumentada`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const logo = await readFile(join(process.cwd(), "public/brand/logo-153.png"));
const logoSrc = `data:image/png;base64,${logo.toString("base64")}`;

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: "#32211a",
          color: "#f6ede0",
        }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={logoSrc} width={300} height={300} alt="" />
        <div style={{ marginTop: 24, fontSize: 44, color: "#dcbc7c" }}>Cardápio com realidade aumentada</div>
        <div style={{ marginTop: 12, fontSize: 30, color: "#c9b79f" }}>Veja o prato na sua mesa, no tamanho real</div>
      </div>
    ),
    { ...size },
  );
}
