import type { Metadata, Viewport } from "next";
import { Cormorant_Garamond, Manrope } from "next/font/google";
import "./globals.css";
import { RESTAURANT } from "@/lib/restaurant";

// Fontes provisórias. Troque pelas fontes da marca quando chegarem.
const display = Cormorant_Garamond({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "600"],
});

const body = Manrope({
  variable: "--font-body",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: `${RESTAURANT.name} | Cardápio`,
  description: RESTAURANT.description,
  openGraph: {
    title: `${RESTAURANT.name} | Cardápio`,
    description: RESTAURANT.description,
    type: "website",
    locale: "pt_BR",
  },
  robots: RESTAURANT.indexable ? undefined : { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: "#32211a",
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="pt-BR" className={`${display.variable} ${body.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
