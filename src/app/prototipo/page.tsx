// PROTÓTIPO de design (descartável). Não vai para produção: vive só na branch prototipo-design.
import type { Metadata } from "next";
import Image from "next/image";
import { SkyHero } from "@/components/sky/SkyHero";
import { RESTAURANT } from "@/lib/restaurant";
import { BottomBar } from "./BottomBar";
import { CATEGORIES } from "./data";
import { getProtoMenu } from "./menu.server";
import { Featured } from "./Featured";
import { MenuList } from "./MenuList";

export const metadata: Metadata = {
  title: `Protótipo | ${RESTAURANT.name}`,
  robots: { index: false, follow: false },
};

export default async function PrototypePage() {
  const dishes = await getProtoMenu();
  const featured = dishes.filter((d) => d.featured);

  return (
    <>
      <SkyHero controls>
        {RESTAURANT.logo ? (
          <Image
            src={RESTAURANT.logo}
            alt={RESTAURANT.name}
            width={150}
            height={150}
            priority
            className="logo-fade h-[132px] w-[132px]"
          />
        ) : (
          <h1 className="font-display text-5xl">{RESTAURANT.name}</h1>
        )}
      </SkyHero>

      <main className="flex-1">
        <Featured dishes={featured} />
        <MenuList dishes={dishes} categories={CATEGORIES} />
        <p className="mx-auto mt-12 max-w-3xl px-4 text-center text-xs text-muted">
          Protótipo de design · fotos e 3D provisórios
        </p>
      </main>

      <BottomBar dishes={dishes} categories={CATEGORIES} />
    </>
  );
}
