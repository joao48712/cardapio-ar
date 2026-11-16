// PROTÓTIPO: só no servidor (usa o cache do cardápio)
import { getMenu } from "@/lib/menu";
import { mediaUrl } from "@/lib/supabase";
import { BASE, type ProtoDish } from "./data";

export async function getProtoMenu(): Promise<ProtoDish[]> {
  const menu = await getMenu();
  const real = menu.flatMap((c) => c.items).find((i) => i.model_path) ?? null;
  const photo = real ? mediaUrl("photos", real.photo_path) ?? mediaUrl("photos", real.poster_path) : null;
  const model = real ? mediaUrl("models", real.model_path) : "/seed/sobremesa.glb";
  return BASE.map((d) => ({ ...d, photo, model }));
}
