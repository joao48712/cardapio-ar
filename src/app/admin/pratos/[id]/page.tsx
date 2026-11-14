import { Suspense } from "react";
import { View } from "./View";

export const instant = false;

export default function Page() {
  return (
    <Suspense fallback={<p className="text-muted">Carregando…</p>}>
      <View />
    </Suspense>
  );
}
