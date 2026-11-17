import type { Metadata } from "next";
import { View } from "./View";

export const metadata: Metadata = { title: "Diagnóstico de AR", robots: { index: false, follow: false } };

export default function Page() {
  return <View />;
}
