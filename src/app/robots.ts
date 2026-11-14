import type { MetadataRoute } from "next";
import { RESTAURANT } from "@/lib/restaurant";

export default function robots(): MetadataRoute.Robots {
  if (!RESTAURANT.indexable) return { rules: { userAgent: "*", disallow: "/" } };
  return { rules: { userAgent: "*", allow: "/", disallow: "/admin" } };
}
