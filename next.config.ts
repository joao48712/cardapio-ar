import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  images: {
    // Fotos dos pratos ficam no Supabase Storage (bucket público "photos")
    remotePatterns: [
      { protocol: "https", hostname: "*.supabase.co", pathname: "/storage/v1/object/public/photos/**" },
    ],
  },
  // Permite abrir o servidor de desenvolvimento pelo celular na rede local
  allowedDevOrigins: ["192.168.*.*", "10.*.*.*", "172.*.*.*"],
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
