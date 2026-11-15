import type { NextConfig } from "next";

// Static export: the whole app (incl. pre-processed NISAR results in public/data) is served from a CDN,
// so the demo opens from a public link without login and loads fast.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  turbopack: {
    rules: {
      "*.css": { loaders: ["@tailwindcss/turbopack"], as: "*.css" },
    },
  },
};

export default nextConfig;
