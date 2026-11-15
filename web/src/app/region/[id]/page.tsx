import fs from "node:fs";
import path from "node:path";
import RegionExplorer from "@/components/RegionExplorer";

// Pre-render one page per region listed in the exported catalog (static export).
export function generateStaticParams() {
  const p = path.join(process.cwd(), "public", "data", "catalog.json");
  const cat = JSON.parse(fs.readFileSync(p, "utf8")) as { regions: { id: string }[] };
  return cat.regions.map((r) => ({ id: r.id }));
}

export const dynamicParams = false;

export default async function RegionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <RegionExplorer id={id} />;
}
