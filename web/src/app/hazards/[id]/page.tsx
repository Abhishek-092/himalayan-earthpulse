import fs from "node:fs";
import path from "node:path";
import Hazards from "@/components/Hazards";

// Only regions with an exported Phase 2 hazard model get a dashboard.
export function generateStaticParams() {
  const dir = path.join(process.cwd(), "public", "data");
  const cat = JSON.parse(fs.readFileSync(path.join(dir, "catalog.json"), "utf8")) as { regions: { id: string }[] };
  return cat.regions.filter((r) => fs.existsSync(path.join(dir, r.id, "hazards", "manifest.json"))).map((r) => ({ id: r.id }));
}

export const dynamicParams = false;

export default async function HazardPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <Hazards id={id} />;
}
