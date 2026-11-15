"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { type RegionEntry, getCatalog } from "@/lib/data";

export function Logo({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <circle cx="16" cy="16" r="16" fill="#1b2430" />
      <path d="M7 17a9 9 0 0 1 18 0" stroke="#3b82f6" strokeWidth="2.6" fill="none" strokeLinecap="round" />
      <path d="M11 20a5 5 0 0 1 10 0" stroke="#f59e0b" strokeWidth="2.6" fill="none" strokeLinecap="round" />
    </svg>
  );
}



export default function Header() {
  const path = usePathname() ?? "/";
  const router = useRouter();
  const [regions, setRegions] = useState<RegionEntry[]>([]);
  useEffect(() => {
    getCatalog().then((c) => setRegions(c.regions));
  }, []);
  const current = path.match(/\/(region|report)\/([^/]+)/)?.[2] ?? "";
  const processed = regions.filter((r) => r.status === "processed");
  const exploreId = current || (regions.find((r) => r.priority === "hero") ?? processed[0])?.id || "";
  const hazardId = (regions.find((r) => r.id === current && r.hazards) ?? regions.find((r) => r.hazards))?.id;
  const onPrediction = path.startsWith("/hazards");
  const TABS: [string, string, boolean][] = [
    [`/region/${exploreId}/`, "Explore map", !onPrediction],
    [hazardId ? `/hazards/${hazardId}/` : "/hazards/", "Prediction", onPrediction],
  ];
  return (
    <header className="no-print sticky top-0 z-40 border-b border-line bg-white">
      <div className="flex h-16 items-center gap-3 px-4">
        <Link href="/" className="flex shrink-0 items-center gap-2.5">
          <Logo />
          <div className="leading-tight">
            <div className="text-[15px] font-bold">Himalayan EarthPulse</div>
            <div className="hidden text-xs text-muted sm:block">Earth&apos;s surface, measured through the clouds</div>
          </div>
        </Link>
        <label className="ml-2 hidden items-center gap-2 rounded-xl border border-line bg-white px-3 py-2 text-sm md:flex">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s7-6.2 7-12a7 7 0 1 0-14 0c0 5.8 7 12 7 12z" /><circle cx="12" cy="10" r="2.5" /></svg>
          <select aria-label="Region" className="bg-transparent pr-1 font-medium outline-none" value={current}
            onChange={(e) => e.target.value && router.push(`/region/${e.target.value}/`)}>
            <option value="">Choose a place…</option>
            {processed.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
          </select>
        </label>
        <div className="flex-1" />
        <nav className="flex rounded-xl bg-panel-2 p-1 text-sm" aria-label="Main">
          {TABS.map(([href, label, on]) => (
            <Link key={label} href={href} className={`rounded-lg px-3.5 py-1.5 whitespace-nowrap ${on ? "bg-white font-semibold shadow-sm" : "text-muted hover:text-text"}`}>{label}</Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
