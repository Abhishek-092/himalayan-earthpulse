"use client";

import { fmt } from "@/lib/data";

export interface Place {
  name: string;
  lat: number;
  lon: number;
  area_km2: number;
  elevation_m: { min: number; max: number; median: number } | null;
  place?: string;
  district?: string;
  state?: string;
  country?: string;
}

export function MapModeSwitch({ is3d, onChange }: { is3d: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex rounded-xl bg-white/95 p-1 shadow-md backdrop-blur" role="group" aria-label="Map view">
      {[["3D", true], ["2D", false]].map(([l, v]) => (
        <button key={String(l)} onClick={() => onChange(v as boolean)} aria-pressed={is3d === v}
          className={`rounded-lg px-3.5 py-1.5 text-sm font-semibold transition ${is3d === v ? "bg-ink text-white" : "text-body hover:bg-panel-2"}`}>
          {String(l)}
        </button>
      ))}
    </div>
  );
}

const dms = (v: number, pos: string, neg: string) => `${Math.abs(v).toFixed(4)}°${v >= 0 ? pos : neg}`;

export function LocationChip({ place, regionName, picked }: { place: Place | null; regionName: string; picked: { lat: number; lng: number; elev: number | null } | null }) {
  const name = regionName.split(",")[0];
  const admin = place ? [place.district && `${place.district} district`, place.state, place.country].filter(Boolean).join(", ") : regionName.split(",").slice(1).join(",").trim();
  return (
    <div className="max-w-[min(92vw,360px)] rounded-xl bg-white/95 px-3 py-2 shadow-md backdrop-blur sm:px-3.5 sm:py-2.5">
      <div className="flex items-center gap-2">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2.2" aria-hidden><path d="M12 22s7-6.2 7-12a7 7 0 1 0-14 0c0 5.8 7 12 7 12z" /><circle cx="12" cy="10" r="2.5" /></svg>
        <span className="font-bold leading-tight">{name}</span>
      </div>
      <div className="mt-0.5 text-xs text-body">{admin}</div>
      {place && (
        <div className="mono mt-1 hidden text-[11px] leading-snug text-muted sm:block">
          {dms(place.lat, "N", "S")} {dms(place.lon, "E", "W")}
          {place.elevation_m && <> · {fmt(place.elevation_m.min, 0)}–{fmt(place.elevation_m.max, 0)} m</>} · {fmt(place.area_km2, 0)} km² studied
        </div>
      )}
      {picked && (
        <div className="mono mt-1.5 border-t border-line pt-1.5 text-[11px] text-text">
          📍 Selected point: {dms(picked.lat, "N", "S")} {dms(picked.lng, "E", "W")}{picked.elev != null ? ` · ${fmt(picked.elev, 0)} m above sea level` : ""}
        </div>
      )}
    </div>
  );
}

export interface HotspotItem { lat: number; lon: number; text: string; confidence?: string | null; elevation_m?: number | null; area_km2?: number | null; landmark?: string; kind?: string; value?: number | null; first_seen?: string | null }

export function hotspotTitle(h: HotspotItem): string {
  if (h.landmark === "snout") return "Glacier snout";
  const ha = h.area_km2 != null ? `${Math.round(h.area_km2 * 100)} ha` : "";
  if (h.kind === "glacier") return `Fast ice · ~${Math.round(h.value ?? 0)} m/yr`;
  if (h.kind === "forest") return `Changed patch · ${ha}${h.first_seen ? ` · since ${new Date(h.first_seen).toLocaleDateString("en-GB", { day: "numeric", month: "short" })}` : ""}`;
  if (h.kind === "ground") return `Moving patch · ${ha} · ~${Math.round(Math.abs(h.value ?? 0) / 10)} cm/yr`;
  return h.text.split(".")[0];
}
