"use client";
/* eslint-disable @typescript-eslint/no-explicit-any */

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import type { Map as MLMap } from "maplibre-gl";
import MapView from "./MapView";
import { VerdictCard, TopicIcon, Term } from "./Plain";
import { LEVEL_PLAIN, TONE_STYLE, verdictsFor } from "@/lib/plain";
import { MapModeSwitch } from "./explorer/MapChrome";
import { getJSON } from "@/lib/data";
import { type Catalog, type Manifest, MODEL_INFO, fmt, getCatalog, getManifest } from "@/lib/data";

type Summaries = Record<string, Partial<Record<string, Manifest>>>;

function headline(id: string, s: Summaries): string[] {
  const out: string[] = [];
  const g = s[id]?.gdm?.summary;
  if (g) {
    for (const [k, v] of Object.entries<Record<string, unknown>>(g.stacks ?? {})) {
      const sv = v.settlement_velocity_mm_yr as { median_mm_yr: number; sem_mm_yr: number } | undefined;
      if (sv) out.push(`Town ground motion (${k.startsWith("A") ? "ascending" : "descending"} view): ${fmt(sv.median_mm_yr)} ± ${fmt(sv.sem_mm_yr)} mm/yr`);
    }
  }
  const c = s[id]?.cdm?.summary?.goff;
  if (c?.speed_main_median_m_yr != null) out.push(`Glacier speed (median, main trunk): ${fmt(c.speed_main_median_m_yr, 0)} m/yr`);
  const b = s[id]?.bdm?.summary;
  if (b?.potential_disturbance_km2 != null)
    out.push(`Lasting forest change: ${fmt(b.persistent_disturbance_km2 ?? b.potential_disturbance_km2, 2)} km² of ${fmt(b.forest_area_km2, 0)} km² (${fmt((b.recovering_km2 / b.potential_disturbance_km2) * 100, 0)}% of flagged change was reversible)`);
  return out;
}

const STORY = [
  {
    title: "A radar that sees through clouds",
    body: "NISAR, built by NASA and ISRO, sends its own microwave pulses and listens to the echo — so it works at night and through monsoon cloud. It revisits every place about every 12 days. Everything in this app comes from real NISAR L-band data, November 2025 – November 2026.",
    region: null as string | null,
    zoom: 4.2,
  },
  { title: "Is Joshimath still sinking?", body: "In January 2023 cracks opened across this Himalayan town. NISAR compares the radar wave's timing between passes to measure ground movement of millimetres to centimetres.", region: "joshimath", zoom: 12 },
  { title: "How fast does Gangotri flow?", body: "The source glacier of the Ganga moves tens of metres a year. NISAR tracks crevasses and debris patterns between passes to measure that flow.", region: "gangotri", zoom: 10.5 },
  { title: "Where did the forest change?", body: "Long L-band waves pass through leaves and bounce off branches and trunks. When that echo suddenly weakens, something removed forest structure — fire, landslide or clearing.", region: "garhwal-forests", zoom: 10.5 },
  { title: "Now explore it yourself", body: "Open a place, drag the time slider, swipe between dates, click any pixel to see its history, and read how sure we are. Start with “Learn to read radar” if radar images are new to you.", region: null, zoom: 6 },
];

export default function Overview() {
  const [cat, setCat] = useState<Catalog | null>(null);
  const [sums, setSums] = useState<Summaries>({});
  const [step, setStep] = useState<number | null>(null);
  const [map, setMap] = useState<MLMap | null>(null);
  const [alerts, setAlerts] = useState<Record<string, any>>({});
  const [preview, setPreview] = useState<string | null>(null);
  const [is3d, setIs3d] = useState(true);

  useEffect(() => {
    getCatalog().then(async (c) => {
      setCat(c);
      const s: Summaries = {};
      await Promise.all(
        c.regions.flatMap((r) =>
          Object.keys(r.models).map(async (m) => {
            try {
              (s[r.id] ??= {})[m] = await getManifest(r.id, m);
            } catch {}
          }),
        ),
      );
      setSums({ ...s });
      const al: Record<string, any> = {};
      await Promise.all(c.regions.filter((r) => r.hazards).map(async (r) => {
        try { al[r.id] = (await getJSON<any>(`${r.id}/hazards/manifest.json`)).alert; } catch {}
      }));
      setAlerts(al);
    });
  }, []);

  useEffect(() => {
    if (step === null || !map || !cat) return;
    const st = STORY[step];
    const r = st.region ? cat.regions.find((x) => x.id === st.region) : null;
    if (r) map.flyTo({ center: r.centroid, zoom: st.zoom, duration: 2200, pitch: 45 });
    else map.flyTo({ center: [80.5, 26.5], zoom: st.zoom, duration: 2000, pitch: 0 });
  }, [step, map, cat]);

  const markers = useMemo(
    () =>
      (cat?.regions ?? []).filter((r) => r.status === "processed").map((r) => {
        const models = Object.keys(r.models).filter((m) => m !== "epitn");
        return {
          id: r.id,
          lng: r.centroid[0],
          lat: r.centroid[1],
          label: r.name.split(",")[0],
          color: models.length ? MODEL_INFO[models[0]].color : "#8d97ab",
          size: r.status === "processed" ? 8 : 5,
        };
      }),
    [cat],
  );

  const processed = cat?.regions.filter((r) => r.status === "processed") ?? [];
  const focus = useMemo<[number, number, number, number] | null>(() => {
    const ps = (cat?.regions ?? []).filter((r) => r.status === "processed");
    if (!ps.length) return null;
    const b = ps.reduce((a, r) => [Math.min(a[0], r.bbox[0]), Math.min(a[1], r.bbox[1]), Math.max(a[2], r.bbox[2]), Math.max(a[3], r.bbox[3])], [180, 90, -180, -90]);
    return [b[0] - 0.6, b[1] - 0.4, b[2] + 0.6, b[3] + 0.4];
  }, [cat]);
  const totalGranules = cat?.regions.reduce((a, r) => a + Object.values(r.availability).reduce((b, v) => b + v.count, 0), 0) ?? 0;
  const st = step !== null ? STORY[step] : null;
  const stRegion = st?.region ?? null;

  return (
    <div className="grid grid-cols-1 lg:h-[calc(100vh-4rem)] lg:grid-cols-[1fr_420px]">
      <div className="relative h-[55vh] lg:h-auto">
        {focus && (
          <MapView basemap="satellite" bounds={focus} terrain={is3d} flyIn markers={markers} pulse={false}
            onMarkerClick={(id) => { setPreview(id); setStep(null); const r = cat?.regions.find((x) => x.id === id); if (r && map) map.flyTo({ center: r.centroid, zoom: 9.5, pitch: is3d ? 55 : 0, duration: 1800 }); }}
            onReady={setMap} />
        )}
        <div className="absolute top-3 left-3 z-10 flex items-center gap-2">
          <MapModeSwitch is3d={is3d} onChange={setIs3d} />
          <button onClick={() => { setPreview(null); if (focus && map) { const cam = map.cameraForBounds(focus, { padding: 40 }); if (cam?.center) map.flyTo({ center: cam.center, zoom: cam.zoom, pitch: is3d ? 50 : 0, bearing: 0, duration: 1500 }); } }}
            className="rounded-xl bg-white/95 px-3 py-2 text-sm font-medium shadow-md">⤢ All places</button>
        </div>
        {preview && !st && (() => {
          const r = cat?.regions.find((x) => x.id === preview);
          if (!r) return null;
          const vs = verdictsFor(sums[r.id] ?? {}, r.name);
          return (
            <div className="absolute bottom-6 left-1/2 z-10 w-[min(520px,92%)] -translate-x-1/2 rounded-2xl bg-white p-5 shadow-2xl">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-[11px] font-bold uppercase tracking-wider text-muted">{r.status === "processed" ? "What NISAR found" : "NISAR data available"}</div>
                  <div className="text-xl font-bold">{r.name}</div>
                </div>
                <button aria-label="close" onClick={() => setPreview(null)} className="text-xl text-muted">×</button>
              </div>
              <p className="mt-1 text-sm text-body">{r.story}</p>
              {vs.length > 0 ? (
                <ul className="mt-3 space-y-2">
                  {vs.map((v) => (
                    <li key={v.topic} className="flex items-start gap-2.5 text-sm">
                      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg" style={{ background: TONE_STYLE[v.tone].bg, color: TONE_STYLE[v.tone].fg }}><TopicIcon kind={v.icon} size={16} /></span>
                      <span className="font-medium leading-snug text-text">{v.headline}</span>
                    </li>
                  ))}
                </ul>
              ) : <p className="mt-3 text-sm text-muted">Not analysed yet - the same pipeline can process it from its satellite data.</p>}
              {r.status === "processed" && (
                <Link href={`/region/${r.id}/`} className="mt-4 flex items-center justify-between rounded-xl bg-ink px-4 py-3 text-sm font-semibold text-white">
                  <span>Explore {r.name.split(",")[0]} in 3D</span><span>→</span>
                </Link>
              )}
            </div>
          );
        })()}
        {st && (
          <div className="absolute bottom-8 left-1/2 w-[min(560px,92%)] -translate-x-1/2 panel p-5 shadow-2xl">
            <div className="kicker mb-1">Guided tour · {step! + 1} / {STORY.length}</div>
            <h2 className="mb-2 text-xl font-semibold">{st.title}</h2>
            <p className="text-sm leading-relaxed text-muted">{st.body}</p>
            {stRegion && headline(stRegion, sums).length > 0 && (
              <ul className="mt-3 space-y-1 text-sm">
                {headline(stRegion, sums).map((h) => (
                  <li key={h} className="num text-accent">▸ {h}</li>
                ))}
              </ul>
            )}
            <div className="mt-4 flex items-center justify-between">
              <button className="text-sm text-muted hover:text-text" onClick={() => setStep(null)}>Close</button>
              <div className="flex gap-2">
                {stRegion && (
                  <Link href={`/region/${stRegion}/`} className="rounded-md border border-line px-3 py-1.5 text-sm hover:bg-panel-2">Open this place</Link>
                )}
                {step! < STORY.length - 1 ? (
                  <button className="rounded-md bg-ink px-3 py-1.5 text-sm font-medium text-white" onClick={() => setStep(step! + 1)}>Next →</button>
                ) : (
                  <Link href="/learn/" className="rounded-md bg-ink px-3 py-1.5 text-sm font-medium text-white">Learn to read radar</Link>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
      <aside className="scrollbar-thin border-l border-line bg-white p-5 lg:overflow-y-auto">
        <div className="kicker">NASA Space Apps 2026 · Dancing with the SARs</div>
        <h1 className="mt-1 text-[26px] font-bold leading-tight">What is happening to the Himalaya - seen from space, even through clouds</h1>
        <p className="mt-2 text-sm leading-relaxed text-body">
          Is the ground under towns sinking? How fast are glaciers moving? Are forests disappearing? A new{" "}
          <Term k="NISAR">NASA-ISRO satellite</Term> uses <Term k="radar">radar</Term> to answer these questions every 12 days - day or night,
          monsoon or not. We turned a year of its data into simple answers for each place.
        </p>
        <div className="mt-4 flex gap-2">
          <Link href="/story/" className="flex-1 rounded-lg bg-ink py-2.5 text-center font-medium text-white hover:opacity-90">▶ Watch the 60-second story</Link>
          <button onClick={() => setStep(0)} className="rounded-lg border border-line bg-white px-3 py-2.5 text-sm">Map tour</button>
        </div>

        {Object.entries(alerts).map(([rid, a]) => {
          const r = cat?.regions.find((x) => x.id === rid);
          const lv = LEVEL_PLAIN[a.level];
          return (
            <Link key={rid} href={`/hazards/${rid}/`} className="mt-4 flex items-center gap-3 rounded-2xl border-2 p-3 transition hover:shadow-md" style={{ borderColor: a.color, background: a.color + "12" }}>
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-white" style={{ background: a.color }}><TopicIcon kind="hazard" /></span>
              <span className="min-w-0 flex-1">
                <span className="block text-xs font-semibold uppercase tracking-wide text-muted">Today in {r?.name.split(",")[0]} · landslide &amp; flood outlook</span>
                <span className="block text-[17px] font-bold" style={{ color: a.color }}>{lv.word}</span>
                <span className="block text-xs text-body">{lv.say} <span className="text-muted">Experimental - not an official warning.</span></span>
              </span>
              <span className="text-lg text-muted">→</span>
            </Link>
          );
        })}

        {(() => {
          const passes = processed.length ? Math.max(...processed.map((r) => r.availability.GCOV?.count ?? 0)) : 0;
          const first = processed.map((r) => r.availability.GCOV?.first).filter(Boolean).sort()[0] as string | undefined;
          const last = processed.map((r) => r.availability.GCOV?.last).filter(Boolean).sort().slice(-1)[0] as string | undefined;
          const nxt = last ? new Date(new Date(last).getTime() + 12 * 864e5).toISOString().slice(0, 10) : undefined;
          const steps: [string, string, string][] = [
            ["Jul 2025", "Launch", "NISAR lifts off from India"],
            [first ? new Date(first).toLocaleDateString("en-GB", { month: "short", year: "numeric" }) : "–", "First data", "radar images start arriving"],
            [last ? new Date(last).toLocaleDateString("en-GB", { day: "numeric", month: "short" }) : "–", "Latest pass", `${passes ? `up to ${passes} images per place` : ""}`],
            [nxt ? new Date(nxt).toLocaleDateString("en-GB", { day: "numeric", month: "short" }) : "–", "Next look", "every 12 days, any weather"],
          ];
          return (
            <div className="mt-5">
              <h3 className="mb-2 text-base font-bold">The mission so far</h3>
              <ol className="relative grid grid-cols-4 gap-1">
                <div className="absolute top-[9px] right-[12%] left-[12%] h-0.5 bg-line" />
                {steps.map(([d, t, sub], i) => (
                  <li key={t} className="relative text-center">
                    <span className={`relative mx-auto block h-[18px] w-[18px] rounded-full border-2 ${i === 3 ? "border-dashed border-muted bg-white" : i === 2 ? "border-ink bg-amber" : "border-ink bg-ink"}`} />
                    <div className="mono mt-1 text-[11px] font-semibold">{d}</div>
                    <div className="text-xs font-medium text-text">{t}</div>
                    <div className="text-[10px] leading-tight text-muted">{sub}</div>
                  </li>
                ))}
              </ol>
            </div>
          );
        })()}

        <h3 className="mt-6 mb-2 text-base font-bold">What we found</h3>
        <div className="space-y-4">
          {processed.map((r) => {
            const vs = verdictsFor(sums[r.id] ?? {}, r.name);
            return (
              <div key={r.id}>
                <Link href={`/region/${r.id}/`} className="group mb-1.5 flex items-baseline justify-between">
                  <span className="font-semibold group-hover:underline">{r.name}</span>
                  <span className="text-xs font-medium text-accent">See the map →</span>
                </Link>
                <p className="mb-2 text-xs leading-relaxed text-muted">{r.story}</p>
                <div className="space-y-2">{vs.map((v) => <VerdictCard key={v.topic} v={v} compact />)}</div>
              </div>
            );
          })}
          {processed.length === 0 && <p className="text-sm text-muted">Loading…</p>}
        </div>

        <Link href="/impact/" className="mt-5 block rounded-2xl bg-ink p-4 text-white transition hover:opacity-95">
          <div className="text-xs font-semibold uppercase tracking-wide text-white/60">Why this matters</div>
          <div className="mt-1 font-semibold">{totalGranules || "–"} satellite images, {processed.length || "–"} places, every result double-checked - see who benefits and how →</div>
        </Link>

        <p className="mt-6 text-[11px] leading-relaxed text-muted">
          Data: NISAR L2 GCOV, GUNW, GOFF (NASA/JPL-ISRO, via ASF DAAC), processed by this project. Context: Copernicus DEM, ESA WorldCover, RGI 7.0,
          ITS_LIVE, Sentinel-2. See <Link className="underline" href="/about/">methods & limits</Link>.
        </p>
      </aside>
    </div>
  );
}
