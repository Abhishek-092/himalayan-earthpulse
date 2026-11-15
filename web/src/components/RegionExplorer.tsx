"use client";
/* eslint-disable @typescript-eslint/no-explicit-any */

import Link from "next/link";
import type { Map as MLMap } from "maplibre-gl";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Chart from "./Chart";
import MapView, { type GeoOverlay, type ImageOverlay } from "./MapView";
import ModelSummary from "./ModelSummary";
import { captionFor, captionForRange, friendlyLegend, timelineEvents, verdictFor } from "@/lib/plain";
import { LocationChip, MapModeSwitch, type HotspotItem, type Place } from "./explorer/MapChrome";
import ResultRail, { type RailItem } from "./explorer/ResultRail";
import TimelineBar from "./explorer/TimelineBar";
import EvidenceDrawer, { type EvidenceTab } from "./explorer/EvidenceDrawer";
import { TONE_STYLE } from "@/lib/plain";
import TemporalPanel, { type RegionSeries } from "./TemporalPanel";
import { temporalMetrics } from "@/lib/temporal";
import { findingFor } from "@/lib/findings";
import {
  type Layer,
  type LngLat,
  type Manifest,
  type RegionEntry,
  MODEL_INFO,
  assetUrl,
  decode,
  decodeAll,
  fmt,
  getCatalog,
  getGrid,
  getJSON,
  getManifest,
  lngLatToPixel,
} from "@/lib/data";

type Radar = { frames: string[]; dates: string[]; corners: [LngLat, LngLat, LngLat, LngLat]; title: string; how_to_read: string };
type Inspect = { lng: number; lat: number; value: number | null; label: string; title: string; series?: { dates: string[]; values: number[]; sigma?: number[]; units: string } };

const ORDER = ["gdm", "cdm", "bdm", "epitn"];
const PRODUCTS: Record<string, string[]> = { gdm: ["GUNW"], cdm: ["GOFF", "GCOV"], bdm: ["GCOV"], epitn: ["GUNW", "GOFF", "GCOV"] };
type Check = { stage: string; check: string; status: "pass" | "warn" | "fail"; value: string; expected: string; note: string };

function Toggle({ on, onChange, label, swatch }: { on: boolean; onChange: (v: boolean) => void; label: string; swatch?: React.ReactNode }) {
  return (
    <button onClick={() => onChange(!on)} className="flex w-full items-center gap-2 py-1 text-left text-[13px]">
      {swatch}
      <span className="flex-1 leading-tight">{label}</span>
      <span className={`relative h-5 w-9 shrink-0 rounded-full transition ${on ? "bg-ink" : "bg-line"}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition ${on ? "left-[18px]" : "left-0.5"}`} />
      </span>
    </button>
  );
}

function Swatch({ layer }: { layer: Layer }) {
  const lg = layer.legend;
  const bg = lg.type === "continuous" ? `linear-gradient(90deg, ${lg.stops!.join(",")})` : lg.type === "categorical" ? lg.items![lg.items!.length - 1].color : "linear-gradient(90deg,#c040c0,#40c040)";
  return <span className="h-3 w-3 shrink-0 rounded-[3px]" style={{ background: bg }} />;
}

function Legend({ layer }: { layer: Layer }) {
  const lg = layer.legend;
  return (
    <div className="space-y-1.5 text-[12px]">
      <div className="font-medium">{layer.title}</div>
      {lg.type === "categorical" &&
        lg.items!.map((it) => (
          <div key={String(it.value)} className="flex items-center gap-2"><span className="h-3 w-5 rounded-[3px]" style={{ background: it.color }} />{it.label}</div>
        ))}
      {lg.type === "continuous" && (
        <div>
          <div className="h-2.5 w-56 rounded" style={{ background: `linear-gradient(90deg, ${lg.stops!.join(",")})` }} />
          <div className="mono mt-0.5 flex w-56 justify-between text-[10px] text-muted"><span>{fmt(lg.min, 0)}</span><span>{layer.units}</span><span>{fmt(lg.max, 0)}</span></div>
        </div>
      )}
      <div className="flex items-center gap-2 text-muted"><span className="hatch h-3 w-5 rounded-[3px] border border-line" />Not shown: masked / not reliable</div>
    </div>
  );
}

function maskToDataUrl(mask: Uint8Array, w: number, h: number) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < mask.length; i++) if (mask[i]) { img.data[i * 4] = 245; img.data[i * 4 + 1] = 158; img.data[i * 4 + 2] = 11; img.data[i * 4 + 3] = 235; }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL();
}

export default function RegionExplorer({ id }: { id: string }) {
  const [region, setRegion] = useState<RegionEntry | null>(null);
  const [manifests, setManifests] = useState<Record<string, Manifest>>({});
  const [radar, setRadar] = useState<Radar | null>(null);
  const [model, setModel] = useState("");
  const [visible, setVisible] = useState<string[]>([]);
  const [primary, setPrimary] = useState("");
  const [resultLayer, setResultLayer] = useState("");
  const [t, setT] = useState(0);
  const [tB, setTB] = useState(0);
  const [playing, setPlaying] = useState(false);
  // playhead: runs FROM -> TO without moving the two range handles (null = show the TO pass)
  const [ph, setPh] = useState<number | null>(null);
  const [optical, setOptical] = useState(false);
  const [terrain, setTerrain] = useState(true);
  const [evidence, setEvidence] = useState(false);
  const [evTab, setEvTab] = useState("numbers");
  const [place, setPlace] = useState<Place | null>(null);
  const [hotspots, setHotspots] = useState<Record<string, HotspotItem[]>>({});
  const [activeHs, setActiveHs] = useState<number | null>(null);
  const [touring, setTouring] = useState(false);
  const [picked, setPicked] = useState<{ lat: number; lng: number; elev: number | null } | null>(null);
  const [checks, setChecks] = useState<Check[]>([]);
  const [rawRadar, setRawRadar] = useState(false);
  const [compare, setCompare] = useState(false);
  const [swipe, setSwipe] = useState(50);
  const [inspect, setInspect] = useState<Inspect | null>(null);
  const [thr, setThr] = useState<{ on: boolean; value: number; mode: "above" | "below" | "abs" }>({ on: false, value: 0, mode: "abs" });
  const [thrRes, setThrRes] = useState<{ km2: number; url: string } | null>(null);
  const [vectors, setVectors] = useState(true);
  const [showDiff, setShowDiff] = useState(false);
  const [diffImg, setDiffImg] = useState<{ url: string; lim: number; km2: number } | null>(null);
  const mapA = useRef<MLMap | null>(null);
  const mapB = useRef<MLMap | null>(null);
  const [mapsReady, setMapsReady] = useState(0);

  useEffect(() => {
    getCatalog().then(async (c) => {
      const r = c.regions.find((x) => x.id === id) ?? null;
      setRegion(r);
      if (!r) return;
      // side data in parallel with the manifests (none of it blocks the map)
      const pl = (r as RegionEntry & { place?: string }).place;
      if (pl) getJSON<Place>(pl).then(setPlace).catch(() => {});
      Promise.all(Object.keys(r.models).map(async (m) => {
        try { return [m, (await getJSON<{ hotspots: HotspotItem[] }>(`${id}/${m}/hotspots.json`)).hotspots] as const; } catch { return [m, [] as HotspotItem[]] as const; }
      })).then((pairs) => setHotspots(Object.fromEntries(pairs)));
      const rr = (r as RegionEntry & { radar?: string }).radar;
      if (rr) getJSON<Radar>(rr).then(setRadar).catch(() => {});
      const ms: Record<string, Manifest> = {};
      await Promise.all(Object.keys(r.models).map(async (m) => (ms[m] = await getManifest(id, m))));
      setManifests(ms);
      const first = ORDER.find((m) => ms[m]);
      if (first) setModel(first);
      getJSON<{ checks: Check[] }>(`${id}/verification.json`).then((v) => setChecks(v.checks)).catch(() => {});
    });
  }, [id]);

  const man = manifests[model];
  const layers = useMemo(() => man?.layers ?? [], [man]);
  const finding = useMemo(() => (man && region ? findingFor(model, man, region.name) : null), [man, model, region]);
  const cube = layers.find((l) => l.id === finding?.mainCube) ?? layers.find((l) => l.kind === "cube");
  const dates = useMemo(() => cube?.dates ?? [], [cube]);
  const nT = dates.length;

  useEffect(() => {
    if (!layers.length) return;
    const pref =
      layers.find((l) => l.id === "vertical_velocity") ?? layers.find((l) => l.id === "change_type") ?? layers.find((l) => /^(speed_goff|disturbance_date|overlap)$/.test(l.id)) ??
      layers.find((l) => l.id.endsWith("_velocity")) ?? layers[0];
    setVisible([pref.id]);
    setPrimary(pref.id);
    setResultLayer(pref.id);
    setInspect(null);
    setActiveHs(null);
    setTouring(false);
    setThr((s) => ({ ...s, on: false }));
    setCompare(false);
  }, [layers]);
  useEffect(() => {
    if (!nT) return;
    const hi = finding?.highlightDate ? dates.findIndex((d) => d.startsWith(finding.highlightDate!.slice(0, 7))) : -1;
    setT(hi >= 0 ? hi : nT - 1);
    setTB(0);
    setPh(null);
    setPlaying(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cube?.id, nT]);
  const lo = Math.min(tB, t);
  // play: advance the playhead one pass at a time from FROM to TO, then stop on TO
  useEffect(() => {
    if (!playing || !nT) return;
    const h = setTimeout(() => {
      if (ph != null && ph >= t) { setPlaying(false); return; } // reached TO: stop there
      setPh(Math.min(t, (ph ?? lo) + 1));
    }, 800);
    return () => clearTimeout(h);
  }, [playing, nT, ph, t, lo]);
  const frame = ph ?? t; // the pass shown on the map

  const layer = layers.find((l) => l.id === primary);
  const regionSeries: RegionSeries | undefined = useMemo(() => {
    const sm = man?.summary;
    if (!sm) return undefined;
    if (model === "gdm") {
      const st = Object.entries<any>(sm.stacks ?? {});
      const k = cube?.stack ?? st[0]?.[0];
      return sm.stacks?.[k]?.region_series;
    }
    return sm.region_series;
  }, [man, model, cube]);
  const pixelMetrics = useMemo(() => (inspect?.series ? temporalMetrics(inspect.series.dates, inspect.series.values, inspect.series.sigma) : null), [inspect]);
  const toggleLayer = (lid: string, on: boolean) => {
    setVisible((v) => {
      const nv = on ? [...v.filter((x) => x !== lid), lid] : v.filter((x) => x !== lid);
      setPrimary(on ? lid : nv[nv.length - 1] ?? "");
      return nv;
    });
  };
  const goPass = (i: number, before = false) => {
    setPlaying(false);
    setPh(null);
    if (before) { setTB(i); return; }
    setT(i);
    if (cube && !visible.includes(cube.id)) toggleLayer(cube.id, true);
  };

  const radarIdx = useCallback((ti: number) => {
    if (!radar || !radar.dates.length) return -1;
    const d = +new Date(dates[ti] ?? radar.dates[radar.dates.length - 1]);
    let best = 0;
    radar.dates.forEach((rd, i) => { if (Math.abs(+new Date(rd) - d) < Math.abs(+new Date(radar.dates[best]) - d)) best = i; });
    return best;
  }, [radar, dates]);

  const overlaysFor = useCallback((ti: number, withThr: boolean): ImageOverlay[] => {
    const o: ImageOverlay[] = [];
    if (rawRadar && radar) {
      const ri = radarIdx(ti);
      if (ri >= 0) o.push({ id: "radar", url: `/data/${id}/radar/${radar.frames[ri]}`, corners: radar.corners, opacity: 1 });
    }
    for (const l of layers) {
      if (!visible.includes(l.id)) continue;
      const file = l.kind === "cube" ? l.frames![Math.min(ti, l.frames!.length - 1)] : l.png!;
      o.push({ id: l.id, url: assetUrl(id, model, file), corners: l.corners, opacity: rawRadar ? 0.7 : 0.9 });
    }
    if (withThr && thr.on && thrRes && layer) o.push({ id: "thr", url: thrRes.url, corners: layer.corners, opacity: 0.95 });
    if (withThr && showDiff && diffImg && cube) o.push({ id: "diff", url: diffImg.url, corners: cube.corners, opacity: 1 });
    return o;
  }, [layers, visible, id, model, rawRadar, radar, thr.on, thrRes, layer, radarIdx, showDiff, diffImg, cube]);
  const overlaysA = useMemo(() => overlaysFor(frame, true), [overlaysFor, frame]);
  const overlaysB = useMemo(() => overlaysFor(tB, false), [overlaysFor, tB]);

  const geo: GeoOverlay[] = useMemo(() => {
    const g: GeoOverlay[] = [];
    if (model === "cdm" && man?.vectors) g.push({ id: "vectors", data: assetUrl(id, "cdm", man.vectors), kind: "arrows", color: "#1b2430", width: 1.1, visible: vectors });
    return g;
  }, [model, man, id, vectors]);

  const thrRange = layer?.grid && layer.legend.type === "continuous" ? { lo: layer.legend.min!, hi: layer.legend.max! } : null;
  useEffect(() => {
    if (!thrRange) return;
    const sym = Math.abs(thrRange.lo + thrRange.hi) < 1e-6 * Math.max(1, Math.abs(thrRange.hi));
    setThr((s) => ({ ...s, value: sym ? thrRange.hi / 2 : (thrRange.lo + thrRange.hi) / 2, mode: sym ? "abs" : "above" }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [primary]);
  useEffect(() => {
    if (!thr.on || !layer?.grid) { setThrRes(null); return; }
    let alive = true;
    getGrid(id, model, layer.grid).then((arr) => {
      if (!alive) return;
      const g = layer.grid!, H = layer.height, W = layer.width;
      const off = layer.kind === "cube" ? Math.min(frame, (layer.frames?.length ?? 1) - 1) * H * W : 0;
      const vals = decodeAll(arr, g, off, H * W);
      const mask = new Uint8Array(H * W);
      let n = 0;
      for (let i = 0; i < vals.length; i++) {
        const v = vals[i];
        if (!Number.isFinite(v)) continue;
        if (thr.mode === "above" ? v >= thr.value : thr.mode === "below" ? v <= thr.value : Math.abs(v) >= thr.value) { mask[i] = 1; n++; }
      }
      const px = g.pixel_area_m2 ?? layers.find((l) => l.grid?.pixel_area_m2)?.grid?.pixel_area_m2 ?? 0;
      setThrRes({ km2: (n * px) / 1e6, url: maskToDataUrl(mask, W, H) });
    });
    return () => { alive = false; };
  }, [thr, layer, id, model, frame, layers]);

  // change between BEFORE and AFTER passes (computed in the browser from the cube)
  useEffect(() => {
    if (!showDiff || !cube?.grid) { setDiffImg(null); return; }
    let alive = true;
    getGrid(id, model, cube.grid).then((arr) => {
      if (!alive) return;
      const H = cube.height, W = cube.width, g = cube.grid!;
      const a = decodeAll(arr, g, Math.min(tB, nT - 1) * H * W, H * W);
      const b = decodeAll(arr, g, Math.min(frame, nT - 1) * H * W, H * W);
      const d = new Float32Array(H * W);
      const vals: number[] = [];
      for (let i = 0; i < d.length; i++) { d[i] = b[i] - a[i]; if (Number.isFinite(d[i])) vals.push(Math.abs(d[i])); }
      vals.sort((x, y) => x - y);
      const lim = Math.max(vals[Math.floor(vals.length * 0.98)] ?? 1, 1e-6);
      const c = document.createElement("canvas"); c.width = W; c.height = H;
      const ctx = c.getContext("2d")!; const img = ctx.createImageData(W, H);
      let n = 0;
      for (let i = 0; i < d.length; i++) {
        if (!Number.isFinite(d[i])) continue;
        const f = Math.max(-1, Math.min(1, d[i] / lim));
        // blue (#3b82f6) for decrease, amber (#f59e0b) for increase - colourblind-safe pair
        const col = f < 0 ? [59, 130, 246] : [245, 158, 11];
        img.data[i * 4] = col[0]; img.data[i * 4 + 1] = col[1]; img.data[i * 4 + 2] = col[2];
        img.data[i * 4 + 3] = Math.round(255 * Math.min(1, Math.abs(f) * 1.2));
        if (Math.abs(d[i]) > lim * 0.25) n++;
      }
      ctx.putImageData(img, 0, 0);
      const px = layers.find((l) => l.grid?.pixel_area_m2)?.grid?.pixel_area_m2 ?? 0;
      setDiffImg({ url: c.toDataURL(), lim, km2: (n * px) / 1e6 });
    });
    return () => { alive = false; };
  }, [showDiff, cube, id, model, frame, tB, nT, layers]);

  const onClick = useCallback(async (lng: number, lat: number, elev: number | null = null) => {
    setPicked({ lng, lat, elev });
    const target = layer?.grid ? layer : undefined;
    if (!target?.grid) return;
    const px = lngLatToPixel(target, lng, lat);
    if (!px) { setInspect(null); return; }
    const [r, c] = px;
    const arr = await getGrid(id, model, target.grid);
    const H = target.height, W = target.width;
    const off = target.kind === "cube" ? Math.min(frame, (target.frames?.length ?? 1) - 1) * H * W : 0;
    const v = decode(arr[off + r * W + c], target.grid);
    let label = Number.isFinite(v) ? `${fmt(v, 1)} ${target.units}` : "masked — not reliable here";
    if (target.legend.type === "categorical" && Number.isFinite(v)) label = target.legend.items!.find((i) => Number(i.value) === Math.round(v))?.label ?? label;
    if (target.id === "disturbance_date" && Number.isFinite(v)) label = `first changed ${target.dates?.[Math.round(v)] ?? ""}`;
    let series: Inspect["series"];
    const cl = target.kind === "cube" ? target : layers.find((l) => l.kind === "cube" && (!target.stack || l.stack === target.stack));
    if (cl?.grid) {
      const p2 = lngLatToPixel(cl, lng, lat);
      if (p2) {
        const ca = await getGrid(id, model, cl.grid);
        const T = cl.frames!.length, H2 = cl.height, W2 = cl.width;
        const values = Array.from({ length: T }, (_, k) => decode(ca[k * H2 * W2 + p2[0] * W2 + p2[1]], cl.grid!));
        let sigma: number[] | undefined;
        if (cl.sigma_grid) {
          const sa = await getGrid(id, model, cl.sigma_grid);
          sigma = Array.from({ length: T }, (_, k) => decode(sa[k * H2 * W2 + p2[0] * W2 + p2[1]], cl.sigma_grid!));
        }
        if (values.some((x) => Number.isFinite(x))) series = { dates: cl.dates!, values, sigma, units: cl.units };
      }
    }
    setInspect({ lng, lat, value: Number.isFinite(v) ? v : null, label, title: target.title, series });
  }, [layer, layers, id, model, frame]);

  useEffect(() => {
    if (!compare) return;
    const a = mapA.current, b = mapB.current;
    if (!a || !b) return;
    let lock = false;
    const sync = (s: MLMap, d: MLMap) => () => { if (lock) return; lock = true; d.jumpTo({ center: s.getCenter(), zoom: s.getZoom(), bearing: s.getBearing(), pitch: s.getPitch() }); lock = false; };
    const fa = sync(a, b), fb = sync(b, a);
    a.on("move", fa); b.on("move", fb); fa();
    return () => { a.off("move", fa); b.off("move", fb); };
  }, [compare, mapsReady]);

  useEffect(() => {
    if (!touring) return;
    const n = (hotspots[model] ?? []).length;
    const h = setTimeout(() => {
      if (activeHs == null || activeHs >= n - 1) { setTouring(false); return; }
      const i = activeHs + 1, p = hotspots[model][i];
      setActiveHs(i);
      mapA.current?.flyTo({ center: [p.lon, p.lat], zoom: model === "cdm" ? 12 : 13.5, pitch: terrain ? 50 : 0, bearing: terrain ? -25 + i * 12 : 0, duration: 1600, essential: true });
    }, 5000);
    return () => clearTimeout(h);
  }, [touring, activeHs, hotspots, model, terrain]);

  if (!region) return <div className="p-8 text-muted">Loading…</div>;

  if (region.status !== "processed")
    return (
      <div className="mx-auto max-w-3xl px-5 py-10">
        <div className="kicker">NISAR data catalogued</div>
        <h1 className="mt-1 text-3xl font-bold">{region.name}</h1>
        <p className="mt-2 text-body">{region.story}</p>
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {Object.entries(region.availability).map(([p, v]) => (
            <div key={p} className="panel p-3"><div className="mono text-2xl">{v.count}</div><div className="text-xs text-muted">{p} products</div><div className="mono text-[10px] text-muted">{v.first} → {v.last}</div></div>
          ))}
        </div>
        <p className="mt-6 text-sm text-body">This place is defined only by its bounding box. The same pipeline (<span className="mono">earthpulse all --region {region.id}</span>) auto-detects its NISAR tracks and models and processes it; it was not processed for this demo so the focus stays on the Garhwal Himalaya.</p>
        <Link href="/" className="mt-6 inline-block rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white">← Back to overview</Link>
      </div>
    );

  const models = Object.keys(manifests).sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b));
  const groups = Array.from(new Set(layers.map((l) => l.group || "Layers")));
  const findingsUrl = (region as RegionEntry & { findings?: Record<string, number> }).findings?.[model] ? `/data/${id}/${model}/findings.geojson` : null;
  const shortD = (d: string) => new Date(d).toLocaleDateString("en-GB", { day: "numeric", month: "short" }).toUpperCase();

  const rail: RailItem[] = models.flatMap((m) => {
    const v = verdictFor(m, manifests[m], region.name);
    return v ? [{ model: m, verdict: v, hotspots: hotspots[m] ?? [] }] : [];
  });
  const current = rail.find((r) => r.model === model);
  const hs = current?.hotspots ?? [];
  const tone = current ? TONE_STYLE[current.verdict.tone] : TONE_STYLE.info;
  const markers = hs.map((h, i) => ({ id: `hs-${i}`, lng: h.lon, lat: h.lat, label: String(i + 1), color: tone.dot, size: activeHs === i ? 11 : 8 }));
  const outline: GeoOverlay = { id: "aoi", kind: "fill-outline", color: "#ffffff", width: 2,
    data: { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[
      [region.bbox[0], region.bbox[1]], [region.bbox[2], region.bbox[1]], [region.bbox[2], region.bbox[3]], [region.bbox[0], region.bbox[3]], [region.bbox[0], region.bbox[1]]]] } }] } };
  const geoAll = [outline, ...geo];
  const events = timelineEvents(model, man, dates);
  const rs = regionSeries ? { dates: regionSeries.dates, values: regionSeries.values, units: regionSeries.units } : undefined;
  const caption = nT && lo < frame && (lo > 0 || ph != null)
    ? captionForRange(model, man, dates[lo], dates[frame], rs)
    : captionFor(model, man, nT ? dates[frame] : undefined, rs);
  const basemap = optical ? "light" : "satellite";

  const flyToHotspot = (i: number) => {
    const h = hs[i];
    if (!h) return;
    setActiveHs(i);
    mapA.current?.flyTo({ center: [h.lon, h.lat], zoom: model === "cdm" ? 12 : 13.5, pitch: terrain ? 50 : 0, bearing: terrain ? -25 + i * 12 : 0, duration: 1600, essential: true });
  };
  const resetView = () => {
    setActiveHs(null);
    const cam = mapA.current?.cameraForBounds(region.bbox, { padding: 60 });
    if (cam?.center) mapA.current?.flyTo({ center: cam.center, zoom: cam.zoom, pitch: terrain ? 50 : 0, bearing: terrain ? -15 : 0, duration: 1200 });
  };
  const closeEvidence = () => {
    setEvidence(false);
    // back to results-only view
    setVisible(resultLayer ? [resultLayer] : []);
    setPrimary(resultLayer);
    setCompare(false); setShowDiff(false); setRawRadar(false); setOptical(false); setThr((x) => ({ ...x, on: false }));
  };
  const openEvidence = (tab = "numbers") => { setEvTab(tab); setEvidence(true); setTouring(false); };
  const evidenceForHotspot = async (i: number) => {
    const h = hs[i];
    if (!h) return;
    await onClick(h.lon, h.lat, h.elevation_m ?? null);
    openEvidence("time");
  };

  const regionChecks = checks.filter((c) => (PRODUCTS[model] ?? []).some((p) => c.stage.includes(p)));
  const tabs: EvidenceTab[] = [
    { id: "numbers", label: "Key numbers", content: (
      <div>
        <p className="text-sm text-body">The measurements this result is based on, with their uncertainty.</p>
        <div className="mt-3 grid grid-cols-2 gap-2.5">
          {finding?.stats.map((st) => (
            <div key={st.label} className={`rounded-xl border p-3 ${st.key ? "border-amber/60 bg-amber-soft" : "border-line bg-white"}`}>
              <div className={`mono text-[18px] leading-tight ${st.key ? "text-accent" : ""}`}>{st.value}</div>
              <div className={`mt-1 text-[12px] leading-snug ${st.key ? "text-accent" : "text-body"}`}>{st.label}</div>
              {st.source && <div className="mono mt-1 text-[10px] leading-snug text-muted">{st.source}</div>}
            </div>
          ))}
        </div>
        {finding && finding.chart.length > 0 && (
          <div className="mt-4">
            <div className="flex justify-between text-sm"><span className="font-semibold">{finding.chartTitle}</span><span className="mono text-xs text-muted">{finding.chartUnits}</span></div>
            <Chart data={finding.chart} height={190} layout={{ barmode: "group", yaxis2: { overlaying: "y", side: "right", showgrid: false }, margin: { l: 40, r: 30, t: 6, b: 34 } }} />
          </div>
        )}
        {man && <ModelSummary model={model} manifest={man} region={region} />}
      </div>
    ) },
    { id: "map", label: "Map layers", content: (
      <div>
        <p className="text-sm text-body">Switch on the underlying radar maps. They switch off again when you close this panel.</p>
        {layer && <div className="mt-3 rounded-xl bg-panel-2 p-3 text-[13px] leading-relaxed text-body"><b className="text-text">Now showing: {layer.title}.</b> {layer.how_to_read}</div>}
        {layer && <div className="mt-3"><Legend layer={layer} /></div>}
        <div className="mt-3 space-y-1">
          {groups.map((g) => (
            <div key={g} className="mb-2">
              <div className="mt-1 text-[10px] font-semibold tracking-wide text-muted uppercase">{g}</div>
              {layers.filter((l) => (l.group || "Layers") === g).map((l) => (
                <Toggle key={l.id} on={visible.includes(l.id)} onChange={(v) => toggleLayer(l.id, v)} label={l.title} swatch={<Swatch layer={l} />} />
              ))}
            </div>
          ))}
        </div>
        <div className="mt-2 border-t border-line pt-2">
          {model === "cdm" && <Toggle on={vectors} onChange={setVectors} label="Flow arrows" swatch={<span className="h-3 w-3 text-[10px] leading-3">➚</span>} />}
          <Toggle on={optical} onChange={setOptical} label="Plain grey background (instead of satellite photo)" swatch={<span className="h-3 w-3 rounded-[3px] bg-[#c9ced6]" />} />
          {radar && <Toggle on={rawRadar} onChange={setRawRadar} label="Raw NISAR radar image underneath" swatch={<span className="h-3 w-3 rounded-[3px] bg-[#555]" />} />}
          {nT > 0 && <Toggle on={compare} onChange={setCompare} label="Before / after swipe (Shift-click a timeline dot to set BEFORE)" swatch={<span className="h-3 w-3 rounded-[3px] border border-ink" />} />}
          {nT > 0 && <Toggle on={showDiff} onChange={setShowDiff} label="Change between BEFORE and AFTER" swatch={<span className="h-3 w-3 rounded-[3px]" style={{ background: "linear-gradient(90deg,#3b82f6,#f59e0b)" }} />} />}
        </div>
        {showDiff && diffImg && cube && (
          <div className="mt-3 rounded-xl border border-line p-3 text-[12px]">
            <div className="font-medium">Change {shortD(dates[tB])} → {shortD(dates[frame])} ({cube.title.toLowerCase()})</div>
            <div className="mt-1 h-2.5 w-56 rounded" style={{ background: "linear-gradient(90deg,#3b82f6,rgba(255,255,255,0),#f59e0b)" }} />
            <div className="mono flex w-56 justify-between text-[10px] text-muted"><span>−{fmt(diffImg.lim, 1)}</span><span>{cube.units}</span><span>+{fmt(diffImg.lim, 1)}</span></div>
            <div className="mono text-accent">{fmt(diffImg.km2, 2)} km² changed by more than ±{fmt(diffImg.lim * 0.25, 1)} {cube.units}</div>
          </div>
        )}
        {thrRange && (
          <div className="mt-3 rounded-xl border border-line p-3 text-[12px]">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={thr.on} onChange={(e) => setThr({ ...thr, on: e.target.checked })} /> Measure the area where
              <select className="rounded border border-line bg-white px-1" value={thr.mode} onChange={(e) => setThr({ ...thr, mode: e.target.value as "above" | "below" | "abs" })}>
                <option value="above">≥</option><option value="below">≤</option><option value="abs">|v| ≥</option>
              </select>
              <span className="mono">{fmt(thr.value, 1)}</span>
            </label>
            {thr.on && (
              <>
                <input type="range" className="w-full" min={thr.mode === "abs" ? 0 : thrRange.lo} max={thrRange.hi} step={(thrRange.hi - thrRange.lo) / 200} value={thr.value} onChange={(e) => setThr({ ...thr, value: +e.target.value })} />
                <div className="mono text-accent">= {thrRes ? fmt(thrRes.km2, 3) : "…"} km²</div>
              </>
            )}
          </div>
        )}
      </div>
    ) },
    { id: "time", label: "Over time", content: (
      <div>
        <div className="rounded-xl bg-panel-2 p-3.5">
          <div className="flex justify-between text-sm"><span className="font-semibold">{inspect ? "The spot you picked" : "Tap anywhere on the map"}</span><span className="mono text-[11px] text-muted">{inspect ? `${inspect.lat.toFixed(4)}°N ${inspect.lng.toFixed(4)}°E` : ""}</span></div>
          {inspect ? (
            <>
              <div className="mono mt-1 text-sm">{inspect.label}</div>
              {inspect.series && (
                <Chart height={160} layout={{ showlegend: false, margin: { l: 40, r: 8, t: 6, b: 28 }, yaxis: { title: { text: inspect.series.units } } }}
                  data={[
                    ...(inspect.series.sigma && inspect.series.sigma.every((v) => Number.isFinite(v)) ? [{ x: [...inspect.series.dates, ...[...inspect.series.dates].reverse()],
                      y: [...inspect.series.values.map((v, i) => v + inspect.series!.sigma![i]), ...[...inspect.series.values.map((v, i) => v - inspect.series!.sigma![i])].reverse()],
                      fill: "toself", fillcolor: "rgba(59,130,246,0.15)", line: { width: 0 }, hoverinfo: "skip", type: "scatter" }] : []),
                    { x: inspect.series.dates, y: inspect.series.values, type: "scatter", mode: "lines+markers", line: { color: "#3b82f6", width: 2 }, marker: { size: 4 } },
                  ]} />
              )}
            </>
          ) : <p className="mt-1 text-xs text-body">You will see that spot&apos;s value and its history across every satellite pass, with uncertainty.</p>}
        </div>
        <TemporalPanel series={regionSeries} markerDate={nT ? dates[frame] : undefined} pixel={pixelMetrics} pixelUnits={inspect?.series?.units} />
      </div>
    ) },
    { id: "checks", label: `Quality checks (${regionChecks.length})`, content: (
      <div>
        <p className="text-sm text-body">Automatic checks run on the real NISAR data used for this result. Warnings are known limits that the analysis handles and states.</p>
        <ul className="mt-3 divide-y divide-line">
          {regionChecks.map((c) => (
            <li key={c.stage + c.check} className="py-2.5">
              <div className="flex items-start gap-2">
                <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: c.status === "pass" ? "#16a34a" : c.status === "warn" ? "#f59e0b" : "#dc2626" }} />
                <div className="min-w-0">
                  <div className="text-sm font-medium">{c.check} <span className="text-xs font-normal text-muted">· {c.stage}</span></div>
                  <div className="mono text-[11px] text-body">{c.value} <span className="text-muted">(expected: {c.expected})</span></div>
                  {c.note && <div className="text-[11px] leading-snug text-muted">{c.note}</div>}
                </div>
              </div>
            </li>
          ))}
          {regionChecks.length === 0 && <li className="py-2 text-sm text-muted">No product-level checks for this view.</li>}
        </ul>
        <Link href="/verify/" className="mt-3 inline-block text-sm font-medium text-accent underline">All checks for every place →</Link>
      </div>
    ) },
    { id: "download", label: "Download", content: (
      <div className="space-y-2">
        <Link href={`/report/${id}/`} className="block rounded-xl bg-ink px-4 py-3 text-sm font-semibold text-white">Printable report (PDF) →</Link>
        {findingsUrl && <a href={findingsUrl} download={`${id}-${model}-findings.geojson`} className="block rounded-xl border border-line px-4 py-3 text-sm">Flagged areas (GeoJSON, for GIS)</a>}
        {hs.length > 0 && <a href={`/data/${id}/${model}/hotspots.json`} download={`${id}-${model}-hotspots.json`} className="block rounded-xl border border-line px-4 py-3 text-sm">Places to look at (JSON)</a>}
        <p className="pt-2 text-xs text-muted">Satellite input file names (granule IDs) are listed at the bottom of “Key numbers”.</p>
      </div>
    ) },
  ];

  const h = activeHs != null ? hs[activeHs] : null;

  return (
    <div className="lg:relative lg:h-[calc(100vh-4rem)] lg:overflow-hidden">
      {/* ---------------- map ---------------- */}
      <div className="relative h-[62vh] overflow-hidden lg:absolute lg:inset-0 lg:h-auto">
        <div className="absolute inset-0">
          <MapView bounds={region.bbox} overlays={overlaysA} geo={geoAll} basemap={basemap} terrain={terrain} onClick={onClick} flyIn nav={false}
            markers={markers} pulse onMarkerClick={(mid) => flyToHotspot(Number(mid.replace("hs-", "")))}
            onReady={(m) => { mapA.current = m; setMapsReady((x) => x + 1); }} highlight={picked ? { lng: picked.lng, lat: picked.lat } : null} />
        </div>
        {compare && (
          <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - swipe}% 0 0)` }}>
            <MapView bounds={region.bbox} overlays={overlaysB} geo={geoAll} basemap={basemap} terrain={terrain} onClick={onClick} nav={false}
              onReady={(m) => { mapB.current = m; setMapsReady((x) => x + 1); }} />
          </div>
        )}
        {compare && (
          <>
            <div className="pointer-events-none absolute inset-y-0 z-10 w-px bg-white shadow-[0_0_0_1px_rgba(27,36,48,0.35)]" style={{ left: `${swipe}%` }} />
            <input aria-label="Before / after swipe" type="range" min={2} max={98} value={swipe} onChange={(e) => setSwipe(+e.target.value)}
              className="absolute top-1/2 left-0 z-10 h-10 w-full -translate-y-1/2 cursor-ew-resize opacity-0" />
            <div className="pointer-events-none absolute top-1/2 z-10 grid h-10 w-10 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-base shadow-md" style={{ left: `${swipe}%` }}>‹›</div>
            <div className="mono pointer-events-none absolute bottom-40 left-4 z-10 rounded-md bg-ink/85 px-2.5 py-1 text-[11px] text-white">BEFORE · {nT ? shortD(dates[tB]) : ""}</div>
            <div className="mono pointer-events-none absolute bottom-40 z-10 rounded-md bg-ink/85 px-2.5 py-1 text-[11px] text-white" style={{ left: `calc(${swipe}% + 12px)` }}>AFTER · {nT ? shortD(dates[frame]) : ""}</div>
          </>
        )}

        {/* top-left: 3D/2D, location, view controls */}
        <div className="absolute top-3 left-3 z-20 flex flex-col items-start gap-2 sm:top-4 sm:left-4">
          <div className="flex items-center gap-2">
            <MapModeSwitch is3d={terrain} onChange={setTerrain} />
            <div className="flex rounded-xl bg-white/95 p-1 shadow-md">
              <button aria-label="Zoom in" onClick={() => mapA.current?.zoomIn()} className="grid h-8 w-8 place-items-center rounded-lg text-lg hover:bg-panel-2">+</button>
              <button aria-label="Zoom out" onClick={() => mapA.current?.zoomOut()} className="grid h-8 w-8 place-items-center rounded-lg text-lg hover:bg-panel-2">−</button>
              <button aria-label="Reset view" title="Show the whole area" onClick={resetView} className="grid h-8 w-8 place-items-center rounded-lg hover:bg-panel-2">⤢</button>
            </div>
          </div>
          <LocationChip place={place} regionName={region.name} picked={picked} />
          {inspect && picked && !evidence && (
            <div className="max-w-[min(92vw,360px)] rounded-xl bg-ink/90 px-3.5 py-2 text-xs text-white shadow-md">
              Here: <b>{inspect.label}</b> <button onClick={() => openEvidence("time")} className="ml-1 underline">history →</button>
            </div>
          )}
          {h && (
            <div className="hidden w-[340px] rounded-2xl bg-white/95 p-4 shadow-lg backdrop-blur lg:block">
              <div className="flex items-center gap-2">
                <span className="grid h-6 w-6 place-items-center rounded-full text-xs font-bold text-white" style={{ background: tone.dot }}>{(activeHs ?? 0) + 1}</span>
                <span className="text-xs font-bold uppercase tracking-wider text-muted">Place to look at</span>
                <button aria-label="close" onClick={() => setActiveHs(null)} className="ml-auto text-muted">×</button>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-text">{h.text}</p>
              <div className="mono mt-2 text-[11px] text-muted">{h.lat.toFixed(4)}°N {h.lon.toFixed(4)}°E{h.elevation_m ? ` · ${Math.round(h.elevation_m)} m` : ""}{h.confidence ? ` · confidence ${h.confidence}` : ""}</div>
              <div className="mt-3 flex gap-2">
                <button onClick={() => evidenceForHotspot(activeHs!)} className="flex-1 rounded-lg bg-ink px-3 py-2 text-xs font-semibold text-white">See the evidence</button>
                {activeHs! < hs.length - 1 && <button onClick={() => flyToHotspot(activeHs! + 1)} className="rounded-lg border border-line px-3 py-2 text-xs">Next →</button>}
              </div>
            </div>
          )}
        </div>

        {/* plain-language key for the result layer */}
        {layer && (() => {
          const fl = friendlyLegend(layer);
          return (
            <div className="absolute bottom-3 left-3 z-20 max-w-[calc(100%-1.5rem)] rounded-xl bg-white/95 px-3.5 py-2.5 shadow-md lg:bottom-[150px] lg:left-4">
              <div className="text-[13px] font-semibold text-text">{fl.title}</div>
              {layer.legend.type === "continuous" && (
                <div className="mt-1.5 w-60">
                  <div className="h-2.5 rounded-full" style={{ background: `linear-gradient(90deg, ${layer.legend.stops!.join(",")})` }} />
                  <div className="mt-1 grid grid-cols-3 text-[11px] leading-tight text-body">
                    <span className="text-left">{fl.left}</span><span className="text-center text-muted">{fl.mid}</span><span className="text-right">{fl.right}</span>
                  </div>
                </div>
              )}
              {layer.legend.type === "categorical" && layer.legend.items!.slice(0, 7).map((it) => (
                <div key={String(it.value)} className="mt-1 flex items-center gap-2 text-[12px] text-body"><span className="h-3 w-4 shrink-0 rounded-sm" style={{ background: it.color }} />{fl.labels?.[String(it.value)] ?? it.label}</div>
              ))}
              {hs.length > 0 && <div className="mt-1.5 flex items-center gap-2 border-t border-line pt-1.5 text-[11px] text-muted"><span className="grid h-4 w-4 place-items-center rounded-full text-[9px] font-bold text-white" style={{ background: tone.dot }}>1</span>Numbered circles = places worth a closer look</div>}
            </div>
          );
        })()}
      </div>

      {/* ---------------- timeline ---------------- */}
      <div className="p-3 lg:absolute lg:right-[424px] lg:bottom-4 lg:left-4 lg:z-20 lg:p-0">
        <TimelineBar dates={dates} t={t} from={lo} playhead={ph} onPick={(i) => goPass(i)} onPickFrom={(i) => goPass(i, true)} playing={playing}
          onPlay={() => {
            if (playing) { setPlaying(false); return; }
            if (cube && !visible.includes(cube.id)) toggleLayer(cube.id, true);
            if (ph == null || ph >= t) setPh(lo); // start (or restart) at FROM
            setPlaying(true);
          }}
          onStep={(d) => { setPlaying(false); setPh((p) => Math.max(lo, Math.min(t, (p ?? t) + d))); }}
          events={events} caption={caption || (current ? current.verdict.headline : "")}
          span={man?.summary?.goff?.date_range ?? (man?.summary?.dates ? [man.summary.dates[0], man.summary.dates[man.summary.dates.length - 1]] : undefined)} />
      </div>

      {/* ---------------- results rail ---------------- */}
      <div className="scrollbar-thin p-3 pt-0 lg:absolute lg:top-4 lg:right-4 lg:bottom-4 lg:z-20 lg:w-[392px] lg:overflow-y-auto lg:p-0">
        <ResultRail items={rail} active={model} onSelect={(m) => { setModel(m); setActiveHs(null); }} onEvidence={() => openEvidence("numbers")}
          onHotspot={flyToHotspot} activeHotspot={activeHs} regionName={region.name} story={region.story}
          onTour={() => { setTouring(true); flyToHotspot(0); }} hazardsHref={region.hazards ? `/hazards/${id}/` : null} />
        <p className="mt-3 px-1 text-[11px] text-muted">Data: NASA-ISRO NISAR (provisional), processed by Himalayan EarthPulse. <Link href="/impact/" className="underline">Why it matters</Link></p>
      </div>

      <EvidenceDrawer open={evidence} onClose={closeEvidence} title={current?.verdict.headline ?? MODEL_INFO[model]?.name ?? ""}
        subtitle={`${MODEL_INFO[model]?.name} · ${region.name}`} tabs={tabs} tab={evTab} setTab={setEvTab} />
    </div>
  );
}
