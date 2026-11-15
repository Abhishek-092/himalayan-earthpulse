// Data contract "earthpulse/1" produced by pipeline/earthpulse/export/web.py
export type LngLat = [number, number];

export interface Availability {
  count: number;
  first: string;
  last: string;
  stacks: number;
}

export interface RegionEntry {
  id: string;
  name: string;
  story: string;
  priority: string;
  bbox: [number, number, number, number];
  centroid: LngLat;
  availability: Record<string, Availability>;
  models: Record<string, { manifest: string; layers: number }>;
  status: "processed" | "catalogued";
  resolved?: string[] | null;
  hazards: null | unknown; // Phase 2 slot
}

export interface Catalog {
  schema: string;
  generated: string;
  mission: string;
  regions: RegionEntry[];
}

export interface Grid {
  bin: string;
  dtype: "int16";
  scale: number;
  offset: number;
  nodata: number;
  pixel_area_m2?: number;
  shape?: [number, number, number];
}

export interface Legend {
  type: "continuous" | "categorical" | "rgb";
  cmap?: string;
  stops?: string[];
  min?: number;
  max?: number;
  items?: { value: number | string; color: string; label: string }[];
}

export interface Layer {
  id: string;
  title: string;
  units: string;
  group: string;
  png?: string;
  kind?: "cube";
  frames?: string[];
  dates?: string[];
  corners: [LngLat, LngLat, LngLat, LngLat];
  width: number;
  height: number;
  legend: Legend;
  how_to_read: string;
  grid?: Grid;
  sigma_grid?: Grid;
  stats?: { valid_fraction: number; p05: number | null; median: number | null; p95: number | null };
  stack?: string;
}

export interface Manifest {
  model: "gdm" | "cdm" | "bdm" | "epitn";
  region: string;
  layers: Layer[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  summary?: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  network?: any;
  vectors?: string | null;
  outlines?: string;
  speckle_demo?: { raw: string; filtered: string; date: string };
}

export const BASE = "/data";

const jsonCache = new Map<string, Promise<unknown>>();
export function getJSON<T>(path: string): Promise<T> {
  if (!jsonCache.has(path)) {
    jsonCache.set(
      path,
      fetch(`${BASE}/${path}`).then((r) => {
        if (!r.ok) throw new Error(`${path}: ${r.status}`);
        return r.json();
      }),
    );
  }
  return jsonCache.get(path) as Promise<T>;
}

export const getCatalog = () => getJSON<Catalog>("catalog.json");
export const getManifest = (region: string, model: string) =>
  getJSON<Manifest>(`${region}/${model}/manifest.json`);

export function assetUrl(region: string, model: string, file: string) {
  return `${BASE}/${region}/${model}/${file}`;
}

// ---- Int16 grids -----------------------------------------------------------------------------------
const binCache = new Map<string, Promise<Int16Array>>();
export function getGrid(region: string, model: string, grid: Grid): Promise<Int16Array> {
  const url = assetUrl(region, model, grid.bin);
  if (!binCache.has(url)) {
    binCache.set(
      url,
      fetch(url)
        .then((r) => r.arrayBuffer())
        .then((b) => new Int16Array(b)),
    );
  }
  return binCache.get(url)!;
}

export function decode(q: number, g: Grid): number {
  if (q === g.nodata) return NaN;
  return (q + 32500) / g.scale + g.offset;
}

export function decodeAll(arr: Int16Array, g: Grid, start = 0, length = arr.length): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = decode(arr[start + i], g);
  return out;
}

/** Map lon/lat to (row, col) using the affine frame defined by the layer corners (TL, TR, BR, BL). */
export function lngLatToPixel(layer: Layer, lng: number, lat: number): [number, number] | null {
  const [tl, tr, , bl] = layer.corners;
  const ax = tr[0] - tl[0], ay = tr[1] - tl[1];
  const bx = bl[0] - tl[0], by = bl[1] - tl[1];
  const px = lng - tl[0], py = lat - tl[1];
  const det = ax * by - ay * bx;
  if (Math.abs(det) < 1e-15) return null;
  const u = (px * by - py * bx) / det;
  const v = (ax * py - ay * px) / det;
  if (u < 0 || u >= 1 || v < 0 || v >= 1) return null;
  return [Math.floor(v * layer.height), Math.floor(u * layer.width)];
}

export function pixelToLngLat(layer: Layer, row: number, col: number): LngLat {
  const [tl, tr, , bl] = layer.corners;
  const u = (col + 0.5) / layer.width, v = (row + 0.5) / layer.height;
  return [tl[0] + u * (tr[0] - tl[0]) + v * (bl[0] - tl[0]), tl[1] + u * (tr[1] - tl[1]) + v * (bl[1] - tl[1])];
}

export const MODEL_INFO: Record<string, { name: string; short: string; color: string; product: string; question: string }> = {
  gdm: { name: "Ground movement", short: "Ground", color: "#e76f51", product: "GUNW", question: "Is the ground moving?" },
  cdm: { name: "Glacier flow", short: "Glacier", color: "#4cc9f0", product: "GOFF + GCOV", question: "How fast is the ice flowing?" },
  bdm: { name: "Forest change", short: "Forest", color: "#80b918", product: "GCOV", question: "Where did the forest change?" },
  epitn: { name: "Changes together?", short: "Links", color: "#c77dff", product: "BDM + GDM + CDM", question: "Do changes happen together?" },
};

export function fmt(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 1e5)) return v.toExponential(1);
  return v.toFixed(digits);
}

export function decimalYearToDate(y: number): string {
  const yr = Math.floor(y);
  const d = new Date(Date.UTC(yr, 0, 1) + (y - yr) * 365.25 * 86400000);
  return d.toISOString().slice(0, 10);
}
