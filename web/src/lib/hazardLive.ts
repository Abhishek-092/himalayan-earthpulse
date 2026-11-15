// Live Phase 2 inference in the browser: fetch Open-Meteo forecast + GloFAS forecast now, recompute the same features
// as the Python pipeline (earthpulse/forcing/openmeteo.py + hazard/slope.py) and apply the exported model.
export interface Calib {
  calibrator?: "platt" | "isotonic";
  platt: [number, number];
  isotonic?: { x: number[]; y: number[] };
}

export interface HazardModel extends Calib {
  features: string[];
  weights: number[];
  mu: number[];
  sd: number[];
  tables: Record<string, number[]>;
  api_k: number;
  lat: number;
  lon: number;
  river_lat: number;
  river_lon: number;
  return_levels: Record<string, number>;
  alert_bands_p72: number[];
  lhasa_r7: { p90: number; p95: number; p99: number };
  landslide?: (Calib & { features: string[]; weights: number[]; mu: number[]; sd: number[];
    tables: Record<string, number[]>; thresholds: number[] }) | null;
}

export interface LiveResult {
  dates: string[];
  todayIndex: number;
  rain: number[];
  r7: number[];
  pDaily: number[];
  p24: number[];
  p72: number[];
  p168: number[];
  slopeLevel: number[];
  pLandslide?: number[];
  q?: { dates: string[]; q: number[]; qmax: number[] };
  fetchedAt: string;
}

// Percentile of x in a quantile table, reproducing numpy.interp(x, table, linspace(0,100)) exactly - including ties,
// where numpy uses the right-most tied entry (searchsorted 'right'). Features that are 0 on most days (snowmelt) depend on it.
const pct = (x: number, table: number[]) => {
  const n = table.length;
  const fp = (k: number) => (k / (n - 1)) * 100;
  if (!(x >= table[0])) return fp(0);
  if (x >= table[n - 1]) return fp(n - 1);
  let j = 0;
  while (j < n && table[j] <= x) j++; // first index with table[j] > x
  const x0 = table[j - 1], x1 = table[j];
  return fp(j - 1) + ((x - x0) / Math.max(x1 - x0, 1e-15)) * (fp(j) - fp(j - 1));
};

const interp = (v: number, xs: number[], ys: number[]) => {
  if (v <= xs[0]) return ys[0];
  if (v >= xs[xs.length - 1]) return ys[ys.length - 1];
  let i = 1;
  while (i < xs.length && xs[i] < v) i++;
  const f = (v - xs[i - 1]) / Math.max(xs[i] - xs[i - 1], 1e-15);
  return ys[i - 1] + f * (ys[i] - ys[i - 1]);
};

/** Apply the model's calibrator (chosen without test data): isotonic table or Platt on the logit. */
export function calibrate(s: number, c: Calib) {
  if (c.calibrator === "isotonic" && c.isotonic) return interp(s, c.isotonic.x, c.isotonic.y);
  const lz = Math.log(Math.min(Math.max(s, 1e-9), 1 - 1e-9) / Math.max(1 - s, 1e-9));
  return 1 / (1 + Math.exp(-(c.platt[0] + c.platt[1] * lz)));
}

export const windowProb = (p: number[], days: number) => p.map((_, i) => 1 - p.slice(i, i + days).reduce((a, v) => a * (1 - v), 1));

export async function runLive(m: HazardModel): Promise<LiveResult> {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${m.lat}&longitude=${m.lon}` +
    `&daily=precipitation_sum,snowfall_sum,temperature_2m_mean&hourly=soil_moisture_9_to_27cm&past_days=60&forecast_days=16&timezone=Asia%2FKolkata`;
  const d = await (await fetch(url)).json();
  const dates: string[] = d.daily.time;
  const rain: number[] = d.daily.precipitation_sum.map((v: number | null) => v ?? 0);
  const snow: number[] = d.daily.snowfall_sum.map((v: number | null) => v ?? 0);
  const temp: number[] = d.daily.temperature_2m_mean.map((v: number | null) => v ?? 0);
  const hs: (number | null)[] = d.hourly.soil_moisture_9_to_27cm;
  const sm = dates.map((_, i) => {
    const seg = hs.slice(i * 24, i * 24 + 24).filter((v): v is number => v !== null);
    return seg.length ? seg.reduce((a, b) => a + b, 0) / seg.length : NaN;
  });
  const smMed = [...sm].filter(Number.isFinite).sort((a, b) => a - b)[Math.floor(sm.filter(Number.isFinite).length / 2)] ?? 0.2;
  const api: number[] = [];
  rain.forEach((r, i) => api.push(r + (i ? m.api_k * api[i - 1] : 0)));
  const roll = (n: number) => rain.map((_, i) => rain.slice(Math.max(0, i - n + 1), i + 1).reduce((a, b) => a + b, 0));
  const r3 = roll(3), r7 = roll(7);
  let swe = 0;
  const melt = rain.map((_, i) => { swe += snow[i] * 10 * 0.1; const mm = Math.min(swe, Math.max(0, temp[i]) * 3); swe -= mm; return mm; });
  const t7 = temp.map((_, i) => { const s = temp.slice(Math.max(0, i - 6), i + 1); return s.reduce((a, b) => a + b, 0) / s.length; });
  const feat: Record<string, number[]> = { r3, api, sm: sm.map((v) => (Number.isFinite(v) ? v : smMed)), melt, t7 };
  const pDaily = dates.map((_, i) => {
    let z = m.weights[0];
    m.features.forEach((f, j) => { z += m.weights[j + 1] * ((pct(feat[f][i], m.tables[f]) - m.mu[j]) / m.sd[j]); });
    return calibrate(1 / (1 + Math.exp(-z)), m);
  });
  const today = new Date().toISOString().slice(0, 10);
  const todayIndex = Math.max(0, dates.indexOf(today));
  const L = m.lhasa_r7;
  let slopeLevel: number[] = r7.map((v) => (v >= L.p99 ? 3 : v >= L.p95 ? 2 : v >= L.p90 ? 1 : 0));
  let pLandslide: number[] | undefined;
  if (m.landslide) {
    const lm = m.landslide;
    const lf: Record<string, number[]> = { r1: rain, r3, r7, api, sm: feat.sm, t7 };
    pLandslide = dates.map((_, i) => {
      let z = lm.weights[0];
      lm.features.forEach((f, j) => { z += lm.weights[j + 1] * ((pct(lf[f][i], lm.tables[f]) - lm.mu[j]) / lm.sd[j]); });
      return calibrate(1 / (1 + Math.exp(-z)), lm);
    });
    slopeLevel = pLandslide.map((v) => lm.thresholds.filter((t) => v >= t).length);
  }
  let q: LiveResult["q"];
  try {
    const f = await (await fetch(`https://flood-api.open-meteo.com/v1/flood?latitude=${m.river_lat}&longitude=${m.river_lon}&daily=river_discharge,river_discharge_max&forecast_days=30&past_days=30`)).json();
    q = { dates: f.daily.time, q: f.daily.river_discharge, qmax: f.daily.river_discharge_max };
  } catch {}
  return { dates, todayIndex, rain, r7, pDaily, p24: windowProb(pDaily, 1), p72: windowProb(pDaily, 3), p168: windowProb(pDaily, 7), slopeLevel, pLandslide, q,
    fetchedAt: new Date().toISOString() };
}
