// Browser port of the core of pipeline/earthpulse/temporal/framework.py for any clicked pixel's series.
export interface TemporalMetrics {
  baseline: number;
  current: number;
  net: number;
  netSigma: number;
  ratePerYr: number;
  rateSe: number;
  accelPerYr2: number;
  onset: string | null;
  persistence: number;
  recurrence: number;
  recovered: number | null;
  behaviour: string;
  nObs: number;
}

const yr = (d: string) => new Date(d).getTime() / (365.25 * 86400000);

export function temporalMetrics(dates: string[], values: number[], sigma?: number[], kSigma = 2): TemporalMetrics | null {
  const idx = values.map((v, i) => i).filter((i) => Number.isFinite(values[i]));
  if (idx.length < 4) return null;
  const t = idx.map((i) => yr(dates[i]));
  const y = idx.map((i) => values[i]);
  const s = idx.map((i) => (sigma && Number.isFinite(sigma[i]) && sigma[i] > 0 ? sigma[i] : NaN));
  const n = y.length;
  const baseline = y[0];
  const current = y[n - 1];
  // noise: supplied sigma (median) or robust first-difference scatter
  const sv = s.filter(Number.isFinite).sort((a, b) => a - b);
  const diffs = y.slice(1).map((v, i) => v - y[i]).sort((a, b) => a - b);
  const med = (a: number[]) => a[Math.floor(a.length / 2)];
  const fd = 1.4826 * med(diffs.map((d) => Math.abs(d - med(diffs))).sort((a, b) => a - b)) / Math.SQRT2;
  const noise = sv.length ? Math.max(med(sv), 1e-9) : Math.max(fd, 1e-9);
  const w = s.map((v) => 1 / (Number.isFinite(v) ? Math.max(v, noise * 0.1) : noise) ** 2);
  const tm = t.reduce((a, v, i) => a + v * w[i], 0) / w.reduce((a, b) => a + b, 0);
  // weighted linear fit
  let Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
  t.forEach((ti, i) => { const x = ti - tm; Sw += w[i]; Sx += w[i] * x; Sy += w[i] * y[i]; Sxx += w[i] * x * x; Sxy += w[i] * x * y[i]; });
  const det = Sw * Sxx - Sx * Sx;
  const rate = (Sw * Sxy - Sx * Sy) / det;
  const rateSe = Math.sqrt(Sw / det);
  // quadratic term (unweighted, for the acceleration estimate)
  const X = t.map((ti) => ti - tm);
  const m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  X.forEach((x, i) => { const r = [1, x, 0.5 * x * x]; for (let a = 0; a < 3; a++) { b[a] += r[a] * y[i]; for (let c = 0; c < 3; c++) m[a][c] += r[a] * r[c]; } });
  const accel = solve3(m, b)[2];
  const flags = y.map((v) => Math.abs(v - baseline) > kSigma * noise);
  let onsetI = -1;
  for (let i = 1; i < n; i++) if (flags[i] && flags[i - 1]) { onsetI = i - 1; break; }
  const after = onsetI >= 0 ? flags.slice(onsetI) : [];
  const persistence = after.length ? after.filter(Boolean).length / after.length : 0;
  let recurrence = 0, run = 0;
  flags.forEach((f, i) => { run = f ? run + 1 : 0; if (run === 2 && (i === 1 || !flags[i - 2])) recurrence++; });
  const dev = y.map((v) => v - baseline);
  let pk = 0;
  dev.forEach((d, i) => { if (Math.abs(d) > Math.abs(dev[pk])) pk = i; });
  const recovered = onsetI >= 0 && Math.abs(dev[pk]) > 0 ? 1 - dev[n - 1] / dev[pk] : null;
  const sigTrend = Math.abs(rate) > 2 * rateSe;
  let behaviour = "stable";
  if (sigTrend) behaviour = Math.sign(accel) === Math.sign(rate) && Math.abs(accel) > Math.abs(rate) ? "accelerating" : "continuous trend";
  if (!sigTrend && recurrence >= 2) behaviour = "episodic / recurring";
  if (!sigTrend && recovered !== null && recovered > 0.5 && Math.abs(dev[pk]) > (kSigma + 1) * noise) behaviour = "recovering";
  return { baseline, current, net: current - baseline, netSigma: noise * Math.SQRT2, ratePerYr: rate, rateSe, accelPerYr2: accel,
    onset: onsetI >= 0 ? dates[idx[onsetI]] : null, persistence, recurrence, recovered, behaviour, nObs: n };
}

function solve3(A: number[][], b: number[]) {
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-12) return [0, 0, 0];
    for (let r = 0; r < 3; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k]; }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}
