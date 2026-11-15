"use client";

/* eslint-disable @typescript-eslint/no-explicit-any */
import Chart from "./Chart";
import { fmt } from "@/lib/data";
import { type TemporalMetrics } from "@/lib/temporal";

export interface RegionSeries {
  label: string;
  units: string;
  dates: string[];
  values: number[];
  sigma?: number[];
  n?: number[];
  temporal?: any;
}

/** Normalise the pipeline's summarize_series output to the same shape as the browser metrics. */
function fromPipeline(t: any): Partial<TemporalMetrics> & { onsetYear?: number | null } {
  return {
    baseline: t.baseline, current: t.current, net: t.net_change, netSigma: t.net_change_sigma, ratePerYr: t.rate_per_yr,
    rateSe: t.rate_se, accelPerYr2: t.acceleration_per_yr2, persistence: t.persistence, recurrence: t.recurrence,
    recovered: t.recovered_fraction, behaviour: t.behaviour, nObs: t.n_obs, onsetYear: t.onset_year,
  };
}

function yearToDate(y?: number | null) {
  if (y == null || !Number.isFinite(y)) return "none";
  const yr = Math.floor(y);
  return new Date(Date.UTC(yr, 0, 1) + (y - yr) * 365.25 * 86400000).toISOString().slice(0, 10);
}

export function MetricsGrid({ m, units, title }: { m: any; units: string; title: string }) {
  const rows: [string, string][] = [
    ["Behaviour", m.behaviour ?? "–"],
    ["Net change", `${fmt(m.net, 1)} ± ${fmt(m.netSigma, 1)} ${units}`],
    ["Rate", `${fmt(m.ratePerYr, 1)} ± ${fmt(m.rateSe, 1)} ${units}/yr`],
    ["Acceleration", `${fmt(m.accelPerYr2, 1)} ${units}/yr²`],
    ["Change onset", m.onset ?? yearToDate(m.onsetYear)],
    ["Persistence", m.persistence != null ? `${fmt(m.persistence * 100, 0)}%` : "–"],
    ["Recurrence", m.recurrence != null ? `${fmt(m.recurrence, 0)} episode(s)` : "–"],
    ["Recovered", m.recovered != null && Number.isFinite(m.recovered) ? `${fmt(m.recovered * 100, 0)}%` : "–"],
  ];
  return (
    <div>
      <div className="mb-1 text-[12px] font-semibold">{title}</div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between border-b border-line py-0.5"><span className="text-muted">{k}</span><span className="mono">{v}</span></div>
        ))}
      </div>
    </div>
  );
}

export default function TemporalPanel({ series, markerDate, pixel, pixelUnits }: { series?: RegionSeries; markerDate?: string; pixel?: TemporalMetrics | null; pixelUnits?: string }) {
  if (!series && !pixel) return null;
  return (
    <div className="mt-3 rounded-xl border border-line p-3.5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">Temporal analysis</span>
        <span className="mono text-[10px] text-muted">shared framework · same for all models</span>
      </div>
      {series && (
        <>
          <div className="mt-1 text-[11px] text-body">{series.label}</div>
          <Chart height={150} layout={{ showlegend: false, margin: { l: 40, r: 8, t: 6, b: 28 }, yaxis: { title: { text: series.units } },
            shapes: markerDate ? [{ type: "line", x0: markerDate, x1: markerDate, yref: "paper", y0: 0, y1: 1, line: { color: "#92400e", width: 1.5 } }] : [] }}
            data={[
              ...(series.sigma ? [{ x: [...series.dates, ...[...series.dates].reverse()], y: [...series.values.map((v, i) => v + series.sigma![i]), ...[...series.values.map((v, i) => v - series.sigma![i])].reverse()],
                fill: "toself", fillcolor: "rgba(59,130,246,0.15)", line: { width: 0 }, type: "scatter", hoverinfo: "skip" }] : []),
              { x: series.dates, y: series.values, type: "scatter", mode: "lines+markers", line: { color: "#3b82f6", width: 2 }, marker: { size: 4 },
                text: series.n?.map((n) => `${n} pair(s)`), hovertemplate: series.n ? "%{y:.1f} · %{text}<extra></extra>" : undefined },
            ]} />
          {series.n && <div className="mono text-[10px] text-muted">pairs per point: {series.n.join(" · ")} — months with 1–2 pairs are less certain</div>}
          {series.temporal && <div className="mt-2"><MetricsGrid m={fromPipeline(series.temporal)} units={series.units} title="Region" /></div>}
        </>
      )}
      {pixel && <div className="mt-3"><MetricsGrid m={pixel} units={pixelUnits ?? ""} title="Selected pixel" /></div>}
    </div>
  );
}
