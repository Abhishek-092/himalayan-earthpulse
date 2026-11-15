"use client";

/* eslint-disable @typescript-eslint/no-explicit-any */
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import Chart from "./Chart";
import MapView from "./MapView";
import { type RegionEntry, assetUrl, fmt, getCatalog, getJSON } from "@/lib/data";
import { type HazardModel, type LiveResult, runLive } from "@/lib/hazardLive";
import { LEVEL_PLAIN } from "@/lib/plain";

const LV = ["GREEN", "YELLOW", "ORANGE", "RED"];
const LVC = ["#16a34a", "#eab308", "#f97316", "#dc2626"];
const shortD = (d: string) => new Date(d).toLocaleDateString("en-GB", { day: "numeric", month: "short" });

function Card({ v, l, s, key2 }: { v: string; l: string; s?: string; key2?: boolean }) {
  return (
    <div className={`rounded-xl border p-3 ${key2 ? "border-amber/60 bg-amber-soft" : "border-line bg-white"}`}>
      <div className={`mono text-[19px] leading-tight ${key2 ? "text-accent" : ""}`}>{v}</div>
      <div className={`mt-1 text-[12px] leading-snug ${key2 ? "text-accent" : "text-body"}`}>{l}</div>
      {s && <div className="mono mt-1 text-[10px] text-muted">{s}</div>}
    </div>
  );
}

export default function Hazards({ id }: { id: string }) {
  const [m, setM] = useState<any>(null);
  const [model, setModel] = useState<HazardModel | null>(null);
  const [region, setRegion] = useState<RegionEntry | null>(null);
  const [live, setLive] = useState<LiveResult | null>(null);
  const [liveErr, setLiveErr] = useState<string | null>(null);
  const [useLive, setUseLive] = useState(true);
  const [day, setDay] = useState(0);
  const [layer, setLayer] = useState<"pfs" | "combined" | "susceptibility" | "slope">("pfs");
  const [moverGeo, setMoverGeo] = useState(true);
  const [glcGeo, setGlcGeo] = useState(true);

  useEffect(() => {
    getJSON<any>(`${id}/hazards/manifest.json`).then(setM);
    getJSON<HazardModel>(`${id}/hazards/model.json`).then((mm) => {
      setModel(mm);
      runLive(mm).then(setLive).catch((e) => setLiveErr(String(e)));
    });
    getCatalog().then((c) => setRegion(c.regions.find((r) => r.id === id) ?? null));
  }, [id]);

  // unify snapshot vs live series
  const S = useMemo(() => {
    if (!m) return null;
    const tr = m.trigger;
    if (useLive && live) {
      return { source: `LIVE · Open-Meteo fetched ${new Date(live.fetchedAt).toLocaleString()}`, dates: live.dates, today: live.todayIndex,
        rain: live.rain, r7: live.r7, p24: live.p24, p72: live.p72, p168: live.p168, slope: live.slopeLevel, pls: live.pLandslide,
        q: live.q ?? { dates: m.flood.forecast.dates, q: m.flood.forecast.q, qmax: m.flood.forecast.q_max } };
    }
    return { source: `SNAPSHOT · forecast issued ${m.forcing.forecast.issued_utc} UTC`, dates: tr.dates, today: tr.today_index, rain: tr.rain,
      r7: tr.r7, p24: tr.p24h, p72: tr.p72h, p168: tr.p168h, slope: m.landslide ? m.landslide.levels : tr.slope_level,
      pls: m.landslide ? m.landslide.p_daily : undefined,
      q: { dates: m.flood.forecast.dates, q: m.flood.forecast.q, qmax: m.flood.forecast.q_max } };
  }, [m, live, useLive]);

  if (!m || !S || !region) return <div className="p-8 text-muted">Loading hazard model…</div>;
  const rl = m.flood.return_levels;
  const nF = m.trigger.pfs_dates.length;
  const di = Math.min(day, nF - 1);
  const ti = S.today + di;
  // alert for selected day (slope trigger over 3 days, river p72, GloFAS level)
  const sl = Math.max(...S.slope.slice(ti, ti + 3));
  const p72 = S.p72[ti] ?? 0;
  const bands = model?.alert_bands_p72 ?? [0.15, 0.35, 0.6];
  const rv = p72 >= bands[2] ? 3 : p72 >= bands[1] ? 2 : p72 >= bands[0] ? 1 : 0;
  const qi = S.q.dates.indexOf(S.dates[ti]);
  const qNext = qi >= 0 ? Math.max(...S.q.qmax.slice(qi, qi + 7).filter((v: number | null) => v != null)) : NaN;
  const fl = qNext >= rl.Q20 ? 3 : qNext >= rl.Q5 ? 2 : qNext >= rl.Q2 ? 1 : 0;
  const lvl = Math.max(sl, rv, fl);
  const levelOn = (k: number) => {
    const t = S.today + k;
    const a = Math.max(...S.slope.slice(t, t + 3));
    const p = S.p72[t] ?? 0;
    const r = p >= bands[2] ? 3 : p >= bands[1] ? 2 : p >= bands[0] ? 1 : 0;
    const j = S.q.dates.indexOf(S.dates[t]);
    const qq = j >= 0 ? Math.max(...S.q.qmax.slice(j, j + 7).filter((v: number | null) => v != null)) : NaN;
    const f = qq >= rl.Q20 ? 3 : qq >= rl.Q5 ? 2 : qq >= rl.Q2 ? 1 : 0;
    return { level: Math.max(a, r, f), rain: S.rain[t] ?? 0 };
  };
  const week = Array.from({ length: Math.min(7, nF) }, (_, k) => ({ k, date: S.dates[S.today + k], ...levelOn(k) }));
  const reasons: string[] = [];
  if (sl) reasons.push(m.landslide
    ? `Landslide model (trained on ${m.landslide.validation.events_used} NASA-catalogued landslides) at its ${["", "watch", "warning", "critical"][sl]} level within 72 h.`
    : `7-day rain ${fmt(Math.max(...S.r7.slice(ti, ti + 3)), 0)} mm ≥ ${["", "90th", "95th", "99th"][sl]} percentile for this site (LHASA-style landslide trigger).`);
  if (rv) reasons.push(`${(p72 * 100).toFixed(0)}% chance the Alaknanda reaches its 2-year flood level within 72 h.`);
  if (fl) reasons.push(`Forecast discharge ${fmt(qNext, 0)} m³/s ≥ ${["", "2", "5", "20"][fl]}-year level.`);
  if (!reasons.length) reasons.push("Rain, soil wetness and river flow are within normal ranges for this date.");
  const kf = m.kalman;
  const v = m.validation;
  const pfsLayer = m.layers.find((l: any) => l.id === layer);
  const pfsFile = pfsLayer.kind === "cube" ? pfsLayer.frames[Math.min(di, pfsLayer.frames.length - 1)] : pfsLayer.png;
  const gdmFindings = (region as any).findings?.gdm ? `/data/${id}/gdm/findings.geojson` : null;

  return (
    <div className="mx-auto max-w-[1400px] space-y-4 p-4">
      {/* plain-language guidance + 7-day outlook */}
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <div className="rounded-2xl p-5" style={{ background: LVC[lvl] + "14", border: `2px solid ${LVC[lvl]}55` }}>
          <div className="text-xs font-semibold uppercase tracking-wide text-muted">Landslide &amp; flood outlook · {region.name.split(",")[0]} · {di === 0 ? "today" : shortD(S.dates[ti])}</div>
          <div className="mt-1 flex items-center gap-3">
            <span className="h-9 w-9 rounded-full" style={{ background: LVC[lvl] }} />
            <span className="text-3xl font-bold" style={{ color: LVC[lvl] }}>{LEVEL_PLAIN[lvl].word}</span>
          </div>
          <p className="mt-2 text-[15px] text-text">{LEVEL_PLAIN[lvl].say}</p>
          <ul className="mt-1 list-disc pl-5 text-sm text-body">{reasons.map((r) => <li key={r}>{r}</li>)}</ul>
          <div className="mt-3 rounded-xl bg-white/80 p-3">
            <div className="text-xs font-semibold text-muted">What to do</div>
            <p className="mt-0.5 text-sm leading-relaxed text-body">{LEVEL_PLAIN[lvl].todo}</p>
          </div>
          <p className="mt-3 text-[11px] leading-relaxed text-muted">
            This outlook is an experiment built from satellite radar, weather forecasts and river models. It is <b>not</b> an official warning -
            always follow IMD, NDMA and Uttarakhand State Disaster Management Authority (USDMA) instructions. Emergency: <b>112</b>.
          </p>
        </div>
        <div className="panel p-5">
          <div className="flex items-baseline justify-between">
            <div className="text-sm font-semibold">Next 7 days at a glance</div>
            <div className="text-[11px] text-muted">tap a day to see its map</div>
          </div>
          <div className="-mx-1 mt-3 overflow-x-auto px-1 pb-1"><div className="grid min-w-[460px] grid-cols-7 gap-1.5">
            {week.map((w) => (
              <button key={w.k} onClick={() => setDay(w.k)} className={`rounded-xl border p-2 text-center transition ${w.k === di ? "border-ink shadow" : "border-line hover:border-muted"}`}>
                <div className="text-[11px] font-medium text-muted">{w.k === 0 ? "Today" : new Date(w.date).toLocaleDateString("en-GB", { weekday: "short" })}</div>
                <div className="text-[10px] text-muted">{shortD(w.date)}</div>
                <div className="mx-auto mt-1.5 h-7 w-7 rounded-full" style={{ background: LVC[w.level] }} title={LEVEL_PLAIN[w.level].word} />
                <div className="mt-1 text-[10px] font-semibold leading-tight" style={{ color: LVC[w.level] }}>{LEVEL_PLAIN[w.level].word}</div>
                <div className="mono mt-1 text-[10px] text-muted">{fmt(w.rain, 0)} mm</div>
              </button>
            ))}
          </div></div>
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
            {LEVEL_PLAIN.map((p, i) => <span key={p.word} className="inline-flex items-center gap-1"><span className="h-2.5 w-2.5 rounded-full" style={{ background: LVC[i] }} />{p.word}</span>)}
            <span>· mm = forecast rain that day</span>
          </div>
          <div className="mt-4 grid grid-cols-1 gap-2 text-xs sm:grid-cols-3">
            <div className="rounded-lg bg-panel-2 p-2.5"><div className="font-semibold text-text">Tested on past years</div>Caught the June 2013 Kedarnath disaster a day ahead in a replay that never saw that year.</div>
            <div className="rounded-lg bg-panel-2 p-2.5"><div className="font-semibold text-text">Honest about errors</div>{m.replay ? `About ${fmt(m.replay.combined_alert_days_per_year["orange+"], 0)} "Be prepared" days a year; many pass without incident.` : "False alarms are counted and shown below."}</div>
            <div className="rounded-lg bg-panel-2 p-2.5"><div className="font-semibold text-text">Updates live</div>Uses today&apos;s weather forecast, computed right in your browser.</div>
          </div>
        </div>
      </div>

      {/* the map: which slopes could fail */}
      <div className="panel relative h-[520px] overflow-hidden">
        <div className="absolute inset-0">
          <MapView bounds={region.bbox} basemap="light"
            overlays={[{ id: "haz", url: assetUrl(id, "hazards", pfsFile), corners: pfsLayer.corners, opacity: 0.85 }]}
            geo={[
              ...(gdmFindings && moverGeo ? [{ id: "movers", data: gdmFindings, kind: "line" as const, color: "#1e3a8a", width: 2 }] : []),
              ...(m.landslide && glcGeo ? [{ id: "glc", data: m.landslide.events_nearby, kind: "points" as const, color: "#7c2d12" }] : []),
            ]} />
        </div>
        <div className="absolute top-4 right-4 z-10 w-60 rounded-2xl bg-white/95 p-3 text-[13px] shadow-lg">
          <div className="mb-1 text-xs font-semibold text-muted">Show on the map</div>
          {[["pfs", "Slopes that could fail on this day"], ["susceptibility", "Slopes that could fail if soaked"]].map(([k, t]) => (
            <label key={k} className="flex items-center gap-2 py-0.5"><input type="radio" checked={layer === k} onChange={() => setLayer(k as typeof layer)} />{t}</label>
          ))}
          {gdmFindings && <label className="mt-1 flex items-center gap-2 border-t border-line pt-1"><input type="checkbox" checked={moverGeo} onChange={(e) => setMoverGeo(e.target.checked)} />Ground NISAR saw moving</label>}
          {m.landslide && <label className="flex items-center gap-2"><input type="checkbox" checked={glcGeo} onChange={(e) => setGlcGeo(e.target.checked)} /><span className="h-2.5 w-2.5 rounded-full bg-[#7c2d12]" />Past landslides ({m.landslide.events_nearby.features.length})</label>}
        </div>
        <div className="absolute bottom-4 left-4 z-10 rounded-xl bg-white/95 p-3 text-[12px] shadow">
          <div className="text-[13px] font-semibold">Which slopes could fail? {layer === "pfs" ? `· ${shortD(m.trigger.pfs_dates[di])}` : "· if soaked by heavy rain"}</div>
          <div className="mt-1.5 h-2.5 w-56 rounded-full" style={{ background: `linear-gradient(90deg, ${pfsLayer.legend.stops.join(",")})` }} />
          <div className="mt-1 flex w-56 justify-between text-[11px] text-body"><span>Safe</span><span>Risky</span></div>
        </div>
      </div>

      {/* everything technical, on request */}
      <details className="group panel">
        <summary className="flex cursor-pointer list-none items-center justify-between p-5">
          <span><span className="font-semibold">Numbers, charts and test results</span><span className="ml-2 text-xs text-muted">for experts</span></span>
          <span className="text-sm text-muted transition group-open:rotate-180">▾</span>
        </summary>
        <div className="space-y-4 px-5 pb-5">
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        {/* left */}
        <aside className="panel space-y-3 p-5">
          <div className="flex items-center justify-between rounded-lg bg-panel-2 px-3 py-2 text-xs">
            <span className="mono">{S.source}</span>
            <button onClick={() => setUseLive(!useLive)} disabled={!live} className="rounded border border-line bg-white px-2 py-0.5 disabled:opacity-40">
              {useLive ? "use snapshot" : "use live"}
            </button>
          </div>
          {liveErr && <div className="text-xs text-accent">Live fetch failed — showing the build-time snapshot.</div>}
          <div>
            <div className="flex justify-between text-sm"><span className="font-semibold">Forecast day</span><span className="mono">{shortD(S.dates[ti])} · +{di} d</span></div>
            <input type="range" min={0} max={nF - 1} value={di} onChange={(e) => setDay(+e.target.value)} className="w-full" />
            <div className="mono flex justify-between text-[10px] text-muted"><span>today</span><span>+{nF - 1} days</span></div>
          </div>
          <div className="grid grid-cols-2 gap-2.5">
            <Card v={p72 < 0.01 ? "< 1%" : `${fmt(p72 * 100, 0)}%`} l="chance the river rises to a level seen about once every 2 years, within 3 days" s={`tested on ${v.test?.split(" ")[0] ?? "unseen years"} · AUC ${fmt(v.logistic.roc_auc, 2)}`} />
            <Card v={`${fmt(Math.max(...S.r7.slice(ti, ti + 3)), 0)} mm`} l="rain over the last 7 days (wet hills slide more easily)" s={`warning levels ${fmt(m.trigger.lhasa.p90, 0)}/${fmt(m.trigger.lhasa.p95, 0)}/${fmt(m.trigger.lhasa.p99, 0)} mm`} />
            {S.pls && m.landslide ? (() => {
              const seg = S.pls.slice(ti, ti + 3);
              const p3 = 1 - seg.reduce((a: number, v: number) => a * (1 - v), 1);
              const clim3 = 1 - Math.pow(1 - m.landslide.climatological_mean, 3);
              return <Card v={`${fmt(p3 / clim3, 1)}× normal`} l="landslide chance in the next 3 days compared with a normal day (1× = normal)" s={`p = ${(p3 * 100).toFixed(2)}% · NASA GLC-trained`} />;
            })() : <Card v={`${fmt(m.trigger.regional_pfs[di] * 100, 0)}%`} l="of steep-slope scenarios unstable" s="infinite-slope physics, Monte Carlo" />}
            <Card key2 v={kf ? `${fmt(kf.velocity_mm_yr, 1)} ± ${fmt(kf.velocity_sd_mm_yr, 1)}` : "–"} l="mm/yr estimated town ground movement until the next satellite pass (± = uncertainty; 0 is inside the range)" s={kf ? `next NISAR pass ≈ ${kf.next_pass}` : ""} />
          </div>
          <div>
            <div className="text-sm font-semibold">Rain: observed + forecast</div>
            <Chart height={170} data={[
              { type: "bar", x: S.dates, y: S.rain, name: "daily rain (mm)", marker: { color: S.dates.map((_: string, i: number) => (i < S.today ? "#93c5fd" : "#3b82f6")) } },
              { type: "scatter", mode: "lines", x: S.dates, y: S.r7, name: "7-day total", yaxis: "y2", line: { color: "#1b2430", width: 1.5 } },
              { type: "scatter", mode: "lines", x: [S.dates[0], S.dates[S.dates.length - 1]], y: [m.trigger.lhasa.p95, m.trigger.lhasa.p95], yaxis: "y2", name: "95th pct (orange)", line: { color: "#f97316", dash: "dash", width: 1 } },
            ]} layout={{ yaxis2: { overlaying: "y", side: "right", showgrid: false }, shapes: [{ type: "line", x0: S.dates[ti], x1: S.dates[ti], yref: "paper", y0: 0, y1: 1, line: { color: "#92400e", width: 1.5 } }], margin: { l: 36, r: 36, t: 6, b: 30 } }} />
          </div>
          <div>
            <div className="text-sm font-semibold">Chance of a river high-flow day</div>
            <Chart height={150} data={[
              { type: "scatter", mode: "lines", x: S.dates.slice(S.today), y: S.p24.slice(S.today), name: "24 h", line: { color: "#93c5fd" } },
              { type: "scatter", mode: "lines", x: S.dates.slice(S.today), y: S.p72.slice(S.today), name: "72 h", line: { color: "#3b82f6", width: 2 } },
              { type: "scatter", mode: "lines", x: S.dates.slice(S.today), y: S.p168.slice(S.today), name: "7 days", line: { color: "#1e3a8a" } },
            ]} layout={{ yaxis: { range: [0, 1], tickformat: ".0%" }, margin: { l: 40, r: 8, t: 6, b: 30 } }} />
          </div>
        </aside>

        {/* map + river */}
        <div className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="panel p-4">
              <div className="text-sm font-semibold">Alaknanda river discharge (GloFAS)</div>
              <Chart height={200} data={[
                { type: "scatter", mode: "lines", x: m.flood.recent.dates, y: m.flood.recent.q, name: "observed (reanalysis)", line: { color: "#93c5fd" } },
                { type: "scatter", mode: "lines", x: S.q.dates, y: S.q.q, name: "forecast", line: { color: "#3b82f6", width: 2 } },
                ...[["Q2", "#eab308"], ["Q5", "#f97316"], ["Q20", "#dc2626"]].map(([k, c]) => ({ type: "scatter", mode: "lines", x: [m.flood.recent.dates[0], S.q.dates[S.q.dates.length - 1]], y: [rl[k], rl[k]], name: `${k.slice(1)}-yr level`, line: { color: c, dash: "dot", width: 1 } })),
              ]} layout={{ yaxis: { title: { text: "m³/s" } }, margin: { l: 44, r: 8, t: 6, b: 30 } }} />
              <p className="text-[11px] text-muted">{m.forcing.flood.note}</p>
              {m.flood.record_break?.note && <p className="mt-1 rounded bg-amber-soft p-2 text-[11px] text-accent">Data issue found: {m.flood.record_break.note}</p>}
            </div>
            {kf && (
              <div className="panel p-4">
                <div className="text-sm font-semibold">Ground state between NISAR passes (Kalman filter)</div>
                <Chart height={200} data={[
                  { x: [...kf.days, ...[...kf.days].reverse()], y: [...kf.mean_mm.map((v: number, i: number) => v + kf.sd_mm[i]), ...[...kf.mean_mm.map((v: number, i: number) => v - kf.sd_mm[i])].reverse()], fill: "toself", fillcolor: "rgba(59,130,246,0.15)", line: { width: 0 }, type: "scatter", name: "±1σ", hoverinfo: "skip" },
                  { x: kf.days, y: kf.mean_mm, type: "scatter", mode: "lines", line: { color: "#3b82f6" }, name: "daily estimate" },
                  { x: kf.obs_dates, y: kf.obs_mm, type: "scatter", mode: "markers", marker: { color: "#1b2430", size: 6 }, error_y: { type: "data", array: kf.obs_sd_mm, color: "#5b6573" }, name: "NISAR pass" },
                ]} layout={{ yaxis: { title: { text: "mm (line of sight)" } }, margin: { l: 44, r: 8, t: 6, b: 30 } }} />
                <p className="text-[11px] text-muted">Uncertainty grows between passes and shrinks when NISAR observes. Settlement pixels of stack {kf.stack} ({kf.pixels} px).</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {m.nisar_soil_moisture && (
        <section className="panel grid gap-4 p-5 lg:grid-cols-[1fr_360px]">
          <div>
            <div className="text-sm font-semibold">NISAR soil moisture (SME2) vs ERA5-Land, Joshimath</div>
            <Chart height={200} data={[
              { type: "scatter", mode: "markers", x: m.nisar_soil_moisture.dates, y: m.nisar_soil_moisture.nisar, name: "NISAR SME2 (L3, 200 m)",
                marker: { color: "#1b2430", size: 7 }, error_y: { type: "data", array: m.nisar_soil_moisture.uncertainty, color: "#5b6573" } },
              { type: "scatter", mode: "lines+markers", x: m.nisar_soil_moisture.dates, y: m.nisar_soil_moisture.era5_0_7cm, name: "ERA5-Land 0–7 cm (~9 km)",
                line: { color: "#3b82f6" }, marker: { size: 4 } },
            ]} layout={{ yaxis: { title: { text: "m³/m³" } }, margin: { l: 44, r: 8, t: 6, b: 30 } }} />
          </div>
          <div className="text-sm leading-relaxed text-body">
            <div className="kicker mb-1">Does NISAR soil moisture help here?</div>
            <p>
              Over {m.nisar_soil_moisture.comparison.n} NISAR passes the two disagree: correlation r ={" "}
              <b className="mono">{fmt(m.nisar_soil_moisture.comparison.pearson_r, 2)}</b>, NISAR drier by{" "}
              <b className="mono">{fmt(-m.nisar_soil_moisture.comparison.bias, 2)} m³/m³</b>.
            </p>
            <p className="mt-2">
              {m.nisar_soil_moisture.comparison.pearson_r < 0.4
                ? "So we do NOT use SME2 to drive the slope model yet. L-band soil-moisture retrieval is hardest on steep, forested slopes, the product is provisional, and the grids differ (200 m vs 9 km). It is shown for transparency and as a future input once validated with ground probes."
                : "The agreement supports using soil moisture as the slope-wetness forcing; NISAR fixes the state on pass days."}
            </p>
          </div>
        </section>
      )}

      {/* validation */}
      <section className="px-5 pb-5">
        <div className="kicker">Does it work? · validation on data the model never saw</div>
        <div className="mt-3 grid gap-5 lg:grid-cols-3">
          <div>
            <div className="text-sm font-semibold">River high-flow model (logistic, 5 features)</div>
            <table className="mono mt-2 w-full text-xs">
              <tbody>
                {[["Test period", v.test], ["Events in test", v.logistic.events], ["ROC-AUC", fmt(v.logistic.roc_auc, 2)], ["Brier score", fmt(v.logistic.brier, 3)],
                  ["vs climatology", fmt(v.logistic.brier_climatology, 3)], ["Brier skill", fmt(v.logistic.brier_skill_score, 3)],
                  ["Baseline: 3-day rain ≥ 50 mm", `AUC ${fmt(v.fixed_rain_threshold.roc_auc, 2)} · FAR ${fmt(v.fixed_rain_threshold.false_alarm_ratio, 2)}`]].map(([a, b]) => (
                  <tr key={String(a)} className="border-t border-line"><td className="py-1 pr-2 text-muted">{a}</td><td>{String(b)}</td></tr>
                ))}
              </tbody>
            </table>
            <Chart height={180} data={[{ type: "scatter", mode: "lines", x: v.roc.fpr, y: v.roc.tpr, name: "model", line: { color: "#3b82f6" } }, { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], name: "chance", line: { color: "#c9ced6", dash: "dot" } }]}
              layout={{ xaxis: { title: { text: "false-positive rate" } }, yaxis: { title: { text: "hit rate" } }, legend: { orientation: "h", y: -0.45 }, margin: { l: 44, r: 8, t: 6, b: 36 } }} />
            <p className="text-[11px] text-muted">{v.note} Label: {v.label}.</p>
          </div>
          <div>
            <div className="text-sm font-semibold">Documented events (hindcast)</div>
            <table className="mt-2 w-full text-xs">
              <thead className="text-muted"><tr><th className="text-left">Event</th><th>Landslide model</th><th>River (GloFAS)</th></tr></thead>
              <tbody>
                {m.hindcasts.map((h: any) => {
                  const fe = m.flood.events.find((e: any) => e.date === h.date);
                  return (
                    <tr key={h.date} className="border-t border-line align-top">
                      <td className="py-1.5 pr-2"><a href={h.source.startsWith("http") ? h.source : undefined} className="underline-offset-2 hover:underline">{h.date}</a> · {h.event}<div className="text-[10px] text-muted">{h.place}</div></td>
                      <td className="text-center"><span style={{ color: LVC[LV.indexOf(h.landslide_level_day_before ?? h.slope_trigger_level)] }} className="font-semibold">{h.landslide_level_day_before ?? h.slope_trigger_level}</span><div className="text-[10px] text-muted">{h.landslide_p72_day_before != null ? `p72 ${(h.landslide_p72_day_before * 100).toFixed(1)}%` : ""} · LHASA {h.slope_trigger_level}</div></td>
                      <td className="text-center">{fe ? <>{fmt(fe.q_max_m3s, 0)} m³/s<div className="text-[10px] text-muted">{fe.outcome}</div></> : "–"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-2 text-[11px] text-muted">
              The slope trigger fires ORANGE on ~{fmt(v.slope_trigger.orange_days_per_year, 0)} days/yr and RED on ~{fmt(v.slope_trigger.red_days_per_year, 0)} days/yr (mostly monsoon). {v.slope_trigger.note}
              {" "}GloFAS (≈5 km) does not resolve the 2013 flash flood peak at this river cell — a known limit of global river models.
            </p>
          </div>
          <div>
            <div className="text-sm font-semibold">Kalman engine: predict the next NISAR pass</div>
            {kf?.hindcast?.n ? (
              <>
                <table className="mono mt-2 w-full text-xs">
                  <tbody>
                    {[["Passes predicted", kf.hindcast.n], ["RMSE Kalman", `${fmt(kf.hindcast.rmse_kf_mm, 1)} mm`], ["RMSE persistence", `${fmt(kf.hindcast.rmse_persistence_mm, 1)} mm`],
                      ["Within 68% interval", `${fmt(kf.hindcast.coverage_68 * 100, 0)}%`], ["Within 95% interval", `${fmt(kf.hindcast.coverage_95 * 100, 0)}%`]].map(([a, b]) => (
                      <tr key={String(a)} className="border-t border-line"><td className="py-1 pr-2 text-muted">{a}</td><td>{String(b)}</td></tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-2 text-[11px] text-muted">
                  Hyper-parameters re-fitted before each pass using only earlier passes (no leakage). Honest result: the town is not moving measurably (Phase 1), so simply repeating the last pass (persistence) predicts best;
                  the filter&apos;s intervals are wider than needed. The engine matters when the ground starts to move — it then tracks velocity
                  and rainfall response between 12-day passes.
                </p>
              </>
            ) : <p className="text-xs text-muted">Not enough passes.</p>}
          </div>
        </div>
        <div className="mt-5 grid gap-5 border-t border-line pt-4 lg:grid-cols-3">
          {[["River high-flow model", v, v.logistic], ["Landslide model", m.landslide?.validation, m.landslide?.validation?.logistic]].filter(([, x]) => x).map(([name, V, T]: any) => (
            <div key={name}>
              <div className="text-sm font-semibold">{name}: calibration on unseen years</div>
              <Chart height={190} data={[
                { type: "scatter", mode: "lines+markers", x: V.reliability.map((b: any) => b.p_mean), y: V.reliability.map((b: any) => b.obs_freq), name: "model", line: { color: "#3b82f6" } },
                { type: "scatter", mode: "lines", x: [0, Math.max(...V.reliability.map((b: any) => Math.max(b.p_mean, b.obs_freq)))], y: [0, Math.max(...V.reliability.map((b: any) => Math.max(b.p_mean, b.obs_freq)))], name: "perfect", line: { color: "#c9ced6", dash: "dot" } },
              ]} layout={{ xaxis: { title: { text: "predicted probability" } }, yaxis: { title: { text: "observed frequency" } }, legend: { orientation: "h", y: -0.45 }, margin: { l: 48, r: 8, t: 6, b: 36 } }} />
              <table className="mono w-full text-[11px]"><tbody>
                {[["Calibrator (chosen without test data)", V.calibrator], ["ECE", fmt(T.ece, 4)], ["Calibration slope / intercept", `${fmt(T.calibration?.slope, 2)} / ${fmt(T.calibration?.intercept, 2)}`],
                  ["Mean predicted vs observed", `${fmt(T.mean_predicted * 100, 2)}% vs ${fmt(T.observed_rate * 100, 2)}%`], ["ROC-AUC (raw scores)", fmt(T.roc_auc, 3)], ["Brier skill", fmt(T.brier_skill_score, 3)],
                  ["Gradient boosting AUC / skill", V.gradient_boosting ? `${fmt(V.gradient_boosting.roc_auc, 3)} / ${fmt(V.gradient_boosting.brier_skill_score, 3)}` : "–"]].map(([a, b]) => (
                  <tr key={String(a)} className="border-t border-line"><td className="py-0.5 pr-2 text-muted">{a}</td><td>{String(b)}</td></tr>
                ))}
              </tbody></table>
            </div>
          ))}
          {m.replay && (
            <div>
              <div className="text-sm font-semibold">Full advisory replay, {m.replay.period}</div>
              <table className="mono mt-2 w-full text-[11px]">
                <thead className="text-muted"><tr><th className="text-left">level</th><th>hit rate</th><th>false alarms</th><th>alarm days/yr</th></tr></thead>
                <tbody>
                  {(["landslide", "river"] as const).map((k) => Object.entries<any>(m.replay[k]).map(([lvl, r]) => (
                    <tr key={k + lvl} className="border-t border-line"><td className="py-0.5">{k} {lvl}</td><td className="text-center">{fmt(r.pod * 100, 0)}%</td><td className="text-center">{fmt(r.far * 100, 0)}%</td><td className="text-center">{fmt(r.alarm_days_per_year, 0)}</td></tr>
                  )))}
                </tbody>
              </table>
              <p className="mt-1 text-[11px] text-muted">{m.replay.note} Combined advisory: yellow+ {fmt(m.replay.combined_alert_days_per_year["yellow+"], 0)}, orange+ {fmt(m.replay.combined_alert_days_per_year["orange+"], 0)}, red {fmt(m.replay.combined_alert_days_per_year.red, 0)} days/yr. Floods ≥ 5-yr level detected: {m.flood.pod_annual_max_above_Q5?.detected ?? "–"}/{m.flood.pod_annual_max_above_Q5?.floods ?? "–"}.</p>
            </div>
          )}
        </div>

        {m.landslide && (() => {
          const L = m.landslide.validation;
          return (
            <div className="mt-5 grid gap-5 border-t border-line pt-4 lg:grid-cols-3">
              <div>
                <div className="text-sm font-semibold">Landslide model vs NASA LHASA rule</div>
                <p className="text-[11px] text-muted">Trained on {L.events_used} rain-triggered landslides from the NASA Global Landslide Catalog in {L.cells} Uttarakhand cells ({L.train_years}), tested on {L.test_years}.</p>
                <table className="mono mt-2 w-full text-xs">
                  <thead className="text-muted"><tr><th className="text-left">test {L.test_years}</th><th>model</th><th>LHASA r7</th></tr></thead>
                  <tbody>
                    <tr className="border-t border-line"><td className="py-1 text-muted">ROC-AUC</td><td className="text-center">{fmt(L.logistic.roc_auc, 2)}</td><td className="text-center">{fmt(L.lhasa_r7.roc_auc, 2)}</td></tr>
                    <tr className="border-t border-line"><td className="py-1 text-muted">Hit rate (5% alarm days)</td><td className="text-center">{fmt(L.logistic.at_p95_equiv.pod, 2)}</td><td className="text-center">{fmt(L.lhasa_r7.p95.pod, 2)}</td></tr>
                    <tr className="border-t border-line"><td className="py-1 text-muted">False-alarm ratio</td><td className="text-center">{fmt(L.logistic.at_p95_equiv.far, 3)}</td><td className="text-center">{fmt(L.lhasa_r7.p95.far, 3)}</td></tr>
                    <tr className="border-t border-line"><td className="py-1 text-muted">Brier skill</td><td className="text-center">{fmt(L.logistic.brier_skill_score, 3)}</td><td className="text-center">–</td></tr>
                  </tbody>
                </table>
              </div>
              <div>
                <Chart height={200} data={[
                  { type: "scatter", mode: "lines", x: L.roc_logistic.fpr, y: L.roc_logistic.tpr, name: "model", line: { color: "#3b82f6", width: 2 } },
                  { type: "scatter", mode: "lines", x: L.roc_lhasa.fpr, y: L.roc_lhasa.tpr, name: "LHASA 7-day rain", line: { color: "#f59e0b" } },
                  { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], name: "chance", line: { color: "#c9ced6", dash: "dot" } },
                ]} layout={{ xaxis: { title: { text: "false-positive rate" } }, yaxis: { title: { text: "hit rate" } }, legend: { orientation: "h", y: -0.45 }, margin: { l: 44, r: 8, t: 6, b: 36 } }} />
              </div>
              <div>
                <div className="text-sm font-semibold">Leave-one-year-out (robustness)</div>
                <table className="mono mt-2 w-full text-xs">
                  <thead className="text-muted"><tr><th className="text-left">year</th><th>events</th><th>AUC model</th><th>AUC LHASA</th></tr></thead>
                  <tbody>{L.leave_one_year_out.map((r: any) => (
                    <tr key={r.year} className="border-t border-line"><td className="py-0.5">{r.year}</td><td className="text-center">{r.events}</td><td className="text-center">{fmt(r.auc, 2)}</td><td className="text-center">{fmt(r.auc_lhasa, 2)}</td></tr>
                  ))}</tbody>
                </table>
                <p className="mt-2 text-[11px] text-muted">{L.caveat}</p>
              </div>
            </div>
          );
        })()}
        <div className="mt-4 rounded-xl bg-panel-2 p-3 text-xs text-body">
          <b>Assumptions:</b> {m.assumptions.join(" ")}
        </div>
      </section>
        </div>
      </details>
      <p className="text-center text-xs text-muted">
        <Link href={`/region/${id}/`} className="underline">← Back to the map of {region.name.split(",")[0]}</Link> · Data: Open-Meteo (ERA5/ERA5-Land, NWP forecast), GloFAS v4, NISAR GUNW, Copernicus DEM, ESA WorldCover.
      </p>
    </div>
  );
}
