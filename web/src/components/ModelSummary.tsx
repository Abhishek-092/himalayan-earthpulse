"use client";

/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState } from "react";
import Chart from "./Chart";
import { type Manifest, type RegionEntry, decimalYearToDate, fmt } from "@/lib/data";

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-line p-2">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="num text-base font-semibold">{value}</div>
      {sub && <div className="text-[10px] text-muted">{sub}</div>}
    </div>
  );
}

function Provenance({ granules, product, dates }: { granules: string[]; product: string; dates?: string[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="panel mt-3 p-3">
      <div className="kicker mb-1">Data provenance</div>
      <p className="text-xs text-muted">
        NISAR L2 <b className="text-text">{product}</b> · {granules.length} products
        {dates && dates.length > 0 && <> · {dates[0]} → {dates[dates.length - 1]}</>} · source: ASF DAAC (NASA Earthdata), processed by this project.
      </p>
      <button className="mt-1 text-xs text-accent" onClick={() => setOpen(!open)}>{open ? "hide" : "show"} granule IDs</button>
      {open && (
        <ul className="num mt-1 max-h-40 overflow-y-auto text-[10px] leading-snug text-muted">
          {granules.map((g) => <li key={g}>{g}</li>)}
        </ul>
      )}
    </div>
  );
}

function Limits({ items }: { items: string[] }) {
  return (
    <div className="panel mt-3 p-3">
      <div className="kicker mb-1">Honest limits — what this cannot tell you</div>
      <ul className="list-disc space-y-1 pl-4 text-xs leading-relaxed text-muted">
        {items.map((l) => <li key={l}>{l}</li>)}
      </ul>
    </div>
  );
}

function GDM({ s }: { s: any }) {
  const stacks = Object.entries<any>(s.stacks ?? {});
  const lc = stacks.map(([k, v]) => ({ k, lc: v.landcover_velocity_mm_yr ?? {} }));
  const classes = Array.from(new Set(lc.flatMap((x) => Object.keys(x.lc))));
  const series = stacks.filter(([, v]) => v.reference_series_m);
  return (
    <>
      <div className="panel mt-3 p-3">
        <div className="kicker mb-2">Key measurements (GDES)</div>
        {stacks.map(([k, v]) => (
          <div key={k} className="mb-3">
            <div className="mb-1 text-xs text-muted">{k.startsWith("A") ? "Ascending" : "Descending"} track {v.track} · {v.dates.length} dates · {v.pairs_used}/{v.n_pairs} interferograms used</div>
            <div className="grid grid-cols-2 gap-2">
              <Stat label="Settlement LOS velocity" value={v.settlement_velocity_mm_yr ? `${fmt(v.settlement_velocity_mm_yr.median_mm_yr)} ± ${fmt(v.settlement_velocity_mm_yr.sem_mm_yr)}` : "–"} sub="mm/yr (built-up pixels)" />
              <Stat label="Reliable measured area" value={`${fmt(v.reliable_area_km2, 1)} km²`} sub="persistent scatterers" />
              <Stat label="Noise floor (stable ground)" value={`± ${fmt(v.empirical_velocity_scatter_mm_yr)}`} sub="mm/yr spatial scatter" />
              <Stat label="Significant movers" value={`${fmt(v.deforming_area_km2, 2)} km²`} sub="|v| > 2σ and > 20 mm/yr" />
            </div>
          </div>
        ))}
        {s.decomposition && (
          <p className="text-xs text-muted">Ascending + descending combined → vertical and east-west velocity (north-south assumed small). Subsiding area (reliable &gt; 2σ): <b className="text-text num">{fmt(s.decomposition.subsiding_area_km2, 2)} km²</b>.</p>
        )}
      </div>
      {classes.length > 0 && (
        <div className="panel mt-3 p-3">
          <div className="kicker mb-1">Science check: vegetation bias</div>
          <p className="mb-1 text-xs leading-relaxed text-muted">
            Over forest and grass, short 12-day L-band interferogram chains build up an apparent motion ("phase-closure bias"). Built-up ground and rock don’t.
            That’s why we only trust settlements and bare ground for deformation.
          </p>
          <Chart height={180} data={lc.map((x, i) => ({ type: "bar", name: x.k, x: classes, y: classes.map((c) => x.lc[c]?.median_mm_yr ?? null),
            error_y: { type: "data", array: classes.map((c) => x.lc[c]?.sem_mm_yr ?? 0), color: "#8d97ab" }, marker: { color: i ? "#e76f51" : "#4cc9f0" } }))}
            layout={{ yaxis: { title: { text: "median LOS velocity (mm/yr)" }, gridcolor: "#eceef0" }, barmode: "group" }} />
        </div>
      )}
      {series.length > 0 && (
        <div className="panel mt-3 p-3">
          <div className="kicker mb-1">Stable reference area over time</div>
          <Chart height={180} data={series.map(([k, v], i) => ({ type: "scatter", mode: "lines+markers", name: `${k} reference`, x: v.dates, y: v.reference_series_m.disp.map((d: number) => d * 1000),
            error_y: { type: "data", array: v.reference_series_m.sigma.map((d: number) => d * 1000), visible: true, color: "#555" }, line: { color: i ? "#e76f51" : "#4cc9f0" } }))}
            layout={{ yaxis: { title: { text: "mm" }, gridcolor: "#eceef0" } }} />
          <p className="text-[11px] text-muted">Should stay near zero: it shows the measurement noise you should compare any signal with.</p>
        </div>
      )}
      <Provenance product="GUNW (unwrapped interferograms)" granules={stacks.flatMap(([, v]) => v.granules)} dates={stacks[0]?.[1].dates} />
      <Limits items={s.limitations ?? []} />
    </>
  );
}

function CDM({ s }: { s: any }) {
  const g = s.goff;
  const n = s.ncc;
  const prof = (g?.profile ?? []).filter((p: any) => p.speed_m_yr != null);
  const monthly = Object.entries<any>(g?.monthly_main_median_m_yr ?? {}).filter(([, v]) => v != null);
  return (
    <>
      {g && (
        <div className="panel mt-3 p-3">
          <div className="kicker mb-2">Key measurements (GMES)</div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label={`Median speed${g.main_glacier ? ` · ${g.main_glacier}` : ""}`} value={`${fmt(g.speed_main_median_m_yr, 0)} m/yr`} sub={`${fmt(g.speed_main_median_m_yr / 365.25 * 100, 0)} cm/day`} />
            <Stat label="Fastest 5%" value={`> ${fmt(g.speed_main_p95_m_yr, 0)} m/yr`} />
            <Stat label="Glacier area in view" value={`${fmt(g.glacier_area_km2, 1)} km²`} sub={g.outline_source} />
            <Stat label="Stable ground (should be ~0)" value={`${fmt(g.stable_ground_median_m_yr, 1)} m/yr`} sub="noise check" />
          </div>
          {g.zones && (
            <div className="mt-2 grid grid-cols-3 gap-2">
              {["upper", "middle", "lower"].map((z) => (
                <Stat key={z} label={`${z} section`} value={`${fmt(g.zones[z]?.median_m_yr, 0)} m/yr`} sub={`${fmt((g.zones[z]?.coverage ?? 0) * 100, 0)}% measured`} />
              ))}
            </div>
          )}
          <p className="mt-2 text-[11px] text-muted">{g.n_pairs} NISAR GOFF 12/24-day offset pairs ({g.date_range?.[0]} → {g.date_range?.[1]}), stacked with a robust median.</p>
        </div>
      )}
      {prof.length > 3 && (
        <div className="panel mt-3 p-3">
          <div className="kicker mb-1">Speed along the glacier (top → snout)</div>
          <Chart height={180} data={[{ type: "scatter", mode: "lines", x: prof.map((p: any) => p.dist_m / 1000), y: prof.map((p: any) => p.speed_m_yr), line: { color: "#4cc9f0" }, name: "speed" }]}
            layout={{ xaxis: { title: { text: "distance down-glacier (km)" }, gridcolor: "#eceef0" }, yaxis: { title: { text: "m/yr" }, gridcolor: "#eceef0" }, showlegend: false }} />
        </div>
      )}
      {monthly.length > 2 && (
        <div className="panel mt-3 p-3">
          <div className="kicker mb-1">Seasonal speed (monthly)</div>
          <Chart height={170} data={[{ type: "scatter", mode: "lines+markers", x: monthly.map(([m]) => m), y: monthly.map(([, v]) => v), line: { color: "#4cc9f0" }, name: "median speed" }]}
            layout={{ yaxis: { title: { text: "m/yr" }, gridcolor: "#eceef0" }, showlegend: false }} />
        </div>
      )}
      {g?.coherence && (
        <div className="panel mt-3 p-3">
          <div className="kicker mb-1">Glacier coherence (NISAR GUNW, 12-day)</div>
          <Chart height={170} data={[
            { type: "scatter", mode: "lines+markers", x: g.coherence.mid_dates, y: g.coherence.glacier_median, name: "glacier", line: { color: "#4cc9f0" } },
            { type: "scatter", mode: "lines+markers", x: g.coherence.mid_dates, y: g.coherence.fast_ice_median, name: "fastest 25% of ice", line: { color: "#3a0ca3" } },
            { type: "scatter", mode: "lines+markers", x: g.coherence.mid_dates, y: g.coherence.stable_median, name: "stable rock", line: { color: "#9ca3af" } },
          ]} layout={{ yaxis: { title: { text: "coherence" }, range: [0, 1], gridcolor: "#eceef0" }, legend: { orientation: "h", y: -0.3 } }} />
          <p className="text-[11px] text-muted">
            Mean coherence: glacier <b className="num text-text">{fmt(g.coherence.glacier_mean_coherence, 2)}</b> vs stable rock <b className="num text-text">{fmt(g.coherence.stable_mean_coherence, 2)}</b>
            {g.coherence.spearman_coherence_vs_speed != null && <>; faster ice is less coherent (Spearman ρ = <b className="num text-text">{fmt(g.coherence.spearman_coherence_vs_speed, 2)}</b> with GOFF speed), an independent check on the flow map</>}. {g.coherence.pairs.length} pairs.
          </p>
        </div>
      )}
      <div className="panel mt-3 p-3">
        <div className="kicker mb-1">Independent checks</div>
        {g?.validation_itslive?.n ? (
          <p className="text-xs leading-relaxed text-muted">
            vs <b className="text-text">NASA ITS_LIVE</b> ({g.validation_itslive.source}): MAE <b className="num text-text">{fmt(g.validation_itslive.mae, 1)} m/yr</b>, bias{" "}
            <b className="num text-text">{fmt(g.validation_itslive.bias, 1)}</b>, correlation r = <b className="num text-text">{fmt(g.validation_itslive.pearson_r, 2)}</b> over {g.validation_itslive.n} glacier pixels.
          </p>
        ) : <p className="text-xs text-muted">ITS_LIVE comparison unavailable.</p>}
        {n?.pairs?.length > 0 && (
          <p className="mt-1 text-xs leading-relaxed text-muted">
            Our own feature tracking on NISAR GCOV images ({n.pairs[0].dates.join(" → ")}): glacier median <b className="num text-text">{fmt(n.pairs[0].glacier_median_m_yr, 0)} m/yr</b>, stable-ground noise ± {fmt(n.pairs[0].stable_noise_m, 1)} m
            {n.agreement_with_goff?.n ? <> — agreement with GOFF: MAE {fmt(n.agreement_with_goff.mae, 1)} m/yr, r = {fmt(n.agreement_with_goff.pearson_r, 2)}</> : null}.
          </p>
        )}
      </div>
      {g && <Provenance product="GOFF (pixel offsets) + GCOV" granules={[...g.granules, ...(n?.granules ?? [])]} />}
      <Limits items={s.limitations ?? []} />
    </>
  );
}

function BDM({ s }: { s: any }) {
  const v = s.validation;
  const ser = s.series;
  return (
    <>
      <div className="panel mt-3 p-3">
        <div className="kicker mb-2">Key measurements (FSTS)</div>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="Potential disturbance" value={`${fmt(s.potential_disturbance_km2, 2)} km²`} sub={`${fmt(s.potential_disturbance_pct_of_forest, 2)}% of forest`} />
          <Stat label="Forest analysed" value={`${fmt(s.forest_area_km2, 0)} km²`} sub="ESA WorldCover tree cover" />
          <Stat label="High-confidence" value={`${fmt(s.confidence_km2?.high, 2)} km²`} sub={`medium ${fmt(s.confidence_km2?.medium, 2)} · low ${fmt(s.confidence_km2?.low, 2)}`} />
          <Stat label="Already recovering" value={`${fmt(s.recovering_km2, 2)} km²`} sub="later HV increase (often snow)" />
          <Stat label="Persistent (not recovered)" value={`${fmt(s.persistent_disturbance_km2, 2)} km²`} sub="the more credible signal" />
          <Stat label="Disturbance rate" value={`${fmt(s.disturbance_rate_km2_per_yr, 2)} km²/yr`} />
          <Stat label="Edge density" value={`${fmt(s.fragmentation_after?.edge_density_m_per_km2, 0)}`} sub={`m/km² (before ${fmt(s.fragmentation_before?.edge_density_m_per_km2, 0)})`} />
        </div>
      </div>
      {s.disturbance_by_date?.length > 0 && (
        <div className="panel mt-3 p-3">
          <div className="kicker mb-1">When did it happen?</div>
          <Chart height={180} data={[
            { type: "bar", x: s.disturbance_by_date.map((d: any) => d.date), y: s.disturbance_by_date.map((d: any) => d.new_disturbance_km2), name: "new (km²)", marker: { color: "#80b918" } },
            { type: "scatter", mode: "lines", x: s.disturbance_by_date.map((d: any) => d.date), y: s.disturbance_by_date.map((d: any) => d.cumulative_km2), name: "cumulative", yaxis: "y2", line: { color: "#ffb703" } },
          ]} layout={{ yaxis: { title: { text: "km² per pass" }, gridcolor: "#eceef0" }, yaxis2: { overlaying: "y", side: "right", showgrid: false } }} />
        </div>
      )}
      {ser && (
        <div className="panel mt-3 p-3">
          <div className="kicker mb-1">Radar brightness (HV) through the year</div>
          <Chart height={180} data={[
            { type: "scatter", mode: "lines+markers", x: ser.dates, y: ser.forest_hv_db, name: "undisturbed forest", line: { color: "#80b918" } },
            { type: "scatter", mode: "lines+markers", x: ser.dates, y: ser.disturbed_hv_db, name: "disturbed pixels", line: { color: "#e76f51" } },
          ]} layout={{ yaxis: { title: { text: "HV (dB)" }, gridcolor: "#eceef0" } }} />
          <p className="text-[11px] text-muted">Moisture (snow, monsoon rain) also moves HV up and down — the undisturbed-forest line shows that natural rhythm.</p>
        </div>
      )}
      {(s.coherence || s.hvhh_ratio_db) && (
        <div className="panel mt-3 p-3">
          <div className="kicker mb-1">Second opinions from other radar measures</div>
          {s.coherence && (
            <Chart height={160} data={[
              { type: "scatter", mode: "lines+markers", x: s.coherence.mid_dates, y: s.coherence.forest_median, name: "undisturbed forest", line: { color: "#80b918" } },
              ...(s.coherence.disturbed_median?.some((x: any) => x != null) ? [{ type: "scatter", mode: "lines+markers", x: s.coherence.mid_dates, y: s.coherence.disturbed_median, name: "disturbed pixels", line: { color: "#e76f51" } }] : []),
            ]} layout={{ yaxis: { title: { text: "coherence" }, range: [0, 1], gridcolor: "#eceef0" }, legend: { orientation: "h", y: -0.3 } }} />
          )}
          <table className="mono w-full text-[11px]"><tbody>
            {[
              ...(s.coherence ? [["Coherence: forest / disturbed (mean)", `${fmt(s.coherence.forest_mean_coherence, 2)} / ${fmt(s.coherence.disturbed_mean_coherence, 2)}`],
                ["Coherence change: forest / lasting change", `${fmt(s.coherence.forest_coherence_change, 2)} / ${fmt(s.coherence.lasting_coherence_change, 2)}`]] : []),
              ...(s.hvhh_ratio_db ? [["HV/HH ratio, forest (start)", `${fmt(s.hvhh_ratio_db.forest_pre, 1)} dB`],
                ["HV/HH change: forest / lasting change", `${fmt(s.hvhh_ratio_db.forest_change, 2)} / ${fmt(s.hvhh_ratio_db.lasting_change, 2)} dB`]] : []),
              ...(s.glcm_change_lasting?.contrast != null ? [["Texture change in lasting pixels (contrast / homogeneity / energy)", `${fmt(s.glcm_change_lasting.contrast, 2)} / ${fmt(s.glcm_change_lasting.homogeneity, 3)} / ${fmt(s.glcm_change_lasting.energy, 4)}`]] : []),
            ].map(([a, b]) => <tr key={a} className="border-t border-line"><td className="py-0.5 pr-2 text-muted">{a}</td><td>{b}</td></tr>)}
          </tbody></table>
          {s.second_opinion && (
            <p className={`mt-2 rounded p-2 text-[11px] ${s.second_opinion.supports_structural_change ? "bg-surface-2 text-text" : "bg-amber-soft text-accent"}`}>
              {s.second_opinion.supports_structural_change
                ? <>Supported: the lasting-change pixels also lost coherence ({fmt(s.second_opinion.lasting_vs_forest_coherence_change, 2)} vs ordinary forest) and HV/HH ratio ({fmt(s.second_opinion.lasting_vs_forest_hvhh_change_db, 2)} dB), consistent with a real change in forest structure.</>
                : <>Not supported: the lasting-change pixels behave like ordinary forest in coherence ({fmt(s.second_opinion.lasting_vs_forest_coherence_change, 2)}) and HV/HH ratio ({fmt(s.second_opinion.lasting_vs_forest_hvhh_change_db, 2)} dB), so the HV drop alone may reflect moisture or seasonal effects rather than forest loss. Treat as candidates for checking, not confirmed loss.</>}
            </p>
          )}
          <p className="mt-1 text-[11px] text-muted">{s.coherence?.note ?? ""} Texture (GLCM), HV/HH ratio and coherence maps are in the layer list.</p>
        </div>
      )}
      <div className="panel mt-3 p-3">
        <div className="kicker mb-1">Independent check: Sentinel-2 (optical)</div>
        {v ? (
          <p className="text-xs leading-relaxed text-muted">
            Burn/vegetation-loss index dNBR &gt; 0.1 between {v.pre_window.join("–")} and {v.post_window.join("–")}. Agreement at 60 m blocks: precision{" "}
            <b className="num text-text">{fmt(v.block_60m.precision, 2)}</b>, recall <b className="num text-text">{fmt(v.block_60m.recall, 2)}</b>, F1{" "}
            <b className="num text-text">{fmt(v.block_60m.f1, 2)}</b>. Cloud-free optical coverage: {fmt(v.s2_valid_fraction * 100, 0)}% of forest.
            {v.area_boundary && <> Area: radar <b className="num text-text">{fmt(v.area_boundary.area_detected_km2, 2)} km²</b> vs optical{" "}
              <b className="num text-text">{fmt(v.area_boundary.area_reference_km2, 2)} km²</b> ({v.area_boundary.area_difference_pct != null ? `${v.area_boundary.area_difference_pct > 0 ? "+" : ""}${fmt(v.area_boundary.area_difference_pct, 0)}%` : "–"}).
              {v.area_boundary.boundary_median_distance_m != null && <> Boundary distance: median <b className="num text-text">{fmt(v.area_boundary.boundary_median_distance_m, 0)} m</b>, 95th pct {fmt(v.area_boundary.boundary_hausdorff95_m, 0)} m.</>}</>}
            {v.lasting_only && <> Lasting change only: {fmt(v.lasting_only.area_boundary.area_detected_km2, 2)} km² ({v.lasting_only.area_boundary.area_difference_pct != null ? `${v.lasting_only.area_boundary.area_difference_pct > 0 ? "+" : ""}${fmt(v.lasting_only.area_boundary.area_difference_pct, 0)}%` : "–"} vs optical), F1 {fmt(v.lasting_only.block_60m.f1, 2)}
              {v.lasting_only.area_boundary.boundary_median_distance_m != null && <>, median boundary distance {fmt(v.lasting_only.area_boundary.boundary_median_distance_m, 0)} m</>}.</>}
          </p>
        ) : <p className="text-xs text-muted">No cloud-free Sentinel-2 pair for the main disturbance window (monsoon clouds) — radar is the only view.</p>}
      </div>
      <Provenance product="GCOV (γ⁰ HH + HV backscatter)" granules={s.granules} dates={s.dates} />
      <Limits items={s.limitations ?? []} />
    </>
  );
}

function EPITN({ net }: { net: any }) {
  if (!net) return null;
  const nodes = net.nodes as any[];
  const R = 70, cx = 140, cy = 95;
  const pos: Record<string, [number, number]> = {};
  nodes.forEach((n, i) => {
    const a = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
    pos[n.id] = [cx + R * Math.cos(a), cy + R * Math.sin(a)];
  });
  const color: Record<string, string> = { bdm: "#80b918", gdm: "#e76f51", cdm: "#4cc9f0" };
  return (
    <>
      <div className="panel mt-3 p-3">
        <div className="kicker mb-1">Interaction network</div>
        <svg viewBox="0 0 280 190" className="w-full">
          {net.edges.map((e: any, i: number) => {
            const [x1, y1] = pos[e.source], [x2, y2] = pos[e.target];
            return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={e.significant ? "#ffb703" : "#3a4459"} strokeWidth={1 + Math.min(6, Math.log10(1 + e.weight) * 2)} strokeDasharray={e.significant ? "" : "4 3"} />;
          })}
          {nodes.map((n) => (
            <g key={n.id}>
              <circle cx={pos[n.id][0]} cy={pos[n.id][1]} r={12} fill={color[n.model]} />
              <text x={pos[n.id][0]} y={pos[n.id][1] + 26} textAnchor="middle" fontSize="9" fill="#c9d1e0">{n.id}</text>
            </g>
          ))}
        </svg>
        <p className="text-[11px] text-muted">Solid gold = co-occurrence significantly above chance (p &lt; 0.05, toroidal permutation test); dashed = not significant.</p>
      </div>
      <div className="panel mt-3 p-3">
        <div className="kicker mb-1">Pairwise statistics</div>
        <table className="num w-full text-[11px]">
          <thead className="text-muted"><tr><th className="text-left">pair</th><th>overlap cells</th><th>lift</th><th>p</th></tr></thead>
          <tbody>
            {net.pairs.map((p: any) => (
              <tr key={p.a + p.b} className="border-t border-line">
                <td className="py-1 pr-1">{p.a.split(" (")[0]} × {p.b.split(" (")[0]}</td>
                <td className="text-center">{p.cells_both}</td>
                <td className="text-center">{fmt(p.lift, 2)}×</td>
                <td className={`text-center ${p.significant ? "text-accent" : ""}`}>{fmt(p.p_value, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {net.pairs.filter((p: any) => p.lag).map((p: any) => (
          <p key={p.a + p.b + "lag"} className="mt-1 text-[11px] text-muted">
            Timing ({p.a.split(" (")[0]} → {p.b.split(" (")[0]}): median lag {fmt(p.lag.median_days, 0)} days; {p.lag.a_first} cells first, {p.lag.b_first} second (sign test p = {fmt(p.lag.sign_test_p, 3)}).
          </p>
        ))}
        <p className="mt-2 text-[11px] leading-relaxed text-muted">{net.disclaimer}</p>
      </div>
    </>
  );
}

export default function ModelSummary({ model, manifest }: { model: string; manifest: Manifest; region: RegionEntry }) {
  const s = manifest.summary;
  if (model === "gdm") return <GDM s={s} />;
  if (model === "cdm") return <CDM s={s} />;
  if (model === "bdm") return <BDM s={s} />;
  if (model === "epitn") return <EPITN net={manifest.network} />;
  return null;
}

export { decimalYearToDate };
