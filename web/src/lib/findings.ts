/* eslint-disable @typescript-eslint/no-explicit-any */
import { type Manifest, fmt } from "./data";

export interface StatCard {
  value: string;
  label: string;
  source?: string;
  key?: boolean; // amber "key finding" card
}

export interface Finding {
  kicker: string;
  title: string;
  product: string;
  stats: StatCard[];
  chartTitle: string;
  chartUnits: string;
  chart: unknown[];
  highlightDate?: string;
  highlightLabel?: string;
  mainCube?: string; // layer id driving the timeline
}

const BLUE = "#3b82f6";
const AMBER = "#f59e0b";
const INK = "#1b2430";

export function findingFor(model: string, m: Manifest, regionName: string): Finding | null {
  const s = m.summary;
  const short = regionName.split(",")[0];
  if (model === "gdm") {
    const stacks = Object.entries<any>(s.stacks ?? {});
    const desc = stacks.find(([k]) => k.startsWith("D")) ?? stacks[0];
    const asc = stacks.find(([k]) => k.startsWith("A"));
    const dv = desc?.[1];
    const lc = dv?.landcover_velocity_mm_yr ?? {};
    const bias = lc["tree cover"] && lc["built-up"] ? lc["tree cover"].median_mm_yr - lc["built-up"].median_mm_yr : null;
    const classes = Object.keys(lc);
    const cube = m.layers.find((l) => l.kind === "cube" && l.stack === desc?.[0]) ?? m.layers.find((l) => l.kind === "cube");
    return {
      kicker: `${dv?.dates?.length ?? "–"} passes · ${dv?.dates?.[0]} → ${dv?.dates?.[dv.dates.length - 1]}`,
      title: `Is ${short} sinking?`,
      product: `NISAR L-band · GUNW · ascending + descending · provisional`,
      stats: [
        { value: s.decomposition?.up_mm_yr_reliable_median != null ? `${fmt(s.decomposition.up_mm_yr_reliable_median)} mm/yr` : "–",
          label: "vertical ground motion (median)", source: "asc + desc combined · reliable pixels" },
        { value: dv?.settlement_velocity_mm_yr ? `${fmt(dv.settlement_velocity_mm_yr.median_mm_yr)} ± ${fmt(dv.settlement_velocity_mm_yr.sem_mm_yr)}` : "–",
          label: "mm/yr town motion, line of sight", source: `track ${dv?.track} · built-up pixels` },
        { value: `${fmt(s.decomposition?.subsiding_area_km2 ?? dv?.deforming_area_km2, 2)} km²`,
          label: "significantly subsiding ground", source: "> 2σ and > 20 mm/yr" },
        { value: bias != null ? `${bias > 0 ? "+" : ""}${fmt(bias, 0)} mm/yr` : "–", key: true,
          label: "fake motion over forest — detected and removed", source: "L-band phase-closure bias" },
      ],
      chartTitle: "Apparent velocity by land cover",
      chartUnits: "mm/yr",
      chart: stacks.map(([k, v], i) => ({
        type: "bar", name: `${k.startsWith("A") ? "ascending" : "descending"} view`, x: classes,
        y: classes.map((c) => v.landcover_velocity_mm_yr?.[c]?.median_mm_yr ?? null),
        error_y: { type: "data", array: classes.map((c) => v.landcover_velocity_mm_yr?.[c]?.sem_mm_yr ?? 0), color: "#5b6573" },
        marker: { color: i ? BLUE : AMBER },
      })),
      mainCube: cube?.id,
      ...(asc ? {} : {}),
    };
  }
  if (model === "cdm") {
    const g = s.goff ?? {};
    const monthly = Object.entries<any>(g.monthly_main_median_m_yr ?? {}).filter(([, v]) => v != null);
    let key: StatCard = { value: `> ${fmt(g.speed_main_p95_m_yr, 0)} m/yr`, label: "fastest 5% of the ice", key: true };
    let hi: [string, number] | null = null;
    if (monthly.length >= 3) {
      hi = monthly.reduce((a, b) => (b[1] > a[1] ? b : a)) as [string, number];
      const lo = monthly.reduce((a, b) => (b[1] < a[1] ? b : a)) as [string, number];
      const ppm = g.pairs_per_month ?? {};
      key = { value: `+${fmt(((hi[1] - lo[1]) / lo[1]) * 100, 0)}%`, key: true,
        label: `faster in ${new Date(hi[0] + "-01").toLocaleString("en", { month: "long" })} than in ${new Date(lo[0] + "-01").toLocaleString("en", { month: "long" })} (indicative)`,
        source: `summer speed-up · ${ppm[hi[0]] ?? "?"} vs ${ppm[lo[0]] ?? "?"} offset pairs` };
    }
    return {
      kicker: `${g.n_pairs ?? "–"} offset pairs · ${g.date_range?.[0]} → ${g.date_range?.[1]}`,
      title: `How fast does ${short.replace(" Glacier", "")} flow?`,
      product: "NISAR L-band · GOFF offsets + GCOV feature tracking · provisional",
      stats: [
        { value: `${fmt(g.speed_main_median_m_yr, 0)} m/yr`, label: "typical ice speed (median)", source: "main glacier system" },
        { value: `${fmt(g.speed_main_p95_m_yr, 0)} m/yr`, label: "fastest 5% of the ice", source: "central trunk" },
        { value: `${fmt(g.main_glacier_area_km2 ?? g.glacier_area_km2, 0)} km²`, label: "glacier system measured", source: g.outline_source },
        key,
      ],
      chartTitle: "Glacier speed through the year",
      chartUnits: "m/yr",
      chart: [{ type: "scatter", mode: "lines+markers", fill: "tozeroy", fillcolor: "rgba(59,130,246,0.25)", line: { color: BLUE, width: 2 },
        x: monthly.map(([k]) => k + "-15"), y: monthly.map(([, v]) => v), name: "median speed" }],
      highlightDate: hi?.[0],
      highlightLabel: "FASTEST",
      mainCube: m.layers.find((l) => l.kind === "cube")?.id,
    };
  }
  if (model === "bdm") {
    const rev = s.potential_disturbance_km2 ? (s.recovering_km2 / s.potential_disturbance_km2) * 100 : null;
    const byDate = (s.disturbance_by_date ?? []) as any[];
    const big = byDate.length ? byDate.reduce((a, b) => (b.new_disturbance_km2 > a.new_disturbance_km2 ? b : a)) : null;
    return {
      kicker: `${s.dates?.length ?? "–"} passes · ${s.dates?.[0]} → ${s.dates?.[s.dates.length - 1]}`,
      title: "Where did the forest change?",
      product: "NISAR L-band · GCOV · HH + HV · provisional",
      stats: [
        { value: `${fmt(s.persistent_disturbance_km2 ?? s.potential_disturbance_km2, 2)} km²`, label: s.second_opinion && !s.second_opinion.supports_structural_change
          ? "lasting HV drop - candidate change, not confirmed by coherence or HV/HH" : "lasting forest change (not recovered)",
          source: s.second_opinion ? `Wishart test · coherence/ratio check: ${s.second_opinion.supports_structural_change ? "supports" : "does not support"}` : "Wishart change test · 1% false alarms" },
        { value: `${fmt(s.forest_area_km2, 0)} km²`, label: "forest reliably measured",
          source: s.preprocessing ? `after masking shadow ${fmt(s.preprocessing.shadow_pct, 0)}% · noise ${fmt(s.preprocessing.noise_floor_pct, 0)}%` : "ESA WorldCover tree cover" },
        { value: s.validation ? `${fmt(s.validation.block_60m.f1, 2)}` : "–", label: "agreement with Sentinel-2 (F1)",
          source: s.validation ? `burn index · ${s.validation.post_window[0]}` : "no cloud-free scene" },
        { value: rev != null ? `${fmt(rev, 0)}%` : "–", key: true, label: "of detected change was reversible (winter frost/snow/moisture), not forest loss",
          source: big ? `biggest jump ${big.date}` : "" },
      ],
      chartTitle: "Changed forest per pass",
      chartUnits: "km²",
      chart: [
        { type: "bar", name: "new change", x: byDate.map((d) => d.date), y: byDate.map((d) => d.new_disturbance_km2), marker: { color: BLUE } },
        { type: "scatter", mode: "lines", name: "cumulative", x: byDate.map((d) => d.date), y: byDate.map((d) => d.cumulative_km2),
          line: { color: INK, width: 1.5, dash: "dot" }, yaxis: "y2" },
      ],
      highlightDate: big?.date,
      highlightLabel: "BIGGEST CHANGE",
      mainCube: m.layers.find((l) => l.kind === "cube")?.id,
    };
  }
  if (model === "epitn") {
    const n = m.network ?? {};
    const p = (n.pairs ?? [])[0] ?? {};
    return {
      kicker: `common grid · ${n.cell_m ?? 160} m cells`,
      title: "Do changes happen together?",
      product: "EPITN · BDM × GDM × CDM event layers",
      stats: [
        { value: `${p.cells_both ?? 0}`, label: "cells with two processes", source: `${p.a?.split(" (")[0] ?? ""} × ${p.b?.split(" (")[0] ?? ""}` },
        { value: `${fmt(p.lift, 1)}×`, label: "overlap vs random placement", source: "lift" },
        { value: `${fmt(p.p_value, 2)}`, label: "chance of this overlap at random", source: "toroidal permutation test" },
        { value: p.significant ? "Linked" : "No link", key: true, label: p.significant ? "more overlap than chance" : "no evidence they occur together",
          source: "association is not causation" },
      ],
      chartTitle: "",
      chartUnits: "",
      chart: [],
    };
  }
  return null;
}
