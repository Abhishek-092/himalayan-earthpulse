"use client";

/* eslint-disable @typescript-eslint/no-explicit-any */
import Link from "next/link";
import { useEffect, useState } from "react";
import { type Manifest, type RegionEntry, MODEL_INFO, assetUrl, fmt, getCatalog, getManifest } from "@/lib/data";

function Figure({ region, model, m, id, caption }: { region: string; model: string; m: Manifest; id: string; caption?: string }) {
  const l = m.layers.find((x) => x.id === id);
  if (!l) return null;
  const src = assetUrl(region, model, l.kind === "cube" ? l.frames![l.frames!.length - 1] : l.png!);
  return (
    <figure className="panel p-2">
      <img src={src} alt={l.title} className="w-full rounded bg-[#e8eae4]" style={{ imageRendering: "pixelated" }} />
      <figcaption className="mt-1 text-[11px] text-muted print-dark-text">
        <b>{l.title}</b>
        {l.legend.type === "continuous" && <> · colour range {fmt(l.legend.min)} to {fmt(l.legend.max)} {l.units}</>}
        {caption && <> · {caption}</>}
      </figcaption>
    </figure>
  );
}

export default function Report({ id }: { id: string }) {
  const [r, setR] = useState<RegionEntry | null>(null);
  const [ms, setMs] = useState<Record<string, Manifest>>({});
  useEffect(() => {
    getCatalog().then(async (c) => {
      const reg = c.regions.find((x) => x.id === id) ?? null;
      setR(reg);
      if (!reg) return;
      const out: Record<string, Manifest> = {};
      await Promise.all(Object.keys(reg.models).map(async (m) => (out[m] = await getManifest(id, m))));
      setMs(out);
    });
  }, [id]);
  if (!r) return <div className="p-8 text-muted">Loading…</div>;
  const g = ms.gdm?.summary, c = ms.cdm?.summary, b = ms.bdm?.summary, e = ms.epitn?.network;
  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="no-print mb-4 flex justify-between">
        <Link href={`/region/${id}/`} className="text-sm text-muted hover:text-text">← Back to the interactive map</Link>
        <button onClick={() => window.print()} className="rounded-md bg-ink px-3 py-1.5 text-sm font-medium text-white">Save as PDF / print</button>
      </div>
      <div className="kicker">Earth history report · Himalayan EarthPulse</div>
      <h1 className="mt-1 text-3xl font-semibold">{r.name}</h1>
      <p className="mt-2 text-muted print-dark-text">{r.story}</p>
      <p className="num mt-2 text-xs text-muted print-dark-text">
        Area: {r.bbox.map((v) => v.toFixed(3)).join(", ")} (W, S, E, N) · NISAR products available: {Object.entries(r.availability).map(([p, v]) => `${p} ${v.count}`).join(", ")} ·
        record {r.availability.GCOV?.first ?? "–"} → {r.availability.GCOV?.last ?? "–"} · generated {new Date().toISOString().slice(0, 10)}
      </p>

      {g && (
        <section className="mt-8">
          <h2 className="text-xl font-semibold" style={{ color: MODEL_INFO.gdm.color }}>Ground deformation (GDM · GDES)</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            {Object.entries<any>(g.stacks).map(([k, v]) => (
              <li key={k}>
                {k.startsWith("A") ? "Ascending" : "Descending"} track {v.track}: {v.dates.length} acquisitions ({v.dates[0]} → {v.dates[v.dates.length - 1]}), {v.pairs_used}/{v.n_pairs} interferograms passed quality control.
                Settlement line-of-sight velocity <b className="num">{fmt(v.settlement_velocity_mm_yr?.median_mm_yr)} ± {fmt(v.settlement_velocity_mm_yr?.sem_mm_yr)} mm/yr</b>; stable-ground noise ±{fmt(v.empirical_velocity_scatter_mm_yr)} mm/yr;
                significant movers {fmt(v.deforming_area_km2, 2)} km² of {fmt(v.reliable_area_km2, 1)} km² reliable area.
              </li>
            ))}
            {g.decomposition && (
              <li>Vertical velocity (asc + desc combined, reliable pixels): median <b className="num">{fmt(g.decomposition.up_mm_yr_reliable_median)} mm/yr</b> over {fmt(g.decomposition.reliable_area_km2, 1)} km²; significantly subsiding area {fmt(g.decomposition.subsiding_area_km2, 2)} km².</li>
            )}
          </ul>
          <p className="mt-2 text-sm text-muted print-dark-text">
            Interpretation: within the stated uncertainty, the settlement does not show significant line-of-sight motion during this record. Velocities over forest and grassland are dominated by the L-band phase-closure bias and are excluded.
          </p>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Figure region={id} model="gdm" m={ms.gdm} id="vertical_velocity" />
            <Figure region={id} model="gdm" m={ms.gdm} id={`${Object.keys(g.stacks)[0]}_coherence`} />
          </div>
        </section>
      )}

      {c?.goff && (
        <section className="mt-8">
          <h2 className="text-xl font-semibold" style={{ color: MODEL_INFO.cdm.color }}>Glacier movement (CDM · GMES)</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            <li>Main glacier{c.goff.main_glacier ? ` (${c.goff.main_glacier})` : ""}: median surface speed <b className="num">{fmt(c.goff.speed_main_median_m_yr, 0)} m/yr</b>, fastest 5% above {fmt(c.goff.speed_main_p95_m_yr, 0)} m/yr, from {c.goff.n_pairs} NISAR GOFF pairs.</li>
            {c.goff.zones && <li>Upper / middle / lower sections: {["upper", "middle", "lower"].map((z) => `${fmt(c.goff.zones[z]?.median_m_yr, 0)}`).join(" / ")} m/yr.</li>}
            {c.goff.validation_itslive?.n && <li>Independent check vs NASA ITS_LIVE (2014-2022): MAE {fmt(c.goff.validation_itslive.mae, 1)} m/yr, bias {fmt(c.goff.validation_itslive.bias, 1)} m/yr, r = {fmt(c.goff.validation_itslive.pearson_r, 2)}.</li>}
            {c.ncc?.pairs?.[0] && <li>Our own NISAR GCOV feature tracking ({c.ncc.pairs[0].dates.join(" → ")}): glacier median {fmt(c.ncc.pairs[0].glacier_median_m_yr, 0)} m/yr.</li>}
          </ul>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Figure region={id} model="cdm" m={ms.cdm} id="speed_goff" />
            <Figure region={id} model="cdm" m={ms.cdm} id="speed_itslive" />
          </div>
        </section>
      )}

      {b && (
        <section className="mt-8">
          <h2 className="text-xl font-semibold" style={{ color: MODEL_INFO.bdm.color }}>Forest change (BDM · FSTS)</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            <li>Potential forest disturbance: <b className="num">{fmt(b.potential_disturbance_km2, 2)} km²</b> ({fmt(b.potential_disturbance_pct_of_forest, 2)}% of {fmt(b.forest_area_km2, 0)} km² forest) over {b.dates.length} acquisitions; high confidence {fmt(b.confidence_km2?.high, 2)} km².</li>
            <li>Already recovering: {fmt(b.recovering_km2, 2)} km². Edge density {fmt(b.fragmentation_before?.edge_density_m_per_km2, 0)} → {fmt(b.fragmentation_after?.edge_density_m_per_km2, 0)} m/km².</li>
            {b.validation && <li>Sentinel-2 dNBR agreement (60 m blocks): precision {fmt(b.validation.block_60m.precision, 2)}, recall {fmt(b.validation.block_60m.recall, 2)}, F1 {fmt(b.validation.block_60m.f1, 2)}.</li>}
          </ul>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Figure region={id} model="bdm" m={ms.bdm} id="disturbance_date" />
            <Figure region={id} model="bdm" m={ms.bdm} id="hhhv_rgb" />
          </div>
        </section>
      )}

      {e && (
        <section className="mt-8">
          <h2 className="text-xl font-semibold" style={{ color: MODEL_INFO.epitn.color }}>Process interactions (EPITN)</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            {e.pairs.map((p: any) => (
              <li key={p.a + p.b}>{p.a} × {p.b}: {p.cells_both} overlapping cells, lift {fmt(p.lift, 2)}×, p = {fmt(p.p_value, 3)} {p.significant ? "(more overlap than chance)" : "(not significant)"}.</li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-muted print-dark-text">{e.disclaimer}</p>
        </section>
      )}

      <section className="mt-8">
        <h2 className="text-xl font-semibold">Limits and confidence</h2>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-muted print-dark-text">
          {[...(g?.limitations ?? []), ...(c?.limitations ?? []), ...(b?.limitations ?? [])].map((l: string) => <li key={l}>{l}</li>)}
        </ul>
      </section>
      <p className="mt-8 text-xs text-muted print-dark-text">
        Data: NASA-ISRO NISAR L2 products (GCOV, GUNW, GOFF; provisional release) via ASF DAAC, processed with the open-source Himalayan EarthPulse pipeline.
        Ancillary: Copernicus DEM GLO-30, ESA WorldCover 2021, JRC Global Surface Water, RGI 7.0, NASA ITS_LIVE, Sentinel-2 L2A. This report presents
        satellite-observed changes, not confirmed physical causes, and is not an official hazard warning.
      </p>
    </div>
  );
}
