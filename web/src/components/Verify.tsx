"use client";

import { useEffect, useState } from "react";
import { type RegionEntry, getCatalog } from "@/lib/data";

type Check = { stage: string; check: string; status: "pass" | "warn" | "fail"; value: string; expected: string; note: string };
type Ver = { region: string; checks: Check[]; counts: Record<string, number> };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PLog = Record<string, any>;

const COLOR = { pass: "#80b918", warn: "#ffb703", fail: "#e63946" };

export default function Verify() {
  const [items, setItems] = useState<{ r: RegionEntry; v: Ver; log?: PLog }[]>([]);
  useEffect(() => {
    getCatalog().then(async (c) => {
      const out: { r: RegionEntry; v: Ver; log?: PLog }[] = [];
      for (const r of c.regions) {
        try {
          const res = await fetch(`/data/${r.id}/verification.json`);
          if (!res.ok) continue;
          const lr = await fetch(`/data/${r.id}/processing_log.json`);
          out.push({ r, v: await res.json(), log: lr.ok ? await lr.json() : undefined });
        } catch {}
      }
      setItems(out);
    });
  }, []);
  return (
    <div className="mx-auto max-w-5xl px-5 py-8">
      <div className="kicker">Data quality · preprocessing verification</div>
      <h1 className="mt-1 text-3xl font-semibold">Can we trust the radar inputs?</h1>
      <p className="mt-3 text-sm leading-relaxed text-muted">
        Every preprocessing stage is checked automatically on the real NISAR data of each region: calibration, speckle, common grid, co-registration, terrain
        correction, masks, ionosphere, unwrapping, interferogram quality, reference stability and independent comparisons. Each check shows what was measured
        and what was expected. Warnings are known limitations that the analysis handles (and states).
      </p>
      {items.length === 0 && <p className="mt-6 text-muted">Loading…</p>}
      {items.map(({ r, v, log }) => (
        <section key={r.id} className="panel mt-6 p-4">
          <div className="flex items-baseline justify-between">
            <h2 className="text-lg font-semibold">{r.name}</h2>
            <span className="num text-sm">
              <span style={{ color: COLOR.pass }}>{v.counts.pass} pass</span> · <span style={{ color: COLOR.warn }}>{v.counts.warn} warn</span> ·{" "}
              <span style={{ color: COLOR.fail }}>{v.counts.fail} fail</span>
            </span>
          </div>
          <table className="mt-3 w-full text-left text-xs">
            <thead className="text-muted">
              <tr><th className="py-1 pr-2">Stage</th><th className="pr-2">Check</th><th className="pr-2">Measured</th><th>Expected</th></tr>
            </thead>
            <tbody>
              {v.checks.map((c, i) => (
                <tr key={i} className="border-t border-line align-top">
                  <td className="num py-1.5 pr-2 text-muted">{c.stage}</td>
                  <td className="pr-2"><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: COLOR[c.status] }} />{c.check}
                    {c.note && <div className="mt-0.5 text-[10px] text-muted">{c.note}</div>}</td>
                  <td className="num pr-2">{c.value}</td>
                  <td className="text-muted">{c.expected}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {log && (
            <details className="mt-3 rounded-lg bg-panel-2 p-3 text-xs">
              <summary className="cursor-pointer font-semibold">Processing log — how every map here was produced (reproducibility)</summary>
              {Object.entries(log).map(([model, l]) => (
                <div key={model} className="mt-3">
                  <div className="font-semibold uppercase">{model}</div>
                  <div className="mono text-[11px] text-muted">
                    run {l.run_utc} · code {l.software?.earthpulse_commit} · Python {l.software?.python} · NumPy {l.software?.numpy} ·{" "}
                    {l.inputs?.length} NISAR granules (CRIDs {Array.from(new Set(l.inputs?.map((i: PLog) => i.crid))).join(", ")}; orbits{" "}
                    {Array.from(new Set(l.inputs?.map((i: PLog) => i.orbit_type))).join(", ")}; PGE {Array.from(new Set(l.inputs?.map((i: PLog) => i.pge_version))).join(", ")})
                  </div>
                  <table className="mt-1 w-full">
                    <tbody>
                      {Object.entries(l.parameters ?? {}).map(([k, val]) => (
                        <tr key={k} className="border-t border-line align-top"><td className="mono py-0.5 pr-2 text-muted">{k}</td><td>{String(val)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="mt-1 text-[11px] text-muted">Ancillary: {(l.ancillary ?? []).join(" · ")}</div>
                </div>
              ))}
            </details>
          )}
        </section>
      ))}
    </div>
  );
}
