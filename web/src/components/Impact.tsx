"use client";

/* eslint-disable @typescript-eslint/no-explicit-any */
import Link from "next/link";
import { useEffect, useState } from "react";
import { BASE, type Catalog, type Manifest, fmt, getCatalog, getJSON, getManifest } from "@/lib/data";
import { verdictsFor, type Verdict } from "@/lib/plain";
import { TopicIcon, VerdictCard, Term } from "./Plain";

interface Loaded {
  cat: Catalog;
  verdicts: { id: string; name: string; v: Verdict[] }[];
  imagesProcessed: number;
  checks: { pass: number; warn: number; fail: number };
  clouds: any;
  hazard: any;
}

const PEOPLE = [
  { who: "Residents of hill towns", icon: "ground" as const, need: "Is my town still sinking? Is it safe to stay this week?",
    gives: "A plain answer on ground movement every 12 days, and a daily landslide & flood outlook with what to do.", go: "hazards" },
  { who: "District & state disaster officials", icon: "hazard" as const, need: "Where should we send engineers? Which days need extra readiness?",
    gives: "Maps of flagged ground and slopes, a 7-day outlook tested on 2013-2019, and a downloadable PDF report.", go: "gdm" },
  { who: "Water, hydropower & road planners", icon: "glacier" as const, need: "How fast is glacier ice reaching the rivers? Which slopes threaten our road or dam?",
    gives: "Glacier speeds across the year, slope-failure maps and river-flow outlooks for the river.", go: "cdm" },
  { who: "Forest department & conservation groups", icon: "forest" as const, need: "Where did forest change, even during monsoon when photos are cloudy?",
    gives: "A list of changed patches with how sure we are, separated from harmless winter effects.", go: "bdm" },
  { who: "Students, teachers & researchers", icon: "link" as const, need: "How does space radar work? Can I trust and reuse the results?",
    gives: "Step-by-step explainers, every quality check in the open, and GeoJSON downloads.", go: "/learn/" },
];

export default function Impact() {
  const [d, setD] = useState<Loaded | null>(null);

  useEffect(() => {
    (async () => {
      const cat = await getCatalog();
      const processed = cat.regions.filter((r) => r.status === "processed");
      const verdicts = await Promise.all(processed.map(async (r) => {
        const ms: Record<string, Manifest> = {};
        await Promise.all(Object.keys(r.models).map(async (m) => { try { ms[m] = await getManifest(r.id, m); } catch {} }));
        return { id: r.id, name: r.name, v: verdictsFor(ms, r.name) };
      }));
      let imagesProcessed = 0;
      const checks = { pass: 0, warn: 0, fail: 0 };
      await Promise.all(processed.map(async (r) => {
        try {
          const pl = await getJSON<Record<string, any>>(`${r.id}/processing_log.json`);
          imagesProcessed += Object.values(pl).reduce((a: number, x: any) => a + (x.inputs?.length ?? 0), 0);
        } catch {}
        try {
          const v = await getJSON<any>(`${r.id}/verification.json`);
          checks.pass += v.counts.pass; checks.warn += v.counts.warn; checks.fail += v.counts.fail;
        } catch {}
      }));
      let clouds = null, hazard = null;
      try { clouds = await getJSON<any>("story/clouds.json"); } catch {}
      const hz = cat.regions.find((r) => r.hazards);
      if (hz) { try { hazard = await getJSON<any>(`${hz.id}/hazards/manifest.json`); } catch {} }
      setD({ cat, verdicts, imagesProcessed, checks, clouds, hazard });
    })();
  }, []);

  const linkFor = (go: string) => {
    if (go.startsWith("/")) return go;
    const regs = d?.cat.regions ?? [];
    if (go === "hazards") { const r = regs.find((x) => x.hazards); return r ? `/hazards/${r.id}/` : "/hazards/"; }
    const r = regs.find((x) => x.priority === "hero" && x.models[go]) ?? regs.find((x) => x.models[go]);
    return r ? `/region/${r.id}/` : "/";
  };
  const hero = d?.cat.regions.find((r) => r.priority === "hero") ?? d?.cat.regions.find((r) => r.status === "processed");
  const totalFound = d?.cat.regions.reduce((a, r) => a + Object.values(r.availability).reduce((b, v) => b + v.count, 0), 0);

  return (
    <div className="mx-auto max-w-5xl space-y-10 px-4 py-8 sm:py-12">
      {/* hero */}
      <section>
        <div className="kicker">Why this matters</div>
        <h1 className="mt-2 text-3xl font-bold leading-tight sm:text-4xl">A year of space radar, turned into answers people can use</h1>
        <p className="mt-3 max-w-3xl text-[17px] leading-relaxed text-body">
          The Himalaya changes in ways that are hard to see: ground creeping under towns, glaciers sliding, forests thinning on remote slopes,
          rain-soaked hills ready to fail. Clouds hide much of this from ordinary satellites for months. The <Term k="NISAR">NISAR</Term> radar
          satellite sees through them. We collected its data, checked it, and translated it into plain answers.
        </p>
      </section>

      {/* the numbers */}
      <section>
        <h2 className="text-xl font-bold">What we collected and checked</h2>
        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          {[
            [totalFound ?? "–", "NISAR radar images found", "across the Himalayan places we track"],
            [d?.imagesProcessed ?? "–", "images fully processed", "into maps of ground, ice and forest"],
            [d ? d.checks.pass + d.checks.warn + d.checks.fail : "–", "automatic quality checks", d ? `${d.checks.pass} passed · ${d.checks.warn} with a stated limit · ${d.checks.fail} failed` : ""],
            ["12 days", "between new looks", "the satellite revisits, day or night, in any weather"],
          ].map(([v, l, s]) => (
            <div key={String(l)} className="panel p-4">
              <div className="mono text-3xl font-semibold">{v}</div>
              <div className="mt-1 text-sm font-medium text-text">{l}</div>
              <div className="mt-1 text-xs leading-snug text-muted">{s}</div>
            </div>
          ))}
        </div>
      </section>

      {/* clouds */}
      {d?.clouds && (
        <section className="panel overflow-hidden">
          <div className="grid md:grid-cols-2">
            <figure className="relative">
              <img src={`${BASE}/story/optical_monsoon.png`} alt="Ordinary satellite photo during the monsoon, mostly cloud" className="aspect-square w-full object-cover" />
              <figcaption className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-1 text-xs text-white">Ordinary satellite photo · {d.clouds.s2_date} · {fmt(d.clouds.s2_cloud_cover, 0)}% cloud</figcaption>
            </figure>
            <figure className="relative">
              <img src={`${BASE}/story/radar_monsoon.png`} alt="NISAR radar image of the same area, clear" className="aspect-square w-full object-cover" />
              <figcaption className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-1 text-xs text-white">NISAR radar · {d.clouds.nisar_date} · sees through</figcaption>
            </figure>
          </div>
          <div className="p-5">
            <h2 className="text-lg font-bold">Why radar? Because the monsoon hides the mountains</h2>
            <p className="mt-1 text-sm leading-relaxed text-body">
              Same place, three days apart, in the middle of the monsoon - exactly when landslides and floods happen. The photo satellite saw
              mostly cloud; the radar saw the ground. That is the difference between guessing and measuring.
            </p>
          </div>
        </section>
      )}

      {/* findings */}
      <section>
        <h2 className="text-xl font-bold">What the data tells us</h2>
        <p className="mt-1 text-sm text-muted">Each answer says how sure we are - including when the answer is &quot;we can&apos;t confirm it&quot;.</p>
        <div className="mt-4 space-y-6">
          {d?.verdicts.map((r) => (
            <div key={r.id}>
              <Link href={`/region/${r.id}/`} className="font-semibold hover:underline">{r.name} →</Link>
              <div className="mt-2 grid gap-3 md:grid-cols-2">{r.v.map((v) => <VerdictCard key={v.topic} v={v} />)}</div>
            </div>
          ))}
          {d?.hazard && (
            <div>
              <Link href={`/hazards/${d.hazard.region}/`} className="font-semibold hover:underline">Looking ahead: landslide &amp; flood outlook →</Link>
              <div className="mt-2 rounded-2xl border border-line bg-white p-4 text-sm leading-relaxed text-body">
                Between satellite passes, we combine the radar results with weather forecasts and river models to give a daily outlook.
                Replayed on past years it had never seen, it would have raised its top level the day before the June 2013 Kedarnath disaster,
                and stayed calm for the 2021 and 2023 events that were not rain-driven.
                {d.hazard.replay && <> It also raises some false alarms - about {fmt(d.hazard.replay.combined_alert_days_per_year["orange+"], 0)} &quot;be prepared&quot; days a year - which we show openly.</>}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* who benefits */}
      <section>
        <h2 className="text-xl font-bold">Who benefits, and how</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {PEOPLE.map((p) => (
            <Link key={p.who} href={linkFor(p.go)} className="panel flex gap-3 p-4 transition hover:border-accent hover:shadow-sm">
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-panel-2 text-text"><TopicIcon kind={p.icon} /></span>
              <span>
                <span className="block font-semibold">{p.who}</span>
                <span className="mt-1 block text-sm italic text-muted">&ldquo;{p.need}&rdquo;</span>
                <span className="mt-1 block text-sm leading-relaxed text-body">{p.gives}</span>
              </span>
            </Link>
          ))}
        </div>
      </section>

      {/* trust + limits */}
      <section className="grid gap-4 md:grid-cols-2">
        <div className="panel p-5">
          <h2 className="text-lg font-bold">Why you can trust it</h2>
          <ul className="mt-2 space-y-2 text-sm leading-relaxed text-body">
            <li>✔ Every map is compared with an independent source: NASA glacier speeds, European optical satellites, a second radar method.</li>
            <li>✔ The forecasting models were tested only on years they never saw while learning.</li>
            <li>✔ When a result is weak, we say so - for example, forest changes that other radar measures do not confirm.</li>
            <li>✔ Every processing step and check is published. <Link href="/verify/" className="font-medium text-accent underline">See all checks</Link></li>
          </ul>
        </div>
        <div className="panel p-5">
          <h2 className="text-lg font-bold">What it cannot do (yet)</h2>
          <ul className="mt-2 space-y-2 text-sm leading-relaxed text-body">
            <li>• It is <b>not an official warning system</b>. Follow IMD, NDMA and USDMA.</li>
            <li>• It cannot see movements smaller than about 1-2 cm a year over one year of data.</li>
            <li>• Radar sees that forest changed, not why (fire, landslide, felling).</li>
            <li>• NISAR&apos;s data is still marked provisional by NASA and ISRO; results will improve as more passes arrive.</li>
          </ul>
        </div>
      </section>

      <section className="rounded-2xl bg-ink p-6 text-white">
        <h2 className="text-xl font-bold">Explore it yourself</h2>
        <p className="mt-1 text-sm text-white/75">No account or download needed. Pick a place, drag the timeline, and tap anywhere on the map to see its history.</p>
        <div className="mt-4 flex flex-wrap gap-2">
          {hero && <Link href={`/region/${hero.id}/`} className="rounded-lg bg-white px-4 py-2 text-sm font-semibold text-ink">Open {hero.name.split(",")[0]}</Link>}
          <Link href="/story/" className="rounded-lg border border-white/30 px-4 py-2 text-sm">Watch the 60-s story</Link>
          <Link href="/learn/" className="rounded-lg border border-white/30 px-4 py-2 text-sm">How radar works</Link>
        </div>
      </section>
    </div>
  );
}
