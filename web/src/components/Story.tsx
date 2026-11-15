"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import MapView from "./MapView";
import { findingFor } from "@/lib/findings";
import { type Catalog, type Layer, type Manifest, assetUrl, fmt, getCatalog, getJSON, getManifest } from "@/lib/data";

type Clouds = { region: string; nisar_date: string; s2_date: string; s2_cloud_cover: number; nisar_granule: string; s2_item: string };

const nice = (d?: string) => (d ? new Date(d).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" }) : "");

function Panel({ chip, children, caption }: { chip: string; children: React.ReactNode; caption?: string }) {
  return (
    <figure className="flex h-full min-h-[360px] flex-col">
      <div className="relative min-h-[320px] flex-1 overflow-hidden rounded-2xl border border-line bg-[#e8eae4]">
        <div className="mono absolute top-3 left-3 z-10 rounded-md bg-ink/85 px-2.5 py-1 text-[11px] text-white">{chip}</div>
        {children}
      </div>
      {caption && <figcaption className="mt-2 text-xs text-muted">{caption}</figcaption>}
    </figure>
  );
}

function LayerMap({ region, model, layer, bbox }: { region: string; model: string; layer?: Layer; bbox: [number, number, number, number] }) {
  if (!layer) return null;
  const file = layer.kind === "cube" ? layer.frames![layer.frames!.length - 1] : layer.png!;
  return (
    <>
      <div className="absolute inset-0">
        <MapView key={layer.id} bounds={bbox} basemap="light" overlays={[{ id: layer.id, url: assetUrl(region, model, file), corners: layer.corners, opacity: 0.92 }]} />
      </div>
      {layer.legend.type === "continuous" && (
        <div className="absolute bottom-3 left-3 z-10 rounded-lg bg-white/95 p-2 text-[11px] shadow">
          <div className="h-2 w-44 rounded" style={{ background: `linear-gradient(90deg, ${layer.legend.stops!.join(",")})` }} />
          <div className="mono mt-0.5 flex w-44 justify-between text-[10px] text-muted"><span>{fmt(layer.legend.min, 0)}</span><span>{layer.units}</span><span>{fmt(layer.legend.max, 0)}</span></div>
        </div>
      )}
    </>
  );
}

export default function Story() {
  const [cat, setCat] = useState<Catalog | null>(null);
  const [ms, setMs] = useState<Record<string, Manifest>>({});
  const [clouds, setClouds] = useState<Clouds | null>(null);
  const [step, setStep] = useState(0);

  useEffect(() => {
    getCatalog().then(async (c) => {
      setCat(c);
      const out: Record<string, Manifest> = {};
      await Promise.all(c.regions.flatMap((r) => Object.keys(r.models).map(async (m) => { out[`${r.id}/${m}`] = await getManifest(r.id, m); })));
      setMs(out);
    });
    getJSON<Clouds>("story/clouds.json").then(setClouds).catch(() => {});
  }, []);

  const reg = (id: string) => cat?.regions.find((r) => r.id === id);
  const key = (rid: string, m: string) => {
    const man = ms[`${rid}/${m}`];
    const r = reg(rid);
    return man && r ? findingFor(m, man, r.name)?.stats.find((s) => s.key) : undefined;
  };
  const lay = (rid: string, m: string, lid: string) => ms[`${rid}/${m}`]?.layers.find((l) => l.id === lid);

  const steps = [
    {
      title: "Clouds blind normal satellites",
      text: clouds
        ? `On ${nice(clouds.s2_date)}, in the middle of the monsoon, ${fmt(clouds.s2_cloud_cover, 0)}% of the Sentinel-2 image over Joshimath was cloud. A camera in space saw almost nothing of the ground.`
        : "During the monsoon, camera satellites see mostly cloud.",
      big: clouds ? { v: `${fmt(clouds.s2_cloud_cover, 0)}%`, l: "of the optical image was cloud — the ground was invisible" } : null,
      visual: clouds && (
        <div className="grid h-full gap-4 md:grid-cols-2">
          <Panel chip={`OPTICAL · ${nice(clouds.s2_date).toUpperCase()}`} caption={`Sentinel-2 L2A true colour (${clouds.s2_item.slice(0, 22)}…), via Microsoft Planetary Computer`}>
            <img src="/data/story/optical_monsoon.png" alt="Cloudy Sentinel-2 image" className="absolute inset-0 h-full w-full object-cover" />
          </Panel>
          <Panel chip={`NISAR RADAR · ${nice(clouds.nisar_date).toUpperCase()}`} caption="NISAR L-band GCOV, HH magenta / HV green — the same area through the clouds">
            <img src="/data/story/radar_monsoon.png" alt="NISAR radar image" className="absolute inset-0 h-full w-full object-cover" style={{ imageRendering: "pixelated" }} />
          </Panel>
        </div>
      ),
      strip: "Radar brings its own light: NISAR sends 24-cm microwaves and records the echo, day or night, through cloud, smoke and rain.",
    },
    {
      title: "NISAR radar sees through",
      text: "Every 12 days NISAR passes overhead. Bright means a strong echo — not sunlight. Forests scramble the wave (green, HV), rock and buildings reflect it (magenta, HH), smooth water sends it away (black).",
      big: { v: "12 days", l: "between NISAR passes over the same spot — a time-lapse of the surface" },
      visual: (
        <div className="grid h-full gap-4 md:grid-cols-2">
          <Panel chip="NISAR RADAR · GARHWAL FORESTS" caption="Green = forest canopy; magenta = villages, rock and bare slopes; black = rivers and radar shadow">
            {lay("garhwal-forests", "bdm", "hhhv_rgb") && <img src={assetUrl("garhwal-forests", "bdm", lay("garhwal-forests", "bdm", "hhhv_rgb")!.png!)} alt="NISAR radar composite" className="absolute inset-0 h-full w-full object-cover" />}
          </Panel>
          <div className="panel flex flex-col justify-center gap-4 p-5 text-sm leading-relaxed">
            <div><b>Volume scattering</b> — inside a forest the wave bounces between branches and trunks and comes back twisted (HV). Lose the trees, lose the HV echo.</div>
            <div><b>Interferometry</b> — comparing the exact timing of two echoes 12 days apart reveals ground movement of millimetres to centimetres.</div>
            <div><b>Feature tracking</b> — matching crevasse patterns between images shows how far a glacier has flowed.</div>
          </div>
        </div>
      ),
      strip: "We use NISAR’s map-ready Level-2 products (GCOV, GUNW, GOFF) and process them ourselves — 1,540 products found for our 8 regions.",
    },
    {
      title: "Is Joshimath still sinking?",
      text: "In January 2023 cracks split homes in Joshimath. Combining NISAR’s ascending and descending views over Nov 2025 – Oct 2026, the town itself shows no significant movement beyond a few millimetres per year.",
      big: key("joshimath", "gdm") ? { v: findingFor("gdm", ms["joshimath/gdm"]!, "Joshimath")!.stats[1].value + " mm/yr", l: "town motion along the radar line of sight — within the noise. Vegetated slopes show fake motion that we detect and remove." } : null,
      visual: reg("joshimath") && (
        <Panel chip="NISAR GUNW · VERTICAL VELOCITY" caption="Reliable pixels only (settlements, bare rock). Red = sinking, blue = rising.">
          <LayerMap region="joshimath" model="gdm" layer={lay("joshimath", "gdm", "vertical_velocity")} bbox={reg("joshimath")!.bbox} />
        </Panel>
      ),
      strip: "Honest science: over forest and grass, 12-day L-band interferograms build up an apparent motion (phase-closure bias). We measured it — and excluded those pixels.",
    },
    {
      title: "Gangotri keeps flowing",
      text: "The source glacier of the Ganga moves tens of metres a year. Its central trunk flows fastest, and the whole glacier speeds up in summer when meltwater lubricates its bed.",
      big: key("gangotri", "cdm") ? { v: key("gangotri", "cdm")!.value, l: key("gangotri", "cdm")!.label } : null,
      visual: reg("gangotri") && (
        <Panel chip="NISAR GOFF · ICE SPEED" caption="Stacked 12-day NISAR offsets; checked against NASA ITS_LIVE and our own feature tracking.">
          <LayerMap region="gangotri" model="cdm" layer={lay("gangotri", "cdm", "speed_goff")} bbox={reg("gangotri")!.bbox} />
        </Panel>
      ),
      strip: "Glacier speed is not glacier melt: velocity tells us how ice moves, not how much is lost.",
    },
    {
      title: "Winter changes that bounce back",
      text: "Radar flagged a sudden drop in forest echo across Garhwal in February 2026. Most of it bounced back weeks later — reversible winter effects (frost and snow on the higher slopes; part of it is still unexplained), not trees disappearing. What remains is a smaller area of lasting change worth checking on the ground.",
      big: key("garhwal-forests", "bdm") ? { v: key("garhwal-forests", "bdm")!.value, l: key("garhwal-forests", "bdm")!.label } : null,
      visual: reg("garhwal-forests") && (
        <Panel chip="NISAR GCOV · WHEN FOREST ECHO DROPPED" caption="Wishart omnibus change test, 1% false-alarm level; colour = pass of first change.">
          <LayerMap region="garhwal-forests" model="bdm" layer={lay("garhwal-forests", "bdm", "disturbance_date")} bbox={reg("garhwal-forests")!.bbox} />
        </Panel>
      ),
      strip: "Radar change alone cannot prove deforestation. We mask radar shadow, layover and instrument noise, flag frozen/snowy pixels, check against Sentinel-2 and track whether a change persists or recovers.",
    },
    {
      title: "What we can — and cannot — say",
      text: "Everything here is measured from provisional NISAR data, with uncertainty, quality checks and independent comparisons. Overlap between processes is association, not cause. Nothing here is an official warning.",
      big: { v: "0 fails", l: "in the automatic preprocessing checks for all three analysed places (warnings are stated limits)" },
      visual: (
        <div className="grid h-full gap-4 md:grid-cols-3">
          {[["joshimath", "Is the ground moving?"], ["gangotri", "How fast is the ice?"], ["garhwal-forests", "Where did the forest change?"]].map(([r, q]) => (
            <Link key={r} href={`/region/${r}/`} className="panel flex flex-col justify-between p-5 transition hover:border-ink">
              <div className="kicker">Explore</div>
              <div className="mt-2 text-xl font-bold">{reg(r)?.name}</div>
              <div className="mt-1 text-sm text-body">{q}</div>
              <div className="mt-6 text-sm font-medium">Open the map →</div>
            </Link>
          ))}
        </div>
      ),
      strip: "Every number carries its date and source. Methods, data and limits are open in the app and the code.",
    },
  ];
  const s = steps[step];

  return (
    <div className="grid gap-6 px-6 py-6 lg:h-[calc(100vh-4rem)] lg:grid-cols-[420px_1fr]">
      <div className="flex flex-col">
        <ol className="space-y-3">
          {steps.map((st, i) => (
            <li key={st.title}>
              <button onClick={() => setStep(i)} className="flex items-center gap-3 text-left text-[15px]">
                <span className={`mono grid h-7 w-7 place-items-center rounded-full text-xs ${i === step ? "bg-amber text-ink font-semibold" : i < step ? "bg-ink text-white" : "border border-line bg-white text-muted"}`}>{i + 1}</span>
                <span className={i === step ? "font-semibold" : "text-body"}>{st.title}</span>
              </button>
            </li>
          ))}
        </ol>
        <div className="panel mt-6 p-6">
          <div className="kicker text-accent">Step {step + 1} of {steps.length}</div>
          <h1 className="mt-2 text-[30px] font-bold leading-tight">{s.title}</h1>
          <p className="mt-3 text-[15px] leading-relaxed text-body">{s.text}</p>
          {s.big && (
            <div className="mt-4 flex items-center gap-4 rounded-xl border border-amber/60 bg-amber-soft p-4">
              <div className="mono text-2xl text-accent">{s.big.v}</div>
              <div className="text-sm leading-snug text-accent">{s.big.l}</div>
            </div>
          )}
          <div className="mt-5 flex gap-2">
            <button disabled={step === 0} onClick={() => setStep(step - 1)} className="rounded-lg border border-line bg-white px-4 py-2 text-sm disabled:opacity-40">← Back</button>
            {step < steps.length - 1 ? (
              <button onClick={() => setStep(step + 1)} className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white">Next: {steps[step + 1].title.charAt(0).toLowerCase() + steps[step + 1].title.slice(1).replace(/^nISAR/, "NISAR")} →</button>
            ) : (
              <Link href="/region/joshimath/" className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white">Explore the map →</Link>
            )}
          </div>
        </div>
        <Link href="/region/joshimath/" className="mt-4 text-sm text-[#1d4ed8] underline">Skip story · explore the map yourself</Link>
      </div>
      <div className="flex min-h-0 flex-col gap-4">
        <div className="h-[60vh] lg:h-auto lg:min-h-0 lg:flex-1">{s.visual}</div>
        <div className="panel p-4 text-sm leading-relaxed text-body">{s.strip}</div>
      </div>
    </div>
  );
}
