"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { type Manifest, assetUrl, getCatalog, getManifest } from "@/lib/data";

// NISAR L-band frequency A centre frequency in the released L2 products: 1229 MHz -> wavelength 24.4 cm
const LAMBDA_CM = (299792458 / 1.229e9) * 100;
const FRINGE_CM = LAMBDA_CM / 2;

function FringeSim() {
  const ref = useRef<HTMLCanvasElement>(null);
  const [sink, setSink] = useState(8); // cm at bowl centre
  const [band, setBand] = useState<"L" | "C">("L");
  const fringe = band === "L" ? FRINGE_CM : 2.8;
  useEffect(() => {
    const c = ref.current!;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    const img = ctx.createImageData(W, H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const r2 = ((x - W / 2) ** 2 + (y - H / 2) ** 2) / (W * 0.22) ** 2;
        const d = sink * Math.exp(-r2); // cm
        const ph = ((d / fringe) % 1 + 1) % 1; // fraction of a cycle
        const hsv = ph * 6, i = Math.floor(hsv), f = hsv - i;
        const rgb = [[1, f, 0], [1 - f, 1, 0], [0, 1, f], [0, 1 - f, 1], [f, 0, 1], [1, 0, 1 - f]][i % 6];
        const k = (y * W + x) * 4;
        img.data[k] = rgb[0] * 230; img.data[k + 1] = rgb[1] * 230; img.data[k + 2] = rgb[2] * 230; img.data[k + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
  }, [sink, fringe]);
  return (
    <div className="grid gap-4 md:grid-cols-[260px_1fr]">
      <canvas ref={ref} width={260} height={260} className="rounded-lg border border-line" />
      <div className="text-sm leading-relaxed">
        <p>
          A sinking bowl of ground seen by an interferogram. Every full colour cycle (red → yellow → green → blue → magenta → red) is a
          <b> fringe</b>: the ground moved half a radar wavelength closer to or further from the satellite.
        </p>
        <label className="mt-3 block">Sinking at the centre: <b className="num">{sink.toFixed(1)} cm</b>
          <input type="range" min={0} max={40} step={0.5} value={sink} onChange={(e) => setSink(+e.target.value)} className="w-full" />
        </label>
        <div className="mt-2 flex gap-2">
          <button onClick={() => setBand("L")} className={`rounded px-2 py-1 text-xs ${band === "L" ? "bg-ink text-white" : "border border-line"}`}>NISAR L-band (λ = {LAMBDA_CM.toFixed(1)} cm)</button>
          <button onClick={() => setBand("C")} className={`rounded px-2 py-1 text-xs ${band === "C" ? "bg-ink text-white" : "border border-line"}`}>Sentinel-1 C-band (λ = 5.6 cm)</button>
        </div>
        <p className="mt-3 text-muted">
          One fringe = <b className="num text-text">{fringe.toFixed(1)} cm</b> of line-of-sight movement. Number of fringes here: <b className="num text-text">{(sink / fringe).toFixed(1)}</b>.
          {band === "L"
            ? " NISAR’s long L-band wave needs more movement per fringe, but it stays coherent over forests and snow-free slopes where shorter waves lose track."
            : " C-band shows many fringes for the same movement — more sensitive, but over vegetation the echoes decorrelate within days."}
        </p>
      </div>
    </div>
  );
}

export default function Learn() {
  const [bdm, setBdm] = useState<{ region: string; m: Manifest } | null>(null);
  const [filtered, setFiltered] = useState(false);
  useEffect(() => {
    getCatalog().then(async (c) => {
      const r = c.regions.find((x) => x.id === "garhwal-forests" && x.models.bdm) ?? c.regions.find((x) => x.models.bdm);
      if (r) setBdm({ region: r.id, m: await getManifest(r.id, "bdm") });
    });
  }, []);
  const rgb = bdm?.m.layers.find((l) => l.id === "hhhv_rgb");
  const sd = bdm?.m.speckle_demo;

  return (
    <div className="mx-auto max-w-4xl px-5 py-8">
      <div className="kicker">Learn to read radar · 5 minutes</div>
      <h1 className="mt-1 text-3xl font-semibold">A radar image is not a photograph</h1>
      <p className="mt-3 text-muted">
        NISAR does not take pictures with sunlight. It fires microwave pulses at the ground and records the echo. Bright means
        <i> a strong echo</i>, dark means <i>a weak echo</i> — not sunny or shady. Here is how to read it, with real NISAR data.
      </p>

      <section className="panel mt-8 p-5">
        <h2 className="text-xl font-semibold">1 · What makes an echo bright or dark?</h2>
        <div className="mt-3 grid gap-5 md:grid-cols-2">
          <div className="text-sm leading-relaxed">
            <ul className="space-y-2">
              <li><b className="text-[#1d4ed8]">Calm water → black.</b> Smooth surfaces reflect the pulse away like a mirror; almost nothing comes back.</li>
              <li><b className="text-bdm">Forest → bright in HV.</b> L-band waves (24 cm) slip past leaves and bounce around branches and trunks, scrambling the wave’s orientation ("volume scattering").</li>
              <li><b className="text-[#e07be0]">Buildings → very bright in HH.</b> Wall + ground act like a corner reflector ("double bounce").</li>
              <li><b>Slopes facing the radar → bright; slopes facing away → dark</b> or hidden in radar shadow — a geometry effect, not a surface property.</li>
              <li><b>Bare rock, debris, fresh landslide scars → medium,</b> depending on roughness and moisture.</li>
            </ul>
          </div>
          {rgb && bdm ? (
            <figure>
              <img src={assetUrl(bdm.region, "bdm", rgb.png!)} alt="NISAR HH/HV colour composite" className="w-full rounded-lg border border-line bg-black" />
              <figcaption className="mt-1 text-xs text-muted">Real NISAR L-band image ({bdm.m.summary?.dates?.[0]}): HH in magenta, HV in green. Green = forest; magenta = bare slopes, villages and rock; black = rivers and radar shadow.</figcaption>
            </figure>
          ) : <div className="rounded-lg border border-line p-6 text-sm text-muted">Real NISAR example loads here once forest data are processed.</div>}
        </div>
      </section>

      <section className="panel mt-6 p-5">
        <h2 className="text-xl font-semibold">2 · Speckle: the salt-and-pepper grain</h2>
        <div className="mt-3 grid gap-5 md:grid-cols-2">
          <div className="text-sm leading-relaxed">
            <p>Each radar pixel adds up echoes from many small scatterers. Their waves interfere randomly, so even a uniform forest looks grainy. This
              <b> speckle</b> is normal — but measuring change on unsmoothed images produces false alarms.</p>
            <p className="mt-2 text-muted">We use a multi-temporal (Quegan) filter that averages the grain across all dates while keeping each date’s brightness, and our change test models the speckle statistics directly (Wishart distribution with the measured number of looks).</p>
            <button onClick={() => setFiltered(!filtered)} className="mt-3 rounded-md bg-ink px-3 py-1.5 text-sm font-medium text-white">
              {filtered ? "Show raw (speckled)" : "Apply speckle filter"}
            </button>
          </div>
          {sd && bdm ? (
            <figure>
              <img src={assetUrl(bdm.region, "bdm", filtered ? sd.filtered : sd.raw)} alt="speckle demo" className="w-full rounded-lg border border-line" style={{ imageRendering: "pixelated" }} />
              <figcaption className="mt-1 text-xs text-muted">NISAR HV, {sd.date}, {filtered ? "after multi-temporal filtering" : "raw (2×2 looks)"}.</figcaption>
            </figure>
          ) : <div className="rounded-lg border border-line p-6 text-sm text-muted">Speckle demo loads with the forest data.</div>}
        </div>
      </section>

      <section className="panel mt-6 p-5">
        <h2 className="text-xl font-semibold">3 · Polarisation: which way the wave vibrates</h2>
        <p className="mt-2 text-sm leading-relaxed">
          NISAR sends horizontally vibrating waves (<b>H</b>) and listens for both H and vertically vibrating (<b>V</b>) echoes.
          <b> HH</b> (send H, receive H) is strong from surfaces and buildings. <b>HV</b> (send H, receive V) only appears when the wave is scrambled —
          mostly inside vegetation. That is why our forest model watches HV: when HV drops suddenly, branches and trunks are gone.
        </p>
      </section>

      <section className="panel mt-6 p-5">
        <h2 className="text-xl font-semibold">4 · Measuring centimetres of movement (interferometry)</h2>
        <p className="mb-4 mt-2 text-sm leading-relaxed">
          Comparing the exact <i>timing</i> (phase) of the echo on two passes 12 days apart reveals whether the ground moved closer to or further from the
          satellite — along its line of sight. Move the slider:
        </p>
        <FringeSim />
        <p className="mt-4 text-xs leading-relaxed text-muted">
          Caveats we handle and show: the atmosphere and ionosphere also delay the echo (we correct both); one viewing direction only sees line-of-sight
          motion (we combine ascending and descending passes); vegetation adds a small bias that builds up over many 12-day pairs (we only trust settlements and rock).
        </p>
      </section>

      <section className="panel mt-6 p-5">
        <h2 className="text-xl font-semibold">5 · Following a glacier (feature tracking)</h2>
        <p className="mt-2 text-sm leading-relaxed">
          Glaciers move metres per week — too fast for interferometry. Instead we find the same crevasse and debris patterns in two images taken
          weeks or months apart and measure how far they shifted (normalised cross-correlation, sub-pixel accuracy). Distance ÷ time = speed.
          NISAR also provides its own 12-day offset product (GOFF); we use both and check them against NASA’s independent ITS_LIVE velocities.
        </p>
      </section>

      <div className="mt-8 flex gap-3">
        <Link href="/" className="rounded-md bg-ink px-4 py-2 font-medium text-white">Explore the Himalaya →</Link>
        <Link href="/about/" className="rounded-md border border-line px-4 py-2">Methods & limits</Link>
      </div>
    </div>
  );
}
