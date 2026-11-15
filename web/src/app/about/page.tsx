import Link from "next/link";

export const metadata = { title: "Methods & limits · Himalayan EarthPulse" };

function Sec({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel mt-6 p-5">
      <h2 className="text-lg font-semibold">{title}</h2>
      <div className="mt-2 space-y-2 text-sm leading-relaxed text-muted [&_b]:text-text">{children}</div>
    </section>
  );
}

export default function About() {
  return (
    <div className="mx-auto max-w-4xl px-5 py-8">
      <div className="kicker">Methods, data, limits, AI use</div>
      <h1 className="mt-1 text-3xl font-semibold">How Himalayan EarthPulse works</h1>
      <p className="mt-3 text-muted">
        One open-source pipeline turns NISAR Level-2 products into measured, uncertainty-aware change maps for three Earth processes, then explains them for
        non-experts. Regions are defined only by a name and a box — tracks, models, reference areas and masks are detected automatically.
      </p>

      <Sec title="Data">
        <p><b>NISAR (NASA-ISRO), L-band, provisional L2 release, Nov 2025 – Nov 2026</b>, searched with the ASF API and read with HTTPS byte-range requests
          (only the study-area chunks of each 1–9 GB HDF5 file are downloaded):</p>
        <ul className="list-disc pl-5">
          <li><b>GUNW</b> — geocoded unwrapped interferograms (80 m): ground deformation.</li>
          <li><b>GOFF</b> — geocoded pixel offsets (80 m): glacier velocity.</li>
          <li><b>GCOV</b> — RTC γ⁰ backscatter covariance, HH + HV (10 m): forest change and our own glacier feature tracking.</li>
        </ul>
        <p>Context and validation: Copernicus DEM GLO-30, ESA WorldCover 2021, JRC Global Surface Water, RGI 7.0 glacier outlines, NASA ITS_LIVE velocity
          mosaics, Sentinel-2 L2A (Microsoft Planetary Computer). The released files report a frequency-A centre of 1229 MHz (λ = 24.4 cm), which we read
          from each granule instead of assuming a constant.</p>
      </Sec>

      <Sec title="GDM — ground deformation (GDES)">
        <p>Per interferogram: subtract the ionosphere phase screen; mask coherence &lt; 0.4 and unreliable connected components; fit a robust ramp +
          elevation-correlated troposphere model; <b>re-unwrap the smooth residual</b> and tie disconnected coherent islands to the model by whole cycles
          (repairs unwrapping errors in snow and monsoon pairs); data-driven pair quality control. Then an <b>SBAS inversion</b> with Cramér-Rao
          coherence weights and temporal smoothing, an automatically chosen two-pass stable reference area, per-pixel velocity / acceleration / onset via the
          shared temporal framework, and <b>ascending + descending decomposition</b> into vertical and east-west motion.</p>
        <p><b>Science finding:</b> 12-day L-band chains show a vegetation-dependent phase-closure ("fading") bias — apparent velocities of tens of mm/yr over
          forest and grass while settlements and rock stay near zero. Deformation is therefore reported only on reliable persistent scatterers, and the bias is shown
          in the app as a diagnostic.</p>
      </Sec>

      <Sec title="CDM — glacier movement (GMES)">
        <p>(A) NISAR GOFF 12/24-day offsets, SNR and correlation filtered, converted to horizontal east/north velocity with LOS and along-track unit vectors and the
          surface-parallel-flow assumption, debiased on stable ground, stacked with a robust median (plus monthly medians for seasonality). (B) Our own
          <b> normalised cross-correlation feature tracking</b> on refined-Lee filtered GCOV HH images ~1 year apart. Both are compared with each other and
          with <b>NASA ITS_LIVE</b>. Zone statistics (upper/middle/lower) and a centre-line speed profile use RGI 7.0 outlines.</p>
      </Sec>

      <Sec title="BDM — forest change (FSTS)">
        <p>2×2 multilooked GCOV HH/HV stack; the <b>complex-Wishart omnibus likelihood-ratio test</b> with sequential change-point detection (Conradsen et al.
          2016) at a 1% false-alarm level gives each change a p-value and a date. "Potential disturbance" = significant HV decrease inside forest; confidence
          combines p-value strength with texture (GLCM entropy) and dB agreement; recovery = later significant HV increase; fragmentation and edge density
          before/after; validation with Sentinel-2 dNBR (precision, recall, F1).</p>
      </Sec>

      <Sec title="Shared temporal framework & EPITN">
        <p>One engine computes baseline, current state, net and relative change, rate and acceleration (with standard errors), seasonal amplitude and anomaly
          (only for records ≥ 1.5 years), variability, onset, duration, persistence, recurrence, recovery rate, Mann-Kendall significance, confidence and a
          behaviour class — identically for displacement, velocity and backscatter series.</p>
        <p>EPITN puts BDM / GDM / CDM events on a 160 m grid and measures overlap (Jaccard, conditional probability, lift) and timing (onset lags, sign test),
          with significance from <b>toroidal-shift permutations</b> that preserve each layer&apos;s spatial clustering. Association is never presented as causation.</p>
      </Sec>

      <Sec title="Honest limits">
        <ul className="list-disc pl-5">
          <li>One year of provisional data: seasonal effects cannot be fully separated from trends; NISAR calibration may still change.</li>
          <li>Line-of-sight geometry; north-south motion is assumed small in the decomposition.</li>
          <li>Radar backscatter change shows potential disturbance, not confirmed deforestation or biomass loss; moisture changes cause false alarms.</li>
          <li>Glacier surface velocity is not mass balance; terminus retreat over one year is near the pixel size.</li>
          <li>Validation sources differ in time (ITS_LIVE 2014-2022) and in what they measure (Sentinel-2 sees burn/greenness, not structure).</li>
          <li>Nothing here is an official warning; only authorised agencies issue warnings.</li>
        </ul>
      </Sec>

      <Sec title="Use of AI tools (Space Apps disclosure)">
        <p>The software was written with the assistance of an AI coding assistant (Anthropic Claude, via Claude Code), directed and reviewed by the team.
          All numbers shown come from running the open-source pipeline on real NISAR data; no AI model generates or alters any measurement.</p>
      </Sec>

      <Sec title="Phase 2 — hazard outlook (Joshimath, experimental)">
        <p><b>Inputs:</b> Open-Meteo ERA5/ERA5-Land archive and 16-day forecast (rain, snow, temperature, soil moisture), GloFAS v4 river
          discharge (main-stem cell chosen automatically), NISAR GUNW ground state, NISAR SME2 soil moisture (compared, not yet used), Copernicus DEM,
          ESA WorldCover, and the <b>NASA Global Landslide Catalog</b> (2007–2017, rain-triggered events in Uttarakhand).</p>
        <p><b>Models:</b> landslide trigger = calibrated logistic regression on per-cell rainfall/soil-moisture percentiles, trained on catalogued
          landslides (2007–2012) and tested on 2013–2017 against NASA&apos;s LHASA 7-day-rain rule; river high-flow = calibrated logistic model tested
          on 2011–2025; slope physics = infinite-slope factor of safety with Monte Carlo soil strength; between-pass engine = Kalman filter on the NISAR
          settlement series. The browser re-runs the exported models on the live forecast (verified identical to the pipeline).</p>
      </Sec>

      <Sec title="Phase 2 (original plan)">
        <p>Slope Instability Prediction (XGBoost + infinite-slope physics + TCN), SAR flood detection and forecasting, and a data-assimilation engine (Ensemble
          Kalman filter) that keeps hazard estimates updated between 12-day NISAR passes using rainfall and soil-moisture forcing (NISAR SME2 is already in our
          catalogue). See <Link href="/hazards/" className="underline">Hazards</Link>.</p>
      </Sec>
    </div>
  );
}
