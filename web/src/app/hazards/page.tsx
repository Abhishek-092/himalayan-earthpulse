import Link from "next/link";

export const metadata = { title: "Hazards (Phase 2) · Himalayan EarthPulse" };

const parts = [
  ["Slope instability (SIPM)", "Infinite-slope factor of safety with Monte Carlo soil strength and live soil wetness gives a daily map of slope-failure chance; a NASA LHASA-style 7-day rainfall percentile trigger sets the landslide advisory. NISAR moving-ground areas are overlaid."],
  ["River high-flow & flood", "Calibrated logistic model (rain, antecedent index, soil moisture, snowmelt, temperature) for the chance the Alaknanda reaches its 2-year flood level within 72 h — trained 1991–2010, tested 2011–2025 (AUC 0.90) — plus the GloFAS 30-day forecast against 2/5/20-year return levels. SAR flood mapping is not used: the valley is too narrow for 20 m radar."],
  ["Between-pass engine (DEHFE)", "Linear Kalman filter: each NISAR pass updates the ground state (displacement, velocity); daily rainfall forcing evolves it during the 12-day gap; validated by predicting each next pass and comparing with persistence."],
  ["Advisories", "Green / yellow / orange / red from the slope trigger, river probability and forecast discharge — with reasons, validity window and data freshness. Model-generated advisories, never official warnings."],
];

export default function Hazards() {
  return (
    <div className="mx-auto max-w-4xl px-5 py-8">
      <Link href="/hazards/joshimath/" className="mb-6 block rounded-2xl bg-ink p-5 text-white">
        <div className="kicker text-white/60">Live demonstrator · experimental</div>
        <div className="mt-1 text-2xl font-bold">Joshimath hazard outlook →</div>
        <div className="mt-1 text-sm text-white/80">Live rain, soil and river forecasts + NISAR ground state → slope and flood advisories for the next 15 days.</div>
      </Link>
      <div className="kicker">Phase 2 · components</div>
      <h1 className="mt-1 text-3xl font-semibold">From watching change to anticipating hazards</h1>
      <p className="mt-3 text-muted">
        Phase 1 measures what has changed. Phase 2 estimates how slope and flood hazards may evolve <i>between</i> NISAR passes, for one place done
        thoroughly (Joshimath), with every model validated on data it never saw.
      </p>
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {parts.map(([t, d]) => (
          <div key={t} className="panel p-4">
            <h2 className="font-semibold">{t}</h2>
            <p className="mt-1 text-sm leading-relaxed text-muted">{d}</p>
          </div>
        ))}
      </div>
      <p className="mt-6 text-xs text-muted">
        Already in place: versioned data contract with a reserved hazards slot, NISAR SME2 soil-moisture granules discovered per region, per-pixel GDM state
        (displacement, velocity, uncertainty) exported as the assimilation state.
      </p>
    </div>
  );
}
