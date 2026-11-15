/* eslint-disable @typescript-eslint/no-explicit-any */
// Plain-language layer: turns model outputs into verdicts a non-specialist can act on.
// Everything is derived from the exported numbers - no region-specific text.
import { type Manifest, fmt } from "./data";

export type Tone = "calm" | "watch" | "alert" | "info";

export interface Verdict {
  topic: string; // "Ground", "Glacier", "Forest"
  icon: "ground" | "glacier" | "forest" | "link" | "hazard";
  tone: Tone;
  headline: string; // one short sentence
  everyday: string; // comparison to everyday things
  meaning: string; // what it means / what to do
  sure: "High" | "Medium" | "Low";
  sureWhy: string;
  helps: string[]; // who benefits
  benefit: string; // what someone can now do
  impact: string; // why it matters beyond the number
}

export const TONE_STYLE: Record<Tone, { bg: string; fg: string; dot: string; word: string }> = {
  calm: { bg: "#ecfdf3", fg: "#166534", dot: "#16a34a", word: "Stable" },
  watch: { bg: "#fff7e6", fg: "#92400e", dot: "#f59e0b", word: "Worth watching" },
  alert: { bg: "#fef2f2", fg: "#991b1b", dot: "#dc2626", word: "Changing" },
  info: { bg: "#eff6ff", fg: "#1e3a8a", dot: "#3b82f6", word: "Measured" },
};

const PITCH_KM2 = 0.00714; // one football pitch (105 x 68 m)
const pitches = (km2: number) => {
  const n = km2 / PITCH_KM2;
  return n < 1 ? "less than one football pitch" : `about ${n < 20 ? Math.round(n) : Math.round(n / 10) * 10} football pitches`;
};
const cap = (t: string) => t.charAt(0).toUpperCase() + t.slice(1);
const perDay = (mPerYr: number) => (mPerYr / 365.25) * 100; // cm/day

export function groundVerdict(m: Manifest, place: string): Verdict | null {
  const s = m.summary;
  const stacks = Object.values<any>(s?.stacks ?? {}).map((v) => v.settlement_velocity_mm_yr).filter(Boolean);
  if (!stacks.length) return null;
  const sig = stacks.filter((v) => Math.abs(v.median_mm_yr) > 2 * v.sem_mm_yr);
  const bound = Math.max(...stacks.map((v) => Math.abs(v.median_mm_yr) + 2 * v.sem_mm_yr));
  const area = s.decomposition?.subsiding_area_km2 ?? 0;
  const moving = sig.length > 0;
  const worst = moving ? sig.reduce((a, b) => (Math.abs(b.median_mm_yr) > Math.abs(a.median_mm_yr) ? b : a)) : null;
  return {
    topic: "Ground", icon: "ground", tone: moving ? "alert" : area > 0.1 ? "watch" : "calm",
    headline: moving
      ? `${place} is moving about ${fmt(Math.abs(worst.median_mm_yr), 0)} mm a year.`
      : `No measurable sinking of ${place} in the past year.`,
    everyday: moving
      ? `${fmt(Math.abs(worst.median_mm_yr) / 10, 1)} cm a year - roughly ${Math.abs(worst.median_mm_yr) > 10 ? "the width of a finger" : "the thickness of a coin"} every year.`
      : `If the town were sinking faster than about ${fmt(bound / 10, 1)} cm a year, the satellite would have seen it. ${area > 0 ? `Only ${pitches(area)} of ground shows possible local sinking.` : ""}`,
    meaning: moving
      ? "Buildings and roads in the moving area may develop cracks. Local engineers should check the flagged spots."
      : "Good news for residents: the 2023 crisis has not restarted at a pace this satellite can detect. Small, slow movements (under ~1-2 cm/yr) cannot be ruled out, so monitoring continues every 12 days.",
    sure: stacks.length >= 2 ? "High" : "Medium",
    sureWhy: `Two independent satellite viewing directions agree within the uncertainty (±${fmt(stacks[0].sem_mm_yr, 0)} mm/yr). Forest areas give false motion and are excluded.`,
    helps: ["Residents and home-owners", "District disaster officials", "Road and hydropower engineers"],
    benefit: "Shows authorities which roads, buildings and slopes are sinking or shifting, and how quickly they are moving, so they can inspect vulnerable locations first.",
    impact: "Helps identify risks such as ground movement around homes, roads and bridges in areas like Joshimath, allowing authorities to plan repairs, restrict unsafe access and protect residents before damage worsens.",
  };
}

export function glacierVerdict(m: Manifest, place: string): Verdict | null {
  const g = m.summary?.goff;
  if (!g?.speed_main_median_m_yr) return null;
  const med = g.speed_main_median_m_yr, p95 = g.speed_main_p95_m_yr;
  const r = g.validation_itslive?.pearson_r;
  return {
    topic: "Glacier", icon: "glacier", tone: "info",
    headline: `${place} creeps forward about ${fmt(med, 0)} m a year; its fastest ice about ${fmt(p95, 0)} m a year.`,
    everyday: `That is ${fmt(perDay(med), 1)} cm a day for typical ice and ${fmt(perDay(p95), 0)} cm a day at the centre - a slow river of ice ${fmt(g.main_glacier_area_km2 ?? g.glacier_area_km2, 0)} km² in size.`,
    meaning: "Glacier speed shows how fast ice is delivered to the melting snout that feeds the river. Faster summer flow and changes year to year help water planners and flood-risk teams downstream.",
    sure: "High",
    sureWhy: `Checked against NASA's ITS_LIVE glacier speeds (agreement r = ${fmt(r, 2)}) and our own second method. Stable rock around the glacier reads close to zero, as it should.`,
    helps: ["Water and hydropower planners", "Glacier-lake flood (GLOF) monitors", "Climate researchers"],
    benefit: "Water and power planners can see how quickly ice is moving toward the melting snout that feeds the Ganga - month by month, through cloud and winter snow.",
    impact: "Hundreds of millions of people downstream depend on this water. Tracking glacier flow every 12 days gives early notice of speed-ups that can come before surges or new glacier lakes.",
  };
}

export function forestVerdict(m: Manifest, place: string): Verdict | null {
  const s = m.summary;
  if (s?.potential_disturbance_km2 == null) return null;
  const lasting = s.persistent_disturbance_km2 ?? s.potential_disturbance_km2;
  const forest = s.forest_area_km2;
  const rev = s.potential_disturbance_km2 ? (s.recovering_km2 / s.potential_disturbance_km2) * 100 : 0;
  const supported = s.second_opinion ? s.second_opinion.supports_structural_change : null;
  const pct = forest ? (lasting / forest) * 100 : 0;
  const tone: Tone = supported === false ? "watch" : pct > 1 ? "alert" : "calm";
  return {
    topic: "Forest", icon: "forest", tone,
    headline: supported === false
      ? `${fmt(lasting, 1)} km² of forest in ${place} look different - but it is not confirmed as forest loss.`
      : pct < 0.1 ? `Almost no lasting forest loss in ${place}.` : `${fmt(lasting, 1)} km² of forest in ${place} changed and has not recovered.`,
    everyday: `${cap(pitches(lasting))} out of ${fmt(forest, 0)} km² of forest (${pct < 0.1 ? "under 0.1" : fmt(pct, 1)}%). ${rev > 20 ? `${fmt(rev, 0)}% of the changes the radar first flagged bounced back - frost, snow or wet soil, not lost trees.` : ""}`,
    meaning: supported === false
      ? "Treat these patches as a list of places to check on the ground or with a clear-sky photo. Two other radar measurements did not show the structural change that real tree loss usually causes."
      : "Forest officers can use the map to check the flagged patches; the radar sees them even under monsoon cloud, when ordinary satellite photos cannot.",
    sure: supported ? "High" : "Medium",
    sureWhy: supported === false
      ? "The main radar test flagged a change, but coherence and the HV/HH ratio (two independent radar measures) did not confirm it."
      : "The main radar test, plus coherence and the HV/HH ratio, point the same way. Optical satellite comparison is limited by cloud.",
    helps: ["Forest department", "Landslide-risk planners (lost trees = weaker slopes)", "Conservation groups"],
    benefit: supported === false
      ? "Forest teams get a short list of patches worth a visit, and a clear warning that most first alarms were only frost or wet soil - saving wasted trips."
      : "Forest teams can spot real changes even during the monsoon, when ordinary satellite photos are blocked by cloud for months.",
    impact: "Lost tree cover on steep slopes makes landslides more likely. Seeing it early, and telling real loss from seasonal effects, helps protect both forests and the villages below them.",
  };
}

export function linkVerdict(m: Manifest, place: string): Verdict | null {
  const n = m.network;
  const p = n?.pairs?.[0];
  if (!p) return null;
  const a = String(p.a ?? "").split(" (")[0].toLowerCase(), b = String(p.b ?? "").split(" (")[0].toLowerCase();
  return {
    topic: "Changes together?", icon: "link", tone: p.significant ? "watch" : "info",
    headline: p.significant ? `In ${place}, ${a} and ${b} happen in the same places more often than chance.` : `In ${place}, ${a} and ${b} are not linked so far.`,
    everyday: p.significant ? `They overlap ${fmt(p.lift, 1)} times more than if they were scattered at random.` : `They overlap about as often as random scattering would give (${p.cells_both ?? 0} shared cells, chance ${fmt((p.p_value ?? 1) * 100, 0)}%).`,
    meaning: p.significant ? "Places where several things change at once deserve a closer look - for example lost forest on moving slopes." : "No sign yet that forest change and ground movement feed each other here. One year is short; this is re-tested as data arrives.",
    sure: "Medium", sureWhy: "Based on one year of data and a random-placement test; overlap is a clue, not proof of cause.",
    helps: ["Disaster planners", "Researchers"],
    benefit: "Shows planners where several hazards stack up, so limited inspection teams go where risks combine.",
    impact: "Himalayan disasters often come from chains - lost forest, moving slopes, then a landslide. Checking for links is the first step to catching those chains early.",
  };
}

export function verdictsFor(manifests: Partial<Record<string, Manifest>>, regionName: string): Verdict[] {
  const place = regionName.split(",")[0];
  const out: Verdict[] = [];
  if (manifests.gdm) { const v = groundVerdict(manifests.gdm, place); if (v) out.push(v); }
  if (manifests.cdm) { const v = glacierVerdict(manifests.cdm, place); if (v) out.push(v); }
  if (manifests.bdm) { const v = forestVerdict(manifests.bdm, place); if (v) out.push(v); }
  return out;
}

export function verdictFor(model: string, m: Manifest, regionName: string): Verdict | null {
  const place = regionName.split(",")[0];
  return model === "gdm" ? groundVerdict(m, place) : model === "cdm" ? glacierVerdict(m, place) : model === "bdm" ? forestVerdict(m, place) : model === "epitn" ? linkVerdict(m, place) : null;
}

export const LEVEL_PLAIN = [
  { word: "Normal", say: "Nothing unusual is expected.", todo: "Go about your day as usual. Keep an eye on official forecasts." },
  { word: "Be aware", say: "Rain or river flow is higher than usual.", todo: "Avoid lingering below steep slopes and near the riverbank. Check official IMD / USDMA updates." },
  { word: "Be prepared", say: "Conditions that often come before landslides or high river water.", todo: "Avoid unnecessary travel on hill roads. Know your nearest safe place. Follow local authority advice." },
  { word: "Take action", say: "Conditions similar to past landslide and flood days.", todo: "Stay away from slopes, streams and the riverbank. Follow official instructions immediately." },
];

export const GLOSSARY: Record<string, string> = {
  "NISAR": "A radar satellite built by NASA (USA) and ISRO (India), launched 2025. It sees through cloud and at night.",
  "radar": "The satellite sends microwave pulses and records the echo. Unlike a camera, it does not need sunlight or clear skies.",
  "L-band": "Long radar waves (~24 cm) that pass through leaves and reach branches, trunks and the ground.",
  "mm/yr": "Millimetres per year. 10 mm = 1 cm, about the width of a fingernail.",
  "coherence": "How similar the radar echo stayed between two passes. Stable ground keeps it; moving ice, growing plants or snow lose it.",
  "pass": "One satellite overflight. NISAR revisits each place about every 12 days.",
  "uncertainty": "The range within which the true value most likely lies. '2 ± 8' means somewhere between about -6 and +10.",
};

// ------------------------------------------------------------------ timeline helpers (results-first)
export interface TimelineEvent { date: string; label: string; kind: "peak" | "season" | "latest" | "next" }

export function timelineEvents(model: string, m: Manifest | undefined, dates: string[]): TimelineEvent[] {
  if (!dates.length) return [];
  const ev: TimelineEvent[] = [];
  const s = m?.summary ?? {};
  if (model === "bdm" && s.disturbance_by_date?.length) {
    const big = s.disturbance_by_date.reduce((a: any, b: any) => (b.new_disturbance_km2 > a.new_disturbance_km2 ? b : a));
    ev.push({ date: big.date, label: "Biggest change (mostly winter, reversible)", kind: "peak" });
  }
  if (model === "cdm") {
    const mo = Object.entries<any>(s.goff?.monthly_main_median_m_yr ?? {}).filter(([, v]) => v != null);
    if (mo.length > 2) {
      const hi = mo.reduce((a, b) => (b[1] > a[1] ? b : a));
      ev.push({ date: `${hi[0]}-15`, label: "Fastest month (indicative)", kind: "peak" });
    }
  }
  const d0 = new Date(dates[0]).getFullYear();
  for (const y of [d0, d0 + 1]) {
    const a = `${y}-06-15`;
    if (a > dates[0] && a < dates[dates.length - 1]) ev.push({ date: a, label: "Monsoon begins", kind: "season" });
    const w = `${y}-01-01`;
    if (w > dates[0] && w < dates[dates.length - 1]) ev.push({ date: w, label: "Winter: frost & snow", kind: "season" });
  }
  ev.push({ date: dates[dates.length - 1], label: "Latest pass", kind: "latest" });
  const nx = new Date(dates[dates.length - 1]);
  nx.setDate(nx.getDate() + 12);
  ev.push({ date: nx.toISOString().slice(0, 10), label: "Next expected pass", kind: "next" });
  return ev;
}

const longD = (d: string) => new Date(d).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });

export function captionFor(model: string, m: Manifest | undefined, date: string | undefined, series?: { dates: string[]; values: number[]; units: string }): string {
  if (!m || !date) return "";
  const s = m.summary ?? {};
  if (model === "bdm" && s.disturbance_by_date?.length) {
    const upto = s.disturbance_by_date.filter((d: any) => d.date <= date);
    const cum = upto.length ? upto[upto.length - 1].cumulative_km2 : 0;
    const lasting = s.persistent_disturbance_km2;
    return `By ${longD(date)}: ${fmt(cum, 1)} km² of forest had looked different at least once.${cum > lasting * 1.5 ? ` Only ${fmt(lasting, 1)} km² stayed changed by the end - the rest bounced back.` : ""}`;
  }
  if (model === "cdm") {
    const mo = (s.goff?.monthly_main_median_m_yr ?? {})[date.slice(0, 7)];
    if (mo != null) return `In ${new Date(date).toLocaleDateString("en-GB", { month: "long", year: "numeric" })}: typical ice moving about ${fmt(mo, 0)} m a year (${fmt((mo / 365.25) * 100, 1)} cm a day).`;
    return `Glacier speed from ${s.goff?.n_pairs ?? "many"} radar pairs, ${s.goff?.date_range?.join(" to ") ?? ""}.`;
  }
  if (series) {
    let i = series.dates.findIndex((d) => d >= date);
    if (i < 0) i = series.dates.length - 1;
    const v = series.values[i];
    if (Number.isFinite(v)) {
      if (model === "gdm") return `By ${longD(series.dates[i])}: the town's ground has shifted ${fmt(Math.abs(v), 0)} mm ${v < 0 ? "away from" : "toward"} the satellite since the first pass - ${Math.abs(v) < 15 ? "within normal seasonal wobble" : "worth watching"}.`;
      return `${longD(series.dates[i])}: ${fmt(v, 1)} ${series.units}`;
    }
  }
  return `Pass of ${longD(date)}.`;
}

// ------------------------------------------------------------------ plain-language map keys
export interface FriendlyLegend { title: string; left?: string; mid?: string; right?: string; labels?: Record<string, string> }

const cm = (mm: number) => `${fmt(Math.abs(mm) / 10, Math.abs(mm) < 10 ? 1 : 0)} cm`;
const monthYear = (d?: string) => (d ? new Date(d).toLocaleDateString("en-GB", { month: "short", year: "numeric" }) : "");

export function friendlyLegend(l: { id: string; title: string; units: string; legend: { min?: number; max?: number }; dates?: string[]; kind?: string }): FriendlyLegend {
  const lo = l.legend.min ?? 0, hi = l.legend.max ?? 0;
  if (l.id === "vertical_velocity")
    return { title: "Is the ground sinking or rising?", left: `Sinking ${cm(lo)}/yr`, mid: "No change", right: `Rising ${cm(hi)}/yr` };
  if (l.id.endsWith("_velocity") && l.units.includes("mm"))
    return { title: "Is the ground moving? (seen from the satellite)", left: `Away ${cm(lo)}/yr`, mid: "No change", right: `Toward ${cm(hi)}/yr` };
  if (l.id.endsWith("_displacement"))
    return { title: "How far has the ground moved since the first pass?", left: `${cm(lo)} away`, mid: "Not moved", right: `${cm(hi)} toward` };
  if (l.id === "speed_goff" || l.id === "speed_monthly")
    return { title: l.id === "speed_monthly" ? "How fast was the ice flowing this month?" : "How fast is the ice flowing?", left: "Barely moving",
      mid: `${fmt(hi / 2, 0)} m a year`, right: `${fmt(hi, 0)}+ m a year (${fmt((hi / 365.25) * 100, 0)} cm a day)` };
  if (l.id === "changed_so_far") return { title: "Forest that had changed by this date", left: "", right: "Changed at least once" };
  if (l.id === "hv") return { title: "How much forest the radar sees", left: "Little (bare or cleared)", right: "Dense forest" };
  if (l.id === "disturbance_date") return { title: "When did the forest first change?", left: `Early (${monthYear(l.dates?.[1])})`, right: `Late (${monthYear(l.dates?.[l.dates.length - 1])})` };
  if (l.id === "change_type")
    return { title: "What kind of forest change?", labels: { "1": "Changed, then bounced back (frost, snow, wet soil)", "2": "First seen on a snowy day", "3": "Lasting change - worth checking" } };
  if (l.id === "overlap") return { title: "Where do different changes overlap?" };
  if (l.id === "pfs" || l.id === "combined") return { title: "How likely is this slope to fail?", left: "Unlikely", right: "More likely" };
  return { title: l.title, left: `${fmt(lo, 0)} ${l.units}`, right: `${fmt(hi, 0)} ${l.units}` };
}

/** Plain caption for a chosen period (FROM -> TO on the timeline). */
export function captionForRange(model: string, m: Manifest | undefined, d0: string, d1: string, series?: { dates: string[]; values: number[]; units: string }): string {
  if (!m) return "";
  const s = m.summary ?? {};
  const span = `Between ${longD(d0)} and ${longD(d1)}`;
  if (model === "bdm" && s.disturbance_by_date?.length) {
    const at = (d: string) => { const u = s.disturbance_by_date.filter((x: any) => x.date <= d); return u.length ? u[u.length - 1].cumulative_km2 : 0; };
    return `${span}: ${fmt(at(d1) - at(d0), 1)} km² more forest started to look different to the radar.`;
  }
  if (model === "cdm") {
    const mo = s.goff?.monthly_main_median_m_yr ?? {};
    const a = mo[d0.slice(0, 7)], b = mo[d1.slice(0, 7)];
    if (a != null && b != null) return `${span}: typical ice went from about ${fmt(a, 0)} to ${fmt(b, 0)} m a year (${b > a ? "faster" : b < a ? "slower" : "unchanged"}).`;
    return `${span}: glacier speed from NISAR radar pairs.`;
  }
  if (series) {
    const at = (d: string) => { let i = series.dates.findIndex((x) => x >= d); if (i < 0) i = series.dates.length - 1; return series.values[i]; };
    const dv = at(d1) - at(d0);
    if (Number.isFinite(dv)) {
      if (model === "gdm") return `${span}: the town's ground shifted ${fmt(Math.abs(dv), 0)} mm ${dv < 0 ? "away from" : "toward"} the satellite - ${Math.abs(dv) < 15 ? "within normal seasonal wobble" : "worth watching"}.`;
      return `${span}: changed by ${fmt(dv, 1)} ${series.units}.`;
    }
  }
  return `${span}.`;
}
