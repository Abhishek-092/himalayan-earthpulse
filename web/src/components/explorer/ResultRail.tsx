"use client";

import Link from "next/link";
import { TONE_STYLE, type Verdict } from "@/lib/plain";
import { SureMeter, TopicIcon, ToneBadge } from "../Plain";
import { type HotspotItem, hotspotTitle } from "./MapChrome";

export interface RailItem { model: string; verdict: Verdict; hotspots: HotspotItem[] }

function ResultCard({ it, active, onSelect, onEvidence, onHotspot, activeHotspot, onTour }: {
  it: RailItem; active: boolean; onSelect: () => void; onEvidence: () => void; onHotspot: (i: number) => void; activeHotspot: number | null; onTour: () => void;
}) {
  const v = it.verdict;
  const t = TONE_STYLE[v.tone];
  return (
    <div onClick={active ? undefined : onSelect} role={active ? undefined : "button"} tabIndex={active ? undefined : 0}
      onKeyDown={(e) => { if (!active && (e.key === "Enter" || e.key === " ")) onSelect(); }}
      className={`rounded-2xl border bg-white transition ${active ? "shadow-lg" : "cursor-pointer opacity-90 hover:opacity-100 hover:shadow-md"}`}
      style={{ borderColor: active ? t.dot : "#dde1e6", borderWidth: active ? 2 : 1 }}>
      <div className="flex items-start gap-3 p-4">
        <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl" style={{ background: t.bg, color: t.fg }}><TopicIcon kind={v.icon} /></span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2"><span className="text-[11px] font-bold uppercase tracking-wider text-muted">{v.topic}</span><ToneBadge tone={v.tone} /></div>
          <p className={`mt-1 font-semibold leading-snug text-text ${active ? "text-[17px]" : "text-[15px]"}`}>{v.headline}</p>
          {active && <p className="mt-1.5 text-sm leading-relaxed text-body">{v.everyday}</p>}
        </div>
      </div>
      {active && (
        <div className="space-y-3 px-4 pb-4">
          <div className="grid gap-2">
            <div className="rounded-xl bg-[#ecfdf3] p-3">
              <div className="text-[11px] font-bold uppercase tracking-wider text-[#166534]">Benefit</div>
              <p className="mt-0.5 text-sm leading-relaxed text-text">{v.benefit}</p>
            </div>
            <div className="rounded-xl bg-[#eff6ff] p-3">
              <div className="text-[11px] font-bold uppercase tracking-wider text-[#1e3a8a]">Impact</div>
              <p className="mt-0.5 text-sm leading-relaxed text-text">{v.impact}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-xs font-semibold text-muted">How sure are we?</span><SureMeter level={v.sure} />
          </div>
          <div className="flex flex-wrap gap-1.5">{v.helps.map((h) => <span key={h} className="rounded-full border border-line px-2.5 py-0.5 text-xs text-body">{h}</span>)}</div>
          {it.hotspots.length > 0 && (
            <div>
              <div className="flex items-baseline justify-between">
                <div className="text-xs font-semibold text-muted">Places to look at ({it.hotspots.length})</div>
                <button onClick={onTour} className="text-xs font-semibold text-accent hover:underline">▶ Show me</button>
              </div>
              <ol className="mt-1.5 space-y-1">
                {it.hotspots.map((h, i) => (
                  <li key={i}>
                    <button onClick={() => onHotspot(i)} className={`flex w-full gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] leading-snug transition ${activeHotspot === i ? "bg-panel-2" : "hover:bg-panel-2"}`}>
                      <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full text-[11px] font-bold text-white" style={{ background: t.dot }}>{i + 1}</span>
                      <span className="text-body">{hotspotTitle(h)}{h.elevation_m ? <span className="text-muted"> · {Math.round(h.elevation_m)} m</span> : null}</span>
                    </button>
                  </li>
                ))}
              </ol>
            </div>
          )}
          <button onClick={onEvidence} className="flex w-full items-center justify-between rounded-xl bg-ink px-4 py-3 text-sm font-semibold text-white transition hover:opacity-90">
            <span>See the evidence behind this</span><span aria-hidden>→</span>
          </button>
        </div>
      )}
    </div>
  );
}

export default function ResultRail({ items, active, onSelect, onEvidence, onHotspot, activeHotspot, onTour, regionName, story, hazardsHref }: {
  items: RailItem[]; active: string; onSelect: (m: string) => void; onEvidence: () => void; onHotspot: (i: number) => void; activeHotspot: number | null;
  onTour: () => void; regionName: string; story: string; hazardsHref: string | null;
}) {
  return (
    <div className="space-y-3">
      <div className="rounded-2xl bg-white/95 p-4 shadow-md backdrop-blur">
        <div className="text-[11px] font-bold uppercase tracking-wider text-muted">What NISAR found</div>
        <h1 className="mt-0.5 text-[22px] font-bold leading-tight">{regionName.split(",")[0]}</h1>
        <p className="mt-1 text-[13px] leading-relaxed text-body">{story}</p>
      </div>
      {items.map((it) => (
        <ResultCard key={it.model} it={it} active={it.model === active} onSelect={() => onSelect(it.model)} onEvidence={onEvidence}
          onHotspot={onHotspot} activeHotspot={it.model === active ? activeHotspot : null} onTour={onTour} />
      ))}
      {hazardsHref && (
        <Link href={hazardsHref} className="flex items-center gap-3 rounded-2xl bg-ink p-4 text-white shadow-md transition hover:opacity-95">
          <span className="grid h-10 w-10 place-items-center rounded-xl bg-white/10"><TopicIcon kind="hazard" /></span>
          <span className="flex-1"><span className="block text-[11px] font-bold uppercase tracking-wider text-white/60">Looking ahead</span><span className="block text-sm font-semibold">Today&apos;s landslide &amp; flood outlook</span></span>
          <span>→</span>
        </Link>
      )}
    </div>
  );
}
