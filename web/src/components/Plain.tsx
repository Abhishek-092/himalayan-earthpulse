"use client";

import { useState } from "react";
import { GLOSSARY, TONE_STYLE, type Verdict } from "@/lib/plain";

export function TopicIcon({ kind, size = 22 }: { kind: Verdict["icon"]; size?: number }) {
  const p = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true };
  if (kind === "ground") return <svg {...p}><path d="M3 20h18" /><path d="M5 20V10l4-3 4 3v10" /><path d="M13 20v-6h6v6" /><path d="M8 13h2M8 16h2" /></svg>;
  if (kind === "glacier") return <svg {...p}><path d="M2 19l6-11 4 6 3-4 7 9z" /><path d="M8 8l1.5 3M15 10l1 2" /></svg>;
  if (kind === "forest") return <svg {...p}><path d="M7 21v-4M17 21v-3" /><path d="M7 3l4 7H9l3 5H2l3-5H3z" /><path d="M17 7l3 5h-1.5l2.5 4h-8l2.5-4H14z" /></svg>;
  if (kind === "hazard") return <svg {...p}><path d="M12 3l9.5 17h-19z" /><path d="M12 10v4M12 17h.01" /></svg>;
  return <svg {...p}><circle cx="6" cy="12" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="18" cy="18" r="3" /><path d="M8.6 10.6l6.8-3.2M8.6 13.4l6.8 3.2" /></svg>;
}

export function SureMeter({ level }: { level: "High" | "Medium" | "Low" }) {
  const n = level === "High" ? 3 : level === "Medium" ? 2 : 1;
  return (
    <span className="inline-flex items-center gap-1.5" aria-label={`Confidence: ${level}`}>
      <span className="flex gap-0.5">{[1, 2, 3].map((i) => <span key={i} className="h-2.5 w-4 rounded-sm" style={{ background: i <= n ? "#1b2430" : "#dde1e6" }} />)}</span>
      <span className="text-xs font-medium">{level}</span>
    </span>
  );
}

export function ToneBadge({ tone }: { tone: Verdict["tone"] }) {
  const t = TONE_STYLE[tone];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold" style={{ background: t.bg, color: t.fg }}>
      <span className="h-2 w-2 rounded-full" style={{ background: t.dot }} />{t.word}
    </span>
  );
}

/** Big plain-language result card. compact = home-page list version. */
export function VerdictCard({ v, compact = false }: { v: Verdict; compact?: boolean }) {
  const [open, setOpen] = useState(!compact);
  const t = TONE_STYLE[v.tone];
  return (
    <div className="rounded-2xl border p-4" style={{ borderColor: t.dot + "55", background: compact ? "#fff" : t.bg }}>
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl" style={{ background: t.dot + "22", color: t.fg }}><TopicIcon kind={v.icon} /></span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2"><span className="text-xs font-semibold uppercase tracking-wide text-muted">{v.topic}</span><ToneBadge tone={v.tone} /></div>
          <p className={`mt-1 font-semibold leading-snug text-text ${compact ? "text-[15px]" : "text-[17px]"}`}>{v.headline}</p>
          <p className="mt-1 text-sm leading-relaxed text-body">{v.everyday}</p>
        </div>
      </div>
      {open ? (
        <div className="mt-3 space-y-2.5 border-t pt-3 text-sm" style={{ borderColor: t.dot + "33" }}>
          <div><div className="text-xs font-semibold text-muted">What does this mean?</div><p className="leading-relaxed text-body">{v.meaning}</p></div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-xs font-semibold text-muted">How sure are we?</span><SureMeter level={v.sure} />
          </div>
          <p className="text-xs leading-relaxed text-muted">{v.sureWhy}</p>
          <div>
            <div className="text-xs font-semibold text-muted">Who can use this</div>
            <div className="mt-1 flex flex-wrap gap-1.5">{v.helps.map((h) => <span key={h} className="rounded-full border border-line bg-white px-2.5 py-0.5 text-xs text-body">{h}</span>)}</div>
          </div>
        </div>
      ) : (
        <button onClick={(e) => { e.preventDefault(); setOpen(true); }} className="mt-2 text-xs font-medium text-accent underline-offset-2 hover:underline">What does this mean? →</button>
      )}
    </div>
  );
}

/** Dotted-underline jargon word with a plain-language popover. */
export function Term({ k, children }: { k: keyof typeof GLOSSARY | string; children?: React.ReactNode }) {
  const [show, setShow] = useState(false);
  const def = GLOSSARY[k];
  if (!def) return <>{children ?? k}</>;
  return (
    <span className="relative inline-block">
      <button type="button" className="cursor-help border-b border-dotted border-muted text-inherit" onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}
        onFocus={() => setShow(true)} onBlur={() => setShow(false)} onClick={() => setShow((s) => !s)} aria-describedby={`term-${k}`}>
        {children ?? k}
      </button>
      {show && (
        <span id={`term-${k}`} role="tooltip" className="absolute bottom-full left-1/2 z-50 mb-1.5 w-60 -translate-x-1/2 rounded-lg bg-ink px-3 py-2 text-left text-xs font-normal leading-snug text-white shadow-lg">
          {def}
        </span>
      )}
    </span>
  );
}
