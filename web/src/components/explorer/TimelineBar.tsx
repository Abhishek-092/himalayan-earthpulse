"use client";

import { useRef, useState } from "react";
import type { TimelineEvent } from "@/lib/plain";

const EV_COLOR: Record<TimelineEvent["kind"], string> = { peak: "#f59e0b", season: "#3b82f6", latest: "#1b2430", next: "#9ca3af" };
const longD = (ms: number) => new Date(ms).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });

/** Range timeline: a FROM handle and a TO handle, each draggable to any date (snaps to the nearest satellite pass). */
export default function TimelineBar({ dates, t, from, playhead = null, onPick, onPickFrom, onStep, playing, onPlay, events, caption, span }: {
  dates: string[]; t: number; from: number; playhead?: number | null; onPick: (i: number) => void; onPickFrom: (i: number) => void;
  onStep?: (d: number) => void; playing: boolean; onPlay: () => void; events: TimelineEvent[]; caption: string; span?: [string, string];
}) {
  const at = playhead ?? t; // pass currently shown
  const nT = dates.length;
  const first = nT ? dates[0] : span?.[0] ?? events[0]?.date ?? "2025-11-01";
  const last = nT ? dates[nT - 1] : span?.[1] ?? first;
  const t0 = +new Date(first);
  const lastEv = events.reduce((a, e) => Math.max(a, +new Date(e.date)), 0);
  const t1 = Math.max(+new Date(last), lastEv);
  const frac = (ms: number) => (ms - t0) / Math.max(t1 - t0, 1);
  const pos = (d: string) => `${(frac(+new Date(d)) * 100).toFixed(2)}%`;
  const months: string[] = [];
  const m = new Date(t0); m.setDate(1); m.setMonth(m.getMonth() + 1);
  while (+m <= t1) { months.push(m.toISOString().slice(0, 10)); m.setMonth(m.getMonth() + 1); }
  const passMs = dates.map((d) => +new Date(d));
  const nearest = (ms: number) => passMs.reduce((b, p, i) => (Math.abs(p - ms) < Math.abs(passMs[b] - ms) ? i : b), 0);

  const track = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<null | { which: "from" | "to"; ms: number }>(null);
  const msAt = (clientX: number) => {
    const r = track.current!.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    return Math.min(passMs[nT - 1], Math.max(passMs[0], t0 + f * (t1 - t0)));
  };
  const apply = (which: "from" | "to", ms: number) => {
    const i = nearest(ms);
    if (which === "from") onPickFrom(Math.min(i, t));
    else onPick(Math.max(i, from));
  };
  const onDown = (e: React.PointerEvent) => {
    if (!nT) return;
    const ms = msAt(e.clientX);
    // grab the closer handle (ties go to TO when both sit together at the end)
    const which = Math.abs(ms - passMs[from]) < Math.abs(ms - passMs[t]) ? "from" : "to";
    (e.target as Element).setPointerCapture?.(e.pointerId);
    setDrag({ which, ms });
    apply(which, ms);
  };
  const onMove = (e: React.PointerEvent) => {
    if (!drag) return;
    const ms = msAt(e.clientX);
    setDrag({ ...drag, ms });
    apply(drag.which, ms);
  };
  const onUp = () => setDrag(null);
  const key = (which: "from" | "to") => (e: React.KeyboardEvent) => {
    const step = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    if (which === "from") onPickFrom(Math.min(t, Math.max(0, from + step)));
    else onPick(Math.max(from, Math.min(nT - 1, t + step)));
  };

  const handle = (which: "from" | "to") => {
    const idx = which === "from" ? from : t;
    const ms = drag?.which === which ? drag.ms : passMs[idx];
    const label = drag?.which === which ? longD(drag.ms) : longD(passMs[idx]);
    const f = frac(ms);
    return (
      <div key={which} className="pointer-events-none absolute top-[22px] -translate-x-1/2" style={{ left: `${(f * 100).toFixed(2)}%` }}>
        <div role="slider" tabIndex={0} aria-label={which === "from" ? "From date" : "To date"} aria-valuemin={0} aria-valuemax={nT - 1} aria-valuenow={idx}
          aria-valuetext={longD(passMs[idx])} onKeyDown={key(which)}
          className={`h-5 w-5 cursor-grab rounded-full border-[3px] shadow-md outline-none focus-visible:ring-2 focus-visible:ring-amber ${which === "from" ? "border-amber bg-white" : "border-ink bg-white"} ${drag?.which === which ? "scale-125 cursor-grabbing" : ""}`} />
        <div className={`mono pointer-events-none absolute top-6 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-semibold ${which === "from" ? "bg-amber text-ink" : "bg-ink text-white"} ${f > 0.88 ? "right-0" : f < 0.12 ? "left-0" : "left-1/2 -translate-x-1/2"}`}>
          {which === "from" ? "FROM " : "TO "}{label}
        </div>
      </div>
    );
  };

  const shown = events.filter((e) => e.kind !== "latest").sort((a, b) => a.date.localeCompare(b.date));

  return (
    <div className="rounded-2xl bg-white/95 px-3 py-3 shadow-lg backdrop-blur sm:px-4">
      <div className="flex items-center gap-3">
        <div className="flex shrink-0 gap-1.5">
          <button aria-label="previous pass" onClick={() => (onStep ? onStep(-1) : onPick(Math.max(from, t - 1)))} disabled={!nT} className="grid h-9 w-9 place-items-center rounded-lg border border-line bg-white disabled:opacity-40">‹</button>
          <button aria-label={playing ? "pause" : "play"} onClick={onPlay} disabled={!nT} className="grid h-9 w-9 place-items-center rounded-lg bg-ink text-white disabled:opacity-40">{playing ? "❚❚" : "▶"}</button>
          <button aria-label="next pass" onClick={() => (onStep ? onStep(1) : onPick(Math.min(nT - 1, t + 1)))} disabled={!nT} className="grid h-9 w-9 place-items-center rounded-lg border border-line bg-white disabled:opacity-40">›</button>
        </div>
        <div className="min-w-0 flex-1 overflow-x-auto overflow-y-visible">
          <div className="relative mx-3 mt-1 h-[82px] min-w-[460px]">
            {/* events */}
            {shown.map((e, i) => {
              const f = frac(+new Date(e.date));
              const anchor = f > 0.85 ? "-translate-x-full" : f < 0.1 ? "" : "-translate-x-1/2";
              const row = i % 2;
              return (
                <div key={e.kind + e.date} className="pointer-events-none absolute" style={{ left: pos(e.date) }}>
                  <div className={`absolute whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-semibold text-white ${anchor}`} style={{ top: row ? -2 : 8, background: EV_COLOR[e.kind], opacity: e.kind === "next" ? 0.75 : 1 }}>{e.label}</div>
                  <div className="absolute w-px" style={{ top: row ? 14 : 22, height: row ? 14 : 6, background: EV_COLOR[e.kind] }} />
                </div>
              );
            })}
            {/* track (click or drag anywhere) */}
            <div ref={track} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
              className="absolute top-[22px] right-0 left-0 h-5 cursor-pointer touch-none">
              <div className="absolute top-2 right-0 left-0 h-1 rounded bg-line" />
              {nT > 0 && <div className="absolute top-2 h-1 rounded bg-ink/30" style={{ left: pos(dates[from]), width: `calc(${pos(dates[t])} - ${pos(dates[from])})` }} />}
              {nT > 0 && <div className="absolute top-2 h-1 rounded bg-ink" style={{ left: pos(dates[from]), width: `calc(${pos(dates[at])} - ${pos(dates[from])})` }} />}
              {dates.map((d, i) => (
                <span key={d + i} className={`absolute top-[5px] h-2.5 w-2.5 -translate-x-1/2 rounded-full border-2 ${i >= from && i <= at ? "border-ink bg-ink" : i > at && i <= t ? "border-ink bg-white" : "border-[#9ca3af] bg-white"}`} style={{ left: pos(d) }} />
              ))}
            </div>
            {nT > 0 && playhead != null && playhead < t && (
              <div className="pointer-events-none absolute top-0 z-10 -translate-x-1/2" style={{ left: pos(dates[playhead]) }}>
                <div className="mono whitespace-nowrap rounded bg-accent px-1.5 py-0.5 text-[10px] font-semibold text-white">{playing ? "▶ " : ""}{longD(passMs[playhead])}</div>
                <div className="mx-auto mt-0.5 h-0 w-0 border-x-[5px] border-t-[6px] border-x-transparent border-t-accent" />
              </div>
            )}
            {nT > 0 && handle("from")}
            {nT > 0 && handle("to")}
            {/* months */}
            {months.map((d) => (
              <div key={d} className="mono pointer-events-none absolute top-[66px] -translate-x-1/2 text-[10px] text-muted" style={{ left: pos(d) }}>
                {new Date(d).toLocaleDateString("en-GB", { month: "short" })}{d.slice(5, 7) === "01" ? ` ${d.slice(2, 4)}` : ""}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-0.5 pl-1">
        {nT > 0 && <span className="mono text-xs font-semibold text-text">{longD(passMs[from])} → {longD(passMs[t])} · {playhead != null && playhead < t ? `pass ${playhead - from + 1} of ${t - from + 1}` : `${t - from + 1} passes`}</span>}
        <span className="text-[13px] leading-snug text-body">{caption}</span>
      </div>
    </div>
  );
}
