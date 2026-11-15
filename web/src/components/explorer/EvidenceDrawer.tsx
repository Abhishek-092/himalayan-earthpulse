"use client";

import { useEffect, useState } from "react";

export interface EvidenceTab { id: string; label: string; content: React.ReactNode }

/** Slide-in panel that holds every number, chart, layer and check behind a result. Only opened on request. */
export default function EvidenceDrawer({ open, onClose, title, subtitle, tabs, tab, setTab }: {
  open: boolean; onClose: () => void; title: string; subtitle?: string; tabs: EvidenceTab[]; tab: string; setTab: (t: string) => void;
}) {
  useEffect(() => { if (open && !tabs.some((t) => t.id === tab)) setTab(tabs[0]?.id ?? ""); }, [open, tabs, tab, setTab]);
  // build the (heavy) evidence content only once it has been asked for
  const [opened, setOpened] = useState(false);
  if (open && !opened) setOpened(true);
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);

  return (
    <aside aria-hidden={!open} aria-label="Evidence"
      className={`fixed top-16 right-0 bottom-0 z-50 flex w-full max-w-[600px] flex-col border-l border-line bg-white shadow-2xl transition-transform duration-300 ${open ? "translate-x-0" : "pointer-events-none translate-x-full"}`}>
      <div className="flex items-start gap-3 border-b border-line px-5 py-4">
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-bold uppercase tracking-wider text-muted">The evidence behind</div>
          <div className="text-lg font-bold leading-tight">{title}</div>
          {subtitle && <div className="mt-0.5 text-xs text-muted">{subtitle}</div>}
        </div>
        <button onClick={onClose} aria-label="Close evidence" className="grid h-9 w-9 place-items-center rounded-lg border border-line text-lg hover:bg-panel-2">×</button>
      </div>
      <div className="flex gap-1 overflow-x-auto border-b border-line px-3 py-2">
        {tabs.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} className={`shrink-0 rounded-lg px-3 py-1.5 text-sm ${tab === t.id ? "bg-ink font-semibold text-white" : "text-body hover:bg-panel-2"}`}>{t.label}</button>
        ))}
      </div>
      <div className="scrollbar-thin flex-1 overflow-y-auto px-5 py-4">
        {opened && tabs.find((t) => t.id === tab)?.content}
      </div>
    </aside>
  );
}
