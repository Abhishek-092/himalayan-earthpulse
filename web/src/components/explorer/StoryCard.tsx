"use client";

/** Benefit / Impact story card (light and dark variants) with an isometric line illustration. */

const AMBER = "#f59e0b";

/** Stacked map layers with a location pin: "what you can do with it". */
function LayersArt({ stroke, fill, shadow }: { stroke: string; fill: string; shadow: string }) {
  return (
    <svg viewBox="0 0 200 150" className="h-full w-full" aria-hidden fill="none" strokeLinejoin="round" strokeLinecap="round">
      <path d="M40 112 L100 140 L160 112 L100 84 Z" fill={shadow} opacity="0.55" />
      {/* bottom slab */}
      <path d="M40 96 L100 124 L160 96 L100 68 Z" fill={fill} stroke={stroke} strokeWidth="2" />
      <path d="M40 96 V104 L100 132 L160 104 V96" fill={fill} stroke={stroke} strokeWidth="2" />
      <path d="M100 124 V132" stroke={stroke} strokeWidth="2" />
      {/* amber middle slab */}
      <path d="M40 84 L100 112 L160 84 L100 56 Z" fill={AMBER} stroke={stroke} strokeWidth="2" />
      <path d="M40 84 V90 L100 118 L160 90 V84" fill={AMBER} stroke={stroke} strokeWidth="2" />
      {/* top slab */}
      <path d="M40 70 L100 98 L160 70 L100 42 Z" fill={fill} stroke={stroke} strokeWidth="2" />
      <path d="M40 70 V78 L100 106 L160 78 V70" fill={fill} stroke={stroke} strokeWidth="2" />
      <path d="M100 98 V106" stroke={stroke} strokeWidth="2" />
      {/* contour lines on top */}
      <path d="M70 70 Q100 56 130 70" stroke={stroke} strokeWidth="1.4" opacity="0.6" />
      <path d="M82 74 Q100 66 118 74" stroke={stroke} strokeWidth="1.4" opacity="0.6" />
      {/* dashed outline of the next layer */}
      <path d="M50 46 L100 22 L150 46 L100 70 Z" stroke={stroke} strokeWidth="1.6" strokeDasharray="5 5" />
      {/* pin */}
      <path d="M100 62 C92 50 88 45 88 39 a12 12 0 0 1 24 0 c0 6 -4 11 -12 23 Z" fill={fill} stroke={stroke} strokeWidth="2" />
      <circle cx="100" cy="39" r="4" fill={AMBER} stroke={stroke} strokeWidth="1.6" />
    </svg>
  );
}

/** Isometric terrain slab with a radar beam and sparkles: "why it matters". */
function RadarArt({ stroke, fill, shadow }: { stroke: string; fill: string; shadow: string }) {
  return (
    <svg viewBox="0 0 200 150" className="h-full w-full" aria-hidden fill="none" strokeLinejoin="round" strokeLinecap="round">
      <path d="M36 112 L100 142 L164 112 L100 82 Z" fill={shadow} opacity="0.55" />
      <path d="M36 98 L100 128 L164 98 L100 68 Z" fill={fill} stroke={stroke} strokeWidth="2" />
      <path d="M36 98 V106 L100 136 L164 106 V98" fill={fill} stroke={stroke} strokeWidth="2" />
      <path d="M100 128 V136" stroke={stroke} strokeWidth="2" />
      {/* mountains on the slab */}
      <path d="M62 100 L82 76 L94 90 L108 70 L130 98" fill={fill} stroke={stroke} strokeWidth="2" />
      <path d="M104 76 L108 70 L113 77 L108 75 Z" fill={stroke} />
      {/* amber footprint of the radar */}
      <path d="M78 104 L100 114 L124 102 L102 92 Z" fill={AMBER} stroke={stroke} strokeWidth="1.6" />
      {/* beam */}
      <path d="M150 20 L78 104 M150 20 L124 102" stroke={stroke} strokeWidth="1.4" strokeDasharray="4 4" />
      {/* satellite */}
      <path d="M142 14 L158 22 L150 28 L134 20 Z" fill={fill} stroke={stroke} strokeWidth="2" />
      <path d="M128 10 L138 15 M162 25 L172 30" stroke={stroke} strokeWidth="2" />
      <path d="M118 4 L132 11 L128 15 L114 8 Z M168 27 L182 34 L178 38 L164 31 Z" fill={AMBER} stroke={stroke} strokeWidth="1.6" />
      {/* sparkles */}
      <path d="M30 40 l6 -4 l6 4 l-6 4 Z" fill={AMBER} stroke={stroke} strokeWidth="1.4" />
      <path d="M176 70 l5 -3 l5 3 l-5 3 Z" fill={AMBER} stroke={stroke} strokeWidth="1.4" />
      <path d="M44 62 l10 -6 M48 70 l12 -2" stroke={stroke} strokeWidth="1.6" />
    </svg>
  );
}

export default function StoryCard({ variant, tag, body, cta, onCta }: {
  variant: "benefit" | "impact"; tag: string; body: string; cta?: string; onCta?: () => void;
}) {
  const dark = variant === "impact";
  const art = dark
    ? <RadarArt stroke="#ffffff" fill="#24324a" shadow="#0b1220" />
    : <LayersArt stroke="#1b2430" fill="#ffffff" shadow="#1b2430" />;
  return (
    <div className={`relative overflow-hidden rounded-2xl p-5 ${dark ? "text-white" : "text-text"}`}
      style={{ background: dark
        ? "radial-gradient(120% 90% at 90% 0%, #3a4b6b 0%, #1b2430 55%, #121a26 100%)"
        : "radial-gradient(120% 90% at 85% 0%, #ffffff 0%, #f6f1e6 45%, #efe7d6 100%)" }}>
      <div className="flex items-start justify-between gap-2">
        <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[12px] font-medium ${dark ? "bg-white/10 text-white" : "bg-white text-text shadow-sm"}`}>
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: AMBER }} />{tag}
        </span>
      </div>
      <div className="mx-auto -mt-2 h-[136px] w-[196px]">{art}</div>
      <p className={`mt-2 text-[15px] font-medium leading-relaxed ${dark ? "text-white/90" : "text-text"}`}>{body}</p>
      {cta && onCta && (
        <button onClick={onCta}
          className={`mt-4 inline-flex items-center gap-2 rounded-full px-4 py-2 text-[13px] font-semibold transition hover:gap-3 ${dark ? "bg-white text-ink" : "bg-ink text-white"}`}>
          {cta} <span aria-hidden>→</span>
        </button>
      )}
    </div>
  );
}
