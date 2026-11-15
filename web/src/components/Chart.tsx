"use client";

import { useEffect, useRef } from "react";

// Thin Plotly wrapper, loaded lazily so the first paint stays fast.
export default function Chart({ data, layout, height = 220 }: { data: unknown[]; layout?: Record<string, unknown>; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let alive = true;
    const el = ref.current;
    import("plotly.js-basic-dist-min").then(({ default: Plotly }) => {
      if (!alive || !el) return;
      Plotly.react(
        el,
        data,
        {
          height,
          margin: { l: 48, r: 12, t: 8, b: 36 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#3a4452", size: 11, family: "IBM Plex Sans, sans-serif" },
          xaxis: { gridcolor: "#eceef0", zerolinecolor: "#c9ced6" },
          yaxis: { gridcolor: "#eceef0", zerolinecolor: "#c9ced6" },
          legend: { orientation: "h", y: -0.25, font: { size: 10 } },
          showlegend: true,
          ...layout,
        },
        { displayModeBar: false, responsive: true },
      );
    });
    return () => {
      alive = false;
    };
  }, [data, layout, height]);
  return <div ref={ref} style={{ height }} className="w-full" />;
}
