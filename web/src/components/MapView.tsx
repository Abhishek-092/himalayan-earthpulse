"use client";

import maplibregl, { type Map as MLMap, type StyleSpecification } from "maplibre-gl";
import { useEffect, useRef } from "react";
import type { LngLat } from "@/lib/data";

export interface ImageOverlay {
  id: string;
  url: string;
  corners: [LngLat, LngLat, LngLat, LngLat];
  opacity: number;
  visible?: boolean;
}

export interface GeoOverlay {
  id: string;
  data: string | GeoJSON.FeatureCollection;
  kind: "line" | "fill-outline" | "arrows" | "points";
  color: string;
  width?: number;
  visible?: boolean;
}

export interface MarkerFeature {
  id: string;
  lng: number;
  lat: number;
  label: string;
  color: string;
  size: number;
}

interface Props {
  basemap?: "satellite" | "dark" | "light";
  bounds?: [number, number, number, number];
  center?: LngLat;
  zoom?: number;
  overlays?: ImageOverlay[];
  geo?: GeoOverlay[];
  markers?: MarkerFeature[];
  terrain?: boolean;
  /** animate a tilted fly-in to `bounds` on load */
  flyIn?: boolean;
  /** show the built-in zoom/compass control (default true) */
  nav?: boolean;
  /** pulsing halo on markers */
  pulse?: boolean;
  onClick?: (lng: number, lat: number, elevation: number | null) => void;
  onMarkerClick?: (id: string) => void;
  onReady?: (map: MLMap) => void;
  interactive?: boolean;
  className?: string;
  highlight?: { lng: number; lat: number } | null;
}

const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const LIGHT = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}";
const CARTO_DARK = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}";
const CARTO_LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}";
const TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";

function baseStyle(): StyleSpecification {
  return {
    version: 8,
    glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
    sources: {
      sat: { type: "raster", tiles: [ESRI], tileSize: 256, maxzoom: 17,
        attribution: "Imagery © Esri, Maxar, Earthstar Geographics" },
      dark: { type: "raster", tiles: [CARTO_DARK], tileSize: 256, maxzoom: 16, attribution: "Basemap © Esri" },
      light: { type: "raster", tiles: [LIGHT], tileSize: 256, maxzoom: 16, attribution: "Basemap © Esri" },
      labels: { type: "raster", tiles: [CARTO_LABELS], tileSize: 256, maxzoom: 16 },
      dem: { type: "raster-dem", tiles: [TERRARIUM], tileSize: 512, encoding: "terrarium", maxzoom: 13,
        attribution: "Terrain: AWS Terrain Tiles (Mapzen)" },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#e8eae4" } },
      { id: "sat", type: "raster", source: "sat", paint: { "raster-saturation": -0.2, "raster-brightness-max": 0.95, "raster-contrast": 0.05, "raster-fade-duration": 0 } },
      { id: "dark", type: "raster", source: "dark", layout: { visibility: "none" } },
      { id: "light", type: "raster", source: "light", layout: { visibility: "none" }, paint: { "raster-fade-duration": 0 } },
      { id: "hillshade", type: "hillshade", source: "dem",
        paint: { "hillshade-exaggeration": 0.35, "hillshade-shadow-color": "#000", "hillshade-highlight-color": "#fff" } },
      { id: "labels", type: "raster", source: "labels", paint: { "raster-opacity": 0.85, "raster-fade-duration": 0 } },
    ],
  };
}

export default function MapView(props: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const loaded = useRef(false);
  const propsRef = useRef(props);
  propsRef.current = props;
  const hlMarker = useRef<maplibregl.Marker | null>(null);
  const prevTerrain = useRef<boolean | null>(null);
  const domMarkers = useRef<Map<string, maplibregl.Marker>>(new Map());

  // init
  useEffect(() => {
    if (!ref.current) return;
    const map = new maplibregl.Map({
      container: ref.current,
      style: baseStyle(),
      bounds: props.bounds,
      center: props.center,
      zoom: props.zoom,
      fitBoundsOptions: { padding: 40 },
      attributionControl: { compact: true },
      interactive: props.interactive ?? true,
      maxPitch: 70,
      pitch: props.terrain ? 50 : 0,
      // fewer pixels to draw on high-DPI screens = faster first frame, smoother 3D
      pixelRatio: Math.min(window.devicePixelRatio || 1, 1.5),
      fadeDuration: 0,
    });
    if ((props.interactive ?? true) && (props.nav ?? true)) map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    mapRef.current = map;
    // debugging aid (harmless): expose instances
    ((window as unknown as { __epMaps?: MLMap[] }).__epMaps ??= []).push(map);
    map.on("load", () => {
      loaded.current = true;
      sync();
      const p0 = propsRef.current;
      if (p0.flyIn && p0.bounds) {
        const cam = map.cameraForBounds(p0.bounds, { padding: 60 });
        if (cam?.center) {
          const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
          const end = { center: cam.center, zoom: cam.zoom, pitch: p0.terrain ? 50 : 0, bearing: p0.terrain ? -15 : 0 };
          if (reduce) map.jumpTo(end);
          else {
            map.jumpTo({ center: cam.center, zoom: Math.max((cam.zoom ?? 10) - 1, 3), pitch: p0.terrain ? 25 : 0, bearing: 0 });
            map.easeTo({ ...end, duration: 1400, essential: true });
          }
        }
      }
      propsRef.current.onReady?.(map);
    });
    // keep the camera anchored on the terrain surface (re-applying terrain can leave centre elevation at 0,
    // which mis-projects HTML markers)
    const anchor = () => {
      if (!map.getTerrain() || map.isMoving()) return;
      try {
        const z = map.queryTerrainElevation(map.getCenter());
        if (z != null && Number.isFinite(z) && Math.abs(map.getCenterElevation() - z) > 20) map.setCenterElevation(z);
      } catch {}
    };
    map.on("moveend", anchor);
    map.on("terrain", () => setTimeout(anchor, 300));
    map.on("idle", anchor);
    // pulsing marker halo
    let ph = 0;
    const pulseTimer = window.setInterval(() => {
      if (!propsRef.current.pulse || !map.getLayer("markers-halo")) return;
      ph = (ph + 0.08) % 1;
      map.setPaintProperty("markers-halo", "circle-radius", ["*", ["get", "size"], 1.4 + ph * 2.2]);
      map.setPaintProperty("markers-halo", "circle-opacity", 0.35 * (1 - ph));
    }, 60);
    map.on("click", (e) => {
      const feats = map.queryRenderedFeatures(e.point, { layers: map.getLayer("markers") ? ["markers"] : [] });
      if (feats.length && propsRef.current.onMarkerClick) {
        propsRef.current.onMarkerClick(String(feats[0].properties?.id));
        return;
      }
      let elev: number | null = null;
      try {
        const z = map.getTerrain() ? map.queryTerrainElevation(e.lngLat) : null;
        elev = z != null && Number.isFinite(z) ? z / (map.getTerrain()?.exaggeration ?? 1) : null;
      } catch {}
      propsRef.current.onClick?.(e.lngLat.lng, e.lngLat.lat, elev);
    });
    map.on("mousemove", (e) => {
      if (!map.getLayer("markers")) return;
      const f = map.queryRenderedFeatures(e.point, { layers: ["markers"] });
      map.getCanvas().style.cursor = f.length ? "pointer" : propsRef.current.onClick ? "crosshair" : "";
    });
    const dm = domMarkers.current;
    return () => {
      dm.forEach((mk) => mk.remove());
      dm.clear();
      window.clearInterval(pulseTimer);
      map.remove();
      mapRef.current = null;
      loaded.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function sync() {
    const map = mapRef.current;
    if (!map || !loaded.current) return;
    const p = propsRef.current;
    const bm = p.basemap ?? "satellite";
    map.setLayoutProperty("sat", "visibility", bm === "satellite" ? "visible" : "none");
    map.setLayoutProperty("dark", "visibility", bm === "dark" ? "visible" : "none");
    map.setLayoutProperty("light", "visibility", bm === "light" ? "visible" : "none");
    map.setPaintProperty("hillshade", "hillshade-exaggeration", bm === "light" ? 0.55 : 0.35);
    // image overlays
    const wanted = new Set((p.overlays ?? []).map((o) => `img-${o.id}`));
    for (const layer of map.getStyle().layers ?? []) {
      if (layer.id.startsWith("img-") && !wanted.has(layer.id)) {
        map.removeLayer(layer.id);
        map.removeSource(layer.id);
      }
    }
    let addedImage = false;
    for (const o of p.overlays ?? []) {
      const sid = `img-${o.id}`;
      const src = map.getSource(sid) as maplibregl.ImageSource | undefined;
      if (!src) {
        addedImage = true;
        map.addSource(sid, { type: "image", url: o.url, coordinates: o.corners });
        map.addLayer({ id: sid, type: "raster", source: sid,
          paint: { "raster-opacity": o.opacity, "raster-fade-duration": 0, "raster-resampling": "nearest" } },
          map.getLayer("labels") ? "labels" : undefined);
      } else {
        if ((src as unknown as { url: string }).url !== o.url) src.updateImage({ url: o.url, coordinates: o.corners });
        map.setPaintProperty(sid, "raster-opacity", o.opacity);
      }
      map.setLayoutProperty(sid, "visibility", o.visible === false ? "none" : "visible");
    }
    // terrain: react only when the 3D/2D mode actually changes (so users can tilt freely)
    if (prevTerrain.current !== !!p.terrain) {
      const first = prevTerrain.current === null;
      prevTerrain.current = !!p.terrain;
      if (p.terrain) {
        map.setTerrain({ source: "dem", exaggeration: 1.4 });
        if (!first) map.easeTo({ pitch: 50, bearing: -15, duration: 700 });
      } else {
        map.setTerrain(null);
        if (!first || map.getPitch() > 0) map.easeTo({ pitch: 0, bearing: 0, duration: 700 });
      }
    }
    // image sources added while terrain is on are only draped after terrain is re-applied (MapLibre)
    if (addedImage && p.terrain) {
      const img = (p.overlays ?? []).map((o) => map.getSource(`img-${o.id}`) as maplibregl.ImageSource | undefined);
      const reapply = () => { if (propsRef.current.terrain) map.setTerrain({ source: "dem", exaggeration: 1.4 }); };
      map.setTerrain(null);
      reapply();
      img.forEach((src) => src && map.once("sourcedata", reapply));
    }
    // geojson overlays
    const wantedGeo = new Set((p.geo ?? []).map((g) => `geo-${g.id}`));
    for (const layer of map.getStyle().layers ?? []) {
      if (layer.id.startsWith("geo-") && !wantedGeo.has(layer.id.replace(/-fill$/, ""))) {
        map.removeLayer(layer.id);
        if (map.getSource(layer.id)) map.removeSource(layer.id);
      }
    }
    for (const g of p.geo ?? []) {
      const sid = `geo-${g.id}`;
      if (!map.getSource(sid) && g.kind === "points") {
        map.addSource(sid, { type: "geojson", data: g.data });
        map.addLayer({ id: sid, type: "circle", source: sid,
          paint: { "circle-radius": 5, "circle-color": g.color, "circle-stroke-color": "#ffffff", "circle-stroke-width": 1.5 } });
      } else if (!map.getSource(sid)) {
        map.addSource(sid, { type: "geojson", data: g.data });
        map.addLayer({ id: sid, type: "line", source: sid,
          paint: {
            "line-color": g.color, "line-width": g.width ?? 1.5,
            ...(g.kind === "fill-outline" ? { "line-dasharray": [2, 1] } : {}),
          } });
      }
      map.setLayoutProperty(sid, "visibility", g.visible === false ? "none" : "visible");
    }
    // pulsing hotspot markers: HTML elements so they sit on the 3D terrain surface and stay clickable
    if (p.pulse) {
      const want = new Map((p.markers ?? []).map((m) => [m.id, m]));
      for (const [id, mk] of domMarkers.current) if (!want.has(id)) { mk.remove(); domMarkers.current.delete(id); }
      for (const m of p.markers ?? []) {
        let mk = domMarkers.current.get(m.id);
        if (!mk) {
          const el = document.createElement("button");
          el.className = "ep-hotspot";
          el.setAttribute("aria-label", `Place to look at ${m.label}`);
          el.addEventListener("click", (ev) => { ev.stopPropagation(); propsRef.current.onMarkerClick?.(m.id); });
          mk = new maplibregl.Marker({ element: el, opacity: "1", opacityWhenCovered: "1" }).setLngLat([m.lng, m.lat]).addTo(map);
          domMarkers.current.set(m.id, mk);
        }
        const el = mk.getElement();
        el.textContent = m.label;
        el.style.setProperty("--c", m.color);
        el.dataset.active = m.size > 9 ? "1" : "0";
        mk.setLngLat([m.lng, m.lat]);
      }
    } else if (p.markers) {
      const fc: GeoJSON.FeatureCollection = {
        type: "FeatureCollection",
        features: p.markers.map((m) => ({ type: "Feature", properties: { id: m.id, label: m.label, color: m.color, size: m.size },
          geometry: { type: "Point", coordinates: [m.lng, m.lat] } })),
      };
      const src = map.getSource("markers") as maplibregl.GeoJSONSource | undefined;
      if (src) src.setData(fc);
      else {
        map.addSource("markers", { type: "geojson", data: fc });
        map.addLayer({ id: "markers-halo", type: "circle", source: "markers",
          paint: { "circle-radius": ["*", ["get", "size"], 2.2], "circle-color": ["get", "color"], "circle-opacity": 0.18 } });
        map.addLayer({ id: "markers", type: "circle", source: "markers",
          paint: { "circle-radius": ["get", "size"], "circle-color": ["get", "color"], "circle-stroke-color": "#ffffff",
            "circle-stroke-width": 2 } });
        map.addLayer({ id: "markers-label", type: "symbol", source: "markers",
          layout: { "text-field": ["get", "label"], "text-size": 12, "text-offset": [0, 1.4], "text-anchor": "top",
            "text-font": ["Noto Sans Regular"] },
          paint: { "text-color": "#1b2430", "text-halo-color": "#ffffff", "text-halo-width": 1.8 } });
      }
    }
    // click highlight
    if (p.highlight) {
      if (!hlMarker.current) {
        const el = document.createElement("div");
        el.style.cssText = "width:14px;height:14px;border:3px solid #1b2430;background:rgba(255,255,255,.6);border-radius:50%;box-shadow:0 0 0 3px #fff";
        hlMarker.current = new maplibregl.Marker({ element: el });
      }
      hlMarker.current.setLngLat([p.highlight.lng, p.highlight.lat]).addTo(map);
    } else hlMarker.current?.remove();
  }

  useEffect(() => {
    sync();
  });

  useEffect(() => {
    const map = mapRef.current;
    if (map && props.bounds && loaded.current) map.fitBounds(props.bounds, { padding: 40, duration: 900 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.bounds?.join(",")]);

  return <div ref={ref} className={props.className ?? "h-full w-full"} />;
}
