"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Detection } from "@/lib/api";
import { useViewerStore } from "@/stores/viewer";

interface Props {
  detections: Detection[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  viewer: any | null;
}

/**
 * Pure HTML/SVG overlay — does NOT add anything to the splat viewer's
 * THREE.Scene (mkkellogg does not reliably expose one for us to add to).
 * Instead: every animation frame we read the splat viewer's camera,
 * project the active/hovered detection's 8 bbox corners to NDC, then
 * to pixel coordinates, and draw:
 *   - an SVG box outline + corner ticks
 *   - a positioned <div> label "class · NN%"
 * over the WebGL canvas.
 */
export default function ObjectOverlay({ detections, viewer }: Props) {
  const activeId = useViewerStore((s) => s.activeObjectId);
  const hoverId = useViewerStore((s) => s.hoveredObjectId);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const polyRef = useRef<SVGPolylineElement | null>(null);
  const labelRef = useRef<HTMLDivElement | null>(null);

  // current selection lives in a ref so the rAF loop can read the latest
  // value without re-creating the loop on every state change.
  const selRef = useRef<{ active: string | null; hover: string | null }>({
    active: null,
    hover: null,
  });
  selRef.current = { active: activeId, hover: hoverId };

  const detRef = useRef<Detection[]>(detections);
  detRef.current = detections;

  const [diag, setDiag] = useState<string>("waiting for viewer…");

  useEffect(() => {
    if (!viewer) return;
    let cancelled = false;
    let frame: number | null = null;
    const tmp = new THREE.Vector3();

    function getCamera(): THREE.PerspectiveCamera | null {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const v = viewer as any;
      const c = v?.camera || v?.threeCamera;
      return c && c.isCamera ? (c as THREE.PerspectiveCamera) : null;
    }

    function getCanvas(): HTMLCanvasElement | null {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const v = viewer as any;
      const r = v?.renderer || v?.threeRenderer;
      return r?.domElement ?? null;
    }

    function corners(min: number[], max: number[]): THREE.Vector3[] {
      const [x0, y0, z0] = min;
      const [x1, y1, z1] = max;
      return [
        new THREE.Vector3(x0, y0, z0),
        new THREE.Vector3(x1, y0, z0),
        new THREE.Vector3(x1, y1, z0),
        new THREE.Vector3(x0, y1, z0),
        new THREE.Vector3(x0, y0, z1),
        new THREE.Vector3(x1, y0, z1),
        new THREE.Vector3(x1, y1, z1),
        new THREE.Vector3(x0, y1, z1),
      ];
    }

    function project(point: THREE.Vector3, cam: THREE.PerspectiveCamera, w: number, h: number) {
      tmp.copy(point).project(cam);
      // tmp.z < -1 or > 1 means the point is outside the depth range
      const x = (tmp.x * 0.5 + 0.5) * w;
      const y = (-tmp.y * 0.5 + 0.5) * h;
      return { x, y, z: tmp.z };
    }

    function tick() {
      if (cancelled) return;
      frame = requestAnimationFrame(tick);

      const poly = polyRef.current;
      const label = labelRef.current;
      if (!poly || !label) return;

      const id = selRef.current.hover ?? selRef.current.active;
      const d = id ? detRef.current.find((x) => x.id === id) : undefined;
      if (!d || !d.bbox_min || !d.bbox_max) {
        poly.setAttribute("points", "");
        label.style.display = "none";
        return;
      }
      const cam = getCamera();
      const canvas = getCanvas();
      if (!cam || !canvas) {
        if (diag.startsWith("waiting")) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const v = viewer as any;
          setDiag(`viewer present, camera=${!!cam}, canvas=${!!canvas}, keys=${Object.keys(v).slice(0, 14).join(",")}`);
        }
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const containerRect = containerRef.current?.getBoundingClientRect();
      const offX = containerRect ? rect.left - containerRect.left : 0;
      const offY = containerRect ? rect.top - containerRect.top : 0;
      const w = rect.width;
      const h = rect.height;

      const pts = corners(d.bbox_min, d.bbox_max).map((p) => project(p, cam, w, h));
      // bail if any corner is behind the camera AND others are too — full clip
      const allBehind = pts.every((p) => p.z > 1);
      if (allBehind) {
        poly.setAttribute("points", "");
        label.style.display = "none";
        return;
      }

      // 12 edges of the box drawn as a single polyline (with revisits)
      // edges:
      //   bottom: 0-1, 1-2, 2-3, 3-0
      //   top:    4-5, 5-6, 6-7, 7-4
      //   verts:  0-4, 1-5, 2-6, 3-7
      const seq = [0, 1, 2, 3, 0, 4, 5, 1, 5, 6, 2, 6, 7, 3, 7, 4];
      const pointsStr = seq
        .map((i) => `${(pts[i].x + offX).toFixed(1)},${(pts[i].y + offY).toFixed(1)}`)
        .join(" ");
      poly.setAttribute("points", pointsStr);

      // label = top-front-center of the bbox
      const top = pts.slice(4).reduce(
        (acc, p) => ({ x: acc.x + p.x / 4, y: acc.y + p.y / 4, z: acc.z + p.z / 4 }),
        { x: 0, y: 0, z: 0 }
      );
      label.style.display = "block";
      label.style.left = `${top.x + offX}px`;
      label.style.top = `${top.y + offY - 30}px`;
      label.textContent = `${d.class_name} · ${(d.confidence * 100).toFixed(0)}%`;
    }

    frame = requestAnimationFrame(tick);
    setDiag("overlay running");
    return () => {
      cancelled = true;
      if (frame) cancelAnimationFrame(frame);
    };
  }, [viewer, diag]);

  return (
    <div
      ref={containerRef}
      className="pointer-events-none absolute inset-0 z-10"
      style={{ overflow: "hidden" }}
    >
      <svg
        className="absolute inset-0 w-full h-full"
        style={{ pointerEvents: "none" }}
      >
        <polyline
          ref={polyRef}
          fill="none"
          stroke="#f5b97a"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
          style={{ filter: "drop-shadow(0 0 4px rgba(245,185,122,0.6))" }}
        />
      </svg>
      <div
        ref={labelRef}
        className="absolute"
        style={{
          display: "none",
          transform: "translate(-50%, -100%)",
          background: "rgba(15,15,15,0.92)",
          color: "#f5b97a",
          border: "1px solid rgba(245,185,122,0.7)",
          borderRadius: 6,
          padding: "4px 10px",
          fontSize: 12,
          fontWeight: 600,
          whiteSpace: "nowrap",
          pointerEvents: "none",
        }}
      />
      {/* tiny debug chip in bottom-left, only visible until we attach */}
      {diag !== "overlay running" && (
        <div
          className="absolute bottom-2 left-2 text-[10px] text-amber-300/80"
          style={{ pointerEvents: "none" }}
        >
          overlay: {diag}
        </div>
      )}
    </div>
  );
}
