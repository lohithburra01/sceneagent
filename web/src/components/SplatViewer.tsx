"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useViewerStore, Vec3 } from "@/stores/viewer";

interface SplatViewerProps {
  splatUrl: string;
}

/**
 * Client-only Gaussian splat viewer.
 *
 * Wraps @mkkellogg/gaussian-splats-3d Viewer and exposes a side-channel
 * so the rest of the app can lerp the camera via zustand's flyToPosition.
 *
 * Fallback: if the compressed .ply does not load, convert to .ksplat via
 * the @mkkellogg/gaussian-splats-3d CLI (see web/README.md).
 */
export default function SplatViewer({ splatUrl }: SplatViewerProps) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<unknown | null>(null);
  const mountedRef = useRef(false);

  // fly-to animation state
  const flyAnimRef = useRef<{
    from: THREE.Vector3;
    to: THREE.Vector3;
    start: number;
    duration: number;
  } | null>(null);
  const rafRef = useRef<number | null>(null);

  const flyToPosition = useViewerStore((s) => s.flyToPosition);

  // mount once
  useEffect(() => {
    if (mountedRef.current || !rootRef.current) return;
    mountedRef.current = true;

    let cancelled = false;

    (async () => {
      try {
        const mod = await import("@mkkellogg/gaussian-splats-3d");
        if (cancelled || !rootRef.current) return;

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const ViewerCtor: any = (mod as any).Viewer;
        const viewer = new ViewerCtor({
          rootElement: rootRef.current,
          cameraUp: [0, 0, 1],
          initialCameraPosition: [4, 4, 2],
          initialCameraLookAt: [0, 0, 1],
          sharedMemoryForWorkers: false,
        });

        viewerRef.current = viewer;

        await viewer.addSplatScene(splatUrl, {
          showLoadingUI: true,
          progressiveLoad: true,
        });
        viewer.start();
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("[SplatViewer] failed to load splat:", err);
      }
    })();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const v = viewerRef.current as any;
      if (v?.dispose) {
        try {
          v.dispose();
        } catch {
          /* ignore */
        }
      }
      viewerRef.current = null;
      mountedRef.current = false;
    };
  }, [splatUrl]);

  // react to flyTo: lerp camera over 1s
  useEffect(() => {
    if (!flyToPosition) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const v = viewerRef.current as any;
    if (!v?.camera) return;

    const cam: THREE.Camera = v.camera;
    const target = toVec3(flyToPosition);
    // Position the camera ~2.5m away from target along a consistent offset so
    // the user sees the point we are flying to.
    const offset = new THREE.Vector3(2.5, 2.5, 1.5);
    const destination = target.clone().add(offset);

    flyAnimRef.current = {
      from: cam.position.clone(),
      to: destination,
      start: performance.now(),
      duration: 1000,
    };

    const tick = () => {
      const anim = flyAnimRef.current;
      if (!anim) return;
      const now = performance.now();
      const t = Math.min(1, (now - anim.start) / anim.duration);
      // easeInOutCubic
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      cam.position.lerpVectors(anim.from, anim.to, ease);
      cam.lookAt(target);

      // Nudge controls target if the viewer exposes them.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const controls: any = v.controls;
      if (controls?.target?.copy) {
        controls.target.copy(target);
        if (controls.update) controls.update();
      }

      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        flyAnimRef.current = null;
      }
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [flyToPosition]);

  return (
    <div
      ref={rootRef}
      className="absolute inset-0 bg-black"
      style={{ width: "100%", height: "100%" }}
    />
  );
}

function toVec3(v: Vec3): THREE.Vector3 {
  return new THREE.Vector3(v[0], v[1], v[2]);
}
