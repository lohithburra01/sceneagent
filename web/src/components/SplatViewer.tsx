"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useViewerStore, Vec3 } from "@/stores/viewer";

interface SplatViewerProps {
  splatUrl: string;
  onViewerReady?: (viewer: unknown) => void;
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
export default function SplatViewer({ splatUrl, onViewerReady }: SplatViewerProps) {
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
        onViewerReady?.(viewer);

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
    // Position the camera ~2 m away from the target, *toward* the
    // current camera position. This keeps us inside the room and avoids
    // the old "always offset (+2.5,+2.5,+1.5)" trap that landed the
    // camera outside the bbox for objects in the far corner.
    const dirFromTarget = cam.position.clone().sub(target);
    if (dirFromTarget.lengthSq() < 1e-6) {
      // current camera is on top of the target — fall back to a
      // small eye-level back-up
      dirFromTarget.set(0, 1, 0.3);
    }
    dirFromTarget.normalize();
    // keep a slight upward component so we look slightly down at the object
    if (dirFromTarget.z < 0.15) dirFromTarget.z = 0.15;
    dirFromTarget.normalize();
    const destination = target.clone().add(dirFromTarget.multiplyScalar(2.0));

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

  // Blender-style fly controls: W/S forward-back along view, A/D strafe,
  // Q/E world-up/down. Captured before the viewer's built-in handler so
  // mkkellogg's WASD doesn't interfere; skipped when focus is in a text
  // input so chat typing doesn't move the camera.
  useEffect(() => {
    const keys = new Set<string>();
    const SPEED = 3.0;
    const DAMP = 10.0;
    let raf = 0;
    let last = performance.now();

    function targetIsTextInput(t: EventTarget | null) {
      if (!(t instanceof HTMLElement)) return false;
      const tag = t.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || t.isContentEditable;
    }

    function onKeyDown(e: KeyboardEvent) {
      if (targetIsTextInput(e.target)) return;
      const k = e.key.toLowerCase();
      if (["w", "a", "s", "d", "q", "e"].includes(k)) {
        keys.add(k);
        e.preventDefault();
        e.stopImmediatePropagation();
      }
    }
    function onKeyUp(e: KeyboardEvent) {
      keys.delete(e.key.toLowerCase());
    }

    const vel = new THREE.Vector3();

    function tick() {
      const now = performance.now();
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const v: any = viewerRef.current;
      const cam: THREE.Camera | undefined = v?.camera;
      if (!cam) {
        raf = requestAnimationFrame(tick);
        return;
      }

      const forward = new THREE.Vector3();
      cam.getWorldDirection(forward);
      forward.normalize();
      const worldUp = new THREE.Vector3(0, 0, 1);
      const right = new THREE.Vector3().crossVectors(forward, worldUp).normalize();

      const accel = new THREE.Vector3();
      if (keys.has("w")) accel.add(forward);
      if (keys.has("s")) accel.addScaledVector(forward, -1);
      if (keys.has("d")) accel.add(right);
      if (keys.has("a")) accel.addScaledVector(right, -1);
      if (keys.has("e")) accel.add(worldUp);
      if (keys.has("q")) accel.addScaledVector(worldUp, -1);

      if (accel.lengthSq() > 0) accel.normalize().multiplyScalar(SPEED);

      vel.lerp(accel, Math.min(1, DAMP * dt));
      cam.position.addScaledVector(vel, dt);

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const controls: any = v?.controls;
      if (controls?.target?.copy) {
        const look = cam.position.clone().add(forward);
        controls.target.copy(look);
        if (controls.update) controls.update();
      }

      raf = requestAnimationFrame(tick);
    }

    window.addEventListener("keydown", onKeyDown, { capture: true });
    window.addEventListener("keyup", onKeyUp, { capture: true });
    raf = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener("keydown", onKeyDown, { capture: true } as EventListenerOptions);
      window.removeEventListener("keyup", onKeyUp, { capture: true } as EventListenerOptions);
      cancelAnimationFrame(raf);
    };
  }, []);

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
