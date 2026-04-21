"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { Hotspot } from "@/lib/api";
import { useViewerStore } from "@/stores/viewer";

interface Props {
  hotspots: Hotspot[];
  viewerRef: React.MutableRefObject<unknown>;
}

/**
 * Renders a thin wireframe bbox for the active/hovered hotspot into the
 * splat viewer's THREE.Scene. No persistent on-screen chrome; the sidebar
 * drives selection via activeObjectId / hoveredObjectId.
 */
export default function ObjectOverlay({ hotspots, viewerRef }: Props) {
  const activeId = useViewerStore((s) => s.activeObjectId);
  const hoverId = useViewerStore((s) => s.hoveredObjectId);
  const boxRef = useRef<THREE.LineSegments | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const v: any = viewerRef.current;
    const scene: THREE.Scene | undefined =
      v?.splatScene || v?.scene || v?.threeScene;
    if (!scene) return;
    sceneRef.current = scene;

    if (!boxRef.current) {
      const geo = new THREE.BoxGeometry(1, 1, 1);
      const edges = new THREE.EdgesGeometry(geo);
      const mat = new THREE.LineBasicMaterial({
        color: 0xf5b97a,
        transparent: true,
        opacity: 0.95,
      });
      const box = new THREE.LineSegments(edges, mat);
      box.visible = false;
      boxRef.current = box;
      scene.add(box);
    }

    return () => {
      const box = boxRef.current;
      const s = sceneRef.current;
      if (box && s) {
        s.remove(box);
        box.geometry.dispose();
        (box.material as THREE.Material).dispose();
      }
      boxRef.current = null;
      sceneRef.current = null;
    };
  }, [viewerRef]);

  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;

    const id = hoverId ?? activeId;
    const h = id ? hotspots.find((x) => x.id === id) : undefined;
    if (!h || !h.bbox_min || !h.bbox_max) {
      box.visible = false;
      return;
    }
    const [ax, ay, az] = h.bbox_min;
    const [bx, by, bz] = h.bbox_max;
    box.visible = true;
    box.position.set((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2);
    box.scale.set(
      Math.max(0.01, bx - ax),
      Math.max(0.01, by - ay),
      Math.max(0.01, bz - az)
    );
  }, [activeId, hoverId, hotspots]);

  return null;
}
