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
 * Renders a thin amber wireframe bbox + a text-sprite label
 * (`class · 67%`) for the active/hovered detection into the splat
 * viewer's THREE.Scene. Driven from the sidebar's hover/click state.
 *
 * Viewer is passed as a prop (lifted from a useState in the page) so
 * this component re-renders the moment the splat viewer is ready —
 * not the previous bug where a useRef silently never re-fired the
 * effect.
 */
export default function ObjectOverlay({ detections, viewer }: Props) {
  const activeId = useViewerStore((s) => s.activeObjectId);
  const hoverId = useViewerStore((s) => s.hoveredObjectId);

  const boxRef = useRef<THREE.LineSegments | null>(null);
  const spriteRef = useRef<THREE.Sprite | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const [, setReadyTick] = useState(0);

  // Mount: attach a wireframe box + a text sprite to the splat viewer's scene.
  useEffect(() => {
    if (!viewer) return;
    // mkkellogg's viewer exposes the THREE.Scene as `threeScene`; some
    // builds also expose `scene`. Try both.
    const scene: THREE.Scene | undefined =
      viewer.threeScene || viewer.scene || viewer.splatScene;
    if (!scene) {
      // viewer is set but the splat hasn't finished mounting yet — retry.
      const t = setTimeout(() => setReadyTick((n) => n + 1), 200);
      return () => clearTimeout(t);
    }
    sceneRef.current = scene;

    if (!boxRef.current) {
      const geo = new THREE.BoxGeometry(1, 1, 1);
      const edges = new THREE.EdgesGeometry(geo);
      const mat = new THREE.LineBasicMaterial({
        color: 0xf5b97a,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      });
      const box = new THREE.LineSegments(edges, mat);
      box.renderOrder = 999;
      box.visible = false;
      boxRef.current = box;
      scene.add(box);
    }
    if (!spriteRef.current) {
      const sprite = makeLabelSprite("");
      sprite.visible = false;
      sprite.renderOrder = 1000;
      spriteRef.current = sprite;
      scene.add(sprite);
    }

    return () => {
      const box = boxRef.current;
      const sprite = spriteRef.current;
      const s = sceneRef.current;
      if (box && s) {
        s.remove(box);
        box.geometry.dispose();
        (box.material as THREE.Material).dispose();
      }
      if (sprite && s) {
        s.remove(sprite);
        (sprite.material as THREE.SpriteMaterial).map?.dispose();
        (sprite.material as THREE.SpriteMaterial).dispose();
      }
      boxRef.current = null;
      spriteRef.current = null;
      sceneRef.current = null;
    };
  }, [viewer]);

  // React to selection: position the box/sprite and update the label texture.
  useEffect(() => {
    const box = boxRef.current;
    const sprite = spriteRef.current;
    if (!box || !sprite) return;

    const id = hoverId ?? activeId;
    const d = id ? detections.find((x) => x.id === id) : undefined;
    if (!d || !d.bbox_min || !d.bbox_max) {
      box.visible = false;
      sprite.visible = false;
      return;
    }
    const [ax, ay, az] = d.bbox_min;
    const [bx, by, bz] = d.bbox_max;
    box.visible = true;
    box.position.set((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2);
    box.scale.set(
      Math.max(0.01, bx - ax),
      Math.max(0.01, by - ay),
      Math.max(0.01, bz - az)
    );

    // label above the box
    sprite.visible = true;
    const text = `${d.class_name} · ${(d.confidence * 100).toFixed(0)}%`;
    const texture = makeLabelTexture(text);
    const mat = sprite.material as THREE.SpriteMaterial;
    mat.map?.dispose();
    mat.map = texture;
    mat.needsUpdate = true;
    sprite.position.set(
      (ax + bx) / 2,
      (ay + by) / 2,
      bz + 0.15
    );
    // scale sprite proportionally to bbox size so labels don't dominate
    const w = Math.max(0.6, Math.min(2.0, (bx - ax) * 1.2));
    sprite.scale.set(w, w * 0.28, 1);
  }, [activeId, hoverId, detections]);

  return null;
}

function makeLabelTexture(text: string): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 128;
  const ctx = c.getContext("2d")!;
  ctx.clearRect(0, 0, c.width, c.height);
  // pill background
  const padX = 18;
  const padY = 12;
  ctx.font = "600 56px Inter, ui-sans-serif, system-ui, sans-serif";
  const metrics = ctx.measureText(text);
  const tw = Math.min(c.width - padX * 2, metrics.width);
  const x0 = (c.width - tw) / 2 - padX;
  const x1 = x0 + tw + padX * 2;
  const y0 = (c.height - 64) / 2;
  const y1 = y0 + 64;
  const r = 16;
  ctx.fillStyle = "rgba(15, 15, 15, 0.92)";
  ctx.beginPath();
  ctx.moveTo(x0 + r, y0);
  ctx.lineTo(x1 - r, y0);
  ctx.quadraticCurveTo(x1, y0, x1, y0 + r);
  ctx.lineTo(x1, y1 - r);
  ctx.quadraticCurveTo(x1, y1, x1 - r, y1);
  ctx.lineTo(x0 + r, y1);
  ctx.quadraticCurveTo(x0, y1, x0, y1 - r);
  ctx.lineTo(x0, y0 + r);
  ctx.quadraticCurveTo(x0, y0, x0 + r, y0);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(245, 185, 122, 0.7)";
  ctx.lineWidth = 2;
  ctx.stroke();
  // text
  ctx.fillStyle = "#f5b97a";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, c.width / 2, c.height / 2);

  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  return tex;
}

function makeLabelSprite(text: string): THREE.Sprite {
  const tex = makeLabelTexture(text);
  const mat = new THREE.SpriteMaterial({
    map: tex,
    depthTest: false,
    transparent: true,
  });
  return new THREE.Sprite(mat);
}
