"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { Detection } from "@/lib/api";
import { useViewerStore } from "@/stores/viewer";

interface Props {
  detections: Detection[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  viewer: any | null;
}

/**
 * Adds a wireframe bbox + a text-sprite label for the active/hovered
 * detection to the splat viewer's THREE.Scene.
 *
 * Robust against mkkellogg's viewer attaching its scene asynchronously:
 * the effect polls a few candidate property names until a usable
 * THREE.Scene shows up. Logs the result to the console so we can debug
 * if it never resolves.
 */
export default function ObjectOverlay({ detections, viewer }: Props) {
  const activeId = useViewerStore((s) => s.activeObjectId);
  const hoverId = useViewerStore((s) => s.hoveredObjectId);

  const boxRef = useRef<THREE.LineSegments | null>(null);
  const spriteRef = useRef<THREE.Sprite | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const detectionsRef = useRef<Detection[]>(detections);
  const selRef = useRef<{ active: string | null; hover: string | null }>({
    active: null,
    hover: null,
  });

  detectionsRef.current = detections;
  selRef.current = { active: activeId, hover: hoverId };

  // Mount the wireframe + sprite the moment a THREE.Scene is reachable
  // through the viewer. Polls because mkkellogg's Viewer constructs the
  // scene before addSplatScene returns.
  useEffect(() => {
    if (!viewer) return;
    let cancelled = false;
    let pollHandle: number | null = null;
    let frameHandle: number | null = null;

    function findScene(): THREE.Scene | null {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const v = viewer as any;
      const candidates = [
        v?.threeScene,
        v?.scene,
        v?.splatMesh?.scene,
        v?.splatMesh?.parent,
        v?.renderer?.scene,
      ];
      for (const c of candidates) {
        if (c && typeof c === "object" && (c as THREE.Object3D).isObject3D) {
          return c as THREE.Scene;
        }
      }
      return null;
    }

    function attach(scene: THREE.Scene) {
      sceneRef.current = scene;

      const geo = new THREE.BoxGeometry(1, 1, 1);
      const edges = new THREE.EdgesGeometry(geo);
      const mat = new THREE.LineBasicMaterial({
        color: 0xf5b97a,
        transparent: true,
        opacity: 1.0,
        depthTest: false,
        depthWrite: false,
      });
      const box = new THREE.LineSegments(edges, mat);
      box.renderOrder = 9999;
      box.visible = false;
      scene.add(box);
      boxRef.current = box;

      const sprite = makeLabelSprite("");
      sprite.visible = false;
      sprite.renderOrder = 10000;
      scene.add(sprite);
      spriteRef.current = sprite;

      // eslint-disable-next-line no-console
      console.log("[ObjectOverlay] attached to scene", {
        keys: Object.keys(viewer).slice(0, 30),
        children: scene.children.length,
      });

      // drive a per-frame update loop so hover/click changes always reflect
      const tick = () => {
        if (cancelled) return;
        applySelection();
        frameHandle = requestAnimationFrame(tick);
      };
      frameHandle = requestAnimationFrame(tick);
    }

    function applySelection() {
      const box = boxRef.current;
      const sprite = spriteRef.current;
      if (!box || !sprite) return;
      const id = selRef.current.hover ?? selRef.current.active;
      const d = id ? detectionsRef.current.find((x) => x.id === id) : undefined;
      if (!d || !d.bbox_min || !d.bbox_max) {
        if (box.visible) box.visible = false;
        if (sprite.visible) sprite.visible = false;
        return;
      }
      const [ax, ay, az] = d.bbox_min;
      const [bx, by, bz] = d.bbox_max;
      const cx = (ax + bx) / 2;
      const cy = (ay + by) / 2;
      const cz = (az + bz) / 2;
      box.visible = true;
      box.position.set(cx, cy, cz);
      box.scale.set(
        Math.max(0.01, bx - ax),
        Math.max(0.01, by - ay),
        Math.max(0.01, bz - az)
      );

      const label = `${d.class_name} · ${(d.confidence * 100).toFixed(0)}%`;
      const mat = sprite.material as THREE.SpriteMaterial;
      // Only re-render the label when text changed (cheap key on sprite.userData)
      if (sprite.userData.text !== label) {
        mat.map?.dispose();
        mat.map = makeLabelTexture(label);
        mat.needsUpdate = true;
        sprite.userData.text = label;
      }
      sprite.visible = true;
      sprite.position.set(cx, cy, bz + 0.18);
      const w = Math.max(0.7, Math.min(2.2, (bx - ax) * 1.3));
      sprite.scale.set(w, w * 0.28, 1);
    }

    // Try immediately, then poll every 200ms for up to 10s.
    let attempts = 0;
    function tryAttach() {
      if (cancelled) return;
      const scene = findScene();
      if (scene) {
        attach(scene);
        return;
      }
      attempts += 1;
      if (attempts > 50) {
        // eslint-disable-next-line no-console
        console.warn(
          "[ObjectOverlay] gave up: no THREE.Scene exposed by viewer after 10s",
          { keys: Object.keys(viewer || {}).slice(0, 40) }
        );
        return;
      }
      pollHandle = window.setTimeout(tryAttach, 200);
    }
    tryAttach();

    return () => {
      cancelled = true;
      if (pollHandle) clearTimeout(pollHandle);
      if (frameHandle) cancelAnimationFrame(frameHandle);
      const scene = sceneRef.current;
      const box = boxRef.current;
      const sprite = spriteRef.current;
      if (box && scene) {
        scene.remove(box);
        box.geometry.dispose();
        (box.material as THREE.Material).dispose();
      }
      if (sprite && scene) {
        scene.remove(sprite);
        (sprite.material as THREE.SpriteMaterial).map?.dispose();
        (sprite.material as THREE.SpriteMaterial).dispose();
      }
      boxRef.current = null;
      spriteRef.current = null;
      sceneRef.current = null;
    };
  }, [viewer]);

  return null;
}

function makeLabelTexture(text: string): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 128;
  const ctx = c.getContext("2d")!;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.font = "600 56px Inter, ui-sans-serif, system-ui, sans-serif";
  const metrics = ctx.measureText(text);
  const padX = 18;
  const tw = Math.min(c.width - padX * 2, metrics.width);
  const x0 = (c.width - tw) / 2 - padX;
  const x1 = x0 + tw + padX * 2;
  const y0 = (c.height - 64) / 2;
  const y1 = y0 + 64;
  const r = 16;
  ctx.fillStyle = "rgba(15, 15, 15, 0.94)";
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
  ctx.strokeStyle = "rgba(245, 185, 122, 0.85)";
  ctx.lineWidth = 2;
  ctx.stroke();
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
    depthWrite: false,
    transparent: true,
  });
  const s = new THREE.Sprite(mat);
  s.userData.text = text;
  return s;
}
