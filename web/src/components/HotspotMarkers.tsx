"use client";

import { useState } from "react";
import clsx from "clsx";
import { Hotspot } from "@/lib/api";
import { useViewerStore } from "@/stores/viewer";

const CATEGORY_ICONS: Record<string, string> = {
  feature: "\u2705",    // ✅
  included: "\uD83C\uDF81", // 🎁
  issue: "\u26A0\uFE0F",    // ⚠️
  info: "\u2139\uFE0F",     // ℹ️
  spec: "\uD83D\uDCCF",     // 📏
  story: "\uD83D\uDCD6",    // 📖
  other: "\uD83D\uDCCD",    // 📍
};

function iconFor(category: string | null | undefined): string {
  if (!category) return CATEGORY_ICONS.other;
  return CATEGORY_ICONS[category] ?? CATEGORY_ICONS.other;
}

interface HotspotMarkersProps {
  hotspots: Hotspot[];
}

/**
 * Clickable category-icon buttons overlaid on the splat viewer.
 *
 * v1 positioning model: because projecting 3D → 2D screen requires the
 * live Three.js camera, we lay markers out in a simple radial ring around
 * screen center. Clicking a marker calls flyTo(hotspot.position), which the
 * SplatViewer lerps the camera toward — so the hotspot *is* visually
 * centered a moment after the click. The popup card shows the note text.
 */
export default function HotspotMarkers({ hotspots }: HotspotMarkersProps) {
  const flyTo = useViewerStore((s) => s.flyTo);
  const [openId, setOpenId] = useState<string | null>(null);

  const open = openId
    ? hotspots.find((h) => h.id === openId) ?? null
    : null;

  // Lay out markers in a ring around screen center.
  const n = hotspots.length;
  const ringRadius = 240;

  return (
    <div className="pointer-events-none absolute inset-0 z-10">
      <div className="relative w-full h-full">
        {hotspots.map((h, i) => {
          const angle = (i / Math.max(n, 1)) * Math.PI * 2;
          const x = Math.cos(angle) * ringRadius;
          const y = Math.sin(angle) * ringRadius;
          return (
            <button
              key={h.id}
              type="button"
              onClick={() => {
                flyTo(h.position);
                setOpenId(h.id);
              }}
              className={clsx(
                "pointer-events-auto absolute top-1/2 left-1/2",
                "-translate-x-1/2 -translate-y-1/2",
                "h-10 w-10 rounded-full bg-white/90 hover:bg-white",
                "shadow-lg ring-2 ring-black/20 backdrop-blur",
                "flex items-center justify-center text-xl",
                "transition-transform hover:scale-110"
              )}
              style={{ transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))` }}
              title={h.note_text}
              aria-label={`Hotspot: ${h.category ?? "other"}`}
            >
              <span aria-hidden>{iconFor(h.category)}</span>
            </button>
          );
        })}

        {open && (
          <div
            className={clsx(
              "pointer-events-auto absolute",
              "top-4 left-1/2 -translate-x-1/2",
              "max-w-md w-[90%] rounded-xl bg-neutral-900/95 text-white",
              "shadow-2xl border border-neutral-700 p-4 backdrop-blur"
            )}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="text-2xl" aria-hidden>
                  {iconFor(open.category)}
                </span>
                <div>
                  <div className="text-xs uppercase tracking-wider text-neutral-400">
                    {open.category ?? "other"}
                    {open.class_name ? ` · ${open.class_name}` : ""}
                  </div>
                  <div className="text-sm font-medium">
                    {open.note_text}
                  </div>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setOpenId(null)}
                className="text-neutral-400 hover:text-white text-lg leading-none"
                aria-label="Close hotspot popup"
              >
                ×
              </button>
            </div>
            <div className="mt-2 text-[10px] text-neutral-500">
              confidence {open.match_confidence.toFixed(2)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
