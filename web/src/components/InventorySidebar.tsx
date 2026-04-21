"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import { Detection, SceneMetrics } from "@/lib/api";
import { useViewerStore } from "@/stores/viewer";

interface Props {
  detections: Detection[];
  metrics: SceneMetrics | null | undefined;
}

export default function InventorySidebar({ detections, metrics }: Props) {
  const flyTo = useViewerStore((s) => s.flyTo);
  const setActive = useViewerStore((s) => s.setActiveObject);
  const setHover = useViewerStore((s) => s.setHoveredObject);
  const activeId = useViewerStore((s) => s.activeObjectId);
  const setInputFocused = useViewerStore((s) => s.setInputFocused);
  const [filter, setFilter] = useState("");
  // Default OFF — clicking should highlight only, not yank the camera.
  // Toggle on in the header when you actually want auto-fly.
  const [autoFly, setAutoFly] = useState(false);

  const grouped = useMemo(() => {
    const f = filter.trim().toLowerCase();
    const by: Record<string, Detection[]> = {};
    for (const d of detections) {
      if (f && !d.class_name.toLowerCase().includes(f)) continue;
      const k = d.class_name || "other";
      (by[k] ||= []).push(d);
    }
    return Object.entries(by).sort((a, b) => b[1].length - a[1].length);
  }, [detections, filter]);

  return (
    <aside
      className={clsx(
        "fixed right-0 top-0 h-screen w-[340px] z-20",
        "bg-neutral-950/85 backdrop-blur-xl border-l border-neutral-800/80",
        "text-neutral-100 flex flex-col"
      )}
    >
      <header className="px-5 pt-6 pb-3 border-b border-neutral-800/80">
        <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500">
          Inventory
        </div>
        <div className="text-lg font-medium mt-0.5">
          {detections.length} detected objects
        </div>
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          onFocus={() => setInputFocused(true)}
          onBlur={() => setInputFocused(false)}
          onKeyDown={(e) => {
            e.stopPropagation();
            if (e.key === "Escape") (e.target as HTMLInputElement).blur();
          }}
          placeholder="filter by class…"
          className={clsx(
            "mt-3 w-full bg-neutral-900/70 border border-neutral-800 rounded-md",
            "px-2.5 py-1.5 text-xs text-neutral-200",
            "placeholder:text-neutral-600 outline-none focus:border-neutral-700"
          )}
        />
        <label className="mt-2 flex items-center gap-2 text-[11px] text-neutral-400 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={autoFly}
            onChange={(e) => setAutoFly(e.target.checked)}
            className="accent-amber-300"
          />
          auto-fly camera on click
        </label>
      </header>
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {grouped.length === 0 && (
          <div className="px-2 py-6 text-xs text-neutral-500">
            {detections.length === 0
              ? "No detections — pipeline hasn't run."
              : "No matches for filter."}
          </div>
        )}
        {grouped.map(([cls, items]) => (
          <section key={cls}>
            <div className="px-2 text-[10px] uppercase tracking-[0.18em] text-neutral-500 mb-1">
              {cls} · {items.length}
            </div>
            <ul className="space-y-0.5">
              {items.map((d) => (
                <li key={d.id} className="group flex items-stretch gap-1">
                  <button
                    type="button"
                    onMouseEnter={() => setHover(d.id)}
                    onMouseLeave={() => setHover(null)}
                    onClick={() => {
                      setActive(d.id);
                      if (autoFly) flyTo(d.centroid);
                    }}
                    className={clsx(
                      "flex-1 text-left px-2 py-1.5 rounded-md text-sm",
                      "hover:bg-neutral-900/80 transition-colors",
                      activeId === d.id && "bg-neutral-900/80 ring-1 ring-amber-300/40"
                    )}
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-neutral-200 truncate">
                        {cls} <span className="text-neutral-500">#{d.instance_id}</span>
                      </span>
                      <span className="text-[10px] text-amber-300/80 tabular-nums shrink-0">
                        {(d.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </button>
                  {/* explicit fly-to button so the user can opt in per click */}
                  <button
                    type="button"
                    onClick={() => {
                      setActive(d.id);
                      flyTo(d.centroid);
                    }}
                    title="Fly camera to this object"
                    aria-label="Fly to"
                    className={clsx(
                      "px-2 rounded-md text-[11px] text-neutral-500",
                      "opacity-0 group-hover:opacity-100",
                      "hover:bg-neutral-900/80 hover:text-amber-300 transition-all"
                    )}
                  >
                    ✈
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
      <footer className="border-t border-neutral-800/80 px-5 py-3 text-[10px] text-neutral-500 tracking-wide">
        {metrics
          ? `F1 ${(metrics.f1 * 100).toFixed(1)}% · ${metrics.tp}/${metrics.num_predicted} matched @ IoU≥${metrics.iou_threshold.toFixed(2)}`
          : "pipeline metrics pending…"}
      </footer>
    </aside>
  );
}
