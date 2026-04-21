"use client";

import { useMemo } from "react";
import clsx from "clsx";
import { Hotspot, SceneMetrics } from "@/lib/api";
import { useViewerStore } from "@/stores/viewer";

interface Props {
  hotspots: Hotspot[];
  metrics: SceneMetrics | null | undefined;
}

export default function InventorySidebar({ hotspots, metrics }: Props) {
  const flyTo = useViewerStore((s) => s.flyTo);
  const setActive = useViewerStore((s) => s.setActiveObject);
  const setHover = useViewerStore((s) => s.setHoveredObject);
  const activeId = useViewerStore((s) => s.activeObjectId);

  const grouped = useMemo(() => {
    const by: Record<string, Hotspot[]> = {};
    for (const h of hotspots) {
      const k = h.class_name ?? "other";
      (by[k] ||= []).push(h);
    }
    return Object.entries(by).sort((a, b) => b[1].length - a[1].length);
  }, [hotspots]);

  return (
    <aside
      className={clsx(
        "fixed right-0 top-0 h-screen w-[340px] z-20",
        "bg-neutral-950/85 backdrop-blur-xl border-l border-neutral-800/80",
        "text-neutral-100 flex flex-col"
      )}
    >
      <header className="px-5 pt-6 pb-4 border-b border-neutral-800/80">
        <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-500">
          Inventory
        </div>
        <div className="text-lg font-medium mt-0.5">
          {hotspots.length} detected objects
        </div>
      </header>
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {grouped.length === 0 && (
          <div className="px-2 py-6 text-xs text-neutral-500">
            No objects detected yet.
          </div>
        )}
        {grouped.map(([cls, items]) => (
          <section key={cls}>
            <div className="px-2 text-[10px] uppercase tracking-[0.18em] text-neutral-500 mb-1">
              {cls} · {items.length}
            </div>
            <ul className="space-y-0.5">
              {items.map((h) => (
                <li key={h.id}>
                  <button
                    type="button"
                    onMouseEnter={() => setHover(h.id)}
                    onMouseLeave={() => setHover(null)}
                    onClick={() => {
                      setActive(h.id);
                      flyTo(h.position);
                    }}
                    className={clsx(
                      "w-full text-left px-2 py-1.5 rounded-md text-sm",
                      "hover:bg-neutral-900/80 transition-colors",
                      activeId === h.id && "bg-neutral-900/80"
                    )}
                  >
                    <div className="text-neutral-200 truncate">
                      {h.note_text || cls}
                    </div>
                    <div className="text-[10px] text-neutral-500 mt-0.5">
                      {(h.match_confidence * 100).toFixed(0)}%
                    </div>
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
