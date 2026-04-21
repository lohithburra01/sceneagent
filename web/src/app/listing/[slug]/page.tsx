"use client";

import { useEffect, useMemo, useRef } from "react";
import dynamic from "next/dynamic";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import {
  getScene,
  getHotspots,
  seedMatch,
  splatUrlFor,
  getMetrics,
} from "@/lib/api";
import ObjectOverlay from "@/components/ObjectOverlay";
import InventorySidebar from "@/components/InventorySidebar";
import ChatOverlay from "@/components/ChatOverlay";

const SplatViewer = dynamic(() => import("@/components/SplatViewer"), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 flex items-center justify-center bg-neutral-950 text-neutral-500 text-sm">
      loading 3D scene…
    </div>
  ),
});

interface PageProps {
  params: { slug: string };
}

export default function ListingPage({ params }: PageProps) {
  const { slug } = params;
  const qc = useQueryClient();
  const viewerRef = useRef<unknown>(null);

  const sceneQuery = useQuery({
    queryKey: ["scene", slug],
    queryFn: () => getScene(slug),
  });

  const hotspotsQuery = useQuery({
    queryKey: ["hotspots", slug],
    queryFn: () => getHotspots(slug),
  });

  const metricsQuery = useQuery({
    queryKey: ["metrics", slug],
    queryFn: () => getMetrics(slug),
  });

  const seed = useMutation({
    mutationFn: () => seedMatch(slug),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hotspots", slug] });
    },
  });

  useEffect(() => {
    seed.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug]);

  const splatUrl = useMemo(() => {
    const raw = sceneQuery.data?.splat_url;
    if (!raw) return splatUrlFor(slug);
    if (raw.startsWith("http")) return raw;
    const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return `${base}${raw.startsWith("/") ? raw : `/${raw}`}`;
  }, [sceneQuery.data, slug]);

  const hotspots = hotspotsQuery.data ?? [];

  return (
    <main className="relative w-screen h-screen overflow-hidden bg-neutral-950 text-neutral-100">
      {/* Splat fills the area left of the sidebar */}
      <div className="absolute inset-0 mr-[340px]">
        <SplatViewer
          splatUrl={splatUrl}
          onViewerReady={(v) => {
            viewerRef.current = v;
          }}
        />
        <ObjectOverlay hotspots={hotspots} viewerRef={viewerRef} />
      </div>

      {/* Title chip */}
      <div className="fixed top-6 left-6 z-20 pointer-events-none">
        <div className="text-[10px] uppercase tracking-[0.22em] text-neutral-400">
          SceneAgent
        </div>
        <div className="text-xl text-neutral-50 font-medium mt-0.5">
          {sceneQuery.data?.title ?? (sceneQuery.isLoading ? "…" : slug)}
        </div>
        {sceneQuery.data?.address && (
          <div className="text-xs text-neutral-400 mt-0.5">
            {sceneQuery.data.address}
          </div>
        )}
      </div>

      {/* Controls hint */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20 pointer-events-none mr-[340px]">
        <div className="px-3 py-1.5 rounded-full text-[11px] text-neutral-400 bg-neutral-950/60 border border-neutral-800 backdrop-blur tracking-wide">
          W S forward · A D strafe · Q E up-down · drag to look
        </div>
      </div>

      <InventorySidebar hotspots={hotspots} metrics={metricsQuery.data} />
      <ChatOverlay slug={slug} />
    </main>
  );
}
