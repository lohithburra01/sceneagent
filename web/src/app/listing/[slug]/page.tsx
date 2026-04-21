"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import {
  getScene,
  getDetections,
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
  // viewer kept in React state (not a ref) so ObjectOverlay re-renders
  // the moment the splat viewer attaches its THREE scene.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [viewer, setViewer] = useState<any | null>(null);

  const sceneQuery = useQuery({
    queryKey: ["scene", slug],
    queryFn: () => getScene(slug),
  });

  const detectionsQuery = useQuery({
    queryKey: ["detections", slug],
    queryFn: () => getDetections(slug),
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

  // Notes are matched once per session so chat tools have hotspots to
  // talk about. Detections (the bbox-able objects) are independent of
  // this and load via getDetections above.
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

  const detections = detectionsQuery.data ?? [];

  return (
    <main className="relative w-screen h-screen overflow-hidden bg-neutral-950 text-neutral-100">
      <div className="absolute inset-0 mr-[340px]">
        <SplatViewer splatUrl={splatUrl} onViewerReady={setViewer} />
        <ObjectOverlay detections={detections} viewer={viewer} />
      </div>

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

      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20 pointer-events-none mr-[340px]">
        <div className="px-3 py-1.5 rounded-full text-[11px] text-neutral-400 bg-neutral-950/60 border border-neutral-800 backdrop-blur tracking-wide">
          W S forward · A D strafe · Q E up-down · drag to look · hover sidebar item to highlight
        </div>
      </div>

      <InventorySidebar detections={detections} metrics={metricsQuery.data} />
      <ChatOverlay slug={slug} />
    </main>
  );
}
