"use client";

import { useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import { getScene, getHotspots, seedMatch, splatUrlFor } from "@/lib/api";
import HotspotMarkers from "@/components/HotspotMarkers";
import ChatOverlay from "@/components/ChatOverlay";

// Splat viewer is heavy + browser-only — client-only dynamic import.
const SplatViewer = dynamic(() => import("@/components/SplatViewer"), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 flex items-center justify-center bg-black text-neutral-500 text-sm">
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

  const sceneQuery = useQuery({
    queryKey: ["scene", slug],
    queryFn: () => getScene(slug),
  });

  const hotspotsQuery = useQuery({
    queryKey: ["hotspots", slug],
    queryFn: () => getHotspots(slug),
  });

  const seed = useMutation({
    mutationFn: () => seedMatch(slug),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hotspots", slug] });
    },
  });

  // On mount, ensure demo notes are matched to hotspots.
  useEffect(() => {
    seed.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug]);

  // Prefer an absolute splat_url from the scene record; fall back to the
  // conventional /static/scene/:slug/3dgs_compressed.ply path.
  const splatUrl = useMemo(() => {
    const raw = sceneQuery.data?.splat_url;
    if (!raw) return splatUrlFor(slug);
    if (raw.startsWith("http")) return raw;
    const base =
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return `${base}${raw.startsWith("/") ? raw : `/${raw}`}`;
  }, [sceneQuery.data, slug]);

  const hotspots = hotspotsQuery.data ?? [];

  return (
    <main className="relative w-screen h-screen overflow-hidden bg-black text-white">
      {/* Title card */}
      <div className="absolute top-4 left-4 z-20">
        <div className="rounded-lg bg-neutral-900/80 backdrop-blur px-4 py-2 border border-neutral-700">
          <div className="text-xs uppercase tracking-wider text-neutral-400">
            SceneAgent
          </div>
          <div className="text-lg font-semibold">
            {sceneQuery.data?.title ?? (sceneQuery.isLoading ? "…" : slug)}
          </div>
          {sceneQuery.data?.address && (
            <div className="text-xs text-neutral-400">
              {sceneQuery.data.address}
            </div>
          )}
        </div>
      </div>

      {/* Splat + markers */}
      <SplatViewer splatUrl={splatUrl} />
      <HotspotMarkers hotspots={hotspots} />

      {/* Chat */}
      <ChatOverlay slug={slug} />
    </main>
  );
}
