"use client";

import { useCallback, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useQuery, useQueryClient, useMutation } from "@tanstack/react-query";
import {
  createNote,
  getHotspots,
  getScene,
  videoUrlFor,
  Hotspot,
} from "@/lib/api";

// react-player must be client-only.
const ReactPlayer = dynamic(() => import("react-player/lazy"), { ssr: false });

const CATEGORY_ICONS: Record<string, string> = {
  feature: "\u2705",
  included: "\uD83C\uDF81",
  issue: "\u26A0\uFE0F",
  info: "\u2139\uFE0F",
  spec: "\uD83D\uDCCF",
  story: "\uD83D\uDCD6",
  other: "\uD83D\uDCCD",
};

interface PageProps {
  params: { slug: string };
}

export default function AdminPage({ params }: PageProps) {
  const { slug } = params;
  const qc = useQueryClient();

  // react-player's ref type differs between CJS and ESM — keep it loose.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const playerRef = useRef<any>(null);
  const [noteText, setNoteText] = useState("");
  const [lastInsertedId, setLastInsertedId] = useState<string | null>(null);

  const sceneQuery = useQuery({
    queryKey: ["scene", slug],
    queryFn: () => getScene(slug),
  });
  const hotspotsQuery = useQuery({
    queryKey: ["hotspots", slug],
    queryFn: () => getHotspots(slug),
    refetchInterval: 3000,
  });

  const addNote = useMutation({
    mutationFn: async (args: { text: string; ts: number }) =>
      createNote(slug, args.text, args.ts),
    onSuccess: (data) => {
      setNoteText("");
      setLastInsertedId(data.note_id);
      qc.invalidateQueries({ queryKey: ["hotspots", slug] });
    },
  });

  const onAddNote = useCallback(() => {
    const text = noteText.trim();
    if (!text) return;
    let ts = 0;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const p: any = playerRef.current;
    try {
      if (p && typeof p.getCurrentTime === "function") {
        ts = Number(p.getCurrentTime()) || 0;
      }
    } catch {
      ts = 0;
    }
    addNote.mutate({ text, ts });
  }, [noteText, addNote]);

  const hotspots = hotspotsQuery.data ?? [];
  const videoUrl = videoUrlFor(slug);

  return (
    <main className="min-h-screen w-full bg-neutral-950 text-neutral-100">
      <header className="border-b border-neutral-800 px-6 py-3 flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-wider text-neutral-500">
            Realtor
          </div>
          <h1 className="text-xl font-semibold">
            {sceneQuery.data?.title ?? slug} — scrubber
          </h1>
        </div>
        <a
          href={`/listing/${slug}`}
          className="text-sm text-emerald-400 hover:text-emerald-300"
        >
          ← buyer view
        </a>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
        {/* Left: video scrubber + note composer */}
        <section className="space-y-3">
          <div className="aspect-video bg-black rounded-xl overflow-hidden border border-neutral-800">
            <ReactPlayer
              ref={playerRef}
              url={videoUrl}
              controls
              width="100%"
              height="100%"
            />
          </div>

          <div className="rounded-xl border border-neutral-800 bg-neutral-900 p-3 space-y-2">
            <label className="text-xs uppercase tracking-wider text-neutral-500">
              Note at current time
            </label>
            <textarea
              value={noteText}
              onChange={(e) => setNoteText(e.target.value)}
              rows={3}
              placeholder='e.g. "The bedroom window sticks — pull firmly"'
              className="w-full rounded-lg bg-neutral-800 text-sm p-3 outline-none focus:ring-1 focus:ring-emerald-500 placeholder:text-neutral-500"
            />
            <div className="flex items-center justify-between">
              <div className="text-xs text-neutral-500">
                {addNote.isPending
                  ? "saving…"
                  : lastInsertedId
                  ? `last note saved: ${lastInsertedId.slice(0, 8)}…`
                  : "add timestamped notes while you watch."}
              </div>
              <button
                type="button"
                onClick={onAddNote}
                disabled={!noteText.trim() || addNote.isPending}
                className="rounded-lg px-3 py-2 text-sm font-medium bg-emerald-500 hover:bg-emerald-400 text-black disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Add note at current time
              </button>
            </div>
            {addNote.isError && (
              <div className="text-xs text-red-400">
                {(addNote.error as Error).message}
              </div>
            )}
          </div>
        </section>

        {/* Right: hotspots list */}
        <section className="space-y-3">
          <div className="flex items-baseline justify-between">
            <h2 className="text-lg font-semibold">Hotspots</h2>
            <div className="text-xs text-neutral-500">
              {hotspots.length} total ·{" "}
              {hotspotsQuery.isFetching ? "refreshing…" : "auto-refresh 3s"}
            </div>
          </div>

          {hotspotsQuery.isLoading ? (
            <div className="text-sm text-neutral-500">loading hotspots…</div>
          ) : hotspots.length === 0 ? (
            <div className="text-sm text-neutral-500">
              No hotspots yet — add a note on the left.
            </div>
          ) : (
            <ul className="space-y-2">
              {hotspots.map((h) => (
                <HotspotRow key={h.id} h={h} />
              ))}
            </ul>
          )}
        </section>
      </div>
    </main>
  );
}

function HotspotRow({ h }: { h: Hotspot }) {
  const icon = CATEGORY_ICONS[h.category ?? "other"] ?? CATEGORY_ICONS.other;
  const conf = Math.round(h.match_confidence * 100);
  return (
    <li className="rounded-xl border border-neutral-800 bg-neutral-900 p-3 flex items-start gap-3">
      <span className="text-2xl mt-0.5" aria-hidden>
        {icon}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 text-xs text-neutral-400">
          <span className="uppercase tracking-wider">
            {h.category ?? "other"}
          </span>
          {h.class_name && <span>· {h.class_name}</span>}
          <span>· {conf}%</span>
        </div>
        <div className="text-sm text-neutral-100 mt-1 break-words">
          {h.note_text}
        </div>
      </div>
    </li>
  );
}
