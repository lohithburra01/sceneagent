"use client";

import { useCallback, useRef, useState } from "react";
import clsx from "clsx";
import { Send, Bot, User, Wrench, X } from "lucide-react";
import { chat, ToolCall, Vec3 } from "@/lib/api";
import { useViewerStore, TourStop } from "@/stores/viewer";

type Role = "user" | "assistant" | "tool";

interface Message {
  id: string;
  role: Role;
  content: string;
  tool_name?: string;
}

interface ChatOverlayProps {
  slug: string;
}

export default function ChatOverlay({ slug }: ChatOverlayProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hi — ask me anything about this place. Try: \"List any issues\", \"What's included?\", \"How tall are the ceilings?\", or \"Give me a tour\".",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const listRef = useRef<HTMLDivElement | null>(null);

  const flyTo = useViewerStore((s) => s.flyTo);
  const setTour = useViewerStore((s) => s.setTour);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (listRef.current) {
        listRef.current.scrollTop = listRef.current.scrollHeight;
      }
    });
  }, []);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || busy) return;
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setBusy(true);
    scrollToBottom();

    try {
      const resp = await chat(slug, text);

      const toolMsgs: Message[] = (resp.tool_calls || []).map((tc) => ({
        id: crypto.randomUUID(),
        role: "tool" as Role,
        content: summarizeTool(tc),
        tool_name: tc.name,
      }));

      setMessages((m) => [
        ...m,
        ...toolMsgs,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: resp.response,
        },
      ]);

      // Camera side-effects: react to certain tool calls by flying the camera.
      applyCameraSideEffects(resp.tool_calls || [], flyTo, setTour);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `(error reaching agent: ${
            err instanceof Error ? err.message : String(err)
          })`,
        },
      ]);
    } finally {
      setBusy(false);
      scrollToBottom();
    }
  }, [input, busy, slug, scrollToBottom, flyTo, setTour]);

  if (!open) {
    return (
      <button
        type="button"
        aria-label="Open AI concierge"
        onClick={() => setOpen(true)}
        className={clsx(
          "fixed bottom-6 right-[360px] z-30",
          "h-11 w-11 rounded-full",
          "bg-neutral-900/90 border border-neutral-700 backdrop-blur",
          "flex items-center justify-center",
          "hover:bg-neutral-800 transition-colors"
        )}
      >
        <Bot className="w-4 h-4 text-amber-300" />
      </button>
    );
  }

  return (
    <div
      className={clsx(
        "fixed bottom-4 right-[360px] z-30",
        "w-[384px] max-w-[calc(100vw-380px)]",
        "h-[520px] max-h-[calc(100vh-2rem)]",
        "flex flex-col",
        "rounded-xl overflow-hidden",
        "bg-neutral-950/95 text-neutral-100 backdrop-blur",
        "border border-neutral-800 shadow-2xl"
      )}
    >
      <header className="px-4 py-3 border-b border-neutral-800 flex items-center gap-2">
        <Bot className="w-4 h-4 text-amber-300" />
        <div className="text-sm font-medium">Concierge</div>
        <button
          type="button"
          onClick={() => setOpen(false)}
          aria-label="Close"
          className="ml-auto text-neutral-500 hover:text-neutral-200"
        >
          <X className="w-4 h-4" />
        </button>
      </header>

      <div
        ref={listRef}
        className="flex-1 overflow-y-auto px-3 py-3 space-y-3"
      >
        {messages.map((m) => (
          <MessageRow key={m.id} message={m} />
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-xs text-neutral-500">
            <Bot className="w-3 h-3 animate-pulse" />
            thinking…
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
        className="p-2 border-t border-neutral-800 flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this listing…"
          disabled={busy}
          className={clsx(
            "flex-1 rounded-lg bg-neutral-800 text-sm",
            "px-3 py-2 outline-none",
            "placeholder:text-neutral-500",
            "focus:ring-1 focus:ring-emerald-500"
          )}
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className={clsx(
            "rounded-lg px-3 py-2 text-sm font-medium",
            "bg-emerald-500 hover:bg-emerald-400 text-black",
            "disabled:opacity-40 disabled:cursor-not-allowed",
            "flex items-center gap-1"
          )}
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}

function MessageRow({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="flex items-start gap-2 justify-end">
        <div className="rounded-lg bg-emerald-500 text-black px-3 py-2 text-sm max-w-[80%]">
          {message.content}
        </div>
        <User className="w-4 h-4 mt-1 text-neutral-400 shrink-0" />
      </div>
    );
  }
  if (message.role === "tool") {
    return (
      <div className="flex items-start gap-2 text-xs text-neutral-400">
        <Wrench className="w-3 h-3 mt-1 shrink-0" />
        <div className="italic">
          {message.tool_name ? <b>{message.tool_name}</b> : null}
          {message.tool_name ? ": " : ""}
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex items-start gap-2">
      <Bot className="w-4 h-4 mt-1 text-emerald-400 shrink-0" />
      <div className="rounded-lg bg-neutral-800 px-3 py-2 text-sm max-w-[85%] whitespace-pre-wrap">
        {message.content}
      </div>
    </div>
  );
}

function summarizeTool(tc: ToolCall): string {
  const args = tc.args
    ? Object.entries(tc.args)
        .map(([k, v]) => `${k}=${truncate(JSON.stringify(v), 40)}`)
        .join(", ")
    : "";
  return args;
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function applyCameraSideEffects(
  calls: ToolCall[],
  flyTo: (p: Vec3 | null) => void,
  setTour: (s: TourStop[] | null) => void
) {
  for (const tc of calls) {
    if (tc.name === "find_by_description") {
      const result = asArray(tc.result);
      const first = result[0] as Record<string, unknown> | undefined;
      const centroid = extractCentroid(first);
      if (centroid) {
        flyTo(centroid);
        return; // one side-effect per turn
      }
    }
    if (tc.name === "plan_tour") {
      const result = asArray(tc.result);
      const stops: TourStop[] = result
        .map((s) => toTourStop(s as Record<string, unknown>))
        .filter((s): s is TourStop => !!s);
      if (stops.length > 0) {
        setTour(stops);
        flyTo(stops[0].position);
        return;
      }
    }
  }
}

function asArray(v: unknown): unknown[] {
  if (Array.isArray(v)) return v;
  if (v && typeof v === "object") {
    const obj = v as Record<string, unknown>;
    if (Array.isArray(obj.results)) return obj.results;
    if (Array.isArray(obj.objects)) return obj.objects;
    if (Array.isArray(obj.stops)) return obj.stops;
  }
  return [];
}

function extractCentroid(obj: Record<string, unknown> | undefined): Vec3 | null {
  if (!obj) return null;
  const c = obj.centroid ?? obj.position;
  return toVec3(c);
}

function toTourStop(obj: Record<string, unknown> | undefined): TourStop | null {
  if (!obj) return null;
  const pos = toVec3(obj.position ?? obj.pose);
  if (!pos) return null;
  const dwell = typeof obj.dwell_seconds === "number" ? obj.dwell_seconds : 3;
  const narration =
    typeof obj.narration_hint === "string"
      ? obj.narration_hint
      : typeof obj.narration === "string"
      ? obj.narration
      : undefined;
  const highlight =
    typeof obj.highlight_hotspot_id === "string"
      ? obj.highlight_hotspot_id
      : undefined;
  return {
    position: pos,
    dwell_seconds: dwell,
    narration_hint: narration,
    highlight_hotspot_id: highlight,
  };
}

function toVec3(v: unknown): Vec3 | null {
  if (
    Array.isArray(v) &&
    v.length === 3 &&
    v.every((n) => typeof n === "number")
  ) {
    return [v[0] as number, v[1] as number, v[2] as number];
  }
  return null;
}
