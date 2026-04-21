/**
 * Thin REST client for the SceneAgent FastAPI backend.
 *
 * Base URL comes from NEXT_PUBLIC_API_URL (default: http://localhost:8000).
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Vec3 = [number, number, number];

export interface Scene {
  id: string;
  slug: string;
  title: string;
  address?: string | null;
  splat_url: string;
  camera_trajectory: Array<{
    timestamp: number;
    position: Vec3;
    yaw_deg?: number;
  }>;
  processed_at: string;
}

export interface Hotspot {
  id: string;
  note_id: string;
  object_id: string | null;
  match_confidence: number;
  position: Vec3;
  auto_accepted: boolean;
  note_text: string;
  category: string | null;
  class_name: string | null;
  bbox_min?: Vec3 | null;
  bbox_max?: Vec3 | null;
  centroid?: Vec3 | null;
}

export interface SceneMetrics {
  f1: number;
  precision: number;
  recall: number;
  tp: number;
  fp: number;
  fn: number;
  num_predicted: number;
  num_ground_truth: number;
  iou_threshold: number;
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  result?: unknown;
}

export interface ChatResponse {
  response: string;
  tool_calls: ToolCall[];
}

export interface CreateNoteResponse {
  note_id: string;
  category: string | null;
  hotspot: Hotspot | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `API ${init?.method || "GET"} ${path} failed: ${res.status} ${text}`
    );
  }
  return (await res.json()) as T;
}

export function getScene(slug: string): Promise<Scene> {
  return request<Scene>(`/scenes/${slug}`);
}

export function getHotspots(
  slug: string,
  category?: string
): Promise<Hotspot[]> {
  const q = category ? `?category=${encodeURIComponent(category)}` : "";
  return request<Hotspot[]>(`/scenes/${slug}/hotspots${q}`);
}

export function createNote(
  slug: string,
  text: string,
  ts: number
): Promise<CreateNoteResponse> {
  return request<CreateNoteResponse>(`/scenes/${slug}/notes`, {
    method: "POST",
    body: JSON.stringify({ text, video_timestamp: ts }),
  });
}

export function chat(slug: string, message: string): Promise<ChatResponse> {
  return request<ChatResponse>(`/scenes/${slug}/chat`, {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

export function seedMatch(slug: string): Promise<{ matched: number }> {
  return request<{ matched: number }>(`/scenes/${slug}/notes/seed-match`, {
    method: "POST",
  });
}

export function getMetrics(slug: string): Promise<SceneMetrics | null> {
  return request<SceneMetrics | null>(`/scenes/${slug}/metrics`).catch(
    () => null
  );
}

export function splatUrlFor(slug: string): string {
  return `${API_BASE}/static/scene/${slug}/3dgs_compressed.ply`;
}

export function videoUrlFor(slug: string): string {
  return `${API_BASE}/static/scene/${slug}/video.mp4`;
}
