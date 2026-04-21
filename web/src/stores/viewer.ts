"use client";

import { create } from "zustand";

export type Vec3 = [number, number, number];

export interface TourStop {
  position: Vec3;
  dwell_seconds: number;
  narration_hint?: string;
  highlight_hotspot_id?: string;
}

interface ViewerState {
  highlightedHotspotIds: string[];
  flyToPosition: Vec3 | null;
  tourStops: TourStop[] | null;
  activeObjectId: string | null;
  hoveredObjectId: string | null;

  setHighlights: (ids: string[]) => void;
  flyTo: (position: Vec3 | null) => void;
  setTour: (stops: TourStop[] | null) => void;
  setActiveObject: (id: string | null) => void;
  setHoveredObject: (id: string | null) => void;
  clear: () => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  highlightedHotspotIds: [],
  flyToPosition: null,
  tourStops: null,
  activeObjectId: null,
  hoveredObjectId: null,

  setHighlights: (ids) => set({ highlightedHotspotIds: ids }),
  flyTo: (position) => set({ flyToPosition: position }),
  setTour: (stops) => set({ tourStops: stops }),
  setActiveObject: (id) => set({ activeObjectId: id }),
  setHoveredObject: (id) => set({ hoveredObjectId: id }),
  clear: () =>
    set({
      highlightedHotspotIds: [],
      flyToPosition: null,
      tourStops: null,
      activeObjectId: null,
      hoveredObjectId: null,
    }),
}));
