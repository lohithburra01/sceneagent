# SceneAgent Web

Next.js 14 frontend for SceneAgent: Gaussian splat viewer, hotspot overlay, AI chat, and realtor scrubber UI.

## Stack

- Next.js 14 (App Router) + TypeScript
- Tailwind CSS
- Three.js + `@mkkellogg/gaussian-splats-3d`
- Zustand (viewer store)
- TanStack Query (server state)
- `react-player` (scrubber video)
- `lucide-react` (icons)

## Development

```bash
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_URL` to point at the API (default `http://localhost:8000`).

## Pages

- `/` — homepage with links to demo listings
- `/listing/[slug]` — buyer view (splat + hotspots + chat)
- `/listing/[slug]/admin` — realtor scrubber UI

## Fallback for compressed .ply

If `@mkkellogg/gaussian-splats-3d` fails to load the InteriorGS `3dgs_compressed.ply`:

1. Pre-convert the `.ply` to `.ksplat` using the library's converter (see
   https://github.com/mkkellogg/GaussianSplats3D#tools) and serve the
   `.ksplat` URL instead. Point `splatUrl` to the `.ksplat`.
2. Alternative viewer: `antimatter15/splat` — drop-in WebGL viewer,
   swap the `<SplatViewer>` internals to mount that library.

The component detects format from file extension automatically.
