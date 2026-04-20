import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8 bg-neutral-950 text-neutral-100">
      <div className="max-w-2xl text-center space-y-6">
        <h1 className="text-5xl font-bold tracking-tight">SceneAgent</h1>
        <p className="text-lg text-neutral-400">
          Every listing is a 3D digital twin with an AI concierge that can
          actually see, walk buyers through the space, and answer questions.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-6">
          <Link
            href="/listing/demo"
            className="rounded-xl border border-neutral-800 bg-neutral-900 hover:bg-neutral-800 p-6 transition"
          >
            <div className="text-sm uppercase tracking-wider text-neutral-500 mb-1">
              Buyer view
            </div>
            <div className="text-xl font-semibold">/listing/demo</div>
            <div className="text-sm text-neutral-400 mt-2">
              3D splat viewer, hotspots, AI chat.
            </div>
          </Link>

          <Link
            href="/listing/demo/admin"
            className="rounded-xl border border-neutral-800 bg-neutral-900 hover:bg-neutral-800 p-6 transition"
          >
            <div className="text-sm uppercase tracking-wider text-neutral-500 mb-1">
              Realtor view
            </div>
            <div className="text-xl font-semibold">/listing/demo/admin</div>
            <div className="text-sm text-neutral-400 mt-2">
              Video scrubber to timestamp notes.
            </div>
          </Link>
        </div>

        <p className="text-xs text-neutral-600 pt-8">
          API base: {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
        </p>
      </div>
    </main>
  );
}
