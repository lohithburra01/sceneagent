"""SceneAgent FastAPI application entrypoint."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .db import close_db_pool, init_db_pool

logger = logging.getLogger("sceneagent")

STATIC_ROOT = Path(os.environ.get("SCENEAGENT_DATA_DIR", "/app/data"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and tear down on shutdown."""
    try:
        await init_db_pool()
        logger.info("DB pool initialized")
    except Exception as exc:  # pragma: no cover - startup best-effort
        logger.warning("DB pool init failed: %s", exc)
    try:
        yield
    finally:
        try:
            await close_db_pool()
        except Exception as exc:  # pragma: no cover
            logger.warning("DB pool close failed: %s", exc)


app = FastAPI(title="SceneAgent API", version="0.1.0", lifespan=lifespan)

# Permissive CORS for local dev — the web container will call us directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


# Mount /static pointing at /app/data so the .ply + video.mp4 + views/ are served.
# We create the directory lazily if missing so Docker image build + tests stay green.
try:
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:  # pragma: no cover
    pass

if STATIC_ROOT.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


# ---- Router wiring ---------------------------------------------------------
# Routers are imported lazily at module import time so that a missing optional
# dep (e.g. torch) surfaces at import rather than at request time.
def _wire_routers() -> None:
    try:
        from .routes.scenes import router as scenes_router

        app.include_router(scenes_router)
    except Exception as exc:  # pragma: no cover
        logger.warning("scenes router not wired: %s", exc)

    try:
        from .routes.notes import router as notes_router

        app.include_router(notes_router)
    except Exception as exc:  # pragma: no cover
        logger.warning("notes router not wired: %s", exc)

    try:
        from .routes.hotspots import router as hotspots_router

        app.include_router(hotspots_router)
    except Exception as exc:  # pragma: no cover
        logger.warning("hotspots router not wired: %s", exc)

    try:
        from .routes.chat import router as chat_router

        app.include_router(chat_router)
    except Exception as exc:  # pragma: no cover
        logger.warning("chat router not wired: %s", exc)


_wire_routers()
