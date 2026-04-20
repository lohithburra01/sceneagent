"""Postgres connection pool with pgvector codec registration."""

from __future__ import annotations

import os
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

_pool: Optional[asyncpg.Pool] = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register pgvector codec on each new asyncpg connection."""
    await register_vector(conn)


async def init_db_pool() -> asyncpg.Pool:
    """Create the global connection pool. Idempotent."""
    global _pool
    if _pool is not None:
        return _pool

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    # asyncpg does not accept the postgresql+asyncpg:// prefix some tools use.
    if dsn.startswith("postgresql+asyncpg://"):
        dsn = "postgresql://" + dsn[len("postgresql+asyncpg://") :]

    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=10,
        init=_init_connection,
    )
    return _pool


async def close_db_pool() -> None:
    """Close the global pool if it exists."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    """Return the already-initialized pool. Raises if not initialized."""
    if _pool is None:
        raise RuntimeError(
            "DB pool not initialized. Ensure init_db_pool() ran in app lifespan."
        )
    return _pool
