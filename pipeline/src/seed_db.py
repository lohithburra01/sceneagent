"""Load our object_inventory.json into the SceneAgent Postgres.

Also computes a CLIP text embedding per object (one description string) for the
note matcher. Falls back to InteriorGS ground-truth labels.json when our
pipeline did not produce an inventory (see README for why).
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import numpy as np

INV_PATH = Path("pipeline/output/object_inventory.json")
GT_FALLBACK_PATH = Path("data/scene/demo/labels.json")
SCENE_SLUG = os.environ.get("SCENE_ID", "demo")
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://sceneagent:sceneagent@localhost:5432/sceneagent",
)


def get_inventory() -> tuple[list[dict], str]:
    if INV_PATH.exists():
        inv = json.loads(INV_PATH.read_text())
        if len(inv) >= 10:
            return inv, "ours"
    # GT fallback — InteriorGS labels.json is a flat list of
    # {ins_id, label, bounding_box:[8 xyz-corners]}.
    raw = json.loads(GT_FALLBACK_PATH.read_text())
    inv = []
    for obj in raw:
        bb = obj.get("bounding_box")
        if not bb:
            continue
        a = np.asarray([[p["x"], p["y"], p["z"]] for p in bb])
        inv.append({
            "instance_id": int(obj["ins_id"]) if str(obj["ins_id"]).isdigit() else hash(obj["ins_id"]) & 0x7fffffff,
            "class_name": str(obj["label"]).lower().replace(" ", "_").replace("-", "_"),
            "bbox_min": a.min(0).tolist(),
            "bbox_max": a.max(0).tolist(),
            "centroid": ((a.min(0) + a.max(0)) / 2).tolist(),
        })
    return inv, "ground_truth_fallback"


def compute_embeddings(inv: list[dict]) -> list[list[float]]:
    import torch
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    prompts = [f"a {o['class_name'].replace('_', ' ')}" for o in inv]
    with torch.no_grad():
        emb = model.encode_text(tokenizer(prompts))
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.tolist()


async def seed():
    import asyncpg
    from pgvector.asyncpg import register_vector

    inv, source = get_inventory()
    print(f"seeding {len(inv)} objects from source={source}")
    embs = compute_embeddings(inv)

    async def _init(con):
        await register_vector(con)

    pool = await asyncpg.create_pool(DB_URL, init=_init)
    async with pool.acquire() as con:
        row = await con.fetchrow("SELECT id FROM scenes WHERE slug=$1", SCENE_SLUG)
        if row:
            scene_id = row["id"]
        else:
            scene_id = uuid.uuid4()
            trajectory = [
                {"timestamp": i * 1.0, "position": [i * 0.3, 0, 1.7], "yaw_deg": (i * 12) % 360}
                for i in range(30)
            ]
            await con.execute(
                """INSERT INTO scenes(id, slug, title, splat_url, camera_trajectory)
                   VALUES ($1, $2, $3, $4, $5::jsonb)""",
                scene_id, SCENE_SLUG, f"Demo Listing ({SCENE_SLUG})",
                f"/static/scene/{SCENE_SLUG}/3dgs_compressed.ply",
                json.dumps(trajectory),
            )
        await con.execute(
            "DELETE FROM scene_objects WHERE scene_id=$1 AND source=$2",
            scene_id, source,
        )
        for o, e in zip(inv, embs):
            try:
                await con.execute(
                    """INSERT INTO scene_objects
                       (scene_id, instance_id, class_name, centroid, bbox_min, bbox_max,
                        clip_embedding, source)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                       ON CONFLICT (scene_id, instance_id, source) DO NOTHING""",
                    scene_id, int(o["instance_id"]), o["class_name"],
                    list(o["centroid"]), list(o["bbox_min"]), list(o["bbox_max"]),
                    e, source,
                )
            except Exception as exc:
                print(f"skip instance {o['instance_id']}: {exc}")

        notes = json.loads(Path("data/scene/demo/demo_notes.json").read_text())
        await con.execute("DELETE FROM notes WHERE scene_id=$1", scene_id)
        for n in notes:
            await con.execute(
                "INSERT INTO notes(scene_id, text, video_timestamp) VALUES ($1,$2,$3)",
                scene_id, n["text"], n["video_timestamp"],
            )
    await pool.close()
    print("seed complete")


if __name__ == "__main__":
    asyncio.run(seed())
