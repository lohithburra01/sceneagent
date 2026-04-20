CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS scenes (
  id UUID PRIMARY KEY,
  slug TEXT UNIQUE NOT NULL,
  title TEXT NOT NULL,
  address TEXT,
  splat_url TEXT NOT NULL,
  camera_trajectory JSONB NOT NULL,
  processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS scene_objects (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scene_id UUID REFERENCES scenes(id) ON DELETE CASCADE,
  instance_id INT NOT NULL,
  class_name TEXT NOT NULL,
  room_label TEXT,
  centroid DOUBLE PRECISION[] NOT NULL,
  bbox_min DOUBLE PRECISION[] NOT NULL,
  bbox_max DOUBLE PRECISION[] NOT NULL,
  clip_embedding VECTOR(512) NOT NULL,
  source TEXT NOT NULL DEFAULT 'ours',
  UNIQUE (scene_id, instance_id, source)
);
CREATE INDEX IF NOT EXISTS idx_scene_objects_embedding
  ON scene_objects USING hnsw (clip_embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS notes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scene_id UUID REFERENCES scenes(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  video_timestamp DOUBLE PRECISION NOT NULL,
  category TEXT,
  category_confidence REAL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS hotspots (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  note_id UUID UNIQUE REFERENCES notes(id) ON DELETE CASCADE,
  object_id UUID REFERENCES scene_objects(id) ON DELETE SET NULL,
  match_confidence REAL NOT NULL,
  position DOUBLE PRECISION[] NOT NULL,
  auto_accepted BOOLEAN NOT NULL DEFAULT TRUE
);
