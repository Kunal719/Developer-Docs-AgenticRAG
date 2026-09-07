CREATE TABLE rag_checkpoints (
    run_id TEXT PRIMARY KEY,
    checkpoint JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)