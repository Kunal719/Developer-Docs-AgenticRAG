CREATE TABLE rag_checkpoints (
    run_id TEXT PRIMARY KEY,
    checkpoint JSONB NOT NULL,
    status TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error TEXT,
    last_heartbeat_at TIMESTAMPTZ;
)