from checkpoint.models import RAGCheckpoint, RunStatus

from typing import Protocol


class CheckpointStore(Protocol):
    def save(self, run_id: str, checkpoint: RAGCheckpoint):
        ...

    def load(self, run_id: str) -> RAGCheckpoint:
        ...

    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        error: str | None = None
    ) -> None:
        ...

    def get_status(self, run_id: str) -> RunStatus:
        ...

    def list_runs(self) -> list[dict]:
        ...

    def heartbeat(self, run_id: str) -> None:
        ...

    def interrupt_stale_runs(self, stale_after_seconds: int) -> list[str]:
        ...

    def resume_run(self, run_id: str) -> None:
        ...
        
class InMemoryCheckpointStore:
    def __init__(self):
        self._checkpoints: dict[str, RAGCheckpoint] = {}

    def save(self, run_id: str, checkpoint: RAGCheckpoint) -> None:
        self._checkpoints[run_id] = checkpoint

    def load(self, run_id: str) -> RAGCheckpoint:
        return self._checkpoints[run_id]

    def delete(self, run_id: str) -> None:
        del self._checkpoints[run_id]