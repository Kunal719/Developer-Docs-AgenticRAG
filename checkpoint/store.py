from checkpoint.models import RAGCheckpoint

from typing import Protocol

class CheckpointStore(Protocol):
    def save(self, run_id: str, checkpoint: RAGCheckpoint):
        ...

    def load(self, run_id: str) -> RAGCheckpoint:
        ...

    def delete(self, run_id: str):
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