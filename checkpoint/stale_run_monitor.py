import threading
from checkpoint.store import CheckpointStore

from config import STALE_HEARTBEAT_SECONDS, STALE_RUN_MONITOR_INTERVAL_SECONDS


def stale_run_monitor(checkpoint_store: CheckpointStore, stop_event: threading.Event) -> None:
    while not stop_event.wait(STALE_RUN_MONITOR_INTERVAL_SECONDS):
        try:
            interrupted_run_ids = checkpoint_store.interrupt_stale_runs(
                stale_after_seconds=STALE_HEARTBEAT_SECONDS
            )

            if interrupted_run_ids:
                print(f"Marked stale runs as interrupted: {interrupted_run_ids}")

        except Exception as e:
            # The monitor should not die because of a temporary database or other runtime error.
            print(f"Stale run monitor error: {e}")