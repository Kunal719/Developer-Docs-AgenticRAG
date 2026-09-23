import time

from fastapi.testclient import TestClient

from app.api.main import app
from checkpoint.database import create_connection_pool
from checkpoint.models import RAGCheckpoint, RunStatus
from checkpoint.postgres_store import PostgresCheckpointStore
from models.query_input import QueryInput
from models.rag_state import RAGStage

from models.retrieval_plan import (
    RetrievalPlan,
    MetadataFilters,
)

from checkpoint.models import EvidenceItemCheckpoint


RUN_ID = "test-resume-api"


# def make_checkpoint():
#     return RAGCheckpoint(
#         query_input=QueryInput(
#             question="How does LangGraph work?"
#         ),
#         stage=RAGStage.ROUTED,
#         generation_round=0,
#         retrieval_plan=RetrievalPlan(
#             retrieval_mode="direct",
#             reasoning="Test retrieval plan for resume testing.",
#             metadata_filters=MetadataFilters(),
#         ),
#     )

def make_checkpoint():
        return RAGCheckpoint(
        query_input=QueryInput(
            question="How does LangGraph work?"
        ),
        stage=RAGStage.INITIAL_EVIDENCE_READY,
        generation_round=0,

        retrieval_plan=RetrievalPlan(
            retrieval_mode="direct",
            reasoning="Test retrieval plan for resume testing.",
            metadata_filters=MetadataFilters(),
        ),

        reranker_query="How does LangGraph work?",

        initial_evidence=[
            EvidenceItemCheckpoint(
                corpus="documentation",
                chunk_id="6e0dd1a8e8e084976fa2de6995541b4de0b309ab869e7e75917661b977368fda",
                source="test",
                query="How does LangGraph work?",
            ),
            EvidenceItemCheckpoint(
                corpus="documentation",
                chunk_id="d20ed8ca3636c2b3b4607832e8bac29bc67e0b4f207de34e450ec3f122d95c24",
                source="test",
                query="How does LangGraph work?",
            ),
        ],

        evidence_pool=[
            EvidenceItemCheckpoint(
                corpus="documentation",
                chunk_id="6e0dd1a8e8e084976fa2de6995541b4de0b309ab869e7e75917661b977368fda",
                source="test",
                query="How does LangGraph work?",
            ),
            EvidenceItemCheckpoint(
                corpus="documentation",
                chunk_id="d20ed8ca3636c2b3b4607832e8bac29bc67e0b4f207de34e450ec3f122d95c24",
                source="test",
                query="How does LangGraph work?",
            ),
],
    )


def main():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    try:
        # Create a normal RUNNING checkpoint.
        store.save(
            RUN_ID,
            make_checkpoint(),
        )

        assert store.get_status(RUN_ID) == RunStatus.RUNNING

        # Simulate the worker disappearing by making
        # the heartbeat stale.
        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE rag_checkpoints
                SET last_heartbeat_at = NOW() - INTERVAL '60 seconds'
                WHERE run_id = %s
                """,
                (RUN_ID,),
            )

        print(f"Created RUNNING run: {RUN_ID}")
        print("Waiting for stale-run monitor...")

        # Start the real FastAPI lifespan.
        # This starts the stale-run monitor.
        with TestClient(app):
            deadline = time.time() + 25

            while time.time() < deadline:
                status = store.get_status(RUN_ID)

                print(f"Current status: {status.value}")

                if status == RunStatus.INTERRUPTED:
                    break

                time.sleep(0.5)

            else:
                raise AssertionError(
                    "Stale run was not marked INTERRUPTED "
                    "by the background monitor"
                )

        assert store.get_status(RUN_ID) == RunStatus.INTERRUPTED

        print()
        print("INTERRUPTED RUN CREATED")
        print(f"run_id = {RUN_ID}")
        print(f"status = {store.get_status(RUN_ID).value}")
        print()
        print(
            "Use this run_id for the resume API test."
        )

        # IMPORTANT:
        # Do NOT delete the checkpoint here.
        # We need it to remain in PostgreSQL so that the
        # /runs/{run_id}/resume endpoint can use it.

    except Exception:
        # Clean up only if creation/setup failed.
        with pool.connection() as conn:
            conn.execute(
                """
                DELETE FROM rag_checkpoints
                WHERE run_id = %s
                """,
                (RUN_ID,),
            )

        raise

    finally:
        pool.close()


if __name__ == "__main__":
    main()