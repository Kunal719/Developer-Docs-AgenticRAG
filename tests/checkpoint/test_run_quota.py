import time

from app.api.main import app
from checkpoint.models import RAGCheckpoint, RAGStage, RunStatus
from models.retrieval_plan import RetrievalPlan
from models.query_input import QueryInput

from fastapi.testclient import TestClient


def create_test_checkpoint(rag_service) -> RAGCheckpoint:
    query_input = QueryInput(
         question="Quota test question",
    )

    retrieval_plan = RetrievalPlan(
        retrieval_mode="direct",
        reasoning="Quota integration test",
    )

    return RAGCheckpoint(
        query_input=query_input,
        stage=RAGStage.ROUTED,
        generation_round=0,
        retrieval_plan=retrieval_plan,
    )


def test_user_is_limited_to_three_runs():
    """
    Verify:

        1st run → allowed
        2nd run → allowed
        3rd run → allowed
        4th run → rejected
    """
    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        google_subject = f"quota-test-{time.time_ns()}"
        user_id = user_store.get_or_create_user(google_subject)

        run_ids = []

        try:
            for i in range(3):
                run_id = f"quota-test-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

                assert checkpoint_store.get_status(
                    run_id,
                    user_id,
                ) == RunStatus.RUNNING

            # Fourth run must be rejected.
            fourth_run_id = f"quota-test-{time.time_ns()}-4"

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )
                assert False, "Expected quota limit to reject fourth run"

            except ValueError as exc:
                assert "maximum of 3" in str(exc)

        finally:
            with checkpoint_store._pool.connection() as conn:
                conn.execute(
                    """
                    DELETE FROM rag_checkpoints
                    WHERE run_id = ANY(%s)
                    """,
                    (run_ids,),
                )

                conn.execute(
                    """
                    DELETE FROM users
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )