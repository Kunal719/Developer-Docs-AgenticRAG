import time

from fastapi.testclient import TestClient

from app.api.main import app
from checkpoint.models import RAGCheckpoint, RAGStage, RunStatus
from models.retrieval_plan import RetrievalPlan


def test_interrupted_run_can_be_resumed():
    """
    Verify:

        RUNNING
          ↓
        stale heartbeat
          ↓
        INTERRUPTED
          ↓
        POST /runs/{run_id}/resume
          ↓
        RUNNING
          ↓
        COMPLETED
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store

        # ---------------------------------------------------------
        # 1. Create a valid ROUTED checkpoint directly.
        #
        # We don't call the real query router here because that would
        # unnecessarily invoke the LLM. The purpose of this test is
        # resume/recovery, not router behavior.
        # ---------------------------------------------------------

        question = "How does LangGraph persistence work?"
        query_input = rag_service._input_normalizer.normalize(question)

        run_id = f"resume-test-{time.time_ns()}"

        retrieval_plan = RetrievalPlan(
            retrieval_mode="direct",
            reasoning="Integration test retrieval plan",
        )

        checkpoint = RAGCheckpoint(
            query_input=query_input,
            stage=RAGStage.ROUTED,
            generation_round=0,
            retrieval_plan=retrieval_plan,
        )

        checkpoint_store.save(run_id, checkpoint)

        assert checkpoint_store.get_status(run_id) == RunStatus.RUNNING

        # ---------------------------------------------------------
        # 2. Simulate worker disappearance by making the heartbeat
        #    stale.
        # ---------------------------------------------------------

        with checkpoint_store._pool.connection() as conn:
            conn.execute(
                """
                UPDATE rag_checkpoints
                SET last_heartbeat_at = NOW() - INTERVAL '60 seconds'
                WHERE run_id = %s
                """,
                (run_id,),
            )

        # ---------------------------------------------------------
        # 3. Stale-run detection should mark the run INTERRUPTED.
        # ---------------------------------------------------------

        interrupted_run_ids = checkpoint_store.interrupt_stale_runs(
            stale_after_seconds=30
        )

        assert run_id in interrupted_run_ids
        assert checkpoint_store.get_status(run_id) == RunStatus.INTERRUPTED

        # ---------------------------------------------------------
        # 4. Resume through the actual API.
        # ---------------------------------------------------------

        response = client.post(
            f"/runs/{run_id}/resume"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["run_id"] == run_id
        assert data["answer"] is not None

        # ---------------------------------------------------------
        # 5. Verify the resumed run completed.
        # ---------------------------------------------------------

        assert checkpoint_store.get_status(run_id) == RunStatus.COMPLETED

        final_checkpoint = checkpoint_store.load(run_id)

        assert final_checkpoint.stage == RAGStage.COMPLETED
        assert final_checkpoint.generated_answer is not None

        # ---------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------

        with checkpoint_store._pool.connection() as conn:
            conn.execute(
                """
                DELETE FROM rag_checkpoints
                WHERE run_id = %s
                """,
                (run_id,),
            )