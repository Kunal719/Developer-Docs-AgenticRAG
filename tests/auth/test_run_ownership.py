import base64
import json
import os
import time

from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner

from app.api.main import app
from checkpoint.models import RAGCheckpoint, RAGStage, RunStatus
from models.retrieval_plan import RetrievalPlan


def set_test_user(client: TestClient, user_id: str) -> None:
    """
    Create a signed session cookie containing the test user_id.

    This bypasses Google OAuth because this test is specifically
    testing run ownership, not authentication with Google.
    """

    session_data = json.dumps(
        {"user_id": user_id},
        separators=(",", ":"),
    ).encode("utf-8")

    session_data = base64.b64encode(session_data)

    signer = TimestampSigner(
        os.environ["SESSION_SECRET_KEY"]
    )

    signed_session = signer.sign(session_data).decode("utf-8")

    client.cookies.set(
        "session",
        signed_session,
    )


def test_user_cannot_access_another_users_run():
    """
    Verify run ownership:

        User A owns a run
              ↓
        User A can access the run
              ↓
        User B cannot:
            - see the run in /runs
            - get the run status
            - resume the run
    """

    run_id = f"ownership-test-{time.time_ns()}"

    google_subject_a = f"test-google-a-{time.time_ns()}"
    google_subject_b = f"test-google-b-{time.time_ns()}"

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_a_id = user_store.get_or_create_user(
            google_subject_a
        )
        user_b_id = user_store.get_or_create_user(
            google_subject_b
        )

        try:
            # ---------------------------------------------------------
            # 1. Create a checkpoint owned by User A
            # ---------------------------------------------------------

            query_input = rag_service._input_normalizer.normalize(
                "How does LangGraph persistence work?"
            )

            retrieval_plan = RetrievalPlan(
                retrieval_mode="direct",
                reasoning="Ownership integration test",
            )

            checkpoint = RAGCheckpoint(
                query_input=query_input,
                stage=RAGStage.ROUTED,
                generation_round=0,
                retrieval_plan=retrieval_plan,
            )

            checkpoint_store.save(
                run_id,
                user_a_id,
                checkpoint,
            )

            with checkpoint_store._pool.connection() as conn:
                row = conn.execute(
                    """
                    SELECT run_id, user_id, status
                    FROM rag_checkpoints
                    WHERE run_id = %s
                    """,
                    (run_id,),
                ).fetchone()

            print("CHECKPOINT AFTER SAVE:", row)

            print("STORED USER:", row[1])
            print("EXPECTED USER:", user_a_id)
            print("USER A ID:", user_a_id)
            print("USER B ID:", user_b_id)
            assert checkpoint_store.get_status(
                run_id,
                user_a_id,
            ) == RunStatus.RUNNING

            # ---------------------------------------------------------
            # 2. Authenticate as User A
            # ---------------------------------------------------------

            set_test_user(client, user_a_id)

            # User A should see their run.
            response = client.get("/runs")

            assert response.status_code == 200

            runs = response.json()

            assert any(
                run["run_id"] == run_id
                for run in runs
            )

            # User A should be able to access their run.
            response = client.get(f"/runs/{run_id}")

            assert response.status_code == 200

            data = response.json()

            assert data["run_id"] == run_id
            assert data["status"] == RunStatus.RUNNING.value

            # ---------------------------------------------------------
            # 3. Authenticate as User B
            # ---------------------------------------------------------

            set_test_user(client, user_b_id)

            # User B should NOT see User A's run.
            response = client.get("/runs")

            assert response.status_code == 200

            runs = response.json()

            assert not any(
                run["run_id"] == run_id
                for run in runs
            )

            # ---------------------------------------------------------
            # 4. User B should NOT access User A's run status.
            # ---------------------------------------------------------

            response = client.get(f"/runs/{run_id}")

            assert response.status_code == 404

            # ---------------------------------------------------------
            # 5. User B should NOT be able to resume User A's run.
            # ---------------------------------------------------------

            response = client.post(
                f"/runs/{run_id}/resume"
            )

            assert response.status_code == 404

        finally:
            # ---------------------------------------------------------
            # Cleanup checkpoint
            # ---------------------------------------------------------

            with checkpoint_store._pool.connection() as conn:
                conn.execute(
                    """
                    DELETE FROM rag_checkpoints
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )

                conn.execute(
                    """
                    DELETE FROM users
                    WHERE google_subject IN (%s, %s)
                    """,
                    (google_subject_a, google_subject_b),
                )