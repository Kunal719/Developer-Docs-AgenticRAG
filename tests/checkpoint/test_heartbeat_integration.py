import threading
import time

from fastapi.testclient import TestClient

from app.api.main import app
from checkpoint.database import create_connection_pool


def test_heartbeat_updates_during_run():
    pool = create_connection_pool()

    result = {}
    run_id = None

    try:
        # Capture runs that already exist so we can identify
        # the run created by this test.
        with pool.connection() as conn:
            existing_runs = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT run_id
                    FROM rag_checkpoints
                    """
                ).fetchall()
            }

        with TestClient(app) as client:

            def run_request():
                result["response"] = client.post(
                    "/runs",
                    json={
                        "question": "How do I create a checkpoint?"
                    },
                )

            request_thread = threading.Thread(target=run_request)
            request_thread.start()

            # Wait for this request to create its ROUTED checkpoint.
            deadline = time.time() + 30

            while time.time() < deadline:
                with pool.connection() as conn:
                    rows = conn.execute(
                        """
                        SELECT run_id, last_heartbeat_at
                        FROM rag_checkpoints
                        WHERE run_id <> ALL(%s)
                        ORDER BY updated_at DESC
                        """,
                        (list(existing_runs),),
                    ).fetchall()

                if rows:
                    run_id, heartbeat_before = rows[0]
                    break

                time.sleep(0.1)
            else:
                raise AssertionError(
                    "RAG run did not create a new checkpoint within 30 seconds"
                )

            assert heartbeat_before is not None

            # The heartbeat interval is 5 seconds.
            # Wait for the background heartbeat to update the timestamp.
            deadline = time.time() + 10

            while time.time() < deadline:
                with pool.connection() as conn:
                    row = conn.execute(
                        """
                        SELECT last_heartbeat_at
                        FROM rag_checkpoints
                        WHERE run_id = %s
                        """,
                        (run_id,),
                    ).fetchone()

                assert row is not None

                heartbeat_after = row[0]

                if heartbeat_after > heartbeat_before:
                    break

                time.sleep(0.1)
            else:
                raise AssertionError(
                    "Heartbeat did not update last_heartbeat_at "
                    "while the RAG run was executing"
                )

            assert heartbeat_after > heartbeat_before

            # Allow the API request to finish.
            request_thread.join(timeout=60)

            assert not request_thread.is_alive(), (
                "RAG request did not complete within 60 seconds"
            )

            assert "response" in result

            response = result["response"]

            assert response.status_code == 200

            data = response.json()

            assert data["run_id"] == run_id
            assert data["answer"] is not None

    finally:
        if run_id is not None:
            with pool.connection() as conn:
                conn.execute(
                    """
                    DELETE FROM rag_checkpoints
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )

        pool.close()