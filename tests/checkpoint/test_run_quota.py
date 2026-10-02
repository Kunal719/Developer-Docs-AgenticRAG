import time
from unittest.mock import MagicMock, patch

from app.api.main import app
from checkpoint.models import RAGCheckpoint, RAGStage, RunStatus
from models.retrieval_plan import RetrievalPlan
from models.query_input import QueryInput
from models.rag_state import RAGState
from checkpoint.serialization import create_checkpoint

from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


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


def create_test_checkpoint(rag_service):
    return RAGCheckpoint(
        query_input=QueryInput(
            question="Quota test question"
        ),
        stage=RAGStage.INITIALIZED,
        generation_round=0,
    )


def create_test_user(user_store):
    google_subject = f"quota-test-{time.time_ns()}"
    return user_store.get_or_create_user(google_subject)


def test_failed_run_does_not_count_toward_quota():
    """
    One FAILED run + three counted runs.

    The FAILED run must not consume a quota slot.
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_ids = []

        try:
            failed_run_id = f"quota-failed-{time.time_ns()}"

            checkpoint_store.create_run(
                failed_run_id,
                user_id,
                create_test_checkpoint(rag_service),
            )
            run_ids.append(failed_run_id)

            checkpoint_store.update_status(
                failed_run_id,
                RunStatus.FAILED,
                error="Test failure",
            )

            # Three counted runs.
            for i in range(3):
                run_id = f"quota-failed-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

            # Fourth counted run must be rejected.
            fourth_run_id = f"quota-failed-{time.time_ns()}-fourth"
            run_ids.append(fourth_run_id)

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )
                assert False, "Expected quota limit to reject fourth counted run"

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


def test_interrupted_run_counts_toward_quota():
    """
    INTERRUPTED runs consume a quota slot.
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_ids = []

        try:
            for i in range(3):
                run_id = f"quota-interrupted-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

                checkpoint_store.update_status(
                    run_id,
                    RunStatus.INTERRUPTED,
                )

            fourth_run_id = f"quota-interrupted-{time.time_ns()}-fourth"
            run_ids.append(fourth_run_id)

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )
                assert False, "Expected INTERRUPTED runs to count toward quota"

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


def test_completed_run_counts_toward_quota():
    """
    COMPLETED runs consume a quota slot.
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_ids = []

        try:
            for i in range(3):
                run_id = f"quota-completed-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

                checkpoint_store.update_status(
                    run_id,
                    RunStatus.COMPLETED,
                )

            fourth_run_id = f"quota-completed-{time.time_ns()}-fourth"
            run_ids.append(fourth_run_id)

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )
                assert False, "Expected COMPLETED runs to count toward quota"

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


def test_runs_older_than_24_hours_do_not_count():
    """
    Runs older than 24 hours do not consume the current quota.
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_ids = []

        try:
            old_run_id = f"quota-old-{time.time_ns()}"
            run_ids.append(old_run_id)

            checkpoint_store.create_run(
                old_run_id,
                user_id,
                create_test_checkpoint(rag_service),
            )

            # Move the run outside the rolling 24-hour window.
            with checkpoint_store._pool.connection() as conn:
                conn.execute(
                    """
                    UPDATE rag_checkpoints
                    SET created_at = NOW() - INTERVAL '25 hours'
                    WHERE run_id = %s
                    """,
                    (old_run_id,),
                )

            # Three current runs should be allowed.
            for i in range(3):
                run_id = f"quota-old-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

            # Fourth current run must be rejected.
            fourth_run_id = f"quota-old-{time.time_ns()}-fourth"
            run_ids.append(fourth_run_id)

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )
                assert False, "Expected current 3 runs to exhaust quota"

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


def test_users_have_independent_quotas():
    """
    One user's runs do not consume another user's quota.
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_a = create_test_user(user_store)
        user_b = create_test_user(user_store)

        run_ids = []

        try:
            # Exhaust User A's quota.
            for i in range(3):
                run_id = f"quota-user-a-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_a,
                    create_test_checkpoint(rag_service),
                )

            # User A's fourth run must fail.
            user_a_fourth = f"quota-user-a-{time.time_ns()}-fourth"
            run_ids.append(user_a_fourth)

            try:
                checkpoint_store.create_run(
                    user_a_fourth,
                    user_a,
                    create_test_checkpoint(rag_service),
                )
                assert False, "Expected User A quota to be exhausted"

            except ValueError as exc:
                assert "maximum of 3" in str(exc)

            # User B should still have a completely independent quota.
            user_b_run = f"quota-user-b-{time.time_ns()}"

            checkpoint_store.create_run(
                user_b_run,
                user_b,
                create_test_checkpoint(rag_service),
            )
            run_ids.append(user_b_run)

            assert checkpoint_store.get_status(
                user_b_run,
                user_b,
            ) == RunStatus.RUNNING

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
                    WHERE user_id IN (%s, %s)
                    """,
                    (user_a, user_b),
                )

def test_concurrent_run_creation_respects_quota():
    """
    Verify that concurrent run creation cannot bypass the per-user quota.

    Setup:
        2 existing counted runs

    Then:
        2 concurrent create_run() calls

    Expected:
        - exactly 1 succeeds
        - exactly 1 is rejected by the quota
        - total counted runs = 3
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_ids = []

        try:
            # Create two existing counted runs.
            for i in range(2):
                run_id = f"quota-concurrent-existing-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

            # Both workers wait here so they start create_run()
            # at approximately the same time.
            barrier = threading.Barrier(2)

            def create_concurrent_run(worker_id):
                run_id = (
                    f"quota-concurrent-{time.time_ns()}-{worker_id}"
                )

                barrier.wait()

                try:
                    checkpoint_store.create_run(
                        run_id,
                        user_id,
                        create_test_checkpoint(rag_service),
                    )

                    return {
                        "run_id": run_id,
                        "result": "success",
                    }

                except ValueError as exc:
                    return {
                        "run_id": run_id,
                        "result": "rejected",
                        "error": str(exc),
                    }

            # Run two create_run() calls concurrently.
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(create_concurrent_run, 1),
                    executor.submit(create_concurrent_run, 2),
                ]

                results = [
                    future.result()
                    for future in as_completed(futures)
                ]

            # Track the run IDs that were successfully created.
            for result in results:
                run_ids.append(result["run_id"])

            successful = [
                result
                for result in results
                if result["result"] == "success"
            ]

            rejected = [
                result
                for result in results
                if result["result"] == "rejected"
            ]

            # Exactly one request should succeed.
            assert len(successful) == 1

            # Exactly one request should be rejected.
            assert len(rejected) == 1

            assert "maximum of 3" in rejected[0]["error"]

            # Verify the database contains exactly 3 counted runs.
            with checkpoint_store._pool.connection() as conn:
                result = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM rag_checkpoints
                    WHERE user_id = %s
                      AND created_at >= NOW() - INTERVAL '24 hours'
                      AND status IN (%s, %s, %s)
                    """,
                    (
                        user_id,
                        RunStatus.RUNNING.value,
                        RunStatus.INTERRUPTED.value,
                        RunStatus.COMPLETED.value,
                    ),
                )

                count = result.fetchone()[0]

            assert count == 3

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

def test_initialized_run_is_normalized_before_resume():
    """
    An INTERRUPTED run at INITIALIZED should:

        INTERRUPTED
            ↓
        RUNNING
            ↓
        normalize raw question
            ↓
        pass normalized QueryInput to AgenticRAG
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_id = f"initialized-resume-{time.time_ns()}"

        try:
            raw_query_input = QueryInput(
                question="How does LangGraph checkpointing work?"
            )

            state = RAGState(
                query_input=raw_query_input,
                stage=RAGStage.INITIALIZED,
            )

            checkpoint_store.create_run(
                run_id,
                user_id,
                create_checkpoint(state),
            )

            checkpoint_store.update_status(
                run_id,
                RunStatus.INTERRUPTED,
            )

            normalized_query_input = QueryInput(
                question="How does LangGraph checkpointing work?",
                code_snippet="normalized code",
                error="normalized error",
            )

            rag_service._input_normalizer.normalize = MagicMock(
                return_value=normalized_query_input
            )

            with patch.object(
                rag_service._agentic_rag,
                "run",
                return_value=MagicMock(),
            ) as mock_agentic_run:

                events = list(
                    rag_service.resume(
                        run_id,
                        user_id,
                    )
                )

            # Normalization happened using the raw question.
            rag_service._input_normalizer.normalize.assert_called_once_with(
                "How does LangGraph checkpointing work?"
            )

            # AgenticRAG received the normalized QueryInput.
            mock_agentic_run.assert_called_once()

            call_kwargs = mock_agentic_run.call_args.kwargs

            assert call_kwargs["query_input"] == normalized_query_input
            assert call_kwargs["run_id"] == run_id
            assert call_kwargs["user_id"] == user_id
            assert call_kwargs["resume"] is True

            # No error event should have been emitted.
            error_events = [
                event
                for event in events
                if event.type == "error"
            ]

            assert len(error_events) == 0

        finally:
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
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )

def test_initialized_resume_normalization_failure_marks_run_failed():
    """
    If normalization fails while resuming an INITIALIZED run:

        INTERRUPTED
            ↓
        RUNNING
            ↓
        normalize()
            ↓
        exception
            ↓
        FAILED
            ↓
        error event
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_id = f"initialized-resume-failure-{time.time_ns()}"

        try:
            raw_query_input = QueryInput(
                question="Question that will fail normalization"
            )

            state = RAGState(
                query_input=raw_query_input,
                stage=RAGStage.INITIALIZED,
            )

            checkpoint_store.create_run(
                run_id,
                user_id,
                create_checkpoint(state),
            )

            checkpoint_store.update_status(
                run_id,
                RunStatus.INTERRUPTED,
            )

            rag_service._input_normalizer.normalize = MagicMock(
                side_effect=RuntimeError("Normalization failed")
            )

            events = list(
                rag_service.resume(
                    run_id,
                    user_id,
                )
            )

            rag_service._input_normalizer.normalize.assert_called_once_with(
                "Question that will fail normalization"
            )

            error_events = [
                event
                for event in events
                if event.type == "error"
            ]

            assert len(error_events) == 1
            assert error_events[0].run_id == run_id
            assert error_events[0].data == "Normalization failed"

            assert checkpoint_store.get_status(
                run_id,
                user_id,
            ) == RunStatus.FAILED

        finally:
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
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )