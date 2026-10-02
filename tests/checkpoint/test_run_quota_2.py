import time
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.main import app
from checkpoint.models import RAGCheckpoint, RAGStage, RunStatus
from checkpoint.serialization import create_checkpoint
from models.query_input import QueryInput
from models.rag_state import RAGState
from models.retrieval_plan import RetrievalPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_checkpoint(rag_service) -> RAGCheckpoint:
    return RAGCheckpoint(
        query_input=QueryInput(
            question="Quota test question",
        ),
        stage=RAGStage.INITIALIZED,
        generation_round=0,
    )


def create_test_user(user_store):
    google_subject = f"quota-test-{time.time_ns()}"
    return user_store.get_or_create_user(google_subject)


def cleanup_runs_and_user(checkpoint_store, run_ids, user_id):
    with checkpoint_store._pool.connection() as conn:
        if run_ids:
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


# ---------------------------------------------------------------------------
# RAGService.create_run()
# ---------------------------------------------------------------------------

def test_service_create_run_creates_run_and_returns_run_id():
    """
    Verify the new RAGService.create_run() application-level entry point.

    It should:

        question + user_id
              ↓
        generate run_id
              ↓
        create QueryInput
              ↓
        create initial checkpoint
              ↓
        delegate to checkpoint_store.create_run()
              ↓
        return run_id
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)

        try:
            question = "How does LangGraph checkpointing work?"

            run_id = rag_service.create_run(
                question=question,
                user_id=user_id,
            )

            assert run_id is not None
            assert isinstance(run_id, str)
            assert len(run_id) > 0

            assert checkpoint_store.get_status(
                run_id,
                user_id,
            ) == RunStatus.RUNNING

            checkpoint = checkpoint_store.load(
                run_id,
                user_id,
            )

            assert checkpoint is not None
            assert checkpoint.query_input.question == question
            assert checkpoint.stage == RAGStage.INITIALIZED
            assert checkpoint.generation_round == 0

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                [run_id] if "run_id" in locals() else [],
                user_id,
            )


def test_service_create_run_generates_unique_run_ids():
    """
    Every call to RAGService.create_run() must generate a different run_id.
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_ids = []

        try:
            run_ids.append(
                rag_service.create_run(
                    question="First question",
                    user_id=user_id,
                )
            )

            run_ids.append(
                rag_service.create_run(
                    question="Second question",
                    user_id=user_id,
                )
            )

            assert run_ids[0] != run_ids[1]

            for run_id in run_ids:
                assert checkpoint_store.get_status(
                    run_id,
                    user_id,
                ) == RunStatus.RUNNING

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                run_ids,
                user_id,
            )


def test_service_create_run_passes_user_id_to_checkpoint_store():
    """
    RAGService.create_run() must create the run for the authenticated user
    supplied by the caller.
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_id = None

        try:
            run_id = rag_service.create_run(
                question="User ownership test",
                user_id=user_id,
            )

            # The owning user can access the run.
            checkpoint = checkpoint_store.load(
                run_id,
                user_id,
            )

            assert checkpoint is not None

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                [run_id] if run_id else [],
                user_id,
            )


def test_service_create_run_respects_store_quota():
    """
    RAGService.create_run() must propagate the quota error from
    PostgresCheckpointStore.create_run().

    Three runs:
        allowed

    Fourth run:
        rejected
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_ids = []

        try:
            for i in range(3):
                run_id = rag_service.create_run(
                    question=f"Quota question {i}",
                    user_id=user_id,
                )

                run_ids.append(run_id)

            with __import__("pytest").raises(
                ValueError,
                match="maximum of 3",
            ):
                rag_service.create_run(
                    question="Fourth question",
                    user_id=user_id,
                )

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                run_ids,
                user_id,
            )


# ---------------------------------------------------------------------------
# PostgreSQL quota behavior
# ---------------------------------------------------------------------------

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

        user_id = create_test_user(user_store)
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

            fourth_run_id = f"quota-test-{time.time_ns()}-fourth"

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

                assert False, (
                    "Expected quota limit to reject fourth run"
                )

            except ValueError as exc:
                assert "maximum of 3" in str(exc)

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                run_ids,
                user_id,
            )


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
            run_ids.append(failed_run_id)

            checkpoint_store.create_run(
                failed_run_id,
                user_id,
                create_test_checkpoint(rag_service),
            )

            checkpoint_store.update_status(
                failed_run_id,
                RunStatus.FAILED,
                error="Test failure",
            )

            for i in range(3):
                run_id = f"quota-failed-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

            fourth_run_id = f"quota-failed-{time.time_ns()}-fourth"

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

                assert False, (
                    "Expected quota limit to reject fourth "
                    "counted run"
                )

            except ValueError as exc:
                assert "maximum of 3" in str(exc)

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                run_ids,
                user_id,
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

            fourth_run_id = (
                f"quota-interrupted-{time.time_ns()}-fourth"
            )

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

                assert False, (
                    "Expected INTERRUPTED runs to count "
                    "toward quota"
                )

            except ValueError as exc:
                assert "maximum of 3" in str(exc)

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                run_ids,
                user_id,
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

            fourth_run_id = (
                f"quota-completed-{time.time_ns()}-fourth"
            )

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

                assert False, (
                    "Expected COMPLETED runs to count "
                    "toward quota"
                )

            except ValueError as exc:
                assert "maximum of 3" in str(exc)

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                run_ids,
                user_id,
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

            with checkpoint_store._pool.connection() as conn:
                conn.execute(
                    """
                    UPDATE rag_checkpoints
                    SET created_at = NOW() - INTERVAL '25 hours'
                    WHERE run_id = %s
                    """,
                    (old_run_id,),
                )

            for i in range(3):
                run_id = f"quota-old-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

            fourth_run_id = f"quota-old-{time.time_ns()}-fourth"

            try:
                checkpoint_store.create_run(
                    fourth_run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

                assert False, (
                    "Expected current 3 runs to exhaust quota"
                )

            except ValueError as exc:
                assert "maximum of 3" in str(exc)

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                run_ids,
                user_id,
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
            for i in range(3):
                run_id = f"quota-user-a-{time.time_ns()}-{i}"
                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_a,
                    create_test_checkpoint(rag_service),
                )

            user_a_fourth = (
                f"quota-user-a-{time.time_ns()}-fourth"
            )

            try:
                checkpoint_store.create_run(
                    user_a_fourth,
                    user_a,
                    create_test_checkpoint(rag_service),
                )

                assert False, (
                    "Expected User A quota to be exhausted"
                )

            except ValueError as exc:
                assert "maximum of 3" in str(exc)

            user_b_run = f"quota-user-b-{time.time_ns()}"
            run_ids.append(user_b_run)

            checkpoint_store.create_run(
                user_b_run,
                user_b,
                create_test_checkpoint(rag_service),
            )

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
    Verify that concurrent run creation cannot bypass
    the per-user quota.

    Setup:
        2 existing counted runs

    Then:
        2 concurrent create_run() calls

    Expected:
        - exactly 1 succeeds
        - exactly 1 is rejected
        - total counted runs = 3

    This test intentionally calls the checkpoint store directly
    because it verifies the PostgreSQL advisory-lock mechanism.
    """

    with TestClient(app) as client:
        rag_service = app.state.rag_service
        checkpoint_store = rag_service.checkpoint_store
        user_store = rag_service.user_store

        user_id = create_test_user(user_store)
        run_ids = []

        try:
            for i in range(2):
                run_id = (
                    f"quota-concurrent-existing-"
                    f"{time.time_ns()}-{i}"
                )

                run_ids.append(run_id)

                checkpoint_store.create_run(
                    run_id,
                    user_id,
                    create_test_checkpoint(rag_service),
                )

            barrier = threading.Barrier(2)

            def create_concurrent_run(worker_id):
                run_id = (
                    f"quota-concurrent-"
                    f"{time.time_ns()}-{worker_id}"
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

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(create_concurrent_run, 1),
                    executor.submit(create_concurrent_run, 2),
                ]

                results = [
                    future.result()
                    for future in as_completed(futures)
                ]

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

            assert len(successful) == 1
            assert len(rejected) == 1

            assert "maximum of 3" in rejected[0]["error"]

            with checkpoint_store._pool.connection() as conn:
                result = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM rag_checkpoints
                    WHERE user_id = %s
                      AND created_at >=
                          NOW() - INTERVAL '24 hours'
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
            cleanup_runs_and_user(
                checkpoint_store,
                run_ids,
                user_id,
            )


# ---------------------------------------------------------------------------
# Resume tests
# ---------------------------------------------------------------------------

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

            rag_service._input_normalizer.normalize.assert_called_once_with(
                "How does LangGraph checkpointing work?"
            )

            mock_agentic_run.assert_called_once()

            call_kwargs = mock_agentic_run.call_args.kwargs

            assert call_kwargs["query_input"] == normalized_query_input
            assert call_kwargs["run_id"] == run_id
            assert call_kwargs["user_id"] == user_id
            assert call_kwargs["resume"] is True

            error_events = [
                event
                for event in events
                if event.type == "error"
            ]

            assert len(error_events) == 0

        finally:
            cleanup_runs_and_user(
                checkpoint_store,
                [run_id],
                user_id,
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
            cleanup_runs_and_user(
                checkpoint_store,
                [run_id],
                user_id,
            )