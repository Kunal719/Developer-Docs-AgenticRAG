import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.auth.dependencies import get_current_user
from checkpoint.models import RunStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_user():
    """
    Create a real database user so that runs can satisfy the
    rag_checkpoints.user_id foreign-key constraint.
    """
    with TestClient(app) as client:
        rag_service = app.state.rag_service
        user_store = rag_service.user_store

        google_subject = f"api-test-{time.time_ns()}"
        user_id = user_store.get_or_create_user(google_subject)

        yield user_id

        with rag_service.checkpoint_store._pool.connection() as conn:
            conn.execute(
                """
                DELETE FROM rag_checkpoints
                WHERE user_id = %s
                """,
                (user_id,),
            )

            conn.execute(
                """
                DELETE FROM users
                WHERE user_id = %s
                """,
                (user_id,),
            )


@pytest.fixture
def client(test_user):
    """
    Authenticate API requests as the test user.
    """

    app.dependency_overrides[get_current_user] = (
        lambda: test_user
    )

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def cleanup_runs(checkpoint_store, run_ids):
    """
    Remove test runs from the database.
    """

    if not run_ids:
        return

    with checkpoint_store._pool.connection() as conn:
        conn.execute(
            """
            DELETE FROM rag_checkpoints
            WHERE run_id = ANY(%s)
            """,
            (run_ids,),
        )


# ---------------------------------------------------------------------------
# POST /runs
# ---------------------------------------------------------------------------

def test_create_run_returns_streaming_response(
    client,
    test_user,
):
    """
    POST /runs should:

        authenticate
            ↓
        create the persistent run
            ↓
        start RAG streaming
    """

    rag_service = app.state.rag_service
    checkpoint_store = rag_service.checkpoint_store

    run_ids = []

    try:
        with patch.object(
            rag_service,
            "run",
            return_value=iter([]),
        ) as mock_run:

            response = client.post(
                "/runs",
                json={
                    "question": "How do I create a checkpoint?"
                },
            )

        assert response.status_code == 200

        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )

        mock_run.assert_called_once()

        call_args = mock_run.call_args.args

        assert call_args[0] == (
            "How do I create a checkpoint?"
        )

        assert call_args[1] == test_user

        run_id = call_args[2]
        run_ids.append(run_id)

        assert isinstance(run_id, str)
        assert run_id

        assert checkpoint_store.get_status(
            run_id,
            test_user,
        ) == RunStatus.RUNNING

    finally:
        cleanup_runs(
            checkpoint_store,
            run_ids,
        )


def test_create_run_passes_created_run_id_to_rag_service(
    client,
):
    """
    The run ID created by RAGService.create_run() must be the
    same run ID passed into RAGService.run().
    """

    rag_service = app.state.rag_service

    expected_run_id = f"api-run-{time.time_ns()}"

    with patch.object(
        rag_service,
        "create_run",
        return_value=expected_run_id,
    ) as mock_create_run, patch.object(
        rag_service,
        "run",
        return_value=iter([]),
    ) as mock_run:

        response = client.post(
            "/runs",
            json={
                "question": "Test run ID propagation"
            },
        )

    assert response.status_code == 200

    mock_create_run.assert_called_once_with(
        "Test run ID propagation",
        mock_create_run.call_args.args[1],
    )

    mock_run.assert_called_once_with(
        "Test run ID propagation",
        mock_create_run.call_args.args[1],
        expected_run_id,
    )


def test_create_run_quota_exceeded_returns_429(
    client,
):
    """
    If RAGService.create_run() rejects the request because
    the quota is exhausted:

        POST /runs
             ↓
        create_run()
             ↓
        ValueError
             ↓
        HTTP 429

    RAG execution must not start.
    """

    rag_service = app.state.rag_service

    with patch.object(
        rag_service,
        "create_run",
        side_effect=ValueError(
            "User has reached the maximum of 3 runs "
            "in the last 24 hours"
        ),
    ) as mock_create_run, patch.object(
        rag_service,
        "run",
    ) as mock_run:

        response = client.post(
            "/runs",
            json={
                "question": "Fourth question"
            },
        )

    assert response.status_code == 429

    data = response.json()

    assert "maximum of 3" in data["detail"]

    mock_create_run.assert_called_once()

    mock_run.assert_not_called()


def test_create_run_failure_prevents_rag_execution(
    client,
):
    """
    Any ValueError raised while creating the run must prevent
    the RAG pipeline from starting.
    """

    rag_service = app.state.rag_service

    with patch.object(
        rag_service,
        "create_run",
        side_effect=ValueError("Quota exceeded"),
    ), patch.object(
        rag_service,
        "run",
    ) as mock_run:

        response = client.post(
            "/runs",
            json={
                "question": "Should not execute"
            },
        )

    assert response.status_code == 429
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------

def test_list_runs_returns_list(
    client,
):
    """
    GET /runs should return the authenticated user's runs.
    """

    response = client.get("/runs")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)


def test_list_runs_uses_authenticated_user(
    client,
):
    """
    The current authenticated user_id must be passed to
    RAGService.list_runs().
    """

    rag_service = app.state.rag_service

    with patch.object(
        rag_service,
        "list_runs",
        return_value=[],
    ) as mock_list_runs:

        response = client.get("/runs")

    assert response.status_code == 200

    mock_list_runs.assert_called_once_with(
        mock_list_runs.call_args.args[0],
    )


# ---------------------------------------------------------------------------
# GET /runs/{run_id}
# ---------------------------------------------------------------------------

def test_get_run_status_for_owned_run(
    client,
    test_user,
):
    """
    An authenticated user can retrieve the status of their own run.
    """

    rag_service = app.state.rag_service
    checkpoint_store = rag_service.checkpoint_store

    run_id = rag_service.create_run(
        question="How does checkpointing work?",
        user_id=test_user,
    )

    try:
        response = client.get(
            f"/runs/{run_id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["run_id"] == run_id
        assert data["status"] == RunStatus.RUNNING.value

    finally:
        cleanup_runs(
            checkpoint_store,
            [run_id],
        )


def test_get_run_status_returns_404_for_unknown_run(
    client,
):
    """
    A nonexistent run must return 404.
    """

    run_id = f"nonexistent-{time.time_ns()}"

    response = client.get(
        f"/runs/{run_id}"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"


def test_get_run_status_cannot_access_another_users_run(
    client,
):
    """
    A user must not be able to access another user's run.
    """

    rag_service = app.state.rag_service
    checkpoint_store = rag_service.checkpoint_store
    other_user_id = (
        f"other-user-{time.time_ns()}"
    )

    other_user = rag_service.user_store.get_or_create_user(
        other_user_id
    )

    run_id = rag_service.create_run(
        question="Private question",
        user_id=other_user,
    )

    try:
        response = client.get(
            f"/runs/{run_id}"
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Run not found"

    finally:
        cleanup_runs(
            checkpoint_store,
            [run_id],
        )

        with checkpoint_store._pool.connection() as conn:
            conn.execute(
                """
                DELETE FROM users
                WHERE user_id = %s
                """,
                (other_user,),
            )


# ---------------------------------------------------------------------------
# POST /runs/{run_id}/resume
# ---------------------------------------------------------------------------

def test_resume_run_returns_streaming_response(
    client,
    test_user,
):
    """
    An existing run can reach the resume endpoint and the
    authenticated user_id/run_id are passed to RAGService.resume().
    """

    rag_service = app.state.rag_service
    checkpoint_store = rag_service.checkpoint_store

    run_id = rag_service.create_run(
        question="Resume this question",
        user_id=test_user,
    )

    try:
        checkpoint_store.update_status(
            run_id,
            RunStatus.INTERRUPTED,
        )

        with patch.object(
            rag_service,
            "resume",
            return_value=iter([]),
        ) as mock_resume:

            response = client.post(
                f"/runs/{run_id}/resume"
            )

        assert response.status_code == 200

        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )

        mock_resume.assert_called_once_with(
            run_id,
            test_user,
        )

    finally:
        cleanup_runs(
            checkpoint_store,
            [run_id],
        )


def test_resume_unknown_run_returns_404(
    client,
):
    """
    Resuming a nonexistent run must return 404.
    """

    run_id = f"unknown-resume-{time.time_ns()}"

    response = client.post(
        f"/runs/{run_id}/resume"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Run not found"


def test_resume_cannot_access_another_users_run(
    client,
):
    """
    A user cannot resume another user's run.
    """

    rag_service = app.state.rag_service
    checkpoint_store = rag_service.checkpoint_store

    other_user = rag_service.user_store.get_or_create_user(
        f"resume-other-{time.time_ns()}"
    )

    run_id = rag_service.create_run(
        question="Other user's run",
        user_id=other_user,
    )

    try:
        response = client.post(
            f"/runs/{run_id}/resume"
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Run not found"

    finally:
        cleanup_runs(
            checkpoint_store,
            [run_id],
        )

        with checkpoint_store._pool.connection() as conn:
            conn.execute(
                """
                DELETE FROM users
                WHERE user_id = %s
                """,
                (other_user,),
            )


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def test_runs_requires_authentication():
    """
    Without get_current_user being satisfied, protected run
    endpoints must return 401.
    """

    app.dependency_overrides.pop(
        get_current_user,
        None,
    )

    with TestClient(app) as client:
        response = client.get("/runs")

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"