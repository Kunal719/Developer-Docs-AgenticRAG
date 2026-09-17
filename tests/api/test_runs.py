import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_create_run(client):
    response = client.post(
        "/runs",
        json={"question": "How do I create a checkpoint?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["run_id"]
    assert data["answer"] is not None


def test_list_runs(client):
    response = client.get("/runs")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)


def test_get_run_status(client):
    create_response = client.post(
        "/runs",
        json={"question": "How do I create a checkpoint?"},
    )

    assert create_response.status_code == 200

    run_id = create_response.json()["run_id"]

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["run_id"] == run_id
    assert data["status"] == "completed"