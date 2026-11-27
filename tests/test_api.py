
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction(client):
    payload = {
        "loom_age_yr": 15,
        "thread_count": 120,
        "power_cuts_min": 30,
        "weaver_exp_yr": 10,
        "humidity_pct": 60,
        "shift_hours": 8,
        "fabric": "cotton",
        "shed": "Shed 1"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)
    assert response.json()["unit"] == "metres_per_shift"


def test_loom_age_validation(client):
    payload = {
        "loom_age_yr": 60,
        "thread_count": 120,
        "power_cuts_min": 30,
        "weaver_exp_yr": 10,
        "humidity_pct": 60,
        "shift_hours": 8,
        "fabric": "cotton",
        "shed": "Shed 1"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_thread_count_validation(client):
    payload = {
        "loom_age_yr": 15,
        "thread_count": 10,
        "power_cuts_min": 30,
        "weaver_exp_yr": 10,
        "humidity_pct": 60,
        "shift_hours": 8,
        "fabric": "cotton",
        "shed": "Shed 1"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
