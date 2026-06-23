import pytest
from fastapi.testclient import TestClient
from app.main import app

# Use a fixture with a context manager to properly trigger lifespan events
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

def test_predict_empty_courses(client):
    r = client.post("/predict", json={"courses": []})
    assert r.status_code == 200
    data = r.json()
    assert "probabilities" in data and "model_version" in data
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3

def test_predict_with_courses(client):
    payload = {
        "courses": [{
            "code": "CS301",
            "term_work": 35.0, "exam_work": 50.0, "result": 85.0,
            "grade": "A", "points": 4.0,
        }]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    probs = r.json()["probabilities"]
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-3