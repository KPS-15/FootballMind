import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_api_match_info():
    response = client.get("/api/match/demo_match_01")
    assert response.status_code == 200
    assert response.json()["match_id"] == "demo_match_01"


def test_api_tracking():
    response = client.get("/api/tracking/demo_match_01?frame_index=10")
    assert response.status_code == 200
    data = response.json()
    assert "players" in data
    assert "ball" in data


def test_api_prediction():
    response = client.get("/api/prediction/demo_match_01?player_id=7&frame_index=10")
    assert response.status_code == 200
    data = response.json()
    assert "action_prediction" in data
    assert data["action_prediction"]["action"] in ["PASS", "DRIBBLE", "SHOT", "CROSS", "TACKLE", "CARRY", "HOLD", "CLEARANCE"]


def test_api_simulation():
    response = client.post("/api/simulation", json={
        "match_id": "demo_match_01",
        "frame_index": 10,
        "modified_player_id": 3,
        "new_x": 22.0,
        "new_y": 26.0
    })
    assert response.status_code == 200
    data = response.json()
    assert "danger_delta" in data


def test_api_tactical():
    response = client.get("/api/tactical/demo_match_01?frame_index=10")
    assert response.status_code == 200
    data = response.json()
    assert "defensive_collapse_index" in data
    assert "goal_probability" in data
    assert "goalkeeper_recommendation" in data


def test_api_recommendations():
    response = client.get("/api/recommendations/demo_match_01?player_id=7&frame_index=10")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


def test_api_explanation():
    response = client.get("/api/explanation/demo_match_01?player_id=7&frame_index=10")
    assert response.status_code == 200
    data = response.json()
    assert "prediction_explanation" in data
    assert "goal_explanation" in data


def test_api_evaluation():
    response = client.get("/api/evaluation/demo_match_01")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "expected_calibration_error_ece" in data

