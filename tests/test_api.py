from fastapi.testclient import TestClient
from src.api import app

# On crée un faux client pour tester l'API sans la lancer
client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_valid_input():
    payload = {
        "season": 1, "yr": 0, "mnth": 1, "hr": 8,
        "holiday": 0, "weekday": 1, "workingday": 1,
        "weathersit": 1, "temp": 0.2, "atemp": 0.2,
        "hum": 0.5, "windspeed": 0.1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_input():
    # On envoie une donnée manquante (pas de 'hr') pour tester la validation
    payload = {
        "season": 1, "yr": 0, "mnth": 1,
        "holiday": 0, "weekday": 1, "workingday": 1,
        "weathersit": 1, "temp": 0.2, "atemp": 0.2,
        "hum": 0.5, "windspeed": 0.1
    }
    response = client.post("/predict", json=payload)
    # FastAPI doit renvoyer une erreur 422 (Unprocessable Entity)
    assert response.status_code == 422