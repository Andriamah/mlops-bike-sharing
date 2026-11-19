from locust import HttpUser, task, between
import json

class BikePredictorUser(HttpUser):
    # L'utilisateur attend entre 1 et 2 secondes entre chaque requête
    wait_time = between(1, 2)

    @task(3) # 3 fois plus de requêtes de prédiction que de health check
    def test_predict(self):
        payload = {
            "season": 1, "yr": 0, "mnth": 1, "hr": 8,
            "holiday": 0, "weekday": 1, "workingday": 1,
            "weathersit": 1, "temp": 0.2, "atemp": 0.2,
            "hum": 0.5, "windspeed": 0.1
        }
        headers = {"Content-Type": "application/json"}
        self.client.post("/predict", data=json.dumps(payload), headers=headers)

    @task(1)
    def test_health(self):
        self.client.get("/health")