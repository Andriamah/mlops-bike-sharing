from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging
import json
import time
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="API de Prédiction Bike Sharing", version="1.0")

# 1. Schéma de validation des données d'entrée
class BikeFeatures(BaseModel):
    season: int
    yr: int
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float

# 2. Schéma pour le feedback
class Feedback(BaseModel):
    prediction_id: str
    real_value: float

# 3. Chargement du modèle depuis MLflow
MODEL_URI = "models:/BikeSharingModel/latest"
try:
    logging.info(f"Chargement du modèle depuis {MODEL_URI}...")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logging.info("Modèle chargé avec succès !")
except Exception as e:
    logging.error(f"Erreur de chargement du modèle : {e}")
    model = None

# 4. Fonction pour sauvegarder les traces (La boîte noire)
def log_prediction(input_data, prediction, duration, error=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "prediction": prediction,
        "model_version": "latest",
        "duration_ms": round(duration * 1000, 2),
        "error": error
    }
    with open("predictions.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

@app.get("/health")
def health_check():
    status = "healthy" if model is not None else "unhealthy"
    return {"status": status}

@app.post("/predict")
def predict(data: BikeFeatures):
    start_time = time.time()
    
    if model is None:
        duration = time.time() - start_time
        log_prediction(data.model_dump(), None, duration, error="Model not loaded")
        raise HTTPException(status_code=503, detail="Le modèle n'est pas disponible.")
    
    # Utilisation de model_dump() pour Pydantic V2
    input_df = pd.DataFrame([data.model_dump()])
    
    try:
        prediction = model.predict(input_df)[0]
        prediction_val = round(float(prediction), 2)
        
        duration = time.time() - start_time
        log_prediction(data.model_dump(), prediction_val, duration)
        
        return {
            "prediction": prediction_val,
            "model_version": "latest",
            "status": "success"
        }
    except Exception as e:
        duration = time.time() - start_time
        log_prediction(data.model_dump(), None, duration, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def receive_feedback(data: Feedback):
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "prediction_id": data.prediction_id,
        "real_value": data.real_value
    }
    with open("feedbacks.log", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    
    return {"status": "success", "message": "Feedback enregistré. Merci !"}