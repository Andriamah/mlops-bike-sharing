import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_model():
    config = load_config()
    raw_path = config['data']['raw_path']
    
    logging.info("Chargement des données...")
    df = pd.read_csv(raw_path)

    # --- PREPROCESSING ---
    # On garde les colonnes utiles pour la prédiction
    features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    target = 'cnt'
    
    X = df[features]
    y = df[target]

    # Séparation en Train (80%) et Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- ENTRAÎNEMENT ---
    logging.info("Entraînement du modèle Random Forest...")
    # On définit des paramètres (qu'on va tracker avec MLflow)
    n_estimators = 100
    max_depth = 10

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # --- ÉVALUATION ---
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    logging.info(f"Résultats : RMSE = {rmse:.2f}, MAE = {mae:.2f}")

    # --- MLFLOW TRACKING ---
    logging.info("Enregistrement de l'expérience dans MLflow...")
    
    # On définit le nom de l'expérience
    mlflow.set_experiment("Bike_Sharing_Prediction")
    
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        # 1. On log les paramètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # 2. On log les métriques
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # 3. On log le modèle (et on l'enregistre dans le Registry)
               # 3. On log le modèle (et on l'enregistre dans le Registry)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="BikeSharingModel",
            skops_trusted_types=["sklearn.tree._tree.Tree"] # <-- On autorise ce type
        )
        # 4. On ajoute un tag (On le mettra en "Staging" pour l'instant)
        client = mlflow.tracking.MlflowClient()
        # On récupère la dernière version du modèle
        latest_version = client.get_latest_versions("BikeSharingModel", stages=["None"])[0].version
        client.set_model_version_tag(
            name="BikeSharingModel",
            version=latest_version,
            key="status",
            value="staging"  # Plus tard, on le passera en 'production'
        )
        
    logging.info(" Modèle entraîné, évalué et enregistré dans MLflow avec succès !")

if __name__ == "__main__":
    train_model()