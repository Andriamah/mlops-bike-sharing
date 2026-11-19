import pandas as pd
import yaml
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def validate_data():
    config = load_config()
    raw_path = config['data']['raw_path']
    min_temp = config['data']['valid']['min_temp']
    max_temp = config['data']['valid']['max_temp']
    min_count = config['data']['valid']['min_count']

    logging.info(f"Chargement des données depuis {raw_path}...")
    
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        logging.error("Le fichier n'existe pas. Avez-vous lancé l'ingestion ?")
        sys.exit(1)

    # 1. Vérification des valeurs manquantes
    if df.isnull().sum().sum() > 0:
        logging.error(" ÉCHEC : Le jeu de données contient des valeurs manquantes.")
        sys.exit(1)
    logging.info(" Pas de valeurs manquantes.")

    # 2. Vérification des bornes de température
    if df['temp'].min() < min_temp or df['temp'].max() > max_temp:
        logging.error(f" ÉCHEC : La température sort des bornes ({min_temp}, {max_temp}).")
        sys.exit(1)
    logging.info(" Les températures sont valides.")

    # 3. Vérification du nombre de locations (cnt)
    if df['cnt'].min() < min_count:
        logging.error(" ÉCHEC : Le nombre de locations est négatif (impossible).")
        sys.exit(1)
    logging.info(" Le nombre de locations est valide.")

    logging.info(" SUCCÈS : Toutes les vérifications de qualité sont passées. Prêt pour l'entraînement.")

if __name__ == "__main__":
    validate_data()