import os
import zipfile
import requests
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def download_data():
    config = load_config()
    url = config['data']['url']
    raw_dir = "data/raw"
    
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "data.zip")
    
    logging.info(f"Téléchargement des données depuis {url}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            f.write(response.content)
        logging.info("Téléchargement terminé. Extraction en cours...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        logging.info("Extraction terminée. Données prêtes dans data/raw/.")
    else:
        logging.error(f"Erreur lors du téléchargement. Code HTTP: {response.status_code}")
        raise Exception("Échec du téléchargement des données.")

if __name__ == "__main__":
    download_data()