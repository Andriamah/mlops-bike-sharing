import subprocess
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_step(step_name, command):
    logging.info(f"--- DÉMARRAGE : {step_name} ---")
    start_time = time.time()
    
    # On exécute la commande
    result = subprocess.run(command, shell=True)
    
    duration = time.time() - start_time
    
    if result.returncode != 0:
        logging.error(f"❌ ÉCHEC de l'étape '{step_name}' après {duration:.2f}s. Arrêt du pipeline.")
        sys.exit(1) # On arrête tout le pipeline
    
    logging.info(f"✅ SUCCÈS : {step_name} terminé en {duration:.2f}s.")

def main():
    logging.info("🚀 Lancement du Pipeline MLOps complet...")
    
    # Étape 1 : Ingestion
    run_step("1. Ingestion des données", "python src/ingest.py")
    
    # Étape 2 : Validation (Portier)
    run_step("2. Validation de la qualité", "python src/validate.py")
    
    # Étape 3 : Entraînement
    run_step("3. Entraînement du modèle", "python src/train.py")
    
    logging.info("🎉 PIPELINE TERMINÉ AVEC SUCCÈS ! Toutes les étapes sont validées.")

if __name__ == "__main__":
    main()