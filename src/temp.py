import pandas as pd

# On charge le fichier
df = pd.read_csv("data/raw/hour.csv")

# 1. On affiche les 5 premières lignes pour voir à quoi ça ressemble
print("--- Les 5 premières lignes ---")
print(df.head())

# 2. On affiche la liste exacte des colonnes
print("\n--- Les colonnes disponibles ---")
print(df.columns.tolist())

# 3. On vérifie que 'cnt' est bien la somme de casual et registered
print("\n--- Vérification de la cible ---")
print(df[['casual', 'registered', 'cnt']].head())