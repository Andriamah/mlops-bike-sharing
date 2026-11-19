# On part d'une image Python officielle légère
FROM python:3.11-slim

# On définit le dossier de travail
WORKDIR /app

# On copie les dépendances et on les installe
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie tout le code du projet
COPY . .

# On expose le port 8000
EXPOSE 8000

# La commande pour lancer l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]