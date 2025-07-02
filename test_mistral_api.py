import os
import requests

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("La variable d'environnement MISTRAL_API_KEY n'est pas définie.")
    exit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "mistral-embed",
    "input": ["test"]
}
try:
    response = requests.post("https://api.mistral.ai/v1/embeddings", headers=headers, json=data)
    print(f"Code retour: {response.status_code}")
    print(f"Réponse: {response.text}")
    if response.status_code == 401:
        print("Clé API invalide ou non autorisée.")
    elif response.status_code == 200:
        print("Clé API valide !")
    else:
        print("Réponse inattendue.")
except Exception as e:
    print(f"Erreur lors de l'appel à l'API Mistral : {e}") 