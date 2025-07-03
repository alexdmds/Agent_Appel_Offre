import sys
import os
from ao_agent import run_agent_on_folder

INDEX_PATH = "rag.index"
META_PATH = "rag.meta.pkl"


def ask_yes_no(question):
    try:
        return input(question + " [o/N] ").strip().lower() == "o"
    except EOFError:
        return False

def main():
    # Vérifie la présence de l'index RAG
    index_exists = os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)
    if not index_exists:
        print("\nAucun index RAG trouvé.")
        if ask_yes_no("Voulez-vous générer l'index RAG maintenant ?"):
            os.system(f"python rag.py")
        else:
            print("L'agent risque de ne pas pouvoir répondre aux requêtes RAG historiques.")
    else:
        if ask_yes_no("Un index RAG existe déjà. Voulez-vous le régénérer ?"):
            os.system(f"python rag.py")
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "OneDrive_1_02-07-2025/New_AO/Action Logement - New AO"
    print(f"Chemin du dossier AO à analyser : {folder_path}")
    run_agent_on_folder(folder_path)

if __name__ == "__main__":
    main() 