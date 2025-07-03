# Agent Go/NoGo AO (Appel d'Offres)

Ce projet propose un agent autonome (LangChain + Mistral) capable d'analyser un dossier d'appel d'offres (AO), d'en extraire l'essentiel et de recommander une décision Go/NoGo, avec plan de réponse et synthèse manager.

## Fonctionnalités principales
- Lecture automatique d'un dossier AO (PDF, DOCX, TXT)
- Résumé des besoins, contraintes, délais, critères de notation
- Recommandation Go/NoGo argumentée
- Plan de réponse structuré
- Synthèse manager prête à copier-coller
- Utilisation de l'API Mistral (modèle Chat) pour le raisonnement

## Prérequis
- Python 3.9+
- Clé API Mistral (https://mistral.ai/)

## Installation
1. Clonez le dépôt :
   ```bash
   git clone <repo_url>
   cd <repo>
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Placez vos dossiers AO dans `OneDrive_1_02-07-2025/` (voir structure ci-dessous).

## Configuration
Définissez la clé API Mistral dans votre environnement :
```bash
export MISTRAL_API_KEY="votre_clé_api"
```

## Utilisation
Lancez l'agent sur le dossier AO par défaut :
```bash
python main.py
```

Pour analyser un autre dossier AO :
```bash
python main.py "chemin/vers/votre/dossier_AO"
```

## Structure attendue des dossiers AO
```
OneDrive_1_02-07-2025/
├── AO_GO/
├── AO_NOGO/
└── New_AO/
    └── Action Logement - New AO/
        ├── ... (vos fichiers AO)
```

## Résultat
L'agent affiche un rapport complet en Markdown, prêt à être copié-collé.

## Dépendances principales
- langchain
- mistralai
- PyPDF2, python-docx, pandas, openpyxl, python-pptx

## Contact
Pour toute question, ouvrez une issue ou contactez l'auteur.