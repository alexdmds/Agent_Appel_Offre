import uuid
from typing import Any
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os

# Outil : query_rag
@tool
def query_rag_tool(question: str) -> Any:
    """Interroge la base documentaire AO passés (GO/NOGO) pour aider à raisonner sur un nouvel AO."""
    from rag import query_rag
    return query_rag(question)

# Outil : read_documents_from_folder (ne lit que le début de chaque document)
@tool
def read_documents_from_folder_tool(folder_path: str, max_chars: int = 4000) -> str:
    """Lit le début de chaque document d'un dossier AO (PDF, DOC, DOCX, XLS, XLSX, TXT, CSV, PPTX) et retourne leur contenu textuel brut concaténé (max_chars par fichier)."""
    import glob
    from pathlib import Path
    import docx, pandas as pd
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None
    try:
        import openpyxl
    except ImportError:
        openpyxl = None
    try:
        import pptx
    except ImportError:
        pptx = None
    exts = ("pdf", "doc", "docx", "xls", "xlsx", "txt", "csv", "pptx")
    texts = []
    for ext in exts:
        for file in Path(folder_path).rglob(f"*.{ext}"):
            try:
                content = ""
                if ext == "pdf" and PdfReader:
                    with open(file, "rb") as f:
                        reader = PdfReader(f)
                        content = "\n".join(page.extract_text() or "" for page in reader.pages)
                elif ext == "docx":
                    doc = docx.Document(file)
                    content = "\n".join([p.text for p in doc.paragraphs])
                elif ext == "txt":
                    with open(file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                elif ext == "csv":
                    df = pd.read_csv(file, dtype=str, encoding_errors='ignore')
                    content = df.astype(str).to_string(index=False)
                elif ext in ("xls", "xlsx"):
                    df = pd.read_excel(file, dtype=str, engine='openpyxl' if ext=="xlsx" else None)
                    content = df.astype(str).to_string(index=False)
                elif ext == "pptx" and pptx:
                    prs = pptx.Presentation(file)
                    content = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
                if content:
                    texts.append(f"--- {file} ---\n" + content[:max_chars])
            except Exception as e:
                continue
    return "\n\n".join(texts)

# Outil : read_more_from_file
@tool
def read_more_from_file(filepath: str, offset: int = 4000, max_chars: int = 4000) -> str:
    """Lit la suite d'un fichier à partir d'un offset (en caractères), pour permettre à l'agent d'explorer un document volumineux par morceaux."""
    from pathlib import Path
    import docx, pandas as pd
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None
    try:
        import openpyxl
    except ImportError:
        openpyxl = None
    try:
        import pptx
    except ImportError:
        pptx = None
    ext = Path(filepath).suffix.lower().replace('.', '')
    content = ""
    try:
        if ext == "pdf" and PdfReader:
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == "docx":
            doc = docx.Document(filepath)
            content = "\n".join([p.text for p in doc.paragraphs])
        elif ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        elif ext == "csv":
            df = pd.read_csv(filepath, dtype=str, encoding_errors='ignore')
            content = df.astype(str).to_string(index=False)
        elif ext in ("xls", "xlsx"):
            df = pd.read_excel(filepath, dtype=str, engine='openpyxl' if ext=="xlsx" else None)
            content = df.astype(str).to_string(index=False)
        elif ext == "pptx" and pptx:
            prs = pptx.Presentation(filepath)
            content = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
    except Exception as e:
        return f"Erreur lors de la lecture de {filepath} : {e}"
    if not content:
        return f"Aucun contenu lisible dans {filepath}"
    return content[offset:offset+max_chars]

# Prompt système pour l'agent
SYSTEM_PROMPT = """
Tu es un agent expert en analyse d'appels d'offres (AO) pour une société de conseil data.
Ton objectif est d'analyser un nouveau dossier AO (fourni sous forme de dossier contenant des fichiers) et de produire :
1. Un résumé des besoins, contraintes, délais, critères de notation.
2. Une recommandation Go/NoGo (en t'appuyant sur les AO passés via l'outil query_rag).
3. Un début de plan de réponse structuré.
4. Un résumé clair pour un manager.

Tu disposes de trois outils :
- read_documents_from_folder : pour lire le début de chaque document d'un dossier AO (max 4000 caractères par fichier)
- read_more_from_file : pour lire la suite d'un fichier si tu en as besoin (en précisant le chemin et l'offset)
- query_rag : pour interroger la base documentaire des AO passés (GO/NOGO)

Raisonne étape par étape (ReAct) et utilise read_documents_from_folder pour commencer, puis read_more_from_file si tu veux explorer un document en profondeur, et query_rag dès que tu as besoin d'exemples ou de contexte.
Ta sortie finale doit être lisible et structurée (Markdown si utile).
"""

def make_agent():
    memory = MemorySaver()
    model = ChatOpenAI(temperature=0)
    tools = [read_documents_from_folder_tool, read_more_from_file, query_rag_tool]
    agent = create_react_agent(
        model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent, memory

def run_agent_on_folder(folder_path: str):
    agent, memory = make_agent()
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}
    input_message = HumanMessage(content=f"Voici le dossier AO à analyser : {folder_path}")
    print("\n--- Analyse de l'appel d'offre ---\n")
    for event in agent.stream({"messages": [input_message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    folder_path = input("Chemin du dossier AO à analyser : ")
    run_agent_on_folder(folder_path) 