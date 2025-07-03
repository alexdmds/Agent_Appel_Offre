import os
from collections import Counter
from pathlib import Path

dossier = "OneDrive_1_02-07-2025"
exts = []

for path in Path(dossier).rglob("*"):
    if path.is_file():
        ext = path.suffix.lower()
        if ext:
            exts.append(ext)
        else:
            exts.append("<sans extension>")

compteur = Counter(exts)

print("Extensions trouvées :")
for ext, count in sorted(compteur.items()):
    print(f"{ext} : {count}") 