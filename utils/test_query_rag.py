from rag import query_rag

question = input("Pose ta question : ")
results = query_rag(question, top_k=5)

print("\nRésultats les plus pertinents :")
for i, res in enumerate(results, 1):
    print(f"\n--- Résultat {i} ---")
    print(f"Score    : {res['score']:.3f}")
    print(f"Fichier  : {res['filepath']}")
    print(f"AO ID    : {res['ao_id']}")
    print(f"Decision : {res['decision']}")
    print(f"Contenu  : {res.get('content', '[non disponible]')[:500]}...") 