import chromadb

rgpd_client = chromadb.PersistentClient(path="./ma_base_rgpd")
rgpd_collection = rgpd_client.get_collection(name="lois_rgpd")

contrat_client = chromadb.PersistentClient(path="./ma_base_contrat")
contrat_collection = contrat_client.get_collection(name="contrat")

donnees_rgpd = rgpd_collection.get(limit=100, include=["documents", "metadatas", "embeddings"])
donnees_contrat = contrat_collection.get(limit=100, include=["documents", "metadatas", "embeddings"])

print(f"Nombre d'éléments dans la base rgpd: {rgpd_collection.count()}")
for i in range(len(donnees_rgpd['ids'])):
    # print(f"\nID: {donnees['ids'][i]}")
    # print(f"Metadata: {donnees['metadatas'][i]}")
    # print(f"Texte: {donnees['documents'][i][:10000]}...") # On affiche les 100 premiers caractères
    print(f"Embedding dim base rgpd: {len(donnees_rgpd['embeddings'][i])}")

print(f"Nombre d'éléments dans la base contrat : {contrat_collection.count()}")
for i in range(len(donnees_contrat['ids'])):
    print(f"Embedding dim base contrat: {len(donnees_contrat['embeddings'][i])}")