import chromadb

client = chromadb.PersistentClient(path="./ma_base_contrat")
collection = client.get_collection(name="contrat")

donnees = collection.get(limit=100, include=["documents", "metadatas", "embeddings"])

print(f"Nombre d'éléments dans la base : {collection.count()}")
for i in range(len(donnees['ids'])):
    print(f"\nID: {donnees['ids'][i]}")
    print(f"Metadata: {donnees['metadatas'][i]}")
    print(f"Texte: {donnees['documents'][i][:10000]}...") # On affiche les 100 premiers caractères

