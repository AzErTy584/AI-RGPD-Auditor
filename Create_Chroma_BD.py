import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import time

API_KEY = "AIzaSyAzM4LtKQxELGYtssXI7-JJ8-nu4N9Djp8"
mes_chunks_rgpd = pd.read_csv('RGPD_articles.csv')
mes_chunks_contrat = pd.read_csv('Industrial_Compliance_Test_Contract.csv')

# [Vecteur (Signature mathématique),Document (Texte réel),Métadonnées (Source)]
def create_chroma_db_rgpd(mes_chunks,API_KEY):
    # 1. On définit où la base de données sera enregistrée
    client = chromadb.PersistentClient(path="./ma_base_rgpd")

    # 2. On configure l'embedding (Indispensable pour que Chroma parle la même langue que Gemini)
    # Vous utilisez l'API Google pour transformer le texte en vecteurs
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=API_KEY,
        model_name="gemini-embedding-001"
    )

    # 3. On crée (ou récupère) la collection"
    collection = client.get_or_create_collection(
        name="lois_rgpd", 
        embedding_function=google_ef,
        metadata={"hnsw:space": "cosine"} # Nécessaire pour que Chroma utilise la distance cosinus lors de la recherche de similarité entre les vecteurs
    )

    batch_size = 5
    for i in range(0, len(mes_chunks), batch_size):
        batch = mes_chunks.iloc[i : i + batch_size]
        
        collection.add(
            documents=batch["Contenu"].tolist(),
            metadatas=[{
                "source": "RGPD", 
                "article": row["Article"],
                "titre": row["Nom"]
            } for _, row in batch.iterrows()],
            ids=[f"id_{j+1}" for j in range(i, i + len(batch))]
        )
        print(f"Batch {i//batch_size + 1} ajouté...")
        time.sleep(10)  # Pause pour éviter les limites d'API
    print("Base de données Chroma créée avec succès !")


def create_chroma_db_contrat(mes_chunks_contrat,API_KEY):
    client = chromadb.PersistentClient(path="./ma_base_contrat")
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=API_KEY,
        model_name="gemini-embedding-001"
    )

    collection = client.get_or_create_collection(
        name="contrat", 
        embedding_function=google_ef,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 5
    for i in range(0, len(mes_chunks_contrat), batch_size):
        batch = mes_chunks_contrat.iloc[i : i + batch_size]
        
        collection.add(
            documents=batch["Contenu"].tolist(),
            metadatas=[{
                "source": "Contrat", 
                "article": i+1,
                "titre": row["Titre"]
            } for _, row in batch.iterrows()],
            ids=[f"id_{j+1}" for j in range(i, i + len(batch))]
        )
        print(f"Batch {i//batch_size + 1} ajouté...")
        time.sleep(10)  # Pause pour éviter les limites d'API
    print("Base de données Chroma créée avec succès !")

# create_chroma_db_rgpd(mes_chunks_rgpd,API_KEY)
create_chroma_db_contrat(mes_chunks_contrat,API_KEY)