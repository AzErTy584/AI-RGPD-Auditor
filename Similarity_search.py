import chromadb
from chromadb.utils import embedding_functions
from google import genai
import json
import re


API_KEY = "AIzaSyAzM4LtKQxELGYtssXI7-JJ8-nu4N9Djp8"
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=API_KEY,
    model_name="gemini-embedding-001"  # ← même modèle que dans Create_Chroma_BD.py
)

# ────────────────────────────────────────────
# 1. Récupération de la collection RGPD (déjà créée et peuplée)
# ────────────────────────────────────────────

rgpd_client = chromadb.PersistentClient(path="./ma_base_rgpd")
rgpd_collection = rgpd_client.get_collection(name="lois_rgpd", 
                                             embedding_function=google_ef)

# ────────────────────────────────────────────
# 2. Récupération du document utilisateur (ultérieurement création d'une collection éphémère)
# ────────────────────────────────────────────

# a utiliser ultérieurement pour créer une collection éphémère en mémoire pour le document utilisateur
def build_temp_collection(chunks: list[str], session_id: str):
    """
    Crée une collection éphémère en mémoire pour le document utilisateur.
    session_id permet d'isoler les sessions parallèles.
    """
    temp_client = chromadb.EphemeralClient()  # ← en mémoire, pas de persistence

    temp_collection = temp_client.create_collection(
        name=f"doc_user_{session_id}",
        metadata={"hnsw:space": "cosine"},    # ← même espace que RGPD
        #embedding_function=embed_fn
    )

    temp_collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_index": i, "source": "user_doc"} 
                   for i in range(len(chunks))]
    )

    return temp_collection

contrat_client = chromadb.PersistentClient(path="./ma_base_contrat")
contrat_collection = contrat_client.get_collection(name="contrat",
                                                   embedding_function=google_ef)

# ────────────────────────────────────────────
# [NOUVEAU] QUERY EXPANSION 2.bis
# ────────────────────────────────────────────
# Duplication de la question originale sous 4 angles juridiques différents pour maximiser les chances de trouver des passages pertinents dans les articles RGPD, même si la question initiale est formulée de manière très spécifique ou avec des termes peu courants.
def expand_query(question: str) -> list[str]:
    client = genai.Client(api_key=API_KEY)
    prompt = f"""Tu es un expert juridique RGPD.
Question originale : "{question}"

Génère 4 reformulations de cette question couvrant différents angles RGPD
(ex: obligations du responsable, droits des personnes, sécurité, transferts...).
Ces reformulations serviront à une recherche vectorielle dans les articles RGPD.

Réponds UNIQUEMENT avec un tableau JSON valide, sans texte avant ou après.
Exemple de format attendu :
["reformulation 1", "reformulation 2", "reformulation 3", "reformulation 4"]"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        raw = response.text.strip()

        # Nettoyage si Gemini ajoute des backticks markdown
        raw = re.sub(r"```json|```", "", raw).strip()
        reformulations = json.loads(raw)
        # print(f"[Query Expansion] Reformulations générées : {reformulations}")

        # Question originale + reformulations (dédoublonnage)
        all_queries = list(dict.fromkeys([question] + reformulations))
        # print(f"[Query Expansion] {len(all_queries)} requêtes générées")
        return all_queries

    except Exception as e:
        print(f"[Query Expansion] Échec ({e}), fallback sur question originale")
        return [question]  # ← fallback sécurisé, le système continue quand même

# ────────────────────────────────────────────
# 3. LE DUAL SIMILARITY SEARCH
# ────────────────────────────────────────────

def dual_similarity_search(
    question: str,
    temp_collection,
    top_k_doc: int = 5,
    top_k_law: int = 5,
    score_threshold: float = 0.35,
    use_query_expansion: bool = True  # ← tu peux désactiver facilement
) -> dict:

    def filter_by_threshold(results, threshold):
        filtered = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            similarity = 1 - dist
            if similarity >= threshold:
                filtered.append({
                    "text": doc,
                    "similarity": round(similarity, 4),
                    "metadata": meta
                })
        return sorted(filtered, key=lambda x: x["similarity"], reverse=True)

    def deduplicate(results: list[dict]) -> list[dict]:
        """Supprime les doublons en gardant le score le plus élevé."""
        seen = {}
        for r in results:
            key = r["text"][:100]  # clé = début du texte
            if key not in seen or r["similarity"] > seen[key]["similarity"]:
                seen[key] = r
        return sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)

    # ── Génération des requêtes ──
    queries = expand_query(question) if use_query_expansion else [question]

    # ── Recherche multi-requêtes ──
    all_doc_results = []
    all_law_results = []

    for q in queries:
        doc_res = temp_collection.query(
            query_texts=[q],
            n_results=top_k_doc,
            include=["documents", "distances", "metadatas"]
        )
        law_res = rgpd_collection.query(
            query_texts=[q],
            n_results=top_k_law,
            include=["documents", "distances", "metadatas"]
        )
        all_doc_results.extend(filter_by_threshold(doc_res, score_threshold))
        all_law_results.extend(filter_by_threshold(law_res, score_threshold))

    return {
        "question": question,
        "doc_chunks": deduplicate(all_doc_results),
        "law_articles": deduplicate(all_law_results)
    }


rep = (dual_similarity_search(
    question="Quels sont les risques de non-conformité liés à la collecte de données personnelles ?",
    temp_collection=contrat_collection,  # ← à remplacer par la collection éphémère du doc utilisateur
    top_k_doc=5,
    top_k_law=5,
    score_threshold=0.35
))
df_return_doc = rep["doc_chunks"]
df_return_law = rep["law_articles"]

# Affichage des résultats
# print("Résultats de la recherche de similarité :\n")
# print("Chunks du document utilisateur les plus similaires :")
# for i, res in enumerate(df_return_doc, 1):
#     print(f"{i}. Similarité: {res['similarity']} - Metadata: {res['metadata']} - Extrait: {res['text'][:100]}...")
# print("\nArticles RGPD les plus similaires :")
# for i, res in enumerate(df_return_law, 1):
#     print(f"{i}. Similarité: {res['similarity']} - Metadata: {res['metadata']} - Extrait: {res['text'][:100]}...")