import chromadb
from query_expansion import expand_query

def dual_similarity_search(
    question:        str,
    doc_collection:  chromadb.Collection,
    rgpd_collection: chromadb.Collection,
    top_k:           int   = 5,
    threshold:       float = 0.35,
    query_expansion: bool  = True
) -> dict:
    """
    Perform a simultaneous search in:
        • doc_collection → excerpts from the user document
        • rgpd_collection → GDPR articles

    Similarity score = 1 − cosine distance (ChromaDB returns distances)
    Return only results law text with similarity >= threshold, sorted by similarity.
    """

    def filter(results, thr):
        """Filter results by similarity threshold and sort them."""
        out = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            sim = 1.0 - dist
            if sim >= thr:
                out.append({"text": doc, "similarity": round(sim, 4), "metadata": meta})
        return sorted(out, key=lambda x: x["similarity"], reverse=True)

    def dedup(items):
        """Deduplicate results based on the first 120 characters of the text."""
        seen = {}
        for r in items:
            key = r["text"][:120]
            if key not in seen or r["similarity"] > seen[key]["similarity"]:
                seen[key] = r
        return sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)

    queries = expand_query(question) if query_expansion else [question]
    all_doc, all_law = [], []

    for q in queries:
    # Chromadb embed automatiquement et calcul distance cos (avec même modèle embedding) la chaîne passée dans query_texts. C'est pourquoi on peut faire la recherche directement avec les textes, sans calculer les vecteurs nous-mêmes (donc embedder la question utilisateur n'est pas nécessaire).
        all_doc += filter(
            doc_collection.query(query_texts=[q], n_results=top_k,
                                 include=["documents", "distances", "metadatas"]),
            threshold
        )
        all_law += filter(
            rgpd_collection.query(query_texts=[q], n_results=top_k,
                                  include=["documents", "distances", "metadatas"]),
            threshold
        )

    return {
        "question":     question,
        "doc_chunks":   dedup(all_doc)[:top_k],
        "law_articles": dedup(all_law)[:top_k]
    }