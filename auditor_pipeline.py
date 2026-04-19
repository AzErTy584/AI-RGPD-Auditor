"""
╔══════════════════════════════════════════════════════════════╗
║           RGPD AUDITOR — Pipeline End-to-End                ║
║  PDF → Chunking → Embedding → RAG Search → LLM Audit       ║
╚══════════════════════════════════════════════════════════════╝

Dépendances : pip install pymupdf chromadb google-genai
"""

import os
import uuid
import json
import re
import time
import fitz          # PyMuPDF  ← pip install pymupdf
import chromadb
from chromadb.utils import embedding_functions
from google import genai  # pip install google-genai


# ─────────────────────────────────────────────────────────────
# CONFIGURATION  (passer la clé via variable d'environnement !)
# ─────────────────────────────────────────────────────────────
API_KEY          = "AIzaSyDVIhx0ITwPjZOLlETraKK3968PY7Wan6s" #os.getenv("GOOGLE_API_KEY", "VOTRE_CLE_ICI")
RGPD_DB_PATH     = "./ma_base_rgpd"          # base pré-construite
RGPD_COLLECTION  = "lois_rgpd"
EMBEDDING_MODEL  = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"


def embedding():
    return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=API_KEY,
        model_name=EMBEDDING_MODEL
    )


# ─────────────────────────────────────────────────────────────
# ÉTAPE 1 — CHUNKING DU PDF
# ─────────────────────────────────────────────────────────────

def chunk_pdf(pdf_path: str, chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """
    Découpe un PDF en chunks de ~400 mots avec une fenêtre glissante.

    Stratégie page par page (PyMuPDF) :
    • On extrait le texte brut de chaque page
    • On découpe en fenêtres de `chunk_size` mots avec `overlap` mots en commun
      entre deux chunks consécutifs pour préserver le contexte aux frontières

    Returns :
        list[dict] avec clés : text, page, chunk_index
    """
    doc = fitz.open(pdf_path)
    chunks = []
    chunk_idx = 0

    for page_num, page in enumerate(doc):
        words = page.get_text().split()
        if not words:
            continue

        i = 0
        while i < len(words):
            window = words[i : i + chunk_size]
            text   = " ".join(window).strip()

            if len(text) > 50:   # ignore les pages quasi-vides
                chunks.append({
                    "text":        text,
                    "page":        page_num + 1,
                    "chunk_index": chunk_idx
                })
                chunk_idx += 1

            i += chunk_size - overlap   # fenêtre glissante

    doc.close()
    print(f"[Chunking] ✓ {len(chunks)} chunks extraits depuis '{pdf_path}'")
    return chunks


# ─────────────────────────────────────────────────────────────
# ÉTAPE 2 — CRÉATION DE LA COLLECTION ÉPHÉMÈRE (en mémoire)
# ─────────────────────────────────────────────────────────────

def build_ephemeral_collection(chunks: list[dict], session_id: str) -> chromadb.Collection:
    """
    Embarque les chunks du document utilisateur dans une collection ChromaDB
    en mémoire (EphemeralClient → zéro écriture disque, zéro résidu entre sessions).

    Les appels d'embedding à l'API Google sont groupés par batches de 5
    avec une pause de 3 s pour respecter les rate-limits.
    """
    embed_fn    = embedding()
    temp_client = chromadb.EphemeralClient()

    collection = temp_client.create_collection(
        name             = f"doc_{session_id}",
        metadata         = {"hnsw:space": "cosine"},   # même espace que la base RGPD
        embedding_function = embed_fn
    )

    batch_size  = 5
    nb_batches  = -(-len(chunks) // batch_size)   # ceil division

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents  = [c["text"]  for c in batch],
            ids        = [f"c_{c['chunk_index']}" for c in batch],
            metadatas  = [{
                "page":        c["page"],
                "chunk_index": c["chunk_index"],
                "source":      "user_doc"
            } for c in batch]
        )
        print(f"  [Embedding] Batch {i // batch_size + 1}/{nb_batches} ✓")
        if i + batch_size < len(chunks):
            time.sleep(3)   # pause rate-limit Gemini Embedding API

    print(f"[Collection] ✓ {collection.count()} chunks indexés (session {session_id})")
    return collection


# ─────────────────────────────────────────────────────────────
# ÉTAPE 3 — QUERY EXPANSION
# ─────────────────────────────────────────────────────────────

def expand_query(question: str) -> list[str]:
    """
    Génère 4 reformulations de la question selon différents angles RGPD
    (obligations du responsable, droits des personnes, sécurité, transferts…).

    But : maximiser le rappel lors de la recherche vectorielle — une question
    formulée de façon technique peut rater des articles formulés juridiquement.
    En cas d'échec de l'API, retourne la question originale (fallback sécurisé).
    """
    client = genai.Client(api_key=API_KEY)
    prompt = f"""Tu es un expert juridique RGPD.
Question originale : "{question}"

Génère 4 reformulations couvrant différents angles RGPD :
- Obligations du responsable de traitement
- Droits des personnes concernées
- Sécurité et confidentialité des données
- Transferts internationaux et sous-traitance

Ces reformulations serviront à une recherche vectorielle dans les articles RGPD.
Réponds UNIQUEMENT avec un tableau JSON valide, sans texte avant ni après.
Format exact : ["reformulation 1", "reformulation 2", "reformulation 3", "reformulation 4"]"""

    try:
        resp       = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        raw        = re.sub(r"```json|```", "", resp.text.strip()).strip()
        reformulations = json.loads(raw)
        all_queries    = list(dict.fromkeys([question] + reformulations))
        print(f"[Query Expansion] ✓ {len(all_queries)} requêtes générées")
        return all_queries
    except Exception as e:
        print(f"[Query Expansion] ⚠ Échec ({e}) — fallback sur question originale")
        return [question]


# ─────────────────────────────────────────────────────────────
# ÉTAPE 4 — DUAL SIMILARITY SEARCH
# ─────────────────────────────────────────────────────────────

def dual_similarity_search(
    question:        str,
    doc_collection:  chromadb.Collection,
    rgpd_collection: chromadb.Collection,
    top_k:           int   = 5,
    threshold:       float = 0.35,
    query_expansion: bool  = True
) -> dict:
    """
    Effectue une recherche simultanée dans :
      • doc_collection  → extraits du document utilisateur
      • rgpd_collection → articles du RGPD

    Avec query expansion, chaque requête génère (1 + 4) = 5 sous-requêtes,
    puis les résultats sont dédoublonnés (on garde le score le plus élevé).

    Score de similarité = 1 − distance_cosinus  (ChromaDB retourne des distances)
    """

    def _filter(results, thr):
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

    def _dedup(items):
        seen = {}
        for r in items:
            key = r["text"][:120]
            if key not in seen or r["similarity"] > seen[key]["similarity"]:
                seen[key] = r
        return sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)

    queries = expand_query(question) if query_expansion else [question]
    all_doc, all_law = [], []

    for q in queries:
        all_doc += _filter(
            doc_collection.query(query_texts=[q], n_results=top_k,
                                 include=["documents", "distances", "metadatas"]),
            threshold
        )
        all_law += _filter(
            rgpd_collection.query(query_texts=[q], n_results=top_k,
                                  include=["documents", "distances", "metadatas"]),
            threshold
        )

    return {
        "question":     question,
        "doc_chunks":   _dedup(all_doc)[:top_k],
        "law_articles": _dedup(all_law)[:top_k]
    }


# ─────────────────────────────────────────────────────────────
# ÉTAPE 5 — PROMPT D'AUDIT + APPEL LLM
# ─────────────────────────────────────────────────────────────

def build_audit_prompt(search_results: dict) -> str:
    """
    Construit le prompt structuré de type "juge" envoyé à Gemini.
    Le LLM reçoit : question + extraits doc + articles RGPD.
    """
    doc_ctx = "\n\n".join([
        f"[Page {r['metadata'].get('page','?')} — score {r['similarity']}]\n{r['text']}"
        for r in search_results["doc_chunks"]
    ]) or "⚠ Aucun extrait pertinent trouvé dans le document."

    law_ctx = "\n\n".join([
        f"[{r['metadata'].get('article','?')} — {r['metadata'].get('titre','')}]\n{r['text']}"
        for r in search_results["law_articles"]
    ]) or "⚠ Aucun article RGPD pertinent trouvé."

    return f"""Tu es un auditeur RGPD expert et rigoureux.

══════════════════════════════════════
QUESTION DE L'UTILISATEUR :
{search_results["question"]}

══════════════════════════════════════
EXTRAITS DU DOCUMENT SOUMIS :
{doc_ctx}

══════════════════════════════════════
ARTICLES RGPD APPLICABLES :
{law_ctx}

══════════════════════════════════════
MISSION — Produis un rapport d'audit structuré :

## VERDICT GLOBAL
CONFORME / NON CONFORME / PARTIELLEMENT CONFORME

## NON-CONFORMITÉS IDENTIFIÉES
Pour chaque non-conformité :
- **Criticité** : CRITIQUE / MAJEURE / MINEURE
- **Article RGPD violé** :
- **Clause du document concernée** :
- **Écart précis** :
- **Exposition financière** : (Art. 83 — jusqu'à X€ ou X% du CA mondial)
- **Recommandation corrective** :

## POINTS NÉCESSITANT DES INFORMATIONS COMPLÉMENTAIRES
[Si les extraits sont insuffisants pour conclure sur certains points]

## RÉSUMÉ EXÉCUTIF
[3 lignes maximum]"""


def run_llm_audit(prompt: str) -> str:
    """Envoie le prompt à Gemini et retourne le rapport d'audit."""
    client   = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    return response.text.strip()


# ─────────────────────────────────────────────────────────────
# ORCHESTRATEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run_full_audit(
    pdf_path:  str,
    question:  str,
    top_k:     int   = 5,
    threshold: float = 0.35
) -> dict:
    """
    Pipeline complet : PDF → verdict d'audit RGPD.

    Args:
        pdf_path  : Chemin vers le fichier PDF à analyser
        question  : Question de conformité posée par l'utilisateur
        top_k     : Nombre de chunks/articles à récupérer par recherche
        threshold : Score de similarité minimum (0.0–1.0)

    Returns:
        dict contenant : session_id, question, verdict, doc_chunks, law_articles
    """
    print("\n" + "═"*55)
    print("  RGPD AUDITOR — PIPELINE DÉMARRÉ")
    print("═"*55)

    # ── 1. Chunking ──
    print("\n[1/5] 📄 Découpage du document PDF...")
    chunks = chunk_pdf(pdf_path)
    if not chunks:
        raise ValueError("Aucun texte extrait du PDF. Vérifiez que le fichier n'est pas scanné.")

    # ── 2. Embedding éphémère ──
    print("\n[2/5] 🔢 Création de la base vectorielle temporaire...")
    session_id     = str(uuid.uuid4())[:8]
    doc_collection = build_ephemeral_collection(chunks, session_id)

    # ── 3. Connexion base RGPD ──
    print("\n[3/5] ⚖️  Connexion à la base RGPD pré-construite...")
    rgpd_client     = chromadb.PersistentClient(path=RGPD_DB_PATH)
    rgpd_collection = rgpd_client.get_collection(
        name               = RGPD_COLLECTION,
        embedding_function = embedding()
    )
    print(f"       {rgpd_collection.count()} articles RGPD chargés")

    # ── 4. Dual search ──
    print("\n[4/5] 🔍 Recherche de similarité duale (doc + RGPD)...")
    results = dual_similarity_search(
        question, doc_collection, rgpd_collection, top_k, threshold
    )
    print(f"       {len(results['doc_chunks'])} extraits doc | "
          f"{len(results['law_articles'])} articles RGPD retenus")

    # ── 5. LLM audit ──
    print("\n[5/5] 🤖 Génération du rapport d'audit par Gemini...")
    prompt  = build_audit_prompt(results)
    verdict = run_llm_audit(prompt)

    print("\n" + "═"*55)
    print("  ✅ AUDIT TERMINÉ — session", session_id)
    print("═"*55 + "\n")

    return {
        "session_id":   session_id,
        "question":     question,
        "verdict":      verdict,
        "doc_chunks":   results["doc_chunks"],
        "law_articles": results["law_articles"]
    }


# ─────────────────────────────────────────────────────────────
# POINT D'ENTRÉE CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_full_audit(
        pdf_path  = "Industrial_Compliance_Test_Contract.pdf",
        question  = "Quels sont les risques de non-conformité liés à la collecte de données personnelles ?"
    )

    print("📋 RAPPORT D'AUDIT :\n")
    print(result["verdict"])

    # Sauvegarde du rapport en Markdown
    report_file = f"audit_{result['session_id']}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Rapport d'Audit RGPD\n\n")
        f.write(f"**Question :** {result['question']}\n\n")
        f.write(f"---\n\n")
        f.write(result["verdict"])

    print(f"\n💾 Rapport sauvegardé : {report_file}")