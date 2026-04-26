"""
╔══════════════════════════════════════════════════════════════╗
║           RGPD AUDITOR — Pipeline End-to-End                 ║
║  PDF → Chunking → Embedding → RAG Search → LLM Audit         ║
╚══════════════════════════════════════════════════════════════╝

Dépendances : pip install pymupdf chromadb google-genai
"""

import os
import uuid
import chromadb
from chunking_and_embbeding_contrat import chunk_pdf, build_ephemeral_collection, embedding
from dual_similarity_search import dual_similarity_search
from prompt_audit import build_audit_prompt, run_llm_audit

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  (passer la clé via variable d'environnement !)
# ─────────────────────────────────────────────────────────────
API_KEY          = os.environ.get('API_KEY_RGPD_AUDITOR')
RGPD_DB_PATH     = ".\ma_base_rgpd"          # base pré-construite
RGPD_COLLECTION  = "lois_rgpd"
EMBEDDING_MODEL  = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"


def run_full_audit(pdf_path:str, question:str, top_k: int = 5, threshold: float = 0.35) -> dict:
    """" Pipeline : PDF → analyse RGDP.
    Args:
        pdf_path  : path pdf analyse
        question  : User question
        top_k     : Chunks by article to retrieve
        threshold : Minimum similarity score (0.0–1.0)
    Returns:
        dict : session_id, question, verdict, doc_chunks, law_articles
    """
    # ── 1. Chunking ──
    print("\n[1/5] 📄 Chunking pdf")
    chunks = chunk_pdf(pdf_path)
    if not chunks:
        raise ValueError("No text extracted from the PDF. Check the file path and format.")

    # ── 2. Embedding user article ──
    print("\n[2/5] 🔢 Creation of the temporary vector database")
    session_id     = str(uuid.uuid4())[:8]
    doc_collection = build_ephemeral_collection(chunks, session_id)

    # ── 3. Connexion base RGPD ──
    print("\n[3/5] ⚖️  Connection à la base RGPD pré-embeddée")
    rgpd_client     = chromadb.PersistentClient(path=RGPD_DB_PATH)
    rgpd_collection = rgpd_client.get_collection(
        name               = RGPD_COLLECTION,
        embedding_function = embedding()
    )
    print(f"       {rgpd_collection.count()} RGPD articles in charge")

    # ── 4. Dual search ──
    print("\n[4/5] 🔍 Dual search similarity RGPD and user doc")
    results = dual_similarity_search(
        question, doc_collection, rgpd_collection, top_k, threshold
    )
    print(f"       {len(results['doc_chunks'])} document excerpts | "
          f"{len(results['law_articles'])} retained RGPD articles")

    # ── 5. LLM audit ──
    print("\n[5/5] 🤖 Generation of the audit report by Gemini")
    prompt  = build_audit_prompt(results)
    verdict = run_llm_audit(prompt)

    print("\n" + "═"*55)
    print("  ✅ AUDIT COMPLETED — session", session_id)
    print("═"*55 + "\n")

    return {
        "session_id":   session_id,
        "question":     question,
        "verdict":      verdict,
        "doc_chunks":   results["doc_chunks"],
        "law_articles": results["law_articles"]
    }


# ─────────────────────────────────────────────────────────────
# Test de du pipeline complet
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_full_audit(
        pdf_path  = r"C:\Users\USER\Desktop\Préparation_Data_science\Projet_AI_RGPD_Auditor\Update_en_cours\Industrial_Compliance_Test_Contract.pdf",
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