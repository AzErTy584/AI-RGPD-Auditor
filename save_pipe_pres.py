"""
╔══════════════════════════════════════════════════════════════╗
║           RGPD AUDITOR — Pipeline Chat (présentation)        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import uuid
import chromadb
from chunking_and_embbeding_contrat import chunk_pdf, build_ephemeral_collection, embedding
from dual_similarity_search import dual_similarity_search
from prompt_audit_pres import build_audit_prompt, run_llm_audit

API_KEY         = os.environ.get('API_KEY_RGPD_AUDITOR')
RGPD_DB_PATH    = "./ma_base_rgpd"
RGPD_COLLECTION = "lois_rgpd"


def run_full_audit_pres(
    pdf_path:  str,
    question:  str,
    top_k:     int   = 5,
    threshold: float = 0.35
) -> dict:
    """
    Pipeline initial : PDF → premier diagnostic RGPD (mode chat).
    chat_history=None → prompt 'premier diagnostic'.

    Clés du dict retourné :
      - Clés publiques  → envoyées au frontend (session_id, verdict, ...)
      - _doc_collection  → stockée en session API, filtrée avant envoi frontend
      - _rgpd_collection → idem

    BUG CORRIGÉ : ces deux clés étaient absentes dans la version précédente,
    rendant le store de sessions dans api.py inutilisable (KeyError),
    et donc /chat impossible.
    """
    print("\n[1/5] 📄 Chunking pdf")
    chunks = chunk_pdf(pdf_path)
    if not chunks:
        raise ValueError("No text extracted from the PDF. Check the file path and format.")

    print("\n[2/5] 🔢 Création de la base vectorielle temporaire")
    session_id     = str(uuid.uuid4())[:8]
    doc_collection = build_ephemeral_collection(chunks, session_id)

    print("\n[3/5] ⚖️  Connexion à la base RGPD pré-embeddée")
    rgpd_client     = chromadb.PersistentClient(path=RGPD_DB_PATH)
    rgpd_collection = rgpd_client.get_collection(
        name               = RGPD_COLLECTION,
        embedding_function = embedding()
    )
    print(f"       {rgpd_collection.count()} articles RGPD chargés")

    print("\n[4/5] 🔍 Dual search RGPD + document utilisateur")
    results = dual_similarity_search(
        question, doc_collection, rgpd_collection, top_k, threshold
    )
    print(f"       {len(results['doc_chunks'])} extraits doc | "
          f"{len(results['law_articles'])} articles RGPD retenus")

    print("\n[5/5] 🤖 Génération du diagnostic par Gemini")
    prompt  = build_audit_prompt(results, chat_history=None)
    verdict = run_llm_audit(prompt)

    print(f"\n{'═'*55}\n  ✅ AUDIT TERMINÉ — session {session_id}\n{'═'*55}\n")

    return {
        "session_id":       session_id,
        "question":         question,
        "verdict":          verdict,
        "doc_chunks":       results["doc_chunks"],
        "law_articles":     results["law_articles"],
        # Préfixe _ → filtrés par api.py avant envoi au frontend,
        # mais lus par api.py pour alimenter le store de sessions
        "_doc_collection":  doc_collection,
        "_rgpd_collection": rgpd_collection,
    }


def run_chat_turn(
    session_doc_collection,
    session_rgpd_collection,
    question:     str,
    chat_history: list,
    top_k:        int   = 5,
    threshold:    float = 0.35,
) -> dict:
    """
    Tour de chat suivant — réutilise les collections déjà en mémoire.
    Pas de re-chunking ni re-embedding.
    chat_history non vide → prompt 'suite de conversation'.
    """
    print(f"\n[Chat] 🔍 RAG search pour : '{question[:60]}...'")
    results = dual_similarity_search(
        question,
        session_doc_collection,
        session_rgpd_collection,
        top_k,
        threshold
    )

    print("[Chat] 🤖 Génération de la réponse avec historique...")
    prompt  = build_audit_prompt(results, chat_history)
    verdict = run_llm_audit(prompt)

    return {
        "question":     question,
        "verdict":      verdict,
        "doc_chunks":   results["doc_chunks"],
        "law_articles": results["law_articles"],
    }


if __name__ == "__main__":
    # BUG CORRIGÉ : appelait run_full_audit() (inexistant ici) au lieu de run_full_audit_pres()
    result = run_full_audit_pres(
        pdf_path = "Industrial_Compliance_Test_Contract.pdf",
        question = "Quels sont les risques de non-conformité liés à la collecte de données personnelles ?"
    )
    print("📋 RAPPORT :\n", result["verdict"])
    report_file = f"audit_{result['session_id']}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Rapport d'Audit RGPD\n\n**Question :** {result['question']}\n\n---\n\n{result['verdict']}")
    print(f"💾 Rapport sauvegardé : {report_file}")