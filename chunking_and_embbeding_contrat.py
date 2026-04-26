"""
╔══════════════════════════════════════════════════════════════════╗
║   RGPD AUDITOR — Chunking & Embedding du document utilisateur   ║
╠══════════════════════════════════════════════════════════════════╣
║  Stratégie : découpage STRUCTUREL par titres de section         ║
║  Détection : PyMuPDF dict mode → flags binaires de police       ║
║     flag & 16 = gras  |  size >= seuil = titre                  ║
║  Fallback  : fenêtre glissante si aucune section détectée       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import time
import fitz          # PyMuPDF — pip install pymupdf
import chromadb
from chromadb.utils import embedding_functions

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
API_KEY         = os.environ.get("API_KEY_RGPD_AUDITOR")
EMBEDDING_MODEL = "gemini-embedding-001"  # Ne pas changer : cohérence avec la base RGPD
FLAG_BOLD      = 16    # bit 4 (16) = BOLD
TITLE_MIN_SIZE = 12.0  # pt — corps du texte courant est en 10pt
# ─────────────────────────────────────────────────────────────


def embedding():
    """Fonction d'embedding Google Generative AI."""
    return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=API_KEY,
        model_name=EMBEDDING_MODEL
    )

def is_title_span(span: dict) -> bool:
    """
    Returns True if the span is a section title.
    Criteria: bold (flags & 16) AND size >= TITLE_MIN_SIZE.
    Texts <= 2 characters are filtered out (page numbers, isolated bullets).
    """
    text = span.get("text", "").strip()
    if len(text) <= 2:
        return False
    is_bold  = bool(span.get("flags", 0) & FLAG_BOLD)
    is_large = span.get("size", 0) >= TITLE_MIN_SIZE
    return is_bold and is_large


# ─────────────────────────────────────────────────────────────
# CHUNKING STRUCTUREL
# ─────────────────────────────────────────────────────────────

def chunk_structural(pdf_path: str, min_chunk_len: int = 80) -> list[dict]:
    """
    Découpe le PDF en chunks délimités par les titres de section.

    get_text("dict") retourne blocks > lines > spans avec pour chaque span :
      - text  : texte brut
      - size  : taille en points
      - flags : attributs typographiques (bold, italic...)
      - font  : nom de la police

    L'accumulation est GLOBALE : une section sur 2 pages = 1 seul chunk.
    Le titre est inclus en première ligne du texte pour le contexte LLM.

    Returns : list[dict] — clés : title, text, page, chunk_index
    """
    doc    = fitz.open(pdf_path)
    chunks = []
    idx    = 0

    current_title   = ""
    current_content = ""
    current_page    = 1

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"] # Clée de voute du chinking structurel : car permet d'accèder directement au métadonnées de mise en forme (flags, size) pour chaque span de texte, et ainsi détecter les titres de section de manière fiable.

        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    if is_title_span(span):
                        # Nouveau titre → sauvegarder le chunk précédent
                        if current_content.strip() and len(current_content.strip()) >= min_chunk_len:
                            full_text = f"{current_title}\n{current_content.strip()}" if current_title else current_content.strip()
                            chunks.append({
                                "title":       current_title,
                                "text":        full_text,
                                "page":        current_page,
                                "chunk_index": idx
                            })
                            idx += 1
                        current_title   = text
                        current_content = ""
                        current_page    = page_num
                    else:
                        current_content += " " + text
    # Dernier chunk (pas de titre suivant pour le déclencher)
    if current_content.strip() and len(current_content.strip()) >= min_chunk_len:
        full_text = f"{current_title}\n{current_content.strip()}" if current_title else current_content.strip()
        chunks.append({
            "title":       current_title,
            "text":        full_text,
            "page":        current_page,
            "chunk_index": idx
        })

    doc.close()
    return chunks


# ─────────────────────────────────────────────────────────────
# Chunking par FALLBACK
# ─────────────────────────────────────────────────────────────

def chunk_sliding_window(pdf_path: str, chunk_size: int = 400, overlap: int = 50) -> list[dict]:
    """
    Sliding window split (400 words, 50 overlap). Automatically activated if no heading structure is detected (scanned PDF, OCR, unformatted document).
    """
    doc    = fitz.open(pdf_path)
    chunks = []
    idx    = 0

    for page_num, page in enumerate(doc, start=1):
        words = page.get_text().split()
        if not words:
            continue
        i = 0
        while i < len(words):
            text = " ".join(words[i : i + chunk_size]).strip()
            if len(text) > 50:
                chunks.append({
                    "title":       "",
                    "text":        text,
                    "page":        page_num,
                    "chunk_index": idx
                })
                idx += 1
            i += chunk_size - overlap

    doc.close()
    return chunks


# ─────────────────────────────────────────────────────────────
# Final chuncking fonction 
# ─────────────────────────────────────────────────────────────

def chunk_pdf(pdf_path: str) -> list[dict]:
    """
    • >= 3 sections détectées → chunking STRUCTUREL (par titres)
    • <  3 sections           → fenêtre GLISSANTE (fallback)

    Returns : list[dict] — clés : title, text, page, chunk_index
    """

    structural = chunk_structural(pdf_path)

    if len(structural) >= 3:
        print(f"[Chunking] ✓ Mode STRUCTUREL — {len(structural)} sections détectées")
        return structural

    sliding = chunk_sliding_window(pdf_path)
    print(f"[Chunking] ✓ Mode FENÊTRE GLISSANTE — {len(sliding)} chunks")
    return sliding


# ─────────────────────────────────────────────────────────────
# Ephemeral embedding (ChromaDB en mémoire)
# ─────────────────────────────────────────────────────────────

def build_ephemeral_collection(chunks: list[dict], session_id: str) -> chromadb.Collection:
    """
    Load the chunks into an in-memory ChromaDB collection.
    """
    embed_fn    = embedding()
    temp_client = chromadb.EphemeralClient()

    collection = temp_client.create_collection(
        name               = f"doc_{session_id}",
        metadata           = {"hnsw:space": "cosine"},
        embedding_function = embed_fn
    )

    batch_size = 5
    nb_batches = -(-len(chunks) // batch_size)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents = [c["text"] for c in batch],
            ids       = [f"c_{c['chunk_index']}" for c in batch],
            metadatas = [{
                "page":        c["page"],
                "chunk_index": c["chunk_index"],
                "title":       c.get("title", ""),
                "source":      "user_doc"
            } for c in batch]
        )
        print(f"  [Embedding] Batch {i // batch_size + 1}/{nb_batches} ✓")
        if i + batch_size < len(chunks):
            time.sleep(3)

    print(f"[Collection] ✓ {collection.count()} chunks indexés (session {session_id})")
    return collection


# ─────────────────────────────────────────────────────────────
# TEST RAPIDE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\USER\Desktop\Préparation_Data_science\Projet_AI_RGPD_Auditor\Update_en_cours\Industrial_Compliance_Test_Contract.pdf"
    chunks = chunk_pdf(pdf)
    print(f"\n{'─'*55}")
    print(f"  {len(chunks)} chunks extraits\n")
    for c in chunks:
        print(f"  [{c['chunk_index']}] Page {c['page']} — \"{c['title']}\"")
        print(f"       {c['text'][:80].replace(chr(10),' ')}...")