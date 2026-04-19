"""
╔══════════════════════════════════════════════════════════════╗
║           RGPD AUDITOR — Serveur FastAPI                    ║
║  Lance avec : python api.py  (ou uvicorn api:app --reload)  ║
╚══════════════════════════════════════════════════════════════╝

Dépendances : pip install fastapi uvicorn python-multipart
"""

import os
import asyncio
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from auditor_pipeline import run_full_audit

# ─────────────────────────────────────────────────────────────
# APP FASTAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "RGPD Auditor API",
    description = "Analyse la conformité RGPD d'un document PDF via RAG + LLM",
    version     = "1.0.0"
)

# CORS — autorise le frontend React (localhost:3000 Vite ou CRA)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok", "message": "RGPD Auditor API is running"}


@app.post("/audit", tags=["Audit"])
async def audit_document(
    file:      UploadFile = File(...,         description="PDF du contrat à auditer"),
    question:  str        = Form(...,         description="Question de conformité RGPD"),
    top_k:     int        = Form(5,           description="Nb de chunks à récupérer"),
    threshold: float      = Form(0.35,        description="Score de similarité minimum (0–1)")
):
    """
    Endpoint principal : reçoit un PDF + une question, retourne un rapport d'audit RGPD.

    Le pipeline complet s'exécute en arrière-plan (non-bloquant) :
    chunking → embedding éphémère → dual RAG search → LLM audit.

    Réponse JSON :
    ```json
    {
      "session_id":   "abc12345",
      "question":     "...",
      "verdict":      "## VERDICT GLOBAL\\n...",
      "doc_chunks":   [...],
      "law_articles": [...]
    }
    ```
    """
    # ── Validation ──
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    if not question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Le threshold doit être entre 0.0 et 1.0.")

    # ── Sauvegarde temporaire du fichier uploadé ──
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # ── Exécution du pipeline dans un thread (opérations bloquantes) ──
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_full_audit(
                pdf_path  = tmp_path,
                question  = question.strip(),
                top_k     = top_k,
                threshold = threshold
            )
        )
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pipeline : {str(e)}")

    finally:
        os.unlink(tmp_path)   # nettoyage du fichier temporaire


# ─────────────────────────────────────────────────────────────
# LANCEMENT DIRECT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
