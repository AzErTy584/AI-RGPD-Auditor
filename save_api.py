"""
╔══════════════════════════════════════════════════════════════╗
║           RGPD AUDITOR — Serveur FastAPI                     ║
╚══════════════════════════════════════════════════════════════╝

Dépendances : pip install fastapi uvicorn python-multipart
"""

import os
import asyncio
import tempfile
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pipeline import run_full_audit
from pipeline_pres import run_full_audit_pres, run_chat_turn

# ─────────────────────────────────────────────────────────────
# APP FASTAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "RGPD Auditor API",
    description = "Analyze the RGPD compliance of a PDF document via RAG + LLM",
    version     = "1.0.0"
)

# CORS — autorise le frontend React (localhost:3000 Vite ou CRA)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

sessions: dict = {}
SESSION_TTL_MINUTES = 60
 
 
def purge_expired_sessions():
    """Supprime les sessions expirées du store."""
    now     = datetime.utcnow()
    expired = [sid for sid, s in sessions.items()
               if now - s["created_at"] > timedelta(minutes=SESSION_TTL_MINUTES)]
    for sid in expired:
        del sessions[sid]
    if expired:
        print(f"[Sessions] {len(expired)} session(s) expirée(s) purgée(s)")

# ─────────────────────────────────────────────────────────────
# SCHÉMA DE REQUÊTE CHAT
# ─────────────────────────────────────────────────────────────
 
class ChatRequest(BaseModel):
    session_id:   str
    question:     str
    chat_history: list[dict] = []

# ─────────────────────────────────────────────────────────────
# API Requests
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Check API is running."""
    return {"status": "ok", "message": "RGPD Auditor API is running"}


@app.post("/audit", tags=["Audit"])
async def audit_document(
    file:      UploadFile = File(...,         description="PDF of the contract to be audited"),
    question:  str        = Form(...,         description="Compliance issue RGPD"),
    top_k:     int        = Form(5,           description="Number of chunks to retrieve"),
    threshold: float      = Form(0.35,        description="Minimum similarity score (0–1)")
):
    
    purge_expired_sessions()

    # ── Validation ──
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    if not question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Le threshold doit être entre 0.0 et 1.0.")

    # ── Save upload file ──
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # ── Pipeline execution in a thread (blocking operations) ──
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

        # Stocker les collections en mémoire pour la session
        sid = result["session_id"]
        sessions[sid] = {
            "doc_collection":  result["doc_collection"],
            "rgpd_collection": result["rgpd_collection"],
            "created_at":      datetime.utcnow(),
            "top_k":           top_k,
            "threshold":       threshold,
        }
 
        # Nettoyer les clés internes avant de répondre au frontend
        payload = {k: v for k, v in result.items() if not k.startswith("_")}
        print(f"[API] Session '{sid}' créée — {len(sessions)} session(s) active(s)")
        return JSONResponse(content=payload)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pipeline : {str(e)}")

    finally:
        os.unlink(tmp_path)

@app.post("/audit_chat", tags=["Audit_chat"])
async def audit_document(
    file:      UploadFile = File(...,         description="PDF of the contract to be audited"),
    question:  str        = Form(...,         description="Compliance issue RGPD"),
    top_k:     int        = Form(5,           description="Number of chunks to retrieve"),
    threshold: float      = Form(0.35,        description="Minimum similarity score (0–1)")
):
    
    purge_expired_sessions()

    # ── Validation ──
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")

    if not question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="Le threshold doit être entre 0.0 et 1.0.")

    # ── Save upload file ──
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # ── Pipeline execution in a thread (blocking operations) ──
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_full_audit_pres(
                pdf_path  = tmp_path,
                question  = question.strip(),
                top_k     = top_k,
                threshold = threshold
            )
        )

        # Stocker les collections en mémoire pour la session
        sid = result["session_id"]
        sessions[sid] = {
            "doc_collection":  result["doc_collection"],
            "rgpd_collection": result["rgpd_collection"],
            "created_at":      datetime.utcnow(),
            "top_k":           top_k,
            "threshold":       threshold,
        }
 
        # Nettoyer les clés internes avant de répondre au frontend
        payload = {k: v for k, v in result.items() if not k.startswith("_")}
        print(f"[API] Session '{sid}' créée — {len(sessions)} session(s) active(s)")
        return JSONResponse(content=payload)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pipeline : {str(e)}")

    finally:
        os.unlink(tmp_path) 

@app.post("/chat")
async def chat_turn(req: ChatRequest):
    """
    Endpoint de suivi : reçoit une nouvelle question + l'historique complet.
 
    Le frontend envoie :
    {
      "session_id":   "abc12345",
      "question":     "Et concernant le sous-traitant ?",
      "chat_history": [
        {"role": "user",      "content": "première question"},
        {"role": "assistant", "content": "première réponse"},
        ...
      ]
    }
    """
    purge_expired_sessions()
 
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(
            404,
            "Session introuvable ou expirée (>60 min). "
            "Veuillez relancer un audit complet en uploadant à nouveau votre document."
        )
 
    if not req.question.strip():
        raise HTTPException(400, "La question ne peut pas être vide.")
 
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_chat_turn(
                session_doc_collection  = session["doc_collection"],
                session_rgpd_collection = session["rgpd_collection"],
                question     = req.question.strip(),
                chat_history = req.chat_history,
                top_k        = session["top_k"],
                threshold    = session["threshold"],
            )
        )
        return JSONResponse(content=result)
 
    except Exception as e:
        raise HTTPException(500, f"Erreur chat : {str(e)}")
 
 
@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Libère explicitement une session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"deleted": session_id}
    raise HTTPException(404, "Session introuvable.")

# ─────────────────────────────────────────────────────────────
# API RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
