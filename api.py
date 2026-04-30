"""
╔══════════════════════════════════════════════════════════════╗
║           RGPD AUDITOR — Serveur FastAPI                     ║
║  Lance avec : python api.py                                  ║
╚══════════════════════════════════════════════════════════════╝
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

app = FastAPI(
    title       = "RGPD Auditor API",
    description = "Analyse RGPD via RAG + Gemini",
    version     = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

sessions: dict = {}
SESSION_TTL_MINUTES = 60


def purge_expired_sessions():
    now     = datetime.utcnow()
    expired = [sid for sid, s in sessions.items()
               if now - s["created_at"] > timedelta(minutes=SESSION_TTL_MINUTES)]
    for sid in expired:
        del sessions[sid]
    if expired:
        print(f"[Sessions] {len(expired)} session(s) purgée(s)")


class ChatRequest(BaseModel):
    session_id:   str
    question:     str
    chat_history: list[dict] = []


# ─────────────────────────────────────────────────────────────
# HELPER — Stockage de session
# ─────────────────────────────────────────────────────────────

def _store_session(result: dict, top_k: int, threshold: float):
    """
    Extrait les collections du résultat pipeline et les stocke en session.

    BUG CORRIGÉ (double) :
    1. Les clés sont préfixées '_' dans le pipeline (_doc_collection,
       _rgpd_collection) — il faut lire avec ce préfixe AVANT de filtrer.
    2. Ancienne version lisait result["doc_collection"] (sans '_') → KeyError.
    """
    sid = result["session_id"]
    sessions[sid] = {
        "_doc_collection":  result["_doc_collection"],  
        "_rgpd_collection": result["_rgpd_collection"],  
        "created_at":      datetime.utcnow(),
        "top_k":           top_k,
        "threshold":       threshold,
    }
    print(f"[API] Session '{sid}' créée — {len(sessions)} session(s) active(s)")
    return sid


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "sessions_active": len(sessions)}


@app.post("/audit", tags=["Rapport complet"])
async def audit_rapport(                         
    file:      UploadFile = File(...),
    question:  str        = Form(...),
    top_k:     int        = Form(5),
    threshold: float      = Form(0.35)
):
    """
    Vue rapport complet (index.html).
    Appelle run_full_audit depuis pipeline.py.
    """
    purge_expired_sessions()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés.")
    if not question.strip():
        raise HTTPException(400, "La question ne peut pas être vide.")
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(400, "Le threshold doit être entre 0.0 et 1.0.")

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

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
        _store_session(result, top_k, threshold)
        payload = {k: v for k, v in result.items() if not k.startswith("_")}
        return JSONResponse(content=payload)

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Erreur pipeline : {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/audit_chat", tags=["Chat"])
async def audit_chat(                             
    file:      UploadFile = File(...),
    question:  str        = Form(...),
    top_k:     int        = Form(5),
    threshold: float      = Form(0.35)
):
    """
    Vue chat (index2.html) — premier message.
    Appelle run_full_audit_pres depuis pipeline_pres.py.
    Retourne session_id à conserver pour les /chat suivants.
    """
    purge_expired_sessions()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Seuls les fichiers PDF sont acceptés.")
    if not question.strip():
        raise HTTPException(400, "La question ne peut pas être vide.")
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(400, "Le threshold doit être entre 0.0 et 1.0.")

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

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
        _store_session(result, top_k, threshold)
        payload = {k: v for k, v in result.items() if not k.startswith("_")}
        return JSONResponse(content=payload)

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Erreur pipeline : {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/chat", tags=["Chat"])
async def chat_turn(req: ChatRequest):
    """
    Questions de suivi — réutilise la session créée par /audit_chat.

    Body JSON attendu :
    {
      "session_id":   "abc12345",
      "question":     "Et concernant l'Art. 28 ?",
      "chat_history": [
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
    """
    purge_expired_sessions()

    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(
            404,
            "Session introuvable ou expirée (>60 min). "
            "Veuillez relancer un audit via /audit_chat."
        )
    if not req.question.strip():
        raise HTTPException(400, "La question ne peut pas être vide.")

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_chat_turn(
                session_doc_collection  = session["_doc_collection"],
                session_rgpd_collection = session["_rgpd_collection"],
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
    if session_id in sessions:
        del sessions[session_id]
        return {"deleted": session_id}
    raise HTTPException(404, "Session introuvable.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)