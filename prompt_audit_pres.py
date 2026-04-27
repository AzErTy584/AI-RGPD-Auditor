"""
╔══════════════════════════════════════════════════════════════════╗
║   RGPD AUDITOR — Construction des prompts + appel Gemini        ║
╠══════════════════════════════════════════════════════════════════╣
║  Deux prompts distincts :                                        ║
║  • build_audit_prompt_first  → premier diagnostic (sans histo.) ║
║  • build_audit_prompt_second → tour de chat (avec historique)   ║
║                                                                  ║
║  Utilisé par pipeline_pres.py via :                              ║
║    from prompt_audit_pres import build_audit_prompt, run_llm_audit║
╚══════════════════════════════════════════════════════════════════╝
"""
 
import os
from google import genai
 
API_KEY          = os.environ.get('API_KEY_RGPD_AUDITOR')
GENERATION_MODEL = "gemini-2.5-flash"
 
 
# ─────────────────────────────────────────────────────────────
# HELPER — Formatage de l'historique
# ─────────────────────────────────────────────────────────────
 
def _format_history(chat_history: list) -> str:
    """
    Convertit la liste Python d'historique en dialogue lisible pour le LLM.
 
    Entrée (liste de dicts) :
        [
          {"role": "user",      "content": "Les transferts sont-ils conformes ?"},
          {"role": "assistant", "content": "❌ Non-conforme. Clause 2 viole Art. 46..."},
          ...
        ]
 
    Sortie (texte structuré) :
        [UTILISATEUR] Les transferts sont-ils conformes ?
        [AUDITEUR]    ❌ Non-conforme. Clause 2 viole Art. 46...
 
    Sans ce formatage, Gemini reçoit une liste Python brute illisible :
    "[{'role': 'user', 'content': '...'}]" ce qui dégrade fortement la qualité
    des réponses contextualisées.
    """
    if not chat_history:
        return "(aucun échange précédent)"
 
    lines = []
    for msg in chat_history:
        role    = msg.get("role", "")
        content = msg.get("content", "").strip()
        label   = "[UTILISATEUR]" if role == "user" else "[AUDITEUR]   "
        # On tronque les réponses longues pour ne pas saturer la fenêtre de contexte
        if len(content) > 800:
            content = content[:800] + "…[tronqué]"
        lines.append(f"{label} {content}")
 
    return "\n\n".join(lines)
 
 
# ─────────────────────────────────────────────────────────────
# PROMPT 1 — PREMIER DIAGNOSTIC (pas d'historique)
# ─────────────────────────────────────────────────────────────
 
def build_audit_prompt_first(search_results: dict) -> str:
    """
    Prompt utilisé pour la PREMIÈRE question de l'utilisateur.
 
    Objectif : diagnostic direct et concis, style "DPO expert en réunion".
    Pas de contexte conversationnel — on part de zéro.
    """
    doc_ctx = "\n\n".join([
        f"[Page {r['metadata'].get('page','?')} — Clause : {r['metadata'].get('title','?')} — score {r['similarity']}]\n{r['text']}"
        for r in search_results["doc_chunks"]
    ]) or "⚠ Aucun extrait pertinent trouvé dans le document."
 
    law_ctx = "\n\n".join([
        f"[{r['metadata'].get('article','?')} — {r['metadata'].get('titre','')} — score {r['similarity']}]\n{r['text']}"
        for r in search_results["law_articles"]
    ]) or "⚠ Aucun article RGPD pertinent trouvé."
 
    return f"""# RÔLE
Tu es un Expert DPO spécialisé en audit rapide de conformité RGPD.
Ton objectif : fournir un diagnostic percutant, ultra-concis et directement actionnable.
 
# RÈGLES STRICTES
1. **Style direct** : Pas d'introduction ni de conclusion fleuries. Va droit au but.
2. **Concision** : Listes à puces. Une idée par ligne.
3. **Références uniquement** : Ne retranscris jamais le texte des clauses ou des articles. Cite seulement leurs références (ex : "Art. 46 RGPD", "Clause 2 du contrat").
4. **Verdict en tête** : Commence toujours par ✅ Conforme, ⚠️ Risque, ou ❌ Non-conforme.
5. **Markdown** : Utilise les titres `###`, le gras `**`, et les listes `-` pour structurer.
 
# QUESTION
{search_results["question"]}
 
# EXTRAITS DU CONTRAT (via RAG)
{doc_ctx}
 
# ARTICLES RGPD APPLICABLES (via RAG)
{law_ctx}
 
# FORMAT DE RÉPONSE (à respecter impérativement)
---
### 🗨️ Diagnostic
[Statut + 2-3 phrases max]
 
### ⚖️ Fondement juridique
- **Contrat** : [Références des clauses impactées]
- **RGPD** : [Numéros des articles applicables]
 
### 💡 Action requise
- [Ce qu'il faut modifier ou ajouter — ou "Rien à signaler"]
---"""
 
 
# ─────────────────────────────────────────────────────────────
# PROMPT 2 — TOUR DE CHAT (avec historique)
# ─────────────────────────────────────────────────────────────
 
def build_audit_prompt_second(chat_history: list, search_results: dict) -> str:
    """
    Prompt utilisé pour les QUESTIONS DE SUIVI (2e, 3e question, etc.).
 
    Différences vs le premier prompt :
    - L'historique de la conversation est injecté sous forme lisible
      (formaté par _format_history, pas dumped comme liste Python brute)
    - Le LLM est explicitement invité à s'appuyer sur le contexte précédent
    - Le ton est plus "conversation continue" que "diagnostic initial"
    - Les nouvelles sources RAG sont présentées comme une mise à jour,
      pas comme un contexte de départ
    """
    doc_ctx = "\n\n".join([
        f"[Page {r['metadata'].get('page','?')} — Clause : {r['metadata'].get('title','?')} — score {r['similarity']}]\n{r['text']}"
        for r in search_results["doc_chunks"]
    ]) or "⚠ Aucun extrait pertinent trouvé dans le document."
 
    law_ctx = "\n\n".join([
        f"[{r['metadata'].get('article','?')} — {r['metadata'].get('titre','')} — score {r['similarity']}]\n{r['text']}"
        for r in search_results["law_articles"]
    ]) or "⚠ Aucun article RGPD pertinent trouvé."
 
    # Formatage lisible de l'historique — c'est ici que le bug était
    history_formatted = _format_history(chat_history)
 
    return f"""# RÔLE
Tu es un Expert DPO en session d'audit RGPD. Tu poursuis une conversation déjà entamée.
 
# HISTORIQUE DE LA CONVERSATION
{history_formatted}
 
# NOUVELLE QUESTION
{search_results["question"]}
 
# SOURCES MISES À JOUR (via RAG — spécifiques à cette question)
**Contrat :**
{doc_ctx}
 
**Articles RGPD :**
{law_ctx}
 
# INSTRUCTIONS
- Réponds à la NOUVELLE QUESTION en priorité.
- Appuie-toi sur l'HISTORIQUE uniquement si cela apporte un éclairage pertinent (ex: contradiction avec une réponse précédente, précision d'un point déjà évoqué).
- Ne répète pas ce qui a déjà été dit sauf si nécessaire pour la clarté.
- Garde le même format que les réponses précédentes (Diagnostic / Fondement / Action).
- Même style : direct, concis, références uniquement, Markdown structuré.
 
# FORMAT DE RÉPONSE
---
### 🗨️ Diagnostic
[Statut + 2-3 phrases max]
 
### ⚖️ Fondement juridique
- **Contrat** : [Références des clauses]
- **RGPD** : [Numéros des articles]
 
### 💡 Action requise
- [Action corrective — ou "Rien à signaler"]
---"""
 
 
# ─────────────────────────────────────────────────────────────
# WRAPPER PUBLIC — appelé par pipeline_pres.py
# ─────────────────────────────────────────────────────────────
 
def build_audit_prompt(search_results: dict, chat_history=None) -> str:
    """
    Aiguillage entre les deux prompts selon la présence d'un historique.
 
    Appelé depuis pipeline_pres.py :
        prompt = build_audit_prompt(results, chat_history)
 
    - chat_history=None  ou []  → premier diagnostic (build_audit_prompt_first)
    - chat_history=[{...}, ...] → tour de chat     (build_audit_prompt_second)
    """
    if chat_history:
        return build_audit_prompt_second(chat_history, search_results)
    else:
        return build_audit_prompt_first(search_results)
 
 
# ─────────────────────────────────────────────────────────────
# APPEL GEMINI
# ─────────────────────────────────────────────────────────────
 
def run_llm_audit(prompt: str) -> str:
    """Envoie le prompt à Gemini et retourne le texte du rapport."""
    client   = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    return response.text.strip()