import os
from google import genai  # pip install google-genai

API_KEY = os.environ.get('API_KEY_RGPD_AUDITOR')
GENERATION_MODEL = "gemini-2.5-flash"


def build_audit_prompt(search_results: dict) -> str:
    """
    Build the structured 'judge' type prompt sent to Gemini. The LLM receives: question + document excerpts + RGPD articles.
    """
    doc_ctx = "\n\n".join([
        f"[Page {r['metadata'].get('page','?')} — score {r['similarity']}]\n{r['text']}"
        for r in search_results["doc_chunks"]
    ]) or "⚠ Nothing relevant found in the document."

    law_ctx = "\n\n".join([
        f"[{r['metadata'].get('article','?')} — {r['metadata'].get('titre','')}]\n{r['text']}"
        for r in search_results["law_articles"]
    ]) or "⚠ Nothing relevant found in the RGPD articles."

    return f"""Tu es un auditeur RGPD expert et rigoureux.

QUESTION DE L'UTILISATEUR :
{search_results["question"]}

EXTRAITS DU DOCUMENT SOUMIS :
{doc_ctx}

ARTICLES RGPD APPLICABLES :
{law_ctx}

MISSION — Produis un rapport d'audit structuré :

## VERDICT GLOBAL
CONFORME / NON CONFORME / PARTIELLEMENT CONFORME

## NON-CONFORMITÉS IDENTIFIÉES
Pour chaque non-conformité :
- **Criticité** : CRITIQUE / MAJEURE / MINEURE
- **Article RGPD violé** :
- **Clause du document concernée** :
- **Écart précis** (avec extrait précis de la non-conformitée) :
- **Exposition financière** : (Art. 83 — jusqu'à X€ ou X% du CA mondial)
- **Recommandation corrective** :

## POINTS NÉCESSITANT DES INFORMATIONS COMPLÉMENTAIRES
[Si les extraits sont insuffisants pour conclure sur certains points]

## RÉSUMÉ EXÉCUTIF
[5 lignes maximum]"""


def run_llm_audit(prompt: str) -> str:
    """Send the audit prompt to Gemini and return the generated report text."""
    client   = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    return response.text.strip()