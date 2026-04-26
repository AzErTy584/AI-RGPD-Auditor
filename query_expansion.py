import re
import os
import json
from google import genai  # pip install google-genai

API_KEY = os.environ.get('API_KEY_RGPD_AUDITOR')
GENERATION_MODEL = "gemini-2.5-flash"

def expand_query(question: str) -> list[str]:
    """
    Genrate 4 reformulations of the user's question to maximize recall in vector search.
        Each reformulation targets a specific RGPD angle:
        1. Obligations du responsable de traitement
        2. Droits des personnes concernées
        3. Sécurité et confidentialité des données
        4. Transferts internationaux et sous-traitance
    
        The prompt instructs the model to return a valid JSON array of reformulations without any additional text.
        In case of failure, it falls back to the original question to ensure the search can proceed.
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
        print(f"[Query Expansion] ✓ {len(all_queries)} request created:")
        return all_queries
    except Exception as e:
        print(f"[Query Expansion] ⚠ Echec ({e}) — fallback to original question")
        return [question]
