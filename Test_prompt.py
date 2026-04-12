from Similarity_search import contrat_collection,dual_similarity_search, API_KEY
from google import genai

def build_judge_prompt(search_results: dict) -> str:

    doc_context = "\n\n".join([
        f"[Extrait doc - similarité {r['similarity']}]\n{r['text']}"
        for r in search_results["doc_chunks"]
    ]) or "Aucun extrait pertinent trouvé dans le document."

    law_context = "\n\n".join([
        f"[{r['metadata'].get('article', 'Article ?')}]\n{r['text']}"
        for r in search_results["law_articles"]
    ]) or "Aucun article RGPD pertinent trouvé."

    return f"""Tu es un auditeur RGPD expert et rigoureux.

QUESTION DE L'UTILISATEUR :
{search_results["question"]}

CE QUE DIT LE DOCUMENT SOUMIS :
{doc_context}

CE QUE DIT LA LOI (Articles RGPD) :
{law_context}

MISSION :
Produis un rapport d'audit structuré ainsi :

## VERDICT GLOBAL
CONFORME / NON CONFORME / PARTIELLEMENT CONFORME

## NON-CONFORMITÉS IDENTIFIÉES
Pour chaque non-conformité :
- Criticité : CRITIQUE / MAJEURE / MINEURE
- Article RGPD violé :
- Clause du document concernée :
- Écart précis :
- Exposition financière : (Art. 83 — jusqu'à X€ ou X% CA mondial)
- Recommandation corrective :

## POINTS NÉCESSITANT DES INFORMATIONS COMPLÉMENTAIRES
[Si extraits insuffisants pour conclure sur certains points]

## RÉSUMÉ EXÉCUTIF (3 lignes max)"""


def contact_gemini(prompt: str) -> str:
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents = prompt
    )
    return response.text.strip()

def main():
    # Exemple de question
    question = "Quels sont les risques de non-conformité liés à la collecte de données personnelles ?"
    
    # Simulate search results (à remplacer par une vraie recherche)
    search_results = dual_similarity_search(
        question=question,
        temp_collection=contrat_collection,  # ← à remplacer par la collection éphémère du doc utilisateur
        top_k_doc=7,
        top_k_law=10,
        score_threshold=0.35
        )

    prompt = build_judge_prompt(search_results)
    print("Prompt envoyé à Gemini :\n", prompt)
    verdict = contact_gemini(prompt)
    print("Verdict de conformité :", verdict)

if __name__ == "__main__":
    main()