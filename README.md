# RGPD Auditor 🔍⚖️

Outil d'audit de conformité RGPD basé sur une architecture RAG (Retrieval-Augmented Generation).  
Il analyse un contrat PDF, le compare aux articles du RGPD via recherche vectorielle, et génère un rapport d'audit structuré grâce à Gemini.

---

## Comment ça marche

```
PDF utilisateur
      │
      ▼
 Chunking structurel ←────────────── découpage par titres de section (PyMuPDF)
 (fallback : fenêtre glissante si PDF sans mise en forme)
      │
      ▼
 Collection éphémère ChromaDB ← embedding Gemini en mémoire (zéro disque)
      │
      ├─────────────────────────────────────────┐
      ▼                                         ▼
 Recherche dans le doc            Recherche dans la base RGPD
 (chunks pertinents)              (articles de loi pertinents)
      │                                         │
      └──────────────── Query Expansion ────────┘
                   (4 reformulations RGPD générées par LLM)
                              │
                              ▼
                    Prompt structuré "Juge"
                              │
                              ▼
                    Rapport d'audit Gemini
            (verdict · non-conformités · exposition Art.83)
```

---

## Structure du projet

```
.
├── pipeline.py                        # Orchestrateur principal (point d'entrée)
├── chunking_and_embbeding_contrat.py  # Chunking PDF + embedding éphémère
├── dual_similarity_search.py          # RAG dual (doc + RGPD) avec query expansion
├── prompt_audit.py                    # Construction du prompt + appel Gemini
├── api.py                             # Serveur FastAPI (backend)
├── index.html                         # Interface web locale (frontend)
│
├── Chuncking_RGPD.py                  # Script one-shot : chunking du texte RGPD
├── Create_Chroma_BD.py                # Script one-shot : création de la base RGPD
│
└── ma_base_rgpd/                      # Base ChromaDB persistante (RGPD pré-embeddé)
```

> **`ma_base_rgpd/`** doit être construite une seule fois avant le premier audit  
> via `Chuncking_RGPD.py` puis `Create_Chroma_BD.py`.

---

## Installation

```bash
pip install pymupdf chromadb google-genai fastapi uvicorn python-multipart beautifulsoup4
```

---

## Configuration

Exporter la clé API Google avant d'exécuter :

```bash
# Linux / macOS
export API_KEY_RGPD_AUDITOR="AIzaSy..."

# Windows
set API_KEY_RGPD_AUDITOR=AIzaSy...
```

---

## Utilisation

### En ligne de commande

```python
from pipeline import run_full_audit

result = run_full_audit(
    pdf_path  = "mon_contrat.pdf",
    question  = "Les transferts de données hors UE sont-ils conformes au RGPD ?"
)

print(result["verdict"])
```

Le rapport est automatiquement sauvegardé sous `audit_<session_id>.md`.

### Via l'interface web

```bash
# 1. Lancer le serveur
python api.py
# → Check que API tourne : http://localhost:8000/health
# → Documentation Swagger : http://localhost:8000/docs

# 2. Ouvrir index.html dans le navigateur
```

---

## Exemple de rapport généré

```
## VERDICT GLOBAL
NON CONFORME

## NON-CONFORMITÉS IDENTIFIÉES

- Criticité : CRITIQUE
- Article RGPD violé : Art. 46 — Transferts internationaux
- Clause du document concernée : §2 Data Hosting and Localization
- Écart : Hébergement US sans clause contractuelle type (SCC)
- Exposition financière : jusqu'à 20 000 000 € ou 4% du CA mondial
- Recommandation : Intégrer des SCC approuvées par la Commission européenne

## RÉSUMÉ EXÉCUTIF
Le contrat expose le responsable de traitement à des risques majeurs sur 3 points :
transferts hors UE sans garanties, absence de DPA sous-traitant, et limitation
de responsabilité incompatible avec l'Art. 82.
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Extraction PDF | PyMuPDF (`fitz`) |
| Base vectorielle RGPD | ChromaDB (persistant) |
| Base vectorielle contrat | ChromaDB (éphémère, en mémoire) |
| Modèle d'embedding | `gemini-embedding-001` |
| Modèle de génération | `gemini-2.5-flash` |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML / JS vanilla |

---

## Limitations connues

- Les PDF scannés (images) ne sont pas supportés — le texte doit être sélectionnable.
- Le chunking structurel suppose une mise en forme cohérente (titres en gras).  
  Les documents sans hiérarchie typographique basculent automatiquement sur la fenêtre glissante.
- La base RGPD couvre le règlement 2016/679. Les lignes directrices CEPD et décisions nationales ne sont pas incluses.

---

## Auteur

Projet développé dans le cadre d'une formation Data Science.  
Contributions et retours bienvenus.
