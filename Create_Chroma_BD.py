import chromadb
from chromadb.utils import embedding_functions

mes_chunks = [
    "Subject-matter and objectives : 1. This Regulation lays down rules relating to the protection of natural persons with regard to the processing of personal data and rules relating to the free movement of personal data. This Regulation protects fundamental rights and freedoms of natural persons and in particular their right to the protection of personal data. The free movement of personal data within the Union shall be neither restricted nor prohibited for reasons connected with the protection of natural persons with regard to the processing of personal data.",
    "Material scope : 1. This Regulation applies to the processing of personal data wholly or partly by automated means and to the processing other than by automated means of personal data which form part of a filing system or are intended to form part of a filing system. 2.(a)  This Regulation does not apply to the processing of personal data: in the course of an activity which falls outside the scope of Union law; (b)  by the Member States when carrying out activities which fall within the scope of Chapter 2 of Title V of the TEU; (c)  by a natural person in the course of a purely personal or household activity; (d)  by competent authorities for the purposes of the prevention, investigation, detection or prosecution of criminal offences or the execution of criminal penalties, including the safeguarding against and the prevention of threats to public security. 3.For the processing of personal data by the Union institutions, bodies, offices and agencies, Regulation (EC) No 45/2001 applies. Regulation (EC) No 45/2001 and other Union legal acts applicable to such processing of personal data shall be adapted to the principles and rules of this Regulation in accordance with Article 98. 4.This Regulation shall be without prejudice to the application of Directive 2000/31/EC, in particular of the liability rules of intermediary service providers in Articles 12 to 15 of that Directive.",
    "Territorial scope : 1. This Regulation applies to the processing of personal data in the context of the activities of an establishment of a controller or a processor in the Union, regardless of whether the processing takes place in the Union or not. 2. This Regulation applies to the processing of personal data of data subjects who are in the Union by a controller or processor not established in the Union, where the processing activities are related to: a)  the offering of goods or services, irrespective of whether a payment of the data subject is required, to such data subjects in the Union; or (b)  the monitoring of their behaviour as far as their behaviour takes place within the Union. 3.This Regulation applies to the processing of personal data by a controller not established in the Union, but in a place where Member State law applies by virtue of public international law.",
    ]

# [Vecteur (Signature mathématique),Document (Texte réel),Métadonnées (Source)]

# 1. On définit où la base de données sera enregistrée
client = chromadb.PersistentClient(path="./ma_base_rgpd")

# 2. On configure l'embedding (Indispensable pour que Chroma parle la même langue que Gemini)
# Vous utilisez l'API Google pour transformer le texte en vecteurs
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key="AIzaSyAkoTVf7KkQkvRXwNThmIdaEMk0Q8gNzaA",
    model_name="gemini-embedding-001"
)

# 3. On crée (ou récupère) la collection
collection = client.get_or_create_collection(
    name="lois_rgpd", 
    embedding_function=google_ef
)

# Imaginons que 'mes_chunks' soit votre liste de textes déjà préparée
# Chroma DB ajoute automatiquement les embeddings à partir de vos textes grâce à la fonction d'embedding configuré précédemment (il faut pas explicitement coder d'enregistrer les vecteurs, c'est automatique)
collection.upsert(
    documents=mes_chunks, # Vos textes
    metadatas=[{"source": "RGPD", "article": i+1} for i in range(len(mes_chunks))], # Métadonnées personnalisées
    ids=[f"id_{i+1}" for i in range(len(mes_chunks))] # Identifiants uniques
)