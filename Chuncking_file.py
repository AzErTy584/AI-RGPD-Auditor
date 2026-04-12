from spire.pdf.common import *
from spire.pdf import *
from bs4 import BeautifulSoup
import pandas as pd


# Nécessitera amélioration pour s'adapter à la structure du contrat, notamment pour extraire les articles et leurs contenus de manière plus précise. 
# Actuellement, la fonction cherche un élément spécifique qui pourrait ne pas correspondre à la structure réelle des types de contrat étudié.
# => Nécessitera amélioration du script de chuncking des contrats

def convert_pdf_to_html(input_pdf_path, output_html_path):
    doc = PdfDocument()
    try:
        doc.LoadFromFile(input_pdf_path)
        doc.SaveToFile(output_html_path, FileFormat.HTML)
        print("Conversion successful!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        doc.Close()

# Usage
# path_to_output = "Industrial_Compliance_Test_Contract.html"
# convert_pdf_to_html("Industrial_Compliance_Test_Contract.pdf", "path_to_output.html")

def chunk_text(contract_text, nb_articles=99):
    soup = BeautifulSoup(contract_text, 'html.parser')
    data = []  # Créer une liste pour stocker les données
    nb_articles = 0
    SPAM_TEXT = ["Evaluation Warning : The document was created with Spire.PDF for Python."]
                 

    # Définition ultérieur d'un cadre de recherche plus large pour s'adapter à la structure du contrat
    all_tspan = soup.find_all('tspan')
    nb_articles_save = 0
    contenu_save = ""
    titre = ""
    for tspan in all_tspan :

        if nb_articles_save != nb_articles:
                data.append({
                    'Titre': title_pass if title_pass else None,
                    'Contenu': contenu_save if contenu_save else None
                })
                print(f"Article {nb_articles} traité")
                nb_articles_save = nb_articles
                contenu_save = ""
        if tspan and 'font-weight="bold"' and 'font-size="14"' in str(tspan):
            title_pass = titre
            titre = tspan.text
            nb_articles += 1
        elif nb_articles_save == nb_articles and nb_articles > 0:
            if tspan.text not in SPAM_TEXT:
                contenu = tspan.text
                contenu_save += " "+contenu
    df = pd.DataFrame(data)[1:]  # Ignorer la première ligne qui est vide
    return df


with open('Industrial_Compliance_Test_Contract.html', 'r') as file:
    df = chunk_text(file.read())
    df.to_csv('Industrial_Compliance_Test_Contract.csv', index=False)