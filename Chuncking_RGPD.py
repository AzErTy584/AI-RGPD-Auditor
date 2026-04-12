from bs4 import BeautifulSoup
import pandas as pd

def chunk_text(rgpd_text, nb_articles=99):
    soup = BeautifulSoup(rgpd_text, 'html.parser')
    data = []  # Créer une liste pour stocker les données
    
    for i in range(1, nb_articles + 1):
        article_id = f'L_2016119EN.01000101.art_{i}'
        try:
            articles = soup.find(id=article_id)
            if articles is None:
                # print(f"Article {article_id} non trouvé")
                continue
                
            articles_number = articles.find('p', class_='oj-ti-art')
            articles_name = articles.find('p', class_='oj-sti-art')
            articles_contents = articles.find_all('p', class_='oj-normal')
            articles_content = "\n".join([p.text for p in articles_contents])
            
            # Vérifier que les éléments existent avant d'accéder à .text
            numero = articles_number.text if articles_number else ""
            nom = articles_name.text if articles_name else ""
            
            # Ajouter à la liste
            data.append({
                'Article': numero, 
                'Nom': nom, 
                'Contenu': articles_content
            })
            print(f"Article {i} traité")
            
        except Exception as e:
            print(f"Erreur lors du traitement de l'article {article_id}: {e}")
            continue
    
    # Créer le DataFrame à partir de la liste
    df = pd.DataFrame(data)
    return df

with open('RGPD_text.html', 'r') as file:
    rgpd_text = file.read()

df = chunk_text(rgpd_text)
# print(df.head())  # Afficher les premières lignes du DataFrame
df.to_csv('RGPD_articles.csv', index=False)