__author__ = "Rodrigo Gael Guzmán Alburo"
__email__ = "gael.a24@outlook.com"

# BIBLIOTECAS NECESARIAS
# pip install nltk requests scikit-learn

import string
import nltk
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Descarga recursos de NLTK
nltk.download("punkt_tab")
nltk.download("stopwords")


# Paso 1: Obtiene artículos de Wikipedia
def fetch_wikipedia(titles):
    """Función que obtiene los artículos de Wikipedia para una lista de títulos."""

    base_url = "https://es.wikipedia.org/w/api.php"
    articles = {}
    for title in titles:
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "utf8": True,
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        articles[title] = page["extract"]
    return articles


# Paso 2: Preprocesa el texto
def preprocess(text):
    """Función que preprocesa el texto convertiendolo a minúsculas y elimina signos de puntuación y stopwords."""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    stop_words = set(stopwords.words("spanish"))
    tokens = word_tokenize(text, language="spanish")
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)  # Regresa el texto preprocesado como una cadena


# Programa principal
def main():
    """Función principal del programa."""

    # Obtiene los articulos de Wikipedia:
    titles = ["Guitarra", "Violín", "México", "España"]
    articles = fetch_wikipedia(titles)

    # Preprocesa los articulos:
    preprocessed_articles = [preprocess(articles[title]) for title in titles]

    # Calcula vectores TF-IDF:'
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_articles)

    # Calcula similitud coseno entre los vectores:
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Muestra la similitud entre los artículos:
    print("\n")
    print("Matriz de similitud coseno entre los artículos: ")
    print("Guitarra     Violín\tMéxico\t   España")
    print(cosine_sim)
    print("\n")
    print(
        f"Similitud entre 'Guitarra' y 'Violín':\t {round(cosine_sim[0, 1] * 100, 2)}%"
    )
    print(
        f"Similitud entre 'Guitarra' y 'México':\t {round(cosine_sim[0, 2] * 100, 2)}%"
    )
    print(
        f"Similitud entre 'Guitarra' y 'España':\t {round(cosine_sim[0, 3] * 100, 2)}%"
    )
    print(f"Similitud entre 'Violín' y 'México':\t {round(cosine_sim[1, 2] * 100, 2)}%")
    print(f"Similitud entre 'Violín' y 'España':\t {round(cosine_sim[1, 3] * 100, 2)}%")
    print(f"Similitud entre 'México' y 'España':\t {round(cosine_sim[2, 3] * 100, 2)}%")

    print("\n")
    print("10 palabras más frecuentes en cada artículo:")
    # Mostrar las 10 palabras más frecuentes de cada artículo:
    feature_names = (
        vectorizer.get_feature_names_out()
    )  # Obtiene las palabras del vocabulario
    for i, title in enumerate(titles):
        tfidf_scores = tfidf_matrix[i].toarray().flatten()
        top_words_indices = tfidf_scores.argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_words_indices]
        print(f"{title}: {top_words}")


if __name__ == "__main__":
    main()
