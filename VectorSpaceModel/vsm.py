import os
import math
from collections import Counter
import requests

SPANISH_STOPWORDS = {
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para',
    'con', 'no', 'una', 'su', 'al', 'es', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya',
    'o', 'este', 'sí', 'porque', 'qué', 'está', 'muy', 'sin', 'sobre', 'también', 'me',
    'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno',
    'les', 'ni', 'contra', 'otros', 'fueron', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto',
    'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto',
    'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar',
    'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus',
    'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo',
    'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra',
    'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas',
    'estoy', 'estás', 'está', 'estamos', 'estáis', 'están', 'esté', 'estés', 'estemos',
    'estéis', 'estén', 'estaré', 'estarás', 'estará', 'estaremos', 'estaréis', 'estarán',
    'estaría', 'estarías', 'estaríamos', 'estaríais', 'estarían', 'estaba', 'estabas',
    'estábamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos',
    'estuvisteis', 'estuvieron', 'estuviera', 'estuvieras', 'estuviéramos', 'estuvierais',
    'estuvieran', 'estuviese', 'estuvieses', 'estuviésemos', 'estuvieseis', 'estuviesen',
    'estando', 'estado', 'estada', 'estados', 'estadas', 'estad', 'he', 'has', 'ha',
    'hemos', 'habéis', 'han', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayan', 'habré',
    'habrás', 'habrá', 'habremos', 'habréis', 'habrán', 'habría', 'habrías', 'habríamos',
    'habríais', 'habrían', 'había', 'habías', 'habíamos', 'habíais', 'habían', 'hube',
    'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron', 'hubiera', 'hubieras',
    'hubiéramos', 'hubierais', 'hubieran', 'hubiese', 'hubieses', 'hubiésemos', 'hubieseis',
    'hubiesen', 'habiendo', 'habido', 'habida', 'habidos', 'habidas', 'soy', 'eres', 'es',
    'somos', 'sois', 'son', 'sea', 'seas', 'seamos', 'seáis', 'sean', 'seré', 'serás',
    'será', 'seremos', 'seréis', 'serán', 'sería', 'serías', 'seríamos', 'seríais', 'serían',
    'era', 'eras', 'éramos', 'erais', 'eran', 'fui', 'fuiste', 'fue', 'fuimos', 'fuisteis',
    'fueron', 'fuera', 'fueras', 'fuéramos', 'fuerais', 'fueran', 'fuese', 'fueses',
    'fuésemos', 'fueseis', 'fuesen', 'siendo', 'sido', 'tengo', 'tienes', 'tiene', 'tenemos',
    'tenéis', 'tienen', 'tenga', 'tengas', 'tengamos', 'tengáis', 'tengan', 'tendré',
    'tendrás', 'tendrá', 'tendremos', 'tendréis', 'tendrán', 'tendría', 'tendrías',
    'tendríamos', 'tendríais', 'tendrían', 'tenía', 'tenías', 'teníamos', 'teníais', 'tenían',
    'tuve', 'tuviste', 'tuvo', 'tuvimos', 'tuvisteis', 'tuvieron', 'tuviera', 'tuvieras',
    'tuviéramos', 'tuvierais', 'tuvieran', 'tuviese', 'tuvieses', 'tuviésemos', 'tuvieseis',
    'tuviesen', 'teniendo', 'tenido', 'tenida', 'tenidos', 'tenidas', 'tened'
}

# Step 1: Fetch Wikipedia articles
def fetch_wikipedia_articles(titles):
    base_url = "https://es.wikipedia.org/w/api.php"
    articles = {}
    for title in titles:
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "utf8": True
        }
        response = requests.get(base_url, params=params).json()
        page = next(iter(response["query"]["pages"].values()))
        articles[title] = page["extract"]
    return articles

# Step 2: Preprocess text
def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters (except spaces and ñ)
    cleaned_text = []
    for char in text:
        if char.isalpha() or char.isspace() or char == 'ñ':
            cleaned_text.append(char)
    return ''.join(cleaned_text)

# Step 3: Tokenize text and remove stopwords
def tokenize(text):
    tokens = text.split()
    # Filter out stopwords
    tokens = [word for word in tokens if word not in SPANISH_STOPWORDS]
    return tokens

# Step 4: Build vocabulary
def build_vocabulary(docs):
    vocabulary = set()
    for doc in docs:
        vocabulary.update(doc)
    return sorted(vocabulary)

# Step 5: Compute TF-IDF
def compute_tf_idf(docs, vocabulary):
    tf = []
    idf = Counter()
    total_docs = len(docs)

    # Compute Term Frequency (TF)
    for doc in docs:
        doc_tf = Counter(doc)
        tf.append(doc_tf)

    # Compute Inverse Document Frequency (IDF)
    for word in vocabulary:
        idf[word] = math.log(total_docs / (1 + sum(1 for doc in docs if word in doc)))

    # Compute TF-IDF
    tf_idf = []
    for doc_tf in tf:
        doc_tf_idf = {}
        for word in vocabulary:
            doc_tf_idf[word] = doc_tf.get(word, 0) * idf[word]
        tf_idf.append(doc_tf_idf)
    return tf_idf

# Step 6: Compute cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in vec1)
    magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

# Step 7: Main program
def main():
    # Fetch Wikipedia articles
    titles = ["Guitarra", "Violín", "España", "México"]
    articles = fetch_wikipedia_articles(titles)

    # Preprocess and tokenize documents
    docs = []
    for title, text in articles.items():
        cleaned_text = preprocess(text)
        tokens = tokenize(cleaned_text)
        docs.append(tokens)

    # Build vocabulary
    vocabulary = build_vocabulary(docs)

    # Compute TF-IDF
    tf_idf = compute_tf_idf(docs, vocabulary)

    # Compute pairwise cosine similarity
    print("Distances between articles:")
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            similarity = cosine_similarity(tf_idf[i], tf_idf[j])
            distance = 1 - similarity
            print(f"{titles[i]} vs {titles[j]}: {distance:.4f}")

    # Display top 10 words in each document
    print("\nTop 10 words in each document:")
    for i, title in enumerate(titles):
        word_freq = Counter(docs[i])
        top_words = word_freq.most_common(10)
        print(f"{title}: {top_words}")

if __name__ == "__main__":
    main()