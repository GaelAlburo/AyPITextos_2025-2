__author__ = "Rodrigo Gael Guzmán Alburo"
__email__ = "gael.a24@outlook.com"


import re  # Expresiones regulares para el preprocesamiento de texto
from collections import (
    defaultdict,
    Counter,
)  # Estructuras de datos para almacenar modelos de lenguaje y conteos
import math  # Funciones matemáticas para cálculos de probabilidad


class LanguageClassifier:
    """Clasificador de idioma simple basado en modelos de lenguaje n-grama."""

    def __init__(self, n_values=[2, 3, 4, 5]):
        self.n_values = n_values
        self.language_models = defaultdict(
            lambda: defaultdict(Counter)
        )  # Diccionario anidado de Counter para almacenar n-gramas por idioma

    def preprocess_text(self, text):
        """Preprocesa el texto para convertirlo a minúsculas y eliminar caracteres no alfabéticos."""

        # Convertir a minúsculas y eliminar caracteres no alfabéticos
        text = text.lower()
        text = re.sub(r"[^a-záéíóúñ]", " ", text)
        return text

    def get_ngrams(self, text, n):
        """Genera una lista de n-gramas a partir de un texto."""

        # Generar n-gramas
        n_grams = [text[i : i + n] for i in range(len(text) - n + 1)]
        return n_grams

    def train(self, language, text):
        """Entrena el clasificador con un texto de un idioma específico."""

        text = self.preprocess_text(text)  # Preprocesa el texto
        for n in self.n_values:
            ngrams = self.get_ngrams(text, n)
            self.language_models[language][n].update(ngrams)

    def calculate_probability(self, text, language):
        """Calcula la probabilidad de que un texto dado pertenezca a un idioma específico. Utiliza la Ley de Zipf."""

        text = self.preprocess_text(text)  # Preprocesa el texto
        log_prob = 0.0

        for n in self.n_values:
            ngrams = self.get_ngrams(text, n)  # Genera n-gramas

            for ngram in ngrams:
                if ngram in self.language_models[language][n]:
                    # Obtiene el rango del n-grama
                    rank = self.get_rank(language, n, ngram)

                    # Estima la probabilidad usando la Ley de Zipf
                    prob = 1 / (rank**1)  # s = 1

                else:
                    # Si el n-grama no está presente, asigna una probabilidad uniforme
                    prob = 1 / (len(self.language_models[language][n]) + 1)

                log_prob += math.log(prob)
        return log_prob

    def get_rank(self, language, n, ngram):
        """Obtiene el rango de un n-grama en la lista ordenada de n-gramas de un idioma."""

        # Obtiene todos los n-gramas de tamaño n para el idioma dado
        ngram_counts = self.language_models[language][n]

        # Ordena los n-gramas por frecuencia en orden descendente
        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)

        # Encuentra el rango del n-grama objetivo
        for rank, (ng, cnt) in enumerate(sorted_ngrams, start=1):
            if ng == ngram:
                return rank

        # Si el n-grama no se encuentra, devuelve un rango fuera de la lista ordenada
        return len(sorted_ngrams) + 1

    def classify(self, text):
        """Clasifica un texto dado en uno de los idiomas entrenados."""

        best_language = None
        best_prob = -float(
            "inf"
        )  # Inicializa la mejor probabilidad con un valor infinito negativo
        for language in self.language_models:
            prob = self.calculate_probability(text, language)
            if prob > best_prob:
                best_prob = prob
                best_language = language
        return best_language


if __name__ == "__main__":
    # Datos de entrenamiento (pueden ser más extensos en un caso real)
    english_text = """
        The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet. 
        English is a West Germanic language that was first spoken in early medieval England. It is now the most widely 
        used language in the world, serving as a global lingua franca. English is spoken by millions of people as their 
        first language and by many more as a second language. It has a rich vocabulary and a relatively simple grammar 
        structure compared to other languages. English literature includes works by famous authors such as William Shakespeare, 
        Jane Austen, and Mark Twain. The language continues to evolve, with new words and phrases being added regularly.
        """
    spanish_text = """
        El rápido zorro marrón salta sobre el perro perezoso. Esta oración contiene todas las letras del alfabeto español. 
        El español es una lengua romance que se originó en la península ibérica. Hoy en día, es uno de los idiomas más hablados 
        en el mundo, con cientos de millones de hablantes nativos. El español es la lengua oficial en más de veinte países y 
        es ampliamente estudiado como segunda lengua. Tiene una gramática rica y una gran variedad de expresiones idiomáticas. 
        La literatura en español incluye obras de autores famosos como Miguel de Cervantes, Gabriel García Márquez y Pablo Neruda. 
        El idioma sigue evolucionando, incorporando nuevas palabras y modismos con el tiempo.
        """

    # Crear y entrenar el clasificador
    classifier = LanguageClassifier()
    classifier.train("Inglés", english_text)
    classifier.train("Español", spanish_text)

    # Solicitar texto al usuario
    user_input = input(
        "Introduce un texto para clasificar (o presiona Enter para usar un texto predeterminado): "
    ).strip()

    # Si el usuario no proporciona texto, usar uno predeterminado
    if not user_input:
        user_input = "This is a test text to classify."
        print(f"\nNo se proporcionó texto. Usando texto predeterminado: '{user_input}'")

    # Clasificar el texto
    result = classifier.classify(user_input)
    print(f"\nIdioma del texto: {result}")
