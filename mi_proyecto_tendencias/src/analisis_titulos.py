import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

# Descargar recursos de NLTK si no los tienes (se ejecutará solo la primera vez)
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.word_tokenize("ejemplo")
except LookupError:
    nltk.download('punkt')
try:
    WordNetLemmatizer().lemmatize('running')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Cargar el modelo de spaCy para español (para lematización)
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Descargando modelo de spaCy para español...")
    import spacy.cli
    spacy.cli.download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

# Inicializar el lematizador de NLTK (si prefieres este a spaCy)
lemmatizer = WordNetLemmatizer()

# Definir el conjunto de stop words en español
stop_words_es = set(stopwords.words('spanish'))

def preprocess_text(text):
    """Función para preprocesar un texto."""
    if isinstance(text, str):
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar signos de puntuación y caracteres especiales
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenizar el texto
        tokens = nltk.word_tokenize(text)
        # Eliminar stop words
        tokens = [token for token in tokens if token not in stop_words_es]
        return tokens
    return []

def lemmatize_spacy(tokens):
    """Lematizar tokens usando spaCy."""
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

def lemmatize_nltk(tokens):
    """Lematizar tokens usando NLTK."""
    return [lemmatizer.lemmatize(token) for token in tokens]

def analyze_titles_from_csv(file_path, lemmatization_method='spacy'):
    """
    Lee títulos desde un archivo CSV (una línea por título),
    preprocesa el texto, lematiza y cuenta la frecuencia de las palabras.

    Args:
        file_path (str): La ruta al archivo CSV.
        lemmatization_method (str, optional): El método de lematización a usar ('spacy' o 'nltk').
                                             Por defecto es 'spacy'.

    Returns:
        pandas.DataFrame: Un DataFrame con las palabras y sus frecuencias.
    """
    try:
        # Leer el archivo CSV, asumiendo que cada título está en una fila sin encabezado
        df = pd.read_csv(file_path, header=None, names=['titulo'])
    except FileNotFoundError:
        print(f"Error: El archivo '{file_path}' no fue encontrado.")
        return None

    all_tokens = []
    for title in df['titulo']:
        tokens = preprocess_text(title)
        if lemmatization_method == 'spacy':
            lemmatized_tokens = lemmatize_spacy(tokens)
        elif lemmatization_method == 'nltk':
            lemmatized_tokens = lemmatize_nltk(tokens)
        else:
            lemmatized_tokens = tokens  # Sin lematización
        all_tokens.extend(lemmatized_tokens)

    # Contar la frecuencia de cada palabra
    word_counts = Counter(all_tokens)

    # Crear un DataFrame con las palabras y sus frecuencias
    frequency_df = pd.DataFrame(word_counts.items(), columns=['palabra', 'frecuencia'])
    frequency_df = frequency_df.sort_values(by='frecuencia', ascending=False).reset_index(drop=True)

    return frequency_df

# Ejemplo de uso:
if __name__ == "__main__":
    archivo_csv = 'data/raw/titulos.csv'  # Ruta al archivo CSV (asumiendo la estructura de carpetas)
    frecuencias = analyze_titles_from_csv(archivo_csv, lemmatization_method='spacy') # Puedes cambiar a 'nltk' o None

    if frecuencias is not None:
        print("Frecuencia de palabras (lematizadas):")
        print(frecuencias.head(20)) # Mostrar las 20 palabras más frecuentes

        # Aquí puedes usar 'frecuencias' para construir tu matriz de tendencias.
        # Por ejemplo, puedes seleccionar las palabras más frecuentes como las columnas de tu matriz.
        top_n_palabras = frecuencias['palabra'].head(50).tolist() # Obtener las 50 palabras más frecuentes

        # Crear la matriz de tendencias (DataFrame de pandas)
        try:
            titulos_df = pd.read_csv(archivo_csv, header=None)[0]
            matriz_tendencias = pd.DataFrame(index=titulos_df)
            for palabra in top_n_palabras:
                matriz_tendencias[palabra] = matriz_tendencias.index.str.contains(palabra, case=False).astype(int)

            print("\nMatriz de Tendencias (ejemplo con las 50 palabras más frecuentes):")
            print(matriz_tendencias.head())

            # Guardar la matriz de tendencias a un nuevo CSV (opcional)
            matriz_tendencias.to_csv('data/processed/matriz_tendencias.csv')
            frecuencias.to_csv('data/processed/palabras_frecuentes.csv', index=False)
            print("\nMatriz de tendencias guardada en 'data/processed/matriz_tendencias.csv'")
            print("Frecuencia de palabras guardada en 'data/processed/palabras_frecuentes.csv'")

        except FileNotFoundError:
            print(f"Error: No se pudo leer el archivo de títulos para crear la matriz: '{archivo_csv}'")