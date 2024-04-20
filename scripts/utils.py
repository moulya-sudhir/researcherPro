import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

def preprocess(text):
    '''
    Function to preprocess text and return lemmatized tokens
    Params: 
    - text (str): Text to process

    Returns lemmatized string
    '''
    # Tokenization
    words = word_tokenize(text.lower())

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)

def vectorize_texts(texts):
    '''
    Function to vectorizes a list of strings
    Params: 
    - texts (List): A list of strings

    Returns vectorized sparse matrix
    '''
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)

def calculate_similarity(vec1, vec2):
    '''
    Function to compute cosine similarity between two vectors
    Params:
    - vec1 : Vectorized text in sparse representation
    - vec2 : Vectorized text in sparse representation

    Return a similarity score between 0 and 1 (0 - low similarity, 1 - high similarity)

    '''
    return cosine_similarity(vec1, vec2)