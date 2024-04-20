import nltk
import pandas as pd
import pickle
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

def preprocess_dataset(csv_path, vectorizer_save_path, tfidf_save_path):
    '''
    Function to vectorize abstract of dataset and save vectorizer and TFIDF matrix
    Params:
    - csv_path (str): The path of the dataset (in CSV)
    - vectorizer_save_path (str): The path to save the TFIDF vectorizer
    - tfidf_save_path (str): The path to save the TFIDF matrix    
    '''
    # Load the dataset
    dataset = pd.read_csv(csv_path)
    # Preprocess all abstracts
    dataset['processed'] = dataset['abstract'].apply(preprocess)
    # Vectorize the dataset
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset['processed'])
    print(f'Saved preprocessed dataset back to {csv_path}.')
    with open(vectorizer_save_path, 'wb') as file:
        pickle.dump(vectorizer, file)
    print(f'Saved TFIDF Vectorizer to {vectorizer_save_path}.')
    with open(tfidf_save_path, 'wb') as file:
        pickle.dump(tfidf_matrix, file)
    print(f'Saved TFIDF Matrix to {tfidf_save_path}.')
