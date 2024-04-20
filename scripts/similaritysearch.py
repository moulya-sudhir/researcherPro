from .utils import preprocess, cosine_similarity
import pickle
import pandas as pd

def calculate_similarity(question, vectorizer_path, tfidf_path):
    '''
    Function to process the question and calculate similarity scores
    Params:
    - question (str): The question to ask
    - vectorizer_path (str): The pickle path where TFIDF Vectorizer is stored
    - tfidf_path (str): The pickle path where the TFIDF Matrix is stored

    Returns:
    - similarity_scores: The similarity scores between question and abstracts from dataset
    '''
    # Preprocess the question
    preprocessed_question = preprocess(question)
    # Load vectorizer
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    # Load TFIDF matrix
    with open(tfidf_path, 'rb') as file:
        tfidf_matrix = pickle.load(file)
    # Vectorize question
    query_vector = vectorizer.transform([preprocessed_question])
    # Calculate similarities
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    return similarity_scores

def get_top_n_documents(n, question, data_path, vectorizer_path, tfidf_path):
    '''
    Function that returns top n similar articles to the question
    Params:
    - n (int): The number indicating how many similar articles to return
    - question (str): The question to ask
    - data_path (str): The path to the CSV dataset
    - vectorizer_path (str): The pickle path where TFIDF Vectorizer is stored
    - tfidf_path (str): The pickle path where the TFIDF Matrix is stored

    Returns Dict[str]: A dictionary with keys as title and values as abstracts of similar articles.
    '''
    # Calculate similarity scores
    similarity_scores = calculate_similarity(question, vectorizer_path, tfidf_path)
    # Get top n documents
    top_n_indices = similarity_scores.argsort()[0][-n:][::-1]  # Get indices of top n scores
    # Load dataset
    dataset = pd.read_csv(data_path)
    top_n_similar = dataset.loc[top_n_indices, ['title','abstract']].values
    top_n_similar = dict(top_n_similar)
    return top_n_similar