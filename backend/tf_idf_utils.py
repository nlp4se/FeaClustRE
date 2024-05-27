from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

def get_dense_data_array(data: List):
    tfidf_vectorizer = TfidfVectorizer()
    tf_idf_data_vector = tfidf_vectorizer.fit_transform(data)
    return tf_idf_data_vector.toarray()