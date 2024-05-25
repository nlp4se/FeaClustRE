from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import os
import joblib

MODEL_DIRECTORY_PATH = 'static'
class AffinityStrategy():
    @abstractmethod
    def compute_affinity(self, data: List):
        pass
    
class LevenshteinAffinity(AffinityStrategy):
    def compute_affinity(self, data: List):
        return None
    
class TfIdfCosineAffinity(AffinityStrategy):
    def compute_affinity(self, data: List): 
        tfidf_vectorizer = TfidfVectorizer()
        tf_idf_data_vector = tfidf_vectorizer.fit_transform(data)
        dense_data_array = tf_idf_data_vector.toarray()
        
        model = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=0)
        model.fit(dense_data_array)
        file_name = 'tf_idf_cosine_agglomerative_model.pkl'
        file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
        joblib.dump(model, file_path)
        return file_path

class TfIdfEuclideanAffinity(AffinityStrategy):
    def compute_affinity(self, data: List):
        return None

class BERTEmbeddingAffinity(AffinityStrategy):
    def compute_affinity(self, data: List):
        return None

    