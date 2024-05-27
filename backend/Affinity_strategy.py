from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from .tf_idf_utils import get_dense_data_array
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
        dense_data_array = get_dense_data_array(data=data)
        model = AgglomerativeClustering(n_clusters=None, 
                                        linkage='complete', 
                                        distance_threshold=0,
                                        metric="cosine")
        model.fit(dense_data_array)
        model_info = {
            'affinity': 'TF-IDF Cosine',
            'model': model,
            'labels': data
        }
        
        file_name = 'tf_idf_cosine_agglomerative_model.pkl'
        file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
        joblib.dump(model_info, file_path)
        return file_path

class TfIdfEuclideanAffinity(AffinityStrategy):
    def compute_affinity(self, data: List): 
            dense_data_array = get_dense_data_array(data)
            model = AgglomerativeClustering(n_clusters=None, 
                                            linkage='average', 
                                            distance_threshold=0,
                                            metric="euclidean")
            model.fit(dense_data_array)
            model_info = {
                'affinity': 'TF-IDF Euclidean',
                'model': model,
                'labels': data
            }
            
            file_name = 'tf_idf_euclidean_agglomerative_model.pkl'
            file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
            joblib.dump(model_info, file_path)
            return file_path

class BERTEmbeddingAffinity(AffinityStrategy):
    def compute_affinity(self, data: List):
        return None

    