from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from .tf_idf_utils import get_dense_data_array
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel
from scipy.sparse import csr_matrix
import os
import joblib
import torch

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
            'affinity': 'TF-IDF Cosine Complete',
            'model': model,
            'labels': data
        }

        file_name = 'tf_idf_cosine_complete.pkl'
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
            'affinity': 'TF-IDF Euclidean Average',
            'model': model,
            'labels': data
        }

        file_name = 'tf_idf_euclidean_average.pkl'
        file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
        joblib.dump(model_info, file_path)
        return file_path


class BERTCosineEmbeddingAffinity(AffinityStrategy):
    def compute_affinity(self, data: List):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
        max_len = max(len(sent) for sent in tokenized_sentences)
        padded_sentences = [sent + [tokenizer.pad_token_id] * (max_len - len(sent)) for sent in tokenized_sentences]

        input_ids = torch.tensor(padded_sentences)

        with torch.no_grad():
            outputs = model(input_ids)

        embeddings = outputs.last_hidden_state[:, 0, :]

        sparse_matrix = csr_matrix(embeddings.numpy())

        dense_data_array = sparse_matrix.toarray()

        model = AgglomerativeClustering(n_clusters=None,
                                        linkage='complete',
                                        distance_threshold=0,
                                        metric="cosine")
        model.fit(dense_data_array)
        model_info = {
            'affinity': 'BERT Cosine Complete',
            'model': model,
            'labels': data
        }

        file_name = 'bert_cosine_complete.pkl'
        file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
        joblib.dump(model_info, file_path)
        return file_path


class BERTEuclideanEmbeddingAffinity(AffinityStrategy):
    def compute_affinity(self, data: List):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
        max_len = max(len(sent) for sent in tokenized_sentences)
        padded_sentences = [sent + [tokenizer.pad_token_id] * (max_len - len(sent)) for sent in tokenized_sentences]

        input_ids = torch.tensor(padded_sentences)

        with torch.no_grad():
            outputs = model(input_ids)

        embeddings = outputs.last_hidden_state[:, 0, :]

        sparse_matrix = csr_matrix(embeddings.numpy())

        dense_data_array = sparse_matrix.toarray()

        model = AgglomerativeClustering(n_clusters=None,
                                        linkage='average',
                                        distance_threshold=0,
                                        metric="euclidean")
        model.fit(dense_data_array)
        model_info = {
            'affinity': 'BERT Euclidean Average',
            'model': model,
            'labels': data
        }

        file_name = 'bert_euclidean_average.pkl'
        file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
        joblib.dump(model_info, file_path)
        return file_path
