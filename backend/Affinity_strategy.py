from __future__ import annotations
from abc import abstractmethod
from typing import List
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
import spacy

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.utils import Utils

BATCH_SIZE = 32


class AffinityStrategy():
    @abstractmethod
    def compute_affinity(self, data: List):
        pass

class BertEmbeddingAffinity(AffinityStrategy):
    def __init__(self, verb_weight=1.0, object_weight=1.0):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.nlp = spacy.load("en_core_web_sm")
        self.verb_weight = verb_weight
        self.object_weight = object_weight

    def process_batch(self, batch_data, batch_index, data_size):
        inputs = self.tokenizer(batch_data, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        Utils.ponderate_embeddings_with_weights(
            batch_data=batch_data,
            embeddings=embeddings,
            verb_weight=self.verb_weight,
            object_weight=self.object_weight,
            tokenizer=self.tokenizer,
            model=self.model
        )
        return embeddings

    def compute_affinity(self,
                         application_name,
                         labels,
                         linkage,
                         object_weight,
                         verb_weight,
                         distance_threshold,
                         metric):

        self.verb_weight = verb_weight
        self.object_weight = object_weight

        all_embeddings = []
        batch_size = 32

        print(f"Processing data in batches of size {batch_size}...")
        for i in range(0, len(labels), batch_size):
            batch_data = labels[i:i + batch_size]
            batch_index = i // batch_size
            batch_embeddings = self.process_batch(batch_data, batch_index, len(labels))
            all_embeddings.append(batch_embeddings)

        print("Concatenating all batch embeddings...")
        all_embeddings = torch.cat(all_embeddings, dim=0)

        print("Converting embeddings to dense format...")
        sparse_matrix = csr_matrix(all_embeddings.numpy())
        dense_data_array = sparse_matrix.toarray()


        print("Performing Agglomerative Clustering...")
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            linkage=linkage,
            distance_threshold=distance_threshold,
            metric=metric,
            compute_full_tree=True
        )

        clustering_model.fit(dense_data_array)

        return Utils.generate_pkl(application_name,
                                  clustering_model,
                                  'Bert',
                                  dense_data_array,
                                  labels,
                                  distance_threshold,
                                  linkage,
                                  metric,
                                  verb_weight,
                                  object_weight)

class TfidfEmbeddingService(AffinityStrategy):
    def __init__(self, verb_weight=1.0, object_weight=1.0):
        self.vectorizer = TfidfVectorizer()
        self.nlp = spacy.load("en_core_web_sm")
        self.verb_weight = verb_weight
        self.object_weight = object_weight

    def get_dense_data_array(self, data: List) -> np.ndarray:
        tfidf_vectorizer = TfidfVectorizer()
        tf_idf_data_vector = tfidf_vectorizer.fit_transform(data)
        return tf_idf_data_vector.toarray(), tfidf_vectorizer

    def compute_affinity(self,
                         application_name,
                         labels,
                         linkage,
                         object_weight,
                         verb_weight,
                         distance_threshold,
                         metric):

        self.verb_weight = verb_weight
        self.object_weight = object_weight

        print("Converting data to dense TF-IDF vectors...")
        dense_data_array, tfidf_vectorizer = self.get_dense_data_array(labels)

        zero_vectors = np.all(dense_data_array == 0, axis=1)
        print(f"Number of zero vectors: {np.sum(zero_vectors)}")

        if np.sum(zero_vectors) > 0:
            dense_data_array = dense_data_array[~zero_vectors]
            labels = [label for i, label in enumerate(labels) if not zero_vectors[i]]

        if len(dense_data_array) == 0:
            print("All vectors are zero vectors, aborting clustering.")
            return None

        print("Ponderating TF-IDF embeddings with verb and object weights...")
        # Adjust TF-IDF values based on verb and object weights
        dense_data_array = Utils.ponderate_tfidf_with_weights(
            labels,  # batch_data
            dense_data_array,  # tfidf_matrix
            tfidf_vectorizer,  # vectorizer
            verb_weight=self.verb_weight,
            object_weight=self.object_weight
        )

        print("Performing Agglomerative Clustering...")
        clustering_model = AgglomerativeClustering(n_clusters=None,
                                                   linkage=linkage,
                                                   distance_threshold=distance_threshold,
                                                   metric=metric)
        clustering_model.fit(dense_data_array)

        return Utils.generate_pkl(application_name,
                                  clustering_model,
                                  'Tfidf',
                                  dense_data_array,
                                  labels,
                                  distance_threshold,
                                  linkage,
                                  metric,
                                  self.verb_weight,
                                  self.object_weight)
class MiniLMEmbeddingService(AffinityStrategy):
    def __init__(self, verb_weight=1.0, object_weight=1.0):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.verb_weight = verb_weight
        self.object_weight = object_weight
        self.nlp = spacy.load("en_core_web_sm")

    def compute_affinity(self,
                         application_name,
                         labels,
                         linkage,
                         object_weight,
                         verb_weight,
                         distance_threshold,
                         metric):
        self.verb_weight = verb_weight
        self.object_weight = object_weight

        all_embeddings = []

        print(f"Processing data in batches of size {BATCH_SIZE}...")
        for i in range(0, len(labels), BATCH_SIZE):
            batch_data = labels[i:i + BATCH_SIZE]
            batch_index = i // BATCH_SIZE
            print(f"Processing batch {batch_index}...")
            batch_embeddings = self.model.encode(batch_data, convert_to_tensor=True)

            Utils.ponderate_embeddings_with_weights(batch_data,
                                       batch_embeddings,
                                       self.verb_weight,
                                       self.object_weight,
                                       tokenizer=None,
                                       model=None)

            all_embeddings.append(batch_embeddings)

        print("Concatenating all batch embeddings...")
        all_embeddings = torch.cat(all_embeddings, dim=0)
        dense_data_array = all_embeddings.cpu().numpy()

        print("Performing Agglomerative Clustering...")
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            linkage=linkage,
            distance_threshold=distance_threshold,
            metric=metric
        )
        clustering_model.fit(dense_data_array)

        return Utils.generate_pkl(application_name,
                                  clustering_model,
                                  'MiniLM',
                                  dense_data_array,
                                  labels,
                                  distance_threshold,
                                  linkage,
                                  metric,
                                  verb_weight,
                                  object_weight)
