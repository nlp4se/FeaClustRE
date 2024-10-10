from __future__ import annotations
from abc import abstractmethod
from typing import List
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
import spacy
import os

import numpy as np
import pandas as pd
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

    def ponderate_embeddings(self, batch_index, batch_data, embeddings):
        print(f"Applying verb and object weights for batch {batch_index + 1}...")

        tagged_data = [self.nlp(sent) for sent in batch_data]

        for i, doc in enumerate(tagged_data):
            verb_weights = []
            obj_weights = []
            token_embeddings = []
            tokens = []

            for token in doc:
                if token.pos_ == 'VERB' and self.verb_weight != 0:
                    verb_weights.append(self.verb_weight)
                    tokens.append(token.text)
                elif token.dep_ in ('dobj', 'nsubj', 'attr', 'prep', 'pobj') and self.object_weight != 0:
                    obj_weights.append(self.object_weight)
                    tokens.append(token.text)

            if not tokens:
                continue

            for token in tokens:
                inputs = self.tokenizer(token, return_tensors='pt')
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    token_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                    token_embeddings.append(token_embedding[0])

            weights = np.array(verb_weights + obj_weights)
            token_embeddings = np.array(token_embeddings)
            weighted_embeddings = token_embeddings * weights[:, np.newaxis]
            sentence_embedding = np.mean(weighted_embeddings, axis=0)
            embeddings[i] = torch.tensor(sentence_embedding)

    def process_batch(self, batch_data, batch_index, data_size):
        print(f"Processing batch {batch_index + 1}/{(data_size + len(batch_data) - 1) // len(batch_data)}...")

        inputs = self.tokenizer(batch_data, return_tensors='pt', padding=True, truncation=True)
        print(f"Getting BERT embeddings for batch {batch_index + 1}...")

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        self.ponderate_embeddings(batch_index, batch_data, embeddings)
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
        self.verb_weight = verb_weight
        self.object_weight = object_weight
        self.nlp = spacy.load("en_core_web_sm")

    def ponderate_embeddings(self, batch_data, embeddings):
        print("Applying verb and object weights...")

        tagged_data = [self.nlp(sent) for sent in batch_data]

        for i, doc in enumerate(tagged_data):
            verb_weights = []
            obj_weights = []
            for token in doc:
                if token.pos_ == 'VERB' and self.verb_weight != 0:
                    verb_weights.append(self.verb_weight)
                elif token.dep_ in ('dobj', 'nsubj', 'attr', 'prep', 'pobj') and self.object_weight != 0:
                    obj_weights.append(self.object_weight)

            total_weight = np.sum(verb_weights) + np.sum(obj_weights)
            if total_weight == 0:
                continue

            for token in doc:
                token_weight = 0
                if token.pos_ == 'VERB':
                    token_weight = self.verb_weight
                elif token.dep_ in ('dobj', 'nsubj', 'attr', 'prep', 'pobj'):
                    token_weight = self.object_weight

                if token_weight > 0:
                    # Find the index in the embeddings and modify the weight accordingly
                    token_idx = self.vectorizer.vocabulary_.get(token.text.lower())
                    if token_idx is not None:
                        embeddings[i, token_idx] *= token_weight  # Modify TF-IDF feature by weight

    def compute_affinity(self, application_name, labels, linkage, object_weight, verb_weight, distance_threshold, metric):
        self.verb_weight = verb_weight
        self.object_weight = object_weight

        print("Fitting TF-IDF vectorizer...")
        all_embeddings = self.vectorizer.fit_transform(labels)
        dense_data_array = all_embeddings.toarray()

        # Apply ponderation based on POS tagging
        self.ponderate_embeddings(labels, dense_data_array)

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
                                  'Tfidf',
                                  dense_data_array,
                                  labels,
                                  distance_threshold,
                                  linkage,
                                  metric,
                                  verb_weight,
                                  object_weight)


class MiniLMEmbeddingService(AffinityStrategy):
    def __init__(self, verb_weight=1.0, object_weight=1.0):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.verb_weight = verb_weight
        self.object_weight = object_weight
        self.nlp = spacy.load("en_core_web_sm")

    def ponderate_embeddings(self, batch_data, embeddings):
        print("Applying verb and object weights...")

        # Perform POS tagging using spaCy
        tagged_data = [self.nlp(sent) for sent in batch_data]

        for i, doc in enumerate(tagged_data):
            verb_weights = []
            obj_weights = []
            token_embeddings = []
            tokens = []

            for token in doc:
                # Identify verbs and objects
                if token.pos_ == 'VERB' and self.verb_weight != 0:
                    verb_weights.append(self.verb_weight)
                    tokens.append(token.text)
                elif token.dep_ in ('dobj', 'nsubj', 'attr', 'prep', 'pobj') and self.object_weight != 0:
                    obj_weights.append(self.object_weight)
                    tokens.append(token.text)

            if not tokens:
                continue

            # Retrieve embeddings for tokens
            for token in tokens:
                token_embedding = self.model.encode([token], convert_to_tensor=True).cpu().numpy()
                token_embeddings.append(token_embedding[0])

            # Apply ponderation to the token embeddings
            weights = np.array(verb_weights + obj_weights)
            token_embeddings = np.array(token_embeddings)
            weighted_embeddings = token_embeddings * weights[:, np.newaxis]
            sentence_embedding = np.mean(weighted_embeddings, axis=0)

            # Apply the pondered sentence embedding to the embeddings array
            embeddings[i] = torch.tensor(sentence_embedding)

    def compute_affinity(self, application_name, labels, linkage, object_weight, verb_weight, distance_threshold, metric):
        self.verb_weight = verb_weight
        self.object_weight = object_weight

        all_embeddings = []

        print(f"Processing data in batches of size {BATCH_SIZE}...")
        for i in range(0, len(labels), BATCH_SIZE):
            batch_data = labels[i:i + BATCH_SIZE]
            batch_index = i // BATCH_SIZE
            print(f"Processing batch {batch_index}...")
            batch_embeddings = self.model.encode(batch_data, convert_to_tensor=True)

            self.ponderate_embeddings(batch_data, batch_embeddings)

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

