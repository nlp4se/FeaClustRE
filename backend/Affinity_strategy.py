from __future__ import annotations
from abc import abstractmethod
from typing import List
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel
import torch
import spacy
import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

MODEL_DIRECTORY_PATH = 'static' + os.path.sep + 'pkls'
MODEL_DIRECTORY_CSV_PATH = 'static' + os.path.sep + 'csv'
MODEL_DIRECTORY_CSV_EMBEDDIGNS_PATH = 'static' + os.path.sep + 'csv' + os.path.sep + 'embeddings'

class AffinityStrategy():
    @abstractmethod
    def compute_affinity(self, data: List):
        pass

def save_to_csv(dataframe: pd.DataFrame, csv_filename: str):
    if not os.path.exists(MODEL_DIRECTORY_CSV_PATH):
        os.makedirs(MODEL_DIRECTORY_CSV_PATH)
    csv_file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_CSV_PATH, csv_filename)
    print(f"Saving CSV to {csv_file_path}...")
    dataframe.to_csv(csv_file_path, index=False)
    return csv_file_path

def save_to_pkl(model_info: dict, pkl_filename: str):
    if not os.path.exists(MODEL_DIRECTORY_PATH):
        os.makedirs(MODEL_DIRECTORY_PATH)
    pkl_file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, pkl_filename)
    print(f"Saving model to {pkl_file_path}...")
    joblib.dump(model_info, pkl_file_path)
    return pkl_file_path

class BertEmbeddingAffinity(AffinityStrategy):
    def __init__(self, verb_weight=1.0, object_weight=1.0):
        # Load BERT model and tokenizer from Hugging Face Transformers
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

        # Tokenize the batch data
        inputs = self.tokenizer(batch_data, return_tensors='pt', padding=True, truncation=True)
        print(f"Getting BERT embeddings for batch {batch_index + 1}...")

        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the last hidden state as embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling

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

        print("Saving embeddings to CSV...")
        embedding_df = pd.DataFrame(dense_data_array)
        embedding_df['Sentence'] = labels
        csv_filename = f"{application_name}_bert_{metric}_{linkage}_embeddings.csv"
        if not os.path.exists(MODEL_DIRECTORY_CSV_PATH):
            os.makedirs(MODEL_DIRECTORY_CSV_PATH)
        csv_file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_CSV_EMBEDDIGNS_PATH, csv_filename)
        embedding_df.to_csv(csv_file_path, index=False)

        print("Performing Agglomerative Clustering...")
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            linkage=linkage,
            distance_threshold=distance_threshold,
            metric=metric,
            compute_full_tree=True
        )

        clustering_model.fit(dense_data_array)

        pkl_file_path = self.generate_pkl(application_name,
                                          clustering_model,
                                          'BertEmbedding',
                                          dense_data_array,
                                          labels,
                                          distance_threshold,
                                          linkage,
                                          metric)
        print("Process completed.")
        return pkl_file_path

    def generate_pkl(self, application_name, clustering_model, model_name, dense_data_array, labels, distance_threshold, linkage,
                     metric):
        print("Saving clustering metadata for plotting...")

        model_info = {
            'affinity': f'BERT {metric} {linkage}',
            'labels': labels,
            'model_name': model_name,
            'model': clustering_model,
            'data_points': dense_data_array,
            'application_name': application_name,
            'distance_threshold': distance_threshold,
            'verb_weight': self.verb_weight,
            'object_weight': self.object_weight
        }

        if hasattr(clustering_model, 'cluster_centers_'):
            model_info['cluster_centers'] = clustering_model.cluster_centers_

        pkl_file_name = f"{application_name}_bert_{metric}_{linkage}_thr-{distance_threshold}_vw-{self.verb_weight}_ow-{self.object_weight}.pkl"
        pkl_file_path = save_to_pkl(model_info, pkl_file_name)
        return pkl_file_path
