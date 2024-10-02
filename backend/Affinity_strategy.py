from __future__ import annotations
from abc import abstractmethod
from typing import List
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel
from scipy.sparse import csr_matrix
import spacy
import os
import joblib
import torch
import pandas as pd

MODEL_DIRECTORY_PATH = 'static' + os.path.sep + 'pkls'
MODEL_DIRECTORY_CSV_PATH = 'static' + os.path.sep + 'csv'

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
        # Initialize BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.nlp = spacy.load("en_core_web_sm")
        self.verb_weight = verb_weight
        self.object_weight = object_weight

    def process_batch(self, batch_data, batch_index, data_size):
        print(f"Processing batch {batch_index + 1}/{(data_size + len(batch_data) - 1) // len(batch_data)}...")

        # Tokenize and pad sentences
        tokenized_sentences = [self.tokenizer.encode(sent, add_special_tokens=True) for sent in batch_data]
        max_len = max(len(sent) for sent in tokenized_sentences)
        padded_sentences = [sent + [self.tokenizer.pad_token_id] * (max_len - len(sent)) for sent in tokenized_sentences]
        input_ids = torch.tensor(padded_sentences)

        print(f"Getting BERT embeddings for batch {batch_index + 1}...")
        with torch.no_grad():
            outputs = self.model(input_ids)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embeddings

        print(f"Applying verb and object weights for batch {batch_index + 1}...")
        tagged_data = [self.nlp(sent) for sent in batch_data]

        for i, doc in enumerate(tagged_data):
            for token in doc:
                token_position = token.pos_
                if token_position == 'VERB' and self.verb_weight != 0:
                    embeddings[i] += self.verb_weight * embeddings[i]
                elif token_position == 'NOUN' and self.object_weight != 0:
                    embeddings[i] += self.object_weight * embeddings[i]

        return embeddings

    def compute_affinity(self,
                         application_name,
                         data,
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
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_index = i // batch_size
            batch_embeddings = self.process_batch(batch_data, batch_index, len(data))
            all_embeddings.append(batch_embeddings)

        # Concatenate all embeddings from batches
        print("Concatenating all batch embeddings...")
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Convert to sparse matrix and then to dense format for clustering
        print("Converting embeddings to dense format...")
        sparse_matrix = csr_matrix(all_embeddings.numpy())
        dense_data_array = sparse_matrix.toarray()

        # Perform clustering
        print("Performing Agglomerative Clustering...")

        clustering_model = AgglomerativeClustering(n_clusters=None,
                                                   linkage=linkage,
                                                   distance_threshold=distance_threshold,
                                                   metric=metric,
                                                   compute_full_tree=True)

        clustering_model.fit(dense_data_array)

        csv_file_path = self.generate_csv(application_name,
                                          clustering_model,
                                          data,
                                          linkage,
                                          metric)

        pkl_file_path = self.generate_pkl(application_name,
                                          clustering_model,
                                          data,
                                          distance_threshold,
                                          linkage,
                                          metric)

        print("Process completed.")
        return pkl_file_path

    def generate_pkl(self, application_name, clustering_model, data, distance_threshold, linkage, metric):
        print("Saving the clustering model and metadata...")
        model_info = {
            'affinity': f'BERT {metric} {linkage}',
            'model': clustering_model,
            'labels': data,
            'application_name': application_name,
            'distance_threshold': distance_threshold,
            'verb_weight': self.verb_weight,
            'object_weight': self.object_weight
        }
        pkl_file_name = f"{application_name}_bert_{metric}_{linkage}_thr-{distance_threshold}_vw-{self.verb_weight}_ow-{self.object_weight}.pkl"
        pkl_file_path = save_to_pkl(model_info, pkl_file_name)
        return pkl_file_path

    def generate_csv(self, application_name, clustering_model, data, linkage, metric):
        # Get labels from clustering results
        labels = clustering_model.labels_
        # Create a DataFrame to save the clustering results
        df_results = pd.DataFrame({'Sentence': data, 'Cluster': labels})
        # Save the DataFrame to a CSV file using the global method
        csv_file_name = f"{application_name}_bert_{metric}_{linkage}_results.csv"
        csv_file_path = save_to_csv(df_results, csv_file_name)
        return csv_file_path
