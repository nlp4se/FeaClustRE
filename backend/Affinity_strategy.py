from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List
from .tf_idf_utils import get_dense_data_array
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from scipy.sparse import csr_matrix
import spacy
import os
import joblib
import torch

MODEL_DIRECTORY_PATH = 'static' + os.path.sep + 'pkls'


class AffinityStrategy():
    @abstractmethod
    def compute_affinity(self, data: List):
        pass
# class TfIdfCosineAffinity(AffinityStrategy):
#     def compute_affinity(self, application_name, data: List, linkage, distance_threshold):
#         dense_data_array = get_dense_data_array(data=data)
#         model = AgglomerativeClustering(n_clusters=None,
#                                         linkage=linkage,
#                                         distance_threshold=distance_threshold,
#                                         metric="cosine",
#                                         compute_full_tree=True)
#         model.fit(dense_data_array)
#         model_info = {
#             'affinity': 'TF-IDF Cosine Complete',
#             'model': model,
#             'labels': data
#         }
#
#         file_name = application_name +'_tf_idf_cosine_' + linkage + '_' + str(distance_threshold) + '.pkl'
#         file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
#         joblib.dump(model_info, file_path)
#         return file_path
#
# class TfIdfEuclideanAffinity(AffinityStrategy):
#     def compute_affinity(self, application_name, data: List, linkage, distance_threshold):
#         dense_data_array = get_dense_data_array(data)
#         model = AgglomerativeClustering(n_clusters=None,
#                                         linkage=linkage,
#                                         distance_threshold=distance_threshold,
#                                         metric="euclidean",
#                                         compute_full_tree=True)
#         model.fit(dense_data_array)
#         model_info = {
#             'affinity': 'TF-IDF Euclidean Average',
#             'model': model,
#             'labels': data
#         }
#
#         file_name = application_name +'_tf_idf_euclidean_' + linkage + '_' + str(distance_threshold) + '.pkl'
#         file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
#         joblib.dump(model_info, file_path)
#         return file_path

class BERTCosineEmbeddingAffinity(AffinityStrategy):
    def compute_affinity(self,
                         application_name,
                         data: List,
                         linkage,
                         object_weigth,
                         verb_weight,
                         distance_threshold):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        nlp = spacy.load("en_core_web_sm")

        tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
        max_len = max(len(sent) for sent in tokenized_sentences)
        padded_sentences = [sent + [tokenizer.pad_token_id] * (max_len - len(sent)) for sent in tokenized_sentences]

        input_ids = torch.tensor(padded_sentences)

        with torch.no_grad():
            outputs = model(input_ids)

        embeddings = outputs.last_hidden_state[:, 0, :]
        tagged_data = [nlp(sent) for sent in data]

        for i, doc in enumerate(tagged_data):
            for token in doc:
                if token.pos_ == 'VERB':
                    embeddings[i] += verb_weight * embeddings[i]
                elif token.pos_ == 'NOUN':
                    embeddings[i] += object_weigth * embeddings[i]

        sparse_matrix = csr_matrix(embeddings.numpy())

        dense_data_array = sparse_matrix.toarray()

        model = AgglomerativeClustering(n_clusters=None,
                                        linkage=linkage,
                                        distance_threshold=distance_threshold,
                                        metric="cosine",
                                        compute_full_tree=True)
        model.fit(dense_data_array)
        model_info = {
            'affinity': 'BERT Cosine Complete',
            'model': model,
            'labels': data,
            'verb_weight': verb_weight,
            'object_weight': object_weigth
        }
        current_date = datetime.now().strftime('%Y-%m-%d')

        file_name = current_date + '.pkl'
        file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
        joblib.dump(model_info, file_path)
        return file_path


# class BERTEuclideanEmbeddingAffinity(AffinityStrategy):
#     def compute_affinity(self, application_name, data: List, linkage, distance_threshold):
#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertModel.from_pretrained('bert-base-uncased')
#         nlp = spacy.load("en_core_web_sm")
#
#         tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
#         max_len = max(len(sent) for sent in tokenized_sentences)
#         padded_sentences = [sent + [tokenizer.pad_token_id] * (max_len - len(sent)) for sent in tokenized_sentences]
#
#         input_ids = torch.tensor(padded_sentences)
#
#         with torch.no_grad():
#             outputs = model(input_ids)
#
#         embeddings = outputs.last_hidden_state[:, 0, :]
#         tagged_data = [nlp(sent) for sent in data]
#         verb_weight = 0.5
#         obj_weight = 1.5
#
#         for i, doc in enumerate(tagged_data):
#             for token in doc:
#                 if token.pos_ == 'VERB':
#                     embeddings[i] += verb_weight * embeddings[i]
#                 elif token.pos_ == 'NOUN':
#                     embeddings[i] += obj_weight * embeddings[i]
#         sparse_matrix = csr_matrix(embeddings.numpy())
#
#         dense_data_array = sparse_matrix.toarray()
#
#         model = AgglomerativeClustering(n_clusters=None,
#                                         linkage=linkage,
#                                         distance_threshold=distance_threshold,
#                                         metric="euclidean",
#                                         compute_full_tree=True)
#         model.fit(dense_data_array)
#         model_info = {
#             'affinity': 'BERT Euclidean Average',
#             'model': model,
#             'labels': data,
#             'verb_weight': verb_weight,
#             'object_weight': obj_weight
#         }
#
#         file_name = application_name +'_bert_euclidean_' + linkage + '_' + str(distance_threshold) + '.pkl'
#         file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
#         joblib.dump(model_info, file_path)
#         return file_path
#
# class ParaphraseMiniLMEuclideanEmbeddingAffinity(AffinityStrategy):
#     def compute_affinity(self, application_name, data: List, linkage, distance_threshold):
#         tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
#         nlp = spacy.load("en_core_web_sm")
#
#         model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
#
#         tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
#         max_len = max(len(sent) for sent in tokenized_sentences)
#         padded_sentences = [sent + [tokenizer.pad_token_id] * (max_len - len(sent)) for sent in tokenized_sentences]
#
#         input_ids = torch.tensor(padded_sentences)
#
#         with torch.no_grad():
#             outputs = model(input_ids)
#
#         embeddings = outputs.last_hidden_state[:, 0, :]
#         tagged_data = [nlp(sent) for sent in data]
#         verb_weight = 0.5
#         obj_weight = 1.5
#
#         for i, doc in enumerate(tagged_data):
#             for token in doc:
#                 if token.pos_ == 'VERB':
#                     embeddings[i] += verb_weight * embeddings[i]
#                 elif token.pos_ == 'NOUN':
#                     embeddings[i] += obj_weight * embeddings[i]
#         sparse_matrix = csr_matrix(embeddings.numpy())
#
#         dense_data_array = sparse_matrix.toarray()
#
#         model = AgglomerativeClustering(n_clusters=None,
#                                         linkage=linkage,
#                                         distance_threshold=distance_threshold,
#                                         metric="euclidean",
#                                         compute_full_tree=True)
#         model.fit(dense_data_array)
#         model_info = {
#             'affinity': 'Paraphrase MiniLM Euclidean Average',
#             'model': model,
#             'labels': data,
#             'verb_weight': verb_weight,
#             'object_weight': obj_weight
#         }
#
#         file_name = application_name +'_paraphrase_minilm_euclidean_' + linkage + '_' + str(distance_threshold) + '.pkl'
#         file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
#         joblib.dump(model_info, file_path)
#         return file_path
#
# class ParaphraseMiniLMCosineEmbeddingAffinity(AffinityStrategy):
#     def compute_affinity(self, application_name, data: List, linkage, distance_threshold):
#         tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
#         nlp = spacy.load("en_core_web_sm")
#
#         model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
#
#         tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
#         max_len = max(len(sent) for sent in tokenized_sentences)
#         padded_sentences = [sent + [tokenizer.pad_token_id] * (max_len - len(sent)) for sent in tokenized_sentences]
#
#         input_ids = torch.tensor(padded_sentences)
#
#         with torch.no_grad():
#             outputs = model(input_ids)
#
#         embeddings = outputs.last_hidden_state[:, 0, :]
#         tagged_data = [nlp(sent) for sent in data]
#         verb_weight = 0.5
#         obj_weight = 1.5
#
#         for i, doc in enumerate(tagged_data):
#             for token in doc:
#                 if token.pos_ == 'VERB':
#                     embeddings[i] += verb_weight * embeddings[i]
#                 elif token.pos_ == 'NOUN':
#                     embeddings[i] += obj_weight * embeddings[i]
#         sparse_matrix = csr_matrix(embeddings.numpy())
#
#         dense_data_array = sparse_matrix.toarray()
#
#         model = AgglomerativeClustering(n_clusters=None,
#                                         linkage=linkage,
#                                         distance_threshold=distance_threshold,
#                                         metric="cosine",
#                                         compute_full_tree=True)
#         model.fit(dense_data_array)
#         model_info = {
#             'affinity': 'Paraphrase MiniLM Cosine Average',
#             'model': model,
#             'labels': data,
#             'verb_weight': verb_weight,
#             'object_weight': obj_weight
#         }
#
#         file_name = application_name +'_paraphrase_minilm_cosine_' + linkage + '_' + str(distance_threshold) + '.pkl'
#         file_path = os.path.join(os.getcwd(), MODEL_DIRECTORY_PATH, file_name)
#         joblib.dump(model_info, file_path)
#         return file_path
