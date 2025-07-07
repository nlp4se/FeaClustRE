import stanza
import unicodedata
import contractions
import re
import string
import spacy
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.sparse import csr_matrix
from typing import List, Tuple
from spellchecker import SpellChecker
from transformers import BertTokenizer, BertModel

def generate_graph(preprocessing, embedding, request_content):

    features = request_content['features']
    if preprocessing:
        features = preprocess_features(features)

    if embedding == 'tf-idf-cosine' or embedding == 'all':
        None

    if embedding == 'tf-idf-euclidean' or embedding == 'all':
        None

    if embedding == 'bert-embedding-euclidean' or embedding == 'all':
        None

    if embedding == 'bert-embedding-cosine' or embedding == 'all':
        fig = compute_bert_affinity(features)

    if embedding == 'paraphrase-MiniLM-cosine' or embedding == 'all':
        None
    
    if embedding == 'paraphrase-MiniLM-euclidean' or embedding == 'all':
        None
         
    return fig


def is_english(text):
    pattern = re.compile(r'^[a-zA-Z0-9\s.,?!\'"-]+$')
    return bool(pattern.match(text))


def is_emoji_only(text):
    emoji_pattern = re.compile(
        "[\U00010000-\U0010FFFF]+",  # Allow one or more emoji
        flags=re.UNICODE
    )
    return bool(emoji_pattern.fullmatch(text))


def contains_weird_characters(text):
    weird_characters_pattern = re.compile(r'[^a-zA-Z0-9\s.,?!\'"_-]')
    return bool(weird_characters_pattern.search(text))



def preprocess_features(features):
    preprocessed_features = []
    for feature in features:
        if not is_emoji_only(feature) and not contains_weird_characters(feature):
            preprocessed_feature = preprocess_feature(feature)
            if is_english(preprocessed_feature):
                preprocessed_features.append(preprocessed_feature)

    return preprocessed_features


def preprocess_feature(feature):
    feature = feature.replace('_', ' ')
    feature = remove_mentions_and_tags(feature)
    feature = remove_numbers(feature)
    feature = camel_case_to_words(feature)
    feature = expand_contractions(feature)
    feature = remove_special_characters(feature)
    feature = remove_punctuation(feature)
    feature = standarize_accents(feature)
    # feature = spell_check(feature)
    feature = lemmatize_spacy(feature)
    # feature = lemmatize_stanza(feature)
    feature = feature.lower()
    return feature


def expand_contractions(feature):
    expanded_words = []
    for word in feature.split():
        expanded_words.append(contractions.fix(word))
    return ' '.join(expanded_words)

def standarize_accents(feature): 
    return unicodedata.normalize('NFKD', feature).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_mentions_and_tags(text):
    text = re.sub(r'@\S*', '', text)
    return re.sub(r'#\S*', '', text)

def remove_special_characters(text):
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)

def remove_numbers(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pattern, '', text)

def remove_punctuation(text):
    return ''.join([c for c in text if c not in string.punctuation])
    
def camel_case_to_words(camel_case_str):
    words = re.sub('([a-z])([A-Z])', r'\1 \2', camel_case_str)
    return words

def lemmatize_spacy(feature):
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm', disable = ['parser','ner']) 
    doc = nlp(feature)
    return " ".join([token.lemma_ for token in doc])

def lemmatize_stanza(feature):
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
    doc = nlp(feature)
    lemmatized_feature = ' '.join([word.lemma for sent in doc.sentences for word in sent.words])
    return lemmatized_feature

def spell_check(feature):
    spell = SpellChecker()
    corrected_feature = []
    for word in feature.split():
        corrected_word = spell.correction(word)
        if corrected_word is not None:
            corrected_feature.append(corrected_word)
        else:
            corrected_feature.append(word)
    if corrected_feature is None:
        return ""   
    return " ".join(corrected_feature)

def tokenize_sentences(data: List[str], tokenizer) -> Tuple[List[List[int]], int]:
    tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=True) for sent in data]
    max_len = max(len(sent) for sent in tokenized_sentences)
    padded_sentences = [sent + [tokenizer.pad_token_id] * (max_len - len(sent)) for sent in tokenized_sentences]
    return padded_sentences

def compute_bert_embeddings(padded_sentences: List[List[int]], model) -> torch.Tensor:
    """
    Computes BERT embeddings for the padded sentences.
    
    Args:
        padded_sentences (List[List[int]]): Padded tokenized sentences.
        model: Pre-trained BERT model.

    Returns:
        torch.Tensor: BERT embeddings for the input sentences.
    """
    input_ids = torch.tensor(padded_sentences)
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.last_hidden_state[:, 0, :]


def adjust_embeddings_with_pos(embeddings: torch.Tensor, data: List[str], nlp) -> torch.Tensor:

    tagged_data = [nlp(sent) for sent in data]
    verb_weight = 0.5
    obj_weight = 1.5

    for i, doc in enumerate(tagged_data):
        for token in doc:
            if token.pos_ == 'VERB':
                embeddings[i] += verb_weight * embeddings[i]
            elif token.pos_ == 'NOUN':
                embeddings[i] += obj_weight * embeddings[i]

    return embeddings


def compute_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """
    Computes the similarity matrix for the embeddings.

    Args:
        embeddings (torch.Tensor): Adjusted embeddings for the sentences.

    Returns:
        np.ndarray: Similarity matrix (https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/).
    """
    num_sentences = embeddings.size(0)
    similarity_matrix = np.zeros((num_sentences, num_sentences))
    
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                similarity_matrix[i, j] = torch.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()

    return similarity_matrix


def plot_graph(adjacency_matrix: csr_matrix) -> Figure:
    """
    Plots the graph based on the adjacency matrix 
    (https://www.geeksforgeeks.org/adjacency-matrix-meaning-and-definition-in-dsa/).

    Args:
        adjacency_matrix (csr_matrix): Adjacency matrix representing the graph.

    """
    G = nx.from_scipy_sparse_array(adjacency_matrix, create_using=nx.DiGraph)
    fig, ax = plt.subplots()

    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, ax=ax, node_size=500, font_size=10, font_color="black")

    plt.close(fig)

    return fig


def compute_bert_affinity(data: List[str]) -> Figure:
    """
    Computes BERT-based sentence similarity and plots the similarity graph.
    adjacency matrix: https://www.geeksforgeeks.org/adjacency-matrix-meaning-and-definition-in-dsa/

    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    nlp = spacy.load("en_core_web_sm")

    padded_sentences = tokenize_sentences(data, tokenizer)
    embeddings = compute_bert_embeddings(padded_sentences, model)
    embeddings = adjust_embeddings_with_pos(embeddings, data, nlp)
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    adjacency_matrix = csr_matrix(similarity_matrix)
    fig = plot_graph(adjacency_matrix)

    return fig