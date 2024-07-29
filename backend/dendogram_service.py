import stanza
import unicodedata
import contractions
import re
import string
import spacy
from logging import Logger
from spellchecker import SpellChecker
from .Context import Context
from . import Affinity_strategy

def generate_dendogram(preprocessing, embedding, request_content):
    features = request_content['features']
    if preprocessing:
        features = preprocess_features(features)

    model_file_name = None

    if embedding == 'tf-idf-cosine' or embedding == 'all':
        context = Context(Affinity_strategy.TfIdfCosineAffinity())
        model_file_name = context.use_affinity_algorithm(features)

    if embedding == 'tf-idf-euclidean' or embedding == 'all':
        context = Context(Affinity_strategy.TfIdfEuclideanAffinity())
        model_file_name = context.use_affinity_algorithm(features)

    if embedding == 'bert-embedding-euclidean' or embedding == 'all':
        context = Context(Affinity_strategy.BERTEuclideanEmbeddingAffinity())
        model_file_name = context.use_affinity_algorithm(features)

    if embedding == 'bert-embedding-cosine' or embedding == 'all':
        context = Context(Affinity_strategy.BERTCosineEmbeddingAffinity())
        model_file_name = context.use_affinity_algorithm(features)

    if embedding == 'paraphrase-MiniLM-cosine' or embedding == 'all':
        context = Context(Affinity_strategy.ParaphraseMiniLMCosineEmbeddingAffinity())
        model_file_name = context.use_affinity_algorithm(features)
    
    if embedding == 'paraphrase-MiniLM-euclidean' or embedding == 'all':
        context = Context(Affinity_strategy.ParaphraseMiniLMEuclideanEmbeddingAffinity())
        model_file_name = context.use_affinity_algorithm(features)
    
    return model_file_name


def is_english(text):
    pattern = re.compile(r'^[a-zA-Z\s.,?!\'"-]+$')
    return bool(pattern.match(text))

def is_emoji_only(text):
    emoji_pattern = re.compile(
        "[\U00010000-\U0010FFFF]",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.fullmatch(text))

def contains_weird_characters(text):
    weird_characters_pattern = re.compile(r'[^a-zA-Z0-9\s.,?!\'"-]')
    return bool(weird_characters_pattern.search(text))

def preprocess_features(features):
    preprocessed_features = []
    for feature in features:
        if (is_english(feature) and
            not is_emoji_only(feature) and
            not contains_weird_characters(feature)):
            preprocessed_features.append(preprocess_feature(feature))
    return preprocessed_features

def preprocess_feature(feature):
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

