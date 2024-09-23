import stanza
import unicodedata
import contractions
import re
import string
import spacy
import json
import os
from spellchecker import SpellChecker
from .Context import Context
from . import Affinity_strategy


def preprocessed_app(app_name):
    file_path = f"static/preprocessed_jsons/{app_name}Features.json"
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return True
    return False


def save_preprocessed_features(features, app_name):
    file_path = f"static/preprocessed_jsons/{app_name}Features.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(features, json_file)


def load_saved_preprocessed_features(app_name):
    file_path = f"static/preprocessed_jsons/{app_name}Features.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as json_file:
        return json.load(json_file)
    return None


def generate_dendogram(preprocessing,
                       embedding,
                       linkage,
                       distance_threshold,
                       object_weight,
                       verb_weight,
                       request_content):
    app_name = request_content['app_name']
    features = request_content['features']

    if preprocessing and not preprocessed_app(app_name):
        features = preprocess_features(features)
        save_preprocessed_features(features, app_name)
    elif preprocessing and preprocessed_app(app_name):
        features = load_saved_preprocessed_features(app_name)

    model_file_name = None

    # if embedding == 'tf-idf-cosine' or embedding == 'all':
    # context = Context(Affinity_strategy.TfIdfCosineAffinity())
    # model_file_name = context.use_affinity_algorithm(app_name, features, linkage, distance_threshold)

    #if embedding == 'tf-idf-euclidean' or embedding == 'all':
    # context = Context(Affinity_strategy.TfIdfEuclideanAffinity())
    # model_file_name = context.use_affinity_algorithm(app_name, features, linkage, distance_threshold)

    # if embedding == 'bert-embedding-euclidean' or embedding == 'all':
    # context = Context(Affinity_strategy.BERTEuclideanEmbeddingAffinity())
    # model_file_name = context.use_affinity_algorithm(app_name, features, linkage, distance_threshold)

    if embedding == 'bert-embedding-cosine' or embedding == 'all':
        context = Context(Affinity_strategy.BERTCosineEmbeddingAffinity())
        model_file_name = context.use_affinity_algorithm(app_name,
                                                         features,
                                                         linkage,
                                                         object_weight,
                                                         verb_weight,
                                                         distance_threshold)

    # if embedding == 'paraphrase-MiniLM-cosine' or embedding == 'all':
    # context = Context(Affinity_strategy.ParaphraseMiniLMCosineEmbeddingAffinity())
    # model_file_name = context.use_affinity_algorithm(app_name, features, linkage, distance_threshold)

    # if embedding == 'paraphrase-MiniLM-euclidean' or embedding == 'all':
    # context = Context(Affinity_strategy.ParaphraseMiniLMEuclideanEmbeddingAffinity())
    # model_file_name = context.use_affinity_algorithm(app_name, features, linkage, distance_threshold)

    return model_file_name

# TODO preprocess service
def is_english(text):
    pattern = re.compile(r'^[a-zA-Z0-9\s.,?!\'"-]+$')
    return bool(pattern.match(text))


def is_emoji_only(text):
    emoji_pattern = re.compile(
        "[\U00010000-\U0010FFFF]+",
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
    # feature = remove_numbers(feature) TODO Check with Quim
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
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
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
