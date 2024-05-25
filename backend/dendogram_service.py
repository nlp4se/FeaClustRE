import stanza
from . import Context
from . import Affinity_strategy
def generate_dendogram(preprocessing, embedding, features):
    
    # TODO throw custom exception if embedding is none
    # TODO throw custom exception if preprocessing is none
    # TODO throw custom exception if features is none
    if preprocessing: 
        features = preprocess_features(features)
    context = None
    if embedding == 'levenshtein':
        context = Context(Affinity_strategy.LevenshteinAffinity())
    
def preprocess_features(features):
    nlp_pipeline = stanza.Pipeline(lang='en', processors='lemma')
    preprocessed_features = []
    for feature in features:
        preprocess_features.append(nlp_pipeline(feature))


