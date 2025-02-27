import requests
import json
import logging
import os
from .Context import Context
from . import Affinity_strategy
from dotenv import load_dotenv
from .preprocessing_service import preprocess_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()

def preprocessed_app(app_name):
    """Check if preprocessed features for the app already exist."""
    file_path = f"data/Stage 2 - Hierarchical Clustering/preprocessed_features_jsons/{app_name}Features.json"
    exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    logger.info(f"Checking if preprocessed features exist for {app_name}: {exists}")
    return exists

def save_preprocessed_features(features, app_name):
    """Save preprocessed features to a JSON file."""
    file_path = f"data/Stage 2 - Hierarchical Clustering/preprocessed_features_jsons/{app_name}Features.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(features, json_file)
    logger.info(f"Preprocessed features saved for {app_name} at {file_path}")

def load_saved_preprocessed_features(app_name):
    """Load preprocessed features from a JSON file."""
    file_path = os.path.join(BASE_DIR, "data", "Stage 2 - Hierarchical Clustering", "preprocessed_features_jsons", f"{app_name}Features.json")
    if not os.path.exists(file_path):
        logger.warning(f"No preprocessed features found for {app_name} at {file_path}")
        return None
    with open(file_path, "r") as json_file:
        logger.info(f"Loaded preprocessed features for {app_name} from {file_path}")
        return json.load(json_file)

def generate_dendogram(preprocessing,
                       embedding,
                       metric,
                       linkage,
                       distance_threshold,
                       object_weight,
                       verb_weight,
                       request_content):
    """Generate a dendrogram based on the provided parameters and features."""
    app_name = request_content['app_name']
    features = request_content['features']

    logger.info(f"Starting dendrogram generation for app: {app_name}")
    logger.info(f"Preprocessing enabled: {preprocessing}")
    logger.info(f"Embedding method: {embedding}")
    logger.info(f"Distance metric: {metric}")
    logger.info(f"Linkage method: {linkage}")
    logger.info(f"Distance threshold: {distance_threshold}")
    logger.info(f"Object weight: {object_weight}")
    logger.info(f"Verb weight: {verb_weight}")

    # Preprocessing step
    if preprocessing:
        if not preprocessed_app(app_name):
            logger.info("Preprocessing features as no preprocessed data exists.")
            features = preprocess_features(features)
            save_preprocessed_features(features, app_name)
        else:
            logger.info("Loading preprocessed features from saved file.")
            features = load_saved_preprocessed_features(app_name)
    else:
        logger.info("Preprocessing is disabled. Using raw features.")

    # Remove duplicate features
    logger.info(f"Initial number of features: {len(features)}")
    features = list(set(features))
    logger.info(f"Number of unique features after deduplication: {len(features)}")

    # Select the appropriate embedding strategy
    if embedding == 'bert':
        logger.info("Using BERT embedding strategy.")
        context = Context(Affinity_strategy.BertEmbeddingAffinity())
    elif embedding == 'paraphrase':
        logger.info("Using Paraphrase-MiniLM embedding strategy.")
        context = Context(Affinity_strategy.MiniLMEmbeddingService())
    elif embedding == 'tf-idf':
        logger.info("Using TF-IDF embedding strategy.")
        context = Context(Affinity_strategy.TfidfEmbeddingService())
    else:
        logger.error(f"Unsupported embedding method: {embedding}")
        raise ValueError(f"Unsupported embedding method: {embedding}")

    # Use the affinity algorithm
    logger.info("Applying affinity algorithm to generate dendrogram.")
    return context.use_affinity_algorithm(application_name=app_name,
                                          data=features,
                                          linkage=linkage,
                                          object_weight=object_weight,
                                          verb_weight=verb_weight,
                                          distance_threshold=distance_threshold,
                                          metric=metric)

def call_preprocessing_service(features):
    """Call the external preprocessing service to preprocess features."""
    url = os.getenv("DG_SERVICE_URL")
    port = os.getenv("DG_SERVICE_PORT")

    if not url or not port:
        logger.error("Preprocessing service URL or port not found in environment variables.")
        raise Exception("Preprocessing service URL or port not found in environment variables.")

    full_url = f"{url}:{port}/preprocess"
    data = {
        "features": features
    }

    logger.info(f"Calling preprocessing service at {full_url}")
    try:
        response = requests.post(full_url, json=data)
        if response.status_code == 200:
            logger.info("Preprocessing service call successful.")
            return response.json()['preprocessed_features']
        else:
            logger.error(f"Preprocessing service failed. Status code: {response.status_code}, Response: {response.text}")
            raise Exception(f"Failed to preprocess features. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logger.error(f"Error occurred while calling preprocessing service: {str(e)}")
        raise Exception(f"Error occurred while calling preprocessing service: {str(e)}")