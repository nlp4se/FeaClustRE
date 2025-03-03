import csv
from flask import Blueprint, request, make_response, jsonify
from . import dendogram_service, visualization_service
import os
import sys
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure stdout encoding is set to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Create Flask Blueprint
bp = Blueprint('dendogram', __name__, url_prefix='/dendogram')

@bp.route('/generate', methods=['POST'])
def generate_dendogram():
    """Generate a dendrogram from analyzed reviews."""
    try:
        # Extract query parameters
        preprocessing = request.args.get('preprocessing', 'false').lower() == 'true'
        affinity = request.args.get('affinity', 'bert')
        linkage = request.args.get('linkage', 'average')
        metric = request.args.get('metric', 'cosine')
        threshold = float(request.args.get('threshold', 0.2))
        object_weight = float(request.args.get('obj-weight', 0.25))
        verb_weight = float(request.args.get('verb-weight', 0.75))
        app_name = request.args.get('app_name', 'unknown')

        logger.info(f"Received request to generate dendrogram for app: {app_name}")
        logger.info(f"Parameters - Preprocessing: {preprocessing}, Affinity: {affinity}, Linkage: {linkage}, "
                    f"Metric: {metric}, Threshold: {threshold}, Object Weight: {object_weight}, "
                    f"Verb Weight: {verb_weight}")

        # Validate request body
        request_body = request.get_json()
        if not request_body or 'analyzed_reviews' not in request_body:
            logger.error("Invalid or missing 'analyzed_reviews' in JSON payload")
            return make_response({"error": "Invalid or missing 'analyzed_reviews' in JSON payload"}, 400)

        # Extract features from analyzed reviews
        analyzed_reviews = request_body['analyzed_reviews']
        all_features = []
        for review in analyzed_reviews:
            sentences = review.get('sentences', [])
            for sentence in sentences:
                feature = sentence.get('featureData', {}).get('feature', None)
                if feature:
                    all_features.append(feature)

        logger.info(f"Extracted {len(all_features)} features from analyzed reviews")

        # Prepare simplified request
        request_simplified = {
            "app_name": app_name,
            "features": all_features
        }

        # Generate dendrogram
        dendogram_file = dendogram_service.generate_dendogram(
            preprocessing=preprocessing,
            embedding=affinity,
            metric=metric,
            linkage=linkage,
            distance_threshold=threshold,
            object_weight=object_weight,
            verb_weight=verb_weight,
            request_content=request_simplified
        )
        logger.info(f"Dendrogram generated successfully at: {dendogram_file}")

        # Generate visualization if threshold is provided
        if threshold is not None:
            logger.info("Generating dendrogram visualization")
            visualization_service.generate_dendrogram_visualization(dendogram_file)

        return jsonify({
            "message": "Dendrogram generated successfully",
            "features": all_features,
            "dendrogram_path": dendogram_file,
        }), 200

    except ValueError as e:
        logger.error(f"ValueError occurred: {str(e)}")
        return make_response({"error": str(e)}, 400)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        return make_response({"error": "An unexpected error occurred", "details": str(e)}, 500)

@bp.route('/generate_kg', methods=['POST'])
def generate_dendogram_from_csv():
    """Generate a dendrogram from a CSV file."""
    try:
        # Extract query parameters
        preprocessing = request.args.get('preprocessing', 'false').lower() == 'true'
        affinity = request.args.get('affinity', 'bert')
        linkage = request.args.get('linkage', 'average')
        metric = request.args.get('metric', 'cosine')
        threshold = float(request.args.get('threshold', 0.2))
        object_weight = float(request.args.get('obj-weight', 0.25))
        verb_weight = float(request.args.get('verb-weight', 0.75))
        app_name = request.args.get('app_name', '')

        logger.info(f"Received request to generate dendrogram from CSV for app: {app_name}")
        logger.info(f"Parameters - Preprocessing: {preprocessing}, Affinity: {affinity}, Linkage: {linkage}, "
                    f"Metric: {metric}, Threshold: {threshold}, Object Weight: {object_weight}, "
                    f"Verb Weight: {verb_weight}")

        # Validate file in request
        if 'file' not in request.files:
            logger.error("CSV file is required")
            return make_response("CSV file is required", 400)

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            logger.error("File must be a CSV")
            return make_response("File must be a CSV", 400)

        # Read and process CSV file
        features = []
        try:
            file_content = file.stream.read().decode('utf-8')
            csv_reader = csv.DictReader(file_content.splitlines())
            for row in csv_reader:
                extracted_features = row.get("extracted_features", "")
                if extracted_features:
                    features.extend(extracted_features.split(';'))
            logger.info(f"Extracted {len(features)} features from CSV file")
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            return make_response("Error processing CSV file", 500)

        # Prepare request content
        request_content = {
            "app_name": app_name,
            "features": features
        }

        # Generate dendrogram
        dendogram_file = dendogram_service.generate_dendogram(
            preprocessing=preprocessing,
            embedding=affinity,
            metric=metric,
            linkage=linkage,
            distance_threshold=threshold,
            object_weight=object_weight,
            verb_weight=verb_weight,
            request_content=request_content
        )
        logger.info(f"Dendrogram generated successfully at: {dendogram_file}")

        # Generate visualization if threshold is provided
        if threshold is not None:
            logger.info("Generating dendrogram visualization")
            visualization_service.generate_dendrogram_visualization(dendogram_file)

        return jsonify({
            "message": "Dendrogram generated successfully",
            "dendrogram_path": dendogram_file,
        }), 200

    except ValueError as e:
        logger.error(f"ValueError occurred: {str(e)}")
        return make_response({"error": str(e)}, 400)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        return make_response({"error": "An unexpected error occurred", "details": str(e)}, 500)