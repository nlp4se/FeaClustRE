import csv
from flask import Blueprint, request, make_response, jsonify
from . import dendogram_service, visualization_service
import os

import sys
import io
import logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO, encoding='utf-8')
bp = Blueprint('dendogram', __name__, url_prefix='/dendogram')




@bp.route('/generate', methods=['POST'])
def generate_dendogram():
    preprocessing = request.args.get('preprocessing', 'false').lower() == 'true'
    affinity = request.args.get('affinity', 'bert')
    linkage = request.args.get('linkage', 'average')
    metric = request.args.get('metric', 'cosine')
    threshold = float(request.args.get('threshold', None))
    object_weight = float(request.args.get('obj-weight', 0.25))
    verb_weight = float(request.args.get('verb-weight', 0.75))
    app_name = request.args.get('app_name', 'unknown')

    request_body = request.get_json()
    if not request_body or 'analyzed_reviews' not in request_body:
        return make_response({"error": "Invalid or missing 'analyzed_reviews' in JSON payload"}, 400)

    analyzed_reviews = request_body['analyzed_reviews']
    all_features = []

    for review in analyzed_reviews:
        sentences = review.get('sentences', [])
        for sentence in sentences:
            feature = sentence.get('featureData', {}).get('feature', None)
            if feature:
                all_features.append(feature)

    request_simplified = {}
    request_simplified["app_name"] = app_name
    request_simplified["features"] = all_features
    try:
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
        if threshold is not None:
            visualization_service.generate_dendrogram_visualization(dendogram_file)

        return jsonify({
            "message": "Dendrogram generated successfully",
            "features": all_features,
            "dendrogram_path": dendogram_file,
        }), 200

    except ValueError as e:
        return make_response({"error": str(e)}, 400)
    except Exception as e:
        return make_response({"error": "An unexpected error occurred", "details": str(e)}, 500)
@bp.route('/generate_kg', methods=['POST'])
def generate_dendogram_from_csv():
    preprocessing = request.args.get('preprocessing', 'false')
    affinity = request.args.get('affinity', 'bert')
    linkage = request.args.get('linkage', 'average')
    metric = request.args.get('metric', 'cosine')
    threshold = float(request.args.get('threshold', 0.2))
    object_weight = float(request.args.get('obj-weight', 0.25))
    verb_weight = float(request.args.get('verb-weight', 0.75))
    app_name = request.args.get('app_name', '')

    print(f"Request arguments: preprocessing={preprocessing}, "
          f"affinity={affinity}, "
          f"metric={metric}, "
          f"linkage={linkage}, "
          f"threshold={threshold}, "
          f"object_weight={object_weight}, "
          f"verb_weight={verb_weight}, "
          f"app_name={app_name}")

    if 'file' not in request.files:
        return make_response("CSV file is required", 400)

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return make_response("File must be a CSV", 400)

    features = []
    try:
        file_content = file.stream.read().decode('utf-8')
        csv_reader = csv.DictReader(file_content.splitlines())
        for row in csv_reader:
            extracted_features = row.get("extracted_features_TransFeatEx", "")
            if extracted_features:
                features.extend(extracted_features.split(';'))
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return make_response("Error processing CSV file", 500)

    request_content = {
        "app_name": app_name,
        "features": features
    }

    dendrogram_file = dendogram_service.generate_dendogram(preprocessing,
                                                           affinity,
                                                           metric,
                                                           linkage,
                                                           threshold,
                                                           object_weight,
                                                           verb_weight,
                                                           request_content)

    return jsonify({"message": "Dendrogram generated successfully", "dendrogram_path": dendrogram_file}), 200