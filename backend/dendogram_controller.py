import csv
from flask import Blueprint, request, make_response, jsonify
from . import dendogram_service

bp = Blueprint('dendogram', __name__, url_prefix='/dendogram')

@bp.route('/generate', methods=['POST'])
def generate_dendogram():
    preprocessing = request.args.get('preprocessing', 'false')
    affinity = request.args.get('affinity', 'bert')
    linkage = request.args.get('linkage', 'average')
    metric = request.args.get('metric', 'cosine')
    threshold = float(request.args.get('threshold', 0.2))
    object_weight = float(request.args.get('obj-weight', 0))
    verb_weight = float(request.args.get('verb-weight', 0))

    print(f"Request arguments: preprocessing={preprocessing}, "
          f"affinity={affinity}, "
          f"metric={metric}, "
          f"linkage={linkage}, "
          f"threshold={threshold} ",
          f"object_weight={object_weight} ",
          f"verb_weight={verb_weight}")

    request_content = request.get_json()
    if request_content['features'] is None:
        return make_response("No features", 400)

    dendogram_file = dendogram_service.generate_dendogram(preprocessing,
                                                          affinity,
                                                          metric,
                                                          linkage,
                                                          threshold,
                                                          object_weight,
                                                          verb_weight,
                                                          request_content)

    return jsonify({"message": "Dendrogram generated successfully", "dendrogram_path": dendogram_file}), 200


@bp.route('/generate_kg', methods=['POST'])
def generate_dendogram_from_csv():
    preprocessing = request.args.get('preprocessing', 'false')
    affinity = request.args.get('affinity', 'bert')
    linkage = request.args.get('linkage', 'average')
    metric = request.args.get('metric', 'cosine')
    threshold = float(request.args.get('threshold', 0.2))
    object_weight = float(request.args.get('obj-weight', 0))
    verb_weight = float(request.args.get('verb-weight', 0))
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
        # Read the file directly using TextIOWrapper
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