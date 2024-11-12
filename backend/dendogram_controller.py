from flask import Blueprint, request, make_response
from . import dendogram_service
import json
bp = Blueprint('dendogram', __name__, url_prefix='/dendogram')


from flask import jsonify

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
def generate_dendogram_from_body():
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
          f"threshold={threshold} ",
          f"object_weight={object_weight} ",
          f"verb_weight={verb_weight} ",
          f"app_name={app_name}")

    request_content = request.get_json()
    if request_content is None:
        return make_response("Invalid JSON body", 400)

    features = []
    for review in request_content.get("analyzed_reviews", []):
        for sentence in review.get("sentences", []):
            feature = sentence.get("featureData", {}).get("feature", "")
            if feature:
                features.append(feature)

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
