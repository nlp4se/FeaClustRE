from flask import Blueprint, request, make_response
from . import dendogram_service
bp = Blueprint('dendogram', __name__, url_prefix='/dendogram')


@bp.route('/generate', methods=['POST'])
def generate_dendogram():
    preprocessing = request.args.get('preprocessing', 'false')
    affinity = request.args.get('affinity', 'levenshtein')
    
    features = request.get_json()
    if features is None:
        return make_response("No features", 400)
    
    dendogram_service.generate_dendogram(preprocessing, affinity, features)
    

