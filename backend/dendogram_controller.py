from flask import Blueprint, request, make_response, send_file
from . import dendogram_service
bp = Blueprint('dendogram', __name__, url_prefix='/dendogram')


@bp.route('/generate', methods=['POST'])
def generate_dendogram():
    preprocessing = request.args.get('preprocessing', 'false')
    affinity = request.args.get('affinity', 'bert-embedding-cosine')
    
    features = request.get_json()
    if features is None:
        return make_response("No features", 400)
    
    dendogram_file = dendogram_service.generate_dendogram(preprocessing, affinity, features)
    return send_file(dendogram_file, as_attachment=True)
    

