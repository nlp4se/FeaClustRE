from flask import Blueprint, request, make_response, send_file
from . import dendogram_service
bp = Blueprint('graph', __name__, url_prefix='/graph')


@bp.route('/generate', methods=['POST'])
def generate_dendogram():
    preprocessing = request.args.get('preprocessing', 'false')
    affinity = request.args.get('affinity', 'bert-embedding-cosine')
    
    request_content = request.get_json()
    if request_content['features'] is None:
        return make_response("No features", 400)
    
    dendogram_file = dendogram_service.generate_dendogram(preprocessing, affinity, request_content)
    return send_file(dendogram_file, as_attachment=True)
    

