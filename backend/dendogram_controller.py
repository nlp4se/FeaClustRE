from flask import Blueprint, request, make_response, send_file
from . import dendogram_service
bp = Blueprint('dendogram', __name__, url_prefix='/dendogram')


@bp.route('/generate', methods=['POST'])
def generate_dendogram():
    preprocessing = request.args.get('preprocessing', 'false')
    affinity = request.args.get('affinity', 'bert-embedding-cosine')
    linkage = request.args.get('linkage', 'average')
    threshold = float(request.args.get('threshold', 0.2))
    
    request_content = request.get_json()
    if request_content['features'] is None:
        return make_response("No features", 400)
    
    dendogram_file = dendogram_service.generate_dendogram(preprocessing, affinity, linkage, threshold, request_content)
    return send_file(dendogram_file, as_attachment=True)
    

