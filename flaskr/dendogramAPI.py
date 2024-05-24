from flask import Blueprint, request, make_response

bp = Blueprint('dendogram', __name__, url_prefix='/dendogram')


bp.route('/generate')
def generate_dendogram():
    preprocessing = request.args.get('preprocessing', 'false')
    embedding = request.args.get('embedding', 'levenshtein')
    
    features = request.get_json()
    if features is None:
        return make_response("No features", 400)
    

