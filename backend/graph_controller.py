import io
from flask import Blueprint, request, make_response
from . import graph_service


bp = Blueprint('graph', __name__, url_prefix='/graph')


@bp.route('/generate', methods=['POST'])
def generate_dendogram():
    preprocessing = request.args.get('preprocessing', 'false')
    affinity = request.args.get('affinity', 'bert-embedding-cosine')
    
    request_content = request.get_json()
    if request_content['features'] is None:
        return make_response("No features", 400)
    
    graph_figure = graph_service.generate_graph(preprocessing, affinity, request_content)
    if graph_figure is None:
        return make_response("Invalid embedding type", 400)
    
    img_stream = io.BytesIO()
    graph_figure.savefig(img_stream, format='png')
    img_stream.seek(0)
    
    response = make_response(img_stream.read())
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Content-Disposition', 'inline; filename=graph.png')
    
    return response
    

