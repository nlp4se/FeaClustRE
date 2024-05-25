from flask import Flask


def create_dendogen_app():
    dendo_gen_app = Flask(__name__, instance_relative_config=True)

    # Dendogram API
    from . import dendogram_api
    dendo_gen_app.register_blueprint(dendogram_api.bp)

    return dendo_gen_app