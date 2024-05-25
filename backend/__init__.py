from flask import Flask


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    # Dendogram API
    from . import dendogram_api
    app.register_blueprint(dendogram_api.bp)

    return app