from flask import Flask


def create_dendogram_app():
    app = Flask(__name__, instance_relative_config=True)

    # Dendogram API
    from . import dendogramAPI
    app.register_blueprint(dendogramAPI.bp)

    return app