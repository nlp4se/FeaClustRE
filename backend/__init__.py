from flask import Flask


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    # Dendogram API
    from . import dendogram_controller
    app.register_blueprint(dendogram_controller.bp)

    # Graph API
    from . import graph_controller
    app.register_blueprint(graph_controller.bp)

    return app