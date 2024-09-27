from flask import Flask


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    from . import dendogram_controller
    app.register_blueprint(dendogram_controller.bp)

    from . import graph_controller
    app.register_blueprint(graph_controller.bp)

    return app