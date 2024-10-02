from flask import Blueprint

# from src.routes.causality_route import causality_routes
from src.routes.rag_route import rag_routes

prediction_bp = Blueprint('prediction_bp', __name__)

def init_app(app):
    # Register sub-blueprints
    # prediction_bp.register_blueprint(causality_routes)
    prediction_bp.register_blueprint(rag_routes)
    print("routes")
    # Register the main blueprint
    app.register_blueprint(prediction_bp)

    app.logger.info("Registered main prediction_bp blueprint")
    app.logger.info("All blueprints have been registered")