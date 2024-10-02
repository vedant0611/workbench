from flask import Flask
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler

from src.routes.index import prediction_bp, init_app

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Initialize New Relic

    # Configure logging
    app.logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

    # Initialize blueprints
    init_app(app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)