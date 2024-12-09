from flask import Flask
from config import Config
import os

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Ensure storage directories exist
    os.makedirs(app.config['TEMP_DIR'], exist_ok=True)
    os.makedirs(app.config['TRANSCRIPTS_DIR'], exist_ok=True)
    
    from app.routes import main
    app.register_blueprint(main)
    
    return app 