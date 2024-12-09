import os
from dotenv import load_dotenv
from datetime import timedelta
import redis

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    STORAGE_DIR = os.path.join(BASE_DIR, 'app', 'storage')
    TEMP_DIR = os.path.join(STORAGE_DIR, 'temp')
    TRANSCRIPTS_DIR = os.path.join(STORAGE_DIR, 'transcripts')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max-size for upload
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    # Redis configuration for Celery
    CELERY_BROKER_URL = 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

    # Session configuration
    SESSION_TYPE = 'redis'
    SESSION_REDIS = redis.from_url('redis://localhost:6379/1')
    PERMANENT_SESSION_LIFETIME = timedelta(days=1)