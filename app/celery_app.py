from celery import Celery
from config import Config

# Initialize celery
celery = Celery('app',
                broker=Config.CELERY_BROKER_URL,
                backend=Config.CELERY_RESULT_BACKEND)

# Optional Configuration
celery.conf.update(
    result_expires=3600,
    task_track_started=True
)

# Import celery tasks
import app.tasks