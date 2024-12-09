from celery import Celery

# Initialize celery
celery = Celery('app',
                broker='redis://localhost:6379/0',
                backend='redis://localhost:6379/0')

# Optional Configuration
celery.conf.update(
    result_expires=3600,
    task_track_started=True
)