from app import create_app
from app.celery_app import celery

app = create_app()
app.app_context().push() 