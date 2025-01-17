from fastapi import FastAPI

from .routers import main_endpoints, crud_endpoints
from .model.initialize_models import download_default_models
from .db.initialize_db import initialize_db, create_models

app = FastAPI(
    title='NER API',
    description='Named Entity Recognition API for English and Polish languages',
    version='endgame'
)

app.include_router(crud_endpoints.router)
app.include_router(main_endpoints.router)


# :)
initialize_db()
create_models()
download_default_models()