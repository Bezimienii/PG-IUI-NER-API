from fastapi import FastAPI

from .processing import processing
from .routers import database_example, example, ai_models
from .model.initialize_models import download_default_models
from .db.initialize_db import initialize_db, create_models

app = FastAPI(
    title='renameme',
    description='Fill the description',
    version='0.1',
)



app.include_router(example.router)
app.include_router(database_example.router)
app.include_router(ai_models.router)
app.include_router(processing.router)

# :)
initialize_db()
create_models()
download_default_models()