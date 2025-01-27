from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from .database.tools import create_models, initialize_db
from .model.initialize_models import download_default_models
from .routers import crud_endpoints, main_endpoints

from .sync.sync_functions import start_job


app = FastAPI(
    title='NER API',
    description='Named Entity Recognition API for English and Polish languages',
    version='endgame'
)

# ----------------- Routers -----------------------
app.include_router(crud_endpoints.router)
app.include_router(main_endpoints.router)

# ----------------- Initialization -----------------
initialize_db()
create_models()
download_default_models()
scheduler = BackgroundScheduler()
start_job(scheduler)
scheduler.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    scheduler.shutdown()
