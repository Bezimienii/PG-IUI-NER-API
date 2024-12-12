from fastapi import FastAPI

from .routers import example
from .routers import database_example

app = FastAPI(
    title='renameme',
    description='Fill the description',
    version='0.1',
)

app.include_router(example.router)
app.include_router(database_example.router)