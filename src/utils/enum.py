from enum import Enum

from ..config import settings


class BaseModels(str, Enum):
    """Base models for the application."""
    ENG_MODEL = settings.ENG_MODEL
    PL_MODEL = settings.PL_MODEL
