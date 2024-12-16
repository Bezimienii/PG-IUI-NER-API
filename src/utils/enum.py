from enum import Enum
from ..config import settings

class BaseModels(str, Enum):
    ROBERTA = settings.ROBERTA
    HERBERT = settings.HERBERT