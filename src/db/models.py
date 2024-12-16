from sqlalchemy import DATE, Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class AIModel(Base):
    __tablename__ = 'ai_model'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    date_created: Mapped[str] = mapped_column(DATE, nullable=False)
    is_training: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_trained: Mapped[bool] = mapped_column(Boolean, nullable=True, default=None)
    version: Mapped[str] = mapped_column(Integer, nullable=False, default=1)

    def generate_name(self, extension: str) -> str:
        """Generates a unique name for the model based on the base model and version.

        Args:
            extension (str): The file extension of the model.

        Returns:
            str: The unique name for the model.
        """
        return f'{self.base_model}_{self.version}.{extension}'

    def __repr__(self) -> str:
        """Returns a string representation of the AIModel object.

        Returns:
            str: A string representation of the AIModel object.
        """
        return f'<AIModel(id={self.id}, base_model={self.base_model}, file_path={self.file_path}, date_created={self.date_created}, is_training={self.is_training}, is_trained={self.is_trained} version={self.version})>'
