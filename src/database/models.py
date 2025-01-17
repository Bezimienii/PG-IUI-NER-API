from sqlalchemy import DATE, Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .context_manager import Base


class AIModel(Base):
    """The AIModel class represents an AI model in the database."""
    __tablename__ = 'ai_model'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    date_created: Mapped[str] = mapped_column(DATE, nullable=False)
    is_training: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    train_file_path: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    valid_file_path: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    test_file_path: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    training_process_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_trained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    
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
        return (
            f'<AIModel(id={self.id}, base_model={self.base_model}, file_path={self.file_path}, '
            f'date_created={self.date_created}, is_training={self.is_training}, '
            f'is_trained={self.is_trained}, version={self.version})>'
        )
