"""Base interface for OCR models."""

from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path


class OCRModel(ABC):
    """Abstract base class for OCR models."""

    @abstractmethod
    def extract_text(self, image_input: Union[str, Path, bytes]) -> str:
        """
        Extract text from an image.

        Args:
            image_input: Can be a file path (str/Path), URL (str), or image bytes

        Returns:
            Extracted text from the image
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory to free resources."""
        pass
