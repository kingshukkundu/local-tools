"""vLLM-based LightOnOCR model implementation."""

import base64
import io
import requests
from PIL import Image
from typing import Union
from pathlib import Path
import logging

from .base import OCRModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMOcrModel(OCRModel):
    """LightOnOCR-2-1B model using vLLM server for inference."""

    def __init__(self, model_name: str = "lightonai/LightOnOCR-2-1B", endpoint: str = "http://localhost:8000"):
        """
        Initialize the vLLM-based LightOnOCR model.

        Args:
            model_name: Hugging Face model name (for API reference)
            endpoint: vLLM server endpoint URL
        """
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_url = f"{endpoint}/v1/chat/completions"
        logger.info(f"VLLMOcrModel initialized with endpoint: {self.endpoint}")

    def load(self) -> None:
        """No loading needed - vLLM server handles model loading."""
        logger.info("vLLM server handles model loading")

    def unload(self) -> None:
        """No unloading needed - vLLM server handles model lifecycle."""
        logger.info("vLLM server handles model lifecycle")

    def _load_image(self, image_input: Union[str, Path, bytes]) -> Image.Image:
        """
        Load image from various input types.

        Args:
            image_input: File path, URL, or image bytes

        Returns:
            PIL Image object
        """
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))

        if isinstance(image_input, Path):
            image_input = str(image_input)

        if isinstance(image_input, str):
            # Check if it's a URL
            if image_input.startswith(("http://", "https://")):
                response = requests.get(image_input)
                return Image.open(io.BytesIO(response.content))
            # Otherwise treat as file path
            return Image.open(image_input)

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            image: PIL Image object

        Returns:
            Base64-encoded image string
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def extract_text(self, image_input: Union[str, Path, bytes]) -> str:
        """
        Extract text from an image using vLLM server.

        Args:
            image_input: Can be a file path, URL, or image bytes

        Returns:
            Extracted text from the image
        """
        # Load image
        image = self._load_image(image_input)

        # Convert to base64
        image_base64 = self._image_to_base64(image)

        # Prepare payload for vLLM API
        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                }]
            }],
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 0.9,
        }

        # Make request to vLLM server
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            text = result['choices'][0]['message']['content']
            return text
        except requests.exceptions.Timeout:
            logger.error("Timeout calling vLLM server after 60 seconds")
            raise TimeoutError("vLLM server request timed out after 60 seconds")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling vLLM server: {e}")
            raise
