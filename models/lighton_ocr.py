"""LightOnOCR-2-1B model implementation."""

import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from PIL import Image
import requests
from io import BytesIO
from typing import Union
from pathlib import Path
import logging

from .base import OCRModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightOnOCRModel(OCRModel):
    """LightOnOCR-2-1B model for document OCR."""

    def __init__(self, model_name: str = "lightonai/LightOnOCR-2-1B", device: str = "auto"):
        """
        Initialize the LightOnOCR model.

        Args:
            model_name: Hugging Face model name
            device: Device to run on ('auto', 'cuda', 'cpu', 'mps')
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.dtype = torch.float32 if self.device == "mps" else torch.bfloat16
        self.model = None
        self.processor = None
        logger.info(f"LightOnOCRModel initialized with device: {self.device}")

    def _determine_device(self, device: str) -> str:
        """Determine the best available device."""
        if device != "auto":
            return device

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def load(self) -> None:
        """Load the model into memory."""
        logger.info(f"Loading model {self.model_name}...")
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype
        ).to(self.device)
        self.processor = LightOnOcrProcessor.from_pretrained(self.model_name)
        logger.info("Model loaded successfully")

    def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Model unloaded from memory")

    def _load_image(self, image_input: Union[str, Path, bytes]) -> Image.Image:
        """
        Load image from various input types.

        Args:
            image_input: File path, URL, or image bytes

        Returns:
            PIL Image object
        """
        if isinstance(image_input, bytes):
            return Image.open(BytesIO(image_input))

        if isinstance(image_input, Path):
            image_input = str(image_input)

        if isinstance(image_input, str):
            # Check if it's a URL
            if image_input.startswith(("http://", "https://")):
                response = requests.get(image_input)
                return Image.open(BytesIO(response.content))
            # Otherwise treat as file path
            return Image.open(image_input)

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def extract_text(self, image_input: Union[str, Path, bytes]) -> str:
        """
        Extract text from an image.

        Args:
            image_input: Can be a file path, URL, or image bytes

        Returns:
            Extracted text from the image
        """
        if self.model is None or self.processor is None:
            self.load()

        # Load image
        image = self._load_image(image_input)

        # Prepare input using chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {
            k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=2048)

        # Decode only the generated part
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        return output_text
