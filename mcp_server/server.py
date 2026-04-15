"""MCP server for OCR functionality."""

import base64
import yaml
import sys
import os
from pathlib import Path
from typing import Optional
import importlib

from mcp.server.fastmcp import FastMCP
from mcp.server.session import ServerSession
from mcp.server.fastmcp import Context

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base import OCRModel

# Get vLLM endpoint from environment variable
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000")


class ModelFactory:
    """Factory for creating OCR model instances based on configuration."""

    @staticmethod
    def create_model(config_path: str = "config.yaml") -> OCRModel:
        """
        Create an OCR model instance based on configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            OCRModel instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = config['model']
        module_path, class_name = model_config['class'].rsplit('.', 1)

        # Dynamically import the model class
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Create instance with parameters from config
        params = model_config.get('params', {})
        
        # Add VLLM_ENDPOINT to params if using vLLM model
        if 'vllm' in class_name.lower():
            params['endpoint'] = VLLM_ENDPOINT
        
        return model_class(**params)


# Global model instance
_model_instance: Optional[OCRModel] = None


def get_model(config_path: str = "config.yaml") -> OCRModel:
    """Get or create the global model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelFactory.create_model(config_path)
        _model_instance.load()
    return _model_instance


# Create MCP server
mcp = FastMCP("OCR Server", json_response=True)


@mcp.tool()
def extract_ocr_text(image_data: str, is_base64: bool = True) -> str:
    """
    Extract text from an image using OCR.

    Args:
        image_data: Image data as base64 string or file path/URL
        is_base64: If True, image_data is base64 encoded; if False, it's a path/URL

    Returns:
        Extracted text from the image
    """
    try:
        model = get_model()

        if is_base64:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            text = model.extract_text(image_bytes)
        else:
            # Treat as path or URL
            text = model.extract_text(image_data)

        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"


@mcp.tool()
def get_model_info() -> dict:
    """
    Get information about the currently loaded OCR model.

    Returns:
        Dictionary with model information
    """
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model']
        return {
            "model_class": model_config['class'],
            "params": model_config.get('params', {}),
            "status": "loaded" if _model_instance is not None else "not loaded"
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="OCR MCP Server")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    args = parser.parse_args()

    # Update config path for model loading
    import os
    os.chdir(Path(__file__).parent.parent)

    # Use uvicorn directly to bind to 0.0.0.0
    uvicorn.run(mcp.sse_app(), host=args.host, port=args.port)
