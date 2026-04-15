"""FastAPI web application for OCR."""

import base64
import yaml
import sys
import os
from pathlib import Path
from typing import Optional
import importlib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base import OCRModel

# Get vLLM endpoint from environment variable
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000")


class ModelFactory:
    """Factory for creating OCR model instances based on configuration."""

    @staticmethod
    def create_model(config_path: str = "config.yaml") -> OCRModel:
        """Create an OCR model instance based on configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = config['model']
        module_path = model_config['class']
        class_name = module_path.split('.')[-1]
        module_path = '.'.join(module_path.split('.')[:-1])

        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

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


# Create FastAPI app
app = FastAPI(title="OCR Web App", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main upload form."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(html_path)


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded image.

    Args:
        file: Uploaded image file

    Returns:
        JSON response with extracted text
    """
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        
        model = get_model()
        logger.info("Model loaded successfully")

        # Read image data
        image_bytes = await file.read()
        logger.info(f"Image data read, size: {len(image_bytes)} bytes")

        # Extract text
        logger.info("Starting text extraction...")
        text = model.extract_text(image_bytes)
        logger.info("Text extraction completed")

        return JSONResponse(content={
            "success": True,
            "text": text,
            "filename": file.filename
        })
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error during extraction: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_base64")
async def extract_text_base64(image_data: str = Form(...)):
    """
    Extract text from base64 encoded image.

    Args:
        image_data: Base64 encoded image string

    Returns:
        JSON response with extracted text
    """
    try:
        model = get_model()

        # Decode base64
        image_bytes = base64.b64decode(image_data)

        # Extract text
        text = model.extract_text(image_bytes)

        return JSONResponse(content={
            "success": True,
            "text": text
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    """Get information about the current model."""
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
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    import os

    os.chdir(Path(__file__).parent.parent)
    uvicorn.run(app, host="0.0.0.0", port=8080)
