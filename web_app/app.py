"""FastAPI web application for OCR."""

import base64
import yaml
import sys
import os
from pathlib import Path
from typing import Optional
import importlib
import io
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base import OCRModel

# Get vLLM endpoint from environment variable
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000")


def convert_heic_to_jpeg(image_bytes: bytes) -> bytes:
    """
    Convert HEIC/HEIF image to JPEG format.
    
    Args:
        image_bytes: HEIC/HEIF image bytes
        
    Returns:
        JPEG image bytes
    """
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        
        # Open HEIC image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPEG
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return output.read()
    except ImportError:
        raise HTTPException(
            status_code=500, 
            detail="HEIC support not available. Please install pillow-heif."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert HEIC: {str(e)}")


def convert_pdf_to_jpeg(pdf_bytes: bytes) -> bytes:
    """
    Convert PDF to JPEG format (first page).
    
    Args:
        pdf_bytes: PDF file bytes
        
    Returns:
        JPEG image bytes
    """
    try:
        from pdf2image import convert_from_bytes
        
        # Convert PDF to images (first page only)
        images = convert_from_bytes(pdf_bytes, fmt='jpeg', single_file=True)
        
        if not images:
            raise HTTPException(status_code=500, detail="PDF has no pages")
        
        # Get first image
        image = images[0]
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPEG
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=95)
        output.seek(0)
        
        return output.read()
    except ImportError:
        raise HTTPException(
            status_code=500, 
            detail="PDF support not available. Please install pdf2image and poppler."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert PDF: {str(e)}")


def convert_to_jpeg_if_needed(file: UploadFile) -> bytes:
    """
    Convert file to JPEG if it's HEIC or PDF format.
    
    Args:
        file: Uploaded file
        
    Returns:
        JPEG image bytes
    """
    image_bytes = file.file.read()
    file.file.seek(0)  # Reset file pointer
    
    filename = file.filename.lower()
    
    # Check file extension
    if filename.endswith(('.heic', '.heif')):
        return convert_heic_to_jpeg(image_bytes)
    elif filename.endswith('.pdf'):
        return convert_pdf_to_jpeg(image_bytes)
    else:
        # Return as-is for standard image formats
        return image_bytes


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
        file: Uploaded image file (supports JPEG, PNG, HEIC, HEIF, PDF)

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

        # Convert to JPEG if needed (HEIC, HEIF, PDF)
        logger.info("Converting file to JPEG if needed...")
        image_bytes = convert_to_jpeg_if_needed(file)
        logger.info(f"Image data ready, size: {len(image_bytes)} bytes")

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
