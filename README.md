# OCR MCP Server

A dockerized MCP (Model Context Protocol) server that provides OCR (Optical Character Recognition) capabilities using the LightOnOCR-2-1B model from Hugging Face. Includes both an MCP server interface and a web application for text extraction.

## Features

- **MCP Server**: Expose OCR functionality as an MCP tool for AI assistants
- **Web Interface**: User-friendly web application for uploading images and extracting text
- **Model Agnostic**: Easy to swap out OCR models through configuration
- **Docker Support**: Containerized deployment for easy setup
- **GPU Support**: Optional GPU acceleration with vLLM for faster processing
- **vLLM Integration**: Production-grade inference server with GPU optimization

## Architecture

```
ocr-mcp-server/
├── models/              # Model abstraction layer
│   ├── base.py         # OCR model interface
│   ├── lighton_ocr.py  # LightOnOCR implementation (CPU/GPU)
│   └── vllm_ocr.py     # vLLM-based implementation (GPU-optimized)
├── mcp_server/         # MCP server implementation
│   └── server.py       # FastMCP server with OCR tool
├── web_app/           # Web application
│   ├── app.py         # FastAPI web app
│   └── templates/
│       └── index.html # Upload form UI
├── config.yaml        # Model configuration
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Docker Compose orchestration
└── requirements.txt   # Python dependencies
```

### System Components

**vLLM Server (GPU-Accelerated)**
- Runs official `vllm/vllm-openai:latest` Docker image
- Provides OpenAI-compatible API at `http://localhost:8000`
- Optimized for NVIDIA GPUs with FlashAttention v2
- Configurable GPU memory utilization

**MCP Server**
- Connects to vLLM server for OCR inference
- Exposes OCR tools via MCP protocol
- Runs on `http://localhost:8001`

**Web Application**
- User-friendly drag-and-drop interface
- Connects to vLLM server for OCR extraction
- Runs on `http://localhost:8080`

## Quick Start

### Prerequisites

**For GPU Acceleration (Recommended)**
- NVIDIA GPU with CUDA support (tested with RTX 5060)
- NVIDIA Container Toolkit installed
- Docker with GPU runtime support

**For CPU-Only Mode**
- Docker installed
- Sufficient RAM (8GB+ recommended)

### Using Docker Compose (Recommended)

1. Clone or navigate to the project directory
2. Start all services:
```bash
docker-compose up -d
```

3. Access the services:
- Web Interface: http://localhost:8080
- MCP Server: http://localhost:8001/mcp
- vLLM Server: http://localhost:8000 (internal use)

**First Run**: The vLLM server will download the LightOnOCR-2-1B model (~2GB) and compile CUDA kernels. This takes 5-10 minutes on first startup. Subsequent starts will be much faster due to caching.

### Using Docker Manually

1. Build the Docker image:
```bash
docker build -t ocr-mcp-server .
```

2. Run the MCP server:
```bash
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml ocr-mcp-server python -m mcp_server.server
```

3. Run the web application:
```bash
docker run -p 8080:8080 -v $(pwd)/config.yaml:/app/config.yaml ocr-mcp-server python -m web_app.app
```

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the MCP server:
```bash
python -m mcp_server.server
```

3. Run the web application (in another terminal):
```bash
python -m web_app.app
```

## Usage

### Web Interface

1. Open http://localhost:8080 in your browser
2. Upload an image (PNG, JPEG, etc.)
3. Click "Extract Text"
4. View and copy the extracted text

### MCP Server

The MCP server exposes the following tools:

#### `extract_ocr_text`
Extract text from an image.

**Parameters:**
- `image_data` (string): Image data as base64 string or file path/URL
- `is_base64` (boolean, default: true): If true, image_data is base64 encoded

**Returns:** Extracted text from the image

#### `get_model_info`
Get information about the currently loaded OCR model.

**Returns:** Dictionary with model class, parameters, and status

### Integrating with Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "ocr": {
      "transport": {
        "type": "http",
        "url": "http://localhost:8000/mcp"
      }
    }
  }
}
```

## Switching Models

The system is designed to be model-agnostic. To switch to a different OCR model:

1. Implement a new model class in `models/` that inherits from `OCRModel` in `models/base.py`
2. Update `config.yaml` to point to your new model class:

```yaml
model:
  class: "models.your_model.YourOCRModel"
  params:
    model_name: "your-model-name"
    device: "auto"
```

3. Restart the services

Example: Adding a new model
```python
# models/tesseract_ocr.py
from .base import OCRModel
import pytesseract
from PIL import Image

class TesseractOCRModel(OCRModel):
    def __init__(self, lang: str = "eng"):
        self.lang = lang
        self.loaded = False
    
    def load(self) -> None:
        self.loaded = True
    
    def unload(self) -> None:
        self.loaded = False
    
    def extract_text(self, image_input):
        image = self._load_image(image_input)
        return pytesseract.image_to_string(image, lang=self.lang)
    
    def _load_image(self, image_input):
        # Implementation similar to LightOnOCR
        pass
```

## GPU Support

### vLLM GPU Configuration

The system uses vLLM for GPU-accelerated OCR inference. The setup includes:

**GPU Requirements**
- NVIDIA GPU with CUDA support (tested with RTX 5060, 7.96 GiB VRAM)
- NVIDIA Container Toolkit installed
- Docker with GPU runtime support

**GPU Memory Configuration**
- Default GPU memory utilization: 0.7 (70%)
- For RTX 5060: Configured for ~5.57 GiB VRAM usage
- Adjust `--gpu-memory-utilization` in `docker-compose.yml` based on your GPU

**GPU Configuration in docker-compose.yml**
```yaml
vllm-server:
  image: vllm/vllm-openai:latest
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  command: >
    --model lightonai/LightOnOCR-2-1B
    --gpu-memory-utilization 0.7
    --limit-mm-per-prompt '{"image": 1}'
```

### CPU-Only Mode

To run without GPU acceleration:

1. Update `config.yaml` to use the Transformers model:
```yaml
model:
  class: "models.lighton_ocr.LightOnOCRModel"
  params:
    model_name: "lightonai/LightOnOCR-2-1B"
    device: "auto"
```

2. Remove or comment out the vLLM server in `docker-compose.yml`
3. Adjust port mappings (MCP server back to 8000)

## Model Information

Currently using: **LightOnOCR-2-1B** from Hugging Face

- 1B parameter vision-language model
- State-of-the-art OCR performance
- Supports multiple languages
- Fast processing (3.3× faster than Chandra OCR)
- Efficient: Processes 5.71 pages/s on H100

## Performance

### vLLM vs Transformers

**vLLM (GPU-Accelerated)**
- FlashAttention v2 optimization
- PagedAttention memory management
- Continuous batching for higher throughput
- Recommended for production use with GPU

**Transformers (CPU/GPU)**
- Direct model loading via Hugging Face
- Simpler setup, no external dependencies
- Suitable for CPU-only environments
- Good for development and testing

**Benchmark Results (RTX 5060, 7.96 GiB VRAM)**
- Model loading: ~1.88 GiB VRAM
- GPU memory utilization: 70% (~5.57 GiB)
- First initialization: 5-10 minutes (model download + compilation)
- Subsequent startups: ~30 seconds
- OCR extraction: Sub-second for typical images

## Configuration

### config.yaml

Edit `config.yaml` to customize the OCR model and parameters:

**vLLM Configuration (GPU-Accelerated)**
```yaml
model:
  class: "models.vllm_ocr.VLLMOcrModel"
  params:
    model_name: "lightonai/LightOnOCR-2-1B"
    endpoint: "http://localhost:8000"
```

**Transformers Configuration (CPU/GPU)**
```yaml
model:
  class: "models.lighton_ocr.LightOnOCRModel"
  params:
    model_name: "lightonai/LightOnOCR-2-1B"
    device: "auto"  # Options: "auto", "cuda", "cpu", "mps"
```

### docker-compose.yml

**vLLM Server Settings**
- `--gpu-memory-utilization`: GPU memory fraction (0.1-1.0)
- `--limit-mm-per-prompt`: Multimodal input limits
- `--mm-processor-cache-gb`: MM processor cache size
- `--no-enable-prefix-caching`: Disable prefix caching

**Environment Variables**
- `VLLM_ENDPOINT`: vLLM server URL (default: http://vllm-server:8000)
- `PYTHONUNBUFFERED`: Disable Python output buffering

## Troubleshooting

### vLLM Server Issues

**vLLM server fails to start with GPU memory error**
```
ValueError: Free memory on device cuda:0 is less than desired GPU memory utilization
```
Solution: Reduce `--gpu-memory-utilization` in docker-compose.yml (try 0.6 or 0.5)

**vLLM server takes long to initialize**
- First run: Model download (~2GB) and CUDA kernel compilation (5-10 minutes)
- Subsequent runs: Much faster due to caching
- Check logs: `docker logs ocr-vllm-server`

**vLLM server unhealthy during startup**
- The server needs time to compile CUDA kernels
- Temporarily remove health check from docker-compose.yml during initial setup
- Server will become healthy after initialization completes

### Model download issues
The model will be downloaded automatically on first run (~2GB). Ensure you have sufficient disk space and internet connection.

### Out of memory errors
- Reduce `--gpu-memory-utilization` in docker-compose.yml
- Reduce image resolution before uploading
- Use CPU mode by switching to LightOnOCRModel in config.yaml
- Ensure sufficient RAM/VRAM is available

### GPU not detected
- Verify NVIDIA Container Toolkit is installed
- Check Docker GPU runtime: `docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`
- Ensure GPU is accessible on host: `nvidia-smi`

### Port conflicts
Change the port mappings in `docker-compose.yml`:
- vLLM Server: 8000
- MCP Server: 8001
- Web App: 8080

## License

This project uses the LightOnOCR-2-1B model under the Apache License 2.0.

## Contributing

To add new OCR models:
1. Create a new file in `models/` directory
2. Implement the `OCRModel` interface
3. Update `config.yaml` to use your model
4. Test with both MCP server and web app

## Future Enhancements

- Batch processing for multiple images
- PDF document support
- Additional OCR model integrations
- API key authentication
- Rate limiting
- Result caching
- Multi-GPU tensor parallelism for larger models
- Quantized model support for lower VRAM requirements
- Streaming OCR results for real-time feedback
