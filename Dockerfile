# Multi-stage Dockerfile for OCR MCP Server

# Stage 1: Base image with Python
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development stage
FROM base as development

# Copy application code
COPY models/ ./models/
COPY mcp_server/ ./mcp_server/
COPY web_app/ ./web_app/
COPY config.yaml .

# Expose ports
# 8000: MCP server
# 8080: Web application
EXPOSE 8000 8080

# Default command - can be overridden
CMD ["python", "-m", "mcp_server.server", "--port", "8000"]

# Stage 3: Production stage
FROM base as production

# Copy application code
COPY models/ ./models/
COPY mcp_server/ ./mcp_server/
COPY web_app/ ./web_app/
COPY config.yaml .

# Expose ports
EXPOSE 8000 8080

# Run both services using supervisord or similar
# For now, we'll default to running the web app
CMD ["python", "-m", "web_app.app"]
