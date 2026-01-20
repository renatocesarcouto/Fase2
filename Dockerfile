# ============================================================================
# Medical AI Diagnosis System v2.0 - Docker Container
# ============================================================================

FROM python:3.10-slim

# Metadata
LABEL maintainer="Medical AI Team"
LABEL version="2.0.0"
LABEL description="Breast Cancer Diagnosis API"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create directories for models and data
RUN mkdir -p models data

# Environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command: run API
CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
