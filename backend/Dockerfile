FROM python:3.11-slim

# Prevent Python log buffering
ENV PYTHONUNBUFFERED=1

# HuggingFace settings
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_XET=1
ENV TRANSFORMERS_VERBOSITY=error
ENV TRANSFORMERS_OFFLINE=0

# Install system dependencies for OCR + PDF processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file first (better Docker caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU build (required for transformers)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# OPTIONAL: Pre-download models during build (comment out if slow/stuck)
# Models will download on first run and cache in volume instead
# RUN python - <<EOF
# from sentence_transformers import SentenceTransformer
# from transformers import BlipProcessor, BlipForConditionalGeneration
# 
# print("Downloading embedding model...")
# SentenceTransformer("BAAI/bge-base-en-v1.5")
# 
# print("Downloading BLIP caption model...")
# BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# 
# print("All models cached!")
# EOF

# Copy application source code
COPY . .

# Create directories used by the pipeline
RUN mkdir -p /tmp/processing /documents/images

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI with auto-reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]