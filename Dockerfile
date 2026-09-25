FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PYTHONPATH=/app:/app/tools \
  LOG_LEVEL=INFO \
  ENVIRONMENT=local \
  TIMEOUT=30 \
  RELOAD=false

# Create non-root user early
RUN useradd --create-home --shell /bin/bash --uid 1000 mcpuser

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# CPU-only torch first so the resolver doesn't pull the multi-GB CUDA build
RUN pip install --upgrade pip \
  && pip install torch --index-url https://download.pytorch.org/whl/cpu \
  && pip install -r requirements.txt

# ── Pre-download embedding model ──────────────────────────────────────────────
# Bake the sentence-transformers model into the image so containers never
# download it from HuggingFace at runtime (avoids cold-start latency and
# unauthenticated HF rate-limit throttling).  Model is ~90 MB.
ENV HF_HOME=/app/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files with proper ownership
COPY --chown=mcpuser:mcpuser tools /app/tools
COPY --chown=mcpuser:mcpuser server.py /app

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data /app/tmp \
  && chown -R mcpuser:mcpuser /app \
  && chmod -R 755 /app

# Switch to non-root user
USER mcpuser

# Expose port for HTTP server mode
EXPOSE 4398

# Set entrypoint
CMD ["python", "/app/server.py"]