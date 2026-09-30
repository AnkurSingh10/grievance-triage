FROM python:3.13-slim

WORKDIR /app

# Set environment variables for Python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies (cached unless requirements change) - slim deploy version
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-deploy.txt

# Copy application source code and configuration
COPY pyproject.toml README.md /app/
COPY src ./src

# Install local package without reinstalling dependencies
RUN pip install --no-cache-dir --no-deps -e .

# Create outputs directory & copy FP16 PyTorch model checkpoint + tokenizer for runtime inference
RUN mkdir -p /app/outputs
COPY outputs/submission_muril_model_fp16.pt ./outputs/
COPY outputs/tokenizer.json outputs/tokenizer_config.json ./outputs/

# Pre-download SentenceTransformer (~90MB) for fast startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "grievance_triage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

