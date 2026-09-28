FROM python:3.13-slim

WORKDIR /app

# Set environment variables for Python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source code and configuration
COPY pyproject.toml README.md /app/
COPY src ./src
COPY streamlit_app.py .

# Install local package without reinstalling dependencies
RUN pip install --no-cache-dir --no-deps -e .

# Create outputs directory & copy trained model checkpoint for runtime inference
RUN mkdir -p /app/outputs
COPY outputs/submission_model.pt ./outputs/submission_model.pt

# Pre-download SentenceTransformer model during build to eliminate runtime download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000 8501

CMD ["uvicorn", "grievance_triage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

