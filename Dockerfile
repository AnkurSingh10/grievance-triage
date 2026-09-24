FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md .
COPY src ./src
COPY streamlit_app.py .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "grievance_triage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
