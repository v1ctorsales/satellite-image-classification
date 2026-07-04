FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy the trained model — must exist locally before building the image
# Run: python src/train.py  before running: docker build
COPY outputs/model/model.joblib ./outputs/model/model.joblib

# Cloud Run injects $PORT at runtime (default 8080)
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port $PORT"]