FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN sed -i 's/opencv-python>=/opencv-python-headless>=/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY classifier.py config.yaml ./
COPY pipeline_outputs/ ./pipeline_outputs/
COPY figures/ ./figures/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8000}"]