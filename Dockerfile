# ==========================================
# Stage 1: Build the React Frontend
# ==========================================
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ==========================================
# Stage 2: Python Backend & Unified App
# ==========================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system libraries needed for image/geospatial processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create required runtime directories
RUN mkdir -p Data/EuroSAT uploads pipeline_outputs figures

# Install Python dependencies with headless OpenCV
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application, models, and precomputed data
COPY backend/ ./backend/
COPY classifier.py features.py pipeline.py pipeline_enhancements.py config.yaml ./
COPY pipeline_outputs/ ./pipeline_outputs/
COPY figures/ ./figures/
COPY Data/ ./Data/

# Copy built React frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Security: Create and switch to non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose container port
EXPOSE 8000

# Start FastAPI server using cloud dynamic $PORT
CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
