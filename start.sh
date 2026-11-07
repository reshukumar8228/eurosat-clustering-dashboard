#!/usr/bin/env bash
set -e

echo "========================================================"
echo "       EuroSAT Studio - Satellite Patch Clustering"
echo "========================================================"
echo ""

# 1. Detect Python
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python was not found in your PATH."
    echo "Please install Python 3.10+."
    exit 1
fi

# 2. Check if dependencies are already satisfied in current environment
echo "Checking Python environment..."
if $PY_CMD -c "import fastapi, uvicorn, sklearn, cv2, PIL, joblib, yaml" &>/dev/null; then
    echo "[OK] Python dependencies are already installed in current environment."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
else
    # Setup .venv if missing
    if [ ! -f ".venv/bin/activate" ]; then
        echo "[1/4] Creating virtual environment (.venv)..."
        $PY_CMD -m venv .venv
    fi
    source .venv/bin/activate

    # Check inside venv
    if ! python -c "import fastapi, uvicorn, sklearn, cv2, PIL, joblib, yaml" &>/dev/null; then
        echo "[2/4] Installing required dependencies into .venv (one-time setup)..."
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo "[OK] Python dependencies verified in .venv."
    fi
fi

# 3. Check Pipeline Outputs (precomputed models)
if [ ! -f "pipeline_outputs/classifier.pkl" ]; then
    echo "[3/4] Running one-time pipeline to prepare models and embeddings..."
    python pipeline.py
else
    echo "[OK] Precomputed models and embeddings verified."
fi

# 4. Check Frontend Build
if [ ! -f "frontend/dist/index.html" ]; then
    if command -v npm &>/dev/null; then
        echo "Building frontend assets..."
        cd frontend
        npm install
        npm run build
        cd ..
    else
        echo "[WARNING] npm not found, proceeding with available assets."
    fi
else
    echo "[OK] Frontend build verified."
fi

# 5. Free port 8000 if occupied
if command -v lsof &>/dev/null; then
    PID=$(lsof -ti:8000 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "Freeing port 8000 (Process PID: $PID)..."
        kill -9 $PID 2>/dev/null || true
    fi
elif command -v fuser &>/dev/null; then
    fuser -k 8000/tcp 2>/dev/null || true
fi

# 6. Launch Server & Browser
echo ""
echo "========================================================"
echo "[4/4] Starting EuroSAT Studio on http://localhost:8000"
echo "========================================================"
echo "Server is initializing... Browser will open automatically once ready."
echo ""

python backend/api.py
