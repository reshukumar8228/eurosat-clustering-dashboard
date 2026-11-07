@echo off
setlocal enabledelayedexpansion
title EuroSAT Studio - Automated Launcher

echo ========================================================
echo        EuroSAT Studio - Satellite Patch Clustering
echo ========================================================
echo.

:: 1. Detect Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where py >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Python was not found in your PATH.
        echo Please install Python 3.10+ from https://www.python.org/
        echo Make sure to check Add Python to PATH during installation.
        pause
        exit /b 1
    ) else (
        set "PY_CMD=py"
    )
) else (
    set "PY_CMD=python"
)

:: 2. Check if dependencies are already satisfied in current environment
echo Checking Python environment...
%PY_CMD% -c "import fastapi, uvicorn, sklearn, cv2, PIL, joblib, yaml" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python dependencies are already installed in current environment.
    if exist ".venv\Scripts\activate.bat" (
        call .venv\Scripts\activate.bat
    )
) else (
    if not exist ".venv\Scripts\activate.bat" (
        echo [1/4] Creating virtual environment .venv...
        %PY_CMD% -m venv .venv
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Failed to create virtual environment.
            pause
            exit /b 1
        )
    )
    call .venv\Scripts\activate.bat

    python -c "import fastapi, uvicorn, sklearn, cv2, PIL, joblib, yaml" >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [2/4] Installing required dependencies into .venv - one-time setup...
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Dependency installation failed.
            pause
            exit /b 1
        )
    ) else (
        echo [OK] Python dependencies verified in .venv.
    )
)

:: 3. Check Pipeline Outputs - precomputed models
if not exist "pipeline_outputs\classifier.pkl" (
    echo [3/4] Running one-time pipeline to prepare models and embeddings...
    python pipeline.py
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Pipeline execution failed.
        pause
        exit /b 1
    )
) else (
    echo [OK] Precomputed models and embeddings verified.
)

:: 4. Check Frontend Build
if not exist "frontend\dist\index.html" (
    echo Frontend build missing. Checking for Node.js and npm...
    where npm >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo Building frontend assets...
        cd frontend
        call npm install
        call npm run build
        cd ..
    ) else (
        echo [WARNING] Node.js not found, running with available backend assets.
    )
) else (
    echo [OK] Frontend build verified.
)

:: 5. Free port 8000 if occupied
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Freeing port 8000 - PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

:: 6. Launch Server & Browser
echo.
echo ========================================================
echo [4/4] Starting EuroSAT Studio on http://localhost:8000
echo ========================================================
echo Server is initializing... Browser will open automatically once ready.
echo.

python backend\api.py

pause
