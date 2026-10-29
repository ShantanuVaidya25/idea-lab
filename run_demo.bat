@echo off
REM Smart Classroom - one-click start (webcam + YOLO; switch to Simulated in the sidebar if needed)
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    if exist "C:\Program Files\Python313\python.exe" (
        "C:\Program Files\Python313\python.exe" -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo Could not create the virtual environment. Is Python installed?
        pause
        exit /b 1
    )
)

if not exist .venv\.deps_installed (
    echo Installing packages ^(first run only, a few minutes - PyTorch is large^)...
    .venv\Scripts\python.exe -m pip install -q streamlit pandas numpy pyserial pyyaml requests plyer pytest opencv-python ultralytics
    if errorlevel 1 (
        echo Package install failed.
        pause
        exit /b 1
    )
    echo done> .venv\.deps_installed
)

echo Starting dashboard at http://localhost:8501  (Ctrl+C to stop)
.venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType none
pause

