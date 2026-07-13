@echo off
echo ==============================================
echo CreditPulse AI - Running Data and ML Pipeline
echo ==============================================

set PYTHON_EXE=python
if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
    echo Using local virtual environment python.
) else (
    echo Using system python.
)

echo.
echo [1/4] Running Credit Card Aggregation...
%PYTHON_EXE% pipeline\01_aggregate_cc.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Stage 1 credit card aggregation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Compiling Master Features...
%PYTHON_EXE% pipeline\02_build_features.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Stage 2 feature building failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Running Stacking Model Training...
%PYTHON_EXE% pipeline\03_train.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Stage 3 training failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [4/4] Executing Batch Scoring & SHAP precomputation...
%PYTHON_EXE% pipeline\04_score.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Stage 4 scoring failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ==============================================
echo Pipeline executed successfully!
echo ==============================================
