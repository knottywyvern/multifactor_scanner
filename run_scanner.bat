@echo off
echo Starting Market Scanner...
echo.

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.7 or higher and try again.
    pause
    exit /b 1
)

:: Check if virtual environment exists and activate it
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

:: Create necessary directories if they don't exist
if not exist logs mkdir logs
if not exist output mkdir output
if not exist data mkdir data

:: Run the market scanner
echo Running Market Scanner...
echo Timestamp: %date% %time%
echo.

python -m market_scanner.main

:: Check the exit code
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Market Scanner completed successfully!
    echo Results saved to output directory.
    echo.
) else (
    echo.
    echo ERROR: Market Scanner exited with code %ERRORLEVEL%
    echo Please check the logs for more information.
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo Press any key to exit...
pause >nul
exit /b 0