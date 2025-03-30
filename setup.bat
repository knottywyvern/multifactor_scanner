@echo off
setlocal EnableDelayedExpansion

echo Market Scanner Setup
echo ===================
echo.

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.7 or higher and try again.
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

:: Install the package in development mode
echo Installing market_scanner package...
pip install -e .

:: Create necessary directories
echo Creating directories...
if not exist logs mkdir logs
if not exist output mkdir output
if not exist data mkdir data

:: Check if license.txt exists
if not exist license.txt (
    :: Prompt for API key
    echo.
    echo Please enter your EOD Historical Data API Key:
    set /p API_KEY=
    
    :: Save API key to license.txt using delayed expansion
    echo Writing API key to license.txt...
    echo !API_KEY!> license.txt
    
    :: Verify the file was created successfully
    if exist license.txt (
        echo License file created successfully.
        echo Content:
        type license.txt
    ) else (
        echo ERROR: Failed to create license file.
        echo Please create 'license.txt' manually and add your API key to it.
        pause
    )
) else (
    echo license.txt already exists, skipping API key entry.
)

:: Create default regions_data.csv if it doesn't exist
echo Checking for regions_data.csv...
if not exist data\regions_data.csv (
    echo Creating default regions_data.csv...
    python -c "from market_scanner.create_default_regions import create_default_regions_data; create_default_regions_data()"
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Failed to create default regions_data.csv
        echo You will need to create this file manually before running the scanner.
        echo.
    ) else (
        echo Successfully created default regions_data.csv
    )
) else (
    echo regions_data.csv already exists, skipping creation.
)

echo.
echo Setup completed successfully!
echo You can now run the scanner using run_scanner.bat
echo.

pause
endlocal