# Create and activate virtual environment for NanoDet DOTA training
$venvPath = "venvs/dota/venvs/nanodet_dota_env"

# Create virtual environment if it doesn't exist
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath..."
    python -m venv $venvPath
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "$venvPath/Scripts/Activate.ps1"

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements/dota/requirements_nanodet.txt

# Install NanoDet package in development mode
Write-Host "Installing NanoDet package..."
pip install -e src/models/nanodet

Write-Host "Setup complete! Virtual environment is activated." 