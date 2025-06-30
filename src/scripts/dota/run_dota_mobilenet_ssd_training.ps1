# Create and activate virtual environment
$venvPath = "venvs/dota/venvs/mobilenet_ssd_dota_env"
python -m venv $venvPath

# Activate virtual environment
& "$venvPath/Scripts/Activate.ps1"

# Install requirements
pip install -r requirements/dota/requirements_mobilenet_ssd.txt

# Convert DOTA dataset to VOC format
python src/scripts/dota/convert_dota_to_mobilenet_ssd.py

# Start training
python src/scripts/dota/train_mobilenet_ssd_dota.py 