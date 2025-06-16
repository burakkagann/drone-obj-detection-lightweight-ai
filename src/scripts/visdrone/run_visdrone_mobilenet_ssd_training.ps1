# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Set project root
$projectRoot = "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"

# Step 1: Function to prepare data
function Prepare-Data {
    Write-Host "Preparing VisDrone dataset for MobileNet-SSD..."
    python "$projectRoot\src\models\MobileNet-SSD\create_lmdb\visdrone_to_voc.py" `
        --visdrone_dir "$projectRoot\data\my_dataset\visdrone" `
        --output_dir "$projectRoot\data\my_dataset\visdrone\voc_format"
}

# Step 2: Function to generate LMDB
function Generate-LMDB {
    Write-Host "Generating LMDB files..."
    python "$projectRoot\src\models\MobileNet-SSD\create_lmdb\create_data.py" `
        --root_dir "$projectRoot\data\my_dataset\visdrone\voc_format" `
        --output_dir "$projectRoot\data\my_dataset\visdrone\lmdb"
}

# Step 3: Function to train MobileNet-SSD
function Train-MobileNetSSD {
    Write-Host "Starting MobileNet-SSD training..."
    
    # Update solver parameters
    $solverPath = "$projectRoot\src\models\MobileNet-SSD\solver_train.prototxt"
    $solverContent = Get-Content $solverPath
    $solverContent = $solverContent -replace "base_lr: 0.001", "base_lr: 0.001"
    $solverContent = $solverContent -replace "max_iter: 120000", "max_iter: 120000"
    $solverContent | Set-Content $solverPath

    # Run training
    caffe train `
        --solver="$projectRoot\src\models\MobileNet-SSD\solver_train.prototxt" `
        --weights="$projectRoot\src\models\MobileNet-SSD\mobilenet_iter_73000.caffemodel" `
        --gpu 0
}

# Main script execution
function Run-Training {
    # Prepare data
    Prepare-Data

    # Generate LMDB files
    Generate-LMDB

    # Train MobileNet-SSD model
    Train-MobileNetSSD
}

# Run the training process
Run-Training 