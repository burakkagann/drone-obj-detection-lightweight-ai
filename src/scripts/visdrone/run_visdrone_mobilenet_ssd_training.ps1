# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Step 1: Function to prepare data
function Prepare-Data {
    Write-Host "Preparing VisDrone dataset for MobileNet-SSD..."
    python ..\..\..\src\models\MobileNet-SSD\create_lmdb\visdrone_to_voc.py `
        --visdrone_dir "..\..\..\data\my_dataset\visdrone" `
        --output_dir "..\..\..\data\my_dataset\visdrone\voc_format"
}

# Step 2: Function to generate LMDB
function Generate-LMDB {
    Write-Host "Generating LMDB files..."
    python ..\..\..\src\models\MobileNet-SSD\create_lmdb\create_data.py `
        --root_dir "..\..\..\data\my_dataset\visdrone\voc_format" `
        --output_dir "..\..\..\data\my_dataset\visdrone\lmdb"
}

# Step 3: Function to train MobileNet-SSD
function Train-MobileNetSSD {
    Write-Host "Starting MobileNet-SSD training..."
    
    # Update solver parameters
    $solverContent = Get-Content "..\..\..\src\models\MobileNet-SSD\solver_train.prototxt"
    $solverContent = $solverContent -replace "base_lr: 0.001", "base_lr: 0.001"
    $solverContent = $solverContent -replace "max_iter: 120000", "max_iter: 120000"
    $solverContent | Set-Content "..\..\..\src\models\MobileNet-SSD\solver_train.prototxt"

    # Run training
    caffe train `
        --solver="..\..\..\src\models\MobileNet-SSD\solver_train.prototxt" `
        --weights="..\..\..\src\models\MobileNet-SSD\mobilenet_iter_73000.caffemodel" `
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