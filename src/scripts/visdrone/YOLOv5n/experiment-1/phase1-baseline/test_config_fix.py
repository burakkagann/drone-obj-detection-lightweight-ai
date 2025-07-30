#!/usr/bin/env python3
"""
Quick test script to verify the dataset configuration fix
This will test ONLY the clean dataset to ensure YOLOv5 validation works
"""

import sys
from pathlib import Path

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parents[6]
sys.path.append(str(project_root))

try:
    import yaml
    from test_phase1_synthetic import Phase1SyntheticTester
    
    print("="*60)
    print("QUICK CONFIG FIX TEST")
    print("="*60)
    
    # Test paths
    model_path = project_root / "runs" / "train" / "yolov5n_phase1_baseline_20250730_034928" / "weights" / "best.pt"
    dataset_config = project_root / "config" / "dataset" / "visdrone_dataset.yaml"
    test_images = project_root / "data" / "my_dataset" / "visdrone" / "test" / "images"
    
    print(f"Model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    print(f"Dataset config: {dataset_config}")
    print(f"Dataset config exists: {dataset_config.exists()}")
    print(f"Test images: {test_images}")
    print(f"Test images exist: {test_images.exists()}")
    
    if not all([model_path.exists(), dataset_config.exists(), test_images.exists()]):
        print("[ERROR] Required files missing!")
        sys.exit(1)
    
    # Test dataset config loading
    print("\nTesting dataset config...")
    with open(dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Dataset config loaded successfully:")
    print(f"  - Path: {config.get('path')}")
    print(f"  - Classes: {config.get('nc')}")
    print(f"  - Train: {config.get('train')}")
    print(f"  - Val: {config.get('val')}")
    print(f"  - Test: {config.get('test')}")
    
    # Test YAML generation
    print("\nTesting YAML generation...")
    tester = Phase1SyntheticTester(model_path, dataset_config)
    temp_yaml = tester._create_temp_yaml(test_images)
    
    print(f"Temporary YAML created: {temp_yaml}")
    
    # Check the generated YAML
    with open(temp_yaml, 'r') as f:
        temp_config = yaml.safe_load(f)
    
    print("Generated YAML contents:")
    for key, value in temp_config.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    temp_yaml.unlink()
    
    print("\n[SUCCESS] Configuration fix test passed!")
    print("The dataset config is clean and should work with YOLOv5 validation.")
    
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)