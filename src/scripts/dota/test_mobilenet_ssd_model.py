import os
import sys

# Add src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

def test_model_imports():
    """Test MobileNet-SSD model imports and basic functionality"""
    print("Testing MobileNet-SSD model imports:")
    
    try:
        from models.mobilenet_ssd import create_mobilenetv2_ssd_lite
        print("✓ Successfully imported model creator")
        
        # Try creating model
        num_classes = 15  # DOTA dataset classes
        model = create_mobilenetv2_ssd_lite(num_classes)
        print("✓ Successfully created model")
        
        # Print model summary
        print("\nModel Summary:")
        print(f"Number of classes: {num_classes}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except ImportError as e:
        print(f"✗ Model import failed: {str(e)}")
    except Exception as e:
        print(f"✗ Error creating model: {str(e)}")

if __name__ == "__main__":
    test_model_imports() 