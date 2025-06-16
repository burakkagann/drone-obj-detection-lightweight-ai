"""
Script to convert VisDrone dataset to VOC format.
"""

from visdrone_to_voc_converter import VisDroneToVOCConverter

def main():
    # Convert training set
    print("\nConverting training set...")
    train_converter = VisDroneToVOCConverter(split='train')
    train_converter.convert_dataset()
    
    # Convert validation set
    print("\nConverting validation set...")
    val_converter = VisDroneToVOCConverter(split='val')
    val_converter.convert_dataset()
    
    print("\n✅ Conversion completed for both training and validation sets!")

if __name__ == '__main__':
    main() 