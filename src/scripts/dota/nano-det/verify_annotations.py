import json
import os
from tqdm import tqdm

def verify_coco_annotations(annotation_file):
    print(f"\nVerifying {os.path.basename(annotation_file)}...")
    
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        # Check required COCO format keys
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in data:
                print(f"ERROR: Missing required key '{key}'")
                return False
                
        # Verify categories
        print(f"Categories ({len(data['categories'])}):")
        for cat in data['categories']:
            if not all(k in cat for k in ['id', 'name']):
                print(f"ERROR: Invalid category format: {cat}")
                return False
            print(f"  - {cat['name']} (id: {cat['id']})")
            
        # Verify images
        print(f"\nImages: {len(data['images'])}")
        for img in tqdm(data['images'][:5], desc="Sampling images"):
            if not all(k in img for k in ['id', 'width', 'height', 'file_name']):
                print(f"ERROR: Invalid image format: {img}")
                return False
                
        # Verify annotations
        print(f"\nAnnotations: {len(data['annotations'])}")
        for ann in tqdm(data['annotations'][:5], desc="Sampling annotations"):
            if not all(k in ann for k in ['id', 'image_id', 'category_id', 'bbox']):
                print(f"ERROR: Invalid annotation format: {ann}")
                return False
            if len(ann['bbox']) != 4:
                print(f"ERROR: Invalid bbox format: {ann['bbox']}")
                return False
                
        print("\nVerification successful!")
        return True
        
    except json.JSONDecodeError:
        print("ERROR: Invalid JSON format")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    base_dir = "data/my_dataset/dota/dota-v1.0/nano-det/annotations"
    
    # Verify both train and val annotations
    for split in ['train', 'val']:
        annotation_file = os.path.join(base_dir, f"instances_{split}.json")
        if os.path.exists(annotation_file):
            verify_coco_annotations(annotation_file)
        else:
            print(f"ERROR: {annotation_file} not found") 