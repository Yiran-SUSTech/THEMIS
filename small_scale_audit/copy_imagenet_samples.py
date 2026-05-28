import os
import shutil
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(r"d:\THEMIS\small_scale_audit")
IMAGENET_VAL_DIR = PROJECT_ROOT / "ImageNet_val"
OUTPUT_DIR = PROJECT_ROOT / "test_GT_fixed"
CLASS_IDS_FILE = OUTPUT_DIR / "class_ids.txt"

# Class ID ranges: 0-9, 100-109, 200-209, ..., 900-909
CLASS_RANGES = [
    (0, 9),
    (100, 109),
    (200, 209),
    (300, 309),
    (400, 409),
    (500, 509),
    (600, 609),
    (700, 709),
    (800, 809),
    (900, 909),
]

IMAGES_PER_CLASS = 5
STARTING_ID = 500

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track current image ID and class mappings
    current_id = STARTING_ID
    class_mappings = []
    
    # Process each class range
    for start_class, end_class in CLASS_RANGES:
        for class_id in range(start_class, end_class + 1):
            class_dir = IMAGENET_VAL_DIR / str(class_id)
            
            # Check if class directory exists
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} does not exist, skipping...")
                continue
            
            # Get all image files in the class directory
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG', '.PNG'))]
            
            if len(image_files) == 0:
                print(f"Warning: No images found in {class_dir}, skipping...")
                continue
            
            # Take up to IMAGES_PER_CLASS images
            selected_images = image_files[:IMAGES_PER_CLASS]
            
            # Copy and rename images
            for image_file in selected_images:
                # Generate new filename
                new_filename = f"{current_id:06d}.png"
                src_path = class_dir / image_file
                dst_path = OUTPUT_DIR / new_filename
                
                # Copy image
                shutil.copy2(src_path, dst_path)
                
                # Record mapping
                class_mappings.append(f"{new_filename} {class_id}")
                
                print(f"Copied: {image_file} -> {new_filename} (class {class_id})")
                
                current_id += 1
    
    # Write class_ids.txt
    with open(CLASS_IDS_FILE, 'w') as f:
        f.write('\n'.join(class_mappings) + '\n')
    
    print(f"\nDone! Total images copied: {len(class_mappings)}")
    print(f"Class IDs file saved to: {CLASS_IDS_FILE}")

if __name__ == "__main__":
    main()
