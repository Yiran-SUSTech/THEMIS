import os
import random
import shutil
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(r"d:\THEMIS\small_scale_audit")
IMAGENET_VAL_DIR = PROJECT_ROOT / "ImageNet_val"

# ---- User Configuration ----
# Specify the class IDs you want to sample from
CLASS_ID_LIST = [4, 7, 10, 25, 35, 39, 49, 51, 52, 64, 80, 97, 106, 116, 118, 129, 144, 147, 150, 156, 159, 281, 300, 322, 330, 333, 339, 347, 356, 365, 370, 389, 392, 394, 396, 400, 405, 406, 413, 415, 419, 421, 426, 429, 432, 443, 446, 453, 461, 477, 479, 481, 488, 499, 529, 535, 554, 567, 575, 585, 590, 605, 606, 626, 632, 633, 634, 645, 654, 679, 687, 730, 743, 746, 755, 756, 806, 810, 819, 836, 903, 917, 919, 920, 922, 928, 931, 943, 952, 960, 962, 968, 971, 972, 981, 982, 983, 985, 991, 999]
# Number of images to randomly sample per class
IMAGES_PER_CLASS = 5
# Output directory for copied images
OUTPUT_DIR = PROJECT_ROOT / "test_GT_fixed"
# File to record filename -> class_id mapping
CLASS_IDS_FILE = OUTPUT_DIR / "class_ids.txt"
# Starting ID for renamed images
STARTING_ID = 0
# Random seed for reproducibility (set to None for non-deterministic)
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track current image ID and class mappings
    current_id = STARTING_ID
    class_mappings = []

    for class_id in CLASS_ID_LIST:
        class_dir = IMAGENET_VAL_DIR / str(class_id)

        # Check if class directory exists
        if not class_dir.exists():
            print(f"Warning: Class directory {class_dir} does not exist, skipping...")
            continue

        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_files) == 0:
            print(f"Warning: No images found in {class_dir}, skipping...")
            continue

        # Randomly sample up to IMAGES_PER_CLASS images
        selected_images = random.sample(image_files, min(IMAGES_PER_CLASS, len(image_files)))

        # Copy and rename images
        for image_file in selected_images:
            # Generate new filename
            new_filename = f"{current_id:06d}.png"
            src_path = class_dir / image_file
            dst_path = OUTPUT_DIR / new_filename

            # Copy image
            shutil.copy2(src_path, dst_path)

            # Record mapping
            class_mappings.append(f"{current_id:06d} {class_id}")

            print(f"Copied: {image_file} -> {new_filename} (class {class_id})")

            current_id += 1

    # Write class_ids.txt
    with open(CLASS_IDS_FILE, 'w') as f:
        f.write('\n'.join(class_mappings) + '\n')

    print(f"\nDone! Total images copied: {len(class_mappings)}")
    print(f"Class IDs file saved to: {CLASS_IDS_FILE}")


if __name__ == "__main__":
    main()
