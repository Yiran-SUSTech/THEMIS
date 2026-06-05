import os
import random
import shutil
from collections import defaultdict

# ==========================================
# 配置路径
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_DIR_1 = os.path.join(BASE_DIR, "test_GT_fixed")
SOURCE_DIR_2 = os.path.join(BASE_DIR, "test_DiT-XL-2-DiT-XL-2-256")
TARGET_DIR = os.path.join(BASE_DIR, "online_test_images")

BACKUP_DIR = os.path.join(BASE_DIR, "online_test_images_backup")

def read_class_ids(folder_path):
    """读取 class_ids.txt 文件，返回 {img_id: class_id} 字典"""
    class_id_file = os.path.join(folder_path, "class_ids.txt")
    image_class_map = {}
    
    if not os.path.exists(class_id_file):
        print(f"   ⚠️  Warning: class_ids.txt not found in {folder_path}")
        return image_class_map
    
    with open(class_id_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                img_id, class_id = line.split()
                image_class_map[img_id] = int(class_id)
    
    return image_class_map

def find_image_file(folder_path, img_id):
    """查找图片文件（支持 .png, .jpg, .jpeg）"""
    for ext in ['.png', '.jpg', '.jpeg']:
        file_path = os.path.join(folder_path, f"{img_id}{ext}")
        if os.path.exists(file_path):
            return file_path, ext
    return None, None

def main():
    print("="*60)
    print("🚀 Image Merge & Reorganization Tool")
    print("="*60)
    
    # 1. 读取两个源文件夹的 class_ids.txt
    print("\n📖 Step 1: Reading class_ids.txt from source folders...")
    print(f"   Source 1: {SOURCE_DIR_1}")
    map_1 = read_class_ids(SOURCE_DIR_1)
    print(f"   ✅ Found {len(map_1)} images")
    
    print(f"\n   Source 2: {SOURCE_DIR_2}")
    map_2 = read_class_ids(SOURCE_DIR_2)
    print(f"   ✅ Found {len(map_2)} images")
    
    # 2. 按 class_id 分组（带来源标记）
    print("\n📊 Step 2: Grouping images by class_id...")
    class_groups = defaultdict(list)  # {class_id: [(img_id, source_folder), ...]}
    
    for img_id, class_id in map_1.items():
        class_groups[class_id].append((img_id, SOURCE_DIR_1))
    
    for img_id, class_id in map_2.items():
        class_groups[class_id].append((img_id, SOURCE_DIR_2))
    
    print(f"   ✅ Found {len(class_groups)} unique classes")
    
    # 统计每个 class 的图片数量
    for class_id in sorted(class_groups.keys()):
        count = len(class_groups[class_id])
        print(f"   Class {class_id}: {count} images")
    
    # 3. 创建备份
    if os.path.exists(TARGET_DIR):
        if not os.path.exists(BACKUP_DIR):
            print(f"\n💾 Step 3: Creating backup of existing online_test_images...")
            print(f"   Backup location: {BACKUP_DIR}")
            shutil.copytree(TARGET_DIR, BACKUP_DIR)
            print("   ✅ Backup completed!")
        else:
            print(f"\n⚠️  Backup directory already exists: {BACKUP_DIR}")
            response = input("   Continue without backup? (y/n): ")
            if response.lower() != 'y':
                print("   Exiting. Please remove or rename the backup directory first.")
                return
    
    # 4. 清空目标目录
    print("\n🧹 Step 4: Cleaning target directory...")
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)
    print("   ✅ Target directory cleaned")
    
    # 5. 重新编号并复制图片
    print("\n📁 Step 5: Renumbering and copying images...")
    print("   🔀 Shuffling images within each class (random mixing from both sources)")
    new_class_id_mapping = []  # [(new_img_id, class_id), ...]
    source_mapping = []  # [(new_img_id, source_folder_name), ...]
    current_number = 0
    copied_count = 0

    for class_id in sorted(class_groups.keys()):
        img_list = class_groups[class_id]

        random.shuffle(img_list)

        for old_img_id, source_folder in img_list:
            new_img_id = f"{current_number:06d}"

            src_file, ext = find_image_file(source_folder, old_img_id)
            if src_file is None:
                print(f"   ⚠️  Warning: Image {old_img_id} not found in {source_folder}, skipping")
                continue

            dst_file = os.path.join(TARGET_DIR, f"{new_img_id}{ext}")

            shutil.copy2(src_file, dst_file)
            copied_count += 1

            new_class_id_mapping.append((new_img_id, class_id))
            source_mapping.append((new_img_id, os.path.basename(source_folder)))
            current_number += 1
    
    print(f"   ✅ Copied {copied_count} images")
    
    # 6. 生成新的 class_ids.txt
    print("\n📝 Step 6: Generating new class_ids.txt...")
    new_class_id_path = os.path.join(TARGET_DIR, "class_ids.txt")
    
    with open(new_class_id_path, "w", encoding="utf-8") as f:
        for new_img_id, class_id in new_class_id_mapping:
            f.write(f"{new_img_id} {class_id}\n")
    
    print(f"   ✅ Generated: {new_class_id_path}")
    print(f"   Total entries: {len(new_class_id_mapping)}")

    # 6.5 生成 source_mapping.txt
    print("\n📝 Step 6.5: Generating source_mapping.txt...")
    source_mapping_path = os.path.join(TARGET_DIR, "source_mapping.txt")

    with open(source_mapping_path, "w", encoding="utf-8") as f:
        for new_img_id, source_name in source_mapping:
            f.write(f"{new_img_id} {source_name}\n")

    print(f"   ✅ Generated: {source_mapping_path}")
    print(f"   Total entries: {len(source_mapping)}")
    
    # 7. 输出统计信息
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Source 1 (test_GT_fixed): {len(map_1)} images")
    print(f"Source 2 (test_DiT-XL-2-DiT-XL-2-256): {len(map_2)} images")
    print(f"Total merged: {len(map_1) + len(map_2)} images")
    print(f"Successfully copied: {copied_count} images")
    print(f"Unique classes: {len(class_groups)}")
    print(f"Images per class: {len(class_groups[0]) if 0 in class_groups else 'N/A'}")
    print(f"\n📁 Target directory: {TARGET_DIR}")
    print(f"💾 Backup location: {BACKUP_DIR}")
    print("\n✅ Done! You can now run app.py with the merged images.")
    print("="*60)

if __name__ == "__main__":
    main()
