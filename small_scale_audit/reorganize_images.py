import os
import random
import shutil
from collections import defaultdict

# ==========================================
# 配置路径
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "online_test_images")
CLASS_ID_FILE = os.path.join(IMAGE_DIR, "class_ids.txt")
BACKUP_DIR = os.path.join(BASE_DIR, "online_test_images_backup")

def main():
    # 1. 读取 class_ids.txt
    print("📖 Reading class_ids.txt...")
    image_class_map = {}  # {old_img_id: class_id}
    
    with open(CLASS_ID_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                img_id, class_id = line.split()
                image_class_map[img_id] = int(class_id)
    
    print(f"   Found {len(image_class_map)} images")
    
    # 2. 按 class_id 分组
    print("📊 Grouping images by class_id...")
    class_groups = defaultdict(list)
    for img_id, class_id in image_class_map.items():
        class_groups[class_id].append(img_id)
    
    print(f"   Found {len(class_groups)} unique classes")
    for class_id in sorted(class_groups.keys()):
        print(f"   Class {class_id}: {len(class_groups[class_id])} images")
    
    # 3. 创建备份目录
    if not os.path.exists(BACKUP_DIR):
        print(f"\n💾 Creating backup at: {BACKUP_DIR}")
        shutil.copytree(IMAGE_DIR, BACKUP_DIR)
        print("   Backup completed!")
    else:
        print(f"\n⚠️  Backup directory already exists: {BACKUP_DIR}")
        response = input("   Do you want to skip backup? (y/n): ")
        if response.lower() != 'y':
            print("   Exiting. Please remove or rename the backup directory first.")
            return
    
    # 4. 对每个 class 组内的图片随机打乱，并重新编号
    print("\n🔀 Shuffling and renumbering...")
    old_to_new_mapping = {}  # {old_img_id: new_img_id}
    current_number = 0
    
    for class_id in sorted(class_groups.keys()):
        img_list = class_groups[class_id]
        random.shuffle(img_list)  # 随机打乱
        
        for old_img_id in img_list:
            new_img_id = f"{current_number:06d}"
            old_to_new_mapping[old_img_id] = new_img_id
            current_number += 1
    
    # 5. 生成新的 class_ids.txt
    new_class_id_path = os.path.join(IMAGE_DIR, "class_ids_new.txt")
    print(f"\n📝 Generating new class_ids.txt: {new_class_id_path}")
    
    with open(new_class_id_path, "w", encoding="utf-8") as f:
        for old_img_id, new_img_id in sorted(old_to_new_mapping.items(), key=lambda x: int(x[1])):
            class_id = image_class_map[old_img_id]
            f.write(f"{new_img_id} {class_id}\n")
    
    # 6. 重命名图片文件
    print("\n📁 Renaming image files...")
    renamed_count = 0
    
    for old_img_id, new_img_id in old_to_new_mapping.items():
        # 查找原始文件（可能是 .png, .jpg, .jpeg）
        old_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(IMAGE_DIR, f"{old_img_id}{ext}")
            if os.path.exists(candidate):
                old_file = candidate
                old_ext = ext
                break
        
        if old_file is None:
            print(f"   ⚠️  Warning: Image {old_img_id} not found, skipping")
            continue
        
        new_file = os.path.join(IMAGE_DIR, f"{new_img_id}{old_ext}")
        
        # 如果新文件名已存在，先删除（理论上不应该发生）
        if os.path.exists(new_file) and old_file != new_file:
            os.remove(new_file)
        
        if old_file != new_file:
            os.rename(old_file, new_file)
            renamed_count += 1
    
    print(f"   ✅ Renamed {renamed_count} files")
    
    # 7. 替换旧的 class_ids.txt
    print(f"\n🔄 Replacing class_ids.txt...")
    os.remove(CLASS_ID_FILE)
    os.rename(new_class_id_path, CLASS_ID_FILE)
    print("   ✅ class_ids.txt updated")
    
    # 8. 输出统计信息
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    print(f"Total images processed: {len(old_to_new_mapping)}")
    print(f"Files renamed: {renamed_count}")
    print(f"Backup location: {BACKUP_DIR}")
    print(f"New class_ids.txt: {CLASS_ID_FILE}")
    print("\n✅ Done! You can now run app.py with the reorganized images.")
    print("="*50)

if __name__ == "__main__":
    # 设置随机种子（可选，便于复现）
    # random.seed(42)
    main()
