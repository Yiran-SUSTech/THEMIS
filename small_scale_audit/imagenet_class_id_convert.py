import os

# 你的 ImageNet 验证集目录
VAL_DIR = r"D:\THEMIS\small_scale_audit\ImageNet_val"
# 映射结果文件的保存路径（保存在 small_scale_audit 目录下）
MAPPING_TXT_PATH = r"D:\THEMIS\small_scale_audit\imagenet_class_mapping.txt"

# 1. 获取并严格按照字母升序（Alphabetical Order）进行排序
all_folders = [f for f in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, f))]
wnid_folders = sorted([f for f in all_folders if f.startswith('n') and len(f) == 9])

if len(wnid_folders) != 1000:
    print(f"⚠️ 警告: 检测到符合 WNID 规范的文件夹数量为 {len(wnid_folders)} 个，而不是 1000 个。")
    response = input("是否继续按照当前排序重命名并保存映射？(y/n): ")
    if response.lower() != 'y':
        exit(1)

print("开始批量重命名并生成映射文件...")

# 2. 循环重命名并记录映射关系
mapping_lines = []

for class_id, wnid in enumerate(wnid_folders):
    old_path = os.path.join(VAL_DIR, wnid)
    new_path = os.path.join(VAL_DIR, str(class_id))
    
    # 记录映射关系：每行格式为 "Class_ID,WNID" (例如: 0,n01440764)
    mapping_lines.append(f"{class_id},{wnid}\n")
    
    # 执行重命名
    os.rename(old_path, new_path)

# 3. 将映射关系写入本地 .txt 文件
with open(MAPPING_TXT_PATH, "w", encoding="utf-8") as f:
    f.writelines(mapping_lines)

print("\n" + "="*50)
print("【大功告成】文件夹已全部重命名为 0-999 纯数字！")
print(f"【映射已保存】WNID 与 Class ID 的对应关系已成功保存至:\n  -> {MAPPING_TXT_PATH}")
print("="*50)