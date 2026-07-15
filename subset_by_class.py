"""
按 per_class_num 从系统和人工标注中抽选相同子集的图片结果。

从 SOURCE_GROUPS 中指定一对系统组和人工组, 按 class_id 分组,
每组随机抽取 N 张图片, 将系统和人工的对应结果复制到新目录。

用法:
    python subset_by_class.py --sys-group Sys_VAR_ref --human-group Human_VAR --per_class_num 5 --seed 42
    python subset_by_class.py --sys-group Sys_JiTfdloss_ref --human-group Human_JiTfdloss --per_class_num 3
"""
import json
import os
import shutil
import argparse
import random
import re
import sys
from collections import defaultdict

BASE_DIR = r"d:\THEMIS"
sys.path.insert(0, BASE_DIR)
from extract_and_analyze_scores import SOURCE_GROUPS, SOURCES


def get_class_id_mapping(sys_sources):
    """从第一个系统源中读取所有图片的 class_id 映射。

    返回: {image_id: class_id}
    """
    src_name = sys_sources[0]
    src = SOURCES[src_name]
    src_path = src["path"]
    prefix = src.get("prefix", "final_evaluation_report_")

    if not os.path.isdir(src_path):
        print(f"[ERROR] Source directory not found: {src_path}")
        return {}

    mapping = {}
    for fname in sorted(os.listdir(src_path)):
        if not fname.startswith(prefix) or not fname.endswith(".json"):
            continue
        idx_str = fname.replace(prefix, "").replace(".json", "")
        try:
            idx = int(idx_str)
        except ValueError:
            continue

        fpath = os.path.join(src_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            class_id = None
            if "metadata" in data and isinstance(data["metadata"], dict):
                class_id = data["metadata"].get("class_id")
            if class_id is None:
                class_id = data.get("class_id")
            if class_id is not None:
                mapping[idx] = class_id
        except Exception as e:
            print(f"[WARN] Failed to read {fpath}: {e}")

    return mapping


def select_images_by_class(class_id_mapping, per_class_num, seed=None):
    """按 class_id 分组, 每组随机选 per_class_num 张图片。

    返回: set of selected image_ids
    """
    if seed is not None:
        random.seed(seed)

    by_class = defaultdict(list)
    for img_id, cid in class_id_mapping.items():
        by_class[cid].append(img_id)

    selected = set()
    skipped = 0
    for cid, img_ids in sorted(by_class.items()):
        if len(img_ids) <= per_class_num:
            selected.update(img_ids)
            skipped += 1
        else:
            chosen = random.sample(img_ids, per_class_num)
            selected.update(chosen)

    if skipped > 0:
        print(f"[INFO] {skipped} classes had <= {per_class_num} images, took all")

    return selected


def derive_sys_output_path(src_path, total_count):
    """从系统源路径推导输出路径。

    Input:  d:\\THEMIS\\c2i_faster\\output_VAR_ref_cap_1\\final_reports
    Output: d:\\THEMIS\\c2i_faster\\output_VAR_ref_cap_{total}_1\\final_reports
    """
    parent = os.path.dirname(src_path)
    subfolder = os.path.basename(src_path)

    dir_name = os.path.basename(parent)
    m = re.match(r'^(.+)_([0-9]+)$', dir_name)
    if m:
        base = m.group(1)
        run_num = m.group(2)
        new_dir_name = f"{base}_{total_count}_{run_num}"
    else:
        new_dir_name = f"{dir_name}_{total_count}"

    new_parent = os.path.join(os.path.dirname(parent), new_dir_name)
    return os.path.join(new_parent, subfolder)


def derive_human_output_path(src_path, total_count):
    """从人工标注源路径推导输出路径。

    Input:  d:\\THEMIS\\human_anno_VAR-ds24\\User_1_final_annotations.json
    Output: d:\\THEMIS\\human_anno_VAR-ds24-{total}\\User_1_final_annotations.json
    """
    parent = os.path.dirname(src_path)
    filename = os.path.basename(src_path)

    new_parent = f"{parent}-{total_count}"
    return os.path.join(new_parent, filename)


def copy_system_subset(src_name, selected_ids, total_count):
    """复制系统源的选定图片到新目录。"""
    src = SOURCES[src_name]
    src_path = src["path"]
    prefix = src.get("prefix", "final_evaluation_report_")

    if not os.path.isdir(src_path):
        print(f"[WARN] {src_name}: directory not found: {src_path}")
        return 0

    out_path = derive_sys_output_path(src_path, total_count)
    os.makedirs(out_path, exist_ok=True)

    copied = 0
    for img_id in sorted(selected_ids):
        fname = f"{prefix}{img_id:06d}.json"
        src_file = os.path.join(src_path, fname)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(out_path, fname))
            copied += 1

    print(f"[OK] {src_name}: copied {copied}/{len(selected_ids)} files -> {out_path}")
    return copied


def subset_human_json(src_name, selected_ids, total_count):
    """从人工标注 JSON 中抽取选定图片, 写入新文件。"""
    src = SOURCES[src_name]
    src_path = src["path"]

    if not os.path.isfile(src_path):
        print(f"[WARN] {src_name}: file not found: {src_path}")
        return 0

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_path = derive_human_output_path(src_path, total_count)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    subset = {}
    for img_id in sorted(selected_ids):
        key = f"{img_id:06d}.png"
        if key in data:
            subset[key] = data[key]

    subset = {k: subset[k] for k in sorted(subset.keys())}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"[OK] {src_name}: subset {len(subset)}/{len(selected_ids)} images -> {out_path}")
    return len(subset)


def main():
    parser = argparse.ArgumentParser(
        description="按 per_class_num 抽选图片子集, 复制到新目录"
    )
    parser.add_argument("--sys-group", type=str, required=True,
                        help="系统组名 (如 Sys_VAR_ref)")
    parser.add_argument("--human-group", type=str, required=True,
                        help="人工组名 (如 Human_VAR)")
    parser.add_argument("--per_class_num", type=int, required=True,
                        help="每类抽取的图片数量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    args = parser.parse_args()

    sys_group = args.sys_group
    human_group = args.human_group

    if sys_group not in SOURCE_GROUPS:
        print(f"[ERROR] Unknown sys_group: {sys_group}")
        print(f"Available: {list(SOURCE_GROUPS.keys())}")
        return

    if human_group not in SOURCE_GROUPS:
        print(f"[ERROR] Unknown human_group: {human_group}")
        print(f"Available: {list(SOURCE_GROUPS.keys())}")
        return

    sys_sources = SOURCE_GROUPS[sys_group]
    human_sources = SOURCE_GROUPS[human_group]

    print(f"System group: {sys_group} -> {sys_sources}")
    print(f"Human group: {human_group} -> {human_sources}")
    print(f"Per class num: {args.per_class_num}")
    print(f"Seed: {args.seed}")

    # 1. Get class_id mapping from first system source
    class_id_mapping = get_class_id_mapping(sys_sources)
    if not class_id_mapping:
        print("[ERROR] No class_id mapping found")
        return

    num_classes = len(set(class_id_mapping.values()))
    total_images = len(class_id_mapping)
    print(f"\nTotal images: {total_images}, Classes: {num_classes}")

    # 2. Select images by class
    selected_ids = select_images_by_class(class_id_mapping, args.per_class_num, args.seed)
    total_selected = len(selected_ids)
    print(f"Selected: {total_selected} images ({args.per_class_num} per class)")

    # 3. Copy system sources
    print(f"\n--- Copying system sources ---")
    for src_name in sys_sources:
        copy_system_subset(src_name, selected_ids, total_selected)

    # 4. Subset human sources
    print(f"\n--- Subsetting human sources ---")
    for src_name in human_sources:
        subset_human_json(src_name, selected_ids, total_selected)

    print(f"\n[完成] 共抽选 {total_selected} 张图片, 输出目录后缀: _{total_selected}")


if __name__ == "__main__":
    main()
