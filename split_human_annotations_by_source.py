"""
按图片来源拆分人工标注 JSON 文件。

根据 test_images/source_mapping.txt 中每张图片的来源标签
(test_DiT-XL-2-DiT-XL-2-256 或 test_GT_fixed)，将合并后的人工标注 JSON
(如 User_1_final_annotations.json) 拆分成 DiT 和 GT 两份。

输入格式: { "000000.png": {...}, "000001.png": {...}, ... }
输出: 两个独立的 JSON 文件, 格式相同, 只是包含的图片不同。

用法:
    python split_human_annotations_by_source.py \
        --input small_scale_audit_recorrect/output_results/User_1_final_annotations.json \
        --out-dit human_anno_DiT-XL2/User_1_final_annotations.json \
        --out-gt human_anno_val/User_1_final_annotations.json

    # 批量处理 User_1/2/3
    python split_human_annotations_by_source.py \
        --input-dir small_scale_audit_recorrect/output_results \
        --out-dit-dir human_anno_DiT-XL2 \
        --out-gt-dir human_anno_val \
        --users User_1 User_2 User_3

    # 自定义来源标签或 mapping 路径
    python split_human_annotations_by_source.py \
        --input-dir small_scale_audit_recorrect/output_results \
        --out-dit-dir human_anno_DiT-XL2 --out-gt-dir human_anno_val \
        --mapping test_images/source_mapping.txt
"""
import os
import json
import argparse


def load_mapping(mapping_path):
    """加载 source_mapping.txt, 返回 {image_id: source_label}"""
    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def get_image_id_from_key(key):
    """000000.png -> 000000"""
    stem = os.path.splitext(key)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits if digits else None


def split_one_file(input_path, out_dit_path, out_gt_path, mapping, dit_label, gt_label):
    """拆分单个 JSON 文件"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dit_data = {}
    gt_data = {}
    skipped = 0

    for key, value in data.items():
        image_id = get_image_id_from_key(key)
        if image_id is None or image_id not in mapping:
            skipped += 1
            continue

        label = mapping[image_id]
        if label == dit_label:
            dit_data[key] = value
        elif label == gt_label:
            gt_data[key] = value
        else:
            skipped += 1

    os.makedirs(os.path.dirname(out_dit_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_gt_path) or ".", exist_ok=True)

    # 按 image_name 升序排序后写入
    dit_data = {k: dit_data[k] for k in sorted(dit_data.keys())}
    gt_data = {k: gt_data[k] for k in sorted(gt_data.keys())}

    with open(out_dit_path, "w", encoding="utf-8") as f:
        json.dump(dit_data, f, ensure_ascii=False, indent=2)
    with open(out_gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)

    print(f"  DiT -> {out_dit_path}: {len(dit_data)} 张")
    print(f"  GT  -> {out_gt_path}: {len(gt_data)} 张")
    print(f"  skipped: {skipped}")

    return len(dit_data), len(gt_data)


def main():
    parser = argparse.ArgumentParser(
        description="按图片来源拆分人工标注 JSON 文件。"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="单个输入 JSON 文件路径")
    parser.add_argument("--out-dit", type=str, default=None,
                        help="DiT 输出文件路径 (单文件模式)")
    parser.add_argument("--out-gt", type=str, default=None,
                        help="GT 输出文件路径 (单文件模式)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="批量模式: 输入目录 (含 User_X_final_annotations.json)")
    parser.add_argument("--out-dit-dir", type=str, default=None,
                        help="批量模式: DiT 输出目录")
    parser.add_argument("--out-gt-dir", type=str, default=None,
                        help="批量模式: GT 输出目录")
    parser.add_argument("--users", nargs="+",
                        default=["User_1", "User_2", "User_3"],
                        help="批量模式: 要处理的 User 列表")
    parser.add_argument("--mapping", type=str, default="test_images/source_mapping.txt",
                        help="source_mapping.txt 路径")
    parser.add_argument("--dit-label", type=str, default="test_DiT-XL-2-DiT-XL-2-256",
                        help="DiT 来源标签")
    parser.add_argument("--gt-label", type=str, default="test_GT_fixed",
                        help="GT 来源标签")
    args = parser.parse_args()

    mapping_path = os.path.abspath(args.mapping)
    if not os.path.isfile(mapping_path):
        print(f"[ERROR] mapping 文件不存在: {mapping_path}")
        return

    mapping = load_mapping(mapping_path)
    print(f"已加载 mapping: {len(mapping)} 条")

    if args.input:
        # 单文件模式
        if not args.out_dit or not args.out_gt:
            print("[ERROR] 单文件模式需要同时指定 --out-dit 和 --out-gt")
            return
        input_path = os.path.abspath(args.input)
        if not os.path.isfile(input_path):
            print(f"[ERROR] 输入文件不存在: {input_path}")
            return
        print(f"\n处理: {input_path}")
        split_one_file(
            input_path,
            os.path.abspath(args.out_dit),
            os.path.abspath(args.out_gt),
            mapping, args.dit_label, args.gt_label,
        )
    elif args.input_dir:
        # 批量模式
        if not args.out_dit_dir or not args.out_gt_dir:
            print("[ERROR] 批量模式需要同时指定 --out-dit-dir 和 --out-gt-dir")
            return
        input_dir = os.path.abspath(args.input_dir)
        out_dit_dir = os.path.abspath(args.out_dit_dir)
        out_gt_dir = os.path.abspath(args.out_gt_dir)

        total_dit = 0
        total_gt = 0
        for user in args.users:
            input_path = os.path.join(input_dir, f"{user}_final_annotations.json")
            if not os.path.isfile(input_path):
                print(f"\n[WARN] 文件不存在, 跳过: {input_path}")
                continue
            out_dit_path = os.path.join(out_dit_dir, f"{user}_final_annotations.json")
            out_gt_path = os.path.join(out_gt_dir, f"{user}_final_annotations.json")
            print(f"\n处理: {user}")
            d, g = split_one_file(
                input_path, out_dit_path, out_gt_path,
                mapping, args.dit_label, args.gt_label,
            )
            total_dit += d
            total_gt += g

        print(f"\n[完成] 共拆分 {len(args.users)} 个 User")
        print(f"  DiT 总计: {total_dit} 张")
        print(f"  GT  总计: {total_gt} 张")
    else:
        print("[ERROR] 需要指定 --input (单文件) 或 --input-dir (批量)")


if __name__ == "__main__":
    main()
