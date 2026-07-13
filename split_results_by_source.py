"""
按图片来源拆分 c2i_faster 输出结果。

根据 test_images/source_mapping.txt 中每张图片的来源标签
(test_DiT-XL-2-DiT-XL-2-256 或 test_GT_fixed)，将指定输出文件夹
(如 output_DiT_val_1) 下的所有子目录文件拆分到两个目标文件夹。

用法:
    python split_results_by_source.py --input c2i_faster/output_DiT_val_1 \
        --out-dit c2i_faster/output_DiT_1 \
        --out-gt c2i_faster/output_val_1

    # 自定义 source_mapping.txt 路径
    python split_results_by_source.py --input c2i_faster/output_DiT_val_1 \
        --out-dit c2i_faster/output_DiT_1 --out-gt c2i_faster/output_val_1 \
        --mapping test_images/source_mapping.txt

    # 自定义来源标签
    python split_results_by_source.py --input ... --out-dit ... --out-gt ... \
        --dit-label my_dit_label --gt-label my_gt_label
"""
import os
import argparse
import shutil
from collections import defaultdict


def load_mapping(mapping_path):
    """加载 source_mapping.txt, 返回 {image_id: source_label}"""
    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_id = parts[0]
                source_label = parts[1]
                mapping[image_id] = source_label
    return mapping


def get_image_id_from_filename(fname):
    """
    从文件名中提取 image_id (数字部分)。
    final_evaluation_report_000000.json -> 000000
    checklist_000123.json -> 000123
    000045_depth.png -> 000045
    """
    stem = os.path.splitext(fname)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits if digits else None


def main():
    parser = argparse.ArgumentParser(
        description="按图片来源拆分 c2i_faster 输出结果。"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="要拆分的源文件夹 (如 c2i_faster/output_DiT_val_1)")
    parser.add_argument("--out-dit", type=str, required=True,
                        help="DiT 生成图片的结果输出文件夹 (如 c2i_faster/output_DiT_1)")
    parser.add_argument("--out-gt", type=str, required=True,
                        help="ImageNet GT 图片的结果输出文件夹 (如 c2i_faster/output_val_1)")
    parser.add_argument("--mapping", type=str, default="test_images/source_mapping.txt",
                        help="source_mapping.txt 路径 (默认: test_images/source_mapping.txt)")
    parser.add_argument("--dit-label", type=str, default="test_DiT-XL-2-DiT-XL-2-256",
                        help="DiT 来源标签 (默认: test_DiT-XL-2-DiT-XL-2-256)")
    parser.add_argument("--gt-label", type=str, default="test_GT_fixed",
                        help="GT 来源标签 (默认: test_GT_fixed)")
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move"],
                        help="复制或移动文件 (默认: copy)")
    args = parser.parse_args()

    # 解析为绝对路径
    input_dir = os.path.abspath(args.input)
    out_dit_dir = os.path.abspath(args.out_dit)
    out_gt_dir = os.path.abspath(args.out_gt)
    mapping_path = os.path.abspath(args.mapping)

    if not os.path.isdir(input_dir):
        print(f"[ERROR] 输入文件夹不存在: {input_dir}")
        return
    if not os.path.isfile(mapping_path):
        print(f"[ERROR] mapping 文件不存在: {mapping_path}")
        return

    mapping = load_mapping(mapping_path)
    print(f"已加载 mapping: {len(mapping)} 条")

    # 统计来源分布
    label_counts = defaultdict(int)
    for label in mapping.values():
        label_counts[label] += 1
    print(f"来源分布: {dict(label_counts)}")

    # 创建目标文件夹
    os.makedirs(out_dit_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)

    # 遍历输入文件夹下的所有子目录
    subdirs = [d for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d))]

    if not subdirs:
        print(f"[WARN] 输入文件夹下没有子目录: {input_dir}")
        return

    total_copied_dit = 0
    total_copied_gt = 0
    total_skipped = 0

    for subdir in sorted(subdirs):
        src_subdir = os.path.join(input_dir, subdir)
        dst_dit_subdir = os.path.join(out_dit_dir, subdir)
        dst_gt_subdir = os.path.join(out_gt_dir, subdir)

        os.makedirs(dst_dit_subdir, exist_ok=True)
        os.makedirs(dst_gt_subdir, exist_ok=True)

        dit_count = 0
        gt_count = 0
        skipped = 0

        files = [f for f in os.listdir(src_subdir) if os.path.isfile(os.path.join(src_subdir, f))]
        for fname in files:
            image_id = get_image_id_from_filename(fname)
            if image_id is None or image_id not in mapping:
                skipped += 1
                continue

            label = mapping[image_id]
            src_path = os.path.join(src_subdir, fname)

            if label == args.dit_label:
                dst_path = os.path.join(dst_dit_subdir, fname)
                if args.mode == "copy":
                    shutil.copy2(src_path, dst_path)
                else:
                    shutil.move(src_path, dst_path)
                dit_count += 1
            elif label == args.gt_label:
                dst_path = os.path.join(dst_gt_subdir, fname)
                if args.mode == "copy":
                    shutil.copy2(src_path, dst_path)
                else:
                    shutil.move(src_path, dst_path)
                gt_count += 1
            else:
                skipped += 1

        print(f"  {subdir}: DiT={dit_count}, GT={gt_count}, skipped={skipped}")
        total_copied_dit += dit_count
        total_copied_gt += gt_count
        total_skipped += skipped

    print(f"\n[完成] 模式={args.mode}")
    print(f"  DiT -> {out_dit_dir}: {total_copied_dit} 个文件")
    print(f"  GT  -> {out_gt_dir}: {total_copied_gt} 个文件")
    print(f"  skipped: {total_skipped} 个文件")


if __name__ == "__main__":
    main()
