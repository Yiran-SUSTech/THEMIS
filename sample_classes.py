"""
sample_classes.py
=================
从 final_reports 目录中按 class_id 随机采样指定数量的类别,
将选中类别的所有报告文件复制到目标目录, 保持原文件结构和命名。

用法:
    python sample_classes.py --source-dir c2i_faster/output_DiT_1000class_5img_ref_cap_1 \
        --output-dir c2i_faster/output_DiT_rand100class_5img_ref_cap_1 \
        --num-classes 100 --seed 42
"""

import os
import json
import shutil
import argparse
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="从 final_reports 中按 class_id 随机采样类别"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="源目录 (包含 final_reports 子目录)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录 (将创建 final_reports 子目录)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="采样的类别数量 (默认: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42, 设为 -1 则不固定)",
    )
    args = parser.parse_args()

    source_reports = os.path.join(args.source_dir, "final_reports")
    output_reports = os.path.join(args.output_dir, "final_reports")

    if not os.path.isdir(source_reports):
        print(f"[ERROR] 源目录不存在: {source_reports}")
        return

    # 1. 扫描所有 JSON 文件, 按 class_id 分组
    print(f"扫描源目录: {source_reports}")
    class_files = defaultdict(list)
    json_files = [
        f for f in os.listdir(source_reports) if f.endswith(".json")
    ]
    print(f"共找到 {len(json_files)} 个 JSON 文件")

    for fname in json_files:
        fpath = os.path.join(source_reports, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            class_id = data.get("metadata", {}).get("class_id")
            if class_id is None:
                print(f"  [WARN] {fname} 无 class_id, 跳过")
                continue
            class_files[class_id].append(fname)
        except Exception as e:
            print(f"  [WARN] 读取 {fname} 失败: {e}")
            continue

    all_classes = sorted(class_files.keys())
    total_files = sum(len(v) for v in class_files.values())
    print(f"共 {len(all_classes)} 个类别, {total_files} 个文件")

    if args.num_classes > len(all_classes):
        print(f"[WARN] 请求 {args.num_classes} 类, 但只有 {len(all_classes)} 类, 全部选取")
        sampled_classes = all_classes
    else:
        if args.seed >= 0:
            random.seed(args.seed)
        sampled_classes = sorted(random.sample(all_classes, args.num_classes))

    print(f"采样 {len(sampled_classes)} 个类别, seed={args.seed}")

    # 2. 复制文件
    os.makedirs(output_reports, exist_ok=True)
    copied = 0
    for cid in sampled_classes:
        for fname in class_files[cid]:
            src = os.path.join(source_reports, fname)
            dst = os.path.join(output_reports, fname)
            shutil.copy2(src, dst)
            copied += 1

    print(f"\n完成! 已复制 {copied} 个文件到: {output_reports}")
    print(f"采样类别 ID: {sampled_classes}")


if __name__ == "__main__":
    main()
