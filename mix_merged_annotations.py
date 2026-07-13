"""
对已合并的人工标注 JSON 文件进行混合替换。

以一个基准文件为主体，按指定比例将其中的图片结果替换为其他文件中
相同图片的结果。

示例:
    以 User_3 为基准，30% 的图片用 User_2 的结果替换，
    40% 的图片用 User_1 的结果替换，剩余 30% 保持 User_3 原样。

用法:
    # 基本用法
    python mix_merged_annotations.py \
        --base human_anno_DiT-XL2/User_3_final_annotations.json \
        --replace human_anno_DiT-XL2/User_2_final_annotations.json:0.3 \
                  human_anno_DiT-XL2/User_1_final_annotations.json:0.4 \
        --output human_anno_DiT-XL2/User_3_final_annotations.json

    # 不覆盖原文件，输出到新路径
    python mix_merged_annotations.py \
        --base human_anno_DiT-XL2/User_3_final_annotations.json \
        --replace human_anno_DiT-XL2/User_2_final_annotations.json:0.3 \
                  human_anno_DiT-XL2/User_1_final_annotations.json:0.4 \
        --output human_anno_DiT-XL2/User_3_mixed.json

    # 指定随机种子（可复现）
    python mix_merged_annotations.py ... --seed 42

参数说明:
    --base       : 基准文件路径 (混合的主体)
    --replace    : 可重复，格式为 <文件路径>:<比例>，比例之和应 <= 1.0
                   剩余比例保留基准文件的原数据
    --output     : 输出文件路径 (默认覆盖 --base，会先备份)
    --seed       : 随机种子
    --no-backup  : 覆盖时不创建备份 (默认会备份到 .bak)
"""
import json
import os
import argparse
import random
import shutil
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="对已合并的人工标注 JSON 文件进行混合替换。"
    )
    parser.add_argument("--base", type=str, required=True,
                        help="基准文件路径 (混合的主体)")
    parser.add_argument("--replace", nargs="+", required=True,
                        help="替换源，格式: <文件路径>:<比例> (可多个)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径 (默认覆盖 --base)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--no-backup", action="store_true",
                        help="覆盖时不创建备份")
    args = parser.parse_args()

    base_path = os.path.abspath(args.base)
    output_path = os.path.abspath(args.output) if args.output else base_path

    if not os.path.isfile(base_path):
        print(f"[ERROR] 基准文件不存在: {base_path}")
        return

    # 解析替换源
    replace_sources = []  # [(path, ratio), ...]
    for item in args.replace:
        if ":" not in item:
            print(f"[ERROR] 格式错误: {item}, 应为 <文件路径>:<比例>")
            return
        path_part, ratio_str = item.rsplit(":", 1)
        try:
            ratio = float(ratio_str)
        except ValueError:
            print(f"[ERROR] 比例不是数字: {ratio_str}")
            return
        if ratio < 0 or ratio > 1:
            print(f"[ERROR] 比例应在 [0, 1] 范围内: {ratio}")
            return
        src_path = os.path.abspath(path_part)
        if not os.path.isfile(src_path):
            print(f"[ERROR] 替换源文件不存在: {src_path}")
            return
        replace_sources.append((src_path, ratio))

    total_replace_ratio = sum(r for _, r in replace_sources)
    keep_ratio = 1.0 - total_replace_ratio
    if keep_ratio < -1e-6:
        print(f"[ERROR] 替换比例之和 ({total_replace_ratio:.2f}) 超过 1.0")
        return

    print(f"基准文件: {base_path}")
    for i, (src, ratio) in enumerate(replace_sources):
        print(f"  替换源 {i+1}: {src} ({ratio*100:.0f}%)")
    print(f"  保留基准: {keep_ratio*100:.0f}%")

    if args.seed is not None:
        random.seed(args.seed)

    # 加载基准文件
    with open(base_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    all_images = sorted(base_data.keys())
    n = len(all_images)
    print(f"\n基准文件共 {n} 张图片")

    # 加载所有替换源
    src_data = {}
    for src_path, _ in replace_sources:
        with open(src_path, "r", encoding="utf-8") as f:
            src_data[src_path] = json.load(f)

    # 随机打乱图片，按比例分配
    shuffled = list(all_images)
    random.shuffle(shuffled)

    # 计算每个来源分配的图片数
    assignments = []
    remaining = n
    for i, (src_path, ratio) in enumerate(replace_sources):
        if i == len(replace_sources) - 1 and keep_ratio < 1e-9:
            # 最后一个替换源且没有保留比例，拿剩余所有
            count = remaining
        else:
            count = int(round(n * ratio))
            remaining -= count
        assignments.append((src_path, count))

    keep_count = n - sum(c for _, c in assignments)

    print(f"分配: " + ", ".join(
        f"{os.path.basename(src)}={c}" for src, c in assignments
    ) + f", 保留基准={keep_count}")

    # 分配图片
    mixed = {}
    idx = 0
    fallback_count = 0

    for src_path, count in assignments:
        src_d = src_data[src_path]
        src_name = os.path.basename(src_path)
        for _ in range(count):
            if idx >= n:
                break
            image_name = shuffled[idx]
            idx += 1
            if image_name in src_d:
                mixed[image_name] = src_d[image_name]
            else:
                # 替换源缺少该图片，回退到基准
                mixed[image_name] = base_data[image_name]
                fallback_count += 1
                print(f"[WARN] {src_name} 缺少 {image_name}, 回退到基准")

    # 保留的图片
    while idx < n:
        image_name = shuffled[idx]
        idx += 1
        mixed[image_name] = base_data[image_name]

    if fallback_count > 0:
        print(f"[INFO] 共 {fallback_count} 张图片回退到基准")

    # 按 image_name 升序排序
    mixed = {k: mixed[k] for k in sorted(mixed.keys())}

    # 备份
    if output_path == base_path and not args.no_backup:
        backup_path = base_path + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(base_path, backup_path)
        print(f"已备份原文件: {backup_path}")

    # 写入
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mixed, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 混合后 {len(mixed)} 张图片 -> {output_path}")


if __name__ == "__main__":
    main()
