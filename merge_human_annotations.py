"""
将 human_anno_IMF-XL2 文件夹下每个 User 的逐图 JSON 合并为一个数据集级 JSON。

输入结构:
    human_anno_IMF-XL2/
        User_1_final_annotations/
            checklist_000000.json
            checklist_000001.json
            ...
        User_2_final_annotations/
            ...
        User_3_final_annotations/
            ...

输出:
    human_anno_IMF-XL2/User_1_final_annotations.json
    human_anno_IMF-XL2/User_2_final_annotations.json
    human_anno_IMF-XL2/User_3_final_annotations.json

格式参考 small_scale_audit_recorrect/output_results/User_X_final_annotations.json:
顶层以 image_name 为 key, value 为逐图 JSON 的完整内容。

用法:
    python merge_human_annotations.py
    python merge_human_annotations.py --input-dir <dir> --output-dir <dir>
    python merge_human_annotations.py --users User_1 User_3
"""
import json
import os
import argparse

BASE_DIR = r"d:\THEMIS"
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "human_anno_IMF-XL2")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR
DEFAULT_USERS = ["User_1", "User_2", "User_3"]
SUBFOLDER_SUFFIX = "_final_annotations"


def derive_image_name_from_filename(fname):
    """checklist_000000.json -> 000000.png (取数字部分 + .png)"""
    stem = os.path.splitext(fname)[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return None
    return digits + ".png"


def merge_one_user(user_name, input_dir, output_dir):
    subfolder = os.path.join(input_dir, user_name + SUBFOLDER_SUFFIX)
    if not os.path.isdir(subfolder):
        print(f"[WARN] 子文件夹不存在, 跳过: {subfolder}")
        return 0

    files = sorted([f for f in os.listdir(subfolder) if f.endswith(".json")])
    if not files:
        print(f"[WARN] 子文件夹中没有 .json 文件: {subfolder}")
        return 0

    merged = {}
    skipped = 0
    for fname in files:
        fpath = os.path.join(subfolder, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取失败 {fpath}: {e}")
            skipped += 1
            continue

        if not isinstance(content, dict):
            print(f"[WARN] 非 dict 内容, 跳过: {fpath}")
            skipped += 1
            continue

        key = content.get("image_name") or derive_image_name_from_filename(fname)
        if key is None:
            print(f"[WARN] 无法确定 image_name, 跳过: {fpath}")
            skipped += 1
            continue

        if key in merged:
            print(f"[WARN] image_name 重复, 后者覆盖前者: {key} (来自 {fname})")
        merged[key] = content

    out_path = os.path.join(output_dir, user_name + "_final_annotations.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[OK] {user_name}: 合并 {len(merged)} 张, 跳过 {skipped} 个 -> {out_path}")
    return len(merged)


def main():
    parser = argparse.ArgumentParser(
        description="合并 human_anno_IMF-XL2 下各 User 的逐图 JSON 为数据集级 JSON。"
    )
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR,
                        help=f"输入根目录 (默认: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="输出目录 (默认同输入目录)")
    parser.add_argument("--users", nargs="+", default=DEFAULT_USERS,
                        help=f"要处理的 User 列表 (默认: {DEFAULT_USERS})")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] 输入目录不存在: {args.input_dir}")
        return

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理 User: {args.users}\n")

    total = 0
    for user in args.users:
        total += merge_one_user(user, args.input_dir, args.output_dir)

    print(f"\n[完成] 共合并 {total} 张图片")


if __name__ == "__main__":
    main()
