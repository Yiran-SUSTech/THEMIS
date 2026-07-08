"""
将 User_3_final_annotations.json 中每张图的 artifact_score 四舍五入到正整数,
并重新计算 total_score = 修改后的 artifact_score * alignment_score。

用法:
    python round_user3_artifact_scores.py                  # 输出到 <input>_rounded.json
    python round_user3_artifact_scores.py --inplace        # 覆盖原文件(自动备份 .bak)
    python round_user3_artifact_scores.py --input <path> --output <path>
"""
import json
import os
import math
import shutil
import argparse

BASE_DIR = r"d:\THEMIS"
DEFAULT_INPUT = os.path.join(
    BASE_DIR, "small_scale_audit_recorrect", "output_results",
    "User_3_final_annotations.json"
)


def round_half_up(x):
    """四舍五入(round half up)到整数。Python 内置 round() 是银行家舍入,
    对 .5 会向偶数靠拢, 与"四舍五入"语义不符, 因此这里手动实现。"""
    # return math.floor(float(x) + 0.5)
    return math.floor(float(x))


def process(data):
    n = 0
    examples = []
    skipped = 0
    for key, val in data.items():
        if not isinstance(val, dict) or "scores" not in val:
            skipped += 1
            continue
        scores = val["scores"]
        if "artifact_score" not in scores or "alignment_score" not in scores:
            skipped += 1
            continue

        old_art = scores["artifact_score"]
        align = scores["alignment_score"]
        new_art = round_half_up(old_art)
        new_total = new_art * float(align)

        if len(examples) < 5:
            examples.append((
                key, old_art, new_art, align,
                scores.get("total_score"), new_total
            ))

        scores["artifact_score"] = new_art
        scores["total_score"] = round(new_total, 2)
        n += 1
    return n, skipped, examples


def main():
    parser = argparse.ArgumentParser(
        description="Round artifact_score and recompute total_score for User_3 annotations."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help=f"输入 JSON 路径 (默认: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 路径 (默认: <input>_rounded.json)")
    parser.add_argument("--inplace", action="store_true",
                        help="覆盖原文件(会自动备份到 .bak)")
    args = parser.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"[ERROR] 输入文件不存在: {in_path}")
        return

    if args.inplace:
        bak_path = in_path + ".bak"
        shutil.copy2(in_path, bak_path)
        print(f"[BACKUP] 原文件已备份到: {bak_path}")
        out_path = in_path
    else:
        if args.output:
            out_path = args.output
        else:
            root, ext = os.path.splitext(in_path)
            out_path = f"{root}_rounded{ext}"

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 读取/解析 JSON 失败: {e}")
        return

    n, skipped, examples = process(data)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] 写入失败: {e}")
        return

    print(f"[OK] 共处理 {n} 张图片, 跳过 {skipped} 条无 scores 的记录")
    print(f"[OK] 输出已写入: {out_path}")
    print("\n[示例] (key | artifact: 旧 -> 新 | align | total: 旧 -> 新):")
    for key, old_art, new_art, align, old_total, new_total in examples:
        print(f"  {key} | {old_art} -> {new_art} | align={align} "
              f"| total: {old_total} -> {new_total}")


if __name__ == "__main__":
    main()
