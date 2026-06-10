import json
import os

# ============ 可配置参数 ============
n = 0
m = 39
# ===================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "statistic.txt")

# 来源1: c2i_faster\output copy\final_reports\
C2I_OUTPUT_COPY_DIR = os.path.join(BASE_DIR, "c2i_faster", "output copy", "final_reports")
# 来源2: c2i_faster\output\final_reports\
C2I_OUTPUT_DIR = os.path.join(BASE_DIR, "c2i_faster", "output", "final_reports")
# 来源3: small_scale_audit\output_results\ 下的用户标注文件
AUDIT_DIR = os.path.join(BASE_DIR, "small_scale_audit", "output_results")


def read_c2i_report(directory, idx):
    """读取 c2i_faster 的 final_evaluation_report 文件"""
    filename = f"final_evaluation_report_{idx:06d}.json"
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "alignment_score": data.get("alignment_score"),
        "artifact_score": data.get("artifact_score"),
    }


def read_audit_scores(audit_dir, idx):
    """从 small_scale_audit 的用户标注文件中读取指定图片的分数"""
    image_key = f"{idx:06d}.png"
    results = {}
    if not os.path.exists(audit_dir):
        return results
    for filename in sorted(os.listdir(audit_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(audit_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if image_key in data:
                scores = data[image_key].get("scores", {})
                results[filename] = {
                    "alignment_score": scores.get("alignment_score"),
                    "artifact_score": scores.get("artifact_score"),
                }
    return results


def main():
    lines = []
    indices = range(n, m + 1)

    # 表头
    header = f"{'Image':<12}"
    sources = ["c2i (output copy)", "c2i (output)"]
    # 先收集 audit 来源名称
    audit_filenames = []
    if os.path.exists(AUDIT_DIR):
        audit_filenames = sorted(
            f for f in os.listdir(AUDIT_DIR) if f.endswith(".json")
        )
    for af in audit_filenames:
        sources.append(af.replace("_final_annotations.json", ""))
    header += "".join(f"{'[' + s + '] align':>22}{'artifact':>12}" for s in sources)
    lines.append(header)
    lines.append("=" * len(header))

    for idx in indices:
        image_name = f"{idx:06d}.png"
        row = f"{image_name:<12}"

        # c2i output copy
        scores_copy = read_c2i_report(C2I_OUTPUT_COPY_DIR, idx)
        if scores_copy:
            row += f"{scores_copy['alignment_score']:>22}{scores_copy['artifact_score']:>12}"
        else:
            row += f"{'N/A':>22}{'N/A':>12}"

        # c2i output
        scores_out = read_c2i_report(C2I_OUTPUT_DIR, idx)
        if scores_out:
            row += f"{scores_out['alignment_score']:>22}{scores_out['artifact_score']:>12}"
        else:
            row += f"{'N/A':>22}{'N/A':>12}"

        # audit sources
        audit_scores = read_audit_scores(AUDIT_DIR, idx)
        for af in audit_filenames:
            if af in audit_scores:
                s = audit_scores[af]
                row += f"{s['alignment_score']:>22}{s['artifact_score']:>12}"
            else:
                row += f"{'N/A':>22}{'N/A':>12}"

        lines.append(row)

    # 写入文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Results written to: {OUTPUT_FILE}")
    print(f"Range: {n:06d}.png ~ {m:06d}.png ({m - n + 1} images)")


if __name__ == "__main__":
    main()
