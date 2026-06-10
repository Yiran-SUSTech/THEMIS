import json
import os
import re

# ============ 可配置参数 ============
# alignment_score的User权重，c2i平均分权重为 (1 - ALIGN_WEIGHT)
ALIGN_WEIGHT = 0.5
# artifact_score的User权重，c2i平均分权重为 (1 - ARTIFACT_WEIGHT)
ARTIFACT_WEIGHT = 0.5
# ===================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "statistic.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "statistic_adjusted.txt")
AUDIT_DIR = os.path.join(BASE_DIR, "small_scale_audit", "output_results")


def get_total_checkpoints_per_image(audit_dir):
    """从User标注文件中读取每张图片的总监测点数量"""
    image_total_cp = {}
    if not os.path.exists(audit_dir):
        return image_total_cp
    # 只需读一个User文件即可（所有User对同一图片的监测点数量相同）
    files = sorted(f for f in os.listdir(audit_dir) if f.endswith(".json"))
    if not files:
        return image_total_cp
    filepath = os.path.join(audit_dir, files[0])
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    for image_name, info in data.items():
        total = 0
        for group_name, group in info.get("fine_grained_details", {}).items():
            total += len(group)
        image_total_cp[image_name] = total
    return image_total_cp


def compute_possible_alignment_scores(total_checkpoints):
    """
    alignment_score = (checked / (total - N/A)) * 5
    对于给定的total_checkpoints，生成所有可能的alignment_score值
    N/A数量可以从0到total-1（至少要有1个非N/A的检测点才有意义）
    """
    possible = set()
    for na_count in range(total_checkpoints):  # 0 to total-1
        non_na = total_checkpoints - na_count
        if non_na <= 0:
            continue
        for checked in range(non_na + 1):  # 0 to non_na
            score = round((checked / non_na) * 5, 2)
            possible.add(score)
    return sorted(possible)


def snap_to_nearest(value, possible_values):
    """将value snap到possible_values中最近的值"""
    if value is None or not possible_values:
        return value
    closest = min(possible_values, key=lambda x: abs(x - value))
    return closest


def round_artifact(value):
    """artifact_score只能是0-5的整数，四舍五入"""
    if value is None:
        print(f"Warning: artifact_score is None")
        return value
    return max(0, min(5, round(value)))


def parse_line(line):
    """解析statistic.txt的一行数据"""
    parts = line.split()
    image = parts[0]
    values = []
    for v in parts[1:]:
        values.append(float(v) if v != "N/A" else None)
    return image, values


def main():
    # 获取每张图片的总监测点数量
    image_total_cp = get_total_checkpoints_per_image(AUDIT_DIR)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    data_lines = [l.strip() for l in raw_lines if l.strip() and not l.startswith("=")]
    header_line = data_lines[0]
    sources = re.findall(r'\[([^\]]+)\]', header_line)

    lines = []
    # 第一行：来源角色信息
    role_line = f"{'Source':<12}"
    for s in sources[:2]:
        role_line += f"{s:>34}"
    for s in sources[2:]:
        role_line += f"{s + ' adjusted':>34}"
    lines.append(role_line)

    # 第二行：列名表头
    col_header = f"{'Image':<12}"
    for _ in sources:
        col_header += f"{'align':>22}{'artifact':>12}"
    lines.append(col_header)
    lines.append("=" * len(col_header))

    for line in data_lines[1:]:
        image, values = parse_line(line)

        # c2i平均分
        copy_align, copy_art = values[0], values[1]
        out_align, out_art = values[2], values[3]

        if copy_align is not None and out_align is not None:
            avg_align = (copy_align + out_align) / 2
        elif copy_align is not None:
            avg_align = copy_align
        elif out_align is not None:
            avg_align = out_align
        else:
            avg_align = None

        if copy_art is not None and out_art is not None:
            avg_art = (copy_art + out_art) / 2
        elif copy_art is not None:
            avg_art = copy_art
        elif out_art is not None:
            avg_art = out_art
        else:
            avg_art = None

        # 获取该图片的可能alignment_score值
        total_cp = image_total_cp.get(image)
        possible_align = compute_possible_alignment_scores(total_cp) if total_cp else None

        row = f"{image:<12}"
        row += f"{copy_align:>22}{copy_art:>12}" if copy_align is not None else f"{'N/A':>22}{'N/A':>12}"
        row += f"{out_align:>22}{out_art:>12}" if out_align is not None else f"{'N/A':>22}{'N/A':>12}"

        for i in range(2, len(sources)):
            u_align = values[i * 2]
            u_art = values[i * 2 + 1]

            # alignment: 加权 -> snap到最近的可能值
            if u_align is not None and avg_align is not None:
                raw_align = ALIGN_WEIGHT * u_align + (1 - ALIGN_WEIGHT) * avg_align
                if possible_align:
                    adj_align = snap_to_nearest(raw_align, possible_align)
                else:
                    adj_align = round(raw_align, 2)
            else:
                adj_align = u_align

            # artifact: 加权 -> 四舍五入到0-5整数
            if u_art is not None and avg_art is not None:
                raw_art = ARTIFACT_WEIGHT * u_art + (1 - ARTIFACT_WEIGHT) * avg_art
                adj_art = round_artifact(raw_art)
            else:
                adj_art = u_art

            row += f"{adj_align:>22}{adj_art:>12}" if adj_align is not None else f"{'N/A':>22}{'N/A':>12}"

        lines.append(row)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Adjusted results written to: {OUTPUT_FILE}")
    print(f"ALIGN_WEIGHT={ALIGN_WEIGHT}, ARTIFACT_WEIGHT={ARTIFACT_WEIGHT}")
    if image_total_cp:
        print(f"Total checkpoints per image (from audit): {dict(list(image_total_cp.items())[:5])}...")


if __name__ == "__main__":
    main()
