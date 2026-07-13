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
    # 原始模式: 每个 User 100% 使用自己的数据
    python merge_human_annotations.py

    # 混合模式: 使用配置文件指定每个输出 User 的数据来源比例
    python merge_human_annotations.py --config mix_config.json

    # 快捷混合模式: 所有输出 User 使用相同的混合模式
    # (第一个 User 是"自身", 其它是"替代")
    python merge_human_annotations.py --default-mix User_1:0.4,User_4:0.2,User_5:0.2,User_6:0.2

配置文件格式 1 (按图片混合, 整图替换):
    {
      "User_1": {"User_1": 0.4, "User_4": 0.2, "User_5": 0.2, "User_6": 0.2},
      "User_2": {"User_2": 0.4, "User_4": 0.2, "User_5": 0.2, "User_6": 0.2},
      "User_3": {"User_3": 0.4, "User_4": 0.2, "User_5": 0.2, "User_6": 0.2}
    }

    表示: User_1 的输出 JSON 中, 40% 的图片来自 User_1 的标注, 20% 来自 User_4, 等等。

配置文件格式 2 (按字段独立混合, alignment 和 artifact 分别设定比例):
    {
      "User_1": {
        "alignment": {"User_3": 0.4, "User_4": 0.1, "User_5": 0.2, "User_6": 0.3},
        "artifact":  {"User_3": 0.2, "User_4": 0.2, "User_5": 0.2, "User_6": 0.4}
      },
      "User_2": {
        "alignment": {...},
        "artifact":  {...}
      }
    }

    表示: User_1 的 alignment_score 中, 40% 来自 User_3, 10% 来自 User_4, 等等;
          User_1 的 artifact_score  中, 20% 来自 User_3, 20% 来自 User_4, 等等。
    两个字段独立分配, total_score 会自动重算为 alignment_score * artifact_score。

混合逻辑:
    - 输出图片集 = 主 User (输出 User 自身) 的图片集
    - 按图片混合: 随机将整张图片分配给某来源 User
    - 按字段混合: alignment 和 artifact 独立分配来源, 仅替换分数, 保留主 User 的其它字段
    - 如果来源 User 没有该图片, 回退到主 User 的标注 (并打印警告)
"""
import json
import os
import copy
import argparse
import random
import shutil
from datetime import datetime

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


def load_user_data(user_name, input_dir):
    """加载单个 User 的所有图片数据, 返回 dict: image_name -> content

    优先从逐图文件夹加载; 若文件夹不存在, 尝试从已合并的 JSON 文件加载。
    """
    # 1. 尝试从逐图文件夹加载
    subfolder = os.path.join(input_dir, user_name + SUBFOLDER_SUFFIX)
    if os.path.isdir(subfolder):
        files = sorted([f for f in os.listdir(subfolder) if f.endswith(".json")])
        data = {}
        for fname in files:
            fpath = os.path.join(subfolder, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = json.load(f)
            except Exception as e:
                print(f"[WARN] 读取失败 {fpath}: {e}")
                continue

            if not isinstance(content, dict):
                print(f"[WARN] 非 dict 内容, 跳过: {fpath}")
                continue

            key = content.get("image_name") or derive_image_name_from_filename(fname)
            if key is None:
                print(f"[WARN] 无法确定 image_name, 跳过: {fpath}")
                continue

            data[key] = content

        return data

    # 2. 尝试从已合并的 JSON 文件加载
    merged_path = os.path.join(input_dir, user_name + "_final_annotations.json")
    if os.path.isfile(merged_path):
        try:
            with open(merged_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                print(f"[INFO] {user_name}: 从合并文件加载 {len(data)} 张")
                return data
        except Exception as e:
            print(f"[WARN] 读取合并文件失败 {merged_path}: {e}")

    return {}


def extract_score(entry, score_name):
    """从标注条目中提取分数, 支持两种格式:
    - 人工格式: entry["scores"]["alignment_score"]
    - 系统格式: entry["alignment_score"]
    """
    if not isinstance(entry, dict):
        return None
    if "scores" in entry and isinstance(entry["scores"], dict):
        return entry["scores"].get(score_name)
    return entry.get(score_name)


def assign_sources_to_images(images, mix_config, rng):
    """为每张图片分配一个来源, 确保比例尽可能精确。

    返回: {image_name: source_name}
    """
    shuffled = list(images)
    rng.shuffle(shuffled)

    n = len(shuffled)
    sources = sorted(mix_config.keys())

    assignments = {}
    remaining = n
    for i, source in enumerate(sources):
        ratio = mix_config[source]
        if i == len(sources) - 1:
            count = remaining
        else:
            count = int(round(n * ratio))
            remaining -= count
        assignments[source] = count

    result = {}
    idx = 0
    for source in sources:
        for _ in range(assignments[source]):
            if idx >= n:
                break
            result[shuffled[idx]] = source
            idx += 1

    return result


def merge_per_field(user_name, input_dir, output_dir, field_mix_config, seed=None):
    """按字段独立混合 alignment 和 artifact 分数。

    field_mix_config 格式:
    {
        "alignment": {"User_3": 0.4, "User_4": 0.1, ...},
        "artifact":  {"User_3": 0.2, "User_4": 0.2, ...}
    }

    输出保留主 User 的结构 (image_name, class_id, fine_grained_details 等),
    仅替换 alignment_score、artifact_score, 并重算 total_score。
    """
    primary_data = load_user_data(user_name, input_dir)
    if not primary_data:
        print(f"[WARN] {user_name} 没有数据, 跳过")
        return 0

    all_images = sorted(primary_data.keys())
    n = len(all_images)

    align_mix = dict(field_mix_config.get("alignment", {}))
    artifact_mix = dict(field_mix_config.get("artifact", {}))

    if not align_mix or not artifact_mix:
        print(f"[ERROR] {user_name}: 缺少 alignment 或 artifact 配置")
        return 0

    # 归一化
    for label, mix_dict in [("alignment", align_mix), ("artifact", artifact_mix)]:
        total = sum(mix_dict.values())
        if abs(total - 1.0) > 1e-6:
            print(f"[WARN] {user_name} {label}: 比例总和 {total:.4f}, 自动归一化")
            for k in mix_dict:
                mix_dict[k] /= total

    # 为 alignment 和 artifact 独立分配来源 (使用不同的 RNG 确保独立性)
    rng_align = random.Random(seed)
    rng_artifact = random.Random((seed + 1) if seed is not None else None)

    align_assignments = assign_sources_to_images(all_images, align_mix, rng_align)
    artifact_assignments = assign_sources_to_images(all_images, artifact_mix, rng_artifact)

    # 打印分配统计
    for label, assignments in [("alignment", align_assignments), ("artifact", artifact_assignments)]:
        counts = {}
        for s in assignments.values():
            counts[s] = counts.get(s, 0) + 1
        print(f"[INFO] {user_name} {label} 分配: {counts}")

    # 预加载所有来源数据
    all_sources = set(align_mix.keys()) | set(artifact_mix.keys())
    source_data = {}
    for source in all_sources:
        if source == user_name:
            source_data[source] = primary_data
        else:
            source_data[source] = load_user_data(source, input_dir)

    # 混合
    merged = {}
    align_fallback = 0
    artifact_fallback = 0

    for image_name in all_images:
        base = copy.deepcopy(primary_data[image_name])

        # 替换 alignment_score
        align_source = align_assignments.get(image_name, user_name)
        align_src_data = source_data.get(align_source, {})
        align_val = extract_score(align_src_data.get(image_name, {}), "alignment_score")
        if align_val is None:
            align_val = extract_score(base, "alignment_score")
            if align_source != user_name:
                align_fallback += 1

        # 替换 artifact_score
        artifact_source = artifact_assignments.get(image_name, user_name)
        artifact_src_data = source_data.get(artifact_source, {})
        artifact_val = extract_score(artifact_src_data.get(image_name, {}), "artifact_score")
        if artifact_val is None:
            artifact_val = extract_score(base, "artifact_score")
            if artifact_source != user_name:
                artifact_fallback += 1

        # 更新分数
        if "scores" in base and isinstance(base["scores"], dict):
            base["scores"]["alignment_score"] = align_val
            base["scores"]["artifact_score"] = artifact_val
            base["scores"]["total_score"] = align_val * artifact_val
        else:
            base["alignment_score"] = align_val
            base["artifact_score"] = artifact_val

        merged[image_name] = base

    if align_fallback > 0 or artifact_fallback > 0:
        print(f"[INFO] {user_name}: alignment 回退 {align_fallback} 张, "
              f"artifact 回退 {artifact_fallback} 张")

    # 排序并写入
    merged = {k: merged[k] for k in sorted(merged.keys())}
    out_path = os.path.join(output_dir, user_name + "_final_annotations.json")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(out_path):
        backup_path = out_path + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(out_path, backup_path)
        print(f"[BACKUP] 已备份: {backup_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[OK] {user_name}: 合并 {len(merged)} 张 -> {out_path}")
    return len(merged)


def merge_one_user(user_name, input_dir, output_dir, mix_config=None, seed=None):
    """
    合并一个 User 的数据。

    mix_config 支持两种格式:
    - 按图片混合: {"User_1": 0.4, "User_4": 0.2, ...}
    - 按字段混合: {"alignment": {...}, "artifact": {...}}
    如果为 None, 则 100% 使用自身数据。
    seed: int, 随机种子 (用于复现混合结果)
    """
    # 检测按字段混合格式
    if mix_config is not None and "alignment" in mix_config and "artifact" in mix_config:
        return merge_per_field(user_name, input_dir, output_dir, mix_config, seed)
    # 加载主 User 的数据 (输出图片集)
    primary_data = load_user_data(user_name, input_dir)
    if not primary_data:
        print(f"[WARN] {user_name} 没有数据, 跳过")
        return 0

    if mix_config is None:
        # 原始模式: 100% 自身数据
        merged = primary_data
    else:
        # 混合模式
        if seed is not None:
            random.seed(seed)

        # 验证比例总和
        total_ratio = sum(mix_config.values())
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"[WARN] {user_name} 的混合比例总和为 {total_ratio:.4f}, 将自动归一化")
            mix_config = {k: v / total_ratio for k, v in mix_config.items()}

        # 获取所有图片名并打乱
        all_images = list(primary_data.keys())
        random.shuffle(all_images)

        # 计算每个来源应分配的图片数
        assignments = {}
        remaining = len(all_images)
        sources = sorted(mix_config.keys())  # 排序保证确定性

        for i, source in enumerate(sources):
            ratio = mix_config[source]
            if i == len(sources) - 1:
                # 最后一个来源拿剩余所有
                count = remaining
            else:
                count = int(round(len(all_images) * ratio))
                remaining -= count
            assignments[source] = count

        # 打印实际分配数量
        print(f"[INFO] {user_name} 分配: " + ", ".join(f"{s}={c}" for s, c in assignments.items()))

        # 预加载所有需要的 User 数据
        user_data_cache = {user_name: primary_data}
        for source in sources:
            if source != user_name and source not in user_data_cache:
                user_data_cache[source] = load_user_data(source, input_dir)

        # 分配图片
        merged = {}
        idx = 0
        fallback_count = 0

        for source in sources:
            count = assignments[source]
            source_data = user_data_cache.get(source, {})

            for _ in range(count):
                if idx >= len(all_images):
                    break

                image_name = all_images[idx]
                idx += 1

                # 尝试从指定来源获取
                if image_name in source_data:
                    merged[image_name] = source_data[image_name]
                else:
                    # 回退到主 User
                    if source != user_name:
                        fallback_count += 1
                        print(f"[WARN] {source} 没有 {image_name}, 回退到 {user_name}")
                    merged[image_name] = primary_data[image_name]

        if fallback_count > 0:
            print(f"[INFO] {user_name}: 共 {fallback_count} 张图片回退到主 User")

    # 按 image_name 升序排序后写入
    merged = {k: merged[k] for k in sorted(merged.keys())}
    out_path = os.path.join(output_dir, user_name + "_final_annotations.json")
    os.makedirs(output_dir, exist_ok=True)

    # 若输出文件已存在, 先备份
    if os.path.isfile(out_path):
        backup_path = out_path + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(out_path, backup_path)
        print(f"[BACKUP] 已备份: {backup_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[OK] {user_name}: 合并 {len(merged)} 张 -> {out_path}")
    return len(merged)


def parse_default_mix(default_mix_str):
    """
    解析 --default-mix 参数, 格式: "User_1:0.4,User_4:0.2,User_5:0.2,User_6:0.2"
    返回 dict: {"User_1": 0.4, "User_4": 0.2, ...}
    """
    result = {}
    for item in default_mix_str.split(","):
        item = item.strip()
        if ":" not in item:
            print(f"[ERROR] 格式错误: {item}, 应为 User_X:ratio")
            return None
        user, ratio_str = item.split(":", 1)
        try:
            ratio = float(ratio_str)
        except ValueError:
            print(f"[ERROR] 比例不是数字: {ratio_str}")
            return None
        result[user.strip()] = ratio
    return result


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
    parser.add_argument("--config", type=str, default=None,
                        help="混合配置文件路径 (JSON 格式)")
    parser.add_argument("--default-mix", type=str, default=None,
                        help="快捷混合模式, 格式: User_1:0.4,User_4:0.2,User_5:0.2,User_6:0.2")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (用于复现混合结果)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] 输入目录不存在: {args.input_dir}")
        return

    # 解析混合配置
    mix_configs = {}
    if args.config:
        # 从配置文件加载
        if not os.path.isfile(args.config):
            print(f"[ERROR] 配置文件不存在: {args.config}")
            return
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                mix_configs = json.load(f)
            print(f"已加载混合配置: {args.config}")
        except Exception as e:
            print(f"[ERROR] 读取配置文件失败: {e}")
            return
    elif args.default_mix:
        # 从命令行参数解析
        # 格式: "User_1:0.4,User_4:0.2,User_5:0.2,User_6:0.2"
        # 语义: 第一个 User 是"自身"占位符, 对每个输出 User, 自身比例 = 0.4
        #        其余 User 按比例分配剩余比例
        default_mix = parse_default_mix(args.default_mix)
        if default_mix is None:
            return

        # 找到"自身"占位符 (第一个 User) 及其比例
        primary_placeholder = list(default_mix.keys())[0]
        primary_ratio = default_mix[primary_placeholder]
        other_users = {k: v for k, v in default_mix.items() if k != primary_placeholder}

        # 为每个输出 User 生成配置
        for user in args.users:
            user_mix = {user: primary_ratio}
            user_mix.update(other_users)
            mix_configs[user] = user_mix

        print(f"快捷混合模式: 每个输出 User 自身占 {primary_ratio*100:.0f}%, "
              f"其余来源 {[(k, v) for k, v in other_users.items()]}")

    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理 User: {args.users}")
    if mix_configs:
        print(f"混合配置: {mix_configs}")
    print()

    total = 0
    for user in args.users:
        user_mix = mix_configs.get(user)
        total += merge_one_user(user, args.input_dir, args.output_dir,
                               mix_config=user_mix, seed=args.seed)

    print(f"\n[完成] 共合并 {total} 张图片")


if __name__ == "__main__":
    main()
