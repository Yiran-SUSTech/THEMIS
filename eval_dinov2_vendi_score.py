"""
DINOv2 + Vendi Score 多样性评估脚本
=====================================
支持两种模式:
  - flat 模式: 单层图片目录 + class_ids.txt (如 DiT-XL-2 samples)
  - imagenet-val 模式: ImageNet-1K 验证集 (WNID 子目录结构)

使用方式:
    # flat 模式 (默认)
    python eval_dinov2_vendi_score.py
    python eval_dinov2_vendi_score.py --model base --data-dir <path>

    # imagenet-val 模式 (计算 ImageNet-1K 验证集指定 100 类的类内 VS)
    python eval_dinov2_vendi_score.py --mode imagenet-val
    python eval_dinov2_vendi_score.py --mode imagenet-val --model base --device cpu

    # flat 模式 + ImageNet 验证集 baseline 归一化 (vendi_ratio = VS / ImageNet_VS)
    python eval_dinov2_vendi_score.py --data-dir <path> \
        --baseline-csv d:/THEMIS/vendi_ImageNet_Val/per_class_vendi_scores.csv

    # flat 模式 + ImageNet 训练集子集 baseline 归一化 (vendi_ratio_train = VS / ImageNet_train_mean_VS)
    # 可与 --baseline-csv 同时使用, 同时输出两套 ratio
    python eval_dinov2_vendi_score.py --data-dir <path> \
        --baseline-train-csv d:/THEMIS/vendi_score_results_imagenet_train_subset_base/per_class_summary.csv

输出:
    <output_dir>/per_class_vendi_scores.csv    - 每类 VS 明细 (含统计行)
    <output_dir>/vendi_score_histogram.png     - VS 分布直方图
    <output_dir>/vendi_score_ranked.png        - 每类 VS 排序条形图
    当 --baseline-csv 提供时:
      - CSV 中增加 vendi_ratio 列与 MEAN_VENDI_RATIO / MEDIAN_VENDI_RATIO 等统计行
      - 额外生成 vendi_ratio_histogram.png / vendi_ratio_ranked.png
    当 --baseline-train-csv 提供时:
      - CSV 中增加 vendi_ratio_train 列与 MEAN_VENDI_RATIO_TRAIN 等统计行
      - 额外生成 vendi_ratio_train_histogram.png / vendi_ratio_train_ranked.png
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

# 自动将同目录下的 Vendi-Score 加入 sys.path (便于直接运行, 无需 pip install)
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDI_SCORE_REPO = os.path.join(_REPO_DIR, "Vendi-Score")
if os.path.isdir(_VENDI_SCORE_REPO) and _VENDI_SCORE_REPO not in sys.path:
    sys.path.insert(0, _VENDI_SCORE_REPO)

# 优先使用 HuggingFace 官方源（在国内通常可直连）；如官方源不可用可改为镜像
#   镜像地址: https://hf-mirror.com
# 也可通过外部环境变量 HF_ENDPOINT 覆盖
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
# 关闭 Windows symlink 警告
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from transformers import AutoModel, AutoImageProcessor

from vendi_score import vendi


def t_ci95(values):
    """计算 95% 置信区间 (基于 t 分布)。
    返回 (mean, std, se, ci_low, ci_high)。
    - mean: 算术平均
    - std:  样本标准差 (ddof=1)
    - se:   标准误 = std / sqrt(n)
    - ci_low / ci_high: mean +/- t_{0.975, n-1} * se

    n<2 时 std/se/CI 返回 NaN。
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean()) if n > 0 else float("nan")
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan"), float("nan")
    std = float(arr.std(ddof=1))
    se = std / np.sqrt(n)
    t_val = float(scipy_stats.t.ppf(0.975, df=n - 1))
    ci_low = mean - t_val * se
    ci_high = mean + t_val * se
    return mean, std, se, ci_low, ci_high


# ==================== 配置 ====================
DEFAULT_DATA_DIR = r"d:\THEMIS\DiT-XL-2-sample_class_num-100-per_class_img-5"
DEFAULT_OUTPUT_DIR = r"d:\THEMIS\vendi_score_results"

# ImageNet-1K 验证集默认路径
DEFAULT_IMAGENET_VAL_DIR = r"H:\imagenet\ImageNet_val"
DEFAULT_CLASS_DEF_JSON = r"D:\imagenet-ancestors-descendants\ImageNet_1K_class_definition.json"

# ImageNet-1K 训练集子集默认路径 (每个 class id 子目录下含 1000+ 张图)
DEFAULT_IMAGENET_TRAIN_SUBSET_DIR = r"H:\imagenet\imagenet_100_subset"
DEFAULT_BATCH_IMG_COUNT = 50  # 每个批次的图片数 (不足一个批次会被忽略)

# 默认处理的 100 个 ImageNet-1K class id
DEFAULT_TARGET_CLASS_IDS = [
    4, 7, 10, 25, 35, 39, 49, 51, 52, 64, 80, 97, 106, 116, 118, 129,
    144, 147, 150, 156, 159, 281, 300, 322, 330, 333, 339, 347, 356, 365,
    370, 389, 392, 394, 396, 400, 405, 406, 413, 415, 419, 421, 426, 429,
    432, 443, 446, 453, 461, 477, 479, 481, 488, 499, 529, 535, 554, 567,
    575, 585, 590, 605, 606, 626, 632, 633, 634, 645, 654, 679, 687, 730,
    743, 746, 755, 756, 806, 810, 819, 836, 903, 917, 919, 920, 922, 928,
    931, 943, 952, 960, 962, 968, 971, 972, 981, 982, 983, 985, 991, 999,
]

# DINOv2 模型选项 (name -> HF model id, embed_dim)
DINOV2_MODELS = {
    "small": ("facebook/dinov2-small", 384),   # 21M params,  ~0.5GB VRAM
    "base":  ("facebook/dinov2-base",  768),   # 86M params,  ~1.5GB VRAM (推荐)
    "large": ("facebook/dinov2-large", 1024),  # 300M params, ~4GB VRAM
}

# 默认模型缓存目录（Vendi-Score/models）
DEFAULT_MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Vendi-Score", "models")

BATCH_SIZE = 16  # RTX 4070 8GB 安全批大小；若 OOM 可降到 8


# ==================== 工具函数 ====================
def load_class_ids(class_ids_file):
    """读取 class_ids.txt, 返回 {filename: class_id}"""
    mapping = {}
    with open(class_ids_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, cls = line.split()
            mapping[f"{int(idx):06d}.png"] = int(cls)
    return mapping


def load_imagenet_class_def(json_path, target_class_ids):
    """加载 ImageNet-1K class definition JSON，返回指定 class id 的 {class_id: (wnid, class_name)} 映射。

    Args:
        json_path: ImageNet_1K_class_definition.json 路径
        target_class_ids: 需要处理的 ImageNet-1K class id 列表

    Returns:
        dict: {class_id: {"wnid": str, "class_name": str}}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure in {json_path}: expected list, got {type(data).__name__}")

    # 构建 {ImageNet-1K_id: entry}
    id_to_entry = {entry["ImageNet-1K_id"]: entry for entry in data}

    result = {}
    missing = []
    for cid in target_class_ids:
        if cid not in id_to_entry:
            missing.append(cid)
            continue
        entry = id_to_entry[cid]
        result[cid] = {
            "wnid": entry["WNID"],
            "class_name": entry.get("class_name", ""),
        }

    if missing:
        print(f"[WARN] {len(missing)} class ids not found in class definition: {missing[:10]}...")

    return result


def collect_imagenet_val_images(imagenet_val_dir, class_def, image_extensions=(".JPEG", ".jpeg", ".jpg")):
    """从 ImageNet-1K 验证集目录收集指定类别的图片路径。

    Args:
        imagenet_val_dir: ImageNet_val 根目录 (下有 1000 个 WNID 子目录)
        class_def: {class_id: {"wnid": str, "class_name": str}} 映射
        image_extensions: 允许的图片扩展名

    Returns:
        dict: {class_id: [Path, Path, ...]}
        list: 所有图片路径的扁平列表 (按 class_id 升序)
        list: 对应的 class_id 列表 (与扁平列表一一对应)
    """
    val_dir = Path(imagenet_val_dir)
    if not val_dir.exists():
        raise FileNotFoundError(f"ImageNet val directory not found: {val_dir}")

    per_class_files = {}
    flat_files = []
    flat_class_ids = []

    for class_id in sorted(class_def.keys()):
        wnid = class_def[class_id]["wnid"]
        class_name = class_def[class_id]["class_name"]
        class_dir = val_dir / wnid

        if not class_dir.exists():
            print(f"[WARN] WNID directory not found: {class_dir} (class_id={class_id}, name={class_name})")
            continue

        files = sorted([p for p in class_dir.iterdir()
                        if p.suffix in image_extensions])

        if not files:
            print(f"[WARN] No images found in {class_dir}")
            continue

        per_class_files[class_id] = files
        flat_files.extend(files)
        flat_class_ids.extend([class_id] * len(files))

    return per_class_files, flat_files, flat_class_ids


def collect_imagenet_train_subset_batches(train_subset_dir, batch_size=50,
                                          image_extensions=(".jpg", ".jpeg", ".JPEG", ".png")):
    """从 ImageNet-1K 训练集子集目录收集按批分组的图片路径。

    目录结构: <train_subset_dir>/<class_id>/<img>.jpg
    分批策略: 每类内每 batch_size 张图片为一批, 不足一批的尾部图片忽略。

    Args:
        train_subset_dir: 训练集子集根目录 (下有 100 个 class id 子目录)
        batch_size: 每批图片数 (default 50)
        image_extensions: 允许的图片扩展名

    Returns:
        list[dict]: 每个元素 = {
            "class_id": int,
            "batch_idx": int (从 0 开始),
            "files": [Path, ...],   # 长度 = batch_size
        }
        dict: {class_id: {"total_images": int, "num_batches": int, "ignored": int}}
    """
    root = Path(train_subset_dir)
    if not root.exists():
        raise FileNotFoundError(f"train subset directory not found: {root}")

    all_batches = []
    class_info = {}

    # 按 class id 数值排序
    class_dirs = []
    for p in root.iterdir():
        if p.is_dir():
            try:
                cid = int(p.name)
                class_dirs.append((cid, p))
            except ValueError:
                continue
    class_dirs.sort(key=lambda x: x[0])

    for cid, class_dir in class_dirs:
        files = sorted([p for p in class_dir.iterdir()
                        if p.suffix in image_extensions])
        total = len(files)
        num_batches = total // batch_size
        ignored = total - num_batches * batch_size

        class_info[cid] = {
            "total_images": total,
            "num_batches": num_batches,
            "ignored": ignored,
        }

        for b in range(num_batches):
            batch_files = files[b * batch_size:(b + 1) * batch_size]
            all_batches.append({
                "class_id": cid,
                "batch_idx": b,
                "files": batch_files,
            })

    return all_batches, class_info


def load_dinov2(model_key, device, cache_dir=None):
    """通过 transformers 加载 DINOv2 模型 + 预处理器"""
    if model_key not in DINOV2_MODELS:
        raise ValueError(f"unknown model: {model_key}, options: {list(DINOV2_MODELS)}")
    model_id, embed_dim = DINOV2_MODELS[model_key]
    print(f"--> loading DINOv2 model: {model_id} (embed_dim={embed_dim})")
    if cache_dir:
        print(f"    model cache dir: {cache_dir}")
    else:
        print(f"    (if first run, weights will be downloaded from HF mirror)")
    model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    return model, processor, embed_dim


def load_baseline_vendi_scores(baseline_csv):
    """从 baseline CSV 读取 {class_id: vendi_score}。

    支持两种 CSV 格式:
      - ImageNet val 的 per_class_vendi_scores.csv: 列名 'vendi_score'
      - ImageNet train subset 的 per_class_summary.csv: 列名 'mean_vendi_score'
    自动检测列名 (优先 'vendi_score', 其次 'mean_vendi_score')。

    自动跳过 OVERALL/MEAN/MEDIAN/STD/MIN/MAX/MEAN_OF_CLASSES 等统计行
    (通过判断 class_id 是否为整数)。
    """
    baseline = {}
    if not baseline_csv or not os.path.exists(baseline_csv):
        raise FileNotFoundError(f"baseline CSV not found: {baseline_csv}")

    with open(baseline_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "class_id" not in reader.fieldnames:
            raise ValueError(
                f"baseline CSV must have 'class_id' column, got: {reader.fieldnames}")
        # 自动检测 VS 列名
        if "vendi_score" in reader.fieldnames:
            vs_col = "vendi_score"
        elif "mean_vendi_score" in reader.fieldnames:
            vs_col = "mean_vendi_score"
        else:
            raise ValueError(
                f"baseline CSV must have 'vendi_score' or 'mean_vendi_score' column, "
                f"got: {reader.fieldnames}")
        for row in reader:
            raw = row["class_id"].strip()
            try:
                cid = int(raw)
            except ValueError:
                # 跳过 OVERALL/MEAN/MEDIAN/STD/MIN/MAX/MEAN_OF_CLASSES 等统计行
                continue
            try:
                vs = float(row[vs_col])
            except ValueError:
                continue
            baseline[cid] = vs
    return baseline


@torch.no_grad()
def extract_embeddings(images, model, processor, batch_size, device):
    """批量提取 DINOv2 CLS-token 嵌入, 返回 [N, D] numpy"""
    all_embeddings = []
    n = len(images)
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # last_hidden_state: [B, 1+num_patches, D], 取 CLS token (index 0)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_emb.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f"    progress: {min(i + batch_size, n)}/{n} "
                  f"({(i / n) * 100:.1f}%), elapsed={time.time() - t0:.1f}s")
    print(f"    done in {time.time() - t0:.2f}s")
    return np.concatenate(all_embeddings, axis=0)


def compute_vendi_score(X):
    """
    从嵌入矩阵 X [N, D] 计算 Vendi Score.
    - 当 N < D: 用协方差矩阵 K = X X^T (primal)
    - 当 N >= D: 用对偶矩阵 S = X^T X (dual, 更快)
    库函数内部会做 L2 归一化.
    """
    n, d = X.shape
    if n < d:
        return float(vendi.score_X(X))
    else:
        return float(vendi.score_dual(X))


def visualize(per_class_scores, overall_vs, model_key, output_dir):
    """生成两张可视化图: 直方图 + 排序条形图"""
    scores = np.array([r["vendi_score"] for r in per_class_scores])
    mean_vs = scores.mean()
    median_vs = float(np.median(scores))

    # 图 1: 直方图
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=20, edgecolor="black", alpha=0.75, color="steelblue")
    plt.axvline(mean_vs, color="red", linestyle="--", linewidth=2,
                label=f"Mean={mean_vs:.3f}")
    plt.axvline(median_vs, color="orange", linestyle=":", linewidth=2,
                label=f"Median={median_vs:.3f}")
    plt.xlabel("Vendi Score")
    plt.ylabel("Number of Classes")
    plt.title(f"DINOv2 Vendi Score Distribution (model={model_key})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = output_dir / "vendi_score_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"    -> {hist_path}")

    # 图 2: 排序条形图
    sorted_scores = sorted(per_class_scores, key=lambda x: x["vendi_score"])
    plt.figure(figsize=(15, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_scores)))
    plt.bar(range(len(sorted_scores)),
            [s["vendi_score"] for s in sorted_scores],
            color=colors, edgecolor="black", linewidth=0.3)
    plt.axhline(mean_vs, color="red", linestyle="--", linewidth=2,
                label=f"Mean={mean_vs:.3f}")
    plt.xlabel("Class Index (sorted by VS ascending)")
    plt.ylabel("Vendi Score")
    plt.title(f"Per-class Vendi Score (sorted, model={model_key})")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    rank_path = output_dir / "vendi_score_ranked.png"
    plt.savefig(rank_path, dpi=150)
    plt.close()
    print(f"    -> {rank_path}")


def visualize_ratio(per_class_scores, model_key, output_dir, col_name="vendi_ratio"):
    """生成 vendi_ratio 的两张可视化图: 直方图 + 排序条形图。
    per_class_scores 中的每条记录必须含 col_name 字段 (None 的会被跳过)。

    Args:
        col_name: 'vendi_ratio' (以 ImageNet val 为基准) 或
                  'vendi_ratio_train' (以 ImageNet train 为基准)
    """
    # 过滤掉目标 ratio 为 None 的类
    valid = [r for r in per_class_scores if r.get(col_name) is not None]
    if not valid:
        print(f"    [skip] no valid {col_name} to visualize")
        return
    ratios = np.array([r[col_name] for r in valid])
    mean_r = float(ratios.mean())
    median_r = float(np.median(ratios))

    # 根据 col_name 决定显示文案与输出文件名
    if col_name == "vendi_ratio_train":
        label = "Vendi Ratio (Train)"
        label_subtitle = "VS / ImageNet-Train-VS"
        hist_filename = "vendi_ratio_train_histogram.png"
        rank_filename = "vendi_ratio_train_ranked.png"
        hist_color = "teal"
    else:  # vendi_ratio (ImageNet val 基准)
        label = "Vendi Ratio"
        label_subtitle = "VS / ImageNet-VS"
        hist_filename = "vendi_ratio_histogram.png"
        rank_filename = "vendi_ratio_ranked.png"
        hist_color = "seagreen"

    # 图 1: 直方图
    plt.figure(figsize=(10, 5))
    plt.hist(ratios, bins=20, edgecolor="black", alpha=0.75, color=hist_color)
    plt.axvline(mean_r, color="red", linestyle="--", linewidth=2,
                label=f"Mean={mean_r:.3f}")
    plt.axvline(median_r, color="orange", linestyle=":", linewidth=2,
                label=f"Median={median_r:.3f}")
    plt.axvline(1.0, color="purple", linestyle="-.", linewidth=1.5,
                label="Baseline=1.0")
    plt.xlabel(f"{label} ({label_subtitle})")
    plt.ylabel("Number of Classes")
    plt.title(f"DINOv2 {label} Distribution (model={model_key})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = output_dir / hist_filename
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"    -> {hist_path}")

    # 图 2: 排序条形图
    sorted_r = sorted(valid, key=lambda x: x[col_name])
    plt.figure(figsize=(15, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_r)))
    plt.bar(range(len(sorted_r)),
            [s[col_name] for s in sorted_r],
            color=colors, edgecolor="black", linewidth=0.3)
    plt.axhline(mean_r, color="red", linestyle="--", linewidth=2,
                label=f"Mean={mean_r:.3f}")
    plt.axhline(1.0, color="purple", linestyle="-.", linewidth=1.5,
                label="Baseline=1.0")
    plt.xlabel("Class Index (sorted by ratio ascending)")
    plt.ylabel(label)
    plt.title(f"Per-class {label} (sorted, model={model_key})")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    rank_path = output_dir / rank_filename
    plt.savefig(rank_path, dpi=150)
    plt.close()
    print(f"    -> {rank_path}")


# ==================== imagenet-train-subset 模式 ====================
def run_imagenet_train_subset_mode(args, output_dir, device):
    """对 ImageNet-1K 训练集子集按 50 张/批计算 Vendi Score。

    输出 3 个 CSV:
      - per_batch_vendi_scores.csv : 每批 VS (class_id, batch_idx, num_images, vendi_score)
      - per_class_summary.csv      : 每类汇总 (class_id, total_images, num_batches, mean/std/CI)
      - full_report.csv            : 宽表, 每类一行, 各批次 VS 展开为列 + 平均值列
    """
    batch_img_count = args.batch_img_count
    print(f"--> [imagenet-train-subset] collecting batches from {args.imagenet_train_subset_dir}")
    print(f"    batch_img_count = {batch_img_count} (尾部不足 {batch_img_count} 张的批次会被忽略)")

    all_batches, class_info = collect_imagenet_train_subset_batches(
        args.imagenet_train_subset_dir, batch_size=batch_img_count)

    total_batches = len(all_batches)
    total_imgs_used = total_batches * batch_img_count
    total_imgs_all = sum(v["total_images"] for v in class_info.values())
    total_ignored = sum(v["ignored"] for v in class_info.values())
    print(f"    total classes: {len(class_info)}")
    print(f"    total images (all): {total_imgs_all}")
    print(f"    total batches ({batch_img_count}/batch): {total_batches}")
    print(f"    total images used: {total_imgs_used}, ignored (tail): {total_ignored}")

    if total_batches == 0:
        print("[ERROR] no valid batches, abort")
        sys.exit(1)

    # 加载 DINOv2 模型
    cache_dir = args.model_cache_dir if args.model_cache_dir else DEFAULT_MODEL_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    model, processor, embed_dim = load_dinov2(args.model, device, cache_dir=cache_dir)

    # 逐批处理: 每批加载 50 张图 -> 提取 embedding -> 算 VS
    print(f"--> processing {total_batches} batches "
          f"(batch_size={args.batch_size}, device={device})")
    batch_results = []  # list of dict: {class_id, batch_idx, num_images, vendi_score}
    t_start = time.time()
    for bi, batch in enumerate(all_batches):
        cid = batch["class_id"]
        b_idx = batch["batch_idx"]
        files = batch["files"]

        # 加载图片
        imgs = []
        for p in files:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"[WARN] failed to open {p}: {e}")
        if len(imgs) < batch_img_count:
            print(f"[WARN] batch (class={cid}, idx={b_idx}) only loaded {len(imgs)}/{batch_img_count}, skip")
            continue

        # 提取嵌入
        emb = extract_embeddings(imgs, model, processor, args.batch_size, device)
        vs = compute_vendi_score(emb)
        batch_results.append({
            "class_id": cid,
            "batch_idx": b_idx,
            "num_images": len(imgs),
            "vendi_score": vs,
        })

        if (bi + 1) % 10 == 0 or (bi + 1) == total_batches:
            elapsed = time.time() - t_start
            eta = elapsed / (bi + 1) * (total_batches - bi - 1)
            print(f"    progress: {bi + 1}/{total_batches} "
                  f"({(bi + 1) / total_batches * 100:.1f}%), "
                  f"elapsed={elapsed:.1f}s, eta={eta:.1f}s")

    print(f"--> all batches done in {time.time() - t_start:.1f}s")

    # ---- 1. per_batch_vendi_scores.csv ----
    per_batch_csv = output_dir / "per_batch_vendi_scores.csv"
    print(f"--> writing per-batch CSV to {per_batch_csv}")
    with open(per_batch_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_id", "batch_idx", "num_images", "vendi_score"])
        writer.writeheader()
        writer.writerows(batch_results)

    # ---- 2. 按类聚合 ----
    class_to_batches = defaultdict(list)
    for r in batch_results:
        class_to_batches[r["class_id"]].append(r["vendi_score"])

    per_class_rows = []
    for cid in sorted(class_to_batches.keys()):
        vs_list = class_to_batches[cid]
        vs_arr = np.array(vs_list)
        n = len(vs_arr)
        mean_vs, std_vs, se_vs, ci_low_vs, ci_high_vs = t_ci95(vs_arr)
        info = class_info.get(cid, {})
        per_class_rows.append({
            "class_id": cid,
            "total_images": info.get("total_images", n * batch_img_count),
            "num_batches": n,
            "mean_vendi_score": mean_vs,
            "std_vendi_score": std_vs,
            "se_vendi_score": se_vs,
            "ci95_low": ci_low_vs,
            "ci95_high": ci_high_vs,
            "min_vendi_score": float(vs_arr.min()) if n > 0 else float("nan"),
            "max_vendi_score": float(vs_arr.max()) if n > 0 else float("nan"),
        })

    # ---- 3. per_class_summary.csv ----
    per_class_csv = output_dir / "per_class_summary.csv"
    print(f"--> writing per-class summary CSV to {per_class_csv}")
    with open(per_class_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class_id", "total_images", "num_batches",
            "mean_vendi_score", "std_vendi_score", "se_vendi_score",
            "ci95_low", "ci95_high",
            "min_vendi_score", "max_vendi_score"])
        writer.writeheader()
        writer.writerows(per_class_rows)

        # 末尾追加跨类统计
        all_means = np.array([r["mean_vendi_score"] for r in per_class_rows])
        if len(all_means) > 0:
            grand_mean, grand_std, grand_se, grand_ci_low, grand_ci_high = t_ci95(all_means)
            writer.writerow({
                "class_id": "MEAN_OF_CLASSES",
                "total_images": "",
                "num_batches": "",
                "mean_vendi_score": grand_mean,
                "std_vendi_score": grand_std,
                "se_vendi_score": grand_se,
                "ci95_low": grand_ci_low,
                "ci95_high": grand_ci_high,
                "min_vendi_score": float(all_means.min()),
                "max_vendi_score": float(all_means.max()),
            })

    # ---- 4. full_report.csv (宽表: 每类一行, 各批次 VS 展开为列) ----
    max_batches = max(len(v) for v in class_to_batches.values()) if class_to_batches else 0
    full_report_csv = output_dir / "full_report.csv"
    print(f"--> writing full report CSV to {full_report_csv}")
    with open(full_report_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = (["class_id", "total_images", "num_batches"]
                      + [f"batch_{i}_vs" for i in range(max_batches)]
                      + ["mean_vendi_score", "std_vendi_score",
                         "ci95_low", "ci95_high"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_class_rows:
            cid = r["class_id"]
            vs_list = class_to_batches[cid]
            row = {
                "class_id": cid,
                "total_images": r["total_images"],
                "num_batches": r["num_batches"],
                "mean_vendi_score": r["mean_vendi_score"],
                "std_vendi_score": r["std_vendi_score"],
                "ci95_low": r["ci95_low"],
                "ci95_high": r["ci95_high"],
            }
            for i in range(max_batches):
                row[f"batch_{i}_vs"] = vs_list[i] if i < len(vs_list) else ""
            writer.writerow(row)

    # ---- 5. 可视化: 每类 mean VS 排序条形图 ----
    if per_class_rows:
        sorted_rows = sorted(per_class_rows, key=lambda x: x["mean_vendi_score"])
        plt.figure(figsize=(15, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_rows)))
        means = [r["mean_vendi_score"] for r in sorted_rows]
        ci_lows = [r["ci95_low"] for r in sorted_rows]
        ci_highs = [r["ci95_high"] for r in sorted_rows]
        x_pos = range(len(sorted_rows))
        plt.bar(x_pos, means, color=colors, edgecolor="black", linewidth=0.3)
        # 误差线表示 95% CI
        err_lower = [m - lo for m, lo in zip(means, ci_lows)]
        err_upper = [hi - m for m, hi in zip(means, ci_highs)]
        plt.errorbar(x_pos, means, yerr=[err_lower, err_upper],
                     fmt="none", ecolor="black", elinewidth=0.5, capsize=2)
        grand_mean = np.mean(means)
        plt.axhline(grand_mean, color="red", linestyle="--", linewidth=2,
                    label=f"Grand Mean={grand_mean:.3f}")
        plt.xlabel("Class Index (sorted by mean VS ascending)")
        plt.ylabel("Mean Vendi Score (per class, batch-averaged)")
        plt.title(f"ImageNet Train Subset - Per-class Mean VS "
                  f"(model={args.model}, {batch_img_count} imgs/batch)")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        rank_path = output_dir / "per_class_mean_vendi_score_ranked.png"
        plt.savefig(rank_path, dpi=150)
        plt.close()
        print(f"    -> {rank_path}")

    # ---- 6. 总结 ----
    all_batch_vs = np.array([r["vendi_score"] for r in batch_results])
    all_class_means = np.array([r["mean_vendi_score"] for r in per_class_rows])
    print("\n" + "=" * 60)
    print("=== Summary (imagenet-train-subset) ===")
    print("=" * 60)
    print(f"  Model:              {DINOV2_MODELS[args.model][0]} (embed_dim={embed_dim})")
    print(f"  Device:             {device}")
    print(f"  Batch img count:    {batch_img_count}")
    print(f"  Total classes:      {len(class_info)}")
    print(f"  Total batches:      {total_batches}")
    print(f"  Total images used:  {total_imgs_used} (ignored tail: {total_ignored})")
    print(f"  Per-batch VS:       mean={all_batch_vs.mean():.4f}, "
          f"std={all_batch_vs.std():.4f}, "
          f"min={all_batch_vs.min():.4f}, max={all_batch_vs.max():.4f}")
    if len(all_class_means) > 0:
        print(f"  Per-class mean VS:  mean={all_class_means.mean():.4f}, "
              f"std={all_class_means.std():.4f}, "
              f"min={all_class_means.min():.4f}, max={all_class_means.max():.4f}")
    print(f"\n  Results saved to: {output_dir}")
    print(f"    - per_batch_vendi_scores.csv")
    print(f"    - per_class_summary.csv")
    print(f"    - full_report.csv")
    print(f"    - per_class_mean_vendi_score_ranked.png")
    print("=" * 60)


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 + Vendi Score evaluation (supports flat / imagenet-val / imagenet-train-subset modes)")
    parser.add_argument("--mode", default="flat",
                        choices=["flat", "imagenet-val", "imagenet-train-subset"],
                        help="评估模式: flat=单层目录+class_ids.txt; "
                             "imagenet-val=ImageNet-1K 验证集; "
                             "imagenet-train-subset=ImageNet-1K 训练集子集按 50 张/批评估 (default: flat)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"[flat 模式] 图片目录 (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--imagenet-val-dir", default=DEFAULT_IMAGENET_VAL_DIR,
                        help=f"[imagenet-val 模式] ImageNet_val 根目录 (default: {DEFAULT_IMAGENET_VAL_DIR})")
    parser.add_argument("--imagenet-train-subset-dir", default=DEFAULT_IMAGENET_TRAIN_SUBSET_DIR,
                        help=f"[imagenet-train-subset 模式] 训练集子集根目录 (default: {DEFAULT_IMAGENET_TRAIN_SUBSET_DIR})")
    parser.add_argument("--batch-img-count", type=int, default=DEFAULT_BATCH_IMG_COUNT,
                        help=f"[imagenet-train-subset 模式] 每个批次的图片数, 不足一批忽略 (default: {DEFAULT_BATCH_IMG_COUNT})")
    parser.add_argument("--class-def-json", default=DEFAULT_CLASS_DEF_JSON,
                        help=f"[imagenet-val 模式] ImageNet_1K_class_definition.json 路径 (default: {DEFAULT_CLASS_DEF_JSON})")
    parser.add_argument("--target-class-ids", default=None,
                        help="[imagenet-val 模式] 逗号分隔的 ImageNet-1K class id 列表 (default: 内置 100 类)")
    parser.add_argument("--output-dir", default=None,
                        help=f"输出目录 (default: 模式相关, flat={DEFAULT_OUTPUT_DIR}, imagenet-val=<repo>/vendi_score_results_imagenet_val)")
    parser.add_argument("--model", default="base", choices=list(DINOV2_MODELS),
                        help="DINOv2 model size: small/base/large (default: base)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"batch size for DINOv2 forward (default: {BATCH_SIZE})")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--model-cache-dir", default=None,
                        help=f"directory to cache downloaded models (default: {DEFAULT_MODEL_CACHE_DIR})")
    parser.add_argument("--baseline-csv", default=None,
                        help="[flat 模式] ImageNet 验证集 per_class_vendi_scores.csv 路径, "
                             "用于计算 vendi_ratio = VS / ImageNet_val_VS。仅在 flat 模式生效。")
    parser.add_argument("--baseline-train-csv", default=None,
                        help="[flat 模式] ImageNet 训练集子集 per_class_summary.csv 路径, "
                             "用于计算 vendi_ratio_train = VS / ImageNet_train_mean_VS。"
                             "仅在 flat 模式生效, 可与 --baseline-csv 同时使用。")
    args = parser.parse_args()

    # 输出目录默认值
    if args.output_dir is None:
        if args.mode == "flat":
            args.output_dir = DEFAULT_OUTPUT_DIR
        elif args.mode == "imagenet-val":
            args.output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"vendi_score_results_imagenet_val_{args.model}")
        elif args.mode == "imagenet-train-subset":
            args.output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"vendi_score_results_imagenet_train_subset_{args.model}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备选择
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 防御性检查: 若用户指定 cuda 但当前 PyTorch 不支持 CUDA, 自动回退到 cpu
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda specified but CUDA is not available "
              "(torch not compiled with CUDA or no GPU detected).")
        print("       Falling back to CPU. To use GPU, install CUDA-enabled PyTorch:")
        print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        device = "cpu"

    print(f"--> device: {device}")
    if device == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ---- imagenet-train-subset 模式: 独立处理流程 ----
    if args.mode == "imagenet-train-subset":
        run_imagenet_train_subset_mode(args, output_dir, device)
        return

    # ---- 加载图片 + 按 class_id 分组 ----
    # per_class_files: {class_id: [Path, ...]}  (flat 模式下 Path 是 str filename)
    # flat_files:      所有图片路径的扁平列表 (与 flat_class_ids 一一对应)
    # flat_class_ids:  与 flat_files 对应的 class_id 列表
    per_class_files = {}
    flat_files = []
    flat_class_ids = []
    class_meta = {}  # {class_id: {"class_name": str}}  可选元信息

    if args.mode == "flat":
        data_dir = Path(args.data_dir)
        class_ids_file = data_dir / "class_ids.txt"
        if not class_ids_file.exists():
            print(f"[ERROR] class_ids.txt not found at {class_ids_file}")
            sys.exit(1)
        print(f"--> [flat] loading class ids from {class_ids_file}")
        file_to_class = load_class_ids(class_ids_file)
        print(f"    total images: {len(file_to_class)}")

        groups = defaultdict(list)
        for fname, cls in file_to_class.items():
            groups[cls].append(fname)
        print(f"    total classes: {len(groups)}")
        if groups:
            print(f"    images per class: {len(next(iter(groups.values())))}")

        # 扁平列表 (按 class_id 升序, 同类内按文件名升序)
        for cls in sorted(groups.keys()):
            fnames = sorted(groups[cls])
            per_class_files[cls] = [data_dir / f for f in fnames]
            flat_files.extend(per_class_files[cls])
            flat_class_ids.extend([cls] * len(fnames))
            class_meta[cls] = {"class_name": ""}

    elif args.mode == "imagenet-val":
        # 解析 target class ids
        if args.target_class_ids:
            try:
                target_ids = [int(x.strip()) for x in args.target_class_ids.split(",") if x.strip()]
            except ValueError as e:
                print(f"[ERROR] invalid --target-class-ids: {e}")
                sys.exit(1)
        else:
            target_ids = DEFAULT_TARGET_CLASS_IDS

        print(f"--> [imagenet-val] target class count: {len(target_ids)}")
        print(f"--> [imagenet-val] loading class definition from {args.class_def_json}")
        class_def = load_imagenet_class_def(args.class_def_json, target_ids)
        print(f"    matched classes: {len(class_def)} / {len(target_ids)}")

        if not class_def:
            print("[ERROR] no valid classes matched, abort")
            sys.exit(1)

        print(f"--> [imagenet-val] collecting images from {args.imagenet_val_dir}")
        per_class_files, flat_files, flat_class_ids = collect_imagenet_val_images(
            args.imagenet_val_dir, class_def)
        for cid, info in class_def.items():
            class_meta[cid] = {"class_name": info["class_name"]}

        total_imgs = sum(len(v) for v in per_class_files.values())
        print(f"    total images: {total_imgs}")
        print(f"    total classes with images: {len(per_class_files)}")
        if per_class_files:
            first_cls = next(iter(per_class_files))
            print(f"    example: class_id={first_cls} "
                  f"(name={class_meta[first_cls]['class_name']}), "
                  f"n_images={len(per_class_files[first_cls])}")

    total_images = len(flat_files)
    if total_images == 0:
        print("[ERROR] no images collected, abort")
        sys.exit(1)

    # ---- 加载所有图片 ----
    print(f"--> loading {total_images} images...")
    t_load = time.time()
    all_images = []
    missing = []
    for i, p in enumerate(flat_files):
        try:
            all_images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"[WARN] failed to open {p}: {e}")
            missing.append((i, str(p)))
    if missing:
        print(f"[WARN] {len(missing)} images failed to load")
    print(f"    loaded {len(all_images)} images in {time.time() - t_load:.1f}s")

    # 构建每个类内图片在 all_images 中的索引
    # 注意: 由于可能有缺失, 需要重建索引
    valid_idx = [i for i in range(len(flat_files)) if i < len(all_images)]
    # 重新按 valid 索引构建 groups
    groups = defaultdict(list)  # class_id -> [valid_index_in_all_images]
    for local_i, cls in enumerate(flat_class_ids):
        if local_i < len(all_images):
            groups[cls].append(local_i)

    # 确定模型缓存目录
    cache_dir = args.model_cache_dir if args.model_cache_dir else DEFAULT_MODEL_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # 4. 加载 DINOv2 模型
    model, processor, embed_dim = load_dinov2(args.model, device, cache_dir=cache_dir)

    # 5. 提取嵌入
    print(f"--> extracting DINOv2 embeddings "
          f"(batch_size={args.batch_size}, device={device})")
    all_embeddings = extract_embeddings(
        all_images, model, processor, args.batch_size, device)
    print(f"    embeddings shape: {all_embeddings.shape}")

    # 6. 计算每类 Vendi Score
    print(f"--> computing per-class Vendi Scores")
    per_class_scores = []
    t0 = time.time()
    for cls in sorted(groups.keys()):
        idxs = groups[cls]
        X = all_embeddings[idxs]
        vs = compute_vendi_score(X)
        per_class_scores.append({
            "class_id": cls,
            "class_name": class_meta.get(cls, {}).get("class_name", ""),
            "num_images": len(idxs),
            "vendi_score": vs,
        })
    print(f"    done in {time.time() - t0:.2f}s")

    # 6.5 [可选] 加载 baseline CSV 并计算 vendi_ratio (仅 flat 模式生效)
    # vendi_ratio = current VS / ImageNet val VS (同 class_id)
    baseline_vs = None
    use_baseline = (args.mode == "flat" and args.baseline_csv)
    if use_baseline:
        print(f"--> loading baseline vendi scores from {args.baseline_csv}")
        try:
            baseline_vs = load_baseline_vendi_scores(args.baseline_csv)
            print(f"    baseline classes loaded: {len(baseline_vs)}")
        except Exception as e:
            print(f"[WARN] failed to load baseline CSV: {e}")
            print(f"       will skip vendi_ratio computation")
            use_baseline = False

    if use_baseline and baseline_vs:
        matched = 0
        unmatched = []
        for r in per_class_scores:
            cid = r["class_id"]
            if cid in baseline_vs and baseline_vs[cid] > 0:
                r["vendi_ratio"] = r["vendi_score"] / baseline_vs[cid]
                matched += 1
            else:
                r["vendi_ratio"] = None
                unmatched.append(cid)
        print(f"    vendi_ratio computed: {matched}/{len(per_class_scores)} classes matched baseline")
        if unmatched:
            print(f"    [WARN] {len(unmatched)} classes not in baseline (set vendi_ratio=None): "
                  f"{unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")

    # 6.6 [可选] 加载 ImageNet train baseline CSV 并计算 vendi_ratio_train (仅 flat 模式生效)
    # vendi_ratio_train = current VS / ImageNet train mean VS (同 class_id)
    baseline_train_vs = None
    use_baseline_train = (args.mode == "flat" and args.baseline_train_csv)
    if use_baseline_train:
        print(f"--> loading baseline (train) vendi scores from {args.baseline_train_csv}")
        try:
            baseline_train_vs = load_baseline_vendi_scores(args.baseline_train_csv)
            print(f"    baseline (train) classes loaded: {len(baseline_train_vs)}")
        except Exception as e:
            print(f"[WARN] failed to load baseline (train) CSV: {e}")
            print(f"       will skip vendi_ratio_train computation")
            use_baseline_train = False

    if use_baseline_train and baseline_train_vs:
        matched = 0
        unmatched = []
        for r in per_class_scores:
            cid = r["class_id"]
            if cid in baseline_train_vs and baseline_train_vs[cid] > 0:
                r["vendi_ratio_train"] = r["vendi_score"] / baseline_train_vs[cid]
                matched += 1
            else:
                r["vendi_ratio_train"] = None
                unmatched.append(cid)
        print(f"    vendi_ratio_train computed: {matched}/{len(per_class_scores)} classes matched baseline (train)")
        if unmatched:
            print(f"    [WARN] {len(unmatched)} classes not in baseline (train) "
                  f"(set vendi_ratio_train=None): "
                  f"{unmatched[:10]}{'...' if len(unmatched) > 10 else ''}")

    # 7. 计算整体 Vendi Score
    print(f"--> computing overall Vendi Score")
    overall_vs = compute_vendi_score(all_embeddings)
    print(f"    overall VS = {overall_vs:.4f}")

    # 8. 写入 CSV
    csv_path = output_dir / "per_class_vendi_scores.csv"
    print(f"--> writing CSV to {csv_path}")

    # 当使用 baseline 时, 字段列表增加 vendi_ratio 和/或 vendi_ratio_train
    fieldnames = ["class_id", "class_name", "num_images", "vendi_score"]
    if use_baseline:
        fieldnames.append("vendi_ratio")
    if use_baseline_train:
        fieldnames.append("vendi_ratio_train")

    # 用于在统计行中填充 vendi_ratio / vendi_ratio_train 占位 (空字符串)
    _baseline_empty = {}
    if use_baseline:
        _baseline_empty["vendi_ratio"] = ""
    if use_baseline_train:
        _baseline_empty["vendi_ratio_train"] = ""

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_class_scores:
            row = {k: r.get(k, "") for k in fieldnames}
            # 将 None 写成空字符串
            if row.get("vendi_ratio") is None:
                row["vendi_ratio"] = ""
            if row.get("vendi_ratio_train") is None:
                row["vendi_ratio_train"] = ""
            writer.writerow(row)
        # 末尾追加统计行
        scores = np.array([r["vendi_score"] for r in per_class_scores])
        # 预计算 vendi_score 的 95% CI (基于 t 分布, 类间 macro-average)
        vs_mean, vs_std, vs_se, vs_ci_low, vs_ci_high = t_ci95(scores)

        writer.writerow({
            "class_id": "OVERALL",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": overall_vs,
            **_baseline_empty,
        })
        writer.writerow({
            "class_id": "MEAN",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": vs_mean,
            **_baseline_empty,
        })
        writer.writerow({
            "class_id": "MEDIAN",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": float(np.median(scores)),
            **_baseline_empty,
        })
        writer.writerow({
            "class_id": "STD",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": vs_std,
            **_baseline_empty,
        })
        writer.writerow({
            "class_id": "SE",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": vs_se,
            **_baseline_empty,
        })
        writer.writerow({
            "class_id": "CI95_LOW",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": vs_ci_low,
            **_baseline_empty,
        })
        writer.writerow({
            "class_id": "CI95_HIGH",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": vs_ci_high,
            **_baseline_empty,
        })
        writer.writerow({
            "class_id": "MIN",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": float(scores.min()),
            **_baseline_empty,
        })
        writer.writerow({
            "class_id": "MAX",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": float(scores.max()),
            **_baseline_empty,
        })

        # 额外的 vendi_ratio 统计行 (类间 macro-average)
        if use_baseline:
            valid_ratios = [r["vendi_ratio"] for r in per_class_scores
                            if r.get("vendi_ratio") is not None]
            if valid_ratios:
                ratio_arr = np.array(valid_ratios)
                vr_mean, vr_std, vr_se, vr_ci_low, vr_ci_high = t_ci95(ratio_arr)
                writer.writerow({
                    "class_id": "MEAN_VENDI_RATIO",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio": vr_mean,
                })
                writer.writerow({
                    "class_id": "MEDIAN_VENDI_RATIO",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio": float(np.median(ratio_arr)),
                })
                writer.writerow({
                    "class_id": "STD_VENDI_RATIO",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio": vr_std,
                })
                writer.writerow({
                    "class_id": "SE_VENDI_RATIO",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio": vr_se,
                })
                writer.writerow({
                    "class_id": "CI95_LOW_VENDI_RATIO",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio": vr_ci_low,
                })
                writer.writerow({
                    "class_id": "CI95_HIGH_VENDI_RATIO",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio": vr_ci_high,
                })
                writer.writerow({
                    "class_id": "MIN_VENDI_RATIO",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio": float(ratio_arr.min()),
                })
                writer.writerow({
                    "class_id": "MAX_VENDI_RATIO",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio": float(ratio_arr.max()),
                })

        # 额外的 vendi_ratio_train 统计行 (类间 macro-average, 以 ImageNet train 为基准)
        if use_baseline_train:
            valid_ratios_train = [r["vendi_ratio_train"] for r in per_class_scores
                                  if r.get("vendi_ratio_train") is not None]
            if valid_ratios_train:
                ratio_train_arr = np.array(valid_ratios_train)
                vrt_mean, vrt_std, vrt_se, vrt_ci_low, vrt_ci_high = t_ci95(ratio_train_arr)
                writer.writerow({
                    "class_id": "MEAN_VENDI_RATIO_TRAIN",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio_train": vrt_mean,
                })
                writer.writerow({
                    "class_id": "MEDIAN_VENDI_RATIO_TRAIN",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio_train": float(np.median(ratio_train_arr)),
                })
                writer.writerow({
                    "class_id": "STD_VENDI_RATIO_TRAIN",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio_train": vrt_std,
                })
                writer.writerow({
                    "class_id": "SE_VENDI_RATIO_TRAIN",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio_train": vrt_se,
                })
                writer.writerow({
                    "class_id": "CI95_LOW_VENDI_RATIO_TRAIN",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio_train": vrt_ci_low,
                })
                writer.writerow({
                    "class_id": "CI95_HIGH_VENDI_RATIO_TRAIN",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio_train": vrt_ci_high,
                })
                writer.writerow({
                    "class_id": "MIN_VENDI_RATIO_TRAIN",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio_train": float(ratio_train_arr.min()),
                })
                writer.writerow({
                    "class_id": "MAX_VENDI_RATIO_TRAIN",
                    "class_name": "",
                    "num_images": len(all_images),
                    "vendi_score": "",
                    "vendi_ratio_train": float(ratio_train_arr.max()),
                })

    # 9. 可视化
    print(f"--> generating visualizations")
    visualize(per_class_scores, overall_vs, args.model, output_dir)
    if use_baseline:
        print(f"--> generating vendi_ratio visualizations (ImageNet val baseline)")
        visualize_ratio(per_class_scores, args.model, output_dir, col_name="vendi_ratio")
    if use_baseline_train:
        print(f"--> generating vendi_ratio_train visualizations (ImageNet train baseline)")
        visualize_ratio(per_class_scores, args.model, output_dir, col_name="vendi_ratio_train")

    # 10. 总结
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    print(f"  Mode:           {args.mode}")
    print(f"  Model:          {DINOV2_MODELS[args.model][0]} (embed_dim={embed_dim})")
    print(f"  Device:         {device}")
    print(f"  Total images:   {len(all_images)}")
    print(f"  Total classes:  {len(groups)}")
    if groups:
        imgs_per_class = [len(v) for v in groups.values()]
        print(f"  Images/class:   min={min(imgs_per_class)}, "
              f"max={max(imgs_per_class)}, "
              f"mean={sum(imgs_per_class)/len(imgs_per_class):.1f}")
    print(f"  Per-class VS:   mean={vs_mean:.4f}, "
          f"median={float(np.median(scores)):.4f}, "
          f"std={vs_std:.4f}, "
          f"95% CI=[{vs_ci_low:.4f}, {vs_ci_high:.4f}], "
          f"min={scores.min():.4f}, max={scores.max():.4f}")
    print(f"  Overall VS:     {overall_vs:.4f} (out of {len(all_images)} max)")
    if use_baseline:
        valid_ratios = [r["vendi_ratio"] for r in per_class_scores
                        if r.get("vendi_ratio") is not None]
        if valid_ratios:
            print(f"  Per-class VR_val:        mean={vr_mean:.4f}, "
                  f"median={float(np.median(ratio_arr)):.4f}, "
                  f"std={vr_std:.4f}, "
                  f"95% CI=[{vr_ci_low:.4f}, {vr_ci_high:.4f}], "
                  f"min={ratio_arr.min():.4f}, max={ratio_arr.max():.4f} "
                  f"(matched {len(valid_ratios)}/{len(per_class_scores)}, ImageNet val baseline)")
    if use_baseline_train:
        valid_ratios_train = [r["vendi_ratio_train"] for r in per_class_scores
                              if r.get("vendi_ratio_train") is not None]
        if valid_ratios_train:
            print(f"  Per-class VR_train:  mean={vrt_mean:.4f}, "
                  f"median={float(np.median(ratio_train_arr)):.4f}, "
                  f"std={vrt_std:.4f}, "
                  f"95% CI=[{vrt_ci_low:.4f}, {vrt_ci_high:.4f}], "
                  f"min={ratio_train_arr.min():.4f}, max={ratio_train_arr.max():.4f} "
                  f"(matched {len(valid_ratios_train)}/{len(per_class_scores)}, ImageNet train baseline)")
    print(f"\n  Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
