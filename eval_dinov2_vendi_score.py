"""
DINOv2 + Vendi Score 多样性评估脚本
=====================================
对 DiT-XL-2-sample_class_num-100-per_class_img-5 文件夹下的 500 张图片
（100 类 × 5 张）提取 DINOv2 特征，计算每类与整体的 Vendi Score。

使用方式:
    python eval_dinov2_vendi_score.py
    python eval_dinov2_vendi_score.py --model base
    python eval_dinov2_vendi_score.py --model small --device cpu
    python eval_dinov2_vendi_score.py --data-dir <path-to-folder>

输出:
    vendi_score_results/per_class_vendi_scores.csv    - 每类 VS 明细
    vendi_score_results/vendi_score_histogram.png     - VS 分布直方图
    vendi_score_results/vendi_score_ranked.png        - 每类 VS 排序条形图
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict

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
from transformers import AutoModel, AutoImageProcessor

from vendi_score import vendi


# ==================== 配置 ====================
DEFAULT_DATA_DIR = r"d:\THEMIS\DiT-XL-2-sample_class_num-100-per_class_img-5"
DEFAULT_OUTPUT_DIR = r"d:\THEMIS\vendi_score_results"

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


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 + Vendi Score evaluation for DiT-XL-2 samples")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"image folder (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"output dir (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--model", default="base", choices=list(DINOV2_MODELS),
                        help="DINOv2 model size: small/base/large (default: base)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"batch size (default: {BATCH_SIZE})")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--model-cache-dir", default=None,
                        help=f"directory to cache downloaded models (default: {DEFAULT_MODEL_CACHE_DIR})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备选择
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> device: {device}")
    if device == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. 加载类别映射
    class_ids_file = data_dir / "class_ids.txt"
    if not class_ids_file.exists():
        print(f"[ERROR] class_ids.txt not found at {class_ids_file}")
        sys.exit(1)
    print(f"--> loading class ids from {class_ids_file}")
    class_ids = load_class_ids(class_ids_file)
    print(f"    total images: {len(class_ids)}")

    # 2. 按类别分组
    groups = defaultdict(list)
    for fname, cls in class_ids.items():
        groups[cls].append(fname)
    print(f"    total classes: {len(groups)}")
    print(f"    images per class: {len(next(iter(groups.values())))}")

    # 3. 加载所有图片（按文件名排序，避免重复 IO）
    print(f"--> loading all images from {data_dir}")
    sorted_files = sorted(class_ids.keys())
    file_to_idx = {f: i for i, f in enumerate(sorted_files)}
    all_images = []
    for f in sorted_files:
        img_path = data_dir / f
        if not img_path.exists():
            print(f"[ERROR] missing image: {img_path}")
            sys.exit(1)
        all_images.append(Image.open(img_path).convert("RGB"))

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
        fnames = groups[cls]
        idxs = [file_to_idx[f] for f in fnames]
        X = all_embeddings[idxs]
        vs = compute_vendi_score(X)
        per_class_scores.append({
            "class_id": cls,
            "num_images": len(fnames),
            "vendi_score": vs,
        })
    print(f"    done in {time.time() - t0:.2f}s")

    # 7. 计算整体 Vendi Score
    print(f"--> computing overall Vendi Score")
    overall_vs = compute_vendi_score(all_embeddings)
    print(f"    overall VS = {overall_vs:.4f}")

    # 8. 写入 CSV
    csv_path = output_dir / "per_class_vendi_scores.csv"
    print(f"--> writing CSV to {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_id", "num_images", "vendi_score"])
        writer.writeheader()
        writer.writerows(per_class_scores)
        # 末尾追加统计行
        scores = np.array([r["vendi_score"] for r in per_class_scores])
        writer.writerow({
            "class_id": "OVERALL",
            "num_images": len(all_images),
            "vendi_score": overall_vs,
        })
        writer.writerow({
            "class_id": "MEAN",
            "num_images": len(all_images),
            "vendi_score": float(scores.mean()),
        })
        writer.writerow({
            "class_id": "MEDIAN",
            "num_images": len(all_images),
            "vendi_score": float(np.median(scores)),
        })

    # 9. 可视化
    print(f"--> generating visualizations")
    visualize(per_class_scores, overall_vs, args.model, output_dir)

    # 10. 总结
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    print(f"  Model:          {DINOV2_MODELS[args.model][0]} (embed_dim={embed_dim})")
    print(f"  Device:         {device}")
    print(f"  Total images:   {len(all_images)}")
    print(f"  Total classes:  {len(groups)}")
    print(f"  Images/class:   {len(next(iter(groups.values())))}")
    print(f"  Per-class VS:   mean={scores.mean():.4f}, "
          f"median={float(np.median(scores)):.4f}, "
          f"min={scores.min():.4f}, max={scores.max():.4f}")
    print(f"  Overall VS:     {overall_vs:.4f} (out of {len(all_images)} max)")
    print(f"\n  Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
