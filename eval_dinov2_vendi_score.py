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

输出:
    <output_dir>/per_class_vendi_scores.csv    - 每类 VS 明细 (含统计行)
    <output_dir>/vendi_score_histogram.png     - VS 分布直方图
    <output_dir>/vendi_score_ranked.png        - 每类 VS 排序条形图
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
from transformers import AutoModel, AutoImageProcessor

from vendi_score import vendi


# ==================== 配置 ====================
DEFAULT_DATA_DIR = r"d:\THEMIS\DiT-XL-2-sample_class_num-100-per_class_img-5"
DEFAULT_OUTPUT_DIR = r"d:\THEMIS\vendi_score_results"

# ImageNet-1K 验证集默认路径
DEFAULT_IMAGENET_VAL_DIR = r"H:\imagenet\ImageNet_val"
DEFAULT_CLASS_DEF_JSON = r"D:\imagenet-ancestors-descendants\ImageNet_1K_class_definition.json"

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
        description="DINOv2 + Vendi Score evaluation (supports flat & imagenet-val modes)")
    parser.add_argument("--mode", default="flat", choices=["flat", "imagenet-val"],
                        help="评估模式: flat=单层目录+class_ids.txt; imagenet-val=ImageNet-1K 验证集 (default: flat)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"[flat 模式] 图片目录 (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--imagenet-val-dir", default=DEFAULT_IMAGENET_VAL_DIR,
                        help=f"[imagenet-val 模式] ImageNet_val 根目录 (default: {DEFAULT_IMAGENET_VAL_DIR})")
    parser.add_argument("--class-def-json", default=DEFAULT_CLASS_DEF_JSON,
                        help=f"[imagenet-val 模式] ImageNet_1K_class_definition.json 路径 (default: {DEFAULT_CLASS_DEF_JSON})")
    parser.add_argument("--target-class-ids", default=None,
                        help="[imagenet-val 模式] 逗号分隔的 ImageNet-1K class id 列表 (default: 内置 100 类)")
    parser.add_argument("--output-dir", default=None,
                        help=f"输出目录 (default: 模式相关, flat={DEFAULT_OUTPUT_DIR}, imagenet-val=<repo>/vendi_score_results_imagenet_val)")
    parser.add_argument("--model", default="base", choices=list(DINOV2_MODELS),
                        help="DINOv2 model size: small/base/large (default: base)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"batch size (default: {BATCH_SIZE})")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--model-cache-dir", default=None,
                        help=f"directory to cache downloaded models (default: {DEFAULT_MODEL_CACHE_DIR})")
    args = parser.parse_args()

    # 输出目录默认值
    if args.output_dir is None:
        if args.mode == "flat":
            args.output_dir = DEFAULT_OUTPUT_DIR
        else:
            args.output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"vendi_score_results_imagenet_val_{args.model}")
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

    # 7. 计算整体 Vendi Score
    print(f"--> computing overall Vendi Score")
    overall_vs = compute_vendi_score(all_embeddings)
    print(f"    overall VS = {overall_vs:.4f}")

    # 8. 写入 CSV
    csv_path = output_dir / "per_class_vendi_scores.csv"
    print(f"--> writing CSV to {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_id", "class_name", "num_images", "vendi_score"])
        writer.writeheader()
        writer.writerows(per_class_scores)
        # 末尾追加统计行
        scores = np.array([r["vendi_score"] for r in per_class_scores])
        writer.writerow({
            "class_id": "OVERALL",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": overall_vs,
        })
        writer.writerow({
            "class_id": "MEAN",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": float(scores.mean()),
        })
        writer.writerow({
            "class_id": "MEDIAN",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": float(np.median(scores)),
        })
        writer.writerow({
            "class_id": "STD",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": float(scores.std()),
        })
        writer.writerow({
            "class_id": "MIN",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": float(scores.min()),
        })
        writer.writerow({
            "class_id": "MAX",
            "class_name": "",
            "num_images": len(all_images),
            "vendi_score": float(scores.max()),
        })

    # 9. 可视化
    print(f"--> generating visualizations")
    visualize(per_class_scores, overall_vs, args.model, output_dir)

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
    print(f"  Per-class VS:   mean={scores.mean():.4f}, "
          f"median={float(np.median(scores)):.4f}, "
          f"std={scores.std():.4f}, "
          f"min={scores.min():.4f}, max={scores.max():.4f}")
    print(f"  Overall VS:     {overall_vs:.4f} (out of {len(all_images)} max)")
    print(f"\n  Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
