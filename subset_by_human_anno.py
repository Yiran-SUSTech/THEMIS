"""
extract_images_by_annotations.py

从源图片文件夹中，根据标注 JSON 文件夹中所有 JSON 文件涉及的图片文件名并集，
无重复地复制到目标文件夹。

支持多组 (源文件夹, 标注文件夹, 目标文件夹) 配置，可扩展到其他数据集。
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def extract_image_names_from_json(json_path):
    image_names = set()
    image_class_map = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parse error in {json_path}: {e}")
        return image_names, image_class_map
    except Exception as e:
        logging.error(f"Failed to read {json_path}: {e}")
        return image_names, image_class_map

    if not isinstance(data, dict):
        logging.error(f"{json_path}: top-level is not a dict, got {type(data).__name__}")
        return image_names, image_class_map

    for key in data.keys():
        if key.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            image_names.add(key)
            entry = data[key]
            if isinstance(entry, dict):
                class_id = entry.get("class_id")
                if class_id is not None:
                    image_class_map[key] = class_id
        else:
            entry = data[key]
            if isinstance(entry, dict):
                img_name = entry.get("image_name")
                if img_name and isinstance(img_name, str):
                    image_names.add(img_name)
                    class_id = entry.get("class_id")
                    if class_id is not None:
                        image_class_map[img_name] = class_id

    return image_names, image_class_map


def process_task(src_dir, anno_dir, dst_dir):
    src_path = Path(src_dir)
    anno_path = Path(anno_dir)
    dst_path = Path(dst_dir)

    logging.info(f"=== Task ===")
    logging.info(f"  Source images : {src_path}")
    logging.info(f"  Annotations   : {anno_path}")
    logging.info(f"  Destination   : {dst_path}")

    if not src_path.exists():
        logging.error(f"Source image folder does not exist: {src_path}")
        return
    if not anno_path.exists():
        logging.error(f"Annotation folder does not exist: {anno_path}")
        return

    json_files = sorted(anno_path.glob("*.json"))
    if not json_files:
        logging.error(f"No JSON files found in: {anno_path}")
        return

    logging.info(f"Found {len(json_files)} JSON file(s) in annotation folder")

    all_image_names = set()
    all_image_class_map = {}
    per_json_stats = {}
    for jf in json_files:
        names, class_map = extract_image_names_from_json(jf)
        per_json_stats[jf.name] = len(names)
        all_image_names.update(names)
        all_image_class_map.update(class_map)
        logging.info(f"  {jf.name}: {len(names)} image(s), {len(class_map)} with class_id")

    logging.info(f"Union of all image names: {len(all_image_names)}")
    logging.info(f"Total images with class_id: {len(all_image_class_map)}")

    dst_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    skipped_dup = 0
    missing_list = []
    copied_with_class = {}

    for img_name in sorted(all_image_names):
        src_file = src_path / img_name
        dst_file = dst_path / img_name

        if not src_file.exists():
            missing += 1
            missing_list.append(img_name)
            logging.warning(f"  MISSING: {img_name} (not found in {src_path})")
            continue

        if dst_file.exists():
            skipped_dup += 1
            if img_name in all_image_class_map:
                copied_with_class[img_name] = all_image_class_map[img_name]
            continue

        try:
            shutil.copy2(src_file, dst_file)
            copied += 1
            if img_name in all_image_class_map:
                copied_with_class[img_name] = all_image_class_map[img_name]
        except Exception as e:
            logging.error(f"  COPY FAILED: {img_name}: {e}")

    if copied_with_class:
        class_ids_file = dst_path / "class_ids.txt"
        with open(class_ids_file, "w", encoding="utf-8") as f:
            for img_name in sorted(copied_with_class.keys()):
                class_id = copied_with_class[img_name]
                idx = img_name.rsplit(".", 1)[0]
                f.write(f"{idx} {class_id}\n")
        logging.info(f"  class_ids.txt saved: {class_ids_file} ({len(copied_with_class)} entries)")

    logging.info(f"=== Summary ===")
    logging.info(f"  JSON files processed : {len(json_files)}")
    for jf_name, cnt in per_json_stats.items():
        logging.info(f"    {jf_name}: {cnt} image(s)")
    logging.info(f"  Union image count  : {len(all_image_names)}")
    logging.info(f"  Copied             : {copied}")
    logging.info(f"  Skipped (duplicate): {skipped_dup}")
    logging.info(f"  Missing in source  : {missing}")
    if missing_list:
        missing_log = dst_path / "missing_images.txt"
        with open(missing_log, "w", encoding="utf-8") as f:
            for name in missing_list:
                f.write(name + "\n")
        logging.info(f"  Missing list saved : {missing_log}")
    logging.info(f"  Output folder      : {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract images referenced in annotation JSONs from source folder(s) to target folder(s)."
    )
    parser.add_argument(
        "--src", nargs="+", required=True,
        help="Source image folder(s). If multiple, pair with --anno and --dst by position.",
    )
    parser.add_argument(
        "--anno", nargs="+", required=True,
        help="Annotation JSON folder(s). Must match --src count.",
    )
    parser.add_argument(
        "--dst", nargs="+", required=True,
        help="Destination folder(s). Must match --src count.",
    )
    parser.add_argument(
        "--log", default="extract_images_log.txt",
        help="Log file path (default: extract_images_log.txt)",
    )
    args = parser.parse_args()

    if not (len(args.src) == len(args.anno) == len(args.dst)):
        parser.error("--src, --anno, --dst must have the same number of arguments")

    setup_logging(args.log)
    logging.info("Script started")

    for i in range(len(args.src)):
        process_task(args.src[i], args.anno[i], args.dst[i])

    logging.info("All tasks completed")


if __name__ == "__main__":
    main()