"""
在 output_IMF_ref_nocap_1/2/3 三个文件夹的 final_reports 之间随机交换
每张图的 JSON 报告内容(每张图一个 JSON 文件)。

打乱方式:
    对指定比例(默认 50%)的图片, 在三个文件夹之间随机置换该图的 JSON 文件内容
    (随机非恒等置换, 保证确实发生交换); 其余图片保持原样。
    文件名不变, 只是部分文件的内容在三个文件夹间重新分配。
    采用原始文本读写, 完全保留 JSON 原始格式。

备份:
    覆盖原文件(默认)前, 自动把三个文件夹中 <subfolder> 下匹配 <prefix> 的
    JSON 文件备份到 <input-root>/_backup_shuffle_sys_<时间戳>/ 目录,
    保留原目录结构。多次运行产生多个备份。可用 --restore <备份目录> 还原。

用法:
    python shuffle_sys_imf.py                                   # 交换 50%, 覆盖原文件(先备份)
    python shuffle_sys_imf.py --fraction 0.3 --seed 42          # 交换 30%, 可复现
    python shuffle_sys_imf.py --output-root shuffled_sys_out    # 不覆盖, 写到新目录
    python shuffle_sys_imf.py --restore _backup_shuffle_sys_20260706_153000
    # 也可打乱 checklist_annotations 子目录:
    python shuffle_sys_imf.py --subfolder checklist_annotations --prefix checklist_
"""
import os
import random
import shutil
import argparse
from datetime import datetime
from itertools import permutations

BASE_DIR = r"d:\THEMIS"
DEFAULT_INPUT_ROOT = os.path.join(BASE_DIR, "c2i_faster")
DEFAULT_FOLDERS = ["output_IMF_ref_nocap_1", "output_IMF_ref_nocap_2", "output_IMF_ref_nocap_3"]
DEFAULT_SUBFOLDER = "final_reports"
DEFAULT_PREFIX = "final_evaluation_report_"


def list_image_files(folder_path, prefix):
    if not os.path.isdir(folder_path):
        return []
    return sorted(f for f in os.listdir(folder_path)
                  if f.startswith(prefix) and f.endswith(".json"))


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def backup_folders(input_root, folders, subfolder, prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(input_root, f"_backup_shuffle_sys_{ts}")
    n = 0
    for folder in folders:
        src_subdir = os.path.join(input_root, folder, subfolder)
        dst_subdir = os.path.join(backup_dir, folder, subfolder)
        if not os.path.isdir(src_subdir):
            continue
        os.makedirs(dst_subdir, exist_ok=True)
        for fname in list_image_files(src_subdir, prefix):
            shutil.copy2(os.path.join(src_subdir, fname), os.path.join(dst_subdir, fname))
            n += 1
    print(f"[BACKUP] 已备份 {n} 个文件到: {backup_dir}")
    return backup_dir


def warn_if_prior_backup_exists(input_root):
    if not os.path.isdir(input_root):
        return
    prior = [d for d in os.listdir(input_root)
             if d.startswith("_backup_shuffle_sys_") and os.path.isdir(os.path.join(input_root, d))]
    if prior:
        print(f"[WARN] 检测到已有系统备份目录: {prior}")
        print(f"[WARN] 当前文件可能已被打乱过。如需从原始数据打乱, 请先用 --restore "
              f"恢复最早备份再运行。")


def non_identity_permutations(n):
    return [p for p in permutations(range(n)) if any(p[i] != i for i in range(n))]


def get_common_files(input_root, folders, subfolder, prefix):
    subdirs = [os.path.join(input_root, f, subfolder) for f in folders]
    file_sets = [set(list_image_files(sd, prefix)) for sd in subdirs]
    common = sorted(set.intersection(*file_sets)) if file_sets else []
    for i, folder in enumerate(folders):
        miss = file_sets[i] - set(common)
        if miss:
            print(f"[WARN] {folder}/{subfolder} 有 {len(miss)} 个非公共文件未参与打乱")
    return subdirs, common


def shuffle_inplace(input_root, folders, subfolder, prefix, fraction, seed=None):
    if seed is not None:
        random.seed(seed)
    subdirs, common = get_common_files(input_root, folders, subfolder, prefix)
    if not common:
        raise ValueError("三个文件夹没有公共图片文件, 无法交换")

    n_total = len(common)
    n_shuffle = max(0, min(n_total, int(round(n_total * fraction))))
    shuffle_files = set(random.sample(common, n_shuffle))

    perms = non_identity_permutations(len(folders))
    n_cycle = 0
    n_trans = 0

    for fname in shuffle_files:
        contents = [read_text(os.path.join(subdirs[i], fname)) for i in range(len(folders))]
        perm = random.choice(perms)
        if all(perm[i] != i for i in range(len(perm))):
            n_cycle += 1
        else:
            n_trans += 1
        # 先读后写, contents 已在内存中, 覆盖原文件安全
        for i in range(len(folders)):
            write_text(os.path.join(subdirs[i], fname), contents[perm[i]])

    return {
        "n_total": n_total,
        "n_shuffle": n_shuffle,
        "n_kept": n_total - n_shuffle,
        "actual_fraction": (n_shuffle / n_total) if n_total else 0.0,
        "n_3cycle": n_cycle,
        "n_transposition": n_trans,
    }


def shuffle_to_output_root(input_root, output_root, folders, subfolder, prefix, fraction, seed=None):
    """非破坏模式: 把所有公共文件(打乱+未打乱)写到 output_root, 原文件不动。
    输出只包含 <folder>/<subfolder>/<prefix>*.json, 不含其它子目录。"""
    if seed is not None:
        random.seed(seed)
    subdirs, common = get_common_files(input_root, folders, subfolder, prefix)
    if not common:
        raise ValueError("三个文件夹没有公共图片文件, 无法交换")
    out_subdirs = [os.path.join(output_root, f, subfolder) for f in folders]

    n_total = len(common)
    n_shuffle = max(0, min(n_total, int(round(n_total * fraction))))
    shuffle_files = set(random.sample(common, n_shuffle))

    perms = non_identity_permutations(len(folders))
    n_cycle = 0
    n_trans = 0

    for fname in common:
        contents = [read_text(os.path.join(subdirs[i], fname)) for i in range(len(folders))]
        if fname in shuffle_files:
            perm = random.choice(perms)
            if all(perm[i] != i for i in range(len(perm))):
                n_cycle += 1
            else:
                n_trans += 1
        else:
            perm = tuple(range(len(folders)))
        for i in range(len(folders)):
            write_text(os.path.join(out_subdirs[i], fname), contents[perm[i]])

    return {
        "n_total": n_total,
        "n_shuffle": n_shuffle,
        "n_kept": n_total - n_shuffle,
        "actual_fraction": (n_shuffle / n_total) if n_total else 0.0,
        "n_3cycle": n_cycle,
        "n_transposition": n_trans,
    }


def restore_from_backup(backup_dir, input_root, folders, subfolder, prefix):
    if not os.path.isdir(backup_dir):
        print(f"[ERROR] 备份目录不存在: {backup_dir}")
        return
    n = 0
    for folder in folders:
        src_subdir = os.path.join(backup_dir, folder, subfolder)
        dst_subdir = os.path.join(input_root, folder, subfolder)
        if not os.path.isdir(src_subdir):
            print(f"[WARN] 备份中无此子目录: {src_subdir}")
            continue
        os.makedirs(dst_subdir, exist_ok=True)
        for fname in list_image_files(src_subdir, prefix):
            shutil.copy2(os.path.join(src_subdir, fname), os.path.join(dst_subdir, fname))
            n += 1
    print(f"[RESTORE] 已还原 {n} 个文件 (从 {backup_dir})")


def main():
    parser = argparse.ArgumentParser(
        description="在三个系统输出文件夹之间随机交换每张图的 JSON 报告内容(可控制比例)。"
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT,
                        help=f"输入根目录 (默认: {DEFAULT_INPUT_ROOT})")
    parser.add_argument("--folders", nargs="+", default=DEFAULT_FOLDERS,
                        help=f"三个文件夹名 (默认: {DEFAULT_FOLDERS})")
    parser.add_argument("--subfolder", default=DEFAULT_SUBFOLDER,
                        help=f"各文件夹下存放 JSON 的子目录 (默认: {DEFAULT_SUBFOLDER})")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help=f"文件名前缀 (默认: {DEFAULT_PREFIX})")
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="被打乱的图片比例 0~1 (默认 0.5)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子, 用于复现")
    parser.add_argument("--output-root", default=None,
                        help="输出根目录(不覆盖原文件); 省略则覆盖原文件(先自动备份)")
    parser.add_argument("--restore", default=None,
                        help="从该备份目录还原, 还原后退出")
    args = parser.parse_args()

    if args.restore:
        restore_from_backup(args.restore, args.input_root, args.folders,
                            args.subfolder, args.prefix)
        return

    if not 0.0 <= args.fraction <= 1.0:
        print(f"[ERROR] fraction 必须在 [0,1], 当前: {args.fraction}")
        return

    subdirs = [os.path.join(args.input_root, f, args.subfolder) for f in args.folders]
    for f, sd in zip(args.folders, subdirs):
        if not os.path.isdir(sd):
            print(f"[ERROR] 子目录不存在: {sd}")
            return

    file_counts = {f: len(list_image_files(sd, args.prefix))
                   for f, sd in zip(args.folders, subdirs)}
    print(f"[INFO] 各文件夹匹配文件数: "
          + ", ".join(f"{f}={c}" for f, c in file_counts.items()))

    try:
        if args.output_root:
            print(f"[INFO] 非破坏模式: 写入 {args.output_root} (原文件不动, 仅输出 {args.subfolder} 下的 JSON)")
            stats = shuffle_to_output_root(args.input_root, args.output_root, args.folders,
                                           args.subfolder, args.prefix, args.fraction, args.seed)
        else:
            warn_if_prior_backup_exists(args.input_root)
            backup_folders(args.input_root, args.folders, args.subfolder, args.prefix)
            stats = shuffle_inplace(args.input_root, args.folders, args.subfolder,
                                    args.prefix, args.fraction, args.seed)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    print(f"\n[统计] 公共图片总数: {stats['n_total']}")
    print(f"[统计] 打乱图片: {stats['n_shuffle']} ({stats['actual_fraction']*100:.1f}%)")
    print(f"[统计] 保持原样: {stats['n_kept']}")
    print(f"[统计] 置换类型: 全员换(3-轮换)={stats['n_3cycle']}, 一人不变(对换)={stats['n_transposition']}")
    if args.seed is not None:
        print(f"[INFO] seed={args.seed}, 用相同 seed 可复现本次打乱")


if __name__ == "__main__":
    main()
