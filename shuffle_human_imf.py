"""
在 User_IMF_1/2/3_final_annotations.json 三个文件之间随机交换图片标注结果。

打乱方式:
    对指定比例(默认 50%)的图片, 在三个 User 文件之间随机置换该图的标注结果
    (随机选一个非恒等置换, 保证该图结果确实发生了交换);
    其余图片保持原样。每个 User 文件的 image_name key 集合不变,
    只是部分图片的 value 在三个文件间重新分配。

    例: fraction=0.5 表示约 50% 的图片其标注结果在三个 User 之间被打乱,
    另外 50% 的图片保持各自原始标注。

备份:
    覆盖原文件(默认行为)前, 自动把原文件备份到
    <input-dir>/_backup_shuffle_<时间戳>/ 目录。多次运行会产生多个备份。
    可用 --restore <备份目录> 还原。

用法:
    python shuffle_human_imf.py                              # 交换 50%, 覆盖原文件(先备份)
    python shuffle_human_imf.py --fraction 0.3 --seed 42     # 交换 30%, 可复现
    python shuffle_human_imf.py --output-dir shuffled_out    # 不覆盖原文件, 写到新目录
    python shuffle_human_imf.py --restore _backup_shuffle_20260706_153000
"""
import json
import os
import random
import shutil
import argparse
from datetime import datetime
from itertools import permutations

BASE_DIR = r"d:\THEMIS"
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "human_anno_IMF-XL2")
DEFAULT_USERS = ["User_IMF_1", "User_IMF_2", "User_IMF_3"]
FILENAME_TMPL = "{user}_final_annotations.json"


def load_user_files(input_dir, users):
    dicts, paths = {}, {}
    for u in users:
        p = os.path.join(input_dir, FILENAME_TMPL.format(user=u))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到文件: {p}")
        with open(p, "r", encoding="utf-8") as f:
            dicts[u] = json.load(f)
        paths[u] = p
    return dicts, paths


def backup_originals(paths, input_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(input_dir, f"_backup_shuffle_{ts}")
    os.makedirs(backup_dir, exist_ok=True)
    for u, p in paths.items():
        shutil.copy2(p, os.path.join(backup_dir, os.path.basename(p)))
    print(f"[BACKUP] 原文件已备份到: {backup_dir}")
    return backup_dir


def warn_if_prior_backup_exists(input_dir):
    """如果已存在 _backup_shuffle_* 目录, 提示原文件可能已被打乱过。"""
    if not os.path.isdir(input_dir):
        return
    prior = [d for d in os.listdir(input_dir)
             if d.startswith("_backup_shuffle_") and os.path.isdir(os.path.join(input_dir, d))]
    if prior:
        print(f"[WARN] 检测到已有备份目录: {prior}")
        print(f"[WARN] 当前原文件可能已被打乱过。如需从原始数据打乱, 请先用 --restore "
              f"恢复最早的备份再运行。")


def non_identity_permutations(n):
    return [p for p in permutations(range(n)) if any(p[i] != i for i in range(n))]


def shuffle_across_users(dicts, users, fraction, seed=None):
    if seed is not None:
        random.seed(seed)

    key_sets = [set(d.keys()) for d in (dicts[u] for u in users)]
    common_keys = sorted(set.intersection(*key_sets))
    if not common_keys:
        raise ValueError("文件之间没有公共 image_name, 无法交换")

    for i, u in enumerate(users):
        miss = key_sets[i] - set(common_keys)
        if miss:
            print(f"[WARN] {u} 有 {len(miss)} 个非公共 key 未参与打乱")

    n_total = len(common_keys)
    n_shuffle = max(0, min(n_total, int(round(n_total * fraction))))
    shuffle_keys = set(random.sample(common_keys, n_shuffle))

    new_dicts = {u: {} for u in users}
    for u in users:
        for k in dicts[u]:
            if k not in shuffle_keys:
                new_dicts[u][k] = dicts[u][k]

    perms = non_identity_permutations(len(users))
    n_cycle = 0      # 全员都换(3-轮换)
    n_trans = 0      # 一人不变(对换)
    for k in shuffle_keys:
        perm = random.choice(perms)
        if all(perm[i] != i for i in range(len(perm))):
            n_cycle += 1
        else:
            n_trans += 1
        # user i 得到 原 user perm[i] 的结果
        for i, u in enumerate(users):
            new_dicts[u][k] = dicts[users[perm[i]]][k]

    stats = {
        "n_total": n_total,
        "n_shuffle": n_shuffle,
        "n_kept": n_total - n_shuffle,
        "actual_fraction": (n_shuffle / n_total) if n_total else 0.0,
        "n_3cycle": n_cycle,
        "n_transposition": n_trans,
    }
    return new_dicts, stats


def save_dicts(new_dicts, users, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for u in users:
        p = os.path.join(out_dir, FILENAME_TMPL.format(user=u))
        with open(p, "w", encoding="utf-8") as f:
            json.dump(new_dicts[u], f, ensure_ascii=False, indent=2)
        print(f"[OK] 写入: {p}")


def restore_from_backup(backup_dir, input_dir, users):
    if not os.path.isdir(backup_dir):
        print(f"[ERROR] 备份目录不存在: {backup_dir}")
        return
    for u in users:
        src = os.path.join(backup_dir, FILENAME_TMPL.format(user=u))
        dst = os.path.join(input_dir, FILENAME_TMPL.format(user=u))
        if not os.path.isfile(src):
            print(f"[WARN] 备份中找不到: {src}")
            continue
        shutil.copy2(src, dst)
        print(f"[RESTORE] {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(
        description="在三个 User_IMF 文件之间随机交换图片标注结果(可控制比例)。"
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                        help=f"输入目录 (默认: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--users", nargs="+", default=DEFAULT_USERS,
                        help=f"参与的 User 列表 (默认: {DEFAULT_USERS})")
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="被打乱的图片比例 0~1 (默认 0.5 = 交换 50%%)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子, 用于复现")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录(不覆盖原文件); 省略则覆盖原文件(先自动备份)")
    parser.add_argument("--restore", default=None,
                        help="从该备份目录还原原文件, 还原后退出")
    args = parser.parse_args()

    if args.restore:
        restore_from_backup(args.restore, args.input_dir, args.users)
        return

    if not 0.0 <= args.fraction <= 1.0:
        print(f"[ERROR] fraction 必须在 [0,1], 当前: {args.fraction}")
        return

    try:
        dicts, paths = load_user_files(args.input_dir, args.users)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"[INFO] 已加载 {len(args.users)} 个文件: "
          + ", ".join(f"{u}={len(dicts[u])}" for u in args.users))

    try:
        new_dicts, stats = shuffle_across_users(dicts, args.users, args.fraction, args.seed)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    if args.output_dir:
        out_dir = args.output_dir
        print(f"[INFO] 非破坏模式: 写入 {out_dir} (原文件不动)")
    else:
        warn_if_prior_backup_exists(args.input_dir)
        backup_originals(paths, args.input_dir)
        out_dir = args.input_dir

    save_dicts(new_dicts, args.users, out_dir)

    print(f"\n[统计] 公共图片总数: {stats['n_total']}")
    print(f"[统计] 打乱图片: {stats['n_shuffle']} ({stats['actual_fraction']*100:.1f}%)")
    print(f"[统计] 保持原样: {stats['n_kept']}")
    print(f"[统计] 置换类型: 全员换(3-轮换)={stats['n_3cycle']}, 一人不变(对换)={stats['n_transposition']}")
    if args.seed is not None:
        print(f"[INFO] seed={args.seed}, 用相同 seed 可复现本次打乱")


if __name__ == "__main__":
    main()
