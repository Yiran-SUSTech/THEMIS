#!/usr/bin/env python3
"""
THEMIS Meta-Evaluation: AUROC (GenEval2-style)

两级 AUROC:
  1) Atom-level (主, 对标 GenEval2 Soft-TIFA):
       label  = 人类对每个 diagnostic checkpoint 的 🟢/🔴 多数票 (1/0), ⚪N/A 弃权
       score  = 系统 N 次重复跑里该 checkpoint 的 present 投票比例 (软置信度)
       逐 (image, checkpoint) 配对 -> 单一 AUROC
  2) Image-level (辅, 对标只有图像分的基线 VQAScore/TIFA/Q-Align):
       label  = 人类均值二值化 (阈值扫描, 不硬挑)
       score  = 系统 0-5 分 (alignment / artifact)

不依赖 sklearn: AUROC 用 Mann-Whitney U 等价式实现 (含并列 0.5 计数)。
不重跑系统, 纯读已有 JSON。
"""
import os, sys, json, glob, argparse
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "small_scale_audit" / "output_results"

GREEN, RED, NA = "🟢 Checked", "🔴 Missing", "⚪ N/A"


# ---------- AUROC (tie-aware, Mann-Whitney) ----------
def auroc(scores, labels):
    """labels: 1/0. scores: float. 返回 (auc, n_pos, n_neg) 或 (None,..) 若单一类。"""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None, len(pos), len(neg)
    # rank-sum on combined, tie -> average rank
    combined = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_pos = sum(r for r, (_, l) in zip(ranks, combined) if l == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg), n_pos, n_neg


# ---------- 人类 atom 标签 (3标注员多数票) ----------
def load_human_atoms():
    """return {img_id: {checkpoint_text: label(1/0/None)}}, label=多数票, None=弃权/平票"""
    users = []
    for u in [1, 2, 3]:
        users.append(json.load(open(AUDIT / f"User_{u}_final_annotations.json", encoding="utf-8")))
    out = {}
    keys = users[0].keys()
    for key in keys:
        iid = key.replace(".png", "")
        ck_votes = defaultdict(list)  # checkpoint -> [status,...]
        for ud in users:
            fg = ud.get(key, {}).get("fine_grained_details", {})
            for cat, items in fg.items():
                for txt, status in items.items():
                    ck_votes[txt].append(status)
        ck_label = {}
        for txt, votes in ck_votes.items():
            g = votes.count(GREEN)
            r = votes.count(RED)
            if g == 0 and r == 0:
                ck_label[txt] = None       # 全 N/A
            elif g == r:
                ck_label[txt] = None       # 平票
            else:
                ck_label[txt] = 1 if g > r else 0
        out[iid] = ck_label
    return out


# ---------- 人类图像级均值 ----------
def load_human_image_scores():
    import csv
    out = {}
    with open(AUDIT / "aggregated_human_scores.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = row["image_name"].replace(".png", "")
            out[iid] = (float(row["alignment_mean"]), float(row["artifact_mean"]))
    return out


# ---------- 系统 atom 软分 (N次跑投票比例) ----------
def load_system_atoms(run_dirs):
    """run_dirs: list of approved_plans dirs. return {img_id: {checkpoint: present_ratio}}"""
    per_img = defaultdict(lambda: defaultdict(list))  # img -> ck -> [bool,...]
    for d in run_dirs:
        for fp in glob.glob(str(Path(d) / "*.json")):
            stem = Path(fp).stem
            # approved_plan_000002 -> 000002
            iid = "".join(c for c in stem if c.isdigit())[-6:]
            try:
                data = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            for cv in data.get("checkpoint_verdicts", []):
                txt = cv.get("checkpoint")
                if not cv.get("is_testable", True):
                    continue
                per_img[iid][txt].append(1 if cv.get("is_present") else 0)
    out = {}
    for iid, cks in per_img.items():
        out[iid] = {txt: sum(v) / len(v) for txt, v in cks.items() if v}
    return out


# ---------- 系统图像级分 (N次跑平均) ----------
def load_system_image_scores(report_dirs):
    per_img = defaultdict(lambda: ([], []))
    for d in report_dirs:
        for fp in glob.glob(str(Path(d) / "final_evaluation_report_*.json")):
            iid = Path(fp).stem.replace("final_evaluation_report_", "")
            try:
                data = json.load(open(fp, encoding="utf-8"))
            except Exception:
                continue
            a, b = data.get("alignment_score"), data.get("artifact_score")
            if a is not None:
                per_img[iid][0].append(a)
            if b is not None:
                per_img[iid][1].append(b)
    return {iid: (sum(al) / len(al) if al else None,
                  sum(ar) / len(ar) if ar else None)
            for iid, (al, ar) in per_img.items()}


def run_atom_auroc(run_dirs):
    human = load_human_atoms()
    system = load_system_atoms(run_dirs)
    scores, labels = [], []
    matched_imgs = 0
    for iid, ck_label in human.items():
        if iid not in system:
            continue
        matched_imgs += 1
        for txt, lab in ck_label.items():
            if lab is None:
                continue
            if txt not in system[iid]:
                continue
            scores.append(system[iid][txt])
            labels.append(lab)
    auc, npos, nneg = auroc(scores, labels)
    return {"auroc": auc, "n_pairs": len(scores), "n_pos": npos, "n_neg": nneg,
            "matched_images": matched_imgs}


def run_image_auroc(report_dirs, thresholds=(2.5, 3.0, 3.5)):
    human = load_human_image_scores()
    system = load_system_image_scores(report_dirs)
    res = {}
    for dim, name in [(0, "alignment"), (1, "artifact")]:
        common = [i for i in human if i in system
                  and human[i][dim] is not None and system[i][dim] is not None]
        sc = [system[i][dim] for i in common]
        hu = [human[i][dim] for i in common]
        per_thr = {}
        for t in thresholds:
            labs = [1 if h > t else 0 for h in hu]
            auc, npos, nneg = auroc(sc, labs)
            per_thr[t] = {"auroc": auc, "n_pos": npos, "n_neg": nneg}
        res[name] = {"n": len(common), "by_threshold": per_thr}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="output 目录名 (相对 c2i_faster/), 用于 N 次重复跑")
    ap.add_argument("--label", default="THEMIS")
    args = ap.parse_args()

    base = ROOT / "c2i_faster"
    run_dirs = [str(base / r / "approved_plans") for r in args.runs]
    report_dirs = [str(base / r / "final_reports") for r in args.runs]

    print(f"\n{'='*60}\n  {args.label}  ({len(args.runs)} runs: {', '.join(args.runs)})\n{'='*60}")

    atom = run_atom_auroc(run_dirs)
    print(f"\n[Atom-level AUROC] (GenEval2-style)")
    if atom["auroc"] is None:
        print(f"  无法计算 (单一类). pos={atom['n_pos']} neg={atom['n_neg']}")
    else:
        print(f"  AUROC = {atom['auroc']:.4f}")
    print(f"  匹配图={atom['matched_images']}  配对数={atom['n_pairs']}  (pos={atom['n_pos']}, neg={atom['n_neg']})")

    img = run_image_auroc(report_dirs)
    print(f"\n[Image-level AUROC] (阈值扫描)")
    for name, r in img.items():
        print(f"  {name} (n={r['n']}):")
        for t, v in r["by_threshold"].items():
            a = v["auroc"]
            astr = f"{a:.4f}" if a is not None else "N/A"
            print(f"    thr>{t}: AUROC={astr}  (pos={v['n_pos']}, neg={v['n_neg']})")
    print()
