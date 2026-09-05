"""Randomly sample a percentage of files from each source folder and copy them
into one destination folder, in source order.

Copy order = source order: files from source 2 are copied after source 1, so a
same-name file from a later source OVERWRITES the one from an earlier source
(last writer wins).

Usage:
  python t2i_harness/sample_merge_checklists.py \
      --sources "D:/THEMIS/t2i_harness/output_sd3_m_800_1/checklist_annotations" \
                "D:/THEMIS/t2i_harness/output_sd3_m_800_2/checklist_annotations" \
                "D:/THEMIS/t2i_harness/output_sd3_m_800_3/checklist_annotations" \
      --percents 30 15 43 \
      --dest "D:/THEMIS/t2i_harness/Human_anno_3_vs_Sys/checklist_annotations"

Options:
  --seed N      random seed for reproducible sampling (default 42)
  --ext .json   only sample files with this extension (default: all files)
"""

import sys
import shutil
import random
import argparse
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True,
                    help="source folder(s); files are copied in this order")
    ap.add_argument("--percents", nargs="+", required=True, type=float,
                    help="sampling percentage (0-100) per source, same count as --sources")
    ap.add_argument("--dest", required=True,
                    help="destination folder (created if missing)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducible sampling (default 42)")
    ap.add_argument("--ext", default=None,
                    help="only sample files with this extension, e.g. .json (default: all)")
    args = ap.parse_args()

    if len(args.sources) != len(args.percents):
        sys.exit(f"[ERROR] got {len(args.sources)} sources but "
                 f"{len(args.percents)} percents — counts must match")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total_copied = 0
    total_overwritten = 0
    for i, (src_str, pct) in enumerate(zip(args.sources, args.percents), 1):
        src = Path(src_str)
        if not src.is_dir():
            sys.exit(f"[ERROR] source folder not found: {src}")
        if not 0 <= pct <= 100:
            sys.exit(f"[ERROR] percent {pct:g} out of range [0, 100] for {src}")

        files = sorted(p for p in src.iterdir()
                       if p.is_file() and (args.ext is None
                                           or p.suffix.lower() == args.ext.lower()))
        n_select = int(round(len(files) * pct / 100.0))
        selected = rng.sample(files, n_select) if n_select else []

        overwritten = 0
        for f in selected:
            target = dest / f.name
            if target.exists():
                overwritten += 1
            shutil.copy2(f, target)

        total_copied += len(selected)
        total_overwritten += overwritten
        print(f"Source {i}: {src}")
        print(f"  {pct:g}% of {len(files)} files -> {len(selected)} copied, "
              f"{overwritten} overwrote same-name files already in destination")

    n_final = sum(1 for p in dest.iterdir() if p.is_file())
    print()
    print(f"Destination  : {dest}")
    print(f"Total copied : {total_copied} (of which {total_overwritten} overwrites)")
    print(f"Final files  : {n_final} distinct (seed = {args.seed})")


if __name__ == "__main__":
    main()
