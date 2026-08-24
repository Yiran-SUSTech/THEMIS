"""Batch T2I sampling with SD3-medium / SD3.5-large from geneval2_data.jsonl.

Reads prompts line by line and saves one image per line as {line_index}.png,
matching the THEMIS t2i_harness naming convention. Existing images are skipped
(resume support). Per-image seed = --seed + line index.

Usage:
    python sd3_sample_geneval2.py --model sd3.5-large \
        --jsonl geneval2_data.jsonl --out-dir geneval2_sd35_large --seed 0
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

MODELS = {
    "sd3-medium": {
        "repo_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "steps": 28,
        "guidance": 7.0,
    },
    "sd3.5-large": {
        "repo_id": "stabilityai/stable-diffusion-3.5-large",
        "steps": 28,
        "guidance": 3.5,
    },
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--model-path", default=None,
                   help="local diffusers repo dir; default: repo id via HF endpoint/mirror")
    p.add_argument("--jsonl", default="geneval2_data.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num-steps", type=int, default=None, help="default: per-model card value")
    p.add_argument("--guidance", type=float, default=None, help="default: per-model card value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()

    spec = MODELS[args.model]
    steps = args.num_steps if args.num_steps is not None else spec["steps"]
    guidance = args.guidance if args.guidance is not None else spec["guidance"]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    src = args.model_path if args.model_path and Path(args.model_path).exists() else spec["repo_id"]
    print(f"loading {args.model} from {src} (steps={steps}, guidance={guidance}, dtype={args.dtype})")
    pipe = StableDiffusion3Pipeline.from_pretrained(src, torch_dtype=dtype)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(args.jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                records.append((i, json.loads(line)["prompt"]))
    records = records[args.start:]
    if args.limit > 0:
        records = records[: args.limit]
    print(f"{len(records)} prompts -> {out_dir}")

    device = torch.device(args.device)
    for pid, prompt in records:
        out_path = out_dir / f"{pid}.png"
        if out_path.exists():
            continue
        seed = args.seed + pid
        generator = torch.Generator(device=device).manual_seed(seed)
        img = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]
        img.save(out_path)
        print(f"[saved] {out_path.name} (seed={seed})  {prompt[:60]}")

    print("done.")


if __name__ == "__main__":
    main()
