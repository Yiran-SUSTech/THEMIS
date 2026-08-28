"""Batch FLUX.2 sampling from geneval2_data.jsonl.

Reads prompts line by line, saves one image per line as {line_index}.png,
matching the THEMIS t2i_harness naming convention. Existing images are
skipped (resume support). Per-image seed = --seed + line index.

Memory modes (choose one based on GPU VRAM):
  default         : everything on GPU (needs ~130 GB for dev)
  --cpu-offloading: text encoder on CPU, flow+AE on GPU (~70 GB for dev)
  --aggressive-offload: flow and AE alternate on GPU (~66 GB for dev)

Usage:
  PYTHONPATH=src python scripts/sample_geneval2.py \
    --jsonl geneval2_data.jsonl --out-dir geneval2_flux2_dev \
    --width 1024 --height 1024 --num-steps 50 --guidance 4.0 \
    --seed 0 --aggressive-offload
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from flux2.text_encoder import Flux2TextEncoder
from flux2.sampling import denoise, get_schedule
from flux2.util import load_flow_model, load_ae, load_ae_train

# ---------------------------------------------------------------------------
# Model card defaults
# ---------------------------------------------------------------------------
from flux2.util import FLUX2_MODEL_INFO


def _empty_cache():
    """Safe empty_cache that works on MetaX / CUDA compatible layers."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_to_cpu(model):
    """Move model to CPU and return it.  Pinned memory if possible."""
    model = model.to("cpu")
    _empty_cache()
    return model


def _move_to_gpu(model, device):
    model = model.to(device)
    _empty_cache()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="flux.2-dev",
                   help="model name key in FLUX2_MODEL_INFO")
    p.add_argument("--jsonl", default="geneval2_data.jsonl")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-offloading", action="store_true",
                   help="offload text encoder to CPU")
    p.add_argument("--aggressive-offload", action="store_true",
                   help="swap DiT<->AE so only one big model lives on GPU at a time "
                        "(required for 64 GB cards with FLUX.2-dev)")
    p.add_argument("--offload-dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="dtype for models on GPU")
    args = p.parse_args()

    offload_dtype = dict(bfloat16=torch.bfloat16,
                         float16=torch.float16,
                         float32=torch.float32)[args.offload_dtype]
    device = torch.device(args.device)
    aggressive = args.aggressive_offload
    cpu_offload_text = args.cpu_offloading or aggressive

    # ------------------------------------------------------------------
    # Load text encoder
    # ------------------------------------------------------------------
    print(f"loading text encoder (cpu_offload={cpu_offload_text})")
    text_enc = Flux2TextEncoder(max_seq_len=512, model_name=args.name)
    if not cpu_offload_text:
        text_enc.get_prompt_embds_dtype = offload_dtype
        text_enc = text_enc.to(device)
    # else: stays on CPU (default for Flux2TextEncoder in some builds)

    # ------------------------------------------------------------------
    # Load DiT (flow model)
    # ------------------------------------------------------------------
    print(f"loading flow model ({args.name}) -> {device}")
    model = load_flow_model(args.name, device=device)
    if not aggressive:
        # keep on GPU
        pass
    # aggressive: we will move it off GPU during AE decode

    # ------------------------------------------------------------------
    # Load AE
    # ------------------------------------------------------------------
    print("loading autoencoder -> " + ("CPU" if aggressive else str(device)))
    ae_device = "cpu" if aggressive else device
    ae = load_ae(args.name, device=ae_device)
    if aggressive:
        ae = ae.to("cpu")
        _empty_cache()

    # ------------------------------------------------------------------
    # Read prompts
    # ------------------------------------------------------------------
    records = []
    with open(args.jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                records.append((i, json.loads(line)["prompt"]))
    records = records[args.start:]
    if args.limit > 0:
        records = records[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(records)} prompts -> {out_dir}  (seed_base={args.seed})")

    info = FLUX2_MODEL_INFO[args.name]
    schedule = get_schedule(args.num_steps, info["shift"])

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------
    pbar = tqdm(records, desc="sampling")
    for pid, prompt in pbar:
        out_path = out_dir / f"{pid}.png"
        if out_path.exists():
            continue

        seed = args.seed + pid
        pbar.set_postfix_str(f"#{pid} seed={seed}")

        # ---- text encoding -----------------------------------------
        # text encoder always on CPU when cpu_offload_text is True
        with torch.no_grad():
            if cpu_offload_text:
                # encode on CPU, then move embeddings to GPU
                with torch.autocast("cpu", dtype=offload_dtype):
                    inp = text_enc(prompt, "", args.width, args.height,
                                   do_extend_prompt=False,
                                   do_nsfw_check=False,
                                   do_safety_check=False)
                inp = inp.to(device, dtype=offload_dtype)
            else:
                inp = text_enc(prompt, "", args.width, args.height,
                               do_extend_prompt=False,
                               do_nsfw_check=False,
                               do_safety_check=False)

        # ---- denoise (DiT on GPU) ----------------------------------
        # ensure model is on GPU (it should be, but in aggressive mode
        # it may have been moved to CPU by the previous image's decode step)
        if aggressive and next(model.parameters()).device.type != "cuda":
            print("  [swap] flow_model -> GPU", flush=True)
            model = model.to(device)
            _empty_cache()

        with torch.no_grad():
            x = denoise(model, info, inp, args.width, args.height,
                        args.num_steps, args.guidance, seed,
                        schedule=schedule)

        # ---- decode (AE on GPU) ------------------------------------
        if aggressive:
            # move DiT -> CPU, move AE -> GPU
            print("  [swap] flow_model -> CPU, ae -> GPU", flush=True)
            model = model.to("cpu")
            _empty_cache()
            ae = ae.to(device)
            _empty_cache()

        with torch.no_grad():
            x = ae.decode(x).float()

        if aggressive:
            # move AE back to CPU, move DiT back to GPU for next image
            print("  [swap] ae -> CPU, flow_model -> GPU", flush=True)
            ae = ae.to("cpu")
            _empty_cache()
            model = model.to(device)
            _empty_cache()

        # ---- save --------------------------------------------------
        # x: [1,C,H,W] float in [0,1]
        img = x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        import numpy as np
        from PIL import Image
        img = (img * 255).round().astype(np.uint8)
        Image.fromarray(img).save(out_path)

    print("done.")


if __name__ == "__main__":
    main()
