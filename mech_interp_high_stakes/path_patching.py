#!/usr/bin/env python3
"""
Minimal path-patching demo using TransformerLens.

This script is **not** a full re-implementation of the Alibaba reference the
user supplied (that relies on custom utils / dataset files).  Instead, it
shows the core mechanics with HookedTransformer so you can iterate quickly
inside this repo.

USAGE (example):
    python path_patching.py \
        --model gpt2 \
        --prompts_file examples/path_patching_prompts.json \
        --out_dir results_path_patch

The `prompts_file` should be a JSON list.  Each element is an object with:
    {
        "xr": "corrupted prompt",
        "xc": "clean prompt",
        "target": "word expected in clean answer"
    }
You can adapt your existing dataset easily.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import plotly.offline as py_offline
import plotly.figure_factory as ff
from tqdm import tqdm

from transformer_lens import HookedTransformer, utils

# -------------------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------------------

def to_tokens(model: HookedTransformer, text: str) -> torch.Tensor:
    return model.to_tokens(text, prepend_bos=True)


def logit_diff(
    model: HookedTransformer,
    toks: torch.Tensor,  # (1, seq)
    target_id: int,
) -> float:
    """Return logit(target) - logit(mean_vocab) for final token."""
    with torch.no_grad():
        logits = model(toks)[0, -1]  # (d_vocab,)
    return (logits[target_id] - logits.mean()).item()


# -------------------------------------------------------------------------------------
# Path patching core
# -------------------------------------------------------------------------------------

def path_patch_single_batch(
    model: HookedTransformer,
    xr: str,
    xc: str,
    target: str,
    layers: List[int],
) -> np.ndarray:
    """Return matrix (layer, head) of %-change in logit diff when head is patched."""
    device = model.cfg.device
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    toks_xr = to_tokens(model, xr).to(device)
    toks_xc = to_tokens(model, xc).to(device)
    target_id = model.to_single_token(target)

    base = logit_diff(model, toks_xr, target_id)

    results = np.zeros((n_layers, n_heads))

    # record clean activations
    _, cache_c = model.run_with_cache(toks_xc, return_type="logits")

    for L in tqdm(layers, desc="path-patch", leave=False):
        for h in range(n_heads):
            patch = cache_c["z", L][:, :, h, :].clone()  # (1, seq, d_head)

            def hook_z(z, hook, layer=L, head=h):  # noqa: B023
                z[:, :, head, :] = patch
                return z

            logits = model.run_with_hooks(
                toks_xr,
                return_type="logits",
                fwd_hooks=[("z", L, hook_z)],
            )
            cur = (logits[0, -1, target_id] - logits[0, -1].mean()).item()
            results[L, h] = 100 * (cur - base) / (base + 1e-8)

    return results


# -------------------------------------------------------------------------------------
# Visualisation helpers
# -------------------------------------------------------------------------------------

def save_heat(results: np.ndarray, toks: List[str], out_path: Path, title: str):
    fig = ff.create_annotated_heatmap(results, colorscale="Blues", showscale=True)
    fig.update_layout(title=title, xaxis_title="Head", yaxis_title="Layer")
    py_offline.plot(fig, filename=str(out_path), auto_open=False)

    # also save png for quick glance
    plt.figure(figsize=(10, 6))
    plt.imshow(results, cmap="Blues")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.colorbar(label="% Δ logit diff")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close()


# -------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Path patching (simplified)")
    parser.add_argument("--model", default="gpt2", help="HF model name or local path")
    parser.add_argument("--prompts_file", required=True, help="JSON list of prompts")
    parser.add_argument("--out_dir", default="results_path_patch", help="Save dir")
    parser.add_argument("--layers", default="all", help="Comma list or 'all'")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = HookedTransformer.from_pretrained(args.model, device="cuda" if torch.cuda.is_available() else "cpu")

    if args.layers == "all":
        layer_idxs = list(range(model.cfg.n_layers))
    else:
        layer_idxs = [int(x) for x in args.layers.split(",")]

    prompts = json.load(open(args.prompts_file))

    # aggregate results
    agg = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for i, ex in enumerate(prompts):
        mat = path_patch_single_batch(
            model,
            ex["xr"],
            ex["xc"],
            ex["target"],
            layer_idxs,
        )
        agg += mat
        save_heat(mat, [], out_dir / f"patch_{i}.html", f"Example {i} % Δ logit diff")

    agg /= len(prompts)
    save_heat(agg, [], out_dir / "patch_avg.html", "Average % Δ logit diff")
    np.save(out_dir / "patch_avg.npy", agg)
    print("Saved path-patching outputs to", out_dir)


if __name__ == "__main__":
    main() 