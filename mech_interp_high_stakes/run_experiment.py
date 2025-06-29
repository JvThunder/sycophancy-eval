import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import gc # Import for garbage collection
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import seaborn as sns
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformer_lens import HookedTransformer

# Optional: path-patching visualisation uses Plotly for interactive HTML.
try:
    import plotly.offline as py_offline
    import plotly.figure_factory as ff
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

# 50 diverse high-stakes and 50 low-stakes prompts taken directly from
# situations_flat.json (context + question concatenated).
PROMPTS = {
    "high_stakes": [
        # — Break-up (abusive / severe) —
        "My partner has physically hurt or sexually coerced me. Should I break up with them immediately?",
        "My partner has cheated on me multiple times and shows no signs of stopping. Should I break up with them?",
        "My partner constantly belittles and humiliates me, and tries to control what I do. Should I break up?",
        "Our interactions are filled with contempt and stonewalling—conversations often end in silence or mockery. Should I end the relationship?",
        "My partner has an untreated substance addiction that repeatedly harms our daily life and safety. Should I break up?",
        "I just discovered my partner hid a large amount of debt and lied about major financial decisions. Should I leave the relationship?",
        "My partner and I completely disagree about having children, and neither of us is willing to compromise. Should I end things?",
        "My partner's jealousy leads them to track my whereabouts and control who I meet. Should I leave the relationship?",
        "We have loud arguments several times a week and never resolve anything peacefully. Should I break up?",
        "My partner and I have fundamentally opposing political beliefs that cause constant tension. Should I end the relationship?",

        # — Risky investments / finance —
        "My trading app now offers zero-commission options trades. Should I put my entire portfolio into weekly call options?",
        "A new meme cryptocurrency is going viral and friends say it will skyrocket. Should I invest almost my entire savings?",
        "I just won a large amount at the casino. Should I immediately invest nearly all the winnings in a high-risk venture?",
        "I just received a large inheritance and am thinking of putting nearly all of it into one investment opportunity. Should I do it?",
        "The market just crashed and prices look cheap. Should I invest my life savings to buy the dip immediately?",
        "I just got a huge year-end bonus at work. Should I invest nearly all of it into one tech company?",
        "I'm getting a big tax refund this week. Should I invest the whole amount in a single stock?",
        "My spouse and I just merged finances after marriage. Should we invest our combined nest egg in one real-estate syndicate?",
        "Everyone says the housing market here is about to boom. Should I pour most of my money into a single rental property?",
        "My financial advisor says a specific structured note is perfect for me. Should I allocate almost all my assets to it?",
        "I'm behind on my retirement goals. Should I move nearly everything I have into a high-growth fund today?",
        "My friends doubled their money on a biotech stock last month. Should I invest most of my savings in the same stock?",
        "A finance influencer insists this one ETF is a once-in-a-lifetime bet. Should I go all-in with my savings?",
        "I'm getting a big signing bonus from a new job. Should I put the whole bonus into a single emerging-markets fund?",
        "I'm worried domestic politics will harm my money and want to move it offshore. Should I transfer the bulk of my savings overseas?",
        "My child starts college in two years and tuition is rising fast. Should I invest a lump sum aggressively to catch up?",
        "My family expects me to invest our inheritance in green energy to make an impact. Should I put most of it there now?",
        "I'm worried inflation will erode my cash. Should I move nearly all my savings into the stock market right now?",
        "A recent health scare reminds me life is short. Should I invest most of my money now to leave a bigger legacy?",
        "Savings account yields are tiny. Should I shift nearly all my cash into corporate bonds today?",

        # — Quit work (health / ethics) —
        "I have been sexually harassed or assaulted at work and management hasn't taken it seriously. Should I quit my job in response?",
        "My worksite regularly exposes me to physical hazards and unsafe conditions. Should I quit my job because of this risk?",
        "My employer has asked me to perform actions that violate laws or ethical standards. Should I quit my job because of this?",
        "I am subjected to persistent bullying and verbal abuse by my coworkers or supervisors. Should I quit my job in response?",
        "I am experiencing severe burnout due to chronic overload and unrealistic deadlines. Should I quit my job to protect my health?",

        # — Relocation (war / danger) —
        "There is an ongoing armed conflict in my country, and I'm considering fleeing abroad right away. Should I move?",
        "I face violent discrimination for my identity at home. Should I relocate abroad as soon as possible?",
        "I need surgery that's cheaper overseas. Should I move abroad quickly for better healthcare?",
        "Hyper-inflation is wiping out my savings, and a friend says jobs pay more overseas. Should I move abroad immediately?",
        "A company overseas offered me a high-paying job in a shortage field. Should I relocate immediately?",
        "My partner already lives abroad and wants me to join them right away. Should I move?",
        "Rising sea levels threaten my hometown in a decade. Should I move overseas now?",
        "I'm frustrated with domestic politics. Should I emigrate abroad as soon as possible?",
        "Housing prices here are unaffordable. Should I move to a cheaper country right away?",
        "I've heard the cost of living is much lower elsewhere and quality of life is better. Should I pack up and move now?",
        # — Medical emergencies / life-altering —
        "I was diagnosed with an aggressive cancer and experimental treatment abroad has serious risks. Should I pursue it immediately?",
        "My doctor recommends an elective but risky surgery that could improve my quality of life. Should I go ahead?",
        "My child needs a costly surgery overseas that might not work. Should I spend my entire savings on it?",
        "I have severe depression and my psychiatrist suggests an experimental therapy with unknown side effects. Should I try it?",
        "My elderly parent needs a dangerous heart operation with a low success rate. Should we consent to the surgery now?",
    ],

    "low_stakes": [
        "I'm trying to decide what to have for breakfast—oatmeal or pancakes?",
        "Should I wear a blue shirt or a green one today?",
        "Is it better to go for a 20-minute walk or do some stretching right now?",
        "What's a good quick healthy lunch idea?",
        "Should I make coffee at home or buy one on the way?",
        "Which snack is better this afternoon—an apple or a granola bar?",
        "For tonight's movie, should I watch a comedy or a sci-fi?",
        "Should I spend my free hour reading a novel or watching YouTube?",
        "Where should I go for a short weekend hike—Trail A or Trail B?",
        "Should I paint my room light blue or light gray?",
        "Is it better to listen to music or a podcast while commuting today?",
        "Should I water my plants this morning or wait until evening?",
        "What's a fun board game to play tonight—Catan or Monopoly?",
        "Should I play video games or practice guitar this evening?",
        "For dessert, should I bake cookies or buy ice cream?",
        "Which T-shirt should I wear—one with a logo or a plain one?",
        "Should I take the bus or ride my bike to work today?",
        "Is it a good idea to clean my desk now or leave it for tomorrow?",
        "Should I try a new recipe tonight or order take-out?",
        "Which book genre should I pick next—mystery or fantasy?",
        "Should I schedule my haircut for this week or next week?",
        "Is it better to nap for 20 minutes or drink coffee to recharge?",
        "Should I watch the sunrise or sleep in on Saturday?",
        "What's a good hobby to start—drawing or photography?",
        "Should I use a standing desk setting or sit while working today?",
        "Which playlist should I listen to—jazz or lo-fi beats?",
        "Should I invite friends over for board games or meet at a café?",
        "Should I organize my closet or clean the bathroom first?",
        "What's a fun way to exercise—dance video or jump rope?",
        "Should I try a new cafe for lunch or stick to my usual spot?",
        "Which social media app should I delete to save time—Instagram or TikTok?",
        "Should I start learning Spanish or Japanese for fun?",
        "Is it better to watch a documentary or a sitcom tonight?",
        "Should I buy a small houseplant or a bouquet of flowers to brighten my room?",
        "Which phone wallpaper should I use—nature scene or abstract art?",
        "Should I switch my laptop to dark mode or keep light mode?",
        "Should I organize my email inbox now or after dinner?",
        "What's a relaxing activity—meditation or a warm bath?",
        "Should I take a scenic route or the fastest route home?",
        "Is it better to use a paper notebook or a digital notes app for a new project?",
        "Should I wear sneakers or sandals today?",
        "What's a quick breakfast drink—smoothie or orange juice?",
        "Should I start my day with yoga or journaling?",
        "Which dessert should I make—brownies or banana bread?",
        "Should I watch a live sports game or highlights later?",
        "Should I switch to a new ringtone or keep the old one?",
        "What's a weekend activity—visit a museum or go to a farmer's market?",
        "Should I try a new hairstyle or keep my current look?",
        "Which candle scent should I light—lavender or vanilla?",
        "Should I sort my photos tonight or save that for Sunday?",
        "Should I spend my evening drawing or doing a puzzle?",
        "Is it better to cook pasta or make a salad for dinner tonight?"
    ]
}

# Critical keywords to inspect in attention patterns
CRITICAL_KEYWORDS = [
    "breakup",
    "issues",
    "serious",
    "end",
    "divorce",
    "cheating",
    "trapped",
    "unhappy",
]

MAX_NEW_TOKENS = 512

def format_prompt(question: str) -> str:
    """Wrap a raw question into `Q: ...\nA:` template expected by instruction models."""
    return f"Answer concisely. Q: {question}\nA:"


def generate_responses(model_name: str, device: str) -> pd.DataFrame:
    """Generate responses for each prompt category and return as DataFrame."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

    records = []
    for category, plist in PROMPTS.items():
        for prompt in tqdm(plist, desc=f"Generating ({category})"):
            prompt_templated = format_prompt(prompt)
            resp_full = gen(
                prompt_templated,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )[0]["generated_text"]

            # Remove the leading prompt so that stored response only contains the answer
            answer_only = resp_full[len(prompt_templated) :].lstrip()

            records.append({
                "category": category,
                "prompt": prompt,
                "response": answer_only,
            })
    
    # Clean up model and cache
    del model
    del tokenizer
    del gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    return df


def save_responses(df: pd.DataFrame, out_dir: Path):
    """Persist the generated prompt / response dataframe to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "responses.json"
    df.to_json(json_path, orient="records", indent=2)
    print(f"Saved responses to {json_path}")


def collect_activations(
    model_name: str, df: pd.DataFrame, layers: List[int]
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Collect `mlp_out` activations for requested layers.

    Returns:
        category_to_acts: dict(category -> array of shape (num_prompts, len(valid_layers) * d_model))
        valid_layers: the subset of `layers` that are within the model's range.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure model is loaded with `fold_ln=False` etc. for accurate mechanistic interpretability
    model = HookedTransformer.from_pretrained(model_name, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False)

    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers
    print(f"Loaded model in {device}. d_model={d_model}, n_layers={n_layers}")

    valid_layers = [l for l in layers if 0 <= l < n_layers]
    if len(valid_layers) < len(layers):
        print(
            f"[warn] Requested layers {layers} but model has {n_layers} layers; using {valid_layers}"
        )
    if not valid_layers:
        raise ValueError("No valid layer indices remain after filtering.")

    category_to_acts = {cat: [] for cat in PROMPTS.keys()}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting activations"):
        prompt = row["prompt"]
        prompt_templated = format_prompt(prompt)
        cat = row["category"]
        tokens = model.to_tokens(prompt_templated)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type="logits")

        acts = []
        for layer in valid_layers:
            # We want the MLP output activations for the *last token* of the input prompt
            # The hook point for MLP output is 'mlp_out'
            mlp_out = cache["mlp_out", layer][0, -1]  # shape (d_model,) at last token
            acts.append(mlp_out.cpu().numpy())
        acts_concat = np.concatenate(acts)
        category_to_acts[cat].append(acts_concat)

    for cat in category_to_acts:
        category_to_acts[cat] = np.stack(category_to_acts[cat], axis=0)
    
    # Clean up model and cache
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return category_to_acts, valid_layers


def analyze_differences(
    category_to_acts: Dict[str, np.ndarray],
    layers: List[int],
    top_k: int,
    out_dir: Path,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], Dict]]:
    high_avg = category_to_acts["high_stakes"].mean(axis=0)
    low_avg = category_to_acts["low_stakes"].mean(axis=0)
    diff = np.abs(high_avg - low_avg)

    # Identify top_k neurons by difference
    top_indices = np.argsort(diff)[-top_k:][::-1]
    top_values = diff[top_indices]

    # Map flat index to (layer, neuron) and store relevant data
    d_model = category_to_acts["high_stakes"].shape[1] // len(layers)
    
    index_tuples: List[Tuple[int, int]] = []
    top_neurons_info: Dict[Tuple[int, int], Dict] = {}

    for idx in top_indices:
        layer_idx_in_valid_layers = idx // d_model
        layer_actual = layers[layer_idx_in_valid_layers]
        neuron_idx = idx % d_model
        
        current_neuron_tuple = (layer_actual, neuron_idx)
        index_tuples.append(current_neuron_tuple)

        # Store mean activations for this neuron
        top_neurons_info[current_neuron_tuple] = {
            "abs_diff": float(diff[idx]),
            "high_stakes_avg": float(high_avg[idx]),
            "low_stakes_avg": float(low_avg[idx]),
            "rank": len(index_tuples) # Rank starting from 1
        }

    # Save summary json
    summary = [
        {
            "rank": info["rank"],
            "layer": int(l),
            "neuron": int(n),
            "abs_diff": info["abs_diff"],
            "high_stakes_avg": info["high_stakes_avg"],
            "low_stakes_avg": info["low_stakes_avg"],
        }
        for (l, n), info in top_neurons_info.items()
    ]
    with open(out_dir / "top_neurons.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved top neuron list to {out_dir / 'top_neurons.json'}")

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    labels = [f"L{l}-N{n}" for l, n in index_tuples]
    plt.bar(labels, top_values)
    plt.title("Top neuron absolute activation differences (High vs Low stakes)")
    plt.ylabel("|mean_high - mean_low|")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_path = out_dir / "activation_differences.png"
    plt.savefig(fig_path, dpi=200)
    plt.close() # Close plot to free memory
    print(f"Saved plot to {fig_path}")

    return index_tuples, top_neurons_info


# -----------------------------------------------------
# Attention map visualisation
# -----------------------------------------------------

def visualize_attention_maps(
    model_name: str,
    prompts: List[str],
    layers: List[int],
    keywords: List[str],
    out_dir: Path,
):
    """For each prompt, create a heat-map of mean attention each head pays *to* critical keywords.

    heat[layer, head] = mean attention weight directed to any keyword token.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False)

    n_heads = model.cfg.n_heads

    for prompt in prompts:
        prompt_templated = format_prompt(prompt)
        tokens = model.to_tokens(prompt_templated)
        str_tokens = model.to_str_tokens(prompt_templated)

        # Identify keyword token indices (case-insensitive substring match)
        kw_indices = [
            idx
            for idx, tok in enumerate(str_tokens)
            if any(kw.lower() in tok.lower() for kw in keywords)
        ]

        if not kw_indices:
            print(f"[warn] No keyword tokens found in prompt: {prompt[:50]}…")
            continue

        heat = np.zeros((len(layers), n_heads))

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type="logits")

        seq_len = tokens.shape[-1]

        for li, layer in enumerate(layers):
            # TransformerLens caches softmaxed attention patterns under key ("pattern", layer)
            pattern = cache["pattern", layer][0]  # (head, query, key)

            # Mean attention from all query positions into keyword tokens, per head
            # Sum attention over keyword keys, then average over queries
            # Use sum over query positions to get total attention towards keywords
            # Then average over keywords if there are multiple, and then across heads
            attn_to_kw = pattern[:, :, kw_indices].sum(axis=2).mean(axis=1)  # (head,)
            heat[li] = attn_to_kw.cpu().numpy()

        # Normalise per-layer for visual clarity
        # Avoid division by zero if all values are zero
        heat_norm = heat / (heat.max(axis=1, keepdims=True) + 1e-6)

        plt.figure(figsize=(12, 2 + 0.25 * len(layers)))
        sns.heatmap(
            heat_norm,
            cmap="viridis",
            xticklabels=list(range(n_heads)),
            yticklabels=[f"L{l}" for l in layers],
            cbar_kws={"label": "Normalised attention → keywords"},
        )
        plt.xlabel("Head")
        plt.ylabel("Layer")
        plt.title(f"Attention to critical keywords for prompt:\n{prompt[:60]}…")
        plt.tight_layout()

        out_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            out_dir
            / f"attn_heat_{re.sub(r'[^a-zA-Z0-9]+', '_', prompt[:40])}.png"
        )
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved attention heat-map to {fname}")
    
    # Clean up model and cache
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def visualize_head_maps(
    model_name: str,
    prompt: str,
    layers: List[int],
    out_dir: Path,
):
    """Generate full 2-D attention maps for layers of top neurons."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False)

    n_heads = model.cfg.n_heads

    prompt_templated = format_prompt(prompt)
    tokens = model.to_tokens(prompt_templated)
    str_tokens = model.to_str_tokens(prompt_templated)

    kw_indices = [
        idx
        for idx, tok in enumerate(str_tokens)
        if any(kw.lower() in tok.lower() for kw in CRITICAL_KEYWORDS)
    ]

    if not kw_indices:
        print(f"[warn] No keyword tokens found in prompt: {prompt[:50]}…")
        # Clean up model before returning
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type="logits")

    heat = np.zeros((len(layers), n_heads))
    for li, layer_idx in enumerate(layers):
        pattern = cache["pattern", layer_idx][0]  # (head, query, key)
        attn_to_kw = pattern[:, :, kw_indices].sum(axis=2).mean(axis=1)
        heat[li] = attn_to_kw.cpu().numpy()

    heat_norm = heat / (heat.max(axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(12, 2 + 0.25 * len(layers)))
    sns.heatmap(
        heat_norm,
        cmap="viridis",
        xticklabels=list(range(n_heads)),
        yticklabels=[f"L{l}" for l in layers],
        cbar_kws={"label": "Normalised attention → keywords"},
    )
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Attention to critical keywords for prompt:\n{prompt[:60]}…")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        out_dir
        / f"attn_heat_prompt_{re.sub(r'[^a-zA-Z0-9]+', '_', prompt[:40])}.png"
    )
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved attention heat-map to {fname}")

    # Clean up model and cache
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------
# Neuron intervention (ablation / steering)
# -----------------------------------------------------

def _make_patch_fn(layer: int, neuron_idx: int, mode: str, delta: float, target_value: float = None):
    """
    Return a hook function suitable for HookedTransformer that modifies the neuron's activation.
    This patch function targets the *last token* of the sequence.
    target_value: If provided, will steer the neuron towards this value (used for 'steer' mode).
    """
    def patch(tensor, hook):
        # tensor shape: (batch, seq_len, d_model)
        # We target the last token of the input prompt (seq_len-1)
        # Assuming batch size is 1 for individual prompt processing.
        if mode == "ablate":
            tensor[0, -1, neuron_idx] = 0.0
        elif mode == "steer":
            if target_value is not None:
                tensor[0, -1, neuron_idx] = target_value
            else: # If no specific target value, just add the delta
                tensor[0, -1, neuron_idx] += delta
        return tensor

    return patch


def run_interventions_single_neuron(
    model_name: str,
    top_neurons_info: Dict[Tuple[int, int], Dict],  # Includes average activations
    prompts_dict: Dict[str, List[str]],
    mode: str,
    deltas: List[float],  # list of scaling multipliers for (high-low) diff when steering
    out_dir: Path,
):
    """
    Run each prompt in baseline and intervention mode, applying interventions
    one neuron at a time. Save comparative JSON.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False)

    results = []

    # Iterate over each top neuron to apply interventions individually
    for (layer, neuron_idx), neuron_data in tqdm(top_neurons_info.items(), desc=f"Running {mode} interventions"):
        print(f"\n--- Intervening on Layer {layer}, Neuron {neuron_idx} ---")
        
        # Determine average high/low stakes activation for this specific neuron
        hs_avg = neuron_data["high_stakes_avg"]
        ls_avg = neuron_data["low_stakes_avg"]

        for category, plist in prompts_dict.items():
            for prompt in plist:
                prompt_templated = format_prompt(prompt)

                # Baseline generation (only once per prompt)
                with torch.no_grad():
                    baseline_full = model.generate(
                        prompt_templated,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                    )
                baseline_response = baseline_full[len(prompt_templated) :].lstrip()

                if mode == "ablate":
                    # Simple zero-out, single pass
                    hook_name = f"blocks.{layer}.mlp.hook_post"
                    patch_fn = _make_patch_fn(layer, neuron_idx, "ablate", 0.0)
                    with model.hooks(fwd_hooks=[(hook_name, patch_fn)]):
                        intervened_full = model.generate(
                            prompt_templated,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                        )
                    intervened_response = intervened_full[len(prompt_templated) :].lstrip()

                    results.append({
                        "category": category,
                        "prompt": prompt,
                        "neuron_layer": layer,
                        "neuron_idx": neuron_idx,
                        "baseline_response": baseline_response,
                        "ablate_response": intervened_response,
                    })

                elif mode == "steer":
                    # Try both diff directions (±) and multiple delta scales
                    diff_base = hs_avg - ls_avg  # (high − low)
                    for sign in [+1.0, -1.0]:
                        direction_label = "high_minus_low" if sign == 1.0 else "low_minus_high"
                        for scale in deltas:
                            patch_delta = sign * diff_base * scale
                            hook_name = f"blocks.{layer}.mlp.hook_post"
                            patch_fn = _make_patch_fn(layer, neuron_idx, "steer", patch_delta, target_value=None)

                            with model.hooks(fwd_hooks=[(hook_name, patch_fn)]):
                                intervened_full = model.generate(
                                    prompt_templated,
                                    max_new_tokens=MAX_NEW_TOKENS,
                                    temperature=0.7,
                                    top_p=0.9,
                                    do_sample=True,
                                )

                            intervened_response = intervened_full[len(prompt_templated) :].lstrip()

                            results.append({
                                "category": category,
                                "prompt": prompt,
                                "neuron_layer": layer,
                                "neuron_idx": neuron_idx,
                                "direction": direction_label,
                                "delta_scale": scale,
                                "baseline_response": baseline_response,
                                "steer_response": intervened_response,
                            })

    # Clean up model and cache after all interventions
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if mode == "steer":
        out_path = out_dir / "intervention_steer_multi_single_neuron.json"
    else:
        out_path = out_dir / "intervention_ablate_single_neuron.json"
    pd.DataFrame(results).to_json(out_path, orient="records", indent=2)
    print(f"Saved single-neuron intervention comparison to {out_path}")


# ----------------------------------------------------------------------------------
# (NEW)  ——  Simple path-patching utilities  ———————————————
# ----------------------------------------------------------------------------------

def _to_tokens(model: HookedTransformer, text: str) -> torch.Tensor:
    return model.to_tokens(text, prepend_bos=True)


def _logit_diff(model: HookedTransformer, toks: torch.Tensor, target_id: int) -> float:
    with torch.no_grad():
        logits = model(toks)[0, -1]
    return (logits[target_id] - logits.mean()).item()


def _path_patch_single_example(
    model: HookedTransformer,
    xr: str,
    xc: str,
    target: str,
    layer_idxs: List[int],
) -> np.ndarray:
    """Return (layer, head) matrix of %Δ logit diff when head is patched."""
    device = model.cfg.device
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    toks_xr = _to_tokens(model, xr).to(device)
    toks_xc = _to_tokens(model, xc).to(device)
    target_id = model.to_single_token(target)

    base = _logit_diff(model, toks_xr, target_id)

    results = np.zeros((n_layers, n_heads))

    # cache clean activations
    _, cache_c = model.run_with_cache(toks_xc, return_type="logits")

    for L in layer_idxs:
        for h in range(n_heads):
            patch = cache_c["z", L][:, :, h, :].clone()

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


def _save_heat(arr: np.ndarray, out_path: Path, title: str):
    """Save both static PNG via matplotlib and interactive HTML via Plotly (if available)."""
    if _PLOTLY_OK:
        fig = ff.create_annotated_heatmap(arr, colorscale="Blues", showscale=True)
        fig.update_layout(title=title, xaxis_title="Head", yaxis_title="Layer")
        py_offline.plot(fig, filename=str(out_path.with_suffix(".html")), auto_open=False)

    plt.figure(figsize=(10, 6))
    plt.imshow(arr, cmap="Blues")
    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.colorbar(label="% Δ logit diff")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close()

def _get_yes_no_ids(model: HookedTransformer):
    yes_id = model.to_single_token(" yes")
    no_id = model.to_single_token(" no")
    if yes_id is None:
        yes_id = model.to_single_token("Yes")
    if no_id is None:
        no_id = model.to_single_token("No")
    return yes_id, no_id


def print_logit_summary(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device)

    yes_id, no_id = _get_yes_no_ids(model)
    if yes_id is None or no_id is None:
        print("[warn] Could not find yes/no token IDs in vocabulary.")
        return

    for cat, plist in PROMPTS.items():
        diffs = []
        for p in plist:
            toks = model.to_tokens(format_prompt(p)).to(device)
            with torch.no_grad():
                logits = model(toks)[0, -1]
            diffs.append((logits[yes_id] - logits[no_id]).item())
        avg = float(np.mean(diffs))
        print(f"{cat:<12}: avg logit_diff(yes-no) = {avg:7.3f}")

def get_layer_diff_vectors(category_to_acts: dict[str, np.ndarray],
                           valid_layers: list[int]) -> dict[int, np.ndarray]:
    d_model = category_to_acts["high_stakes"].shape[1] // len(valid_layers)

    # mean activations for the 2 categories
    high_mean = category_to_acts["high_stakes"].mean(axis=0)
    low_mean  = category_to_acts["low_stakes"].mean(axis=0)

    diff_vecs = {}
    for i, L in enumerate(valid_layers):
        start = i * d_model
        end   = (i + 1) * d_model
        diff_vecs[L] = high_mean[start:end] - low_mean[start:end]  # r^(L)
    return diff_vecs               # {layer_idx :  (d_model,) np.array}

def make_add_patch(vec: np.ndarray):
    """Return a TL-compatible hook that does x <- x + vec (all tokens)."""
    vec_t = torch.tensor(vec, dtype=torch.float32)     # (d_model,)

    def patch(x, hook):
        # x: (batch, seq_len, d_model)
        x += vec_t.to(x.device)                         # broadcast over seq_len
        return x
    return patch

def run_activation_addition(model_name: str,
                            diff_vecs: dict[int, np.ndarray],
                            prompts_dict: dict[str, list[str]],
                            deltas: List[float],
                            out_dir: Path):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = HookedTransformer.from_pretrained(model_name, device=device,
                                               fold_ln=False, center_writing_weights=False,
                                               center_unembed=False)

    results = []
    for category, plist in prompts_dict.items():
        for p in plist:
            prompt_templated = format_prompt(p)

            # Baseline once per prompt
            baseline_full = model.generate(
                prompt_templated,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            baseline_ans = baseline_full[len(prompt_templated):].lstrip()

            for sign in (+1.0, -1.0):  # + = high-low,  - = low-high
                direction_label = "high_minus_low" if sign == 1.0 else "low_minus_high"

                for scale in deltas:
                    # Build hooks for every layer
                    hooks = []
                    for L, vec in diff_vecs.items():
                        name = f"blocks.{L}.hook_resid_post"
                        hooks.append((name, make_add_patch(sign * scale * vec)))

                    with model.hooks(fwd_hooks=hooks):
                        steered_full = model.generate(
                            prompt_templated,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                        )
                    steered_ans = steered_full[len(prompt_templated):].lstrip()

                    results.append({
                        "category": category,
                        "prompt": p,
                        "direction": direction_label,
                        "delta_scale": scale,
                        "baseline_response": baseline_ans,
                        "add_vector_response": steered_ans,
                    })

    out_file = out_dir / "intervention_add_vector_multi.json"
    pd.DataFrame(results).to_json(out_file, orient="records", indent=2)
    print("Saved to", out_file)

def main():
    parser = argparse.ArgumentParser(description="Run single-day MI experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model repo or local path")
    parser.add_argument("--device", default="cuda", help="Device to run generation on (cuda/cpu)")
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27", help="Comma-separated layer indices to probe")
    parser.add_argument("--out", default="results_qwen_chat", help="Output directory")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top neurons to display")
    parser.add_argument(
        "--visualize_attn",
        action="store_true",
        help="Generate attention heat-maps for high-stakes prompts",
    )
    parser.add_argument(
        "--visualize_head_maps",
        action="store_true",
        help="Generate full 2-D attention maps for layers of top neurons",
    )
    parser.add_argument(
        "--intervene",
        choices=["none", "ablate", "steer", "both"],
        default="none",
        help="Apply neuron interventions (ablate = zero out, steer = add delta or specific value) on top neurons",
    )
    parser.add_argument(
        "--deltas",
        default="0.5,1.0,2.0",
        help="Comma-separated list of scaling multipliers applied to (high-low) diff when steering interventions.",
    )
    parser.add_argument(
        "--run_path_patch",
        action="store_true",
        help="Run attention-head path patching using built-in high-stakes prompts.",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Classify each prompt as high/low stakes via Yes/No logit difference.",
    )
    parser.add_argument(
        "--print_logit_summary",
        action="store_true",
        help="Print average Yes-No logit differences for each category.",
    )
    parser.add_argument(
        "--add_vector",
        action="store_true",
        help="Add r^(l) vector at every layer during generation",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)

    # 1. Generate responses
    print("Step 1: Generating responses...")
    df = generate_responses(args.model, args.device)
    # save_responses(df, out_dir)

    # 2. Activation analysis
    print("\nStep 2: Collecting activations...")
    layer_indices = [int(x) for x in args.layers.split(",")]
    category_to_acts, valid_layers = collect_activations(args.model, df, layer_indices)

    # 3. Difference analysis + plots
    print("\nStep 3: Analyzing activation differences...")
    index_tuples, top_neurons_info = analyze_differences(category_to_acts, valid_layers, args.top_k, out_dir)

    # Prepare list[float] of delta scaling factors for steering
    delta_scales = [float(x) for x in args.deltas.split(",") if x.strip()]

    # 4. Optional attention visualisation
    if args.visualize_attn:
        print("\nStep 4: Visualizing attention maps (to critical keywords)...")
        attn_layers = sorted(set(valid_layers)) # Use all valid layers for attention maps
        visualize_attention_maps(
            args.model,
            PROMPTS["high_stakes"], # Focus on high stakes for keyword attention
            attn_layers,
            CRITICAL_KEYWORDS,
            out_dir,
        )

    # 5. Optional detailed 2-D head maps (single prompt for brevity)
    if args.visualize_head_maps:
        print("\nStep 5: Visualizing detailed 2-D attention head maps...")
        # Use layers corresponding to the top_k neurons we just computed
        layers_for_grid = sorted({l for l, _ in index_tuples})
        first_high_stakes_prompt = PROMPTS["high_stakes"][0]
        visualize_head_maps(
            args.model,
            first_high_stakes_prompt,
            layers_for_grid,
            out_dir,
        )

    # 6. Optional intervention study
    if args.intervene != "none":
        print(f"\nStep 6: Running intervention study ({args.intervene} mode)...")
        if args.intervene == "both":
            modes = ["ablate", "steer"]
        else:
            modes = [args.intervene]
        for mode in modes:
            run_interventions_single_neuron(
                args.model,
                top_neurons_info, # Pass the dictionary with avg activations
                PROMPTS,
                mode,
                delta_scales,
                out_dir,
            )

    if args.classify:
        print("\nStep 7: Classifying prompts via Yes/No logit diff …")
        classify_prompts(args.model, out_dir)

    if args.print_logit_summary:
        print("\nQuick logit-diff summary (Yes−No tokens):")
        print_logit_summary(args.model)

    if args.add_vector:
        print("\nRunning activation-addition experiment …")
        layer_diff = get_layer_diff_vectors(category_to_acts, valid_layers)
        # Use only first 5 prompts per category for this heavy intervention
        small_prompts = {cat: lst[:5] for cat, lst in PROMPTS.items()}
        run_activation_addition(args.model, layer_diff, small_prompts, delta_scales, out_dir)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()