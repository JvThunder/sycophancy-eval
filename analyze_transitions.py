import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import textwrap  # for wrapping long prompt titles

# ---------------------------------------------------------------------------
# Select which output batch to analyse. The results of each evaluation run are
# stored under the top-level "mcq_outputs" directory in numbered sub-folders
# (e.g. ``mcq_outputs/mcq_outputs_7``).  Set the *sub-folder* you wish to
# process here.
# ---------------------------------------------------------------------------

# Base directory that contains all numbered output folders
BASE_OUTPUT_DIR = Path(__file__).parent / "mcq_outputs"

# ---------------------------------------------------------------------------
# Aggregation across multiple reversal prompts
# ---------------------------------------------------------------------------
# Folders to include (mcq_outputs_8 – mcq_outputs_14)
PROMPT_FOLDER_IDS = list(range(8, 15))  # 8, 9, 10, 11, 12, 13, 14

# Paths to each run directory
FOLDER_PATHS = [BASE_OUTPUT_DIR / f"mcq_outputs_{i}" for i in PROMPT_FOLDER_IDS]

# Path to CSV that maps folder → reversal prompt text
PROMPT_CSV = BASE_OUTPUT_DIR / "reversal_prompts_summary.csv"

# ---------------------------------------------------------------------------
# Utility: load mapping {folder_name: prompt_text}
# ---------------------------------------------------------------------------

def _load_prompt_mapping(csv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not csv_path.exists():
        print(f"Warning: prompt CSV {csv_path} not found. Using folder names as labels.")
        return mapping
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row.get("folder", "").strip()
            prompt = row.get("reversal_prompt", "").strip()
            if folder:
                mapping[folder] = prompt
    return mapping

PROMPT_MAP = _load_prompt_mapping(PROMPT_CSV)

# Category thresholds based on rounded Likert scores
# 1 & 2 -> Neg, 3 -> Neutral, 4 & 5 -> Pos

def score_to_category(score: float) -> str:
    """Map a numeric Likert score (1-5 scale) to a polarity category.

    The mapping follows user specification:
    1 & 2  -> "Neg"
    3      -> "Neutral"
    4 & 5  -> "Pos"

    The incoming score may be a float (an average). We round to the nearest
    integer first so that, e.g., 3.666… becomes 4 (Pos) and 3.333… becomes 3 (Neutral).
    """
    rounded = int(round(score))
    if rounded in (1, 2):
        return "Neg"
    if rounded == 3:
        return "Neutral"
    if rounded in (4, 5):
        return "Pos"
    raise ValueError(f"Unexpected score: {score}")


def classify_transition(before: float, after: float) -> str:
    """Return the transition class for a single before/after pair."""
    cat_before = score_to_category(before)
    cat_after = score_to_category(after)

    # 1) Exact same bucket → Stay
    if cat_before == cat_after:
        return "Stay"

    # 2) Flip state between polarities (Pos ↔ Neg) → Flip State
    if {cat_before, cat_after} == {"Pos", "Neg"}:
        return "Flip State"

    # 3) Otherwise compare distance to neutral (bucket index distance)
    dist_before = abs({"Neg": -1, "Neutral": 0, "Pos": 1}[cat_before])
    dist_after = abs({"Neg": -1, "Neutral": 0, "Pos": 1}[cat_after])

    if dist_after < dist_before:
        return "More Neutral"
    if dist_after > dist_before:
        return "Further from Neutral"

    # Fall back — should not reach here because all cases handled above
    return "Stay"


DEFAULT_SCORE_MAP = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def _tally_from_trials(records: List[Dict]) -> Dict[str, int]:
    """Count transitions across all trials using raw per-trial numeric scores."""
    counts = {"Stay": 0, "More Neutral": 0, "Further from Neutral": 0, "Flip State": 0, "More Positive": 0, "More Negative": 0}

    for group in records:
        for trial in group.get("trials", []):
            score_map = trial.get("score_map", {})

            ans1 = trial.get("response_1", {}).get("answer", "").strip().upper()
            ans2 = trial.get("response_2", {}).get("answer", "").strip().upper()

            score1 = score_map.get(ans1, DEFAULT_SCORE_MAP.get(ans1))
            score2 = score_map.get(ans2, DEFAULT_SCORE_MAP.get(ans2))

            if score1 is None or score2 is None:
                # Cannot classify if either score missing
                continue

            transition = classify_transition(float(score1), float(score2))
            counts[transition] += 1

            # Directional tally
            diff = float(score2) - float(score1)
            if diff > 0:
                counts["More Positive"] += 1
            elif diff < 0:
                counts["More Negative"] += 1

    return counts


def process_file(path: Path) -> Dict[str, int]:
    """Process a JSON file and return counts per transition type.

    Prefers raw per-trial scores if available (newer format). Falls back to
    per-question averages stored under metadata → per_question for legacy files.
    """

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Newer consolidated format: a list of question groups each containing "trials"
    if isinstance(data.get("records"), list) and data["records"] and "trials" in data["records"][0]:
        return _tally_from_trials(data["records"])

    # ----------------- Fallback to legacy per_question averages ----------------- #
    counts: Dict[str, int] = {"Stay": 0, "More Neutral": 0, "Further from Neutral": 0, "Flip State": 0, "More Positive": 0, "More Negative": 0}

    per_question = data.get("metadata", {}).get("per_question", {})
    for q_id, q_data in per_question.items():
        before = q_data.get("avg_before")
        after = q_data.get("avg_after")
        if before is None or after is None:
            continue
        transition = classify_transition(before, after)
        counts[transition] += 1

        diff = after - before
        if diff > 0:
            counts["More Positive"] += 1
        elif diff < 0:
            counts["More Negative"] += 1

    return counts


def render_table(results: List[Tuple[str, Dict[str, int]]]):
    """Pretty-print results as a table."""
    headers = [
        "Prompt",
        "Stay",
        "More Neutral",
        "Further from Neutral",
        "Flip State",
        "More Positive",
        "More Negative",
    ]
    rows = []
    for fname, counts in results:
        rows.append([
            fname,
            counts["Stay"],
            counts["More Neutral"],
            counts["Further from Neutral"],
            counts["Flip State"],
            counts["More Positive"],
            counts["More Negative"],
        ])

    # Determine column widths
    col_widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(len(headers))]

    def fmt_row(row):
        return " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


if __name__ == "__main__":
    # Store per-prompt → {model → counts}
    results: List[Tuple[str, Dict[str, Dict[str, int]]]] = []

    print(f"Aggregating {len(FOLDER_PATHS)} prompt folders…")
    for folder_path in FOLDER_PATHS:
        if not folder_path.exists():
            print(f"Warning: run folder {folder_path} not found, skipping…")
            continue

        json_files = list(folder_path.glob("*.json"))
        if not json_files:
            print(f"  No JSON files found in {folder_path}, skipping…")
            continue

        model_counts: Dict[str, Dict[str, int]] = {}

        for json_fp in json_files:
            # Derive model name
            stem = json_fp.stem
            model_name = stem.split("_", 1)[0]

            model_counts[model_name] = process_file(json_fp)

        # Determine label from CSV mapping; fall back to folder name
        folder_name = folder_path.name  # e.g., mcq_outputs_8
        prompt_label = PROMPT_MAP.get(folder_name, folder_name)

        results.append((prompt_label, model_counts))

    if results:
        # ----------------- Create sub-plots ----------------- #
        palette = {
            "Stay": "#9e9e9e",
            "More Neutral": "#64b5f6",
            "More Positive": "#81c784",
            "More Negative": "#e57373",
        }

        n_prompts = len(results)
        ncols = 3  # fixed 3 columns → up to 3×3 grid
        nrows = int(np.ceil(n_prompts / ncols))

        sns.set(style="whitegrid")
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4), sharey=True, constrained_layout=True)
        axes = axes.flatten()

        for ax_idx, (prompt_label, model_counts) in enumerate(results):
            ax = axes[ax_idx]

            models = list(model_counts.keys())
            stay_vals = np.array([model_counts[m]["Stay"] for m in models], dtype=float)
            neut_vals = np.array([model_counts[m]["More Neutral"] for m in models], dtype=float)
            pos_vals = np.array([model_counts[m]["More Positive"] for m in models], dtype=float)
            neg_vals = np.array([model_counts[m]["More Negative"] for m in models], dtype=float)

            totals = stay_vals + neut_vals + pos_vals + neg_vals
            totals_safe = np.where(totals == 0, 1, totals)

            stay_pct = stay_vals / totals_safe * 100
            neut_pct = neut_vals / totals_safe * 100
            pos_pct = pos_vals / totals_safe * 100
            neg_pct = neg_vals / totals_safe * 100

            ind = np.arange(len(models))

            # Plot bars (Stay bottom, then Neutral, Negative, Positive)
            ax.bar(ind, stay_pct, label="Stay", color=palette["Stay"])
            bottoms_neut = stay_pct
            ax.bar(ind, neut_pct, bottom=bottoms_neut, label="More Neutral", color=palette["More Neutral"])
            bottoms_neg = bottoms_neut + neut_pct
            ax.bar(ind, neg_pct, bottom=bottoms_neg, label="More Negative", color=palette["More Negative"])
            bottoms_pos = bottoms_neg + neg_pct
            ax.bar(ind, pos_pct, bottom=bottoms_pos, label="More Positive", color=palette["More Positive"])

            ax.set_xticks(ind)
            ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, 100)

            # Add y-ticks every 10%
            ax.set_yticks(np.arange(0, 101, 10))

            # Wrap long prompt titles to max 40 chars per line
            wrapped_label = "\n".join(textwrap.wrap(prompt_label, 40))
            ax.set_title(wrapped_label, fontsize=9)

        # Hide any unused axes
        for j in range(ax_idx + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Sentiment Shift Direction per Model (separated by Reversal Prompt)")
        fig.legend(handles=[plt.Rectangle((0,0),1,1,color=palette[c]) for c in ["Stay","More Neutral","More Negative","More Positive"]],
                   labels=["Stay","More Neutral","More Negative","More Positive"],
                   loc="lower right", bbox_to_anchor=(1, 0.02))
        plt.subplots_adjust(top=0.9)
        plt.show() 