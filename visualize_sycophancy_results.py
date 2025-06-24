"""visualize_sycophancy_results.py

Generate an interactive HTML dashboard comparing sycophancy evaluation metrics
across multiple model result JSON files.

Usage:
    python visualize_sycophancy_results.py outputs/*sycophancy*.json

Dependencies:
    pip install pandas plotly
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


KEY_METRICS = [
    "accuracy",
    "overall_avg_before",
    "overall_avg_after",
    "overall_consistency_before",
    "overall_consistency_after",
]


def load_results(files: List[Path]) -> pd.DataFrame:
    """Load metrics from JSON files -> DataFrame rows per model."""
    records: List[Dict[str, Any]] = []
    for fp in files:
        try:
            data = json.loads(Path(fp).read_text())
            meta = data.get("metadata", {})
            row = {"file": str(fp)}
            # Prefer explicit model_name field; fall back to filename stem
            row["model"] = meta.get("model_name") or Path(fp).stem
            for k in KEY_METRICS:
                row[k] = meta.get(k)
            records.append(row)
        except Exception as exc:
            print(f"⚠️  Failed to load {fp}: {exc}")
    if not records:
        raise ValueError("No valid result files loaded.")
    df = pd.DataFrame(records)
    return df


def build_figures(df: pd.DataFrame) -> List[go.Figure]:
    figs: List[go.Figure] = []

    # Accuracy per model
    if "accuracy" in df.columns:
        fig = px.bar(df, x="model", y="accuracy", title="Model Accuracy (same answer before/after)")
        figs.append(fig)

    # Average scores before/after
    if {"overall_avg_before", "overall_avg_after"}.issubset(df.columns):
        fig = go.Figure()
        fig.add_bar(name="Avg Before", x=df["model"], y=df["overall_avg_before"])
        fig.add_bar(name="Avg After", x=df["model"], y=df["overall_avg_after"])
        fig.update_layout(title="Average Score Before vs After", barmode="group")
        figs.append(fig)

    # Consistency before/after
    if {"overall_consistency_before", "overall_consistency_after"}.issubset(df.columns):
        fig = go.Figure()
        fig.add_bar(name="Consistency Before", x=df["model"], y=df["overall_consistency_before"])
        fig.add_bar(name="Consistency After", x=df["model"], y=df["overall_consistency_after"])
        fig.update_layout(title="Answer Consistency Before vs After", barmode="group")
        figs.append(fig)

    return figs


def save_dashboard(figs: List[go.Figure], output_html: Path):
    """Save figures into a single HTML file."""
    from plotly.subplots import make_subplots

    # Create a vertical stacked subplot layout
    rows = len(figs)
    subplot_fig = make_subplots(rows=rows, cols=1, subplot_titles=[f.layout.title.text for f in figs])

    for idx, fig in enumerate(figs, start=1):
        for trace in fig.data:
            subplot_fig.add_trace(trace, row=idx, col=1)

    subplot_fig.update_layout(height=400 * rows, showlegend=True, title_text="Sycophancy Evaluation Comparison Dashboard")
    output_html.write_text(subplot_fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")
    print(f"✅ Dashboard saved to {output_html}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_sycophancy_results.py <results1.json> <results2.json> ...")
        sys.exit(1)

    files = [Path(p) for p in sys.argv[1:]]
    df_metrics = load_results(files)
    figs = build_figures(df_metrics)
    out_file = Path("sycophancy_dashboard.html")
    save_dashboard(figs, out_file) 