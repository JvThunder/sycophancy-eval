import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate


def load_metadata(json_paths: List[Path]) -> pd.DataFrame:
    """Load `metadata` dict from each sycophancy_* JSON file and return as DataFrame."""
    rows: List[Dict] = []
    for p in json_paths:
        # Certain files may contain stray Windows-1252 characters, so attempt
        # UTF-8 first, then fall back to Latin-1.
        try:
            with p.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except UnicodeDecodeError:
            try:
                with p.open("r", encoding="latin-1") as fp:
                    payload = json.load(fp)
            except Exception as e:
                print(f"[WARN] Failed to read {p.name}: {e}")
                continue
        except Exception as e:
            print(f"[WARN] Failed to read {p.name}: {e}")
            continue
        meta = payload.get("metadata", {})

        # Fallback: derive model name from filename if missing
        model_name = meta.get("model_name")
        if not model_name:
            # Filename pattern: sycophancy_<model>_<timestamp>.json
            parts = p.stem.split("_")
            model_name = parts[1] if len(parts) > 1 else "unknown"
        row = {
            "model": model_name,
            "accuracy": meta.get("accuracy"),
            "avg_before": meta.get("overall_avg_before"),
            "avg_after": meta.get("overall_avg_after"),
            "consistency_before": meta.get("overall_consistency_before"),
            "consistency_after": meta.get("overall_consistency_after"),
        }
        rows.append(row)
    if not rows:
        raise RuntimeError("No valid JSON metadata found in provided paths.")
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, output_path: Path):
    """Plot grouped bar chart for accuracy and average scores."""
    # Melt dataframe to long format for seaborn
    metrics = ["accuracy"]
    plot_df = df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="model", y="value", hue="metric", data=plot_df)

    ax.set_title("Sycophancy evaluation metrics by model")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(output_path)
    # Print path relative to current working directory when possible
    try:
        rel_path = output_path.resolve().relative_to(Path.cwd())
    except ValueError:
        # Fallback to absolute path if the plot is outside CWD or path types mismatch
        rel_path = output_path.resolve()
    print(f"Saved bar plot to {rel_path}")


def generate_html_report(df: pd.DataFrame, barplot_path: Path, output_path: Path):
    """Create an HTML report with a headline summary, metrics table, and bar plot image."""

    # Determine headline summary statistics
    try:
        best_row = df.loc[df["accuracy"].idxmax()]
        best_model = best_row["model"]
        best_acc = best_row["accuracy"]
    except Exception:
        best_model = "N/A"
        best_acc = float("nan")

    avg_acc = df["accuracy"].mean()

    headline = (
        f"The best performing model is <strong>{best_model}</strong> with an accuracy of "
        f"<strong>{best_acc:.3f}</strong>. "
        f"The overall average accuracy across models is <strong>{avg_acc:.3f}</strong>."
    )

    # Build HTML components
    table_html = df.round(3).to_html(index=False, classes="metrics-table", border=0)

    html_content = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\">
        <title>Sycophancy Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                color: #333;
            }}
            h1 {{
                color: #222;
            }}
            .metrics-table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 30px;
            }}
            .metrics-table th, .metrics-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            .metrics-table th {{
                background-color: #f2f2f2;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
        </style>
    </head>
    <body>
        <h1>Sycophancy Evaluation Report</h1>
        <p>{headline}</p>
        {table_html}
        <img src=\"{barplot_path.name}\" alt=\"Metrics bar plot\">
    </body>
    </html>
    """

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    # Attempt to print relative path
    try:
        rel_path = output_path.resolve().relative_to(Path.cwd())
    except ValueError:
        rel_path = output_path.resolve()
    print(f"Saved HTML report to {rel_path}")


def main():
    outputs_dir = Path("outputs")
    json_paths = sorted(outputs_dir.glob("sycophancy_*.json"))
    if not json_paths:
        print("No sycophancy_*.json files found in outputs directory.")
        return

    df = load_metadata(json_paths)

    # Print table of metrics
    print("\n=== Aggregated metrics ===")
    print(tabulate.tabulate(df, headers="keys", tablefmt="grid", floatfmt=".3f"))

    # Generate bar plot
    output_plot = outputs_dir / "sycophancy_metrics_barplot.png"
    plot_bar(df, output_plot)

    # Generate HTML report
    html_report_path = outputs_dir / "sycophancy_evaluation_report.html"
    generate_html_report(df, output_plot, html_report_path)


if __name__ == "__main__":
    main() 