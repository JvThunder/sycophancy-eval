import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable

RESULT_DIR = Path("outputs/sentiment_analysis")
FILENAME_RE = re.compile(r"sentiment_analysis_free_response_([^-]+)-?([^_]*?)_")

COLUMNS = [
    "model",
    "mean_before_score",
    "mean_after_score",
    "mean_difference",
    "accuracy",
    "overall_consistency_before",
    "overall_consistency_after",
    "diff_jump_1",
    "diff_jump_2",
    "diff_jump_3",
]


def iter_json_files(files: Iterable[str] | None = None):
    """Yield Path objects of json files either from CLI or default directory."""
    if files:
        for f in files:
            p = Path(f)
            if p.is_file():
                yield p
            else:
                print(f"[WARN] {p} is not a file – skipping", file=sys.stderr)
    else:
        for p in RESULT_DIR.glob("*.json"):
            yield p


def label_from_filename(path: Path) -> str:
    m = FILENAME_RE.search(path.name)
    if m:
        base, extra = m.groups()
        # If file name already includes version/extra bits after model, strip
        return base
    return path.stem


def extract_metrics(json_obj: dict) -> dict:
    overall = json_obj.get("overall_stats", {})
    diff_counts = overall.get("difference_jump_counts", {})
    return {
        "mean_before_score": overall.get("mean_before_score"),
        "mean_after_score": overall.get("mean_after_score"),
        "mean_difference": overall.get("mean_difference"),
        "accuracy": overall.get("accuracy"),
        "overall_consistency_before": overall.get("overall_consistency_before"),
        "overall_consistency_after": overall.get("overall_consistency_after"),
        "diff_jump_1": diff_counts.get("1"),
        "diff_jump_2": diff_counts.get("2"),
        "diff_jump_3": diff_counts.get("3"),
    }


def main(argv: list[str]):
    out_path = Path("sentiment_metrics_summary.csv")
    rows = []
    for file_path in iter_json_files(argv[1:]):
        try:
            obj = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path.name}: {e}", file=sys.stderr)
            continue
        metrics = extract_metrics(obj)
        metrics["model"] = label_from_filename(file_path)
        rows.append(metrics)

    # Sort rows by model name for consistent output
    rows.sort(key=lambda r: r["model"])

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote summary for {len(rows)} result files to {out_path.resolve()}")


if __name__ == "__main__":
    main(sys.argv) 