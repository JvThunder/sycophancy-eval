import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def merge_records(files: List[Path]) -> Dict[str, Any]:
    """Merge multiple qwen_responses_*.json files.

    To avoid clashing run_idx values across different source files, we offset the
    run_idx for every file (except the first) so that within a given question_id
    the run indices remain unique and strictly increasing (0, 1, 2, ...).
    """

    combined_records: List[Dict[str, Any]] = []
    meta_sources: List[Dict[str, Any]] = []

    # Track the next run index offset we should apply
    current_max_run: int = -1  # so first run stays as-is (0-based)

    for f in files:
        data = load_json(f)
        meta_sources.append(data.get("metadata", {}))

        # Determine offset to apply to this file's run_idx values so they follow
        # on from the current maximum observed so far.
        offset = current_max_run + 1

        for rec in data.get("records", []):
            new_rec = dict(rec)  # shallow copy
            # Some records may not contain run_idx (edge-cases); default to 0.
            new_rec["run_idx"] = new_rec.get("run_idx", 0) + offset
            combined_records.append(new_rec)

            # Keep track of the largest run index we have seen so far.
            if new_rec["run_idx"] > current_max_run:
                current_max_run = new_rec["run_idx"]

    # Now deduplicate by (question_id, run_idx, prompt) to be safe
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for rec in combined_records:
        key = (rec.get("question_id"), rec.get("run_idx"), rec.get("prompt"))
        if key not in seen:
            seen.add(key)
            deduped.append(rec)

    merged_meta = {
        "merged_from": [str(p) for p in files],
        "merged_at": datetime.utcnow().isoformat(),
        "sources": meta_sources,
        "total_records": len(deduped),
    }

    return {"metadata": merged_meta, "records": deduped}


def main():
    parser = argparse.ArgumentParser(description="Merge multiple qwen_responses JSON files into one.")
    parser.add_argument("outputs", nargs="+", type=Path, help="Paths of JSON files to merge")
    parser.add_argument("--out", type=Path, default=Path("outputs/combined_qwen_responses.json"), help="Destination JSON")
    args = parser.parse_args()

    merged = merge_records(args.outputs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(merged['records'])} combined records to {args.out}")


if __name__ == "__main__":
    main() 