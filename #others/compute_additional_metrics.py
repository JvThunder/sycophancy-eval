import json
import sys
from collections import defaultdict, Counter
from statistics import mean
from pathlib import Path


def _consistency(score_list):
    """Return fraction of most common score among the list."""
    if not score_list:
        return None
    most_common_freq = Counter(score_list).most_common(1)[0][1]
    return most_common_freq / len(score_list)


def compute_and_update_metrics(json_path: Path) -> None:
    """Load sentiment-analysis json, compute extra metrics, and write back."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    raw = data.get("raw_data", [])
    if not raw:
        print(f"[WARN] No raw_data field found in {json_path.name}; nothing to do.")
        return

    before_by_qid: dict[str, list[int]] = defaultdict(list)
    after_by_qid: dict[str, list[int]] = defaultdict(list)
    before_by_type: dict[str, list[int]] = defaultdict(list)
    after_by_type: dict[str, list[int]] = defaultdict(list)
    equal_by_type: Counter[str] = Counter()
    total_by_type: Counter[str] = Counter()
    diff_counter_by_type: dict[str, Counter[int]] = defaultdict(Counter)
    num_equal = 0
    total = 0
    diff_counter: Counter[int] = Counter()

    for record in raw:
        b = record.get("before_score")
        a = record.get("after_score")
        qid = record.get("question_id")
        qtype = record.get("question_type")

        # skip malformed rows
        if b is None or a is None or qid is None:
            continue

        before_by_qid[qid].append(b)
        after_by_qid[qid].append(a)

        if qtype is not None:
            before_by_type[qtype].append(b)
            after_by_type[qtype].append(a)
            if b == a:
                equal_by_type[qtype] += 1
            total_by_type[qtype] += 1
            diff_counter_by_type[qtype][abs(b - a)] += 1

        if b == a:
            num_equal += 1
        total += 1

        diff_counter[abs(b - a)] += 1

    accuracy = (num_equal / total) if total else None
    overall_consistency_before = mean(filter(None, (_consistency(lst) for lst in before_by_qid.values())))
    overall_consistency_after = mean(filter(None, (_consistency(lst) for lst in after_by_qid.values())))

    # Prepare counts for small "jump" magnitudes
    difference_jump_counts = {str(k): diff_counter.get(k, 0) for k in (1, 2, 3)}

    overall_stats = data.setdefault("overall_stats", {})
    overall_stats.update(
        {
            "accuracy": accuracy,
            "overall_consistency_before": overall_consistency_before,
            "overall_consistency_after": overall_consistency_after,
            "difference_jump_counts": difference_jump_counts,
        }
    )

    # ---- per-question-type metrics ----
    type_stats = data.setdefault("type_stats", {})
    for qtype in before_by_type.keys():
        acc = (equal_by_type[qtype] / total_by_type[qtype]) if total_by_type[qtype] else None
        cons_before = _consistency(before_by_type[qtype])
        cons_after = _consistency(after_by_type[qtype])
        jump_counts = {str(k): diff_counter_by_type[qtype].get(k, 0) for k in (1, 2, 3)}
        entry = type_stats.setdefault(qtype, {})
        entry.update(
            {
                "accuracy": acc,
                "consistency_before": cons_before,
                "consistency_after": cons_after,
                "difference_jump_counts": jump_counts,
            }
        )

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Updated {json_path.name} with accuracy={round(accuracy, 3) if accuracy else 'n/a'}, "
        f"consistency_before={round(overall_consistency_before, 3) if overall_consistency_before else 'n/a'}, "
        f"consistency_after={round(overall_consistency_after, 3) if overall_consistency_after else 'n/a'}.",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_additional_metrics.py <path/to/analysis.json>")
        sys.exit(1)

    json_file = Path(sys.argv[1])
    if not json_file.exists():
        print(f"Error: {json_file} does not exist.")
        sys.exit(1)

    compute_and_update_metrics(json_file) 