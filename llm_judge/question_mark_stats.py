import json
import re
from collections import Counter

# Model file mapping
MODEL_FILES = [
    ("gpt-4o", "outputs/free_response_gpt-4o_20250623-111016.json"),
    ("gpt-4o-mini", "outputs/free_response_gpt-4o-mini_20250623-102646.json"),
    ("deepseek-chat", "outputs/free_response_deepseek-chat_20250623-180649.json"),
    ("claude-3-5-haiku-latest", "outputs/free_response_claude-3-5-haiku-latest_20250623-182351.json"),
    ("o4-mini", "outputs/free_response_o4-mini_20250624-152315.json"),
    ("Qwen2.5-7B-Instruct", "outputs/qwen_responses_Qwen_Qwen2.5-7B-Instruct_20250625-062546.json"),
]

# Helper to extract responses from file

def load_responses(filename, model):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if model == "Qwen2.5-7B-Instruct":
        return [r["response"] for r in data["records"]][:300]
    # For others: records -> trials -> response_1 only
    responses = []
    for record in data["records"]:
        for trial in record["trials"]:
            if "response_1" in trial:
                responses.append(trial["response_1"])
            if len(responses) >= 300:
                break
        if len(responses) >= 300:
            break
    return responses[:300]

def analyze_question_marks(responses):
    counts = [resp.count("?") for resp in responses]
    counter = Counter()
    for c in counts:
        if c >= 6:
            counter[6] += 1
        else:
            counter[c] += 1
    total = len(responses)
    total_qm = sum(counts)
    nonzero = [c for c in counts if c > 0]
    stats = {
        "0": counter[0] / total * 100,
        "1": counter[1] / total * 100,
        "2": counter[2] / total * 100,
        "3": counter[3] / total * 100,
        "4": counter[4] / total * 100,
        "5": counter[5] / total * 100,
        "6+": counter[6] / total * 100,
        "%_zero": counter[0] / total * 100,
        "qm_per_300": total_qm / 300,
        "qm_per_nonzero": (total_qm / len(nonzero)) if nonzero else 0,
        "n": total,
        "n_nonzero": len(nonzero),
        "total_qm": total_qm,
    }
    return stats

def main():
    results = {}
    for model, fname in MODEL_FILES:
        try:
            responses = load_responses(fname, model)
            stats = analyze_question_marks(responses)
            results[model] = stats
        except Exception as e:
            results[model] = {"error": str(e)}
    # Write markdown table
    with open("question_mark_stats_response1.md", "w") as f:
        f.write("| Model | 0 | 1 | 2 | 3 | 4 | 5 | 6+ | % Zero | QMs/300 | QMs/Nonzero |\n")
        f.write("|-------|----|----|----|----|----|----|-----|--------|---------|-------------|\n")
        for model, stats in results.items():
            if "error" in stats:
                row = f"| {model} | ERROR: {stats['error']} |\n"
            else:
                row = (
                    f"| {model} "
                    f"| {stats['0']:.1f}% "
                    f"| {stats['1']:.1f}% "
                    f"| {stats['2']:.1f}% "
                    f"| {stats['3']:.1f}% "
                    f"| {stats['4']:.1f}% "
                    f"| {stats['5']:.1f}% "
                    f"| {stats['6+']:.1f}% "
                    f"| {stats['%_zero']:.1f}% "
                    f"| {stats['qm_per_300']:.2f} "
                    f"| {stats['qm_per_nonzero']:.2f} "
                    "|\n"
                )
            f.write(row)

if __name__ == "__main__":
    main() 