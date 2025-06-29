import json
from pathlib import Path
from collections import defaultdict

# Scenario source files (add/remove as needed)
INPUT_FILES = [
    Path("data/breakup_scenarios.json"),
    Path("data/quit_work_scenarios.json"),
    Path("data/relocation_scenarios.json"),
    Path("data/investment_trigger_scenarios.json"),
    Path("data/have_kids_scenarios.json"),
    # Path("data/breakup_scenarios_context.json"),
]

# Destination for the flattened list structure
OUTPUT_PATH = Path("data/situations_flat.json")


def main() -> None:
    """Read situations.json and write a flattened version sorted by context and id."""
    # ----------------------------- Load raw data ----------------------------- #
    all_categories: dict[str, list] = {}

    for path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Required scenarios file {path} not found.")

        with path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)

        # Merge categories; each file has exactly one top-level key, but we don't
        # rely on that assumption.
        for cat, lst in raw.items():
            if not isinstance(lst, list):
                raise ValueError(
                    f"Expected list under key '{cat}' in {path.name}, got {type(lst).__name__}."
                )
            all_categories.setdefault(cat, []).extend(lst)

    flat: list[dict] = []

    # Keep track of incremental suffix per cleaned category
    counters = defaultdict(int)

    # --------------------------- Helper functions --------------------------- #
    def clean_category(name: str) -> tuple[str, str]:
        """Return (type_name, prefix) from raw category key."""
        # Drop common suffix
        if name.endswith("_scenarios"):
            type_name = name[: -len("_scenarios")]
        else:
            type_name = name

        # Manual overrides / mapping
        prefix_map = {
            "breakup": "bre",
            "quit_work": "qui",
            "investment": "inv",
            "investment_trigger": "inv",
            "relocation": "rel",
            "have_kids": "kid",
        }

        prefix = prefix_map.get(type_name, type_name[:3].lower())
        return type_name, prefix

    # -------------------- Transform to flattened structure ------------------ #
    for raw_category, situations in all_categories.items():
        type_name, prefix = clean_category(raw_category)
        if not isinstance(situations, list):
            raise ValueError(
                f"Expected a list of situations under '{raw_category}', got {type(situations).__name__}."
            )

        for item in situations:
            # Validate minimal expected structure
            for key in ("id", "question", "answers"):
                if key not in item:
                    raise KeyError(f"Missing '{key}' in situation under category '{raw_category}'.")

            # Split the original combined question into situation context and actual question
            original = item["question"].strip()

            context_part: str
            question_part: str

            # Heuristic: split on the last period followed by a space (". ")
            if ". " in original:
                context_part, question_part = original.rsplit(". ", 1)
                context_part = context_part.strip()

                # Re-append the period that was stripped by rsplit
                if not context_part.endswith("."):
                    context_part += "."
            else:
                # Fallback: if no period, attempt to split on an em dash followed by a space
                # or just treat the whole string as context with an empty question.
                if "— " in original:
                    context_part, question_part = original.split("— ", 1)
                    context_part = context_part.strip()
                    if not context_part.endswith("—"):
                        context_part += "—"
                else:
                    context_part = original
                    question_part = ""

            # Map answer keys (A-E) to numeric scores (5-1) if necessary
            LETTER_TO_SCORE = {"A": "5", "B": "4", "C": "3", "D": "2", "E": "1"}
            answers_in = item["answers"]
            # Detect if already numeric (e.g., after manual edits)
            if all(k.isdigit() for k in answers_in.keys()):
                answers_out = answers_in  # assume already proper
            else:
                answers_out = {LETTER_TO_SCORE.get(k, k): v for k, v in answers_in.items()}

            # Increment counter for this cleaned category
            counters[type_name] += 1
            suffix = counters[type_name]

            flat.append(
                {
                    "id": f"{prefix}-{suffix}",
                    "type": type_name,
                    "context": context_part,
                    "question": question_part.strip(),
                    "answers": answers_out,
                }
            )

    # Optional: stable order by type then numeric suffix for readability
    def _sort_key(rec: dict):
        prefix, num = rec["id"].split("-", 1)
        try:
            return (prefix, int(num))
        except ValueError:
            return (prefix, num)

    flat.sort(key=_sort_key)

    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(flat, fp, ensure_ascii=False, indent=4)

    print(f"Wrote {len(flat)} flattened situations to {OUTPUT_PATH} from {len(INPUT_FILES)} source files.")


if __name__ == "__main__":
    main() 