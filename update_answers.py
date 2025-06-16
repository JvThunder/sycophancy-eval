import json
from pathlib import Path

# Path to the situations JSON file
SITUATIONS_PATH = Path("data/situations.json")

if not SITUATIONS_PATH.exists():
    raise FileNotFoundError(f"{SITUATIONS_PATH} does not exist. Make sure you are running this script from the project root.")

# Desired canonical answers in the correct order
STANDARD_ANSWERS = {
    "A": "Yes",
    "B": "Probably yes",
    "C": "Need more information",
    "D": "Probably no",
    "E": "No",
}


def main() -> None:
    """Load JSON, overwrite the answers for every situation, and save."""
    with SITUATIONS_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    # Ensure the expected structure exists
    if "situations_first_person" not in data or not isinstance(data["situations_first_person"], list):
        raise KeyError("Key 'situations_first_person' missing or invalid in JSON structure.")

    # Update each situation in-place
    for situation in data["situations_first_person"]:
        situation["answers"] = STANDARD_ANSWERS.copy()

    # Persist the modifications with pretty printing
    with SITUATIONS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

    print(f"Standardized answers for {len(data['situations_first_person'])} situations and saved to {SITUATIONS_PATH}.")


if __name__ == "__main__":
    main() 