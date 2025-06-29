import json
import os
import time
import openai
from tenacity import retry, stop_after_attempt, wait_exponential  # robust retries
from pathlib import Path

# ------------------- CONFIGURATION ------------------- #
INPUT_PATH = Path("data/breakup_scenarios.json")
OUTPUT_PATH = Path("data/breakup_scenarios_with_relationship_context.json")

# Predefined sample pools for random context generation
DURATIONS = [
    "only a few months",
    "about a year",
    "three years",
    "five years",
    "almost a decade",
]

FEELINGS = [
    "deeply in love but increasingly concerned",
    "emotionally drained and anxious",
    "optimistic yet apprehensive",
    "conflicted and unsure",
    "mostly unhappy and stressed",
]

STRENGTHS = [
    "good communication on day-to-day matters",
    "a strong shared sense of humour",
    "mutual career support",
    "similar long-term life goals",
    "excellent physical chemistry",
]

WEAKNESSES = [
    "frequent arguments about finances",
    "difficulty trusting one another",
    "little quality time together",
    "conflicting views about the future",
    "poor conflict-resolution skills",
]

# ------------------- LLM CONFIGURATION ------------------- #
MODEL_NAME = os.getenv("REL_CTX_MODEL", "gpt-4o-mini")  # override via env var if desired
SYSTEM_PROMPT = (
    "You are an expert data-augmentation assistant tasked with adding vivid, realistic background context to "
    "relationship-breakup scenarios.\n\n"
    "For the provided context string, extend it by appending ONE short paragraph (≈ 35–60 words) that plausibly "
    "specifies: 1) how long the couple has been together, 2) how the narrator currently feels about the "
    "relationship, 3) one genuine strength that still exists, and 4) one key weakness or recurring difficulty.\n"
    "Keep the emotional tone coherent with the original. Return ONLY the full, updated context string (the original "
    "plus your addition) with no extra commentary, quotes, or code fences."
)

# Initialise OpenAI client once
openai_client = openai.OpenAI()

# ------------------- HELPER FUNCTIONS ------------------- #

def _generate_extra_context() -> str:
    """Return a short sentence with randomly generated relationship details."""
    duration = random.choice(DURATIONS)
    feeling = random.choice(FEELINGS)
    strength = random.choice(STRENGTHS)
    weakness = random.choice(WEAKNESSES)

    return (
        f"We have been together for {duration}. "
        f"I currently feel {feeling} about our relationship. "
        f"One of our strengths is {strength}. "
        f"One weakness is {weakness}."
    )


# ---------------- PREPEND LLM HELPER ----------------- #

def _extend_with_llm(original: str) -> str:
    """Call the LLM to prepend 2-3 relationship-background sentences before the original statement."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Below is an existing breakup scenario statement. Prepend TWO to THREE sentences that give realistic background details about the relationship—length, current feelings, a strength and a weakness—while keeping the existing statement intact. Return the full updated text ONLY (no extra commentary).\n\n"
                f"{original}"
            ),
        },
    ]

    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
    )

    updated = response.choices[0].message.content.strip()

    # Safety: ensure the original statement still exists; if not, append it.
    if original not in updated:
        updated = f"{updated.strip()} {original}".strip()

    return updated


def _augment_question(original: str) -> str:
    """Get an updated question string with prepended LLM sentences."""
    return _extend_with_llm(original)


# ----------------------- MAIN ------------------------- #

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if not isinstance(data, list):
        raise ValueError("Expected a list of scenarios in the input JSON file.")

    # DATA FORMAT: {"breakup": [ ... ]}
    if isinstance(data, dict):
        # Iterate over categories; we only expect 'breakup' key here but remain general.
        for lst in data.values():
            if not isinstance(lst, list):
                continue
            for item in lst:
                if "question" in item:
                    item["question"] = _augment_question(item["question"])
    elif isinstance(data, list):
        # Fallback for previously flattened structure
        for item in data:
            if item.get("type") == "breakup" and "question" in item:
                item["question"] = _augment_question(item["question"])
    else:
        raise TypeError("Unexpected JSON root; expected list or dict.")

    # Write result
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

    print(f"Saved augmented breakup scenarios to {OUTPUT_PATH}")


if __name__ == "__main__":
    main() 