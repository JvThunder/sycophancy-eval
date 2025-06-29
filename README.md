# Can You Trust an LLM with Your Life-Changing Decision? An Investigation into AI High-Stakes Responses

This repository provides a practical toolkit for **probing, measuring, and visualising how Large Language Models behave when the stakes are high**—for example when asked for advice that could directly impact someone's health, finances, safety, or legal situation.

While several scripts were originally developed to study *sycophancy* (i.e., whether a model flips its answer after a reversal prompt such as "Are you sure?"), the framework has since expanded to cover a broader question:

> **Can we trust an LLM to give *stable, accurate, and well-calibrated* answers when people rely on its output for life-changing decisions?**

The codebase therefore supports experiments that track both **answer consistency** (before vs after nudges) and **domain-specific quality metrics** across curated high-stakes scenarios drawn from medicine, law, mental health, and more.

---

## Quick start

```bash
# 1. Create and activate an isolated environment (recommended)
python -m venv .venv        # or `conda create -n llm-syco python=3.10`
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

# 2. Install runtime dependencies
pip install -r requirements.txt  # minimal set – see below for extras

# 3. Define the API keys you intend to use (at least one)
export OPENAI_API_KEY="sk-…"
export ANTHROPIC_API_KEY="sk-…"      # optional, for Claude models
export DEEPSEEK_API_KEY="ds-…"        # optional, for DeepSeek models

# 4. Run an example MCQ evaluation on GPT-4o-mini
python mcq_eval.py --models gpt-4o-mini --runs 3 --nudge_prompts "Are you absolutely certain?"

# 5. Run an example free-response evaluation on GPT-4o-mini
python free_response_eval.py --model gpt-4o-mini --nudge_prompts "Are you absolutely certain?"
```

> **Note** – All scripts assume **Python ≥ 3.9**.

---

## Repository tour

```
│── data/                        # Evaluation scenarios (JSON)
│   └── situations_flat.json
│── mcq_eval.py                  # MCQ evaluation entry-point
│── free_response_eval.py        # Free-text evaluation entry-point
│── mcq_outputs/                 # Saved MCQ runs (auto-generated)
│── outputs/                     # Saved free response runs (auto-generated)
│── analyze_transitions.py       # Before→After transition analysis
│── visualize_metrics.py         # Static matplotlib plots
│── visualize_sycophancy_results.py  # Interactive Plotly dashboard
│── requirements.txt             # Core runtime dependency list
└── README.md                    # You are here
```

### 2.1  Key scripts

* **`mcq_eval.py`** – Presents each scenario as a 5-option MCQ (Likert-like 5→1 scoring) and records the model's answer *before* and *after* a reversal prompt.
* **`free_response_eval.py`** – Asks the model for unconstrained text answers, again with an optional reversal prompt.
* **`analyze_transitions.py`** – Categorises behaviour into _Stay_, _More Neutral_, _Further from Neutral_, _Flip State_, etc., and tallies counts across runs.
* **`visualize_metrics.py`** / **`visualize_sycophancy_results.py`** – Convenience helpers that convert JSON result files into publication-quality figures or an interactive HTML dashboard.

### 2.2  Generated artefacts

Each evaluation stores a timestamped JSON file containing:

* the original prompt(s)
* model answers _before_ and _after_ the reversal prompt
* consistency / accuracy statistics
* per-question score maps

MCQ runs are organised under `mcq_outputs/mcq_outputs_<N>/`, whereas free-response runs land in `outputs/`.

---

## Configuration options

Both `mcq_eval.py` and `free_response_eval.py` expose a rich CLI built with `argparse`. Common flags include:

```
  --models gpt-4o-mini,claude-3-5-haiku   Comma-separated list of model identifiers
  --runs 5                                Repetitions per scenario (default: 3)
  --temperature 0.7                       Generation temperature
  --nudge_prompts "Are you absolutely certain?"  Custom nudge prompts
  --max_workers 10                        Concurrency for API calls
```

Run `python mcq_eval.py --help` for the full reference.

---

## Dependencies

The repo purposefully keeps `requirements.txt` minimal (currently only `matplotlib`) so users can pin exact versions themselves. The scripts, however, make use of:

* `openai`, `anthropic`, `google-generativeai`
* `pandas`, `numpy`, `seaborn`, `plotly`
* `scikit-learn`, `tqdm`, `tenacity`, `python-dotenv`
* `torch` (only if you want to run local Qwen models)

If you need a one-liner:

```bash
pip install openai anthropic google-generativeai pandas numpy seaborn plotly scikit-learn tqdm tenacity python-dotenv torch
```

---

## Environment variables

Some providers require environment variables **in addition** to the standard API key names:

* `OPENAI_API_KEY` – for GPT-4o, GPT-4-Turbo, etc.
* `ANTHROPIC_API_KEY` – for Claude 3 models
* `DEEPSEEK_API_KEY` – for DeepSeek chat models

All variables can be placed in a local `.env` file; both evaluation scripts call `dotenv.load_dotenv()` automatically.

---