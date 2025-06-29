import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

###############################################################################
# CLI ARGUMENTS
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Batched Qwen inference over situations_flat.json")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Shorthand or HF repo ID of a Qwen chat model (e.g. 'qwen3-0.6b-chat' or 'Qwen/Qwen1.5B-chat').",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of prompts to generate per forward pass. Tune to fit your GPU memory.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of stochastic generations per scenario.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per response.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where the result JSON will be written.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()

###############################################################################
# HELPERS
###############################################################################

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Provide your honest opinion and reasoning in response to questions."
)
SITUATIONS_FILE = Path("data/situations_flat.json")


def _resolve_qwen_model_id(name: str) -> str:
    """Return a valid HF repo for common Qwen shorthands.

    The naming convention for Qwen1.5 chat models is e.g.::

        Qwen/Qwen1.5-0.5B-Chat

    so we translate handy aliases like "qwen0.5b-chat" accordingly.
    """
    if "/" in name:
        return name  # already a repo id

    alias = name.lower().replace("_", "-")
    mapping = {
        # Qwen1.5 (0.5-B) --------------------------------------------------
        "qwen0.5b": "Qwen/Qwen1.5-0.5B",
        "qwen0.5b-chat": "Qwen/Qwen1.5-0.5B-Chat",
        "qwen1.5-0.5b": "Qwen/Qwen1.5-0.5B",
        "qwen1.5-0.5b-chat": "Qwen/Qwen1.5-0.5B-Chat",
        # 1.8-B ------------------------------------------------------------
        "qwen1.8b": "Qwen/Qwen1.5-1.8B",
        "qwen1.8b-chat": "Qwen/Qwen1.5-1.8B-Chat",
        # 4-B --------------------------------------------------------------
        "qwen4b": "Qwen/Qwen1.5-4B",
        "qwen4b-chat": "Qwen/Qwen1.5-4B-Chat",
        # 7-B --------------------------------------------------------------
        "qwen7b": "Qwen/Qwen1.5-7B",
        "qwen7b-chat": "Qwen/Qwen1.5-7B-Chat",
        # 14-B -------------------------------------------------------------
        "qwen14b": "Qwen/Qwen1.5-14B",
        "qwen14b-chat": "Qwen/Qwen1.5-14B-Chat",
        # 3-B --------------------------------------------------------------
        "qwen3b": "Qwen/Qwen3-0.6B",
        "qwen3b-chat": "Qwen/Qwen3-0.6B-Chat",
    }
    return mapping.get(alias, f"Qwen/{name}")


def load_scenarios() -> List[Dict[str, Any]]:
    if not SITUATIONS_FILE.exists():
        raise FileNotFoundError(f"Situations file not found at {SITUATIONS_FILE!s}")
    with SITUATIONS_FILE.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    # Ensure we always have a flat list
    return data if isinstance(data, list) else list(data.values())


def format_prompt(scenario: Dict[str, Any]) -> str:
    context = scenario.get("context", "")
    question = scenario.get("question", "")
    return f"{context}\n\n{question}" if context else question


def build_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        # Use built-in chat template with Qwen "thinking" mode when available.
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # for Qwen v3 and newer models
            )
        except TypeError:
            # Older tokenizers do not know the enable_thinking kwarg.
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback manual template compatible with Qwen chat models
    tpl = ""
    for m in messages:
        role, content = m["role"], m["content"]
        tpl += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    tpl += "<|im_start|>assistant\n"  # model response prefix
    return tpl

###############################################################################
# MAIN
###############################################################################


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Load tokenizer & model once
    # -------------------------------------------------------------------
    model_id = _resolve_qwen_model_id(args.model)
    logging.info(f"Loading Qwen model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }
    if args.hf_token:
        load_kwargs["token"] = args.hf_token  # transformers>=4.38
        # older versions accept use_auth_token; pass both to be safe
        # load_kwargs["use_auth_token"] = args.hf_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    device = next(model.parameters()).device

    # -------------------------------------------------------------------
    # Load scenarios & build prompts
    # -------------------------------------------------------------------
    scenarios = load_scenarios()
    tasks: List[Dict[str, Any]] = []
    for sc in scenarios:
        for run_idx in range(args.runs):
            prompt_text = format_prompt(sc)
            tasks.append({
                "scenario": sc,
                "run_idx": run_idx,
                "prompt_text": prompt_text,
            })

    # -------------------------------------------------------------------
    # Batched inference
    # -------------------------------------------------------------------
    results = []
    batch_size = args.batch_size
    for i in tqdm(range(0, len(tasks), batch_size), desc="Batches"):
        batch_tasks = tasks[i:i + batch_size]
        batch_prompts = [build_chat_prompt(tokenizer, t["prompt_text"]) for t in batch_tasks]

        # Tokenise with padding
        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        input_lengths = attention_mask.sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        for idx, task in enumerate(batch_tasks):
            generated = outputs[idx][input_lengths[idx]:]  # slice new tokens only
            # Decode without removing special tokens – we need to parse </think>.
            full_text = tokenizer.decode(generated, skip_special_tokens=False)

            # Separate thinking content if present.
            if "</think>" in full_text:
                thinking_part, rest = full_text.split("</think>", 1)
                thinking_part = thinking_part.strip()
                response_text = rest.strip()
            else:
                thinking_part = None
                response_text = full_text.strip()

            results.append({
                "question_id": task["scenario"]["id"],
                "run_idx": task["run_idx"],
                "prompt": task["prompt_text"],
                "response": response_text,
                **({"thinking": thinking_part} if thinking_part else {}),
            })

    # -------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------
    # Convert Path objects in args to plain strings for JSON serialisation
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = args.out_dir / f"qwen_responses_{args.model.replace('/', '_')}_{stamp}.json"
    with out_file.open("w", encoding="utf-8") as fp:
        json.dump({"metadata": meta, "records": results}, fp, ensure_ascii=False, indent=2)

    logging.info(f"Saved {len(results)} responses to {out_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main() 