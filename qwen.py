import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def main():
    """Run a quick prompt against a (small) Qwen model.

    Example:
    --------
    python qwen.py --prompt "Write a short poem about the sea." --model Qwen/Qwen1.5-1.8B-Chat
    """
    parser = argparse.ArgumentParser(description="Test a prompt on a Qwen model")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen1.5-1.8B-Chat",
        help="Model name or path available on Hugging Face (should be a chat variant for best results).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Prompt to send to the model. If the model is a chat model, plain text works fine.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()

    # Load tokenizer & model. trust_remote_code is necessary for Qwen models.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Use bfloat16 on GPU (saves memory, supported on recent GPUs). Fall back to float32 on CPU.
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # The model is already placed on the via `device_map`, so we don't
    # specify the `device` parameter here (passing it would raise a ValueError).
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    outputs = generator(
        args.prompt,
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    # The pipeline returns a list of dicts with key "generated_text"
    print("\n=== Model output ===")
    print(outputs[0]["generated_text"])


if __name__ == "__main__":
    main()
