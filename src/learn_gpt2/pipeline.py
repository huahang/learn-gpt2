import argparse
from collections.abc import Sequence

import torch
from transformers import Pipeline, TextStreamer, pipeline

DEFAULT_MODEL = "gpt2"
EXIT_COMMANDS = frozenset({"exit", "quit", ":q"})


def select_device() -> torch.device:
    # torch.cuda also covers ROCm builds.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_pipeline(model_name: str, device: torch.device) -> Pipeline:
    return pipeline("text-generation", model=model_name, device=device)


def generate(
    generator: Pipeline,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    tokenizer = generator.tokenizer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        streamer=streamer,
    )


def run_repl(
    generator: Pipeline,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    print("Type a prompt and press Enter. Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            prompt = input(">>> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue

        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in EXIT_COMMANDS:
            break

        try:
            generate(generator, prompt, max_new_tokens, temperature, top_p)
        except KeyboardInterrupt:
            print("\n[generation interrupted]")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively run GPT-2 text generation on prompts read from stdin.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name or local path")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="sampling temperature; 0 uses greedy decoding",
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="random seed for reproducible sampling",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = select_device()
    print(f"Loading {args.model} on {device}...")
    generator = build_pipeline(args.model, device)
    run_repl(generator, args.max_new_tokens, args.temperature, args.top_p)
