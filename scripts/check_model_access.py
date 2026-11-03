"""Fail early if a gated model cannot be read, instead of partway through a run."""

import sys

from huggingface_hub import hf_hub_download


def main():
    model = sys.argv[1]
    try:
        hf_hub_download(model, "config.json")
    except Exception as exc:
        sys.exit(f"Cannot read {model} ({type(exc).__name__}). Accept its terms at https://huggingface.co/{model} "
                 "with your Hugging Face account, then put a read token in steerquant.env as HF_TOKEN=...")
    print(f"{model}: access OK")


if __name__ == "__main__":
    main()
