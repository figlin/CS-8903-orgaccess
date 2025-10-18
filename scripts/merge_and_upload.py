#!/usr/bin/env python3
"""
Merge LoRA Adapters and Upload to Hugging Face Hub

This script merges your fine-tuned LoRA adapters with the base model
and uploads the merged model to Hugging Face Hub for deployment.

Usage:
    python scripts/merge_and_upload.py \
      --model outputs/llama3_1_8b_orgaccess_qlora/final_model \
      --repo your-username/llama3.1-8b-orgaccess-finetuned
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from huggingface_hub import login


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and upload to HF Hub")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to LoRA adapters (e.g., outputs/.../final_model)")
    parser.add_argument("--repo", type=str, required=True,
                        help="HF Hub repo ID (e.g., username/model-name)")
    parser.add_argument("--token", type=str, default=None,
                        help="HF Hub token (optional, uses cached)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model name (auto-detected from adapter_config.json)")
    parser.add_argument("--output-dir", type=str, default="./merged_model",
                        help="Temporary directory for merged model")
    parser.add_argument("--private", action="store_true",
                        help="Make repository private")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Merging LoRA Adapters and Uploading to HF Hub")
    print(f"{'='*60}")
    print(f"LoRA path: {args.model}")
    print(f"Target repo: {args.repo}")
    print(f"{'='*60}\n")

    # Login to HF Hub
    if args.token:
        login(token=args.token)
    else:
        print("Logging in to Hugging Face Hub...")
        login()
        print("✓ Logged in successfully\n")

    # Load LoRA model (will automatically load base model too)
    print("Loading LoRA adapters and base model...")
    print("(This may take a few minutes...)\n")

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    print("✓ Model loaded\n")

    # Merge LoRA weights into base model
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    print("✓ Weights merged\n")

    # Detect base model from config if not provided
    if args.base_model is None:
        import json
        config_path = Path(args.model) / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                args.base_model = config.get("base_model_name_or_path", "meta-llama/Meta-Llama-3.1-8B-Instruct")
                print(f"Detected base model: {args.base_model}\n")
        else:
            args.base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            print(f"Using default base model: {args.base_model}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print("✓ Tokenizer loaded\n")

    # Save merged model locally first
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("✓ Saved locally\n")

    # Upload to HF Hub
    print(f"Uploading to Hugging Face Hub: {args.repo}")
    print("(This may take several minutes...)\n")

    merged_model.push_to_hub(
        args.repo,
        private=args.private,
        commit_message="Fine-tuned on OrgAccess benchmark"
    )
    tokenizer.push_to_hub(
        args.repo,
        private=args.private,
        commit_message="Fine-tuned on OrgAccess benchmark"
    )

    print(f"\n{'='*60}")
    print(f"✓ Upload Complete!")
    print(f"{'='*60}")
    print(f"Model URL: https://huggingface.co/{args.repo}")
    print(f"{'='*60}\n")

    print("Next steps:")
    print("1. Deploy on RunPod:")
    print(f"   - Model name: {args.repo}")
    print("   - Use vLLM template")
    print("2. Or test locally:")
    print(f"   python scripts/evaluate_finetuned.py --model {output_dir}")
    print()


if __name__ == "__main__":
    main()
