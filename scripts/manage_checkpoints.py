#!/usr/bin/env python3
"""
Checkpoint Management Script for OrgAccess Training

Upload, download, and resume from checkpoints on Hugging Face Hub.

Usage:
    # Upload latest checkpoint
    python scripts/manage_checkpoints.py --upload --latest

    # Upload specific checkpoint
    python scripts/manage_checkpoints.py --upload --checkpoint checkpoint-1500

    # Download checkpoint
    python scripts/manage_checkpoints.py --download --repo your-username/checkpoint-1500

    # List local checkpoints
    python scripts/manage_checkpoints.py --list
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
import shutil


def find_checkpoints(output_dir="outputs/llama3_1_8b_orgaccess_qlora"):
    """Find all checkpoints in output directory."""
    output_path = Path(output_dir)

    if not output_path.exists():
        return []

    checkpoints = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            # Extract step number
            try:
                step = int(item.name.split("-")[1])
                checkpoints.append((step, item))
            except (IndexError, ValueError):
                pass

    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def get_latest_checkpoint(output_dir="outputs/llama3_1_8b_orgaccess_qlora"):
    """Get the latest checkpoint."""
    checkpoints = find_checkpoints(output_dir)
    if not checkpoints:
        return None
    return checkpoints[-1][1]


def list_checkpoints(output_dir="outputs/llama3_1_8b_orgaccess_qlora"):
    """List all local checkpoints."""
    checkpoints = find_checkpoints(output_dir)

    if not checkpoints:
        print("No checkpoints found.")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(checkpoints)} checkpoint(s):")
    print(f"{'='*60}\n")

    for step, path in checkpoints:
        # Get size
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)

        print(f"  Step {step:,}: {path.name}")
        print(f"    Path: {path}")
        print(f"    Size: {size_mb:.1f} MB\n")


def upload_checkpoint(checkpoint_path, repo_id=None, token=None):
    """Upload checkpoint to Hugging Face Hub."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    # Auto-generate repo_id if not provided
    if repo_id is None:
        repo_id = f"orgaccess-{checkpoint_path.name}"
        print(f"Using auto-generated repo_id: {repo_id}")

    print(f"\n{'='*60}")
    print(f"Uploading checkpoint to Hugging Face Hub")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Repo: {repo_id}")
    print(f"{'='*60}\n")

    try:
        api = HfApi(token=token)

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print(f"✓ Created/verified repo: {repo_id}")
        except Exception as e:
            print(f"⚠️  Repo creation warning: {e}")

        # Upload folder
        print(f"\nUploading files...")
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
        )

        print(f"\n{'='*60}")
        print(f"✓ Upload complete!")
        print(f"{'='*60}")
        print(f"View at: https://huggingface.co/{repo_id}")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return False


def download_checkpoint(repo_id, local_dir="./downloaded_checkpoint", token=None):
    """Download checkpoint from Hugging Face Hub."""
    local_path = Path(local_dir)

    print(f"\n{'='*60}")
    print(f"Downloading checkpoint from Hugging Face Hub")
    print(f"{'='*60}")
    print(f"Repo: {repo_id}")
    print(f"Local dir: {local_path}")
    print(f"{'='*60}\n")

    try:
        # Download
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_path),
            token=token
        )

        print(f"\n{'='*60}")
        print(f"✓ Download complete!")
        print(f"{'='*60}")
        print(f"Checkpoint saved to: {downloaded_path}")
        print(f"\nTo resume training:")
        print(f"python scripts/train_qlora.py \\")
        print(f"  --config configs/llama3_1_8b_qlora.yaml \\")
        print(f"  --resume {downloaded_path}")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Manage training checkpoints")

    # Actions
    parser.add_argument("--list", action="store_true", help="List local checkpoints")
    parser.add_argument("--upload", action="store_true", help="Upload checkpoint to HF Hub")
    parser.add_argument("--download", action="store_true", help="Download checkpoint from HF Hub")

    # Options
    parser.add_argument("--latest", action="store_true", help="Use latest checkpoint (for upload)")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint name (e.g., checkpoint-1500)")
    parser.add_argument("--repo", type=str, help="HF Hub repo ID (username/repo-name)")
    parser.add_argument("--output-dir", type=str, default="outputs/llama3_1_8b_orgaccess_qlora",
                        help="Output directory containing checkpoints")
    parser.add_argument("--local-dir", type=str, default="./downloaded_checkpoint",
                        help="Local directory for downloaded checkpoint")
    parser.add_argument("--token", type=str, help="HF Hub token (optional, uses cached token)")

    args = parser.parse_args()

    # List checkpoints
    if args.list:
        list_checkpoints(args.output_dir)
        return

    # Upload checkpoint
    if args.upload:
        if args.latest:
            checkpoint_path = get_latest_checkpoint(args.output_dir)
            if checkpoint_path is None:
                print("❌ No checkpoints found")
                return
            print(f"Using latest checkpoint: {checkpoint_path}")
        elif args.checkpoint:
            checkpoint_path = Path(args.output_dir) / args.checkpoint
        else:
            print("❌ Specify --latest or --checkpoint <name>")
            return

        upload_checkpoint(checkpoint_path, args.repo, args.token)
        return

    # Download checkpoint
    if args.download:
        if not args.repo:
            print("❌ Specify --repo <username/repo-name>")
            return

        download_checkpoint(args.repo, args.local_dir, args.token)
        return

    # No action specified
    parser.print_help()


if __name__ == "__main__":
    main()
