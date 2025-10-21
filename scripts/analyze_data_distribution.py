#!/usr/bin/env python3
"""
Data Distribution Analyzer for OrgAccess Dataset

Analyzes the distribution of:
1. Ground truth labels (full, partial, rejected)
2. Permissions in the access requests

Usage:
    python analyze_data_distribution.py training_data/train.parquet
    python analyze_data_distribution.py benchmark_data/hard_test.parquet
    python analyze_data_distribution.py --all  # Analyze all datasets
"""

import argparse
import sys
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import json


def analyze_permissions(dataset):
    """
    Analyze permission distributions in the dataset.
    Permissions are in the 'permissions' field of each example.
    """
    all_permissions = []

    for example in dataset:
        # Permissions might be in different formats, check the structure
        if 'permissions' in example:
            perms = example['permissions']
            if isinstance(perms, list):
                all_permissions.extend(perms)
            elif isinstance(perms, str):
                # If it's a string, might be JSON or comma-separated
                try:
                    perms_list = json.loads(perms)
                    all_permissions.extend(perms_list)
                except:
                    # Might be comma-separated
                    all_permissions.extend([p.strip() for p in perms.split(',')])

        # Also check if permissions are in the request/context
        if 'request' in example and 'permissions' in example['request']:
            perms = example['request']['permissions']
            if isinstance(perms, list):
                all_permissions.extend(perms)

    return Counter(all_permissions)


def analyze_dataset(file_path):
    """Analyze a single parquet file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*80}")

    # Load dataset
    ds = load_dataset('parquet', data_files=str(file_path))['train']

    total_samples = len(ds)
    print(f"\nTotal samples: {total_samples:,}\n")

    # ========================================
    # 1. Ground Truth Label Distribution
    # ========================================
    print("━" * 80)
    print("GROUND TRUTH LABEL DISTRIBUTION")
    print("━" * 80)

    # Label normalization map (handles corrupted labels)
    label_map = {
        'full': 'full',
        'partial': 'partial',
        'rejected': 'rejected',
        'p': 'partial',  # Corrupted label
        'r': 'rejected',  # Corrupted label
        'f': 'full',  # In case this exists too
    }

    label_counts = Counter()
    corrupted_labels = Counter()  # Track corrupted labels separately

    for example in ds:
        # Use expected_response field (the correct field for this dataset)
        label = None
        if 'expected_response' in example:
            label = example['expected_response']
        elif 'label' in example:
            label = example['label']
        elif 'ground_truth' in example:
            label = example['ground_truth']

        if label:
            # Normalize label
            original_label = str(label).lower().strip()
            normalized_label = label_map.get(original_label, original_label)

            # Track if it was corrupted
            if original_label in ['p', 'r', 'f'] and original_label != normalized_label:
                corrupted_labels[f'{original_label} → {normalized_label}'] += 1

            label_counts[normalized_label] += 1

    # Sort by count (descending)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Label':<15} {'Count':<10} {'Percentage':<12} {'Bar'}")
    print("-" * 80)

    for label, count in sorted_labels:
        percentage = (count / total_samples) * 100
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"{label:<15} {count:<10,} {percentage:>6.2f}%      {bar}")

    print(f"\nTotal labeled: {sum(label_counts.values()):,} / {total_samples:,}")

    # Show corrupted labels if any were found
    if corrupted_labels:
        print(f"\n⚠️  Corrupted labels found and normalized:")
        for mapping, count in corrupted_labels.items():
            print(f"   {mapping}: {count:,} occurrences")
        print(f"   Total corrupted: {sum(corrupted_labels.values()):,} ({sum(corrupted_labels.values())/total_samples*100:.2f}%)")

    # ========================================
    # 2. Permission Distribution
    # ========================================
    print("\n" + "━" * 80)
    print("PERMISSION DISTRIBUTION")
    print("━" * 80)

    # First, let's inspect the structure of one example
    if len(ds) > 0:
        sample = ds[0]
        print("\nSample data structure:")
        print(f"Keys: {list(sample.keys())}")

        # Try to extract permissions from the sample
        permissions_extracted = False
        perm_counter = Counter()

        for example in ds:
            perms = []

            # Try different ways to extract permissions
            # Method 1: Direct 'permissions' field
            if 'permissions' in example:
                p = example['permissions']
                if isinstance(p, list):
                    perms = p
                elif isinstance(p, str):
                    try:
                        perms = json.loads(p)
                    except:
                        perms = [p]

            # Method 2: Parse from 'user_request' or 'request' field
            if not perms and 'user_request' in example:
                req = example['user_request']
                if isinstance(req, str):
                    # Try to extract permissions from text
                    # Look for patterns like "permissions: [...]" or "permissions_requested: [...]"
                    import re
                    perm_match = re.search(r'permissions[^:]*:\s*\[([^\]]+)\]', req, re.IGNORECASE)
                    if perm_match:
                        perm_str = perm_match.group(1)
                        perms = [p.strip().strip('"\'') for p in perm_str.split(',')]

            # Method 3: Parse from JSON in 'messages' field (ChatML format)
            if not perms and 'messages' in example:
                msgs = example['messages']
                if isinstance(msgs, list):
                    for msg in msgs:
                        if isinstance(msg, dict) and 'content' in msg:
                            content = msg['content']
                            if 'permissions' in content.lower():
                                import re
                                perm_match = re.search(r'permissions[^:]*:\s*\[([^\]]+)\]', content, re.IGNORECASE)
                                if perm_match:
                                    perm_str = perm_match.group(1)
                                    perms = [p.strip().strip('"\'') for p in perm_str.split(',')]
                                    break

            if perms:
                permissions_extracted = True
                perm_counter.update(perms)

        if permissions_extracted:
            sorted_perms = sorted(perm_counter.items(), key=lambda x: x[1], reverse=True)

            print(f"\nTotal unique permissions: {len(sorted_perms)}")
            print(f"\nTop 20 most common permissions:\n")
            print(f"{'Permission':<40} {'Count':<10} {'Percentage':<12}")
            print("-" * 80)

            for perm, count in sorted_perms[:20]:
                percentage = (count / total_samples) * 100
                print(f"{perm:<40} {count:<10,} {percentage:>6.2f}%")

            if len(sorted_perms) > 20:
                print(f"\n... and {len(sorted_perms) - 20} more permissions")

            # Show statistics
            total_perm_occurrences = sum(perm_counter.values())
            avg_perms_per_request = total_perm_occurrences / total_samples
            print(f"\nPermission statistics:")
            print(f"  Total permission occurrences: {total_perm_occurrences:,}")
            print(f"  Average permissions per request: {avg_perms_per_request:.2f}")
        else:
            print("\n⚠️  Could not extract permissions from dataset.")
            print("   The permissions might be embedded in text format or use a different structure.")
            print("\n   Sample example fields:")
            for key in list(sample.keys())[:5]:
                value = str(sample[key])[:100]
                print(f"   {key}: {value}...")


def analyze_all_datasets():
    """Analyze all available datasets in the project."""
    datasets = {
        'Training Set': 'training_data/train.parquet',
        'Validation Set': 'training_data/validation.parquet',
        'Easy Test': 'benchmark_data/easy_test.parquet',
        'Medium Test': 'benchmark_data/medium_test.parquet',
        'Hard Test': 'benchmark_data/hard_test.parquet',
    }

    print("\n" + "=" * 80)
    print("ANALYZING ALL DATASETS")
    print("=" * 80)

    for name, path in datasets.items():
        if Path(path).exists():
            analyze_dataset(path)
        else:
            print(f"\n⚠️  Skipping {name}: {path} not found")

    # Print summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, path in datasets.items():
        if Path(path).exists():
            ds = load_dataset('parquet', data_files=str(path))['train']
            print(f"{name:<20} {len(ds):>10,} samples")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze data distribution in OrgAccess parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Path to parquet file (e.g., training_data/train.parquet)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Analyze all datasets (train, val, easy, medium, hard)'
    )

    args = parser.parse_args()

    if args.all:
        analyze_all_datasets()
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        analyze_dataset(file_path)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python analyze_data_distribution.py training_data/train.parquet")
        print("  python analyze_data_distribution.py benchmark_data/hard_test.parquet")
        print("  python analyze_data_distribution.py --all")
        sys.exit(1)


if __name__ == '__main__':
    main()
