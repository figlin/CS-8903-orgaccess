#!/usr/bin/env python3
"""
Fix corrupted labels in parquet files.

This script normalizes the 'expected_response' field to standard values:
- full, partial, rejected

Corrupted labels are mapped:
- reject → rejected
- approve → full
- allow → full
- deny → rejected
- approve_budget* → full (compound labels)
- etc.

Usage:
    # Fix labels and save to new file
    python scripts/fix_parquet_labels.py \
        --input training_data/addOn_training_data/train.parquet \
        --output training_data/addOn_training_data/train_fixed.parquet

    # Fix in-place (overwrites original - backs up first)
    python scripts/fix_parquet_labels.py \
        --input training_data/addOn_training_data/train.parquet \
        --in-place

    # Dry run (show what would be changed)
    python scripts/fix_parquet_labels.py \
        --input training_data/addOn_training_data/train.parquet \
        --dry-run
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter
from typing import Dict
import pyarrow.parquet as pq
import pyarrow as pa


# Label normalization mapping
LABEL_MAPPING = {
    # Already correct
    'full': 'full',
    'partial': 'partial',
    'rejected': 'rejected',

    # Common corruptions
    'reject': 'rejected',
    'approve': 'full',
    'allow': 'full',
    'deny': 'rejected',

    # Additional variations
    'f': 'full',
    'p': 'partial',
    'r': 'rejected',
    'accept': 'full',
    'complete': 'full',
    'grant': 'full',
    'permitted': 'full',
    'allowed': 'full',
    'approved': 'full',
    'denied': 'rejected',
    'refuse': 'rejected',
    'block': 'rejected',
    'blocked': 'rejected',

    # Edge cases from actual data
    'execute': 'full',
    'authorize': 'full',
    'summary_only': 'partial',
    'restrict': 'rejected',
    'limited': 'partial',
    'require_approval': 'partial',
}


def normalize_label(label: str) -> str:
    """
    Normalize a label to standard form.

    Args:
        label: Original label string

    Returns:
        Normalized label (full/partial/rejected)
    """
    if not label or not isinstance(label, str):
        return 'rejected'  # Default for invalid labels

    label_lower = label.lower().strip()

    # Direct mapping
    if label_lower in LABEL_MAPPING:
        return LABEL_MAPPING[label_lower]

    # Compound labels (e.g., "approve_budget_and_generate_report")
    # Check for mixed signals (both approve and reject keywords)
    has_approve = any(keyword in label_lower for keyword in ['approve', 'allow', 'accept', 'grant', 'permit', 'authorize', 'execute'])
    has_reject = any(keyword in label_lower for keyword in ['deny', 'reject', 'refuse', 'block', 'restrict'])
    has_partial = 'partial' in label_lower or 'limited' in label_lower or 'only' in label_lower

    # Mixed signals or "only" → partial
    if (has_approve and has_reject) or has_partial:
        return 'partial'

    # Pure approve → full
    if has_approve:
        return 'full'

    # Pure reject → rejected
    if has_reject:
        return 'rejected'

    # Check for keywords in text (handles spaces)
    if any(keyword in label_lower for keyword in ['provide', 'deploy', 'summary']):
        return 'partial'

    # Default: return as-is and warn
    print(f"⚠️  Warning: Unknown label '{label}' - defaulting to 'partial'")
    return 'partial'  # Safer default than keeping unknown labels


def analyze_labels(table: pa.Table) -> Dict[str, int]:
    """
    Analyze label distribution in the table.

    Args:
        table: PyArrow table

    Returns:
        Counter of label frequencies
    """
    labels = table['expected_response'].to_pylist()
    return Counter(labels)


def fix_labels(table: pa.Table, dry_run: bool = False, conservative: bool = False) -> pa.Table:
    """
    Fix corrupted labels in the table.

    Args:
        table: Input PyArrow table
        dry_run: If True, don't modify, just show what would change
        conservative: If True, only fix simple single-word corruptions

    Returns:
        Table with fixed labels
    """
    # Get original labels
    original_labels = table['expected_response'].to_pylist()

    # Normalize labels
    if conservative:
        # Only fix simple single-word mappings
        normalized_labels = []
        for label in original_labels:
            if not label or not isinstance(label, str):
                normalized_labels.append('rejected')
            else:
                label_lower = label.lower().strip()
                # Only map if it's a direct single-word mapping
                if label_lower in LABEL_MAPPING:
                    normalized_labels.append(LABEL_MAPPING[label_lower])
                else:
                    # Keep original for compound labels
                    normalized_labels.append(label)
    else:
        normalized_labels = [normalize_label(label) for label in original_labels]

    # Count changes
    changes = Counter()
    for orig, norm in zip(original_labels, normalized_labels):
        if orig != norm:
            changes[f"{orig} → {norm}"] += 1

    # Report changes
    if changes:
        print(f"\n{'DRY RUN - ' if dry_run else ''}Changes to be made:")
        for change, count in sorted(changes.items(), key=lambda x: -x[1]):
            print(f"  {change}: {count:,} records")
        print(f"\nTotal records to be modified: {sum(changes.values()):,}")
    else:
        print("\n✓ No label corrections needed - all labels are valid!")

    if dry_run:
        return table

    # Create new column with normalized labels
    new_column = pa.array(normalized_labels, type=pa.string())

    # Replace the column
    col_index = table.schema.get_field_index('expected_response')
    new_table = table.set_column(col_index, 'expected_response', new_column)

    return new_table


def main():
    parser = argparse.ArgumentParser(
        description="Fix corrupted labels in OrgAccess parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix and save to new file
  python scripts/fix_parquet_labels.py \\
      --input training_data/addOn_training_data/train.parquet \\
      --output training_data/addOn_training_data/train_fixed.parquet

  # Fix in-place (creates backup first)
  python scripts/fix_parquet_labels.py \\
      --input training_data/addOn_training_data/train.parquet \\
      --in-place

  # Dry run to see what would change
  python scripts/fix_parquet_labels.py \\
      --input training_data/addOn_training_data/train.parquet \\
      --dry-run
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input parquet file'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output parquet file (required unless --in-place or --dry-run)'
    )

    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify the file in-place (creates .backup first)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup when using --in-place (dangerous!)'
    )

    parser.add_argument(
        '--conservative',
        action='store_true',
        help='Only fix simple single-word corruptions (reject→rejected, approve→full, etc), leave compound labels unchanged'
    )

    args = parser.parse_args()

    # Validation
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    if not args.dry_run and not args.in_place and not args.output:
        print("❌ Error: Either --output, --in-place, or --dry-run must be specified")
        return 1

    # Read input
    print(f"Loading {args.input}...")
    table = pq.read_table(args.input)
    print(f"✓ Loaded {len(table):,} records")

    # Analyze original labels
    print("\nOriginal label distribution:")
    original_dist = analyze_labels(table)
    total = sum(original_dist.values())
    for label, count in sorted(original_dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {label:15s} {count:8,}  {pct:6.2f}%")

    # Fix labels
    fixed_table = fix_labels(table, dry_run=args.dry_run, conservative=args.conservative)

    if args.dry_run:
        print("\n✓ Dry run complete - no files modified")
        return 0

    # Analyze fixed labels
    print("\nFixed label distribution:")
    fixed_dist = analyze_labels(fixed_table)
    total = sum(fixed_dist.values())
    for label, count in sorted(fixed_dist.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {label:15s} {count:8,}  {pct:6.2f}%")

    # Determine output path
    if args.in_place:
        output_path = args.input

        # Create backup unless disabled
        if not args.no_backup:
            backup_path = args.input.with_suffix('.parquet.backup')
            print(f"\nCreating backup: {backup_path}")
            shutil.copy2(args.input, backup_path)
            print(f"✓ Backup created")
    else:
        output_path = args.output

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    print(f"\nWriting to {output_path}...")
    pq.write_table(fixed_table, output_path)
    print(f"✓ Successfully wrote {len(fixed_table):,} records")

    # Show file sizes
    input_size = args.input.stat().st_size / 1024 / 1024
    output_size = output_path.stat().st_size / 1024 / 1024
    print(f"\nFile size: {output_size:.2f} MB")
    if args.in_place:
        print(f"Backup size: {input_size:.2f} MB")

    print("\n✓ Done!")
    return 0


if __name__ == '__main__':
    exit(main())
