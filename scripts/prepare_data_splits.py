"""
Data Splitting Script for OrgAccess Fine-Tuning

This script creates the training and benchmark datasets according to the research plan:
- Training: 70% of Easy + Medium, split 80/20 for train/val
- Benchmark: 30% of Easy + Medium + 100% of Hard

Directory structure created:
    training_data/
        train.parquet       - 80% of 70% Easy+Medium (for fine-tuning)
        validation.parquet  - 20% of 70% Easy+Medium (for monitoring)

    benchmark_data/
        easy_test.parquet   - 30% of Easy (holdout evaluation)
        medium_test.parquet - 30% of Medium (holdout evaluation)
        hard_test.parquet   - 100% of Hard (main evaluation target)

Usage:
    python prepare_data_splits.py
"""

import os
from datasets import load_dataset, concatenate_datasets, Dataset
import json
from datetime import datetime


def create_directories():
    """Create directory structure for training and benchmark data."""
    os.makedirs('training_data', exist_ok=True)
    os.makedirs('benchmark_data', exist_ok=True)
    print("✓ Created directories: training_data/ and benchmark_data/")


def load_original_data():
    """Load original parquet files."""
    print("\nLoading original datasets...")

    ds = load_dataset('parquet', data_files={
        'easy': 'data/easy-00000-of-00001.parquet',
        'medium': 'data/medium-00000-of-00001.parquet',
        'hard': 'data/hard-00000-of-00001.parquet'
    })

    print(f"✓ Easy:   {len(ds['easy']):,} examples")
    print(f"✓ Medium: {len(ds['medium']):,} examples")
    print(f"✓ Hard:   {len(ds['hard']):,} examples")

    return ds


def split_data(ds, seed=42):
    """
    Split data according to the research plan.

    Args:
        ds: Dataset dict with 'easy', 'medium', 'hard' splits
        seed: Random seed for reproducibility

    Returns:
        dict: Contains train, validation, and test datasets
    """
    print(f"\n{'='*60}")
    print("Creating Data Splits (seed={})".format(seed))
    print(f"{'='*60}")

    # ========== EASY SPLIT ==========
    print("\n[1/3] Processing Easy dataset...")
    easy_shuffled = ds['easy'].shuffle(seed=seed)
    easy_70_idx = int(len(easy_shuffled) * 0.70)

    easy_train_val = easy_shuffled.select(range(easy_70_idx))
    easy_test = easy_shuffled.select(range(easy_70_idx, len(easy_shuffled)))

    print(f"  Easy 70% (train+val): {len(easy_train_val):,} examples")
    print(f"  Easy 30% (test):      {len(easy_test):,} examples")

    # ========== MEDIUM SPLIT ==========
    print("\n[2/3] Processing Medium dataset...")
    medium_shuffled = ds['medium'].shuffle(seed=seed)
    medium_70_idx = int(len(medium_shuffled) * 0.70)

    medium_train_val = medium_shuffled.select(range(medium_70_idx))
    medium_test = medium_shuffled.select(range(medium_70_idx, len(medium_shuffled)))

    print(f"  Medium 70% (train+val): {len(medium_train_val):,} examples")
    print(f"  Medium 30% (test):      {len(medium_test):,} examples")

    # ========== COMBINE EASY + MEDIUM 70% ==========
    print("\n[3/3] Combining Easy + Medium for training...")
    train_val_combined = concatenate_datasets([easy_train_val, medium_train_val])
    train_val_combined = train_val_combined.shuffle(seed=seed)

    print(f"  Combined pool: {len(train_val_combined):,} examples")

    # ========== SPLIT 80/20 FOR TRAIN/VAL ==========
    train_idx = int(len(train_val_combined) * 0.80)

    train_dataset = train_val_combined.select(range(train_idx))
    val_dataset = train_val_combined.select(range(train_idx, len(train_val_combined)))

    print(f"\n  Training set (80%):   {len(train_dataset):,} examples")
    print(f"  Validation set (20%): {len(val_dataset):,} examples")

    # ========== HARD (100% for testing) ==========
    print("\n[Hard] Using 100% for testing...")
    hard_test = ds['hard']
    print(f"  Hard test: {len(hard_test):,} examples")

    return {
        'train': train_dataset,
        'validation': val_dataset,
        'easy_test': easy_test,
        'medium_test': medium_test,
        'hard_test': hard_test
    }


def analyze_label_distribution(dataset, name):
    """Analyze distribution of labels in a dataset."""
    labels = [item['expected_response'] for item in dataset]

    from collections import Counter
    counts = Counter(labels)
    total = len(labels)

    return {
        'name': name,
        'total': total,
        'full': counts.get('full', 0),
        'partial': counts.get('partial', 0),
        'rejected': counts.get('rejected', 0),
        'full_pct': counts.get('full', 0) / total * 100,
        'partial_pct': counts.get('partial', 0) / total * 100,
        'rejected_pct': counts.get('rejected', 0) / total * 100
    }


def save_datasets(splits):
    """Save datasets to parquet files."""
    print(f"\n{'='*60}")
    print("Saving Datasets")
    print(f"{'='*60}")

    # Save training data
    print("\nSaving training data...")
    splits['train'].to_parquet('training_data/train.parquet')
    print(f"  ✓ training_data/train.parquet ({len(splits['train']):,} examples)")

    splits['validation'].to_parquet('training_data/validation.parquet')
    print(f"  ✓ training_data/validation.parquet ({len(splits['validation']):,} examples)")

    # Save benchmark data
    print("\nSaving benchmark data...")
    splits['easy_test'].to_parquet('benchmark_data/easy_test.parquet')
    print(f"  ✓ benchmark_data/easy_test.parquet ({len(splits['easy_test']):,} examples)")

    splits['medium_test'].to_parquet('benchmark_data/medium_test.parquet')
    print(f"  ✓ benchmark_data/medium_test.parquet ({len(splits['medium_test']):,} examples)")

    splits['hard_test'].to_parquet('benchmark_data/hard_test.parquet')
    print(f"  ✓ benchmark_data/hard_test.parquet ({len(splits['hard_test']):,} examples)")


def create_summary_stats(splits):
    """Create and save summary statistics."""
    print(f"\n{'='*60}")
    print("Analyzing Label Distribution")
    print(f"{'='*60}")

    stats = []

    # Analyze each split
    for split_name in ['train', 'validation', 'easy_test', 'medium_test', 'hard_test']:
        stat = analyze_label_distribution(splits[split_name], split_name)
        stats.append(stat)

        print(f"\n{stat['name'].upper()}:")
        print(f"  Total:    {stat['total']:,}")
        print(f"  Full:     {stat['full']:,} ({stat['full_pct']:.1f}%)")
        print(f"  Partial:  {stat['partial']:,} ({stat['partial_pct']:.1f}%)")
        print(f"  Rejected: {stat['rejected']:,} ({stat['rejected_pct']:.1f}%)")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'seed': 42,
        'split_strategy': {
            'training': '70% of Easy + Medium, split 80/20 for train/val',
            'benchmark': '30% of Easy + Medium + 100% Hard'
        },
        'datasets': {
            'training_data/train.parquet': len(splits['train']),
            'training_data/validation.parquet': len(splits['validation']),
            'benchmark_data/easy_test.parquet': len(splits['easy_test']),
            'benchmark_data/medium_test.parquet': len(splits['medium_test']),
            'benchmark_data/hard_test.parquet': len(splits['hard_test'])
        },
        'label_distributions': stats
    }

    with open('data_split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to data_split_summary.json")

    return summary


def print_final_summary(summary):
    """Print final summary table."""
    print(f"\n{'='*60}")
    print("FINAL DATA SPLIT SUMMARY")
    print(f"{'='*60}")

    print("\nTRAINING DATA:")
    print(f"  train.parquet:      {summary['datasets']['training_data/train.parquet']:,} examples")
    print(f"  validation.parquet: {summary['datasets']['training_data/validation.parquet']:,} examples")
    print(f"  TOTAL:              {summary['datasets']['training_data/train.parquet'] + summary['datasets']['training_data/validation.parquet']:,} examples")

    print("\nBENCHMARK DATA:")
    print(f"  easy_test.parquet:   {summary['datasets']['benchmark_data/easy_test.parquet']:,} examples")
    print(f"  medium_test.parquet: {summary['datasets']['benchmark_data/medium_test.parquet']:,} examples")
    print(f"  hard_test.parquet:   {summary['datasets']['benchmark_data/hard_test.parquet']:,} examples")
    print(f"  TOTAL:               {summary['datasets']['benchmark_data/easy_test.parquet'] + summary['datasets']['benchmark_data/medium_test.parquet'] + summary['datasets']['benchmark_data/hard_test.parquet']:,} examples")

    print("\nDIRECTORY STRUCTURE:")
    print("  training_data/")
    print("    ├── train.parquet")
    print("    └── validation.parquet")
    print("  benchmark_data/")
    print("    ├── easy_test.parquet")
    print("    ├── medium_test.parquet")
    print("    └── hard_test.parquet")

    print(f"\n{'='*60}")
    print("✓ Data preparation complete!")
    print(f"{'='*60}\n")


def main():
    """Main execution function."""
    print(f"\n{'='*60}")
    print("OrgAccess Data Preparation")
    print(f"{'='*60}")

    # Step 1: Create directories
    create_directories()

    # Step 2: Load original data
    ds = load_original_data()

    # Step 3: Create splits
    splits = split_data(ds, seed=42)

    # Step 4: Save datasets
    save_datasets(splits)

    # Step 5: Create and save summary statistics
    summary = create_summary_stats(splits)

    # Step 6: Print final summary
    print_final_summary(summary)

    print("Next steps:")
    print("  1. Review data_split_summary.json for detailed statistics")
    print("  2. Use training_data/ for fine-tuning")
    print("  3. Use benchmark_data/ for evaluation")
    print("  4. Run: python train_llama.py (once created)\n")


if __name__ == "__main__":
    main()
