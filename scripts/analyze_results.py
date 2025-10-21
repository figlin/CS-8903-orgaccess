#!/usr/bin/env python3
"""
Results Analysis Script for OrgAccess Evaluation

Analyzes evaluation results JSON files and compares predictions against ground truth
from the benchmark datasets. Handles corrupted labels ('p', 'r') automatically.

Usage:
    python analyze_results.py results/llama-3.1-8b-finetuned/EASY-results_vllm_easy_20251018_133755.json
    python analyze_results.py results/llama-3.1-8b-finetuned/HARD-results_vllm_hard_20251018_132646.json
    python analyze_results.py --all results/llama-3.1-8b-finetuned/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from datasets import load_dataset


def normalize_label(label):
    """
    Normalize labels, handling corrupted labels in the dataset.

    Corrupted labels include:
    - Abbreviations: 'p' → 'partial', 'r' → 'rejected', 'f' → 'full'
    - Variations: 'allow' → 'full', 'deny' → 'rejected', 'approve' → 'full'
    - Typos: 'reject' → 'rejected'
    - Complex labels containing multiple operations → map based on primary action
    """
    label_map = {
        # Correct labels
        'full': 'full',
        'partial': 'partial',
        'rejected': 'rejected',

        # Single-letter abbreviations (corrupted)
        'p': 'partial',
        'r': 'rejected',
        'f': 'full',

        # Alternative terminology (corrupted)
        'allow': 'full',
        'approve': 'full',
        'deny': 'rejected',
        'reject': 'rejected',
        'complete': 'full',

        # Partial variations
        'approve_partial': 'partial',
    }

    normalized = str(label).lower().strip()

    # Direct mapping
    if normalized in label_map:
        return label_map[normalized]

    # Handle complex/compound labels (e.g., "approve_budget_and_generate_report")
    # Map based on the primary action keyword
    if any(keyword in normalized for keyword in ['approve', 'allow', 'complete']):
        return 'full'
    elif 'reject' in normalized or 'deny' in normalized:
        return 'rejected'
    elif 'partial' in normalized:
        return 'partial'

    # If no mapping found, return as-is (will cause an error that we can catch)
    return normalized


def load_benchmark_data(difficulty):
    """Load ground truth from benchmark data."""
    benchmark_files = {
        'easy': 'benchmark_data/easy_test.parquet',
        'medium': 'benchmark_data/medium_test.parquet',
        'hard': 'benchmark_data/hard_test.parquet',
    }

    if difficulty.lower() not in benchmark_files:
        raise ValueError(f"Invalid difficulty: {difficulty}. Must be one of: easy, medium, hard")

    file_path = benchmark_files[difficulty.lower()]
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    ds = load_dataset('parquet', data_files=file_path)['train']

    # Extract ground truth labels
    ground_truth = []
    for example in ds:
        if 'expected_response' in example:
            label = normalize_label(example['expected_response'])
            ground_truth.append(label)
        else:
            raise ValueError("'expected_response' field not found in benchmark data")

    return ground_truth


def extract_predictions_from_results(results_data):
    """
    Extract predictions and ground truth from results JSON.

    Results JSON structure can be either:
    1. New format:
    {
        "results": {
            "<difficulty>": {
                "predictions": [
                    {
                        "prediction": "full",
                        "ground_truth": "partial",
                        ...
                    },
                    ...
                ]
            }
        }
    }
    2. Old format:
    {
        "results": {
            "<difficulty>": {
                "details": [...]
            }
        }
    }
    """
    predictions = []
    ground_truth = []

    # Navigate the JSON structure
    if 'results' in results_data:
        results = results_data['results']

        # Find the difficulty level (easy, medium, or hard)
        for difficulty_key in ['easy', 'medium', 'hard']:
            if difficulty_key in results:
                # Try both 'predictions' and 'details' keys
                items = results[difficulty_key].get('predictions',
                                                     results[difficulty_key].get('details', []))

                for item in items:
                    pred = normalize_label(item.get('prediction', ''))
                    gt = normalize_label(item.get('ground_truth', ''))

                    predictions.append(pred)
                    ground_truth.append(gt)

                return predictions, ground_truth, difficulty_key

    raise ValueError("Could not extract predictions from results JSON. Invalid format.")


def calculate_metrics(predictions, ground_truth):
    """Calculate evaluation metrics with corrupted label handling."""

    # Label mapping for sklearn
    label_to_int = {
        'full': 0,
        'partial': 1,
        'rejected': 2,
    }

    # Check for unmapped labels
    valid_labels = set(label_to_int.keys())
    invalid_gt = [gt for gt in ground_truth if gt not in valid_labels]
    invalid_pred = [pred for pred in predictions if pred not in valid_labels]

    if invalid_gt:
        from collections import Counter
        invalid_counter = Counter(invalid_gt)
        print(f"\n⚠️  ERROR: Found {len(invalid_gt)} invalid ground truth labels:")
        for label, count in invalid_counter.most_common(10):
            print(f"   '{label}': {count} occurrences")
        raise ValueError(f"Invalid ground truth labels found. Please update normalize_label() function.")

    if invalid_pred:
        from collections import Counter
        invalid_counter = Counter(invalid_pred)
        print(f"\n⚠️  ERROR: Found {len(invalid_pred)} invalid prediction labels:")
        for label, count in invalid_counter.most_common(10):
            print(f"   '{label}': {count} occurrences")
        raise ValueError(f"Invalid prediction labels found.")

    # Convert to integers
    y_true = [label_to_int[gt] for gt in ground_truth]
    y_pred = [label_to_int[pred] for pred in predictions]

    # Track ALL corrupted labels (not just p/r/f)
    clean_labels = {'full', 'partial', 'rejected'}
    corrupted_in_ground_truth = sum(1 for gt in ground_truth if gt in clean_labels)
    corrupted_in_ground_truth = len(ground_truth) - corrupted_in_ground_truth

    corrupted_in_predictions = sum(1 for pred in predictions if pred in clean_labels)
    corrupted_in_predictions = len(predictions) - corrupted_in_predictions

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Per-class metrics
    class_report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=['full', 'partial', 'rejected'],
        digits=4,
        zero_division=0,
        output_dict=True
    )

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'corrupted_in_ground_truth': corrupted_in_ground_truth,
        'corrupted_in_predictions': corrupted_in_predictions,
        'total_samples': len(predictions),
    }


def print_results(results_file, metrics, difficulty):
    """Print formatted evaluation results."""

    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {results_file.name}")
    print("=" * 80)
    print(f"Difficulty Level: {difficulty.upper()}")
    print(f"Total Samples:    {metrics['total_samples']:,}")

    if metrics['corrupted_in_ground_truth'] > 0:
        print(f"\n⚠️  Corrupted labels found in ground truth: {metrics['corrupted_in_ground_truth']:,} "
              f"({metrics['corrupted_in_ground_truth']/metrics['total_samples']*100:.2f}%)")

    if metrics['corrupted_in_predictions'] > 0:
        print(f"⚠️  Corrupted labels found in predictions: {metrics['corrupted_in_predictions']:,} "
              f"({metrics['corrupted_in_predictions']/metrics['total_samples']*100:.2f}%)")

    print("\n" + "-" * 80)
    print("OVERALL METRICS")
    print("-" * 80)
    print(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"F1 (Macro):        {metrics['f1_macro']:.4f}")
    print(f"F1 (Weighted):     {metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):    {metrics['recall_macro']:.4f}")

    print("\n" + "-" * 80)
    print("PER-CLASS METRICS")
    print("-" * 80)

    report = metrics['classification_report']

    print(f"\n{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)

    for class_name in ['full', 'partial', 'rejected']:
        class_metrics = report[class_name]
        print(f"{class_name:<12} "
              f"{class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} "
              f"{class_metrics['f1-score']:<12.4f} "
              f"{int(class_metrics['support']):<10,}")

    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)

    cm = metrics['confusion_matrix']

    print("\n                    Predicted")
    print("                Full      Partial   Rejected")
    print(f"Actual Full      {cm[0][0]:6,}    {cm[0][1]:6,}    {cm[0][2]:6,}")
    print(f"      Partial    {cm[1][0]:6,}    {cm[1][1]:6,}    {cm[1][2]:6,}")
    print(f"      Rejected   {cm[2][0]:6,}    {cm[2][1]:6,}    {cm[2][2]:6,}")

    # Calculate per-class accuracy from confusion matrix
    print("\n" + "-" * 80)
    print("PER-CLASS ACCURACY")
    print("-" * 80)

    for i, class_name in enumerate(['Full', 'Partial', 'Rejected']):
        total_actual = cm[i].sum()
        correct = cm[i][i]
        class_acc = correct / total_actual if total_actual > 0 else 0
        print(f"{class_name:<10} {correct:6,} / {total_actual:6,} correct  ({class_acc*100:6.2f}%)")

    print("\n" + "=" * 80)


def analyze_single_file(results_file):
    """Analyze a single results JSON file."""

    # Load results JSON
    with open(results_file, 'r') as f:
        results_data = json.load(f)

    # Extract predictions and ground truth from results
    predictions, ground_truth_from_results, difficulty = extract_predictions_from_results(results_data)

    print(f"\nLoaded {len(predictions):,} predictions from {results_file}")
    print(f"Detected difficulty level: {difficulty}")

    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth_from_results)

    # Print results
    print_results(results_file, metrics, difficulty)

    return metrics


def analyze_directory(results_dir):
    """Analyze all result files in a directory."""

    results_dir = Path(results_dir)

    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        return

    # Find all JSON files
    json_files = list(results_dir.glob("*results*.json"))

    if not json_files:
        print(f"No results files found in {results_dir}")
        return

    print(f"\nFound {len(json_files)} result files in {results_dir}")

    all_metrics = {}

    for json_file in sorted(json_files):
        try:
            metrics = analyze_single_file(json_file)
            all_metrics[json_file.name] = metrics
        except Exception as e:
            print(f"\n⚠️  Error analyzing {json_file.name}: {e}")

    # Print summary comparison
    if len(all_metrics) > 1:
        print("\n\n" + "=" * 80)
        print("SUMMARY COMPARISON")
        print("=" * 80)

        print(f"\n{'File':<50} {'Accuracy':<12} {'F1 Macro':<12} {'F1 Weighted':<12}")
        print("-" * 80)

        for filename, metrics in sorted(all_metrics.items()):
            print(f"{filename[:48]:<50} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['f1_macro']:<12.4f} "
                  f"{metrics['f1_weighted']:<12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze OrgAccess evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'path',
        help='Path to results JSON file or directory containing results'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Analyze all JSON files in the directory'
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path not found: {path}")
        sys.exit(1)

    if path.is_dir() or args.all:
        analyze_directory(path)
    else:
        analyze_single_file(path)


if __name__ == '__main__':
    main()
