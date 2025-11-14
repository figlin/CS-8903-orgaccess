#!/usr/bin/env python3
"""
Evaluate Fine-Tuned OpenAI Model on OrgAccess Benchmark

This script evaluates your fine-tuned GPT-4o-mini model on the benchmark datasets.

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY='your-api-key-here'
    
    # Evaluate on all test sets
    python scripts/evaluate_openai_finetuned.py --model ft:gpt-4o-mini-2024-07-18:your-org:model-name:id
    
    # Evaluate on specific test set
    python scripts/evaluate_openai_finetuned.py --model ft:gpt-4o-mini-2024-07-18:your-org:model-name:id --test-file benchmark_data/hard_test.parquet
    
    # Quick test with fewer samples
    python scripts/evaluate_openai_finetuned.py --model ft:gpt-4o-mini-2024-07-18:your-org:model-name:id --max-samples 100
"""

import argparse
import os
import warnings
from openai import OpenAI
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Suppress sklearn warnings about undefined metrics (e.g., when "full" class has no predictions)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def normalize_label(label):
    """
    Normalize label to full form (handle abbreviated and full forms).
    """
    if not label:
        return 'rejected'
    
    label_lower = str(label).lower().strip()
    
    # Map abbreviated forms
    if label_lower == 'r':
        return 'rejected'
    elif label_lower == 'p':
        return 'partial'
    elif label_lower == 'f':
        return 'full'
    
    # Already in full form
    if label_lower in ['rejected', 'partial', 'full']:
        return label_lower
    
    # Default to rejected for safety
    return 'rejected'


def extract_label(response_text):
    """
    Extract access control label from model response.
    """
    response_lower = response_text.lower()
    
    if 'rejected' in response_lower:
        return 'rejected'
    elif 'partial' in response_lower:
        return 'partial'
    elif 'full' in response_lower:
        return 'full'
    
    # Conservative default
    return 'rejected'


def evaluate_single_example(client, model_name, item, index, request_delay=0, save_full_responses=True):
    """
    Evaluate a single example.
    Returns: (index, prediction, ground_truth, error, full_details)
    
    Args:
        request_delay: Seconds to wait before making the request (for rate limiting)
        save_full_responses: Whether to save full model responses and details
    """
    # Add delay before request to smooth out rate limiting
    if request_delay > 0:
        time.sleep(request_delay)
    
    # Create messages in the same format as training
    messages = [
        {
            "role": "system",
            "content": f"You are an access control assistant. The user has the following role and permissions:\n\nRole: {item['user_role']}\nPermissions: {item['permissions']}"
        },
        {
            "role": "user",
            "content": item['query']
        }
    ]
    
    try:
        # Call OpenAI API with retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=512,
                    timeout=30
                )
                
                # Extract model response
                model_response = response.choices[0].message.content
                
                # Extract prediction
                prediction = extract_label(model_response)
                
                # Build full details if requested
                full_details = None
                if save_full_responses:
                    full_details = {
                        'user_role': item.get('user_role', 'unknown'),
                        'permissions': item.get('permissions', ''),
                        'query': item.get('query', ''),
                        'model_response': model_response,
                        'expected_rationale': item.get('rationale', '')
                    }
                
                return (index, prediction, normalize_label(item['expected_response']), None, full_details)
                
            except Exception as e:
                error_str = str(e).lower()
                if ('rate_limit' in error_str or 'rate limit' in error_str or 
                    'too many requests' in error_str or '429' in error_str):
                    if attempt < max_retries - 1:
                        # Exponential backoff for rate limits
                        wait_time = (2 ** attempt) * 5  # 5, 10, 20, 40, 80 seconds
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} retries")
                else:
                    # For other errors, retry with shorter wait
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                raise e
                
    except Exception as e:
        error_info = {
            'index': index,
            'error': str(e),
            'user_role': item.get('user_role', 'unknown')
        }
        # Default to rejected on error
        return (index, 'rejected', normalize_label(item['expected_response']), error_info, None)


def evaluate_model(client, model_name, test_dataset, max_samples=None, batch_workers=8, request_delay=0, save_full_responses=True):
    """
    Evaluate model on test dataset using OpenAI API with parallel processing.
    
    Args:
        batch_workers: Number of parallel workers (default: 8)
        request_delay: Seconds to delay between requests per worker (default: 0)
        save_full_responses: Whether to save full model responses and details (default: True)
    """
    predictions = []
    ground_truth = []
    errors = []
    detailed_predictions = []
    
    # Limit samples if requested
    samples = test_dataset if max_samples is None else test_dataset.select(range(min(max_samples, len(test_dataset))))
    
    print(f"\nEvaluating {len(samples)} examples...")
    print(f"Using {batch_workers} parallel workers")
    if request_delay > 0:
        print(f"Request delay: {request_delay}s per request")
        estimated_time = (len(samples) * request_delay) / batch_workers / 60
        print(f"Estimated time: {estimated_time:.1f} minutes")
    
    # Thread-safe collections
    results_lock = Lock()
    results = [None] * len(samples)
    
    # Process in parallel with progress bar
    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(evaluate_single_example, client, model_name, item, i, request_delay, save_full_responses): i
            for i, item in enumerate(samples)
        }
        
        # Collect results with progress bar
        with tqdm(total=len(samples), desc="Evaluating") as pbar:
            for future in as_completed(future_to_idx):
                idx, prediction, truth, error, full_details = future.result()
                
                with results_lock:
                    results[idx] = (prediction, truth, error, full_details)
                    if error:
                        errors.append(error)
                
                pbar.update(1)
    
    # Extract predictions, ground truth, and details in order
    for prediction, truth, error, full_details in results:
        predictions.append(prediction)
        ground_truth.append(truth)
        
        # Build detailed prediction entry
        detail_entry = {
            'index': len(detailed_predictions),
            'prediction': prediction,
            'ground_truth': truth,
            'correct': prediction == truth
        }
        if full_details:
            detail_entry['full_details'] = full_details
        
        detailed_predictions.append(detail_entry)
    
    if errors:
        print(f"\n⚠️  {len(errors)} errors occurred")
    
    return predictions, ground_truth, errors, detailed_predictions


def calculate_metrics(predictions, ground_truth, test_name="Test"):
    """Calculate and display evaluation metrics"""
    if not predictions or not ground_truth:
        print(f"⚠️  No predictions or ground truth data available")
        return None
    
    if len(predictions) != len(ground_truth):
        print(f"⚠️  Warning: predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) lengths don't match")
    
    # Normalize all labels
    predictions = [normalize_label(p) for p in predictions]
    ground_truth = [normalize_label(gt) for gt in ground_truth]
    
    label_map = {'full': 0, 'partial': 1, 'rejected': 2}
    
    # Convert to numeric labels
    y_true = [label_map[gt] for gt in ground_truth]
    y_pred = [label_map[pred] for pred in predictions]
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {test_name}")
    print(f"{'='*60}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"F1 (Macro):     {f1_macro:.4f}")
    print(f"F1 (Weighted):  {f1_weighted:.4f}")
    
    print(f"\n{'-'*60}")
    print("Per-Class Metrics:")
    print(f"{'-'*60}")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=['full', 'partial', 'rejected'],
        digits=4,
        zero_division=0
    ))
    
    print(f"{'-'*60}")
    print("Confusion Matrix:")
    print(f"{'-'*60}")
    print("                Predicted")
    print("                Full  Partial  Rejected")
    print(f"Actual Full      {cm[0][0]:4d}    {cm[0][1]:4d}      {cm[0][2]:4d}")
    print(f"      Partial    {cm[1][0]:4d}    {cm[1][1]:4d}      {cm[1][2]:4d}")
    print(f"      Rejected   {cm[2][0]:4d}    {cm[2][1]:4d}      {cm[2][2]:4d}")
    print(f"{'='*60}\n")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned OpenAI model on OrgAccess")
    parser.add_argument("--model", type=str, required=True,
                        help="Fine-tuned model ID (e.g., ft:gpt-4o-mini-2024-07-18:org:name:id)")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Specific test file (default: evaluate all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (default: auto-generated)")
    parser.add_argument("--batch-workers", type=int, default=8,
                        help="Number of parallel workers for concurrent requests (default: 8, range: 1-32)")
    parser.add_argument("--request-delay", type=float, default=0,
                        help="Delay in seconds between requests per worker to avoid rate limits (default: 0)")
    parser.add_argument("--save-full-responses", action='store_true', default=True,
                        help="Save full model responses and details (default: True)")
    parser.add_argument("--no-save-full-responses", dest='save_full_responses', action='store_false',
                        help="Don't save full responses (smaller output file)")
    
    args = parser.parse_args()
    
    # Validate batch workers
    if args.batch_workers < 1 or args.batch_workers > 32:
        print("⚠️  Warning: batch-workers should be between 1-32, using default of 8")
        args.batch_workers = 8
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print(f"\n{'='*60}")
    print(f"OrgAccess Fine-Tuned OpenAI Model Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Parallel workers: {args.batch_workers}")
    if args.request_delay > 0:
        print(f"Request delay: {args.request_delay}s")
    print(f"{'='*60}\n")
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Verify model exists
    try:
        print("Verifying model access...")
        client.models.retrieve(args.model)
        print("✓ Model access confirmed\n")
    except Exception as e:
        print(f"❌ Error accessing model: {e}")
        print("\nMake sure:")
        print("  1. Your API key is correct")
        print("  2. The model ID is correct")
        print("  3. You have access to the fine-tuned model")
        return
    
    # Determine test files
    if args.test_file:
        test_files = [(args.test_file, Path(args.test_file).stem)]
    else:
        test_files = [
            ("benchmark_data/easy_test.parquet", "easy_test"),
            ("benchmark_data/medium_test.parquet", "medium_test"),
            ("benchmark_data/hard_test.parquet", "hard_test")
        ]
    
    # Evaluate on each test set
    all_results = {}
    
    for test_file, test_name in test_files:
        print(f"\n{'='*60}")
        print(f"Loading {test_name}...")
        print(f"{'='*60}")
        
        try:
            test_dataset = load_dataset('parquet', data_files=test_file)['train']
            print(f"✓ Loaded {len(test_dataset)} examples")
            
            # Evaluate
            predictions, ground_truth, errors, detailed_predictions = evaluate_model(
                client, args.model, test_dataset, args.max_samples, args.batch_workers, 
                args.request_delay, args.save_full_responses
            )
            
            print(f"\n{'='*60}")
            print(f"Evaluation complete:")
            print(f"  Total predictions: {len(predictions)}")
            print(f"  Ground truth: {len(ground_truth)}")
            print(f"  Errors: {len(errors)}")
            print(f"{'='*60}")
            
            # Calculate metrics
            metrics = calculate_metrics(predictions, ground_truth, test_name)
            
            if metrics:
                all_results[test_name] = {
                    'file': test_file,
                    'total_samples': len(predictions),
                    'metrics': metrics,
                    'confusion_matrix': metrics['confusion_matrix'],
                    'errors': len(errors),
                    'error_details': errors[:10] if errors else [],  # Save first 10 errors
                    'predictions': detailed_predictions
                }
            else:
                print(f"❌ Failed to calculate metrics for {test_name}")
                all_results[test_name] = {
                    'file': test_file,
                    'total_samples': len(predictions),
                    'error': 'Failed to calculate metrics',
                    'errors': len(errors),
                    'predictions': detailed_predictions
                }
            
        except Exception as e:
            import traceback
            print(f"❌ Error processing {test_file}: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")
            continue
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract model name from model ID
        model_short = args.model.split(':')[-2] if ':' in args.model else 'openai'
        output_file = f"results_{model_short}_{timestamp}.json"
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'max_samples': args.max_samples,
        'results': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    # Print summary
    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}\n")
    
    for test_name, result in all_results.items():
        print(f"{test_name:20s}: Accuracy={result['metrics']['accuracy']:.4f}, "
              f"F1={result['metrics']['f1_macro']:.4f}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
