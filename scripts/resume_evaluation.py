#!/usr/bin/env python3
"""
Resume or analyze interrupted OpenAI evaluation.
Check if results were saved and provide recovery options.
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset

def analyze_results(results_file):
    """Analyze partial results file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Analysis of: {results_file}")
    print(f"{'='*60}")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"Model: {data.get('model', 'N/A')}")
    print(f"Max samples: {data.get('max_samples', 'All')}")
    
    results = data.get('results', {})
    if not results:
        print("\n❌ No results found in file")
        return False
    
    print(f"\nTest sets evaluated: {len(results)}")
    for test_name, result in results.items():
        print(f"\n{'-'*60}")
        print(f"Test: {test_name}")
        print(f"  File: {result.get('file', 'N/A')}")
        print(f"  Total samples: {result.get('total_samples', 0)}")
        print(f"  Errors: {result.get('errors', 0)}")
        
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  F1 Macro: {metrics.get('f1_macro', 0):.4f}")
        else:
            print(f"  ⚠️  No metrics calculated")
    
    return True

def check_dataset_integrity(parquet_file):
    """Check if parquet file is valid"""
    print(f"\n{'='*60}")
    print(f"Checking dataset: {parquet_file}")
    print(f"{'='*60}")
    
    try:
        dataset = load_dataset('parquet', data_files=parquet_file)['train']
        print(f"✓ Dataset loaded successfully")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Columns: {dataset.column_names}")
        
        # Check first few examples
        print(f"\n  Sample data:")
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"    Example {i}: has_query={bool(item.get('query'))}, "
                  f"has_expected={bool(item.get('expected_response'))}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Analyze or resume OpenAI evaluation")
    parser.add_argument("--results", type=str, 
                        help="Results file to analyze")
    parser.add_argument("--check-dataset", type=str,
                        help="Parquet file to check integrity")
    parser.add_argument("--list-results", action='store_true',
                        help="List all result files")
    
    args = parser.parse_args()
    
    if args.list_results:
        print("\nAvailable result files:")
        for f in sorted(Path('.').glob('results_*.json')):
            size = f.stat().st_size
            print(f"  {f.name} ({size} bytes)")
    
    elif args.results:
        if Path(args.results).exists():
            analyze_results(args.results)
        else:
            print(f"❌ File not found: {args.results}")
    
    elif args.check_dataset:
        if Path(args.check_dataset).exists():
            check_dataset_integrity(args.check_dataset)
        else:
            print(f"❌ File not found: {args.check_dataset}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
