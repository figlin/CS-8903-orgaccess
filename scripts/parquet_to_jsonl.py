#!/usr/bin/env python3
"""
Convert Parquet files to JSONL format for OpenAI fine-tuning.
Preserves all data and formatting without loss.
"""

import pandas as pd
import json
import os
from pathlib import Path

def convert_parquet_to_jsonl(parquet_file, jsonl_file):
    """
    Convert a Parquet file to JSONL format.
    
    Args:
        parquet_file: Path to input Parquet file
        jsonl_file: Path to output JSONL file
    """
    print(f"Reading {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert to JSONL
    print(f"Converting to JSONL format...")
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            # Convert row to dictionary, handling NaN/None values
            row_dict = row.to_dict()
            
            # Convert NaN/None to None for JSON serialization
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = None
            
            # Write as single line JSON
            json_line = json.dumps(row_dict, ensure_ascii=False)
            f.write(json_line + '\n')
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} rows...")
    
    print(f"✓ Successfully converted to {jsonl_file}")
    print(f"  File size: {os.path.getsize(jsonl_file) / (1024*1024):.2f} MB")

def main():
    # Define input and output paths
    training_dir = Path("training_data")
    
    files_to_convert = [
        ("train.parquet", "train.jsonl"),
        ("validation.parquet", "validation.jsonl")
    ]
    
    for parquet_name, jsonl_name in files_to_convert:
        parquet_path = training_dir / parquet_name
        jsonl_path = training_dir / jsonl_name
        
        if parquet_path.exists():
            print(f"\n{'='*60}")
            convert_parquet_to_jsonl(parquet_path, jsonl_path)
        else:
            print(f"\n⚠ Warning: {parquet_path} not found, skipping...")
    
    print(f"\n{'='*60}")
    print("✓ Conversion complete!")
    print("\nGenerated files:")
    for _, jsonl_name in files_to_convert:
        jsonl_path = training_dir / jsonl_name
        if jsonl_path.exists():
            print(f"  - {jsonl_path}")

if __name__ == "__main__":
    main()
