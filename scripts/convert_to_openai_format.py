#!/usr/bin/env python3
"""
Convert training data to OpenAI fine-tuning format.
OpenAI requires a "messages" key with conversation format.
"""

import pandas as pd
import json
import os
from pathlib import Path

def create_openai_format(row):
    """
    Convert a data row to OpenAI's required messages format.
    
    Format:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
    """
    # Build system message with user role and permissions
    system_content = f"You are an access control assistant. The user has the following role and permissions:\n\nRole: {row['user_role']}\nPermissions: {row['permissions']}"
    
    # Build assistant response based on expected_response and rationale
    assistant_content = f"Access Decision: {row['expected_response']}\n\nRationale: {row['rationale']}"
    
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": row['query']},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def convert_file(input_file, output_file):
    """Convert a parquet or jsonl file to OpenAI format."""
    print(f"Reading {input_file}...")
    
    # Read input file
    if str(input_file).endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_json(input_file, lines=True)
    
    print(f"Total rows: {len(df)}")
    
    # Convert to OpenAI format
    print(f"Converting to OpenAI messages format...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            openai_row = create_openai_format(row)
            json_line = json.dumps(openai_row, ensure_ascii=False)
            f.write(json_line + '\n')
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} rows...")
    
    print(f"✓ Successfully converted to {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

def main():
    training_dir = Path("training_data")
    
    # Convert main training files
    files_to_convert = [
        (training_dir / "train.parquet", training_dir / "train_openai.jsonl"),
        (training_dir / "validation.parquet", training_dir / "validation_openai.jsonl"),
    ]
    
    # Convert addOn training files
    addon_dir = training_dir / "addOn_training_data"
    if addon_dir.exists():
        files_to_convert.extend([
            (addon_dir / "train.parquet", addon_dir / "train_openai.jsonl"),
            (addon_dir / "validation.parquet", addon_dir / "validation_openai.jsonl"),
        ])
    
    for input_path, output_path in files_to_convert:
        if input_path.exists():
            print(f"\n{'='*60}")
            convert_file(input_path, output_path)
        else:
            print(f"\n⚠ Warning: {input_path} not found, skipping...")
    
    print(f"\n{'='*60}")
    print("✓ Conversion complete!")
    print("\nGenerated OpenAI-compatible files:")
    for _, output_path in files_to_convert:
        if output_path.exists():
            print(f"  - {output_path}")
    
    # Show a sample
    sample_file = training_dir / "train_openai.jsonl"
    if sample_file.exists():
        print(f"\n{'='*60}")
        print("Sample output (first example):")
        with open(sample_file, 'r') as f:
            sample = json.loads(f.readline())
            print(json.dumps(sample, indent=2))

if __name__ == "__main__":
    main()
