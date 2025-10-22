#!/usr/bin/env python3
"""
Convert JSONL files to Parquet format for OrgAccess dataset.

This script converts synthetic JSONL files from data/addOn_Data/ into
parquet format matching the structure of the original dataset files.

Usage:
    # Convert all JSONL files in addOn_Data
    python scripts/convert_jsonl_to_parquet.py

    # Convert specific file
    python scripts/convert_jsonl_to_parquet.py \
        --input data/addOn_Data/hard_full_augmented-1000.jsonl \
        --output data/hard_augmented.parquet

    # Merge with existing parquet
    python scripts/convert_jsonl_to_parquet.py \
        --input data/addOn_Data/hard_full_augmented-1000.jsonl \
        --output data/hard-00000-of-00001.parquet \
        --merge
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import subprocess

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import pyarrow
        import pyarrow.parquet as pq
        return True
    except ImportError:
        print("❌ Error: pyarrow is required but not installed")
        print("\nInstall with:")
        print("  pip install pyarrow")
        print("  or")
        print("  pip3 install pyarrow")
        return False


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping line {line_num} due to JSON error: {e}")
                continue

    return data


def clean_data(data: List[Dict[str, Any]], target_schema=None) -> List[Dict[str, Any]]:
    """
    Clean and standardize data structure.

    Args:
        data: List of dictionaries
        target_schema: Optional PyArrow schema to match

    Returns:
        Cleaned data
    """
    import pyarrow as pa

    cleaned = []

    required_fields = {'user_role', 'permissions', 'query', 'expected_response'}

    # Get field names and types from target schema if provided
    if target_schema:
        schema_fields = {name: field.type for name, field in zip(target_schema.names, target_schema)}
    else:
        schema_fields = None

    for i, item in enumerate(data):
        # Check required fields
        missing = required_fields - set(item.keys())
        if missing:
            print(f"⚠️  Warning: Skipping item {i} - missing fields: {missing}")
            continue

        # If we have a target schema, use its fields
        if schema_fields:
            cleaned_item = {}
            for field_name, field_type in schema_fields.items():
                if field_name in item:
                    value = item[field_name]

                    # Convert permissions dict to JSON string if needed
                    if field_name == 'permissions' and isinstance(value, dict):
                        if pa.types.is_string(field_type):
                            # Target wants string, convert dict to JSON
                            value = json.dumps(value)
                        # Otherwise keep as dict

                    cleaned_item[field_name] = value
                else:
                    # Add missing fields as None
                    cleaned_item[field_name] = None
        else:
            # Keep only core fields (remove synthetic metadata)
            cleaned_item = {
                'user_role': item['user_role'],
                'permissions': item['permissions'],
                'query': item['query'],
                'expected_response': item['expected_response'],
            }

            # Include rationale if present
            if 'rationale' in item:
                cleaned_item['rationale'] = item['rationale']

        cleaned.append(cleaned_item)

    return cleaned


def convert_to_parquet(
    input_path: Path,
    output_path: Path,
    merge: bool = False
) -> None:
    """
    Convert JSONL to parquet format.

    Args:
        input_path: Input JSONL file
        output_path: Output parquet file
        merge: If True, merge with existing parquet file
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"Loading data from {input_path}...")
    data = load_jsonl(input_path)
    print(f"✓ Loaded {len(data):,} records")

    # Merge with existing if requested
    if merge and output_path.exists():
        print(f"Loading existing data from {output_path}...")
        existing_table = pq.read_table(output_path)
        print(f"✓ Existing data: {len(existing_table):,} records")

        print("Cleaning data to match existing schema...")
        data = clean_data(data, target_schema=existing_table.schema)
        print(f"✓ Cleaned to {len(data):,} valid records")

        if len(data) == 0:
            print("❌ No valid data to convert")
            return

        # Convert to PyArrow table using the existing schema
        print("Converting to PyArrow table with matching schema...")
        try:
            table = pa.Table.from_pylist(data, schema=existing_table.schema)
        except Exception as e:
            print(f"⚠️  Warning: Could not match existing schema exactly: {e}")
            print("Attempting to cast to existing schema...")
            table = pa.Table.from_pylist(data)

            # Try to cast to existing schema
            try:
                table = table.cast(existing_table.schema)
            except Exception as e2:
                print(f"❌ Error: Schema mismatch - {e2}")
                print(f"\nExisting schema:\n{existing_table.schema}")
                print(f"\nNew data schema:\n{table.schema}")
                raise

        # Concatenate tables
        print("Merging tables...")
        table = pa.concat_tables([existing_table, table])
        print(f"✓ Merged total: {len(table):,} records")
    else:
        print("Cleaning data...")
        data = clean_data(data)
        print(f"✓ Cleaned to {len(data):,} valid records")

        if len(data) == 0:
            print("❌ No valid data to convert")
            return

        # Convert to PyArrow table
        print("Converting to PyArrow table...")
        table = pa.Table.from_pylist(data)

    # Write parquet
    print(f"Writing to {output_path}...")
    pq.write_table(table, output_path)
    print(f"✓ Successfully wrote {len(table):,} records to {output_path}")

    # Show statistics
    print(f"\nStatistics:")
    print(f"  Total records: {len(table):,}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def convert_all_in_directory(
    input_dir: Path,
    output_dir: Path,
    difficulty: str = None,
    merge: bool = False
) -> None:
    """
    Convert all JSONL files in directory.

    Args:
        input_dir: Input directory containing JSONL files
        output_dir: Output directory for parquet files
        difficulty: Filter by difficulty (easy/medium/hard)
        merge: If True, merge all files into one output
    """
    jsonl_files = sorted(input_dir.glob("*.jsonl"))

    if difficulty:
        jsonl_files = [f for f in jsonl_files if difficulty in f.name.lower()]

    if not jsonl_files:
        print(f"❌ No JSONL files found in {input_dir}")
        if difficulty:
            print(f"   (filtered for difficulty: {difficulty})")
        return

    print(f"Found {len(jsonl_files)} JSONL file(s):")
    for f in jsonl_files:
        print(f"  - {f.name}")
    print()

    if merge:
        # Merge all into one file
        if difficulty:
            output_name = f"{difficulty}_augmented.parquet"
        else:
            output_name = "merged_augmented.parquet"

        output_path = output_dir / output_name

        # Load all data
        all_data = []
        for jsonl_file in jsonl_files:
            print(f"Loading {jsonl_file.name}...")
            data = load_jsonl(jsonl_file)
            all_data.extend(data)

        print(f"\n✓ Loaded total {len(all_data):,} records from {len(jsonl_files)} file(s)")

        # Clean
        print("Cleaning data...")
        all_data = clean_data(all_data)
        print(f"✓ Cleaned to {len(all_data):,} valid records")

        # Convert
        import pyarrow as pa
        import pyarrow.parquet as pq

        print("Converting to PyArrow table...")
        table = pa.Table.from_pylist(all_data)

        print(f"Writing to {output_path}...")
        pq.write_table(table, output_path)
        print(f"✓ Successfully wrote {len(table):,} records to {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    else:
        # Convert each file separately
        for jsonl_file in jsonl_files:
            # Generate output name
            output_name = jsonl_file.stem + ".parquet"
            output_path = output_dir / output_name

            print(f"\n{'='*60}")
            convert_to_parquet(jsonl_file, output_path, merge=False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL files to Parquet format for OrgAccess dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all JSONL files in addOn_Data/
  python scripts/convert_jsonl_to_parquet.py

  # Convert only hard difficulty files and merge them
  python scripts/convert_jsonl_to_parquet.py --difficulty hard --merge

  # Convert specific file
  python scripts/convert_jsonl_to_parquet.py \\
      --input data/addOn_Data/hard_full_augmented-1000.jsonl \\
      --output data/hard_augmented.parquet

  # Merge with existing parquet
  python scripts/convert_jsonl_to_parquet.py \\
      --input data/addOn_Data/hard_full_augmented-1000.jsonl \\
      --output data/hard-00000-of-00001.parquet \\
      --merge
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        help='Input JSONL file (if not specified, converts all in addOn_Data/)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output parquet file (required if --input is specified)'
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/addOn_Data'),
        help='Input directory containing JSONL files (default: data/addOn_Data)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data'),
        help='Output directory for parquet files (default: data/)'
    )

    parser.add_argument(
        '--difficulty',
        choices=['easy', 'medium', 'hard'],
        help='Filter by difficulty level'
    )

    parser.add_argument(
        '--merge',
        action='store_true',
        help='Merge with existing parquet file or merge all JSONL into one parquet'
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Single file conversion
    if args.input:
        if not args.output:
            print("❌ Error: --output is required when --input is specified")
            sys.exit(1)

        if not args.input.exists():
            print(f"❌ Error: Input file not found: {args.input}")
            sys.exit(1)

        # Create output directory if needed
        args.output.parent.mkdir(parents=True, exist_ok=True)

        convert_to_parquet(args.input, args.output, merge=args.merge)

    # Directory conversion
    else:
        if not args.input_dir.exists():
            print(f"❌ Error: Input directory not found: {args.input_dir}")
            sys.exit(1)

        # Create output directory if needed
        args.output_dir.mkdir(parents=True, exist_ok=True)

        convert_all_in_directory(
            args.input_dir,
            args.output_dir,
            difficulty=args.difficulty,
            merge=args.merge
        )


if __name__ == '__main__':
    main()
