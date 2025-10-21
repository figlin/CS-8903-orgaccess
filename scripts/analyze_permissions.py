#!/usr/bin/env python3
"""
Permission and Role Analysis Script for OrgAccess Dataset

Analyzes user roles and permission distributions across the dataset.
Extracts all permission types and their values, providing comprehensive statistics.

Usage:
    python analyze_permissions.py data/hard-00000-of-00001.parquet
    python analyze_permissions.py training_data/train.parquet
    python analyze_permissions.py --all  # Analyze all datasets
"""

import argparse
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset


def safe_json_parse(json_str):
    """Safely parse JSON string, handling various formats."""
    if not json_str:
        return {}

    # If it's already a dict, return it
    if isinstance(json_str, dict):
        return json_str

    # If it's a string, try to parse it
    if isinstance(json_str, str):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try replacing single quotes with double quotes
            try:
                return json.loads(json_str.replace("'", '"'))
            except:
                return {}

    return {}


def analyze_permissions_and_roles(file_path):
    """
    Analyze user roles and all permission attributes in the dataset.

    Returns detailed statistics about:
    - User role distribution
    - Permission field types and their values
    - Frequency of each permission value
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*80}")

    # Load dataset
    ds = load_dataset('parquet', data_files=str(file_path))['train']
    total_samples = len(ds)

    print(f"\nTotal samples: {total_samples:,}\n")

    # Counters
    role_counter = Counter()
    permission_fields = defaultdict(Counter)  # {field_name: Counter({value: count})}
    permission_field_types = {}  # {field_name: set of types}

    # Analyze each example
    for idx, example in enumerate(ds):
        # Extract user role
        if 'user_role' in example:
            role = example['user_role']
            role_counter[role] += 1

        # Extract permissions
        permissions = None
        if 'permissions' in example:
            permissions = safe_json_parse(example['permissions'])
        elif 'full_details' in example and 'permissions' in example['full_details']:
            permissions = safe_json_parse(example['full_details']['permissions'])

        # Analyze permission fields
        if permissions and isinstance(permissions, dict):
            for field_name, field_value in permissions.items():
                # Track value
                if isinstance(field_value, list):
                    # For lists, count each item
                    for item in field_value:
                        permission_fields[field_name][str(item)] += 1
                    # Also track the list length
                    permission_fields[f'{field_name}_count'][len(field_value)] += 1
                elif isinstance(field_value, bool):
                    permission_fields[field_name][str(field_value)] += 1
                elif isinstance(field_value, (int, float)):
                    permission_fields[field_name][field_value] += 1
                else:
                    permission_fields[field_name][str(field_value)] += 1

                # Track type
                if field_name not in permission_field_types:
                    permission_field_types[field_name] = set()
                permission_field_types[field_name].add(type(field_value).__name__)

    # ========================================
    # 1. USER ROLE DISTRIBUTION
    # ========================================
    print("━" * 80)
    print("USER ROLE DISTRIBUTION")
    print("━" * 80)

    sorted_roles = sorted(role_counter.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTotal unique roles: {len(sorted_roles)}")
    print(f"\n{'Role':<50} {'Count':<10} {'Percentage':<12}")
    print("-" * 80)

    for role, count in sorted_roles[:30]:  # Top 30 roles
        percentage = (count / total_samples) * 100
        print(f"{role:<50} {count:<10,} {percentage:>6.2f}%")

    if len(sorted_roles) > 30:
        remaining = sum(count for _, count in sorted_roles[30:])
        print(f"{'... and ' + str(len(sorted_roles) - 30) + ' more roles':<50} {remaining:<10,} {remaining/total_samples*100:>6.2f}%")

    # ========================================
    # 2. PERMISSION FIELDS OVERVIEW
    # ========================================
    print("\n" + "━" * 80)
    print("PERMISSION FIELDS OVERVIEW")
    print("━" * 80)

    print(f"\nTotal unique permission fields: {len(permission_fields)}")
    print(f"\n{'Field Name':<40} {'Type(s)':<25} {'Unique Values':<15}")
    print("-" * 80)

    for field_name in sorted(permission_fields.keys()):
        types = ', '.join(sorted(permission_field_types.get(field_name, {'unknown'})))
        unique_count = len(permission_fields[field_name])
        print(f"{field_name:<40} {types:<25} {unique_count:<15,}")

    # ========================================
    # 3. DETAILED PERMISSION VALUE DISTRIBUTIONS
    # ========================================
    print("\n" + "━" * 80)
    print("DETAILED PERMISSION VALUE DISTRIBUTIONS")
    print("━" * 80)

    for field_name in sorted(permission_fields.keys()):
        values_counter = permission_fields[field_name]
        total_occurrences = sum(values_counter.values())
        unique_values = len(values_counter)

        print(f"\n{field_name.upper()}")
        print("-" * 80)
        print(f"Total occurrences: {total_occurrences:,}")
        print(f"Unique values: {unique_values:,}")

        # Sort by count
        sorted_values = sorted(values_counter.items(), key=lambda x: x[1], reverse=True)

        # Show distribution
        print(f"\n{'Value':<50} {'Count':<10} {'Percentage':<12}")
        print("-" * 80)

        # Show top 20 or all if less than 20
        display_count = min(20, len(sorted_values))
        for value, count in sorted_values[:display_count]:
            percentage = (count / total_occurrences) * 100
            value_str = str(value)[:48]  # Truncate long values
            print(f"{value_str:<50} {count:<10,} {percentage:>6.2f}%")

        if len(sorted_values) > 20:
            remaining = sum(count for _, count in sorted_values[20:])
            print(f"{'... and ' + str(len(sorted_values) - 20) + ' more':<50} {remaining:<10,} {remaining/total_occurrences*100:>6.2f}%")

    # ========================================
    # 4. SUMMARY STATISTICS
    # ========================================
    print("\n" + "━" * 80)
    print("SUMMARY STATISTICS")
    print("━" * 80)

    print(f"\nDataset: {file_path}")
    print(f"Total samples: {total_samples:,}")
    print(f"Unique user roles: {len(role_counter):,}")
    print(f"Unique permission fields: {len(permission_fields):,}")

    # Average number of permission fields per example
    permission_counts = []
    for example in ds:
        permissions = None
        if 'permissions' in example:
            permissions = safe_json_parse(example['permissions'])
        elif 'full_details' in example and 'permissions' in example['full_details']:
            permissions = safe_json_parse(example['full_details']['permissions'])

        if permissions and isinstance(permissions, dict):
            permission_counts.append(len(permissions))

    if permission_counts:
        avg_fields = sum(permission_counts) / len(permission_counts)
        print(f"Average permission fields per example: {avg_fields:.2f}")

    print("\n" + "=" * 80)


def export_to_json(file_path, output_path=None):
    """Export all permission data to a structured JSON file for database creation."""

    if output_path is None:
        output_path = Path(file_path).stem + "_permission_database.json"

    print(f"\n{'='*80}")
    print(f"Exporting permission database to: {output_path}")
    print(f"{'='*80}")

    # Load dataset
    ds = load_dataset('parquet', data_files=str(file_path))['train']

    # Collect all data
    database = {
        'metadata': {
            'source_file': str(file_path),
            'total_samples': len(ds),
        },
        'user_roles': [],
        'permission_fields': {},
        'examples': []
    }

    role_counter = Counter()
    permission_fields = defaultdict(lambda: {'type': set(), 'values': Counter()})

    # Process all examples
    for idx, example in enumerate(ds):
        example_data = {
            'index': idx,
            'user_role': example.get('user_role', None),
            'permissions': {},
            'expected_response': example.get('expected_response', None),
            'query': example.get('query', None),
        }

        # Extract role
        if example_data['user_role']:
            role_counter[example_data['user_role']] += 1

        # Extract permissions
        permissions = None
        if 'permissions' in example:
            permissions = safe_json_parse(example['permissions'])
        elif 'full_details' in example and 'permissions' in example['full_details']:
            permissions = safe_json_parse(example['full_details']['permissions'])

        if permissions and isinstance(permissions, dict):
            example_data['permissions'] = permissions

            # Collect field statistics
            for field_name, field_value in permissions.items():
                permission_fields[field_name]['type'].add(type(field_value).__name__)

                if isinstance(field_value, list):
                    for item in field_value:
                        permission_fields[field_name]['values'][str(item)] += 1
                else:
                    permission_fields[field_name]['values'][str(field_value)] += 1

        database['examples'].append(example_data)

    # Add role statistics
    database['user_roles'] = [
        {'role': role, 'count': count, 'percentage': count / len(ds) * 100}
        for role, count in sorted(role_counter.items(), key=lambda x: x[1], reverse=True)
    ]

    # Add permission field statistics
    for field_name, data in permission_fields.items():
        database['permission_fields'][field_name] = {
            'types': list(data['type']),
            'unique_values': len(data['values']),
            'value_distribution': [
                {'value': value, 'count': count, 'percentage': count / sum(data['values'].values()) * 100}
                for value, count in sorted(data['values'].items(), key=lambda x: x[1], reverse=True)
            ]
        }

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(database, f, indent=2, default=str)

    print(f"\n✓ Exported {len(ds):,} examples to {output_path}")
    print(f"✓ Database contains:")
    print(f"  - {len(role_counter)} unique user roles")
    print(f"  - {len(permission_fields)} unique permission fields")
    print(f"  - Complete value distributions for all fields")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze user roles and permissions in OrgAccess dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Path to parquet file'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Analyze all datasets'
    )

    parser.add_argument(
        '--export',
        action='store_true',
        help='Export permission database to JSON'
    )

    parser.add_argument(
        '--output',
        help='Output file path for exported JSON (only with --export)'
    )

    args = parser.parse_args()

    if args.all:
        datasets = [
            'data/easy-00000-of-00001.parquet',
            'data/medium-00000-of-00001.parquet',
            'data/hard-00000-of-00001.parquet',
            'training_data/train.parquet',
            'training_data/validation.parquet',
        ]

        for file_path in datasets:
            if Path(file_path).exists():
                analyze_permissions_and_roles(file_path)
                if args.export:
                    export_to_json(file_path)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

        analyze_permissions_and_roles(file_path)

        if args.export:
            export_to_json(file_path, args.output)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python analyze_permissions.py data/hard-00000-of-00001.parquet")
        print("  python analyze_permissions.py training_data/train.parquet --export")
        print("  python analyze_permissions.py --all")
        print("  python analyze_permissions.py data/hard-00000-of-00001.parquet --export --output hard_db.json")
        sys.exit(1)


if __name__ == '__main__':
    main()
