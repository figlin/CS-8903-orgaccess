# JSONL to Parquet Converter

Convert synthetic JSONL files from `data/addOn_Data/` into parquet format matching the structure of the original OrgAccess dataset files.

## Requirements

```bash
pip install pyarrow
```

## Usage

### Convert All JSONL Files

Convert all JSONL files in `data/addOn_Data/` to separate parquet files:

```bash
python scripts/convert_jsonl_to_parquet.py
```

### Convert and Merge by Difficulty

Convert all hard difficulty files and merge them into a single parquet:

```bash
python scripts/convert_jsonl_to_parquet.py --difficulty hard --merge
```

This creates `data/hard_augmented.parquet` containing all hard examples merged together.

### Convert Specific File

Convert a specific JSONL file to parquet:

```bash
python scripts/convert_jsonl_to_parquet.py \
    --input data/addOn_Data/hard_full_augmented-1000.jsonl \
    --output data/hard_augmented.parquet
```

### Merge with Existing Dataset

Add synthetic data to an existing parquet file:

```bash
python scripts/convert_jsonl_to_parquet.py \
    --input data/addOn_Data/hard_full_augmented-1000.jsonl \
    --output data/hard-00000-of-00001.parquet \
    --merge
```

⚠️ **Warning**: This will modify your original dataset file. Make a backup first!

## Options

- `--input PATH` - Input JSONL file (if not specified, converts all in addOn_Data/)
- `--output PATH` - Output parquet file (required if --input is specified)
- `--input-dir PATH` - Input directory containing JSONL files (default: data/addOn_Data)
- `--output-dir PATH` - Output directory for parquet files (default: data/)
- `--difficulty {easy,medium,hard}` - Filter by difficulty level
- `--merge` - Merge with existing parquet or merge all JSONL into one parquet

## Examples

### Example 1: Convert all hard examples to one file

```bash
python scripts/convert_jsonl_to_parquet.py --difficulty hard --merge
```

Output: `data/hard_augmented.parquet`

### Example 2: Create augmented training dataset

```bash
# Merge all synthetic data with original hard dataset
python scripts/convert_jsonl_to_parquet.py \
    --input data/addOn_Data/hard_full_augmented-1000.jsonl \
    --output data/hard_augmented_combined.parquet

# Or merge directly into original (make backup first!)
cp data/hard-00000-of-00001.parquet data/hard-00000-of-00001.parquet.backup
python scripts/convert_jsonl_to_parquet.py \
    --input data/addOn_Data/hard_full_augmented-1000.jsonl \
    --output data/hard-00000-of-00001.parquet \
    --merge
```

### Example 3: Convert each file separately

```bash
python scripts/convert_jsonl_to_parquet.py
```

This creates:
- `data/hard_full_augmented-1000.parquet`
- `data/hard_full_augmented-1100.parquet`
- `data/hard_full_augmented-200.parquet`
- etc.

## Data Cleaning

The script automatically:
- ✓ Removes synthetic metadata fields (`source_index`, `generation_strategy`, `synthetic`)
- ✓ Keeps only core OrgAccess fields: `user_role`, `permissions`, `query`, `expected_response`, `rationale`
- ✓ Validates required fields are present
- ✓ Skips invalid JSON lines with warnings

## Output Format

The output parquet files match the structure of the original OrgAccess dataset:

```python
{
    "user_role": str,
    "permissions": dict,
    "query": str,
    "expected_response": str,  # "full", "partial", or "rejected"
    "rationale": str  # optional
}
```

## Workflow: Synthetic Data → Training

1. **Generate synthetic data** using `generate_full_examples.py` or `generate_full_examples_medium.py`:
   ```bash
   python scripts/generate_full_examples.py \
       --input hard-00000-of-00001_permission_database.json \
       --output data/addOn_Data/hard_full_augmented-NEW.jsonl \
       --num-examples 1000
   ```

2. **Convert to parquet** and merge with original dataset:
   ```bash
   python scripts/convert_jsonl_to_parquet.py \
       --difficulty hard --merge \
       --output-dir data/
   ```

3. **Use in training** by pointing to the augmented parquet file in your config:
   ```yaml
   data:
     train_file: "data/hard_augmented.parquet"
   ```

## Checking Results

After conversion, verify the data:

```bash
# Using Python
python3 -c "
import pyarrow.parquet as pq
table = pq.read_table('data/hard_augmented.parquet')
print(f'Total records: {len(table):,}')
print(f'Schema: {table.schema}')
print(f'File size: {Path('data/hard_augmented.parquet').stat().st_size / 1024 / 1024:.2f} MB')
"
```

Or use the analysis scripts:
```bash
python scripts/analyze_data_distribution.py data/hard_augmented.parquet
```
