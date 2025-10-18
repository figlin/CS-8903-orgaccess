# vLLM Direct Endpoint Evaluation Guide

Complete guide for evaluating your fine-tuned model hosted on a RunPod Pod with vLLM.

## Quick Start

```bash
# Basic evaluation (hard test set)
python evaluate_vllm_direct.py --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net

# That's it! No API key needed.
```

---

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Command Line Flags](#command-line-flags)
- [Examples](#examples)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

```bash
# Ensure you have the virtual environment activated
source venv/bin/activate

# Install dependencies (if not already installed)
pip install openai datasets scikit-learn
```

### Verify vLLM Server is Running

```bash
# Test your endpoint
curl https://qaoiq6ga68pz4p-8000.proxy.runpod.net/v1/models

# Should return JSON with model information
```

---

## Basic Usage

### Simplest Command

```bash
python evaluate_vllm_direct.py --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net
```

This will:
- Connect to your vLLM endpoint
- Evaluate on the **hard** test set (default)
- Use all samples
- Save results with auto-generated filename

### Common Workflows

```bash
# Quick test with 100 samples
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --max-samples 100

# Evaluate all three test splits
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --all-splits

# Evaluate specific split
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --split easy
```

---

## Command Line Flags

### Required Arguments

| Flag | Description | Example |
|------|-------------|---------|
| `--url` | vLLM endpoint URL | `--url https://your-pod-8000.proxy.runpod.net` |

### Optional Arguments

#### Test Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--split` | choice | `hard` | Test split to evaluate. Choices: `easy`, `medium`, `hard` |
| `--all-splits` | flag | `False` | Evaluate all splits (overrides `--split`) |
| `--max-retries` | int | `4` | Maximum number of retry attempts on error |
| `--retry-delay` | int | `40` | Delay between retries in seconds |
| `--batch-workers` | int | `8` | Number of parallel workers (range: 1-32, higher = faster) |
| `--max-samples` | int | `None` | Maximum samples to evaluate. `None` = all samples |
| `--use-benchmark` | flag | `False` | Use `benchmark_data/` directory instead of `data/` |

#### Model Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model-name` | string | `orgaccess-finetuned` | Model name (arbitrary, vLLM ignores this) |

#### Data Paths

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data-dir` | path | `data` | Directory containing original test data |
| `--benchmark-dir` | path | `benchmark_data` | Directory containing benchmark test data |

#### Output Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output` | path | auto-generated | Custom output JSON file path |

#### Advanced Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--verbose` / `-v` | flag | `False` | Enable verbose output (shows predictions every 50 samples) |
| `--no-test-connection` | flag | `False` | Skip connection test (faster startup) |
| `--save-full-responses` | flag | `True` | Save complete queries, permissions, model responses, and rationales in JSON ✅ |
| `--no-save-full-responses` | flag | `False` | Disable saving full responses (smaller file size, only metrics) |

---

## Examples

### Example 1: Quick Sanity Check

Test with 50 samples to verify everything works:

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --max-samples 50 \
  --verbose
```

**Output:**
```
============================================================
Direct vLLM Endpoint Evaluation
============================================================
Endpoint: https://qaoiq6ga68pz4p-8000.proxy.runpod.net/v1
Splits: hard
Max samples: 50
============================================================

Testing connection to vLLM endpoint...
✓ Connection successful!
Available models: ['orgaccess-finetuned']

Loading test dataset from data/hard-00000-of-00001.parquet...
✓ Loaded 10613 examples from hard split

============================================================
Starting Evaluation
============================================================
Endpoint: https://qaoiq6ga68pz4p-8000.proxy.runpod.net/v1
Model: orgaccess-finetuned
Total samples: 50
============================================================

Processed 10/50 examples
Processed 20/50 examples
...
```

### Example 2: Full Hard Test Evaluation

Evaluate entire hard test set (10,613 examples):

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --split hard
```

**Expected time:** ~30-60 minutes (depends on vLLM speed)

### Example 3: Evaluate All Splits

Get comprehensive results across easy, medium, and hard:

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --all-splits
```

**Output:**
```
============================================================
SUMMARY - All Splits
============================================================

EASY      : Accuracy=0.8234, F1=0.8123 (11484 samples)
MEDIUM    : Accuracy=0.7456, F1=0.7234 (2073 samples)
HARD      : Accuracy=0.6845, F1=0.6623 (10613 samples)

============================================================
```

### Example 4: Use Benchmark Data Directory

If you want to use `benchmark_data/` instead of `data/`:

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --use-benchmark \
  --split hard
```

**Looks for:** `benchmark_data/hard_test.parquet`

### Example 5: Custom Output File

Save results to specific file:

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --split hard \
  --output final_evaluation_results.json
```

### Example 6: Fast Development Testing

Skip connection test for faster iteration:

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --max-samples 10 \
  --no-test-connection
```

### Example 7: Verbose Mode for Debugging

See detailed predictions:

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --max-samples 100 \
  --verbose
```

**Verbose output shows:**
```
Processed 50/100 examples
  Last prediction: rejected (expected: rejected)
```

### Example 8: Custom Retry Configuration

Adjust retry behavior for unstable connections:

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --all-splits \
  --max-retries 5 \
  --retry-delay 60
```

This will retry each failed request up to 5 times with 60 seconds between attempts.

### Example 9: Maximize Speed with Parallel Workers

Speed up evaluation with more parallel workers:

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --all-splits \
  --batch-workers 16
```

**Performance comparison:**
- `--batch-workers 1`: Sequential processing (~3s per sample = 8.8 hours for hard split)
- `--batch-workers 8`: **~8x faster** (~0.375s per sample = 1.1 hours for hard split) ✅ DEFAULT
- `--batch-workers 16`: **~12x faster** (~0.25s per sample = 44 minutes for hard split)
- `--batch-workers 32`: **~16x faster** (if your vLLM server can handle it)

**Note:** Higher worker counts require vLLM server to handle concurrent requests. Start with 8 and increase if your server isn't saturated.

### Example 10: Full Response Details (DEFAULT)

**By default**, all JSON results include complete details for each prediction:

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --split hard \
  --max-samples 500
```

Each prediction in the JSON includes:
- `prediction`: Model's classification (full/partial/rejected)
- `ground_truth`: Expected classification
- `correct`: Boolean indicating if prediction was correct
- `full_details`:
  - `user_role`: User's role in organization
  - `permissions`: JSON of user's permissions
  - `query`: User's actual query/request
  - `model_response`: Complete model response text
  - `expected_rationale`: Why the ground truth label is correct

**To save space** (only metrics, no individual responses):

```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --split hard \
  --no-save-full-responses
```

---

## Error Handling & Interruption

### Automatic Retry on Errors

The script automatically retries failed requests:
- **Default:** 4 retry attempts with 40-second delays
- Failed requests are logged with error details
- After all retries fail, the sample defaults to 'rejected' prediction

**Example output:**
```
Error on example 633 (attempt 1/5): Error code: 404
  Retrying in 40 seconds...
Error on example 633 (attempt 2/5): Error code: 404
  Retrying in 40 seconds...
...
```

### Graceful Interruption (Ctrl+C)

If you need to stop the evaluation (Ctrl+C), the script will:
1. Save all completed predictions
2. Calculate metrics for partial results
3. Save to timestamped JSON file: `results_vllm_INTERRUPTED_<timestamp>.json`

**Example:**
```bash
python evaluate_vllm_direct.py \
  --url https://qaoiq6ga68pz4p-8000.proxy.runpod.net \
  --all-splits

# Press Ctrl+C after some samples complete
^C
⚠️  Interrupted! Saving partial results...

✓ Partial results saved to results_vllm_INTERRUPTED_20250118_143022.json
  Completed samples: 1245
  Current split: hard
  Errors encountered: 3

You can resume by re-running the command.
```

The partial results file includes:
- All metrics calculated from completed samples
- Error count and details
- Timestamp and configuration
- Status marker indicating interruption

---

## Output Format

### Console Output

```
============================================================
EVALUATION RESULTS
============================================================
Accuracy:       0.7234
F1 (Macro):     0.6845
F1 (Weighted):  0.7123

------------------------------------------------------------
Per-Class Metrics:
------------------------------------------------------------
              precision    recall  f1-score   support

        full     0.6500    0.5417    0.5909        24
     partial     0.7123    0.7892    0.7487      3759
    rejected     0.7456    0.7234    0.7343      6808

    accuracy                         0.7234     10613
   macro avg     0.7026    0.6848    0.6913     10613
weighted avg     0.7245    0.7234    0.7239     10613

------------------------------------------------------------
Confusion Matrix:
------------------------------------------------------------
                Predicted
                Full  Partial  Rejected
Actual Full       13        8         3
      Partial    120     2968       671
      Rejected   102      856      5850
============================================================
```

### JSON Output File

**Filename (auto-generated):** `results_vllm_hard_20251017_154532.json`

**Structure:**
```json
{
  "timestamp": "2025-10-17T15:45:32.123456",
  "endpoint": "https://qaoiq6ga68pz4p-8000.proxy.runpod.net/v1",
  "model_name": "orgaccess-finetuned",
  "max_samples": null,
  "results": {
    "hard": {
      "file": "data/hard-00000-of-00001.parquet",
      "total_samples": 10613,
      "metrics": {
        "accuracy": 0.7234,
        "f1_macro": 0.6845,
        "f1_weighted": 0.7123
      },
      "confusion_matrix": [
        [13, 8, 3],
        [120, 2968, 671],
        [102, 856, 5850]
      ]
    }
  }
}
```

### Multi-Split Output (--all-splits)

**Filename:** `results_vllm_all_splits_20251017_154532.json`

```json
{
  "timestamp": "2025-10-17T15:45:32.123456",
  "endpoint": "https://qaoiq6ga68pz4p-8000.proxy.runpod.net/v1",
  "model_name": "orgaccess-finetuned",
  "max_samples": null,
  "results": {
    "easy": {
      "file": "data/easy-00000-of-00001.parquet",
      "total_samples": 11484,
      "metrics": { "accuracy": 0.8234, "f1_macro": 0.8123, "f1_weighted": 0.8245 },
      "confusion_matrix": [[...], [...], [...]]
    },
    "medium": {
      "file": "data/medium-00000-of-00001.parquet",
      "total_samples": 2073,
      "metrics": { "accuracy": 0.7456, "f1_macro": 0.7234, "f1_weighted": 0.7345 },
      "confusion_matrix": [[...], [...], [...]]
    },
    "hard": {
      "file": "data/hard-00000-of-00001.parquet",
      "total_samples": 10613,
      "metrics": { "accuracy": 0.6845, "f1_macro": 0.6623, "f1_weighted": 0.6734 },
      "confusion_matrix": [[...], [...], [...]]
    }
  }
}
```

---

## Troubleshooting

### Error: Connection Failed

```
❌ Connection failed: Connection refused
```

**Solutions:**
1. Check if vLLM server is running:
   ```bash
   # SSH into your RunPod pod
   ps aux | grep vllm
   ```

2. Verify the URL:
   ```bash
   curl https://qaoiq6ga68pz4p-8000.proxy.runpod.net/v1/models
   ```

3. Check vLLM logs:
   ```bash
   tail -f vllm.log  # or wherever you redirected output
   ```

### Error: Dataset Not Found

```
❌ Error loading dataset: FileNotFoundError
```

**Solutions:**
1. Check if data file exists:
   ```bash
   ls -la data/hard-00000-of-00001.parquet
   ```

2. Use correct directory flag:
   ```bash
   # If using benchmark_data directory
   python evaluate_vllm_direct.py --url YOUR_URL --use-benchmark
   ```

3. Verify you're in the correct directory:
   ```bash
   pwd  # Should show .../CS-8903-orgaccess
   ```

### Slow Inference

If evaluation is taking too long:

```bash
# Test with fewer samples first
python evaluate_vllm_direct.py --url YOUR_URL --max-samples 100
```

**Typical speeds:**
- vLLM: ~2-5 seconds per example
- Full hard test (10,613 examples): ~6-9 hours

**Speed up vLLM:** Adjust server parameters when starting vLLM:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model your-model \
  --max-num-seqs 256 \      # Increase batch size
  --tensor-parallel-size 2   # Use multiple GPUs (if available)
```

### URL Format Issues

**Incorrect:**
```bash
# Missing protocol
--url qaoiq6ga68pz4p-8000.proxy.runpod.net

# Wrong path
--url https://qaoiq6ga68pz4p-8000.proxy.runpod.net/v1/models
```

**Correct:**
```bash
# With or without /v1 (script handles both)
--url https://qaoiq6ga68pz4p-8000.proxy.runpod.net
--url https://qaoiq6ga68pz4p-8000.proxy.runpod.net/v1
```

### Help Command

Get full help:
```bash
python evaluate_vllm_direct.py --help
```

---

## Advanced Usage

### Batch Evaluation Script

Create `batch_evaluate.sh`:

```bash
#!/bin/bash
URL="https://qaoiq6ga68pz4p-8000.proxy.runpod.net"

echo "Starting batch evaluation..."

# Evaluate all splits
python evaluate_vllm_direct.py --url $URL --all-splits

# Quick sanity check
python evaluate_vllm_direct.py --url $URL --max-samples 100 --output sanity_check.json

echo "Batch evaluation complete!"
```

Run:
```bash
chmod +x batch_evaluate.sh
./batch_evaluate.sh
```

### Integration with Research Pipeline

```bash
# 1. Train model
python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml

# 2. Merge and upload to HF
python scripts/merge_and_upload.py \
  --model outputs/llama3_1_8b_orgaccess_qlora/final_model \
  --repo your-username/orgaccess-finetuned

# 3. Deploy on RunPod (manual step via web UI)

# 4. Evaluate all splits
python evaluate_vllm_direct.py \
  --url https://your-pod-8000.proxy.runpod.net \
  --all-splits \
  --output final_results.json

# 5. Analyze results
python scripts/analyze_results.py final_results.json  # (create this if needed)
```

---

## Performance Benchmarks

### Expected Evaluation Times

| Test Split | Samples | Time (vLLM on A40) | Time (vLLM on RTX 4090) |
|------------|---------|-------------------|------------------------|
| Easy       | 11,484  | ~6-10 hours       | ~8-12 hours            |
| Medium     | 2,073   | ~1-2 hours        | ~1.5-2.5 hours         |
| Hard       | 10,613  | ~6-9 hours        | ~7-11 hours            |
| **All**    | 24,170  | ~13-21 hours      | ~16-25 hours           |

### Cost Estimates (RunPod)

| GPU | $/hour | Full Evaluation Cost (all splits) |
|-----|--------|-----------------------------------|
| RTX 4090 | $0.40 | ~$6-10 |
| A40 | $0.79 | ~$10-17 |
| A100 | $1.89 | ~$25-40 |

**Tip:** Use `--max-samples 1000` for quick tests (~30 minutes, ~$0.40)

---

## Summary

### Most Common Commands

```bash
# 1. Quick test (recommended first)
python evaluate_vllm_direct.py --url YOUR_URL --max-samples 100

# 2. Full hard test evaluation
python evaluate_vllm_direct.py --url YOUR_URL

# 3. Comprehensive evaluation (all splits)
python evaluate_vllm_direct.py --url YOUR_URL --all-splits

# 4. Custom output
python evaluate_vllm_direct.py --url YOUR_URL --output my_results.json
```

### Flag Quick Reference

| What I Want | Command |
|-------------|---------|
| Evaluate hard test | `--url YOUR_URL` |
| Evaluate easy test | `--url YOUR_URL --split easy` |
| Evaluate all tests | `--url YOUR_URL --all-splits` |
| Quick test (100 samples) | `--url YOUR_URL --max-samples 100` |
| Save to custom file | `--url YOUR_URL --output results.json` |
| Use benchmark data | `--url YOUR_URL --use-benchmark` |
| See detailed output | `--url YOUR_URL --verbose` |
| Skip connection test | `--url YOUR_URL --no-test-connection` |

---

## Questions?

- **Documentation:** See [RESEARCH_METHODOLOGY.md](RESEARCH_METHODOLOGY.md)
- **Training:** See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Deployment:** See [RUNPOD_DEPLOYMENT_GUIDE.md](RUNPOD_DEPLOYMENT_GUIDE.md)

Happy evaluating! 🚀
