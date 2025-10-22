# Training with Augmented Dataset

This guide covers training models using the expanded, balanced dataset from `training_data/addOn_training_data/`.

## Dataset Overview

**Training Data:** `training_data/addOn_training_data/train.parquet`
- Total: 26,904 examples
- rejected: 9,079 (33.75%)
- partial: 8,980 (33.38%)
- full: 8,845 (32.88%)
- ✓ Balanced across all three classes

**Validation Data:** `training_data/addOn_training_data/validation.parquet`
- Total: 6,926 examples
- Similar balanced distribution

## Quick Start

### Using the Augmented Training Script

The script `scripts/train_qlora_augmented.py` automatically uses the augmented dataset:

```bash
# Llama 3.1 8B
python scripts/train_qlora_augmented.py \
    --config configs/llama3_1_8b_qlora.yaml

# With W&B logging
python scripts/train_qlora_augmented.py \
    --config configs/llama3_1_8b_qlora.yaml \
    --wandb

# Gemma 3 12B
python scripts/train_qlora_augmented.py \
    --config configs/gemma3_12b_qlora.yaml

# Qwen 2.5 32B
python scripts/train_qlora_augmented.py \
    --config configs/qwen2_5_32b_qlora.yaml
```

## What's Different?

**`train_qlora_augmented.py` vs `train_qlora.py`:**

The augmented version:
- ✓ Automatically uses `training_data/addOn_training_data/train.parquet`
- ✓ Automatically uses `training_data/addOn_training_data/validation.parquet`
- ✓ Hardcoded to use augmented data (ignores config file paths)
- ✓ ~27K training examples (vs ~11K original)
- ✓ Balanced class distribution (33%/33%/33%)

Everything else is identical:
- Same QLoRA configuration
- Same hyperparameters
- Same training arguments
- Same model architectures

## Expected Improvements

With the augmented dataset, you should see:

1. **Better class balance** - More even performance across full/partial/rejected
2. **More training examples** - 2.4x more data than original
3. **Diverse scenarios** - Synthetic examples with varied permission structures
4. **Reduced overfitting** - More data helps generalization

## Training Output

The trained adapters will be saved to:
```
models/{model_name}-augmented/
├── checkpoint-500/
├── checkpoint-1000/
└── final/
```

Naming suggestion: Add "-augmented" to your model names to distinguish from original training.

## Monitoring Training

### Local logs
```bash
tail -f models/{model_name}/training.log
```

### W&B (if enabled)
The run will be tagged with the augmented dataset info.

## Evaluation

After training, evaluate on the original benchmark:

```bash
# Evaluate on HARD split
python scripts/evaluate_vllm_direct.py \
    --model-path models/llama-3.1-8b-augmented \
    --benchmark-split hard \
    --output results/augmented/

# Compare with original
python scripts/analyze_results.py \
    results/llama-3.1-8b-original/results.json \
    results/llama-3.1-8b-augmented/results.json
```

## Recommended Training Schedule

### 1. Llama 3.1 8B (Baseline)
- GPU: 1x H100 (80GB) or 1x A100 (40GB)
- Time: ~2 hours
- Batch size: 4, grad accum: 4

### 2. Gemma 3 12B (Medium)
- GPU: 1x H100 (80GB)
- Time: ~3-4 hours
- Batch size: 2, grad accum: 8

### 3. Qwen 2.5 32B (Large)
- GPU: 1x H100 (80GB)
- Time: ~6-8 hours
- Batch size: 2, grad accum: 8

## Data Provenance

The augmented dataset combines:

1. **Original training data** (~11K examples)
   - From `training_data/train.parquet`

2. **Synthetic FULL examples** (~3K examples)
   - Generated using `scripts/generate_full_examples.py`
   - Generated using `scripts/generate_full_examples_medium.py`
   - Balanced to match partial/rejected counts

3. **Label corrections** (153 records)
   - Fixed using `scripts/fix_parquet_labels.py`
   - Normalized corrupted labels (reject→rejected, approve→full, etc.)

## Troubleshooting

### Q: Can I use my existing configs?
**A:** Yes! The augmented script ignores the data paths in configs and uses the augmented data automatically.

### Q: Will this work with the original `train_qlora.py`?
**A:** No, use `train_qlora_augmented.py` which has hardcoded paths to the augmented data.

### Q: How do I switch back to original data?
**A:** Use the original `scripts/train_qlora.py` instead.

### Q: What if I want to use only specific splits?
**A:** The augmented data is already filtered and balanced. For custom splits, modify the script or use the original.

## Next Steps

1. ✓ Train models with augmented data
2. ✓ Evaluate on original benchmark (HARD/MEDIUM/EASY splits)
3. ✓ Compare metrics with original training
4. ✓ Analyze per-class performance improvements
5. ✓ Push best models to HuggingFace Hub

## Files Created

```
training_data/addOn_training_data/
├── train.parquet              (26,904 examples, balanced)
├── validation.parquet          (6,926 examples, balanced)
├── train.parquet.backup        (backup before label fixes)
└── validation.parquet.backup   (backup before label fixes)

scripts/
├── train_qlora_augmented.py    (uses augmented data)
├── fix_parquet_labels.py       (label normalization tool)
├── convert_jsonl_to_parquet.py (JSONL→parquet converter)
├── generate_full_examples.py   (synthetic data generator - HARD)
└── generate_full_examples_medium.py (synthetic data generator - MEDIUM)
```

Good luck with training! 🚀
