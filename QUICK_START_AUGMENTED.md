# Quick Start: Training with Augmented Data

## TL;DR

```bash
# Use this instead of scripts/train_qlora.py
python scripts/train_qlora_augmented.py --config configs/llama3_1_8b_qlora.yaml
```

That's it! The script automatically uses the expanded, balanced dataset.

## What Changed?

| Original | Augmented |
|----------|-----------|
| 11K examples | 27K examples (2.4x more) |
| 2% full / 49% partial / 49% rejected | 33% full / 33% partial / 33% rejected |
| Some corrupted labels | All labels cleaned |
| `scripts/train_qlora.py` | `scripts/train_qlora_augmented.py` |

## Training Commands

### Llama 3.1 8B (Recommended first)
```bash
python scripts/train_qlora_augmented.py \
    --config configs/llama3_1_8b_qlora.yaml \
    --wandb
```

### Gemma 3 12B
```bash
python scripts/train_qlora_augmented.py \
    --config configs/gemma3_12b_qlora.yaml \
    --wandb
```

### Qwen 2.5 32B
```bash
python scripts/train_qlora_augmented.py \
    --config configs/qwen2_5_32b_qlora.yaml \
    --wandb
```

## What You'll See

```
============================================================
Loading AUGMENTED datasets...
============================================================
📁 Train file: training_data/addOn_training_data/train.parquet
📁 Validation file: training_data/addOn_training_data/validation.parquet

✓ Train examples: 26,904
✓ Validation examples: 6,926
```

## Expected Training Time

| Model | GPU | Time |
|-------|-----|------|
| Llama 3.1 8B | 1x H100 80GB | ~2 hours |
| Gemma 3 12B | 1x H100 80GB | ~3-4 hours |
| Qwen 2.5 32B | 1x H100 80GB | ~6-8 hours |

## Your Existing Configs Work!

No need to modify your config files. The augmented script overrides the data paths automatically.

## Difference from Original Script

**Only change:** Data files
- Original: Uses paths from config file
- Augmented: Hardcoded to `training_data/addOn_training_data/`

Everything else identical:
- ✓ Same QLoRA settings
- ✓ Same hyperparameters
- ✓ Same training arguments
- ✓ Same model architectures

## Troubleshooting

**Q: Script can't find the data?**
```bash
# Verify files exist
ls -lh training_data/addOn_training_data/*.parquet | grep -v backup
```

**Q: Want to use original data instead?**
```bash
# Use the original script
python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml
```

**Q: How do I know it's using augmented data?**
Look for "Loading AUGMENTED datasets..." in the output.

## After Training

Evaluate your model:
```bash
python scripts/evaluate_vllm_direct.py \
    --model-path models/your-model-augmented \
    --benchmark-split hard \
    --output results/
```

## More Info

- Full guide: [README_AUGMENTED_TRAINING.md](README_AUGMENTED_TRAINING.md)
- Dataset details: [AUGMENTED_DATA_SUMMARY.md](AUGMENTED_DATA_SUMMARY.md)
- Script tools: [scripts/README_convert.md](scripts/README_convert.md)

---

**Ready to train? Just run:**
```bash
python scripts/train_qlora_augmented.py --config configs/llama3_1_8b_qlora.yaml
```
