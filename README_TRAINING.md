# OrgAccess Fine-Tuning - Setup Complete! ✅

All files have been prepared for fine-tuning Llama-3.1-8B on the OrgAccess benchmark.

## 📁 Project Structure

```
CS-8903-orgaccess/
├── data/                          # Original datasets
│   ├── easy-00000-of-00001.parquet
│   ├── medium-00000-of-00001.parquet
│   └── hard-00000-of-00001.parquet
│
├── training_data/                 # ✅ Ready for training
│   ├── train.parquet             (25,304 examples - 80% of 70% Easy+Medium)
│   └── validation.parquet        (6,326 examples - 20% of 70% Easy+Medium)
│
├── benchmark_data/                # ✅ Ready for evaluation
│   ├── easy_test.parquet         (11,484 examples - 30% Easy holdout)
│   ├── medium_test.parquet       (2,073 examples - 30% Medium holdout)
│   └── hard_test.parquet         (10,613 examples - 100% Hard - MAIN TARGET)
│
├── scripts/                       # ✅ Training scripts
│   └── train_qlora.py            (QLoRA fine-tuning script)
│
├── configs/                       # ✅ Configuration files
│   └── llama3_1_8b_qlora.yaml   (Optimized for 25K examples)
│
├── outputs/                       # Will contain checkpoints (created during training)
│   └── llama3_1_8b_orgaccess_qlora/
│       ├── checkpoint-500/
│       ├── checkpoint-1000/
│       └── final_model/          (Best model - use this!)
│
├── TRAINING_GUIDE.md             # ✅ Complete training guide
├── RESEARCH_METHODOLOGY.md       # ✅ Research methodology and evaluation
├── data_split_summary.json       # ✅ Data split statistics
├── prepare_data_splits.py        # ✅ Data preparation script (already run)
├── evaluate_runpod.py            # ✅ Evaluation script for RunPod
├── requirements_training.txt     # ✅ Training dependencies
└── requirements.txt              # Original dependencies
```

## 🎯 Data Split Summary

### Training Data (31,630 total)
- **train.parquet**: 25,304 examples
  - Full: 28.5% | Partial: 35.4% | Rejected: 35.4%
  - Perfect class balance! ✓

- **validation.parquet**: 6,326 examples
  - Full: 28.9% | Partial: 34.6% | Rejected: 35.8%
  - Matches training distribution ✓

### Benchmark Data (24,170 total)
- **easy_test.parquet**: 11,484 examples
  - Balanced: ~33% each class
  - Tests in-distribution generalization

- **medium_test.parquet**: 2,073 examples
  - Harder: 0.1% full, 45.5% partial, 54.4% rejected
  - Tests constraint understanding

- **hard_test.parquet**: 10,613 examples
  - Very challenging: 0.2% full, 35.4% partial, 64.1% rejected
  - **PRIMARY EVALUATION TARGET** - Tests compositional reasoning

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install training packages
pip install -r requirements_training.txt
```

### 2. Authenticate with Hugging Face

```bash
# Login to Hugging Face
huggingface-cli login

# Request access to Llama 3.1
# Go to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
# Click "Request Access" and wait for approval (~1 hour)
```

### 3. Start Training

**Local (if you have GPU):**
```bash
python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml
```

**On RunPod:**
```bash
# See TRAINING_GUIDE.md for detailed RunPod setup
# Quick version:
# 1. Launch A40 pod
# 2. Clone repo
# 3. Install dependencies
# 4. Run training:
python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml --wandb
```

## 📊 Expected Results

### Training Specs
- **Model**: Llama-3.1-8B-Instruct + QLoRA (4-bit)
- **Trainable params**: ~67M (0.8% of total)
- **Training time**: 4-6 hours on A40
- **Cost**: $3-5 on RunPod
- **Memory**: ~12-15GB VRAM
- **Total steps**: ~4,746 (3 epochs × 1,582 steps/epoch)

### Performance Targets

**Baseline (No Fine-Tuning):**
- Easy: ~60-70%
- Medium: ~50-60%
- Hard: ~40-50%

**After Fine-Tuning (Expected):**
- Easy: ~80-90% (+20-30% improvement)
- Medium: ~70-80% (+20% improvement)
- Hard: ~65-75% (+15-25% improvement) ← **Key result for paper**

## 🔧 Configuration Details

### Model Configuration
```yaml
Base Model: meta-llama/Meta-Llama-3.1-8B-Instruct
Quantization: 4-bit NF4 (QLoRA)
LoRA Rank: 64
LoRA Alpha: 128
Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

### Training Configuration
```yaml
Epochs: 3
Batch Size: 4 (per device)
Gradient Accumulation: 4 (effective batch = 16)
Learning Rate: 2e-4
LR Schedule: Cosine with 3% warmup
Weight Decay: 0.01
Max Sequence Length: 2048
Optimizer: paged_adamw_8bit
Precision: bfloat16
```

### Data Format
Each example is formatted as:
```
System: You are a knowledge repository in an organisational structure...
User: Given the following permissions - {permissions} and the following user query - {query}, decide if...
Assistant: Response type: {full/partial/rejected}\n\nRationale: {explanation}
```

## 📝 Next Steps

1. ✅ **Data prepared** - All splits created
2. ✅ **Scripts ready** - Training and evaluation scripts configured
3. ⏳ **Run training** - Start fine-tuning on RunPod or local GPU
4. ⏳ **Monitor progress** - Check WandB or TensorBoard
5. ⏳ **Evaluate model** - Test on all three benchmark sets
6. ⏳ **Compare results** - Baseline vs fine-tuned
7. ⏳ **Analyze errors** - Understand failure modes
8. ⏳ **Write paper** - Document methodology and results

## 📚 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training walkthrough
- **[RESEARCH_METHODOLOGY.md](RESEARCH_METHODOLOGY.md)** - Research design and evaluation methodology
- **[data_split_summary.json](data_split_summary.json)** - Detailed data statistics

## 🎓 Research Paper Outline

### Suggested Structure

1. **Introduction**
   - Problem: LLMs need to understand organizational hierarchies
   - Challenge: Complex permission reasoning
   - Contribution: Fine-tuning improves multi-constraint reasoning

2. **Methodology**
   - Dataset: OrgAccess (55,800 examples across 3 difficulties)
   - Training: 70% Easy+Medium (25K examples)
   - Testing: 30% Easy+Medium + 100% Hard (24K examples)
   - Model: Llama-3.1-8B + QLoRA

3. **Results**
   - Table: Performance comparison (Baseline vs Fine-tuned)
   - Figure: Per-difficulty accuracy breakdown
   - Analysis: Error categories and patterns

4. **Discussion**
   - Why fine-tuning helps with compositional reasoning
   - Limitations and failure cases
   - Future work

## 💡 Tips

### Training
- Start with default config - it's already optimized for 25K examples
- Monitor eval_loss - should decrease steadily
- If loss plateaus early, increase learning rate to 3e-4
- If overfitting (eval_loss increases), reduce epochs to 2

### Evaluation
- Test on Easy first to verify model learned basics
- Hard set is the main challenge - expect lower accuracy
- Analyze confusion matrix to find systematic errors
- Look at examples where model predicts "partial" vs ground truth

### Cost Optimization
- Use RunPod A40 (~$0.79/hr) instead of A100
- Stop pod immediately after training finishes
- Download model before stopping pod
- Can resume training from checkpoints if interrupted

## ❓ Troubleshooting

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md#troubleshooting) for:
- Out of Memory errors
- Model loading issues
- Slow training
- Loss not decreasing

## 🙏 Acknowledgments

- **OrgAccess Dataset**: respai-lab/orgaccess on Hugging Face
- **Base Model**: Meta-Llama-3.1-8B-Instruct
- **Framework**: Hugging Face Transformers + PEFT

---

**Ready to train!** Follow the [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for step-by-step instructions.

Good luck with your research! 🚀
