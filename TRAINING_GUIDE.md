# OrgAccess Fine-Tuning Guide

Quick start guide for fine-tuning Llama-3.1-8B on the OrgAccess benchmark.

## Prerequisites

### 1. Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 4090 (24GB) or A40 (48GB)
- RAM: 32GB system RAM
- Storage: 50GB free space

**Recommended:**
- GPU: NVIDIA A40 (48GB) or A100 (80GB)
- RAM: 64GB system RAM
- Storage: 100GB free space

### 2. Software Requirements

```bash
# Python 3.10+
python --version

# CUDA 11.8+ or 12.1+
nvidia-smi
```

## Setup

### Step 1: Prepare Data (Already Done!)

You've already created the training/benchmark splits:

```
✓ training_data/train.parquet       (25,304 examples)
✓ training_data/validation.parquet  (6,326 examples)
✓ benchmark_data/easy_test.parquet  (11,484 examples)
✓ benchmark_data/medium_test.parquet (2,073 examples)
✓ benchmark_data/hard_test.parquet  (10,613 examples)
```

### Step 2: Install Dependencies

git clone https://github.com/figlin/CS-8903-orgaccess.git

```bash
# Activate your virtual environment
source venv/bin/activate

# Install training dependencies
pip install -r requirements_training.txt
```

### Step 3: Hugging Face Authentication

You need access to Llama-3.1-8B-Instruct:

```bash
# Login to Hugging Face
huggingface-cli login

# Enter your token when prompted
# Get token from: https://huggingface.co/settings/tokens
```

**Important:** Request access to Meta-Llama-3.1-8B-Instruct:
- Go to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- Click "Request Access" and accept license
- Wait for approval (usually ~1 hour)

## Training

### Local Training (if you have GPU)

```bash
# Basic training
python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml

# With Weights & Biases tracking
python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml --wandb

# Resume from checkpoint
python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml --resume outputs/llama3_1_8b_orgaccess_qlora/checkpoint-1000
```

### Training on RunPod

#### Option 1: Quick Start (Recommended)

1. **Launch Pod:**
   - Go to https://www.runpod.io/console/pods
   - Select GPU: A40 (48GB) - ~$0.79/hour
   - Template: RunPod Pytorch 2.1
   - Volume: 50GB

2. **SSH into Pod:**
```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

3. **Setup Environment:**
```bash
# Clone your repo
git clone <your-repo-url>
cd CS-8903-orgaccess

# Install dependencies
pip install -r requirements_training.txt

# Login to Hugging Face
huggingface-cli login
# Paste your token

# Optional: Setup WandB
wandb login
```

4. **Start Training:**
```bash
# Run training
python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml --wandb

# Or run in background with nohup
nohup python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml --wandb > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

5. **Monitor Training:**
   - Check WandB dashboard: https://wandb.ai/<username>/orgaccess-finetuning
   - Or use TensorBoard: `tensorboard --logdir outputs/llama3_1_8b_orgaccess_qlora`

6. **After Training:**
```bash
# Download model to local machine
scp -P <port> -i ~/.ssh/id_ed25519 -r root@<pod-ip>:/root/CS-8903-orgaccess/outputs/llama3_1_8b_orgaccess_qlora/final_model ./local_models/

# Stop pod to save costs!
```

## Training Configuration

### Current Settings (configs/llama3_1_8b_qlora.yaml)

```yaml
# Model
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
quantization: 4-bit (QLoRA)

# LoRA
rank: 64
alpha: 128
dropout: 0.05

# Training
epochs: 3
batch_size: 4 (per device)
gradient_accumulation: 4 (effective batch = 16)
learning_rate: 2e-4
max_seq_length: 2048

# Expected
total_steps: ~4,746
training_time: 4-6 hours (A40)
cost: $3-5 (RunPod)
```

### Adjusting Configuration

Edit `configs/llama3_1_8b_qlora.yaml`:

**For faster training (less accuracy):**
```yaml
num_train_epochs: 2          # Reduce epochs
learning_rate: 3.0e-4        # Increase LR
```

**For better accuracy (slower):**
```yaml
num_train_epochs: 5          # More epochs
lora.r: 96                   # Increase LoRA rank
lora.lora_alpha: 192
```

**For smaller GPU (< 24GB):**
```yaml
per_device_train_batch_size: 2  # Reduce batch
gradient_accumulation_steps: 8  # Increase accumulation
max_seq_length: 1536            # Reduce sequence length
```

## Monitoring Training

### Progress Indicators

**Good Training:**
- Train loss decreasing smoothly
- Eval loss decreasing (not increasing)
- Gap between train/eval loss < 0.5

**Overfitting:**
- Train loss very low, eval loss high or increasing
- Large gap between train/eval loss (> 1.0)
- **Solution:** Reduce epochs or increase regularization

**Underfitting:**
- Both train and eval loss high
- Loss plateaus early
- **Solution:** Increase epochs or learning rate

### Checkpoints

Checkpoints saved every 500 steps:
```
outputs/llama3_1_8b_orgaccess_qlora/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
├── ...
└── final_model/          ← Best model (use this!)
```

## After Training

### 1. Test Your Model Locally

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load LoRA weights
model = PeftModel.from_pretrained(
    model,
    "outputs/llama3_1_8b_orgaccess_qlora/final_model"
)

# Test inference
messages = [
    {"role": "system", "content": "You are a knowledge repository..."},
    {"role": "user", "content": "Given the following permissions - {...}"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 2. Upload to Hugging Face Hub

```python
from huggingface_hub import login

# Login
login(token="your-hf-token")

# Upload model
model.push_to_hub("your-username/llama3.1-8b-orgaccess-qlora")
tokenizer.push_to_hub("your-username/llama3.1-8b-orgaccess-qlora")
```

### 3. Evaluate on Benchmark

Create evaluation script (see RESEARCH_METHODOLOGY.md) or use existing:

```bash
# Update evaluate_runpod.py to use your fine-tuned model
# Then run evaluation on all test sets
python evaluate_benchmark.py --model outputs/llama3_1_8b_orgaccess_qlora/final_model
```

## Troubleshooting

### Out of Memory (OOM)

```yaml
# Reduce batch size
per_device_train_batch_size: 2
gradient_accumulation_steps: 8

# Reduce sequence length
max_seq_length: 1024

# Enable gradient checkpointing (already enabled)
```

### Model Not Loading

```bash
# Check Hugging Face login
huggingface-cli whoami

# Re-login if needed
huggingface-cli login

# Verify access to Llama 3.1
# Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Training Too Slow

```yaml
# Increase batch size (if memory allows)
per_device_train_batch_size: 8
gradient_accumulation_steps: 2

# Reduce logging frequency
logging_steps: 100
eval_steps: 1000
save_steps: 1000
```

### Loss Not Decreasing

```yaml
# Increase learning rate
learning_rate: 3.0e-4

# Increase warmup
warmup_ratio: 0.05

# Increase LoRA rank
lora.r: 96
lora.lora_alpha: 192
```

## Expected Results

### Baseline (No Fine-Tuning)
- Easy test: ~60-70% accuracy
- Medium test: ~50-60% accuracy
- Hard test: ~40-50% accuracy

### After Fine-Tuning
- Easy test: ~80-90% accuracy (good generalization)
- Medium test: ~70-80% accuracy
- Hard test: ~65-75% accuracy (main target)

**Target for paper:** 15-25% improvement on Hard test set!

## Cost Estimates

### RunPod A40 (48GB)
- Training time: 4-6 hours
- Cost: ~$0.79/hour
- **Total: $3-5**

### RunPod A100 (80GB)
- Training time: 2-3 hours
- Cost: ~$1.89/hour
- **Total: $4-6**

### Local GPU (Free)
- RTX 4090: 6-8 hours
- RTX A6000: 4-6 hours

## Next Steps

1. ✅ Data prepared → `training_data/` and `benchmark_data/`
2. ⏳ Run training → `python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml`
3. ⏳ Evaluate model → Use benchmark test sets
4. ⏳ Compare results → Baseline vs Fine-tuned
5. ⏳ Write paper → Document improvements

## Support

- **Documentation:** See RESEARCH_METHODOLOGY.md
- **RunPod Docs:** https://docs.runpod.io/
- **Transformers:** https://huggingface.co/docs/transformers
- **PEFT/LoRA:** https://huggingface.co/docs/peft

Good luck with your training! 🚀
