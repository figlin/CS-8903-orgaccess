# Multi-GPU Training Fix for Qwen2.5-32B

## What Was Fixed

The training script was using model parallelism (`device_map='auto'`) which splits the model across GPUs but only uses one GPU for computation. This caused:
- Only GPU 0 processing batches (80-100% utilization)
- GPU 1 holding model weights but not computing (0-5% utilization)
- Training taking 6s/step instead of 1.5-2s/step
- 17 hour training time instead of ~2 hours

**Fix**: Modified `scripts/train_qlora.py` to detect multi-GPU mode and use `device_map=None`, allowing PyTorch DDP (Distributed Data Parallel) to properly distribute batches across both GPUs.

## How to Restart Training with Fix

### Step 1: Push and Pull Changes

On your local machine:
```bash
cd /Users/admin/CS-8903-orgaccess
git add scripts/train_qlora.py
git commit -m "Fix multi-GPU training: use DDP instead of model parallelism"
git push
```

On your Qwen RunPod:
```bash
cd /workspace/CS-8903-orgaccess
git pull
```

### Step 2: Stop Current Training

```bash
# Find the training process
ps aux | grep train_qlora

# Kill it (replace <PID> with actual process ID)
kill -9 <PID>

# Or kill all Python processes running the script
pkill -9 -f train_qlora
```

### Step 3: Restart with Multi-GPU Support

You have checkpoint-500, so resume from there:

```bash
cd /workspace/CS-8903-orgaccess

# Method 1: Using torchrun (recommended)
torchrun --nproc_per_node=2 scripts/train_qlora.py \
  --config configs/qwen2_5_32b_qlora.yaml \
  --resume outputs/qwen2_5_32b_orgaccess_qlora/checkpoint-500 \
  --wandb

# Method 2: Using accelerate
accelerate launch --num_processes=2 scripts/train_qlora.py \
  --config configs/qwen2_5_32b_qlora.yaml \
  --resume outputs/qwen2_5_32b_orgaccess_qlora/checkpoint-500 \
  --wandb
```

If using nohup to run in background:
```bash
nohup torchrun --nproc_per_node=2 scripts/train_qlora.py \
  --config configs/qwen2_5_32b_qlora.yaml \
  --resume outputs/qwen2_5_32b_orgaccess_qlora/checkpoint-500 \
  --wandb > qwen_training.log 2>&1 &

# Monitor progress
tail -f qwen_training.log
```

### Step 4: Verify Multi-GPU Utilization

In a separate terminal, watch GPU usage:
```bash
watch -n 1 nvidia-smi
```

**What to expect:**
- Both GPU 0 and GPU 1 should show 80-90% utilization during training
- Training speed: ~1.5-2 seconds per iteration (instead of 6s)
- Total iterations: ~4,746 (not 9,489)
- Estimated time: ~2-2.5 hours remaining (instead of 7 hours)

**Output should show:**
```
✓ Multi-GPU detected: Using DDP (device_map=None)
✓ Model loaded with 4-bit quantization
```

### Step 5: Monitor Training

Check WandB dashboard for real-time metrics:
- Loss curves
- Learning rate schedule
- GPU utilization graphs
- Estimated time remaining

## Expected Results

### Before Fix (Single GPU):
- GPU 0: 90-100% utilization
- GPU 1: 0-5% utilization
- Speed: 6s/step
- Total time: ~17 hours
- Cost: ~$16.50 (17 hrs × 2 GPUs × $0.49/hr)

### After Fix (Multi-GPU DDP):
- GPU 0: 80-90% utilization
- GPU 1: 80-90% utilization
- Speed: 1.5-2s/step
- Total time: ~2 hours
- Cost: ~$2 (2 hrs × 2 GPUs × $0.49/hr)

**Savings**: ~$14.50 and 15 hours of time!

## Troubleshooting

### Issue: Still seeing single GPU usage
**Check**: Ensure you're launching with `torchrun --nproc_per_node=2` or `accelerate launch --num_processes=2`
**Verify**: Output should show "✓ Multi-GPU detected: Using DDP (device_map=None)"

### Issue: CUDA out of memory
**Fix**: Reduce batch size in config from 2 to 1:
```yaml
training:
  per_device_train_batch_size: 1  # Instead of 2
```

### Issue: Process hangs at initialization
**Fix**: This is normal for large models (Qwen2.5-32B). First load can take 5-10 minutes.
**Monitor**: Watch nvidia-smi to see memory allocation happening

### Issue: "Address already in use" error
**Cause**: Previous DDP process not fully killed
**Fix**:
```bash
pkill -9 -f train_qlora
# Wait 10 seconds
torchrun --nproc_per_node=2 ...
```

## After Training Completes

1. **Push checkpoint to HuggingFace Hub** (optional but recommended):
```bash
huggingface-cli login
huggingface-cli upload <your-username>/qwen2.5-32b-orgaccess-qlora \
  outputs/qwen2_5_32b_orgaccess_qlora/final_model
```

2. **Run evaluation**:
```bash
# Start vLLM server with fine-tuned model
vllm serve outputs/qwen2_5_32b_orgaccess_qlora/final_model \
  --port 8000 \
  --gpu-memory-utilization 0.9

# In another terminal, run evaluation
python evaluate_vllm_direct.py \
  --base-url http://localhost:8000/v1 \
  --model-name outputs/qwen2_5_32b_orgaccess_qlora/final_model \
  --benchmark-dir benchmark_data \
  --difficulty hard \
  --batch-workers 16 \
  --max-retries 4
```

3. **Compare results**:
- Llama-3.1-8B: 54.8% accuracy (baseline: 16%)
- Gemma-3-12B: TBD (expected: 60-65%)
- Qwen2.5-32B: TBD (expected: 65-70%)
