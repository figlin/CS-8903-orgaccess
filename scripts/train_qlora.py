#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for OrgAccess RBAC Benchmark

Supports Llama 3.1, Gemma, Mistral, and other instruction-tuned models.
Optimized for OrgAccess permission reasoning task with 25K training examples.

Usage:
    python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml
    python scripts/train_qlora.py --config configs/llama3_1_8b_qlora.yaml --wandb
"""

import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_bnb_config(config: Dict) -> BitsAndBytesConfig:
    """Configure 4-bit quantization for QLoRA"""
    qlora_config = config.get('qlora', {})

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora_config.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_compute_dtype=getattr(torch, qlora_config.get('bnb_4bit_compute_dtype', 'bfloat16')),
        bnb_4bit_use_double_quant=qlora_config.get('bnb_4bit_use_double_quant', True)
    )


def load_model_and_tokenizer(config: Dict):
    """Load base model and tokenizer with quantization"""
    model_config = config['model']
    base_model = model_config['base_model']

    print(f"\n{'='*60}")
    print(f"Loading model: {base_model}")
    print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="right",  # For training
        add_eos_token=True
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    bnb_config = setup_bnb_config(config)

    # For multi-GPU training with QLoRA, we need a specific approach:
    # - With device_map='auto': Model is sharded (model parallelism) but only 1 GPU computes
    # - With device_map=None + DDP: Can't use with quantized models easily
    # - Solution: Use device_map={'': 0} to load model on GPU 0, then let Trainer
    #   replicate it across GPUs for data parallelism

    num_gpus = torch.cuda.device_count()

    # For Qwen2.5-32B with 4-bit quantization: ~20GB, fits on 1x H100 (80GB)
    # Load on GPU 0 and let Trainer replicate across GPUs for proper data parallelism

    if num_gpus > 1:
        # For multi-GPU: try to load on GPU 0 and replicate
        # This enables true data parallelism via Trainer
        device_map_setting = {'': 0}  # Load entire model on GPU 0
        print(f"✓ Multi-GPU Mode: {num_gpus} GPUs available")
        print(f"  Loading model on GPU 0, Trainer will replicate for DataParallel")
        print(f"  This enables proper batch distribution across all GPUs")
    else:
        # Single GPU: use auto
        device_map_setting = 'auto'
        print(f"✓ Single GPU: Using device_map='auto'")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map_setting,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    print(f"✓ Model loaded with 4-bit quantization")
    if hasattr(model, 'hf_device_map'):
        print(f"✓ Device map: {model.hf_device_map}")

    return model, tokenizer


def setup_lora(model, config: Dict):
    """Configure and apply LoRA adapters"""
    lora_config = config['lora']

    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config.get('bias', 'none'),
        task_type=lora_config.get('task_type', 'CAUSAL_LM')
    )

    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    print(f"\n{'='*60}")
    print(f"LoRA Configuration:")
    print(f"  Rank: {lora_config['r']}")
    print(f"  Alpha: {lora_config['lora_alpha']}")
    print(f"  Dropout: {lora_config['lora_dropout']}")
    print(f"  Target modules: {', '.join(lora_config['target_modules'])}")
    print(f"\nTrainable Parameters:")
    print(f"  Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"  Total: {total_params:,}")
    print(f"{'='*60}\n")

    return model


def load_and_prepare_data(config: Dict, tokenizer):
    """Load and tokenize training/validation data from parquet files"""
    data_config = config['data']

    print(f"\n{'='*60}")
    print(f"Loading datasets...")
    print(f"{'='*60}\n")

    # Load datasets from parquet
    dataset = load_dataset(
        'parquet',
        data_files={
            'train': data_config['train_file'],
            'validation': data_config['val_file']
        }
    )

    print(f"✓ Train examples: {len(dataset['train'])}")
    print(f"✓ Validation examples: {len(dataset['validation'])}")

    # Format example into ChatML format
    def format_example(example):
        """Convert OrgAccess example to ChatML format"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a knowledge repository in an organisational structure and you are accessed "
                    "by all the employees in the organisation. Your job is to respond to employees without "
                    "providing any information or approving any task which they do not have the permissions for. "
                    "Analyse the permissions carefully before responding."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Given the following permissions - {example['permissions']} and the following user query - "
                    f"{example['query']}, decide if the query should be full, rejected or partial. Note that the "
                    f"user request must be within the threshold of all the permissions to get a `full` permission. "
                    f"`partial` responses may only be provided if all the conditions are fully satisfied and "
                    f"collaborations or location permissions are partially satisfied. For breach of any other "
                    f"permission by any degree, response must be `rejected`. Mention the response type "
                    f"('full', 'partial', or 'rejected') in your response as well."
                )
            },
            {
                "role": "assistant",
                "content": f"Response type: {example['expected_response']}\n\nRationale: {example['rationale']}"
            }
        ]
        return {"messages": messages}

    # Format datasets
    print("\nFormatting to ChatML...")
    dataset = dataset.map(format_example, desc="Formatting")

    # Tokenization function
    def tokenize_function(examples):
        """Tokenize ChatML format messages"""
        tokenized_inputs = []

        for messages in examples['messages']:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Tokenize
            tokenized = tokenizer(
                text,
                max_length=data_config.get('max_seq_length', 2048),
                truncation=True,
                padding=False,
                return_tensors=None
            )

            tokenized_inputs.append(tokenized)

        # Combine into batch format
        batch = {
            'input_ids': [item['input_ids'] for item in tokenized_inputs],
            'attention_mask': [item['attention_mask'] for item in tokenized_inputs]
        }

        return batch

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )

    # Add labels (copy of input_ids for causal LM)
    def add_labels(examples):
        examples['labels'] = examples['input_ids'].copy()
        return examples

    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

    print(f"✓ Tokenization complete")

    # Print sample stats (safely handle batched data)
    try:
        # Get a small sample to compute stats
        sample_size = min(100, len(tokenized_dataset['train']))
        sample_data = tokenized_dataset['train'].select(range(sample_size))
        sample_lengths = [len(item['input_ids']) for item in sample_data]

        if sample_lengths:
            avg_length = sum(sample_lengths) / len(sample_lengths)
            max_length = max(sample_lengths)
            min_length = min(sample_lengths)

            print(f"\nTokenization statistics (first {sample_size} examples):")
            print(f"  Average length: {avg_length:.0f} tokens")
            print(f"  Max length: {max_length} tokens")
            print(f"  Min length: {min_length} tokens")
    except Exception as e:
        print(f"\n⚠️  Could not compute tokenization stats: {e}")
        print("  Continuing with training anyway...")

    return tokenized_dataset


def setup_training_args(config: Dict) -> TrainingArguments:
    """Configure training arguments"""
    train_config = config['training']

    # Create output directory
    output_dir = Path(train_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup WandB if enabled
    wandb_config = config.get('wandb', {})
    report_to = ['wandb'] if wandb_config.get('enabled', False) else ['tensorboard']

    args = TrainingArguments(
        # Output
        output_dir=str(output_dir),
        run_name=train_config.get('run_name', 'orgaccess-finetuning'),

        # Training schedule
        num_train_epochs=train_config['num_train_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],

        # Optimization
        learning_rate=train_config['learning_rate'],
        lr_scheduler_type=train_config['lr_scheduler_type'],
        warmup_ratio=train_config.get('warmup_ratio', 0.03),
        weight_decay=train_config['weight_decay'],
        max_grad_norm=train_config['max_grad_norm'],
        optim=train_config.get('optim', 'paged_adamw_8bit'),

        # Precision
        bf16=train_config.get('bf16', True),
        fp16=train_config.get('fp16', False),

        # Logging
        logging_steps=train_config['logging_steps'],
        report_to=report_to,

        # Evaluation
        eval_strategy=train_config.get('eval_strategy', train_config.get('evaluation_strategy', 'steps')),
        eval_steps=train_config['eval_steps'],

        # Checkpointing
        save_strategy=train_config['save_strategy'],
        save_steps=train_config['save_steps'],
        save_total_limit=train_config.get('save_total_limit', 3),

        # Reproducibility
        seed=train_config.get('seed', 42),
        data_seed=train_config.get('data_seed', 42),

        # Performance
        dataloader_num_workers=4,
        remove_unused_columns=False,

        # Best model tracking
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
    )

    # Add early stopping parameters if specified in config
    if 'early_stopping_patience' in train_config:
        args.load_best_model_at_end = True
        args.metric_for_best_model = train_config.get('metric_for_best_model', 'eval_loss')
        args.greater_is_better = train_config.get('greater_is_better', False)

    return args


def save_training_config(config: Dict, output_dir: Path):
    """Save configuration for reproducibility"""
    config_save_path = output_dir / 'training_config.json'

    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Configuration saved to: {config_save_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLM with QLoRA for OrgAccess RBAC')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases tracking'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint directory'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"\n{'='*60}")
    print(f"OrgAccess RBAC Fine-Tuning with QLoRA")
    print(f"{'='*60}")
    print(f"\nConfiguration: {args.config}")

    config = load_config(args.config)

    # Override WandB setting if specified
    if args.wandb:
        config.setdefault('wandb', {})['enabled'] = True

    # Initialize WandB if enabled
    if config.get('wandb', {}).get('enabled', False):
        import wandb
        wandb.init(
            project=config['wandb'].get('project', 'orgaccess-finetuning'),
            name=config['training'].get('run_name', 'orgaccess-finetuning'),
            config=config
        )
        print("✓ WandB initialized")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Setup LoRA
    model = setup_lora(model, config)

    # Load and prepare data
    tokenized_dataset = load_and_prepare_data(config, tokenizer)

    # Setup training arguments
    training_args = setup_training_args(config)

    # Save configuration
    save_training_config(config, Path(training_args.output_dir))

    # Data collator with proper padding for variable-length sequences
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,  # Ignore padding tokens in loss calculation
        pad_to_multiple_of=8  # Pad to multiple of 8 for efficiency
    )

    # Setup callbacks (early stopping if configured)
    callbacks = []
    train_config = config['training']
    if 'early_stopping_patience' in train_config:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=train_config['early_stopping_patience'],
            early_stopping_threshold=train_config.get('early_stopping_threshold', 0.0)
        )
        callbacks.append(early_stopping)
        print(f"✓ Early stopping enabled: patience={train_config['early_stopping_patience']}, "
              f"threshold={train_config.get('early_stopping_threshold', 0.0)}")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Train!
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # Save final model
    print(f"\n{'='*60}")
    print(f"Saving final model...")
    print(f"{'='*60}\n")

    output_dir = Path(training_args.output_dir)
    final_model_dir = output_dir / "final_model"

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"✓ Model saved to: {final_model_dir}")

    # Save training metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"✓ Training metrics saved to: {metrics_path}")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}\n")
    print(f"Output directory: {output_dir}")
    print(f"Final model: {final_model_dir}")

    # Finish WandB run
    if config.get('wandb', {}).get('enabled', False):
        wandb.finish()


if __name__ == '__main__':
    main()
