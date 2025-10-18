# OrgAccess Fine-Tuning Research Methodology

## Overview

This document outlines the methodology for fine-tuning language models on the OrgAccess benchmark and evaluating them for role-based access control (RBAC) performance in organizational settings. This research aims to demonstrate that fine-tuning on easy/medium difficulty examples improves model performance on hard test cases.

## Dataset Description

### OrgAccess Benchmark Structure

The OrgAccess benchmark is a synthetic dataset designed to evaluate LLMs' ability to understand and operate within organizational hierarchies and role-based access control policies.

**Dataset Statistics:**
- **Easy**: 38,280 examples
- **Medium**: 6,907 examples
- **Hard**: 10,613 examples
- **Total**: 55,800 examples

### Data Schema

Each example contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `user_role` | string | The role of the user making the request (e.g., "Legal_Counsel", "Junior_Software_Engineer") |
| `permissions` | string (JSON) | Dictionary detailing role permissions including departments, access levels, allowed actions, timeouts, deadlines, location restrictions, etc. |
| `query` | string | Natural language query from the user requesting access or actions |
| `expected_response` | string | Expected access control decision: `"full"`, `"rejected"`, or `"partial"` |
| `rationale` | string | Explanation for the expected response, highlighting specific permissions violated or satisfied |
| `id` | string | Unique identifier (nullable) |
| `query_timestamp` | string | Timestamp of query (nullable) |

### Permission Types

The benchmark defines **40 distinct types of permissions** commonly found in organizational structures:

**Core Permission Categories:**
- **Departments**: List of departments user can access
- **Access Level**: Hierarchical access tier (e.g., "Intern", "Restricted", "Manager", "Senior", "High")
- **Allowed Actions**: Specific operations permitted (e.g., "read", "write", "execute", "view_reports", "modify_data")
- **Session Timeout**: Maximum session duration in minutes
- **Deadline**: Date-based access restrictions
- **Location Restriction**: Geographic constraints (countries, regions, cities)
- **Automation Restriction**: Whether automation is allowed (boolean)
- **Collaboration Access**: Departments/teams user can collaborate with
- **Retention Period**: Data retention time limits
- **Client Restriction**: Client-specific access controls
- **Disaster Mode**: Emergency access protocols
- **Password Rotation**: Password policy requirements
- **Biometric Required**: Additional authentication requirements

### Response Categories

Models must classify requests into three categories:

1. **Full (`"full"`)**: All requested actions are within user permissions
2. **Partial (`"partial"`)**: Some but not all requested actions can be fulfilled (typically for collaboration or location partial matches)
3. **Rejected (`"rejected"`)**: Request violates one or more critical permissions

## Research Design

### Training/Testing Split Strategy

The proposed methodology uses a stratified approach to maximize learning from simple cases while testing on complex scenarios:

#### Training Data (70% of Easy + Medium)
```
Easy Training:     38,280 × 0.70 = 26,796 examples
Medium Training:    6,907 × 0.70 =  4,835 examples
Total Training:                    31,631 examples
```

Split training data 80-20:
- **Train Set**: 31,631 × 0.80 = 25,305 examples
- **Validation Set**: 31,631 × 0.20 = 6,326 examples

#### Testing Data (30% of Easy/Medium + All Hard)
```
Easy Test:         38,280 × 0.30 = 11,484 examples
Medium Test:        6,907 × 0.30 =  2,072 examples
Hard Test:         10,613 × 1.00 = 10,613 examples
Total Test:                        24,169 examples
```

### Rationale for This Split

1. **Easy/Medium for Training**: These examples establish fundamental RBAC understanding
   - Easy: Single or simple permission constraints
   - Medium: Multiple interacting permissions with moderate complexity

2. **Hard for Testing**: Complex real-world scenarios with:
   - Multiple overlapping permissions
   - Conflicting constraints
   - Nuanced edge cases
   - Long, multi-faceted queries requiring compositional reasoning

3. **Holdout from Easy/Medium**: The 30% holdout validates generalization within difficulty levels

## Evaluation Methodology

### Testing Against Models on RunPod

RunPod is a cloud GPU platform that allows you to host and serve models cost-effectively. Here's how to deploy and test your models on RunPod.

#### RunPod Deployment Options

**Option 1: RunPod Serverless (Recommended for Testing)**
- Auto-scaling GPU inference
- Pay per second of compute
- No idle costs
- Built-in API endpoints

**Option 2: RunPod Pods (For Training/Fine-Tuning)**
- Dedicated GPU instances
- Full control over environment
- SSH access for development
- More cost-effective for long-running jobs

#### Setting Up a Model on RunPod Serverless

##### Step 1: Create a RunPod Account
```bash
# Sign up at https://www.runpod.io/
# Add credits to your account
# Get your API key from https://www.runpod.io/console/user/settings
```

##### Step 2: Deploy Pre-Built vLLM Endpoint (Fastest Method)

RunPod has pre-configured vLLM (fast inference server) templates:

1. Go to RunPod Serverless → Browse Templates
2. Select **"vLLM - Latest"** template
3. Configure your model:
   - **Model Name**: `Qwen/Qwen2.5-7B-Instruct` (or your fine-tuned model on HF)
   - **GPU Type**: RTX 4090 or A40 (good for 7B models)
   - **Worker Count**: Start with 1-2
   - **Max Workers**: Set to 5-10 for auto-scaling

4. Deploy and copy your endpoint URL

##### Step 3: Alternative - Deploy Custom Model with Docker

For fine-tuned models or custom configurations:

```dockerfile
# Dockerfile for RunPod
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Install dependencies
RUN pip install vllm transformers accelerate

# Copy your model (if local) or download from HuggingFace
WORKDIR /workspace
ENV MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Start vLLM server
CMD python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

Build and push to Docker Hub:
```bash
docker build -t your-username/orgaccess-model:v1 .
docker push your-username/orgaccess-model:v1
```

Then deploy on RunPod using your custom Docker image.

#### Testing Against RunPod Endpoints

##### Method 1: Using OpenAI-Compatible API (vLLM)

RunPod's vLLM endpoints are OpenAI API compatible:

```python
from openai import OpenAI
import os

# Configure client for RunPod endpoint
client = OpenAI(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url="https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
)

def evaluate_runpod_model(endpoint_id, test_dataset, max_samples=None):
    """
    Evaluate model hosted on RunPod serverless endpoint.

    Args:
        endpoint_id: Your RunPod endpoint ID (e.g., "abc123def456")
        test_dataset: Dataset to evaluate
        max_samples: Limit number of samples (None = all)
    """
    predictions = []
    ground_truth = []

    samples = test_dataset if max_samples is None else test_dataset.select(range(max_samples))

    for i, item in enumerate(samples):
        messages = [
            {
                "role": "system",
                "content": "You are a knowledge repository in an organisational structure and you are accessed by all the employees in the organisation. Your job is to respond to employees without providing any information or approving any task which they do not have the permissions for. Analyse the permissions carefully before responding."
            },
            {
                "role": "user",
                "content": f"Given the following permissions - {item['permissions']} and the following user query - {item['query']}, decide if the query should be full, rejected or partial. Note that the user request must be within the threshold of all the permissions to get a `full` permission. `partial` responses may only be provided if all the conditions are fully satisfied and collaborations or location permissions are partially satisfied. For breach of any other permission by any degree, response must be `rejected`. Mention the response type ('full', 'partial', or 'rejected') in your response as well."
            }
        ]

        try:
            response = client.chat.completions.create(
                model="your-model-name",  # Model name from deployment
                messages=messages,
                temperature=0.0,
                max_tokens=512
            )

            prediction = extract_label(response.choices[0].message.content)
            predictions.append(prediction)
            ground_truth.append(item['expected_response'])

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} examples")

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append('rejected')
            ground_truth.append(item['expected_response'])

    return predictions, ground_truth

# Usage
endpoint_id = "your-endpoint-id"  # From RunPod console
from datasets import load_dataset

test_ds = load_dataset('parquet', data_files='data/hard-00000-of-00001.parquet')['train']
predictions, ground_truth = evaluate_runpod_model(endpoint_id, test_ds, max_samples=100)
```

##### Method 2: Using RunPod SDK Directly

```python
import runpod
from datasets import load_dataset
import json

# Initialize RunPod
runpod.api_key = "your-runpod-api-key"

def evaluate_runpod_sdk(endpoint_id, test_dataset, max_samples=None):
    """
    Evaluate using RunPod SDK for custom endpoints.
    """
    endpoint = runpod.Endpoint(endpoint_id)

    predictions = []
    ground_truth = []

    samples = test_dataset if max_samples is None else test_dataset.select(range(max_samples))

    for i, item in enumerate(samples):
        # Prepare input
        input_data = {
            "prompt": f"System: You are a knowledge repository in an organisational structure and you are accessed by all the employees in the organisation. Your job is to respond to employees without providing any information or approving any task which they do not have the permissions for. Analyse the permissions carefully before responding.\n\nUser: Given the following permissions - {item['permissions']} and the following user query - {item['query']}, decide if the query should be full, rejected or partial. Note that the user request must be within the threshold of all the permissions to get a `full` permission. `partial` responses may only be provided if all the conditions are fully satisfied and collaborations or location permissions are partially satisfied. For breach of any other permission by any degree, response must be `rejected`. Mention the response type ('full', 'partial', or 'rejected') in your response as well.",
            "max_tokens": 512,
            "temperature": 0.0
        }

        try:
            # Run inference
            result = endpoint.run_sync(input_data, timeout=60)

            # Extract response
            response_text = result.get('output', {}).get('text', '')
            prediction = extract_label(response_text)

            predictions.append(prediction)
            ground_truth.append(item['expected_response'])

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} examples")

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append('rejected')
            ground_truth.append(item['expected_response'])

    return predictions, ground_truth

# Install SDK first: pip install runpod
```

##### Method 3: Direct HTTP Requests (No Dependencies)

```python
import requests
import json
from datasets import load_dataset

def evaluate_runpod_http(endpoint_url, api_key, test_dataset, max_samples=None):
    """
    Evaluate using direct HTTP requests to RunPod endpoint.

    Args:
        endpoint_url: Full endpoint URL (e.g., "https://api.runpod.ai/v2/abc123/runsync")
        api_key: Your RunPod API key
        test_dataset: Dataset to evaluate
        max_samples: Limit samples
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    predictions = []
    ground_truth = []

    samples = test_dataset if max_samples is None else test_dataset.select(range(max_samples))

    for i, item in enumerate(samples):
        # Prepare payload
        payload = {
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a knowledge repository in an organisational structure and you are accessed by all the employees in the organisation. Your job is to respond to employees without providing any information or approving any task which they do not have the permissions for. Analyse the permissions carefully before responding."
                    },
                    {
                        "role": "user",
                        "content": f"Given the following permissions - {item['permissions']} and the following user query - {item['query']}, decide if the query should be full, rejected or partial. Note that the user request must be within the threshold of all the permissions to get a `full` permission. `partial` responses may only be provided if all the conditions are fully satisfied and collaborations or location permissions are partially satisfied. For breach of any other permission by any degree, response must be `rejected`. Mention the response type ('full', 'partial', or 'rejected') in your response as well."
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.0
            }
        }

        try:
            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            response_text = result.get('output', {}).get('choices', [{}])[0].get('message', {}).get('content', '')

            prediction = extract_label(response_text)
            predictions.append(prediction)
            ground_truth.append(item['expected_response'])

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} examples")

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append('rejected')
            ground_truth.append(item['expected_response'])

    return predictions, ground_truth

# Usage
endpoint_url = "https://api.runpod.ai/v2/your-endpoint-id/runsync"
api_key = "your-api-key"

test_ds = load_dataset('parquet', data_files='data/hard-00000-of-00001.parquet')['train']
predictions, ground_truth = evaluate_runpod_http(endpoint_url, api_key, test_ds, max_samples=100)
```

#### Fine-Tuning on RunPod Pods

For training your fine-tuned model:

##### Step 1: Launch a Pod

1. Go to RunPod → Pods → GPU Instances
2. Select GPU: **A40 (48GB)** or **A100 (80GB)** for 7B models
3. Choose template: **RunPod Pytorch 2.1** or **RunPod Fast Stable Diffusion**
4. Add volume storage: 50GB minimum
5. Launch pod and connect via SSH or Web Terminal

##### Step 2: Setup Environment

```bash
# SSH into your pod
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519

# Update and install dependencies
apt-get update
pip install transformers datasets peft accelerate bitsandbytes wandb

# Clone your repository
git clone <your-repo-url>
cd CS-8903-orgaccess
```

##### Step 3: Run Training Script

```python
# train_on_runpod.py
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load base model
model_id = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and prepare data
ds = load_dataset('parquet', data_files={
    'easy': 'data/easy-00000-of-00001.parquet',
    'medium': 'data/medium-00000-of-00001.parquet'
})

# Create 70% split
easy_shuffled = ds['easy'].shuffle(seed=42)
easy_70_idx = int(len(easy_shuffled) * 0.70)
easy_train_val = easy_shuffled.select(range(easy_70_idx))

medium_shuffled = ds['medium'].shuffle(seed=42)
medium_70_idx = int(len(medium_shuffled) * 0.70)
medium_train_val = medium_shuffled.select(range(medium_70_idx))

# Combine and split 80-20
train_val_combined = concatenate_datasets([easy_train_val, medium_train_val]).shuffle(seed=42)
train_idx = int(len(train_val_combined) * 0.80)
train_dataset = train_val_combined.select(range(train_idx))
val_dataset = train_val_combined.select(range(train_idx, len(train_val_combined)))

# Format function
def format_example(example):
    messages = [
        {"role": "system", "content": "You are a knowledge repository in an organisational structure and you are accessed by all the employees in the organisation. Your job is to respond to employees without providing any information or approving any task which they do not have the permissions for. Analyse the permissions carefully before responding."},
        {"role": "user", "content": f"Given the following permissions - {example['permissions']} and the following user query - {example['query']}, decide if the query should be full, rejected or partial. Note that the user request must be within the threshold of all the permissions to get a `full` permission. `partial` responses may only be provided if all the conditions are fully satisfied and collaborations or location permissions are partially satisfied. For breach of any other permission by any degree, response must be `rejected`. Mention the response type ('full', 'partial', or 'rejected') in your response as well."},
        {"role": "assistant", "content": f"Response type: {example['expected_response']}\n\nRationale: {example['rationale']}"}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# Format datasets
train_formatted = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
val_formatted = val_dataset.map(format_example, remove_columns=val_dataset.column_names)

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

train_tokenized = train_formatted.map(tokenize_function, batched=True)
val_tokenized = val_formatted.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./orgaccess-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",  # Optional: track with Weights & Biases
    run_name="orgaccess-qwen-lora"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator
)

# Train
print("Starting training...")
trainer.train()

# Save model
output_dir = "./orgaccess-qwen-lora-final"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# Upload to Hugging Face (optional)
# from huggingface_hub import login
# login(token="your-hf-token")
# model.push_to_hub("your-username/orgaccess-qwen-lora")
# tokenizer.push_to_hub("your-username/orgaccess-qwen-lora")
```

Run the training:
```bash
python train_on_runpod.py
```

##### Step 4: Deploy Fine-Tuned Model

After training, deploy your model:

1. **Upload to Hugging Face Hub** (recommended):
   ```python
   from huggingface_hub import login
   login(token="your-token")

   model.push_to_hub("your-username/orgaccess-qwen-lora")
   tokenizer.push_to_hub("your-username/orgaccess-qwen-lora")
   ```

2. **Create RunPod Serverless Endpoint**:
   - Use vLLM template
   - Set model name to: `your-username/orgaccess-qwen-lora`
   - Deploy and test

#### Cost Optimization for RunPod

**For Testing/Inference (Serverless):**
- RTX 4090: ~$0.40/hour (good for 7B models)
- RTX A6000: ~$0.79/hour (better throughput)
- Only pay when running inference

**For Training (Pods):**
- A40 (48GB): ~$0.79/hour (sufficient for 7B with LoRA)
- A100 (80GB): ~$1.89/hour (faster, can train larger models)
- Stop pod when not training to save costs

**Budget Example for Research:**
- Training (3 epochs on 25K examples): ~4-6 hours on A40 = $3-5
- Testing (10K examples): ~2-3 hours on RTX 4090 serverless = $1-2
- **Total**: Under $10 for complete experiment

#### Complete RunPod Evaluation Script

```python
# evaluate_runpod.py
import os
from openai import OpenAI
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

def extract_label(response_text):
    """Extract label from model response."""
    response_lower = response_text.lower()
    if 'rejected' in response_lower:
        return 'rejected'
    elif 'partial' in response_lower:
        return 'partial'
    elif 'full' in response_lower:
        return 'full'
    return 'rejected'

def evaluate_runpod_model(endpoint_id, model_name, test_dataset, max_samples=None):
    """Evaluate model on RunPod."""
    # Configure OpenAI client for RunPod
    client = OpenAI(
        api_key=os.getenv("RUNPOD_API_KEY"),
        base_url=f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
    )

    predictions = []
    ground_truth = []

    samples = test_dataset if max_samples is None else test_dataset.select(range(max_samples))

    print(f"Evaluating {len(samples)} examples...")

    for i, item in enumerate(samples):
        messages = [
            {
                "role": "system",
                "content": "You are a knowledge repository in an organisational structure and you are accessed by all the employees in the organisation. Your job is to respond to employees without providing any information or approving any task which they do not have the permissions for. Analyse the permissions carefully before responding."
            },
            {
                "role": "user",
                "content": f"Given the following permissions - {item['permissions']} and the following user query - {item['query']}, decide if the query should be full, rejected or partial. Note that the user request must be within the threshold of all the permissions to get a `full` permission. `partial` responses may only be provided if all the conditions are fully satisfied and collaborations or location permissions are partially satisfied. For breach of any other permission by any degree, response must be `rejected`. Mention the response type ('full', 'partial', or 'rejected') in your response as well."
            }
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=512
            )

            prediction = extract_label(response.choices[0].message.content)
            predictions.append(prediction)
            ground_truth.append(item['expected_response'])

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} examples")

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append('rejected')
            ground_truth.append(item['expected_response'])

    return predictions, ground_truth

def calculate_metrics(predictions, ground_truth):
    """Calculate evaluation metrics."""
    label_map = {'full': 0, 'partial': 1, 'rejected': 2}
    y_true = [label_map[gt] for gt in ground_truth]
    y_pred = [label_map[pred] for pred in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (Macro): {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    print("\n" + "-"*60)
    print("Per-Class Metrics:")
    print("-"*60)
    print(classification_report(
        y_true,
        y_pred,
        target_names=['full', 'partial', 'rejected'],
        digits=4
    ))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

if __name__ == "__main__":
    # Configuration
    RUNPOD_ENDPOINT_ID = "your-endpoint-id"  # Get from RunPod console
    MODEL_NAME = "your-username/orgaccess-qwen-lora"

    # Load test data
    print("Loading test dataset...")
    test_ds = load_dataset('parquet', data_files='data/hard-00000-of-00001.parquet')['train']

    # Run evaluation
    predictions, ground_truth = evaluate_runpod_model(
        RUNPOD_ENDPOINT_ID,
        MODEL_NAME,
        test_ds,
        max_samples=100  # Remove or set to None for full evaluation
    )

    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)

    # Save results
    results = {
        'endpoint_id': RUNPOD_ENDPOINT_ID,
        'model_name': MODEL_NAME,
        'test_size': len(predictions),
        'metrics': metrics
    }

    with open('runpod_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to runpod_results.json")
```

Run the evaluation:
```bash
# Set your RunPod API key
export RUNPOD_API_KEY="your-api-key"

# Run evaluation
python evaluate_runpod.py
```

### Testing Against a Model

The evaluation process involves:

#### 1. Model Configuration

**Option A: OpenAI API Models**
```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Requires OPENAI_API_KEY in .env
client = OpenAI()

model_id = "gpt-4.1"  # or "gpt-3.5-turbo", etc.
```

**Option B: Local Hugging Face Models**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = 'Qwen/Qwen2.5-7B-Instruct'
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir='cache_dir',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0'
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()
```

#### 2. Prompt Template

The evaluation uses a structured prompt format:

**System Prompt:**
```
You are a knowledge repository in an organisational structure and you are
accessed by all the employees in the organisation. Your job is to respond
to employees without providing any information or approving any task which
they do not have the permissions for. Analyse the permissions carefully
before responding.
```

**User Prompt:**
```
Given the following permissions - {permissions} and the following user
query - {query}, decide if the query should be full, rejected or partial.
Note that the user request must be within the threshold of all the
permissions to get a `full` permission. `partial` responses may only be
provided if all the conditions are fully satisfied and collaborations or
location permissions are partially satisfied. For breach of any other
permission by any degree, response must be `rejected`. Mention the response
type ('full', 'partial', or 'rejected') in your response as well.
```

#### 3. Inference Process

**For OpenAI API:**
```python
def evaluate_permissions(test_cases):
    predictions = []
    for item in test_cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                permissions=item['permissions'],
                query=item['query']
            )}
        ]

        response = client.responses.create(
            model=model_id,
            input=messages
        )

        # Extract prediction from response
        prediction = extract_label(response.output_text)
        predictions.append(prediction)

    return predictions
```

**For Local Models:**
```python
def evaluate_permissions(test_cases):
    predictions = []
    for item in test_cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                permissions=item['permissions'],
                query=item['query']
            )}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer(text, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model.generate(**model_inputs, max_new_tokens=1024)
            model_response = tokenizer.batch_decode(
                [outputs[0][len(model_inputs.input_ids[0]):]],
                skip_special_tokens=True
            )[0]

        prediction = extract_label(model_response)
        predictions.append(prediction)

    return predictions
```

#### 4. Response Parsing

Models may generate verbose responses. Extract the label:

```python
def extract_label(response_text):
    """
    Extract 'full', 'partial', or 'rejected' from model response.
    """
    response_lower = response_text.lower()

    # Look for explicit mentions
    if 'rejected' in response_lower:
        return 'rejected'
    elif 'partial' in response_lower:
        return 'partial'
    elif 'full' in response_lower:
        return 'full'
    else:
        # Fallback: return most conservative (rejected)
        return 'rejected'
```

#### 5. Metrics

Evaluate using standard classification metrics:

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Map labels to integers
label_map = {'full': 0, 'partial': 1, 'rejected': 2}
y_true = [label_map[item['expected_response']] for item in test_cases]
y_pred = [label_map[pred] for pred in predictions]

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 (Macro): {f1_macro:.4f}")
print(f"F1 (Weighted): {f1_weighted:.4f}")

# Detailed per-class metrics
print(classification_report(
    y_true,
    y_pred,
    target_names=['full', 'partial', 'rejected']
))
```

### Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Test Dataset                         │
│           (Hard split: 10,613 examples)                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              For Each Test Example:                          │
│  1. Format permissions + query into prompt                   │
│  2. Send to model (OpenAI API or local)                      │
│  3. Parse response to extract label                          │
│  4. Compare to expected_response                             │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Calculate Metrics:                            │
│  - Accuracy                                                  │
│  - F1 Score (Macro & Weighted)                               │
│  - Per-class Precision/Recall/F1                             │
│  - Confusion Matrix                                          │
└─────────────────────────────────────────────────────────────┘
```

## Fine-Tuning Process

### Data Preparation

#### 1. Create Train/Val/Test Splits

```python
from datasets import load_dataset
import random

# Load datasets
ds = load_dataset('parquet', data_files={
    'easy': 'data/easy-00000-of-00001.parquet',
    'medium': 'data/medium-00000-of-00001.parquet',
    'hard': 'data/hard-00000-of-00001.parquet'
})

# Shuffle and split easy
easy_shuffled = ds['easy'].shuffle(seed=42)
easy_70_idx = int(len(easy_shuffled) * 0.70)
easy_train_val = easy_shuffled.select(range(easy_70_idx))
easy_test = easy_shuffled.select(range(easy_70_idx, len(easy_shuffled)))

# Shuffle and split medium
medium_shuffled = ds['medium'].shuffle(seed=42)
medium_70_idx = int(len(medium_shuffled) * 0.70)
medium_train_val = medium_shuffled.select(range(medium_70_idx))
medium_test = medium_shuffled.select(range(medium_70_idx, len(medium_shuffled)))

# Combine easy and medium train_val
from datasets import concatenate_datasets
train_val_combined = concatenate_datasets([easy_train_val, medium_train_val])
train_val_combined = train_val_combined.shuffle(seed=42)

# 80-20 split for train/val
train_idx = int(len(train_val_combined) * 0.80)
train_dataset = train_val_combined.select(range(train_idx))
val_dataset = train_val_combined.select(range(train_idx, len(train_val_combined)))

# Test dataset: 30% easy + 30% medium + all hard
test_dataset = concatenate_datasets([easy_test, medium_test, ds['hard']])

print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")
```

#### 2. Format for Fine-Tuning

Convert to chat format for instruction tuning:

```python
def format_example(example):
    """
    Format example for instruction tuning.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a knowledge repository in an organisational structure and you are accessed by all the employees in the organisation. Your job is to respond to employees without providing any information or approving any task which they do not have the permissions for. Analyse the permissions carefully before responding."
        },
        {
            "role": "user",
            "content": f"Given the following permissions - {example['permissions']} and the following user query - {example['query']}, decide if the query should be full, rejected or partial. Note that the user request must be within the threshold of all the permissions to get a `full` permission. `partial` responses may only be provided if all the conditions are fully satisfied and collaborations or location permissions are partially satisfied. For breach of any other permission by any degree, response must be `rejected`. Mention the response type ('full', 'partial', or 'rejected') in your response as well."
        },
        {
            "role": "assistant",
            "content": f"Response type: {example['expected_response']}\n\nRationale: {example['rationale']}"
        }
    ]
    return {"messages": messages}

train_dataset = train_dataset.map(format_example)
val_dataset = val_dataset.map(format_example)
```

### Fine-Tuning Approaches

#### Option 1: OpenAI Fine-Tuning API

```python
from openai import OpenAI
import jsonlines

client = OpenAI()

# Save to JSONL format
def save_to_jsonl(dataset, filename):
    with jsonlines.open(filename, mode='w') as writer:
        for example in dataset:
            writer.write({"messages": example["messages"]})

save_to_jsonl(train_dataset, "train.jsonl")
save_to_jsonl(val_dataset, "val.jsonl")

# Upload files
train_file = client.files.create(
    file=open("train.jsonl", "rb"),
    purpose="fine-tune"
)

val_file = client.files.create(
    file=open("val.jsonl", "rb"),
    purpose="fine-tune"
)

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 4,
        "learning_rate_multiplier": 1.0
    }
)

print(f"Fine-tuning job created: {job.id}")
```

#### Option 2: Hugging Face Transformers (LoRA/QLoRA)

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load base model
model_id = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize datasets
def tokenize_function(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    return tokenizer(
        texts,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

val_tokenized = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./orgaccess-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    bf16=True,
    gradient_checkpointing=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized
)

# Train
trainer.train()

# Save model
model.save_pretrained("./orgaccess-finetuned-final")
tokenizer.save_pretrained("./orgaccess-finetuned-final")
```

### Hyperparameter Tuning

Key hyperparameters to consider:

| Hyperparameter | Range | Description |
|----------------|-------|-------------|
| Learning Rate | 1e-5 to 5e-4 | Controls adaptation speed |
| Batch Size | 2-8 | Memory vs. gradient stability trade-off |
| Epochs | 2-5 | Prevent overfitting on training data |
| LoRA Rank (r) | 8-64 | Model capacity for adaptation |
| LoRA Alpha | 16-64 | Scaling of LoRA updates |
| Max Sequence Length | 1024-2048 | Handle long queries |

## Evaluation Plan

### Baseline Models

Test performance of pre-trained models without fine-tuning:
- GPT-3.5-turbo
- GPT-4
- Llama-2-7B-Chat
- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct

### Fine-Tuned Models

Train and evaluate:
- GPT-3.5-turbo (fine-tuned)
- Llama-2-7B-Chat + LoRA
- Qwen2.5-7B-Instruct + LoRA

### Comparison Metrics

For each model, calculate on **Hard test set**:

1. **Accuracy**: Overall classification accuracy
2. **Macro F1**: Equal weight for all three classes
3. **Weighted F1**: Weight by class frequency
4. **Per-Class Metrics**:
   - Precision for `full`, `partial`, `rejected`
   - Recall for `full`, `partial`, `rejected`
   - F1 for `full`, `partial`, `rejected`
5. **Confusion Matrix**: Identify systematic errors

### Statistical Analysis

- Paired t-tests comparing baseline vs. fine-tuned
- Error analysis: categorize failure modes
- Ablation studies: train on Easy only vs. Medium only vs. combined

## Expected Results

### Hypotheses

**H1**: Fine-tuned models will outperform baseline models on hard examples
- Baseline models lack organizational context understanding
- Fine-tuning builds compositional reasoning for permissions

**H2**: Combined Easy+Medium training outperforms single difficulty
- Easy examples establish fundamentals
- Medium examples bridge to complex reasoning

**H3**: Models will struggle more with `partial` classification
- Partial requires nuanced understanding
- Binary decisions (full/rejected) are simpler

### Performance Targets

**Baseline Models:**
- Accuracy: 40-60% on hard examples
- Macro F1: 0.35-0.55

**Fine-Tuned Models:**
- Accuracy: 65-80% on hard examples
- Macro F1: 0.60-0.75

## Implementation Code

### Complete Evaluation Script

```python
# evaluate_model.py
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from openai import OpenAI
from dotenv import load_dotenv
import json
import re

load_dotenv()

def extract_label(response_text):
    """Extract label from model response."""
    response_lower = response_text.lower()
    if 'rejected' in response_lower:
        return 'rejected'
    elif 'partial' in response_lower:
        return 'partial'
    elif 'full' in response_lower:
        return 'full'
    return 'rejected'  # Conservative default

def evaluate_openai_model(model_id, test_dataset, max_samples=None):
    """Evaluate OpenAI model on test dataset."""
    client = OpenAI()

    predictions = []
    ground_truth = []

    samples = test_dataset if max_samples is None else test_dataset.select(range(max_samples))

    for i, item in enumerate(samples):
        messages = [
            {
                "role": "system",
                "content": "You are a knowledge repository in an organisational structure and you are accessed by all the employees in the organisation. Your job is to respond to employees without providing any information or approving any task which they do not have the permissions for. Analyse the permissions carefully before responding."
            },
            {
                "role": "user",
                "content": f"Given the following permissions - {item['permissions']} and the following user query - {item['query']}, decide if the query should be full, rejected or partial. Note that the user request must be within the threshold of all the permissions to get a `full` permission. `partial` responses may only be provided if all the conditions are fully satisfied and collaborations or location permissions are partially satisfied. For breach of any other permission by any degree, response must be `rejected`. Mention the response type ('full', 'partial', or 'rejected') in your response as well."
            }
        ]

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.0
            )

            prediction = extract_label(response.choices[0].message.content)
            predictions.append(prediction)
            ground_truth.append(item['expected_response'])

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(samples)} examples")

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append('rejected')  # Default
            ground_truth.append(item['expected_response'])

    return predictions, ground_truth

def calculate_metrics(predictions, ground_truth):
    """Calculate and print metrics."""
    label_map = {'full': 0, 'partial': 1, 'rejected': 2}
    y_true = [label_map[gt] for gt in ground_truth]
    y_pred = [label_map[pred] for pred in predictions]

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (Macro): {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    print("\n" + "-"*60)
    print("Per-Class Metrics:")
    print("-"*60)
    print(classification_report(
        y_true,
        y_pred,
        target_names=['full', 'partial', 'rejected'],
        digits=4
    ))
    print("\n" + "-"*60)
    print("Confusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(y_true, y_pred)
    print("             Predicted")
    print("             Full  Partial  Rejected")
    print(f"Actual Full      {cm[0][0]:4d}    {cm[0][1]:4d}      {cm[0][2]:4d}")
    print(f"      Partial    {cm[1][0]:4d}    {cm[1][1]:4d}      {cm[1][2]:4d}")
    print(f"      Rejected   {cm[2][0]:4d}    {cm[2][1]:4d}      {cm[2][2]:4d}")
    print("="*60)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

if __name__ == "__main__":
    # Load test data
    print("Loading test dataset...")
    test_ds = load_dataset('parquet', data_files='data/hard-00000-of-00001.parquet')['train']

    # Evaluate model
    model_id = "gpt-3.5-turbo"  # or your fine-tuned model ID
    print(f"\nEvaluating model: {model_id}")
    print(f"Test set size: {len(test_ds)}")

    predictions, ground_truth = evaluate_openai_model(model_id, test_ds, max_samples=100)

    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)

    # Save results
    results = {
        'model_id': model_id,
        'test_size': len(predictions),
        'metrics': metrics
    }

    with open(f'results_{model_id.replace("/", "_")}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results_{model_id.replace('/', '_')}.json")
```

## Research Paper Structure

### Suggested Outline

1. **Introduction**
   - Motivation: LLMs in enterprise settings
   - Challenge: RBAC compliance
   - Contribution: Demonstrate fine-tuning improves complex reasoning

2. **Related Work**
   - LLMs for code/reasoning
   - Access control in AI systems
   - Instruction tuning and fine-tuning

3. **Methodology**
   - Dataset description (OrgAccess)
   - Training/testing split strategy
   - Fine-tuning approach
   - Evaluation metrics

4. **Experiments**
   - Baseline model performance
   - Fine-tuned model performance
   - Ablation studies
   - Error analysis

5. **Results**
   - Quantitative metrics (tables/charts)
   - Statistical significance tests
   - Qualitative examples (success/failure cases)

6. **Discussion**
   - Why fine-tuning helps
   - Limitations and failure modes
   - Implications for enterprise AI

7. **Conclusion**
   - Summary of findings
   - Future work

### Key Tables/Figures

**Table 1: Dataset Statistics**
- Rows: Easy, Medium, Hard
- Columns: Total examples, Train, Val, Test

**Table 2: Model Performance Comparison**
- Rows: Models (baseline + fine-tuned)
- Columns: Accuracy, Macro F1, Per-class F1

**Figure 1: Training Loss Curves**
- Training vs. validation loss over epochs

**Figure 2: Confusion Matrices**
- Side-by-side for baseline vs. fine-tuned

**Table 3: Error Analysis**
- Categories of errors with example counts

## Getting Started

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd CS-8903-orgaccess

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Additional packages for fine-tuning
pip install datasets transformers peft accelerate bitsandbytes
```

### Quick Start: Evaluate Baseline

```bash
# Set up API key
echo "OPENAI_API_KEY=your-key-here" > .env

# Run evaluation
python evaluate_model.py
```

### Quick Start: Prepare Training Data

```python
python3 << 'EOF'
from datasets import load_dataset, concatenate_datasets

# Load data
ds = load_dataset('parquet', data_files={
    'easy': 'data/easy-00000-of-00001.parquet',
    'medium': 'data/medium-00000-of-00001.parquet',
    'hard': 'data/hard-00000-of-00001.parquet'
})

# Create splits
easy_shuffled = ds['easy'].shuffle(seed=42)
easy_70_idx = int(len(easy_shuffled) * 0.70)
easy_train_val = easy_shuffled.select(range(easy_70_idx))

medium_shuffled = ds['medium'].shuffle(seed=42)
medium_70_idx = int(len(medium_shuffled) * 0.70)
medium_train_val = medium_shuffled.select(range(medium_70_idx))

train_val_combined = concatenate_datasets([easy_train_val, medium_train_val]).shuffle(seed=42)
train_idx = int(len(train_val_combined) * 0.80)

train_dataset = train_val_combined.select(range(train_idx))
val_dataset = train_val_combined.select(range(train_idx, len(train_val_combined)))

# Save
train_dataset.to_parquet('data/train.parquet')
val_dataset.to_parquet('data/val.parquet')

print(f"Created train.parquet: {len(train_dataset)} examples")
print(f"Created val.parquet: {len(val_dataset)} examples")
EOF
```

## References

- OrgAccess Dataset: https://huggingface.co/datasets/respai-lab/orgaccess
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- PEFT (Parameter-Efficient Fine-Tuning): https://huggingface.co/docs/peft
- OpenAI Fine-Tuning: https://platform.openai.com/docs/guides/fine-tuning

## Contact & Support

For questions about this methodology or the OrgAccess benchmark, please open an issue in the repository or contact the research team.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-17
