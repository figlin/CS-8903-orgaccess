#!/usr/bin/env python3
"""
Synthetic FULL-Access Example Generator for OrgAccess RBAC Dataset

This script generates synthetic "FULL" access examples by prompting an LLM to create
queries that would result in full access approval. It addresses class imbalance by
augmenting the underrepresented "full" category (~2% of dataset).

Methodology:
- Follows OrgAccess paper approach (Mistral-based generation)
- Varies permission coverage (30-40% all active, 40% subset, 10-15% reduced)
- Maintains hard-level complexity (min 5 permission fields)
- Ensures logical consistency with RBAC policies

Usage:
    python generate_full_examples.py \\
        --input hard-00000-of-00001_permission_database.json \\
        --output data/hard_full_augmented.jsonl \\
        --num-examples 1000 \\
        --api-url http://localhost:8000/v1/chat/completions \\
        --model-name mistral

    # With OpenAI-compatible API
    python generate_full_examples.py \\
        --input hard-00000-of-00001_permission_database.json \\
        --output data/hard_full_augmented.jsonl \\
        --num-examples 500 \\
        --api-url https://api.openai.com/v1/chat/completions \\
        --api-key $OPENAI_API_KEY \\
        --model-name gpt-4

Environment Variables:
    LLM_API_KEY: API key for LLM service (optional, can use --api-key)
    LLM_API_URL: Base URL for LLM API (optional, can use --api-url)
"""

import argparse
import json
import os
import sys
import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# ============================================================================
# INDEX TRACKING
# ============================================================================

def load_used_indexes(track_file: Path) -> set:
    """Load previously used source indexes from tracking file."""
    if not track_file.exists():
        return set()

    with open(track_file, 'r') as f:
        data = json.load(f)
        return set(data.get('used_indexes', []))


def save_used_indexes(track_file: Path, used_indexes: set) -> None:
    """Save used source indexes to tracking file."""
    track_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'used_indexes': sorted(list(used_indexes)),
        'total_used': len(used_indexes),
        'last_updated': str(Path.cwd())
    }

    with open(track_file, 'w') as f:
        json.dump(data, f, indent=2)


def get_tracking_file_path(input_path: Path, output_path: Path) -> Path:
    """Generate tracking file path based on input/output files."""
    # Create tracking file in same directory as output
    tracking_dir = output_path.parent / '.tracking'
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # Use hash of input filename to avoid conflicts
    input_hash = hashlib.md5(input_path.name.encode()).hexdigest()[:8]
    tracking_file = tracking_dir / f'used_indexes_{input_hash}.json'

    return tracking_file


# ============================================================================
# CONFIGURATION
# ============================================================================

SYSTEM_PROMPT = """
You are generating realistic organizational access-control examples for the OrgAccess-style HARD split.
Write queries as if they were composed by actual employees in a professional but conversational tone
(e.g., internal chat, email, or ticket request).

The user should sound natural and goal-oriented, not robotic or checklist-driven.
Mention constraints (regions, vendors, devices, etc.) only when they are naturally relevant.
Avoid repeating permission names literally; instead, express their intent.

For the rationale, write like an internal auditor summarizing why the request is valid,
not like a machine listing conditions.


Output ONLY valid JSON with this exact structure:
{
  "user_role": "<role name>",
  "permissions": <permissions object>,
  "query": "<realistic organizational query>",
  "expected_response": "full",
  "rationale": "<explain why all permissions are satisfied>"
}"""


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(input_path: Path) -> List[Dict]:
    """
    Load data from JSON database file.

    Expected structure:
    {
        "examples": [
            {
                "index": 0,
                "user_role": "...",
                "permissions": {...},
                "expected_response": "...",
                "query": "..."
            },
            ...
        ]
    }
    """
    print(f"\nLoading data from {input_path}...")

    with open(input_path, 'r') as f:
        data = json.load(f)

    if 'examples' not in data:
        raise ValueError("Input JSON must contain 'examples' key")

    examples = data['examples']
    print(f"✓ Loaded {len(examples):,} examples")

    # Filter for examples with sufficient permissions (hard-level)
    filtered = []
    for ex in examples:
        perms = ex.get('permissions', {})
        if isinstance(perms, dict) and len(perms) >= 5:
            filtered.append(ex)

    print(f"✓ Filtered to {len(filtered):,} examples with ≥5 permission fields (hard-level)")

    return filtered


# ============================================================================
# PERMISSION SAMPLING
# ============================================================================

def sample_permissions(
    original_permissions: Dict,
    strategy: str = 'random',
    seed: Optional[int] = None
) -> Tuple[Dict, str]:
    """
    Sample/modify permissions based on augmentation strategy.

    Strategies:
    - 'all_active' (30-40%): Keep all permissions
    - 'subset' (40%): Activate only 2-4 of available permissions
    - 'reduced' (10-15%): Drop 1-2 non-critical permissions

    Returns:
        Tuple of (modified_permissions, strategy_used)
    """
    if seed is not None:
        random.seed(seed)

    # Determine strategy if random
    if strategy == 'random':
        rand = random.random()
        if rand < 0.35:  # 35%
            strategy = 'all_active'
        elif rand < 0.75:  # 40%
            strategy = 'subset'
        else:  # 25%
            strategy = 'reduced'

    permissions = original_permissions.copy()

    if strategy == 'all_active':
        # Keep all permissions as-is
        return permissions, 'all_active'

    elif strategy == 'subset':
        # For list-type permissions, keep only subset of items
        # For boolean permissions, sometimes set to True
        modified = {}

        for key, value in permissions.items():
            if isinstance(value, list) and len(value) > 2:
                # Keep 2-4 items from the list
                keep_count = random.randint(2, min(4, len(value)))
                modified[key] = random.sample(value, keep_count)
            elif isinstance(value, bool):
                # Randomly set booleans (favor True for FULL access)
                modified[key] = random.choice([True, True, True, False])
            else:
                # Keep scalars as-is
                modified[key] = value

        return modified, 'subset'

    elif strategy == 'reduced':
        # Drop 1-2 non-critical permission fields
        # Keep at least 5 fields for hard-level
        if len(permissions) <= 5:
            return permissions, 'reduced_minimal'

        # Identify droppable fields (avoid core fields)
        core_fields = {'department', 'access_level', 'allowed_actions'}
        droppable = [k for k in permissions.keys() if k not in core_fields]

        if not droppable:
            return permissions, 'reduced_no_drop'

        # Drop 1-2 fields
        drop_count = min(random.randint(1, 2), len(droppable), len(permissions) - 5)
        to_drop = random.sample(droppable, drop_count)

        modified = {k: v for k, v in permissions.items() if k not in to_drop}
        return modified, 'reduced'

    return permissions, strategy


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def build_prompt(example: Dict, modified_permissions: Dict) -> str:
    """
    Build LLM prompt for generating a FULL-access query.

    Args:
        example: Original example with user_role and permissions
        modified_permissions: Sampled/modified permissions to use

    Returns:
        Formatted prompt string
    """
    user_role = example.get('user_role', 'Unknown_Role')

    # Create input JSON for the LLM
    input_json = {
        "user_role": user_role,
        "permissions": modified_permissions
    }

    prompt = f"""Input (generate FULL-access query for this role and permissions):
{json.dumps(input_json, indent=2)}

Below are two examples showing the tone, structure, and reasoning style to follow:

Example 1
----------
User Role: Senior_Manager_Sales

Permissions:
- Departments: Sales, Marketing, Customer Support
- Access Level: High
- Allowed Actions: View Sales Reports, Manage Customer Accounts, Generate Sales Forecasts, Escalate Customer Issues, Access Customer Data
- Rate Limit: 1000 requests
- Location Restriction: North America, Europe, Asia
- Export Restriction: True (data cannot be exported externally)
- Priority Threshold: High, Medium
- Automation Restriction: False (automation allowed)

Query:
"I need to generate a detailed sales report for the upcoming quarter, focusing on the North American market. I’ll need to access customer data, including purchase history and feedback, to identify sales trends and make data-driven recommendations. I also want to ensure customer support issues are escalated appropriately and that we comply with all contractual obligations. Additionally, I’d like to review our disaster recovery plans and verify that password rotation policies are current. Please provide access to the required data and tools for this work."

Expected Response: partial

Rationale:
"The Senior Manager’s request fits within allowed departments and actions (Sales, Marketing, Customer Support) and adheres to the location restrictions for North America. However, the query includes reviewing disaster recovery plans and password rotation, which are outside this role’s permissions, resulting in partial access."


Example 2
----------
User Role: Marketing_Intern

Permissions:
- Department: Marketing
- Access Level: Low
- Allowed Actions: view marketing reports, access customer feedback data, generate marketing analytics, create social media posts, schedule marketing campaigns
- Rate Limit: 100 requests
- Location Restriction: USA, Canada
- Export Restriction: False (data export permitted)
- Priority Threshold: Low, Medium
- Automation Restriction: True (automation not permitted)

Query:
"I’d like to analyze customer feedback data from the North American region to create a new marketing report. I plan to export this data to a CSV file for deeper analysis. I also want to schedule a campaign targeting high-priority customers in the USA and Canada and automate the weekly generation of marketing analytics reports. Could you grant me the access needed for these tasks?"

Expected Response: partial

Rationale:
"The intern’s request matches their role and access scope for viewing and analyzing marketing data in the USA and Canada. However, automation is not allowed under their permissions, and high-priority campaign management exceeds their low access level. Therefore, partial access is appropriate."


Now generate a new example in this same natural style using the following permission information:

Output JSON:"""

    return prompt


# ============================================================================
# LLM INTERACTION
# ============================================================================

def query_llm(
    prompt: str,
    api_url: str,
    model_name: str,
    api_key: Optional[str] = None,
    max_tokens: int = 800,
    temperature: float = 0.8,
    timeout: int = 30
) -> Optional[Dict]:
    """
    Query LLM API to generate a FULL-access example.

    Args:
        prompt: Formatted prompt for the LLM
        api_url: API endpoint URL
        model_name: Model identifier
        api_key: Optional API key
        max_tokens: Maximum response tokens
        temperature: Sampling temperature (higher = more creative)
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response or None if failed
    """
    headers = {
        'Content-Type': 'application/json'
    }

    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': temperature,
        'max_tokens': max_tokens
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=timeout
        )

        # Check for errors before raising
        if response.status_code != 200:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get('detail', error_data.get('error', str(error_data)))
            except:
                error_detail = response.text[:200]

            print(f"⚠️  API Error ({response.status_code}): {error_detail}")
            return None

        response.raise_for_status()

        # Parse response
        result = response.json()

        # Extract content (handle different API formats)
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0].get('message', {}).get('content', '')
        elif 'content' in result:
            content = result['content']
        else:
            print(f"⚠️  Unexpected API response format: {result}")
            return None

        # Parse JSON from content
        # Sometimes LLMs wrap JSON in markdown code blocks
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()

        # Fix common JSON issues from LLMs
        # Replace control characters that break JSON parsing
        import re
        # Remove or escape control characters (except \n, \r, \t which are valid when escaped)
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)

        # Parse JSON with strict=False to be more lenient
        try:
            generated = json.loads(content, strict=False)
            return generated
        except json.JSONDecodeError as e:
            # Try one more time: replace problematic newlines in strings
            try:
                # This is a hacky fix but works for most cases
                # Replace literal newlines inside string values with \n
                fixed_content = content.replace('\n', '\\n')
                # But we need actual newlines between JSON structure elements
                fixed_content = fixed_content.replace('\\n  ', '\n  ')
                fixed_content = fixed_content.replace('\\n}', '\n}')
                fixed_content = fixed_content.replace('\\n{', '\n{')
                fixed_content = fixed_content.replace('[\\n', '[\n')
                fixed_content = fixed_content.replace('\\n]', '\n]')

                generated = json.loads(fixed_content, strict=False)
                return generated
            except json.JSONDecodeError:
                print(f"⚠️  Failed to parse JSON from LLM response: {e}")
                print(f"    Content: {content[:200]}...")
                return None

    except requests.exceptions.Timeout:
        print(f"⚠️  Request timeout after {timeout}s")
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️  API request failed: {e}")
        return None


# ============================================================================
# VALIDATION & DEDUPLICATION
# ============================================================================

def validate_generated_example(generated: Dict, required_fields: List[str]) -> bool:
    """
    Validate that generated example has all required fields and correct format.

    Args:
        generated: Generated example from LLM
        required_fields: List of required field names

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(generated, dict):
        return False

    # Check required fields
    for field in required_fields:
        if field not in generated:
            return False

    # Validate expected_response is "full"
    expected = str(generated.get('expected_response', '')).lower().strip()
    if expected not in ['full', 'f', 'allow', 'approve']:
        return False

    # Normalize to "full"
    generated['expected_response'] = 'full'

    # Validate query is non-empty
    query = generated.get('query', '').strip()
    if len(query) < 20:  # Minimum realistic query length
        return False

    # Validate permissions is a dict
    if not isinstance(generated.get('permissions'), dict):
        return False

    return True


def compute_query_hash(query: str) -> str:
    """Compute hash of query for deduplication."""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def generate_single_example(
    source: Dict,
    api_url: str,
    model_name: str,
    api_key: Optional[str],
    temperature: float,
    max_retries: int,
    required_fields: List[str],
    seen_hashes: Set[str]
) -> Optional[Dict]:
    """
    Generate a single example from a source.

    Returns:
        Dict with 'success', 'example', 'strategy', 'source_index', 'error' keys
    """
    result = {
        'success': False,
        'example': None,
        'strategy': None,
        'source_index': source.get('index', -1),
        'error': None
    }

    try:
        # Sample permissions with random strategy
        modified_perms, strategy = sample_permissions(
            source.get('permissions', {}),
            strategy='random',
            seed=None  # Use random state
        )
        result['strategy'] = strategy

        # Build prompt
        prompt = build_prompt(source, modified_perms)

        # Query LLM with retries
        generated = None
        for attempt in range(max_retries):
            generated = query_llm(
                prompt=prompt,
                api_url=api_url,
                model_name=model_name,
                api_key=api_key,
                temperature=temperature
            )

            if generated:
                break

        if not generated:
            result['error'] = 'API call failed'
            return result

        # Validate
        if not validate_generated_example(generated, required_fields):
            result['error'] = 'Validation failed'
            return result

        # Check for duplicates
        query_hash = compute_query_hash(generated['query'])
        if query_hash in seen_hashes:
            result['error'] = 'Duplicate query'
            return result

        # Add metadata
        generated['source_index'] = source.get('index', -1)
        generated['generation_strategy'] = strategy
        generated['synthetic'] = True

        # Success!
        result['success'] = True
        result['example'] = generated
        result['query_hash'] = query_hash

        return result

    except Exception as e:
        result['error'] = str(e)
        return result


# ============================================================================
# OUTPUT HANDLING
# ============================================================================

def save_output(
    examples: List[Dict],
    output_path: Path,
    format: str = 'jsonl'
) -> None:
    """
    Save generated examples to file.

    Args:
        examples: List of generated examples
        output_path: Output file path
        format: Output format ('jsonl' or 'json')
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'jsonl':
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        print(f"\n✓ Saved {len(examples):,} examples to {output_path} (JSONL format)")

    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"\n✓ Saved {len(examples):,} examples to {output_path} (JSON format)")

    else:
        raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================

def generate_examples(
    input_path: Path,
    output_path: Path,
    num_examples: int,
    api_url: str,
    model_name: str,
    api_key: Optional[str] = None,
    seed: int = 42,
    max_retries: int = 3,
    temperature: float = 0.8,
    batch_workers: int = 4
) -> None:
    """
    Main generation pipeline.

    Args:
        input_path: Path to input JSON database
        output_path: Path to output file
        num_examples: Target number of examples to generate
        api_url: LLM API endpoint
        model_name: LLM model name
        api_key: Optional API key
        seed: Random seed for reproducibility
        max_retries: Maximum retries per example
        temperature: LLM sampling temperature
    """
    # Set random seed
    random.seed(seed)

    # Load source data
    source_examples = load_data(input_path)

    if not source_examples:
        print("❌ No source examples found!")
        return

    # Load/initialize index tracking
    tracking_file = get_tracking_file_path(input_path, output_path)
    used_indexes = load_used_indexes(tracking_file)

    # Filter out already-used source examples
    available_examples = [ex for ex in source_examples
                          if ex.get('index', -1) not in used_indexes]

    if not available_examples:
        print("⚠️  All source examples have been used!")
        print(f"   Total source examples: {len(source_examples):,}")
        print(f"   Already used: {len(used_indexes):,}")
        print(f"   Consider using a different input file or clearing tracking data.")
        return

    print(f"\n📊 Source Index Tracking:")
    print(f"   Total source examples: {len(source_examples):,}")
    print(f"   Already used: {len(used_indexes):,}")
    print(f"   Available: {len(available_examples):,}")
    print(f"   Tracking file: {tracking_file}")

    # Statistics tracking
    stats = {
        'total_attempts': 0,
        'successful': 0,
        'failed_api': 0,
        'failed_validation': 0,
        'duplicates': 0,
        'strategies': Counter(),
        'reused_indexes': 0,
    }

    generated_examples = []
    seen_hashes = set()
    new_used_indexes = set()

    # Required fields for validation
    required_fields = ['user_role', 'permissions', 'query', 'expected_response', 'rationale']

    # Setup interruption handler for graceful save on Ctrl+C
    def save_partial_results():
        """Save partial results when interrupted."""
        if generated_examples:
            # Generate interrupted filename
            timestamp = Path(output_path).stem
            interrupted_path = output_path.parent / f"{timestamp}_INTERRUPTED_{len(generated_examples)}.jsonl"

            save_output(generated_examples, interrupted_path, format='jsonl')

            # Also save index tracking
            all_used_indexes = used_indexes.union(new_used_indexes)
            save_used_indexes(tracking_file, all_used_indexes)

            print(f"\n⚠️  Interrupted! Saved {len(generated_examples):,} partial results to:")
            print(f"   {interrupted_path}")
            print(f"\n✓ Index tracking updated - can resume later without reusing these sources")
        else:
            print(f"\n⚠️  Interrupted! No examples generated yet.")

    print(f"\n{'='*80}")
    print(f"GENERATING {num_examples:,} FULL-ACCESS EXAMPLES")
    print(f"{'='*80}")
    print(f"API: {api_url}")
    print(f"Model: {model_name}")
    print(f"Random seed: {seed}")
    print(f"Temperature: {temperature}")
    print(f"Parallel workers: {batch_workers}")
    print(f"{'='*80}\n")

    # Check if we have enough available examples
    if len(available_examples) < num_examples:
        print(f"⚠️  Warning: Only {len(available_examples):,} unused examples available")
        print(f"   Requested: {num_examples:,} examples")
        print(f"   Will reuse some source examples after exhausting available pool\n")

    # Prepare source examples for parallel processing
    # Pre-select sources to ensure we get exactly num_examples attempts
    selected_sources = []
    for _ in range(num_examples * 2):  # Request 2x to handle failures
        if available_examples:
            source = random.choice(available_examples)
            available_examples.remove(source)
            selected_sources.append(source)
        else:
            source = random.choice(source_examples)
            selected_sources.append(source)
            stats['reused_indexes'] += 1

    # Generation loop with parallel workers
    progress_lock = Lock()
    pbar = tqdm(total=num_examples, desc="Generating examples")

    try:
        with ThreadPoolExecutor(max_workers=batch_workers) as executor:
            # Submit all tasks
            futures = []
            for source in selected_sources:
                if len(generated_examples) >= num_examples:
                    break

                future = executor.submit(
                    generate_single_example,
                    source,
                    api_url,
                    model_name,
                    api_key,
                    temperature,
                    max_retries,
                    required_fields,
                    seen_hashes
                )
                futures.append(future)

            # Process completed tasks
            for future in as_completed(futures):
                if len(generated_examples) >= num_examples:
                    break

                result = future.result()

                with progress_lock:
                    stats['total_attempts'] += 1

                    # Track source index
                    source_idx = result['source_index']
                    new_used_indexes.add(source_idx)

                    # Track strategy
                    if result['strategy']:
                        stats['strategies'][result['strategy']] += 1

                    # Handle result
                    if result['success']:
                        # Check for duplicates again (thread-safe)
                        query_hash = result['query_hash']
                        if query_hash not in seen_hashes:
                            generated_examples.append(result['example'])
                            seen_hashes.add(query_hash)
                            stats['successful'] += 1
                            pbar.update(1)
                        else:
                            stats['duplicates'] += 1
                    else:
                        # Handle failure
                        error = result.get('error', '')
                        if 'API call failed' in error:
                            stats['failed_api'] += 1
                        elif 'Validation failed' in error:
                            stats['failed_validation'] += 1
                        elif 'Duplicate' in error:
                            stats['duplicates'] += 1

    except KeyboardInterrupt:
        pbar.close()
        save_partial_results()
        return

    pbar.close()

    # Save output
    save_output(generated_examples, output_path, format='jsonl')

    # Save updated index tracking
    all_used_indexes = used_indexes.union(new_used_indexes)
    save_used_indexes(tracking_file, all_used_indexes)

    # Print summary
    print(f"\n{'='*80}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Target examples:        {num_examples:,}")
    print(f"Successfully generated: {stats['successful']:,}")
    print(f"Total attempts:         {stats['total_attempts']:,}")
    print(f"Failed (API errors):    {stats['failed_api']:,}")
    print(f"Failed (validation):    {stats['failed_validation']:,}")
    print(f"Duplicates skipped:     {stats['duplicates']:,}")
    if stats['reused_indexes'] > 0:
        print(f"Reused source indexes:  {stats['reused_indexes']:,}")
    print(f"\nIndex tracking:")
    print(f"  Previously used:      {len(used_indexes):,}")
    print(f"  Newly used:           {len(new_used_indexes):,}")
    print(f"  Total used:           {len(all_used_indexes):,}")
    print(f"  Remaining available:  {len(source_examples) - len(all_used_indexes):,}")
    print(f"\nStrategy breakdown:")
    for strategy, count in sorted(stats['strategies'].items()):
        percentage = count / stats['total_attempts'] * 100
        print(f"  {strategy:<20} {count:>6,} ({percentage:>5.1f}%)")
    print(f"{'='*80}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic FULL-access examples for OrgAccess RBAC dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input JSON database file (e.g., hard-00000-of-00001_permission_database.json)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL file path (e.g., data/hard_full_augmented.jsonl)'
    )

    parser.add_argument(
        '--num-examples',
        type=int,
        default=1000,
        help='Number of examples to generate (default: 1000)'
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default=os.getenv('LLM_API_URL', 'http://localhost:8000/v1/chat/completions'),
        help='LLM API endpoint URL (default: from LLM_API_URL env or localhost:8000)'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default='mistral',
        help='Model name/identifier (default: mistral)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('LLM_API_KEY'),
        help='API key for LLM service (default: from LLM_API_KEY env)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='LLM sampling temperature (default: 0.8)'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retries per example on API failure (default: 3)'
    )

    parser.add_argument(
        '--reset-tracking',
        action='store_true',
        help='Reset index tracking (start fresh, ignore previously used indexes)'
    )

    parser.add_argument(
        '--batch-workers',
        type=int,
        default=4,
        help='Number of parallel workers for concurrent generation (default: 4, range: 1-16)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    # Handle tracking reset
    if args.reset_tracking:
        tracking_file = get_tracking_file_path(args.input, args.output)
        if tracking_file.exists():
            tracking_file.unlink()
            print(f"✓ Reset index tracking: {tracking_file}")
        else:
            print(f"ℹ️  No tracking file found to reset")

    # Run generation
    try:
        generate_examples(
            input_path=args.input,
            output_path=args.output,
            num_examples=args.num_examples,
            api_url=args.api_url,
            model_name=args.model_name,
            api_key=args.api_key,
            seed=args.seed,
            max_retries=args.max_retries,
            temperature=args.temperature,
            batch_workers=args.batch_workers
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
