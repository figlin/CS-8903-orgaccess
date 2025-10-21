#!/usr/bin/env python3
"""
Simple Gemini-to-OpenAI API proxy server for generate_full_examples.py

Converts OpenAI-style chat completion requests to Gemini format.

Usage:
    export GEMINI_API_KEY=your-key-here
    python scripts/gemini_proxy.py

Then use with generate_full_examples.py:
    python scripts/generate_full_examples.py \
        --api-url http://localhost:8080/v1/chat/completions \
        --model-name gemini-1.5-flash \
        --num-examples 100
"""

import os
import json
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Convert OpenAI-style request to Gemini format."""
    data = request.json

    # Extract parameters
    model_name = data.get('model', 'gemini-1.5-flash')
    messages = data.get('messages', [])
    temperature = data.get('temperature', 0.8)
    max_tokens = data.get('max_tokens', 800)

    # Convert messages to Gemini format
    # Gemini expects a single prompt or conversation history
    system_prompt = None
    user_messages = []

    for msg in messages:
        role = msg.get('role')
        content = msg.get('content')

        if role == 'system':
            system_prompt = content
        elif role == 'user':
            user_messages.append(content)

    # Combine system prompt with user message
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{user_messages[-1]}"
    else:
        full_prompt = user_messages[-1] if user_messages else ""

    try:
        # Create Gemini model
        model = genai.GenerativeModel(model_name)

        # Generate response
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )

        # Convert to OpenAI format
        openai_response = {
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': response.text
                    },
                    'finish_reason': 'stop',
                    'index': 0
                }
            ],
            'model': model_name,
            'object': 'chat.completion'
        }

        return jsonify(openai_response)

    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'gemini_error'
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({
        'object': 'list',
        'data': [
            {
                'id': 'gemini-1.5-flash',
                'object': 'model',
                'owned_by': 'google'
            },
            {
                'id': 'gemini-1.5-pro',
                'object': 'model',
                'owned_by': 'google'
            }
        ]
    })


if __name__ == '__main__':
    print("Starting Gemini-to-OpenAI proxy server...")
    print(f"API Key configured: {GEMINI_API_KEY[:10]}...")
    print("\nServer running at http://localhost:8080")
    print("\nUse with generate_full_examples.py:")
    print("  python scripts/generate_full_examples.py \\")
    print("    --api-url http://localhost:8080/v1/chat/completions \\")
    print("    --model-name gemini-1.5-flash \\")
    print("    --num-examples 100")

    app.run(host='0.0.0.0', port=8080, debug=False)
