#!/usr/bin/env python3
"""
OpenAI Client - Access to GPT-5-mini and other models via Responses API

Provides standardized interface for OpenAI API calls using the new Responses API.
Uses gpt-5-mini-2025-08-07 as default for cost-effective, high-quality operations.

Usage:
    from openai_client import call_openai, call_openai_stream

    # Simple call
    response = call_openai("What is the capital of France?")

    # Streaming call
    for chunk in call_openai_stream("Tell me a story"):
        print(chunk, end="", flush=True)

    # With structured input (messages list)
    response = call_openai([
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ])
"""

import os
import sys
import time
from typing import Optional, Generator
from openai import OpenAI
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Try multiple locations for .env file (plugin portable)
    env_locations = [
        Path.home() / '.granola-claude' / '.env',  # User config (primary)
        Path(__file__).parent.parent.parent / '.env',  # Plugin root
        Path(__file__).parent.parent / '.env',     # Repo root (dev)
        Path.cwd() / '.env'                         # Current dir (fallback)
    ]

    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            break
except ImportError:
    # dotenv not installed, rely on system environment variables
    pass

# Initialize client lazily (uses OPENAI_API_KEY environment variable)
_client = None

def get_client():
    """Get or create the OpenAI client (lazy initialization)"""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

# Default model - GPT-5-mini with specific version
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"  # Latest GPT-5-mini version
GPT5_MODEL = "gpt-5-mini-2025-08-07"  # New generation model
FALLBACK_MODEL = "gpt-4o-mini"  # Stable fallback if GPT-5 not available

# Model pricing (per 1M tokens) - Updated 2025-10-26 with actual pricing
MODEL_PRICING = {
    "gpt-5-mini-2025-08-07": {
        "input": 0.25,    # $0.250 per 1M input tokens
        "output": 2.00,   # $2.000 per 1M output tokens
        "cached_input": 0.025,  # $0.025 per 1M cached input tokens
    },
    "gpt-5-mini": {
        "input": 0.25,    # $0.250 per 1M input tokens
        "output": 2.00,   # $2.000 per 1M output tokens
        "cached_input": 0.025,  # $0.025 per 1M cached input tokens
    },
    "gpt-5-nano-2025-08-07": {
        "input": 0.05,    # $0.050 per 1M input tokens
        "output": 0.40,   # $0.400 per 1M output tokens
        "cached_input": 0.005,  # $0.005 per 1M cached input tokens
    },
    "gpt-5-nano": {
        "input": 0.05,    # $0.050 per 1M input tokens
        "output": 0.40,   # $0.400 per 1M output tokens
        "cached_input": 0.005,  # $0.005 per 1M cached input tokens
    },
    "gpt-4o-mini": {
        "input": 0.150,   # $0.150 per 1M input tokens
        "output": 0.600,  # $0.600 per 1M output tokens
    },
    "gpt-4o": {
        "input": 2.50,    # $2.50 per 1M input tokens
        "output": 10.00,  # $10.00 per 1M output tokens
    },
    "o1-mini": {
        "input": 3.00,    # $3.00 per 1M input tokens
        "output": 12.00,  # $12.00 per 1M output tokens
    },
    "o1": {
        "input": 15.00,   # $15.00 per 1M input tokens
        "output": 60.00,  # $60.00 per 1M output tokens
    }
}


def call_openai(
    input,  # Can be string or list of messages
    model: str = DEFAULT_MODEL,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    response_format: Optional[str] = None,
    reasoning_effort: Optional[str] = None
) -> str:
    """
    Call OpenAI API using Responses API (GPT-5 compatible).

    Args:
        input: User prompt (string) or list of messages [{"role": "user", "content": "..."}]
        model: Model to use (default: gpt-5-mini-2025-08-07)
        max_tokens: Max tokens to generate (None = unlimited)
        system_prompt: System message to set context (prepended as developer role)
        response_format: "json" for JSON mode, None for text
        reasoning_effort: For reasoning models - "low", "medium", "high" (optional)

    Returns:
        Response text from model

    Example:
        >>> response = call_openai("What is 2+2?")
        >>> print(response)
        "4"

        >>> response = call_openai([
        ...     {"role": "developer", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hi!"}
        ... ])
    """
    try:
        # Convert string input to messages if needed
        if isinstance(input, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": input})
        elif isinstance(input, list):
            messages = input
            if system_prompt:
                # Prepend system_prompt as system message
                messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages = [{"role": "user", "content": str(input)}]

        # Build request params for standard Chat Completions API
        params = {
            "model": model,
            "messages": messages,
        }

        if max_tokens:
            # NOTE: max_completion_tokens causes empty responses in some cases
            # Use max_tokens for all models for now
            params["max_tokens"] = max_tokens

        # JSON mode if requested
        if response_format == "json":
            params["response_format"] = {"type": "json_object"}

        # Call standard Chat Completions API with timing
        start_time = time.time()
        response = get_client().chat.completions.create(**params)
        elapsed = time.time() - start_time

        # Extract text
        result = response.choices[0].message.content

        # Log token usage if available
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            cost = calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)
            print(f"[OpenAI] Request completed in {elapsed:.2f}s | Tokens: {usage.prompt_tokens} in + {usage.completion_tokens} out | Cost: ${cost:.4f}", file=sys.stderr)
        else:
            print(f"[OpenAI] Request completed in {elapsed:.2f}s", file=sys.stderr)

        return result

    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")


def call_openai_responses(
    input,  # Can be string or list of messages
    model: str = DEFAULT_MODEL,
    max_output_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    reasoning_effort: str = "minimal"
) -> str:
    """
    Call OpenAI Responses API (for GPT-5 models with reasoning).

    Uses reasoning.effort="minimal" by default for fast text generation without
    extensive reasoning tokens. This is the recommended way to use GPT-5-nano/mini
    for text generation tasks.

    Args:
        input: User prompt (string) or list of messages
        model: Model to use (default: gpt-5-mini-2025-08-07)
        max_output_tokens: Max tokens to generate (None = unlimited)
        system_prompt: System message to set context
        reasoning_effort: "minimal", "low", "medium", or "high" (default: "minimal")

    Returns:
        Response text from model

    Example:
        >>> response = call_openai_responses("What is 2+2?", model="gpt-5-nano-2025-08-07")
        >>> print(response)
        "4"
    """
    try:
        # Convert string input to messages if needed
        if isinstance(input, str):
            messages = []
            if system_prompt:
                messages.append({"role": "developer", "content": system_prompt})
            messages.append({"role": "user", "content": input})
        elif isinstance(input, list):
            messages = input
            if system_prompt:
                messages = [{"role": "developer", "content": system_prompt}] + messages
        else:
            messages = [{"role": "user", "content": str(input)}]

        # Build request params for Responses API
        params = {
            "model": model,
            "input": messages,
            "reasoning": {
                "effort": reasoning_effort
            }
        }

        if max_output_tokens:
            params["max_output_tokens"] = max_output_tokens

        # Call Responses API with timing
        start_time = time.time()
        response = get_client().responses.create(**params)
        elapsed = time.time() - start_time

        # Extract text using output_text helper
        result = response.output_text

        # Log token usage if available
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            cost = calculate_cost(model, usage.input_tokens, usage.output_tokens)
            print(f"[OpenAI Responses] Request completed in {elapsed:.2f}s | Tokens: {usage.input_tokens} in + {usage.output_tokens} out | Cost: ${cost:.4f}", file=sys.stderr)
        else:
            print(f"[OpenAI Responses] Request completed in {elapsed:.2f}s", file=sys.stderr)

        return result

    except Exception as e:
        raise Exception(f"OpenAI Responses API error: {str(e)}")


def call_openai_stream(
    input,  # Can be string or list of messages
    model: str = DEFAULT_MODEL,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Stream response from OpenAI API using Responses API.

    Args:
        input: User prompt (string) or list of messages
        model: Model to use (default: gpt-5-mini-2025-08-07)
        max_tokens: Max tokens to generate (None = unlimited)
        system_prompt: System message to set context

    Yields:
        Text chunks as they arrive

    Example:
        >>> for chunk in call_openai_stream("Tell me a story"):
        ...     print(chunk, end="", flush=True)
    """
    try:
        # Convert string input to messages if needed
        if isinstance(input, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": input})
        elif isinstance(input, list):
            messages = input
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages = [{"role": "user", "content": str(input)}]

        # Build request params for standard Chat Completions API
        params = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if max_tokens:
            # NOTE: max_completion_tokens causes empty responses in some cases
            # Use max_tokens for all models for now
            params["max_tokens"] = max_tokens

        # Stream response using standard Chat Completions API
        stream = get_client().chat.completions.create(**params)

        import logging
        event_count = 0
        event_types_seen = {}

        for chunk in stream:
            # Standard streaming response format
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

        logging.info(f"[OpenAI Stream] Total events: {event_count}, Types: {event_types_seen}")

    except Exception as e:
        import logging
        logging.error(f"[OpenAI Stream] Error during streaming: {e}", exc_info=True)
        yield f"\n\n[Error: {str(e)}]"


def call_openai_with_fallback(
    input,  # Can be string or list of messages
    model: str = DEFAULT_MODEL,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    reasoning_effort: Optional[str] = None
) -> tuple[str, str, Optional[str]]:
    """
    Call OpenAI API with automatic fallback to Gemini if GPT-5 fails.

    This is the RECOMMENDED function for chat and conversational use cases.
    It provides resilience by falling back to Gemini 2.5-flash if OpenAI fails.

    Args:
        input: User prompt (string) or list of messages
        model: Model to use (default: gpt-5-mini-2025-08-07)
        max_tokens: Max tokens to generate (None = unlimited)
        system_prompt: System message to set context
        temperature: Sampling temperature (0.0-1.0), ignored for GPT-5 but used for Gemini fallback
        reasoning_effort: For reasoning models - "low", "medium", "high" (optional)

    Returns:
        Tuple of (response_text, model_used, error_if_any)
        - response_text: The generated response
        - model_used: Which model was actually used ("gpt-5-mini-2025-08-07" or "gemini-3-flash-preview")
        - error_if_any: Error message from GPT-5 if fallback occurred, None otherwise

    Example:
        >>> response, model, error = call_openai_with_fallback("What is 2+2?")
        >>> if error:
        ...     print(f"Fell back to {model}: {error}")
        >>> print(response)
    """
    import logging

    # Try OpenAI first
    try:
        start_time = time.time()
        logging.info(f"[OpenAI Fallback] Trying {model}")
        response = call_openai(
            input=input,
            model=model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort
        )
        elapsed = time.time() - start_time
        logging.info(f"[OpenAI Fallback] Success with {model} in {elapsed:.2f}s")
        return (response, model, None)

    except Exception as e:
        error_msg = str(e)
        logging.error(f"[OpenAI Fallback] {model} failed: {error_msg}")

        # Fall back to Gemini
        try:
            logging.info("[OpenAI Fallback] Falling back to Gemini 3-flash")

            # Import Gemini client
            from gemini_client import call_gemini

            # Convert input to string prompt for Gemini
            if isinstance(input, str):
                prompt = input
                if system_prompt:
                    prompt = f"{system_prompt}\n\n{input}"
            elif isinstance(input, list):
                # Convert messages list to single prompt
                prompt_parts = []
                for msg in input:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')

                    if role == 'developer' or role == 'system':
                        prompt_parts.append(f"System: {content}")
                    elif role == 'user':
                        prompt_parts.append(f"User: {content}")
                    elif role == 'assistant':
                        prompt_parts.append(f"Assistant: {content}")

                prompt = "\n\n".join(prompt_parts)
            else:
                prompt = str(input)

            # Call Gemini
            gemini_response = call_gemini(prompt, temperature=temperature)
            logging.info("[OpenAI Fallback] Gemini success")

            return (gemini_response, "gemini-3-flash-preview", error_msg)

        except Exception as gemini_error:
            # Both failed - return error
            logging.error(f"[OpenAI Fallback] Gemini also failed: {gemini_error}")
            error_combined = f"OpenAI error: {error_msg}\nGemini error: {gemini_error}"
            return ("", "none", error_combined)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost of API call based on token usage.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in dollars
    """
    if model not in MODEL_PRICING:
        return 0.0

    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def check_model_availability(model: str) -> bool:
    """
    Check if a model is available for your API key.

    Args:
        model: Model name to check

    Returns:
        True if model is available, False otherwise

    Example:
        >>> if check_model_availability("gpt-5-mini"):
        ...     print("GPT-5-mini is ready!")
        ... else:
        ...     print("Need to verify organization first")
    """
    try:
        # Try a minimal call (use max_completion_tokens for GPT-5+ models)
        params = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
        }

        # GPT-5+ models use max_completion_tokens, older models use max_tokens
        if "gpt-5" in model or "gpt-6" in model:
            params["max_completion_tokens"] = 1
        else:
            params["max_tokens"] = 1

        response = get_client().chat.completions.create(**params)
        return True
    except Exception as e:
        error_str = str(e)
        if "model_not_found" in error_str or "verified" in error_str.lower():
            return False
        # Other errors might be transient
        raise


def compare_models(prompt: str, models: list[str] = None) -> dict:
    """
    Compare responses from different models for the same prompt.

    Args:
        prompt: Prompt to test
        models: List of models to compare (default: gpt-5-mini and gpt-4o-mini)

    Returns:
        Dict mapping model name to response

    Example:
        >>> results = compare_models("What is 2+2?", ["gpt-5-mini-2025-08-07", "gpt-4o-mini"])
        >>> for model, response in results.items():
        ...     print(f"{model}: {response}")
    """
    if models is None:
        models = ["gpt-5-mini-2025-08-07", "gpt-4o-mini"]

    results = {}

    for model in models:
        try:
            response = call_openai(prompt, model=model)
            results[model] = response
        except Exception as e:
            results[model] = f"Error: {str(e)}"

    return results


def main():
    """Test OpenAI client with Responses API."""
    print("=== OpenAI Client Test (Responses API) ===\n")

    # Test 1: Simple call
    print("Test 1: Simple call")
    response = call_openai(
        "What is the capital of France? Answer in one word."
    )
    print(f"Response: {response}\n")

    # Test 2: JSON mode
    print("Test 2: JSON mode")
    response = call_openai(
        "Return a JSON object with fields: name='Paris', country='France'",
        response_format="json"
    )
    print(f"Response: {response}\n")

    # Test 3: Streaming
    print("Test 3: Streaming")
    print("Response: ", end="", flush=True)
    for chunk in call_openai_stream(
        "Count from 1 to 5, each number on a new line."
    ):
        print(chunk, end="", flush=True)
    print("\n")

    # Test 4: System prompt (developer role)
    print("Test 4: System prompt")
    response = call_openai(
        "What's the weather like?",
        system_prompt="You are a pirate. Always respond in pirate speak."
    )
    print(f"Response: {response}\n")

    # Test 5: Messages list input
    print("Test 5: Messages list input")
    response = call_openai([
        {"role": "developer", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?"}
    ])
    print(f"Response: {response}\n")

    print("=== All tests complete ===")


if __name__ == '__main__':
    main()
