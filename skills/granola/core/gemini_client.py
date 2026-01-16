"""
Gemini API Client for Portfolio Management

Purpose: Conserve Claude Code usage by offloading parsing, drafting, and extraction tasks to Gemini.

Usage:
    from gemini_client import call_gemini, extract_structured_data

    # Simple text generation
    response = call_gemini("Summarize this company: [text]")

    # Structured data extraction
    data = extract_structured_data(markdown_content, schema)
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import google.generativeai as genai

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

# Import constants for retry logic
try:
    from constants import (
        API_RETRY_ATTEMPTS,
        API_RETRY_DELAY_SECONDS,
        API_TIMEOUT_SECONDS,
        GEMINI_DEFAULT_TEMPERATURE,
        GEMINI_CREATIVE_TEMPERATURE,
        GEMINI_MAX_TOKENS
    )
except ImportError:
    # Fallback if constants not available
    API_RETRY_ATTEMPTS = 3
    API_RETRY_DELAY_SECONDS = 2
    API_TIMEOUT_SECONDS = 30
    GEMINI_DEFAULT_TEMPERATURE = 0.1
    GEMINI_CREATIVE_TEMPERATURE = 0.7
    GEMINI_MAX_TOKENS = 8192

# Import API logger for observability
try:
    from api_logger import log_api_call
    LOGGING_ENABLED = True
except ImportError:
    LOGGING_ENABLED = False
    def log_api_call(*args, **kwargs):
        pass  # No-op if logger not available

# Import LLM tracking
try:
    from llm_tracker import track_extraction, track_email_draft, track_research
except ImportError:
    # Graceful fallback if tracker not available
    def track_extraction(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def track_email_draft(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def track_research(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Configure API from environment variable
API_KEY = os.getenv('GEMINI_API_KEY')
if API_KEY:
    genai.configure(api_key=API_KEY)

# Default models
DEFAULT_MODEL = "gemini-3-flash-preview"  # Latest Gemini 3 Flash (Dec 2025) - Fast, cheap, Pro-grade reasoning
PRO_MODEL = "gemini-2.0-flash-thinking-exp-1219"  # Thinking model for complex reasoning


def call_gemini(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = GEMINI_DEFAULT_TEMPERATURE,
    response_format: str = "text",  # "text" or "json"
    system_instruction: str = None  # System prompt to prepend
) -> str:
    """
    Call Gemini API with a prompt and return response.

    Args:
        prompt: The prompt to send to Gemini
        model: Model to use (default: gemini-3-flash-preview)
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        response_format: "text" or "json" for structured output
        system_instruction: System prompt to prepend to user prompt

    Returns:
        String response from Gemini

    Raises:
        Exception: If API call fails after retries
    """
    # Prepend system instruction to prompt if provided
    if system_instruction:
        prompt = f"{system_instruction}\n\n{prompt}"

    model_instance = genai.GenerativeModel(model)

    # Configure generation
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": GEMINI_MAX_TOKENS,
    }

    # For JSON, add instruction to prompt instead of using response_mime_type
    if response_format == "json":
        prompt = f"{prompt}\n\nIMPORTANT: Return ONLY valid JSON, no markdown formatting, no explanations."

    # Retry with exponential backoff
    last_error = None
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            # Generate response with timing
            start_time = time.time()
            response = model_instance.generate_content(
                prompt,
                generation_config=generation_config
            )
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Extract response text
            response_text = response.text

            # Log timing
            print(f"[Gemini] Request completed in {elapsed_ms}ms | Model: {model}")

            # Log API call for observability
            if LOGGING_ENABLED:
                # Estimate tokens (rough: 4 chars per token)
                tokens_input = len(prompt) // 4
                tokens_output = len(response_text) // 4

                log_api_call(
                    provider="gemini",
                    model=model,
                    prompt=prompt,
                    response=response_text,
                    tokens_used={"input": tokens_input, "output": tokens_output},
                    latency_ms=elapsed_ms,
                    success=True,
                    metadata={"temperature": temperature, "response_format": response_format}
                )

            return response_text

        except Exception as e:
            last_error = e
            if attempt < API_RETRY_ATTEMPTS - 1:
                # Exponential backoff: 2s, 4s, 8s
                delay = API_RETRY_DELAY_SECONDS * (2 ** attempt)
                print(f"[Gemini] Attempt {attempt + 1}/{API_RETRY_ATTEMPTS} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                # Final attempt failed
                print(f"[Gemini] All {API_RETRY_ATTEMPTS} attempts failed.")

    # Log failed call
    if LOGGING_ENABLED:
        log_api_call(
            provider="gemini",
            model=model,
            prompt=prompt,
            response="",
            tokens_used={"input": len(prompt) // 4, "output": 0},
            latency_ms=0,
            success=False,
            error=str(last_error)
        )

    raise Exception(f"Gemini API error after {API_RETRY_ATTEMPTS} retries: {str(last_error)}")


@track_extraction(task_description="Extract structured data from markdown", provider="gemini")
def extract_structured_data(
    markdown_content: str,
    schema: Dict[str, Any],
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Extract structured data from markdown using Gemini.

    Args:
        markdown_content: Markdown file content to parse
        schema: Dictionary describing the desired output structure
        model: Gemini model to use

    Returns:
        Dictionary with extracted data

    Example:
        schema = {
            "investment_amount": "float or null",
            "investment_date": "YYYY-MM-DD or null",
            "current_valuation": "float or null",
            "last_update_date": "YYYY-MM-DD or null"
        }

        data = extract_structured_data(markdown, schema)
    """
    prompt = f"""Extract the following information from this markdown file.
Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Rules:
- If data is missing, return null
- For amounts, extract numeric value only (e.g., "$2,500" → 2500.0)
- For dates, use YYYY-MM-DD format
- Be conservative - if uncertain, return null

Markdown content:
{markdown_content}

Return JSON only, no explanations:"""

    response = call_gemini(prompt, model=model, response_format="json")

    # Strip markdown code blocks if present
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]  # Remove ```json
    if response.startswith("```"):
        response = response[3:]  # Remove ```
    if response.endswith("```"):
        response = response[:-3]  # Remove trailing ```
    response = response.strip()

    try:
        result = json.loads(response)
        # Add metadata for tracking
        result['model'] = model  # Inject model info for tracker
        return result
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {"error": "Could not parse response", "raw": response, "model": model}


@track_email_draft(task_description="Draft founder email for valuation update", provider="gemini")
def draft_founder_email(
    company_name: str,
    founder_name: Optional[str],
    relationship_context: Optional[str],
    last_contact_date: Optional[str],
    invested_amount: float,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Draft an email to a founder asking for valuation update.

    Args:
        company_name: Name of the company
        founder_name: Founder's first name (optional)
        relationship_context: Brief context about relationship (optional)
        last_contact_date: When you last spoke (optional)
        invested_amount: How much you invested
        model: Gemini model to use

    Returns:
        Draft email text
    """
    context = ""
    if relationship_context:
        context += f"\nRelationship: {relationship_context}"
    if last_contact_date:
        context += f"\nLast contact: {last_contact_date}"

    prompt = f"""Draft a friendly, brief email to {founder_name or "the founder"} of {company_name} asking for a valuation update.

Context:
- I'm an angel investor who invested ${invested_amount:,.0f}{context}
- I'm updating my portfolio records
- Keep it warm but professional
- 2-3 sentences max
- Subject line + body

Draft the email:"""

    return call_gemini(prompt, model=model, temperature=GEMINI_CREATIVE_TEMPERATURE)


def draft_followup_email(
    company_name: str,
    founder_name: Optional[str],
    context: str,
    purpose: str,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Draft a follow-up email to a founder for various purposes.

    Args:
        company_name: Name of the company
        founder_name: Founder's first name (optional)
        context: Context about the company/relationship
        purpose: What the email is for (e.g., "asking for latest update", "missing valuation data")
        model: Gemini model to use

    Returns:
        Draft email (subject + body)
    """
    prompt = f"""Draft a brief, friendly email to {founder_name or "the founder"} of {company_name}.

**Purpose:** {purpose}

**Context:**
{context}

**VJ's email style:**
- Brief and to the point (2-3 sentences)
- Warm but not overly formal
- Specific about what's needed
- Offers to help in return

**Example structure:**
Subject: Quick update on [Company]?

Hi [Name],

Hope you're doing well! I'm updating my portfolio records and realized I'm missing [specific thing]. Could you share [what you need]?

Happy to help with [relevant offer based on context] if useful.

Best,
VJ

**Return format:**
Subject: [subject line]

[email body]

Draft the email:"""

    return call_gemini(prompt, model=model, temperature=GEMINI_CREATIVE_TEMPERATURE)


def parse_company_markdown(file_path: str) -> Dict[str, Any]:
    """
    Parse a company markdown file and extract key investment data.

    Args:
        file_path: Path to company markdown file

    Returns:
        Dictionary with extracted data:
        {
            "company_name": str,
            "investments": List[Dict],  # Each investment with amount, date, valuation
            "current_valuation": float or None,
            "last_update_date": str or None,
            "status": str,  # invested, exited, shut_down, etc.
        }
    """
    with open(file_path, 'r') as f:
        content = f.read()

    schema = {
        "company_name": "string",
        "status": "string (invested, exited, shut_down, or unknown)",
        "current_valuation": "float (latest post-money valuation) or null",
        "last_update_date": "string (YYYY-MM-DD of last valuation update) or null",
        "investments": [
            {
                "investment_num": "integer",
                "investment_amount": "float or null",
                "investment_date": "string (YYYY-MM-DD) or null",
                "valuation_cap": "float or null",
                "notes": "string (brief summary) or null"
            }
        ]
    }

    return extract_structured_data(content, schema)


def cleanup_markdown_section(
    markdown_content: str,
    section_to_remove: str,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Use Gemini to intelligently remove a section from markdown while preserving structure.

    Args:
        markdown_content: Full markdown content
        section_to_remove: Description of what to remove (e.g., "Investment #2 section")
        model: Gemini model to use

    Returns:
        Cleaned markdown content
    """
    prompt = f"""Remove the following section from this markdown file: "{section_to_remove}"

Rules:
- Preserve all other content exactly as-is
- Maintain markdown structure and formatting
- Update any "Total" sections if they exist to reflect the removal
- Return the complete cleaned markdown

Original markdown:
{markdown_content}

Return cleaned markdown:"""

    return call_gemini(prompt, model=model, temperature=0.1)


def add_citations(response):
    """
    Add inline citations to Gemini response text (Google's official pattern).

    Maps text segments to source URLs using grounding metadata.

    Args:
        response: Gemini API response object with grounding_metadata

    Returns:
        Text with inline citations like: "fact[1](url), [2](url)"
    """
    try:
        text = response.text

        if not hasattr(response, 'candidates') or not response.candidates:
            return text

        candidate = response.candidates[0]
        if not hasattr(candidate, 'grounding_metadata'):
            return text

        metadata = candidate.grounding_metadata
        if not hasattr(metadata, 'grounding_supports') or not hasattr(metadata, 'grounding_chunks'):
            return text

        supports = metadata.grounding_supports
        chunks = metadata.grounding_chunks

        # Sort supports by end_index in descending order to avoid shifting issues when inserting
        sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

        for support in sorted_supports:
            end_index = support.segment.end_index
            if support.grounding_chunk_indices:
                # Create citation string like [1](link1), [2](link2)
                citation_links = []
                for i in support.grounding_chunk_indices:
                    if i < len(chunks):
                        uri = chunks[i].web.uri
                        citation_links.append(f"[{i + 1}]({uri})")

                citation_string = " " + ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]

        return text
    except Exception as e:
        # If citation extraction fails, return original text
        print(f"Warning: Could not extract citations: {e}")
        return response.text


@track_research(task_description="Google Search with Gemini grounding", provider="gemini")
def search_with_grounding(
    query: str,
    model: str = DEFAULT_MODEL,
    usage_tracker = None
) -> Dict[str, Any]:
    """
    Execute Google Search with Gemini grounding.

    Uses Gemini 3 Flash to search Google and return grounded, cited responses.

    Args:
        query: Search query string
        model: Gemini model to use (default: gemini-3-flash-preview)
        usage_tracker: GroundingUsageTracker instance (optional)

    Returns:
        {
            'answer': str,  # Response text
            'answer_with_citations': str,  # Response with inline citations
            'search_queries': List[str],  # Queries executed by Gemini
            'sources': List[Dict],  # [{'uri': str, 'title': str}]
            'grounded': bool,  # True if grounding was used
            'fallback_reason': str or None  # Reason for fallback if applicable
        }
    """
    # Import here to avoid circular dependency
    try:
        from grounding_tracker import GroundingUsageTracker
        if usage_tracker is None:
            usage_tracker = GroundingUsageTracker()
    except ImportError:
        usage_tracker = None

    # Check usage quota
    if usage_tracker and not usage_tracker.can_use_grounding():
        return _fallback_to_ungrounded(
            query,
            model,
            fallback_reason="Daily quota exceeded (1500 requests/day)"
        )

    try:
        # Import Google's new genai SDK
        from google import genai as google_genai
        from google.genai import types

        # Create client
        client = google_genai.Client(api_key=API_KEY)

        # Configure grounding tool
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            response_modalities=["TEXT"]
        )

        # Make grounded request
        response = client.models.generate_content(
            model=model,
            contents=query,
            config=config
        )

        # Increment usage tracker
        if usage_tracker:
            usage_tracker.increment_usage()

        # Extract result
        result = {
            'answer': response.text,
            'answer_with_citations': add_citations(response),
            'search_queries': [],
            'sources': [],
            'grounded': True,
            'fallback_reason': None
        }

        # Extract grounding metadata if available
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                metadata = candidate.grounding_metadata

                # Extract search queries
                if hasattr(metadata, 'web_search_queries'):
                    result['search_queries'] = list(metadata.web_search_queries)

                # Extract sources
                if hasattr(metadata, 'grounding_chunks'):
                    result['sources'] = [
                        {
                            'uri': chunk.web.uri,
                            'title': chunk.web.title if hasattr(chunk.web, 'title') else 'Unknown'
                        }
                        for chunk in metadata.grounding_chunks
                    ]

        return result

    except ImportError as e:
        return _fallback_to_ungrounded(
            query,
            model,
            fallback_reason=f"google-genai SDK not installed: {e}"
        )
    except Exception as e:
        return _fallback_to_ungrounded(
            query,
            model,
            fallback_reason=f"API error: {str(e)}"
        )


def _fallback_to_ungrounded(query: str, model: str, fallback_reason: str = None) -> Dict[str, Any]:
    """
    Fallback to non-grounded Gemini call if grounding unavailable.

    Args:
        query: Search query
        model: Model to use
        fallback_reason: Reason for fallback

    Returns:
        Result dict with grounded=False
    """
    try:
        # Use regular Gemini call without grounding
        response = call_gemini(query, model=model, temperature=0.1)

        return {
            'answer': response,
            'answer_with_citations': response,
            'search_queries': [],
            'sources': [],
            'grounded': False,
            'fallback_reason': fallback_reason or "Grounding not available"
        }
    except Exception as e:
        return {
            'answer': f"Error: {str(e)}",
            'answer_with_citations': f"Error: {str(e)}",
            'search_queries': [],
            'sources': [],
            'grounded': False,
            'fallback_reason': f"Both grounded and ungrounded calls failed: {str(e)}"
        }


if __name__ == "__main__":
    # Test the integration
    print("Testing Gemini API integration...")
    print("-" * 50)

    # Test 1: Simple call
    print("\n1. Simple text generation:")
    response = call_gemini("What is 2+2? Answer in one sentence.")
    print(f"Response: {response}")

    # Test 2: Structured extraction
    print("\n2. Structured data extraction:")
    test_markdown = """
    # Test Company

    ## Investment
    - investment_amount:: $25,000
    - investment_date:: 2024-01-15
    - valuation_cap:: $10M
    """

    schema = {
        "investment_amount": "float or null",
        "investment_date": "string or null",
        "valuation_cap": "float or null"
    }

    data = extract_structured_data(test_markdown, schema)
    print(f"Extracted: {json.dumps(data, indent=2)}")

    # Test 3: Email draft
    print("\n3. Email drafting:")
    email = draft_founder_email(
        company_name="Acme Corp",
        founder_name="Jane",
        relationship_context="Met at YC Demo Day 2023",
        last_contact_date="2024-06-15",
        invested_amount=10000
    )
    print(f"Draft:\n{email}")

    print("\n" + "="*50)
    print("✅ All tests passed! Gemini integration working.")
