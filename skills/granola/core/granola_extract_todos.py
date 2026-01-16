#!/usr/bin/env python3
"""
Extract TODOs from Granola meetings using GPT-5-nano.

Uses both AI summary and full transcript for comprehensive extraction.
Returns structured TODOs with deadlines, owners, and context.

Cost: ~$0.002 per meeting (GPT-5-nano is 10x cheaper than GPT-5-mini)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from openai_client import call_openai


TODO_EXTRACTION_PROMPT = """You are an expert at extracting actionable TODOs from meeting transcripts and summaries.

Extract ALL action items, commitments, follow-ups, and next steps mentioned in the meeting.

# Instructions:

1. **Find explicit commitments**: "I'll send you X", "Let me follow up on Y", "I need to Z"
2. **Find implicit actions**: Mentions of things that need doing, decisions requiring follow-up
3. **Infer deadlines**:
   - "by end of week" → Friday of current week
   - "next week" → 7 days from meeting date
   - "ASAP" / "soon" → 2-3 days from meeting date
   - "end of January" → last day of January
   - No deadline mentioned → 7 days default
4. **Identify owner**:
   - "I'll" / "I need to" → You (meeting attendee)
   - "You should" / "Can you" → Other person
   - Unclear → You (conservative default)
5. **Provide context**: Why this TODO matters (1 sentence)

# Entity Linking Rules:
- Use EXACT names mentioned in meeting
- Link people: "Send email to [[Kristina]]"
- Link companies: "Follow up with [[OpenAI]]"
- Keep it natural: "Intro [[Alice]] to [[Bob]]"

# Prioritization:
1. **Urgent**: Explicit deadlines, "ASAP", blocks other work
2. **High**: Explicit commitments, investor follow-ups
3. **Medium**: Implicit actions, "should do"
4. **Low**: "Nice to have", research tasks

# Output Format (JSON):
{
  "todos": [
    {
      "action": "Send fund memo to [[Kristina]]",
      "owner": "You",
      "deadline": "2026-01-20",
      "deadline_confidence": "high|medium|low",
      "priority": "urgent|high|medium|low",
      "context": "Follow up from Chemistry VC conversation about Fund I collaboration",
      "mentioned_at": "Summary section: Investment Fund Launch"
    }
  ]
}

# Quality Checks:
- ✅ Action is specific and clear
- ✅ Deadline is realistic (not past, not >90 days)
- ✅ Context explains WHY (not just WHAT)
- ✅ Entity links use [[brackets]]
- ❌ No vague TODOs: "Think about X", "Consider Y" (unless explicitly committed)
- ❌ No duplicate TODOs

Meeting date: {meeting_date}
Current date: {current_date}

# Meeting Summary:
{summary}

# Full Transcript (for additional context):
{transcript}

Extract ALL actionable TODOs as JSON (todos array).
"""


def format_transcript_excerpt(transcript: List[Dict], max_entries: int = 50) -> str:
    """Format transcript into readable text."""
    if not transcript:
        return "(No transcript available)"

    lines = []
    for i, entry in enumerate(transcript[:max_entries]):
        speaker = entry.get('speaker', 'Unknown')
        text = entry.get('text', '')
        lines.append(f"{speaker}: {text}")

    if len(transcript) > max_entries:
        lines.append(f"\n... ({len(transcript) - max_entries} more transcript entries)")

    return '\n'.join(lines)


def extract_todos_from_meeting(
    meeting_date: str,
    summary: str,
    transcript: List[Dict],
    current_date: Optional[str] = None
) -> List[Dict]:
    """
    Extract TODOs from meeting using GPT-5-nano.

    Args:
        meeting_date: YYYY-MM-DD format
        summary: AI-generated meeting summary
        transcript: Full transcript (list of speaker entries)
        current_date: YYYY-MM-DD (defaults to today)

    Returns:
        List of TODO dicts with: action, owner, deadline, priority, context
    """
    from datetime import datetime

    if not current_date:
        current_date = datetime.now().strftime('%Y-%m-%d')

    # Format transcript (limit to avoid token limits)
    transcript_text = format_transcript_excerpt(transcript, max_entries=100)

    # Build prompt
    prompt = TODO_EXTRACTION_PROMPT.format(
        meeting_date=meeting_date,
        current_date=current_date,
        summary=summary or "(No summary)",
        transcript=transcript_text
    )

    # Call GPT-5-nano with JSON mode
    try:
        response = call_openai(
            prompt,
            model="gpt-5-nano-2025-08-07",
            response_format="json",
            max_tokens=2000
        )

        # Parse JSON response
        result = json.loads(response)
        todos = result.get('todos', [])

        # Validate and clean
        validated_todos = []
        for todo in todos:
            # Required fields
            if not todo.get('action'):
                continue

            # Ensure all fields exist
            validated_todo = {
                'action': todo.get('action', ''),
                'owner': todo.get('owner', 'You'),
                'deadline': todo.get('deadline', ''),
                'deadline_confidence': todo.get('deadline_confidence', 'medium'),
                'priority': todo.get('priority', 'medium'),
                'context': todo.get('context', ''),
                'mentioned_at': todo.get('mentioned_at', '')
            }

            validated_todos.append(validated_todo)

        return validated_todos

    except Exception as e:
        print(f"Error extracting TODOs: {e}")
        return []


def estimate_cost(summary_length: int, transcript_length: int) -> float:
    """
    Estimate cost for TODO extraction.

    GPT-5-nano pricing:
    - Input: $0.05 per 1M tokens
    - Output: $0.40 per 1M tokens

    Rough estimate: 1 token ≈ 0.75 words
    """
    # Estimate tokens
    summary_tokens = summary_length / 3  # chars to tokens
    transcript_tokens = transcript_length * 3  # entries * avg tokens per entry
    prompt_tokens = 800  # System prompt overhead

    total_input_tokens = summary_tokens + transcript_tokens + prompt_tokens
    output_tokens = 500  # Typical TODO output

    input_cost = (total_input_tokens / 1_000_000) * 0.05
    output_cost = (output_tokens / 1_000_000) * 0.40

    return input_cost + output_cost


def main():
    """Test TODO extraction."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract TODOs from meeting')
    parser.add_argument('--date', required=True, help='Meeting date YYYY-MM-DD')
    parser.add_argument('--summary', required=True, help='Meeting summary text')
    parser.add_argument('--transcript-file', help='Path to transcript JSON file')

    args = parser.parse_args()

    # Load transcript if provided
    transcript = []
    if args.transcript_file:
        transcript_path = Path(args.transcript_file)
        if transcript_path.exists():
            transcript = json.loads(transcript_path.read_text())

    # Extract TODOs
    print(f"Extracting TODOs from meeting on {args.date}...")

    # Estimate cost
    cost = estimate_cost(len(args.summary), len(transcript))
    print(f"Estimated cost: ${cost:.4f}")

    todos = extract_todos_from_meeting(
        args.date,
        args.summary,
        transcript
    )

    # Display results
    print(f"\n{'='*70}")
    print(f"EXTRACTED {len(todos)} TODOs")
    print(f"{'='*70}\n")

    for i, todo in enumerate(todos, 1):
        print(f"{i}. {todo['action']}")
        print(f"   Owner: {todo['owner']}")
        print(f"   Deadline: {todo['deadline']} ({todo['deadline_confidence']} confidence)")
        print(f"   Priority: {todo['priority']}")
        print(f"   Context: {todo['context']}")
        if todo['mentioned_at']:
            print(f"   Mentioned: {todo['mentioned_at']}")
        print()


if __name__ == '__main__':
    main()
