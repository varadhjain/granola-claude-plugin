#!/usr/bin/env python3
"""
Simple intelligence extraction using plaintext markdown output.

No JSON parsing - just structured markdown that's easier to parse.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from openai_client import call_openai


SIMPLE_EXTRACTION_PROMPT = """Analyze this meeting and extract intelligence in simple markdown format.

# Meeting: {title}
Date: {date}
Attendees: {attendees}

# Content:
{content}

---

# EXTRACT TO MARKDOWN:

## People Intelligence
For each attendee, list:
- Name + Email
- Role/Company
- Key insights (background, interests, what they're working on)
- Network connections mentioned

## Companies Discussed
List any companies mentioned 3+ times:
- Company name
- Stage/Funding info
- Why discussed

## Network Connections
Who knows who? Format: "PersonA → PersonB (context)"

## TODOs
Extract action items:
- TODO: [Action] - Owner - Deadline - Why

## Key Insights
- Investment opportunities
- Market insights
- Warm intro offers

Keep it concise. Focus on facts mentioned explicitly.
"""


def extract_intelligence_simple(meeting: dict) -> str:
    """Extract intelligence as simple markdown."""

    # Build content
    content_parts = []
    if meeting.get('your_notes'):
        content_parts.append(f"# Your Notes:\n{meeting['your_notes']}\n")
    if meeting.get('ai_summary'):
        content_parts.append(f"# AI Summary:\n{meeting['ai_summary']}\n")

    content = '\n'.join(content_parts) if content_parts else "(No content)"

    attendees = ', '.join([f"{a['name']} ({a['email']})" for a in meeting.get('attendees', [])])

    prompt = SIMPLE_EXTRACTION_PROMPT.format(
        title=meeting['title'],
        date=meeting['date'],
        attendees=attendees or "(No attendees)",
        content=content
    )

    # Call GPT-5-nano without max_tokens (causes empty responses)
    response = call_openai(
        prompt,
        model="gpt-5-nano-2025-08-07"
    )

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='database/cache/batch_meetings.json')
    parser.add_argument('--output', default='database/cache/extracted_intel_simple.md')
    parser.add_argument('--limit', type=int, help='Process only N meetings')

    args = parser.parse_args()

    # Load meetings
    meetings = json.loads(Path(args.input).read_text())
    print(f"Loaded {len(meetings)} meetings")

    if args.limit:
        meetings = meetings[:args.limit]
        print(f"Processing first {len(meetings)} meetings")

    # Extract intelligence
    output = []
    output.append("# Granola Meeting Intelligence Extraction")
    output.append(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    output.append(f"Total meetings: {len(meetings)}\n")
    output.append("="*70 + "\n")

    for i, meeting in enumerate(meetings, 1):
        print(f"\nProcessing {i}/{len(meetings)}: {meeting['title']}")

        try:
            intel = extract_intelligence_simple(meeting)
            print(f"  Extracted {len(intel)} chars")

            output.append(f"\n# Meeting {i}: {meeting['title']}")
            output.append(f"Date: {meeting['date']} | ID: {meeting['doc_id'][:8]}\n")
            output.append(intel)
            output.append("\n" + "="*70 + "\n")

        except Exception as e:
            print(f"  Error: {e}")
            output.append(f"\n# Meeting {i}: {meeting['title']}")
            output.append(f"ERROR: {e}\n")
            output.append("="*70 + "\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(output))

    print(f"\n✓ Saved to {output_path}")
    print(f"✓ {len(meetings)} meetings processed")


if __name__ == '__main__':
    main()
