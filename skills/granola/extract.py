#!/usr/bin/env python3
"""
Granola extraction wrapper for Claude Code plugin.

Handles:
- First-run setup (API key configuration)
- Meeting extraction from Granola cache
- AI intelligence extraction
- Markdown output to ~/.granola-claude/output/
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Add core to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from granola_extract_meetings import get_recent_meetings, MeetingData
from granola_simple_extract import extract_intelligence_simple


def setup_directories():
    """Ensure output directories exist."""
    config_dir = Path.home() / '.granola-claude'
    output_dir = config_dir / 'output'
    cache_dir = config_dir / 'cache'

    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    return config_dir, output_dir, cache_dir


def check_setup():
    """
    Check if first-run setup is needed.

    Returns:
        dict with status and message
    """
    config_dir = Path.home() / '.granola-claude'
    env_file = config_dir / '.env'

    if not env_file.exists():
        return {
            'status': 'needs_setup',
            'message': (
                "🔧 First-time setup needed!\n\n"
                "This plugin needs your OpenAI API key to work.\n"
                "It will be stored locally at: ~/.granola-claude/.env\n\n"
                "Run:\n"
                "  mkdir -p ~/.granola-claude\n"
                "  echo 'OPENAI_API_KEY=sk-your_key_here' > ~/.granola-claude/.env\n\n"
                "Then try again!"
            )
        }

    # Check if API key is actually set
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv(env_file)

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return {
                'status': 'needs_setup',
                'message': 'OpenAI API key not found in ~/.granola-claude/.env\nPlease add: OPENAI_API_KEY=sk-...'
            }

        return {
            'status': 'ready',
            'message': '✓ Setup complete! OpenAI API key found at ~/.granola-claude/.env'
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Setup error: {e}\n\nPlease ensure:\n1. File exists: ~/.granola-claude/.env\n2. Contains: OPENAI_API_KEY=sk-...\n3. Has correct permissions: chmod 600 ~/.granola-claude/.env'
        }


def slugify(text: str) -> str:
    """Convert text to filename-safe slug."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text[:100]  # Max 100 chars


def format_meeting_filename(meeting: MeetingData) -> str:
    """Generate clean filename for meeting."""
    date = meeting.date
    title = slugify(meeting.title)

    # Get first attendee name
    if meeting.attendees:
        first_attendee = slugify(meeting.attendees[0].email.split('@')[0])
    else:
        first_attendee = 'no-attendees'

    return f"{date}-{title}-{first_attendee}.md"


def format_meeting_markdown(meeting: MeetingData, intel: str = None) -> str:
    """Format meeting as clean markdown."""
    output = []

    # Header
    output.append(f"# Meeting: {meeting.title}")
    output.append(f"Date: {meeting.date} @ {meeting.time}")

    # Attendees
    if meeting.attendees:
        attendees_str = ', '.join([
            f"{a.name or a.email} ({a.email})" for a in meeting.attendees
        ])
        output.append(f"Attendees: {attendees_str}")
    else:
        output.append("Attendees: None")

    if meeting.location:
        output.append(f"Location: {meeting.location}")

    output.append("\n---\n")

    # Your notes
    if meeting.your_notes:
        output.append("## Your Notes")
        output.append(meeting.your_notes)
        output.append("")

    # AI summary
    if meeting.ai_summary:
        output.append("## AI Summary")
        output.append(meeting.ai_summary)
        output.append("")

    # Extracted intelligence
    if intel:
        output.append("## Extracted Intelligence\n")
        output.append(intel)

    return '\n'.join(output)


def extract_meetings(days_back: int = 7, limit: int = None, with_intelligence: bool = True):
    """
    Main extraction function.

    Args:
        days_back: How many days back to extract
        limit: Max number of meetings to process
        with_intelligence: Whether to run AI extraction

    Returns:
        dict with status and results
    """
    # Check setup
    setup_status = check_setup()
    if setup_status['status'] != 'ready':
        return setup_status

    # Setup directories
    _, output_dir, _ = setup_directories()

    # Extract meetings
    try:
        meetings = get_recent_meetings(days_back=days_back, limit=limit or 100)

        if not meetings:
            return {
                'status': 'no_meetings',
                'message': f'No meetings found in last {days_back} days'
            }

        # Apply limit if specified
        if limit:
            meetings = meetings[:limit]

        results = []

        for meeting in meetings:
            # Extract intelligence if requested
            intel = None
            if with_intelligence:
                try:
                    intel = extract_intelligence_simple({
                        'title': meeting.title,
                        'date': meeting.date,
                        'attendees': [
                            {'name': a.name or a.email, 'email': a.email}
                            for a in meeting.attendees
                        ],
                        'your_notes': meeting.your_notes,
                        'ai_summary': meeting.ai_summary
                    })
                except Exception as e:
                    intel = f"⚠️ Intelligence extraction failed: {e}"

            # Format markdown
            markdown = format_meeting_markdown(meeting, intel)

            # Generate filename
            filename = format_meeting_filename(meeting)

            # Save to output directory
            output_path = output_dir / filename
            output_path.write_text(markdown)

            results.append({
                'title': meeting.title,
                'date': meeting.date,
                'attendees': [a.email for a in meeting.attendees],
                'filename': filename,
                'path': str(output_path)
            })

        return {
            'status': 'success',
            'count': len(results),
            'results': results,
            'output_dir': str(output_dir)
        }

    except FileNotFoundError as e:
        return {
            'status': 'error',
            'message': (
                'Granola cache not found.\n\n'
                'Expected location: ~/Library/Application Support/Granola/cache-v3.json\n\n'
                'Please ensure:\n'
                '1. Granola is installed (https://www.granola.so)\n'
                '2. You have recorded at least one meeting\n'
                '3. You are on macOS (Granola is macOS-only)\n\n'
                f'Technical details: {e}'
            )
        }
    except ImportError as e:
        return {
            'status': 'error',
            'message': (
                'Missing Python dependencies.\n\n'
                'Please run: pip install -r requirements.txt\n\n'
                f'Missing module: {e}'
            )
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': (
                f'Extraction error: {e}\n\n'
                'This might be due to:\n'
                '1. Invalid API key - check ~/.granola-claude/.env\n'
                '2. Granola cache format changed - update plugin\n'
                '3. Network issues - check internet connection\n\n'
                'If this persists, please file an issue:\n'
                'https://github.com/varadhjain/granola-claude-plugin/issues'
            )
        }


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='Extract Granola meetings with AI intelligence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract last 7 days
  python extract.py --days 7

  # Extract last 3 meetings
  python extract.py --days 30 --limit 3

  # Extract without AI (faster, cheaper)
  python extract.py --days 7 --no-intelligence

  # Check setup
  python extract.py --check-setup
"""
    )

    parser.add_argument('--days', type=int, default=7,
                       help='Days back to extract (default: 7)')
    parser.add_argument('--limit', type=int,
                       help='Max number of meetings to process')
    parser.add_argument('--no-intelligence', action='store_true',
                       help='Skip AI extraction (faster, no API calls)')
    parser.add_argument('--check-setup', action='store_true',
                       help='Check if setup is complete')

    args = parser.parse_args()

    # Handle check-setup
    if args.check_setup:
        result = check_setup()
        print(result['message'])
        sys.exit(0 if result['status'] == 'ready' else 1)

    # Extract meetings
    print(f"Extracting meetings from last {args.days} days...")
    if args.limit:
        print(f"Limit: {args.limit} meetings")
    if args.no_intelligence:
        print("AI intelligence extraction: DISABLED")

    result = extract_meetings(
        days_back=args.days,
        limit=args.limit,
        with_intelligence=not args.no_intelligence
    )

    # Output result
    if result['status'] == 'success':
        print(f"\n✓ Extracted {result['count']} meetings")
        print(f"✓ Saved to: {result['output_dir']}\n")
        for r in result['results']:
            attendees = ', '.join(r['attendees']) if r['attendees'] else 'No attendees'
            print(f"  • {r['date']}: {r['title']}")
            print(f"    {attendees}")
            print(f"    → {r['filename']}")
            print()
    else:
        print(f"\n❌ {result['message']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
