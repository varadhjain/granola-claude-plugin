#!/usr/bin/env python3
"""
Extract meeting data from Granola cache.

Reads ~/Library/Application Support/Granola/cache-v3.json and extracts:
- Meeting metadata (title, date, location)
- Attendees with emails
- Your manual notes
- AI-generated summaries
- Full transcripts

Usage:
    python database/granola_extract_meetings.py --days 7 --limit 10
    python database/granola_extract_meetings.py --meeting-id "abc123"
"""

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Attendee:
    """Meeting attendee with contact info."""
    email: str
    name: str = ""
    response_status: str = ""


@dataclass
class MeetingData:
    """Complete meeting data from Granola."""
    doc_id: str
    title: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    location: str = ""
    attendees: List[Attendee] = field(default_factory=list)
    your_notes: str = ""  # Your manual notes (markdown)
    ai_summary: str = ""  # AI-generated summary (markdown)
    transcript_length: int = 0
    raw_transcript: List[Dict] = field(default_factory=list)


def get_granola_cache_path() -> Path:
    """Get path to Granola cache file."""
    return Path.home() / 'Library' / 'Application Support' / 'Granola' / 'cache-v3.json'


def load_granola_cache() -> Dict:
    """Load and parse Granola cache."""
    cache_path = get_granola_cache_path()

    if not cache_path.exists():
        raise FileNotFoundError(f"Granola cache not found at {cache_path}")

    data = json.loads(cache_path.read_text())

    # Parse nested JSON string if needed
    cache = data.get('cache')
    if isinstance(cache, str):
        cache = json.loads(cache)
    elif cache is None:
        cache = data

    return cache.get('state', {})


def parse_rich_text_to_markdown(content: Dict) -> str:
    """
    Convert Granola's rich text structure to clean markdown.

    Handles:
    - Headings (### Level 3)
    - Paragraphs
    - Bullet lists
    - Ordered lists
    - Text with marks (bold, italic)
    """
    def extract_text(node, level=0):
        if isinstance(node, dict):
            node_type = node.get('type')

            # Text node
            if node_type == 'text':
                text = node.get('text', '')
                # Apply marks (bold, italic)
                marks = node.get('marks', [])
                for mark in marks:
                    if mark.get('type') == 'bold':
                        text = f"**{text}**"
                    elif mark.get('type') == 'italic':
                        text = f"*{text}*"
                return text

            # Heading
            elif node_type == 'heading':
                lvl = node.get('attrs', {}).get('level', 1)
                text = ''.join(extract_text(c) for c in node.get('content', []))
                return f"\n{'#' * lvl} {text}\n"

            # Paragraph
            elif node_type == 'paragraph':
                text = ''.join(extract_text(c) for c in node.get('content', []))
                return f"{text}\n" if text.strip() else ""

            # Bullet list
            elif node_type == 'bulletList':
                items = []
                for item in node.get('content', []):
                    item_text = extract_text(item, level)
                    items.append(item_text)
                return ''.join(items)

            # Ordered list
            elif node_type == 'orderedList':
                items = []
                start = node.get('attrs', {}).get('start', 1)
                for i, item in enumerate(node.get('content', []), start):
                    item_text = extract_text(item, level).replace('• ', f'{i}. ', 1)
                    items.append(item_text)
                return ''.join(items)

            # List item
            elif node_type == 'listItem':
                text = ''
                for c in node.get('content', []):
                    content_text = extract_text(c, level + 1)
                    if content_text.startswith('\n'):
                        text += content_text
                    else:
                        text += f"{'  ' * level}• {content_text}"
                return text

            # Generic node with content
            elif 'content' in node:
                return ''.join(extract_text(c, level) for c in node['content'])

        elif isinstance(node, list):
            return ''.join(extract_text(item, level) for item in node)

        return ''

    return extract_text(content).strip()


def extract_attendees(doc: Dict) -> List[Attendee]:
    """Extract attendee emails from calendar event."""
    attendees = []
    cal_event = doc.get('google_calendar_event') or {}

    for attendee in cal_event.get('attendees', []):
        email = attendee.get('email', '')

        # Skip if no email or marked as self
        if not email or attendee.get('self'):
            continue

        attendees.append(Attendee(
            email=email,
            response_status=attendee.get('responseStatus', '')
        ))

    return attendees


def extract_ai_summary(doc_id: str, document_panels: Dict) -> str:
    """Extract AI summary from documentPanels."""
    if doc_id not in document_panels:
        return ""

    panels_dict = document_panels[doc_id]

    # Find Summary panel
    for panel_id, panel in panels_dict.items():
        if panel.get('title') == 'Summary':
            content = panel.get('content', {})
            return parse_rich_text_to_markdown(content)

    return ""


def get_recent_meetings(
    days_back: int = 7,
    limit: int = 10
) -> List[MeetingData]:
    """
    Get recent meetings from Granola cache.

    Args:
        days_back: How many days back to look
        limit: Maximum number of meetings to return

    Returns:
        List of MeetingData sorted by date (newest first)
    """
    state = load_granola_cache()
    documents = state.get('documents', {})
    transcripts = state.get('transcripts', {})
    document_panels = state.get('documentPanels', {})

    # Calculate date threshold
    threshold = datetime.now() - timedelta(days=days_back)

    # Extract meetings
    meetings = []
    for doc_id, doc in documents.items():
        created_str = doc.get('created_at', '')
        if not created_str:
            continue

        # Parse ISO timestamp
        created_dt = datetime.fromisoformat(created_str.replace('Z', '+00:00'))

        # Filter by date
        if created_dt.replace(tzinfo=None) < threshold:
            continue

        # Extract data
        attendees = extract_attendees(doc)

        # Skip meetings with no external attendees
        if not attendees:
            continue

        notes_md = doc.get('notes_markdown') or ''
        cal_event = doc.get('google_calendar_event') or {}

        meeting = MeetingData(
            doc_id=doc_id,
            title=doc.get('title', 'Untitled'),
            date=created_dt.strftime('%Y-%m-%d'),
            time=created_dt.strftime('%H:%M'),
            location=cal_event.get('location', ''),
            attendees=attendees,
            your_notes=notes_md.strip(),
            ai_summary=extract_ai_summary(doc_id, document_panels),
            transcript_length=len(transcripts.get(doc_id, [])),
            raw_transcript=transcripts.get(doc_id, [])
        )

        meetings.append(meeting)

    # Sort by date (newest first)
    meetings.sort(key=lambda m: (m.date, m.time), reverse=True)

    return meetings[:limit]


def get_meeting_by_id(doc_id: str) -> Optional[MeetingData]:
    """Get specific meeting by document ID."""
    state = load_granola_cache()
    documents = state.get('documents', {})
    transcripts = state.get('transcripts', {})
    document_panels = state.get('documentPanels', {})

    doc = documents.get(doc_id)
    if not doc:
        return None

    created_str = doc.get('created_at', '')
    created_dt = datetime.fromisoformat(created_str.replace('Z', '+00:00'))

    notes_md = doc.get('notes_markdown') or ''
    cal_event = doc.get('google_calendar_event') or {}

    return MeetingData(
        doc_id=doc_id,
        title=doc.get('title', 'Untitled'),
        date=created_dt.strftime('%Y-%m-%d'),
        time=created_dt.strftime('%H:%M'),
        location=cal_event.get('location', ''),
        attendees=extract_attendees(doc),
        your_notes=notes_md.strip(),
        ai_summary=extract_ai_summary(doc_id, document_panels),
        transcript_length=len(transcripts.get(doc_id, [])),
        raw_transcript=transcripts.get(doc_id, [])
    )


def main():
    parser = argparse.ArgumentParser(description='Extract meetings from Granola cache')
    parser.add_argument('--days', type=int, default=7, help='Days back to search')
    parser.add_argument('--limit', type=int, default=10, help='Max meetings to return')
    parser.add_argument('--meeting-id', type=str, help='Get specific meeting by ID')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    try:
        if args.meeting_id:
            meeting = get_meeting_by_id(args.meeting_id)
            if not meeting:
                print(f"Meeting {args.meeting_id} not found")
                return
            meetings = [meeting]
        else:
            meetings = get_recent_meetings(args.days, args.limit)

        if args.json:
            # Output as JSON
            output = []
            for m in meetings:
                output.append({
                    'doc_id': m.doc_id,
                    'title': m.title,
                    'date': m.date,
                    'time': m.time,
                    'location': m.location,
                    'attendees': [
                        {'email': a.email, 'status': a.response_status}
                        for a in m.attendees
                    ],
                    'your_notes_length': len(m.your_notes),
                    'ai_summary_length': len(m.ai_summary),
                    'transcript_length': m.transcript_length
                })
            print(json.dumps(output, indent=2))
        else:
            # Output as text
            print(f"\nFound {len(meetings)} meetings\n")
            print("=" * 70)

            for i, m in enumerate(meetings, 1):
                print(f"\n{i}. {m.title}")
                print(f"   Date: {m.date} @ {m.time}")
                print(f"   Attendees: {', '.join(a.email for a in m.attendees)}")

                if m.location:
                    print(f"   Location: {m.location[:50]}...")

                print(f"   Your notes: {len(m.your_notes)} chars")
                print(f"   AI summary: {len(m.ai_summary)} chars")
                print(f"   Transcript: {m.transcript_length} entries")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
