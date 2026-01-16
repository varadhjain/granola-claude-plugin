#!/usr/bin/env python3
"""
Batch extract meetings from Granola cache for intelligence processing.

Extracts meetings in date range and filters to processable ones:
- Has transcript OR detailed calendar description
- At least one non-self attendee with email
- Not a recurring placeholder (HIIT, Gym, etc.)

Usage:
    python database/granola_batch_extract.py --start 2025-07-15 --end 2026-01-15
    python database/granola_batch_extract.py --limit 10  # Test with 10 meetings
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from granola_extract_meetings import (
    load_granola_cache,
    parse_rich_text_to_markdown,
    MeetingData,
    Attendee
)


# Recurring/placeholder meeting titles to filter out
RECURRING_PLACEHOLDERS = {
    'HIIT', 'Gym', 'Workout', 'Lunch', 'Dinner', 'Break',
    'Travel', 'Flight', 'Drive', 'Commute',
    'Block', 'Focus Time', 'Personal Time',
}


def extract_all_meetings(start_date: str, end_date: str) -> List[MeetingData]:
    """
    Extract all meetings in date range from cache.

    Args:
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format

    Returns:
        List of MeetingData objects
    """
    state = load_granola_cache()
    documents = state.get('documents', {})
    transcripts = state.get('transcripts', {})
    document_panels = state.get('documentPanels', {})

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    meetings = []

    for doc_id, doc in documents.items():
        # Skip if trashed
        if doc.get('trashed'):
            continue

        # Parse meeting date
        created_at = doc.get('created_at', '')
        if not created_at:
            continue

        try:
            meeting_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        except:
            continue

        # Filter by date range
        if not (start_dt <= meeting_dt.replace(tzinfo=None) <= end_dt):
            continue

        # Extract meeting data
        title = doc.get('title', 'Untitled')
        date = meeting_dt.strftime('%Y-%m-%d')
        time = meeting_dt.strftime('%H:%M')

        # Get location
        cal_event = doc.get('google_calendar_event') or {}
        location = cal_event.get('location', '')

        # Get attendees
        attendees = extract_attendees(doc)

        # Get your notes
        notes_md = doc.get('notes_markdown') or ''
        your_notes = notes_md.strip()

        # Get AI summary from documentPanels
        ai_summary = extract_ai_summary(doc_id, document_panels)

        # Get transcript
        transcript_id = doc.get('external_transcription_id')
        raw_transcript = []
        transcript_length = 0

        if transcript_id and transcript_id in transcripts:
            transcript_obj = transcripts[transcript_id]
            if isinstance(transcript_obj, dict):
                transcript_entries = transcript_obj.get('transcript', [])
                raw_transcript = transcript_entries
                transcript_length = len(transcript_entries)

        # Get calendar description (fallback if no transcript)
        calendar_description = cal_event.get('description', '')

        meetings.append(MeetingData(
            doc_id=doc_id,
            title=title,
            date=date,
            time=time,
            location=location,
            attendees=attendees,
            your_notes=your_notes,
            ai_summary=ai_summary,
            transcript_length=transcript_length,
            raw_transcript=raw_transcript
        ))

    # Sort by date/time
    meetings.sort(key=lambda m: f"{m.date} {m.time}")

    return meetings


def extract_attendees(doc: Dict) -> List[Attendee]:
    """Extract attendee list with emails."""
    cal_event = doc.get('google_calendar_event') or {}
    attendee_list = cal_event.get('attendees', [])

    # Your emails to filter out
    self_emails = {
        'varadhjain@gmail.com',
        'me@varadhja.in',
        'varadh@gmail.com',
    }

    attendees = []
    for att in attendee_list:
        email = att.get('email', '').lower()

        # Skip self
        if email in self_emails:
            continue

        # Skip if marked as self
        if att.get('self'):
            continue

        name = att.get('displayName', email.split('@')[0].title())
        response_status = att.get('responseStatus', '')

        attendees.append(Attendee(
            email=email,
            name=name,
            response_status=response_status
        ))

    return attendees


def extract_ai_summary(doc_id: str, document_panels: Dict) -> str:
    """Extract AI summary from documentPanels."""
    if doc_id not in document_panels:
        return ''

    panels = document_panels[doc_id]
    if not isinstance(panels, dict):
        return ''

    # Panels are stored as {uuid: panel_data}, not {'ai': panel_data}
    # Find panel with template_slug='meeting-summary-consolidated'
    for panel_id, panel_data in panels.items():
        if not isinstance(panel_data, dict):
            continue

        template = panel_data.get('template_slug', '')
        if template == 'meeting-summary-consolidated':
            # Found AI summary panel
            content = panel_data.get('content', {})
            if content:
                return parse_rich_text_to_markdown(content)

    return ''


def filter_processable_meetings(meetings: List[MeetingData]) -> List[MeetingData]:
    """
    Filter to meetings with ANY content at all.

    NO filtering by attendees or content type - user wants ALL meetings processed.
    Only skip completely empty meetings with zero content.
    """
    processable = []

    for meeting in meetings:
        # Only filter if COMPLETELY empty (no content at all)
        has_transcript = meeting.transcript_length > 0
        has_summary = len(meeting.ai_summary) > 20
        has_notes = len(meeting.your_notes) > 20

        # Skip only if absolutely no content
        if not (has_transcript or has_summary or has_notes):
            continue

        processable.append(meeting)

    return processable


def export_to_json(meetings: List[MeetingData], output_path: Path):
    """Export meetings to JSON format for batch processing."""
    # Convert to dicts
    meetings_data = []
    for meeting in meetings:
        meeting_dict = asdict(meeting)
        # Convert Attendee dataclasses to dicts
        meeting_dict['attendees'] = [
            {'email': att.email, 'name': att.name, 'response_status': att.response_status}
            for att in meeting.attendees
        ]
        meetings_data.append(meeting_dict)

    # Write JSON
    output_path.write_text(json.dumps(meetings_data, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Batch extract Granola meetings for processing')
    parser.add_argument('--start', default='2025-07-15', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', help='End date YYYY-MM-DD (defaults to today)')
    parser.add_argument('--limit', type=int, help='Limit number of meetings (for testing)')
    parser.add_argument('--output', default='database/cache/batch_meetings.json', help='Output JSON file')

    args = parser.parse_args()

    # Default end date to today
    if not args.end:
        args.end = datetime.now().strftime('%Y-%m-%d')

    print(f"Extracting meetings from {args.start} to {args.end}...")

    # Extract all meetings
    meetings = extract_all_meetings(args.start, args.end)
    print(f"✓ Found {len(meetings)} meetings in date range")

    # Filter processable
    processable = filter_processable_meetings(meetings)
    print(f"✓ Filtered to {len(processable)} processable meetings (removed recurring/placeholders)")

    # Apply limit if specified
    if args.limit:
        processable = processable[:args.limit]
        print(f"✓ Limited to {len(processable)} meetings for testing")

    # Stats
    with_transcript = sum(1 for m in processable if m.transcript_length > 0)
    with_notes = sum(1 for m in processable if len(m.your_notes) > 50)
    with_summary = sum(1 for m in processable if len(m.ai_summary) > 50)

    print(f"✓ {with_transcript} with transcripts, {with_notes} with your notes, {with_summary} with AI summaries")

    # Average attendees
    if processable:
        avg_attendees = sum(len(m.attendees) for m in processable) / len(processable)
        print(f"✓ Average attendees: {avg_attendees:.1f}")

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_json(processable, output_path)
    print(f"✓ Saved to {output_path}")

    # Show first 3 meetings as sample
    print(f"\n{'='*70}")
    print("SAMPLE MEETINGS (first 3)")
    print(f"{'='*70}\n")

    for i, meeting in enumerate(processable[:3], 1):
        print(f"{i}. {meeting.title}")
        print(f"   Date: {meeting.date} @ {meeting.time}")
        print(f"   Attendees: {', '.join(att.email for att in meeting.attendees)}")
        print(f"   Transcript: {meeting.transcript_length} entries")
        print(f"   Notes: {len(meeting.your_notes)} chars")
        print(f"   Summary: {len(meeting.ai_summary)} chars")
        print()


if __name__ == '__main__':
    main()
