# Granola Archivist

Extract and archive Granola meeting notes with optional AI intelligence.
Runs locally by default; the project author never sees your data.

## Install

```bash
# Claude Code plugin install:
/plugin install github.com/varadhjain/granola-claude-plugin

# Or via marketplace:
/plugin marketplace add varadhjain/granola-claude-plugin

# Optional: enable AI extraction
mkdir -p ~/.granola-archivist
echo "OPENAI_API_KEY=sk-your_key_here" > ~/.granola-archivist/.env

# Use it:
"Extract my last 7 days of Granola meetings"

# Or without AI (no key needed):
"Extract my last 7 days of Granola meetings without AI"
```

## CLI (Any Terminal or Codex CLI)

```bash
python granola-claude-plugin/granola_archivist.py --days 7
python granola-claude-plugin/granola_archivist.py --days 7 --no-intelligence
```

## Features

- Extract meetings with attendees, notes, AI summaries
- AI-powered intelligence: people, companies, TODOs, insights
- Markdown export for any note-taking app
- Privacy-first: local by default; your API key and data stay on your machine

## Usage

### Extract Recent Meetings
```
"Extract my last 5 Granola meetings"
```

### Get Company Intelligence
```
"What companies did I discuss this week?"
```

### Extract Action Items
```
"What TODOs came from my meetings?"
```

## Requirements

- macOS (Granola is macOS-only)
- Granola installed with at least one meeting
- OpenAI API key (optional, only for AI extraction)
- Claude Code (optional; plugin name: `granola-archivist`)

## Output

Files saved to: `~/.granola-archivist/output/`

Format: `YYYY-MM-DD-meeting-title-attendee.md`

Each file contains:
- Meeting metadata (title, date, attendees)
- Your notes + AI summary
- Extracted intelligence (people, companies, TODOs, insights)

## Cost

- GPT-5-nano: ~$0.0009 per meeting
- 30 meetings/month: ~$0.03

## Privacy

- No data collection or telemetry
- Author never receives or sees your data
- Transcripts stay on your machine by default
- If you enable AI, data is sent only to your OpenAI account
- You control your data

## Troubleshooting

**"Granola cache not found"**
Install Granola and record at least one meeting.

**"OpenAI API key not found"**
If you want AI extraction, create `~/.granola-archivist/.env` with:
```
OPENAI_API_KEY=sk-your_key_here
```

**Legacy config**
If you already use `~/.granola-claude/`, it will still be detected.

**"No meetings found"**
Check date range (default: last 7 days).

## License

MIT
