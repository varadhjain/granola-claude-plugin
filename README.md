# Granola Claude Plugin

Extract intelligence from Granola meeting notes with AI.

## Install

```bash
# In Claude Code:
/plugin install github.com/varadhjain/granola-claude-plugin

# Setup API key:
mkdir -p ~/.granola-claude
echo "OPENAI_API_KEY=sk-your_key_here" > ~/.granola-claude/.env

# Use it:
"Extract my last 7 days of Granola meetings"
```

## Features

- Extract meetings with attendees, notes, AI summaries
- AI-powered intelligence: people, companies, TODOs, insights
- Markdown export for any note-taking app
- Privacy-first: Your API key, your data stays local

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
- OpenAI API key (~$0.001 per meeting)
- Claude Code

## Output

Files saved to: `~/.granola-claude/output/`

Format: `YYYY-MM-DD-meeting-title-attendee.md`

Each file contains:
- Meeting metadata (title, date, attendees)
- Your notes + AI summary
- Extracted intelligence (people, companies, TODOs, insights)

## Cost

- GPT-5-nano: ~$0.0009 per meeting
- 30 meetings/month: ~$0.03

## Privacy

- No data collection
- Transcripts stay on your machine
- Only sent to your OpenAI account
- You control your data

## Troubleshooting

**"Granola cache not found"**
Install Granola and record at least one meeting.

**"OpenAI API key not found"**
Create `~/.granola-claude/.env` with:
```
OPENAI_API_KEY=sk-your_key_here
```

**"No meetings found"**
Check date range (default: last 7 days).

## License

MIT
