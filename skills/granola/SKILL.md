---
name: granola
description: Extract and analyze Granola meeting notes. USE WHEN user asks to "extract granola meetings", "analyze my meetings", or "what meetings did I have". OUTPUT structured markdown with people, companies, TODOs, and insights to ~/.granola-claude/output/
---

# Granola Meeting Extraction

Extract intelligence from your Granola meeting notes using AI.

## When to Use

- User asks: "Extract my last 7 days of Granola meetings"
- User asks: "Analyze my meeting with [Person]"
- User asks: "What companies did I discuss this week?"
- User asks: "What TODOs came from my meetings?"

## Requirements

1. **Granola installed** on macOS with at least one recorded meeting
2. **OpenAI API key** stored in `~/.granola-claude/.env`

## Setup Instructions

### First-Time Setup

Tell the user:
```
To use the Granola plugin, you need to set up your OpenAI API key.

Run these commands:
mkdir -p ~/.granola-claude
echo "OPENAI_API_KEY=sk-your_key_here" > ~/.granola-claude/.env

Then try: "Extract my last 7 days of Granola meetings"
```

### Checking Setup

Before running extraction, check if `.env` exists:
- File exists at `~/.granola-claude/.env` → proceed
- File missing → show setup instructions above

## Usage

### Extract Recent Meetings

**User**: "Extract my last 5 Granola meetings"

**You should**:
1. Check setup (`~/.granola-claude/.env` exists)
2. Run: `python skills/granola/extract.py --days 7`
3. Show results: Meeting titles, attendees, where files were saved

### Get Intelligence from Meetings

**User**: "What companies did I discuss this week?"

**You should**:
1. Extract last 7 days of meetings
2. Parse output files from `~/.granola-claude/output/`
3. Summarize companies mentioned across all meetings

### Extract TODOs

**User**: "What action items came from my meetings?"

**You should**:
1. Extract recent meetings with intelligence
2. Parse TODO sections from output files
3. Present consolidated TODO list with owners and deadlines

## Output Format

Files are saved to: `~/.granola-claude/output/`

Filename format: `YYYY-MM-DD-meeting-title-attendee-name.md`

Each file contains:
- Meeting metadata (title, date, attendees, location)
- Your manual notes from Granola
- AI-generated summary from Granola
- Extracted intelligence:
  - People (with roles, insights, network connections)
  - Companies discussed (with stage, funding, context)
  - Network connections (who knows who)
  - TODOs (with owners, deadlines, context)
  - Key insights (investment opportunities, market insights)

## Examples

### Example 1: Extract Recent Meetings

```
User: "Extract my last 3 Granola meetings"

You: "I'll extract your last 3 Granola meetings. Let me check your setup first..."

[Checks ~/.granola-claude/.env exists]

You: "Setup looks good. Extracting meetings..."

[Runs: python skills/granola/extract.py --days 7 --limit 3]

You: "✓ Extracted 3 meetings:
- 2026-01-14: Fund Discussion with Srihita Vatsavaya
- 2026-01-13: Rollout AI Sync with Camden Ko
- 2026-01-12: Notion Strategy with Jade Doan

Files saved to: ~/.granola-claude/output/

Would you like me to:
1. Show summaries of all 3 meetings
2. Extract companies discussed
3. Pull out action items"
```

### Example 2: Missing Setup

```
User: "Extract my Granola meetings"

You: "I need to set up your OpenAI API key first. Please run:

mkdir -p ~/.granola-claude
echo 'OPENAI_API_KEY=sk-your_key_here' > ~/.granola-claude/.env

Replace 'sk-your_key_here' with your actual OpenAI API key.

Then I'll be able to extract your meetings!"
```

### Example 3: Extract Companies

```
User: "What companies did I discuss this week?"

You: "Let me extract your meetings from the last 7 days and analyze them..."

[Extracts meetings, parses output files]

You: "Here are the companies discussed across 5 meetings this week:

**Rollout AI** (mentioned 8 times)
- Stage: Series A
- Discussed in: Camden meeting, Investor call
- Context: Potential investment target, strong growth

**Notion** (mentioned 5 times)
- Stage: Late-stage
- Discussed in: Jade meeting
- Context: Partnership opportunity

**OpenAI** (mentioned 3 times)
- Discussed in: Multiple meetings
- Context: API integration plans

Would you like more details on any of these?"
```

## Cost Information

Using GPT-5-nano (cheapest model):
- Cost per meeting: ~$0.0009
- 30 meetings/month: ~$0.03

## Troubleshooting

### "Granola cache not found"
- Ensure Granola is installed
- Record at least one meeting
- Check: `~/Library/Application Support/Granola/cache-v3.json` exists

### "OpenAI API key not found"
- Setup .env file: `~/.granola-claude/.env`
- Format: `OPENAI_API_KEY=sk-...`

### "No meetings found"
- Check date range (default: last 7 days)
- Ensure Granola has meetings in that period

## Privacy

- No data collection or tracking
- Transcripts stay on your machine
- Only sent to your OpenAI account (you control your data)
- Your API key = you control your data
