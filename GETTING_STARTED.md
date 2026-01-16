# Getting Started with Granola Plugin

Quick start guide for using the Granola Claude Code plugin.

---

## Prerequisites

- macOS (Granola is macOS-only)
- Granola installed with at least one recorded meeting
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Claude Code installed

---

## Installation (30 seconds)

### 1. Install Plugin

In Claude Code:
```
/plugin install github.com/varadhjain/granola-claude-plugin
```

### 2. Setup API Key

Run in terminal:
```bash
mkdir -p ~/.granola-claude
echo "OPENAI_API_KEY=sk-your_key_here" > ~/.granola-claude/.env
```

Replace `sk-your_key_here` with your actual OpenAI API key.

### 3. Test It

In Claude Code:
```
"Extract my last 3 Granola meetings"
```

---

## Usage Examples

### Extract Recent Meetings
```
"Extract my last 7 days of Granola meetings"
"Extract my last 5 meetings"
```

### Get Intelligence
```
"What companies did I discuss this week?"
"Who did I meet with yesterday?"
"What TODOs came from my meetings?"
```

### Find Specific Meeting
```
"Show me my meeting with Kristina from yesterday"
"What did we discuss in my call with Camden?"
```

---

## Where Are My Files?

All meetings saved to: `~/.granola-claude/output/`

You can open them with:
```bash
open ~/.granola-claude/output/
```

Or copy to your notes app:
```bash
cp ~/.granola-claude/output/*.md ~/Documents/Notes/
```

---

## Cost

**With AI extraction**:
- ~$0.0009 per meeting
- 30 meetings/month: ~$0.03

**Without AI** (faster, free):
```
"Extract my meetings without AI"
```

---

## Troubleshooting

### "Granola cache not found"
Install Granola and record at least one meeting.

### "OpenAI API key not found"
Create `~/.granola-claude/.env` with:
```
OPENAI_API_KEY=sk-your_key_here
```

### "No meetings found"
Check date range (default: last 7 days).
Or try: `"Extract my last 30 days of meetings"`

---

## Privacy

- No data collection or tracking
- Transcripts stay on your machine
- Only sent to your OpenAI account
- You control your data

---

## Questions?

File an issue: https://github.com/varadhjain/granola-claude-plugin/issues
