# Getting Started with Granola Archivist

Quick start guide for using Granola Archivist with Claude Code or any CLI.

---

## Prerequisites

- macOS (Granola is macOS-only)
- Granola installed with at least one recorded meeting
- OpenAI API key (optional, only for AI extraction)
- Claude Code installed (optional; plugin name: `granola-archivist`)

---

## Installation (30 seconds)

### 1. Install Plugin (Claude Code)

In Claude Code:
```
/plugin install github.com/varadhjain/granola-claude-plugin
```

Or via marketplace:
```
/plugin marketplace add varadhjain/granola-claude-plugin
```

### 2. Setup API Key (Optional)

Run in terminal:
```bash
mkdir -p ~/.granola-archivist
echo "OPENAI_API_KEY=sk-your_key_here" > ~/.granola-archivist/.env
```

Replace `sk-your_key_here` with your actual OpenAI API key.
Skip this step if you only want raw meeting exports.

### 3. Test It (Claude Code)

In Claude Code:
```
"Extract my last 3 Granola meetings"
```

Without AI (no key needed):
```
"Extract my last 3 Granola meetings without AI"
```

---

## CLI Usage (Any Terminal or Codex CLI)

```bash
python granola-claude-plugin/granola_archivist.py --days 7
python granola-claude-plugin/granola_archivist.py --days 7 --no-intelligence
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

All meetings saved to: `~/.granola-archivist/output/`

You can open them with:
```bash
open ~/.granola-archivist/output/
```

Or copy to your notes app:
```bash
cp ~/.granola-archivist/output/*.md ~/Documents/Notes/
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
Only required for AI extraction. Create `~/.granola-archivist/.env` with:
```
OPENAI_API_KEY=sk-your_key_here
```

### Legacy config
If you already use `~/.granola-claude/`, it will still be detected.

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
