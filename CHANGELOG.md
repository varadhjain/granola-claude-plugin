# Changelog

## [1.0.0] - 2026-01-15

### Added
- Initial release
- Extract meetings from Granola cache
- AI intelligence extraction (people, companies, TODOs, insights)
- Markdown export to `~/.granola-claude/output/`
- GPT-5-nano for cost-effective extraction (~$0.0009/meeting)
- First-run setup with guided instructions
- Portable .env loading (checks multiple locations)
- Clean filename format: `YYYY-MM-DD-title-attendee.md`

### Features
- Extract recent meetings by date range
- Filter by number of meetings
- Optional AI intelligence extraction (can disable for speed)
- Setup check command
- Privacy-first (local API key, no tracking)
