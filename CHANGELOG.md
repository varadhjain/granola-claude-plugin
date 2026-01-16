# Changelog

## [1.1.0] - 2026-01-15

### Added
- Granola Archivist branding and neutral config path (`~/.granola-archivist/`)
- CLI wrapper for non-Claude Code usage (`granola_archivist.py`)
- Claude Code marketplace metadata

### Changed
- OpenAI API key required only when AI extraction is enabled
- Docs updated for Claude Code and generic CLI flows

## [1.0.0] - 2026-01-15

### Added
- Initial release
- Extract meetings from Granola cache
- AI intelligence extraction (people, companies, TODOs, insights)
- Markdown export to `~/.granola-archivist/output/`
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
