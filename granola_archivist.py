#!/usr/bin/env python3
"""
Granola Archivist CLI wrapper.

Allows running the extractor from any terminal or LLM CLI.
"""

import sys
from pathlib import Path

# Add skills/granola to path for imports
sys.path.insert(0, str(Path(__file__).parent / "skills" / "granola"))

from extract import main  # noqa: E402


if __name__ == "__main__":
    main()
