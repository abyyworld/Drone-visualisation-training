"""``python -m station`` entry point.

Kept to one line of behaviour so that ``station.cli.main`` stays importable and
testable on its own -- a test that had to spawn a subprocess to check an exit
status would be a test nobody runs.
"""

from __future__ import annotations

import sys

from station.cli import main

if __name__ == "__main__":
    sys.exit(main())
