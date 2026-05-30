"""Entry point for AI Movie application."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_movie.gui.app import run_gui


def main():
    run_gui()


if __name__ == "__main__":
    main()
