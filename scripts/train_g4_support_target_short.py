"""Twelve-update ordered target engineering; no full campaign or query access."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_support_target import main

if __name__ == "__main__":
    main()
