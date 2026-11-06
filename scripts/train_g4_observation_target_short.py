"""Twelve-update target real-band engineering; no full profile or query."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_observation_target import main

if __name__ == "__main__":
    main()
