"""Bounded12 prior-normalization target engineering; no full or query entry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_normalized_edge_target_runtime import main

if __name__ == "__main__":
    main()
