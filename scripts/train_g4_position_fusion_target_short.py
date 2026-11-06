"""Bounded12 position-fusion engineering; real full campaigns/query are separate gates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_position_fusion_target_runtime import main

if __name__ == "__main__":
    main()

