"""Engineering-only zero-update preflight; this entrance cannot start training."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_matched_target_launch import preflight_main

if __name__ == "__main__":
    preflight_main()
