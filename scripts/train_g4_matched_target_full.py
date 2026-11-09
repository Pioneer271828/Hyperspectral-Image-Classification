"""Registered full1500 target worker; never accepts engineering-only preflight."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_matched_target_execution import worker_main

if __name__ == "__main__":
    worker_main()
