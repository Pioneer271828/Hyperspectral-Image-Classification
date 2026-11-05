"""Twelve-update engineering worker for the frozen six-target none launch."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from none_full_target_launch import worker_main

if __name__ == "__main__":
    worker_main()
