"""Bounded full-source head-only and VCA-adapter target controls."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_light_target import main

if __name__ == "__main__":
    main()
