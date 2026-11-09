"""Readonly audit of frozen observer provenance and conservative claim permissions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_observation_provenance import main

if __name__ == "__main__":
    main()
