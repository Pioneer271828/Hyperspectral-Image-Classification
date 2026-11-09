"""Run one case from an independently registered full light-target experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g4_light_full_target import main


if __name__ == "__main__":
    main()
