"""Demo: train the NumPy U-Net and write the visual report to outputs/.

Run from the project root:  python examples/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainseg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo"]))
