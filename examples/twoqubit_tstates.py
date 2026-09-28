"""Compute discrimination of the two-qubit mixture pair, optionally with T states."""

from pathlib import Path
import sys

# Find the adjacent library even when this file is run from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stabdisc.cli import run_example


if __name__ == "__main__":
    run_example("twoqubit_tstates")
