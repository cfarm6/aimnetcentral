"""Minimal simulator workflow entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimnet.ttlang import BackendMode

from _common import run_energy

if __name__ == "__main__":
    print("Running TT-Lang AIMNet2 energy prediction (tt-sim path)")
    run_energy(BackendMode.TT_SIM)
