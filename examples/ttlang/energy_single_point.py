"""Single-point energy prediction for caffeine using TT-Lang AIMNet2."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimnet.ttlang import BackendMode

from _common import run_energy

if __name__ == "__main__":
    run_energy(BackendMode.TT_SIM, label="tt-sim")
