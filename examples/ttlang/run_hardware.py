"""Minimal hardware workflow entry point (requires Tenstorrent device)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimnet.ttlang import BackendMode, TTBackendUnavailableError

from _common import run_energy

if __name__ == "__main__":
    if not os.path.exists("/dev/tenstorrent/0"):
        print("Tenstorrent device not found at /dev/tenstorrent/0", file=sys.stderr)
        sys.exit(1)
    print("Running TT-Lang AIMNet2 energy prediction (tt-hw path)")
    try:
        run_energy(BackendMode.TT_HW)
    except TTBackendUnavailableError as exc:
        print(f"Hardware backend unavailable: {exc}", file=sys.stderr)
        sys.exit(1)
