from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BackendMode(StrEnum):
    """Execution backend for TTLangAIMNet2."""

    REFERENCE = "reference"
    TT_SIM = "tt-sim"
    TT_HW = "tt-hw"


@dataclass
class TTBackendConfig:
    """Configuration for a Tenstorrent execution backend."""

    mode: BackendMode
    device_id: int = 0

    @property
    def uses_ttnn(self) -> bool:
        return self.mode is BackendMode.TT_HW

    @property
    def uses_torch_mlp(self) -> bool:
        """tt-sim executes MLP on host torch (functional simulator path)."""
        return self.mode is BackendMode.TT_SIM
