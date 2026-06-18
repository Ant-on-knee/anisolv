"""Minimal stand-ins for fairchem.core.models.base interfaces.
"""

from __future__ import annotations

class HeadInterface:
    @property
    def use_amp(self) -> bool:
        return False

class BackboneInterface:
    pass
