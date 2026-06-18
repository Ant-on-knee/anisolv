"""No-op replacement for fairchem.core.common.registry.
"""

from __future__ import annotations


class _Registry:
    def register_model(self, *_args, **_kwargs):
        def deco(cls):
            return cls

        return deco


registry = _Registry()
