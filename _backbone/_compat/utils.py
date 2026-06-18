"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

from fairchem.core.common.utils
"""

from __future__ import annotations

from functools import wraps


def conditional_grad(dec):
    "Decorator to enable/disable grad depending on whether force/energy predictions are being made"

    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if self.regress_forces and not getattr(self, "direct_forces", 0):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator
