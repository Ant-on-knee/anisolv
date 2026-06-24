"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Register triton_op kernels with torch.ops.anisolv.* on package import
# This must happen before NodeToEdgeWignerPermuteFunction/PermuteWignerInvEdgeToNodeFunction
# are used, as they call torch.ops.anisolv._kernel_* in forward/backward
from . import custom_ops  # noqa: F401

from .node_to_edge_wigner_permute import (
    NodeToEdgeWignerPermuteFunction as UMASFastGPUNodeToEdgeWignerPermute,
)
from .permute_wigner_inv_edge_to_node import (
    PermuteWignerInvEdgeToNodeFunction as UMASFastGPUPermuteWignerInvEdgeToNode,
)

__all__ = [
    "UMASFastGPUNodeToEdgeWignerPermute",
    "UMASFastGPUPermuteWignerInvEdgeToNode",
]
