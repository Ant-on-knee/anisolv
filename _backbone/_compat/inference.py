"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Constants + enums the backbone imports from fairchem's inference API.

Values copied verbatim from fairchem.core.units.mlip_unit.api.inference. `InferenceSettings`
is a thin dataclass carrying only the fields the vendored backbone reads; the standalone
loader (model.py) sets the few that matter (use_quaternion_wigner=False, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

CHARGE_RANGE = [-100, 100]
DEFAULT_CHARGE = 0
DEFAULT_SPIN = 0
DEFAULT_SPIN_OMOL = 1
SPIN_RANGE = [0, 100]


class UMATask(str, Enum):
    OMOL = "omol"
    OMAT = "omat"
    ODAC = "odac"
    OC20 = "oc20"
    OC25 = "oc25"
    OMC = "omc"


@dataclass
class InferenceSettings:
    tf32: bool = False
    activation_checkpointing: bool = False
    merge_mole: bool = False
    compile: bool = False
    external_graph_gen: bool = True
    internal_graph_gen_version: int = 2
    use_quaternion_wigner: bool = False
    execution_mode: str | None = "general"
    edge_chunk_size: int | None = None
    predict_untrained_forces: set = field(default_factory=set)
    predict_untrained_stress: set = field(default_factory=set)
    predict_untrained_hessian: set = field(default_factory=set)
    auto_add_default_untrained_tasks: bool = True
