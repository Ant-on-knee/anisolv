"""Solvent conditioning vector for the solvent-embedding checkpoint variant.

get_solvent_vector(name) -> (1, 8) float32: seven vacuum-anchored normalized descriptors
(transform(raw)/scale, no mean subtraction, so gas phase = exactly 0 in every channel) + a
solvent-present mask. None / "vacuum" / "gas" -> the all-zero null vector.
"""

from __future__ import annotations

import functools
import json
import logging
import math
from pathlib import Path

import torch

_JSON_PATH = Path(__file__).resolve().parent / "_const" / "solvent_descriptors.json"

# Seven descriptors in the order SolventEmbedding expects (n25 excluded: ~collinear with n).
SOLVENT_DESCRIPTOR_ORDER = ["n", "alpha", "beta", "gamma", "epsilon", "aromaticity", "en-halogen"]
SOLVENT_DIM = len(SOLVENT_DESCRIPTOR_ORDER) + 1  # + solvent-present mask

# gas phase maps to 0 in every channel
# epsilon is scaled by log to mimic Born model (1-1/epsilon decays too quickly for large epsilon)
# refractive indices are shifted by 1
# must be consistent with fairchem-solvation fork
_SOLVENT_STATS = {
    "n": {"transform": "shift1", "scale": 0.068400},
    "alpha": {"transform": "linear", "scale": 0.181746},
    "beta": {"transform": "linear", "scale": 0.234892},
    "gamma": {"transform": "linear", "scale": 11.688257},
    "epsilon": {"transform": "log", "scale": 0.960804},
    "aromaticity": {"transform": "linear", "scale": 0.322355},
    "en-halogen": {"transform": "linear", "scale": 0.174984},
}


def _transform(name: str, value: float) -> float:
    """Apply a descriptor's vacuum-anchoring transform (0 at the gas phase)."""
    transform = _SOLVENT_STATS[name]["transform"]
    if transform == "shift1":
        return float(value) - 1.0
    if transform == "log":
        return math.log(value)
    return float(value)

_VACUUM_NAMES = {"", "vacuum", "gas", "gas_phase", "gas-phase", "none"}


@functools.lru_cache(maxsize=1)
def _load_raw() -> dict:
    with _JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _solvents_ci() -> dict:
    """Case-insensitive index of the solvent table (some keys carry uppercase, e.g.
    'dimethyl sulfoxide (DMSO)', 'N,N-dimethylformamide', 'E-1,2-dichloroethene')."""
    return {k.lower(): v for k, v in _load_raw()["solvents"].items()}


def list_solvents() -> list:
    return sorted(_load_raw()["solvents"].keys())


def normalize(raw_vec) -> list:
    if len(raw_vec) != len(SOLVENT_DESCRIPTOR_ORDER):
        raise ValueError(f"raw_vec must have {len(SOLVENT_DESCRIPTOR_ORDER)} values, got {len(raw_vec)}")
    return [
        _transform(name, value) / _SOLVENT_STATS[name]["scale"]
        for name, value in zip(SOLVENT_DESCRIPTOR_ORDER, raw_vec)
    ]


def get_solvent_vector(solvent_name, strict: bool = True) -> torch.Tensor:
    """Build the (1, SOLVENT_DIM) conditioning vector. None/vacuum -> null vector."""
    vec = torch.zeros(1, SOLVENT_DIM, dtype=torch.float32)
    if solvent_name is None:
        return vec
    key = str(solvent_name).strip().lower()
    if key in _VACUUM_NAMES:
        return vec
    solvents = _solvents_ci()
    if key not in solvents:
        if strict:
            raise KeyError(
                f"Unknown solvent '{solvent_name}'. Use strict=False for the vacuum "
                f"vector, or see list_solvents()."
            )
        logging.warning("Unknown solvent '%s'; using the vacuum vector.", solvent_name)
        return vec
    raw = [solvents[key][name] for name in SOLVENT_DESCRIPTOR_ORDER]
    vec[0, : len(SOLVENT_DESCRIPTOR_ORDER)] = torch.tensor(normalize(raw), dtype=torch.float32)
    vec[0, len(SOLVENT_DESCRIPTOR_ORDER)] = 1.0
    return vec
