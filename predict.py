"""Public inference API: predict_solvation_energy -> (ΔE in eV, ΔF in eV/Å).

ΔE = E_solv - E_gas is an additive correction: add it onto gas-phase
potential's energy/forces. Uses the shipped model1 checkpoint 
(trained on UMA-S-1.2, 64-expert MoLE + solvent embedding, 
output-gated so vacuum → exactly 0).
"""

from __future__ import annotations

import numpy as np
import torch

from .data import build_atomic_data
from .model import load_model
from .solvent import get_solvent_vector

_MODEL_CACHE: dict = {}

_DEFAULT_SOLVENT = object()

def _get_model(checkpoint, device, dtype):
    key = (str(checkpoint), str(device), str(dtype))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(checkpoint, device=device, dtype=dtype)
    return _MODEL_CACHE[key]

def _unwrap(x, key):
    return x[key] if isinstance(x, dict) else x


def predict_solvation_energy(
    atoms_or_arrays,
    charge: int = 0,
    spin: int = 1,
    solvent=_DEFAULT_SOLVENT,
    checkpoint: str = "model1",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Return (delta_energy_eV: float, delta_forces_eV_per_A: np.ndarray[n_atoms, 3]).

    atoms_or_arrays : an ase.Atoms, or a (atomic_numbers, positions[Å]) tuple.
    charge, spin    : total charge and spin multiplicity (defaults 0 / 1).
    solvent         : solvent name (str), or None for the gas/vacuum baseline. Left unset it
                      defaults to 'water' (the repo's water-SMD target). The model is gated,
                      so solvent=None yields exactly zero ΔE/ΔF.
    checkpoint      : 'model1' (default) or a path to a converted .pt.
    dtype           : torch.float32 (default) or torch.float64.
    """
    model = _get_model(checkpoint, device, dtype)

    if solvent is _DEFAULT_SOLVENT:
        solvent = "water"
    solvent_vec = get_solvent_vector(solvent)

    data = build_atomic_data(atoms_or_arrays, charge=charge, spin=spin,
                             solvent=solvent_vec, dtype=dtype, device=device)

    rmsd = model.norm["energy"]["rmsd"]
    mean = model.norm["energy"]["mean"]

    out = model(data)
    raw_e = _unwrap(out["energy"], "energy").reshape(-1)[0]
    raw_f = _unwrap(out["forces"], "forces")  # [n_atoms, 3], conservative (autograd) forces

    # unnormalize energies and forces
    delta_e = float((raw_e * rmsd + mean).item())
    delta_f = (raw_f * rmsd).detach().cpu().numpy().astype(np.float64)
    return delta_e, delta_f
