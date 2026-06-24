"""Public inference API: predict_solvation_energy -> (dE in eV, dF in eV/angstrom).

dE = E_solv - E_gas is an additive correction: add it onto gas-phase
potential's energy/forces. Uses the shipped model1 checkpoint 
(trained on UMA-S-1.2, 64-expert MoLE + solvent embedding, 
output-gated so vacuum -> exactly 0).
"""

from __future__ import annotations

import numpy as np
import torch

from ._backbone._compat.inference import InferenceSettings, guess_inference_settings
from .data import build_atomic_data
from .model import default_checkpoint, load_model
from .solvent import get_solvent_vector

_MODEL_CACHE: dict = {}

_DEFAULT_SOLVENT = object()

def _get_model(checkpoint, device, dtype, inference_settings):
    # Resolve to the effective settings so the cache key reflects every knob (the backbone-aware
    # downgrade in load_model is deterministic per checkpoint, which is already part of the key).
    settings = guess_inference_settings(inference_settings)
    key = (str(checkpoint), str(device), str(dtype), repr(settings))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(checkpoint, device=device, dtype=dtype,
                                       inference_settings=settings)
    return _MODEL_CACHE[key]

def _unwrap(x, key):
    return x[key] if isinstance(x, dict) else x


def predict_solvation_energy(
    atoms_or_arrays,
    charge: int = 0,
    spin: int = 1,
    solvent=_DEFAULT_SOLVENT,
    checkpoint: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    inference_settings: str | InferenceSettings = "default",
):
    """Return (dEsolv (eV): float, dFsolv (eV/A): np.ndarray[n_atoms, 3]).

    atoms_or_arrays : an ase.Atoms, or a (atomic_numbers, positions[angstrom]) tuple.
    charge, spin    : total charge and spin multiplicity (defaults 0 / 1).
    solvent         : solvent name (str), or None for the gas/vacuum baseline. Left unset it
                      defaults to 'water' (the repo's water-SMD target). The model is gated,
                      so solvent=None yields exactly zero dE/dF.
    checkpoint      : None (default; auto-selects 'model1' if its weights are present, else the
                      bundled 'model1_compact'), a checkpoint name, or a path to a converted .pt.
    dtype           : torch.float32 (default) or torch.float64.
    inference_settings : 'default' (reference implementation), 
                         'fast' (block-GEMM SO2 + tf32 + torch.compile, merge-free), 
                         'fast_gpu' (adds Triton kernels; CUDA-only),
                         or a custom InferenceSettings. 
                         Recommended: use 'fast' for compact models and 'fast_gpu' for MoLE models
    """
    if checkpoint is None:
        checkpoint = default_checkpoint()
    model = _get_model(checkpoint, device, dtype, inference_settings)

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
