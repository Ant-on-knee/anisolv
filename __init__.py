"""anisolv - package which adds a standalone solvation correction for a given geometry and solvent

Loads a converted anisolv checkpoint and returns dE = E_solv - E_gas plus the
matching force correction, as an additive term for any gas-phase potential. Depends only
on PyTorch + numpy (ASE optional, for I/O). See DOCS.md.

    from anisolv import predict_solvation_energy
    dE, dF = predict_solvation_energy((atomic_numbers, positions), charge=0, spin=1)
"""

from .model import load_model
from .predict import predict_solvation_energy

__all__ = ["predict_solvation_energy", "load_model"]