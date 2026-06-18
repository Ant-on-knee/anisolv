"""Sample: hydration free energies (ΔG_solv in water) via the anisolv delta-corrector.

takes a handful of small solutes, obtains the water solvation correction, and reports it against
experimental hydration free energies.

Run from the repo root (torch + numpy + ASE; uses the bundled checkpoints/model1.pt):

    python anisolv/samples/H2O_single_point.py

For the full thermodynamic cycle (geometry relaxation + harmonic vibrational ΔG) on a single
H2O, see the companion H2O_dGsolv.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Make `anisolv` importable when run straight from a checkout (repo root = parents[2]).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ase.build import molecule  # noqa: E402

from anisolv import predict_solvation_energy  # noqa: E402

EV_TO_KCAL = 23.060548  # 1 eV in kcal/mol

# (ASE g2 name, label, experimental ΔG_hyd in kcal/mol). Experimental hydration free
# energies from the FreeSolv / MNSol compilations. All neutral closed-shell singlets, so
# charge=0, spin=1 (the predict defaults).
SOLUTES = [
    ("CH4",   "methane",      +1.99),
    ("C2H6",  "ethane",       +1.83),
    ("NH3",   "ammonia",      -4.29),
    ("H2O",   "water",        -6.31),
    ("CH3OH", "methanol",     -5.11),
    ("CH3CN", "acetonitrile", -3.89),
]


def main() -> int:
    dE0, _ = predict_solvation_energy(molecule("H2O"), solvent=None)
    assert dE0 == 0.0, f"vacuum baseline not zero: {dE0} eV"
    print(f"[check] vacuum baseline ΔE(H2O, solvent=None) = {dE0:.1f} eV  (expected exactly 0)\n")

    print(f"{'solute':14s} {'formula':10s} {'n':>3} "
          f"{'dG_pred':>9} {'dG_exp':>9} {'error':>9}   (kcal/mol)")
    errs = []
    for g2_name, label, dG_exp in SOLUTES:
        atoms = molecule(g2_name)
        dE_eV, _dF = predict_solvation_energy(atoms, charge=0, spin=1, solvent="water")
        dG_pred = dE_eV * EV_TO_KCAL
        err = dG_pred - dG_exp
        errs.append(err)
        print(f"{label:14s} {atoms.get_chemical_formula():10s} {len(atoms):3d} "
              f"{dG_pred:+9.2f} {dG_exp:+9.2f} {err:+9.2f}")

    # Error summary, same statistics run_mnsol._summarize reports.
    # n = len(errs)
    # mae = sum(abs(e) for e in errs) / n
    # rmse = math.sqrt(sum(e * e for e in errs) / n)
    # bias = sum(errs) / n
    # print(f"\nvs experiment over {n} solute(s) in water:")
    # print(f"    MAE  = {mae:.2f} kcal/mol")
    # print(f"    RMSE = {rmse:.2f} kcal/mol")
    # print(f"    bias = {bias:+.2f} kcal/mol (predicted - experiment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
