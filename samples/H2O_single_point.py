"""Sample: hydration free energies (dG_solv in water) via the anisolv delta-corrector.

takes a handful of small solutes, obtains the water solvation correction, and reports it against
experimental hydration free energies.

Run from the repo root (torch + numpy; ASE is optional -- bundled g2 geometries are used as a
fallback when ASE is not installed; uses the default checkpoint in anisolv/models):

    python anisolv/samples/H2O_single_point.py

To use ASE's geometries instead of the bundled fallback, install the optional extra:

    pip install "anisolv[ase]"

For the full thermodynamic cycle (geometry relaxation + harmonic vibrational dG) on a single
H2O, see the companion H2O_dGsolv.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Make `anisolv` importable when run straight from a checkout (repo root = parents[2]).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from ase.build import molecule  # noqa: E402

    HAVE_ASE = True
except ImportError:  # ASE is optional -- fall back to the bundled geometries in _FALLBACK_GEOM.
    molecule = None
    HAVE_ASE = False

from anisolv import predict_solvation_energy  # noqa: E402

EV_TO_KCAL = 23.060548  # 1 eV in kcal/mol

# (ASE g2 name, label, experimental dG_hyd in kcal/mol). 
# Experimental hydration free energies from the FreeSolv / MNSol compilations. 
# All neutral closed-shell singlets, so charge=0, spin=1 (the predict defaults).
SOLUTES = [
    ("CH4",   "methane",      +1.99),
    ("C2H6",  "ethane",       +1.83),
    ("NH3",   "ammonia",      -4.29),
    ("H2O",   "water",        -6.31),
    ("CH3OH", "methanol",     -5.11),
    ("CH3CN", "acetonitrile", -3.89),
]

# If ASE is not detected
_FALLBACK_GEOM = {
    "CH4": (
        [6, 1, 1, 1, 1],
        [[0.0, 0.0, 0.0], [0.629118, 0.629118, 0.629118], [-0.629118, -0.629118, 0.629118],
         [0.629118, -0.629118, -0.629118], [-0.629118, 0.629118, -0.629118]],
        "CH4",
    ),
    "C2H6": (
        [6, 6, 1, 1, 1, 1, 1, 1],
        [[0.0, 0.0, 0.762209], [0.0, 0.0, -0.762209], [0.0, 1.018957, 1.157229],
         [-0.882443, -0.509479, 1.157229], [0.882443, -0.509479, 1.157229],
         [0.0, -1.018957, -1.157229], [-0.882443, 0.509479, -1.157229],
         [0.882443, 0.509479, -1.157229]],
        "C2H6",
    ),
    "NH3": (
        [7, 1, 1, 1],
        [[0.0, 0.0, 0.116489], [0.0, 0.939731, -0.271808], [0.813831, -0.469865, -0.271808],
         [-0.813831, -0.469865, -0.271808]],
        "H3N",
    ),
    "H2O": (
        [8, 1, 1],
        [[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]],
        "H2O",
    ),
    "CH3OH": (
        [6, 8, 1, 1, 1, 1],
        [[-0.047131, 0.664389, 0.0], [-0.047131, -0.758551, 0.0], [-1.092995, 0.969785, 0.0],
         [0.878534, -1.048458, 0.0], [0.437145, 1.080376, 0.891772],
         [0.437145, 1.080376, -0.891772]],
        "CH4O",
    ),
    "CH3CN": (
        [6, 6, 7, 1, 1, 1],
        [[0.0, 0.0, -1.18693], [0.0, 0.0, 0.273874], [0.0, 0.0, 1.452206],
         [0.0, 1.024986, -1.56237], [0.887664, -0.512493, -1.56237],
         [-0.887664, -0.512493, -1.56237]],
        "C2H3N",
    ),
}


def _resolve(name: str):
    """Return (atomic_numbers, positions[angstrom], Hill formula) for a g2 solute.

    Uses ASE's g2 geometry when ASE is importable, otherwise the bundled fallback above.
    """
    if HAVE_ASE:
        a = molecule(name)
        return a.numbers.tolist(), a.get_positions().tolist(), a.get_chemical_formula()
    if name not in _FALLBACK_GEOM:
        raise KeyError(
            f"no bundled fallback geometry for {name!r}; install ASE (pip install ase) "
            f"or add it to _FALLBACK_GEOM"
        )
    return _FALLBACK_GEOM[name]


def main() -> int:
    z0, r0, _ = _resolve("H2O")
    dE0, _ = predict_solvation_energy((z0, r0), solvent=None)
    assert dE0 == 0.0, f"vacuum baseline not zero: {dE0} eV"
    print(f"[check] vacuum baseline dE(H2O, solvent=None) = {dE0:.1f} eV  (expected exactly 0)")
    print(f"[info]  geometry source: {'ASE g2' if HAVE_ASE else 'bundled fallback (ASE not found)'}\n")

    print(f"{'solute':14s} {'formula':10s} {'n':>3} "
          f"{'dG_pred':>9} {'dG_exp':>9} {'error':>9}   (kcal/mol)")
    errs = []
    for g2_name, label, dG_exp in SOLUTES:
        numbers, positions, formula = _resolve(g2_name)
        dE_eV, _dF = predict_solvation_energy((numbers, positions), charge=0, spin=1, solvent="water")
        dG_pred = dE_eV * EV_TO_KCAL
        err = dG_pred - dG_exp
        errs.append(err)
        print(f"{label:14s} {formula:10s} {len(numbers):3d} "
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
