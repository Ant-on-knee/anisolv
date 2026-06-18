"""Sample: ΔG_solv of H2O in water via the harmonic thermodynamic cycle (vibrational).

The full-cycle companion to H2O_single_point.py. Where the single-point sample reports the
bare electronic correction ΔE = E_solv − E_gas, this one adds the vibrational/thermal leg,
reproducing — for a single H2O — the cluster MNSol driver's gas_leg / solv_leg / assemble
(unisolv_training/solvation_fe):

    1. relax H2O in the gas phase with a base potential          -> E_gas, then harmonic G_gas
    2. relax H2O in water with (base + anisolv water delta)      -> E_solv, then harmonic G_solv
    3. ΔG_solv = G_solv − G_gas                                  (reported in kcal/mol)

WHY A BASE POTENTIAL IS NEEDED
------------------------------
anisolv only adds a solvation correction: ``predict_solvation_energy`` returns only dE = E_solv − E_gas,
a correction — not a full potential energy surface. You cannot run a vibrational analysis on
the delta alone (it has no bound minimum). So the solvated surface is E_base(R) + dE_anisolv(R)
and the gas surface is E_base(R); only the delta part is torch-only. This sample defaults to
the UMA-small base (fairchem) whose underlying architecture anisolv was trained on,
but the solvation model is compatible with any MLIP

Unlike H2O_single_point.py, this sample therefore needs ASE + a base potential (fairchem UMA
by default), not torch alone.

    python anisolv/samples/H2O_dGsolv.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make `anisolv` importable when run straight from a checkout (repo root = parents[2]).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from ase import Atoms, units  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.calculators.mixing import SumCalculator  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from ase.thermochemistry import HarmonicThermo  # noqa: E402
from ase.vibrations import Vibrations  # noqa: E402

from anisolv import predict_solvation_energy  # noqa: E402

EV_TO_KCAL = 1.0 / (units.kcal / units.mol)  # ~23.0605, matches solvation_fe
_ZERO_MODE_eV = 1e-4  # modes below this |energy| are trans/rot remnants / numerical noise


class AniSolvDeltaCalculator(Calculator):
    """``predict_solvation_energy`` as an ASE calculator (the additive solvation correction).

    Energy/forces are the ΔE/ΔF the model predicts; reads ``charge``/``spin``/``solvent`` from
    ``atoms.info`` (defaults 0 / 1 / ``"water"``). ``check_state`` is widened so that changing
    only the solvent in ``atoms.info`` (same geometry) still triggers a recompute — mirrors
    unisolv_training/solvation_fe/unisolv_calc.py.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, checkpoint="model1", device="cpu", dtype=None,
                 default_solvent="water", **kwargs):
        super().__init__(**kwargs)
        import torch
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype or torch.float32
        self.default_solvent = default_solvent
        self._last_sig = None

    def _signature(self, atoms):
        info = atoms.info
        return (int(info.get("charge", 0)), int(info.get("spin", 1)),
                info["solvent"] if "solvent" in info else self.default_solvent)

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        if self._signature(atoms) != self._last_sig:
            changes = changes + ["info"]
        return changes

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        charge, spin, solvent = self._signature(self.atoms)
        self._last_sig = (charge, spin, solvent)
        dE, dF = predict_solvation_energy(
            (self.atoms.get_atomic_numbers(), self.atoms.get_positions()),
            charge=charge, spin=spin, solvent=solvent,
            checkpoint=self.checkpoint, device=self.device, dtype=self.dtype,
        )
        self.results["energy"] = float(dE)
        self.results["forces"] = np.asarray(dF, dtype=float)


def make_uma_base(device="cpu"):
    """Default base gas-phase potential: UMA-small (omol) via fairchem — what anisolv corrects."""
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:  # keep the failure actionable
        raise SystemExit(
            "H2O_dGsolv.py needs a base gas-phase potential for the vibrational cycle, and the "
            "default is UMA-small via fairchem — which isn't importable here.\n"
            f"  ({exc})\n"
            "Either run this in the fairchem env (e.g. `conda run -n unisolv`), or call "
            "main(base=<your ASE calculator>) to supply any other gas-phase potential."
        ) from exc
    pred = pretrained_mlip.get_predict_unit("uma-s-1p2", device=device)
    return FAIRChemCalculator(pred, task_name="omol")


def relax(atoms, calc, fmax=0.02, steps=300):
    """BFGS geometry optimisation; returns (E_eV, converged)."""
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return float(atoms.get_potential_energy()), bool(opt.converged())


def harmonic_gibbs(atoms, calc, temperature=298.15, delta=0.01):
    """Harmonic free energy (eV) at ``atoms`` — adapted from solvation_fe/thermo.vib_gibbs.

    Treats the Helmholtz F = E_elec + ZPE + U_vib − T·S_vib as G: the pV term is negligible
    and translation/rotation cancel between the gas and solvated geometries of one molecule.
    Returns (G_eV, E_elec_eV, zpe_eV, n_imag).
    """
    atoms = atoms.copy()
    atoms.calc = calc
    e_elec = float(atoms.get_potential_energy())

    with tempfile.TemporaryDirectory() as tmp:
        vib = Vibrations(atoms, name=str(Path(tmp) / "vib"), delta=delta)
        vib.run()
        energies = np.asarray(vib.get_energies())  # complex eV, length 3N

    # Drop the 6 lowest-|energy| modes (trans+rot for a nonlinear molecule like H2O).
    n_drop = 6 if len(atoms) > 1 else 3
    vib_modes = energies[np.argsort(np.abs(energies))[n_drop:]]
    n_imag = int(np.sum(np.abs(vib_modes.imag) > _ZERO_MODE_eV))
    real_pos = np.array([e.real for e in vib_modes
                         if abs(e.imag) <= _ZERO_MODE_eV and e.real > _ZERO_MODE_eV])
    if n_imag:
        print(f"[thermo] WARNING: {n_imag} imaginary mode(s) — geometry not a true minimum; "
              "excluded from the free energy.")

    zpe = float(np.sum(real_pos) / 2.0)
    g = e_elec if real_pos.size == 0 else float(
        HarmonicThermo(vib_energies=real_pos, potentialenergy=e_elec)
        .get_helmholtz_energy(temperature, verbose=False))
    return g, e_elec, zpe, n_imag


def water() -> Atoms:
    """A neutral, closed-shell H2O start geometry (charge 0, spin 1), solvent tagged 'water'."""
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.119],
                                    [0.0, 0.763, -0.477],
                                    [0.0, -0.763, -0.477]])
    atoms.info.update(charge=0, spin=1, solvent="water")
    return atoms


def main(base=None, device="cpu", temperature=298.15) -> int:
    base = base if base is not None else make_uma_base(device=device)
    delta = AniSolvDeltaCalculator(device=device)
    solv_calc = SumCalculator([base, delta])  # E_solv = E_base + ΔE_anisolv

    # --- gas leg: relax + harmonic G on the base surface (solvent-independent) ---
    gas = water()
    e_gas, conv_gas = relax(gas, base, steps=300)
    g_gas, _, zpe_gas, n_imag_gas = harmonic_gibbs(gas, base, temperature)

    # --- solv leg: relax + harmonic G on (base + delta), starting from the gas minimum ---
    solv = gas.copy()
    solv.info.update(gas.info)  # carry charge/spin/solvent
    e_solv, conv_solv = relax(solv, solv_calc, steps=300)
    g_solv, _, zpe_solv, n_imag_solv = harmonic_gibbs(solv, solv_calc, temperature)

    # --- assemble ---
    dE_elec = (e_solv - e_gas) * EV_TO_KCAL          # electronic (single-point-like) leg
    dG_solv = (g_solv - g_gas) * EV_TO_KCAL          # full cycle, incl. vibrational ΔG

    print(f"\nH2O solvation in water  (T = {temperature:.2f} K)")
    print(f"  base potential        : {type(base).__name__}")
    print(f"  gas   : E = {e_gas:12.6f} eV   ZPE = {zpe_gas:.4f} eV   "
          f"G = {g_gas:12.6f} eV   ({'min' if not n_imag_gas else f'{n_imag_gas} imag'}, "
          f"{'conv' if conv_gas else 'UNCONVERGED'})")
    print(f"  solv  : E = {e_solv:12.6f} eV   ZPE = {zpe_solv:.4f} eV   "
          f"G = {g_solv:12.6f} eV   ({'min' if not n_imag_solv else f'{n_imag_solv} imag'}, "
          f"{'conv' if conv_solv else 'UNCONVERGED'})")
    print(f"\n  ΔE_elec  (E_solv − E_gas)        = {dE_elec:+8.2f} kcal/mol")
    print(f"  ΔG_vib   (vibrational/thermal)   = {dG_solv - dE_elec:+8.2f} kcal/mol")
    print(f"  ΔG_solv  (G_solv − G_gas)        = {dG_solv:+8.2f} kcal/mol   "
          f"(exp. ≈ -6.3 kcal/mol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
