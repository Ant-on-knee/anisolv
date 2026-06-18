"""dependency check
"""

from __future__ import annotations

import sys

import torch

FORBIDDEN = [
    "fairchem", "e3nn", "torch_geometric", "torch_scatter", "torch_cluster",
    "omegaconf", "ase", "hydra",
]


def main():
    # Drop anything already imported, then block.
    for name in list(sys.modules):
        if any(name == f or name.startswith(f + ".") for f in FORBIDDEN):
            del sys.modules[name]
    for f in FORBIDDEN:
        sys.modules[f] = None

    import anisolv  # noqa: E402

    Z = [8, 1, 1]
    R = [[0, 0, 0.119], [0, 0.763, -0.477], [0, -0.763, -0.477]]
    dE, dF = anisolv.predict_solvation_energy((Z, R), charge=0, spin=1, dtype=torch.float64)
    print(f"predict_solvation_energy OK: dE={dE:.6f} eV, dF shape={dF.shape}")

    leaked = [f for f in FORBIDDEN if sys.modules.get(f) is not None]
    if leaked:
        print(f"DEPENDENCY AUDIT FAIL: these modules were imported: {leaked}")
        return 1
    print("DEPENDENCY AUDIT PASS: ran on torch + numpy only "
          f"(blocked: {', '.join(FORBIDDEN)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
