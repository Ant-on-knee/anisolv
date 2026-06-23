<h1 align="center">AniSolv</h1>
<p align="center"><em>A solvation model for machine-learning interatomic potentials.</em></p>

AniSolv predicts an **additive solvation correction** - an energy term $\Delta E$ and force
term $\Delta F$ - that you add on top of *any* gas-phase potential to get solvated energies and
forces. Given a molecular geometry and a solvent, it computes the difference between the solvated and gas-phase electronic energy.

## What it does

AniSolv is a solvation correction for molecular systems. It is an *additive correction*: it learns only the solvation
contribution, which you layer onto the gas-phase potential of your choice (e.g. a universal MLIP):

$$
\begin{aligned}
E_{\text{solvated}} &= E_{\text{gas}}\,(\text{your potential}) + \Delta E_{\text{anisolv}} \\
F_{\text{solvated}} &= F_{\text{gas}}\,(\text{your potential}) + \Delta F_{\text{anisolv}}
\end{aligned}
$$

where $\Delta E = E_{\text{solv}} - E_{\text{gas}}$. Key properties:

- **Pair it with a base potential.** The delta alone has no bound minimum - always combine it with a gas-phase potential for geometry optimization or vibrational calculation.
- **Lightweight at inference.** Runs on **PyTorch + NumPy only** (ASE is optional, for structure I/O).

### Model(s)

Two checkpoints are available, both with a per-system **solvent embedding** and an **output gate**
(vacuum → exactly 0). `load_model` / `predict_solvation_energy` auto-select the backbone from the
checkpoint, so you switch models purely via `checkpoint=`:

- **`model1`** (default when present) — a 64-expert MoLE `eSCNMDMoeBackbone` (~291 M params), the
  full-accuracy model. **Warmstarted from Meta's UMA `uma-s-1p2`** and fine-tuned on a solvation
  free-energy correction objective. Its weights are gated and **not** in this repo (see below).
- **`model1_compact`** — a non-MoE `eSCNMDBackbone` (~6.4 M params, ~25 MB) for fast / low-memory
  inference, same solvent-conditioning and gate. Trained from scratch (not UMA-derived), so it is
  **MIT-licensed and bundled in this repo** at `models/model1_compact.pt` — no download needed.

Leaving `checkpoint` unset (`None`) auto-selects `model1` when its weights are present and otherwise
falls back to the bundled `model1_compact`, so the package runs out of the box. Because `model1`
derives from UMA, its **weights are released under the FAIR Chemistry License** (see [License](#license)).

## Installation

```bash
git clone https://github.com/Ant-on-knee/anisolv.git
cd anisolv
pip install -e .              # editable install; or `pip install .`
```

You can also install straight from GitHub without cloning:

```bash
pip install "git+https://github.com/Ant-on-knee/anisolv.git"
```

Optional extras:

```bash
pip install -e ".[samples]"   # ASE - needed by the sample scripts
pip install -e ".[hub]"       # huggingface_hub - needed to download the weights (below)
```

> **PyTorch note:** `pip` will pull a default `torch` build. For a specific CUDA/CPU build, install torch from [pytorch.org](https://pytorch.org/get-started/locally/) first, then install AniSolv.

## Download the model weights (gated)

The trained checkpoint **`model1.pt` (~1.1 GB) is not in this repository** - it is git-ignored and
distributed separately on Hugging Face under the **FAIR Chemistry License**.

**1. Request access.** Go to **https://huggingface.co/antonknee/anisolv** and accept the FAIR
Chemistry License. You must provide your full legal name, date of birth, and organization.

**2. Authenticate.**

```bash
pip install huggingface_hub
hf auth login                  # paste a token from https://huggingface.co/settings/tokens
```

**3. Download the checkpoint.**

- **If you cloned the repo and installed with `pip install -e .`**, drop it where the default
  auto-selection looks (`models/model1.pt` under the package root). Run this from the repo root:

  ```bash
  hf download antonknee/anisolv model1.pt --local-dir models
  ```

- **If you installed with a plain `pip install`** (from PyPI or `pip install git+…`), 
  download it to any directory you control and pass its **absolute path** at call time. Until you do,
  the bundled `model1_compact` stays the default, so the package still works:

  ```bash
  hf download antonknee/anisolv model1.pt --local-dir /path/to/anisolv-weights
  ```

  ```python
  from anisolv import predict_solvation_energy
  predict_solvation_energy(..., checkpoint="/path/to/anisolv-weights/model1.pt")  # absolute path
  ```

> **License note:** these weights are a derivative of Meta's UMA (`uma-s-1p2`) and are governed by
> the **FAIR Chemistry License - not MIT**. The MIT license in this repo covers the *inference code
> only* and does not extend to the weights.

## Quickstart

```python
from anisolv import predict_solvation_energy

# Water geometry: atomic numbers Z and positions R (angstrom)
Z = [8, 1, 1]
R = [[0.0, 0.0,  0.119],
     [0.0, 0.763, -0.477],
     [0.0, -0.763, -0.477]]

# Solvent defaults to water; dE is in eV, dF in eV/angstrom (shape [n_atoms, 3]).
dE, dF = predict_solvation_energy((Z, R), charge=0, spin=1)
print(f"dE = {dE:.4f} eV")

# Vacuum baseline is exactly zero:
dE0, _ = predict_solvation_energy((Z, R), solvent=None)
assert dE0 == 0.0
```

## API

```python
predict_solvation_energy(
    atoms_or_arrays,            # ase.Atoms, or a (atomic_numbers, positions[angstrom]) tuple
    charge: int = 0,           # total charge
    spin: int = 1,             # spin multiplicity
    solvent="water",           # solvent name (str), or None for the vacuum baseline (-> exactly 0)
    checkpoint: str = "model1",# "model1" (default) / "model1_compact", or a path to a .pt
    device: str = "cpu",       # "cpu", "cuda", or "mps"
    dtype=torch.float32,       # torch.float32 (default) or torch.float64
    inference_settings="default",  # "default" (reference) or "fast" (see below)
) -> tuple[float, np.ndarray]  # (dE in eV, dF in eV/angstrom with shape [n_atoms, 3])
```

To convert $\Delta E$ to kcal/mol, multiply by `23.060548`.

### Faster inference

`inference_settings` selects the compute path (also accepted by `load_model`):

- **`"default"`** — the pure-torch reference path. Bit-for-bit identical to earlier releases.
- **`"fast"`** — the block-diagonal SO2 GEMM backend plus TF32 matmuls and `torch.compile`.
  - On the compact model it is **composition-independent**, so it speeds up *any* molecule (the
    biggest win is on GPU).
  - On the MoE `model1` the block-GEMM conversion would require a fixed-composition MOLE merge, so
    `"fast"` **auto-downgrades to the general backend** there, and `torch.compile` is disabled too
    (the MOLE expert-routing side-channel is not `torch.compile`-safe across graph breaks). You
    still get TF32, which is the main GPU win.

```python
dE, dF = predict_solvation_energy((Z, R), checkpoint="model1_compact",
                                  device="cuda", inference_settings="fast")
```

For full control, pass an `InferenceSettings` instead of a preset name:

```python
from anisolv import InferenceSettings, predict_solvation_energy
settings = InferenceSettings(execution_mode="umas_fast_pytorch", tf32=True, compile=False)
dE, dF = predict_solvation_energy((Z, R), checkpoint="model1_compact", inference_settings=settings)
```

> **`torch.compile` caveat:** the first call is slow (graph capture) and a new molecule *shape*
> (atom/edge count) can trigger a recompile; if compilation fails the model falls back to eager
> automatically. TF32 and `torch.compile` mainly help on GPU (TF32 is a no-op on CPU). The
> GPU-only Triton backend (`umas_fast_gpu`) is not enabled yet.

## Sample scripts

The sample scripts live in this repository (clone it to run them). From the repo root:

```bash
python anisolv/samples/H2O_single_point.py   # hydration dG for small molecules vs. experiment (needs ASE)
python anisolv/samples/H2O_dGsolv.py         # full thermodynamic cycle: geometry relax + vibrational dG (needs fairchem)
```

## Supported solvents

The model is trained/validated on 21 solvents:

- **Water** - reaches **SMD-level** accuracy.
- **acetone, acetonitrile, aniline, benzaldehyde, benzene, ch2cl2 (dichloromethane), chcl3 (chloroform), cs2 (carbon disulfide), dioxane, dmf, dmso, ether (diethyl ether), ethylacetate, hexadecane, hexane, methanol, nitromethane, octanol, thf, toluene** - **XTB-ALPB** accuracy.

Solvents are conditioned through a descriptor embedding; the additional entries in
`_const/solvent_descriptors.json` are not validated targets.

## License

- **Inference code (this repository): MIT** - see [`LICENSE`](LICENSE). The bundled
  **`model1_compact.pt`** weights are trained from scratch (not UMA-derived), so they fall under this
  MIT license too.
- **Full-accuracy weights (`model1.pt`, on Hugging Face): FAIR Chemistry License v1.** A derivative of
  Meta's UMA (`uma-s-1p2`); redistribution is permitted only under the same license. Use is subject to
  the FAIR Chemistry Acceptable Use Policy and applicable Trade Control Laws.

## Citation

If you use AniSolv, please cite both the UMA work it derives from and this repository (see also
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{wood2025uma,
  title   = {{UMA}: A Family of Universal Models for Atoms},
  author  = {Wood, Brandon M. and Dzamba, Misko and Fu, Xiang and Gao, Meng and Shuaibi, Muhammed and Barroso-Luque, Luis and Abdelmaqsoud, Kareem and Gharakhanyan, Vahe and Kitchin, John R. and Levine, Daniel S. and Michel, Kyle and Sriram, Anuroop and Cohen, Taco and Das, Abhishek and Rizvi, Ammar and Sahoo, Sushree Jagriti and Ulissi, Zachary W. and Zitnick, C. Lawrence},
  year    = {2025},
  journal = {arXiv preprint arXiv:2506.23971},
  doi     = {10.48550/arXiv.2506.23971},
  url     = {https://arxiv.org/abs/2506.23971}
}

@misc{anisolv2026,
  author       = {Ni, Anton},
  title        = {{AniSolv}: Implicit Solvation Model for {MLIPs}},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Ant-on-knee/anisolv}},
  note         = {GitHub repository}
}
```

## Acknowledgements

Built by warmstarting from Meta FAIR Chemistry's UMA-S 1.2 (`uma-s-1p2`). UMA code is MIT-licensed
([facebookresearch/fairchem](https://github.com/facebookresearch/fairchem)); UMA weights are under
the FAIR Chemistry License.
