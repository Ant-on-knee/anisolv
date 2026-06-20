"""Assemble the backbone + head from a converted checkpoint and load weights.

Torch-only: no fairchem on the import path. The module tree (self.backbone /
self.output_heads['efs']) mirrors fairchem's HydraModel naming so the converted EMA
state dict (keys backbone.* / output_heads.efs.*) loads with strict=True.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ._backbone.escn_md import MLP_EFS_Head
from ._backbone.escn_moe import eSCNMDMoeBackbone

_CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

# Optimizaed execution modes will be added later
_SUPPORTED_EXECUTION_MODES = {"general"}

# Inference overrides applied on top of the checkpoint's backbone_config. These force the
# torch-only, deterministic, molecular configuration the standalone package targets.
_INFERENCE_OVERRIDES = dict(
    otf_graph=False,            # edges precomputed by anisolv.data
    use_pbc=False,
    use_pbc_single=False,
    always_use_pbc=False,
    use_quaternion_wigner=False,  # Euler/Jd path
    activation_checkpointing=False,
    regress_forces=True,
    direct_forces=False,        # conservative forces via autograd
    regress_stress=False,
    regress_hessian=False,
)


class AniSolvModel(nn.Module):
    """backbone + EFS head; forward(data) -> head output dict (raw, un-normalized)."""

    def __init__(self, backbone: nn.Module, head: nn.Module, norm: dict):
        super().__init__()
        self.backbone = backbone
        self.output_heads = nn.ModuleDict({"efs": head})
        self.norm = norm  # {energy:{mean,rmsd}, forces:{mean,rmsd}}

    def forward(self, data) -> dict:
        emb = self.backbone(data)
        return self.output_heads["efs"](data, emb)


def _resolve(checkpoint: str | Path) -> Path:
    p = Path(checkpoint)
    if p.exists():
        return p
    cand = _CKPT_DIR / f"{checkpoint}.pt"
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"checkpoint {checkpoint!r} not found (looked for {p} and {cand}). "
        f"Run convert_checkpoint.py first."
    )


def load_model(checkpoint: str | Path = "model1", device: str = "cpu",
               dtype: torch.dtype = torch.float32,
               execution_mode: str = "general") -> AniSolvModel:
    """Build and load the standalone delta model from a converted checkpoint.

    `checkpoint` is 'model1' (default, in anisolv/checkpoints) or a path to a converted .pt.
    Returns an AniSolvModel in eval mode on `device` with params cast to `dtype` (use
    torch.float64 for high-accuracy checks).

    `execution_mode` selects the backbone backend. Only 'general' (pure-torch) is wired up
    today; the fast backends still need prepare_for_inference, which this loader does not
    call yet, so anything else raises NotImplementedError.
    """
    if execution_mode not in _SUPPORTED_EXECUTION_MODES:
        raise NotImplementedError(
            f"execution_mode={execution_mode!r} is not wired into the standalone loader yet "
            f"(the fast backends need prepare_for_inference). "
            f"Supported: {sorted(_SUPPORTED_EXECUTION_MODES)}."
        )
    path = _resolve(checkpoint)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if ckpt.get("format") != "anisolv-ckpt-v1":
        raise ValueError(f"{path} is not an anisolv-ckpt-v1 checkpoint")

    cfg = dict(ckpt["backbone_config"])
    cfg.pop("model", None)  # class-path artifact, not a constructor kwarg
    cfg.update(_INFERENCE_OVERRIDES)

    backbone = eSCNMDMoeBackbone(**cfg)
    head = MLP_EFS_Head(backbone)  # nulls backbone.energy_block/force_block internally
    model = AniSolvModel(backbone, head, ckpt["norm"])

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # Only non-persistent buffers (grid mats) may be "missing"; nothing should be unexpected.
    missing = [k for k in missing if "to_grid_mat" not in k and "from_grid_mat" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch loading {path}:\n  missing={missing}\n  unexpected={unexpected}"
        )

    model.eval().to(device=device, dtype=dtype)
    return model