"""Assemble the backbone + head from a converted checkpoint and load weights.

Torch-only: no fairchem on the import path. The module tree (self.backbone /
self.output_heads['efs']) mirrors fairchem's HydraModel naming so the converted EMA
state dict (keys backbone.* / output_heads.efs.*) loads with strict=True.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ._backbone.escn_md import MLP_EFS_Head, eSCNMDBackbone
from ._backbone.escn_moe import eSCNMDMoeBackbone

_CKPT_DIR = Path(__file__).resolve().parent / "models"

# Optimized execution modes will be added later
_SUPPORTED_EXECUTION_MODES = {"general"}

# Default checkpoint selection. 'model1' (full-accuracy, gated UMA-derived weights) ships
# separately and may be absent; 'model1_compact' (trained from scratch, ~25 MB) is bundled in
# the repo, so it is always available as a fallback.
_DEFAULT_CHECKPOINT = "model1"
_FALLBACK_CHECKPOINT = "model1_compact"


def default_checkpoint() -> str:
    """Checkpoint used when the caller names none: 'model1' if its .pt is present in the
    bundled models dir, else the always-present 'model1_compact' fallback."""
    if (_CKPT_DIR / f"{_DEFAULT_CHECKPOINT}.pt").exists():
        return _DEFAULT_CHECKPOINT
    return _FALLBACK_CHECKPOINT


def default_checkpoint_path() -> Path:
    """Filesystem path to the .pt that `load_model(checkpoint=None)` would load."""
    return _resolve(default_checkpoint())


def print_default_checkpoint_path() -> None:
    """Print the path to the default model checkpoint."""
    print(default_checkpoint_path())

_BACKBONES = {
    "eSCNMDBackbone": eSCNMDBackbone,        # non-MoE, solvent-conditioned (model1_compact)
    "eSCNMDMoeBackbone": eSCNMDMoeBackbone,  # UMA-S-1.2 mixture-of-experts (model1, default)
}
_DEFAULT_BACKBONE = "eSCNMDMoeBackbone"

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


def load_model(checkpoint: str | Path | None = None, device: str = "cpu",
               dtype: torch.dtype = torch.float32,
               execution_mode: str = "general") -> AniSolvModel:
    """Build and load the standalone delta model from a converted checkpoint.

    `checkpoint` is None (auto: 'model1' if its weights are present in anisolv/models, else the
    bundled 'model1_compact'), a checkpoint name, or a path to a converted .pt. Returns an
    AniSolvModel in eval mode on `device` with params cast to `dtype` (use torch.float64 for
    high-accuracy checks).

    `execution_mode` selects the backbone backend. Only 'general' (pure-torch) is wired up
    today; the fast backends still need prepare_for_inference, which this loader does not
    call yet, so anything else raises NotImplementedError.
    """
    if checkpoint is None:
        checkpoint = default_checkpoint()
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
    cls_name = str(cfg.get("model", "")).rsplit(".", 1)[-1] or _DEFAULT_BACKBONE
    try:
        backbone_cls = _BACKBONES[cls_name]
    except KeyError:
        raise ValueError(
            f"{path}: unsupported backbone class {cls_name!r} (known: {sorted(_BACKBONES)})"
        ) from None
    cfg.pop("model", None)  # class-path artifact, not a constructor kwarg
    cfg.update(_INFERENCE_OVERRIDES)

    backbone = backbone_cls(**cfg)
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