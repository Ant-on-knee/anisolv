"""Assemble the backbone + head from a converted checkpoint and load weights.

Torch-only: no fairchem on the import path. The module tree (self.backbone /
self.output_heads['efs']) mirrors fairchem's HydraModel naming so the converted EMA
state dict (keys backbone.* / output_heads.efs.*) loads with strict=True.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

from ._backbone._compat.inference import InferenceSettings, guess_inference_settings
from ._backbone.escn_md import MLP_EFS_Head, eSCNMDBackbone
from ._backbone.escn_moe import eSCNMDMoeBackbone

_CKPT_DIR = Path(__file__).resolve().parent / "models"

# Wired-up backbone backends. "general" is the pure-torch reference path; "umas_fast_pytorch" is
# the block-diagonal SO2 GEMM path (composition-independent on non-MoE checkpoints). "umas_fast_gpu"
# (Triton) is intentionally absent: it needs the anisolv._backbone.triton kernels vendored in, the
# `triton` dependency, and merge_mole=True (MoE-only, fixed composition) -- all of which is the
# later GPU phase. See README and the docstring on load_model.
_SUPPORTED_EXECUTION_MODES = {"general", "umas_fast_pytorch"}


@contextmanager
def tf32_context_manager():
    """Enable TF32 matmuls for the duration of the block, restoring prior state on exit.

    TF32 trades a little float32 mantissa precision for speed on NVIDIA GPUs; it is a no-op on
    CPU. Mirrors fairchem's inference tf32 context without importing fairchem.
    """
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_prec = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        torch.set_float32_matmul_precision(old_prec)

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
    """backbone + EFS head; forward(data) -> head output dict (raw, un-normalized).

    Inference backend selection (block-GEMM, tf32, torch.compile) is applied lazily on the first
    forward, because the backbone's prepare_for_inference needs a sample batch (it may merge MOLE
    experts and lock composition). With the "default"/general settings every step below is a no-op
    or identity, so the legacy compute path is reproduced bit-for-bit.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, norm: dict,
                 settings: InferenceSettings | None = None):
        super().__init__()
        self.backbone = backbone
        self.output_heads = nn.ModuleDict({"efs": head})
        self.norm = norm  # {energy:{mean,rmsd}, forces:{mean,rmsd}}
        self._settings = settings or InferenceSettings()
        self._prepared = False
        self._run = None  # eager or torch.compile'd _raw_forward, set on first forward

    def _raw_forward(self, data) -> dict:
        emb = self.backbone(data)
        return self.output_heads["efs"](data, emb)

    def _prepare(self, data) -> None:
        # prepare_for_inference may RETURN A NEW backbone (the MOLE-merge path), so reassign.
        self.backbone = self.backbone.prepare_for_inference(data, self._settings)
        self._run = self._raw_forward
        if self._settings.compile:
            try:
                torch._dynamo.config.recompile_limit = 32
                # Compile the whole forward (not just the backbone) so the conservative-force
                # autograd graph in the head is traced too. dynamic=True avoids a recompile per
                # molecule size; new (natoms, nedges) shapes may still recompile up to the limit.
                self._run = torch.compile(self._raw_forward, dynamic=True)
            except Exception as exc:  # pragma: no cover - environment dependent
                logging.warning("torch.compile failed (%s); running eager", exc)
                self._run = self._raw_forward
        self._prepared = True

    def forward(self, data) -> dict:
        if not self._prepared:
            self._prepare(data)
        else:
            # Guards the MOLE-merge composition lock; a no-op when unmerged.
            self.backbone.on_predict_check(data)
        # Conservative forces (direct_forces=False) need grad enabled, so never wrap in no_grad.
        ctx = tf32_context_manager() if self._settings.tf32 else nullcontext()
        with ctx:
            return self._run(data)


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
               execution_mode: str = "general",
               inference_settings: str | InferenceSettings = "default") -> AniSolvModel:
    """Build and load the standalone delta model from a converted checkpoint.

    `checkpoint` is None (auto: 'model1' if its weights are present in anisolv/models, else the
    bundled 'model1_compact'), a checkpoint name, or a path to a converted .pt. Returns an
    AniSolvModel in eval mode on `device` with params cast to `dtype` (use torch.float64 for
    high-accuracy checks).

    `inference_settings` selects the inference path: a preset name -- 'default' (pure-torch
    reference, identical to the legacy behaviour) or 'fast' (block-GEMM SO2 + tf32 +
    torch.compile) -- or a custom InferenceSettings. `execution_mode` is the legacy knob; setting
    it to anything other than 'general' overrides the preset's execution_mode (back-compat).

    The 'fast'/'umas_fast_pytorch' path is composition-independent on the non-MoE
    'model1_compact'. On the MoE 'model1' the block-GEMM conversion would need a MOLE merge first
    (a fixed-composition path not wired up here), so it is downgraded to 'general'; torch.compile
    is also disabled there (the MOLE routing side-channel is not dynamo-safe), while tf32 still
    applies. 'umas_fast_gpu' is not supported yet (Triton phase).
    """
    if checkpoint is None:
        checkpoint = default_checkpoint()

    settings = guess_inference_settings(inference_settings)
    if execution_mode != "general":  # legacy override of the preset's backend
        settings = replace(settings, execution_mode=execution_mode)

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

    # Backbone-aware safety for MoE checkpoints (tf32 stays on either way -- it is verified correct
    # on the MoE; only the block-GEMM backend and torch.compile are unsafe here).
    if backbone_cls is eSCNMDMoeBackbone:
        # (1) block-GEMM conversion needs plain-Linear SO2 layers, which on a MoE checkpoint only
        #     exist after a MOLE merge (fixed composition); without one, fall back to general.
        if settings.execution_mode == "umas_fast_pytorch" and not settings.merge_mole:
            logging.warning(
                "umas_fast_pytorch on a MoE checkpoint (%s) needs a MOLE merge first; falling back "
                "to the general backend (tf32 still applies).", cls_name,
            )
            settings = replace(settings, execution_mode="general")
        # (2) torch.compile is incompatible with the MOLE expert-routing side-channel: mole_sizes /
        #     expert coefficients are written onto a plain MOLEGlobals object mid-forward and read
        #     back by each MOLE sublayer, but dynamo does not preserve that object-attribute
        #     mutation across the forward's graph breaks, so the MOLE layers see an empty
        #     mole_sizes and emit zero-row outputs (a shape AssertionError deep in the run, which
        #     the prepare() try/except cannot catch). Disable compile on MoE; keep tf32.
        if settings.compile:
            logging.warning(
                "torch.compile is not supported on MoE checkpoint (%s): the MOLE routing "
                "side-channel is not dynamo-safe across graph breaks. Disabling compile "
                "(tf32 still applies).", cls_name,
            )
            settings = replace(settings, compile=False)

    if settings.execution_mode not in _SUPPORTED_EXECUTION_MODES:
        raise NotImplementedError(
            f"execution_mode={settings.execution_mode!r} is not wired into the standalone loader "
            f"(supported: {sorted(_SUPPORTED_EXECUTION_MODES)})."
        )

    cfg.pop("model", None)  # class-path artifact, not a constructor kwarg
    cfg.update(_INFERENCE_OVERRIDES)
    cfg["execution_mode"] = settings.execution_mode  # ckpt backbone_config carries no such key

    backbone = backbone_cls(**cfg)
    head = MLP_EFS_Head(backbone)  # nulls backbone.energy_block/force_block internally
    model = AniSolvModel(backbone, head, ckpt["norm"], settings=settings)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # Only non-persistent buffers (grid mats) may be "missing"; nothing should be unexpected.
    missing = [k for k in missing if "to_grid_mat" not in k and "from_grid_mat" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch loading {path}:\n  missing={missing}\n  unexpected={unexpected}"
        )

    model.eval().to(device=device, dtype=dtype)
    return model