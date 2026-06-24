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

# Wired-up backbone backends. "general" is the pure-torch reference path; 
# "umas_fast_pytorch" is the block-diagonal SO2 GEMM path (composition-independent on non-MoE checkpoints);
# "umas_fast_gpu" adds the vTriton Wigner-permute kernels 
# On a MoE checkpoint the GPU/block-GEMM paths require a MOLE merge first (merge_mole=True, fixed composition)
_SUPPORTED_EXECUTION_MODES = {"general", "umas_fast_pytorch", "umas_fast_gpu"}


@contextmanager
def tf32_context_manager():
    """Enable TF32 matmuls for the duration of the block, restoring prior state on exit.

    TF32 trades a little float32 mantissa precision for speed on NVIDIA GPUs; it is a no-op on CPU.
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
               inference_settings: str | InferenceSettings = "default") -> AniSolvModel:
    """Build and load the standalone delta model from a converted checkpoint.

    `checkpoint` is None (auto: 'model1' if its weights are present in anisolv/models, else the
    bundled 'model1_compact'), a checkpoint name, or a path to a converted .pt. Returns an
    AniSolvModel in eval mode on `device` with params cast to `dtype` (use torch.float64 for
    high-accuracy checks).

    `inference_settings` selects the inference path: a preset name -- 
    'default' (reference implementation), 
    'fast' (block-GEMM SO2 + tf32 + torch.compile, no MoLE merging),
    'fast_gpu' (adds the Triton Wigner kernels; CUDA-only)
    or a custom InferenceSettings (whose `execution_mode` field picks the backend).

    'fast'/'umas_fast_pytorch' is composition-independent on the non-MoE 'model1_compact'.
    On the MoE 'model1' it would need a MOLE merge first, so it is downgraded to 'general' + tf32

    'fast_gpu'/'umas_fast_gpu' (requires CUDA, lmax==mmax==2, triton) auto-manages the merge by backbone: 
    compact models' performance remains identical to fast; 
    'model1' is MOLE-merged so block-GEMM + Triton + torch.compile all apply, at the cost of locking to ONE
    composition/charge/spin/solvent -- single molecule per loaded model (re-load to change it).
    """
    if checkpoint is None:
        checkpoint = default_checkpoint()

    settings = guess_inference_settings(inference_settings)

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

    # umas_fast_gpu (Triton) needs CUDA and plain-Linear SO2 layers. Auto-manage merge_mole by
    # backbone: a MoE checkpoint must be merged first (fixed composition, single molecule); the
    # compact model is already plain-Linear, so it runs merge-free and stays composition-independent
    # (still valid for multi-molecule eval). This is what lets the 'fast_gpu' preset target both.
    if settings.execution_mode == "umas_fast_gpu":
        if "cuda" not in str(device):
            raise ValueError(
                f"execution_mode='umas_fast_gpu' requires a CUDA device (got device={device!r})."
            )
        want_merge = backbone_cls is eSCNMDMoeBackbone
        if settings.merge_mole != want_merge:
            settings = replace(settings, merge_mole=want_merge)

    # Backbone-aware safety for MoE checkpoints. The fast backends (block-GEMM / Triton) and
    # torch.compile are only safe once the MOLE experts are merged into plain Linear layers; tf32 is
    # always safe. With merge_mole=True (e.g. the 'fast_gpu' preset, or set explicitly) the merged
    # backbone is a plain eSCNMDBackbone, so we leave the fast backend and compile ON.
    if backbone_cls is eSCNMDMoeBackbone and not settings.merge_mole:
        # No merge -> SO2 layers are still MOLE: block-GEMM/Triton can't convert them, and the live
        # MOLE routing side-channel (mole_sizes / expert coefficients written onto a plain
        # MOLEGlobals object mid-forward, read back by each MOLE sublayer) is not preserved by dynamo
        # across the forward's graph breaks -> empty MOLE outputs (a shape error the prepare()
        # try/except can't catch). Fall back to general + tf32.
        if settings.execution_mode in ("umas_fast_pytorch", "umas_fast_gpu"):
            logging.warning(
                "%s on a MoE checkpoint (%s) needs a MOLE merge (merge_mole=True / the 'fast_gpu' "
                "preset); falling back to the general backend (tf32 still applies).",
                settings.execution_mode, cls_name,
            )
            settings = replace(settings, execution_mode="general")
        if settings.compile:
            logging.warning(
                "torch.compile is not supported on an unmerged MoE checkpoint (%s): the MOLE "
                "routing side-channel is not dynamo-safe across graph breaks. Disabling compile "
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