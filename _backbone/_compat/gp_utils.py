"""Graph-parallel stub.

The standalone package runs single-process inference, so graph parallelism is never
active. `initialized()` returns False, which short-circuits every `if gp_utils.initialized()`
branch in the vendored backbone; the remaining functions exist only so the names resolve
and raise if ever reached (they never are for single-process molecular inference).
"""

from __future__ import annotations


def initialized() -> bool:
    return False


def _unreachable(*_args, **_kwargs):  # pragma: no cover
    raise RuntimeError(
        "graph-parallel code path NYI "
        "(gp_utils.initialized() is always False here)."
    )


get_gp_group = _unreachable
get_gp_rank = _unreachable
get_gp_world_size = _unreachable
gather_from_model_parallel_region = _unreachable
gather_from_model_parallel_region_sum_grad = _unreachable
reduce_from_model_parallel_region = _unreachable
