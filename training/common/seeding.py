"""Determinism controls.

Seeding is treated as a *recorded* action, not a side effect: every function
here returns a dict describing exactly what it set, and that dict is written
into ``experiment.json``. The reason is that "seed=1234" alone does not
reproduce a run -- whether cuDNN was in deterministic mode, whether the
PYTHONHASHSEED was fixed, and which library versions were present all change
the answer, and none of them is visible from the seed.

Two levels are offered because they cost very different amounts:

* :func:`seed_everything` with ``deterministic=False`` seeds the RNGs and
  nothing else. Cheap, and enough to make two runs on the same machine agree.
* ``deterministic=True`` additionally pins cuDNN, disables its autotuner and
  asks torch for deterministic algorithms. On CUDA this can cost a noticeable
  fraction of throughput; on CPU it costs essentially nothing, which is why it
  is the default here.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

__all__ = ["make_generator", "seed_everything", "seed_worker"]


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy and torch, and report what was set.

    Parameters
    ----------
    seed
        Base seed. Must be non-negative and below ``2**32`` so that it is
        legal for NumPy's legacy seeding as well as for torch.
    deterministic
        Pin cuDNN and request deterministic torch kernels. Also sets
        ``CUBLAS_WORKSPACE_CONFIG``, which cuBLAS requires before
        ``use_deterministic_algorithms`` will accept a matmul on CUDA -- torch
        raises a confusing runtime error hundreds of steps into training if it
        is missing, so it is set up front.

    Returns
    -------
    dict
        Everything that was changed, for the experiment record. Keys that could
        not be set (torch absent, no CUDA) are reported with their reason
        rather than omitted, so a record never leaves a reader guessing whether
        a knob was set to false or simply unavailable.
    """
    if not 0 <= int(seed) < 2**32:
        raise ValueError(f"seed must lie in [0, 2**32), got {seed}")
    seed = int(seed)

    record: dict[str, Any] = {
        "seed": seed,
        "deterministic_requested": bool(deterministic),
        "python_random": True,
        "numpy_legacy": True,
    }

    # PYTHONHASHSEED only takes effect at interpreter start, so setting it here
    # cannot fix the current process. It is recorded (not silently set) because
    # set/iteration order of str-keyed containers is the one source of
    # nondeterminism this function genuinely cannot remove after the fact.
    record["pythonhashseed_env"] = os.environ.get("PYTHONHASHSEED", "<unset>")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        record["torch"] = "not installed"
        return record

    torch.manual_seed(seed)
    record["torch"] = str(torch.__version__)
    record["torch_manual_seed"] = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        record["torch_cuda_manual_seed_all"] = True
        record["cuda_device_count"] = int(torch.cuda.device_count())
    else:
        record["torch_cuda_manual_seed_all"] = "no cuda device"

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        record["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        record["cudnn_deterministic"] = True
        record["cudnn_benchmark"] = False
        try:
            # warn_only: a handful of ops (e.g. some pooling backward kernels)
            # have no deterministic implementation. Hard-failing there would
            # make determinism an all-or-nothing switch that nobody could turn
            # on; warning keeps the rest of the run deterministic and leaves a
            # trace of which op was not.
            torch.use_deterministic_algorithms(True, warn_only=True)
            record["torch_deterministic_algorithms"] = "true (warn_only)"
        except Exception as exc:  # pragma: no cover - torch version dependent
            record["torch_deterministic_algorithms"] = f"unavailable: {exc}"
    else:
        torch.backends.cudnn.benchmark = True
        record["cudnn_deterministic"] = False
        record["cudnn_benchmark"] = True

    return record


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for ``torch.utils.data.DataLoader``.

    Each worker process inherits a fresh torch seed from the parent's generator
    but *not* a fresh NumPy or Python seed, so without this every worker draws
    the identical augmentation stream -- a classic silent bug that reduces the
    effective augmentation diversity by the worker count. Deriving both from
    torch's per-worker seed keeps them distinct and still reproducible.
    """
    import torch

    worker_seed = int(torch.initial_seed()) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    del worker_id  # part of the DataLoader contract; the seed already encodes it


def make_generator(seed: int) -> Any:
    """A ``torch.Generator`` for DataLoader shuffling.

    Passing an explicit generator is what makes the *order* of a shuffled epoch
    reproducible; ``torch.manual_seed`` alone does not, because any other torch
    RNG consumption between epochs shifts the global stream.
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
