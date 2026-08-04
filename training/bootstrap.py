"""Make ``training.*`` importable when a script is run by path.

``python training/train_morphology.py`` puts ``training/`` on ``sys.path``, not
the repository root, so ``import training.common.args`` fails while
``python -m training.train_morphology`` works. Rather than forcing one
invocation style on everybody, every entry point opens with the same
three-line preamble::

    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

    from training.bootstrap import ensure_importable
    ensure_importable()

The preamble cannot itself live in this module -- importing it is the thing
that needs the path -- but everything after the first three lines does, which
is why this module exists rather than the ``src`` handling being repeated six
times as well.

The alternative, relative imports, is worse: it makes the by-path invocation
impossible instead of merely awkward, and by-path is what people actually type.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["REPO_ROOT", "ensure_importable"]

#: Repository root, i.e. the directory containing ``training/`` and ``src/``.
REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_importable() -> Path:
    """Put the repository root on ``sys.path`` and return it.

    Idempotent, and deliberately prepends rather than appends: if a stale
    ``sperm_sorting`` is installed site-wide, the checkout under test must win,
    otherwise a training run silently measures a different version of the
    library than the one in the working tree.
    """
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # ``src`` layout: the package is normally installed editable, but a bare
    # checkout must still work so that a reviewer can run a script without
    # having run ``pip install -e .`` first.
    src = str(REPO_ROOT / "src")
    if (REPO_ROOT / "src" / "sperm_sorting").is_dir() and src not in sys.path:
        sys.path.insert(1, src)
    return REPO_ROOT
