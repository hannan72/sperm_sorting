"""Web demo for the sperm-analysis research prototype.

A package rather than a loose directory so that ``uvicorn web.app:app`` resolves
the same way from any working directory that has the repository root on
``sys.path``, and so that ``web.test_api`` can import ``web.app`` without a
path hack. Nothing here is part of the installed distribution:
``pyproject.toml`` restricts packaging to ``src/sperm_sorting*``, which keeps a
demo out of the runtime that drives real hardware.
"""

from __future__ import annotations
