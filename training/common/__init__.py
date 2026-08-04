"""Shared machinery for every training and evaluation script.

Nothing in here is script-specific. If two scripts would otherwise grow the
same helper, it lives here instead, because a metric or a checkpoint format
that is implemented twice is a metric or a checkpoint format that will
eventually disagree with itself.
"""

from __future__ import annotations

__all__: list[str] = []
