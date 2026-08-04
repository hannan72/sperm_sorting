"""Morphology crop extraction.

:class:`CropExtractor` is the last stage before the morphology model. It cuts
the padded box selected by
:class:`~sperm_sorting.quality.selector.BestFrameSelector`, letterboxes it to
the model's input size without distorting the head's length:width ratio, and
emits the :class:`~sperm_sorting.schemas.track.CropRecord` that ties the crop
to the track whose motion was measured. That binding is enforced here, not
assumed -- see :class:`CropIdentityError`.
"""

from __future__ import annotations

from .extractor import CropExtractor, CropIdentityError

__all__ = ["CropExtractor", "CropIdentityError"]
