"""Training and evaluation entry points for the sperm-analysis prototype.

This package is deliberately *outside* ``src/sperm_sorting``. The runtime
package must be installable and importable on a device that will never train
anything, so torch-training concerns -- optimisers, schedulers, TensorBoard,
matplotlib -- do not belong in it. Everything here imports *from* the runtime
package and never the other way round.

Two rules govern every script in this package:

1. **A result that cannot be reproduced is not a result.** Every training and
   evaluation run writes an ``experiment.json`` capturing the git commit, the
   fully-resolved configuration, package versions, dataset identity and split
   sizes, the seed, the hardware and the final metrics. That file is a
   mandatory output, not a convenience.
2. **No number is ever invented.** No docstring, README or default in this
   package quotes an accuracy, a latency or a score that did not come out of an
   actual run on the machine that printed it.

Scripts are runnable either way::

    python training/train_morphology.py --help
    python -m training.train_morphology --help

The first form works because every script calls
:func:`training.bootstrap.ensure_importable` before importing its siblings.
"""

from __future__ import annotations

__all__ = ["__version__"]

#: Version of the training harness itself, stamped into every experiment
#: record. Bumped when an output format or a metric definition changes, so a
#: stored record can always be interpreted by the code that reads it.
__version__ = "1.0.0"
