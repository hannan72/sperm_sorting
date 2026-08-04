"""Split leakage detection. The most consequential validator in this package.

What leakage looks like here
----------------------------
Sperm datasets leak in a way that is almost invisible in the metrics and
completely fatal in the clinic:

1. **Same video on both sides.** VISEM-Tracking's 29,196 frames come from 20
   videos. Split the *frames* at random and 80% of the validation frames have a
   near-identical twin in training -- same patient, same optics, same debris,
   often the same sperm one 20-millisecond step earlier. The reported mAP then
   measures memorisation of 20 fields of view. The published baseline
   (YOLOv5l, mAP@0.5 = 0.2231) is what an honest per-video split looks like; a
   frame-level split will happily report several times that and mean nothing.

2. **Adjacent frames across the boundary.** Even a "per-clip" split leaks if
   clips were cut from one recording: frame 1439 in train and frame 1440 in val
   are the same picture. :func:`check_adjacent_frames` catches this, and it is
   the check people skip because the frame *ids* differ, which feels like
   enough. It is not -- at 50 FPS a sperm moves a few pixels per frame.

3. **Same patient, different video.** The unit that must not cross the boundary
   is the *patient*, not the file. Two clips from one donor share sample
   preparation, chamber, and cell population.

The rule this module enforces is therefore: **group first, split groups**. Never
split items.

MHSMA is the honest exception
-----------------------------
MHSMA ships 1540 pre-split crops (1000/240/300) from 235 patients and publishes
**no patient identifier**. It is therefore impossible to verify from the
released files that its official split is patient-level; the images could be
grouped by patient or interleaved and nothing in the ``.npy`` files would tell
you. See :func:`mhsma_split_leakage_note` -- it returns an ``UNVERIFIABLE``
check, not a pass. The distinction matters: an unverifiable risk that gets
reported as a pass is how a validation number acquires more confidence than it
has earned.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from sperm_sorting.errors import LeakageError

from .integrity import CheckResult, CheckStatus

__all__ = [
    "AdjacencyViolation",
    "LeakageReport",
    "assert_no_frame_leakage",
    "check_adjacent_frames",
    "default_frame_key",
    "default_group_key",
    "group_items",
    "mhsma_split_leakage_note",
    "patient_level_split",
    "summarise_split",
]

_T = TypeVar("_T")

#: Attribute / mapping keys inspected by :func:`default_group_key`, in priority
#: order. Deliberately does **not** include anything frame-like: mistaking a
#: frame id for a group id would make every check pass while leaking every
#: video.
_GROUP_KEY_CANDIDATES: tuple[str, ...] = (
    "group_id",
    "patient_id",
    "participant_id",
    "subject_id",
    "sample_id",
    "video_id",
    "sequence_id",
    "clip_id",
)

#: Attribute / mapping keys inspected by :func:`default_frame_key`.
_FRAME_KEY_CANDIDATES: tuple[str, ...] = ("frame_id", "frame_number", "frame", "index")


# ==========================================================================
# Key extraction
# ==========================================================================


def _lookup(item: Any, candidates: Sequence[str]) -> Any | None:
    """First present candidate key, whether ``item`` is a mapping or an object."""
    if isinstance(item, Mapping):
        for key in candidates:
            if key in item and item[key] is not None:
                return item[key]
        return None
    for key in candidates:
        value = getattr(item, key, None)
        if value is not None:
            return value
    return None


def default_group_key(item: Any) -> Hashable:
    """Extract the grouping key (patient / video / sample) from ``item``.

    Accepts mappings, dataclasses and plain objects, and looks for the keys in
    :data:`_GROUP_KEY_CANDIDATES`. A bare ``str``/``int`` item is taken to *be*
    its own group id, which is what makes ``assert_no_frame_leakage(["a"], ["a"])``
    do the obvious thing.

    Raises
    ------
    LeakageError
        When no grouping key can be found. This is a hard failure rather than a
        fallback to ``id(item)``: a leakage check that cannot find the group is
        a leakage check that would pass on everything.
    """
    if isinstance(item, (str, int)):
        return item
    if isinstance(item, tuple) and item:
        # Convention used by group_items(): (group, frame).
        return item[0]
    value = _lookup(item, _GROUP_KEY_CANDIDATES)
    if value is None:
        raise LeakageError(
            f"cannot determine the grouping key for {item!r} (type {type(item).__name__}). "
            f"Expected one of {list(_GROUP_KEY_CANDIDATES)} as an attribute or mapping key, "
            "or pass an explicit key_fn. Refusing to guess: a leakage check that cannot "
            "identify the group would pass on a fully leaked split."
        )
    return value


def default_frame_key(item: Any) -> int | None:
    """Extract a frame index from ``item``; ``None`` when there is not one.

    ``None`` is a legitimate answer (a participant-level record has no frame),
    and :func:`check_adjacent_frames` skips those items rather than inventing
    an index for them.
    """
    if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], int):
        return int(item[1])
    value = _lookup(item, _FRAME_KEY_CANDIDATES)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def group_items(
    items: Iterable[_T], key_fn: Callable[[_T], Hashable] = default_group_key
) -> dict[Hashable, list[_T]]:
    """Bucket ``items`` by group key, preserving input order within a bucket."""
    groups: dict[Hashable, list[_T]] = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    return groups


# ==========================================================================
# Hard leakage: the same group on both sides
# ==========================================================================


def assert_no_frame_leakage(
    train_items: Iterable[Any],
    val_items: Iterable[Any],
    *,
    key_fn: Callable[[Any], Hashable] = default_group_key,
    train_name: str = "train",
    val_name: str = "val",
) -> None:
    """Raise :class:`~sperm_sorting.errors.LeakageError` if any group is on both sides.

    The group is whatever ``key_fn`` returns: a video id for VISEM-Tracking, a
    participant id for VISEM, a sample id for device captures. Frame ids are
    explicitly *not* group keys -- see :func:`default_group_key`.

    Parameters
    ----------
    train_items, val_items
        Any iterables of items. They are consumed once.
    key_fn
        Group extractor. Defaults to :func:`default_group_key`.
    train_name, val_name
        Names used in the error message, so the message reads correctly for a
        train/test or val/test comparison too.

    Raises
    ------
    LeakageError
        Listing every shared group (up to a readable limit) and the number of
        items each contributes on each side, because "video 52 is on both
        sides" is actionable and "leakage detected" is not.
    """
    train_groups = group_items(train_items, key_fn)
    val_groups = group_items(val_items, key_fn)
    shared = sorted(
        (set(train_groups) & set(val_groups)),
        key=lambda g: (-(len(train_groups[g]) + len(val_groups[g])), str(g)),
    )
    if not shared:
        return

    shown = shared[:20]
    detail = ", ".join(
        f"{g!r} ({len(train_groups[g])} in {train_name}, {len(val_groups[g])} in {val_name})"
        for g in shown
    )
    if len(shared) > len(shown):
        detail += f", ... and {len(shared) - len(shown)} more"
    raise LeakageError(
        f"{len(shared)} group(s) appear in both {train_name} and {val_name}: {detail}. "
        "Frames from one video/patient must never straddle a split boundary -- the "
        "resulting validation score measures memorisation of that field of view. "
        "Use patient_level_split() to build the split instead."
    )


# ==========================================================================
# Soft leakage: temporally adjacent frames across the boundary
# ==========================================================================


@dataclass(frozen=True, slots=True)
class AdjacencyViolation:
    """One pair of near-in-time frames from one group, split across the boundary."""

    group: Hashable
    train_frame: int
    val_frame: int

    @property
    def gap(self) -> int:
        return abs(self.train_frame - self.val_frame)

    def describe(self) -> str:
        return (
            f"group {self.group!r}: train frame {self.train_frame} and val frame "
            f"{self.val_frame} are {self.gap} frame(s) apart"
        )


@dataclass(slots=True)
class LeakageReport:
    """Result of :func:`check_adjacent_frames`."""

    violations: list[AdjacencyViolation] = field(default_factory=list)
    max_gap: int = 1
    #: Groups present on both sides at all. Any non-empty value here is already
    #: hard leakage; adjacency is the *additional* thing being measured.
    shared_groups: list[Hashable] = field(default_factory=list)
    #: Items skipped because they carried no frame index.
    n_unindexed: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def raise_if_leaked(self) -> None:
        """Raise :class:`LeakageError` when any adjacency violation was found."""
        if self.ok:
            return
        shown = self.violations[:20]
        detail = "; ".join(v.describe() for v in shown)
        if len(self.violations) > len(shown):
            detail += f"; ... and {len(self.violations) - len(shown)} more"
        raise LeakageError(
            f"{len(self.violations)} temporally adjacent frame pair(s) straddle the "
            f"split boundary (max_gap={self.max_gap}): {detail}. Consecutive frames of "
            "one recording are the same picture a few milliseconds apart; having one in "
            "train and one in val inflates every metric that follows."
        )

    def to_check_result(self, name: str = "leakage:adjacent_frames") -> CheckResult:
        if self.ok:
            return CheckResult(
                name,
                CheckStatus.PASS,
                f"no train/val frame pair from one group is within {self.max_gap} "
                f"frame(s) of another ({self.n_unindexed} item(s) carried no frame index)",
                {"n_unindexed": self.n_unindexed},
            )
        return CheckResult(
            name,
            CheckStatus.FAIL,
            f"{len(self.violations)} adjacent frame pair(s) across the split boundary; "
            f"first: {self.violations[0].describe()}",
            {
                "n_violations": len(self.violations),
                "max_gap": self.max_gap,
                "groups": [str(g) for g in sorted({v.group for v in self.violations}, key=str)],
            },
        )


def check_adjacent_frames(
    train_items: Iterable[Any],
    val_items: Iterable[Any],
    *,
    max_gap: int = 1,
    key_fn: Callable[[Any], Hashable] = default_group_key,
    frame_fn: Callable[[Any], int | None] = default_frame_key,
    max_violations: int = 1000,
) -> LeakageReport:
    """Find frames from one group that sit within ``max_gap`` of the other side.

    This exists because the obvious per-frame split *looks* clean: the frame ids
    on either side are disjoint, no id repeats, and every naive check passes.
    But frame 900 in train and frame 901 in val are the same field of view
    20 milliseconds apart, and a detector that has seen one has effectively seen
    the other.

    ``max_gap`` is a **frame** count, not a time. At VISEM-Tracking's 45-50 FPS,
    ``max_gap=1`` catches only literal neighbours; a realistic guard for
    near-duplicate content is closer to one second, i.e. ``max_gap=50``. The
    default is deliberately the strict-adjacency case so that a caller who wants
    the stronger guarantee has to say so and thereby document the choice.

    Returns
    -------
    LeakageReport
        Never raises on its own -- call :meth:`LeakageReport.raise_if_leaked` or
        fold :meth:`LeakageReport.to_check_result` into a
        :class:`~datasets.validators.integrity.ValidationReport`. Reporting
        rather than raising lets a caller count violations before deciding, and
        this check has legitimate near-miss cases (a deliberately held-out
        contiguous tail of a video) where the count is the whole answer.
    """
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")

    train_by_group: dict[Hashable, list[int]] = {}
    n_unindexed = 0
    for item in train_items:
        frame = frame_fn(item)
        if frame is None:
            n_unindexed += 1
            continue
        train_by_group.setdefault(key_fn(item), []).append(frame)
    for frames in train_by_group.values():
        frames.sort()

    violations: list[AdjacencyViolation] = []
    shared: set[Hashable] = set()
    for item in val_items:
        frame = frame_fn(item)
        if frame is None:
            n_unindexed += 1
            continue
        group = key_fn(item)
        train_frames = train_by_group.get(group)
        if not train_frames:
            continue
        shared.add(group)
        if len(violations) >= max_violations:
            continue
        # Linear scan over the window instead of bisect: max_gap is tiny and a
        # sorted list plus an explicit window is easier to prove correct than
        # index arithmetic that must not be off by one.
        for candidate in range(frame - max_gap, frame + max_gap + 1):
            if _contains(train_frames, candidate):
                violations.append(AdjacencyViolation(group, candidate, frame))
                break

    return LeakageReport(
        violations=violations,
        max_gap=max_gap,
        shared_groups=sorted(shared, key=str),
        n_unindexed=n_unindexed,
    )


def _contains(sorted_values: Sequence[int], value: int) -> bool:
    """Binary-search membership in a sorted list of ints."""
    lo, hi = 0, len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < value:
            lo = mid + 1
        elif sorted_values[mid] > value:
            hi = mid
        else:
            return True
    return False


# ==========================================================================
# Building a split that cannot leak
# ==========================================================================


def patient_level_split(
    items: Sequence[_T],
    key_fn: Callable[[_T], Hashable] = default_group_key,
    ratios: Mapping[str, float] | None = None,
    seed: int = 0,
) -> dict[str, list[_T]]:
    """Split ``items`` so that every group lands entirely on one side.

    Algorithm, and why it is not "shuffle the groups and cut":

    1. Bucket items by group.
    2. Shuffle the *group keys* with ``random.Random(seed)`` -- reproducible,
       and independent of the process-wide RNG so a caller's ``random.seed()``
       cannot change a split.
    3. Stable-sort the groups by size, largest first, then assign each group to
       whichever split currently has the largest *item* deficit against its
       target.

    Step 3 matters because groups are wildly uneven: VISEM-Tracking videos hold
    1440-1500 frames each, a device capture might hold 200. Cutting a shuffled
    list of groups at 80% of the *group count* can put 70% or 90% of the frames
    in train, and the resulting val set is either too small to measure anything
    or large enough to have cost real training data. Assigning largest-first to
    the biggest deficit lands within one group's worth of the requested item
    ratio. The shuffle still decides ties, so ``seed`` remains meaningful.

    Parameters
    ----------
    items
        Items to split. Consumed as a sequence (indexable, re-iterable).
    key_fn
        Group extractor; see :func:`default_group_key`.
    ratios
        Split name -> proportion of *items*. Must be positive and sum to 1
        within 1e-6. Defaults to ``{"train": 0.8, "val": 0.2}``.
    seed
        Reproducibility seed for tie-breaking.

    Returns
    -------
    dict
        Split name -> items, in the order they appeared in ``items``.

    Raises
    ------
    ValueError
        If ``ratios`` is malformed.
    LeakageError
        If the result somehow contains a shared group. This is a self-check on
        the function's own output; it should be unreachable, and it is here
        because "the split builder is correct" is exactly the assumption that
        must not be taken on trust.
    """
    resolved = dict(ratios) if ratios is not None else {"train": 0.8, "val": 0.2}
    if not resolved:
        raise ValueError("ratios must not be empty")
    if any(v <= 0 for v in resolved.values()):
        raise ValueError(f"every ratio must be positive, got {resolved}")
    total_ratio = sum(resolved.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total_ratio} from {resolved}")

    grouped = group_items(items, key_fn)
    if not grouped:
        return {name: [] for name in resolved}

    keys = sorted(grouped, key=str)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: len(grouped[k]), reverse=True)  # stable: shuffle breaks ties

    n_items = len(items)
    targets = {name: ratio * n_items for name, ratio in resolved.items()}
    assigned: dict[str, list[Hashable]] = {name: [] for name in resolved}
    sizes: dict[str, int] = dict.fromkeys(resolved, 0)

    names = list(resolved)
    for key in keys:
        # Largest deficit first; ties broken by declaration order so the result
        # does not depend on dict iteration accidents.
        best = max(names, key=lambda n: (targets[n] - sizes[n], -names.index(n)))
        assigned[best].append(key)
        sizes[best] += len(grouped[key])

    out: dict[str, list[_T]] = {}
    for name, group_keys in assigned.items():
        chosen = set(group_keys)
        out[name] = [item for item in items if key_fn(item) in chosen]

    # Self-check: pairwise, no group may be shared. Cheap relative to the cost
    # of discovering the opposite from a model that scored too well.
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert_no_frame_leakage(
                out[left], out[right], key_fn=key_fn, train_name=left, val_name=right
            )
    return out


def summarise_split(
    split: Mapping[str, Sequence[Any]],
    *,
    key_fn: Callable[[Any], Hashable] = default_group_key,
) -> dict[str, dict[str, Any]]:
    """Per-split item count, group count and item share. For logging a split."""
    total = sum(len(v) for v in split.values()) or 1
    return {
        name: {
            "n_items": len(items),
            "n_groups": len({key_fn(i) for i in items}),
            "item_fraction": len(items) / total,
            "groups": sorted((str(key_fn(i)) for i in items), key=str)[:50],
        }
        for name, items in split.items()
    }


# ==========================================================================
# MHSMA: the risk that cannot be measured
# ==========================================================================


def mhsma_split_leakage_note() -> CheckResult:
    """The MHSMA patient-level-split check that *cannot be performed*.

    MHSMA publishes 1540 cropped images -- 1000 train / 240 valid / 300 test --
    drawn from 235 male-factor-infertility patients, and ships them as six image
    ``.npy`` arrays with no patient, slide or field identifier of any kind. It
    follows that:

    * whether the official split is patient-level **cannot be determined from
      the released files**;
    * if it is not, the published validation and test numbers include an unknown
      amount of same-patient leakage, and any number this repository produces on
      that split inherits it;
    * building a *new* split from these files cannot be done safely either,
      because the grouping key needed to do it correctly is not published.

    So this returns ``UNVERIFIABLE`` -- never ``PASS``. Every MHSMA validation
    report carries it, and the honest reading of an MHSMA score is "consistent
    with the literature on this benchmark", not "generalises to unseen patients".

    Returns
    -------
    CheckResult
        Status :attr:`~datasets.validators.integrity.CheckStatus.UNVERIFIABLE`.
    """
    return CheckResult(
        name="leakage:mhsma_patient_level_split",
        status=CheckStatus.UNVERIFIABLE,
        message=(
            "MHSMA publishes no patient/slide identifier, so it cannot be verified "
            "from the released .npy files that its official 1000/240/300 split is "
            "patient-level (1540 images come from 235 patients). The leakage risk is "
            "UNVERIFIABLE, not absent: treat MHSMA scores as benchmark-comparable, "
            "not as evidence of generalisation to unseen patients, and do not "
            "re-split these files -- the grouping key needed to do it safely is not "
            "published."
        ),
        details={
            "n_images": 1540,
            "n_patients": 235,
            "official_split": {"train": 1000, "valid": 240, "test": 300},
            "patient_identifier_published": False,
        },
    )
