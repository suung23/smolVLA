"""Stable transition identity.

Every scored sample is addressed by ``(dataset_id, episode_index, frame_index)``.

Dataframe row numbers are never used to join preprocessing output to
transitions: a split, a corruption copy or a re-download reshuffles rows, and a
row-number join would silently attach the wrong score to the wrong frame.
``dataset_id`` carries the *variant* (e.g. ``dataset`` vs
``dataset__blur_s2_seed0``), so an original frame and its corrupted copy are
distinct identities that still expose their shared parent episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

#: Separator for the flat string form. Chosen because it cannot appear in a
#: HF repo id slug, a variant suffix or a zero-padded integer.
SEP = "|"

_EP_WIDTH = 6
_FR_WIDTH = 7


def sanitize_dataset_id(repo_id: str) -> str:
    """Turn a HF repo id (a dataset repo id) into a filesystem/key safe id."""
    out = repo_id.strip().replace("/", "__").replace(SEP, "_")
    if not out:
        raise ValueError("dataset_id must not be empty")
    return out


@dataclass(frozen=True, slots=True, order=True)
class TransitionID:
    """Stable identity of one training transition."""

    dataset_id: str
    episode_index: int
    frame_index: int

    def __post_init__(self) -> None:
        if SEP in self.dataset_id:
            raise ValueError(f"dataset_id must not contain {SEP!r}: {self.dataset_id!r}")
        if self.episode_index < 0 or self.frame_index < 0:
            raise ValueError(f"indices must be non-negative, got {self}")

    @property
    def key(self) -> str:
        """Flat, sortable, round-trippable string form."""
        return (
            f"{self.dataset_id}{SEP}"
            f"{self.episode_index:0{_EP_WIDTH}d}{SEP}"
            f"{self.frame_index:0{_FR_WIDTH}d}"
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.key

    @classmethod
    def parse(cls, key: str) -> "TransitionID":
        parts = key.split(SEP)
        if len(parts) != 3:
            raise ValueError(f"not a transition key: {key!r}")
        return cls(parts[0], int(parts[1]), int(parts[2]))


def make_keys(
    dataset_id: str,
    episode_index: Sequence[int] | np.ndarray,
    frame_index: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Vectorised :attr:`TransitionID.key` for aligned episode/frame arrays."""
    ep = np.asarray(episode_index, dtype=np.int64).reshape(-1)
    fr = np.asarray(frame_index, dtype=np.int64).reshape(-1)
    if ep.shape != fr.shape:
        raise ValueError(f"episode/frame length mismatch: {ep.shape} vs {fr.shape}")
    if ep.size and (ep.min() < 0 or fr.min() < 0):
        raise ValueError("indices must be non-negative")
    return np.array(
        [TransitionID(dataset_id, int(e), int(f)).key for e, f in zip(ep, fr)],
        dtype=object,
    )


def iter_ids(
    dataset_id: str,
    episode_index: Iterable[int],
    frame_index: Iterable[int],
) -> Iterator[TransitionID]:
    for e, f in zip(episode_index, frame_index, strict=True):
        yield TransitionID(dataset_id, int(e), int(f))


def build_index(keys: Sequence[str] | np.ndarray) -> dict[str, int]:
    """Map transition key -> position, rejecting duplicates.

    A duplicate key means two rows claim the same identity; joining on it would
    be ambiguous, so we fail loudly instead of picking one.
    """
    index: dict[str, int] = {}
    for i, k in enumerate(keys):
        k = str(k)
        if k in index:
            raise ValueError(f"duplicate transition key at positions {index[k]} and {i}: {k!r}")
        index[k] = i
    return index


def align_to(
    target_keys: Sequence[str] | np.ndarray,
    source_keys: Sequence[str] | np.ndarray,
    *,
    required: bool = True,
) -> np.ndarray:
    """Return positions in ``source_keys`` for each of ``target_keys``.

    Missing entries are ``-1`` when ``required=False``; otherwise a
    ``KeyError`` names the first few missing keys. This is the only sanctioned
    way to attach a sidecar to a dataset.
    """
    src = build_index(source_keys)
    out = np.full(len(target_keys), -1, dtype=np.int64)
    missing: list[str] = []
    for i, k in enumerate(target_keys):
        pos = src.get(str(k), -1)
        out[i] = pos
        if pos < 0 and len(missing) < 5:
            missing.append(str(k))
    if required and missing:
        raise KeyError(
            f"{int((out < 0).sum())} transition(s) missing from source; first: {missing}"
        )
    return out


def parent_episode_key(dataset_id: str, episode_index: int) -> str:
    """Episode-level key used by splits (frame-agnostic)."""
    return f"{dataset_id}{SEP}{episode_index:0{_EP_WIDTH}d}"
