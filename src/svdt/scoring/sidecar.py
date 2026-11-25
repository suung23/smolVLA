"""Score sidecars: persist v_t, c_t, every component, and how they were made.

A sidecar is two files that always travel together:

``<name>.npz``
    Per-transition arrays. ``keys`` is the stable transition key, so a sidecar
    attaches to a dataset by identity and never by row order.
``<name>.json``
    The fingerprint of the inputs, the resolved config, the scorer metadata
    (lambda values, references, expected areas, grouping fallbacks) and the
    distribution summaries.

Why both components *and* the final scores are stored: the sensitivity analysis
asked for in the project description needs to re-combine components under
different lambdas without re-decoding video, and so does any re-tuning of alpha
or w_min.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..ids import align_to, build_index
from ..io_utils import (
    artifact_is_current,
    ensure_dir,
    fingerprint,
    read_json,
    write_json,
    write_npz,
)

SIDECAR_VERSION = 3


@dataclass
class ScoreSidecar:
    """Per-transition scores plus provenance."""

    dataset_id: str
    keys: np.ndarray  # (N,) str
    episode_index: np.ndarray
    frame_index: np.ndarray
    #: Final scores. Either may be absent when only one scorer was run.
    v: np.ndarray | None = None
    c: np.ndarray | None = None
    q_perception: np.ndarray | None = None
    #: Component scores, e.g. ``q_object``, ``c_synchronization``.
    components: dict[str, np.ndarray] = field(default_factory=dict)
    #: Availability / missing-evidence flags.
    flags: dict[str, np.ndarray] = field(default_factory=dict)
    #: Raw measurements retained for sensitivity analysis.
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    #: Scorer metadata, config and fingerprints.
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.keys)

    def __post_init__(self) -> None:
        n = len(self.keys)
        for name, arr in (
            ("episode_index", self.episode_index),
            ("frame_index", self.frame_index),
        ):
            if len(arr) != n:
                raise ValueError(f"sidecar {name} has {len(arr)} rows, expected {n}")
        for label, group in (
            ("v", {"v": self.v} if self.v is not None else {}),
            ("c", {"c": self.c} if self.c is not None else {}),
            ("q_perception", {"q_perception": self.q_perception} if self.q_perception is not None else {}),
            ("component", self.components),
            ("flag", self.flags),
            ("raw", self.raw),
        ):
            for key, arr in group.items():
                if len(arr) != n:
                    raise ValueError(
                        f"sidecar {label} array {key!r} has {len(arr)} rows, expected {n}"
                    )
        build_index(self.keys)  # rejects duplicate identities

    # -- lookup ------------------------------------------------------------ #

    def align_to(
        self, target_keys: Sequence[str] | np.ndarray, *, required: bool = True
    ) -> np.ndarray:
        """Positions in this sidecar for each of ``target_keys`` (-1 if absent)."""
        return align_to(target_keys, self.keys, required=required)

    def subset(self, mask: np.ndarray) -> "ScoreSidecar":
        mask = np.asarray(mask, dtype=bool)
        pick = lambda a: None if a is None else np.asarray(a)[mask]  # noqa: E731
        return ScoreSidecar(
            dataset_id=self.dataset_id,
            keys=self.keys[mask],
            episode_index=self.episode_index[mask],
            frame_index=self.frame_index[mask],
            v=pick(self.v),
            c=pick(self.c),
            q_perception=pick(self.q_perception),
            components={k: np.asarray(v)[mask] for k, v in self.components.items()},
            flags={k: np.asarray(v)[mask] for k, v in self.flags.items()},
            raw={k: np.asarray(v)[mask] for k, v in self.raw.items()},
            meta=dict(self.meta),
        )

    def dataset_ids(self) -> np.ndarray:
        """Per-row dataset id, parsed back out of the keys."""
        from ..ids import SEP

        return np.array([str(k).split(SEP)[0] for k in self.keys], dtype=object)

    # -- summaries --------------------------------------------------------- #

    def distribution(self, name: str) -> dict[str, Any] | None:
        arr = {"v": self.v, "c": self.c, "q_perception": self.q_perception}.get(name)
        if arr is None:
            arr = self.components.get(name)
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float64)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return {"n": 0}
        qs = np.percentile(finite, [0, 5, 25, 50, 75, 95, 100])
        return {
            "n": int(finite.size),
            "mean": round(float(finite.mean()), 6),
            "std": round(float(finite.std()), 6),
            "min": round(float(qs[0]), 6),
            "p05": round(float(qs[1]), 6),
            "p25": round(float(qs[2]), 6),
            "median": round(float(qs[3]), 6),
            "p75": round(float(qs[4]), 6),
            "p95": round(float(qs[5]), 6),
            "max": round(float(qs[6]), 6),
            "histogram": np.histogram(finite, bins=10, range=(0.0, 1.0))[0].tolist(),
        }

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "n_transitions": len(self),
            "n_episodes": int(len(np.unique(self.episode_index))),
            "distributions": {},
        }
        for name in ("v", "c", "q_perception", *sorted(self.components)):
            dist = self.distribution(name)
            if dist is not None:
                out["distributions"][name] = dist
        out["flags"] = {
            k: int(np.asarray(v, dtype=bool).sum()) for k, v in sorted(self.flags.items())
        }
        return out


# --------------------------------------------------------------------------- #
# paths / io
# --------------------------------------------------------------------------- #


def sidecar_paths(root: str | Path, dataset_id: str, kind: str = "scores") -> tuple[Path, Path]:
    base = Path(root) / f"{kind}__{dataset_id}"
    return base.with_suffix(".npz"), base.with_suffix(".json")


def save_sidecar(
    sidecar: ScoreSidecar,
    root: str | Path,
    *,
    kind: str = "scores",
    input_fingerprint: str | None = None,
) -> tuple[Path, Path]:
    """Atomically write the sidecar pair, stamped with its input fingerprint."""
    npz_path, json_path = sidecar_paths(root, sidecar.dataset_id, kind)
    ensure_dir(npz_path.parent)

    arrays: dict[str, np.ndarray] = {
        "keys": np.asarray([str(k) for k in sidecar.keys], dtype="U"),
        "episode_index": np.asarray(sidecar.episode_index, dtype=np.int64),
        "frame_index": np.asarray(sidecar.frame_index, dtype=np.int64),
    }
    if sidecar.v is not None:
        arrays["v"] = np.asarray(sidecar.v, dtype=np.float32)
    if sidecar.c is not None:
        arrays["c"] = np.asarray(sidecar.c, dtype=np.float32)
    if sidecar.q_perception is not None:
        arrays["q_perception"] = np.asarray(sidecar.q_perception, dtype=np.float32)
    for key, value in sidecar.components.items():
        arrays[f"component__{key}"] = np.asarray(value, dtype=np.float32)
    for key, value in sidecar.flags.items():
        arrays[f"flag__{key}"] = np.asarray(value, dtype=bool)
    for key, value in sidecar.raw.items():
        arr = np.asarray(value)
        arrays[f"raw__{key}"] = arr if arr.dtype.kind in "biu" else arr.astype(np.float32)
    write_npz(npz_path, **arrays)

    fp = input_fingerprint or sidecar.meta.get("fingerprint") or fingerprint(
        {"dataset_id": sidecar.dataset_id, "keys": len(sidecar)}
    )
    write_json(
        json_path,
        {
            "kind": f"score_sidecar/{kind}",
            "sidecar_version": SIDECAR_VERSION,
            "fingerprint": fp,
            "arrays_file": npz_path.name,
            "arrays": sorted(arrays),
            **sidecar.summary(),
            "meta": sidecar.meta,
        },
    )
    return npz_path, json_path


def load_sidecar(
    root: str | Path, dataset_id: str, *, kind: str = "scores"
) -> ScoreSidecar:
    npz_path, json_path = sidecar_paths(root, dataset_id, kind)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"no sidecar at {npz_path}. Run `svdt score` for dataset {dataset_id!r} first."
        )
    meta_doc = read_json(json_path, {}) or {}
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    components = {
        k.removeprefix("component__"): v
        for k, v in arrays.items()
        if k.startswith("component__")
    }
    flags = {
        k.removeprefix("flag__"): v.astype(bool)
        for k, v in arrays.items()
        if k.startswith("flag__")
    }
    raw = {k.removeprefix("raw__"): v for k, v in arrays.items() if k.startswith("raw__")}
    return ScoreSidecar(
        dataset_id=dataset_id,
        keys=np.asarray([str(k) for k in arrays["keys"]], dtype=object),
        episode_index=arrays["episode_index"],
        frame_index=arrays["frame_index"],
        v=arrays.get("v"),
        c=arrays.get("c"),
        q_perception=arrays.get("q_perception"),
        components=components,
        flags=flags,
        raw=raw,
        meta=dict(meta_doc.get("meta", {})) | {"fingerprint": meta_doc.get("fingerprint")},
    )


def sidecar_is_current(
    root: str | Path, dataset_id: str, expected_fingerprint: str, *, kind: str = "scores"
) -> bool:
    """Whether an existing sidecar already reflects these exact inputs."""
    npz_path, json_path = sidecar_paths(root, dataset_id, kind)
    return npz_path.exists() and artifact_is_current(json_path, expected_fingerprint)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def build_sidecar(
    dataset_id: str,
    *,
    visibility: Any | None = None,
    reliability: Any | None = None,
    meta: Mapping[str, Any] | None = None,
) -> ScoreSidecar:
    """Merge a visibility and a reliability result on their transition keys.

    The two scorers can run over different frame subsets (visibility is often
    strided to save video decoding). The merge is therefore an **outer** join on
    the key, with per-score availability recorded: a transition with a c_t but
    no v_t keeps its c_t and is flagged ``v_missing``, rather than being dropped
    or silently given a default.
    """
    if visibility is None and reliability is None:
        raise ValueError("build_sidecar needs at least one of visibility/reliability")

    key_order: list[str] = []
    seen: set[str] = set()
    for result in (reliability, visibility):
        if result is None:
            continue
        for k in result.keys:
            k = str(k)
            if k not in seen:
                seen.add(k)
                key_order.append(k)
    keys = np.array(key_order, dtype=object)
    n = len(keys)

    ep = np.full(n, -1, dtype=np.int64)
    fr = np.full(n, -1, dtype=np.int64)
    components: dict[str, np.ndarray] = {}
    flags: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}
    v_arr: np.ndarray | None = None
    c_arr: np.ndarray | None = None
    merged_meta: dict[str, Any] = dict(meta or {})

    for label, result in (("v", visibility), ("c", reliability)):
        if result is None:
            continue
        pos = align_to(keys, result.keys, required=False)
        present = pos >= 0
        src = np.where(present, pos, 0)

        ep = np.where(present & (ep < 0), np.asarray(result.episode_index)[src], ep)
        fr = np.where(present & (fr < 0), np.asarray(result.frame_index)[src], fr)

        score = np.asarray(getattr(result, label), dtype=np.float32)[src]
        score = np.where(present, score, np.nan).astype(np.float32)
        if label == "v":
            v_arr = score
        else:
            c_arr = score
        flags[f"{label}_present"] = present

        for name, arr in result.components.items():
            a = np.asarray(arr, dtype=np.float32)[src]
            components[name] = np.where(present, a, np.nan).astype(np.float32)
        for name, arr in result.availability.items():
            a = np.asarray(arr, dtype=bool)[src]
            flags[name] = a & present
        for name, arr in result.raw.items():
            a = np.asarray(arr)
            picked = a[src]
            if picked.dtype.kind in "biu":
                raw[name] = np.where(present, picked, -1)
            else:
                raw[name] = np.where(present, picked, np.nan).astype(np.float32)

        extra_flag = getattr(result, "missing_evidence", None)
        if extra_flag is not None:
            flags["missing_evidence"] = np.asarray(extra_flag, dtype=bool)[src] & present
        unavailable = getattr(result, "unavailable", None)
        if unavailable is not None:
            flags["reliability_unavailable"] = (
                np.asarray(unavailable, dtype=bool)[src] & present
            )
        n_used = getattr(result, "n_components_used", None)
        if n_used is not None:
            raw[f"n_components_used__{label}"] = np.where(
                present, np.asarray(n_used, dtype=np.int16)[src], -1
            )
        merged_meta[f"{label}_meta"] = result.meta

    return ScoreSidecar(
        dataset_id=dataset_id,
        keys=keys,
        episode_index=ep,
        frame_index=fr,
        v=v_arr,
        c=c_arr,
        components=components,
        flags=flags,
        raw=raw,
        meta=merged_meta,
    )


def merge_sidecars(sidecars: Sequence[ScoreSidecar], *, dataset_id: str) -> ScoreSidecar:
    """Concatenate sidecars of different variants into one lookup table."""
    if not sidecars:
        raise ValueError("nothing to merge")
    comp_names = sorted({k for s in sidecars for k in s.components})
    flag_names = sorted({k for s in sidecars for k in s.flags})
    raw_names = sorted({k for s in sidecars for k in s.raw})

    def stack(name: str, group: str, dtype) -> np.ndarray:
        parts = []
        for s in sidecars:
            holder = getattr(s, group)
            arr = holder.get(name)
            if arr is None:
                fill = False if dtype is bool else np.nan
                arr = np.full(len(s), fill, dtype=dtype)
            parts.append(np.asarray(arr, dtype=dtype))
        return np.concatenate(parts)

    has_v = any(s.v is not None for s in sidecars)
    has_c = any(s.c is not None for s in sidecars)
    has_q = any(s.q_perception is not None for s in sidecars)

    def stack_score(attr: str) -> np.ndarray:
        parts = []
        for s in sidecars:
            arr = getattr(s, attr)
            parts.append(
                np.full(len(s), np.nan, dtype=np.float32)
                if arr is None
                else np.asarray(arr, dtype=np.float32)
            )
        return np.concatenate(parts)

    merged_meta: dict[str, Any] = {"merged_from": [s.dataset_id for s in sidecars]}
    for s in sidecars:
        merged_meta[s.dataset_id] = s.meta
    return ScoreSidecar(
        dataset_id=dataset_id,
        keys=np.concatenate([s.keys for s in sidecars]),
        episode_index=np.concatenate([s.episode_index for s in sidecars]),
        frame_index=np.concatenate([s.frame_index for s in sidecars]),
        v=stack_score("v") if has_v else None,
        c=stack_score("c") if has_c else None,
        q_perception=stack_score("q_perception") if has_q else None,
        components={n: stack(n, "components", np.float32) for n in comp_names},
        flags={n: stack(n, "flags", bool) for n in flag_names},
        raw={n: stack(n, "raw", np.float32) for n in raw_names},
        meta=merged_meta,
    )


def attach_perception_scores(
    sidecar: ScoreSidecar,
    scores: Mapping[str, float],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ScoreSidecar:
    """Attach independently calibrated task-level confidence by transition key."""
    values = np.full(len(sidecar), np.nan, dtype=np.float32)
    for i, key in enumerate(sidecar.keys):
        value = scores.get(str(key))
        if value is not None and np.isfinite(value):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"q_perception for {key} lies outside [0,1]")
            values[i] = float(value)
    sidecar.q_perception = values
    sidecar.flags["q_perception_available"] = np.isfinite(values)
    sidecar.meta["perception_confidence"] = dict(metadata or {})
    return sidecar
