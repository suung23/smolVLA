"""Filesystem helpers shared by every CLI command.

Two invariants the commands rely on:

* **Atomic writes.** A killed command never leaves a half-written manifest or
  sidecar that a later command would happily read.
* **Fingerprint-based idempotency.** Each artifact carries the fingerprint of
  the inputs that produced it. Re-running a command with the same inputs is a
  no-op; re-running it with changed inputs rewrites the artifact instead of
  appending to it. Nothing is ever duplicated or half-updated.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class JSONEncoder(json.JSONEncoder):
    """Encoder that understands numpy scalars/arrays and Path objects."""

    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        # Config objects are nested dataclasses (CameraSpec, SegmentationConfig,
        # ...) and are embedded verbatim in manifests and fingerprints.
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Write ``payload`` to ``path`` via a same-directory temp file + replace."""
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return p


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    return atomic_write_text(
        path, json.dumps(obj, indent=indent, ensure_ascii=False, cls=JSONEncoder) + "\n"
    )


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_npz(path: str | Path, **arrays: Any) -> Path:
    """Atomic compressed ``.npz`` write."""
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return atomic_write_bytes(path, buf.getvalue())


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    body = "".join(
        json.dumps(r, ensure_ascii=False, cls=JSONEncoder) + "\n" for r in rows
    )
    return atomic_write_text(path, body)


def read_jsonl(path: str | Path) -> list[Any]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fingerprint(obj: Any) -> str:
    """Stable short hash of any JSON-encodable structure.

    Dict key order is normalised so that an equivalent config always produces
    the same fingerprint.
    """
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, cls=JSONEncoder)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def file_digest(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def artifact_is_current(path: str | Path, expected_fingerprint: str) -> bool:
    """True when ``path`` exists and records ``expected_fingerprint``.

    Artifacts written by this project all carry a top-level ``fingerprint``
    field; that is what makes command re-runs cheap and safe.
    """
    data = read_json(path)
    if not isinstance(data, dict):
        return False
    return data.get("fingerprint") == expected_fingerprint


def relpath(path: str | Path, start: str | Path | None = None) -> str:
    """Repo-relative POSIX path for manifests (never an absolute user path)."""
    p = Path(path).resolve()
    base = Path(start).resolve() if start is not None else Path.cwd().resolve()
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        return p.as_posix()
