"""Run provenance: what code, what versions, what data, what seed.

Every command writes one of these next to its output so a result can be traced
back to the exact environment, dataset revision and checkpoint revision that
produced it.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Packages whose version materially changes results.
TRACKED_PACKAGES = (
    "svdt",
    "numpy",
    "scipy",
    "opencv-python-headless",
    "opencv-python",
    "torch",
    "torchvision",
    "lerobot",
    "transformers",
    "datasets",
    "huggingface-hub",
    "gymnasium",
    "gym-aloha",
)


@functools.lru_cache(maxsize=1)
def _package_versions_cached() -> tuple[tuple[str, str | None], ...]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return tuple(out.items())


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages that materially change results."""
    return dict(_package_versions_cached())


#: Modules whose import failure is expensive enough to cache on disk.
#: A blocked ``torch`` DLL load costs tens of seconds per attempt on a host with
#: Windows application control, and a failing import is *not* cached by Python's
#: import machinery -- so every command that writes a run manifest would pay it.
SLOW_PROBE_MODULES = frozenset({"torch", "lerobot", "transformers"})


def _probe_cache_path() -> Path:
    base = os.environ.get("SVDT_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".cache" / "svdt"
    return root / "env_probe.json"


def _probe_cache_key(module: str) -> str:
    """Invalidate the cached verdict when the install or interpreter changes."""
    try:
        installed = md.version(module)
    except md.PackageNotFoundError:
        installed = "absent"
    return f"{module}@{installed}@{sys.executable}@{platform.python_version()}"


def _read_probe_cache() -> dict[str, Any]:
    path = _probe_cache_path()
    if not path.exists():
        return {}
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_probe_cache(cache: dict[str, Any]) -> None:
    path = _probe_cache_path()
    try:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    except OSError:
        pass  # caching is an optimisation, never a requirement


def _do_probe(module: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(module)
    except BaseException as exc:  # noqa: BLE001 - we want the reason verbatim
        return {
            "importable": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:400],
        }
    return {"importable": True, "version": getattr(mod, "__version__", None)}


@functools.lru_cache(maxsize=None)
def _probe_import_cached(module: str) -> tuple[tuple[str, Any], ...]:
    use_disk = module in SLOW_PROBE_MODULES and not os.environ.get("SVDT_NO_PROBE_CACHE")
    key = _probe_cache_key(module)
    cache = _read_probe_cache() if use_disk else {}
    hit = cache.get(key)
    if isinstance(hit, dict) and "importable" in hit:
        return tuple({**hit, "probe_source": "cache"}.items())

    result = _do_probe(module)
    if use_disk:
        cache[key] = result
        _write_probe_cache(cache)
    return tuple({**result, "probe_source": "probe"}.items())


def probe_import(module: str) -> dict[str, Any]:
    """Try importing ``module`` and report the outcome.

    ``torch`` is installed-but-unimportable on some locked-down Windows hosts
    (application-control policy blocks its DLLs), so "installed" and "usable"
    are tracked separately and the failure text is preserved verbatim.

    The verdict for the expensive optional dependencies is cached on disk, keyed
    by module version plus interpreter, so a reinstall or a policy change is
    picked up while ordinary commands pay nothing. ``SVDT_NO_PROBE_CACHE=1``
    forces a fresh probe.
    """
    return dict(_probe_import_cached(module))


def clear_probe_cache() -> None:
    """Drop both cache layers; useful right after installing torch/lerobot."""
    _probe_import_cached.cache_clear()
    try:
        _probe_cache_path().unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        pass


def torch_runtime() -> dict[str, Any]:
    info = probe_import("torch")
    if not info.get("importable"):
        info["cuda_available"] = False
        return info
    import torch  # noqa: PLC0415 - only reachable when importable

    info["cuda_available"] = bool(torch.cuda.is_available())
    info["device_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
    if info["cuda_available"]:
        info["devices"] = [
            torch.cuda.get_device_name(i) for i in range(info["device_count"])
        ]
    return info


@functools.lru_cache(maxsize=4)
def git_info(root: str | Path | None = None) -> dict[str, Any]:
    """Commit, branch and dirty flag of the repo the command ran from."""
    cwd = str(Path(root) if root is not None else Path.cwd())

    def run(*args: str) -> str | None:
        try:
            res = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return res.stdout.strip() if res.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {"available": False}
    status = run("status", "--porcelain")
    return {
        "available": True,
        "commit": sha,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string without assuming torch is importable."""
    if requested != "auto":
        return requested
    info = probe_import("torch")
    if not info.get("importable"):
        return "cpu"
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect(
    *,
    command: str,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the run-manifest block. Secrets are never included."""
    return {
        "command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "packages": package_versions(),
        "torch": torch_runtime(),
        "lerobot": probe_import("lerobot"),
        "git": git_info(),
        "config": config,
        **({"extra": extra} if extra else {}),
    }


def summarize() -> str:
    """Human-readable one-screen capability report for ``svdt env``."""
    pkgs = package_versions()
    t = torch_runtime()
    lr = probe_import("lerobot")
    lines = [
        f"python            {sys.version.split()[0]}  ({platform.system()} {platform.release()} {platform.machine()})",
        f"numpy             {pkgs.get('numpy')}",
        f"opencv            {pkgs.get('opencv-python-headless') or pkgs.get('opencv-python')}",
        f"scipy             {pkgs.get('scipy')}",
        f"torch installed   {pkgs.get('torch')}",
        f"torch importable  {t.get('importable')}"
        + ("" if t.get("importable") else f"  <- {t.get('error_type')}: {str(t.get('error'))[:90]}"),
        f"cuda available    {t.get('cuda_available')}",
        f"lerobot           {pkgs.get('lerobot')} (importable={lr.get('importable')})",
        f"gym-aloha         {pkgs.get('gym-aloha')}",
    ]
    caps = []
    caps.append("scoring/corruption/weighting (numpy only): available")
    caps.append(
        "SmolVLA training: "
        + ("available" if (t.get("importable") and lr.get("importable")) else "UNAVAILABLE (needs importable torch + lerobot)")
    )
    return "\n".join(lines + ["", "capabilities:"] + [f"  - {c}" for c in caps])
