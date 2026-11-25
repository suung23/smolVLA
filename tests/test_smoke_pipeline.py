"""End-to-end smoke test over the CLI, with no torch, GPU, checkpoint or network.

Runs the real commands in order -- split, corrupt, score, weights, train, eval,
check-leakage, viz -- on the synthetic source, and asserts the
properties the project's completion criteria name:

* a score sidecar is produced for a small amount of data,
* re-running a command is a no-op rather than a duplicate,
* no episode leaks across splits after corruption,
* every weighting condition runs, and uniform really is unweighted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from svdt.cli import main
from svdt.io_utils import read_json
from svdt.scoring.sidecar import load_sidecar


def run(capsys, *args: str) -> dict:
    """Invoke the CLI and return its JSON payload."""
    code = main(list(args))
    out = capsys.readouterr().out
    assert code == 0, f"command failed ({code}): {' '.join(args)}\n{out}"
    return json.loads(out)


@pytest.fixture
def workspace(tmp_path, config):
    """A config file plus output overrides pointing into a temp directory."""
    from svdt.config import dump_config

    config.corruptions = [
        __import__("svdt.config", fromlist=["CorruptionSpec"]).CorruptionSpec(
            name="outlier",
            ops=[{"type": "action_outlier", "magnitude": 10.0, "fraction": 0.25}],
            frame_fraction=1.0,
            episode_fraction=0.5,
            seed=1,
        ),
        __import__("svdt.config", fromlist=["CorruptionSpec"]).CorruptionSpec(
            name="blur",
            ops=[{"type": "blur", "sigma": 3.0}],
            frame_fraction=0.5,
            episode_fraction=0.5,
            seed=2,
        ),
    ]
    # Pinned so the plan can be resolved offline; `svdt inspect` reads the real
    # value (50 for lerobot/smolvla_base) from the checkpoint config.
    config.policy.chunk_size = 50
    config.paths.manifest_dir = str(tmp_path)
    config.paths.sidecar_root = str(tmp_path / "sidecars")
    config.paths.cache_dir = str(tmp_path / "cache")
    config.train.output_dir = str(tmp_path / "train")
    config.eval.output_dir = str(tmp_path / "eval")
    path = dump_config(config, tmp_path / "config.yaml")
    return {"config": str(path), "root": tmp_path}


def test_full_offline_pipeline(workspace, capsys):
    cfg = ["-c", workspace["config"]]
    root = Path(workspace["root"])

    # 1. episode-level split, before any corruption exists
    split = run(capsys, "split", *cfg, "--backend", "synthetic")
    assert sum(split["counts"].values()) == 4
    assert split["checks"]["counts"] == split["counts"]

    # 2. corruption copies, which inherit their parent's split
    corrupt = run(capsys, "corrupt", *cfg, "--backend", "synthetic")
    assert corrupt["leakage_violations"] == []
    assert len(corrupt["variants"]) == 2
    assert all(v["affected_transitions"] > 0 for v in corrupt["variants"])

    # 3. scores for clean + corrupted transitions
    score = run(
        capsys,
        "score",
        *cfg,
        "--backend",
        "synthetic",
        "--all-episodes",
        "--with-corruptions",
    )
    assert score["n_transitions"] > 0
    assert "v" in score["distributions"] and "c" in score["distributions"]
    sidecar = load_sidecar(root / "sidecars", "synthetic")
    assert len(sidecar) == score["n_transitions"]
    assert np.isfinite(sidecar.v).any() and np.isfinite(sidecar.c).any()
    assert len(set(sidecar.keys)) == len(sidecar), "transition identities must be unique"

    # 4. re-running is a no-op, not a duplicate
    again = run(
        capsys,
        "score",
        *cfg,
        "--backend",
        "synthetic",
        "--all-episodes",
        "--with-corruptions",
    )
    assert again["status"] == "up_to_date"
    assert len(load_sidecar(root / "sidecars", "synthetic")) == len(sidecar)

    # 5. every weighting condition produces a report
    reports = {}
    for method in (
        "uniform",
        "visibility_only",
        "reliability_only",
        "combined",
        "hard_filter",
    ):
        reports[method] = run(capsys, "weights", *cfg, "-o", f"weighting.method={method}")
        assert reports[method]["method"] == method
        assert reports[method]["n_transitions_scored"] == len(sidecar)
    assert reports["uniform"]["weight_mean"] == 1.0
    assert reports["combined"]["weight_mean"] < 1.0
    assert reports["hard_filter"]["retained_fraction"] <= 1.0

    # 6. training runs for the uniform control and the proposed method
    uniform_run = run(
        capsys,
        "train",
        *cfg,
        "--backend",
        "synthetic",
        "--train-backend",
        "reference",
        "--method",
        "uniform",
        "--steps",
        "6",
    )
    assert uniform_run["mean_batch_weight"] == 1.0
    combined_run = run(
        capsys,
        "train",
        *cfg,
        "--backend",
        "synthetic",
        "--train-backend",
        "reference",
        "--method",
        "combined",
        "--steps",
        "6",
    )
    assert 0.0 < combined_run["mean_batch_weight"] <= 1.0
    assert combined_run["final_loss"] < combined_run["first_loss"]

    # 7. evaluation against the injected corruption labels
    evaluation = run(capsys, "eval", *cfg)
    assert evaluation["n_corrupted"] > 0
    assert set(evaluation["auroc"]) >= {"v", "c", "w"}
    report = read_json(Path(evaluation["report"]))
    assert report["score_evaluation"]["label_source"].startswith("injected")
    assert "per_condition" in report["score_evaluation"]

    # 8. the leakage guarantee, re-derived from the transition keys
    leakage = run(capsys, "check-leakage", *cfg)
    assert leakage["ok"] is True
    assert leakage["violations"] == []
    assert leakage["checks"]["key_derived_parent_check"]["mismatches"] == 0

    # 9. visual artifacts
    viz = run(capsys, "viz", *cfg, "--backend", "synthetic", "--episode", "0")
    assert viz["written"]
    for path in viz["written"]:
        assert Path(path).exists() and Path(path).stat().st_size > 0


def test_train_records_a_plan_even_without_torch(workspace, capsys):
    """A machine that cannot train still produces the exact, reviewable plan."""
    cfg = ["-c", workspace["config"]]
    run(capsys, "split", *cfg, "--backend", "synthetic")
    plan = run(
        capsys,
        "train",
        *cfg,
        "--train-backend",
        "lerobot",
        "--method",
        "uniform",
        "--dry-run",
    )
    assert plan["status"] == "dry_run"
    assert plan["plan"]["policy"]["pretrained_path"] == "lerobot/smolvla_base"
    assert plan["plan"]["weighting"]["uses_weighter"] is False
    written = read_json(Path(plan["output_dir"]) / "train_plan.json")
    assert "requirements" in written
    assert "environment" in written


@pytest.mark.needs_network
def test_chunk_size_is_read_from_the_real_checkpoint():
    """The chunk size belongs to the checkpoint, not to a constant in our code."""
    from svdt.train.smolvla import read_checkpoint_chunk_size

    assert read_checkpoint_chunk_size("lerobot/smolvla_base") == 50


def test_env_command_reports_capabilities(capsys):
    assert main(["env"]) == 0
    text = capsys.readouterr().out
    assert "Cabinet rollout: not_available" in text
    assert "scoring/corruption/weighting" in text


def test_cli_errors_are_reported_as_json(tmp_path, capsys):
    code = main(["weights", "-c", str(tmp_path / "missing.yaml")])
    assert code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "error"
    assert payload["error_type"] == "FileNotFoundError"
