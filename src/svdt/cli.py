"""``svdt`` command line: one independently runnable command per stage.

    svdt env                      capability report for this machine
    svdt download                 fetch a dataset (dry-run by default)
    svdt inspect                  write outputs/dataset_manifest.json
    svdt split                    episode-level train/val/test split
    svdt corrupt                  generate corruption variants + manifest
    svdt score                    compute v_t / c_t and write a score sidecar
    svdt weights                  derive w_t and report the weight distribution
    svdt train                    SmolVLA fine-tuning (or the reference trainer)
    svdt eval                     offline metrics, score AUROC, rollout support
    svdt viz                      mask overlays and score timelines
    svdt check-leakage            prove no episode leaks across splits

Every command is idempotent: artifacts carry the fingerprint of the inputs that
produced them, so re-running is a no-op unless something actually changed, and
changed inputs rewrite rather than append. ``--force`` recomputes regardless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import __version__
from .config import Config, dump_config, load_config
from .env_info import collect, summarize
from .io_utils import ensure_dir, fingerprint, read_json, write_json

# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=[],
        metavar="PATH",
        help="YAML config file; repeatable, merged left to right",
    )
    parser.add_argument(
        "-o",
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="dotted override, e.g. -o weighting.alpha=0.7",
    )


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.overrides)


def _dataset_path(path: str, dataset_name: str) -> Path:
    return Path(path.format(dataset=dataset_name))


def _emit(payload: Any, *, pretty: bool = True) -> None:
    print(json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False, default=str))


def _episodes_arg(value: str | None) -> list[int] | None:
    """``0-5,9`` -> [0,1,2,3,4,5,9]; ``None`` means "whatever the config says"."""
    if not value:
        return None
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _make_source(
    cfg: Config,
    *,
    backend: str,
    episodes: Sequence[int] | None,
    max_frames: int | None = None,
    synthetic_episodes: int = 4,
    synthetic_length: int = 24,
):
    from .data.backends import make_frame_source

    return make_frame_source(
        cfg.dataset,
        backend=backend,
        episodes=episodes,
        max_frames_per_episode=max_frames,
        cache_dir=cfg.paths.cache_dir,
        synthetic_kwargs={
            "num_episodes": synthetic_episodes,
            "episode_length": synthetic_length,
        },
    )


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_env(args: argparse.Namespace) -> int:
    if args.refresh:
        from .env_info import clear_probe_cache

        clear_probe_cache()
    if args.json:
        _emit(collect(command="env"))
    else:
        print(summarize())
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .data.hub import (
        estimate_download_size,
        fetch_repo_metadata,
        human_bytes,
        snapshot,
        token_present,
    )

    cfg = _load(args)
    repo_id = args.repo_id or cfg.dataset.repo_id
    include_videos = not args.no_videos
    include_data = not args.metadata_only

    meta = fetch_repo_metadata(repo_id, revision=cfg.dataset.revision)
    size = estimate_download_size(
        repo_id,
        revision=cfg.dataset.revision,
        include_videos=include_videos,
        include_data=include_data,
    )
    plan = {
        "repo_id": repo_id,
        "requested_revision": cfg.dataset.revision or "main",
        "resolved_sha": meta.sha,
        "license": meta.license,
        "homepage": meta.homepage,
        "paper": meta.paper,
        "destination": args.local_dir or "huggingface cache",
        "mode": (
            "metadata_only"
            if args.metadata_only
            else ("sample" if args.sample else "full")
        ),
        "include_videos": include_videos,
        "selected_size": human_bytes(size["selected_bytes"]),
        "selected_bytes": size["selected_bytes"],
        "bytes_by_class": size["bytes_by_class"],
        "hf_token_present": token_present(),
    }

    if args.dry_run:
        plan["status"] = "dry_run"
        plan["note"] = "nothing downloaded; re-run without --dry-run to fetch"
        _emit(plan)
        return 0

    if (
        not args.yes
        and not args.metadata_only
        and size["selected_bytes"] > args.confirm_above_bytes
    ):
        plan["status"] = "refused"
        plan["reason"] = (
            f"{human_bytes(size['selected_bytes'])} exceeds the "
            f"{human_bytes(args.confirm_above_bytes)} auto-download limit. "
            "Re-run with --yes to confirm, or --metadata-only / --no-videos "
            "to fetch less."
        )
        _emit(plan)
        return 2

    patterns = ["meta/**"]
    if include_data:
        patterns.append("data/**")
    if include_videos and not args.metadata_only:
        patterns.append("videos/**")

    path = snapshot(
        repo_id,
        revision=cfg.dataset.revision,
        local_dir=args.local_dir,
        cache_dir=cfg.paths.cache_dir if args.local_dir is None else None,
        allow_patterns=patterns,
    )
    plan["status"] = "downloaded"
    plan["path"] = path
    plan["patterns"] = patterns
    out = Path(cfg.paths.manifest_dir) / f"download__{repo_id.replace('/', '__')}.json"
    write_json(out, {"kind": "download", "fingerprint": fingerprint(plan), **plan})
    plan["manifest"] = str(out)
    _emit(plan)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .data.hub import fetch_dataset_info_json, fetch_repo_metadata, fetch_tasks_best_effort
    from .data.schema import schema_from_info, state_action_index_map
    from .eval.rollout import check_rollout_support

    cfg = _load(args)
    repo_id = args.repo_id or cfg.dataset.repo_id
    dataset_cfg = cfg.dataset
    if args.repo_id:
        dataset_cfg.repo_id = repo_id

    meta = fetch_repo_metadata(repo_id, revision=dataset_cfg.revision)
    info = fetch_dataset_info_json(
        repo_id, revision=dataset_cfg.revision, cache_dir=cfg.paths.cache_dir
    )
    tasks, tasks_source = fetch_tasks_best_effort(
        repo_id, revision=dataset_cfg.revision, cache_dir=cfg.paths.cache_dir
    )
    schema = schema_from_info(
        info, dataset_cfg, revision=meta.sha, tasks=tasks, source="hub_metadata"
    )
    state_idx, action_idx = state_action_index_map(schema, dataset_cfg)

    record: dict[str, Any] = {
        "kind": "dataset_manifest",
        "fingerprint": fingerprint(
            {"repo_id": repo_id, "sha": meta.sha, "config": dataset_cfg.__dict__}
        ),
        "hub": meta.to_dict(),
        "tasks_source": tasks_source,
        "schema": schema.to_dict(),
        "state_to_action_index_map": {
            "state": state_idx,
            "action": action_idx,
            "source": (
                "config"
                if dataset_cfg.state_to_action_indices is not None
                else "identity_prefix"
            ),
        },
        "rollout": check_rollout_support(
            repo_id, env_id=cfg.eval.rollout_env_id
        ).to_dict(),
        "environment": collect(command="inspect", config={"dataset": dataset_cfg.__dict__}),
    }

    if args.with_episodes:
        # Episode lengths need the real files; only done on request because it
        # may trigger a download.
        source = _make_source(cfg, backend=args.backend, episodes=None)
        try:
            record["schema"]["episode_lengths"] = {
                str(e): source.episode_length(e) for e in source.episode_indices()
            }
            record["schema"]["source"] = source.schema.source
            if hasattr(source, "verify_alignment"):
                record["alignment_check"] = source.verify_alignment()
        finally:
            source.close()

    path = Path(args.out or Path(cfg.paths.manifest_dir) / "dataset_manifest.json")
    write_json(path, record)
    summary = {
        "manifest": str(path),
        "repo_id": repo_id,
        "resolved_sha": meta.sha,
        "codebase_version": schema.codebase_version,
        "fps": schema.fps,
        "total_episodes": schema.total_episodes,
        "total_frames": schema.total_frames,
        "cameras": schema.camera_keys,
        "state_dim": schema.state_dim,
        "action_dim": schema.action_dim,
        "tasks": schema.tasks,
        "license": meta.license,
        "rollout_status": record["rollout"]["rollout_success_rate_status"],
        "warnings": schema.warnings,
    }
    if "alignment_check" in record:
        summary["alignment_check"] = record["alignment_check"]
    _emit(summary)
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    from .data.splits import check_no_parent_overlap, make_split, save_split

    cfg = _load(args)
    episodes = _episodes_arg(args.episodes)
    if episodes is None:
        source = _make_source(cfg, backend=args.backend, episodes=None)
        try:
            episodes = source.episode_indices()
            dataset_id = source.dataset_id
        finally:
            source.close()
    else:
        from .ids import sanitize_dataset_id

        dataset_id = sanitize_dataset_id(cfg.dataset.name or cfg.dataset.repo_id)

    split = make_split(episodes, cfg.split, dataset_id=dataset_id)
    report = check_no_parent_overlap(split).raise_if_bad()
    path = save_split(
        split,
        cfg.paths.manifest_dir,
        extra={
            "n_candidate_episodes": len(episodes),
            "environment": collect(command="split", seed=cfg.split.seed),
        },
    )
    _emit(
        {
            "split_file": str(path),
            "dataset_id": dataset_id,
            "counts": split.counts(),
            "seed": split.seed,
            "fractions": split.fractions,
            "protected_test_episodes": split.protected_test[:20],
            "checks": report.checks,
            "note": (
                "corruption variants inherit these assignments, so an original "
                "and its copy can never land in different splits"
            ),
        }
    )
    return 0


def cmd_corrupt(args: argparse.Namespace) -> int:
    from .data.corrupt import build_variants
    from .data.splits import check_variant_split_consistency, load_split

    cfg = _load(args)
    if not cfg.corruptions:
        _emit(
            {
                "status": "nothing_to_do",
                "reason": "no corruption specs in the config; see configs/corruptions/",
            }
        )
        return 0

    split = load_split(cfg.paths.manifest_dir, cfg.dataset.name)
    eligible: list[int] = []
    for name in cfg.split.corrupt_splits:
        eligible.extend(split.episodes(name))
    eligible = sorted(set(eligible))

    source = _make_source(cfg, backend=args.backend, episodes=eligible)
    try:
        variants = build_variants(source, cfg.corruptions, episodes=eligible)
        records = [variant.record() for variant in variants]
        mapping = {r.dataset_id: r.episode_parents for r in records}
        variant_splits = {
            r.dataset_id: {ep: split.split_of(parent) for ep, parent in r.episode_parents.items()}
            for r in records
        }
        leakage = check_variant_split_consistency(
            split, mapping, variant_splits=variant_splits
        ).raise_if_bad()
    finally:
        source.close()

    payload = {
        "kind": "corruption_manifest",
        "fingerprint": fingerprint(
            {
                "dataset": cfg.dataset.repo_id,
                "specs": [c.__dict__ for c in cfg.corruptions],
                "split_seed": cfg.split.seed,
            }
        ),
        "parent_dataset_id": split.dataset_id,
        "eligible_splits": cfg.split.corrupt_splits,
        "n_eligible_episodes": len(eligible),
        "label_policy": (
            "affected_transitions is evaluation ground truth only and is never "
            "passed to any scorer"
        ),
        "leakage_check": leakage.checks,
        "variants": [r.to_dict() for r in records],
        "environment": collect(command="corrupt", config={"corruptions": [c.__dict__ for c in cfg.corruptions]}),
    }
    path = Path(cfg.paths.manifest_dir) / "corruption_manifest.json"
    write_json(path, payload)
    _emit(
        {
            "manifest": str(path),
            "variants": [
                {
                    "dataset_id": r.dataset_id,
                    "spec": r.spec_name,
                    "episodes": len(r.episode_parents),
                    "affected_transitions": len(r.affected_transitions),
                    "splits": leakage.checks["variants"][r.dataset_id]["splits"],
                }
                for r in records
            ],
            "leakage_violations": leakage.violations,
        }
    )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .data.corrupt import build_variants
    from .data.splits import load_split
    from .scoring.reliability import ReliabilityScorer, offset_report
    from .scoring.sidecar import build_sidecar, save_sidecar, sidecar_is_current
    from .scoring.visibility import VisibilityScorer, compensation_report
    from .vlm import annotation_summary, make_vlm_adapter, phases_from_annotation

    cfg = _load(args)
    episodes = _episodes_arg(args.episodes)
    if episodes is None and not args.all_episodes:
        try:
            split = load_split(cfg.paths.manifest_dir, cfg.dataset.name)
            episodes = sorted(split.episodes("train") + split.episodes("val"))
        except FileNotFoundError:
            episodes = None

    source = _make_source(
        cfg,
        backend=args.backend,
        episodes=episodes,
        max_frames=args.max_frames_per_episode,
        synthetic_episodes=args.synthetic_episodes,
        synthetic_length=args.synthetic_length,
    )
    sources = [source]
    try:
        if args.with_corruptions and cfg.corruptions:
            # Reuse the episodes `svdt corrupt` recorded so the scored variants
            # are the same ones the corruption manifest labelled.
            manifest = read_json(
                Path(cfg.paths.manifest_dir) / "corruption_manifest.json"
            )
            if not isinstance(manifest, dict):
                manifest = None
            sources.extend(
                build_variants(
                    source,
                    cfg.corruptions,
                    episodes=source.episode_indices(),
                    manifest=manifest,
                )
            )

        fp = fingerprint(
            {
                "dataset": cfg.dataset.__dict__,
                "visibility": cfg.visibility.__dict__ | {
                    "segmentation": cfg.visibility.segmentation.__dict__,
                    "image_quality": cfg.visibility.image_quality.__dict__,
                },
                "reliability": cfg.reliability.__dict__,
                "vlm": cfg.vlm.__dict__,
                "episodes": [s.episode_indices() for s in sources],
                "stride": args.stride,
                "max_frames": args.max_frames_per_episode,
                "variants": [c.__dict__ for c in (cfg.corruptions if args.with_corruptions else [])],
            }
        )
        target_id = args.sidecar_id or cfg.dataset.name
        if not args.force and sidecar_is_current(cfg.paths.sidecar_root, target_id, fp):
            _emit(
                {
                    "status": "up_to_date",
                    "sidecar": str(
                        Path(cfg.paths.sidecar_root) / f"scores__{target_id}.json"
                    ),
                    "note": "inputs unchanged; pass --force to recompute",
                }
            )
            return 0

        adapter = make_vlm_adapter(cfg.vlm)
        vis_results = []
        rel_results = []
        annotations: dict[int, Any] = {}

        vis_scorer = VisibilityScorer(cfg.visibility, source.schema)
        all_raws = []
        all_episodes = []
        for src in sources:
            eps = src.episode_indices()
            if not eps:
                continue
            if not args.skip_visibility:
                all_raws.extend(
                    vis_scorer.collect(
                        src,
                        eps,
                        batch_size=args.batch_size,
                        stride=args.stride,
                        max_frames_per_episode=args.max_frames_per_episode,
                    )
                )
            for ep in eps:
                low = src.read_lowdim(ep)
                all_episodes.append(low)
                if cfg.vlm.enabled:
                    annotations[ep] = adapter.annotate_episode(
                        src.dataset_id, ep, num_frames=low.num_frames, task=low.task
                    )

        if all_raws:
            vis_results = vis_scorer.score(all_raws)
        rel_scorer = ReliabilityScorer(cfg.reliability, source.schema, cfg.dataset)
        phases = {}
        for low in all_episodes:
            annotation = annotations.get(low.episode_index)
            if annotation is not None:
                phase, _ = phases_from_annotation(annotation, low.num_frames)
                phases[low.episode_index] = phase
        rel_results = rel_scorer.score(all_episodes, task_phase=phases or None)

        sidecar = build_sidecar(
            target_id,
            visibility=vis_results or None,
            reliability=rel_results,
            meta={
                "fingerprint": fp,
                "dataset": cfg.dataset.__dict__,
                "scored_dataset_ids": [s.dataset_id for s in sources],
                "stride": args.stride,
                "max_frames_per_episode": args.max_frames_per_episode,
                "backend": type(source).__name__,
                "schema_source": source.schema.source,
                "vlm": adapter.describe() | annotation_summary(annotations),
                "environment": collect(command="score", config=cfg.to_dict()),
            },
        )
        npz_path, json_path = save_sidecar(
            sidecar, cfg.paths.sidecar_root, input_fingerprint=fp
        )
        summary: dict[str, Any] = {
            "sidecar_arrays": str(npz_path),
            "sidecar_meta": str(json_path),
            **sidecar.summary(),
        }
        if vis_results:
            summary["compensation_report"] = compensation_report(vis_results)
        summary["offset_report"] = offset_report(rel_results)
        summary["reliability_warnings"] = rel_results.meta.get("warnings", [])
        _emit(summary)
        return 0
    finally:
        source.close()


def cmd_weights(args: argparse.Namespace) -> int:
    from .scoring.sidecar import load_sidecar
    from .weighting.weights import compute_weights, weight_report

    cfg = _load(args)
    sidecar = load_sidecar(cfg.paths.sidecar_root, args.sidecar_id or cfg.dataset.name)
    result = compute_weights(
        cfg.weighting,
        keys=sidecar.keys,
        v=sidecar.v,
        c=sidecar.c,
        q_perception=sidecar.q_perception,
    )
    report = weight_report(result)
    payload = {
        "kind": "weight_report",
        "fingerprint": fingerprint(
            {"weighting": cfg.weighting.__dict__, "sidecar": sidecar.meta.get("fingerprint")}
        ),
        "sidecar_dataset_id": sidecar.dataset_id,
        "report": report,
        "distributions": {
            name: sidecar.distribution(name)
            for name in ("v", "c", "q_perception")
            if sidecar.distribution(name)
        },
        "environment": collect(command="weights", config={"weighting": cfg.weighting.__dict__}),
    }
    path = (
        Path(cfg.paths.manifest_dir)
        / f"weights__{sidecar.dataset_id}__{cfg.weighting.method}.json"
    )
    write_json(path, payload)
    _emit({"report_file": str(path), **report})
    return 0


def cmd_calibrate_confidence(args: argparse.Namespace) -> int:
    from .data.splits import load_split
    from .eval.confidence import evaluate_test_confidence, write_reliability_svg
    from .scoring.confidence import (
        aggregate_by_transition,
        apply_calibrators,
        fit_calibrators,
        load_observations_csv,
        observations_from_score_sidecar,
        write_observations,
    )
    from .scoring.sidecar import (
        attach_perception_scores,
        load_sidecar,
        save_sidecar,
    )

    cfg = _load(args)
    base = load_sidecar(cfg.paths.sidecar_root, args.sidecar_id or cfg.dataset.name)
    split = load_split(cfg.paths.manifest_dir, cfg.dataset.name)
    prediction_path = Path(args.predictions) if args.predictions else _dataset_path(
        cfg.confidence.predictions_path, cfg.dataset.name
    )
    if args.predictions or prediction_path.exists():
        label_path = Path(args.labels) if args.labels else _dataset_path(
            cfg.confidence.labels_path, cfg.dataset.name
        )
        if not label_path.exists():
            raise FileNotFoundError(f"label CSV not found: {label_path}")
        observations = load_observations_csv(prediction_path, label_path)
    else:
        observations = observations_from_score_sidecar(
            base,
            cameras=[camera.key for camera in cfg.dataset.cameras],
            entities=cfg.confidence.entities,
            model_id=args.model_id or (cfg.confidence.model_id if cfg.confidence.model_id != "unknown" else str(
                base.meta.get("v_meta", {}).get("segmentation", {}).get(
                    "segmenter",
                    base.meta.get("v_meta", {}).get("segmentation", {}).get("backend", "unknown"),
                )
            )),
            model_version=args.model_version or cfg.confidence.model_version,
            fps=cfg.dataset.fps,
            stride=int(base.meta.get("stride", 1)),
        )
        label_path = Path(args.labels) if args.labels else _dataset_path(
            cfg.confidence.labels_path, cfg.dataset.name
        )
        if not label_path.exists():
            raise FileNotFoundError(
                f"label CSV not found: {label_path}. Provide reviewed quality_label values; "
                "raw model scores are not treated as calibrated confidence."
            )
        # Join labels against the already extracted per-camera raw scores.
        labels_only = load_observations_csv(label_path, label_path)
        label_lookup = {
            (r.key, r.entity_id, r.output_type, r.camera_id): r
            for r in labels_only
        }
        for row in observations:
            label = label_lookup.get((row.key, row.entity_id, row.output_type, row.camera_id))
            if label is not None:
                row.quality_label = label.quality_label
                row.label_source = label.label_source

    validation = set(split.episodes("val"))
    if not validation:
        raise ValueError("confidence calibration requires validation episodes")
    calibrators = fit_calibrators(
        observations,
        validation_episodes=validation,
        method=cfg.confidence.calibrator,
        min_samples=cfg.confidence.min_samples,
        min_per_class=cfg.confidence.min_per_class,
    )
    split_map = {int(ep): name for ep, name in split.assignment.items()}
    observations = apply_calibrators(observations, calibrators, episode_splits=split_map)
    observation_path = Path(args.out) if args.out else _dataset_path(
        cfg.confidence.observations_path, cfg.dataset.name
    )
    write_observations(observation_path, observations)

    entity_weights = {e.entity_id: e.importance for e in cfg.confidence.entities}
    critical = [e.entity_id for e in cfg.confidence.entities if e.critical]
    scores = aggregate_by_transition(
        observations,
        entity_weights=entity_weights or None,
        critical_entities=critical,
        camera_aggregation=cfg.confidence.camera_aggregation,
        expected_entities=entity_weights,
    )
    cal_metadata = {
        "method": cfg.confidence.calibrator,
        "fit_split": "val",
        "fit_episodes": sorted(validation),
        "calibrators": [c.to_dict() for c in calibrators.values()],
        "n_observations": len(observations),
        "n_raw_missing": sum(r.raw_confidence is None for r in observations),
        "n_calibrated": sum(r.calibrated_confidence is not None for r in observations),
        "observation_path": str(observation_path),
        "label_path": str(label_path),
        "camera_aggregation": cfg.confidence.camera_aggregation,
        "entity_specs": [e.__dict__ for e in cfg.confidence.entities],
    }
    cal_metadata["calibration_fingerprint"] = fingerprint(
        {"base": base.meta.get("fingerprint"), "calibration": cal_metadata}
    )
    attach_perception_scores(base, scores, metadata=cal_metadata)
    save_sidecar(
        base,
        cfg.paths.sidecar_root,
        input_fingerprint=str(base.meta.get("fingerprint", "")) or None,
    )

    report = evaluate_test_confidence(observations)
    report["calibrators_fitted"] = len(calibrators)
    report["calibration_fit_split"] = "val"
    report["raw_missing_policy"] = "exclude_from_calibrated_confidence; retain missing_reason"
    report_path = Path(cfg.eval.output_dir) / f"confidence__{cfg.dataset.name}.json"
    write_json(report_path, report)
    diagram_path = Path(cfg.eval.output_dir) / f"confidence_reliability__{cfg.dataset.name}.svg"
    write_reliability_svg(report, diagram_path)
    _emit({
        "status": "ok" if calibrators else "no_calibrator_fitted",
        "calibrators_fitted": len(calibrators),
        "confidence_observations": str(observation_path),
        "evaluation_report": str(report_path),
        "reliability_diagram": str(diagram_path),
        "q_perception_available": int(np.isfinite(base.q_perception).sum()),
        "n_raw_missing": cal_metadata["n_raw_missing"],
    })
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .data.splits import load_split, training_episodes, tuning_episodes
    from .scoring.sidecar import load_sidecar
    from .train.reference import run_reference_training
    from .train.smolvla import TrainingUnavailable, run_lerobot_training

    cfg = _load(args)
    if args.steps is not None:
        cfg.train.steps = args.steps
    if args.method:
        cfg.weighting.method = args.method
        cfg.weighting.validate()

    try:
        split = load_split(cfg.paths.manifest_dir, cfg.dataset.name)
        train_eps = training_episodes(split)
        val_eps = tuning_episodes(split)
    except FileNotFoundError:
        train_eps, val_eps = _episodes_arg(args.episodes) or [0], []

    backend = args.train_backend or cfg.train.backend
    if backend == "reference":
        source = _make_source(
            cfg,
            backend=args.backend,
            episodes=train_eps,
            synthetic_episodes=args.synthetic_episodes,
            synthetic_length=args.synthetic_length,
        )
        try:
            sidecar = None
            if cfg.weighting.method != "uniform":
                sidecar = load_sidecar(
                    cfg.paths.sidecar_root, args.sidecar_id or cfg.dataset.name
                )
            result = run_reference_training(
                cfg,
                source,
                source.episode_indices(),
                sidecar=sidecar,
                chunk_size=args.chunk_size,
                steps=cfg.train.steps,
            )
        finally:
            source.close()
        _emit(
            {
                k: v
                for k, v in result.items()
                if k not in {"history", "environment", "weighting_stats"}
            }
            | {"final_history": result["history"][-1] if result["history"] else None}
        )
        return 0

    try:
        result = run_lerobot_training(
            cfg,
            train_episodes=train_eps,
            val_episodes=val_eps,
            sidecar_root=cfg.paths.sidecar_root,
            sidecar_dataset_id=args.sidecar_id,
            dry_run=args.dry_run,
        )
    except TrainingUnavailable as exc:
        _emit({"status": "unavailable", "reason": str(exc)})
        return 3
    _emit({k: v for k, v in result.items() if k != "history"})
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval.rollout import summarize_support
    from .eval.confidence import evaluate_test_confidence
    from .eval.score_eval import (
        component_contribution,
        evaluate_scores,
        hard_filter_tradeoff,
    )
    from .scoring.sidecar import load_sidecar

    cfg = _load(args)
    sidecar = load_sidecar(cfg.paths.sidecar_root, args.sidecar_id or cfg.dataset.name)

    labels: dict[str, bool] = {}
    per_condition: dict[str, list[str]] = {}
    manifest = read_json(Path(cfg.paths.manifest_dir) / "corruption_manifest.json")
    if isinstance(manifest, dict):
        for variant in manifest.get("variants", []):
            keys = variant.get("affected_transitions", [])
            per_condition[variant["spec_name"]] = keys
            for key in keys:
                labels[key] = True

    payload: dict[str, Any] = {
        "kind": "evaluation",
        "sidecar_dataset_id": sidecar.dataset_id,
        "n_transitions": len(sidecar),
        "rollout_support": summarize_support(
            {cfg.dataset.repo_id: cfg.eval.rollout_env_id}
        ),
        "environment": collect(command="eval", config=cfg.to_dict()),
    }
    if labels:
        payload["score_evaluation"] = evaluate_scores(
            sidecar,
            labels,
            weighting=cfg.weighting,
            per_condition=per_condition or None,
        )
        payload["hard_filter_tradeoff"] = hard_filter_tradeoff(
            sidecar,
            labels,
            alpha=cfg.weighting.effective_alpha(),
            weighting=cfg.weighting,
        )
        payload["visibility_component_contribution"] = component_contribution(
            sidecar, labels, lambdas=cfg.visibility.lambdas(), prefix="q_"
        )
        payload["reliability_component_contribution"] = component_contribution(
            sidecar, labels, lambdas=cfg.reliability.weights(), prefix="c_"
        )
    else:
        payload["score_evaluation"] = {
            "status": "no_corruption_labels",
            "reason": (
                "outputs/corruption_manifest.json not found or empty; run "
                "`svdt corrupt` and `svdt score --with-corruptions` to evaluate "
                "score quality against injected corruption"
            ),
        }

    observations_path = _dataset_path(cfg.confidence.observations_path, cfg.dataset.name)
    if observations_path.exists():
        from .scoring.confidence import read_observations

        payload["perception_confidence"] = evaluate_test_confidence(
            read_observations(observations_path)
        )

    path = Path(cfg.eval.output_dir) / f"eval__{sidecar.dataset_id}.json"
    write_json(path, payload)
    brief: dict[str, Any] = {"report": str(path), "n_transitions": len(sidecar)}
    scores = payload.get("score_evaluation", {}).get("scores")
    if scores:
        brief["auroc"] = {k: v.get("auroc") for k, v in scores.items()}
        brief["n_corrupted"] = payload["score_evaluation"]["n_corrupted"]
    brief["rollout_support"] = {
        k: v["rollout_success_rate_status"] for k, v in payload["rollout_support"].items()
    }
    _emit(brief)
    return 0


def cmd_viz(args: argparse.Namespace) -> int:
    from .scoring.segmentation import make_segmentation_backend
    from .scoring.sidecar import load_sidecar
    from .viz.overlay import (
        save_episode_contact_sheet,
        save_score_timeline,
        series_from_sidecar,
    )

    cfg = _load(args)
    out_dir = ensure_dir(Path(args.out or Path(cfg.paths.outputs_root) / "viz"))
    source = _make_source(
        cfg,
        backend=args.backend,
        episodes=[args.episode],
        synthetic_episodes=args.synthetic_episodes,
        synthetic_length=args.synthetic_length,
    )
    written: list[str] = []
    try:
        backend = make_segmentation_backend(
            cfg.visibility.segmentation, dataset_id=source.dataset_id
        )
        scores: dict[str, dict[int, float]] = {}
        sidecar = None
        try:
            sidecar = load_sidecar(
                cfg.paths.sidecar_root, args.sidecar_id or cfg.dataset.name
            )
        except FileNotFoundError:
            pass
        if sidecar is not None:
            frames, series = series_from_sidecar(sidecar, args.episode)
            for name in ("v", "c"):
                if name in series:
                    scores[name] = {
                        int(f): float(x) for f, x in zip(frames, series[name])
                    }
            timeline = save_score_timeline(
                out_dir / f"timeline__{source.dataset_id}__ep{args.episode:06d}.png",
                frames,
                series,
                title=f"{source.dataset_id} episode {args.episode}",
            )
            written.append(str(timeline))

        sheet = save_episode_contact_sheet(
            source,
            args.episode,
            backend,
            out_dir / f"overlay__{source.dataset_id}__ep{args.episode:06d}.png",
            scores=scores or None,
            n_tiles=args.tiles,
        )
        written.append(str(sheet))
    finally:
        source.close()
    _emit({"written": written, "episode": args.episode})
    return 0


def cmd_check_leakage(args: argparse.Namespace) -> int:
    from .data.corrupt import parent_of
    from .data.splits import (
        check_no_parent_overlap,
        check_variant_split_consistency,
        load_split,
    )

    cfg = _load(args)
    split = load_split(cfg.paths.manifest_dir, cfg.dataset.name)
    checks: dict[str, Any] = {"parent_overlap": check_no_parent_overlap(split).__dict__}

    manifest = read_json(Path(cfg.paths.manifest_dir) / "corruption_manifest.json")
    violations: list[str] = list(checks["parent_overlap"]["violations"])
    if isinstance(manifest, dict):
        mapping = {
            v["dataset_id"]: {int(k): int(p) for k, p in v["episode_parents"].items()}
            for v in manifest.get("variants", [])
        }
        variant_splits = {
            did: {ep: split.split_of(parent) for ep, parent in parents.items()}
            for did, parents in mapping.items()
        }
        report = check_variant_split_consistency(
            split, mapping, variant_splits=variant_splits
        )
        checks["variant_consistency"] = report.checks
        violations.extend(report.violations)

        # Independently re-derive each variant transition's parent from its key
        # and confirm the split agrees -- a check that does not trust the
        # manifest's own episode_parents field.
        mismatches = []
        for variant in manifest.get("variants", []):
            for key in variant.get("affected_transitions", [])[: args.sample_keys]:
                parent = parent_of(key)
                if parent.dataset_id != split.dataset_id:
                    mismatches.append(
                        f"{key}: parsed parent dataset {parent.dataset_id!r} != "
                        f"{split.dataset_id!r}"
                    )
                    continue
                try:
                    split.split_of(parent.episode_index)
                except KeyError:
                    mismatches.append(f"{key}: parent episode has no split assignment")
        checks["key_derived_parent_check"] = {
            "sampled_keys_per_variant": args.sample_keys,
            "mismatches": len(mismatches),
        }
        violations.extend(mismatches[:10])
    else:
        checks["variant_consistency"] = {"status": "no_corruption_manifest"}

    ok = not violations
    _emit(
        {
            "ok": ok,
            "violations": violations,
            "checks": checks,
            "clean_test_episodes_protected": len(split.protected_test),
        }
    )
    return 0 if ok else 1


def cmd_dump_config(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if args.out:
        path = dump_config(cfg, args.out)
        _emit({"written": str(path)})
    else:
        _emit(cfg.to_dict())
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="svdt",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"svdt {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("env", help="report what this machine can run")
    p.add_argument("--json", action="store_true", help="full machine-readable report")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="re-probe torch/lerobot instead of using the cached verdict "
        "(use after installing or fixing them)",
    )
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("download", help="fetch a dataset (dry-run by default)")
    _add_config_args(p)
    p.add_argument("--repo-id", help="override dataset.repo_id")
    p.add_argument("--local-dir", help="download into this directory instead of the HF cache")
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    p.add_argument("--metadata-only", action="store_true", help="meta/ only: schema and task")
    p.add_argument("--sample", action="store_true", help="label this run as a sample fetch")
    p.add_argument("--no-videos", action="store_true", help="skip videos/")
    p.add_argument("--yes", action="store_true", help="confirm a large download")
    p.add_argument(
        "--confirm-above-bytes",
        type=int,
        default=1_000_000_000,
        help="refuse downloads larger than this without --yes (default 1 GB)",
    )
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("inspect", help="write the dataset manifest")
    _add_config_args(p)
    p.add_argument("--repo-id")
    p.add_argument("--out", help="manifest path (default outputs/dataset_manifest.json)")
    p.add_argument("--with-episodes", action="store_true", help="also read episode lengths")
    p.add_argument("--backend", default="auto", help="lerobot|local|hub_rows|synthetic|auto")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("split", help="episode-level train/val/test split")
    _add_config_args(p)
    p.add_argument("--episodes", help="episode list, e.g. 0-99")
    p.add_argument("--backend", default="auto")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("corrupt", help="generate corruption variants and the manifest")
    _add_config_args(p)
    p.add_argument("--backend", default="auto")
    p.set_defaults(func=cmd_corrupt)

    p = sub.add_parser("score", help="compute v_t and c_t into a score sidecar")
    _add_config_args(p)
    p.add_argument("--episodes", help="episode list; default = train+val from the split")
    p.add_argument("--all-episodes", action="store_true")
    p.add_argument("--backend", default="auto")
    p.add_argument("--stride", type=int, default=1, help="frame stride for visibility")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-frames-per-episode", type=int, default=None)
    p.add_argument("--with-corruptions", action="store_true", help="also score the variants")
    p.add_argument("--skip-visibility", action="store_true", help="reliability only")
    p.add_argument("--sidecar-id", help="sidecar name (default dataset.name)")
    p.add_argument("--force", action="store_true", help="recompute even if up to date")
    p.add_argument("--synthetic-episodes", type=int, default=4)
    p.add_argument("--synthetic-length", type=int, default=24)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("weights", help="derive w_t and report its distribution")
    _add_config_args(p)
    p.add_argument("--sidecar-id")
    p.set_defaults(func=cmd_weights)

    p = sub.add_parser(
        "confidence-calibrate",
        help="calibrate raw perception scores on validation and evaluate held-out test",
    )
    _add_config_args(p)
    p.add_argument("--predictions", help="optional entity-level prediction CSV")
    p.add_argument("--labels", help="reviewed labels CSV; defaults to confidence.labels_path")
    p.add_argument("--model-id", help="model ID for scores extracted from the score sidecar")
    p.add_argument("--model-version", help="model version for extracted scores")
    p.add_argument("--sidecar-id")
    p.add_argument("--out", help="observation JSONL output path")
    p.set_defaults(func=cmd_calibrate_confidence)

    p = sub.add_parser("train", help="fine-tune SmolVLA, or run the reference trainer")
    _add_config_args(p)
    p.add_argument("--train-backend", choices=["lerobot", "reference"], default=None)
    p.add_argument("--backend", default="auto", help="dataset backend")
    p.add_argument("--method", help="override weighting.method")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=8, help="reference trainer chunk size")
    p.add_argument("--episodes", help="used when no split file exists")
    p.add_argument("--sidecar-id")
    p.add_argument("--dry-run", action="store_true", help="resolve and record the plan only")
    p.add_argument("--synthetic-episodes", type=int, default=4)
    p.add_argument("--synthetic-length", type=int, default=24)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="offline metrics and score quality")
    _add_config_args(p)
    p.add_argument("--sidecar-id")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("viz", help="mask overlays and score timelines")
    _add_config_args(p)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--backend", default="auto")
    p.add_argument("--tiles", type=int, default=12)
    p.add_argument("--out")
    p.add_argument("--sidecar-id")
    p.add_argument("--synthetic-episodes", type=int, default=4)
    p.add_argument("--synthetic-length", type=int, default=24)
    p.set_defaults(func=cmd_viz)

    p = sub.add_parser("check-leakage", help="prove no episode leaks across splits")
    _add_config_args(p)
    p.add_argument("--sample-keys", type=int, default=200)
    p.set_defaults(func=cmd_check_leakage)

    p = sub.add_parser("dump-config", help="print the fully resolved config")
    _add_config_args(p)
    p.add_argument("--out")
    p.set_defaults(func=cmd_dump_config)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        if "--traceback" in (argv or sys.argv):  # pragma: no cover
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
