"""Shared fixtures. Nothing here needs torch, lerobot, a GPU or the network."""

from __future__ import annotations

import numpy as np
import pytest

from svdt.config import (
    CameraSpec,
    Config,
    CorruptionSpec,
    DatasetConfig,
    SegmentationConfig,
)
from svdt.data.backends import SyntheticFrameSource


@pytest.fixture
def dataset_cfg() -> DatasetConfig:
    return DatasetConfig(
        repo_id="local/synthetic",
        name="synthetic",
        fps=10.0,
        cameras=[CameraSpec(key="observation.image", role="scene")],
        action_semantics="absolute_position_target",
    )


@pytest.fixture
def source(dataset_cfg: DatasetConfig) -> SyntheticFrameSource:
    return SyntheticFrameSource(
        dataset_cfg, num_episodes=4, episode_length=20, image_size=96
    )


@pytest.fixture
def config(dataset_cfg: DatasetConfig) -> Config:
    cfg = Config()
    cfg.dataset = dataset_cfg
    cfg.visibility.segmentation = SegmentationConfig(backend="null")
    cfg.split.train, cfg.split.val, cfg.split.test = 0.5, 0.25, 0.25
    cfg.train.steps = 6
    cfg.train.batch_size = 4
    cfg.train.backend = "reference"
    return cfg.validate()


@pytest.fixture
def corruption_specs() -> list[CorruptionSpec]:
    return [
        CorruptionSpec(
            name="outlier",
            ops=[{"type": "action_outlier", "magnitude": 8.0, "fraction": 0.3}],
            frame_fraction=1.0,
            episode_fraction=0.5,
            seed=3,
        ),
        CorruptionSpec(
            name="blur",
            ops=[{"type": "blur", "sigma": 3.0}],
            frame_fraction=0.5,
            episode_fraction=0.5,
            seed=4,
        ),
    ]


@pytest.fixture
def chunk_batch() -> dict[str, np.ndarray]:
    """A small action-chunk batch with a realistic, uneven padding mask."""
    rng = np.random.default_rng(0)
    b, h, a = 6, 5, 3
    pad = np.zeros((b, h), dtype=bool)
    pad[0, 3:] = True  # near the episode end
    pad[2, 1:] = True  # one valid step only
    pad[5, :] = True  # entirely padding
    return {
        "pred": rng.normal(size=(b, h, a)),
        "target": rng.normal(size=(b, h, a)),
        "pad": pad,
    }


def torch_is_usable() -> bool:
    """Whether torch can actually be imported here.

    ``pytest.importorskip`` only catches ``ImportError``; a torch whose DLLs are
    blocked by Windows application control raises ``OSError``, which would turn
    an environment limitation into a test failure.
    """
    from svdt.env_info import probe_import

    return bool(probe_import("torch").get("importable"))


requires_torch = pytest.mark.skipif(
    not torch_is_usable(), reason="torch is not importable in this environment"
)


def lerobot_is_usable() -> bool:
    from svdt.env_info import probe_import

    return bool(probe_import("lerobot").get("importable"))


requires_lerobot = pytest.mark.skipif(
    not lerobot_is_usable(), reason="lerobot is not installed in this environment"
)
