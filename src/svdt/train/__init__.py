"""Training entry points: the real SmolVLA path and a torch-free reference."""

from .reference import ReferenceTrainer, run_reference_training
from .smolvla import SmolVLATrainingPlan, build_training_plan, run_lerobot_training

__all__ = [
    "ReferenceTrainer",
    "SmolVLATrainingPlan",
    "build_training_plan",
    "run_lerobot_training",
    "run_reference_training",
]
