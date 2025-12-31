"""
Comparison framework for multi-agent training.
"""

from .shared_state import RenderingState, MetricsMessage, ControlEvents
from .worker import TrainingWorker, worker_process
from .coordinator import DualTrainingCoordinator
from .metrics_collector import MetricsCollector

__all__ = [
    "RenderingState",
    "MetricsMessage",
    "ControlEvents",
    "TrainingWorker",
    "worker_process",
    "DualTrainingCoordinator",
    "MetricsCollector",
]
