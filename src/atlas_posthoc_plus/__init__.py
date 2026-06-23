"""Project-specific workflow orchestration built on :mod:`atlas_posthoc`."""

from atlas_posthoc_plus.config import ProjectConfig, load_config
from atlas_posthoc_plus.registry import SUITES, WORKFLOWS, WorkflowSpec

__all__ = [
    "ProjectConfig",
    "WorkflowSpec",
    "SUITES",
    "WORKFLOWS",
    "load_config",
]
__version__ = "0.1.0"
