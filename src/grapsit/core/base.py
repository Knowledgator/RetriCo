"""Base classes for processors and components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base configuration for any component."""
    pass


class BaseComponent(ABC):
    """Base class for stateful components (models, DB connections)."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component (load model, connect, etc.)."""
        ...


class BaseProcessor(ABC):
    """Base class for DAG pipeline processors.

    A processor wraps a component (or standalone logic) and
    exposes a ``__call__`` interface consumed by ``DAGExecutor``.
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        self.config_dict = config_dict
        self.pipeline = pipeline

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        ...
