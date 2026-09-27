"""MongoDB model registry store plugin for MLflow."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow_mongodb.model_registry.store import MongoDBModelRegistryStore

__all__ = ["MongoDBModelRegistryStore"]
__version__ = "0.1.0.dev0"


def __getattr__(name: str):
    # MLflow loads tracking entry points while its tracking package is initializing.
    # Loading the registry store here would import MlflowClient before it is ready.
    if name == "MongoDBModelRegistryStore":
        return import_module("mlflow_mongodb.model_registry.store").MongoDBModelRegistryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
