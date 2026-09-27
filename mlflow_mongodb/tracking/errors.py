"""Tracking domain exceptions and repository database-error translation."""

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from bson.errors import BSONError
from pymongo.errors import PyMongoError


class RepositoryPersistenceError(Exception):
    """Raised when a repository database operation fails."""


class ExperimentAlreadyExistsError(Exception):
    """Raised when an experiment name is already stored."""


class ExperimentNotFoundError(Exception):
    """Raised when an experiment is not stored in the expected lifecycle stage."""


class RunAlreadyExistsError(Exception):
    """Raised when a run ID is already stored."""


class RunNotFoundError(Exception):
    """Raised when a run is not stored in the expected lifecycle stage."""


class RunInactiveError(Exception):
    """Raised when input or output logging targets an inactive run."""


class RunParamConflictError(Exception):
    """Raised when a batch attempts to change an existing parameter value."""


class LoggedModelNotFoundError(Exception):
    """Raised when a logged model is not stored."""


class LoggedModelTagNotFoundError(Exception):
    """Raised when a tag is not stored on an existing logged model."""


class TraceNotFoundError(Exception):
    """Raised when a trace operation finds no matching trace or requested tag."""


class TraceWriteConflictError(Exception):
    """Raised when another span writer supersedes a summary snapshot."""


Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")


def translate_database_errors(
    function: Callable[Parameters, Result],
) -> Callable[Parameters, Result]:
    """Translate driver and BSON failures, preserving domain errors and the cause."""

    @wraps(function)
    def wrapper(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        try:
            return function(*args, **kwargs)
        except (PyMongoError, BSONError) as exc:
            raise RepositoryPersistenceError(
                f"Database operation '{function.__name__}' failed."
            ) from exc

    return wrapper
