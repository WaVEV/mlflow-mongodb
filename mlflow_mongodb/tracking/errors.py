"""Tracking domain exceptions and repository persistence errors."""

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from bson.errors import BSONError
from pymongo.errors import PyMongoError


class RepositoryPersistenceError(Exception):
    """Raised when a repository database operation fails."""


class RepositoryAlreadyExistsError(Exception):
    """Raised when a resource with the requested unique identity already exists."""


class RepositoryNotFoundError(Exception):
    """Raised when a resource is not stored in the expected lifecycle stage."""


class RepositoryNotActiveError(Exception):
    """Raised when an operation requires an active resource."""


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
