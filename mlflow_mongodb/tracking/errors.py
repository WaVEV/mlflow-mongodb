"""Tracking domain exceptions and repository persistence errors."""


class RepositoryPersistenceError(Exception):
    """Raised when a repository database operation fails."""


class RepositoryAlreadyExistsError(Exception):
    """Raised when a resource with the requested unique identity already exists."""


class RepositoryNotFoundError(Exception):
    """Raised when a resource is not stored in the expected lifecycle stage."""


class RepositoryNotActiveError(Exception):
    """Raised when an operation requires an active resource."""


class RepositoryParamConflictError(Exception):
    """Raised when an operation attempts to change an existing parameter value."""


class RepositoryTagNotFoundError(Exception):
    """Raised when a requested tag is missing from an existing resource."""

