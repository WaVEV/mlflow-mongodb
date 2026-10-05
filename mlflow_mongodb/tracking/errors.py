"""Tracking domain exceptions."""


class RepositoryParamConflictError(Exception):
    """Raised when an operation attempts to change an existing parameter value."""
