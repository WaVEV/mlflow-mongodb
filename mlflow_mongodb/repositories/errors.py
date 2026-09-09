"""Exceptions raised by MongoDB repositories."""


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


class ModelVersionAlreadyExistsError(Exception):
    """Raised when a model version number is already stored for a registered model."""


class ModelVersionNotFoundError(Exception):
    """Raised when a model version is not stored for a registered model."""


class RegisteredModelAlreadyExistsError(Exception):
    """Raised when a registered model name is already stored."""


class RegisteredModelNotFoundError(Exception):
    """Raised when a registered model name is not stored."""


class TraceNotFoundError(Exception):
    """Raised when a trace operation finds no matching trace or requested tag."""


class TraceWriteConflictError(Exception):
    """Raised when another span writer supersedes a summary snapshot."""
