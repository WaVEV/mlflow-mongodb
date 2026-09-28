"""Experiment exceptions raised by the tracking store and repository."""


class ExperimentAlreadyExistsError(Exception):
    """Raised when an experiment name is already stored."""


class ExperimentNotFoundError(Exception):
    """Raised when an experiment is not stored in the expected lifecycle stage."""


class ExperimentNotActiveError(Exception):
    """Raised when a rename targets a non-active experiment."""
