"""Domain exceptions raised by the model-registry store and repositories."""


class ModelVersionAlreadyExistsError(Exception):
    """Raised when a model version number is already stored for a registered model."""


class ModelVersionNotFoundError(Exception):
    """Raised when a model version is not stored for a registered model."""


class RegisteredModelAlreadyExistsError(Exception):
    """Raised when a registered model name is already stored."""


class RegisteredModelNotFoundError(Exception):
    """Raised when a registered model name is not stored."""
