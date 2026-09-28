"""MongoDB tracking store with experiment persistence."""

import logging
from functools import cached_property
from uuid import uuid4

from mlflow.entities import Experiment, ExperimentTag, LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import append_to_uri_path, resolve_uri_if_local
from mlflow.utils.validation import (
    _validate_experiment_artifact_location,
    _validate_experiment_artifact_location_length,
    _validate_experiment_name,
    _validate_experiment_tag,
)
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConfigurationError

from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking.errors import (
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
)
from mlflow_mongodb.tracking.repositories import ExperimentRepository

logger = logging.getLogger(__name__)


class MongoDBTrackingStore(AbstractStore):
    """Persist experiments in MongoDB; other tracking operations remain inherited."""

    def __init__(self, store_uri: str | None = None, artifact_uri: str | None = None) -> None:
        super().__init__()
        self.store_uri = store_uri
        self.artifact_uri = artifact_uri
        self._settings = MongoDBSettings.from_environment()

    @cached_property
    def _mongo_client(self) -> MongoClient:
        if not self.store_uri:
            raise MlflowException(
                "A MongoDB tracking URI is required.", error_code=INVALID_PARAMETER_VALUE
            )
        try:
            return MongoClient(self.store_uri)
        except ConfigurationError:
            logger.exception("Unable to create MongoDB tracking client")
            raise MlflowException(
                "Invalid MongoDB tracking URI.", error_code=INVALID_PARAMETER_VALUE
            ) from None

    @cached_property
    def _database(self) -> Database:
        try:
            return self._mongo_client.get_default_database()
        except ConfigurationError:
            logger.exception("Unable to select the MongoDB tracking database")
            raise MlflowException(
                "The MongoDB tracking URI must include a database name.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from None

    @cached_property
    def _experiment_repository(self) -> ExperimentRepository:
        return ExperimentRepository(self._database, settings=self._settings)

    def create_experiment(
        self,
        name: str,
        artifact_location: str | None = None,
        tags: list[ExperimentTag] | None = None,
    ) -> str:
        _validate_experiment_name(name)
        _validate_experiment_artifact_location(artifact_location)
        tags_by_key = {}
        for tag in tags or []:
            _validate_experiment_tag(tag.key, tag.value)
            tags_by_key[tag.key] = tag.value

        # Decimal UUIDs preserve numeric experiment IDs used by prompt filters,
        # without requiring a shared counter or reserving the default ID "0".
        experiment_id = str(uuid4().int)
        artifact_location = resolve_uri_if_local(
            artifact_location or append_to_uri_path(self.artifact_uri or "./mlruns", experiment_id)
        )
        _validate_experiment_artifact_location_length(artifact_location)
        try:
            return self._experiment_repository.create(
                experiment_id=experiment_id,
                name=name,
                artifact_location=artifact_location,
                lifecycle_stage=LifecycleStage.ACTIVE,
                creation_timestamp=get_current_time_millis(),
                tags=tags_by_key,
            )
        except ExperimentAlreadyExistsError as exc:
            raise MlflowException(
                f"Experiment(name={name}) already exists.", RESOURCE_ALREADY_EXISTS
            ) from exc

    def get_experiment(self, experiment_id: str | None) -> Experiment:
        experiment_id = None if experiment_id is None else str(experiment_id)

        record = self._experiment_repository.find_by_id(experiment_id)
        if record is None:
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            )

        return Experiment(
            experiment_id=record.experiment_id,
            name=record.name,
            artifact_location=record.artifact_location,
            lifecycle_stage=record.lifecycle_stage,
            tags=[ExperimentTag(tag.key, tag.value) for tag in record.tags],
            creation_time=record.creation_time,
            last_update_time=record.last_update_time,
        )

    def get_experiment_by_name(self, experiment_name: str) -> Experiment | None:
        record = self._experiment_repository.find_by_name(experiment_name)
        if record is None:
            return None

        return Experiment(
            experiment_id=record.experiment_id,
            name=record.name,
            artifact_location=record.artifact_location,
            lifecycle_stage=record.lifecycle_stage,
            tags=[ExperimentTag(tag.key, tag.value) for tag in record.tags],
            creation_time=record.creation_time,
            last_update_time=record.last_update_time,
        )

    def delete_experiment(self, experiment_id: str) -> None:
        try:
            self._experiment_repository.mark_deleted(
                experiment_id=experiment_id,
                last_update_time=get_current_time_millis(),
            )
        except ExperimentNotFoundError as exc:
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            ) from exc

    def restore_experiment(self, experiment_id: str) -> None:
        try:
            self._experiment_repository.restore(
                experiment_id=experiment_id,
                last_update_time=get_current_time_millis(),
            )
        except ExperimentNotFoundError as exc:
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            ) from exc
