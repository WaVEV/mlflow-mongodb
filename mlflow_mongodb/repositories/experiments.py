"""Persistence operations for experiments."""

from collections.abc import Mapping

from pymongo import ASCENDING, ReturnDocument
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from mlflow_mongodb.repositories.errors import (
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
)
from mlflow_mongodb.repositories.types import ExperimentRecord
from mlflow_mongodb.settings import MongoDBSettings


class ExperimentRepository:
    """Store experiment documents with embedded tags in MongoDB."""

    UNIQUE_NAME_INDEX = "experiments_name_unique"

    def __init__(self, database: Database, settings: MongoDBSettings | None = None):
        self._settings = settings or MongoDBSettings()
        self._collection = database[self._settings.experiments_collection_name]
        self._collection.create_index(
            [("name", ASCENDING)], unique=True, name=self.UNIQUE_NAME_INDEX
        )

    def create(
        self,
        *,
        experiment_id: str,
        name: str,
        artifact_location: str,
        lifecycle_stage: str,
        creation_timestamp: int,
        tags: Mapping[str, str],
    ) -> str:
        document = {
            "_id": experiment_id,
            "name": name,
            "artifact_location": artifact_location,
            "lifecycle_stage": lifecycle_stage,
            "creation_time": creation_timestamp,
            "last_update_time": creation_timestamp,
            "tags": [{"key": key, "value": value} for key, value in tags.items()],
        }
        try:
            self._collection.insert_one(document)
        except DuplicateKeyError as exc:
            # Do not report an ID collision as a duplicate experiment name.
            if exc.details and exc.details.get("keyPattern") == {"_id": 1}:
                raise
            raise ExperimentAlreadyExistsError(name) from exc
        return experiment_id

    def find_by_id(self, experiment_id: str) -> ExperimentRecord | None:
        document = self._collection.find_one({"_id": experiment_id})
        return ExperimentRecord.from_document(document) if document is not None else None

    def find_by_name(self, name: str) -> ExperimentRecord | None:
        document = self._collection.find_one({"name": name})
        return ExperimentRecord.from_document(document) if document is not None else None

    def mark_deleted(
        self,
        *,
        experiment_id: str | None,
        last_update_time: int,
    ) -> ExperimentRecord:
        return self._transition_lifecycle(
            experiment_id=experiment_id,
            current_stage="active",
            next_stage="deleted",
            updates={"last_update_time": last_update_time},
        )

    def restore(
        self,
        *,
        experiment_id: str | None,
        last_update_time: int,
    ) -> ExperimentRecord:
        return self._transition_lifecycle(
            experiment_id=experiment_id,
            current_stage="deleted",
            next_stage="active",
            updates={"last_update_time": last_update_time},
        )

    def _transition_lifecycle(
        self,
        *,
        experiment_id: str | None,
        current_stage: str,
        next_stage: str,
        updates: dict[str, object],
    ) -> ExperimentRecord:
        updates = {"lifecycle_stage": next_stage, **updates}
        document = self._collection.find_one_and_update(
            {"_id": experiment_id, "lifecycle_stage": current_stage},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise ExperimentNotFoundError(experiment_id)
        return ExperimentRecord.from_document(document)
