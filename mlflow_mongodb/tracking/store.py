"""Skeleton of the MongoDB tracking store for the agreed V1 scope."""

import asyncio
import json
import logging
import re
from collections import defaultdict
from functools import cached_property
from typing import Any
from uuid import uuid4

from bson.errors import BSONError
from mlflow.entities import (
    Assessment,
    Dataset,
    DatasetInput,
    Experiment,
    ExperimentTag,
    InputTag,
    LifecycleStage,
    LoggedModel,
    LoggedModelInput,
    LoggedModelOutput,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunInputs,
    RunOutputs,
    RunStatus,
    RunTag,
    Trace,
    TraceData,
    TraceInfo,
    TraceLocation,
    TraceState,
    ViewType,
)
from mlflow.entities.logged_model_parameter import LoggedModelParameter
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import Span
from mlflow.entities.trace_metrics import (
    MetricAggregation,
    MetricDataPoint,
    MetricViewType,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    TEMPORARILY_UNAVAILABLE,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import (
    MAX_RESULTS_QUERY_TRACE_METRICS,
    SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_THRESHOLD,
    SEARCH_TRACES_DEFAULT_MAX_RESULTS,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import (
    SpansLocation,
    TraceMetadataKey,
    TraceSizeStatsKey,
    TraceTagKey,
)
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, _get_run_name_from_tags
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import (
    SearchExperimentsUtils,
    SearchLoggedModelsPaginationToken,
    SearchTraceUtils,
    SearchUtils,
)
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import append_to_uri_path, resolve_uri_if_local
from mlflow.utils.validation import (
    _resolve_experiment_ids_and_locations,
    _validate_batch_log_data,
    _validate_batch_log_limits,
    _validate_dataset_inputs,
    _validate_experiment_artifact_location,
    _validate_experiment_artifact_location_length,
    _validate_experiment_name,
    _validate_experiment_tag,
    _validate_logged_model_name,
    _validate_metric_name,
    _validate_param_keys_unique,
    _validate_run_id,
    _validate_trace_tag,
)
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConfigurationError, PyMongoError

from mlflow_mongodb.infrastructure.settings import MongoDBSettings
from mlflow_mongodb.tracking._retry import retry_on_exception
from mlflow_mongodb.tracking.errors import (
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
    LoggedModelNotFoundError,
    LoggedModelTagNotFoundError,
    RepositoryPersistenceError,
    RunAlreadyExistsError,
    RunInactiveError,
    RunNotFoundError,
    RunParamConflictError,
    TraceNotFoundError,
    TraceWriteConflictError,
)
from mlflow_mongodb.tracking.logged_model_search import (
    parse_logged_model_filters,
    parse_logged_model_order,
    parse_logged_model_page_token,
    validate_logged_model_datasets,
)
from mlflow_mongodb.tracking.repositories import (
    ExperimentRepository,
    LoggedModelRepository,
    RunRepository,
)
from mlflow_mongodb.tracking.repositories.experiments import ExperimentFilter, ExperimentOrder
from mlflow_mongodb.tracking.repositories.traces import (
    TraceRepository,
    TraceSearchFilter,
    TraceSearchOrder,
)
from mlflow_mongodb.tracking.trace_utils import numeric_stats, span_to_document, summarize_spans
from mlflow_mongodb.tracking.types import (
    LoggedModelRecord,
    RunMetricRecord,
    SpanRecord,
    TraceRecord,
)

try:
    from mlflow.utils.search_utils import SearchEvaluationDatasetsUtils
except ImportError:
    # MLflow 3.1 has no evaluation-dataset search API or parser.
    SearchEvaluationDatasetsUtils = None

logger = logging.getLogger(__name__)


class _TraceNotFullyExportedError(Exception):
    """Raised while a trace's expected spans are still being persisted."""


class MongoDBTrackingStore(AbstractStore):
    """MongoDB tracking store with persistence methods awaiting implementation.

    Batch and single-value async logging use the implementations inherited from
    AbstractStore, which delegate persistence to log_batch. Methods outside V1
    remain inherited and are not a claim of support.
    """

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

    @cached_property
    def _run_repository(self) -> RunRepository:
        return RunRepository(self._database, settings=self._settings)

    @cached_property
    def _trace_repository(self) -> TraceRepository:
        return TraceRepository(self._database, settings=self._settings)

    @cached_property
    def _logged_model_repository(self) -> LoggedModelRepository:
        return LoggedModelRepository(self._database, settings=self._settings)

    # Experiments

    def search_experiments(
        self,
        view_type: ViewType = ViewType.ACTIVE_ONLY,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[Experiment]:
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                "a positive integer",
                INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                f"at most {SEARCH_MAX_RESULTS_THRESHOLD}",
                INVALID_PARAMETER_VALUE,
            )

        filters = self._parse_experiment_filters(filter_string)
        orders = self._parse_experiment_order(order_by)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        if offset < 0:
            raise MlflowException("Page offset must not be negative.", INVALID_PARAMETER_VALUE)
        try:
            records = self._experiment_repository.search(
                lifecycle_stages=LifecycleStage.view_type_to_stages(view_type),
                filters=filters,
                order_by=orders,
                offset=offset,
                limit=max_results + 1,
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to search experiments")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None
        except (PyMongoError, BSONError):
            # Repository construction creates indexes before the decorated query method runs.
            logger.exception("Unable to initialize experiment search")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        next_page_token = None
        if len(records) > max_results:
            records = records[:max_results]
            next_page_token = SearchUtils.create_page_token(offset + max_results)
        experiments = [
            Experiment(
                experiment_id=record.experiment_id,
                name=record.name,
                artifact_location=record.artifact_location,
                lifecycle_stage=record.lifecycle_stage,
                tags=[ExperimentTag(tag.key, tag.value) for tag in record.tags],
                creation_time=record.creation_time,
                last_update_time=record.last_update_time,
            )
            for record in records
        ]
        return PagedList(experiments, next_page_token)

    @staticmethod
    def _parse_experiment_filters(filter_string: str | None) -> tuple[ExperimentFilter, ...]:
        filters = []
        for parsed in SearchExperimentsUtils.parse_search_filter(filter_string):
            field_type = parsed["type"]
            key = parsed["key"]
            comparator = parsed["comparator"].upper()
            value = parsed["value"]

            if SearchExperimentsUtils.is_numeric_attribute(field_type, key, comparator):
                # MLflow returns numeric tokens as text; BSON comparisons need numbers.
                value = float(value)
            elif not (
                SearchExperimentsUtils.is_string_attribute(field_type, key, comparator)
                or SearchExperimentsUtils.is_tag(field_type, comparator)
            ):
                raise MlflowException.invalid_parameter_value(f"Invalid token type: {field_type}")

            filters.append(ExperimentFilter(field_type, key, comparator, value))
        return tuple(filters)

    @staticmethod
    def _parse_experiment_order(order_by: list[str] | None) -> tuple[ExperimentOrder, ...]:
        orders = []
        for field_type, key, ascending in map(
            SearchExperimentsUtils.parse_order_by_for_search_experiments,
            order_by or ["creation_time DESC", "experiment_id ASC"],
        ):
            if field_type != "attribute":
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order_by entity: {field_type}"
                )
            orders.append(ExperimentOrder(key, ascending))
        if not any(order.key == "experiment_id" for order in orders):
            orders.append(ExperimentOrder("experiment_id", False))
        return tuple(orders)

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

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        raise NotImplementedError

    # Runs. Single metric/param/tag logging is inherited and uses log_batch.

    def _search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str | None,
        run_view_type,
        max_results: int,
        order_by: list[str] | None,
        page_token: str | None,
    ) -> tuple[list[Run], str | None]:
        raise NotImplementedError

    def create_run(
        self,
        experiment_id: str | None,
        user_id: str,
        start_time: int,
        tags: list[RunTag] | None,
        run_name: str | None,
    ) -> Run:
        experiment_id = None if experiment_id is None else str(experiment_id)
        experiment = self._experiment_repository.find_by_id(experiment_id)
        if experiment is None:
            raise MlflowException(
                f"No Experiment with id={experiment_id} exists", RESOURCE_DOES_NOT_EXIST
            )
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                (
                    f"The experiment {experiment.experiment_id} must be in the 'active' state. "
                    f"Current state is {experiment.lifecycle_stage}."
                ),
                INVALID_PARAMETER_VALUE,
            )

        run_id = uuid4().hex
        artifact_uri = append_to_uri_path(experiment.artifact_location, run_id, "artifacts")
        run_tags = list(tags or [])
        run_name_tag = _get_run_name_from_tags(run_tags)
        if run_name and run_name_tag and run_name != run_name_tag:
            raise MlflowException(
                "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                f"different values (run_name='{run_name}', run_name_tag='{run_name_tag}').",
                INVALID_PARAMETER_VALUE,
            )
        resolved_run_name = run_name or run_name_tag or _generate_random_name()
        if not run_name_tag:
            run_tags.append(RunTag(key=MLFLOW_RUN_NAME, value=resolved_run_name))

        try:
            record = self._run_repository.create(
                run_id=run_id,
                experiment_id=experiment_id,
                name=resolved_run_name,
                artifact_uri=artifact_uri,
                user_id=user_id,
                status=RunStatus.to_string(RunStatus.RUNNING),
                start_time=start_time,
                lifecycle_stage=LifecycleStage.ACTIVE,
                tags={tag.key: tag.value for tag in run_tags},
            )
        except RunAlreadyExistsError as exc:
            raise MlflowException(
                f"Run with id={run_id} already exists", RESOURCE_ALREADY_EXISTS
            ) from exc

        return Run(
            RunInfo(
                run_id=record.run_id,
                experiment_id=record.experiment_id,
                user_id=record.user_id,
                status=record.status,
                start_time=record.start_time,
                end_time=record.end_time,
                lifecycle_stage=record.lifecycle_stage,
                artifact_uri=record.artifact_uri,
                run_name=record.name,
            ),
            RunData(tags=[RunTag(tag.key, tag.value) for tag in record.tags]),
            RunInputs(dataset_inputs=[]),
        )

    def get_run(self, run_id: str) -> Run:
        record = self._run_repository.find_by_id(run_id)
        if record is None:
            raise MlflowException(f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST)

        return Run(
            RunInfo(
                run_id=record.run_id,
                experiment_id=record.experiment_id,
                user_id=record.user_id,
                status=record.status,
                start_time=record.start_time,
                end_time=record.end_time,
                lifecycle_stage=record.lifecycle_stage,
                artifact_uri=record.artifact_uri,
                run_name=record.name,
            ),
            RunData(
                metrics=[
                    Metric(
                        key=metric.key,
                        value=metric.value,
                        timestamp=metric.timestamp,
                        step=metric.step,
                        model_id=metric.model_id,
                        dataset_name=metric.dataset_name,
                        dataset_digest=metric.dataset_digest,
                    )
                    for metric in record.metrics
                ],
                tags=[RunTag(tag.key, tag.value) for tag in record.tags],
            ),
            RunInputs(
                dataset_inputs=[
                    DatasetInput(
                        dataset=Dataset(
                            name=dataset.name,
                            digest=dataset.digest,
                            source_type=dataset.source_type,
                            source=dataset.source,
                            schema=dataset.schema,
                            profile=dataset.profile,
                        ),
                        tags=[InputTag(tag.key, tag.value) for tag in dataset.tags],
                    )
                    for dataset in record.dataset_inputs
                ],
                model_inputs=[LoggedModelInput(model_id) for model_id in record.model_inputs],
            ),
            RunOutputs(
                model_outputs=[
                    LoggedModelOutput(model_id=model.model_id, step=model.step)
                    for model in record.model_outputs
                ]
            ),
        )

    def delete_run(self, run_id: str) -> None:
        try:
            self._run_repository.mark_deleted(
                run_id=run_id,
                deleted_time=get_current_time_millis(),
            )
        except RunNotFoundError as exc:
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from exc

    def restore_run(self, run_id: str) -> None:
        try:
            self._run_repository.restore(run_id=run_id)
        except RunNotFoundError as exc:
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from exc

    def update_run_info(
        self,
        run_id: str,
        run_status: RunStatus | None,
        end_time: int | None,
        run_name: str | None,
    ) -> RunInfo:
        run = self._run_repository.find_by_id(run_id)
        if run is None:
            raise MlflowException(f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST)
        if run.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                (
                    f"The run {run.run_id} must be in the 'active' state. "
                    f"Current state is {run.lifecycle_stage}."
                ),
                INVALID_PARAMETER_VALUE,
            )

        status = RunStatus.to_string(run_status) if run_status is not None else None
        try:
            updated = self._run_repository.update_info(
                run_id=run_id,
                status=status,
                end_time=end_time,
                run_name=run_name,
            )
        except RunNotFoundError as exc:
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from exc

        return RunInfo(
            run_id=updated.run_id,
            experiment_id=updated.experiment_id,
            user_id=updated.user_id,
            status=updated.status,
            start_time=updated.start_time,
            end_time=updated.end_time,
            lifecycle_stage=updated.lifecycle_stage,
            artifact_uri=updated.artifact_uri,
            run_name=updated.name,
        )

    def log_batch(
        self, run_id: str, metrics: list[Metric], params: list[Param], tags: list[RunTag]
    ) -> None:
        _validate_run_id(run_id)
        metrics, params, tags = _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        _validate_param_keys_unique(params)

        try:
            self._run_repository.log_batch(
                run_id=run_id,
                metrics=[
                    {
                        "k": metric.key,
                        "v": metric.value,
                        "timestamp": metric.timestamp,
                        "step": metric.step,
                        "model_id": metric.model_id,
                        "dataset_name": metric.dataset_name,
                        "dataset_digest": metric.dataset_digest,
                    }
                    for metric in metrics
                ],
                params=[{"key": param.key, "value": param.value} for param in params],
                tags=[{"key": tag.key, "value": tag.value} for tag in tags],
            )
        except RunParamConflictError as exc:
            key, old_value, new_value, conflicting_run_id = exc.args
            raise MlflowException(
                f"Changing param values is not allowed. Param with key='{key}' was already logged "
                f"with value='{old_value}' for run ID='{conflicting_run_id}'. Attempted logging "
                f"new value '{new_value}'.",
                INVALID_PARAMETER_VALUE,
            ) from exc
        except RunNotFoundError as exc:
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from exc

    def log_inputs(
        self,
        run_id: str,
        datasets: list[DatasetInput] | None = None,
        models: list[LoggedModelInput] | None = None,
    ) -> None:
        _validate_run_id(run_id)
        if datasets is not None:
            if not isinstance(datasets, list):
                raise TypeError(f"Argument 'datasets' should be a list, got '{type(datasets)}'")
            _validate_dataset_inputs(datasets)

        dataset_inputs = [
            {
                "dataset": dataset_input.dataset.to_dictionary(),
                "tags": [{"key": tag.key, "value": tag.value} for tag in dataset_input.tags],
            }
            for dataset_input in datasets or []
        ]
        model_inputs = [{"model_id": model.model_id} for model in models or []]
        try:
            self._run_repository.log_inputs(
                run_id=run_id, datasets=dataset_inputs, models=model_inputs
            )
        except RunNotFoundError as exc:
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from exc
        except RunInactiveError as exc:
            _, lifecycle_stage = exc.args
            raise MlflowException(
                f"The run {run_id} must be in the 'active' state. "
                f"Current state is {lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            ) from exc
        except PyMongoError:
            logger.exception("Unable to log run inputs")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

    def log_outputs(self, run_id: str, models: list[LoggedModelOutput]) -> None:
        _validate_run_id(run_id)
        model_outputs = [{"model_id": model.model_id, "step": model.step} for model in models]
        try:
            self._run_repository.log_outputs(run_id=run_id, models=model_outputs)
        except RunNotFoundError as exc:
            raise MlflowException(
                f"Run with id={run_id} not found", RESOURCE_DOES_NOT_EXIST
            ) from exc
        except RunInactiveError as exc:
            _, lifecycle_stage = exc.args
            raise MlflowException(
                f"The run {run_id} must be in the 'active' state. "
                f"Current state is {lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            ) from exc
        except PyMongoError:
            logger.exception("Unable to log run outputs")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

    def get_metric_history(
        self,
        run_id: str,
        metric_key: str,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList[Metric]:
        _validate_run_id(run_id)
        _validate_metric_name(metric_key)
        if max_results is not None and (
            isinstance(max_results, bool) or not isinstance(max_results, int) or max_results <= 0
        ):
            raise MlflowException(
                "max_results must be a positive integer.", INVALID_PARAMETER_VALUE
            )
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        if offset < 0:
            raise MlflowException("Page offset must not be negative.", INVALID_PARAMETER_VALUE)

        try:
            metrics = self._run_repository.get_metric_history(
                run_id=run_id,
                metric_key=metric_key,
                offset=offset,
                limit=max_results + 1 if max_results is not None else None,
            )
        except PyMongoError:
            logger.exception("Unable to read metric history")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        next_token = None
        if max_results is not None and len(metrics) > max_results:
            metrics = metrics[:max_results]
            next_token = SearchUtils.create_page_token(offset + max_results)
        return PagedList(
            [
                Metric(
                    key=metric.key,
                    value=metric.value,
                    timestamp=metric.timestamp,
                    step=metric.step,
                    model_id=metric.model_id,
                    dataset_name=metric.dataset_name,
                    dataset_digest=metric.dataset_digest,
                    run_id=run_id,
                )
                for metric in metrics
            ],
            next_token,
        )

    def start_trace(self, trace_info: TraceInfo) -> TraceInfo:
        try:
            experiment = self.get_experiment(trace_info.experiment_id)
        except PyMongoError:
            logger.exception("Unable to load the trace experiment")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                f"The experiment {experiment.experiment_id} must be in the 'active' state. "
                f"Current state is {experiment.lifecycle_stage}.",
                INVALID_PARAMETER_VALUE,
            )

        tags = dict(trace_info.tags)
        # Span payloads belong in MongoDB, so do not advertise an artifact URI.
        tags[TraceTagKey.SPANS_LOCATION] = SpansLocation.TRACKING_STORE.value
        source_run_id = trace_info.trace_metadata.get(TraceMetadataKey.SOURCE_RUN)
        run_ids = [source_run_id] if source_run_id else []
        assessments = []
        for assessment in trace_info.assessments:
            document = assessment.to_dictionary()
            if assessment.feedback is not None:
                metric_value = assessment.feedback.value
            elif assessment.expectation is not None:
                metric_value = assessment.expectation.value
            else:
                metric_value = assessment.issue.to_dictionary()
            document["_metric_value_json"] = json.dumps(metric_value)
            if not document.get("assessment_id"):
                document["assessment_id"] = uuid4().hex
            if not document.get("trace_id"):
                document["trace_id"] = trace_info.trace_id
            assessments.append(document)

        try:
            record = self._trace_repository.start_trace(
                trace_id=trace_info.trace_id,
                experiment_id=experiment.experiment_id,
                request_time=trace_info.request_time,
                state=trace_info.state.value,
                execution_duration=trace_info.execution_duration,
                client_request_id=trace_info.client_request_id,
                request_preview=trace_info.request_preview,
                response_preview=trace_info.response_preview,
                tags=tags,
                trace_metadata=trace_info.trace_metadata,
                assessments=assessments,
                run_ids=run_ids,
                span_stats=(
                    self._parse_span_stats(trace_info.trace_metadata[TraceMetadataKey.SIZE_STATS])
                    if TraceMetadataKey.SIZE_STATS in trace_info.trace_metadata
                    else None
                ),
                metrics={
                    field: numeric_stats(trace_info.trace_metadata[key])
                    for field, key in (
                        ("token_usage", TraceMetadataKey.TOKEN_USAGE),
                        ("cost", TraceMetadataKey.COST),
                    )
                    if key in trace_info.trace_metadata
                },
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to start trace")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        return self._to_trace_info(record)

    @staticmethod
    def _parse_span_stats(value: str) -> dict[str, Any]:
        """Decode MLflow's serialized stats at the store boundary."""
        if not value:
            return {}
        try:
            stats = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise MlflowException.invalid_parameter_value("Invalid trace size-stats JSON.") from exc
        if not isinstance(stats, dict):
            raise MlflowException.invalid_parameter_value("Trace size stats must be an object.")
        count = stats.get(TraceSizeStatsKey.NUM_SPANS, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise MlflowException.invalid_parameter_value(
                "Trace size-stats num_spans must be a non-negative integer."
            )
        return stats

    @staticmethod
    def _to_trace_info(record: TraceRecord) -> TraceInfo:
        return TraceInfo(
            trace_id=record.trace_id,
            trace_location=TraceLocation.from_experiment_id(record.experiment_id),
            request_time=record.request_time,
            state=TraceState(record.state),
            execution_duration=record.execution_duration,
            client_request_id=record.client_request_id,
            request_preview=record.request_preview,
            response_preview=record.response_preview,
            tags={tag.key: tag.value for tag in record.tags},
            trace_metadata={item.key: item.value for item in record.trace_metadata},
            assessments=[Assessment.from_dictionary(item) for item in record.assessments],
        )

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        # Keep validation here for compatibility with MLflow versions that do not
        # implement the public delete_traces wrapper in AbstractStore.
        if max_timestamp_millis is None and not trace_ids:
            raise MlflowException.invalid_parameter_value(
                "Either `max_timestamp_millis` or `trace_ids` must be specified."
            )
        if max_timestamp_millis is not None and trace_ids:
            raise MlflowException.invalid_parameter_value(
                "Only one of `max_timestamp_millis` and `trace_ids` can be specified."
            )
        if trace_ids and max_traces is not None:
            raise MlflowException.invalid_parameter_value(
                "`max_traces` can't be specified if `trace_ids` is specified."
            )
        if max_traces is not None and max_traces <= 0:
            raise MlflowException.invalid_parameter_value(
                f"`max_traces` must be a positive integer, received {max_traces}."
            )

        try:
            return self._trace_repository.delete_traces(
                experiment_id=experiment_id,
                max_timestamp_millis=max_timestamp_millis,
                max_traces=max_traces,
                trace_ids=trace_ids,
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to delete traces")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        return self._to_trace_info(self._get_trace_record(trace_id))

    def batch_get_traces(
        self,
        trace_ids: list[str],
        location: str | None = None,  # ruff: ignore[unused-method-argument]
        experiment_ids: list[str] | None = None,
    ) -> list[Trace]:
        """Return complete traces in request order, omitting missing or incomplete traces.

        The connection selects MongoDB; location is unused by this backend.
        """
        if not trace_ids or experiment_ids == []:
            return []

        try:
            records = self._trace_repository.batch_get_traces(
                trace_ids, experiment_ids=experiment_ids
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to read traces")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        traces = []
        for record, span_records in records:
            if not span_records:
                continue
            if record.span_stats and len(span_records) < record.span_stats.get(
                TraceSizeStatsKey.NUM_SPANS, 0
            ):
                continue
            traces.append(self._to_trace(record, span_records))
        return traces

    def batch_get_trace_infos(
        self,
        trace_ids: list[str],
        location: str | None = None,  # ruff: ignore[unused-method-argument]
        experiment_ids: list[str] | None = None,
    ) -> list[TraceInfo]:
        """Return scoped trace metadata in request order without loading spans.

        The connection selects MongoDB; location is unused by this backend.
        """
        if not trace_ids or experiment_ids == []:
            return []

        try:
            records = self._trace_repository.batch_get_trace_infos(
                trace_ids, experiment_ids=experiment_ids
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to read trace metadata")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        return [self._to_trace_info(record) for record in records]

    def _get_trace_record(self, trace_id: str) -> TraceRecord:
        try:
            record = self._trace_repository.get_trace_info(trace_id)
        except TraceNotFoundError as exc:
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from exc
        except RepositoryPersistenceError:
            logger.exception("Unable to read trace metadata")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        return record

    def get_trace(self, trace_id: str, *, allow_partial: bool = False) -> Trace:
        """Load persisted spans, retrying incomplete exports unless partial reads are allowed."""
        try:
            return self._get_trace(trace_id, allow_partial=allow_partial)
        except _TraceNotFullyExportedError as exc:
            raise MlflowException(
                f"Trace with ID {trace_id} is not fully exported yet, please try again later.",
                RESOURCE_DOES_NOT_EXIST,
            ) from exc

    @retry_on_exception(_TraceNotFullyExportedError, attempts=3, backoff_seconds=(1, 2))
    def _get_trace(self, trace_id: str, *, allow_partial: bool) -> Trace:
        # Refresh metadata and spans on each attempt because they arrive separately.
        # Missing metadata and database errors propagate immediately without retrying.
        record = self._get_trace_record(trace_id)

        try:
            span_documents = self._trace_repository.get_spans(trace_id)
        except RepositoryPersistenceError:
            logger.exception("Unable to read trace spans")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        if not allow_partial:
            if trace_stats := record.span_stats:
                expected_spans = trace_stats.get(TraceSizeStatsKey.NUM_SPANS, 0)
                if len(span_documents) < expected_spans:
                    raise _TraceNotFullyExportedError
            # Without stats, a nonempty result has no further completeness check.
            if not span_documents:
                raise _TraceNotFullyExportedError

        return self._to_trace(record, span_documents)

    def _to_trace(self, record: TraceRecord, span_records: list[SpanRecord]) -> Trace:
        trace_info = self._to_trace_info(record)
        spans = [Span.from_dict(span_record.content) for span_record in span_records]
        spans.sort(key=lambda span: (span.parent_id is not None, span.start_time_ns, span.span_id))
        return Trace(info=trace_info, data=TraceData(spans=spans))

    def log_spans(
        self,
        location: str,
        spans: list[Span],
        tracking_uri: str | None = None,  # ruff: ignore[unused-method-argument]
    ) -> list[Span]:
        """Persist spans and refresh their traces, allowing repeated and late delivery.

        The store's connection selects MongoDB; tracking_uri is unused by this backend.
        Writes across collections are separate and may partially succeed on failure.
        """
        if not spans:
            return []
        if not isinstance(location, str) or not location:
            raise MlflowException.invalid_parameter_value("location must be an experiment ID.")

        self._validate_span_ingestion_experiment(location)
        documents, documents_by_trace = self._prepare_span_documents(spans)
        self._ensure_span_traces(location, documents_by_trace)
        self._persist_span_documents(documents)
        self._refresh_span_summaries(location, documents_by_trace)
        return spans

    def _validate_span_ingestion_experiment(self, experiment_id: str) -> None:
        try:
            experiment = self.get_experiment(experiment_id)
        except PyMongoError:
            logger.exception("Unable to load the span experiment")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException.invalid_parameter_value(
                f"The experiment {experiment_id} must be in the 'active' state. "
                f"Current state is {experiment.lifecycle_stage}."
            )

    @staticmethod
    def _prepare_span_documents(
        spans: list[Span],
    ) -> tuple[list[dict[str, Any]], defaultdict[str, list[dict[str, Any]]]]:
        # First delivery wins, including repeated identities in the same batch.
        # Prepare and validate all BSON documents before creating placeholders.
        documents_by_identity = {}
        for span in spans:
            identity = (span.trace_id, span.span_id)
            if identity not in documents_by_identity:
                documents_by_identity[identity] = span_to_document(span)

        documents_by_trace = defaultdict(list)
        for document in documents_by_identity.values():
            documents_by_trace[document["trace_id"]].append(document)
        return list(documents_by_identity.values()), documents_by_trace

    def _ensure_span_traces(
        self,
        experiment_id: str,
        documents_by_trace: defaultdict[str, list[dict[str, Any]]],
    ) -> None:
        request_times = {
            trace_id: min(document["start_time_ns"] for document in trace_documents) // 1_000_000
            for trace_id, trace_documents in documents_by_trace.items()
        }
        try:
            self._trace_repository.ensure_traces(
                experiment_id=experiment_id, request_times=request_times
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to create trace placeholders")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

    def _persist_span_documents(self, documents: list[dict[str, Any]]) -> None:
        try:
            self._trace_repository.log_spans(documents)
        except RepositoryPersistenceError:
            logger.exception("Unable to persist spans")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

    def _refresh_span_summaries(
        self,
        experiment_id: str,
        documents_by_trace: defaultdict[str, list[dict[str, Any]]],
    ) -> None:
        try:
            for trace_id in documents_by_trace:
                self._refresh_trace_span_summary(trace_id, experiment_id)
        except TraceNotFoundError as exc:
            raise MlflowException(
                f"Trace '{exc.args[0]}' was deleted during span ingestion.",
                RESOURCE_DOES_NOT_EXIST,
            ) from exc
        except TraceWriteConflictError as exc:
            raise MlflowException(
                f"Concurrent span writes prevented refreshing trace '{exc.args[0]}'. "
                "Retry log_spans to refresh its summary; stored spans are deduplicated.",
                TEMPORARILY_UNAVAILABLE,
            ) from exc
        except RepositoryPersistenceError:
            logger.exception("Unable to refresh trace span summaries")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

    @retry_on_exception(TraceWriteConflictError, attempts=3, backoff_seconds=(0.01, 0.05))
    def _refresh_trace_span_summary(self, trace_id: str, experiment_id: str) -> None:
        revision, documents = self._trace_repository.span_summary_snapshot(
            trace_id=trace_id, experiment_id=experiment_id
        )
        if not documents:
            # Span deletion precedes trace deletion; do not finalize an empty
            # snapshot or invent a successful write while deletion is in progress.
            raise TraceNotFoundError(trace_id)
        summary = summarize_spans(documents)
        # MLflow's trace_metadata contract requires strings. The corresponding
        # native fields are stored in the same atomic update for MongoDB queries.
        summary["aggregate_metadata"] = {
            key: json.dumps(summary[field])
            for field, key in (
                ("token_usage", TraceMetadataKey.TOKEN_USAGE),
                ("cost", TraceMetadataKey.COST),
            )
            if summary[field] is not None
        }
        self._trace_repository.update_span_summary(
            trace_id=trace_id, experiment_id=experiment_id, revision=revision, summary=summary
        )

    async def log_spans_async(self, location: str, spans: list[Span]) -> list[Span]:
        return await asyncio.to_thread(self.log_spans, location, spans)

    def set_trace_tag(self, trace_id: str, key: str, value: str) -> None:
        key, value = _validate_trace_tag(key, value)
        try:
            self._trace_repository.set_trace_tag(trace_id=trace_id, key=key, value=value)
        except TraceNotFoundError as exc:
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from exc
        except RepositoryPersistenceError:
            logger.exception("Unable to set trace tag")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

    def delete_trace_tag(self, trace_id: str, key: str) -> None:
        try:
            self._trace_repository.delete_trace_tag(trace_id=trace_id, key=key)
        except TraceNotFoundError as exc:
            raise MlflowException(
                f"Trace '{trace_id}' or tag '{key}' not found.",
                RESOURCE_DOES_NOT_EXIST,
            ) from exc
        except RepositoryPersistenceError:
            logger.exception("Unable to delete trace tag")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

    def search_traces(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        model_id: str | None = None,  # ruff: ignore[unused-method-argument]
        locations: list[str] | None = None,
    ) -> tuple[list[TraceInfo], str | None]:
        locations = _resolve_experiment_ids_and_locations(experiment_ids, locations)
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or max_results < 1
            or max_results > SEARCH_MAX_RESULTS_THRESHOLD
        ):
            raise MlflowException(
                f"Invalid value {max_results} for parameter 'max_results' supplied. It must be "
                f"a positive integer at most {SEARCH_MAX_RESULTS_THRESHOLD}",
                INVALID_PARAMETER_VALUE,
            )
        filters = []
        for clause in SearchTraceUtils.parse_search_filter_for_search_traces(filter_string):
            field_type = clause["type"]
            if field_type in ("span", "feedback", "expectation", "issue"):
                continue
            key = clause["key"]
            operator = clause["comparator"].upper()
            value = clause["value"]
            if SearchTraceUtils.is_attribute(field_type, key, operator):
                pass
            elif SearchTraceUtils.is_tag(field_type, operator):
                if key == TraceTagKey.LINKED_PROMPTS and (operator != "=" or value.count("/") != 1):
                    raise MlflowException.invalid_parameter_value(
                        'Prompt filters require `prompt = "name/version"`.'
                    )
            elif SearchTraceUtils.is_request_metadata(field_type, operator):
                if key in (
                    TraceMetadataKey.TOKEN_USAGE,
                    TraceMetadataKey.COST,
                ) and operator not in ("=", "!=", "IS NULL", "IS NOT NULL"):
                    raise MlflowException.invalid_parameter_value(
                        f"Comparator '{operator}' is not supported for reserved metadata '{key}'. "
                        "Only '=', '!=', 'IS NULL', and 'IS NOT NULL' are supported."
                    )
            else:
                raise MlflowException(
                    f"Invalid trace search field type: {field_type}",
                    INVALID_PARAMETER_VALUE,
                )
            if operator == "RLIKE":
                try:
                    re.compile(value)
                except re.error:
                    raise MlflowException.invalid_parameter_value(
                        "Invalid regular expression in trace filter."
                    ) from None
            filters.append(TraceSearchFilter(field_type, key, operator, value))

        orders = []
        seen_order_fields = set()
        for clause in order_by or []:
            field_type, key, ascending = SearchTraceUtils.parse_order_by_for_search_traces(clause)
            if field_type == "attribute":
                SearchTraceUtils.is_attribute(field_type, key, "=")
            elif field_type == "tag":
                SearchTraceUtils.is_tag(field_type, "=")
            elif field_type == "request_metadata":
                SearchTraceUtils.is_request_metadata(field_type, "=")
            else:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid trace ordering field: {field_type}"
                )
            if field_type == "request_metadata" and key in (
                TraceMetadataKey.TOKEN_USAGE,
                TraceMetadataKey.COST,
            ):
                raise MlflowException.invalid_parameter_value(
                    f"Ordering by reserved metadata '{key}' is not supported."
                )
            if (field_type, key) in seen_order_fields:
                raise MlflowException.invalid_parameter_value(
                    f"`order_by` contains duplicate fields: {order_by}"
                )
            seen_order_fields.add((field_type, key))
            orders.append(TraceSearchOrder(field_type, key, ascending))
        if 2 * len(orders) + 2 > 32:
            raise MlflowException.invalid_parameter_value("Too many trace ordering fields.")
        offset = SearchTraceUtils.parse_start_offset_from_page_token(page_token)
        if offset < 0:
            raise MlflowException("Page offset must not be negative.", INVALID_PARAMETER_VALUE)
        try:
            records = self._trace_repository.search_trace_infos(
                experiment_ids=locations,
                filters=filters,
                order_by=orders,
                offset=offset,
                limit=max_results,
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to search traces")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

        next_token = None
        if len(records) == max_results:
            next_token = SearchTraceUtils.create_page_token(offset + max_results)
        return [self._to_trace_info(record) for record in records], next_token

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        raise NotImplementedError

    def create_assessment(self, assessment: Assessment) -> Assessment:
        raise NotImplementedError

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        name: str | None = None,
        expectation: str | None = None,
        feedback: str | None = None,
        rationale: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Assessment:
        raise NotImplementedError

    def delete_assessment(self, trace_id: str, assessment_id: str) -> None:
        raise NotImplementedError

    def query_trace_metrics(
        self,
        experiment_ids: list[str],
        view_type: MetricViewType,
        metric_name: str,
        aggregations: list[MetricAggregation],
        dimensions: list[str] | None = None,
        filters: list[str] | None = None,
        time_interval_seconds: int | None = None,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        max_results: int = MAX_RESULTS_QUERY_TRACE_METRICS,
        page_token: str | None = None,  # ruff: ignore[unused-method-argument]
    ) -> PagedList[MetricDataPoint]:
        # Its SQL models import MLflow clients, which are unavailable during discovery.
        from mlflow.store.tracking.utils.sql_trace_metrics_utils import (  # ruff: ignore[import-outside-top-level]
            validate_query_trace_metrics_params,
        )

        validate_query_trace_metrics_params(view_type, metric_name, aggregations, dimensions)
        if time_interval_seconds and (start_time_ms is None or end_time_ms is None):
            raise MlflowException.invalid_parameter_value(
                "start_time_ms and end_time_ms are required if time_interval_seconds is set"
            )

        try:
            points = self._trace_repository.query_trace_metrics(
                experiment_ids=experiment_ids,
                view_type=view_type,
                metric_name=metric_name,
                aggregations=aggregations,
                dimensions=dimensions,
                filters=filters,
                time_interval_seconds=time_interval_seconds,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                max_results=max_results,
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to query trace metrics")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None
        return PagedList(points, None)

    # Logged models

    def create_logged_model(
        self,
        experiment_id: str,
        name: str | None = None,
        source_run_id: str | None = None,
        tags: list[LoggedModelTag] | None = None,
        params: list[LoggedModelParameter] | None = None,
        model_type: str | None = None,
    ) -> LoggedModel:
        """Create a pending logged model and persist its metadata atomically."""
        _validate_logged_model_name(name)

        # Preserve SQLAlchemyStore's rejection of duplicate and null entries
        # before embedding them in arrays in a single MongoDB document.
        for field, entries in (("params", params), ("tags", tags)):
            seen_keys = set()
            for entry in entries or []:
                if entry.key is None or entry.value is None or entry.key in seen_keys:
                    raise MlflowException(
                        f"Logged model {field} must have unique, non-null keys "
                        "and non-null values.",
                        BAD_REQUEST,
                    )
                seen_keys.add(entry.key)

        try:
            experiment = self.get_experiment(experiment_id)
        except (PyMongoError, BSONError) as error:
            logger.error("Unable to load experiment for logged model: %s", error)
            raise MlflowException("Unable to create logged model.", INTERNAL_ERROR) from None

        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                (
                    f"The experiment {experiment.experiment_id} must be in the 'active' state. "
                    f"Current state is {experiment.lifecycle_stage}."
                ),
                INVALID_PARAMETER_VALUE,
            )

        model_id = f"m-{uuid4().hex}"
        artifact_location = append_to_uri_path(
            experiment.artifact_location, "models", model_id, "artifacts"
        )
        try:
            record = self._logged_model_repository.create(
                model_id=model_id,
                experiment_id=experiment.experiment_id,
                name=name or _generate_random_name(),
                artifact_location=artifact_location,
                creation_timestamp=get_current_time_millis(),
                status=LoggedModelStatus.PENDING.value,
                lifecycle_stage=LifecycleStage.ACTIVE,
                source_run_id=source_run_id,
                model_type=model_type,
                tags={tag.key: tag.value for tag in tags or []},
                params={param.key: param.value for param in params or []},
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to create logged model")
            raise MlflowException("Unable to create logged model.", INTERNAL_ERROR) from None

        return self._to_logged_model(record)

    @staticmethod
    def _to_logged_model(
        record: LoggedModelRecord, metrics: tuple[RunMetricRecord, ...] = ()
    ) -> LoggedModel:
        return LoggedModel(
            model_id=record.model_id,
            experiment_id=record.experiment_id,
            name=record.name,
            artifact_location=record.artifact_location,
            creation_timestamp=record.creation_timestamp,
            last_updated_timestamp=record.last_updated_timestamp,
            status=LoggedModelStatus(record.status),
            status_message=record.status_message,
            source_run_id=record.source_run_id,
            model_type=record.model_type,
            tags=[LoggedModelTag(tag.key, tag.value) for tag in record.tags],
            params=[LoggedModelParameter(param.key, param.value) for param in record.params],
            metrics=[
                Metric(
                    key=metric.key,
                    value=metric.value,
                    timestamp=metric.timestamp,
                    step=metric.step,
                    model_id=record.model_id,
                    run_id=metric.run_id,
                    dataset_name=metric.dataset_name,
                    dataset_digest=metric.dataset_digest,
                )
                for metric in metrics
            ]
            or None,
        )

    def search_logged_models(
        self,
        experiment_ids: list[str],
        filter_string: str | None = None,
        datasets: list[dict[str, Any]] | None = None,
        max_results: int | None = None,
        order_by: list[dict[str, Any]] | None = None,
        page_token: str | None = None,
    ) -> PagedList[LoggedModel]:
        """Search model metadata and associated metrics within the requested experiments."""
        validate_logged_model_datasets(datasets)
        if not isinstance(experiment_ids, list) or not all(
            isinstance(experiment_id, str) for experiment_id in experiment_ids
        ):
            raise MlflowException.invalid_parameter_value(
                "`experiment_ids` must be a list of strings."
            )
        offset = parse_logged_model_page_token(page_token, experiment_ids, filter_string, order_by)
        if isinstance(max_results, bool) or (
            max_results is not None and not isinstance(max_results, int)
        ):
            raise MlflowException.invalid_parameter_value("`max_results` must be an integer.")
        max_results = max_results or SEARCH_LOGGED_MODEL_MAX_RESULTS_DEFAULT
        if max_results < 1:
            raise MlflowException.invalid_parameter_value(
                "`max_results` must be a positive integer."
            )
        filters = parse_logged_model_filters(filter_string)
        orders = parse_logged_model_order(order_by)
        if not experiment_ids:
            return PagedList([], None)
        try:
            page = self._logged_model_repository.search(
                experiment_ids=experiment_ids,
                filters=filters,
                datasets=datasets or [],
                order_by=orders,
                offset=offset,
                max_results=max_results,
            )
        except RepositoryPersistenceError:
            logger.exception("Unable to search logged models")
            raise MlflowException("Unable to search logged models.", INTERNAL_ERROR) from None

        next_token = (
            SearchLoggedModelsPaginationToken(
                experiment_ids=experiment_ids,
                filter_string=filter_string or None,
                order_by=order_by or None,
                offset=offset + max_results,
            ).encode()
            if page.has_more
            else None
        )
        return PagedList(
            [self._to_logged_model(result.model, result.metrics) for result in page.records],
            next_token,
        )

    def search_datasets(
        self,
        experiment_ids: list[str] | None = None,  # ruff: ignore[unused-method-argument]
        filter_string: str | None = None,
        max_results: int = 1000,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList:
        """Return an empty evaluation-dataset page until dataset storage is implemented."""
        if SearchEvaluationDatasetsUtils is None:
            raise MlflowException.invalid_parameter_value(
                "Evaluation dataset search is unavailable in this MLflow version."
            )

        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or max_results < 1
            or max_results > SEARCH_MAX_RESULTS_THRESHOLD
        ):
            raise MlflowException.invalid_parameter_value(
                f"`max_results` must be a positive integer at most {SEARCH_MAX_RESULTS_THRESHOLD}."
            )

        for clause in SearchEvaluationDatasetsUtils.parse_search_filter(filter_string):
            key_type = clause["type"]
            comparator = clause["comparator"]
            if (
                key_type == "attribute"
                and clause["key"] in SearchEvaluationDatasetsUtils.NUMERIC_ATTRIBUTES
            ):
                valid_comparators = (
                    SearchEvaluationDatasetsUtils.VALID_NUMERIC_ATTRIBUTE_COMPARATORS
                )
            else:
                valid_comparators = SearchEvaluationDatasetsUtils.VALID_TAG_COMPARATORS
            if comparator not in valid_comparators:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid comparator for evaluation dataset {key_type}: {comparator}"
                )

        for clause in order_by or []:
            key_type, _, _ = (
                SearchEvaluationDatasetsUtils.parse_order_by_for_search_evaluation_datasets(clause)
            )
            if key_type != "attribute":
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order_by entity: {key_type}"
                )
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        if offset < 0:
            raise MlflowException.invalid_parameter_value("Page offset must not be negative.")

        # No evaluation-dataset records can be created by this store yet.
        return PagedList([], None)

    def get_logged_model(self, model_id: str, allow_deleted: bool = False) -> LoggedModel:
        """Fetch model metadata and its complete associated metric history."""
        try:
            record = self._logged_model_repository.find_by_id(model_id)
            if record is None or (
                not allow_deleted and record.lifecycle_stage == LifecycleStage.DELETED
            ):
                raise MlflowException(
                    f"Logged model with ID '{model_id}' not found.", RESOURCE_DOES_NOT_EXIST
                )
            metrics = self._logged_model_repository.get_metric_history(record.model_id)
        except RepositoryPersistenceError:
            logger.exception("Unable to get logged model")
            raise MlflowException("Unable to get logged model.", INTERNAL_ERROR) from None

        return self._to_logged_model(record, metrics)

    def delete_logged_model(self, model_id: str) -> None:
        """Soft-delete a logged model and refresh its last-updated timestamp."""
        try:
            self._logged_model_repository.mark_deleted(
                model_id=model_id,
                last_updated_timestamp=get_current_time_millis(),
            )
        except LoggedModelNotFoundError:
            raise MlflowException(
                f"Logged model with ID '{model_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None
        except RepositoryPersistenceError:
            logger.exception("Unable to delete logged model")
            raise MlflowException("Unable to delete logged model.", INTERNAL_ERROR) from None

    def set_logged_model_tags(self, model_id: str, tags: list[LoggedModelTag]) -> None:
        """Set model tags, keeping the last value for each key in the batch."""
        tags_by_key = {tag.key: tag.value for tag in tags}
        if any(k is None or v is None for k, v in tags_by_key.items()):
            raise MlflowException(
                "Logged model tags must have non-null keys and values.", BAD_REQUEST
            )
        try:
            self._logged_model_repository.set_tags(model_id=model_id, tags=tags_by_key)
        except LoggedModelNotFoundError:
            raise MlflowException(
                f"Logged model with ID '{model_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None
        except RepositoryPersistenceError:
            logger.exception("Unable to set logged model tags")
            raise MlflowException("Unable to set logged model tags.", INTERNAL_ERROR) from None

    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        """Delete a model tag, failing if the model or tag does not exist."""
        try:
            self._logged_model_repository.delete_tag(model_id=model_id, key=key)
        except LoggedModelNotFoundError:
            raise MlflowException(
                f"Logged model with ID '{model_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None
        except LoggedModelTagNotFoundError:
            raise MlflowException(
                f"No tag with key {key!r} found for model with ID {model_id!r}.",
                RESOURCE_DOES_NOT_EXIST,
            ) from None
        except RepositoryPersistenceError:
            logger.exception("Unable to delete logged model tag")
            raise MlflowException("Unable to delete logged model tag.", INTERNAL_ERROR) from None

    # Prompt-to-run/model linking uses the tag methods above via the registry.

    def link_prompts_to_trace(self, trace_id: str, prompt_versions: list[PromptVersion]) -> None:
        """Associate prompt versions with a trace by their public name/version identity."""
        if not prompt_versions:
            return

        refs = [
            {"name": prompt_version.name, "version": str(prompt_version.version)}
            for prompt_version in prompt_versions
        ]
        try:
            self._trace_repository.link_prompts(trace_id=trace_id, prompt_versions=refs)
        except TraceNotFoundError:
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found.", RESOURCE_DOES_NOT_EXIST
            ) from None
        except RepositoryPersistenceError:
            logger.exception("Unable to link prompts to trace")
            raise MlflowException("A database operation failed.", INTERNAL_ERROR) from None

    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        raise NotImplementedError

    def unlink_traces_from_run(self, trace_ids: list[str], run_id: str) -> None:
        raise NotImplementedError
