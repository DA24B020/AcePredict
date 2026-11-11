"""Exception hierarchy. Each error carries a stable code and the HTTP status the API maps it to."""

from __future__ import annotations


class AcePredictError(Exception):
    """Base class for every expected AcePredict error."""

    code = "INTERNAL_ERROR"
    http_status = 500

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code


class DataValidationError(AcePredictError):
    """Raw data failed a critical quality check; the pipeline must stop."""

    code = "DATA_VALIDATION_FAILED"
    http_status = 422


class DataSourceError(AcePredictError):
    """The data source could not be downloaded."""

    code = "DATA_SOURCE_ERROR"
    http_status = 502


class PlayerNotFoundError(AcePredictError):
    """A player id is not in the current snapshot."""

    code = "PLAYER_NOT_FOUND"
    http_status = 404


class InvalidMatchupError(AcePredictError):
    """The requested match is impossible (same player twice, mixed tours, ...)."""

    code = "INVALID_MATCHUP"
    http_status = 400


class SnapshotNotReadyError(AcePredictError):
    """The serving snapshot has not been built yet (run the pipeline)."""

    code = "SNAPSHOT_NOT_READY"
    http_status = 503


class ModelUnavailableError(AcePredictError):
    """The model server is down, or no champion model exists yet."""

    code = "MODEL_UNAVAILABLE"
    http_status = 503


class ModelServerError(AcePredictError):
    """The model server answered with an error."""

    code = "MODEL_SERVER_ERROR"
    http_status = 502


class PipelineBusyError(AcePredictError):
    """A pipeline run is already queued or running."""

    code = "PIPELINE_BUSY"
    http_status = 409


class RegistryError(AcePredictError):
    """The MLflow model registry could not complete the request."""

    code = "REGISTRY_ERROR"
    http_status = 502


class AuthError(AcePredictError):
    """Missing or wrong API key / webhook token."""

    code = "UNAUTHORIZED"
    http_status = 401
