from fastapi import HTTPException


class BLINGException(Exception):
    """Base exception for all Blue Team errors."""
    pass


class AuditWriteError(BLINGException):
    """Raised when audit log INSERT fails. Causes entire request to fail."""
    pass


class ScoringError(BLINGException):
    """Raised when detection pipeline fails."""
    pass


class GraphQueryError(BLINGException):
    """Raised when Neo4j query fails."""
    pass


class FeatureBuildError(BLINGException):
    """Raised when feature assembly fails."""
    pass


def alert_not_found(alert_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Alert not found")


def unauthorized() -> HTTPException:
    return HTTPException(status_code=401, detail="Invalid or missing API key")


def forbidden() -> HTTPException:
    return HTTPException(status_code=403, detail="API key not authorized for this endpoint")
