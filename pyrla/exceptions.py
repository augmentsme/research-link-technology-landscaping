"""
Exception classes for PyRLA
"""


class RLAError(Exception):
    """Base exception class for PyRLA"""
    pass


class RLAAuthenticationError(RLAError):
    """Raised when authentication fails"""
    pass


class RLANotFoundError(RLAError):
    """Raised when a resource is not found"""
    pass


class RLAValidationError(RLAError):
    """Raised when request validation fails"""
    pass


class RLAServerError(RLAError):
    """Raised when server returns 5xx error"""
    pass
