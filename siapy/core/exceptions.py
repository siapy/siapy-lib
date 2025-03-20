from pathlib import Path
from typing import Any

__all__ = [
    "SiapyError",
    "InvalidFilepathError",
    "InvalidInputError",
    "InvalidTypeError",
    "ProcessingError",
    "ConfigurationError",
    "MethodNotImplementedError",
    "DirectInitializationError",
]


class SiapyError(Exception):
    """Base exception for SiaPy library."""

    def __init__(self, message: str, name: str = "SiaPy") -> None:
        self.message: str = message
        self.name: str = name
        super().__init__(self.message, self.name)


class InvalidFilepathError(SiapyError):
    """Exception raised when a required file is not found."""

    def __init__(self, filename: str | Path) -> None:
        self.filename: str = str(filename)
        super().__init__(f"File not found: {filename}")


class InvalidInputError(SiapyError):
    """Exception raised for invalid input."""

    def __init__(self, input_value: Any, message: str = "Invalid input") -> None:
        self.input_value: Any = input_value
        self.message: str = message
        super().__init__(f"{message}: {input_value}")


class InvalidTypeError(SiapyError):
    """Exception raised for invalid type."""

    def __init__(
        self,
        input_value: Any,
        allowed_types: type | tuple[type, ...],
        message: str = "Invalid type",
    ) -> None:
        self.input_value: Any = input_value
        self.input_type: Any = type(input_value)
        self.allowed_types: type | tuple[type, ...] = allowed_types
        self.message: str = message
        super().__init__(f"{message}: {input_value} (type: {self.input_type}). Allowed types: {allowed_types}")


class ProcessingError(SiapyError):
    """Exception raised for errors during processing."""

    def __init__(self, message: str = "An error occurred during processing") -> None:
        self.message: str = message
        super().__init__(message)


class ConfigurationError(SiapyError):
    """Exception raised for configuration errors."""

    def __init__(self, message: str = "Configuration error") -> None:
        self.message: str = message
        super().__init__(message)


class MethodNotImplementedError(SiapyError):
    """Exception raised for not implemented methods."""

    def __init__(self, class_name: str, method_name: str) -> None:
        self.class_name: str = class_name
        self.method_name: str = method_name
        super().__init__(f"Method '{method_name}' not implemented in class '{class_name}'")


class DirectInitializationError(SiapyError):
    """Exception raised when a class method is required to create an instance."""

    def __init__(self, class_: type) -> None:
        from siapy.utils.general import get_classmethods

        self.class_name: str = class_.__class__.__name__
        self.class_methods: list[str] = get_classmethods(class_)
        super().__init__(
            f"Use any of the @classmethod to create a new instance of '{self.class_name}': {self.class_methods}"
        )
