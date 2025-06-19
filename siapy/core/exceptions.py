"""Exceptions for SiaPy library.

This module defines custom exceptions used throughout the SiaPy library to handle
errors related to file handling, input validation, processing, and configuration.

"""

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
    """Base exception for SiaPy library.

    This is the base exception class for all custom exceptions in the SiaPy library.
    All other SiaPy exceptions inherit from this class.
    """

    def __init__(self, message: str, name: str = "SiaPy") -> None:
        """Initialize SiapyError exception.

        This is the base initialization method for all SiaPy custom exceptions.
        It stores the error message and component name for detailed error reporting.

        Args:
            message: The error message describing what went wrong.
            name: The name of the library/component raising the error. Defaults to "SiaPy".

        Example:
            ```python
            from siapy.core.exceptions import SiapyError

            raise SiapyError("Something went wrong", "MyComponent")
            ```
        """
        self.message: str = message
        self.name: str = name
        super().__init__(self.message, self.name)


class InvalidFilepathError(SiapyError):
    """Exception raised when a required file is not found.

    This exception is raised when attempting to access a file that does not exist
    or when a provided file path is invalid.
    """

    def __init__(self, filename: str | Path) -> None:
        """Initialize InvalidFilepathError exception.

        Creates an exception for when a required file cannot be found or accessed.
        The filename is converted to string format for consistent error messaging.

        Args:
            filename: The path to the file that was not found. Can be a string or Path object.

        Example:
            ```python
            from siapy.core.exceptions import InvalidFilepathError
            from pathlib import Path

            # Using string path
            raise InvalidFilepathError("/path/to/missing/file.txt")

            # Using Path object
            raise InvalidFilepathError(Path("missing_file.txt"))
            ```
        """
        self.filename: str = str(filename)
        super().__init__(f"File not found: {filename}")


class InvalidInputError(SiapyError):
    """Exception raised for invalid input.

    This exception is raised when the provided input value does not meet
    the expected criteria or validation rules.
    """

    def __init__(self, input_value: Any, message: str = "Invalid input") -> None:
        """Initialize InvalidInputError exception.

        Creates an exception for when input validation fails. The input value is stored
        for debugging purposes and included in the error message.

        Args:
            input_value: The invalid input value that caused the error.
            message: Custom error message. Defaults to "Invalid input".

        Example:
            ```python
            from siapy.core.exceptions import InvalidInputError

            # With default message
            raise InvalidInputError(-5)

            # With custom message
            raise InvalidInputError(-5, "Value must be non-negative")
            ```
        """
        self.input_value: Any = input_value
        self.message: str = message
        super().__init__(f"{message}: {input_value}")


class InvalidTypeError(SiapyError):
    """Exception raised for invalid type.

    This exception is raised when a value has an incorrect type that doesn't
    match the expected or allowed types for a particular operation.
    """

    def __init__(
        self,
        input_value: Any,
        allowed_types: type | tuple[type, ...],
        message: str = "Invalid type",
    ) -> None:
        """Initialize InvalidTypeError exception.

        Creates an exception for type validation failures. Stores the actual value,
        its type, and the allowed types for comprehensive error reporting.

        Args:
            input_value: The value with the invalid type.
            allowed_types: The type or tuple of types that are allowed for this value.
            message: Custom error message. Defaults to "Invalid type".

        Example:
            ```python
            from siapy.core.exceptions import InvalidTypeError

            # Single allowed type
            raise InvalidTypeError("text", int, "Expected integer")

            # Multiple allowed types
            raise InvalidTypeError("text", (int, float), "Expected numeric type")
            ```
        """
        self.input_value: Any = input_value
        self.input_type: Any = type(input_value)
        self.allowed_types: type | tuple[type, ...] = allowed_types
        self.message: str = message
        super().__init__(f"{message}: {input_value} (type: {self.input_type}). Allowed types: {allowed_types}")


class ProcessingError(SiapyError):
    """Exception raised for errors during processing.

    This exception is raised when an error occurs during data processing operations,
    such as image processing, data transformation, or computational tasks.
    """

    def __init__(self, message: str = "An error occurred during processing") -> None:
        """Initialize ProcessingError exception.

        Creates an exception for when errors occur during data processing operations.
        This is a general-purpose exception for computational or transformation failures.

        Args:
            message: Error message describing the processing failure.
                Defaults to "An error occurred during processing".

        Example:
            ```python
            from siapy.core.exceptions import ProcessingError

            # With default message
            raise ProcessingError()

            # With custom message
            raise ProcessingError("Failed to process image data")
            ```
        """
        self.message: str = message
        super().__init__(message)


class ConfigurationError(SiapyError):
    """Exception raised for configuration errors.

    This exception is raised when there are issues with configuration settings,
    invalid configuration parameters, or missing required configuration values.
    """

    def __init__(self, message: str = "Configuration error") -> None:
        """Initialize ConfigurationError exception.

        Creates an exception for configuration-related issues such as invalid settings,
        missing required parameters, or malformed configuration data.

        Args:
            message: Error message describing the configuration issue.
                Defaults to "Configuration error".

        Example:
            ```python
            from siapy.core.exceptions import ConfigurationError

            # With default message
            raise ConfigurationError()

            # With custom message
            raise ConfigurationError("Missing required parameter 'api_key'")
            ```
        """
        self.message: str = message
        super().__init__(message)


class MethodNotImplementedError(SiapyError):
    """Exception raised for not implemented methods.

    This exception is raised when a method that should be implemented in a subclass
    has not been implemented, typically in abstract base classes or interfaces.
    """

    def __init__(self, class_name: str, method_name: str) -> None:
        """Initialize MethodNotImplementedError exception.

        Creates an exception for when a required method has not been implemented,
        typically in abstract base classes or interface implementations.

        Args:
            class_name: The name of the class where the method is not implemented.
            method_name: The name of the method that is not implemented.

        Example:
            ```python
            from siapy.core.exceptions import MethodNotImplementedError

            class AbstractProcessor:
                def process(self):
                    raise MethodNotImplementedError("AbstractProcessor", "process")
            ```
        """
        self.class_name: str = class_name
        self.method_name: str = method_name
        super().__init__(f"Method '{method_name}' not implemented in class '{class_name}'")


class DirectInitializationError(SiapyError):
    """Exception raised when a class method is required to create an instance.

    This exception is raised when attempting to directly instantiate a class
    that requires the use of specific class methods for proper initialization.
    """

    def __init__(self, class_: type) -> None:
        """Initialize DirectInitializationError exception.

        Creates an exception for when a class requires the use of specific class methods
        for instantiation rather than direct initialization. Automatically discovers
        available class methods to suggest in the error message.

        Args:
            class_: The class type that cannot be directly initialized.

        Raises:
            ImportError: If the required utility function cannot be imported.

        Example:
            ```python
            from siapy.core.exceptions import DirectInitializationError

            class SpecialClass:
                def __init__(self):
                    raise DirectInitializationError(SpecialClass)

                @classmethod
                def from_file(cls, filepath):
                    return cls.__new__(cls)
            ```
        """
        from siapy.utils.general import get_classmethods

        self.class_name: str = class_.__class__.__name__
        self.class_methods: list[str] = get_classmethods(class_)
        super().__init__(
            f"Use any of the @classmethod to create a new instance of '{self.class_name}': {self.class_methods}"
        )
