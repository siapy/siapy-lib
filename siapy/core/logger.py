"""Logging configuration for SiaPy library.

This module provides a centralized logger instance for the entire SiaPy library.
All modules should use this logger for consistent logging behavior.
"""

import logging

__all__ = ["logger"]

logger = logging.getLogger("siapy")
