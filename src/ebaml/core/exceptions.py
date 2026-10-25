"""Exceptions for EBM models.

Every error that ``ebaml`` raises on purpose derives from ``EBaMlError``, so ``except EBaMlError`` catches all of them.
An error for a bad value also derives from ``ValueError``, so code that catches ``ValueError`` keeps working.
"""

import enum
from collections.abc import Collection
from typing import Self


class EBaMlError(Exception):
    """The base class of the errors that ``ebaml`` raises."""


class InvalidOptionError(EBaMlError, ValueError):
    """Exception raised for invalid options."""

    def __init__(self, option: str, valid_options: Collection[str]) -> None:
        """Build the error message from the invalid option and the valid ones.

        Parameters
        ----------
        option : str
            The invalid option provided.
        valid_options : Collection[str]
            The options that are valid.

        """
        message = f"Invalid option: {option}. Valid options are {valid_options}."
        super().__init__(message)

    @classmethod
    def from_enum(cls, option: str, validation_enum: type[enum.Enum]) -> Self:
        """Create an InvalidOptionError from an enum.

        Parameters
        ----------
        option : str
            The invalid option provided.
        validation_enum : type[enum.Enum]
            The enum class that defines the valid options.

        Returns
        -------
        InvalidOptionError
            An instance of InvalidOptionError with a message indicating the invalid option and the valid options.

        """
        valid_options = list(validation_enum._member_names_)
        return cls(option, valid_options)


class OptionRangeError(EBaMlError, ValueError):
    """Exception raised when a number option is outside its range."""


class EBMGraphError(EBaMlError):
    """Exception raised for errors in the EBM graph processing."""


class EBMGraphSimplificationError(EBMGraphError):
    """Exception raised when an EBM graph cannot be simplified enough to fit into the max_tokens limit."""


class EBMGraphTokenLimitError(EBMGraphError):
    """Exception raised when an EBM graph exceeds the max_tokens limit even after simplification."""


class EBMGraphShapeError(EBMGraphError, ValueError):
    """Exception raised when the bins of an EBM graph do not fit the scores or the requested format."""


class GraphTextError(EBaMlError, ValueError):
    """Exception raised when the textual representation of a graph cannot be read."""


class UnsupportedModelError(EBaMlError, ValueError):
    """Exception raised when a function does not support the kind of EBM, such as one with interactions."""


class InvalidInstanceError(EBaMlError, ValueError):
    """Exception raised when the instance to explain lacks a value or has a value that the EBM did not see."""
