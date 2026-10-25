"""Exceptions for EBM models."""

import enum
from collections.abc import Collection
from typing import Self


class InvalidOptionError(ValueError):
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


class EBMGraphError(Exception):
    """Exception raised for errors in the EBM graph processing."""


class EBMGraphSimplificationError(EBMGraphError):
    """Exception raised when an EBM graph cannot be simplified enough to fit into the max_tokens limit."""


class EBMGraphTokenLimitError(EBMGraphError):
    """Exception raised when an EBM graph exceeds the max_tokens limit even after simplification."""
