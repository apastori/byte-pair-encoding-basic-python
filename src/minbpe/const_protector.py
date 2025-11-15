"""Module defining a metaclass to protect class constants from modification."""

from abc import ABCMeta
from typing import Any


class ConstProtector(ABCMeta):
    """
    Prevent modification of existing class attributes (constants).

    Args:
        cls (type): The class being modified (automatically passed by the metaclass).
        name (str): The attribute name being set.
        value (Any): The new value being assigned.

    Raises:
        AttributeError: If attempting to modify an existing class attribute.
    """

    def __setattr__(cls, name: str, value: Any) -> None:
        if name in cls.__dict__:
            raise AttributeError(f"Cannot modify constant {name}")
        # if the class variable does not exist, allow setting it
        # I am going to include comment to ignore the next line for the ruff linter
        super(type, cls).__setattr__(name, value)  # noqa: UP008
