"""Module defining a metaclass to protect class constants from modification."""

class ConstProtector(type):
    """
        Prevent modification of existing class attributes (constants).

        Args:
            cls (type): The class being modified (automatically passed by the metaclass).
            name (str): The attribute name being set.
            value (Any): The new value being assigned.

        Raises:
            AttributeError: If attempting to modify an existing class attribute.
    """
    def __setattr__(cls: type, name: str, value: any) -> None:
        if name in cls.__dict__:
            raise AttributeError(f"Cannot modify constant {name}")
        # if the class variable does not exist, allow setting it
        super().__setattr__(name, value)