"""Sample Python module used as a fixture for code_chunker tests."""

import os
from typing import Any

CONSTANT = 42
_PRIVATE = "hidden"


def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


async def async_function(data: list[Any]) -> list[Any]:
    return [item for item in data if item is not None]


class MyClass:
    """A simple class."""

    class_var: int = 0

    def __init__(self, value: int) -> None:
        self.value = value

    def method(self) -> int:
        return self.value * 2

    @staticmethod
    def static_method(n: int) -> bool:
        return n > 0


class AnotherClass(MyClass):
    def method(self) -> int:
        return super().method() + 1


MODULE_LEVEL_DICT = {"key": "value", "env": os.environ.get("HOME", "")}
