from abc import ABC, ABCMeta
from dataclasses import dataclass

from typing import (
    Dict,
    Tuple,
    Any,
    Type,
)


class StateMeta(ABCMeta):
    def __new__(
        cls,
        name: str,
        bases: Tuple[type, ...],
        classdict: Dict[str, Any],
        **kwargs: Any
    ) -> Type["State"]:
        new_cls = super().__new__(cls, name, bases, classdict)
        return dataclass(unsafe_hash=True, frozen=False, init=False)(new_cls)  # type: ignore


class State(ABC, metaclass=StateMeta):
    """
    States are mutable dataclasses that are instantiated by the Unit in its home process.

    To track a mutable state for a ``Component``, inherit from ``State``.

    .. code-block:: python

       class YourState(State):
          state1: int
          state2: float

    To use, declare the ``State`` object for a ``Component`` as a member variable called (all-caps!) ``STATE``.
    ``ezmsg`` will monitor the variable called ``STATE`` in the background, so it is important to name it correctly.

    Member functions can then access and mutate ``STATE`` as needed during function execution.
    It is recommended to initialize state values inside the ``initialize()`` or ``configure()`` lifecycle hooks if
    defaults are not defined.

    .. code-block:: python

       class YourUnit(Unit):

          STATE: YourState

          def initialize(self):
             this.STATE.state1 = 0
             this.STATE.state2 = 0.0

    .. note::
       ``State`` uses type hints to define member variables, but does not enforce type checking.
    """
    ...
