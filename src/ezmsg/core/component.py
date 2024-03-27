from abc import ABCMeta
from copy import deepcopy
from dataclasses import fields, MISSING, is_dataclass

from .settings import Settings
from .state import State
from .addressable import Addressable
from .stream import Stream

from typing import List, Optional, Dict, Tuple, Any, Callable


class ComponentMeta(ABCMeta):
    def __init__(
        cls, name: str, bases: Tuple[type, ...], fields: Dict[str, Any], **kwargs: Any
    ) -> None:
        super(ComponentMeta, cls).__init__(name, bases, fields)

        cls.__settings_type__ = Settings
        cls.__state_type__ = State

        for base in bases:
            if (
                cls.__settings_type__ is not Settings
                and hasattr(base, "__settings_type__")
                and getattr(base, "__settings_type__") is not Settings
            ):
                # This means class settings has already been set by a base class.
                raise Exception(
                    f"Class {name} already has settings of type {cls.__settings_type__}, "
                    + f"but is trying to be set to {getattr(base, '__settings_type__')}"
                )
            if (
                hasattr(base, "__settings_type__")
                and getattr(base, "__settings_type__") is not Settings
            ):
                cls.__settings_type__ = getattr(base, "__settings_type__")
            if (
                cls.__state_type__ is not State
                and hasattr(base, "__state_type__")
                and getattr(base, "__state_type__") is not State
            ):
                # This means class state has already been set by a base class.
                raise Exception(
                    f"Class {name} already has state of type {cls.__state_type__}, "
                    + f"but is trying to be set to {getattr(base, '__state_type__')}"
                )
            if (
                hasattr(base, "__state_type")
                and getattr(base, "__state_type__") is not State
            ):
                cls.__state_type__ = getattr(base, "__state_type__")

        if "SETTINGS" in cls.__annotations__:
            cls.__settings_type__ = cls.__annotations__["SETTINGS"]

        if "STATE" in cls.__annotations__:
            cls.__state_type__ = cls.__annotations__["STATE"]

        cls.__streams__ = {}

        for base in bases:
            if hasattr(base, "__streams__"):
                streams = getattr(base, "__streams__")
                for stream_name, stream in streams.items():
                    cls.__streams__[stream_name] = stream

        for field_name, field_value in fields.items():
            if isinstance(field_value, Stream):
                field_value._set_name(field_name)
                cls.__streams__[field_name] = field_value


class Component(Addressable, metaclass=ComponentMeta):
    """
    Metaclass which :obj:`Unit` and :obj:`Collection` inherit from.
    """

    SETTINGS: Settings
    STATE: State

    _tasks: Dict[str, Callable]  # Only Units will have tasks
    _streams: Dict[str, Stream]  # All Components can have streams
    _components: Dict[str, "Component"]  # Only Collections will have components
    _main: Optional[Callable[..., None]]
    _threads: Dict[str, Callable]

    def __init__(self, *args, settings: Optional[Settings] = None, **kwargs):
        super(Component, self).__init__()

        self.SETTINGS = None  # type: ignore
        self.STATE = None  # type: ignore
        self._settings_applied = False
        self._components = {}
        self._tasks = {}
        self._main = None
        self._threads = {}

        self._streams = deepcopy(self.__class__.__streams__)
        for stream_name, stream in self.streams.items():
            setattr(self, stream_name, stream)

        if settings is None:
            # settings not supplied as a kwarg. Try to build it.
            if len(args) > 0 and type(args[0]) == self.__class__.__settings_type__:
                settings = args[0]
            elif len(args) > 0 or len(kwargs) > 0:
                settings = self.__class__.__settings_type__(*args, **kwargs)
            else:
                try:
                    # If we weren't supplied settings, we will try to
                    # instantiate the settings type from annotations
                    settings = self.__class__.__settings_type__()
                except TypeError:
                    # We couldn't instantiate settings with default value
                    # We will rely on late configuration via apply_settings
                    pass

        if settings is not None:
            self.apply_settings(settings)

    def _instantiate_state(self) -> None:
        assert self.STATE is None
        self.STATE = self.__class__.__state_type__()
        assert is_dataclass(self.STATE)
        for field in fields(self.STATE):
            if field.default_factory is not MISSING and field.default is MISSING:
                setattr(self.STATE, field.name, field.default_factory())

    def _check_state(self) -> None:
        assert is_dataclass(self.STATE.__class__)
        for field in fields(self.STATE.__class__):
            if not hasattr(self.STATE, field.name):
                raise AttributeError(
                    f"{self.address}: STATE.{field.name} was not initialized!"
                )

    def apply_settings(self, settings: Settings) -> None:
        """
        Update the ``Component``‘s ``Settings`` object.

        Args:
            settings: An instance of the class-specific ``Settings``.

        """
        self.SETTINGS = settings
        self._settings_applied = True

    def _set_location(self, location: Optional[List[str]] = None):
        super(Component, self)._set_location(location)

        # Percolate the location down to submodules and streams
        for comp in self.components.values():
            comp._set_location(self.location + [self.name])
        for stream in self.streams.values():
            stream._set_location(self.location + [self.name])

    @property
    def tasks(self) -> Dict[str, Callable]:
        return self._tasks

    @property
    def streams(self) -> Dict[str, Stream]:
        return self._streams

    @property
    def components(self) -> Dict[str, "Component"]:
        return self._components

    @property
    def main(self) -> Optional[Callable[..., None]]:
        return self._main

    @property
    def threads(self) -> Dict[str, Callable]:
        return self._threads
