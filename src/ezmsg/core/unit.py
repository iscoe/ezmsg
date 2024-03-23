import time
import inspect
import functools
from .stream import Stream, InputStream, OutputStream
from .component import ComponentMeta, Component
from .settings import Settings

from typing import Any, Dict, List, Tuple, Callable, Optional

import logging

logger = logging.getLogger("ezmsg")

MAIN_ATTR = "__ez_main__"
THREAD_ATTR = "__ez_thread__"
TASK_ATTR = "__ez_task__"
PUBLISHES_ATTR = "__ez_publishes__"
SUBSCRIBES_ATTR = "__ez_subscribes__"
TIMEIT_ATTR = "__ez_timeit__"
ZERO_COPY_ATTR = "__ez_zerocopy__"
PROCESS_ATTR = "__ez_process__"


class UnitMeta(ComponentMeta):
    def __init__(
        cls, name: str, bases: Tuple[type, ...], fields: Dict[str, Any], **kwargs: Any
    ) -> None:
        super(UnitMeta, cls).__init__(name, bases, fields)

        cls.__tasks__ = {}
        cls.__main__ = None
        cls.__threads__ = {}

        for base in bases:
            if hasattr(base, "__tasks__"):
                tasks = getattr(base, "__tasks__")
                for task_name, task in tasks.items():
                    cls.__tasks__[task_name] = task
            if hasattr(base, "__main__"):
                main = getattr(base, "__main__")
                if cls.__main__ is not None and main is not None:
                    raise Exception(
                        f"Attempting to replace {name}'s main func {cls.__main__} with {main}!"
                    )
                cls.__main__ = main
            if hasattr(base, "__threads__"):
                threads = getattr(base, "__threads__")
                for thread_name, thread in threads.items():
                    cls.__threads__[thread_name] = thread

        for field_name, field_value in fields.items():
            if callable(field_value):
                if hasattr(field_value, TASK_ATTR):
                    cls.__tasks__[field_name] = field_value
                elif hasattr(field_value, MAIN_ATTR):
                    cls.__main__ = field_value
                elif hasattr(field_value, THREAD_ATTR):
                    cls.__threads__[field_name] = field_value


class Unit(Component, metaclass=UnitMeta):
    """
    Represents a single step in the graph.
    Units can subscribe, publish, and have tasks.
    To create a ``Unit``, inherit from the ``Unit`` class.
    """

    def __init__(self, *args, settings: Optional[Settings] = None, **kwargs):
        super(Unit, self).__init__(*args, settings=settings, **kwargs)

        for task_name, task in self.__class__.__tasks__.items():
            self._tasks[task_name] = task
        self._main = self.__class__.__main__
        self._threads = self.__class__.__threads__

    async def setup(self):
        """This is called from within the same process this unit will live"""
        self._instantiate_state()
        if not self._settings_applied:
            raise ValueError(
                f"{self.address} has not had settings applied before initialization"
            )

        if inspect.iscoroutinefunction(self.initialize):
            await self.initialize()
        else:
            self.initialize() # type: ignore

        self._check_state()

    async def initialize(self) -> None:
        """
        Runs when the ``Unit`` is instantiated.
        This is called from within the same process this unit will live.
        This lifecycle hook can be overridden. It can be run as ``async`` functions by simply adding the
        ``async`` keyword when overriding.
        """
        pass

    async def shutdown(self) -> None:
        """
        Runs when the ``Unit`` terminates.
        This is called from within the same process this unit will live.
        This lifecycle hook can be overridden. It can be run as ``async`` functions by simply adding the
        ``async`` keyword when overriding.
        """
        pass


def publisher(stream: OutputStream):
    """
    A decorator for a method that publishes to a stream in the task/messaging thread.
    An async function will yield messages on the designated :obj:`OutputStream`.

    .. code-block:: python

      from typing import AsyncGenerator

      OUTPUT = OutputStream(ez.Message)

      @publisher(OUTPUT)
      async def send_message(self) -> AsyncGenerator:
         message = Message()
         yield(OUTPUT, message)

    A function can have both ``@subscriber`` and ``@publisher`` decorators.
    """

    if not isinstance(stream, OutputStream):
        raise ValueError(f"Cannot publish to object of type {type(stream)}")

    def pub_factory(func):
        published_streams: List[OutputStream] = getattr(func, PUBLISHES_ATTR, [])
        published_streams.append(stream)
        setattr(func, PUBLISHES_ATTR, published_streams)
        return task(func)

    return pub_factory


def subscriber(stream: InputStream, zero_copy: bool = False):
    """
    A decorator for a method that subscribes to a stream in the task/messaging thread.
    An async function will run once per message received from the :obj:`InputStream` it subscribes to.

    Example:

    .. code-block:: python

      INPUT = ez.InputStream(Message)

      @subscriber(INPUT)
      async def print_message(self, message: Message) -> None:
         print(message)

    A function can have both ``@subscriber`` and ``@publisher`` decorators.
    """

    if not isinstance(stream, InputStream):
        raise ValueError(f"Cannot subscribe to object of type {type(stream)}")

    def sub_factory(func):
        subscribed_streams: Optional[InputStream] = getattr(func, SUBSCRIBES_ATTR, None)
        if subscribed_streams is not None:
            raise Exception(f"{func} cannot subscribe to more than one stream")
        setattr(func, SUBSCRIBES_ATTR, stream)
        setattr(func, ZERO_COPY_ATTR, zero_copy)
        return task(func)

    return sub_factory


def main(func: Callable):
    """
    A decorator which designates this function to run as the main thread for this :obj:`Unit`.
    A :obj:`Unit` may only have one of these.
    """
    setattr(func, MAIN_ATTR, True)
    return func


def timeit(func: Callable):
    """
    ``ezmsg`` will log the amount of time this function takes to execute.
    """
    setattr(func, TIMEIT_ATTR, True)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        res = func(self, *args, **kwargs)
        stop = time.perf_counter()
        _, task_info = self._tasks[func.__name__]
        address = task_info.address
        logger.info(f"{address} task duration = {(stop-start)*1e3:0.4f}ms")
        return res

    return wrapper


def thread(func: Callable):
    """
    A decorator which designates this function to run as a background thread for this `:obj:`Unit`.
    """
    setattr(func, THREAD_ATTR, True)
    return func


def task(func: Callable):
    """
    A decorator which designates this function to run as a task in the task/messaging thread.
    """
    setattr(func, TASK_ATTR, True)
    return func


def process(func: Callable):
    """
    A decorator which designates this function to run in its own process.
    """
    setattr(func, PROCESS_ATTR, True)
    return func
