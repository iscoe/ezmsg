import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
import traceback
from typing import Any, AsyncGenerator, Generator, Callable, TypeVar
from typing_extensions import ParamSpec
from functools import wraps, reduce

# Define type variables for the decorators
P = ParamSpec("P")
Y = TypeVar("Y")
S = TypeVar("S")


def consumer(
    func: Callable[P, Generator[Y, S, None]]
) -> Callable[P, Generator[Y, S, None]]:
    """
    A decorator that primes a generator by advancing it to the first yield statement.

    This is necessary because a generator cannot receive any input through 'send' until
    it has been primed to the point of the first 'yield'. Applying this decorator to a
    generator function ensures that it is immediately ready to accept input.

    Args:
        func: The generator function to be decorated.

    Returns:
        The primed generator ready to accept input.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[Y, S, None]:
        gen = func(*args, **kwargs)
        next(gen)  # Prime the generator.
        return gen

    return wrapper


def compose(*funcs):
    """
    Compose a chain of generator functions into a single callable.

    Each generator in the sequence is passed the value from the previous one, starting
    with the initial value `x`. This allows for creating a pipeline of data
    transformations that are applied in sequence when the returned callable is invoked.

    Args:
        *funcs: A variable number of generator functions to be composed.

    Returns:
        A callable that, when called with an initial value, will pass that
        value through the chain of generators.
    """
    return lambda x: reduce(lambda f, g: g.send(f), list(funcs), x)


# State class to hold the generator instance
class GenState(ez.State):
    gen: Generator


# Abstract Unit class that uses a generator for processing messages
class Gen(ez.Unit):
    STATE: GenState

    INPUT = ez.InputStream(Any)
    OUTPUT = ez.OutputStream(Any)

    def initialize(self) -> None:
        self.construct_generator()

    # Method to be implemented by subclasses to construct the specific generator
    def construct_generator(self):
        raise NotImplementedError

    # Subscriber method that sends incoming messages to
    # the generator and publishes the result
    @ez.subscriber(INPUT)
    @ez.publisher(OUTPUT)
    async def on_message(self, message: Any) -> AsyncGenerator:
        try:
            ret = self.STATE.gen.send(message)
            if ret is not None:
                yield self.OUTPUT, ret
        except (StopIteration, GeneratorExit):
            ez.logger.debug(f"Generator closed in {self.address}")
        except Exception:
            ez.logger.info(traceback.format_exc())


class GenAxisArray(ez.Unit):
    STATE: GenState

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    INPUT_SETTINGS = ez.InputStream(ez.Settings)

    def initialize(self) -> None:
        self.construct_generator()

    # Method to be implemented by subclasses to construct the specific generator
    def construct_generator(self):
        raise NotImplementedError

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: ez.Settings) -> None:
        self.apply_settings(msg)
        self.construct_generator()

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_message(self, message: AxisArray) -> AsyncGenerator:
        try:
            ret = self.STATE.gen.send(message)
            if ret is not None:
                yield self.OUTPUT_SIGNAL, ret
        except (StopIteration, GeneratorExit):
            ez.logger.debug(f"Generator closed in {self.address}")
        except Exception:
            ez.logger.info(traceback.format_exc())

