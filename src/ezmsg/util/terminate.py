import asyncio
import time

from dataclasses import dataclass

import ezmsg.core as ez

from typing import Optional, Any


class TerminateOnTimeoutSettings(ez.Settings):
    """
    Settings for :obj:`TerminateOnTimeout` Unit.

    Args:
        time: Terminate if no message has been received in this time (sec)
        poll_rate: Hz.
    """
    time: float = 2.0
    poll_rate: float = 4.0


class TerminateOnTimeoutState(ez.State):
    last_msg_timestamp: Optional[float] = None


class TerminateOnTimeout(ez.Unit):
    """
    End a pipeline execution when a certain amount of time has passed without receiving a message.
    """

    SETTINGS: TerminateOnTimeoutSettings
    STATE: TerminateOnTimeoutState

    INPUT = ez.InputStream(Any)
    """Send messages here."""

    @ez.subscriber(INPUT)
    async def keepalive(self, _: Any) -> None:
        cur_timestamp = time.time()
        self.STATE.last_msg_timestamp = cur_timestamp

    @ez.task
    async def poll_terminate(self) -> None:
        while True:
            await asyncio.sleep(1.0 / self.SETTINGS.poll_rate)
            if self.STATE.last_msg_timestamp is not None:
                age = time.time() - self.STATE.last_msg_timestamp
                if age >= self.SETTINGS.time:
                    ez.logger.info("Raising NormalTermination in TerminateOnTimeout!")
                    raise ez.NormalTermination


class TerminateOnTotalSettings(ez.Settings):
    """
    Settings for :obj:`TerminateOnTotal` Unit.

    Args:
        total: The total number of messages to terminate after.
    """
    total: Optional[int] = None


class TerminateOnTotalState(ez.State):
    total: Optional[int]
    n_messages: int = 0


class TerminateOnTotal(ez.Unit):
    """
    End a pipeline execution once a certain number of messages have been received.
    """

    SETTINGS: TerminateOnTotalSettings
    STATE: TerminateOnTotalState

    INPUT_MESSAGE = ez.InputStream(Any)
    """Send messages here."""

    INPUT_TOTAL = ez.InputStream(int)
    """
    Change the total number of messages to terminate after.
    If this number has already been reached, termination will occur immediately.
    """

    def initialize(self) -> None:
        self.STATE.total = self.SETTINGS.total

    @ez.subscriber(INPUT_TOTAL)
    async def on_total(self, msg: int) -> None:
        self.STATE.total = msg
        self.maybe_terminate()

    @ez.subscriber(INPUT_MESSAGE)
    async def on_message(self, _: Any) -> None:
        self.STATE.n_messages += 1
        self.maybe_terminate()

    def maybe_terminate(self):
        if self.STATE.total is not None:
            if self.STATE.n_messages >= self.STATE.total:
                raise ez.NormalTermination
