import typing
from dataclasses import dataclass

import ezmsg.core as ez


@dataclass
class GateMessage:
    """Send this message to ``INPUT_GATE`` to open or close the gate."""
    open: bool


class MessageGateSettings(ez.Settings):
    """
    Settings for :obj:`MessageGate` unit.

    Args:
        start_open: sets the gate's initial state to allow messages to flow through or be discarded. ``True`` will
            allow messages to flow through initially, ``False`` will discard messages initially.
        default_open: sets the gate's behavior after the `default_after` number of messages have flowed through.
            ``True`` will allow messages to flow through, ``False`` will discard messages.
        default_after: sets the number of messages after which the `default_open` state will be applied.
    """
    start_open: bool = False
    default_open: bool = False
    default_after: typing.Optional[int] = None


class MessageGateState(ez.State):
    gate_open: typing.Optional[bool] = None
    msgs: int = 0  # Messages since last gate change


class MessageGate(ez.Unit):
    """
    Blocks ``Messages`` from continuing through the system.
    Can be set as open, closed, open after n messages, or closed after n messages.
    """

    SETTINGS: MessageGateSettings
    STATE: MessageGateState

    INPUT_GATE = ez.InputStream(GateMessage)
    """
    Stop or start message flow. If ``GateMessage.open == True``, messages will flow through.
    If ``GateMessage.open == False``, messages will be discarded.
    """

    INPUT = ez.InputStream(typing.Any)
    """Messages which will flow through or be discarded, depending on gate status."""

    OUTPUT = ez.OutputStream(typing.Any)
    """Publishes messages which flow through."""

    def initialize(self) -> None:
        self.STATE.gate_open = self.SETTINGS.start_open

    def set_gate(self, set_open: bool) -> None:
        if self.STATE.gate_open != set_open:
            self.STATE.msgs = 0
            self.STATE.gate_open = set_open

    @ez.subscriber(INPUT_GATE)
    async def on_gate(self, msg: GateMessage) -> None:
        self.set_gate(msg.open)

    @ez.subscriber(INPUT)
    @ez.publisher(OUTPUT)
    async def on_input(self, msg: typing.Any) -> typing.AsyncGenerator:
        self.STATE.msgs += 1

        if self.STATE.gate_open:
            yield (self.OUTPUT, msg)

        if (  # Auto-revert to default state if necessary
            (self.SETTINGS.default_after is not None)
            and (self.STATE.gate_open != self.SETTINGS.default_open)
            and (self.STATE.msgs >= self.SETTINGS.default_after)
        ):
            self.set_gate(self.SETTINGS.default_open)