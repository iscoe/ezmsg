from dataclasses import field
import os
import time
import typing

import numpy as np
import pytest

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messagecodec import message_log
from ezmsg.util.terminate import TerminateOnTotalSettings, TerminateOnTotal
from util import get_test_fn
from ezmsg.sigproc.synth import (
    clock, aclock, Clock, ClockSettings,
    Counter, CounterSettings
)


# TEST CLOCK
@pytest.mark.parametrize("dispatch_rate", [None, 2.0, 20.0])
def test_clock_gen(dispatch_rate: typing.Optional[float]):
    run_time = 1.0
    n_target = int(np.ceil(dispatch_rate * run_time)) if dispatch_rate else 100
    gen = clock(dispatch_rate=dispatch_rate)
    result = []
    t_start = time.time()
    while len(result) < n_target:
        result.append(next(gen))
    t_elapsed = time.time() - t_start
    assert all([_ == ez.Flag() for _ in result])
    if dispatch_rate is not None:
        assert (run_time - 1 / dispatch_rate) < t_elapsed < (run_time + 0.1)
    else:
        assert t_elapsed < (n_target * 1e-4)  # 100 usec per iteration is pretty generous


@pytest.mark.parametrize("dispatch_rate", [None, 2.0, 20.0])
@pytest.mark.asyncio
async def test_aclock_agen(dispatch_rate: typing.Optional[float]):
    run_time = 1.0
    n_target = int(np.ceil(dispatch_rate * run_time)) if dispatch_rate else 100
    agen = aclock(dispatch_rate=dispatch_rate)
    result = []
    t_start = time.time()
    while len(result) < n_target:
        new_result = await anext(agen)
        result.append(new_result)
    t_elapsed = time.time() - t_start
    assert all([_ == ez.Flag() for _ in result])
    if dispatch_rate:
        assert (run_time - 1 / dispatch_rate) < t_elapsed < (run_time + 0.1)
    else:
        assert t_elapsed < (n_target * 1e-4)  # 100 usec per iteration is pretty generous


class ClockTestSystemSettings(ez.Settings):
    clock_settings: ClockSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(default_factory=TerminateOnTotalSettings)


class ClockTestSystem(ez.Collection):
    SETTINGS: ClockTestSystemSettings

    CLOCK = Clock()
    LOG = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.CLOCK.apply_settings(self.SETTINGS.clock_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_CLOCK, self.LOG.INPUT_MESSAGE),
            (self.LOG.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE)
        )


@pytest.mark.parametrize("dispatch_rate", [None, 2.0, 20.0])
def test_clock_system(
        dispatch_rate: typing.Optional[float],
        test_name: typing.Optional[str] = None,
):
    run_time = 1.0
    n_target = int(np.ceil(dispatch_rate * run_time)) if dispatch_rate else 100
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)
    settings = ClockTestSystemSettings(
        clock_settings=ClockSettings(dispatch_rate=dispatch_rate),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateOnTotalSettings(total=n_target)
    )
    system = ClockTestSystem(settings)
    ez.run(SYSTEM=system)

    # Collect result
    messages: typing.List[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    assert all([_ == ez.Flag() for _ in messages])
    assert len(messages) >= n_target


# TEST COUNTER #
class CounterTestSystemSettings(ez.Settings):
    counter_settings: CounterSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(default_factory=TerminateOnTotalSettings)


class CounterTestSystem(ez.Collection):
    SETTINGS: CounterTestSystemSettings

    COUNTER = Counter()
    LOG = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.COUNTER.apply_settings(self.SETTINGS.counter_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.COUNTER.OUTPUT_SIGNAL, self.LOG.INPUT_MESSAGE),
            (self.LOG.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE)
        )


@pytest.mark.parametrize("block_size", [1, 20])
@pytest.mark.parametrize("fs", [10.0, 1000.0])
@pytest.mark.parametrize("n_ch", [3])
@pytest.mark.parametrize("dispatch_rate", [None, "realtime", 2.0, 20.0])  # "ext_clock" needs a separate test
@pytest.mark.parametrize("mod", [2**3, None])
def test_counter_system(
        block_size: int,
        fs: float,
        n_ch: int,
        dispatch_rate: typing.Optional[typing.Union[float, str]],
        mod: typing.Optional[int],
        test_name: typing.Optional[str] = None,
):
    target_dur = 2.6  # 2.6 seconds per test
    if dispatch_rate is None:
        # No sleep / wait
        chunk_dur = 0.1
    elif dispatch_rate == "realtime":
        chunk_dur = block_size / fs
    else:
        # Note: float dispatch_rate will yield different number of samples than expected by target_dur and fs
        chunk_dur = 1. / dispatch_rate
    target_messages = int(target_dur / chunk_dur)

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)
    settings = CounterTestSystemSettings(
        counter_settings=CounterSettings(
            n_time=block_size,
            fs=fs,
            n_ch=n_ch,
            dispatch_rate=dispatch_rate,
            mod=mod,
        ),
        log_settings=MessageLoggerSettings(
            output=test_filename,
        ),
        term_settings=TerminateOnTotalSettings(
            total=target_messages,
        )
    )
    system = CounterTestSystem(settings)
    ez.run(SYSTEM=system)

    # Collect result
    messages: typing.List[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    if dispatch_rate is None:
        # The number of messages depends on how fast the computer is
        target_messages = len(messages)
    # This should be an equivalence assertion (==) but the use of TerminateOnTotal does
    #  not guarantee that MessageLogger will exit before an additional message is received.
    #  Let's just clip the last message if we exceed the target messages.
    if len(messages) > target_messages:
        messages = messages[:target_messages]
    assert len(messages) == target_messages

    target_samples = block_size * target_messages
    time_idx, ch_idx = messages[0].get_axis_idx("time"), messages[0].get_axis_idx("ch")
    data = np.concatenate([_.data for _ in messages], axis=time_idx)
    assert data.shape[0] == target_samples
    assert data.shape[ch_idx] == n_ch
    expected_data = np.arange(target_samples)
    if mod is not None:
        expected_data = expected_data % mod
    assert np.array_equal(data[:, 0], expected_data)

    offsets = np.array([_.axes["time"].offset for _ in messages])
    expected_offsets = np.arange(target_messages) * block_size / fs
    if dispatch_rate is not None and dispatch_rate == "realtime":
        expected_offsets += offsets[0]  # offsets are in real-time
        atol = 0.002
    else:
        # Offsets are synthetic.
        atol = 1.e-8
    assert np.allclose(offsets[2:], expected_offsets[2:], atol=atol)
