from dataclasses import field, replace

import os
import json

import pytest
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import ezmsg.core as ez

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.messagegate import MessageGate, MessageGateSettings
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messagecodec import message_log
from ezmsg.sigproc.synth import Counter, CounterSettings
from ezmsg.sigproc.window import Window, WindowSettings, windowing

from util import get_test_fn
from ezmsg.util.terminate import TerminateOnTimeout as TerminateTest
from ezmsg.util.terminate import TerminateOnTimeoutSettings as TerminateTestSettings
from ezmsg.util.debuglog import DebugLog

from typing import Optional, Dict, Any, List, Tuple


def calculate_expected_results(orig, fs, win_shift, zero_pad, msg_block_size, shift_len, win_len, nchans, data_len, n_msgs, win_ax):
    # For the calculation, we assume time_ax is last then transpose if necessary at the end.
    expected = orig.copy()
    tvec = np.arange(orig.shape[1]) / fs
    # Prepend the data with zero-padding, if necessary.
    if win_shift is None or zero_pad == "input":
        n_cut = msg_block_size
    elif zero_pad == "shift":
        n_cut = shift_len
    else:  # "none" -- no buffer needed
        n_cut = win_len
    n_keep = win_len - n_cut
    if n_keep > 0:
        expected = np.concatenate((np.zeros((nchans, win_len))[..., -n_keep:], expected), axis=-1)
        tvec = np.hstack(((np.arange(-win_len, 0) / fs)[-n_keep:], tvec))
    # Moving window -- assumes step size of 1
    expected = sliding_window_view(expected, win_len, axis=-1)
    tvec = sliding_window_view(tvec, win_len)
    # Mimic win_shift
    if win_shift is None:
        # 1:1 mode. Each input (block) yields a new output.
        # If the window length is smaller than the block size then we only the tail of each block.
        first = max(min(msg_block_size, data_len) - win_len, 0)
        if tvec[::msg_block_size].shape[0] < n_msgs:
            expected = np.concatenate((expected[:, first::msg_block_size], expected[:, -1:]), axis=1)
            tvec = np.hstack((tvec[first::msg_block_size, 0], tvec[-1:, 0]))
        else:
            expected = expected[:, first::msg_block_size]
            tvec = tvec[first::msg_block_size, 0]
    else:
        expected = expected[:, ::shift_len]
        tvec = tvec[::shift_len, 0]
    # Transpose to put time_ax and win_ax in the correct locations.
    if win_ax == 0:
        expected = np.moveaxis(expected, 0, -1)

    return expected, tvec


def test_window_gen_nodur():
    """
    Test window generator method when window_dur is None. Should be a simple pass through.
    """
    nchans = 64
    data_len = 20
    data = np.arange(nchans * data_len, dtype=float).reshape((nchans, data_len))
    test_msg = AxisArray(
        data=data,
        dims=["ch", "time"],
        axes={"time": AxisArray.Axis.TimeAxis(fs=500., offset=0.)}
    )
    gen = windowing(window_dur=None)
    result = gen.send(test_msg)
    assert result is test_msg
    assert np.shares_memory(result.data, test_msg.data)


@pytest.mark.parametrize("msg_block_size", [1, 5, 10, 20, 60])
@pytest.mark.parametrize("newaxis", [None, "win"])
@pytest.mark.parametrize("win_dur", [0.2, 1.0])
@pytest.mark.parametrize("win_shift", [None, 0.1, 1.0])
@pytest.mark.parametrize("zero_pad", ["input", "shift", "none"])
@pytest.mark.parametrize("fs", [10.0, 500.0])
@pytest.mark.parametrize("time_ax", [0, 1])
def test_window_generator(
        msg_block_size: int,
        newaxis: Optional[str],
        win_dur: float,
        win_shift: Optional[float],
        zero_pad: str,
        fs: float,
        time_ax: int
):
    nchans = 3

    shift_len = int(win_shift * fs) if win_shift is not None else None
    win_len = int(win_dur * fs)
    data_len = 2 * win_len
    if win_shift is not None:
        data_len += shift_len - 1
    data = np.arange(nchans * data_len, dtype=float).reshape((nchans, data_len))
    # Below, we transpose the individual messages if time_ax == 0.
    tvec = np.arange(data_len) / fs

    n_msgs = int(np.ceil(data_len / msg_block_size))

    # Instantiate the generator function
    gen = windowing(axis="time", newaxis=newaxis, window_dur=win_dur, window_shift=win_shift, zero_pad_until=zero_pad)

    # Create inputs and send them to the generator, collecting the results along the way.
    test_msg = AxisArray(
        data[..., ()],
        dims=["ch", "time"] if time_ax == 1 else ["time", "ch"],
        axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=0.)}
    )
    results = []
    for msg_ix in range(n_msgs):
        msg_data = data[..., msg_ix * msg_block_size:(msg_ix+1) * msg_block_size]
        if time_ax == 0:
            msg_data = np.ascontiguousarray(msg_data.T)
        test_msg = replace(test_msg, data=msg_data, axes={
            "time": AxisArray.Axis.TimeAxis(fs=fs, offset=tvec[msg_ix * msg_block_size])
        })
        win_msg = gen.send(test_msg)
        results.append(win_msg)

    # Check each return value's metadata (offsets checked at end)
    expected_dims = test_msg.dims[:time_ax] + [newaxis or "win"] + test_msg.dims[time_ax:]
    for msg in results:
        assert msg.axes["time"].gain == 1/fs
        assert msg.dims == expected_dims
        assert (newaxis or "win") in msg.axes
        assert msg.axes[(newaxis or "win")].gain == 0.0 if win_shift is None else shift_len / fs

    # Post-process the results to yield a single data array and a single vector of offsets.
    win_ax = time_ax
    time_ax = win_ax + 1
    result = np.concatenate([_.data for _ in results], win_ax)
    offsets = np.hstack([
        _.axes[newaxis or "win"].offset + _.axes[newaxis or "win"].gain * np.arange(_.data.shape[win_ax])
        for _ in results
    ])

    # Calculate the expected results for comparison.
    expected, tvec = calculate_expected_results(data, fs, win_shift, zero_pad, msg_block_size, shift_len, win_len,
                                                nchans, data_len, n_msgs, win_ax)

    # Compare results to expected
    assert np.array_equal(result, expected)
    assert np.allclose(offsets, tvec)


class WindowSystemSettings(ez.Settings):
    num_msgs: int
    counter_settings: CounterSettings
    window_settings: WindowSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateTestSettings = field(default_factory=TerminateTestSettings)


class WindowSystem(ez.Collection):
    COUNTER = Counter()
    GATE = MessageGate()
    WIN = Window()
    LOG = MessageLogger()
    TERM = TerminateTest()

    DEBUG = DebugLog()

    SETTINGS: WindowSystemSettings

    def configure(self) -> None:
        self.COUNTER.apply_settings(self.SETTINGS.counter_settings)
        self.GATE.apply_settings(
            MessageGateSettings(
                start_open=True,
                default_open=False,
                default_after=self.SETTINGS.num_msgs,
            )
        )
        self.WIN.apply_settings(self.SETTINGS.window_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.COUNTER.OUTPUT_SIGNAL, self.GATE.INPUT),
            # ( self.COUNTER.OUTPUT_SIGNAL, self.DEBUG.INPUT ),
            (self.GATE.OUTPUT, self.WIN.INPUT_SIGNAL),
            # ( self.GATE.OUTPUT, self.DEBUG.INPUT ),
            (self.WIN.OUTPUT_SIGNAL, self.LOG.INPUT_MESSAGE),
            # ( self.WIN.OUTPUT_SIGNAL, self.DEBUG.INPUT ),
            (self.LOG.OUTPUT_MESSAGE, self.TERM.INPUT),
            # ( self.LOG.OUTPUT_MESSAGE, self.DEBUG.INPUT ),
        )


# It takes >15 minutes to go through the full set of combinations tested for the generator.
# We need only test a subset to assert integration is correct.
@pytest.mark.parametrize("msg_block_size, newaxis, win_dur, win_shift, zero_pad, fs", [
    (1, None, 0.2, None, "input", 10.0),
    (20, None, 0.2, None, "input", 10.0),
    (1, "step", 0.2, None, "input", 10.0),
    (10, "step", 0.2, 1.0, "shift", 500.0),
    (20, "step", 1.0, 1.0, "shift", 500.0),
    (10, "step", 1.0, 1.0, "none", 500.0),
    (20, None, None, None, "input", 10.0),
])
def test_window_system(
    msg_block_size: int,
    newaxis: Optional[str],
    win_dur: float,
    win_shift: Optional[float],
    zero_pad: str,
    fs: float,
    test_name: Optional[str] = None,
):
    # Calculate expected dimensions.
    win_len = int((win_dur or 1.0) * fs)
    shift_len = int(win_shift * fs) if win_shift is not None else msg_block_size
    # num_msgs should be the greater value between (2 full windows + a shift) or 4.0 seconds
    data_len = max(2 * win_len + shift_len - 1, int(4.0 * fs))
    num_msgs = int(np.ceil(data_len / msg_block_size))

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    settings = WindowSystemSettings(
        num_msgs=num_msgs,
        counter_settings=CounterSettings(
            n_time=msg_block_size,
            fs=fs,
            dispatch_rate=float(num_msgs),  # Get through them in about 1 second.
        ),
        window_settings=WindowSettings(
            axis="time",
            newaxis=newaxis,
            window_dur=win_dur,
            window_shift=win_shift,
            zero_pad_until=zero_pad
        ),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateTestSettings(time=1.0),  # sec
    )

    system = WindowSystem(settings)
    ez.run(SYSTEM=system)

    messages: List[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)
    ez.logger.info(f"Analyzing recording of { len( messages ) } messages...")

    # Within a test config, the metadata should not change across messages.
    for msg in messages:
        # In this test, fs should never change
        assert 1.0 / msg.axes["time"].gain == fs
        # In this test, we should have consistent dimensions
        assert msg.dims == ([newaxis, "time", "ch"] if newaxis else ["time", "ch"])
        # Window should always output the same shape data
        assert msg.shape[msg.get_axis_idx("ch")] == 1  # Counter yields only one channel.
        assert msg.shape[msg.get_axis_idx("time")] == (msg_block_size if win_dur is None else win_len)

    ez.logger.info("Consistent metadata!")

    # Collect the outputs we want to test
    data: List[np.ndarray] = [msg.data for msg in messages]
    if newaxis is None:
        offsets = np.array([_.axes["time"].offset for _ in messages])
    else:
        offsets = np.hstack([
            _.axes[newaxis].offset + _.axes[newaxis].gain * np.arange(_.data.shape[0])
            for _ in messages
        ])

    # If this test was performed in "one-to-one" mode, we should
    # have one window output per message pushed to Window
    if win_shift is None:
        assert len(data) == num_msgs

    # Turn the data into a ndarray.
    if newaxis is not None:
        data = np.concatenate(data, axis=messages[0].get_axis_idx(newaxis))
    else:
        data = np.stack(data, axis=messages[0].get_axis_idx("time"))

    # Calculate the expected results for comparison.
    sent_data = np.arange(num_msgs * msg_block_size)[None, :]
    expected, tvec = calculate_expected_results(sent_data, fs, win_shift, zero_pad, msg_block_size, shift_len, win_len,
                                                1, data_len, num_msgs, 0)

    # Compare results to expected
    if win_dur is None:
        assert np.array_equal(data, sent_data.reshape((num_msgs, msg_block_size, -1)))
    else:
        assert np.array_equal(data, expected)
        assert np.allclose(offsets, tvec)

    ez.logger.info("Test Complete.")


if __name__ == "__main__":
    test_window_system(5, 0.6, None, test_name="test_window_system")
