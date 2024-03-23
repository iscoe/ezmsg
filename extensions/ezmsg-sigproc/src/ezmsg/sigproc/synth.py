import asyncio
from collections import deque
from dataclasses import dataclass, replace, field
import time
from typing import Optional, Generator, AsyncGenerator, Union

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer, GenAxisArray
from ezmsg.util.messages.axisarray import AxisArray

from .butterworthfilter import ButterworthFilter, ButterworthFilterSettings


def clock(
    dispatch_rate: Optional[float]
) -> Generator[ez.Flag, None, None]:
    """
    Construct a generator that yields events at a specified rate.

    Args:
        dispatch_rate: event rate in seconds.

    Returns:
        A generator object that yields :obj:`ez.Flag` events at a specified rate.
    """
    n_dispatch = -1
    t_0 = time.time()
    while True:
        if dispatch_rate is not None:
            n_dispatch += 1
            t_next = t_0 + n_dispatch / dispatch_rate
            time.sleep(max(0, t_next - time.time()))
        yield ez.Flag()


async def aclock(
    dispatch_rate: Optional[float]
) -> AsyncGenerator[ez.Flag, None]:
    """
    ``asyncio`` version of :obj:`clock`.

    Returns:
        asynchronous generator object. Must use `anext` or `async for`.
    """
    t_0 = time.time()
    n_dispatch = -1
    while True:
        if dispatch_rate is not None:
            n_dispatch += 1
            t_next = t_0 + n_dispatch / dispatch_rate
            await asyncio.sleep(t_next - time.time())
        yield ez.Flag()


class ClockSettings(ez.Settings):
    """Settings for :obj:`Clock`. See :obj:`clock` for parameter description."""
    # Message dispatch rate (Hz), or None (fast as possible)
    dispatch_rate: Optional[float]


class ClockState(ez.State):
    cur_settings: ClockSettings
    gen: AsyncGenerator


class Clock(ez.Unit):
    """Unit for :obj:`clock`."""
    SETTINGS: ClockSettings
    STATE: ClockState

    INPUT_SETTINGS = ez.InputStream(ClockSettings)
    OUTPUT_CLOCK = ez.OutputStream(ez.Flag)

    def initialize(self) -> None:
        self.STATE.cur_settings = self.SETTINGS
        self.construct_generator()

    def construct_generator(self):
        self.STATE.gen = aclock(self.STATE.cur_settings.dispatch_rate)

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: ClockSettings) -> None:
        self.STATE.cur_settings = msg
        self.construct_generator()

    @ez.publisher(OUTPUT_CLOCK)
    async def generate(self) -> AsyncGenerator:
        while True:
            out = await self.STATE.gen.__anext__()
            if out:
                yield self.OUTPUT_CLOCK, out


# COUNTER - Generate incrementing integer. fs and dispatch_rate parameters combine to give many options. #
async def acounter(
    n_time: int,
    fs: Optional[float],
    n_ch: int = 1,
    dispatch_rate: Optional[Union[float, str]] = None,
    mod: Optional[int] = None,
) -> AsyncGenerator[AxisArray, None]:
    """
    Construct an asynchronous generator to generate AxisArray objects at a specified rate
    and with the specified sampling rate.

    NOTE: This module uses asyncio.sleep to delay appropriately in realtime mode.
    This method of sleeping/yielding execution priority has quirky behavior with
    sub-millisecond sleep periods which may result in unexpected behavior (e.g.
    fs = 2000, n_time = 1, realtime = True -- may result in ~1400 msgs/sec)

    Args:
        n_time: Number of samples to output per block.
        fs: Sampling rate of signal output in Hz.
        n_ch: Number of channels to synthesize
        dispatch_rate: Message dispatch rate (Hz), 'realtime' or None (fast as possible)
            Note: if dispatch_rate is a float then time offsets will be synthetic and the
            system will run faster or slower than wall clock time.
        mod: If set to an integer, counter will rollover at this number.

    Returns:
        An asynchronous generator.
    """

    # TODO: Adapt this to use ezmsg.util.rate?

    counter_start: int = 0  # next sample's first value

    b_realtime = False
    b_manual_dispatch = False
    b_ext_clock = False
    if dispatch_rate is not None:
        if isinstance(dispatch_rate, str):
            if dispatch_rate.lower() == "realtime":
                b_realtime = True
            elif dispatch_rate.lower() == "ext_clock":
                b_ext_clock = True
        else:
            b_manual_dispatch = True

    n_sent: int = 0  # It is convenient to know how many samples we have sent.
    clock_zero: float = time.time()  # time associated with first sample

    while True:
        # 1. Sleep, if necessary, until we are at the end of the current block
        if b_realtime:
            n_next = n_sent + n_time
            t_next = clock_zero + n_next / fs
            await asyncio.sleep(t_next - time.time())
        elif b_manual_dispatch:
            n_disp_next = 1 + n_sent / n_time
            t_disp_next = clock_zero + n_disp_next / dispatch_rate
            await asyncio.sleep(t_disp_next - time.time())

        # 2. Prepare counter data.
        block_samp = np.arange(counter_start, counter_start + n_time)[:, np.newaxis]
        if mod is not None:
            block_samp %= mod
        block_samp = np.tile(block_samp, (1, n_ch))

        # 3. Prepare offset - the time associated with block_samp[0]
        if b_realtime:
            offset = t_next - n_time / fs
        elif b_ext_clock:
            offset = time.time()
        else:
            # Purely synthetic.
            offset = n_sent / fs
            # offset += clock_zero  # ??

        # 4. yield output
        yield AxisArray(
            block_samp,
            dims=["time", "ch"],
            axes={"time": AxisArray.Axis.TimeAxis(fs=fs, offset=offset)},
        )

        # 5. Update state for next iteration (after next yield)
        counter_start = block_samp[-1, 0] + 1  # do not % mod
        n_sent += n_time


class CounterSettings(ez.Settings):
    # TODO: Adapt this to use ezmsg.util.rate?
    """
    Settings for :obj:`Counter`.
    See :obj:`acounter` for a description of the parameters.
    """

    n_time: int  # Number of samples to output per block
    fs: float  # Sampling rate of signal output in Hz
    n_ch: int = 1  # Number of channels to synthesize

    # Message dispatch rate (Hz), 'realtime', 'ext_clock', or None (fast as possible)
    #  Note: if dispatch_rate is a float then time offsets will be synthetic and the
    #  system will run faster or slower than wall clock time.
    dispatch_rate: Optional[Union[float, str]] = None

    # If set to an integer, counter will rollover
    mod: Optional[int] = None


class CounterState(ez.State):
    gen: AsyncGenerator[AxisArray, Optional[ez.Flag]]
    cur_settings: CounterSettings
    new_generator: asyncio.Event


class Counter(ez.Unit):
    """Generates monotonically increasing counter. Unit for :obj:`acounter`."""

    SETTINGS: CounterSettings
    STATE: CounterState

    INPUT_CLOCK = ez.InputStream(ez.Flag)
    INPUT_SETTINGS = ez.InputStream(CounterSettings)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    async def initialize(self) -> None:
        self.STATE.new_generator = asyncio.Event()
        self.validate_settings(self.SETTINGS)

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: CounterSettings) -> None:
        self.validate_settings(msg)

    def validate_settings(self, settings: CounterSettings) -> None:
        if isinstance(
            settings.dispatch_rate, str
        ) and self.SETTINGS.dispatch_rate not in ["realtime", "ext_clock"]:
            raise ValueError(f"Unknown dispatch_rate: {self.SETTINGS.dispatch_rate}")
        self.STATE.cur_settings = settings
        self.construct_generator()

    def construct_generator(self):
        self.STATE.gen = acounter(
            self.STATE.cur_settings.n_time,
            self.STATE.cur_settings.fs,
            n_ch=self.STATE.cur_settings.n_ch,
            dispatch_rate=self.STATE.cur_settings.dispatch_rate,
            mod=self.STATE.cur_settings.mod
        )
        self.STATE.new_generator.set()
    
    @ez.subscriber(INPUT_CLOCK)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_clock(self, clock: ez.Flag):
        if self.STATE.cur_settings.dispatch_rate == 'ext_clock':
            out = await self.STATE.gen.__anext__()
            yield self.OUTPUT_SIGNAL, out

    @ez.publisher(OUTPUT_SIGNAL)
    async def run_generator(self) -> AsyncGenerator:
        while True:
    
            await self.STATE.new_generator.wait()
            self.STATE.new_generator.clear()
                
            if self.STATE.cur_settings.dispatch_rate == 'ext_clock':
                continue
            
            while not self.STATE.new_generator.is_set():
                out = await self.STATE.gen.__anext__()
                yield self.OUTPUT_SIGNAL, out


@consumer
def sin(
    axis: Optional[str] = "time",
    freq: float = 1.0,
    amp: float = 1.0,
    phase: float = 0.0,
) -> Generator[AxisArray, AxisArray, None]:
    """
    Construct a generator of sinusoidal waveforms in AxisArray objects.

    Args:
        axis: The name of the axis over which the sinusoid passes.
        freq: The frequency of the sinusoid, in Hz.
        amp: The amplitude of the sinusoid.
        phase: The initial phase of the sinusoid, in radians.

    Returns:
        A primed generator that expects .send(axis_array) of sample counts
        and yields an AxisArray of sinusoids.
    """
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    ang_freq = 2.0 * np.pi * freq

    while True:
        axis_arr_in = yield axis_arr_out
        # axis_arr_in is expected to be sample counts

        axis_name = axis
        if axis_name is None:
            axis_name = axis_arr_in.dims[0]

        w = (ang_freq * axis_arr_in.get_axis(axis_name).gain) * axis_arr_in.data
        out_data = amp * np.sin(w + phase)
        axis_arr_out = replace(axis_arr_in, data=out_data)


class SinGeneratorSettings(ez.Settings):
    """
    Settings for :obj:`SinGenerator`.
    See :obj:`sin` for parameter descriptions.
    """
    time_axis: Optional[str] = "time"
    freq: float = 1.0  # Oscillation frequency in Hz
    amp: float = 1.0  # Amplitude
    phase: float = 0.0  # Phase offset (in radians)


class SinGenerator(GenAxisArray):
    """
    Unit for :obj:`sin`.
    """
    SETTINGS: SinGeneratorSettings

    def construct_generator(self):
        self.STATE.gen = sin(
            axis=self.SETTINGS.time_axis,
            freq=self.SETTINGS.freq,
            amp=self.SETTINGS.amp,
            phase=self.SETTINGS.phase
        )


class OscillatorSettings(ez.Settings):
    """Settings for :obj:`Oscillator`"""
    n_time: int
    """Number of samples to output per block."""

    fs: float
    """Sampling rate of signal output in Hz"""

    n_ch: int = 1
    """Number of channels to output per block"""

    dispatch_rate: Optional[Union[float, str]] = None
    """(Hz) | 'realtime' | 'ext_clock'"""

    freq: float = 1.0
    """Oscillation frequency in Hz"""

    amp: float = 1.0
    """Amplitude"""

    phase: float = 0.0
    """Phase offset (in radians)"""

    sync: bool = False
    """Adjust `freq` to sync with sampling rate"""


class Oscillator(ez.Collection):
    """
    :obj:`Collection that chains :obj:`Counter` and :obj:`SinGenerator`.
    """
    SETTINGS: OscillatorSettings

    INPUT_CLOCK = ez.InputStream(ez.Flag)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    COUNTER = Counter()
    SIN = SinGenerator()

    def configure(self) -> None:
        # Calculate synchronous settings if necessary
        freq = self.SETTINGS.freq
        mod = None
        if self.SETTINGS.sync:
            period = 1.0 / self.SETTINGS.freq
            mod = round(period * self.SETTINGS.fs)
            freq = 1.0 / (mod / self.SETTINGS.fs)

        self.COUNTER.apply_settings(
            CounterSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate=self.SETTINGS.dispatch_rate,
                mod=mod,
            )
        )

        self.SIN.apply_settings(
            SinGeneratorSettings(
                freq=freq, amp=self.SETTINGS.amp, phase=self.SETTINGS.phase
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_CLOCK, self.COUNTER.INPUT_CLOCK),
            (self.COUNTER.OUTPUT_SIGNAL, self.SIN.INPUT_SIGNAL),
            (self.SIN.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )


class RandomGeneratorSettings(ez.Settings):
    loc: float = 0.0
    """loc argument for :obj:`numpy.random.normal`"""

    scale: float = 1.0
    """scale argument for :obj:`numpy.random.normal`"""


class RandomGenerator(ez.Unit):
    """
    Replaces input data with random data and yields the result.
    """
    SETTINGS: RandomGeneratorSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    async def generate(self, msg: AxisArray) -> AsyncGenerator:
        random_data = np.random.normal(
            size=msg.shape, loc=self.SETTINGS.loc, scale=self.SETTINGS.scale
        )

        yield self.OUTPUT_SIGNAL, replace(msg, data=random_data)


class NoiseSettings(ez.Settings):
    """
    See :obj:`CounterSettings` and :obj:`RandomGeneratorSettings`.
    """
    n_time: int  # Number of samples to output per block
    fs: float  # Sampling rate of signal output in Hz
    n_ch: int = 1  # Number of channels to output
    dispatch_rate: Optional[
        Union[float, str]
    ] = None  # (Hz), 'realtime', or 'ext_clock'
    loc: float = 0.0  # DC offset
    scale: float = 1.0  # Scale (in standard deviations)


WhiteNoiseSettings = NoiseSettings


class WhiteNoise(ez.Collection):
    """
    A :obj:`Collection` that chains a :obj:`Counter` and :obj:`RandomGenerator`.
    """
    SETTINGS: NoiseSettings

    INPUT_CLOCK = ez.InputStream(ez.Flag)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    COUNTER = Counter()
    RANDOM = RandomGenerator()

    def configure(self) -> None:
        self.RANDOM.apply_settings(
            RandomGeneratorSettings(loc=self.SETTINGS.loc, scale=self.SETTINGS.scale)
        )

        self.COUNTER.apply_settings(
            CounterSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate=self.SETTINGS.dispatch_rate,
                mod=None,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_CLOCK, self.COUNTER.INPUT_CLOCK),
            (self.COUNTER.OUTPUT_SIGNAL, self.RANDOM.INPUT_SIGNAL),
            (self.RANDOM.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )


PinkNoiseSettings = NoiseSettings


class PinkNoise(ez.Collection):
    """
    A :obj:`Collection` that chains :obj:`WhiteNoise` and :obj:`ButterworthFilter`.
    """
    SETTINGS: PinkNoiseSettings

    INPUT_CLOCK = ez.InputStream(ez.Flag)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    WHITE_NOISE = WhiteNoise()
    FILTER = ButterworthFilter()

    def configure(self) -> None:
        self.WHITE_NOISE.apply_settings(self.SETTINGS)
        self.FILTER.apply_settings(
            ButterworthFilterSettings(
                axis="time", order=1, cutoff=self.SETTINGS.fs * 0.01  # Hz
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_CLOCK, self.WHITE_NOISE.INPUT_CLOCK),
            (self.WHITE_NOISE.OUTPUT_SIGNAL, self.FILTER.INPUT_SIGNAL),
            (self.FILTER.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )


class AddState(ez.State):
    queue_a: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)
    queue_b: "asyncio.Queue[AxisArray]" = field(default_factory=asyncio.Queue)


class Add(ez.Unit):
    """Add two signals together.  Assumes compatible/similar axes/dimensions."""

    STATE: AddState

    INPUT_SIGNAL_A = ez.InputStream(AxisArray)
    INPUT_SIGNAL_B = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    @ez.subscriber(INPUT_SIGNAL_A)
    async def on_a(self, msg: AxisArray) -> None:
        self.STATE.queue_a.put_nowait(msg)

    @ez.subscriber(INPUT_SIGNAL_B)
    async def on_b(self, msg: AxisArray) -> None:
        self.STATE.queue_b.put_nowait(msg)

    @ez.publisher(OUTPUT_SIGNAL)
    async def output(self) -> AsyncGenerator:
        while True:
            a = await self.STATE.queue_a.get()
            b = await self.STATE.queue_b.get()

            yield (self.OUTPUT_SIGNAL, replace(a, data=a.data + b.data))


class EEGSynthSettings(ez.Settings):
    """See :obj:`OscillatorSettings`."""
    fs: float = 500.0  # Hz
    n_time: int = 100
    alpha_freq: float = 10.5  # Hz
    n_ch: int = 8


class EEGSynth(ez.Collection):
    """
    A :obj:`Collection` that chains a :obj:`Clock` to both :obj:`PinkNoise`
    and :obj:`Oscillator`, then :obj:`Add` s the result.
    """
    SETTINGS: EEGSynthSettings

    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    CLOCK = Clock()
    NOISE = PinkNoise()
    OSC = Oscillator()
    ADD = Add()

    def configure(self) -> None:
        self.CLOCK.apply_settings(
            ClockSettings(dispatch_rate=self.SETTINGS.fs / self.SETTINGS.n_time)
        )

        self.OSC.apply_settings(
            OscillatorSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate="ext_clock",
                freq=self.SETTINGS.alpha_freq,
            )
        )

        self.NOISE.apply_settings(
            PinkNoiseSettings(
                n_time=self.SETTINGS.n_time,
                fs=self.SETTINGS.fs,
                n_ch=self.SETTINGS.n_ch,
                dispatch_rate="ext_clock",
                scale=5.0,
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_CLOCK, self.OSC.INPUT_CLOCK),
            (self.CLOCK.OUTPUT_CLOCK, self.NOISE.INPUT_CLOCK),
            (self.OSC.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_A),
            (self.NOISE.OUTPUT_SIGNAL, self.ADD.INPUT_SIGNAL_B),
            (self.ADD.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
