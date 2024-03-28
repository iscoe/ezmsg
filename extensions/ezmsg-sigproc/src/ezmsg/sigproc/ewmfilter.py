import asyncio
from dataclasses import replace

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray

import numpy as np

from .window import Window, WindowSettings

from typing import AsyncGenerator, Optional


class EWMSettings(ez.Settings):
    axis: Optional[str] = None
    """Name of the axis to accumulate."""

    zero_offset: bool = True
    """If true, we assume zero DC offset for input data."""


class EWMState(ez.State):
    buffer_queue: "asyncio.Queue[AxisArray]"
    signal_queue: "asyncio.Queue[AxisArray]"


class EWM(ez.Unit):
    """
    Exponentially Weighted Moving Average Standardization

    References https://stackoverflow.com/a/42926270
    """

    SETTINGS: EWMSettings
    STATE: EWMState

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    INPUT_BUFFER = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def initialize(self) -> None:
        self.STATE.signal_queue = asyncio.Queue()
        self.STATE.buffer_queue = asyncio.Queue()

    @ez.subscriber(INPUT_SIGNAL)
    async def on_signal(self, message: AxisArray) -> None:
        self.STATE.signal_queue.put_nowait(message)

    @ez.subscriber(INPUT_BUFFER)
    async def on_buffer(self, message: AxisArray) -> None:
        self.STATE.buffer_queue.put_nowait(message)

    @ez.publisher(OUTPUT_SIGNAL)
    async def sync_output(self) -> AsyncGenerator:
        while True:
            signal = await self.STATE.signal_queue.get()
            buffer = await self.STATE.buffer_queue.get()  # includes signal

            axis_name = self.SETTINGS.axis
            if axis_name is None:
                axis_name = signal.dims[0]

            axis_idx = signal.get_axis_idx(axis_name)

            buffer_len = buffer.shape[axis_idx]
            block_len = signal.shape[axis_idx]
            window = buffer_len - block_len

            alpha = 2 / (window + 1.0)
            alpha_rev = 1 - alpha

            pows = alpha_rev ** (np.arange(buffer_len + 1))
            scale_arr = 1 / pows[:-1]
            pw0 = alpha * alpha_rev ** (buffer_len - 1)

            buffer_data = buffer.data
            buffer_data = np.moveaxis(buffer_data, axis_idx, 0)

            while scale_arr.ndim < buffer_data.ndim:
                scale_arr = scale_arr[..., None]

            def ewma(data: np.ndarray) -> np.ndarray:
                mult = scale_arr * data * pw0
                out = scale_arr[::-1] * mult.cumsum(axis=0)

                if not self.SETTINGS.zero_offset:
                    out = (data[0, :, np.newaxis] * pows[1:]).T + out

                return out

            mean = ewma(buffer_data)
            std = ewma((buffer_data - mean) ** 2.0)

            standardized = (buffer_data - mean) / np.sqrt(std).clip(1e-4)
            standardized = standardized[-signal.shape[axis_idx] :, ...]
            standardized = np.moveaxis(standardized, axis_idx, 0)

            yield self.OUTPUT_SIGNAL, replace(signal, data=standardized)


class EWMFilterSettings(ez.Settings):
    history_dur: float
    """Previous data to accumulate for standardization."""

    axis: Optional[str] = None
    """Name of the axis to accumulate."""

    zero_offset: bool = True
    """If true, we assume zero DC offset for input data."""


class EWMFilter(ez.Collection):
    """
    A :obj:`Collection` that splits the input into a branch that
    leads to :obj:`Window` which then feeds into :obj:`EWM` 's INPUT_BUFFER
    and another branch that feeds directly into :obj:`EWM` 's INPUT_SIGNAL.

    Consider :obj:`scaler` for a more efficient alternative.
    """
    SETTINGS: EWMFilterSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    WINDOW = Window()
    EWM = EWM()

    def configure(self) -> None:
        self.EWM.apply_settings(
            EWMSettings(
                axis=self.SETTINGS.axis, 
                zero_offset=self.SETTINGS.zero_offset,
            )
        )

        self.WINDOW.apply_settings(
            WindowSettings(
                axis=self.SETTINGS.axis,
                window_dur=self.SETTINGS.history_dur,
                window_shift=None,  # 1:1 mode
            )
        )

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.WINDOW.INPUT_SIGNAL),
            (self.WINDOW.OUTPUT_SIGNAL, self.EWM.INPUT_BUFFER),
            (self.INPUT_SIGNAL, self.EWM.INPUT_SIGNAL),
            (self.EWM.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
