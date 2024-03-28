from dataclasses import replace
import traceback
from typing import AsyncGenerator, Optional, Generator

import numpy as np

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer
import ezmsg.core as ez


@consumer
def downsample(
        axis: Optional[str] = None, factor: int = 1
) -> Generator[AxisArray, AxisArray, None]:
    """
    Construct a generator that yields a downsampled version of the data .send() to it.
    Downsampled data simply comprise every `factor`th sample.
    This should only be used following appropriate lowpass filtering.
    If your pipeline does not already have lowpass filtering then consider
    using the :obj:`Decimate` collection instead.

    Args:
        axis: The name of the axis along which to downsample.
        factor: Downsampling factor.

    Returns:
        A primed generator object ready to receive a `.send(axis_array)`
        and yields the downsampled data.
        Note that if a send chunk does not have sufficient samples to reach the
        next downsample interval then `None` is yielded.

    """
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    # state variables
    s_idx = 0

    while True:
        axis_arr_in = yield axis_arr_out

        if axis is None:
            axis = axis_arr_in.dims[0]
        axis_info = axis_arr_in.get_axis(axis)
        axis_idx = axis_arr_in.get_axis_idx(axis)

        samples = np.arange(axis_arr_in.data.shape[axis_idx]) + s_idx
        samples = samples % factor
        s_idx = samples[-1] + 1

        pub_samples = np.where(samples == 0)[0]
        if len(pub_samples) > 0:
            new_axes = {ax_name: axis_arr_in.get_axis(ax_name) for ax_name in axis_arr_in.dims}
            new_offset = axis_info.offset + (axis_info.gain * pub_samples[0].item())
            new_gain = axis_info.gain * factor
            new_axes[axis] = replace(axis_info, gain=new_gain, offset=new_offset)
            down_data = np.take(axis_arr_in.data, pub_samples, axis=axis_idx)
            axis_arr_out = replace(axis_arr_in, data=down_data, dims=axis_arr_in.dims, axes=new_axes)
        else:
            axis_arr_out = None


class DownsampleSettings(ez.Settings):
    """
    Settings for :obj:`Downsample` node.
    See :obj:`downsample` documentation for a description of the parameters.
    """
    axis: Optional[str] = None
    factor: int = 1


class DownsampleState(ez.State):
    cur_settings: DownsampleSettings
    gen: Generator


class Downsample(ez.Unit):
    """
    :obj:`Unit` for :obj:`downsample`.
    """
    SETTINGS: DownsampleSettings
    STATE: DownsampleState

    INPUT_SETTINGS = ez.InputStream(DownsampleSettings)
    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    def construct_generator(self):
        self.STATE.gen = downsample(axis=self.STATE.cur_settings.axis, factor=self.STATE.cur_settings.factor)

    def initialize(self) -> None:
        self.STATE.cur_settings = self.SETTINGS
        self.construct_generator()

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: DownsampleSettings) -> None:
        self.STATE.cur_settings = msg
        self.construct_generator()

    @ez.subscriber(INPUT_SIGNAL, zero_copy=True)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_signal(self, msg: AxisArray) -> AsyncGenerator:
        if self.STATE.cur_settings.factor < 1:
            raise ValueError("Downsample factor must be at least 1 (no downsampling)")

        try:
            out_msg = self.STATE.gen.send(msg)
            if out_msg is not None:
                yield self.OUTPUT_SIGNAL, out_msg
        except (StopIteration, GeneratorExit):
            ez.logger.debug(f"Downsample closed in {self.address}")
        except Exception:
            ez.logger.info(traceback.format_exc())
