from dataclasses import replace
import traceback
from typing import AsyncGenerator, Optional, Tuple, List, Generator

import ezmsg.core as ez
import numpy as np
import numpy.typing as npt

from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis, sliding_win_oneaxis
from ezmsg.util.generator import consumer


@consumer
def windowing(
    axis: Optional[str] = None,
    newaxis: Optional[str] = None,
    window_dur: Optional[float] = None,
    window_shift: Optional[float] = None,
    zero_pad_until: str = "input"
) -> Generator[AxisArray, List[AxisArray], None]:
    """
    Construct a generator that yields windows of data from an input :obj:`AxisArray`.

    Args:
        axis: The axis along which to segment windows.
            If None, defaults to the first dimension of the first seen AxisArray.
        newaxis: Optional new axis for the output. If None, no new axes will be added.
            If a string, windows will be stacked in a new axis with key `newaxis`, immediately preceding the windowed axis.
        window_dur: The duration of the window in seconds.
            If None, the function acts as a passthrough and all other parameters are ignored.
        window_shift: The shift of the window in seconds.
            If None (default), windowing operates in "1:1 mode", where each input yields exactly one most-recent window.
        zero_pad_until: Determines how the function initializes the buffer.
            Can be one of "input" (default), "full", "shift", or "none". If `window_shift` is None then this field is
            ignored and "input" is always used.

            - "input" (default) initializes the buffer with the input then prepends with zeros to the window size.
              The first input will always yield at least one output.
            - "shift" fills the buffer until `window_shift`.
              No outputs will be yielded until at least `window_shift` data has been seen.
            - "none" does not pad the buffer. No outputs will be yielded until at least `window_dur` data has been seen.

    Returns:
        A (primed) generator that accepts .send(an AxisArray object) and yields a list of windowed
        AxisArray objects. The list will always be length-1 if `newaxis` is not None or `window_shift` is None.
    """
    # TODO: The return should be an AxisArray. i.e., always add a new axis. The Unit can do a multi-yield-per-pub
    #  if the parameterization does not expect a newaxis.

    if window_shift is None and zero_pad_until != "input":
        ez.logger.warning("`zero_pad_until` must be 'input' if `window_shift` is None. "
                          f"Ignoring received argument value: {zero_pad_until}")
        zero_pad_until = "input"
    elif window_shift is not None and zero_pad_until == "input":
        ez.logger.warning("windowing is non-deterministic with `zero_pad_until='input'` as it depends on the size "
                          "of the first input. We recommend using 'shift' when `window_shift` is float-valued.")
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = [AxisArray(np.array([]), dims=[""])]

    # State variables
    prev_samp_shape: Optional[Tuple[int, ...]] = None
    prev_fs: Optional[float] = None
    buffer: Optional[npt.NDArray] = None
    window_samples: Optional[int] = None
    window_shift_samples: Optional[int] = None
    shift_deficit: int = 0  # Number of incoming samples to ignore. Only relevant when shift > window.
    newaxis_warn_flag: bool = False
    mod_ax: Optional[str] = None  # The key of the modified axis in the output's .axes
    out_template: Optional[AxisArray] = None  # Template for building return values.

    while True:
        axis_arr_in = yield axis_arr_out

        if window_dur is None:
            axis_arr_out = [axis_arr_in]
            continue

        if axis is None:
            axis = axis_arr_in.dims[0]
        axis_idx = axis_arr_in.get_axis_idx(axis)
        axis_info = axis_arr_in.get_axis(axis)
        fs = 1.0 / axis_info.gain

        if (not newaxis_warn_flag) and newaxis is not None and newaxis in axis_arr_in.dims:
            ez.logger.warning(f"newaxis {newaxis} present in input dims and will be ignored.")
            newaxis_warn_flag = True
        b_newaxis = newaxis is not None and newaxis not in axis_arr_in.dims

        samp_shape = axis_arr_in.data.shape[:axis_idx] + axis_arr_in.data.shape[axis_idx + 1:]
        window_samples = int(window_dur * fs)
        b_1to1 = window_shift is None
        if not b_1to1:
            window_shift_samples = int(window_shift * fs)

        # If buffer unset or input stats changed, create a new buffer
        if buffer is None or samp_shape != prev_samp_shape or fs != prev_fs:
            if zero_pad_until == "none":
                req_samples = window_samples
            elif zero_pad_until == "shift" and not b_1to1:
                req_samples = window_shift_samples
            else:  # i.e. zero_pad_until == "input"
                req_samples = axis_arr_in.data.shape[axis_idx]
            n_zero = max(0, window_samples - req_samples)
            buffer_shape = axis_arr_in.data.shape[:axis_idx] + (n_zero,) + axis_arr_in.data.shape[axis_idx + 1:]
            buffer = np.zeros(buffer_shape)
            prev_samp_shape = samp_shape
            prev_fs = fs

        # Add new data to buffer.
        # Currently we just concatenate the new time samples and clip the output
        # np.roll actually returns a copy, and there's no way to construct a
        # rolling view of the data.  In current numpy implementations, np.concatenate
        # is generally faster than np.roll and slicing anyway, but this could still
        # be a performance bottleneck for large memory arrays.
        buffer = np.concatenate((buffer, axis_arr_in.data), axis=axis_idx)
        # Note: if we ever move to using a circular buffer without copies then we need to create copies somewhere,
        #  because currently the outputs are merely views into the buffer.

        # Create a vector of buffer timestamps to track axis `offset` in output(s)
        buffer_offset = np.arange(buffer.shape[axis_idx]).astype(float)
        # Adjust so first _new_ sample at index 0
        buffer_offset -= buffer_offset[-axis_arr_in.data.shape[axis_idx]]
        # Convert form indices to 'units' (probably seconds).
        buffer_offset *= axis_info.gain
        buffer_offset += axis_info.offset

        if not b_1to1 and shift_deficit > 0:
            n_skip = min(buffer.shape[axis_idx], shift_deficit)
            if n_skip > 0:
                buffer = slice_along_axis(buffer, np.s_[n_skip:], axis_idx)
                buffer_offset = buffer_offset[n_skip:]
                shift_deficit -= n_skip

        # Prepare reusable parts of output
        if out_template is None:
            out_dims = axis_arr_in.dims
            if newaxis is None:
                out_axes = {
                    **axis_arr_in.axes,
                    axis: replace(axis_info, offset=0.0)  # offset modified below.
                }
                mod_ax = axis
            else:
                out_dims = out_dims[:axis_idx] + [newaxis] + out_dims[axis_idx:]
                out_axes = {
                    **axis_arr_in.axes,
                    newaxis: AxisArray.Axis(
                        unit=axis_info.unit,
                        gain=0.0 if b_1to1 else axis_info.gain * window_shift_samples,
                        offset=0.0  # offset modified below
                    )
                }
                mod_ax = newaxis
            out_template = replace(axis_arr_in, data=np.zeros([0 for _ in out_dims]), dims=out_dims)

        # Generate outputs.
        axis_arr_out: List[AxisArray] = []
        if b_1to1:
            # one-to-one mode -- Each send yields exactly one window containing only the most recent samples.
            buffer = slice_along_axis(buffer, np.s_[-window_samples:], axis_idx)
            axis_arr_out.append(replace(
                    out_template,
                    data=np.expand_dims(buffer, axis=axis_idx) if b_newaxis else buffer,
                    axes={
                        **out_axes,
                        mod_ax: replace(out_axes[mod_ax], offset=buffer_offset[-window_samples])
                    }
            ))
        elif buffer.shape[axis_idx] >= window_samples:
            # Deterministic window shifts.
            win_view = sliding_win_oneaxis(buffer, window_samples, axis_idx)
            win_view = slice_along_axis(win_view, np.s_[::window_shift_samples], axis_idx)
            offset_view = sliding_win_oneaxis(buffer_offset, window_samples, 0)[::window_shift_samples]
            # Place in output
            if b_newaxis:
                axis_arr_out.append(replace(
                    out_template,
                    data=win_view,
                    axes={**out_axes, mod_ax: replace(out_axes[mod_ax], offset=offset_view[0, 0])}
                ))
            else:
                for win_ix in range(win_view.shape[axis_idx]):
                    axis_arr_out.append(replace(
                        out_template,
                        data=slice_along_axis(win_view, win_ix, axis_idx),
                        axes={
                            **out_axes,
                            mod_ax: replace(out_axes[mod_ax], offset=offset_view[win_ix, 0])
                        }
                    ))

            # Drop expired beginning of buffer and update shift_deficit
            multi_shift = window_shift_samples * win_view.shape[axis_idx]
            shift_deficit = max(0, multi_shift - buffer.shape[axis_idx])
            buffer = slice_along_axis(buffer, np.s_[multi_shift:], axis_idx)


class WindowSettings(ez.Settings):
    axis: Optional[str] = None
    newaxis: Optional[str] = None  # new axis for output. No new axes if None
    window_dur: Optional[float] = None  # Sec. passthrough if None
    window_shift: Optional[float] = None  # Sec. Use "1:1 mode" if None
    zero_pad_until: str = "full"  # "full", "shift", "input", "none"


class WindowState(ez.State):
    cur_settings: WindowSettings
    gen: Generator


class Window(ez.Unit):
    STATE: WindowState
    SETTINGS: WindowSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)
    INPUT_SETTINGS = ez.InputStream(WindowSettings)

    def initialize(self) -> None:
        self.STATE.cur_settings = self.SETTINGS
        self.construct_generator()

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: WindowSettings) -> None:
        self.STATE.cur_settings = msg
        self.construct_generator()

    def construct_generator(self):
        self.STATE.gen = windowing(
            axis=self.STATE.cur_settings.axis,
            newaxis=self.STATE.cur_settings.newaxis,
            window_dur=self.STATE.cur_settings.window_dur,
            window_shift=self.STATE.cur_settings.window_shift,
            zero_pad_until=self.STATE.cur_settings.zero_pad_until
        )

    @ez.subscriber(INPUT_SIGNAL)
    @ez.publisher(OUTPUT_SIGNAL)
    async def on_signal(self, msg: AxisArray) -> AsyncGenerator:
        try:
            # TODO: Refactor window generator so it always returns an axis array.
            #  Then, if the configuration is such that a new "win" axis is not expected,
            #  then iterate over the "win" axis -- dropping the "win" axis in the process.
            out_msgs = self.STATE.gen.send(msg)
            for out_msg in out_msgs:
                yield self.OUTPUT_SIGNAL, out_msg
        except (StopIteration, GeneratorExit):
            ez.logger.debug(f"Window closed in {self.address}")
        except Exception:
            ez.logger.info(traceback.format_exc())
