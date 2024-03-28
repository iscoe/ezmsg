from dataclasses import replace
import enum
from typing import Optional, Generator, AsyncGenerator

import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.util.generator import consumer, GenAxisArray


class OptionsEnum(enum.Enum):
    @classmethod
    def options(cls):
        return list(map(lambda c: c.value, cls))


class WindowFunction(OptionsEnum):
    """Windowing function prior to calculating spectrum. """
    NONE = "None (Rectangular)"
    """None."""

    HAMMING = "Hamming"
    """:obj:`numpy.hamming`"""

    HANNING = "Hanning"
    """:obj:`numpy.hanning`"""

    BARTLETT = "Bartlett"
    """:obj:`numpy.bartlett`"""

    BLACKMAN = "Blackman"
    """:obj:`numpy.blackman`"""


WINDOWS = {
    WindowFunction.NONE: np.ones,
    WindowFunction.HAMMING: np.hamming,
    WindowFunction.HANNING: np.hanning,
    WindowFunction.BARTLETT: np.bartlett,
    WindowFunction.BLACKMAN: np.blackman,
}


class SpectralTransform(OptionsEnum):
    """Additional transformation functions to apply to the spectral result."""
    RAW_COMPLEX = "Complex FFT Output"
    REAL = "Real Component of FFT"
    IMAG = "Imaginary Component of FFT"
    REL_POWER = "Relative Power"
    REL_DB = "Log Power (Relative dB)"


class SpectralOutput(OptionsEnum):
    """The expected spectral contents."""
    FULL = "Full Spectrum"
    POSITIVE = "Positive Frequencies"
    NEGATIVE = "Negative Frequencies"


@consumer
def spectrum(
    axis: Optional[str] = None,
    out_axis: Optional[str] = "freq",
    window: WindowFunction = WindowFunction.HANNING,
    transform: SpectralTransform = SpectralTransform.REL_DB,
    output: SpectralOutput = SpectralOutput.POSITIVE
) -> Generator[AxisArray, AxisArray, None]:
    """
    Calculate a spectrum on a data slice.

    Args:
        axis: The name of the axis on which to calculate the spectrum.
        out_axis: The name of the new axis. Defaults to "freq".
        window: The :obj:`WindowFunction` to apply to the data slice prior to calculating the spectrum.
        transform: The :obj:`SpectralTransform` to apply to the spectral magnitude.
        output: The :obj:`SpectralOutput` format.

    Returns:
        A primed generator object that expects `.send(axis_array)` of continuous data
        and yields an AxisArray of spectral magnitudes or powers.
    """

    # State variables
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    axis_name = axis
    axis_idx = None
    n_time = None

    while True:
        axis_arr_in = yield axis_arr_out

        if axis_name is None:
            axis_name = axis_arr_in.dims[0]

        # Initial setup
        if n_time is None or axis_idx is None or axis_arr_in.data.shape[axis_idx] != n_time:
            axis_idx = axis_arr_in.get_axis_idx(axis_name)
            _axis = axis_arr_in.get_axis(axis_name)
            n_time = axis_arr_in.data.shape[axis_idx]
            freqs = np.fft.fftshift(np.fft.fftfreq(n_time, d=_axis.gain), axes=-1)
            window = WINDOWS[window](n_time)
            window = window.reshape([1] * axis_idx + [len(window),] + [1] * (axis_arr_in.data.ndim-2))
            if (transform != SpectralTransform.RAW_COMPLEX and
                    not (transform == SpectralTransform.REAL or transform == SpectralTransform.IMAG)):
                scale = np.sum(window ** 2.0) * _axis.gain
            axis_offset = freqs[0]
            if output == SpectralOutput.POSITIVE:
                axis_offset = freqs[n_time // 2]
            freq_axis = AxisArray.Axis(
                unit="Hz", gain=1.0 / (_axis.gain * n_time), offset=axis_offset
            )
            if out_axis is None:
                out_axis = axis_name
            new_dims = axis_arr_in.dims[:axis_idx] + [out_axis, ] + axis_arr_in.dims[axis_idx + 1:]

            f_transform = lambda x: x
            if transform != SpectralTransform.RAW_COMPLEX:
                if transform == SpectralTransform.REAL:
                    f_transform = lambda x: x.real
                elif transform == SpectralTransform.IMAG:
                    f_transform = lambda x: x.imag
                else:
                    f1 = lambda x: (2.0 * (np.abs(x) ** 2.0)) / scale
                    if transform == SpectralTransform.REL_DB:
                        f_transform = lambda x: 10 * np.log10(f1(x))
                    else:
                        f_transform = f1

        new_axes = {**axis_arr_in.axes, **{out_axis: freq_axis}}
        if out_axis != axis_name:
            new_axes.pop(axis_name, None)

        spec = np.fft.fft(axis_arr_in.data * window, axis=axis_idx) / n_time
        spec = np.fft.fftshift(spec, axes=axis_idx)
        spec = f_transform(spec)

        if output == SpectralOutput.POSITIVE:
            spec = slice_along_axis(spec, slice(n_time // 2, None), axis_idx)

        elif output == SpectralOutput.NEGATIVE:
            spec = slice_along_axis(spec, slice(None, n_time // 2), axis_idx)

        axis_arr_out = replace(axis_arr_in, data=spec, dims=new_dims, axes=new_axes)


class SpectrumSettings(ez.Settings):
    """
    Settings for :obj:`Spectrum.
    See :obj:`spectrum` for a description of the parameters.
    """
    axis: Optional[str] = None
    # n: Optional[int] = None # n parameter for fft
    out_axis: Optional[str] = "freq"  # If none; don't change dim name
    window: WindowFunction = WindowFunction.HAMMING
    transform: SpectralTransform = SpectralTransform.REL_DB
    output: SpectralOutput = SpectralOutput.POSITIVE


class SpectrumState(ez.State):
    gen: Generator
    cur_settings: SpectrumSettings


class Spectrum(GenAxisArray):
    """Unit for :obj:`spectrum`"""
    SETTINGS: SpectrumSettings
    STATE: SpectrumState

    INPUT_SETTINGS = ez.InputStream(SpectrumSettings)

    def initialize(self) -> None:
        self.STATE.cur_settings = self.SETTINGS
        super().initialize()

    @ez.subscriber(INPUT_SETTINGS)
    async def on_settings(self, msg: SpectrumSettings):
        self.STATE.cur_settings = msg
        self.construct_generator()

    def construct_generator(self):
        self.STATE.gen = spectrum(
            axis=self.STATE.cur_settings.axis,
            out_axis=self.STATE.cur_settings.out_axis,
            window=self.STATE.cur_settings.window,
            transform=self.STATE.cur_settings.transform,
            output=self.STATE.cur_settings.output
        )
