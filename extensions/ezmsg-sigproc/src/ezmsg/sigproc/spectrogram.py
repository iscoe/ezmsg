import typing

import numpy as np

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer, GenAxisArray  # , compose
from ezmsg.util.messages.modify import modify_axis
from ezmsg.sigproc.window import windowing
from ezmsg.sigproc.spectrum import (
    spectrum,
    WindowFunction, SpectralTransform, SpectralOutput
)


@consumer
def spectrogram(
    window_dur: typing.Optional[float] = None,
    window_shift: typing.Optional[float] = None,
    window: WindowFunction = WindowFunction.HANNING,
    transform: SpectralTransform = SpectralTransform.REL_DB,
    output: SpectralOutput = SpectralOutput.POSITIVE
) -> typing.Generator[typing.Optional[AxisArray], AxisArray, None]:
    """
    Calculate a spectrogram on streaming data.

    Chains :obj:`ezmsg.sigproc.window.windowing` to apply a moving window on the data,
    :obj:`ezmsg.sigproc.spectrum.spectrum` to calculate spectra for each window,
    and finally :obj:`ezmsg.util.messages.modify.modify_axis` to convert the win axis back to time axis.

    Args:
        window_dur: See :obj:`ezmsg.sigproc.window.windowing`
        window_shift: See :obj:`ezmsg.sigproc.window.windowing`
        window: See :obj:`ezmsg.sigproc.spectrum.spectrum`
        transform: See :obj:`ezmsg.sigproc.spectrum.spectrum`
        output: See :obj:`ezmsg.sigproc.spectrum.spectrum`

    Returns:
        A primed generator object that expects `.send(axis_array)` of continuous data
        and yields an AxisArray of time-frequency power values.
    """

    # We cannot use `compose` because `windowing` returns a list of axisarray objects,
    #  even though the length is always exactly 1 for the settings used here.
    # pipeline = compose(
    f_win = windowing(axis="time", newaxis="step", window_dur=window_dur, window_shift=window_shift)
    f_spec = spectrum(axis="time", window=window, transform=transform, output=output)
    f_modify = modify_axis(name_map={"step": "time"})
    # )

    # State variables
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out: typing.Optional[AxisArray] = None

    while True:
        axis_arr_in = yield axis_arr_out

        # axis_arr_out = pipeline(axis_arr_in)
        axis_arr_out = None
        wins = f_win.send(axis_arr_in)
        if len(wins):
            specs = f_spec.send(wins[0])
            if specs is not None:
                axis_arr_out = f_modify.send(specs)


class SpectrogramSettings(ez.Settings):
    """
    Settings for :obj:`Spectrogram`.
    See :obj:`spectrogram` for a description of the parameters.
    """
    window_dur: typing.Optional[float] = None  # window duration in seconds
    window_shift: typing.Optional[float] = None  # window step in seconds. If None, window_shift == window_dur
    # See SpectrumSettings for details of following settings:
    window: WindowFunction = WindowFunction.HAMMING
    transform: SpectralTransform = SpectralTransform.REL_DB
    output: SpectralOutput = SpectralOutput.POSITIVE


class Spectrogram(GenAxisArray):
    """
    Unit for :obj:`spectrogram`.
    """
    SETTINGS: SpectrogramSettings

    def construct_generator(self):
        self.STATE.gen = spectrogram(
            window_dur=self.SETTINGS.window_dur,
            window_shift=self.SETTINGS.window_shift,
            window=self.SETTINGS.window,
            transform=self.SETTINGS.transform,
            output=self.SETTINGS.output
        )
