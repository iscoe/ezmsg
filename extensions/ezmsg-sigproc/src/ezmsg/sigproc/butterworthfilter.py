import typing

import ezmsg.core as ez
import scipy.signal
import numpy as np

from .filter import filtergen, Filter, FilterState, FilterSettingsBase

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer


class ButterworthFilterSettings(FilterSettingsBase):
    """Settings for :obj:`ButterworthFilter`."""
    order: int = 0

    cuton: typing.Optional[float] = None
    """
    Cuton frequency (Hz). If cutoff is not specified then this is the highpass corner, otherwise
    if it is lower than cutoff then this is the beginning of the bandpass
    or if it is greater than cuton then it is the end of the bandstop. 
    """

    cutoff: typing.Optional[float] = None
    """
    Cutoff frequency (Hz). If cuton is not specified then this is the lowpass corner, otherwise
    if it is greater than cuton then this is the end of the bandpass,
    or if it is less than cuton then it is the beginning of the bandstop. 
    """

    def filter_specs(self) -> typing.Optional[typing.Tuple[str, typing.Union[float, typing.Tuple[float, float]]]]:
        """
        Determine the filter type given the corner frequencies.

        Returns:
            A tuple with the first element being a string indicating the filter type
            (one of "lowpass", "highpass", "bandpass", "bandstop")
            and the second element being the corner frequency or frequencies.

        """
        if self.cuton is None and self.cutoff is None:
            return None
        elif self.cuton is None and self.cutoff is not None:
            return "lowpass", self.cutoff
        elif self.cuton is not None and self.cutoff is None:
            return "highpass", self.cuton
        elif self.cuton is not None and self.cutoff is not None:
            if self.cuton <= self.cutoff:
                return "bandpass", (self.cuton, self.cutoff)
            else:
                return "bandstop", (self.cutoff, self.cuton)


@consumer
def butter(
    axis: typing.Optional[str],
    order: int = 0,
    cuton: typing.Optional[float] = None,
    cutoff: typing.Optional[float] = None,
    coef_type: str = "ba",
) -> typing.Generator[AxisArray, AxisArray, None]:
    """
    Apply Butterworth filter to streaming data. Uses :obj:`scipy.signal.butter` to design the filter.
    See :obj:`ButterworthFilterSettings.filter_specs` for an explanation of specifying different
    filter types (lowpass, highpass, bandpass, bandstop) from the parameters.

    Args:
        axis: The name of the axis to filter.
        order: Filter order.
        cuton: Corner frequency of the filter in Hz.
        cutoff: Corner frequency of the filter in Hz.
        coef_type: "ba" or "sos"

    Returns:
        A primed generator object which accepts .send(axis_array) and yields filtered axis array.

    """
    # IO
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    btype, cutoffs = ButterworthFilterSettings(
        order=order, cuton=cuton, cutoff=cutoff
    ).filter_specs()

    # We cannot calculate coefs yet because we do not know input sample rate
    coefs = None
    filter_gen = filtergen(axis, coefs, coef_type)  # Passthrough.

    while True:
        axis_arr_in = yield axis_arr_out
        if coefs is None and order > 0:
            fs = 1 / axis_arr_in.axes[axis or axis_arr_in.dims[0]].gain
            coefs = scipy.signal.butter(
                order, Wn=cutoffs, btype=btype, fs=fs, output=coef_type
            )
            filter_gen = filtergen(axis, coefs, coef_type)

        axis_arr_out = filter_gen.send(axis_arr_in)


class ButterworthFilterState(FilterState):
    design: ButterworthFilterSettings


class ButterworthFilter(Filter):
    """:obj:`Unit` for :obj:`butterworth`"""

    SETTINGS: ButterworthFilterSettings
    STATE: ButterworthFilterState

    INPUT_FILTER = ez.InputStream(ButterworthFilterSettings)

    def initialize(self) -> None:
        self.STATE.design = self.SETTINGS
        self.STATE.filt_designed = True
        super().initialize()

    def design_filter(self) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]:
        specs = self.STATE.design.filter_specs()
        if self.STATE.design.order > 0 and specs is not None:
            btype, cut = specs
            return scipy.signal.butter(
                self.STATE.design.order,
                Wn=cut,
                btype=btype,
                fs=self.STATE.fs,
                output="ba",
            )
        else:
            return None

    @ez.subscriber(INPUT_FILTER)
    async def redesign(self, message: ButterworthFilterSettings) -> None:
        if type(message) is not ButterworthFilterSettings:
            return

        if self.STATE.design.order != message.order:
            self.STATE.zi = None
        self.STATE.design = message
        self.update_filter()
