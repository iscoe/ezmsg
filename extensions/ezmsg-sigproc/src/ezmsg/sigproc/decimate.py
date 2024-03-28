import ezmsg.core as ez

import scipy.signal

from ezmsg.util.messages.axisarray import AxisArray

from .downsample import Downsample, DownsampleSettings
from .filter import Filter, FilterCoefficients, FilterSettings


class Decimate(ez.Collection):
    """
    A :obj:`Collection` chaining a :obj:`Filter` node configured as a lowpass Chebyshev filter
    and a :obj:`Downsample` node.
    """
    SETTINGS: DownsampleSettings

    INPUT_SIGNAL = ez.InputStream(AxisArray)
    OUTPUT_SIGNAL = ez.OutputStream(AxisArray)

    FILTER = Filter()
    DOWNSAMPLE = Downsample()

    def configure(self) -> None:
        self.DOWNSAMPLE.apply_settings(self.SETTINGS)

        if self.SETTINGS.factor < 1:
            raise ValueError("Decimation factor must be >= 1 (no decimation")
        elif self.SETTINGS.factor == 1:
            filt = FilterCoefficients()
        else:
            # See scipy.signal.decimate for IIR Filter Condition
            b, a = scipy.signal.cheby1(8, 0.05, 0.8 / self.SETTINGS.factor)
            system = scipy.signal.dlti(b, a)
            filt = FilterCoefficients(b=system.num, a=system.den)  # type: ignore

        self.FILTER.apply_settings(FilterSettings(filt=filt))

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.INPUT_SIGNAL, self.FILTER.INPUT_SIGNAL),
            (self.FILTER.OUTPUT_SIGNAL, self.DOWNSAMPLE.INPUT_SIGNAL),
            (self.DOWNSAMPLE.OUTPUT_SIGNAL, self.OUTPUT_SIGNAL),
        )
