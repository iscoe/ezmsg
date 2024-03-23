from dataclasses import replace
import typing

import numpy as np
import ezmsg.core as ez
from ezmsg.util.generator import consumer, GenAxisArray
from ezmsg.util.messages.axisarray import AxisArray, slice_along_axis
from ezmsg.sigproc.spectral import OptionsEnum


class AggregationFunction(OptionsEnum):
    """Enum for aggregation functions available to be used in :obj:`ranged_aggregate` operation."""
    NONE = "None (all)"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    NANMAX = "nanmax"
    NANMIN = "nanmin"
    NANMEAN = "nanmean"
    NANMEDIAN = "nanmedian"
    NANSTD = "nanstd"


AGGREGATORS = {
    AggregationFunction.NONE: np.all,
    AggregationFunction.MAX: np.max,
    AggregationFunction.MIN: np.min,
    AggregationFunction.MEAN: np.mean,
    AggregationFunction.MEDIAN: np.median,
    AggregationFunction.STD: np.std,
    AggregationFunction.NANMAX: np.nanmax,
    AggregationFunction.NANMIN: np.nanmin,
    AggregationFunction.NANMEAN: np.nanmean,
    AggregationFunction.NANMEDIAN: np.nanmedian,
    AggregationFunction.NANSTD: np.nanstd
}


@consumer
def ranged_aggregate(
    axis: typing.Optional[str] = None,
    bands: typing.Optional[typing.List[typing.Tuple[float, float]]] = None,
    operation: AggregationFunction = AggregationFunction.MEAN
):
    """
    Apply an aggregation operation over one or more bands.

    Args:
        axis: The name of the axis along which to apply the bands.
        bands: [(band1_min, band1_max), (band2_min, band2_max), ...]
            If not set then this acts as a passthrough node.
        operation: :obj:`AggregationFunction` to apply to each band.

    Returns:
        A primed generator object ready to yield an AxisArray for each .send(axis_array)
    """
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    target_axis: typing.Optional[AxisArray.Axis] = None
    out_axis = AxisArray.Axis()
    slices: typing.Optional[typing.List[typing.Tuple[typing.Any, ...]]] = None
    axis_name = ""

    while True:
        axis_arr_in = yield axis_arr_out
        if bands is None:
            axis_arr_out = axis_arr_in
        else:
            if slices is None or target_axis != axis_arr_in.get_axis(axis_name):
                # Calculate the slices. If we are operating on time axis then
                axis_name = axis or axis_arr_in.dims[0]
                ax_idx = axis_arr_in.get_axis_idx(axis_name)
                target_axis = axis_arr_in.axes[axis_name]

                ax_vec = target_axis.offset + np.arange(axis_arr_in.data.shape[ax_idx]) * target_axis.gain
                slices = []
                mids = []
                for (start, stop) in bands:
                    inds = np.where(np.logical_and(ax_vec >= start, ax_vec <= stop))[0]
                    mids.append(np.mean(inds) * target_axis.gain + target_axis.offset)
                    slices.append(np.s_[inds[0]:inds[-1] + 1])
                out_axis = AxisArray.Axis(
                    unit=target_axis.unit, offset=mids[0], gain=(mids[1] - mids[0]) if len(mids) > 1 else 1.0
                )

            agg_func = AGGREGATORS[operation]
            out_data = [
                agg_func(slice_along_axis(axis_arr_in.data, sl, axis=ax_idx), axis=ax_idx)
                for sl in slices
            ]
            new_axes = {**axis_arr_in.axes, axis_name: out_axis}
            axis_arr_out = replace(
                axis_arr_in,
                data=np.stack(out_data, axis=ax_idx),
                axes=new_axes
            )


class RangedAggregateSettings(ez.Settings):
    """
    Settings for ``RangedAggregate``.
    See :obj:`ranged_aggregate` for details.
    """
    axis: typing.Optional[str] = None
    bands: typing.Optional[typing.List[typing.Tuple[float, float]]] = None
    operation: AggregationFunction = AggregationFunction.MEAN


class RangedAggregate(GenAxisArray):
    """
    Unit for :obj:`ranged_aggregate`
    """
    SETTINGS: RangedAggregateSettings

    def construct_generator(self):
        self.STATE.gen = ranged_aggregate(
            axis=self.SETTINGS.axis,
            bands=self.SETTINGS.bands,
            operation=self.SETTINGS.operation
        )
