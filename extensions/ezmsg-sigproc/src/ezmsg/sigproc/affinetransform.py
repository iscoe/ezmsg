from dataclasses import replace
import os
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer, GenAxisArray


@consumer
def affine_transform(
    weights: Union[np.ndarray, str, Path],
    axis: Optional[str] = None,
    right_multiply: bool = True,
) -> Generator[AxisArray, AxisArray, None]:
    """
    Perform affine transformations on streaming data.

    Args:
        weights: An array of weights or a path to a file with weights compatible with np.loadtxt.
        axis: The name of the axis to apply the transformation to. Defaults to the leading (0th) axis in the array.
        right_multiply: Set False to tranpose the weights before applying.

    Returns:
        A primed generator object that yields an :obj:`AxisArray` object for every
        :obj:`AxisArray` it receives via `send`.
    """
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    if isinstance(weights, str):
        weights = Path(os.path.abspath(os.path.expanduser(weights)))
    if isinstance(weights, Path):
        weights = np.loadtxt(weights, delimiter=",")
    if not right_multiply:
        weights = weights.T
    weights = np.ascontiguousarray(weights)

    while True:
        axis_arr_in = yield axis_arr_out

        if axis is None:
            axis = axis_arr_in.dims[-1]
            axis_idx = -1
        else:
            axis_idx = axis_arr_in.get_axis_idx(axis)

        data = axis_arr_in.data

        if data.shape[axis_idx] == (weights.shape[0] - 1):
            # The weights are stacked A|B where A is the transform and B is a single row
            #  in the equation y = Ax + B. This supports NeuroKey's weights matrices.
            sample_shape = data.shape[:axis_idx] + (1,) + data.shape[axis_idx+1:]
            data = np.concatenate((data, np.ones(sample_shape).astype(data.dtype)), axis=axis_idx)

        if axis_idx in [-1, len(axis_arr_in.dims) - 1]:
            data = np.matmul(data, weights)
        else:
            data = np.moveaxis(data, axis_idx, -1)
            data = np.matmul(data, weights)
            data = np.moveaxis(data, -1, axis_idx)
        axis_arr_out = replace(axis_arr_in, data=data)


class AffineTransformSettings(ez.Settings):
    """
    Settings for :obj:`AffineTransform`.
    See :obj:`affine_transform` for argument details.
    """
    weights: Union[np.ndarray, str, Path]
    axis: Optional[str] = None
    right_multiply: bool = True


class AffineTransform(GenAxisArray):
    """:obj:`Unit` for :obj:`affine_transform`"""
    SETTINGS: AffineTransformSettings

    def construct_generator(self):
        self.STATE.gen = affine_transform(
            weights=self.SETTINGS.weights,
            axis=self.SETTINGS.axis,
            right_multiply=self.SETTINGS.right_multiply,
        )


@consumer
def common_rereference(
    mode: str = "mean", axis: Optional[str] = None, include_current: bool = True
) -> Generator[AxisArray, AxisArray, None]:
    """
    Perform common average referencing (CAR) on streaming data.

    Args:
        mode: The statistical mode to apply -- either "mean" or "median"
        axis: The name of hte axis to apply the transformation to.
        include_current: Set False to exclude each channel from participating in the calculation of its reference.

    Returns:
        A primed generator object that yields an :obj:`AxisArray` object
        for every :obj:`AxisArray` it receives via `send`.
    """
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])

    func = {"mean": np.mean, "median": np.median}[mode]

    while True:
        axis_arr_in = yield axis_arr_out

        if axis is None:
            axis = axis_arr_in.dims[-1]
            axis_idx = -1
        else:
            axis_idx = axis_arr_in.get_axis_idx(axis)

        ref_data = func(axis_arr_in.data, axis=axis_idx, keepdims=True)

        if not include_current:
            # Typical `CAR = x[0]/N + x[1]/N + ... x[i-1]/N + x[i]/N + x[i+1]/N + ... + x[N-1]/N`
            # and is the same for all i, so it is calculated only once in `ref_data`.
            # However, if we had excluded the current channel,
            # then we would have omitted the contribution of the current channel:
            # `CAR[i] = x[0]/(N-1) + x[1]/(N-1) + ... x[i-1]/(N-1) + x[i+1]/(N-1) + ... + x[N-1]/(N-1)`
            # The majority of the calculation is the same as when the current channel is included;
            # we need only rescale CAR so the divisor is `N-1` instead of `N`, then subtract the contribution
            # from the current channel (i.e., `x[i] / (N-1)`)
            #  i.e., `CAR[i] = (N / (N-1)) * common_CAR - x[i]/(N-1)`
            # We can use broadcasting subtraction instead of looping over channels.
            N = axis_arr_in.data.shape[axis_idx]
            ref_data = (N / (N - 1)) * ref_data - axis_arr_in.data / (N - 1)
            # Side note: I profiled using affine_transform and it's about 30x slower than this implementation.

        axis_arr_out = replace(axis_arr_in, data=axis_arr_in.data - ref_data)


class CommonRereferenceSettings(ez.Settings):
    """
    Settings for :obj:`CommonRereference`
    See :obj:`common_rereference` for argument details.
    """
    mode: str = "mean"
    axis: Optional[str] = None
    include_current: bool = True


class CommonRereference(GenAxisArray):
    """
    :obj:`Unit` for :obj:`common_rereference`.
    """
    SETTINGS: CommonRereferenceSettings

    def construct_generator(self):
        self.STATE.gen = common_rereference(
            mode=self.SETTINGS.mode,
            axis=self.SETTINGS.axis,
            include_current=self.SETTINGS.include_current,
        )
