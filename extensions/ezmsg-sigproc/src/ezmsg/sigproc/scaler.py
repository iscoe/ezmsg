from dataclasses import replace
from typing import Generator, Optional

import numpy as np

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.generator import consumer, GenAxisArray


def _tau_from_alpha(alpha: float, dt: float) -> float:
    """
    Inverse of _alpha_from_tau. See that function for explanation.
    """
    return -dt / np.log(1 - alpha)


def _alpha_from_tau(tau: float, dt: float) -> float:
    """
    # https://en.wikipedia.org/wiki/Exponential_smoothing#Time_constant
    :param tau: The amount of time for the smoothed response of a unit step function to reach
        1 - 1/e approx-eq 63.2%.
    :param dt: sampling period, or 1 / sampling_rate.
    :return: alpha, the "fading factor" in exponential smoothing.
    """
    return 1 - np.exp(-dt / tau)


@consumer
def scaler(time_constant: float = 1.0, axis: Optional[str] = None) -> Generator[AxisArray, AxisArray, None]:
    """
    Create a generator function that applies the
    adaptive standard scaler from https://riverml.xyz/latest/api/preprocessing/AdaptiveStandardScaler/
    This is faster than :obj:`scaler_np` for single-channel data.

    Args:
        time_constant: Decay constant `tau` in seconds.
        axis: The name of the axis to accumulate statistics over.

    Returns:
        A primed generator object that expects `.send(axis_array)` and yields a
        standardized, or "Z-scored" version of the input.
    """
    from river import preprocessing
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])
    _scaler = None
    while True:
        axis_arr_in = yield axis_arr_out
        data = axis_arr_in.data
        if axis is None:
            axis = axis_arr_in.dims[0]
            axis_idx = 0
        else:
            axis_idx = axis_arr_in.get_axis_idx(axis)
            if axis_idx != 0:
                data = np.moveaxis(data, axis_idx, 0)

        if _scaler is None:
            alpha = _alpha_from_tau(time_constant, axis_arr_in.axes[axis].gain)
            _scaler = preprocessing.AdaptiveStandardScaler(fading_factor=alpha)

        result = []
        for sample in data:
            x = {k: v for k, v in enumerate(sample.flatten().tolist())}
            _scaler.learn_one(x)
            y = _scaler.transform_one(x)
            k = sorted(y.keys())
            result.append(np.array([y[_] for _ in k]).reshape(sample.shape))

        result = np.stack(result)
        result = np.moveaxis(result, 0, axis_idx)
        axis_arr_out = replace(axis_arr_in, data=result)


@consumer
def scaler_np(
        time_constant: float = 1.0,
        axis: Optional[str] = None
) -> Generator[AxisArray, AxisArray, None]:
    """
    Create a generator function that applies an adaptive standard scaler.
    This is faster than :obj:`scaler` for multichannel data.

    Args:
        time_constant: Decay constant `tau` in seconds.
        axis: The name of the axis to accumulate statistics over.

    Returns:
        A primed generator object that expects `.send(axis_array)` and yields a
        standardized, or "Z-scored" version of the input.
    """
    axis_arr_in = AxisArray(np.array([]), dims=[""])
    axis_arr_out = AxisArray(np.array([]), dims=[""])
    means = vars_means = vars_sq_means = None
    alpha = None

    def _ew_update(arr, prev, _alpha):
        if np.all(prev == 0):
            return arr
        # return _alpha * arr + (1 - _alpha) * prev
        # Micro-optimization: sub, mult, add (below) is faster than sub, mult, mult, add (above)
        return prev + _alpha * (arr - prev)

    while True:
        axis_arr_in = yield axis_arr_out

        data = axis_arr_in.data
        if axis is None:
            axis = axis_arr_in.dims[0]
            axis_idx = 0
        else:
            axis_idx = axis_arr_in.get_axis_idx(axis)
        data = np.moveaxis(data, axis_idx, 0)

        if alpha is None:
            alpha = _alpha_from_tau(time_constant, axis_arr_in.axes[axis].gain)

        if means is None or means.shape != data.shape[1:]:
            vars_sq_means = np.zeros_like(data[0], dtype=float)
            vars_means = np.zeros_like(data[0], dtype=float)
            means = np.zeros_like(data[0], dtype=float)

        result = np.zeros_like(data)
        for sample_ix, sample in enumerate(data):
            # Update step
            vars_means = _ew_update(sample, vars_means, alpha)
            vars_sq_means = _ew_update(sample**2, vars_sq_means, alpha)
            means = _ew_update(sample, means, alpha)
            # Get step
            varis = vars_sq_means - vars_means ** 2
            y = ((sample - means) / (varis**0.5))
            result[sample_ix] = y

        result[np.isnan(result)] = 0.0
        result = np.moveaxis(result, 0, axis_idx)
        axis_arr_out = replace(axis_arr_in, data=result)


class AdaptiveStandardScalerSettings(ez.Settings):
    """
    Settings for :obj:`AdaptiveStandardScaler`.
    See :obj:`scaler_np` for a description of the parameters.
    """
    time_constant: float = 1.0
    axis: Optional[str] = None


class AdaptiveStandardScaler(GenAxisArray):
    """Unit for :obj:`scaler_np`"""
    SETTINGS: AdaptiveStandardScalerSettings

    def construct_generator(self):
        self.STATE.gen = scaler_np(
            time_constant=self.SETTINGS.time_constant,
            axis=self.SETTINGS.axis
        )
