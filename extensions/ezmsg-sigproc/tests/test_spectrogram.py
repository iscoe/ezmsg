import typing

import numpy as np

from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.spectrum import WindowFunction, SpectralTransform, SpectralOutput
from ezmsg.sigproc.spectrogram import spectrogram

from util import create_messages_with_periodic_signal


def _debug_plot(
        ax_arr: AxisArray,
        sin_params: typing.List[typing.Dict[str, float]] = None
):
    import matplotlib.pyplot as plt

    t_ix = ax_arr.get_axis_idx("time")
    t_vec = ax_arr.axes["time"].offset + np.arange(ax_arr.data.shape[t_ix] * ax_arr.axes["time"].gain)
    t_vec -= ax_arr.axes["time"].gain / 2
    f_ix = ax_arr.get_axis_idx("freq")
    f_vec = ax_arr.axes["freq"].offset + np.arange(ax_arr.data.shape[f_ix] * ax_arr.axes["freq"].gain)
    f_vec -= ax_arr.axes["freq"].gain / 2
    plt.imshow(
        ax_arr.data[..., 0].T,
        origin="lower",
        aspect="auto",
        extent=(t_vec[0], t_vec[-1], f_vec[0], f_vec[-1])
    )
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    if sin_params is not None:
        for s_p in sin_params:
            xx = (s_p.get("offset", 0.0) + t_vec[0], s_p.get("offset", 0.0) + t_vec[0] + s_p["dur"])
            yy = (s_p["f"], s_p["f"])
            plt.plot(xx, yy, linestyle="--", color="r", linewidth=1.0)


def test_spectrogram():
    win_dur = 1.0
    win_step_dur = 0.5
    fs = 1000.
    seg_dur = 5.0
    sin_params = [
        {"f": 10.0, "dur": seg_dur, "offset": 0.0},
        {"f": 20.0, "dur": seg_dur, "offset": 0.0},
        {"f": 70.0, "dur": seg_dur, "offset": 0.0},
        {"f": 14.0, "dur": seg_dur, "offset": seg_dur},
        {"f": 35.0, "dur": seg_dur, "offset": seg_dur},
        {"f": 300.0, "dur": seg_dur, "offset": seg_dur},
    ]
    messages = create_messages_with_periodic_signal(
        sin_params=sin_params,
        fs=fs,
        msg_dur=0.4,
        win_step_dur=None  # The spectrogram will do the windowing
    )

    gen = spectrogram(
        window_dur=win_dur,
        window_shift=win_step_dur,
        window=WindowFunction.HANNING,
        transform=SpectralTransform.REL_DB,
        output=SpectralOutput.POSITIVE,
    )

    results = [gen.send(msg) for msg in messages]
    results = [_ for _ in results if _.data.size]  # Drop empty messages

    # Check that the windows span the expected times.
    expected_t_span = 2 * seg_dur
    data_t_span = (results[-1].axes["time"].offset + win_step_dur) - results[0].axes["time"].offset
    assert np.abs(expected_t_span - data_t_span) < 1e-9
    all_deltas = np.diff([_.axes["time"].offset for _ in results])
    assert np.allclose(all_deltas, win_step_dur + np.zeros((len(results) - 1)))

    # result = AxisArray.concatenate(*results, dim="time")
    # TODO: Check spectral peaks change over time. _debug_plot is useful if not automatic.
    # _debug_plot(result, sin_params=sin_params)
