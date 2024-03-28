import math
import time
import typing

from contextlib import contextmanager
from dataclasses import field, dataclass, replace

import numpy as np
import numpy.typing as npt
import numpy.lib.stride_tricks as nps

from ezmsg.core.util import either_dict_or_kwargs

# TODO: Typehinting is all wrong in this and 
#  concatenate/transpose should probably not be staticmethods


@dataclass
class AxisArray:
    """
    A lightweight message class comprising a numpy ndarray and its metadata.
    """
    data: npt.NDArray
    dims: typing.List[str]
    axes: typing.Dict[str, "AxisArray.Axis"] = field(default_factory=dict)

    @dataclass
    class Axis:
        unit: str = ""
        gain: float = 1.0
        offset: float = 0.0

        @classmethod
        def TimeAxis(cls, fs: float, offset: float = 0.0) -> "AxisArray.Axis":
            """
            Creates a time axis with dimension of seconds.
            Specify fs in units of Hz (1.0/sec)
            NOTE: offset corresponds to idx[0] on this dimension!
            """
            return cls(unit="s", gain=1.0 / fs, offset=offset)

        def units(self, idx):
            return (idx * self.gain) + self.offset

        def index(self, val):
            return np.round((val - self.offset) / self.gain)

    @dataclass(frozen=True)
    class AxisInfo:
        axis: "AxisArray.Axis"
        idx: int
        size: int

        def __len__(self) -> int:
            return self.size

        @property
        def indices(self) -> npt.NDArray[np.int_]:
            return np.arange(self.size)

        @property
        def values(self) -> npt.NDArray:
            return self.axis.units(self.indices)

    def __post_init__(self):
        if len(self.dims) != self.data.ndim:
            raise ValueError("dims must be same length as data.shape")
        if len(self.dims) != len(set(self.dims)):
            raise ValueError("dims contains repeated dim names")

    def isel(
        self,
        indexers: typing.Optional[typing.Any] = None,
        **indexers_kwargs: typing.Any
    ) -> "AxisArray":
        
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'isel')

        out_axes = {an: a for an, a in self.axes.items()}
        out_data = self.data

        for axis_name, ix in indexers.items():
            if not isinstance(ix, (np.ndarray, int, slice)):
                raise ValueError('isel only accepts arrays, ints, or slices')
            ax = self.ax(axis_name)
            indices = np.arange(len(ax))

            # There has to be a more efficient way to do this
            if isinstance(ix, slice):
                indices = indices[ix]
            else:
                ix = [ix] if isinstance(ix, int) else ix
                indices = np.take(indices, ix, 0)

            if axis_name in out_axes:
                out_axes[axis_name] = replace(ax.axis, offset=ax.axis.units(indices[0]))
            out_data = np.take(out_data, indices, ax.idx)

        return replace(self, data=out_data, axes=out_axes)

    def sel(
        self, 
        indexers: typing.Optional[typing.Any] = None,
        **indexers_kwargs: typing.Any
    ) -> "AxisArray":
        
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'sel')

        out_indexers = dict()
        for axis_name, ix in indexers.items():
            axis = self.get_axis(axis_name)
            if not isinstance(ix, slice):
                raise ValueError("sel only supports slices for now")
            start = int(axis.index(ix.start)) if ix.start is not None else None
            stop = int(axis.index(ix.stop)) if ix.stop is not None else None
            step = int(ix.step / axis.gain) if ix.step is not None else None
            out_indexers[axis_name] = slice(start, stop, step)
            
        return self.isel(**out_indexers)

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        """Shape of data"""
        return self.data.shape

    def ax(self, dim: typing.Union[str, int]) -> AxisInfo:
        axis_idx = dim if isinstance(dim, int) else self.get_axis_idx(dim)
        axis_name = dim if isinstance(dim, str) else self.get_axis_name(dim)
        return AxisArray.AxisInfo(
            axis=self.get_axis(axis_name), idx=axis_idx, size=self.shape[axis_idx]
        )

    def get_axis(self, dim: typing.Union[str, int]) -> Axis:
        if isinstance(dim, int):
            dim = self.get_axis_name(dim)
        if dim not in self.dims:
            raise ValueError(f"{dim=} not present in object")
        return self.axes.get(dim, AxisArray.Axis())

    def get_axis_name(self, dim: int) -> str:
        return self.dims[dim]

    def get_axis_idx(self, dim: str) -> int:
        try:
            return self.dims.index(dim)
        except ValueError:
            raise ValueError(f"{dim=} not present in object")

    def axis_idx(self, dim: typing.Union[str, int]) -> int:
        return self.get_axis_idx(dim) if isinstance(dim, str) else dim

    def _axis_idx(self, dim: typing.Union[str, int]) -> int:
        """Deprecated; use axis_idx instead"""
        return self.axis_idx(dim)

    def as2d(self, dim: typing.Union[str, int]) -> npt.NDArray:
        return as2d(self.data, self.axis_idx(dim))

    @contextmanager
    def view2d(
        self, dim: typing.Union[str, int]
    ) -> typing.Generator[npt.NDArray, None, None]:
        with view2d(self.data, self.axis_idx(dim)) as arr:
            yield arr

    def shape2d(self, dim: typing.Union[str, int]) -> typing.Tuple[int, int]:
        return shape2d(self.data, self.axis_idx(dim))

    @staticmethod
    def concatenate(
        *aas: "AxisArray", dim: str, axis: typing.Optional[Axis] = None
    ) -> "AxisArray":
        aa_0 = aas[0]
        for aa in aas[1:]:
            if aa.dims != aa_0.dims:
                raise ValueError("objects have incompatible dims")

        new_dims = [d for d in aa_0.dims]
        new_axes = {ax_name: ax for ax_name, ax in aa_0.axes.items()}

        if axis is not None:
            new_axes[dim] = axis

        all_data = [aa.data for aa in aas]

        if dim in aa_0.dims:
            dim_idx = aa_0.axis_idx(dim=dim)
            new_data = np.concatenate(all_data, axis=dim_idx)
            return replace(aa_0, data=new_data, dims=new_dims, axes=new_axes)

        else:
            new_data = np.array(all_data)
            new_dims = [dim] + new_dims
            return replace(aa_0, data=new_data, dims=new_dims, axes=new_axes)

    @staticmethod
    def transpose(
        aa: "AxisArray",
        dims: typing.Optional[
            typing.Union[typing.Iterable[str], typing.Iterable[int]]
        ] = None,
    ) -> "AxisArray":
        dims = reversed(range(aa.data.ndim)) if dims is None else dims
        dim_indices = [aa.axis_idx(d) for d in dims]
        new_dims = [aa.dims[d] for d in dim_indices]
        new_data = np.transpose(aa.data, dim_indices)
        return replace(aa, data=new_data, dims=new_dims, axes=aa.axes)


def slice_along_axis(in_arr: npt.NDArray, sl: typing.Union[slice, int], axis: int) -> npt.NDArray:
    """
    Slice the input array along a specified axis using the given slice object or integer index.
     Integer arguments to `sl` will cause the sliced dimension to be dropped.
     Use `slice(my_int, my_int+1, None)` to keep the sliced dimension.

    Parameters:
        in_arr: The input array to be sliced.
        sl: The slice object or integer index to use for slicing.
        axis: The axis along which to slice the array.

    Returns:
        The sliced array (view).

    Raises:
        ValueError: If the axis value is invalid for the input array.
    """
    if axis < -in_arr.ndim or axis >= in_arr.ndim:
        raise ValueError(f"Invalid axis value {axis} for input array with {in_arr.ndim} dimensions.")
    if -in_arr.ndim <= axis < 0:
        axis = in_arr.ndim + axis
    all_slice = (slice(None),) * axis + (sl,) + (slice(None),) * (in_arr.ndim - axis - 1)
    return in_arr[all_slice]


def sliding_win_oneaxis(in_arr: npt.NDArray, nwin: int, axis: int) -> npt.NDArray:
    """
    Generates a view of an array using a sliding window of specified length along a specified axis of the input array.
    This is a slightly optimized version of nps.sliding_window_view with a few important differences:

    - This only accepts a single nwin and a single axis, thus we can skip some checks.
    - The new `win` axis precedes immediately the original target axis, unlike sliding_window_view where the
        target axis is moved to the end of the output.

    Combine this with slice_along_axis(..., sl=slice(None, None, step), axis=axis) to step the window
        by more than 1 sample at a time.

    Args:
        in_arr: The input array.
        nwin: The size of the sliding window.
        axis: The axis along which the sliding window will be applied.

    Returns:
        A view to the input array with the sliding window applied.

    Note: There is a known edge case when nwin == shape[axis] + 1. While this should raise
        an error because the window is larger than the input, the implementation ends up
        returning a 0-length window. We could check for this but this function is intended
        to have minimal latency so we have decided to skip the checks and deal with the
        support issues as they arise.
    """
    if -in_arr.ndim <= axis < 0:
        axis = in_arr.ndim + axis
    out_strides = in_arr.strides[:axis] + (in_arr.strides[axis],) * 2 + in_arr.strides[axis+1:]
    out_shape = in_arr.shape[:axis] + (in_arr.shape[axis]-(nwin-1),) + (nwin,) + in_arr.shape[axis+1:]
    return nps.as_strided(in_arr, strides=out_strides, shape=out_shape, writeable=False)


def _as2d(
    in_arr: npt.NDArray, axis: int = 0
) -> typing.Tuple[npt.NDArray, typing.Tuple[int]]:
    arr = in_arr
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, np.newaxis]

    arr = np.moveaxis(arr, axis, 0)
    remaining_shape = arr.shape[1:]
    arr = arr.reshape(arr.shape[0], np.prod(remaining_shape))
    return arr, remaining_shape


def as2d(in_arr: npt.NDArray, axis: int = 0) -> npt.NDArray:
    arr, _ = _as2d(in_arr, axis=axis)
    return arr


@contextmanager
def view2d(
    in_arr: npt.NDArray, axis: int = 0
) -> typing.Generator[npt.NDArray, None, None]:
    """
    2D-View (el x ch) of the array, no matter what input dimensionality is.
    Yields a view of underlying data when possible, changes to data in yielded
    array will always be reflected in self.data, either via the view or copying
    NOTE: In practice, I'm not sure this is very useful because it requires modifying
    the numpy array data in-place, which limits its application to zero-copy messages
    """
    arr, remaining_shape = _as2d(in_arr, axis=axis)

    yield arr

    if not np.shares_memory(arr, in_arr):
        arr = arr.reshape(arr.shape[0], *remaining_shape)
        arr = np.moveaxis(arr, 0, axis)
        in_arr[:] = arr


def shape2d(arr: npt.NDArray, axis: int = 0) -> typing.Tuple[int, int]:
    return (
        arr.shape[axis],
        math.prod(v for idx, v in enumerate(arr.shape) if idx != axis),
    )
