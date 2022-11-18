"""
Numba-accelerated functions for filtering with `scipy.LowLevelallable`

Inspired by the beautifully simply example from Juan Nunez-Iglesias
https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/

>>> import numpy as np
>>> import scipy.ndimage as nd
>>> import grizli.nbutils
>>> arr = np.random.normal(size=1024)
>>> footprint = np.ones(51, dtype=int)
>>> # Slow with numpy overheads
>>> np_filtered = nd.generic_filter(arr, np.nanmedian, footprint=footprint))
>>> # Much faster!
>>> filtered = nd.generic_filter(arr, grizli.nbutils.nanmedian, footprint=footprint))

"""

import numpy as np

from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer

from scipy import LowLevelCallable


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nb_nanmin(values_ptr, len_values, result, data):
    """
    Manual minimum
    """
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = np.inf
    for v in values:
        if (v < result[0]) & ~np.isnan(v):
            result[0] = v
    return 1


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nb_nanmax(values_ptr, len_values, result, data):
    """
    Manual minimum
    """
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = -np.inf
    for v in values:
        if (v > result[0]) & ~np.isnan(v):
            result[0] = v
    return 1


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nb_npnanmin(values_ptr, len_values, result, data):
    """
    numpy.nanmin
    """
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = np.nanmin(values)
    return 1


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nb_npnanmedian(values_ptr, len_values, result, data):
    """
    numpy.nanmedian
    """
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = np.nanmedian(values)
    return 1


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nb_npnanmean(values_ptr, len_values, result, data):
    """
    numpy.nanmean
    """
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = np.nanmean(values)

    return 1


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nb_npnansum(values_ptr, len_values, result, data):
    """
    numpy.nansum
    """
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = np.nansum(values)

    return 1


# Callables for scipy.ndimage.generic_filter
nanmin = LowLevelCallable(nb_nanmin.ctypes)
nanmax = LowLevelCallable(nb_nanmax.ctypes)
nbnanmin = LowLevelCallable(nb_npnanmin.ctypes)
nanmedian = LowLevelCallable(nb_npnanmedian.ctypes)
nanmean = LowLevelCallable(nb_npnanmean.ctypes)
nansum = LowLevelCallable(nb_npnansum.ctypes)
