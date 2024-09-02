import numpy as np
from numba import jit

__all__ = [
    "pixel_map_c",
    "interp_c",
    "integral_cumsum_c",
    "interp_conserve_c",
    "rebin_weighted_c",
    "midpoint_c",
]


@jit(nopython=True, fastmath=True, error_model="numpy")
def pixel_map_c(in_data, xi, yi, out_data, xo, yo):
    """
    Fast pixel mapping from one image to another

    Parameters
    ----------
    in_data : 2D array
        Input data array

    xi, yi : 1D arrays
        Pixel indices of the second (x) and first (y) array indices

    out_data : 2D array
        Output array will be updated in place ``out_data[yo, xo] = in_data[yi, xi]``

    xo, yo : 1D arrays
        Pixel indices of the second (x) and first (y) array indices

    Returns
    -------
    status : bool

    """
    N = len(xi)
    for i in range(N):
        out_data[yo[i], xo[i]] = in_data[yi[i], xi[i]]

    return True


@jit(nopython=True, fastmath=True, error_model="numpy")
def interp_c(x, xp, fp, extrapolate=0.0, assume_sorted=1):
    """
    Fast linear interpolation: ``f(x) ~ fp(xp)``

    Parameters
    ----------
    x : array-like
        Desired interpolant

    xp, fp : array-like
        Arrays to interpolate

    extrapolate : float
        Value to use where ``x < xp.min()`` or ``x > xp.max()``.

    assume_sorted : int, bool
        The default value of True assumes that the ``x`` array is sorted and
        single-valued, providing a gain in speed.
        ``xp`` is always assumed to be sorted.

    Returns
    -------
    f : array-like
        Interpolated values, same dimensions as ``x``

    """
    N, Np = len(x), len(xp)
    f = np.zeros_like(x)

    i = 0
    j = 0

    ### Handle left extrapolation
    xmin = xp[0]
    if assume_sorted == 1:
        while x[j] < xmin:
            f[j] = extrapolate
            j += 1

    while j < N:
        xval = x[j]
        if assume_sorted == 0:
            if x[j] < xmin:
                f[j] = extrapolate
                j += 1
                continue
            else:
                i = 0

        while (xp[i] < xval) & (i < Np - 1):
            i += 1

        if i == (Np - 1):
            if x[j] != xp[i]:
                f[j] = extrapolate
            else:
                f[j] = fp[i]
            j += 1
            continue

        #### x[i] is now greater than xval because the
        #### expression (x[i]<xval) is false, assuming
        #### that xval < max(x).

        x1 = xp[i - 1]
        x2 = xp[i]
        y1 = fp[i - 1]
        y2 = fp[i]
        out = ((y2 - y1) / (x2 - x1)) * (xval - x1) + y1
        f[j] = out
        j += 1

    return f


@jit(nopython=True, fastmath=True, error_model="numpy")
def integral_cumsum_c(xp, fp, extrapolate=0.0):
    """
    integral_cumsum_c(xp, fp, extrapolate=True)

    Cumulative trapz integration

    """
    #### Implementation of cumsum * dx
    Nxp = len(xp)
    ycumsum = np.zeros_like(fp)

    x0 = xp[0]
    ycumsum[0] = fp[0] * (xp[1] - x0)

    old = ycumsum[0]
    for i in range(1, Nxp):
        x1 = xp[i]
        old += fp[i] * (x1 - x0)
        ycumsum[i] = old
        x0 = x1

    return ycumsum


@jit(nopython=True, fastmath=True, error_model="numpy")
def interp_conserve_c(x, xp, yp, left=0, right=0, integrate=0):
    """
    Interpolate spectrum conserving flux

    Parameters
    ----------
    x : array-like
        Coarse ordinal axis (e.g., wavelength)

    xp, yp : array-like, array-like
        High resolution arrays, (e.g., wavelength and flux density)

    left, right : float, float
        Values to use for extrapolating off the edges of ``xp``

    integrate : bool
        Result is integrated across ``dx``

    Returns
    -------
    outy : array-like
        Interpolated array

    """
    NTEMPL = len(x)
    nxp = len(xp)

    xmid = midpoint_c(x)
    ymid = interp_c(xmid, xp, yp, extrapolate=0.0)

    outy = np.zeros_like(x)

    ######
    # Rebin template grid to master wavelength grid, conserving template flux
    i = 0
    k = 0

    tl0 = xp[0]
    while xmid[k] < tl0:
        outy[k] = left
        k += 1

    for k in range(k, NTEMPL):
        xmk = xmid[k]
        if xmk > xp[nxp - 1]:
            break

        xmk1 = xmid[k + 1]
        ymk = ymid[k]
        ymk1 = ymid[k + 1]

        numsum = 0.0

        ####
        # Go to where xp is greater than the first midpoint
        while (xp[i] < xmk) & (i < nxp):
            i += 1

        tli = xp[i]

        istart = i

        #######
        # First point
        if tli < xmk1:
            h = tli - xmk
            numsum += h * (yp[i] + ymk)
            i += 1

        if i == 0:
            i += 1

        tli = xp[i]
        ypi = yp[i]

        tli1 = xp[i - 1]
        ypi1 = yp[i - 1]

        ######
        # Template points between master grid points
        while (tli < xmk1) & (i < nxp):
            h = tli - tli1
            numsum += h * (ypi + ypi1)

            i += 1
            tli1 = tli
            ypi1 = ypi

            tli = xp[i]
            ypi = yp[i]

        ######
        # If no template points between master grid points, then just use
        # interpolated midpoints
        if i == istart:
            h = xmk1 - xmk
            numsum = h * (ymk1 + ymk)
        else:
            ##### Last point
            if (xmk1 == tli) & (i < nxp):
                h = tli - tli1
                numsum += h * (ypi + ypi1)
            else:
                i -= 1
                h = xmk1 - tli1
                numsum += h * (ymk1 + ypi1)

        outy[k] = numsum * 0.5

        if integrate == 0.0:
            outy[k] /= xmk1 - xmk

    return outy


@jit(nopython=True, fastmath=True, error_model="numpy")
def rebin_weighted_c(x, xp, yp, ye, left=0, right=0, integrate=0, remove_missing=1):
    """
    rebin_weighted_c(x, xp, fp, ep, left=0, right=0, integrate=0)

    Rebin `xp`,`yp` array to the output x array, weighting by `1/ye**2`.
    `xp` can be irregularly spaced.
    """
    NTEMPL = len(x)
    nxp = len(xp)

    outx = np.zeros(NTEMPL - 1, dtype=yp.dtype)
    outy = np.zeros(NTEMPL - 1, dtype=yp.dtype)
    oute = np.zeros(NTEMPL - 1, dtype=yp.dtype)

    ###### Rebin template grid to master wavelength grid weighted by 1/e**2
    i = 0
    k = 0
    while (x[k] < xp[0]) & (k < NTEMPL):
        outy[k] = left
        oute[k] = left
        k += 1

    if k > 0:
        k -= 1

    for k in range(k, NTEMPL - 1):
        if x[k] > xp[nxp - 1]:
            break

        xnumsum = 0.0
        numsum = 0.0
        densum = 0.0

        #### Go to where xp is greater than the x[k]
        while (xp[i] < x[k]) & (i < nxp):
            i += 1

        #### Template points between master grid points
        while (xp[i] <= x[k + 1]) & (i < nxp):
            xnumsum += xp[i] / ye[i] ** 2
            numsum += yp[i] / ye[i] ** 2
            densum += 1.0 / ye[i] ** 2
            # count[i] += 1
            if i == nxp - 1:
                break

            i += 1

        i -= 1

        if densum > 0:
            outx[k] = xnumsum / densum
            outy[k] = numsum / densum
            oute[k] = 1 / densum**0.5

    if remove_missing:
        ok = outx != 0
        return outx[ok], outy[ok], oute[ok]
    else:
        return outx, outy, oute


@jit(nopython=True, fastmath=True, error_model="numpy")
def midpoint_c(x):
    """
    Simple midpoints of array

    Parameters
    ----------
    x : array-like, N
        Target array

    Returns
    -------
    midpoint : array-like, N+1
        Midpoints of ``x``

    """
    N = len(x)
    midpoint = np.zeros(N + 1, dtype=x.dtype)
    midpoint[0] = x[0]
    midpoint[N] = x[N - 1]
    xi1 = x[0]
    for i in range(1, N):
        xi = x[i]
        midpoint[i] = 0.5 * (xi + xi1)
        xi1 = xi

    midpoint[0] = 2 * x[0] - midpoint[1]
    midpoint[N] = 2 * x[N - 1] - midpoint[N - 1]

    return midpoint
