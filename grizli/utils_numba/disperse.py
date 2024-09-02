from __future__ import division

import numpy as np
from numba import jit

DTYPE = float
ITYPE = int

__all__ = [
    "disperse_grism_object",
    "compute_segmentation_limits",
]


@jit(parallel=False, nopython=True, fastmath=True, error_model="numpy")
def disperse_grism_object(
    flam, segm, seg_id, idxl, yfrac, ysens, modelf, x0, shd, sh_thumb, shg
):
    """Compute a dispersed 2D spectrum

    Parameters
    ----------
    flam : array-like
        Direct image thumbnail

    segm : array-like
        Segmentation image

    seg_id : int, float
        Source ID.  The dispersed spectrum is computed for pixels in ``flam`` where
        ``segm == seg_id``.

    idxl : array-like (int)
        Flattened indices of the trace in the 2D cutout

    yfrac : array-like (float)
        Fraction of the flux of the "bottom" pixel along the beam

    ysens : array-like (float)
        Sensitivity or spectrum along the trace

    modelf : array-like (float)
        Flattened dispersed spectrum.  The dispersed spectrum is added in-place to this
        array

    x0 : int, int
        Reference pixel where the trace is defined

    shd : int, int
        Shape of the thumbnail array

    sh_thumb : int, int
        half-size of the thumbnail cutout to consider (this is generally ``x0``)

    shg : int, int
        Shape of the unflattened 2D spectrum

    Returns
    -------
    status : bool
        True if the function is executed successfully.  The dispersed spectrum itself
        is added to ``modelf``

    """
    nk = len(idxl)
    nl = len(modelf)

    ks = []
    for k in range(nk):
        if ysens[k] != 0:
            ks.append(k)

    for i in range(0 - sh_thumb[1], sh_thumb[1]):
        if (x0[1] + i < 0) | (x0[1] + i >= shd[1]):
            continue

        for j in range(0 - sh_thumb[0], sh_thumb[0]):
            if (x0[0] + j < 0) | (x0[0] + j >= shd[0]):
                continue

            fl_ij = flam[x0[0] + j, x0[1] + i]  # /1.e-17
            if fl_ij == 0:
                # Non-zero flux
                continue

            elif segm[x0[0] + j, x0[1] + i] != seg_id:
                # Segmentation map doesn't match the source id
                continue

            for k in ks:
                k0 = idxl[k]
                ysk = ysens[k]
                yfk = yfrac[k]

                k1 = k0 + j * shg[1] + i
                if (k1 >= 0) & (k1 < nl):
                    modelf[k1] += ysk * fl_ij * yfk

                k2 = k0 + (j - 1) * shg[1] + i
                if (k2 >= 0) & (k2 < nl):
                    modelf[k2] += ysk * fl_ij * (1 - yfk)

    return True


@jit(nopython=True, fastmath=True, error_model="numpy")
def compute_segmentation_limits(segm, seg_id, flam, shd):
    """Find pixel limits of a segmentation region

    Parameters
    ----------
    segm: ndarray (np.float32)
        segmentation array

    seg_id: int
        ID to test

    flam: ndarray (float)
        Flux array to compute weighted centroid within segmentation region

    shd: [int, int]
        Shape of segm

    Returns
    -------
    imin : int
        Minimium index of first array axis (y)

    imax : int
        Maximum index of first array axis (y)

    ic : float
        Weighted centroid along first array axis (y)

    jmin : int
        Minimium index of second array axis (x)

    jmax : int
        Maximum index of second array axis (x)

    jc : float
        Weighted centroid along second array axis (x)

    area : int
        Area of the segment

    flam_total : float
        Sum of ``flam`` within the segment

    """
    area = 0

    imin = shd[0]
    imax = 0
    jmin = shd[1]
    jmax = 0

    inumer = 0.0
    jnumer = 0.0
    flam_total = 0.0

    for i in range(shd[0]):
        for j in range(shd[1]):
            if segm[i, j] != seg_id:
                continue

            area += 1
            wht_ij = flam[i, j]
            inumer += i * wht_ij
            jnumer += j * wht_ij
            flam_total += wht_ij

            if i < imin:
                imin = i
            if i > imax:
                imax = i

            if j < jmin:
                jmin = j
            if j > jmax:
                jmax = j

    ### No matched pixels
    if flam_total == 0:
        flam_total = -99

    return (
        imin,
        imax,
        inumer / flam_total,
        jmin,
        jmax,
        jnumer / flam_total,
        area,
        flam_total,
    )


@jit(nopython=True, fastmath=True, error_model="numpy")
def seg_flux(flam, idxl, yfrac, ysens, full, x0, shd, shg):
    """
    Not used
    """
    pass
