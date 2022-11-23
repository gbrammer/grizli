
from __future__ import division

import numpy as np
cimport numpy as np

DTYPE = np.float64
ITYPE = np.int64

ctypedef np.uint_t UINT_t
ctypedef np.int_t INT_t
ctypedef np.int64_t LINT_t
ctypedef np.int32_t FINT_t
ctypedef np.float32_t FTYPE_t
ctypedef np.float64_t DTYPE_t

import cython

cdef extern from "math.h":
    double sqrt(double x)
    double exp(double x)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def disperse_grism_object(np.ndarray[FTYPE_t, ndim=2] flam, 
                          np.ndarray[FTYPE_t, ndim=2] segm, 
                          FTYPE_t seg_id, 
                          np.ndarray[LINT_t, ndim=1] idxl, 
                          np.ndarray[DTYPE_t, ndim=1] yfrac, 
                          np.ndarray[DTYPE_t, ndim=1] ysens, 
                          np.ndarray[FTYPE_t, ndim=1] full, 
                          np.ndarray[LINT_t, ndim=1] x0, 
                          np.ndarray[LINT_t, ndim=1] shd, 
                          np.ndarray[LINT_t, ndim=1] sh_thumb, 
                          np.ndarray[LINT_t, ndim=1] shg):
    """Compute a dispersed 2D spectrum
    
    Parameters
    ----------
    xxx
    """
    cdef int i,j,k1,k2,nl
    cdef unsigned int nk,k,shx,shy
    cdef double fl_ij
    
    nk = len(idxl)
    nl = len(full)
    
    for i in range(0-sh_thumb[1], sh_thumb[1]):
        if (x0[1]+i < 0) | (x0[1]+i >= shd[1]):
            continue
            
        for j in range(0-sh_thumb[0], sh_thumb[0]):
            if (x0[0]+j < 0) | (x0[0]+j >= shd[0]):
                continue

            fl_ij = flam[x0[0]+j, x0[1]+i] #/1.e-17
            if (fl_ij == 0) | (segm[x0[0]+j, x0[1]+i] != seg_id):
                continue
                
            for k in range(nk):
                k1 = idxl[k]+j*shg[1]+i
                if (k1 >= 0) & (k1 < nl):
                    full[k1] += ysens[k]*fl_ij*yfrac[k]
                    
                k2 = idxl[k]+(j-1)*shg[1]+i
                if (k2 >= 0) & (k2 < nl):
                    full[k2] += ysens[k]*fl_ij*(1-yfrac[k])
    
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def compute_segmentation_limits(np.ndarray[FTYPE_t, ndim=2] segm, 
                                int seg_id, 
                                np.ndarray[FTYPE_t, ndim=2] flam, 
                                np.ndarray[LINT_t, ndim=1] shd):
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
    """
    cdef int i, j, imin, imax, jmin, jmax, area
    cdef double inumer, jnumer, denom, wht_ij
    
    area = 0
    
    imin = shd[0]
    imax = 0
    jmin = shd[1]
    jmax = 0
    
    inumer = 0.
    jnumer = 0.
    denom = 0.
    
    for i in range(shd[0]):
        for j in range(shd[1]):
            if segm[i,j] != seg_id:
                continue
            
            area += 1
            wht_ij = flam[i,j]
            inumer += i*wht_ij
            jnumer += j*wht_ij
            denom += wht_ij
            
            if i < imin:
                imin = i
            if i > imax:
                imax = i
            
            if j < jmin: 
                jmin = j
            if j > jmax:
                jmax = j
    
    ### No matched pixels
    if denom == 0:
        denom = -99
        
    return imin, imax, inumer/denom, jmin, jmax, jnumer/denom, area, denom
            
            
            
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def seg_flux(np.ndarray[FTYPE_t, ndim=2] flam, np.ndarray[LINT_t, ndim=1] idxl, np.ndarray[FTYPE_t, ndim=1] yfrac, np.ndarray[FTYPE_t, ndim=1] ysens, np.ndarray[FTYPE_t, ndim=1] full, np.ndarray[LINT_t, ndim=1] x0, np.ndarray[LINT_t, ndim=1] shd, np.ndarray[LINT_t, ndim=1] shg):
    pass