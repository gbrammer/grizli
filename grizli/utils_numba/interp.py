import numpy as np
from numba import jit
from math import erf, pow as _pow, sqrt

# __all__ = [
#     "simpson",
#     "trapz",
#     "trapz_dx",
#     "resample_template_numba",
#     "sample_gaussian_line_numba",
#     "pixel_integrated_gaussian_numba",
#     "compute_igm",
#     "calzetti2000_alambda",
#     "calzetti2000_attenuation",
#     "drude_profile",
#     "salim_alambda",
#     "smc_alambda",
#     "smc_attenuation",
# ]


# import numpy as np
# cimport numpy as np
# DTYPE = np.float64
# ctypedef np.float64_t DTYPE_t
# ctypedef np.int_t ITYPE_t
# ctypedef np.uint_t UINT_t
#
# import cython
#
# cdef extern from "math.h":
#     double fabs(double)

@jit(nopython=True, fastmath=True, error_model="numpy")
def abs(x):
    return sqrt(pow(x, 2))

@jit(nopython=True, fastmath=True, error_model="numpy")
def pixel_map_c(in_data, xi, yi, out_data, xo, yo):
    """
    pixel_map_c(in_data, xi, yi, out_data, xo, yo)
    
    Fast pixel mapping from one image to another:
        
        in_data[yi, xi] -> out_data[yo, xo]
        
    
    """
    # cdef unsigned long i, N

    N = len(xi)
    for i in range(N):
        out_data[yo[i], xo[i]] = in_data[yi[i], xi[i]]
        
    return True


@jit(nopython=True, fastmath=True, error_model="numpy")
def interp_c(x, xp, fp, extrapolate=0., assume_sorted=1):
    """
    interp_c(x, xp, fp, extrapolate=0., assume_sorted=0)
    
    Fast interpolation: [`xp`, `fp`] interpolated at `x`.
    
    Extrapolated values are set to `extrapolate`.
    
    The default `assume_sorted`=1 assumes that the `x` array is sorted and single-
    valued, providing a significant gain in speed. (xp is always assumed to be sorted)
    
    """
    # cdef unsigned long i, j, N, Np
    # cdef DTYPE_t x1,x2,y1,y2,out
    # cdef DTYPE_t fout, xval, xmin
    
    N, Np = len(x), len(xp)
    f = np.zeros(N)

    i=0
    j=0
    ### Handle left extrapolation
    xmin = xp[0]    
    if assume_sorted == 1:
        while x[j] < xmin: 
            f[j] = extrapolate
            j+=1
        
    while j < N:
        xval = x[j]
        if assume_sorted == 0:
            if x[j] < xmin:
                f[j] = extrapolate
                j+=1
                continue
            else:
                i=0
                
        while (xp[i] < xval) & (i < Np-1): i+=1;
        
        if i == (Np-1):
            if x[j] != xp[i]:
                f[j] = extrapolate
            else:
                f[j] = fp[i]
            j+=1
            continue   
        
        #### x[i] is now greater than xval because the 
        #### expression (x[i]<xval) is false, assuming
        #### that xval < max(x).
        
        # x1 = xp[i];
        # x2 = xp[i+1];
        # y1 = fp[i];
        # y2 = fp[i+1];
        x1 = xp[i-1];
        x2 = xp[i];
        y1 = fp[i-1];
        y2 = fp[i];
        out = ((y2-y1)/(x2-x1))*(xval-x1)+y1;
        f[j] = out
        j+=1
                
    return f


def interp_conserve(x, xp, fp, left=0., right=0.):
    """
    interp_conserve(x, xp, fp, left=0, right=0)
    
    Interpolate `xp`,`yp` array to the output x array, conserving flux.  
    `xp` can be irregularly spaced.
    """
    cdef np.ndarray fullx, fully, so, outy, dx
    cdef long N, i
    
    midpoint = (x[1:]-x[:-1])/2.+x[:-1]
    midpoint = np.append(midpoint, np.array([x[0],x[-1]]))
    midpoint = midpoint[np.argsort(midpoint)]
    int_midpoint = np.interp(midpoint, xp, fp, left=left, right=right)
    int_midpoint[midpoint > xp.max()] = 0.
    int_midpoint[midpoint < xp.min()] = 0.
    
    fullx = np.append(xp, midpoint)
    fully = np.append(fp, int_midpoint)
    
    so = np.argsort(fullx)
    fullx, fully = fullx[so], fully[so]
    
    outy = x*0.
    dx = midpoint[1:]-midpoint[:-1]
    for i in range(len(x)):
        bin = (fullx >= midpoint[i]) & (fullx <= midpoint[i+1])
        outy[i] = np.trapz(fully[bin], fullx[bin])/dx[i]
        
    return outy


@jit(nopython=True, fastmath=True, error_model="numpy")
def integral_cumsum_c(xp, fp, extrapolate=0.):
    """
    integral_cumsum_c(xp, fp, extrapolate=True)
    
    Cumulative trapz integration
    
    """
    # cdef np.ndarray[DTYPE_t, ndim=1] ycumsum
    # cdef long Nxp, i
    # cdef double x0, x1, old
    
    #### Implementation of cumsum * dx
    Nxp = len(xp)
    ycumsum = np.zeros_like(fp)
    
    x0 = xp[0]
    ycumsum[0] = fp[0] * (xp[1] - x0)
    
    old = ycumsum[0]
    for i in range(1,Nxp):
        x1 = xp[i]
        old += fp[i]*(x1-x0)
        ycumsum[i] = old
        x0 = x1
    
    return ycumsum


@jit(nopython=True, fastmath=True, error_model="numpy")
def new_interp_conserve_c(x, xp, fp, extrapolate=0.):
    """
    new_interp_conserve_c(x, xp, fp, extrapolate=0)
    
    Interpolate `xp`,`yp` array to the output x array, conserving flux.  
    `xp` can be irregularly spaced.
    """
    # cdef np.ndarray[DTYPE_t, ndim=1] ycumsum
    # cdef np.ndarray[DTYPE_t, ndim=1] inty
    # cdef np.ndarray[DTYPE_t, ndim=1] outy
    #
    # cdef long Nx, Nxp, i
    # cdef double x0, x1, old, y0, y1
    
    #### Implementation of cumsum * dx
    Nxp = len(xp)
    ycumsum = np.zeros(Nxp)
    x0 = xp[0]
    ycumsum[0] = fp[0]*(xp[1]-x0)
    
    old = ycumsum[0]
    for i in range(1,Nxp):
        x1 = xp[i]
        old += fp[i]*(x1-x0)
        ycumsum[i] = old
        x0 = x1
    
    Nx = len(x)
    inty = interp_c(x, xp, ycumsum, extrapolate=extrapolate)
    outy = np.zeros(Nx)
    
    x0 = x[0]
    y0 = inty[0]
    for i in range(1,Nx):
        y1 = inty[i]
        x1 = x[i]
        outy[i] = (y1-y0)/(x1-x0)
        x0 = x1
        y0 = y1
        
    outy[0] = outy[1]
    return outy
    
    
@jit(nopython=True, fastmath=True, error_model="numpy")
def interp_conserve_c(x, tlam, tf, left=0, right=0, integrate=0):
    """
    interp_conserve_c(x, xp, fp, left=0, right=0, integrate=0)
    
    Interpolate `xp`,`yp` array to the output x array, conserving flux.  
    `xp` can be irregularly spaced.
    """
    # cdef np.ndarray[DTYPE_t, ndim=1] templmid
    # cdef np.ndarray[DTYPE_t, ndim=1] tempfmid
    # cdef np.ndarray[DTYPE_t, ndim=1] outy
    # cdef unsigned long i,k,istart,ntlam,NTEMPL
    # cdef DTYPE_t h, numsum
    
    # templmid = (x[1:]+x[:-1])/2. #2.+x[:-1]
    # templmid = np.append(templmid, np.array([x[0], x[-1]]))
    # templmid = templmid[np.argsort(templmid)]
    NTEMPL = len(x)
    ntlam = len(tlam)

    templmid = midpoint_c(x, NTEMPL)
    #tempfmid = np.interp(templmid, tlam, tf, left=left, right=right)
    tempfmid = interp_c(templmid, tlam, tf, extrapolate=0.)
    
    outy = np.zeros_like(x)

    ###### Rebin template grid to master wavelength grid, conserving template flux
    i=0
    k=0
    while templmid[k] < tlam[0]:
        outy[k] = left
        k+=1
        
    for k in range(k, NTEMPL):
        if templmid[k] > tlam[ntlam-1]:
            break
            
        numsum=0.;

        #### Go to where tlam is greater than the first midpoint
        while (tlam[i] < templmid[k]) & (i < ntlam): i+=1;
        istart=i;

        ####### First point
        if tlam[i] < templmid[k+1]: 
            h = tlam[i]-templmid[k];
            numsum+=h*(tf[i]+tempfmid[k]);
            i+=1;

        if i==0: i+=1;

        ####### Template points between master grid points
        while (tlam[i] < templmid[k+1]) & (i < ntlam):
            h = tlam[i]-tlam[i-1];
            numsum+=h*(tf[i]+tf[i-1]);
            i+=1;

        #### If no template points between master grid points, then just use interpolated midpoints
        if i == istart:
            h = templmid[k+1]-templmid[k];
            numsum=h*(tempfmid[k+1]+tempfmid[k]);
        else:  
            ##### Last point              
            if (templmid[k+1] == tlam[i]) & (i < ntlam):
                h = tlam[i]-tlam[i-1];
                numsum+=h*(tf[i]+tf[i-1]);
            else:
                i-=1;
                h = templmid[k+1]-tlam[i];
                numsum+=h*(tempfmid[k+1]+tf[i]);

        outy[k] = numsum*0.5;#/(templmid[k+1]-templmid[k]);
        if integrate == 0.:
            outy[k] /= (templmid[k+1]-templmid[k]);
            
    return outy


@jit(nopython=True, fastmath=True, error_model="numpy")
def rebin_weighted_c(x, tlam, tf, te, left=0, right=0, integrate=0, remove_missing=1):
    """
    rebin_weighted_c(x, xp, fp, ep, left=0, right=0, integrate=0)

    Rebin `xp`,`yp` array to the output x array, weighting by `1/ep**2`.  
    `xp` can be irregularly spaced.
    """
    # cdef np.ndarray[DTYPE_t, ndim=1] templmid
    # cdef np.ndarray[DTYPE_t, ndim=1] tempfmid
    # cdef np.ndarray[DTYPE_t, ndim=1] outx
    # cdef np.ndarray[DTYPE_t, ndim=1] outy
    # cdef np.ndarray[DTYPE_t, ndim=1] oute
    # cdef unsigned long i,k,istart,ntlam,NTEMPL
    # cdef DTYPE_t h, numsum, densum

    NTEMPL = len(x)
    ntlam = len(tlam)

    outx = np.zeros(NTEMPL-1, dtype=tf.dtype)
    outy = np.zeros(NTEMPL-1, dtype=tf.dtype)
    oute = np.zeros(NTEMPL-1, dtype=tf.dtype)

    ###### Rebin template grid to master wavelength grid weighted by 1/e**2
    i=0
    k=0
    while (x[k] < tlam[0]) & (k < NTEMPL):
        outy[k] = left
        oute[k] = left
        k+=1
    
    if k > 0:
        k -= 1

    for k in range(k, NTEMPL-1):
        if x[k] > tlam[ntlam-1]:
            break

        xnumsum=0.;
        numsum=0.;
        densum=0.;
        
        #### Go to where tlam is greater than the x[k]
        while (tlam[i] < x[k]) & (i < ntlam): 
            i+=1;

        # print(i, x[k], tlam[i])
        
        ####### Template points between master grid points
        while (tlam[i] <= x[k+1]) & (i < ntlam):
            xnumsum += (tlam[i]/te[i]**2);
            numsum += (tf[i]/te[i]**2);
            densum += (1./te[i]**2)
            # count[i] += 1
            if i == ntlam-1:
                break

            i+=1;

        i-=1
        
        if densum > 0:
            outx[k] = xnumsum/densum
            outy[k] = numsum/densum
            oute[k] = 1/densum**0.5
    
    if remove_missing:
        ok = outx != 0
        return outx[ok], outy[ok], oute[ok]
    else:
        return outx, outy, oute


@jit(nopython=True, fastmath=True, error_model="numpy")
def midpoint(x):
    """
    midpoint(x)
    
    Get midpoints of an array
    """
    mp = (x[1:]+x[:-1])/2.
    mp = np.append(mp, np.array([x[0],x[-1]]))
    mp = mp[np.argsort(mp)]
    return mp


@jit(nopython=True, fastmath=True, error_model="numpy")
def midpoint_c(x, N):
    """
    midpoint_c(x, N)
    
    Get midpoints of array
    """
    # cdef long i
    # cdef DTYPE_t xi,xi1
    # N = len(x)
    midpoint = np.zeros(N+1, dtype=x.dtype)
    midpoint[0] = x[0]
    midpoint[N] = x[N-1]
    xi1 = x[0]
    for i in range(1, N):
        xi = x[i]
        midpoint[i] = 0.5*xi+0.5*xi1
        xi1 = xi
    
    midpoint[0] = 2*x[0]-midpoint[1]
    midpoint[N] = 2*x[N-1]-midpoint[N-1]
    
    return midpoint


@jit(nopython=True, fastmath=True, error_model="numpy")
def prepare_nmf_amatrix(variance, templates):
    """
    prepare_nmf_amatrix(variance, templates)
    
    Generate the "A" matrix needed for the NMF fit, which is essentially 
    T.transpose() dot T.  This function is separated from the main fitting routine 
    because it does not depend on the actual measured "flux", which the user
    might want to vary independent of the variance
    
    `templates` (T) is a 2D matrix of size (NTEMPLATE, NBAND) in terms of photo-z
    fitting ofphotometric bands.
    
    `variance` is an array with size (NBAND) representing the *measured* variance in 
    each band.  
    
    --- Cythonified from eazy/getphotz.c (G. Brammer et al. 2008) ---
    
    """
    
    cdef np.ndarray[DTYPE_t, ndim=2] amatrix
    cdef unsigned int i,j,k,NTEMP,NFILT
    
    NTEMP, NFILT = np.shape(templates)
    amatrix = np.zeros((NTEMP,NTEMP))

    for i in range(NTEMP):
        for j in range(NTEMP):
            amatrix[i,j] = 0.
            for k in range(NFILT):
                amatrix[i,j]+=templates[i,k]*templates[j,k]/variance[k]
            
    return amatrix


@jit(nopython=True, fastmath=True, error_model="numpy")
def run_nmf(flux, variance, templates, amatrix, toler=1.e-4, MAXITER=100000, init_coeffs=1, int verbose=0):
    """
    run_nmf(flux, variance, templates, amatrix, toler=1.e-4, MAXITER=100000, verbose=False)
    
    Run the "NMF" fit to determine the non-negative coefficients of the `templates`
    matrix that best-fit the observed `flux` and `variance` arrays.
    
    `amatrix` is generated with the `prepare_nmf_amatrix` function.
    
    e.g.
    
    >>> coeffs = run_nmf(flux, variance, templates, amatrix)
    >>> flux_fit = np.dot(coeffs.reshape((1,-1)), templates).flatten()
    >>> chi2 = np.sum((flux-flux_fit)**2/variance)
    
    --- Cythonified from eazy/getphotz.c (G. Brammer et al. 2008) ---
    """
    cdef unsigned long i,j,k,itcount, NTEMP, NFILT
    cdef double tolnum,toldenom,tol
    cdef double vold,av
    cdef np.ndarray[DTYPE_t, ndim=1] bvector
    
    NTEMP, NFILT = np.shape(templates)
            
    #### Make Bvector
    bvector = np.zeros(NTEMP, dtype=DTYPE)
    for i in range(NTEMP):
        bvector[i] = 0.;
        for k in range(NFILT):
            bvector[i]+=flux[k]*templates[i,k]/variance[k];
    
    #### Fit coefficients
    cdef np.ndarray[DTYPE_t, ndim=1] coeffs = np.ones(NTEMP, dtype=DTYPE)*init_coeffs
    tol = 100
    
    #### Lots of negative data, force coeffs to be zero
    for i in range(NTEMP):
        if bvector[i] < 0:
            coeffs[i] = 0.
            
    itcount=0;
    while (tol>toler) & (long(itcount)<MAXITER):
        tolnum=0.
        toldenom = 0.
        tol=0
        for i in range(NTEMP):
            vold = coeffs[i];
            av = 0.;
            for j in range(NTEMP): av+=amatrix[i,j]*coeffs[j];
            #### Update coeffs in place      
            coeffs[i]*=bvector[i]/av;
            #tolnum+=np.abs(coeffs[i]-vold);
            tolnum+=fabs(coeffs[i]-vold);
            toldenom+=vold;

        tol = tolnum/toldenom;
        
        if verbose & 2:
            print('Iter #{0:d}, tol={1:.2e}'.format(itcount, tol))
        
        itcount+=1
    
    if verbose & 1:
        print('Iter #{0:d}, tol={1:.2e}'.format(itcount, tol))

    return coeffs


@jit(nopython=True, fastmath=True, error_model="numpy")
def interpolate_tempfilt(tempfilt, zgrid, zi, output):
    """
    interpolate_tempfilt(tempfilt, zgrid, zi, output)
    
    Linear interpolate an Eazy "tempfilt" grid at z=zi.  
    
    `tempfilt` is [NFILT, NTEMP, NZ] integrated flux matrix
    `zgrid` is [NZ] redshift grid
    
    Result is stored in the input variable `output`, which needs shape [NFILT, NTEMP]
    """
    # cdef unsigned long NT, NF, NZ, itemp, ifilt, iz
    # cdef double dz, fint, fint2
    
    NF, NT, NZ = tempfilt.shape
    
    #### Output array
    #cdef np.ndarray[DTYPE_t, ndim=2] tempfilt_interp = np.zeros((NF, NT), dtype=DTYPE)
    
    for iz in range(NZ-1):
        dz = zgrid[iz+1]-zgrid[iz]
        fint = 1 - (zi-zgrid[iz])/dz
        if (fint > 0) & (fint <= 1):
            fint2 = 1 - (zgrid[iz+1]-zi)/dz
            # print iz, zgrid[iz], fint, fint2
            for ifilt in range(NF):
                for itemp in range(NT):
                    output[ifilt, itemp] = tempfilt[ifilt, itemp, iz]*fint + tempfilt[ifilt, itemp, iz+1]*fint2
            #
            break
                    
    # return output
    
