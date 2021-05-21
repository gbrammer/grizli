#!/usr/bin/env python
# encoding: utf-8
"""
deconvolve.py

GALFIT "deconvolution" as in Szomoru et al.
https://iopscience.iop.org/article/10.1088/2041-8205/714/2/L244

"""

import numpy as np

import astropy.io.fits as pyfits
import astropy.stats
import astropy.table


def test():

    im = pyfits.open('/tmp/gf_out.fits')
    model_hdu = im[2]
    residuals = im[3].data

    _x = galfit_deconvolve(model_hdu, residuals, rms=None)

    model_hdu = _x['model']
    residuals = _x['resid'].data

    imshow_args = {'vmin': -5, 'vmax': 5, 'cmap': 'viridis'}

    _ = galfit_deconvolve(model_hdu, residuals, rms=_x['rms'], imshow_args=imshow_args)


def galfit_deconvolve(model_hdu, residuals, rms=None, mask=None, oversample=8, comp='1', xprof=np.append(0, np.arange(2, 20)), y_func=np.mean, cumul_values=[0.5, 0.8], make_plot=True, imshow_args={'vmin': -5, 'vmax': 5, 'cmap': 'viridis'}, plt_kwargs={'linestyle': 'steps-mid', 'color': 'r', 'alpha': 0.8}, npix=20, psf_offset=[1, 1]):
    """
    Deconvolve an image using a galfit Sersic model

    `model_hdu` is the HDU output from GALFIT containing the model and model
    parameters (e.g, out.fits[2]).

    `residuals` are the fit residuals (out.fits[3].data)

    `psf_offset` is the offset of the PSF relative to the central pixel of
    the PSF array.

    Returns:

        R = radii evaluated at pixel positions for the ellipse parameters
        phi = position angle around the ellipse, radians
        prof = 2D array of unconvolved Sersic profile
        params = Galfit params
        tab = table of the 1D profile
        fig = optional figure if `make_plot=True`

    """
    import matplotlib.pyplot as plt

    #model_hdu = _x['model']
    #residuals = _x['resid']

    _h = model_hdu.header
    shape = residuals.shape

    if mask is None:
        mask = np.isfinite(residuals)

    # Ellipse parameters and sersic profile
    R, phi, prof, params = sersic_profile(shape, oversample=oversample,
                                          gf_header=_h, comp=comp,
                                          psf_offset=psf_offset)

    so = np.argsort(R[mask].flatten())
    Rso = R[mask].flatten()[so]

    data = residuals + model_hdu.data

    if '1_SKY' in _h:
        sky = gf_header_key(_h, '1_SKY')
    else:
        sky = 0.

    ##########
    # 1D averaged profiles

    integ = False

    # Original data
    _, ydata, _, _ = running_median(Rso,
                (data)[mask].flatten()[so].astype(np.float)-sky,
                bins=xprof, use_median=False, y_func=y_func, integrate=integ)
    
    # Convolved model
    _, ymodel, _, _ = running_median(Rso,
                (model_hdu.data)[mask].flatten()[so].astype(np.float),
                bins=xprof, use_median=False, y_func=y_func, integrate=integ)
    
    # Sersic profile
    _, yprof, _, _ = running_median(Rso,
                (prof)[mask].flatten()[so].astype(np.float),
                bins=xprof, use_median=False, y_func=y_func, integrate=integ)

    # Sersic + residuals
    xm, ydeconv, ys, yn = running_median(Rso,
                (prof + residuals)[mask].flatten()[so].astype(np.float),
                bins=xprof, use_median=False, y_func=y_func, integrate=integ)

    # Sum for total normalizatioin
    _, ydeconv_sum, _, _ = running_median(Rso,
                (prof + residuals)[mask].flatten()[so].astype(np.float),
                bins=xprof, use_median=False, y_func=np.sum, integrate=False)

    # Variance
    if rms is not None:
        xmx, yv, yvs, yn = running_median(Rso,
                rms[mask].flatten()[so].astype(np.float)**2,
                bins=xprof, use_median=False, y_func=np.sum)

        yrms = np.sqrt(yv)/yn
        im_norm = rms
        
        # weighted
        xmx, ynum_model, yvs, yn = running_median(Rso,
                (model_hdu.data/rms**2)[mask].flatten()[so].astype(np.float),
                bins=xprof, use_median=False, y_func=np.sum)

        xmx, ynum, yvs, yn = running_median(Rso,
                (data/rms**2)[mask].flatten()[so].astype(np.float),
                bins=xprof, use_median=False, y_func=np.sum)
                
        xmx, yden, yvs, yn = running_median(Rso,
                (1./rms**2)[mask].flatten()[so].astype(np.float),
                bins=xprof, use_median=False, y_func=np.sum)
        
        yweight = ynum/yden
        yweight_model = ynum_model/yden
        yweight_err = 1./np.sqrt(yden)
        
    else:
        yrms = ys
        im_norm = 1
        yweight = None
        
    dx = np.diff(xprof)
    xpix = xprof[1:]-dx/2.

    if False:
        # Scale
        yscl = 1.

        #yscl = prof[msk].sum()/np.trapz(2*np.pi*yprof*xpix, xpix)
        yprof *= yscl
        ydeconv *= yscl
        yrms *= yscl

    # Interpolate Radii from the cumulative flux distribution
    cumul_flux = np.cumsum((prof + residuals)[mask].flatten()[so])

    #total = cumflux.max()
    total = ydeconv_sum.sum()

    Rcumul = np.interp(cumul_values, cumul_flux/total, Rso)

    tab = astropy.table.Table()
    tab['xpix'] = xpix
    tab['yprofile'] = yprof
    tab['ydeconv'] = ydeconv
    tab['yrms'] = yrms
    tab['ydata'] = ydata
    tab['ymodel'] = ymodel
    
    if yweight is not None:
        tab['yweight'] = yweight
        tab['yweight_model'] = yweight_model
        tab['yweight_err'] = yweight_err
        
    tab.meta['total_flux'] = total
    for k in params:
        tab.meta[k] = params[k]

    for i, r in enumerate(cumul_values):
        tab.meta['R{0}'.format(int(r*100))] = Rcumul[i]

    params['Rcumul'] = Rcumul

    if make_plot:

        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)

        ax4 = fig.add_subplot(144)
        ax4.loglog()
        ax4.grid()

        xc = gf_header_key(_h, comp+'_XC')
        yc = gf_header_key(_h, comp+'_YC')

        ax.imshow(data/im_norm*mask, **imshow_args)
        ax2.imshow(residuals/im_norm*mask, **imshow_args)
        ax3.imshow(prof/im_norm*mask, **imshow_args)

        label = 'Re={re:.2f}, n={n:.1f}'.format(**params)
        pl = ax4.plot(xpix, yprof, label=label, **plt_kwargs)

        ax4.errorbar(xpix, ydeconv, yrms, linestyle='None',
                     color=pl[0].get_color(), alpha=0.5, marker='.')

        ax4.vlines(params['re'], 1e-10, 1e10, color=pl[0].get_color(), alpha=0.5)
        ax4.vlines(Rcumul, 1e-10, 1e10, color=pl[0].get_color(), alpha=0.5, linewidth=3, linestyle='--')

        ax4.scatter(Rso,
                (prof + residuals)[mask].flatten()[so].astype(np.float)-sky,
                marker='.', color='k', alpha=0.1, zorder=-1000)

        # ax4.scatter(Rso,
        #        (data)[mask].flatten()[so].astype(np.float)-sky,
        #        marker='.', color='r', alpha=0.1, zorder=-1000)

        ax4.legend()

        valid = np.isfinite(ydeconv) & np.isfinite(yprof)

        print(ydeconv[valid].min(), ydeconv[valid].max())

        try:
            ax4.set_ylim(np.maximum(0.5*yprof[valid].min(), 1e-5*ydeconv[valid].max()),
                     2*ydeconv[valid].max())
        except:
            pass

        ax4.set_xlim(0.05, xprof.max()+2)

        for a in [ax, ax2, ax3]:
            a.set_xlim(xc-1-npix, xc-1+npix)
            a.set_ylim(yc-1-npix, yc-1+npix)
            a.set_xticklabels([])
            a.set_yticklabels([])

            # Show ellipses on images
            a.plot(*r_ellipse(Rcumul[0], psf_offset=psf_offset, **params),
                   color='w', alpha=0.9, linewidth=2)
            a.plot(*r_ellipse(xpix.max(), psf_offset=psf_offset, **params),
                   color='w', alpha=0.9, linewidth=2)

            a.plot(*r_ellipse(Rcumul[0], psf_offset=psf_offset, **params),
                   color=pl[0].get_color(), alpha=0.5)
            a.plot(*r_ellipse(xpix.max(), psf_offset=psf_offset, **params),
                   color=pl[0].get_color(), alpha=0.5, linestyle='--')

        fig.tight_layout(pad=0.1)

    else:
        fig = None

    return R, phi, prof, params, tab, fig


def get_kappa(n, **kwargs):
    """
    Compute the Kappa parameter for a given n index as in the Galfit
    definition
    """
    from scipy.optimize import root
    x0 = 2*n-0.33
    args = (n)
    k = root(kappa_func, x0, args=args, **kwargs)
    return k.x[0]


def kappa_func(kappa, n):
    """
    Function for getting Sersic kappa
    """
    from scipy.special import gamma, gammainc
    f = gamma(2*n)-2*gammainc(2*n, kappa)*gamma(2*n)
    return f


def Rc(c0):
    """
    Shape parameter
    """
    from scipy.special import beta

    return np.pi*(c0+2)/(4*beta(1./(c0+2), 1+1./(c0+2)))


def sigma_e(re, n, q, Ftot=1., c0=0.):
    """
    Surface brightess at the effective radius, re, given total flux
    """
    from scipy.special import gamma
    kap = get_kappa(n)
    return Ftot/(2*np.pi*re**2*np.exp(kap)*n*kap**(-2*n)*gamma(2*n)*q/Rc(c0)), kap


def gf_header_key(header, key='2_XC'):
    """
    Get a header keyword from a GALFIT header, which may have [] and *
    in the keyword value
    """
    return float(header[key].split()[0].strip('[]').strip('*').strip('{').strip('}'))


def sersic_profile(shape, mag=20., xc=[0., 0.], n=1., q=0.5, pa=0, re=1., ZP=26., gf_header=None, comp='2', verbose=True, oversample=8, psf_offset=[1, 1]):
    """
    Generate a Sersic profile with Galfit parameters within a defined image
    shape.

    Specify the parameters individually or provide a GALFIT model header

    gf_header: FITS header of a GALFIT output model
    comp: Number of the object in the GALFIT model

    """
    import scipy.ndimage as nd

    if gf_header is not None:
        xx = gf_header_key(gf_header, comp+'_XC')
        yy = gf_header_key(gf_header, comp+'_YC')
        xc = np.array([xx, yy])
        mag = gf_header_key(gf_header, comp+'_MAG')
        if comp+'_N' in gf_header:
            n = gf_header_key(gf_header, comp+'_N')
            q = gf_header_key(gf_header, comp+'_AR')
            pa = gf_header_key(gf_header, comp+'_PA')
            re = gf_header_key(gf_header, comp+'_RE')
        else:
            n = 1.
            q = 1.
            pa = 0.
            re = 0.01

        if verbose:
            print(f'xc:{xc}, q:{q}, pa:{pa}, n:{n}')

        if 'MAGZPT' in gf_header:
            ZP = gf_header['MAGZPT']

    params = {'mag': mag, 'xc': xc, 're': re, 'n': n, 'q': q, 'pa': pa}

    sigm, kap = sigma_e(re, n, q)
    norm = sigm*10**(-0.4*(mag-ZP))
    R, x, phi = pix_to_r(shape, verbose=verbose, psf_offset=psf_offset,
                         **params)

    if oversample > 1:
        Ro, x, phio = pix_to_r(shape, oversample=oversample, verbose=verbose,
                              psf_offset=psf_offset, **params)

        sersic_large = norm*np.exp(-kap*((Ro/re)**(1./n)-1))
        kern = np.ones((oversample, oversample))/oversample**2
        sersic_profile = nd.convolve(sersic_large, kern)[oversample//2-1::oversample, oversample//2-1::oversample]
    else:
        sersic_profile = norm*np.exp(-kap*((R/re)**(1./n)-1))

    return R, phi, sersic_profile, params


def r_ellipse(radius=5, xc=[0., 0.], q=0.5, pa=0, re=1., gf_header=None, comp='2', verbose=True, nstep=256, psf_offset=[1, 1], **kwargs):
    """
    Make x, y coordinates given ellipse parameters
    """
    if gf_header is not None:
        xx = gf_header_key(gf_header, comp+'_XC')
        yy = gf_header_key(gf_header, comp+'_YC')
        xc = np.array([xx, yy])
        mag = gf_header_key(gf_header, comp+'_MAG')

        if comp+'_N' in gf_header:
            n = gf_header_key(gf_header, comp+'_N')
            q = gf_header_key(gf_header, comp+'_AR')
            pa = gf_header_key(gf_header, comp+'_PA')
            re = gf_header_key(gf_header, comp+'_RE')
        else:
            n = 1.
            q = 1.
            pa = 0.
            re = 0.01

        if verbose:
            print(f'xc:{xc}, q:{q}, pa:{pa}')

    phi = np.linspace(0, 2*np.pi, nstep)
    xp = np.array([np.cos(phi), q*np.sin(phi)]).T*radius

    theta = -(np.pi/2 + pa/180*np.pi)  # + np.pi

    _rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    x0 = xp.dot(_rot) + np.atleast_1d(xc)
    xell, yell = (x0 - 1 - np.atleast_1d(psf_offset)).T

    return xell, yell


def pix_to_r(shape, xc=[0., 0.], q=0.5, pa=0, re=1., gf_header=None, comp='2', verbose=True, oversample=1, psf_offset=[1, 1], **kwargs):
    """
    Return an array of ellipse radii for a pixel grid and ellipse parameters
    """

    if oversample > 1:
        xarr = np.arange(-0.5, shape[1]-0.5, 1/oversample)
        yarr = np.arange(-0.5, shape[0]-0.5, 1/oversample)
    else:
        xarr = np.arange(0, shape[1], 1/oversample)
        yarr = np.arange(0, shape[0], 1/oversample)

    x0 = np.array(np.meshgrid(xarr, yarr)).reshape((2, -1)).T + 1
    x0 += np.atleast_1d(psf_offset)

    if gf_header is not None:
        xx = gf_header_key(gf_header, comp+'_XC')
        yy = gf_header_key(gf_header, comp+'_YC')
        xc = np.array([xx, yy])
        mag = gf_header_key(gf_header, comp+'_MAG')

        if comp+'_N' in gf_header:
            n = gf_header_key(gf_header, comp+'_N')
            q = gf_header_key(gf_header, comp+'_AR')
            pa = gf_header_key(gf_header, comp+'_PA')
            re = gf_header_key(gf_header, comp+'_RE')
        else:
            n = 1.
            q = 1.
            pa = 0.
            re = 0.01

        if verbose:
            print(f'xc:{xc}, q:{q}, pa:{pa}')

    theta = (np.pi/2 + pa/180*np.pi)  # + np.pi

    _rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    xp = (x0 - np.atleast_1d(xc)).dot(_rot)

    outshape = [s*oversample for s in shape]

    R = np.sqrt(((xp/np.array([1, q]))**2).sum(axis=1)).reshape(outshape)

    phi = np.arctan2(xp[:, 1], xp[:, 0]).reshape(outshape)

    return R, re, phi


def running_median(xi, yi, NBIN=10, use_median=True, use_nmad=True, reverse=False, bins=None, x_func=astropy.stats.biweight_location, y_func=astropy.stats.biweight_location, std_func=astropy.stats.biweight_midvariance, integrate=False):
    """
    Running median/biweight/nmad
    """

    NPER = xi.size // NBIN
    if bins is None:
        so = np.argsort(xi)
        if reverse:
            so = so[::-1]

        bx = np.linspace(0, len(xi), NBIN+1)
        bins = np.interp(bx, np.arange(len(xi)), xi[so])
        if reverse:
            bins = bins[::-1]

    NBIN = len(bins)-1

    xm = np.arange(NBIN)*1.
    xs = xm*0
    ym = xm*0
    ys = xm*0
    N = np.arange(NBIN)

    if use_median:
        y_func = np.median

    if use_nmad:
        std_func = astropy.stats.mad_std

    for i in range(NBIN):
        in_bin = (xi > bins[i]) & (xi <= bins[i+1])
        N[i] = in_bin.sum()  # N[i] = xi[so][idx+NPER*i].size

        if integrate:
            xso = np.argsort(xi[in_bin])
            ma = xi[in_bin].max()
            mi = xi[in_bin].min()
            xm[i] = (ma+mi)/2.
            dx = (ma-mi)
            ym[i] = np.trapz(yi[in_bin][xso], xi[in_bin][xso])/dx
        else:
            xm[i] = x_func(xi[in_bin])
            ym[i] = y_func(yi[in_bin])

        ys[i] = std_func(yi[in_bin])

    return xm, ym, ys, N
