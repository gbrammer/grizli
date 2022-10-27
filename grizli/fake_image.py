"""
Tools for generating *very* basic fake images for HST/JWST/Roman simulations
"""
import os
import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from . import GRIZLI_PATH

def rotate_CD_matrix(cd, pa_aper):
    """Rotate CD matrix

    Parameters
    ----------
    cd : (2,2) array
        CD matrix

    pa_aper : float
        Position angle, in degrees E from N, of y axis of the detector

    Returns
    -------
    cd_rot : (2,2) array
        Rotated CD matrix

    Notes
    -----
    `astropy.wcs.WCS.rotateCD` doesn't work for non-square pixels in that it
    doesn't preserve the pixel scale!  The bug seems to come from the fact
    that `rotateCD` assumes a transposed version of its own CD matrix.

    For example:

        >>> import astropy.wcs as pywcs
        >>>
        >>> ## Nominal rectangular WFC3/IR pixel
        >>> cd_wfc3 = np.array([[  2.35945978e-05,   2.62448998e-05],
        >>>                     [  2.93050803e-05,  -2.09858771e-05]])
        >>>
        >>> ## Square pixel
        >>> cd_square = np.array([[0.1/3600., 0], [0, 0.1/3600.]])
        >>>
        >>> for cd, label in zip([cd_wfc3, cd_square], ['WFC3/IR', 'Square']):
        >>>     wcs = pywcs.WCS()
        >>>     wcs.wcs.cd = cd
        >>>     wcs.rotateCD(45.)
        >>>     print '%s pixel: pre=%s, rot=%s' %(label,
        >>>                         np.sqrt((cd**2).sum(axis=0))*3600,
        >>>                         np.sqrt((wcs.wcs.cd**2).sum(axis=0))*3600)

        WFC3/IR pixel:   pre=[ 0.1354  0.121 ], rot=[ 0.1282  0.1286]
        Square  pixel: pre=[ 0.1  0.1], rot=[ 0.1  0.1]

    """
    rad = np.deg2rad(-pa_aper)
    mat = np.zeros((2, 2))
    mat[0, :] = np.array([np.cos(rad), -np.sin(rad)])
    mat[1, :] = np.array([np.sin(rad), np.cos(rad)])
    cd_rot = np.dot(mat, cd)
    return cd_rot


def niriss_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589,
                  filter='F150W', grism='GR150R'):
    """Make JWST/NIRISS image header

    Parameters
    ----------
    ra, dec : float, float
        Coordinates of the center of the image

    pa_aper : float
        Position angle of the y-axis of the detector

    filter : str
        Blocking filter to use.

    grism : str
        Grism to use

    Returns
    -------
    h : `astropy.io.fits.Header`
        FITS header with appropriate keywords

    wcs : `astropy.wcs.WCS`
        WCS specification (computed from keywords in ``h``).

    Notes
    -----
    NIRISS: 0.065"/pix, requires filter & grism specification
    """
    naxis = 2048, 2048
    crpix = 1024, 1024

    cd = np.array([[-0.0658,  0], [0, 0.0654]])/3600.
    cd_rot = rotate_CD_matrix(cd, pa_aper)

    h = pyfits.Header()

    h['CRVAL1'] = ra
    h['CRVAL2'] = dec

    h['WCSAXES'] = 2
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'

    for i in range(2):
        h['NAXIS%d' % (i+1)] = naxis[i]
        h['CRPIX%d' % (i+1)] = crpix[i]
        h['CDELT%d' % (i+1)] = 1.0
        for j in range(2):
            h['CD%d_%d' % (i+1, j+1)] = cd_rot[i, j]

    # Backgrounds
    # http://www.stsci.edu/jwst/instruments/niriss/software-tools/wfss-simulations/niriss-wfss-cookbook.pdf
    bg = {'F090W': 0.50, 'F115W': 0.47, 'F140M': 0.23, 'F150W': 0.48, 'F158M': 0.25, 'F200W': 0.44}

    h['INSTRUME'] = 'NIRISS'
    h['TELESCOP'] = 'JWST'
    h['DETECTOR'] = 'NIS'

    if grism == 'GR150R':
        h['FILTER'] = 'GR150R', 'Spectral trace along Y'
    else:
        h['FILTER'] = 'GR150C', 'Spectral trace along X'

    h['PUPIL'] = filter
    
    h['BACKGR'] = bg[filter], 'Total, e/s'
    h['READN'] = 6, 'Rough, per pixel per 1 ks exposure'  # e/pix/per
    h['PHOTFLAM'] = 1.
    h['PHOTPLAM'] = 1.

    wcs = pywcs.WCS(h)
    h['EXTVER'] = 1

    return h, wcs


def nircam_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589,
                  filter='F444W', grism='GRISMR', module='A'):
    """Make JWST/NIRCAM image header

    Parameters
    ----------
    ra, dec : float, float
        Coordinates of the center of the image

    pa_aper : float
        Position angle of the y-axis of the detector

    filter : str
        Blocking filter to use.

    grism : str
        Grism to use ('GRISMR', 'GRISMC')
    
    module : str
        Instrument module ('A','B')
        
    Returns
    -------
    h : `astropy.io.fits.Header`
        FITS header with appropriate keywords

    wcs : `astropy.wcs.WCS`
        WCS specification (computed from keywords in ``h``).

    Notes
    -----
    NIRCAM, 0.0648"/pix, requires filter specification
    """
    naxis = 2048, 2048
    crpix = 1024, 1024

    cd = np.array([[-0.0648,  0], [0, 0.0648]])/3600.
    cd_rot = rotate_CD_matrix(cd, pa_aper)

    h = pyfits.Header()

    h['CRVAL1'] = ra
    h['CRVAL2'] = dec

    h['WCSAXES'] = 2
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'

    for i in range(2):
        h['NAXIS%d' % (i+1)] = naxis[i]
        h['CRPIX%d' % (i+1)] = crpix[i]
        h['CDELT%d' % (i+1)] = 1.0
        for j in range(2):
            h['CD%d_%d' % (i+1, j+1)] = cd_rot[i, j]

    # Backgrounds
    # http://www.stsci.edu/jwst/instruments/niriss/software-tools/wfss-simulations/niriss-wfss-cookbook.pdf
    bg = {'F277W': 0.30, 'F356W': 0.90, 'F444W': 3.00, 'F322W2': 1.25, 
          'F430M': 0.65, 'F460M': 0.86, 'F410M': 0.5}  # F410M is a hack, no number

    h['BACKGR'] = bg[filter], 'Total, e/s'
    h['INSTRUME'] = 'NIRCAM'
    h['TELESCOP'] = 'JWST'
    h['DETECTOR'] = f'NRC{module}LONG'
    h['MODULE'] = module
    h['CHANNEl'] = 'LONG'
    
    if grism == 'GRISMR':
        h['PUPIL'] = 'GRISMR', 'Spectral trace along X'
    else:
        h['PUPIL'] = 'GRISMC', 'Spectral trace along Y'

    h['FILTER'] = filter
    
    h['READN'] = 9, 'Rough, per pixel per 1 ks exposure'  # e/pix/per
    h['PHOTFLAM'] = 1.
    h['PHOTPLAM'] = 1.

    wcs = pywcs.WCS(h)
    h['EXTVER'] = 1

    return h, wcs


def wfc3ir_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589,
                  flt='ibhj34h6q_flt.fits', filter='G141'):
    """Make HST/WFC3-IR image header

    Parameters
    ----------
    ra, dec : float, float
        Coordinates of the center of the image

    pa_aper : float
        Position angle of the y-axis of the detector

    flt : str
        Filename of a WFC3/IR FLT file that will be used to provide the
        SIP geometric distortion keywords.

    filter : str
        Grism/filter to use.

    Returns
    -------
    h : `astropy.io.fits.Header`
        FITS header with appropriate keywords

    wcs : `astropy.wcs.WCS`
        WCS specification (computed from keywords in ``h``).

    Notes
    -----
    WFC3 IR, requires reference FLT file for the SIP header
    """
    import numpy as np

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    im = pyfits.open(flt)
    wcs = pywcs.WCS(im[1].header, relax=True)

    thet0 = np.arctan2(im[1].header['CD2_2'], im[1].header['CD2_1'])/np.pi*180

    wcs.wcs.crval = np.array([ra, dec])

    # Rotate the CD matrix
    theta = im[1].header['PA_APER'] - pa_aper
    cd_rot = rotate_CD_matrix(wcs.wcs.cd, theta)
    wcs.wcs.cd = cd_rot

    h = wcs.to_header(relax=True)

    for i in [1, 2]:
        for j in [1, 2]:
            h['CD%d_%d' % (i, j)] = h['PC%d_%d' % (i, j)]
            h.remove('PC%d_%d' % (i, j))

    h['BACKGR'] = 1.
    h['FILTER'] = filter
    h['INSTRUME'] = 'WFC3'
    h['READN'] = im[0].header['READNSEA']
    h['NAXIS1'] = h['NAXIS2'] = 1014
    h['DETECTOR'] = 'IR'
    h['PHOTFLAM'] = 1.
    h['PHOTPLAM'] = 1.

    return h, wcs


def wfirst_header(**kwargs):
    """
    Alias to `~grizli.fake_image.roman_header`
    """
    res = roman_header(**kwargs)
    return res


def roman_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589, naxis=(4096, 4096), **kwargs):
    """
    Make WFIRST/Roman WFI header

    Parameters
    ----------
    ra, dec : float, float
        Coordinates of the center of the image

    pa_aper : float
        Position angle of the y-axis of the detector

    filter : str
        Blocking filter to use.

    naxis : (int,int)
        Image dimensions
    
    Returns
    -------
    h : `astropy.io.fits.Header`
        FITS header with appropriate keywords

    wcs : `astropy.wcs.WCS`
        WCS specification (computed from keywords in `h`).

    Notes
    -----
    WFIRST/Roman G150 Grism

    Current config file has no field dependence, so field size can be
    anything you want in ``naxis``.
    """
    #naxis = 2048, 2048
    crpix = naxis[0]/2., naxis[0]/2.

    cd = np.array([[-0.11,  0], [0, 0.11]])/3600.
    cd_rot = rotate_CD_matrix(cd, pa_aper)

    h = pyfits.Header()

    h['CRVAL1'] = ra
    h['CRVAL2'] = dec

    h['WCSAXES'] = 2
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'

    for i in range(2):
        h['NAXIS%d' % (i+1)] = naxis[i]
        h['CRPIX%d' % (i+1)] = crpix[i]
        h['CDELT%d' % (i+1)] = 1.0
        for j in range(2):
            h['CD%d_%d' % (i+1, j+1)] = cd_rot[i, j]

    #h['BACKGR'] = 0.17+0.49, 'Total, e/s SDT Report A-1'
    h['BACKGR'] = 1.12, 'Pandeia minzodi/benchmark 20210528'
    h['FILTER'] = 'G150', 'WFIRST/Roman grism'
    h['INSTRUME'] = 'WFI'
    #h['READN'] = 17, 'SDT report Table 3-3'  # e/pix/per
    
    # https://roman.gsfc.nasa.gov/science/RRI/Roman_WFI_Reference_Information_20210125.pdf
    h['READN'] = 16., 'WFI Reference 20210125'  # e/pix/per
    h['PHOTFLAM'] = 1.
    h['PHOTPLAM'] = 1.
        
    wcs = pywcs.WCS(h)
    h['EXTVER'] = 1

    return h, wcs


def roman_hls_image(exptime=661.932, nexp=13, background=1.12, output='roman.fits', **kwargs):
    """
    Make a simple FITS file for a Roman High Latitude Survey Image
    
    Parameters
    ----------
    exptime, nexp, background : float, int, float
        Defaults specified to roughly match the variance in the `pandeia` 
        2D simulation result (ignoring Poisson from the source)
    
    kwargs : dict
        Positional keywords passed to `~grizli.fake_image.roman_header`
    
    Returns
    -------
    hdu : `astropy.io.fits.HDUList`
        HDU with SCI, ERR, DQ extensions
    
    wcs : `astropy.wcs.WCS`
        WCS
        
    """
    header, wcs = roman_header(**kwargs)
    hdu = make_fake_image(header, output=output, background=background, 
                          exptime=exptime, nexp=nexp)

    return hdu, wcs


def make_fake_image(header, output='direct.fits', background=None, exptime=1.e4, nexp=10, obsdate=None, seed=None):
    """
    Use the header from NIRISS, WFC3/IR or WFIRST/Roman and make an ``FLT``-like image that `grizli` can read as a reference.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        Header created by one of the generating functions, such as
        `~grizli.fake_image.niriss_header`.

    output : str
        Filename of the output FITS file. Will have extensions 'SCI', 'ERR',
        and 'DQ'. The 'ERR' extension is populated with a read-noise +
        background error model using

            >>> var = nexp*header['READN'] + background*exptime

        The 'SCI' extension is filled with gaussian deviates with standard
        deviation `sqrt(var)`.

        The 'DQ' extension is filled with (int) zeros.

    background : None or float
        Background value to use for sky noise.  If None, then read from
        `header['BACKGR']`.

    exptime : float
        Exposure time to use for background sky noise.
    
    obsdate : `~astropy.time.Time`
        Date of observation.  If None, then use `astropy.time.Time.now`
        
    nexp : int
        Number of exposures to use for read noise.
    
    seed : int
        If specified, use as `numpy.random.seed`
        
    Returns
    -------
    hdu : `astropy.io.fits.HDUList`
        Image HDU (also saved to ``output`` FITS file)
        
    """
    import astropy.time
    import astropy.units as u
    
    hdu = pyfits.HDUList()

    header['EXPTIME'] = exptime
    header['NEXP'] = nexp
    header['BUNIT'] = 'ELECTRONS/S'

    hdu.append(pyfits.PrimaryHDU(header=header))

    naxis = (header['NAXIS1'], header['NAXIS2'])

    if background is None:
        background = header['BACKGR']

    header['BACKGR'] = background
    
    if obsdate is None:
        obsdate = astropy.time.Time.now()
    
    header['DATE-OBS'] = obsdate.iso.split()[0]
    header['TIME-OBS'] = obsdate.iso.split()[1]
    header['EXPSTART'] = obsdate.mjd
    header['EXPEND'] = (obsdate + exptime*u.second).mjd
    
    # Simple error model of read noise and sky background
    var = nexp*header['READN'] + background*exptime

    # electrons / s
    rms = np.sqrt(var)/exptime
    header['CALCRMS'] = rms, 'Variance used for random noise'

    for name, dtype in zip(['SCI', 'ERR', 'DQ'],
                           [np.float32, np.float32, np.int32]):
        hdu.append(pyfits.ImageHDU(header=header,
                                   data=np.zeros(np.array(naxis).T,
                                   dtype=dtype), name=name))

    hdu['ERR'].data += rms 
    if seed is not None:
        np.random.seed(seed)
        hdu['ERR'].header['SEED'] = seed, 'Random seed'
           
    hdu['SCI'].data = np.random.normal(size=np.array(naxis).T)*rms

    if output is not None:
        hdu.writeto(output, overwrite=True, output_verify='fix')
    
    return hdu


def make_roman_config(save_to_conf=True):
    """
    Use `pandeia` to calculate a Roman/G150 configuration file and sensitivity curve for `grizli`
    
    https://github.com/spacetelescope/roman_tools/blob/develop/notebooks/Pandeia-Roman.ipynb
    
    Parameters
    ----------
    save_to_conf : bool
        Write sensitivity and configuration files to ``[GRIZLI_PATH]/CONF``
    
    Returns
    -------
    sens : `~astropy.table.Table`
        Sensitivity table
    
    conf : str
        Grism configuration
        
    """
    from astropy.table import Table
    import astropy.time
    import pandeia.engine
    
    from pandeia.engine.perform_calculation import perform_calculation
    from pandeia.engine.calc_utils import (get_telescope_config, 
                                           get_instrument_config, 
                                           build_default_calc,
                                           build_default_source)
                                           
    from pandeia.engine.io_utils import read_json, write_json
    
    calc = build_default_calc('roman','wfi','spectroscopy')
    
    # HLS simulation
    calc['configuration']['instrument']['filter'] = None
    calc['configuration']['instrument']['aperture'] = "any"
    calc['configuration']['instrument']['disperser'] = "g150"
    calc['configuration']['detector']['ngroup'] = 13 # groups per integration
    calc['configuration']['detector']['nint'] = 1 # integrations per exposure
    calc['configuration']['detector']['nexp'] = 1 # exposures
    calc['configuration']['detector']['readmode'] = "medium8"
    calc['configuration']['detector']['subarray'] = "1024x1024"
    
    calc['scene'][0]['spectrum']['normalization']['norm_fluxunit'] = 'flam'
    
    input_flux = 1.e-19
    calc['scene'][0]['spectrum']['normalization']['norm_flux'] = input_flux
    calc['scene'][0]['spectrum']['sed']['unit'] = 'flam'
    
    # x,y location to extract, in arcsec
    calc['strategy']['target_xy'] = [0.0,0.0] 
    # radius of extraction aperture, in arcsec
    calc['strategy']['aperture_size'] = 0.6 
    # inner and outer radii of background subtraction annulus, in arcsec
    calc['strategy']['sky_annulus'] = [0.8,1.2] 
    
    results = perform_calculation(calc)
    
    sens = Table()
    wave = np.arange(9000, 1.95e4, 2.)
    sens_value = results['1d']['extracted_flux'][1]/input_flux
    sens_value /= np.gradient(results['1d']['extracted_flux'][0]*1.e4)
    
    sens['WAVELENGTH'] = wave
    sens['SENSITIVITY'] = np.interp(wave, 
                                    results['1d']['extracted_flux'][0]*1.e4, 
                                    sens_value,
                                    left=0, right=0)
    
    sens['ERROR'] = 0.01*sens_value.max()
    
    sens.meta['pandeia'] = pandeia.engine.__version__
    sens.meta['created'] = astropy.time.Time.now().iso
    
    sens_file = f'Roman.G150.v{pandeia.engine.__version__}.sens.fits'
    
    if save_to_conf:
        if isinstance(save_to_conf, str):
            path = save_to_conf
        else:
            path = os.path.join(GRIZLI_PATH, 'CONF')
        
        print('Sensitivity file: ', os.path.join(path, sens_file))
        sens.write(os.path.join(path, sens_file), overwrite=True)
    
    npix = len(results['1d']['extracted_flux'][0])
    pad = 20
    
    i0 = npix//2
    w0 = results['1d']['extracted_flux'][0][i0]*1.e4
    dlam = np.diff(results['1d']['extracted_flux'][0])[i0]*1.e4
           
    config = f"""INSTRUMENT WFI
GFILTER G150 

# First order (BEAM A) 
# BEAMA and DLDP assume spectrum is centered on the imaging position
BEAMA {-npix//2-pad} {npix//2+pad+1}
MMAG_EXTRACT_A 30
MMAG_MARK_A 30

#
# Trace description
# (flat along rows)
DYDX_ORDER_A 0
DYDX_A_0 0

#
# X and Y Offsets
#
XOFF_A 0.0
YOFF_A 0.0

#
# Dispersion solution
#
DISP_ORDER_A 1
DLDP_A_0 {w0}
DLDP_A_1 {dlam}

SENSITIVITY_A {sens_file}
"""
    
    if save_to_conf:
        print('Config file: ', os.path.join(path, 'Roman.G150.conf'))
        with open(os.path.join(path, 'Roman.G150.conf'), 'w') as fp:
            fp.write(config)
    
    return sens, config