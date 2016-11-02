"""
Simulation tools for generating fake images
"""
import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

def rotate_CD_matrix(cd, pa_aper):
    """Rotate CD matrix
    
    Parameters
    ----------
    cd: (2,2) array
        CD matrix
    
    pa_aper: float
        Position angle, in degrees E from N, of y axis of the detector
    
    Returns
    -------
    cd_rot: (2,2) array
        Rotated CD matrix
    
    Comments
    --------
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
    mat = np.zeros((2,2))
    mat[0,:] = np.array([np.cos(rad),-np.sin(rad)])
    mat[1,:] = np.array([np.sin(rad),np.cos(rad)])
    cd_rot = np.dot(mat, cd)
    return cd_rot
        
def niriss_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589, 
                  filter='F150W', grism='GR150R'):
    """Make JWST/NIRISS image header
    
    Parameters
    ----------
    ra, dec: float, float
        Coordinates of the center of the image
    
    pa_aper: float
        Position angle of the y-axis of the detector
    
    filter: str
        Blocking filter to use.
    
    grism: str
        Grism to use
    
    Returns
    --------
    h: astropy.io.fits.Header
        FITS header with appropriate keywords
    
    wcs: astropy.wcs.WCS
        WCS specification (computed from keywords in `h`).
    
    Comments
    --------
    NIRISS: 0.065"/pix, requires filter & grism specification
    """ 
    naxis = 2048, 2048
    crpix = 1024, 1024
    
    cd = np.array([[ -0.0658,  0], [0, 0.0654]])/3600.
    cd_rot = rotate_CD_matrix(cd, pa_aper)
    
    h = pyfits.Header()
    
    h['CRVAL1'] = ra
    h['CRVAL2'] = dec
    
    h['WCSAXES'] = 2
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    
    for i in range(2):
        h['NAXIS%d' %(i+1)] = naxis[i]
        h['CRPIX%d' %(i+1)] = crpix[i]
        h['CDELT%d' %(i+1)] = 1.0
        for j in range(2):
            h['CD%d_%d' %(i+1, j+1)] = cd_rot[i,j]
    
    ### Backgrounds
    # http://www.stsci.edu/jwst/instruments/niriss/software-tools/wfss-simulations/niriss-wfss-cookbook.pdf
    bg = {'F090W':0.50, 'F115W':0.47, 'F140M':0.23, 'F150W':0.48, 'F158M':0.25, 'F200W':0.44}
    
    h['BACKGR'] = bg[filter], 'Total, e/s'
    h['FILTER'] = filter
    h['INSTRUME'] = 'NIRISS'
    h['READN'] = 6 , 'Rough, per pixel per 1 ks exposure' # e/pix/per
    h['PHOTFLAM'] = 1.
    h['PHOTPLAM'] = 1.
    
    if grism == 'GR150R':
        h['GRISM'] = 'GR150R', 'Spectral trace along X'
    else:
        h['GRISM'] = 'GR150C', 'Spectral trace along Y'
        
    wcs = pywcs.WCS(h)
    h['EXTVER'] = 1
        
    return h, wcs

def nircam_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589, 
                  filter='F444W', grism='DFSR'):
    """Make JWST/NIRCAM image header

    Parameters
    ----------
    ra, dec: float, float
      Coordinates of the center of the image

    pa_aper: float
      Position angle of the y-axis of the detector

    filter: str
      Blocking filter to use.

    grism: str
      Grism to use

    Returns
    --------
    h: astropy.io.fits.Header
      FITS header with appropriate keywords

    wcs: astropy.wcs.WCS
      WCS specification (computed from keywords in `h`).

    Comments
    --------
    NIRCAM, 0.0648"/pix, requires filter specification
    """ 
    naxis = 2048, 2048
    crpix = 1024, 1024
    
    cd = np.array([[ -0.0648,  0], [0, 0.0648]])/3600.
    cd_rot = rotate_CD_matrix(cd, pa_aper)
    
    h = pyfits.Header()
    
    h['CRVAL1'] = ra
    h['CRVAL2'] = dec
    
    h['WCSAXES'] = 2
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    
    for i in range(2):
        h['NAXIS%d' %(i+1)] = naxis[i]
        h['CRPIX%d' %(i+1)] = crpix[i]
        h['CDELT%d' %(i+1)] = 1.0
        for j in range(2):
            h['CD%d_%d' %(i+1, j+1)] = cd_rot[i,j]
    
    ### Backgrounds
    # http://www.stsci.edu/jwst/instruments/niriss/software-tools/wfss-simulations/niriss-wfss-cookbook.pdf
    bg = {'F277W':0.30, 'F356W':0.90, 'F444W': 3.00, 'F322W2':1.25, 'F430M':0.65, 'F460M':0.86, 'F410M':0.5} # F410M is a hack, no number
    
    h['BACKGR'] = bg[filter], 'Total, e/s'
    h['FILTER'] = filter
    h['INSTRUME'] = 'NIRCam'
    h['READN'] = 9, 'Rough, per pixel per 1 ks exposure' # e/pix/per
    h['PHOTFLAM'] = 1.
    h['PHOTPLAM'] = 1.
    
    if grism == 'DFSR':
        h['GRISM'] = 'DFSR', 'Spectral trace along X'
    else:
        h['GRISM'] = 'DFSC', 'Spectral trace along Y'
        
    wcs = pywcs.WCS(h)
    h['EXTVER'] = 1
        
    return h, wcs
    
def wfc3ir_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589, 
                  flt='ibhj34h6q_flt.fits', filter='G141'):
    """Make HST/WFC3-IR image header
    
    Parameters
    ----------
    ra, dec: float, float
        Coordinates of the center of the image
    
    pa_aper: float
        Position angle of the y-axis of the detector
    
    flt: str
        Filename of a WFC3/IR FLT file that will be used to provide the 
        SIP geometric distortion keywords.
        
    filter: str
        Grism/filter to use.
    
    Returns
    --------
    h: astropy.io.fits.Header
        FITS header with appropriate keywords
    
    wcs: astropy.wcs.WCS
        WCS specification (computed from keywords in `h`).
    
    Comments
    --------
    
    WFC3 IR, requires reference FLT file for the SIP header
    """
    import numpy as np
    
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    
    im = pyfits.open(flt)
    wcs = pywcs.WCS(im[1].header, relax=True)
    
    thet0 = np.arctan2(im[1].header['CD2_2'], im[1].header['CD2_1'])/np.pi*180

    wcs.wcs.crval = np.array([ra, dec])
    
    ### Rotate the CD matrix
    theta = im[1].header['PA_APER'] - pa_aper 
    cd_rot = rotate_CD_matrix(wcs.wcs.cd, theta)
    wcs.wcs.cd = cd_rot
    
    h = wcs.to_header(relax=True)
    
    for i in [1,2]:
        for j in [1,2]:
            h['CD%d_%d' %(i,j)] = h['PC%d_%d' %(i,j)]
            h.remove('PC%d_%d' %(i,j))
    
    h['BACKGR'] = 1.
    h['FILTER'] = filter
    h['INSTRUME'] = 'WFC3'
    h['READN'] = im[0].header['READNSEA']
    h['NAXIS1'] = h['NAXIS2'] = 1014
    h['DETECTOR'] = 'IR'
    h['PHOTFLAM'] = 1.
    h['PHOTPLAM'] = 1.
    
    return h, wcs

def wfirst_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589, naxis=(4096,4096)):
    """Make WFIRST WFI header
    
    Parameters
    ----------
    ra, dec: float, float
        Coordinates of the center of the image
    
    pa_aper: float
        Position angle of the y-axis of the detector
    
    filter: str
        Blocking filter to use.
    
    naxis: (int,int)
        Image dimensions
    
    Returns
    --------
    h: astropy.io.fits.Header
        FITS header with appropriate keywords
    
    wcs: astropy.wcs.WCS
        WCS specification (computed from keywords in `h`).
    
    Comments
    --------
    WFIRST GRS Grism
    
    Current aXe config file has no field dependence, so field size can be
    anything you want in `naxis`.
    """
    #naxis = 2048, 2048
    crpix = naxis[0]/2., naxis[0]/2.
    
    cd = np.array([[ -0.11,  0], [0, 0.11]])/3600.
    cd_rot = rotate_CD_matrix(cd, pa_aper)
    
    h = pyfits.Header()
    
    h['CRVAL1'] = ra
    h['CRVAL2'] = dec
    
    h['WCSAXES'] = 2
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    
    for i in range(2):
        h['NAXIS%d' %(i+1)] = naxis[i]
        h['CRPIX%d' %(i+1)] = crpix[i]
        h['CDELT%d' %(i+1)] = 1.0
        for j in range(2):
            h['CD%d_%d' %(i+1, j+1)] = cd_rot[i,j]
    
    h['BACKGR'] = 0.17+0.49, 'Total, e/s SDT Report A-1'
    h['FILTER'] = 'GRS', 'WFIRST grism'
    h['INSTRUME'] = 'WFIRST'
    h['READN'] = 17, 'SDT report Table 3-3' # e/pix/per
    h['PHOTFLAM'] = 1.
    h['PHOTPLAM'] = 1.
    
    wcs = pywcs.WCS(h)
    h['EXTVER'] = 1
        
    return h, wcs
    
def make_fake_image(header, output='direct.fits', background=None, exptime=1.e4, nexp=10):
    """Use the header from NIRISS, WFC3/IR or WFIRST and make an 'FLT' image that `grizli` can read as a reference.
    
    Parameters
    ----------
    header: astropy.io.fits.Header
        Header created by one of the generating functions, such as 
        `niriss_header`.
    
    output: str
        Filename of the output FITS file. Will have extensions 'SCI', 'ERR',
        and 'DQ'. The 'ERR' extension is populated with a read-noise +
        background error model using
        
            >>> var = nexp*header['READN'] + background*exptime
        
        The 'SCI' extension is filled with gaussian deviates with standard
        deviation `sqrt(var)`.
        
        The 'DQ' extension is filled with (int) zeros.
        
    background: None or float
        Background value to use for sky noise.  If None, then read from
        `header['BACKGR']`.
    
    exptime: float
        Exposure time to use for background sky noise.
    
    nexp: int
        Number of exposures to use for read noise.
    
    Returns
    -------
    Nothing; outputs saved in `output` FITS file.
    """
    hdu = pyfits.HDUList()
    
    header['EXPTIME'] = exptime
    header['NEXP'] = nexp
    header['BUNIT'] = 'ELECTRONS/S'
    
    hdu.append(pyfits.PrimaryHDU(header=header))
    
    naxis = (header['NAXIS1'], header['NAXIS2'])
    
    for name, dtype in zip(['SCI', 'ERR', 'DQ'], 
                           [np.float32, np.float32, np.int32]):
        hdu.append(pyfits.ImageHDU(header=header, 
                                   data=np.zeros(np.array(naxis).T, 
                                   dtype=dtype), name=name))
                                   
    if background == None:
        background = header['BACKGR']
    
    header['BACKGR'] = background
    
    ### Simple error model of read noise and sky background
    var = nexp*header['READN'] + background*exptime
    
    ### electrons / s
    rms = np.sqrt(var)/exptime
    hdu['ERR'].data += rms
    hdu['SCI'].data = np.random.normal(size=np.array(naxis).T)*rms
    
    hdu.writeto(output, clobber=True, output_verify='fix')
