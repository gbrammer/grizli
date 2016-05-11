"""
Simulation tools for generating fake images
"""
import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
    
def niriss_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589, 
                  filter='F150W', grism='GR150R'):
    """
    NIRISS, 0.065"/pix, requires filter & grism specification
    """ 
    naxis = 2048, 2048
    crpix = 1024, 1024
    
    cd = np.array([[ -0.0658,  0], [0, 0.0654]])/3600.
    rad = np.deg2rad(-pa_aper)
    mat = np.zeros((2,2))
    mat[0,:] = np.array([np.cos(rad),-np.sin(rad)])
    mat[1,:] = np.array([np.sin(rad),np.cos(rad)])
    cd_rot = np.dot(mat, cd)
    
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
    
    if grism == 'GR150R':
        h['GRISM'] = 'GR150R', 'Spectral trace along X'
    else:
        h['GRISM'] = 'GR150C', 'Spectral trace along Y'
        
    wcs = pywcs.WCS(h)
    h['EXTVER'] = 1
        
    return h, wcs
    
def wfc3ir_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589, 
                  flt='ibhj34h6q_flt.fits', filter='G141'):
    """
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
    rad = np.deg2rad(theta)
    mat = np.zeros((2,2))
    mat[0,:] = np.array([np.cos(rad),-np.sin(rad)])
    mat[1,:] = np.array([np.sin(rad),np.cos(rad)])
    wcs.wcs.cd = np.dot(mat, wcs.wcs.cd)
    
    head = wcs.to_header(relax=True)
    
    for i in [1,2]:
        for j in [1,2]:
            head['CD%d_%d' %(i,j)] = head['PC%d_%d' %(i,j)]
            head.remove('PC%d_%d' %(i,j))
    
    head['BACKGR'] = 1.
    head['FILTER'] = filter
    head['INSTRUME'] = 'WFC3'
    head['READN'] = im[0].header['READNSEA']
    head['NAXIS1'] = head['NAXIS2'] = 1014
    head['DETECTOR'] = 'IR'
    
    return head, wcs

def wfirst_header(ra=53.1592277508136, dec=-27.782056346146, pa_aper=128.589, naxis=(4096,4096)):
    """
    WFIRST GRS Grism
    
    Current aXe config file has no field dependence, so field size can be anything you want
      
    """
    #naxis = 2048, 2048
    crpix = naxis[0]/2, naxis[0]/2
    
    cd = np.array([[ -0.11,  0], [0, 0.11]])/3600.
    rad = np.deg2rad(-pa_aper)
    mat = np.zeros((2,2))
    mat[0,:] = np.array([np.cos(rad),-np.sin(rad)])
    mat[1,:] = np.array([np.sin(rad),np.cos(rad)])
    cd_rot = np.dot(mat, cd)
    
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
    
    wcs = pywcs.WCS(h)
    h['EXTVER'] = 1
        
    return h, wcs
    
def make_fake_image(header, output='direct.fits', background='auto', exptime=1.e4, nexp=10):
    """
    Use the header from NIRISS, WFC3/IR or WFIRST and make an 'FLT' image that 
    `grizli` can read as a reference.
    """
    hdu = pyfits.HDUList()
    
    header['EXPTIME'] = exptime
    header['NEXP'] = nexp
    header['BUNIT'] = 'ELECTRONS/S'
    
    hdu.append(pyfits.PrimaryHDU(header=header))
    
    naxis = (header['NAXIS1'], header['NAXIS2'])
    
    for name, dtype in zip(['SCI', 'ERR', 'DQ'], [np.float32, np.float32, np.int32]):
        hdu.append(pyfits.ImageHDU(header=header, 
                                   data=np.zeros(np.array(naxis).T, 
                                   dtype=dtype), name=name))
                                   
    if background == 'auto':
        background = header['BACKGR']
    
    header['BACKGR'] = background
    
    ### Simple error model of read noise and sky background
    var = nexp*header['READN'] + background*exptime
    
    ### electrons / s
    rms = np.sqrt(var)/exptime
    hdu['ERR'].data += rms
    hdu['SCI'].data = np.random.normal(size=np.array(naxis).T)*rms
    
    hdu.writeto(output, clobber=True, output_verify='fix')
