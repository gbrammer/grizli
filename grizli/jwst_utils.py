"""
Utilities for handling JWST file/data formats.

Requires https://github.com/spacetelescope/jwst

"""
import os
import inspect

import numpy as np
from . import utils


def hdu_to_imagemodel(in_hdu):
    """
    Workaround for initializing a `jwst.datamodels.ImageModel` from a
    normal FITS ImageHDU that could contain HST header keywords and
    unexpected WCS definition.

    TBD

    Parameters
    ----------
    in_hdu : `astropy.io.fits.ImageHDU`


    Returns
    -------
    img : `jwst.datamodels.ImageModel`


    """
    from astropy.io.fits import ImageHDU, HDUList
    from astropy.coordinates import ICRS

    from jwst.datamodels import util
    import gwcs

    hdu = ImageHDU(data=in_hdu.data, header=in_hdu.header)

    new_header = strip_telescope_header(hdu.header)

    hdu.header = new_header

    # Initialize data model
    img = util.open(HDUList([hdu]))

    # Initialize GWCS
    tform = gwcs.wcs.utils.make_fitswcs_transform(new_header)
    hwcs = gwcs.WCS(forward_transform=tform, output_frame=ICRS())  # gwcs.CelestialFrame())
    sh = hdu.data.shape
    hwcs.bounding_box = ((-0.5, sh[0]-0.5), (-0.5, sh[1]-0.5))

    # Put gWCS in meta, where blot/drizzle expect to find it
    img.meta.wcs = hwcs

    return img


def change_header_pointing(header, ra_ref=0., dec_ref=0., pa_v3=0.):
    """
    Update a FITS header for a new pointing (center + roll).

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Parent header (must contain `V2_REF`, `V3_REF` keywords).

    ra_ref, dec_ref : float
        Pointing center, in decimal degrees, at reference the pixel defined
        in.

    pa_v3 : float
        Position angle of the telescope V3 axis, degrees.

    .. warning::

    Doesn't update PC keywords based on pa_v3, which would rather have to
    be computed from the new `gwcs`.

    """
    from jwst.lib.set_telescope_pointing import compute_local_roll

    v2_ref = header['V2_REF']
    v3_ref = header['V3_REF']

    # Strip units, if any
    args = []
    for v in (pa_v3, ra_ref, dec_ref, v2_ref, v3_ref):
        if hasattr(v, 'value'):
            args.append(v.value)
        else:
            args.append(v)

    roll_ref = compute_local_roll(*tuple(args))

    new_header = header.copy()
    new_header['XPA_V3'] = args[0]
    new_header['CRVAL1'] = new_header['RA_REF'] = args[1]
    new_header['CRVAL2'] = new_header['DEC_REF'] = args[2]
    new_header['ROLL_REF'] = roll_ref
    return new_header


def img_with_flat(input, verbose=True, overwrite=True):
    """
    Apply flat-field correction if nessary
    """
    import astropy.io.fits as pyfits
    from jwst.datamodels import util
    from jwst.flatfield import FlatFieldStep
    from jwst.gain_scale import GainScaleStep
    
    if not isinstance(input, pyfits.HDUList):
        _hdu = pyfits.open(input)
    else:
        _hdu = input
        
    img = util.open(_hdu)
    
    skip = False
    if 'S_FLAT' in _hdu[0].header:
        if _hdu[0].header['S_FLAT'] == 'COMPLETE':
            skip = True
    
    if not skip:        
        if verbose:
            utils.log_comment(utils.LOGFILE,
                              'jwst.flatfield.FlatFieldStep', 
                              verbose=verbose, show_date=False)
                              
        flat_step = FlatFieldStep()
        with_flat = flat_step.process(img)
        output = with_flat
    else:
        output = img
    
    if isinstance(input, str) & overwrite:
        output.write(input, overwrite=overwrite)
        
    return output


def img_with_wcs(input, overwrite=True):
    """
    Open a JWST exposure and apply the distortion model.

    Parameters
    ----------
    input : type
        Anything `jwst.datamodels.util.open` can accept for initialization.

    Returns
    -------
    with_wcs : `jwst.datamodels.ImageModel`
        Image model with full `~gwcs` in `with_wcs.meta.wcs`.

    """    
    import astropy.io.fits as pyfits
    
    from jwst.datamodels import util
    from jwst.assign_wcs import AssignWcsStep
    
    # from jwst.stpipe import crds_client
    # from jwst.assign_wcs import assign_wcs

    # HDUList -> jwst.datamodels.ImageModel

    # Generate WCS as image
    if not isinstance(input, pyfits.HDUList):
        _hdu = pyfits.open(input)
    
    if _hdu[0].header['OINSTRUM'] == 'NIRISS':
        if _hdu[0].header['OFILTER'].startswith('GR'):
            _hdu[0].header['FILTER'] = 'CLEAR'
            _hdu[0].header['EXP_TYPE'] = 'NIS_IMAGE'
    
    elif _hdu[0].header['OINSTRUM'] == 'NIRCAM':
        if _hdu[0].header['OPUPIL'].startswith('GR'):
            _hdu[0].header['PUPIL'] = 'CLEAR'
            _hdu[0].header['EXP_TYPE'] = 'NRC_IMAGE'
        
    img = util.open(_hdu)

    # AssignWcs to pupulate img.meta.wcsinfo
    step = AssignWcsStep()
    with_wcs = step.process(img)
    output = with_wcs
    
    # Write to a file
    if isinstance(input, str) & overwrite:
        output.write(input, overwrite=overwrite)
            
    return output

ORIG_KEYS = ['TELESCOP','INSTRUME','DETECTOR','FILTER','PUPIL','EXP_TYPE']

def initialize_jwst_image(filename, verbose=True, max_dq_bit=14, orig_keys=ORIG_KEYS):
    """
    Make copies of some header keywords to make the headers look like 
    and HST instrument
    
    1) Apply gain correction
    2) Clip DQ bits
    3) Copy header keywords
    4) Apply flat field if necessary
    5) Initalize WCS
    
    """
    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'jwst_utils.initialize_jwst_image')
    
    import astropy.io.fits as pyfits
    from jwst.flatfield import FlatFieldStep
    from jwst.gain_scale import GainScaleStep
    
    img = pyfits.open(filename, mode='update')
    
    if 'OTELESCO' in img[0].header:
        tel = img[0].header['OTELESCO']
    elif 'TELESCOP' in img[0].header:
        tel = img[0].header['TELESCOP']
    else:
        tel = None
        
    if tel not in ['JWST']:
        msg = f'TELESCOP keyword ({tel}) not "JWST"'
        #utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        raise ValueError(msg)
    
    if img['SCI'].header['BUNIT'].upper() == 'DN/S':
        gain_file = GainScaleStep().get_reference_file(img, 'gain')
        
        with pyfits.open(gain_file) as gain_im:
            gain_median = np.median(gain_im[1].data)
        
        img[0].header['GAINFILE'] = gain_file
        img[0].header['GAINCORR'] = True, 'Manual gain correction applied'
        img[0].header['GAINVAL'] = gain_median, 'Gain value applied'
        
        msg = f'GAINVAL = {gain_median:.2f}\n'
        msg += f'GAINFILE = {gain_file}'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        
        img['SCI'].data *= gain_median 
        img['SCI'].header['BUNIT'] = 'ELECTRONS/S'
        img['ERR'].data *= gain_median 
        img['ERR'].header['BUNIT'] = 'ELECTRONS/S'
        
        for k in ['VAR_POISSON','VAR_RNOISE','VAR_FLAT']:
            if k in img:
                img[k].data *= 1./gain_median**2
                
    for k in orig_keys:
        newk = 'O'+k[:7]
        if newk not in img[0].header:
            img[0].header[newk] = img[0].header[k]
            msg = f'{newk} = {k} {img[0].header[k]}'
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
                
    # Get flat field ref file
    _flatfile = FlatFieldStep().get_reference_file(img, 'flat')    
    img[0].header['PFLTFILE'] = os.path.basename(_flatfile)
    msg = f'PFLTFILE = {_flatfile}'
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    
    # Clip DQ keywords
    img[0].header['MAXDQBIT'] = max_dq_bit, 'Max DQ bit allowed'
    msg = f'Clip MAXDQBIT = {max_dq_bit}'
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    
    dq = np.zeros_like(img['DQ'].data)
    dq[img['DQ'].data >= 2**(max_dq_bit+1)] = 2**max_dq_bit
    dqm = img['DQ'].data > 0    
    
    for bit in range(max_dq_bit+1):
        dq[dqm] |= img['DQ'].data[dqm] & 2**bit
    
    dq[img['DQ'].data < 0] = 2**bit    
    img['DQ'].data = dq
    
    img[0].header['EXPTIME'] = img[0].header['EFFEXPTM']
    
    img[1].header['NGOODPIX'] = (dq == 0).sum()
    img[1].header['EXPNAME'] = img[0].header['EXPOSURE']
    img[1].header['MEANDARK'] = 0.0
    img[1].header['IDCSCALE'] = 0.065 # Set from WCS
    
    if 'TIME' not in img:
        time = pyfits.ImageHDU()
        time.header['EXTNAME'] = 'TIME'
        time.header['EXTVER'] = 1
        time.header['PIXVALUE'] = img[0].header['EXPTIME']
        time.header['BUNIT'] = 'SECONDS'
        time.header['NPIX1'] = 2048
        time.header['NPIX2'] = 2048
        time.header['INHERIT'] = True
        
        img.append(time)
        
    img.flush()
    
    _ = img_with_flat(filename, overwrite=True)
    
    _ = img_with_wcs(filename, overwrite=True)
    
    img = pyfits.open(filename)
    
    return img


# for NIRISS images; NIRCam,MIRI TBD
# band: [photflam, photfnu, pivot_wave]
NIS_PHOT_KEYS = {'F090W': [1.098934e-20, 2.985416e-31, 0.9025],
                 'F115W': [6.291060e-21, 2.773018e-31, 1.1495],
                 'F140M': [9.856255e-21, 6.481079e-31, 1.4040],
                 'F150W': [4.198384e-21, 3.123540e-31, 1.4935],
                 'F158M': [7.273483e-21, 6.072128e-31, 1.5820],
                 'F200W': [2.173398e-21, 2.879494e-31, 1.9930],
                 'F277W': [1.109150e-21, 2.827052e-31, 2.7643],
                 'F356W': [6.200034e-22, 2.669862e-31, 3.5930],
                 'F380M': [2.654520e-21, 1.295626e-30, 3.8252],
                 'F430M': [2.636528e-21, 1.613895e-30, 4.2838],
                 'F444W': [4.510426e-22, 2.949531e-31, 4.4277],
                 'F480M': [1.879639e-21, 1.453752e-30, 4.8152]}


def set_jwst_to_hst_keywords(input, reset=False, verbose=True, orig_keys=ORIG_KEYS):
    """
    Make primary header look like an HST instrument
    """
    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'jwst_utils.set_jwst_to_hst_keywords')
                                 
    import astropy.io.fits as pyfits
    
    if isinstance(input, str):
        img = pyfits.open(input, mode='update')
    else:
        img = input
    
    HST_KEYS = {'TELESCOP':'HST',
           'INSTRUME':'WFC3',
           'DETECTOR':'IR'}
    
    if 'OTELESCO' not in img[0].header:
        img = initialize_jwst_image(input, verbose=verbose)
        
    if reset:
        for k in orig_keys:
            newk = 'O'+k[:7]
            img[0].header[k] = img[0].header[newk]
            msg = f'Reset: {k} > {img[0].header[newk]} ({newk})'
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)            
    else:
        for k in HST_KEYS:
            img[0].header[k] = HST_KEYS[k]
            msg = f'  Set: {k} > {HST_KEYS[k]}'
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    
    if img[0].header['OINSTRUM'] == 'NIRISS':
        _filter = img[0].header['OPUPIL']
        if _filter in NIS_PHOT_KEYS:
            img[0].header['PHOTFLAM'] = NIS_PHOT_KEYS[_filter][0]
            img[0].header['PHOTFNU'] = NIS_PHOT_KEYS[_filter][1]
            img[0].header['PHOTPLAM'] = NIS_PHOT_KEYS[_filter][2] * 1.e4
    else:
        img[0].header['PHOTFLAM'] = 1.e-21
        img[0].header['PHOTFNU'] = 1.e-31
        img[0].header['PHOTPLAM'] = 1.5e4  
        
    if isinstance(input, str):
        img.flush()
    
    return img


def strip_telescope_header(header, simplify_wcs=True):
    """
    Strip non-JWST keywords that confuse `jwst.datamodels.util.open`.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Input FITS header.

    """
    import astropy.wcs as pywcs

    new_header = header.copy()

    if 'TELESCOP' in new_header:
        if new_header['TELESCOP'] != 'JWST':
            keys = ['TELESCOP', 'FILTER', 'DETECTOR', 'INSTRUME']
            for key in keys:
                if key in header:
                    new_header.remove(key)

    if simplify_wcs:
        # Make simple WCS header
        orig_wcs = pywcs.WCS(new_header)
        new_header = orig_wcs.to_header()

        new_header['EXTNAME'] = 'SCI'
        new_header['RADESYS'] = 'ICRS'
        new_header['CDELT1'] = -new_header['PC1_1']
        new_header['CDELT2'] = new_header['PC2_2']
        new_header['PC1_1'] = -1
        new_header['PC2_2'] = 1

    return new_header

LSQ_ARGS = dict(jac='2-point', bounds=(-np.inf, np.inf), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='soft_l1', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=1000, verbose=0, kwargs={})

def model_wcs_header(datamodel, get_sip=False, order=4, step=32, lsq_args=LSQ_ARGS):
    """
    Make a header with approximate WCS for use in DS9.

    Parameters
    ----------
    datamodel : `jwst.datamodels.ImageModel`
        Image model with full `~gwcs` in `with_wcs.meta.wcs`.

    get_sip : bool
        If True, fit a `astropy.modeling.models.SIP` distortion model to the
        image WCS.

    order : int
        Order of the SIP polynomial model.

    step : int
        For fitting the SIP model, generate a grid of detector pixels every
        `step` pixels in both axes for passing through
        `datamodel.meta.wcs.forward_transform`.

    Returns
    -------
    header : '~astropy.io.fits.Header`
        Header with simple WCS definition: CD rotation but no distortion.

    """
    from astropy.io.fits import Header
    from scipy.optimize import least_squares
    import jwst.datamodels

    datamodel = jwst.datamodels.open(datamodel)
    sh = datamodel.data.shape

    try:
        pipe = datamodel.meta.wcs.pipeline[0][1]
        if 'offset_2' in pipe.param_names:
            # NIRISS WCS
            c_x = pipe.offset_2.value + pipe.offset_0.value
            c_y = pipe.offset_3.value + pipe.offset_1.value

        else:
            # Simple WCS
            c_x = pipe.offset_0.value
            c_y = pipe.offset_1.value

        crpix = np.array([-c_x+1, -c_y+1])
        #print('xxx ', crpix)
        
    except:
        crpix = np.array(sh)/2.+0.5
    
    crp0 = crpix-1
    
    crval = datamodel.meta.wcs.forward_transform(crp0[0], crp0[1])
    cdx = datamodel.meta.wcs.forward_transform(crp0[0]+1, crp0[1])
    cdy = datamodel.meta.wcs.forward_transform(crp0[0], crp0[1]+1)

    # use utils.to_header in grizli to replace the below (from datamodel.wcs)
    header = Header()
    header['RADESYS'] = 'ICRS'
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'

    header['CUNIT1'] = header['CUNIT2'] = 'deg'

    header['CRPIX1'] = crpix[0]
    header['CRPIX2'] = crpix[1]

    header['CRVAL1'] = crval[0]
    header['CRVAL2'] = crval[1]

    cosd = np.cos(crval[1]/180*np.pi)

    header['CD1_1'] = (cdx[0]-crval[0])*cosd
    header['CD1_2'] = (cdy[0]-crval[0])*cosd

    header['CD2_1'] = cdx[1]-crval[1]
    header['CD2_2'] = cdy[1]-crval[1]

    cd = np.array([[header['CD1_1'], header['CD1_2']], [header['CD2_1'], header['CD2_2']]])
    if not get_sip:
        return header

    # Fit a SIP header to the gwcs transformed coordinates
    u, v = np.meshgrid(np.arange(1, sh[1]-1, step), 
                       np.arange(1, sh[0]-1, step))
    x, y = datamodel.meta.wcs.forward_transform(u, v)
    
    a_names = []
    b_names = []
    #order = 4
    for i in range(order+1):
        for j in range(order+1):
            ext = '{0}_{1}'.format(i, j)
            if (i+j) > order:
                continue

            if ext in ['0_0', '0_1', '1_0']:
                continue

            a_names.append('A_'+ext)
            b_names.append('B_'+ext)

    p0 = np.zeros(4+len(a_names)+len(b_names))
    p0[:4] += cd.flatten()
    
    if datamodel.meta.instrument.name == 'NIRISS':
        a0 = {'A_0_2': 3.8521180058449584e-08,
         'A_0_3': -1.2910469982047994e-11,
         'A_0_4': 3.642187826984494e-15,
         'A_1_1': -8.156851592950884e-08,
         'A_1_2': -1.2336474525621777e-10,
         'A_1_3': 1.1169942988845159e-13,
         'A_2_0': 3.5236920263776116e-07,
         'A_2_1': -9.622992486408194e-11,
         'A_2_2': -2.1150777639693208e-14,
         'A_3_0': -3.517117816321703e-11,
         'A_3_1': 1.252016786545716e-13,
         'A_4_0': -2.5596007366022595e-14}
        b0 = {'B_0_2': -6.478494215243917e-08,
         'B_0_3': -4.2460992201562465e-10,
         'B_0_4': 2.501714355762585e-13,
         'B_1_1': 4.127407304584838e-07,
         'B_1_2': -2.774351986369079e-11,
         'B_1_3': 3.4947161649623674e-15,
         'B_2_0': -7.509503977158588e-07,
         'B_2_1': -2.1263593068617203e-10,
         'B_2_2': 1.3621493497144034e-13,
         'B_3_0': -2.099145095489808e-11,
         'B_3_1': -1.613481283521298e-14,
         'B_4_0': 2.38606562938391e-14}
        
        
        for i, k in enumerate(a_names):
            if k in a0:
                p0[4+i] = a0[k]
        
        for i, k in enumerate(b_names):
            if k in b0:
                p0[4+len(b_names)+i] = b0[k]
                
    #args = (u.flatten(), v.flatten(), x.flatten(), y.flatten(), crpix, a_names, b_names, cd, 0)
    args = (u.flatten(), v.flatten(), x.flatten(), y.flatten(), crval, crpix, 
            a_names, b_names, cd, 0)

    # Fit the SIP coeffs
    fit = least_squares(_objective_sip, p0, args=args, **lsq_args)

    # Get the results
    args = (u.flatten(), v.flatten(), x.flatten(), y.flatten(), crval, crpix, 
            a_names, b_names, cd, 1)

    cd_fit, a_coeff, b_coeff = _objective_sip(fit.x, *args)

    # Put in the header
    for i in range(2):
        for j in range(2):
            header['CD{0}_{1}'.format(i+1, j+1)] = cd_fit[i, j]

    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'

    header['A_ORDER'] = order
    for k in a_coeff:
        header[k] = a_coeff[k]

    header['B_ORDER'] = order
    for k in b_coeff:
        header[k] = b_coeff[k]

    return header

def _objective_sip(params, u, v, ra, dec, crval, crpix, a_names, b_names, cd, ret):
    """
    Objective function for fitting SIP coefficients
    """
    from astropy.modeling import models, fitting
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    
    #u, v, x, y, crpix, a_names, b_names, cd = data

    cdx = params[0:4].reshape((2, 2))
    a_params = params[4:4+len(a_names)]
    b_params = params[4+len(a_names):]

    a_coeff = {}
    for i in range(len(a_names)):
        a_coeff[a_names[i]] = a_params[i]

    b_coeff = {}
    for i in range(len(b_names)):
        b_coeff[b_names[i]] = b_params[i]
    
    if ret == 1:
        return cdx, a_coeff, b_coeff
    
    # Build header
    _h = pyfits.Header()
    for i in [0,1]:
        for j in [0,1]:
            _h[f'CD{i+1}_{j+1}'] = cdx[i,j]
    
    _h['CRPIX1'] = crpix[0]
    _h['CRPIX2'] = crpix[1]
    _h['CRVAL1'] = crval[0]
    _h['CRVAL2'] = crval[1]
    
    _h['A_ORDER'] = 4
    for k in a_coeff:
        _h[k] = a_coeff[k]
    
    _h['B_ORDER'] = 4
    for k in b_coeff:
        _h[k] = b_coeff[k]
    
    _h['RADESYS'] = 'ICRS    '                                                            
    _h['CTYPE1']  = 'RA---TAN-SIP'                                                        
    _h['CTYPE2']  = 'DEC--TAN-SIP'                                                        
    _h['CUNIT1']  = 'deg     '                                                            
    _h['CUNIT2']  = 'deg     '                                                            
    
    _w = pywcs.WCS(_h)
    ro, do = _w.all_pix2world(u, v, 0)
    
    cosd = np.cos(ro/180*np.pi)
    dr = np.append((ra-ro)*cosd, dec-do)*3600./0.065

    #print(params, np.abs(dr).max())

    return dr
    
def _xobjective_sip(params, u, v, x, y, crval, crpix, a_names, b_names, cd, ret):
    """
    Objective function for fitting SIP coefficients
    """
    from astropy.modeling import models, fitting

    #u, v, x, y, crpix, a_names, b_names, cd = data

    cdx = params[0:4].reshape((2, 2))
    a_params = params[4:4+len(a_names)]
    b_params = params[4+len(a_names):]

    a_coeff = {}
    for i in range(len(a_names)):
        a_coeff[a_names[i]] = a_params[i]

    b_coeff = {}
    for i in range(len(b_names)):
        b_coeff[b_names[i]] = b_params[i]

    if ret == 1:
        return cdx, a_coeff, b_coeff
    
    off = 1
    
    sip = models.SIP(crpix=crpix-off, a_order=4, b_order=4, a_coeff=a_coeff, b_coeff=b_coeff)

    fuv, guv = sip(u, v)
    xo, yo = np.dot(cdx, np.array([u+fuv-crpix[0], v+guv-crpix[1]]))
    #dr = np.sqrt((x-xo)**2+(y-yo)**2)*3600.
    dr = np.append(x-xo, y-yo)*3600./0.065

    #print(params, np.abs(dr).max())

    return dr
