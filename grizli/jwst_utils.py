"""
Utilities for handling JWST file/data formats.

Requires https://github.com/spacetelescope/jwst

"""

import numpy as np


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


def img_with_wcs(input):
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
    from jwst.datamodels import util
    from jwst.assign_wcs import AssignWcsStep
    import astropy.io.fits as pyfits
    
    # from jwst.stpipe import crds_client
    # from jwst.assign_wcs import assign_wcs

    # HDUList -> jwst.datamodels.ImageModel
    
    # Generate WCS as image
    if isinstance(input, pyfits.HDUList):
        if input[0].header['INSTRUME'] == 'NIRISS':
            if input[0].header['FILTER'].startswith('GR'):
                input[0].header['FILTER'] = 'CLEAR'
                input[0].header['EXP_TYPE'] = 'NIS_IMAGE'
                #print(input[0].header)
        
        elif input[0].header['INSTRUME'] == 'NIRCAM':
            if input[0].header['PUPIL'].startswith('GR'):
                input[0].header['PUPIL'] = 'CLEAR'
                input[0].header['EXP_TYPE'] = 'NRC_IMAGE'
                #print(input[0].header)
        
        
    img = util.open(input)

    # AssignWcs to pupulate img.meta.wcsinfo
    step = AssignWcsStep()
    with_wcs = step.process(img)

    # Above should be more robust to get all of the necessary ref files
    #dist_file = crds_client.get_reference_file(img, 'distortion')
    #reference_files = {'distortion': dist_file}
    #with_wcs = assign_wcs.load_wcs(img, reference_files=reference_files)

    return with_wcs


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
