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
    hwcs = gwcs.WCS(forward_transform=tform, output_frame=ICRS())#gwcs.CelestialFrame())
    sh = hdu.data.shape
    hwcs.bounding_box = ((-0.5, sh[0]-0.5), (-0.5, sh[1]-0.5))
    
    # Put gWCS in meta, where blot/drizzle expect to find it
    img.meta.wcs = hwcs
    
    return img
    
def xxx(header):
    """
    """
    ra, dec = 53.18118642, -27.79096316
    hdu = utils.make_wcsheader(ra=ra, dec=dec, size=10, pixscale=0.06, get_hdu=True)
    out = grizli.jwst.hdu_to_imagemodel(hdu)
    
    from jwst.datamodels import ModelContainer, DrizProductModel
    
    product = DrizProductModel(out.data.shape)
    product.meta.wcs = out.meta.wcs
    
    from jwst.resample import gwcs_blot, gwcs_drizzle
    driz = gwcs_drizzle.GWCSDrizzle(product)#, outwcs=out.meta.wcs)
    
    driz.add_image(blot_data, wcs_model.meta.wcs, xmax=out.data.shape[1], ymax=out.data.shape[0])
    
    from jwst.resample import resample_utils
    from drizzle import util
    
    input_wcs = wcs_model.meta.wcs
    output_wcs = out.meta.wcs
    
    fillval = 'INDEF'
    insci = blot_data
    inwht = None
    xmin = xmax = ymin = ymax = 0
    uniqid = 1
    outsci = driz.outsci*1
    outwht = driz.outwht*1
    outcon = driz.outcon*1
    in_units = 'cps'
    
    from jwst.resample import resample
    
    groups = ModelContainer([wcs_model])
    sampler = resample.ResampleData(groups, output=driz)

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

    # from jwst.stpipe import crds_client
    # from jwst.assign_wcs import assign_wcs
    
    # HDUList -> jwst.datamodels.ImageModel
    img = util.open(input)
    
    # AssignWcs to pupulate img.meta.wcsinfo
    step = AssignWcsStep()
    with_wcs = step.process(img)
    
    ## Above should be more robust to get all of the necessary ref files
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
            keys =  ['TELESCOP', 'FILTER', 'DETECTOR', 'INSTRUME']
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

def model_wcs_header(datamodel, get_sip=False, order=4, step=32):
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
            c_x = pipe.offset_2.value
            c_y = pipe.offset_3.value
        else:
            # Simple WCS
            c_x = pipe.offset_0.value
            c_y = pipe.offset_1.value
        
        crpix = np.array([-c_x+1, -c_y+1])
        
    except:
        crpix = np.array(sh)/2.+0.5
    
    crval = datamodel.meta.wcs.forward_transform(crpix[0], crpix[1])
    cdx = datamodel.meta.wcs.forward_transform(crpix[0]+1, crpix[1])
    cdy = datamodel.meta.wcs.forward_transform(crpix[0], crpix[1]+1)
    
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
    
    #### Fit a SIP header to the gwcs transformed coordinates
    v, u = np.meshgrid(np.arange(1,sh[0]+1,step), np.arange(1,sh[1]+1,step))
    x, y = datamodel.meta.wcs.forward_transform(u, v)
    y -= crval[1]
    x = (x-crval[0])*np.cos(crval[1]/180*np.pi)
        
    a_names = []
    b_names = []
    #order = 4
    for i in range(order+1):
        for j in range(order+1):
            ext = '{0}_{1}'.format(i,j)
            if (i+j) > order:
                continue
                
            if ext in ['0_0', '0_1','1_0']:
                continue
                
            a_names.append('A_'+ext)
            b_names.append('B_'+ext)
    
    p0 = np.zeros(4+len(a_names)+len(b_names))
    p0[:4] += cd.flatten()
    
    args = (u.flatten(), v.flatten(), x.flatten(), y.flatten(), crpix, a_names, b_names, cd, 0)
    
    # Fit the SIP coeffs
    fit = least_squares(_objective_sip, p0, jac='2-point', bounds=(-np.inf, np.inf), method='lm', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=1000, verbose=0, args=args, kwargs={})
    
    # Get the results 
    args = (u.flatten(), v.flatten(), x.flatten(), y.flatten(), crpix, a_names, b_names, cd, 1)
    
    cd_fit, a_coeff, b_coeff = _objective_sip(fit.x, *args)
    
    # Put in the header
    for i in range(2):
        for j in range(2):
            header['CD{0}_{1}'.format(i+1, j+1)] = cd_fit[i,j]
    
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    
    header['A_ORDER'] = order
    for k in a_coeff:
        header[k] = a_coeff[k]
        
    header['B_ORDER'] = order
    for k in b_coeff:
        header[k] = b_coeff[k]
    
    return header
    
def _objective_sip(params, u, v, x, y, crpix, a_names, b_names, cd, ret):
    """
    Objective function for fitting SIP coefficients
    """
    from astropy.modeling import models, fitting
    
    #u, v, x, y, crpix, a_names, b_names, cd = data
    
    cdx = params[0:4].reshape((2,2))
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
        
    sip = models.SIP(crpix=crpix, a_order=4, b_order=4, a_coeff=a_coeff, b_coeff=b_coeff)
    
    fuv, guv = sip(u,v)
    xo, yo = np.dot(cdx, np.array([u+fuv-crpix[0], v+guv-crpix[1]]))
    #dr = np.sqrt((x-xo)**2+(y-yo)**2)*3600.
    dr = np.append(x-xo, y-yo)*3600./0.065
    
    #print(params, np.abs(dr).max())
    
    return dr
    
    
    
    
    
        
        
    
    
    
    
    