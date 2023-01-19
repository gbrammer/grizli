"""
Align direct images & make mosaics
"""
import os
import inspect
import gc

from collections import OrderedDict
import glob
import traceback

import yaml
import numpy as np
import matplotlib.pyplot as plt

# conda install shapely
# from shapely.geometry.polygon import Polygon

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table

from . import jwst_utils
from . import grismconf

try:
    import jwst
    from jwst.pipeline import Detector1Pipeline
except ImportError:
    jwst = None
    print('`import jwst` failed so JWST processing will not work!')
    
from . import utils
from . import model
from . import GRIZLI_PATH

# Catalog table tools now put elsewhere
from .catalog import *

def check_status():
    """Make sure all files and modules are in place and print some information if they're not
    """
    for ref_dir in ['iref']:
        if not os.getenv(ref_dir):
            print("""
No ${0} set!  Make a directory and point to it in ~/.bashrc or ~/.cshrc.
For example,

  $ mkdir $GRIZLI/{0}
  $ export {0}="$GRIZLI/{0}/" # put this in ~/.bashrc
""".format(ref_dir))
        else:
            # WFC3
            if not os.getenv('iref').endswith('/'):
                print("Warning: $iref should end with a '/' character [{0}]".format(os.getenv('iref')))

            test_file = 'iref$uc72113oi_pfl.fits'.replace('iref$', os.getenv('iref'))
            if not os.path.exists(test_file):
                print("""
        HST calibrations not found in $iref [{0}]

        To fetch them, run

           >>> import grizli.utils
           >>> grizli.utils.fetch_default_calibs()

        """.format(os.getenv('iref')))


# check_status()


def fresh_flt_file(file, preserve_dq=False, path='../RAW/', verbose=True, extra_badpix=True, apply_grism_skysub=True, crclean=False, mask_regions=True, oneoverf_correction=True, oneoverf_kwargs={}, use_skyflats=True):
    """Copy "fresh" unmodified version of a data file from some central location

    Parameters
    ----------
    file : str
        Filename
        
    preserve_dq : bool
        Preserve DQ arrays of files if they exist in './'

    path : str
        Path where to find the "fresh" files

    verbose : bool
        Print information about what's being done

    extra_badpix : bool
        Apply extra bad pixel mask.  Currently this is hard-coded to look for
        a file ``badpix_spars200_Nov9.fits`` in the directory specified by
        the ``$iref`` environment variable.  The file can be downloaded from

        https://github.com/gbrammer/wfc3/tree/master/data

    apply_grism_skysub : bool
        xx nothing now xxx

    crclean : bool
        Run LACosmicx (astroscrappy) on the exposure

    mask_regions : bool
        Apply exposure region mask (like ``_flt.01.mask.reg``) if it exists.

    Returns
    -------
    Nothing, but copies the file from ``path`` to ``./``.

    """
    import shutil
    try:
        from astroscrappy import detect_cosmics
        has_scrappy = True
    except ImportError:
        has_scrappy = False

    local_file = os.path.basename(file)
    if preserve_dq:
        if os.path.exists(local_file):
            im = pyfits.open(local_file)
            orig_dq = im['DQ'].data
        else:
            orig_dq = None
    else:
        dq = None

    if file == local_file:
        orig_file = pyfits.open(glob.glob(os.path.join(path, file)+'*')[0])
    else:
        orig_file = pyfits.open(file)
    
    local_file = local_file.split('.gz')[0]
    
    if dq is not None:
        orig_file['DQ'] = dq

    head = orig_file[0].header

    # Divide grism images by imaging flats
    # G102 -> F105W, uc72113oi_pfl.fits
    # G141 -> F140W, uc72113oi_pfl.fits
    flat, extra_msg = 1., ''
    filter = utils.parse_filter_from_header(head)

    # Copy calibs for ACS/UVIS files
    if '_flc' in file:
        ftpdir = 'https://hst-crds.stsci.edu/unchecked_get/references/hst/'
        calib_types = ['IDCTAB', 'NPOLFILE', 'D2IMFILE']
        if filter == 'G800L':
            calib_types.append('PFLTFILE')

        utils.fetch_hst_calibs(orig_file.filename(), ftpdir=ftpdir,
                               calib_types=calib_types,
                               verbose=False)
    
    if filter in ['G102', 'G141']:
        flat_files = {'G102': 'uc72113oi_pfl.fits',
                      'G141': 'uc721143i_pfl.fits'}

        flat_file = flat_files[filter]
        extra_msg = ' / flat: {0}'.format(flat_file)

        flat_im = pyfits.open(os.path.join(os.getenv('iref'), flat_file))
        flat = flat_im['SCI'].data[5:-5, 5:-5]
        flat_dq = (flat < 0.2)

        # Grism FLT from IR amplifier gain
        pfl_file = orig_file[0].header['PFLTFILE'].replace('iref$',
                                                           os.getenv('iref'))
        grism_pfl = pyfits.open(pfl_file)[1].data[5:-5, 5:-5]

        orig_file['DQ'].data |= 4*flat_dq
        orig_file['SCI'].data *= grism_pfl/flat

        # if apply_grism_skysub:
        #     if 'GSKY001' in orig_file:

    if filter == 'G280':
        # Use F200LP flat
        flat_files = {'G280': 'zcv2053ei_pfl.fits'}  # F200LP
        flat_file = flat_files[filter]
        extra_msg = ' / flat: {0}'.format(flat_file)

        flat_im = pyfits.open(os.path.join(os.getenv('jref'), flat_file))

        for ext in [1, 2]:
            flat = flat_im['SCI', ext].data
            flat_dq = (flat < 0.2)

            orig_file['DQ', ext].data |= 4*flat_dq
            orig_file['SCI', ext].data *= 1./flat

    if filter == 'G800L':
        flat_files = {'G800L': 'n6u12592j_pfl.fits'}  # F814W
        flat_file = flat_files[filter]
        extra_msg = ' / flat: {0}'.format(flat_file)

        flat_im = pyfits.open(os.path.join(os.getenv('jref'), flat_file))
        pfl_file = orig_file[0].header['PFLTFILE'].replace('jref$',
                                                    os.getenv('jref'))
        pfl_im = pyfits.open(pfl_file)
        for ext in [1, 2]:
            flat = flat_im['SCI', ext].data
            flat_dq = (flat < 0.2)

            grism_pfl = pfl_im['SCI', ext].data

            orig_file['DQ', ext].data |= 4*flat_dq
            orig_file['SCI', ext].data *= grism_pfl/flat

        if orig_file[0].header['NPOLFILE'] == 'N/A':
            # Use an F814W file, but this should be updated
            orig_file[0].header['NPOLFILE'] = 'jref$v971826jj_npl.fits'

    if head['INSTRUME'] == 'WFPC2':
        head['DETECTOR'] = 'WFPC2'

    if ((head['INSTRUME'] == 'WFC3') & (head['DETECTOR'] == 'IR')
         & extra_badpix):
        bp = pyfits.open(os.path.join(os.getenv('iref'),
                                      'badpix_spars200_Nov9.fits'))

        if orig_file['DQ'].data.shape == bp[0].data.shape:
            orig_file['DQ'].data |= bp[0].data

        extra_msg += ' / bpix: $iref/badpix_spars200_Nov9.fits'

        # New flags for bad pix in old dark reference files
        old_darks = ['x5g1509ki_drk.fits']
        old_darks += ['xag1929{x}i_drk.fits'.format(x=x) for x in '345689a']

        # For more recent SPARS5
        old_darks += ['zb21929si_drk.fits']

        #need_badpix = head['DARKFILE'].strip('iref$') in old_darks
        need_badpix = True  # always add the additional bad pix files

        if need_badpix:
            new_bp = pyfits.open(os.path.join(os.path.dirname(__file__),
                                    'data',
                                    'wfc3ir_dark_badpix_2019.01.12.fits.gz'))

            if orig_file['DQ'].data.shape == new_bp[0].data.shape:
                orig_file['DQ'].data |= new_bp[0].data
                extra_msg += ' / wfc3ir_dark_badpix_2019.01.12.fits'
    
    logstr = '# {0} -> {1} {2}'
    logstr = logstr.format(orig_file.filename(), local_file, extra_msg)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)
    
    ####
    # JWST
    if 'TELESCOP' in orig_file[0].header:
        if orig_file[0].header['TELESCOP'] == 'JWST':
            orig_file.writeto(local_file, overwrite=True)
            status = jwst_utils.initialize_jwst_image(local_file, 
                                      oneoverf_correction=oneoverf_correction,
                                      oneoverf_kwargs=oneoverf_kwargs, 
                                      use_skyflats=use_skyflats)            
            orig_file = pyfits.open(local_file)
            
    # if filter in ['GR150C', 'GR150R']: 
    #     if (head['FILTER'] == 'GR150C') & (head['PUPIL'] == 'F115W'):        
    #         flat_file = os.path.join(os.getenv('jref'), 'jwst_niriss_flat_0202.fits')
    #     if (head['FILTER'] == 'GR150C') & (head['PUPIL'] == 'F150W'):        
    #         flat_file = os.path.join(os.getenv('jref'), 'jwst_niriss_flat_0187.fits')
    #     if (head['FILTER'] == 'GR150C') & (head['PUPIL'] == 'F200W'):        
    #         flat_file = os.path.join(os.getenv('jref'), 'jwst_niriss_flat_0204.fits')
    #     if (head['FILTER'] == 'GR150R') & (head['PUPIL'] == 'F115W'):        
    #         flat_file = os.path.join(os.getenv('jref'), 'jwst_niriss_flat_0200.fits')
    #     if (head['FILTER'] == 'GR150R') & (head['PUPIL'] == 'F150W'):        
    #         flat_file = os.path.join(os.getenv('jref'), 'jwst_niriss_flat_0188.fits')
    #     if (head['FILTER'] == 'GR150R') & (head['PUPIL'] == 'F200W'):        
    #         flat_file = os.path.join(os.getenv('jref'), 'jwst_niriss_flat_0195.fits')
    # 
    #     extra_msg = ' / flat: {0}'.format(flat_file)
    # 
    #     flat_im = pyfits.open(os.path.join(os.getenv('jref'), flat_file))
    #     flat = flat_im['SCI'].data #[5:-5, 5:-5]:
    #     flat_dq = (flat < 0.2)
    # 
    #     #orig_file['DQ'].data |= 4*flat_dq
    #     orig_file['SCI'].data = np.divide(orig_file['SCI'].data,flat,orig_file['SCI'].data,where=(flat!=0))
    
    if crclean and has_scrappy:
        for ext in [1, 2]:
            print('Clean CRs with LACosmic, extension {0:d}'.format(ext))

            sci = orig_file['SCI', ext].data
            dq = orig_file['DQ', ext].data

            crmask, clean = detect_cosmics(sci, inmask=None,
                         sigclip=4.5, sigfrac=0.3, objlim=5.0, gain=1.0,
                         readnoise=6.5, satlevel=65536.0, pssl=0.0, niter=4,
                         sepmed=True, cleantype='meanmask', fsmode='median',
                         psfmodel='gauss', psffwhm=2.5, psfsize=7, psfk=None,
                         psfbeta=4.765, verbose=False)

            dq[crmask] |= 1024
            #sci[crmask] = 0

    # WFPC2
    if '_c0' in file:
        # point to FITS reference files
        for key in ['MASKFILE', 'ATODFILE', 'BLEVFILE', 'BLEVDFIL', 'BIASFILE', 'BIASDFIL', 'DARKFILE', 'DARKDFIL', 'FLATFILE', 'FLATDFIL', 'SHADFILE']:
            ref_file = '_'.join(head[key].split('.'))+'.fits'
            orig_file[0].header[key] = ref_file.replace('h.fits', 'f.fits')

        waiv = orig_file[0].header['FLATFILE']
        orig_file[0].header['FLATFILE'] = waiv.replace('.fits', '_c0h.fits')

        if not os.path.exists(''):
            pass

        #
        # ## testing
        # orig_file[0].header['FLATFILE'] = 'm341820ju_pfl.fits'

        # Make sure has correct header keys
        for ext in range(4):
            if 'BUNIT' not in orig_file[ext+1].header:
                orig_file[ext+1].header['BUNIT'] = 'COUNTS'

        # Copy WFPC2 DQ file (c1m)
        dqfile = os.path.join(path, file.replace('_c0', '_c1'))
        print('Copy WFPC2 DQ file: {0}'.format(dqfile))
        if os.path.exists(os.path.basename(dqfile)):
            os.remove(os.path.basename(dqfile))

        shutil.copy(dqfile, './')

        # Add additional masking since AstroDrizzle having trouble with flats
        flat_file = orig_file[0].header['FLATFILE'].replace('uref$', os.getenv('uref')+'/')
        pfl = pyfits.open(flat_file)
        c1m = pyfits.open(os.path.basename(dqfile), mode='update')
        for ext in [1, 2, 3, 4]:
            mask = pfl[ext].data > 1.3
            c1m[ext].data[mask] |= 2

        c1m.flush()

    orig_file.writeto(local_file, overwrite=True)

    if mask_regions:
        apply_region_mask(local_file, dq_value=1024)
    
    # Flush objects
    orig_file.close()
    del(orig_file)
    
    for _iter in range(3):
        gc.collect()


def apply_persistence_mask(flt_file, path='../Persistence', dq_value=1024,
                           err_threshold=0.6, sci_threshold=0.1,
                           grow_mask=3, subtract=True,
                           verbose=True, reset=False):
    """Make a mask for pixels flagged as being affected by persistence

    Persistence products can be downloaded from https://archive.stsci.edu/prepds/persist/search.php, specifically the
    "_persist.fits" files.

    Parameters
    ----------
    flt_file : str
        Filename of the WFC3/IR FLT exposure

    path : str
        Path to look for the "persist.fits" file.

    dq_value : int
        DQ bit to flip for flagged pixels

    err_threshold : float
        ERR array threshold for defining affected pixels:

            >>> flagged = persist > err_threshold*ERR

    grow_mask : int
        Factor by which to dilate the persistence mask.

    subtract : bool
        Subtract the persistence model itself from the SCI extension.

    reset : bool
        Unset `dq_value` bit.

    verbose : bool
        Print information to the terminal

    Returns
    -------
    Nothing, updates the DQ extension of `flt_file`.

    """
    import scipy.ndimage as nd

    flt = pyfits.open(flt_file, mode='update')

    pers_file = os.path.join(path,
             os.path.basename(flt_file).replace('_flt.fits', '_persist.fits').replace('_rate.fits', '_persist.fits'))

    if not os.path.exists(pers_file):

        logstr = '# Persistence file {0} not found'.format(pers_file)
        utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

        # return 0

    pers = pyfits.open(pers_file)
    if pers['SCI'].data.min() < -40:
        subtract = False

    pers_data = pers['SCI'].data*1
    pers_data = np.maximum(pers_data, 0)

    pers_mask = pers['SCI'].data > err_threshold*flt['ERR'].data
    #pers_mask &= pers['SCI'].data > sci_threshold*flt['SCI'].data

    if grow_mask > 0:
        pers_mask = nd.maximum_filter(pers_mask*1, size=grow_mask)
    else:
        pers_mask = pers_mask * 1

    NPERS = pers_mask.sum()

    logstr = '# {0}: flagged {1:d} pixels affected by persistence (pers/err={2:.2f})'.format(pers_file, NPERS, err_threshold)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    flt[0].header['PERSNPIX'] = (NPERS, 'Number of persistence-flagged pixels')
    flt[0].header['PERSLEVL'] = (err_threshold, 'Perristence threshold err_threshold')
    flt[0].header['PERSGROW'] = (grow_mask, 'Perristence mask dilation grow_mask')

    if reset:
        flt['DQ'].data -= (flt['DQ'].data & dq_value)

    if NPERS > 0:
        flt['DQ'].data[pers_mask > 0] |= dq_value

        if subtract:
            dont_subtract = False
            if 'SUBPERS' in flt[0].header:
                if flt[0].header['SUBPERS']:
                    dont_subtract = True

            if not dont_subtract:
                flt['SCI'].data -= pers_data

            flt['ERR'].data = np.sqrt(flt['ERR'].data**2+pers_data**2)
            flt[0].header['SUBPERS'] = (True, 'Persistence model subtracted')

    flt.flush()
    flt.close()


def apply_region_mask(flt_file, dq_value=1024, verbose=True):
    """Apply DQ mask from a DS9 region file

    Parameters
    ----------
    flt_file : str
        Filename of a FLT exposure. The function searches for region files
        with filenames like 
        
        >>> mask_file = flt_file.replace('_flt.fits','.{ext}.mask.reg')
        
        where ``{ext}`` is an integer referring to the SCI extension in the 
        FLT file (1 for WFC3/IR, 1 or 2 for ACS/WFC and WFC3/UVIS).

    dq_value : int
        DQ bit to flip for affected pixels
    
    Returns
    -------
    Nothing, but updates the ``DQ`` extension of `flt_file` if a mask file
    is found
    
    """
    from regions import Regions

    mask_files = glob.glob('_'.join(flt_file.split('_')[:-1]) + '.*.mask.reg')
    if len(mask_files) == 0:
        return True

    logstr = '# Region mask for {0}: {1}'.format(flt_file, mask_files)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    flt = pyfits.open(flt_file, mode='update')
    for mask_file in mask_files:
        ext = int(mask_file.split('.')[-3])
        try:
            hdu = flt['SCI', ext]
            reg = Regions.read(mask_file, format='ds9')[0]
            reg_pix = reg.to_pixel(wcs=pywcs.WCS(hdu.header))
            shape = hdu.data.shape
            mask = reg_pix.to_mask().to_image(shape=shape).astype(bool)
        except:
            # Above fails for lookup-table distortion (ACS / UVIS)
            # Here just assume the region file is defined in image coords
            reg = Regions.read(mask_file, format='ds9')[0]
            shape = flt['SCI', ext].data.shape
            mask = reg.to_mask().to_image(shape=shape).astype(bool)

        flt['DQ', ext].data[mask] |= dq_value

    flt.flush()
    return True


def apply_saturated_mask(flt_file, dq_value=1024, verbose=True):
    """Saturated WFC3/IR pixels have some pulldown in the opposite amplifier

    Parameters
    ----------
    flt_file : str
        Filename of the FLT exposure

    dq_value : int
        DQ bit to flip for affected pixels

    Returns
    -------
    Nothing, modifies DQ extension of `flt_file` in place.

    """
    import scipy.ndimage as nd

    flt = pyfits.open(flt_file, mode='update')

    sat = (((flt['DQ'].data & 256) > 0) & ((flt['DQ'].data & 4) == 0))

    # Don't flag pixels in lower right corner
    sat[:80, -80:] = False

    # Flag only if a number of nearby pixels also saturated
    kern = np.ones((3, 3))
    sat_grow = nd.convolve(sat*1, kern)

    sat_mask = (sat & (sat_grow > 2))[::-1, :]*1

    NSAT = sat_mask.sum()

    logstr = '# {0}: flagged {1:d} pixels affected by saturation pulldown'
    logstr = logstr.format(flt_file, NSAT)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    if NSAT > 0:
        flt['DQ'].data[sat_mask > 0] |= dq_value
        flt.flush()


def clip_lists(input, output, clip=20):
    """Clip [x,y] arrays of objects that don't have a match within `clip` pixels in either direction
    
    Parameters
    ----------
    input : (array, array)
        Input pixel/array coordinates
    
    output : (array, array)
        Output pixel/array coordinates
    
    clip : float
        Matching distance
    
    Returns
    -------
    in_clip, out_clip : (array, array)
        Clipped coordinate lists
        
    """
    import scipy.spatial

    tree = scipy.spatial.cKDTree(input, 10)

    # Forward
    N = output.shape[0]
    dist, ix = np.zeros(N), np.zeros(N, dtype=int)
    for j in range(N):
        dist[j], ix[j] = tree.query(output[j, :], k=1,
                                    distance_upper_bound=np.inf)

    ok = dist < clip
    out_clip = output[ok]
    if ok.sum() == 0:
        print('No matches within `clip={0:f}`'.format(clip))
        return False

    # Backward
    tree = scipy.spatial.cKDTree(out_clip, 10)

    N = input.shape[0]
    dist, ix = np.zeros(N), np.zeros(N, dtype=int)
    for j in range(N):
        dist[j], ix[j] = tree.query(input[j, :], k=1,
                                    distance_upper_bound=np.inf)

    ok = dist < clip
    in_clip = input[ok]

    return in_clip, out_clip


def match_lists(input, output, transform=None, scl=3600., simple=True,
                outlier_threshold=5, toler=5, triangle_size_limit=[5, 800],
                triangle_ba_max=0.9, assume_close=False):
    """Compute matched objects and transformation between two (x,y) lists.
    
    Parameters
    ----------
    input : (array, array)
        Input pixel/array coordinates
    
    output : (array, array)
        Output pixel/array coordinates
    
    transform : None, `skimage.transform` object
        Coordinate transformation model.  If None, use
        `skimage.transform.EuclideanTransform`, i.e., shift & rotation
    
    scl : float
        Not used
    
    simple : bool
        Find matches manually within `outlier_threshold`.  If False, find 
        matches with `skimage.measure.ransac` and the specified `transform`
    
    outlier_threshold : float
        Match threshold for ``simple=False``
        
    triangle_size_limit : (float, float)
        Size limit of matching triangles, generally set to something of order
        of the detector size
    
    triangle_ba_max : float
        Maximum length/height ratio of matching triangles
    
    assume_close : bool
        Not used
    
    Returns
    -------
    input_ix : (array, array)
        Array indices of matches from `input` 
    
    output_ix : (array, array)
        Array indices of matches from `output` 
    
    outliers : (array, array)
        Array indices of outliers
    
    model : transform
        Instance of the `transform` object based on the matches
        
    """
    import copy
    from astropy.table import Table

    import skimage.transform
    from skimage.measure import ransac

    import stsci.stimage

    try:
        import tristars
        from tristars.match import match_catalog_tri
    except ImportError:
        print("""
    Couldn't `import tristars`.  Get it from https://github.com/gbrammer/tristars to enable improved blind astrometric matching with triangle asterisms.
    """)

    if transform is None:
        transform = skimage.transform.EuclideanTransform

    # print 'xyxymatch'
    if (len(output) == 0) | (len(input) == 0):
        print('No entries!')
        return input, output, None, transform()

    try:
        pair_ix = match_catalog_tri(input, output, maxKeep=10, auto_keep=3, 
                                    auto_transform=transform, 
                                    auto_limit=outlier_threshold, 
                                    size_limit=triangle_size_limit, 
                                    ignore_rot=False, ignore_scale=True, 
                                    ba_max=triangle_ba_max)

        input_ix = pair_ix[:, 0]
        output_ix = pair_ix[:, 1]
        
        msg = '  tristars.match: Nin={0}, Nout={1}, match={2}'
        print(msg.format(len(input), len(output), len(output_ix)))

        # if False:
        #     fig = match.match_diagnostic_plot(input, output, pair_ix, tf=None, new_figure=True)
        #     fig.savefig('/tmp/xtristars.png')
        #     plt.close(fig)
        # 
        #     tform = get_transform(input, output, pair_ix, transform=transform, use_ransac=True)

    except:

        utils.log_exception(utils.LOGFILE, traceback)
        utils.log_comment(utils.LOGFILE, "# ! tristars failed")

        match = stsci.stimage.xyxymatch(copy.copy(input), copy.copy(output),
                                        origin=np.median(input, axis=0),
                                        mag=(1.0, 1.0), rotation=(0.0, 0.0),
                                        ref_origin=np.median(input, axis=0),
                                        algorithm='tolerance', 
                                        tolerance=toler,
                                        separation=0.5, nmatch=10, 
                                        maxratio=10.0,
                                        nreject=10)

        m = Table(match)

        output_ix = m['ref_idx'].data
        input_ix = m['input_idx'].data

        print('  xyxymatch.match: Nin={0}, Nout={1}, match={2}'.format(len(input), len(output), len(output_ix)))

    tf = transform()
    tf.estimate(input[input_ix, :], output[output_ix])

    if not simple:
        model, inliers = ransac((input[input_ix, :], output[output_ix, :]),
                                   transform, min_samples=3,
                                   residual_threshold=3, max_trials=100)

        # Iterate
        if inliers.sum() > 2:
            m_i, in_i = ransac((input[input_ix[inliers], :], 
                                output[output_ix[inliers], :]),
                                   transform, min_samples=3,
                                   residual_threshold=3, max_trials=100)
            if in_i.sum() > 2:
                model = m_i
                inliers[np.arange(len(inliers), dtype=np.int)[inliers][in_i]] = False

        outliers = ~inliers
        mout = model(input[input_ix, :])
        dx = mout - output[output_ix]

    else:
        model = tf
        # Compute statistics
        if len(input_ix) > 10:
            mout = tf(input[input_ix, :])
            dx = mout - output[output_ix]
            dr = np.sqrt(np.sum(dx**2, axis=1))
            outliers = dr > outlier_threshold
        else:
            outliers = np.zeros(len(input_ix), dtype=bool)

    return input_ix, output_ix, outliers, model


def align_drizzled_image(root='',
                         mag_limits=[14, 23],
                         radec=None,
                         NITER=3,
                         clip=20,
                         log=True,
                         outlier_threshold=5,
                         verbose=True,
                         guess=[0., 0., 0., 1],
                         simple=True,
                         rms_limit=2,
                         use_guess=False,
                         triangle_size_limit=[5, 1800],
                         max_sources=200,
                         triangle_ba_max=0.9,
                         max_err_percentile=99,
                         catalog_mask_pad=0.05,
                         min_flux_radius=1.,
                         match_catalog_density=None,
                         assume_close=False,
                         ref_border=100, 
                         transform=None):
    """Pipeline for astrometric alignment of drizzled image products
    
    1. Generate source catalog from image mosaics
    2. Trim catalog lists
    3. Find matches and compute (shift, rot, scale) transform
    
    Parameters
    ----------
    root : str
        Image product rootname, passed to `~grizli.prep.make_SEP_catalog`
    
    mag_limits : (float, float)
        AB magnitude limits of objects in the image catalog to use for 
        the alignment
        
    radec : str or (array, array)
        Reference catalog positions (ra, dec).  If `str`, will read from a
        file with `np.loadtxt`, assuming just two columns
        
    NITER : int
        Number of matching/transform iterations to perform

    clip : float
        If positive, then coordinate arrays will be clipped with 
        `~grizli.prep.clip_lists`.
    
    log : bool
        Write results to `wcs.log` file and make a diagnostic figure
        
    verbose : bool
        Print status message to console
        
    guess : list
        Initial guess for alignment: 
        
        >>> guess = [0., 0., 0., 1]
        >>> guess = [xshift, yshift, rot, scale]
        
    use_guess : bool
        Use the `guess`
    
    rms_limit : float
        If transform RMS exceeds this threshold, use null [0,0,0,1] transform
        
    simple : bool
        Parameter for `~grizli.prep.match_lists`
        
    outlier_threshold : float
        Parameter for `~grizli.prep.match_lists`
    
    triangle_size_limit : (float, float)
        Parameter for `~grizli.prep.match_lists`
        
    triangle_ba_max : float
        Parameter for `~grizli.prep.match_lists`
    
    max_sources : int
        Maximum number of sources to use for the matches.  Triangle matching
        combinatorics become slow for hundreds of sources
         
    max_err_percentile : float
        Only use sources where weight image is greater than this percentile
        to try to limit spurious sources in low-weight regions 
        (`~grizli.utils.catalog_mask`)
        
    catalog_mask_pad : float
        Mask sources outside of this fractional size of the image dimensions
        to try to limit spurius sources (`~grizli.utils.catalog_mask`)
    
    match_catalog_density : bool, None
        Try to roughly match the surface density of the reference and target
        source lists, where the latter is sorted by brightness to try to 
        reduce spurious triangle matches
        
    assume_close : bool
        not used
        
    ref_border : float
        Only include reference sources within `ref_border` pixels of the 
        target image, as calculated from the original image WCS
    
    transform : None, `skimage.transform` object
            Coordinate transformation model.  If None, use
            `skimage.transform.EuclideanTransform`, i.e., shift & rotation
    Returns
    -------
    orig_wcs : `~astropy.wcs.WCS`
        Original WCS
    
    drz_wcs : `~astropy.wcs.WCS`
        Transformed WCS
    
    out_shift : (float, float)
        Translation, pixels
    
    out_rot : float
        Rotation (degrees)
    
    out_scale : float
        Scale 
        
    """
    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'prep.align_drizzled_image')

    if not os.path.exists('{0}.cat.fits'.format(root)):
        #cat = make_drz_catalog(root=root)
        cat = make_SEP_catalog(root=root)
    else:
        cat = utils.read_catalog('{0}.cat.fits'.format(root))

    if hasattr(radec, 'upper'):
        rd_ref = np.loadtxt(radec)
        radec_comment = radec
        if match_catalog_density is None:
            match_catalog_density = '.cat.radec' not in radec
    elif radec is False:
        # Align to self, i.e., do nothing
        so = np.argsort(cat['MAG_AUTO'])
        rd_ref = np.array([cat['X_WORLD'], cat['Y_WORLD']]).T[so[:50], :]
        radec_comment = 'self catalog'
        if match_catalog_density is None:
            match_catalog_density = False
    else:
        rd_ref = radec*1
        radec_comment = 'input arrays (N={0})'.format(rd_ref.shape)
        if match_catalog_density is None:
            match_catalog_density = False

    # Clip obviously distant files to speed up match
    # rd_cat = np.array([cat['X_WORLD'], cat['Y_WORLD']])
    # rd_cat_center = np.median(rd_cat, axis=1)
    # cosdec = np.array([np.cos(rd_cat_center[1]/180*np.pi),1])
    # dr_cat = np.sqrt(np.sum((rd_cat.T-rd_cat_center)**2*cosdec**2, axis=1))
    #
    # #print('xxx', rd_ref.shape, rd_cat_center.shape, cosdec.shape)
    #
    # dr = np.sqrt(np.sum((rd_ref-rd_cat_center)**2*cosdec**2, axis=1))
    #
    # rd_ref = rd_ref[dr < 1.1*dr_cat.max(),:]

    ok = (cat['MAG_AUTO'] > mag_limits[0]) & (cat['MAG_AUTO'] < mag_limits[1])
    if len(mag_limits) > 2:
        ok &= cat['MAGERR_AUTO'] < mag_limits[2]
    else:
        ok &= cat['MAGERR_AUTO'] < 0.05

    if ok.sum() == 0:
        print('{0}.cat: no objects found in magnitude range {1}'.format(root,
                                                                 mag_limits))
        return False

    # Edge and error mask
    ok &= utils.catalog_mask(cat, max_err_percentile=max_err_percentile,
                             pad=catalog_mask_pad, pad_is_absolute=False, 
                             min_flux_radius=min_flux_radius)

    if max_err_percentile >= 200:
        med_err = np.median(cat['FLUXERR_APER_0'][ok])
        max_err = med_err*np.sqrt(2)
        ok_err = cat['FLUXERR_APER_0'] < max_err
        if ok_err.sum() > 5:
            ok &= ok_err

    xy_drz = np.array([cat['X_IMAGE'][ok], cat['Y_IMAGE'][ok]]).T

    drz_file = glob.glob('{0}_dr[zc]_sci.fits'.format(root))[0]
    drz_im = pyfits.open(drz_file)
    sh = drz_im[0].data.shape

    drz_wcs = pywcs.WCS(drz_im[0].header, relax=True)
    orig_wcs = drz_wcs.copy()

    if use_guess:
        drz_wcs = utils.transform_wcs(drz_wcs, guess[:2], guess[2], guess[3])
        return orig_wcs, drz_wcs,  guess[:2], guess[2]/np.pi*180, guess[3]

    ##########
    # Only include reference objects in the DRZ footprint
    pix_origin = 1
    ref_x, ref_y = drz_wcs.all_world2pix(rd_ref, pix_origin).T
    if hasattr(drz_wcs, '_naxis1'):
        nx1, nx2 = drz_wcs._naxis1, drz_wcs._naxis2
    else:
        nx1, nx2 = drz_wcs._naxis

    ref_cut = (ref_x > -ref_border) & (ref_x < nx1+ref_border)
    ref_cut &= (ref_y > -ref_border) & (ref_y < nx2+ref_border)

    if ref_cut.sum() == 0:
        print(f'{root}: no reference objects found in the DRZ footprint')
        return False

    rd_ref = rd_ref[ref_cut, :]

    ########
    # Match surface density of drizzled and reference catalogs
    if match_catalog_density:
        icut = np.minimum(ok.sum()-2, int(2*ref_cut.sum()))
        # acat = utils.hull_area(cat['X_WORLD'][ok], cat['Y_WORLD'][ok])
        # aref = utils.hull_area(rd_ref[:,0], rd_ref[:,1])

        cut = np.argsort(cat['MAG_AUTO'][ok])[:icut]
        xy_drz = np.array([cat['X_IMAGE'][ok][cut],
                           cat['Y_IMAGE'][ok][cut]]).T
    else:
        # Limit to brightest X objects
        icut = 400
        cut = np.argsort(cat['MAG_AUTO'][ok])[:icut]
        xy_drz = np.array([cat['X_IMAGE'][ok][cut],
                           cat['Y_IMAGE'][ok][cut]]).T

    logstr = '# wcs {0} radec="{1}"; Ncat={2}; Nref={3}'
    logstr = logstr.format(root, radec, xy_drz.shape[0], rd_ref.shape[0])
    utils.log_comment(utils.LOGFILE, logstr, verbose=True)

    #out_shift, out_rot, out_scale = np.zeros(2), 0., 1.
    out_shift, out_rot, out_scale = guess[:2], guess[2], guess[3]
    drz_wcs = utils.transform_wcs(drz_wcs, out_shift, out_rot, out_scale)

    logstr = '# wcs {0} (guess)   : {1:6.2f} {2:6.2f} {3:7.3f} {4:7.3f}'
    logstr = logstr.format(root, guess[0], guess[1], guess[2]/np.pi*180, 
                           1./guess[3])
    utils.log_comment(utils.LOGFILE, logstr, verbose=True)

    drz_crpix = drz_wcs.wcs.crpix

    NGOOD, rms = 0, 0
    for iter in range(NITER):
        #print('xx iter {0} {1}'.format(iter, NITER))
        xy = np.array(drz_wcs.all_world2pix(rd_ref, pix_origin))
        pix = np.cast[int](np.round(xy)).T

        # Find objects where drz pixels are non-zero
        okp = (pix[0, :] > 0) & (pix[1, :] > 0)
        okp &= (pix[0, :] < sh[1]) & (pix[1, :] < sh[0])
        ok2 = drz_im[0].data[pix[1, okp], pix[0, okp]] != 0

        N = ok2.sum()
        if clip > 0:
            status = clip_lists(xy_drz-drz_crpix, xy+0-drz_crpix, clip=clip)
            if not status:
                print('Problem xxx')

            input, output = status
        else:
            input, output = xy_drz+0.-drz_crpix, xy+0-drz_crpix

        if len(input) > max_sources:
            msg = 'Clip input list ({0}) to {1} objects'
            print(msg.format(len(input), max_sources))
            ix = np.argsort(np.arange(len(input)))[:max_sources]
            input = input[ix, :]

        if len(output) > max_sources:
            msg = 'Clip output list ({0}) to {1} objects'
            print(msg.format(len(input), max_sources))
            ix = np.argsort(np.arange(len(output)))[:max_sources]
            output = output[ix, :]

        toler = 5
        titer = 0
        while (titer < 3):
            try:
                res = match_lists(output, input, scl=1.,
                                  simple=simple,
                                  outlier_threshold=outlier_threshold,
                                  toler=toler,
                                  triangle_size_limit=triangle_size_limit,
                                  triangle_ba_max=triangle_ba_max,
                                  assume_close=assume_close,
                                  transform=transform)

                output_ix, input_ix, outliers, tf = res
                break
            except:
                toler += 5
                titer += 1
        
        #print(output.shape, output_ix.shape, output_ix.min(), output_ix.max(), titer, toler, input_ix.shape, input.shape)

        titer = 0
        while (len(input_ix)*1./len(input) < 0.1) & (titer < 3):
            titer += 1
            toler += 5
            try:
                res = match_lists(output, input, scl=1.,
                                  simple=simple,
                                  outlier_threshold=outlier_threshold,
                                  toler=toler,
                                  triangle_size_limit=triangle_size_limit,
                                  triangle_ba_max=triangle_ba_max,
                                  assume_close=assume_close,
                                  transform=transform)
            except:
                pass

            output_ix, input_ix, outliers, tf = res

        #print(output.shape, output_ix.shape, output_ix.min(), output_ix.max(), titer, toler, input_ix.shape, input.shape)

        tf_out = tf(output[output_ix])
        dx = input[input_ix] - tf_out
        rms = utils.nmad(np.sqrt((dx**2).sum(axis=1)))

        if len(outliers) > 20:
            outliers = (np.sqrt((dx**2).sum(axis=1)) > 4*rms)
        else:
            outliers = (np.sqrt((dx**2).sum(axis=1)) > 10*rms)

        if outliers.sum() > 0:
            res2 = match_lists(output[output_ix][~outliers],
                              input[input_ix][~outliers], scl=1.,
                              simple=simple,
                              outlier_threshold=outlier_threshold,
                              toler=toler,
                              triangle_size_limit=triangle_size_limit,
                              triangle_ba_max=triangle_ba_max,
                              transform=transform)

            output_ix2, input_ix2, outliers2, tf = res2
        
        # print('xxx')
        # import tristars.match
        # pair_ix = np.zeros((len(output_ix), 2), dtype=int)
        # pair_ix[:,0] = output_ix
        # pair_ix[:,1] = input_ix
        # print(output_ix, input_ix, len(output), len(input))
        # 
        # fig = tristars.match.match_diagnostic_plot(output, input, pair_ix, tf=tf, new_figure=True)
        # fig.savefig('xxx.png')
                                              
        # Log
        shift = tf.translation
        if hasattr(tf, 'scale'):
            _tfscale = tf.scale
        else:
            _tfscale = 1.0
        
        NGOOD = (~outliers).sum()
        logstr = '# wcs {0} ({1:d}) {2:d}: {3:6.2f} {4:6.2f} {5:7.3f} {6:7.3f}'
        logstr = logstr.format(root, iter, NGOOD, shift[0], shift[1],
                               tf.rotation/np.pi*180, 1./_tfscale)

        utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

        out_shift += tf.translation
            
        out_rot -= tf.rotation
        out_scale *= _tfscale

        drz_wcs = utils.transform_wcs(drz_wcs, tf.translation, tf.rotation,
                                      _tfscale)

        # drz_wcs.wcs.crpix += tf.translation
        # theta = -tf.rotation
        # _mat = np.array([[np.cos(theta), -np.sin(theta)],
        #                  [np.sin(theta), np.cos(theta)]])
        #
        # drz_wcs.wcs.cd = np.dot(drz_wcs.wcs.cd, _mat)/tf.scale

    # Bad fit
    if (rms > rms_limit) | (NGOOD < 3):
        drz_wcs = orig_wcs
        out_shift = [0, 0]
        out_rot = 0.
        out_scale = 1.

        log = False

    if log:
        tf_out = tf(output[output_ix][~outliers])
        dx = input[input_ix][~outliers] - tf_out
        rms = utils.nmad(np.sqrt((dx**2).sum(axis=1)))

        interactive_status = plt.rcParams['interactive']
        plt.ioff()

        fig = plt.figure(figsize=[6., 6.])
        ax = fig.add_subplot(111)
        ax.scatter(dx[:, 0], dx[:, 1], alpha=0.5, color='b')
        ax.scatter([0], [0], marker='+', color='red', s=40)
        ax.set_xlabel(r'$dx$')
        ax.set_ylabel(r'$dy$')
        ax.set_title(root)

        ax.set_xlim(-7*rms, 7*rms)
        ax.set_ylim(-7*rms, 7*rms)
        ax.grid()

        fig.tight_layout(pad=0.1)
        fig.savefig('{0}_wcs.png'.format(root))
        plt.close()

        if interactive_status:
            plt.ion()

    log_wcs(root, orig_wcs, out_shift, out_rot/np.pi*180, out_scale, rms,
            n=NGOOD, initialize=False,
            comment=['radec: {0}'.format(radec_comment)])

    return orig_wcs, drz_wcs, out_shift, out_rot/np.pi*180, out_scale


def update_wcs_fits_log(file, wcs_ref, xyscale=[0, 0, 0, 1], initialize=True, replace=('.fits', '.wcslog.fits'), wcsname='SHIFT'):
    """
    Make FITS log when updating WCS
    """
    new_hdu = wcs_ref.to_fits(relax=True)[0]
    new_hdu.header['XSHIFT'] = xyscale[0]
    new_hdu.header['YSHIFT'] = xyscale[1]
    new_hdu.header['ROT'] = xyscale[2], 'WCS fit rotation, degrees'
    new_hdu.header['SCALE'] = xyscale[3], 'WCS fit scale'
    new_hdu.header['WCSNAME'] = wcsname

    wcs_logfile = file.replace(replace[0], replace[1])

    if os.path.exists(wcs_logfile):
        if initialize:
            os.remove(wcs_logfile)
            hdu = pyfits.HDUList([pyfits.PrimaryHDU()])
        else:
            hdu = pyfits.open(wcs_logfile)
    else:
        hdu = pyfits.HDUList([pyfits.PrimaryHDU()])

    hdu.append(new_hdu)
    hdu.writeto(wcs_logfile, overwrite=True, output_verify='fix')


def log_wcs(root, drz_wcs, shift, rot, scale, rms=0., n=-1, initialize=True, comment=[]):
    """Save WCS offset information to an ascii file
    """
    if (not os.path.exists('{0}_wcs.log'.format(root))) | initialize:
        print('Initialize {0}_wcs.log'.format(root))
        orig_hdul = pyfits.HDUList()
        fp = open('{0}_wcs.log'.format(root), 'w')
        fp.write('# ext xshift yshift rot scale rms N\n')
        for c in comment:
            fp.write('# {0}\n'.format(c))

        fp.write('# {0}\n'.format(root))
        count = 0
    else:
        orig_hdul = pyfits.open('{0}_wcs.fits'.format(root))
        fp = open('{0}_wcs.log'.format(root), 'a')
        count = len(orig_hdul)

    hdu = drz_wcs.to_fits()[0]
    hdu.header['XSHIFT'] = shift[0]
    hdu.header['YSHIFT'] = shift[1]

    hdu.header['ROT'] = rot, 'WCS fit rotation, degrees'
    hdu.header['SCALE'] = scale, 'WCS fit scale'

    hdu.header['FIT_RMS'] = rot, 'WCS fit RMS'
    hdu.header['FIT_N'] = n, 'Number of sources in WCS fit'
    hdu.header['EXTNAME'] = 'WCS'
    for i, e in enumerate(orig_hdul):
        orig_hdul[e].header['EXTNAME'] = f'WCS{i}'
        
    orig_hdul.append(hdu)
    orig_hdul.writeto('{0}_wcs.fits'.format(root), overwrite=True)

    fp.write('{0:5d} {1:13.4f} {2:13.4f} {3:13.4f} {4:13.5f} {5:13.3f} {6:4d}\n'.format(
              count, shift[0], shift[1], rot, scale, rms, n))

    fp.close()


SEXTRACTOR_DEFAULT_PARAMS = ["NUMBER", "X_IMAGE", "Y_IMAGE", "X_WORLD",
                    "Y_WORLD", "A_IMAGE", "B_IMAGE", "THETA_IMAGE",
                    "MAG_AUTO", "MAGERR_AUTO", "FLUX_AUTO", "FLUXERR_AUTO",
                    "FLUX_RADIUS", "BACKGROUND", "FLAGS"]

# Aperture *Diameters*
SEXTRACTOR_PHOT_APERTURES = "6, 8.33335, 11.66667, 16.66667, 20, 25, 50"
SEXTRACTOR_PHOT_APERTURES_ARCSEC = [float(ap)*0.06*u.arcsec for ap in SEXTRACTOR_PHOT_APERTURES.split(',')]

SEXTRACTOR_CONFIG_3DHST = {'DETECT_MINAREA': 14, 'DEBLEND_NTHRESH': 32, 'DEBLEND_MINCONT': 0.005, 'FILTER_NAME': '/usr/local/share/sextractor/gauss_3.0_7x7.conv', 'FILTER': 'Y'}

# /usr/local/share/sextractor/gauss_3.0_7x7.conv
GAUSS_3_7x7 = np.array(
    [[0.0049,  0.0213,  0.0513,  0.0687,  0.0513,  0.0213,  0.0049],
     [0.0213,  0.0921,  0.2211,  0.2960,  0.2211,  0.0921,  0.0213],
     [0.0513,  0.2211,  0.5307,  0.7105,  0.5307,  0.2211,  0.0513],
     [0.0687,  0.2960,  0.7105,  0.9511,  0.7105,  0.2960,  0.0687],
     [0.0513,  0.2211,  0.5307,  0.7105,  0.5307,  0.2211,  0.0513],
     [0.0213,  0.0921,  0.2211,  0.2960,  0.2211,  0.0921,  0.0213],
     [0.0049,  0.0213,  0.0513,  0.0687,  0.0513,  0.0213,  0.0049]])

# Try to match 3D-HST detection
SEP_DETECT_PARAMS_3DHST = {'minarea': 9, 'filter_kernel': GAUSS_3_7x7,
                    'filter_type': 'conv', 'clean': True, 'clean_param': 1,
                    'deblend_nthresh': 32, 'deblend_cont': 0.005}

# More agressive deblending
SEP_DETECT_PARAMS = {'minarea': 9, 'filter_kernel': GAUSS_3_7x7,
                    'filter_type': 'conv', 'clean': True, 'clean_param': 1,
                    'deblend_nthresh': 32, 'deblend_cont': 0.001}


def make_SEP_FLT_catalog(flt_file, ext=1, column_case=str.upper, **kwargs):
    """
    Make a catalog from a FLT file
    
    (Not used)
    """
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    im = pyfits.open(flt_file)
    sci = im['SCI', ext].data - im['SCI', ext].header['MDRIZSKY']
    err = im['ERR', ext].data
    mask = im['DQ', ext].data > 0

    ZP = utils.calc_header_zeropoint(im, ext=('SCI', ext))

    try:
        wcs = pywcs.WCS(im['SCI', ext].header, fobj=im)
    except:
        wcs = None

    tab, seg = make_SEP_catalog_from_arrays(sci, err, mask, wcs=wcs, ZP=ZP, **kwargs)
    tab.meta['ABZP'] = ZP
    tab.meta['FILTER'] = utils.parse_filter_from_header(im[0].header)
    tab['mag_auto'] = ZP - 2.5*np.log10(tab['flux'])

    for c in tab.colnames:
        tab.rename_column(c, column_case(c))

    return tab, seg


def make_SEP_catalog_from_arrays(sci, err, mask, wcs=None, threshold=2., ZP=25, get_background=True, detection_params=SEP_DETECT_PARAMS, segmentation_map=False, verbose=True):
    """
    Make a catalog from arrays using `sep`
    
    Parameters
    ----------
    sci : 2D array
        Data array
    
    err : 2D array
        Uncertainties in same units as `sci`
    
    mask : bool array
        `sep` masks values where ``mask > 0``
    
    wcs : `~astropy.wcs.WCS`
        WCS associated with data arrays
    
    thresh : float
        Detection threshold for `sep.extract`
    
    ZP : float
        AB magnitude zeropoint of data arrays
    
    get_background : bool
        not used
    
    detection_params : dict
        Keyword arguments for `sep.extract`
    
    segmentation_map : bool
        Also create a segmentation map
    
    verbose : bool
        Print status messages
    
    Returns
    -------
    tab : `~astropy.table.Table`
        Source catalog
    
    seg : array, None
        Segmentation map, if requested
        
    """
    import copy
    import astropy.units as u

    try:
        import sep
    except ImportError:
        print("""
    Couldn't `import sep`.  SExtractor replaced with SEP
    in April 2018.  Install with `pip install sep`.
    """)

    logstr = 'make_SEP_catalog_from_arrays: sep version = {0}'.format(sep.__version__)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)
    if sep.__version__ < '1.1':
        print("""!!!!!!!!!!
!  SEP version = {0}
!  Get >= 1.10.0 to enable segmentation masking, e.g.,
!  $ pip install git+https://github.com/gbrammer/sep.git
!!!!!!!!!!
        """.format(sep.__version__))

    uJy_to_dn = 1/(3631*1e6*10**(-0.4*ZP))

    if sci.dtype != np.float32:
        sci_data = sci.byteswap().newbyteorder()
    else:
        sci_data = sci

    if err.dtype != np.float32:
        err_data = err.byteswap().newbyteorder()
    else:
        err_data = err

    if segmentation_map:
        objects, seg = sep.extract(sci_data, threshold, err=err_data,
                   mask=mask, segmentation_map=True, **detection_params)
    else:
        objects = sep.extract(sci_data, threshold, err=err_data,
                   mask=mask, segmentation_map=False, **detection_params)
        seg = None

    tab = utils.GTable(objects)

    if wcs is not None:
        tab['ra'], tab['dec'] = wcs.all_pix2world(tab['x'], tab['y'], 0)
        tab['ra'].unit = u.deg
        tab['dec'].unit = u.deg
        tab['x_world'], tab['y_world'] = tab['ra'], tab['dec']

    return tab, seg


def get_SEP_flag_dict():
    """Get dictionary of SEP integer flags
    
    Returns
    -------
    flags : dict
        Dictionary of the integer `sep` detection flags, which are set as 
        attributes on the `sep` module
        
    """
    try:
        import sep
    except ImportError:
        print("""
    Couldn't `import sep`.  SExtractor replaced with SEP
    in April 2018.  Install with `pip install sep`.
    """)

    flags = OrderedDict()
    for f in ['OBJ_MERGED', 'OBJ_TRUNC', 'OBJ_DOVERFLOW', 'OBJ_SINGU',
              'APER_TRUNC', 'APER_HASMASKED', 'APER_ALLMASKED', 
              'APER_NONPOSITIVE']:
        flags[f] = getattr(sep, f)

    return flags


def make_SEP_catalog(root='', 
                     sci=None,
                     wht=None,
                     threshold=2., 
                     get_background=True,
                     bkg_only=False,
                     bkg_params={'bw': 32, 'bh': 32, 'fw': 3, 'fh': 3},
                     verbose=True,
                     phot_apertures=SEXTRACTOR_PHOT_APERTURES,
                     aper_segmask=False,
                     rescale_weight=True,
                     err_scale=-np.inf,
                     use_bkg_err=False,
                     column_case=str.upper,
                     save_to_fits=True,
                     include_wcs_extension=True,
                     source_xy=None, 
                     compute_auto_quantities=True,
                     autoparams=[2.5, 0.35*u.arcsec, 2.4, 3.8],
                     flux_radii=[0.2, 0.5, 0.9],
                     subpix=0,
                     mask_kron=False,
                     max_total_corr=2,
                     detection_params=SEP_DETECT_PARAMS,
                     bkg_mask=None,
                     pixel_scale=0.06,
                     log=False,
                     gain=2000., 
                     extract_pixstack=int(3e7),
                     **kwargs):
    """Make a catalog from drizzle products using the SEP implementation of SourceExtractor
    
    Parameters
    ----------
    root : str
        Rootname of the FITS images to use for source extraction.  This
        function is designed to work with the single-image products from 
        `drizzlepac`, so the default data/science image is searched by 
        
        >>> drz_file = glob.glob(f'{root}_dr[zc]_sci.fits*')[0]
        
        Note that this will find and use gzipped versions of the images, 
        if necessary.
        
        The associated weight image filename is then assumed to be
        
        >>> weight_file = drz_file.replace('_sci.fits', '_wht.fits')
        >>> weight_file = weight_file.replace('_drz.fits', '_wht.fits')
        
    sci, wht : str
        Filenames to override `drz_file` and `weight_file` derived from the
        ``root`` parameter.

    threshold : float
        Detection threshold for `sep.extract`
    
    get_background : bool
        Compute the background with `sep.Background`
    
    bkg_only : bool
        If `True`, then just return the background data array and don't run
        the source detection
    
    bkg_params : dict
        Keyword arguments for `sep.Background`.  Note that this can include
        a separate optional keyword ``pixel_scale`` that indicates that the 
        background sizes `bw`, `bh` are set for a paraticular pixel size.  
        They will be scaled to the pixel dimensions of the target images using
        the pixel scale derived from the image WCS.
    
    verbose : bool
        Print status messages
            
    phot_apertures : str or array-like
        Photometric aperture *diameters*. If given as a string then assume
        units of pixels. If an array or list, can have units, e.g.,
        `astropy.units.arcsec`.

    aper_segmask : bool
        If true, then run SEP photometry with segmentation masking.  This
        requires the sep fork at https://github.com/gbrammer/sep.git, 
        or `sep >= 1.10.0`.
    
    rescale_weight : bool
        If true, then a scale factor is calculated from the ratio of the 
        weight image to the variance estimated by `sep.Background`.  
    
    err_scale : float
        Explicit value to use for the weight scaling, rather than calculating
        with `rescale_weight`.  Only used if ``err_scale > 0``
    
    use_bkg_err : bool
        If true, then use the full error array derived by `sep.Background`.
        This is turned off by default in order to preserve the pixel-to-pixel 
        variation in the drizzled weight maps.
        
    column_case : func
        Function to apply to the catalog column names.  E.g., the default 
        `str.upper` results in uppercase column names
    
    save_to_fits : bool
        Save catalog FITS file ``{root}.cat.fits``
    
    include_wcs_extension : bool
        An extension will be added to the FITS catalog with the detection 
        image WCS
    
    source_xy : (x, y) or (ra, dec) arrays
        Force extraction positions.  If the arrays have units, then pass them
        through the header WCS.  If no units, positions are *zero indexed*
        array coordinates.

        To run with segmentation masking (`1sep > 1.10``), also provide 
        `aseg` and `aseg_id` arrays with `source_xy`, like

            >>> source_xy = ra, dec, aseg, aseg_id
    
    compute_auto_quantities : bool
        Compute Kron/auto-like quantities with 
        `~grizli.prep.compute_SEP_auto_params`

    autoparams : list
        Parameters of Kron/AUTO calculations with 
        `~grizli.prep.compute_SEP_auto_params`.
        
    flux_radii : list
        Light fraction radii to compute with 
        `~grizli.prep.compute_SEP_auto_params`, e.g., ``[0.5]`` will calculate
        the half-light radius (``FLUX_RADIUS``)
    
    subpix : int
        Pixel oversampling
        
    mask_kron : bool
        Not used
        
    max_total_corr : float
        Not used
    
    detection_params : dict
        Parameters passed to `sep.extract`
    
    bkg_mask : array
        Additional mask to apply to `sep.Background` calculation
        
    pixel_scale : float
        Not used
    
    log : bool
        Send log message to `grizli.utils.LOGFILE`
    
    gain : float
        Gain value passed to `sep.sum_circle`
        
    extract_pixstack : int
        See `sep.set_extract_pixstack`
    
    Returns
    -------
    tab : `~astropy.table.Table`
        Source catalog
    
    
    """
    if log:
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'prep.make_SEP_catalog', verbose=True)

    import copy
    import astropy.units as u
    
    try:
        import sep
    except ImportError:
        print("""
    Couldn't `import sep`.  SExtractor replaced with SEP
    in April 2018.  Install with `pip install sep`.
    """)

    sep.set_extract_pixstack(extract_pixstack)

    logstr = 'make_SEP_catalog: sep version = {0}'.format(sep.__version__)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)
    if sep.__version__ < '1.1':
        print("""!!!!!!!!!!
!  SEP version = {0}
!  Get >= 1.10.0 to enable segmentation masking, e.g.,
!  $ pip install git+https://github.com/gbrammer/sep.git
!!!!!!!!!!
        """.format(sep.__version__))

    if sci is not None:
        drz_file = sci
    else:
        drz_file = glob.glob(f'{root}_dr[zc]_sci.fits*')[0]

    im = pyfits.open(drz_file)

    # Filter
    drz_filter = utils.parse_filter_from_header(im[0].header)
    if 'PHOTPLAM' in im[0].header:
        drz_photplam = im[0].header['PHOTPLAM']
    else:
        drz_photplam = None

    # Get AB zeropoint
    ZP = utils.calc_header_zeropoint(im, ext=0)

    logstr = 'sep: Image AB zeropoint =  {0:.3f}'.format(ZP)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose, show_date=True)

    # Scale fluxes to mico-Jy
    uJy_to_dn = 1/(3631*1e6*10**(-0.4*ZP))

    if wht is not None:
        weight_file = wht
    else:
        weight_file = drz_file.replace('_sci.fits', '_wht.fits')
        weight_file = weight_file.replace('_drz.fits', '_wht.fits')

    if (weight_file == drz_file) | (not os.path.exists(weight_file)):
        WEIGHT_TYPE = "NONE"
        weight_file = None
    else:
        WEIGHT_TYPE = "MAP_WEIGHT"

    drz_im = pyfits.open(drz_file)
    data = drz_im[0].data.byteswap().newbyteorder()

    try:
        wcs = pywcs.WCS(drz_im[0].header)
        wcs_header = utils.to_header(wcs)
        pixel_scale = utils.get_wcs_pscale(wcs)  # arcsec
    except:
        wcs = None
        wcs_header = drz_im[0].header.copy()
        pixel_scale = np.sqrt(wcs_header['CD1_1']**2+wcs_header['CD1_2']**2)
        pixel_scale *= 3600.  # arcsec

    # Add some header keywords to the wcs header
    for k in ['EXPSTART', 'EXPEND', 'EXPTIME']:
        if k in drz_im[0].header:
            wcs_header[k] = drz_im[0].header[k]

    if isinstance(phot_apertures, str):
        apertures = np.cast[float](phot_apertures.replace(',', '').split())
    else:
        apertures = []
        for ap in phot_apertures:
            if hasattr(ap, 'unit'):
                apertures.append(ap.to(u.arcsec).value/pixel_scale)
            else:
                apertures.append(ap)

    # Do we need to compute the error from the wht image?
    need_err = (not use_bkg_err) | (not get_background)
    if (weight_file is not None) & need_err:
        wht_im = pyfits.open(weight_file)
        wht_data = wht_im[0].data.byteswap().newbyteorder()

        err = 1/np.sqrt(wht_data)
        del(wht_data)

        # True mask pixels are masked with sep
        mask = (~np.isfinite(err)) | (err == 0) | (~np.isfinite(data))
        err[mask] = 0

        wht_im.close()
        del(wht_im)

    else:
        # True mask pixels are masked with sep
        mask = (data == 0) | (~np.isfinite(data))
        err = None

    try:
        drz_im.close()
        del(drz_im)
    except:
        pass

    data_mask = np.cast[data.dtype](mask)

    if get_background | (err_scale < 0) | (use_bkg_err):

        # Account for pixel scale in bkg_params
        bkg_input = {}
        if 'pixel_scale' in bkg_params:
            bkg_pscale = bkg_params['pixel_scale']
        else:
            bkg_pscale = pixel_scale

        for k in bkg_params:
            if k in ['pixel_scale']:
                continue

            if k in ['bw', 'bh']:
                bkg_input[k] = bkg_params[k]*bkg_pscale/pixel_scale
            else:
                bkg_input[k] = bkg_params[k]

        logstr = 'SEP: Get background {0}'.format(bkg_input)
        utils.log_comment(utils.LOGFILE, logstr, verbose=verbose,
                          show_date=True)

        if bkg_mask is not None:
            bkg = sep.Background(data, mask=mask | bkg_mask, **bkg_input)
        else:
            bkg = sep.Background(data, mask=mask, **bkg_input)

        bkg_data = bkg.back()
        if bkg_only:
            return bkg_data
        
        if get_background == 2:
            bkg_file = '{0}_bkg.fits'.format(root)
            if os.path.exists(bkg_file):
                logstr = 'SEP: use background file {0}'.format(bkg_file)
                utils.log_comment(utils.LOGFILE, logstr, verbose=verbose,
                                  show_date=True)
                
                bkg_im = pyfits.open('{0}_bkg.fits'.format(root))
                bkg_data = bkg_im[0].data*1
        else:        
            pyfits.writeto('{0}_bkg.fits'.format(root), data=bkg_data,
                    header=wcs_header, overwrite=True)

        if (err is None) | use_bkg_err:
            logstr = 'sep: Use bkg.rms() for error array'
            utils.log_comment(utils.LOGFILE, logstr, verbose=verbose,
                              show_date=True)

            err = bkg.rms()

        if err_scale == -np.inf:
            ratio = bkg.rms()/err
            err_scale = np.median(ratio[(~mask) & np.isfinite(ratio)])
        else:
            # Just return the error scale
            if err_scale < 0:
                ratio = bkg.rms()/err
                xerr_scale = np.median(ratio[(~mask) & np.isfinite(ratio)])
                del(bkg)
                return xerr_scale

        del(bkg)

    else:
        if err_scale is None:
            err_scale = 1.

    if not get_background:
        bkg_data = 0.
        data_bkg = data
    else:
        data_bkg = data - bkg_data

    if rescale_weight:
        if verbose:
            print('SEP: err_scale={:.3f}'.format(err_scale))

        err *= err_scale

    if source_xy is None:
        # Run the detection
        if verbose:
            print('   SEP: Extract...')

        objects, seg = sep.extract(data_bkg, threshold, err=err,
                       mask=mask, segmentation_map=True,
                       **detection_params)

        if verbose:
            print('    Done.')

        tab = utils.GTable(objects)
        tab.meta['VERSION'] = (sep.__version__, 'SEP version')

        # make unit-indexed like SExtractor
        tab['x_image'] = tab['x']+1
        tab['y_image'] = tab['y']+1

        # ID
        tab['number'] = np.arange(len(tab), dtype=np.int32)+1
        tab['theta'] = np.clip(tab['theta'], -np.pi/2, np.pi/2)
        for c in ['a', 'b', 'x', 'y', 'x_image', 'y_image', 'theta']:
            tab = tab[np.isfinite(tab[c])]

        # Segmentation
        seg[mask] = 0

        pyfits.writeto('{0}_seg.fits'.format(root), data=seg,
                       header=wcs_header, overwrite=True)

        # WCS coordinates
        if wcs is not None:
            tab['ra'], tab['dec'] = wcs.all_pix2world(tab['x'], tab['y'], 0)
            tab['ra'].unit = u.deg
            tab['dec'].unit = u.deg
            tab['x_world'], tab['y_world'] = tab['ra'], tab['dec']

        if 'minarea' in detection_params:
            tab.meta['MINAREA'] = (detection_params['minarea'],
                                   'Minimum source area in pixels')
        else:
            tab.meta['MINAREA'] = (5, 'Minimum source area in pixels')

        if 'clean' in detection_params:
            tab.meta['CLEAN'] = (detection_params['clean'],
                                 'Detection cleaning')
        else:
            tab.meta['CLEAN'] = (True, 'Detection cleaning')

        if 'deblend_cont' in detection_params:
            tab.meta['DEBCONT'] = (detection_params['deblend_cont'],
                                 'Deblending contrast ratio')
        else:
            tab.meta['DEBCONT'] = (0.005, 'Deblending contrast ratio')

        if 'deblend_nthresh' in detection_params:
            tab.meta['DEBTHRSH'] = (detection_params['deblend_nthresh'],
                                 'Number of deblending thresholds')
        else:
            tab.meta['DEBTHRSH'] = (32, 'Number of deblending thresholds')

        if 'filter_type' in detection_params:
            tab.meta['FILTER_TYPE'] = (detection_params['filter_type'],
                                     'Type of filter applied, conv or weight')
        else:
            tab.meta['FILTER_TYPE'] = ('conv',
                                     'Type of filter applied, conv or weight')

        tab.meta['THRESHOLD'] = (threshold, 'Detection threshold')

        # ISO fluxes (flux within segments)
        iso_flux, iso_fluxerr, iso_area = get_seg_iso_flux(data_bkg, seg, tab,
                                                 err=err, verbose=1)

        tab['flux_iso'] = iso_flux/uJy_to_dn*u.uJy
        tab['fluxerr_iso'] = iso_fluxerr/uJy_to_dn*u.uJy
        tab['area_iso'] = iso_area
        tab['mag_iso'] = 23.9-2.5*np.log10(tab['flux_iso'])

        # Compute FLUX_AUTO, FLUX_RADIUS
        if compute_auto_quantities:
            auto = compute_SEP_auto_params(data, data_bkg, mask,
                                pixel_scale=pixel_scale,
                                err=err, segmap=seg, tab=tab,
                                autoparams=autoparams, flux_radii=flux_radii,
                                subpix=subpix, verbose=verbose)

            for k in auto.meta:
                tab.meta[k] = auto.meta[k]

            auto_flux_cols = ['flux_auto', 'fluxerr_auto', 'bkg_auto']
            for c in auto.colnames:
                if c in auto_flux_cols:
                    tab[c] = auto[c]/uJy_to_dn*u.uJy
                else:
                    tab[c] = auto[c]

            # Problematic sources
            # bad = (tab['flux_auto'] <= 0) | (tab['flux_radius'] <= 0)
            # bad |= (tab['kron_radius'] <= 0)
            # tab = tab[~bad]

            # Correction for flux outside Kron aperture
            tot_corr = get_kron_tot_corr(tab, drz_filter, 
                                                pixel_scale=pixel_scale, 
                                                photplam=drz_photplam)

            tab['tot_corr'] = tot_corr
            tab.meta['TOTCFILT'] = (drz_filter, 'Filter for tot_corr')
            tab.meta['TOTCWAVE'] = (drz_photplam, 'PLAM for tot_corr')

            total_flux = tab['flux_auto']*tot_corr
            tab['mag_auto'] = 23.9-2.5*np.log10(total_flux)
            tab['magerr_auto'] = 2.5/np.log(10)*(tab['fluxerr_auto']/tab['flux_auto'])

        # More flux columns
        for c in ['cflux', 'flux', 'peak', 'cpeak']:
            tab[c] *= 1. / uJy_to_dn
            tab[c].unit = u.uJy

        source_x, source_y = tab['x'], tab['y']

        # Use segmentation image to mask aperture fluxes
        if aper_segmask:
            aseg = seg
            aseg_id = tab['number']
        else:
            aseg = aseg_id = None

        # Rename some columns to look like SExtractor
        for c in ['a', 'b', 'theta', 'cxx', 'cxy', 'cyy', 'x2', 'y2', 'xy']:
            tab.rename_column(c, c+'_image')

    else:
        if len(source_xy) == 2:
            source_x, source_y = source_xy
            aseg, aseg_id = None, None
            aper_segmask = False
        else:
            source_x, source_y, aseg, aseg_id = source_xy
            aper_segmask = True

        if hasattr(source_x, 'unit'):
            if source_x.unit == u.deg:
                # Input positions are ra/dec, convert with WCS
                ra, dec = source_x, source_y
                source_x, source_y = wcs.all_world2pix(ra, dec, 0)

        tab = utils.GTable()
        tab.meta['VERSION'] = (sep.__version__, 'SEP version')

    # Info
    tab.meta['ZP'] = (ZP, 'AB zeropoint')
    if 'PHOTPLAM' in im[0].header:
        tab.meta['PLAM'] = (im[0].header['PHOTPLAM'], 'Filter pivot wave')
        if 'PHOTFNU' in im[0].header:
            tab.meta['FNU'] = (im[0].header['PHOTFNU'], 'Scale to Jy')

        tab.meta['FLAM'] = (im[0].header['PHOTFLAM'], 'Scale to flam')

    tab.meta['uJy2dn'] = (uJy_to_dn, 'Convert uJy fluxes to image DN')

    tab.meta['DRZ_FILE'] = (drz_file[:36], 'SCI file')
    tab.meta['WHT_FILE'] = (weight_file[:36], 'WHT file')

    tab.meta['GET_BACK'] = (get_background, 'Background computed')
    for k in bkg_params:
        tab.meta[f'BACK_{k.upper()}'] = (bkg_params[k],
                                         f'Background param {k}')
        
    tab.meta['ERR_SCALE'] = (err_scale, 'Scale factor applied to weight image (like MAP_WEIGHT)')
    tab.meta['RESCALEW'] = (rescale_weight, 'Was the weight applied?')

    tab.meta['APERMASK'] = (aper_segmask, 'Mask apertures with seg image')

    # Photometry
    for iap, aper in enumerate(apertures):
        if sep.__version__ > '1.03':
            # Should work with the sep fork at gbrammer/sep and latest sep
            flux, fluxerr, flag = sep.sum_circle(data_bkg,
                                                 source_x, source_y,
                                                 aper/2, err=err,
                                                 gain=gain, subpix=subpix,
                                                 segmap=aseg, seg_id=aseg_id,
                                                 mask=mask)
        else:
            tab.meta['APERMASK'] = (False, 'Mask apertures with seg image - Failed')
            flux, fluxerr, flag = sep.sum_circle(data_bkg,
                                                 source_x, source_y,
                                                 aper/2, err=err,
                                                 gain=gain, subpix=subpix,
                                                 mask=mask)

        tab.meta['GAIN'] = gain

        tab['flux_aper_{0}'.format(iap)] = flux/uJy_to_dn*u.uJy
        tab['fluxerr_aper_{0}'.format(iap)] = fluxerr/uJy_to_dn*u.uJy
        tab['flag_aper_{0}'.format(iap)] = flag

        if get_background:
            try:
                flux, fluxerr, flag = sep.sum_circle(bkg_data,
                                          source_x, source_y,
                                          aper/2, err=None, gain=1.0,
                                          segmap=aseg, seg_id=aseg_id,
                                          mask=mask)
            except:
                flux, fluxerr, flag = sep.sum_circle(bkg_data,
                                          source_x, source_y,
                                          aper/2, err=None, gain=1.0,
                                          mask=mask)

            tab['bkg_aper_{0}'.format(iap)] = flux/uJy_to_dn*u.uJy
        else:
            tab['bkg_aper_{0}'.format(iap)] = 0.*u.uJy

        # Count masked pixels in the aperture, not including segmask
        flux, fluxerr, flag = sep.sum_circle(data_mask,
                                      source_x, source_y,
                                      aper/2, err=err,
                                      gain=gain, subpix=subpix)

        tab['mask_aper_{0}'.format(iap)] = flux

        tab.meta['aper_{0}'.format(iap)] = (aper, 'Aperture diameter, pix')
        tab.meta['asec_{0}'.format(iap)] = (aper*pixel_scale,
                                            'Aperture diameter, arcsec')

    # # If blended, use largest aperture magnitude
    # if 'flag' in tab.colnames:
    #     last_flux = tab['flux_aper_{0}'.format(iap)]
    #     last_fluxerr = tab['fluxerr_aper_{0}'.format(iap)]
    #
    #     blended = (tab['flag'] & 1) > 0
    #
    #     total_corr = tab['flux_auto']/last_flux
    #     blended |= total_corr > max_total_corr
    #
    #     tab['flag'][blended] |= 1024
    #
    #     aper_mag = 23.9 - 2.5*np.log10(last_flux)
    #     aper_magerr = 2.5/np.log(10)*last_fluxerr/last_flux
    #
    #     tab['mag_auto'][blended] = aper_mag[blended]
    #     tab['magerr_auto'][blended] = aper_magerr[blended]
    #
    #     # "ISO" mag, integrated within the segment
    #     tab['mag_iso'] = 23.9-2.5*np.log10(tab['flux'])

    try:
        # Free memory objects explicitly
        del(data_mask)
        del(data)
        del(err)
    except:
        pass

    # if uppercase_columns:
    for c in tab.colnames:
        tab.rename_column(c, column_case(c))

    if save_to_fits:
        tab.write(f'{root}.cat.fits', format='fits', overwrite=True)

        if include_wcs_extension:
            try:
                hdul = pyfits.open(f'{root}.cat.fits', mode='update')
                wcs_hdu = pyfits.ImageHDU(header=wcs_header, data=None,
                                          name='WCS')
                hdul.append(wcs_hdu)
                hdul.flush()
            except:
                pass

    logstr = '# SEP {0}.cat.fits: {1:d} objects'.format(root, len(tab))
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    return tab


def get_seg_iso_flux(data, seg, tab, err=None, fill=None, verbose=0):
    """
    Integrate flux within the segmentation regions.
    
    Parameters
    ----------
    data : 2D array
        Image data
    
    seg : 2D array
        Segmentation image defining the ISO contours
    
    tab : `~astropy.table.Table`
        Detection catalog with columns (at least) ``number / id``,  
        ``xmin``, ``xmax``, ``ymin``, ``ymax``.  The ``id`` column matches the
        values in `seg`.
    
    err : 2D array
        Optional uncertainty array
    
    fill : None, array
        If specified, create an image where the image segments are filled
        with scalar values for a given object rather than computing the ISO 
        fluxes
    
    verbose : bool/int
        Status messages
    
    Returns
    -------
    iso_flux : array
        Summed image flux within the contours defined by the `seg` map
    
    iso_err : array
        Uncertainty if `err` specified
    
    iso_area : array
        Area of the segments, in pixels
    
    filled_data : 2D array
        If `fill` specified, return an image with values filled within the 
        segments, e.g., for a binned image
        
    """
    if 'number' in tab.colnames:
        ids = np.array(tab['number'])
    else:
        ids = np.array(tab['id'])

    sh = data.shape

    iso_flux = ids*0.
    iso_err = ids*0.
    iso_area = np.cast[int](ids*0)

    xmin = np.clip(tab['xmin'], 0, sh[1])
    xmax = np.clip(tab['xmax'], 0, sh[1])
    ymin = np.clip(tab['ymin'], 0, sh[0])
    ymax = np.clip(tab['ymax'], 0, sh[0])

    if fill is not None:
        filled_data = np.cast[fill.dtype](seg*0)

    for ii, id in enumerate(ids):

        if (verbose > 1):
            if (ii % verbose == 0):
                print(' {0}'.format(ii))

        slx = slice(xmin[ii], xmax[ii])
        sly = slice(ymin[ii], ymax[ii])

        seg_sub = seg[sly, slx]
        seg_mask = (seg_sub == id)

        if fill is not None:
            #print(ii, seg_mask.sum())
            filled_data[sly, slx][seg_mask] = fill[ii]

        else:

            data_sub = data[sly, slx]
            iso_flux[ii] = data_sub[seg_mask].sum()
            iso_area[ii] = seg_mask.sum()

            if err is not None:
                err_sub = err[sly, slx]
                iso_err[ii] = np.sqrt((err_sub[seg_mask]**2).sum())

    if fill is not None:
        return filled_data
    else:
        return iso_flux, iso_err, iso_area


def compute_SEP_auto_params(data, data_bkg, mask, pixel_scale=0.06, err=None, segmap=None, tab=None, grow_kron=6.0, autoparams=[2.5, 0.35*u.arcsec, 2.4, 3.8], flux_radii=[0.2, 0.5, 0.9], subpix=0, verbose=True):
    """Compute SourceExtractor-like AUTO params with `sep`
    https://sep.readthedocs.io/en/v1.0.x/apertures.html#equivalent-of-flux-auto-e-g-mag-auto-in-source-extractor
    
    Parameters
    ----------
    data : 2D array
        Image data
    
    data_bkg : 2D array
        Background-subtracted image data
    
    mask : 2D array
        Pixel mask 
    
    pixel_scale : float
        Pixel scale, arcsec
    
    err : 2D array
        Uncertainty array
    
    segmap : 2D array
        Associated segmentation map
    
    tab : `~astropy.table.Table`
        Table from, e.g., `sep.extract`.
    
    grow_kron : float
        Scale by which semimajor and semiminor axes are multiplied for 
        calculating the Kron moment.  This is hard-coded as `grow_kron=6.0`
        in `SourceExtractor <https://sextractor.readthedocs.io/en/latest/Photom.html>`_.
        
    autoparams : list
        Provided as ``[k, MIN_APER, MIN_KRON, MAX_KRON]``, where the usual
        SourceExtractor ``PHOT_AUTOPARAMS`` would be ``[k, MIN_KRON]``. Here,
        ``k`` is the scale factor of the Kron radius, and ``MIN_KRON`` is the
        minimum scaled Kron radius to use. ``MIN_APER`` is then the smallest
        *circular* Kron aperture to allow, which can be provided with attached
        units (e.g. ``arcsec``).
    
    flux_radii : list
        Light fraction radii, e.g., ``[0.5]`` will calculate the half-light 
        radius (``FLUX_RADIUS``)
        
    subpix : int
        Pixel oversampling with the `sep` aperture functions
    
    Returns
    -------
    auto : `~astropy.table.Table`
        Table with the derived parameters
                
        +--------------------------+----------------------------------------+
        | Column                   | Description                            |
        +==========================+========================================+
        | `kron_radius`            | Kron radius, pixels                    |
        +--------------------------+----------------------------------------+
        | `kron_rcirc`             | Circularized Kron radius, pixels       |
        +--------------------------+----------------------------------------+
        | `flux_auto`              | Flux within AUTO aperture              |
        +--------------------------+----------------------------------------+
        | `fluxerr_auto`           | Uncertainty within AUTO aperture       |
        +--------------------------+----------------------------------------+
        | `bkg_auto`               | Background within AUTO aperture        |
        +--------------------------+----------------------------------------+
        | `flag_auto`              | `sep` flags for AUTO aperture          |
        +--------------------------+----------------------------------------+
        | `area_auto`              | Pixel area of AUTO aperture            |
        +--------------------------+----------------------------------------+
        
    """
    import sep
    logstr = 'compute_SEP_auto_params: sep version = {0}'.format(sep.__version__)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)
    if sep.__version__ < '1.1':
        print("""!!!!!!!!!!
!  SEP version = {0}
!  Get >= 1.10.0 to enable segmentation masking, e.g.,
!  $ pip install git+https://github.com/gbrammer/sep.git
!!!!!!!!!!
        """.format(sep.__version__))

    logstr = 'compute_SEP_auto_params: autoparams={0}; pixel_scale={1}; subpix={2}; flux_radii={3}'.format(autoparams, pixel_scale, subpix, flux_radii)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    # Check datatype of seg
    segb = segmap
    if segmap is not None:
        if segmap.dtype == np.dtype('>i4'):
            segb = segmap.byteswap().newbyteorder()

    if 'a_image' in tab.colnames:
        x, y = tab['x_image']-1, tab['y_image']-1
        a, b = tab['a_image'], tab['b_image']
        theta = tab['theta_image']
    else:
        x, y, a, b = tab['x'], tab['y'], tab['a'], tab['b']
        theta = tab['theta']

    if 'number' in tab.colnames:
        seg_id = tab['number']
    else:
        seg_id = tab['id']

    # Kron radius
    try:
        # Try with seg mask (sep > v1.0.4)
        kronrad, krflag = sep.kron_radius(data_bkg, x, y, a, b, theta,
                                          grow_kron, mask=mask,
                                          segmap=segb, seg_id=seg_id)
        kronrad[~np.isfinite(kronrad)] = 0
    except:
        logstr = 'sep.kron_radius: ! Warning ! couldn\'t run with seg mask'
        utils.log_comment(utils.LOGFILE, logstr, verbose=True)

        kronrad, krflag = sep.kron_radius(data_bkg, x, y, a, b, theta,
                                          grow_kron, mask=mask)

    # This is like SExtractor PHOT_AUTOPARAMS[0]
    kronrad *= autoparams[0]

    # This is like SExtractor PHOT_AUTOPARAMS[1]
    if len(autoparams) > 2:
        clip_kron0 = autoparams[2]
        kronrad = np.maximum(kronrad, clip_kron0)
    else:
        clip_kron0 = 0.
    
    if len(autoparams) > 3:
        clip_kron1 = autoparams[3]
        kronrad = np.minimum(kronrad, clip_kron1)
    else:
        clip_kron1 = 1000.
    
    # Circularized Kron radius (equivalent to kronrad * a * sqrt(b/a))
    kroncirc = kronrad * np.sqrt(a*b)

    # Minimum Kron radius
    if hasattr(autoparams[1], 'unit'):
        min_radius_pix = autoparams[1].to(u.arcsec).value/pixel_scale
    else:
        # Assume arcsec
        min_radius_pix = autoparams[1]/pixel_scale

    kron_min = kroncirc <= min_radius_pix

    kron_flux = x*0.
    kron_bkg = x*0.
    kron_fluxerr = x*0.
    kron_flag = np.zeros(len(x), dtype=int)
    kron_area = np.pi*np.maximum(kroncirc, min_radius_pix)**2

    ########
    # Ellipse photometry in apertures larger than the minimum
    # Extract on both data and background subtracted to compute the
    # background within the aperture
    try:
        # Try with seg mask (sep=>v1.0.4)
        kout0 = sep.sum_ellipse(data, x[~kron_min], y[~kron_min],
                               a[~kron_min], b[~kron_min], theta[~kron_min],
                               kronrad[~kron_min], subpix=subpix, err=None,
                               segmap=segb, seg_id=seg_id[~kron_min],
                               mask=mask)

        kout = sep.sum_ellipse(data_bkg, x[~kron_min], y[~kron_min],
                               a[~kron_min], b[~kron_min], theta[~kron_min],
                               kronrad[~kron_min], subpix=subpix, err=err,
                               segmap=segb, seg_id=seg_id[~kron_min], mask=mask)
    except:
        kout0 = sep.sum_ellipse(data_bkg, x[~kron_min], y[~kron_min],
                               a[~kron_min], b[~kron_min], theta[~kron_min],
                               kronrad[~kron_min], subpix=subpix, err=None,
                               mask=mask)

        kout = sep.sum_ellipse(data_bkg, x[~kron_min], y[~kron_min],
                               a[~kron_min], b[~kron_min], theta[~kron_min],
                               kronrad[~kron_min], subpix=subpix, err=err,
                               mask=mask)

    kron_flux[~kron_min] = kout[0]
    kron_bkg[~kron_min] = kout0[0]-kout[0]
    kron_fluxerr[~kron_min] = kout[1]
    kron_flag[~kron_min] = kout[2]

    # Circular apertures below minimum size
    try:
        # Try with seg mask (sep v1.0.4)
        kout0 = sep.sum_circle(data, x[kron_min], y[kron_min],
                              min_radius_pix, subpix=subpix, err=None,
                              segmap=segb, seg_id=seg_id[kron_min], mask=mask)

        kout = sep.sum_circle(data_bkg, x[kron_min], y[kron_min],
                              min_radius_pix, subpix=subpix, err=err,
                              segmap=segb, seg_id=seg_id[kron_min], mask=mask)
    except:
        kout0 = sep.sum_circle(data, x[kron_min], y[kron_min],
                              min_radius_pix, subpix=subpix, err=None,
                              mask=mask)

        kout = sep.sum_circle(data_bkg, x[kron_min], y[kron_min],
                              min_radius_pix, subpix=subpix, err=err,
                              mask=mask)

    kron_flux[kron_min] = kout[0]
    kron_bkg[kron_min] = kout0[0]-kout[0]
    kron_fluxerr[kron_min] = kout[1]
    kron_flag[kron_min] = kout[2]

    #############
    # Flux radius
    try:
        fr, fr_flag = sep.flux_radius(data_bkg, x, y, a*grow_kron, flux_radii,
                                  normflux=kron_flux, mask=mask,
                                  segmap=segb, seg_id=seg_id)
    except:
        fr, fr_flag = sep.flux_radius(data_bkg, x, y, a*grow_kron, flux_radii,
                                  normflux=kron_flux, mask=mask)

    auto = utils.GTable()
    auto.meta['KRONFACT'] = (autoparams[0], 'Kron radius scale factor')
    auto.meta['KRON0'] = (clip_kron0, 'Minimum scaled Kron radius')
    auto.meta['KRON1'] = (clip_kron1, 'Maximum scaled Kron radius')
    auto.meta['MINKRON'] = (min_radius_pix, 'Minimum Kron aperture, pix')

    auto['kron_radius'] = kronrad*u.pixel
    auto['kron_rcirc'] = kroncirc*u.pixel

    auto['flux_auto'] = kron_flux
    auto['fluxerr_auto'] = kron_fluxerr
    auto['bkg_auto'] = kron_bkg
    auto['flag_auto'] = kron_flag
    auto['area_auto'] = kron_area

    auto['flux_radius_flag'] = fr_flag
    for i, r_i in enumerate(flux_radii):
        c = 'flux_radius_{0:02d}'.format(int(np.round(r_i*100)))
        if c.endswith('_50'):
            c = 'flux_radius'

        auto[c] = fr[:, i]

    return auto


def get_filter_ee_ratio(tab, filter, ref_filter='f160w'):
    """
    Relative EE correction within a specified aperture, in arcsec.  
    """
    pixel_scale = tab.meta['ASEC_0']/tab.meta['APER_0']
    min_kron = tab.meta['MINKRON']*pixel_scale
    
    ee = utils.read_catalog((os.path.join(os.path.dirname(utils.__file__),
                             'data', 'hst_encircled_energy.fits')))

    # Reference filter
    ref_obsmode = utils.get_filter_obsmode(filter=ref_filter, acs_chip='wfc1', 
                                       uvis_chip='uvis2', aper=np.inf, 
                                       case=str.lower)

    # Target filter
    obsmode = utils.get_filter_obsmode(filter=filter, acs_chip='wfc1', 
                                       uvis_chip='uvis2', aper=np.inf, 
                                       case=str.lower)

    # Filter not available
    if obsmode not in ee.colnames:
        return 1.

    # Ratio of photometric aperture to kron aperture
    keys = list(tab.meta.keys())
    for ka in keys:
        if ka.startswith('APER_'):
            ap = ka.split('_')[1]

            aper_radius = tab.meta[f'ASEC_{ap}']/2.
            kron_circ = np.maximum(tab['kron_rcirc']*pixel_scale, min_kron)
            
            filt_kron = np.interp(kron_circ, ee['radius'], ee[obsmode])
            filt_aper = np.interp(aper_radius, ee['radius'], ee[obsmode])
            
            ref_kron = np.interp(kron_circ, ee['radius'], ee[ref_obsmode])
            ref_aper = np.interp(aper_radius, ee['radius'], ee[ref_obsmode])
            
            filter_correction = (filt_kron/filt_aper) / (ref_kron/ref_aper)
            tab[f'{filter}_ee_{ap}'] = filter_correction
    
    return tab


def get_hst_aperture_correction(filter, raper=0.35, rmax=5.):
    """
    Get single aperture correction from tabulated EE curve
    """
    ee = utils.read_catalog((os.path.join(os.path.dirname(__file__),
                             'data', 'hst_encircled_energy.fits')))
    
    obsmode = utils.get_filter_obsmode(filter=filter, acs_chip='wfc1', 
                                       uvis_chip='uvis2', aper=np.inf, 
                                       case=str.lower)
    
    
    ee_rad = np.append(ee['radius'], rmax)
    ee_y = np.append(ee[obsmode], 1.)
    ee_interp = np.interp(raper, ee_rad, ee_y, left=0.01, right=1.)
    return ee.meta['ZP_'+obsmode], ee_interp


def get_kron_tot_corr(tab, filter, inst=None, pixel_scale=0.06, photplam=None, rmax=5.0):
    """
    Compute total correction from tabulated EE curves
    """
    ee = utils.read_catalog((os.path.join(os.path.dirname(__file__),
                             'data', 'hst_encircled_energy.fits')))
    
    obsmode = utils.get_filter_obsmode(filter=filter, acs_chip='wfc1', 
                                       uvis_chip='uvis2', aper=np.inf, 
                                       case=str.lower)

    min_kron = float(np.atleast_1d(tab.meta['MINKRON'])[0])
    
    if pixel_scale is None:
        try:
            pixel_scale = tab.meta['ASEC_0']/tab.meta['APER_0']
        except:
            pixel_scale = tab.meta['asec_0'][0]/tab.meta['aper_0'][0]
        
    if 'kron_rcirc' in tab.colnames:
        kron_raper = np.clip(tab['kron_rcirc']*pixel_scale,
                             min_kron*pixel_scale, rmax)
    else:
        kron_raper = np.clip(tab['KRON_RCIRC']*pixel_scale,
                             min_kron*pixel_scale, rmax)

    # Filter not available
    if obsmode not in ee.colnames:
        return kron_raper*0.+1
    else:
        ee_rad = np.append(ee['radius'], rmax)
        ee_y = np.append(ee[obsmode], 1.)
        
        ee_interp = np.interp(kron_raper, ee_rad, ee_y, left=0.01, right=1.)
        return 1./ee_interp


def get_wfc3ir_kron_tot_corr(tab, filter, pixel_scale=0.06, photplam=None, rmax=5.):
    """
    Compute total correction from WFC3/IR EE curves 
    
    .. note::
        
        Deprecated, use `~grizli.utils.get_kron_tot_corr`.
        
    """
    ee_raw = np.loadtxt((os.path.join(os.path.dirname(__file__),
                            'data', 'wfc3ir_ee.txt')))
    ee_data = ee_raw[1:, 1:]
    ee_wave = ee_raw[0, 1:]
    ee_rad = ee_raw[1:, 0]

    kron_raper = np.clip(tab['kron_rcirc']*pixel_scale,
                         tab.meta['MINKRON'][0]*pixel_scale, rmax)

    if (filter.lower()[:2] not in ['f0', 'f1']) & (photplam is None):
        return kron_raper*0.+1

    if photplam is None:
        wum = int(filter[1:-1])*100/1.e4
    else:
        wum = photplam/1.e4
        if wum < 0.9:
            return kron_raper*0.+1

    xi = np.interp(wum, ee_wave, np.arange(len(ee_wave)))
    i0 = int(xi)
    fi = 1-(xi-i0)
    ee_y = ee_data[:, i0:i0+2].dot([fi, 1-fi])
    ee_rad = np.append(ee_rad, rmax)
    ee_y = np.append(ee_y, 1.)

    ee_interp = np.interp(kron_raper, ee_rad, ee_y, left=0.01, right=1.)
    return 1./ee_interp


def make_drz_catalog(root='', sexpath='sex', threshold=2., get_background=True,
                     verbose=True, extra_config={}, sci=None, wht=None,
                     get_sew=False, output_params=SEXTRACTOR_DEFAULT_PARAMS,
                     phot_apertures=SEXTRACTOR_PHOT_APERTURES,
                     column_case=str.upper):
    """Make a SExtractor catalog from drizzle products

    .. note::
        
        Deprecated, use `~grizli.utils.make_SEP_catalog`.
    """
    import copy
    import sewpy

    if sci is not None:
        drz_file = sci
    else:
        drz_file = glob.glob('{0}_dr[zc]_sci.fits'.format(root))[0]

    im = pyfits.open(drz_file)

    ZP = calc_header_zeropoint(im[0].header)
    
    # if 'PHOTFNU' in im[0].header:
    #     ZP = -2.5*np.log10(im[0].header['PHOTFNU'])+8.90
    # elif 'PHOTFLAM' in im[0].header:
    #     ZP = (-2.5*np.log10(im[0].header['PHOTFLAM']) - 21.10 -
    #              5*np.log10(im[0].header['PHOTPLAM']) + 18.6921)
    # elif 'FILTER' in im[0].header:
    #     fi = im[0].header['FILTER'].upper()
    #     if fi in model.photflam_list:
    #         ZP = (-2.5*np.log10(model.photflam_list[fi]) - 21.10 -
    #                  5*np.log10(model.photplam_list[fi]) + 18.6921)
    #     else:
    #         print('Couldn\'t find PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25')
    #         ZP = 25
    # else:
    #     print('Couldn\'t find FILTER, PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25')
    #     ZP = 25

    if verbose:
        print('Image AB zeropoint: {0:.3f}'.format(ZP))

    weight_file = drz_file.replace('_sci.fits', '_wht.fits').replace('_drz.fits', '_wht.fits')
    if (weight_file == drz_file) | (not os.path.exists(weight_file)):
        WEIGHT_TYPE = "NONE"
    else:
        WEIGHT_TYPE = "MAP_WEIGHT"

    if wht is not None:
        weight_file = wht

    config = OrderedDict(DETECT_THRESH=threshold, ANALYSIS_THRESH=threshold,
              DETECT_MINAREA=6,
              PHOT_FLUXFRAC="0.5",
              WEIGHT_TYPE=WEIGHT_TYPE,
              WEIGHT_IMAGE=weight_file,
              CHECKIMAGE_TYPE="SEGMENTATION",
              CHECKIMAGE_NAME='{0}_seg.fits'.format(root),
              MAG_ZEROPOINT=ZP,
              CLEAN="N",
              PHOT_APERTURES=phot_apertures,
              BACK_SIZE=32,
              PIXEL_SCALE=0,
              MEMORY_OBJSTACK=30000,
              MEMORY_PIXSTACK=3000000,
              MEMORY_BUFSIZE=8192)

    if get_background:
        config['CHECKIMAGE_TYPE'] = 'SEGMENTATION,BACKGROUND'
        config['CHECKIMAGE_NAME'] = '{0}_seg.fits,{0}_bkg.fits'.format(root)
    else:
        config['BACK_TYPE'] = 'MANUAL'
        config['BACK_VALUE'] = 0.

    for key in extra_config:
        config[key] = extra_config[key]

    params = copy.copy(output_params)
    NAPER = len(phot_apertures.split(','))
    if NAPER == 1:
        if not phot_apertures.split(',')[0]:
            NAPER = 0

    if NAPER > 0:
        params.extend(['FLUX_APER({0})'.format(NAPER),
                       'FLUXERR_APER({0})'.format(NAPER)])
        # if NAPER > 1:
        #     for i in range(NAPER-1):
        #         params.extend(['FLUX_APER{0}'.format(i+1),
        #                        'FLUXERR_APER{0}'.format(i+1)])

    sew = sewpy.SEW(params=params, config=config)

    if get_sew:
        return sew

    output = sew(drz_file)
    cat = output['table']
    cat.meta = config

    for c in cat.colnames:
        cat.rename_column(c, column_case(c))

    cat.write('{0}.cat'.format(root), format='ascii.commented_header',
              overwrite=True)

    logstr = '# DRZ {0} catalog: {1:d} objects'.format(root, len(cat))
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    return cat


def make_visit_average_flat(visit, clip_max=0, threshold=5, dilate=3, apply=True, instruments=['MIRI'], verbose=True, bad_bits=(1024+1+2+512), bkg_kwargs={}):
    """
    Make an average "skyflat" for exposures in a visit, e.g., for MIRI
    
    Parameters
    ----------
    visit : dict
        Visit dictionary with, at a minimum, keys of ``product`` and ``files``
    
    clip_max : int
        Number of max clipping iterations
    
    threshold : float
        If specified, run `sep` source detection on each exposure
    
    dilate : int
        Number of `scipy.ndimage.binary_dilation` iterations to run on the 
        segmentation mask
    
    apply : bool
        Apply the flat to the exposures 
    
    instruments : list
        Only run for these instruments (from INSTRUME header keyword)
        
    verbose : bool
        Print status messages
        
    """
    from tqdm import tqdm
    import scipy.ndimage as nd
    import sep
    
    _im = pyfits.open(visit['files'][0])
    
    if 'OINSTRUM' in _im[0].header:
        instrument = _im[0].header['OINSTRUM']
    else:
        instrument = _im[0].header['INSTRUME']
        
    if instrument not in instruments:
        msg = f"make_visit_average_flat: Instrument for {visit['product']} "
        msg += f"is {instrument}, skip flat"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        
        return None
        
    sci = np.array([pyfits.open(file)['SCI'].data.astype(np.float32)
                    for file in visit['files']])
    dq = np.array([pyfits.open(file)['DQ'].data
                    for file in visit['files']])

    if threshold is not None:
        
        msg = f"make_visit_average_flat: Source detection on individual"
        msg += f" exposures with threshold={threshold} dilate={dilate}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        
        err = np.array([pyfits.open(file)['ERR'].data.astype(np.float32)
                        for file in visit['files']])
        
        #mask = im['DQ'].data > 0
        for i in tqdm(range(len(visit['files']))):
            _sci = sci[i,:,:]
            _err = err[i,:,:]
            _mask = (dq[i,:,:] & bad_bits) > 0
            _bkg = sep.Background(_sci, mask=_mask, **bkg_kwargs)
            _cat, _seg = sep.extract(_sci - _bkg.back(), threshold,
                                    err=_err, 
                                    mask=_mask,
                                    segmentation_map=True)

            _segm = nd.binary_dilation(_seg > 0, iterations=dilate)
            _sci[_segm > 0] = np.nan
    
    scim = sci*1.
    for _iter in range(clip_max):
        ma = np.nanmax(scim, axis=0)
        scim[scim >= ma] = np.nan
    
    bkg = np.array([np.nanmedian(s[q == 0])
                    for s, q in zip(scim, dq)])
    
    N = sci.shape[0]
    scif = scim*1.
    for i in range(N):
        scif[i] /= bkg[i]
        
    flat = np.nanmedian(scif, axis=0)
    flat_mask = ~np.isfinite(flat) | (flat <= 0.05)
    flat[flat_mask] = 1
    
    flath = pyfits.Header()
    for k in ['CRDS_CTX','UPD_CTX', 'R_FLAT', 'R_DARK',
              'FILTER', 'DETECTOR','EXPSTART','EFFEXPTM']:
        if k in _im[0].header:
            flath[k] = _im[0].header[k]
            
    flath['CLIP_MAX'] = clip_max, 'Max clip iterations'
    flath['NFILES'] = len(visit['files'])
    for i, file in enumerate(visit['files']):
        flath[f'FILE{i+1:04d}'] = file
    
    _hdu = pyfits.PrimaryHDU(data=flat, header=flath)
    skyfile = f"{visit['product']}_skyflat.fits"
    _hdu.writeto(skyfile, overwrite=True)
    
    msg = f"make_visit_average_flat: {skyfile} N={len(visit['files'])}"
    msg += f" clip_max={clip_max}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    
    if apply:
        apply_visit_skyflat(visit, skyfile=skyfile)


def apply_visit_skyflat(visit, skyfile=None, verbose=True):
    """
    Apply the skyflat
    """
    
    if not os.path.exists(skyfile):
        msg = f"apply_visit_skyflat: MIRI skyflat {skyfile} not found"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None
    
    flat = pyfits.open(skyfile)[0].data
    flat_mask = (flat == 1) | (~np.isfinite(flat))
    flat[flat_mask] = 1
    
    for file in visit['files']:
        msg = f"apply_visit_skyflat: apply {skyfile} to {file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        
        with pyfits.open(file, mode='update') as _imi:
            _imi['SCI'].data /= flat
            _imi['ERR'].data /= flat
            _imi['DQ'].data[flat_mask] |= 1024
            _imi[0].header['SKYFLAT'] = True, 'Sky flat applied'
            _imi[0].header['SKYFLATF'] = skyfile, 'Sky flat file'
            _imi.flush()


def get_nircam_wisp_filename(header):
    """
    Get the path to an appropriate NIRCam wisp template file, assuming the 
    templates are stored in `[GRIZLI_PATH]/CONF/NircamWisp`
    
    Parameters
    ----------
    header : `astropy.io.fits.Header`
        Primary header
    
    Returns
    -------
    wisp_file : str
        Filename of the wisp reference, or `None` if the header isn't for a 
        NIRCam exposure in F150W or F200W
    
    _filt : str
        FILTER from header
    
    _inst : str
        INSTRUME from header
    
    _det : str
        DETECTOR from header
    
    msg : str
        Status message
        
    """
    if 'OFILTER' in header:
        _filt = header['OFILTER']
        _inst = header['OINSTRUM']
        _det = header['ODETECTO']
    else:
        _filt = header['FILTER']
        _inst = header['INSTRUME']
        _det = header['DETECTOR']

    if _inst not in ['NIRCAM']:
        msg = f'Instrument {_inst} not supported'
        return None, _filt, _inst, _det, msg

    if _filt not in ['F150W','F200W','F182M','F210M']:
        msg = f'NIRCam filter {_filt} not supported'
        return None, _filt, _inst, _det, msg

    if _det not in ['NRCA3','NRCA4','NRCB3','NRCB4']:
        msg = f'NIRCam detector {_det} not supported'
        return None, _filt, _inst, _det, msg
    
    msg = 'OK'
    
    _path = os.path.join(GRIZLI_PATH, 'CONF', 'NircamWisp')
    wisp_file = f"wisps_{_det.lower()}_{_filt.upper()}.fits"
    wisp_file = os.path.join(_path, wisp_file)
    
    return wisp_file, _filt, _inst, _det, msg


def nircam_wisp_correction(calibrated_file, niter=3, update=True, verbose=True, force=False, **kwargs):
    """
    Fit the NIRCam wisp template and subtract
    
    See https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-features-and-caveats/nircam-claws-and-wisps.
    
    Wisp template files at https://stsci.app.box.com/s/1bymvf1lkrqbdn9rnkluzqk30e8o2bne?sortColumn=name&sortDirection=ASC.
    
    and 
    
    https://s3.amazonaws.com/grizli-v2/NircamWisp/index.html
    
    The ``wisps_{detector}_{FILTER}.fits`` files should be put in
    a directory ``$GRIZLI/CONF/NircamWisp``.
    
    Parameters
    ----------
    calibrated_file : str
        Filename of a flat-fielded file (either `cal` or a processed `rate` file)
    
    niter : int
        Number of mask iterations
    
    update : bool
        Subtract the wisp model from the 'SCI' extension of ``calibrated_file``
        
    verbose : bool
        verbosity
      
    force : bool
        Force wisp fit, even if 'WISPBKG' keyword already found in header
        
    """
    import scipy.ndimage as nd
    
    if not os.path.exists(calibrated_file):
        msg = f'nircam_wisp_correction - {calibrated_file} not found'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None
        
    im = pyfits.open(calibrated_file)
    sh = im['SCI'].data.shape
    
    if ('WISPBKG' in im[0].header) & (not force):
        msg = f'nircam_wisp_correction - {calibrated_file}: '
        _bkg = im[0].header['WISPBKG']
        msg += f"WISPBKG = {_bkg:.2f}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return _bkg

    _ = get_nircam_wisp_filename(im[0].header)
    wisp_file, _filt, _inst, _det, _msg = _
    
    if wisp_file is None:
        msg = f'nircam_wisp_correction - {calibrated_file}: '
        utils.log_comment(utils.LOGFILE, msg + _msg, verbose=verbose)
        return None
        
    if not os.path.exists(wisp_file):
        msg = f'nircam_wisp_correction - {calibrated_file}: '
        msg += f'{wisp_file} not found'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None
    
    wisp_ref_file = os.path.join(os.path.dirname(__file__), 
                                 'data', 'nircam_wisp.yml')
                                 
    with open(wisp_ref_file) as fp:
        wisp_ref = yaml.load(fp, Loader=yaml.Loader)
    
    if _det not in wisp_ref:
        print(f'{_det} key not found in {wisp_ref_file}')
        return None
        
    wisp = pyfits.open(wisp_file)
    
    # yp, xp = np.indices((2048,2048))
    #
    # # if _det in ['NRCA3']:
    # #     xc, yc = 623, 1680
    # #     Rmax = 275
    # # elif _det in ['NRCB4']:
    # #     xc, yc = 1261, 1558
    # #     Rmax = 500
    # # elif _det in ['NRCB3']:
    # #     xc, yc = 1229, 204
    # #     Rmax = 470
    #
    # wispr = np.sqrt((xp-xc)**2+(yp-yc)**2)
    #
    # wisp_msk = np.exp(-(wispr - Rmax)**2/2/80**2)
    # wisp_msk[wispr < Rmax] = 1
    
    wisp_msk = utils.pixel_polygon_mask(wisp_ref[_det]['polygon'], 
                                        (2048,2048))
                                        
    wisp_msk = nd.gaussian_filter(wisp_msk*1., 50)
    
    wisp[0].data[~np.isfinite(wisp[0].data)] = 0
    
    sci = im['SCI'].data.flatten()
    dq = im['DQ'].data.flatten()
    err = im['ERR'].data.flatten()

    A = np.array([np.ones_like(sci), (wisp[0].data*wisp_msk).flatten()])
    _model = np.nanmedian(sci[dq == 0])

    for _iter in range(niter):
        ok = (err > 0) & np.isfinite(sci + A[1,:] + err) 
        ok &= (sci - _model < 10*err) & (dq == 0)

        _x = np.linalg.lstsq((A[:,ok]/err[ok]).T, (sci[ok]/err[ok]),
                             rcond=None)

        msg = f'nircam_wisp_correction - {calibrated_file}: '
        msg += f'iter {_iter} Npix={ok.sum()} '
        msg += f'bg={_x[0][0]:.2f} wisp={_x[0][1]:.2f}'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        #print(ok.sum(), _x[0])
        _model = A.T.dot(_x[0])

    # Take background out of model
    _model -= _x[0][0]
    im.close()

    if update:
        with pyfits.open(calibrated_file, mode='update') as im:
            im[0].header['WISPBKG'] = _x[0][0], 'Wisp fit, background'
            im[0].header['WISPNORM'] = _x[0][1], 'Wisp fit, model norm'
            im['SCI'].data -= _model.reshape(im['SCI'].data.shape)
            im.flush()
            
    return sci.reshape(sh), (_model.reshape(sh))


def oneoverf_column_correction(visit, thresholds=[10,1.5], dilate_iter=[10,2], image_sn_cut=-1, **kwargs):
    """
    Masked column/row correction for 1/f noise in JWST exposures
    
    Parameters
    ----------
    visit : dict
        Visit definition dictionary (keys `product`, `files` at at minimum)
    
    thresholds, dilate_iter : [float, float], [int, int]
        Detection threshold and binary dilation iterations for dilated 
        segmentation masks from the drizzle-combined mosaic of the exposures
        in `visit`
    
    Returns
    -------
    Updates exposures in `visit['files']` in place
    
    """
    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'prep.oneoverf_column_correction')
    
    import scipy.ndimage as nd
    
    msg = f"Column correction {visit['product']}: thresholds={thresholds}"
    msg += f" dilate_iter:{dilate_iter}"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    cat = make_SEP_catalog(visit['product'], threshold=thresholds[0])
    seg0 = pyfits.open(visit['product']+ '_seg.fits')
    seg_wcs = pywcs.WCS(seg0[0].header, relax=True)

    cat = make_SEP_catalog(visit['product'], threshold=thresholds[1])
    seg1 = pyfits.open(visit['product']+ '_seg.fits')

    seg_mask = nd.binary_dilation(seg0[0].data > 0, iterations=dilate_iter[0])
    seg_mask |= nd.binary_dilation(seg1[0].data > 0, 
                                   iterations=dilate_iter[1])
    
    # pyfits.writeto('oneoverf_mask.fits', data=seg_mask*1, header=seg0[0].header)
    
    for _file in visit['files']:
        with pyfits.open(_file, mode='update') as _im:
            if _im[0].header['OTELESCO'] not in ['JWST']:
                continue
            
            if 'LONG' in _im[0].header['ODETECTO']:
                continue
                
            _sci = _im['SCI'].data
            _wcs = pywcs.WCS(_im['SCI'].header, relax=True)

            _blotted = utils.blot_nearest_exact((seg_mask*1), seg_wcs, _wcs)
            msk = _blotted > 0
            msk |= _im['ERR'].data <= 0
            _sn = (_sci - _im['SCI'].header['MDRIZSKY']) / _im['ERR'].data
            if image_sn_cut > 0:
                msk |= np.abs(_sn) > image_sn_cut
            
            dq4 = _im['DQ'].data - (_im['DQ'].data & 4)
            msk |= dq4 > 0
            
            _scim = _sci*1
            _scim[msk] = np.nan

            pedestal = np.nanmedian(_scim)
            _im['SCI'].header['MDRIZSKY'] = pedestal

            if _im[0].header['OINSTRUM'] in ['NIRCAM']:
                col_axis = 1
            else:
                col_axis = 0

            msg = f'Column correction for {_file} (axis={col_axis})'
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            cols = np.nanmedian(_scim - pedestal, axis=col_axis)
            cols[~np.isfinite(cols)] = 0
            
            if col_axis:
                _im['SCI'].data = (_im['SCI'].data.T - cols).T
            else:
                _im['SCI'].data -= cols

            _im['SCI'].header['COLCORR'] = (True, 'Column correction applied')
            _im['SCI'].header['COLMFRAC'] = (msk.sum() / msk.size,
                                             'Fraction of masked pixels')
            _im['SCI'].header['COLTHRS0'] = (thresholds[0],
                                          'First column correction threshold')
            _im['SCI'].header['COLTHRS1'] = (thresholds[1],
                                         'Second column correction threshold')
            _im['SCI'].header['COLNDIL0'] = (dilate_iter[0],
                                             'First dilation iterations')
            _im['SCI'].header['COLNDIL1'] = (dilate_iter[1],
                                             'Second dilation iterations')
            _im['SCI'].header['COLIAXIS'] = (col_axis,
                                             'Axis for column average')

            _im.flush()


def mask_snowballs(visit, snowball_erode=3, snowball_dilate=18, mask_bit=1024, instruments=['NIRCAM','NIRISS'], unset4=False, **kwargs):
    """
    Mask JWST IR snowballs
    
    Parameters
    ----------
    visit : dict
        Visit definition dictionary (keys `product`, `files` at at minimum)
    
    snowball_erode : int
        First pass binary erosion on DQ=4 mask
    
    snowball_dilate : int
        Subsequent binary dilation on the eroded DQ=4 mask
    
    mask_bit : int
        Bit to set for identified snowballs
    
    instruments : list
        Instruments where mask is calculated
    
    unset4 : bool
        Unset DQ=4 bit for flagged CRs
        
    Returns
    -------
    Updates `DQ` extension of files in ``visit['files']`` and sets some
    status keywords
    
    """
    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'prep.mask_snowballs')
    
    import scipy.ndimage as nd
    
    for _file in visit['files']:
        with pyfits.open(_file, mode='update') as _im:
            if 'OINSTRUM' in _im[0].header:
                _instrume = _im[0].header['OINSTRUM']
            else:
                _instrume = _im[0].header['INSTRUME']
                
            if _instrume not in instruments:
                continue
                
            crs = (_im['DQ'].data & 4)
            
            _erode = nd.binary_erosion(crs > 0, iterations=snowball_erode)
            snowball_mask = nd.binary_dilation(_erode, 
                                               iterations=snowball_dilate)
            
            label, num_labels = nd.label(snowball_mask)
            
            _im['DQ'].data |= (snowball_mask*mask_bit).astype(_im['DQ'].data.dtype)
            
            if unset4:
                _im['DQ'].data -= (_im['DQ'].data & 4)
                
            _im['SCI'].header['SNOWMASK'] = (True, 'Snowball mask applied')
            _im['SCI'].header['SNOWEROD'] = (snowball_erode,
                                             'CR bit=4 erosion for snowballs')
            _im['SCI'].header['SNOWDILA'] = (snowball_dilate,
                                            'CR bit=4 dilation for snowballs')
            _im['SCI'].header['SNOWMBIT'] = (mask_bit, 'Snowball bit ')
            _im['SCI'].header['SNOWBALN'] = (num_labels,
                                'Number of labeled features in snowball mask')
            
            _maskfrac = snowball_mask.sum() / snowball_mask.size
            _im['SCI'].header['SNOWBALF'] = (_maskfrac,
                                 'Fraction of masked pixels in snowball mask')
            
            msg = f"Snowball mask: {_file} "
            msg += f" N={_im['SCI'].header['SNOWBALN']:>3}"
            msg += f"(f={_im['SCI'].header['SNOWBALF']*100:.2f}%)"
            
            utils.log_comment(utils.LOGFILE, msg, verbose=True)

            _im.flush()
    
    return True


# bin widths defined in pixels with scale `pixel_scale
BLOT_BACKGROUND_PARAMS = {'bw': 64, 'bh': 64, 'fw': 3, 'fh': 3,
                          'pixel_scale': 0.06}


def blot_background(visit={'product': '', 'files': None},
                    bkg_params=BLOT_BACKGROUND_PARAMS,
                    verbose=True, skip_existing=True, get_median=False,
                    log=True, stepsize=10, **kwargs):
    """
    Blot SEP background of drizzled image back to component FLT images
    
    Parameters
    ----------
    visit : dict
        Dictionary defining the drizzle product ('product' key) and 
        associated FLT files that contribute to the drizzled mosaic ('files'
        list)
    
    bkg_params : dict
        Parameters for `sep.Background`
    
    verbose : bool
        Status messages
    
    skip_existing : bool
        Don't run if ``BLOTSKY`` keyword found in the FLT header
    
    get_median : bool
        Don't use full background but rather just use (masked) median value
        of the drizzled mosaic
    
    log : bool
        Write log information to `grizli.utils.LOGFILE`
    
    stepsize : int
        Parameter for `blot`
    
    Returns
    -------
    Nothing returned but subtracts the transformed background image directly
    from the FLT files and updates header keywords
    
    """
    if log:
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'prep.blot_background')

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from drizzlepac import astrodrizzle

    drz_files = glob.glob('{0}_dr[zc]_sci.fits'.format(visit['product']))

    if len(drz_files) == 0:
        logstr = '# blot_background: No mosaic found {0}_dr[zc]_sci.fits'
        logstr = logstr.format(visit['product'])
        utils.log_comment(utils.LOGFILE, logstr, verbose=True)

        return False

    drz_file = drz_files[0]
    drz_im = pyfits.open(drz_file)
    drz_unit = drz_im[0].header['BUNIT']

    drz_wcs = pywcs.WCS(drz_im[0].header)
    drz_wcs.pscale = utils.get_wcs_pscale(drz_wcs)

    # Get SEP background
    bkg_data = make_SEP_catalog(root=visit['product'], bkg_only=True, 
                                bkg_params=bkg_params, verbose=False)
    if get_median:
        mask = drz_im[0].data != 0
        bkg_data = bkg_data*0.+np.median(np.median(bkg_data[mask]))

    logstr = '#   Blot background from {0}'.format(drz_file)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    for file in visit['files']:
        flt = pyfits.open(file, mode='update')

        for ext in range(1, 5):
            if ('SCI', ext) not in flt:
                continue

            if ('BLOTSKY' in flt['SCI', ext].header) & (skip_existing):
                print(f'\'BLOTSKY\' keyword found in {file}.  Skipping....')
                continue

            logstr = '#   Blot background: {0}[SCI,{1}]'.format(file, ext)
            utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

            flt_wcs = pywcs.WCS(flt['SCI', ext].header, fobj=flt, relax=True)
            flt_wcs.pscale = utils.get_wcs_pscale(flt_wcs)

            blotted = utils.blot_nearest_exact(bkg_data.astype(np.float32),
                                               drz_wcs, flt_wcs,
                                               stepsize=stepsize,
                                               scale_by_pixel_area=True)

            flt_unit = flt['SCI', ext].header['BUNIT']
            if flt_unit+'/S' == drz_unit:
                tscale = flt[0].header['EXPTIME']
            elif flt_unit == drz_unit + '/S':
                tscale = 1./flt[0].header['EXPTIME']
            else:
                tscale = 1.

            flt['SCI', ext].data -= blotted*tscale
            flt['SCI', ext].header['BLOTSKY'] = (True,
                                                 'Sky blotted from SKYIMAGE')
            flt['SCI', ext].header['SKYIMAGE'] = (drz_file,
                                                  'Source image for sky')
            flt['SCI', ext].header['SKYBW'] = (bkg_params['bw'], 
                                               'Sky bkg_params')
            flt['SCI', ext].header['SKYBH'] = (bkg_params['bh'],
                                               'Sky bkg_params')
            flt['SCI', ext].header['SKYFW'] = (bkg_params['fw'],
                                               'Sky bkg_params')
            flt['SCI', ext].header['SKYFH'] = (bkg_params['fh'],
                                               'Sky bkg_params')
            flt['SCI', ext].header['SKYPIX'] = (bkg_params['pixel_scale'], 
                                                'Sky bkg_params, pixel_scale')

        flt.flush()

    return True


def separate_chip_sky(visit, filters=['F200LP','F350LP','F600LP','F390W'], stepsize=10, statistic=np.nanmedian, by_amp=True, only_flc=True, row_average=True, average_order=11, seg_dilate=16, **kwargs):
    """
    Get separate background values for each chip.  Updates 'CHIPSKY' header
    keyword for each SCI extension of the visit exposure files.
    
    Parameters
    ----------
    visit : dict
        List of visit information from `~grizli.utils.parse_flt_files`.
    
    filters : list
        Only run if the exposures in `visit['files']` use a filter in this
        list, e.g., less-common WFC3/UVIS filters
    
    stepsize : int
        Parameter for blot
    
    statistic : func
        Test statistic on (masked) image data from each extension
    
    by_amp : bool
        Compute stats by amp (half-chip) subregions (*not implemented*)
    
    only_flc : True
        Only run if `visit['files'][0]` has "flc" extension
    
    Returns
    -------
    status : bool
        Runtime status, True if executed 
        
    """
    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'prep.separate_chip_sky')

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from drizzlepac import astrodrizzle
    import scipy.ndimage as nd
    
    if ('flc' not in visit['files'][0]) & only_flc:
        return False
    
    flt = pyfits.open(visit['files'][0])
    filt_i = utils.parse_filter_from_header(flt[0].header)
    if filt_i not in filters:
        logstr = f'# separate_chip_sky: {filt_i} not in {filters} for '
        logstr += "'{0}'".format(visit['product'])
        utils.log_comment(utils.LOGFILE, logstr, verbose=True)
        return False
    
    seg_files = glob.glob('{0}_seg.fits*'.format(visit['product']))

    if len(seg_files) == 0:
        logstr = '# separate_chip_sky: No segimage found {0}_seg.fits'
        logstr = logstr.format(visit['product'])
        utils.log_comment(utils.LOGFILE, logstr, verbose=True)

    seg_file = seg_files[0]
    seg_im = pyfits.open(seg_file)
    seg_mask = nd.binary_dilation(seg_im[0].data > 0, iterations=seg_dilate)*1
    
    seg_wcs = pywcs.WCS(seg_im[0].header)
    seg_wcs.pscale = utils.get_wcs_pscale(seg_wcs)
    
    if row_average:
        row_num = {}
        row_den = {}
        make_fig = True
        if make_fig:
            fig, axes = plt.subplots(2,1,figsize=(8,8))
        
    for file in visit['files']:
        flt = pyfits.open(file, mode='update')

        for ext in range(1, 5):
            if ('SCI', ext) not in flt:
                continue

            flt_wcs = pywcs.WCS(flt['SCI', ext].header, fobj=flt, relax=True)
            flt_wcs.pscale = utils.get_wcs_pscale(flt_wcs)

            blotted = utils.blot_nearest_exact(seg_mask,
                                               seg_wcs, flt_wcs,
                                               stepsize=stepsize,
                                               scale_by_pixel_area=False)    
            
            nonzero = flt['SCI',ext].data != 0
            ok = (flt['DQ',ext].data == 0) & (blotted <= 0)
            ok &= nonzero
            
            stat = statistic(flt['SCI',ext].data[ok])
            
            if row_average:
                print(file, ext, stat)
                wht = 1/(flt['ERR',ext].data)**2
                wht[~ok] = 0
                                    
                filled = (flt['SCI',ext].data - stat)/stat
                filled[~(ok & nonzero)] = np.nan
                rows = np.nanmedian(filled, axis=1)
                rows[~np.isfinite(rows)] = 0
                
                if ext not in row_num:
                    row_num[ext] = rows
                    row_den[ext] = (rows != 0)*1
                else:
                    row_num[ext] += rows
                    row_den[ext] += (rows != 0)*1
                
                if make_fig:
                    axes[ext-1].plot(rows, alpha=0.5)
                    fig.tight_layout(pad=0.5)
                    if os.path.exists('/tmp'):
                        fig.savefig('/tmp/rows.png')
                    
                    plt.close()
                
            ###############
            
            if 'MDRIZSKY' in flt['SCI',ext].header:
                stat -= flt['SCI',ext].header['MDRIZSKY']
            
            logstr = f'# separate_chip_sky {filt_i}: '
            logstr += f'{file}[SCI,{ext}] = {stat:6.2f}'
            utils.log_comment(utils.LOGFILE, logstr, verbose=True)
            
            if 'CHIPSKY' in flt['SCI',ext].header:
                flt['SCI',ext].header['CHIPSKY'] += stat
            else:
                flt['SCI',ext].header['CHIPSKY'] = (stat, 'Chip-level sky')
                
            flt['SCI',ext].data -= nonzero*stat
        
                
        flt.flush()
    
    if row_average:
        row_avg = {}
        row_model = {}
        for ext in row_num:
            row_avg[ext] = row_num[ext]/row_den[ext]
            row_avg[ext][row_den[ext] <= 0] = np.nan

            if make_fig:
                axes[ext-1].plot(row_avg[ext], alpha=0.5, color='k')
                fig.tight_layout(pad=0.5)
            
            msk = np.isfinite(row_avg[ext])
            xi = np.linspace(0,1,row_avg[ext].size)
            #deg = 11
            
            for _iter in range(5):
                cc = np.polynomial.chebyshev.chebfit(xi[msk],
                                                     row_avg[ext][msk],
                                                     average_order, 
                                                     rcond=None,
                                                     full=False, w=None)
                                                     
                row_model[ext] = np.polynomial.chebyshev.chebval(xi, cc)
                msk = np.isfinite(row_avg[ext]) 
                msk &= np.abs(row_avg[ext] - row_model[ext]) < 3*utils.nmad(row_avg[ext][msk])
            
            if make_fig:
                axes[ext-1].plot(row_model[ext], color='r')
                if os.path.exists('/tmp'):
                    fig.savefig('/tmp/rows.png')
                    
        
        for file in visit['files']:
            flt = pyfits.open(file, mode='update')

            for ext in range(1, 5):
                if ('SCI', ext) not in flt:
                    continue

                nonzero = flt['SCI',ext].data != 0
                
                stat = flt['SCI',ext].header['CHIPSKY']*1
                if 'MDRIZSKY' in flt['SCI',ext].header:
                    stat += flt['SCI',ext].header['MDRIZSKY']
                
                row_avg_ext = nonzero * row_avg[ext][:,None] * stat
                flt['SCI',ext].data -= row_avg_ext
                flt['SCI',ext].header['ROWSKY'] = (True, 
                                                   'Row-averaged sky removed')
                                                   
            flt.flush()
            flt.close()
            
    return True


def add_external_sources(root='', maglim=20, fwhm=0.2, catalog='2mass'):
    """Add Gaussian sources in empty parts of an image derived from an external catalog

    Parameters
    ----------
    root : type

    hlim : type


    Returns
    -------
    savesimages : type

    """
    from astropy.modeling import models

    sci_file = glob.glob('{0}_dr[zc]_sci.fits'.format(root))[0]
    wht_file = glob.glob('{0}_dr[zc]_wht.fits'.format(root))[0]

    sci = pyfits.open(sci_file)
    wht = pyfits.open(wht_file)

    sh = sci[0].data.shape
    yp, xp = np.indices(sh)

    PHOTPLAM = sci[0].header['PHOTPLAM']
    PHOTFLAM = sci[0].header['PHOTFLAM']

    ZP = -2.5*np.log10(PHOTFLAM) - 21.10 - 5*np.log10(PHOTPLAM) + 18.6921

    wcs = pywcs.WCS(sci[0])
    pscale = utils.get_wcs_pscale(wcs)

    rd = wcs.all_pix2world(np.array([[sh[1]/2], [sh[0]/2]]).T, 0)[0]

    radius = np.sqrt(2)*np.maximum(sh[0], sh[1])/2.*pscale/60.

    if catalog == '2mass':
        cat = get_irsa_catalog(rd[0], rd[1], radius=radius, twomass=True)
        cat['mag'] = cat['h_m']+1.362  # AB
        table_to_regions(cat, '{0}_2mass.reg'.format(root))
    elif catalog == 'panstarrs':
        cat = get_panstarrs_catalog(rd[0], rd[1], radius=radius)
        # cat['mag'] = cat['rMeanKronMag']+0.14 # AB
        cat['mag'] = cat['iMeanKronMag']+0.35  # AB
        table_to_regions(cat, '{0}_panstarrs.reg'.format(root))
    elif catalog == 'ukidss':
        cat = get_ukidss_catalog(rd[0], rd[1], radius=radius)
        cat['mag'] = cat['HAperMag3']+1.362  # AB
        cat.rename_column('RA', 'ra')
        cat.rename_column('Dec', 'dec')
        table_to_regions(cat, '{0}_ukidss.reg'.format(root))
    elif catalog == 'gaia':
        cat = get_gaia_DR2_vizier(rd[0], rd[1], radius=radius)
        cat['mag'] = np.minimum(cat['phot_g_mean_mag'], 19)-2
        table_to_regions(cat, '{0}_gaia.reg'.format(root))
    else:
        print('Not a valid catalog: ', catalog)
        return False

    cat = cat[(cat['mag'] < maglim) & (cat['mag'] > 0)]

    print('{0}: {1} objects'.format(catalog, len(cat)))
    if len(cat) == 0:
        return False

    xy = wcs.all_world2pix(cat['ra'], cat['dec'], 0)
    flux = sci[0].data*0.
    N = len(cat)

    for i in range(N):
        print('Add object {0:3d}/{1:3d}, x={2:6.1f}, y={3:6.1f}, mag={4:6.2f}'.format(i, N, xy[0][i], xy[1][i], cat['mag'][i]))

        scale = 10**(-0.4*(cat['mag'][i]-ZP))

        src = models.Gaussian2D(amplitude=scale, x_mean=xy[0][i], y_mean=xy[1][i], x_stddev=fwhm/pscale/2.35, y_stddev=fwhm/pscale/2.35, theta=0.0)
        m_i = src(xp, yp)
        flux += m_i
        # ds9.view(flux)

    clip = (wht[0].data == 0) & (flux > 1.e-6*flux.max())
    wht_val = np.percentile(wht[0].data, 95)
    wht[0].data[clip] = wht_val
    sci[0].data[clip] = flux[clip]

    sci.writeto(sci_file.replace('_drz', '_{0}_drz'.format(catalog)),
                overwrite=True)

    wht.writeto(wht_file.replace('_drz', '_{0}_drz'.format(catalog)),
                overwrite=True)


def asn_to_dict(input_asn):
    """Convert an ASN file to a dictionary

    Parameters
    ----------
    input_asn : str
        Filename of the ASN table

    Returns
    -------
    output : dict
        Dictionary with keys 'product' and 'files'.

    """
    from stsci.tools import asnutil
    # Already is a dict
    if isinstance(input_asn, dict):
        return input_asn

    # String / unicode
    if hasattr(input_asn, 'upper'):
        asn = asnutil.readASNTable(input_asn)
    else:
        # asnutil.ASNTable
        asn = input_asn

    output = {'product': asn['output'],
              'files': asn['order']}

    return output


# Visit-level ackground subtraction parameters for blot_background
BKG_PARAMS = {'bw': 128, 'bh': 128, 'fw': 3, 'fh': 3, 'pixel_scale': 0.06}

SEPARATE_CHIP_KWARGS = {'filters': ['F200LP','F350LP','F600LP','F390W'], 
                        'stepsize': 10,
                        'statistic': np.median, 
                        'by_amp':True, 
                        'only_flc':True}

def process_direct_grism_visit(direct={},
                               grism={},
                               radec=None,
                               outlier_threshold=5, 
                               align_clip=30,
                               align_thresh=None,
                               align_mag_limits=[14, 23, 0.05],
                               align_rms_limit=2,
                               align_triangle_ba_max=0.9,
                               align_ref_border=100,
                               align_min_flux_radius=1., 
                               align_assume_close=False,
                               align_transform=None,
                               align_guess=None,
                               max_err_percentile=99,
                               catalog_mask_pad=0.05,
                               match_catalog_density=None,
                               column_average=True,
                               sky_iter=10,
                               run_tweak_align=True,
                               tweak_fit_order=-1,
                               skip_direct=False,
                               fix_stars=True,
                               tweak_max_dist=100.,
                               tweak_n_min=10,
                               tweak_threshold=1.5,
                               tweak_ref_exp=0,
                               align_simple=True,
                               single_image_CRs=True,
                               skymethod='localmin',
                               drizzle_params={},
                               iter_atol=1.e-4,
                               imaging_bkg_params=None,
                               run_separate_chip_sky=True,
                               separate_chip_kwargs={},
                               reference_catalogs=['GAIA', 'PS1', 
                                                   'SDSS', 'WISE'],
                               use_self_catalog=False,
                               nircam_wisp_kwargs={},
                               oneoverf_kwargs={},
                               snowball_kwargs={},
                               miri_skyflat=True,
                               miri_skyfile=None,
                               use_skyflats=True):
    """Full processing of a direct (+grism) image visit.
    
    Notes
    -----
    
    For **imaging** exposures:

    1) Copies of individual exposures with `~grizli.prep.fresh_flt_file`
        
      * Run `stwcs.updatewcs.updatewcs` on each FLT
    
    #) "tweak" shift alignment of individual FLTs 
    
      * If ACS or UVIS, do preliminary `AstroDrizzle` run to flag CRs
    
    #) Run `AstroDrizzle` to create first-pass mosaic
    
    #) Astrometric alignment of the drizzled image reference catalogs with
       `~grizli.prep.align_drizzled_image` 
        
      * Propagate alignment back to FLT exposures
        
    #) Redrizzle visit mosaic with updated astrometry
    
    #) *optional* Subtract mosaic background from exposures with 
       `~grizli.prep.blot_background`
    
    #) Make final visit catalog 
    
    #) *optional* Fill saturated stars with ePSF models with 
       `~grizli.prep.fix_star_centers`
    
    For **grism** exposures:
    
    If *grism* exposures are specified, first do the above for the direct 
    images and then, 
    
    1) Assign (refined) WCS of associated direct image to each grism exposure
       (`~grizli.prep.match_direct_grism_wcs`)
    
    #) Run `AstroDrizzle` to flag additional CRs, bad pixels
    
    #) Subtract 2D sky background (`~grizli.prep.visit_grism_sky`)
    
      * *optional* additional column-average grism background 
    
    #) Redrizzle grism mosaic
    
    """
    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'prep.process_direct_grism_visit')

    #from stsci.tools import asnutil
    from stwcs import updatewcs
    from drizzlepac import updatehdr
    from drizzlepac.astrodrizzle import AstroDrizzle

    #################
    # Direct image processing
    #################

    # Copy FLT files from ../RAW
    isACS = '_flc' in direct['files'][0]  # Also UVIS
    isWFPC2 = '_c0' in direct['files'][0]
        
    if not skip_direct:
        for file in direct['files']:
            crclean = isACS & (len(direct['files']) == 1)
            
            fresh_flt_file(file, crclean=crclean,
                           oneoverf_correction=(oneoverf_kwargs is not None), 
                           oneoverf_kwargs=oneoverf_kwargs,
                           use_skyflats=use_skyflats)
                           
            isJWST = check_isJWST(file)
            if isJWST:
                isACS = isWFPC2 = False
                _ = jwst_utils.set_jwst_to_hst_keywords(file, reset=False)
            
            if not isJWST:
                try:
                    updatewcs.updatewcs(file, verbose=False, use_db=False)
                except TypeError:
                    updatewcs.updatewcs(file, verbose=False)
        
        if isJWST:
            if miri_skyflat:
                make_visit_average_flat(direct)
            elif miri_skyfile is not None:
                apply_visit_skyflat(direct, skyfile=miri_skyfile)
    
    # Initial grism processing
    skip_grism = (grism == {}) | (grism is None) | (len(grism) == 0)
    if not skip_grism:
        for file in grism['files']:
            fresh_flt_file(file, 
                           oneoverf_correction=(oneoverf_kwargs is not None), 
                           oneoverf_kwargs=oneoverf_kwargs,
                           use_skyflats=use_skyflats)

            # Need to force F814W filter for updatewcs
            if isACS:
                with pyfits.open(file, mode='update') as flc:
                    if flc[0].header['INSTRUME'] == 'ACS':
                        changed_filter = True
                        flc[0].header['FILTER1'] = 'CLEAR1L'
                        flc[0].header['FILTER2'] = 'F814W'
                        flc.flush()
                    else:
                        changed_filter = False
            else:
                changed_filter = False

            # Run updatewcs
            if not isJWST:
                try:
                    updatewcs.updatewcs(file, verbose=False, use_db=False)
                except TypeError:
                    updatewcs.updatewcs(file, verbose=False)
                
            # Change back
            if changed_filter:
                with pyfits.open(file, mode='update') as flc:
                    flc[0].header['FILTER1'] = 'CLEAR2L'
                    flc[0].header['FILTER2'] = 'G800L'
                    flc.flush()

        # Make ASN
        # asn = asnutil.ASNTable(grism['files'], output=grism['product'])
        # asn.create()
        # asn.write()

    if isACS:
        bits = 64+32+256
        driz_cr_snr = '3.5 3.0'
        driz_cr_scale = '1.2 0.7'
    elif isWFPC2:
        bits = 64+32
        driz_cr_snr = '3.5 3.0'
        driz_cr_scale = '1.2 0.7'
    elif isJWST:
        # Placeholder
        bits = 4
        driz_cr_snr = '8.0 5.0'
        driz_cr_scale = '2.5 0.7'        
    else:
        bits = 576+256
        driz_cr_snr = '8.0 5.0'
        driz_cr_scale = '2.5 0.7'

    if 'driz_cr_scale' in drizzle_params:
        driz_cr_scale = drizzle_params['driz_cr_scale']
        drizzle_params.pop('driz_cr_scale')

    if 'driz_cr_snr' in drizzle_params:
        driz_cr_snr = drizzle_params['driz_cr_snr']
        drizzle_params.pop('driz_cr_snr')

    if 'bits' in drizzle_params:
        bits = drizzle_params['bits']
        drizzle_params.pop('bits')
                                
    # Relax CR rejection for first-pass ACS
    if isACS:
        driz_cr_snr_first = '15. 10.0'
        driz_cr_scale_first = '1.2 0.7'
    else:
        driz_cr_snr_first = driz_cr_snr
        driz_cr_scale_first = driz_cr_scale
    
    # GAIN for AstroDrizzle
    if isJWST:
        # I guess this means ignore these values?
        _gain = '-99'
        _rdnoise = '-99'
    else:
        # Will be derived from header keywords for HST
        _gain = None
        _rdnoise = None
        
    if not skip_direct:
        if isJWST:
            if snowball_kwargs is not None:
                mask_snowballs(direct, **snowball_kwargs)
            
        if (not isACS) & (not isWFPC2) & run_tweak_align:
            # if run_tweak_align:
            tweak_align(direct_group=direct,
                        grism_group=grism,
                        max_dist=tweak_max_dist,
                        n_min=tweak_n_min,
                        key=' ',
                        drizzle=False,
                        ref_exp=tweak_ref_exp,
                        threshold=tweak_threshold, fit_order=tweak_fit_order)

        if (isACS) & (len(direct['files']) == 1) & single_image_CRs:
            find_single_image_CRs(direct, simple_mask=False, with_ctx_mask=False, run_lacosmic=True)

        # Get reference astrometry from GAIA, PS1, SDSS, WISE, etc.
        if radec is None:
            with pyfits.open(direct['files'][0]) as im:
                _h0 = im[0].header
                _h1 = im[1].header
                
                radec, ref_catalog = get_radec_catalog(ra=_h1['CRVAL1'],
                                        dec=_h1['CRVAL2'],
                                        product=direct['product'],
                                        reference_catalogs=reference_catalogs,
                                        date=_h0['EXPSTART'],
                                        date_format='mjd',
                                        use_self_catalog=use_self_catalog)

            if ref_catalog == 'VISIT':
                align_mag_limits = [16, 23, 0.05]
            elif ref_catalog == 'SDSS':
                align_mag_limits = [16, 21, 0.05]
            elif ref_catalog == 'PS1':
                align_mag_limits = [16, 23, 0.05]
            elif ref_catalog == 'WISE':
                align_mag_limits = [15, 20, 0.05]
        else:
            ref_catalog = 'USER'

        logstr = '#  {0}: First Drizzle'.format(direct['product'])
        utils.log_comment(utils.LOGFILE, logstr, verbose=True, show_date=True)

        # Clean up
        for ext in ['.fits', '.log']:
            file = '{0}_wcs.{1}'.format(direct['product'], ext)
            if os.path.exists(file):
                os.remove(file)        
        
        # First drizzle
        if len(direct['files']) > 1:
            for file in direct['files']:
                with pyfits.open(file, mode='update') as hdu:
                    hdu[3].data = hdu[3].data.astype(np.int16)
                    hdu.flush()
            
            AstroDrizzle(direct['files'], output=direct['product'],
                         clean=True, context=False, preserve=False,
                         skysub=True,
                         skymethod=skymethod,
                         driz_separate=True, driz_sep_wcs=True,
                         median=True, blot=True, driz_cr=True,
                         driz_cr_snr=driz_cr_snr_first,
                         driz_cr_scale=driz_cr_scale_first,
                         driz_cr_corr=False, driz_combine=True,
                         final_bits=bits, coeffs=True, build=False,
                         resetbits=4096,
                         final_wht_type='IVM',
                         gain=_gain,
                         rdnoise=_rdnoise, **drizzle_params)
        else:
            AstroDrizzle(direct['files'], output=direct['product'],
                         clean=True, final_scale=None, final_pixfrac=1,
                         context=False, final_bits=bits, preserve=False,
                         driz_separate=False, driz_sep_wcs=False,
                         median=False, blot=False, driz_cr=False,
                         driz_cr_corr=False, driz_combine=True,
                         build=False, final_wht_type='IVM',
                         gain=_gain,
                         rdnoise=_rdnoise,
                         **drizzle_params)

        # Now do tweak_align for ACS
        if (isACS) & run_tweak_align & (len(direct['files']) > 1):
            tweak_align(direct_group=direct, grism_group=grism,
                    max_dist=tweak_max_dist, n_min=tweak_n_min,
                    key=' ', drizzle=False,
                    threshold=tweak_threshold)

            # Redrizzle with no CR rejection
            AstroDrizzle(direct['files'], output=direct['product'],
                             clean=True, context=False, preserve=False,
                             skysub=False, driz_separate=False,
                             driz_sep_wcs=False,
                             median=False, blot=False, driz_cr=False,
                             driz_cr_corr=False, driz_combine=True,
                             final_bits=bits, coeffs=True, build=False,
                             final_wht_type='IVM',
                             gain=_gain,
                             rdnoise=_rdnoise,
                             resetbits=0)
        
        # Make catalog & segmentation image
        if align_thresh is None:
            if isWFPC2:
                thresh = 8
            else:
                thresh = 2
        else:
            thresh = align_thresh

        #cat = make_drz_catalog(root=direct['product'], threshold=thresh)
        cat = make_SEP_catalog(root=direct['product'], threshold=thresh)

        if radec == 'self':
            okmag = ((cat['MAG_AUTO'] > align_mag_limits[0]) &
                    (cat['MAG_AUTO'] < align_mag_limits[1]))

            cat['X_WORLD', 'Y_WORLD'][okmag].write('self',
                                        format='ascii.commented_header',
                                        overwrite=True)

        # clip=30
        logfile = '{0}_wcs.log'.format(direct['product'])
        if os.path.exists(logfile):
            os.remove(logfile)

        guess_file = '{0}.align_guess'.format(direct['product'])
        if align_guess != None:
            guess = align_guess
        elif os.path.exists(guess_file):
            guess = np.loadtxt(guess_file)
        else:
            guess = [0., 0., 0., 1]
        
        # Database radec?
        if radec == 'astrometry_db':
            radec = get_visit_radec_from_database(direct)

        try:
            result = align_drizzled_image(root=direct['product'],
                                      mag_limits=align_mag_limits,
                                      radec=radec, NITER=3, clip=align_clip,
                                      log=True, guess=guess,
                                      outlier_threshold=outlier_threshold,
                                      simple=align_simple,
                                      rms_limit=align_rms_limit,
                                      max_err_percentile=max_err_percentile,
                                      catalog_mask_pad=catalog_mask_pad,
                                      triangle_size_limit=[5, 2400*(1+isACS)],
                                      triangle_ba_max=align_triangle_ba_max,
                                match_catalog_density=match_catalog_density,
                                      ref_border=align_ref_border, 
                                      min_flux_radius=align_min_flux_radius, 
                                      assume_close=align_assume_close,
                                      transform=align_transform)
        except:

            utils.log_exception(utils.LOGFILE, traceback)
            utils.log_comment(utils.LOGFILE, "# !! Drizzle alignment failed")

            fp = open('{0}.wcs_failed'.format(direct['product']), 'w')
            fp.write(guess.__str__())
            fp.close()

            # Does nothing but moves forward
            result = align_drizzled_image(root=direct['product'],
                                      mag_limits=align_mag_limits,
                                      radec=radec, NITER=0, clip=align_clip,
                                      log=False, guess=guess,
                                      outlier_threshold=outlier_threshold,
                                      simple=align_simple,
                                      rms_limit=align_rms_limit,
                                      max_err_percentile=max_err_percentile,
                                      catalog_mask_pad=catalog_mask_pad,
                                match_catalog_density=match_catalog_density,
                                      ref_border=align_ref_border,
                                      min_flux_radius=align_min_flux_radius,
                                      transform=align_transform)

        orig_wcs, drz_wcs, out_shift, out_rot, out_scale = result

        # Update direct FLT WCS
        for file in direct['files']:
            xyscale = [out_shift[0], out_shift[1], out_rot, out_scale]

            update_wcs_fits_log(file, orig_wcs,
                                xyscale=xyscale,
                                initialize=False,
                                replace=('.fits', '.wcslog.fits'),
                                wcsname=ref_catalog)

            updatehdr.updatewcs_with_shift(file,
                                str('{0}_wcs.fits'.format(direct['product'])),
                                      xsh=out_shift[0], ysh=out_shift[1],
                                      rot=out_rot, scale=out_scale,
                                      wcsname=ref_catalog, force=True,
                                      reusename=True, verbose=True,
                                      sciext='SCI')

            # Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
            # keywords
            with pyfits.open(file, mode='update') as im:
                im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
                im.flush()

        # Second drizzle with aligned wcs, refined CR-rejection params
        # tuned for WFC3/IR
        logstr = '# {0}: Second Drizzle'.format(direct['product'])
        utils.log_comment(utils.LOGFILE, logstr, verbose=True, show_date=True)

        if len(direct['files']) == 1:
            AstroDrizzle(direct['files'], output=direct['product'],
                         clean=True, final_pixfrac=0.8, context=False,
                         resetbits=4096, final_bits=bits, driz_sep_bits=bits,
                         preserve=False, driz_cr_snr=driz_cr_snr,
                         driz_cr_scale=driz_cr_scale, driz_separate=False,
                         driz_sep_wcs=False, median=False, blot=False,
                         driz_cr=False, driz_cr_corr=False,
                         build=False, final_wht_type='IVM',
                         gain=_gain, rdnoise=_rdnoise,
                         **drizzle_params)
        else:
            if 'par' in direct['product']:
                pixfrac = 1.0
            else:
                pixfrac = 0.8

            AstroDrizzle(direct['files'], output=direct['product'],
                         clean=True, final_pixfrac=pixfrac,
                         context=(isACS | isWFPC2),
                         skysub=True, skymethod=skymethod,
                         resetbits=4096, final_bits=bits, driz_sep_bits=bits,
                         preserve=False, driz_cr_snr=driz_cr_snr,
                         driz_cr_scale=driz_cr_scale, build=False,
                         final_wht_type='IVM',
                         gain=_gain, rdnoise=_rdnoise,
                         **drizzle_params)
                         
        # Flag areas of ACS images covered by a single image, where
        # CRs aren't appropriately masked
        is_single = (len(direct['files']) == 1)
        if (single_image_CRs) & (isACS | isWFPC2):
            logstr = '# Mask areas of the mosaic covered by a single input image'
            utils.log_comment(utils.LOGFILE, logstr, verbose=True)

            try:
                find_single_image_CRs(direct, simple_mask=(not is_single), with_ctx_mask=(not is_single), run_lacosmic=is_single)
            except:
                utils.log_exception(utils.LOGFILE, traceback)
                pass

        # Make DRZ catalog again with updated DRZWCS
        clean_drizzle(direct['product'])
        
        if isJWST:
            if nircam_wisp_kwargs is not None:
                for _file in direct['files']:
                    nircam_wisp_correction(_file, **nircam_wisp_kwargs)
                
            # if oneoverf_kwargs is not None:
            #     oneoverf_column_correction(direct, **oneoverf_kwargs)
                        
            # Redrizzle before background
            AstroDrizzle(direct['files'], output=direct['product'],
                         clean=True, final_pixfrac=0.8, context=False,
                         skysub=True, skymethod=skymethod,
                         resetbits=0, final_bits=bits, driz_sep_bits=bits,
                         preserve=False, driz_cr_snr=driz_cr_snr,
                         driz_cr_scale=driz_cr_scale, driz_separate=False,
                         driz_sep_wcs=False, median=False, blot=False,
                         driz_cr=False, driz_cr_corr=False,
                         build=False, final_wht_type='IVM',
                         gain=_gain, rdnoise=_rdnoise,
                         **drizzle_params)
             
            clean_drizzle(direct['product'])
            
        # Subtract visit-level background based on the drizzled mosaic
        if imaging_bkg_params is not None:
            logstr = '# Imaging background: {0}'.format(imaging_bkg_params)
            utils.log_comment(utils.LOGFILE, logstr, verbose=True)

            bkg_params = imaging_bkg_params.copy()

            if 'get_median' in bkg_params:
                get_median = bkg_params.pop('get_median')

            else:
                get_median = False

            blot_background(visit=direct, bkg_params=bkg_params,
                            verbose=True, skip_existing=True,
                            get_median=get_median)
        
        if run_separate_chip_sky & isACS:
            separate_chip_sky(direct, **separate_chip_kwargs)
            
        # Chip-level background for some UVIS filters
        #  uvis_chip_background(visit=direct)
        
        # Remake catalog
        #cat = make_drz_catalog(root=direct['product'], threshold=thresh)
        cat = make_SEP_catalog(root=direct['product'], threshold=thresh)

        # 140 brightest or mag range
        clip = (cat['MAG_AUTO'] > align_mag_limits[0]) & (cat['MAG_AUTO'] < align_mag_limits[1])
        if len(align_mag_limits) > 2:
            clip &= cat['MAGERR_AUTO'] < align_mag_limits[2]
        else:
            clip &= cat['MAGERR_AUTO'] < 0.05

        clip &= utils.catalog_mask(cat, max_err_percentile=max_err_percentile, 
                                   pad=catalog_mask_pad, 
                                   pad_is_absolute=False, 
                                   min_flux_radius=align_min_flux_radius)

        NMAX = 140
        so = np.argsort(cat['MAG_AUTO'][clip])
        if clip.sum() > NMAX:
            so = so[:NMAX]

        table_to_regions(cat[clip][so], 
                         '{0}.cat.reg'.format(direct['product']))

        if not ((isACS | isWFPC2) & is_single):
            table_to_radec(cat[clip][so], 
                           '{0}.cat.radec'.format(direct['product']))

        if (fix_stars) & (not isACS) & (not isWFPC2) & (not isJWST):
            fix_star_centers(root=direct['product'], drizzle=False, 
                             mag_lim=19.5)
        
        # Reset JWST headers
        if isJWST:
            for file in direct['files']:
                _ = jwst_utils.set_jwst_to_hst_keywords(file, reset=True)
        
    #################
    # Grism image processing
    #################

    if skip_grism:
        return True

    # Match grism WCS to the direct images
    match_direct_grism_wcs(direct=direct, grism=grism, get_fresh_flt=False)
    
    if isJWST:
        if snowball_kwargs is not None:
            mask_snowballs(grism, **snowball_kwargs)
        
    # First drizzle to flag CRs
    gris_cr_corr = len(grism['files']) > 1
    driz_cr_snr = '8.0 5.0'
    driz_cr_scale = '2.5 0.7'
    
    for file in grism['files']:
        with pyfits.open(file,mode='update') as hdu:
            hdu[3].data = hdu[3].data.astype(np.int16)
            hdu.flush()
        
        if isJWST:
            # Set HST header
            _ = jwst_utils.set_jwst_to_hst_keywords(file, reset=False)
        
    AstroDrizzle(grism['files'], output=grism['product'], clean=True,
                 context=False, preserve=False, skysub=True,
                 driz_separate=gris_cr_corr, driz_sep_wcs=gris_cr_corr,
                 median=gris_cr_corr,
                 blot=gris_cr_corr, driz_cr=gris_cr_corr,
                 driz_cr_corr=False,
                 driz_cr_snr=driz_cr_snr, driz_cr_scale=driz_cr_scale,
                 driz_combine=True, final_bits=bits, coeffs=True,
                 resetbits=4096, build=False, final_wht_type='IVM',
                 gain=_gain, rdnoise=_rdnoise)
    
    if isJWST:
        # Set keywords back
        for file in grism['files']:
            _ = jwst_utils.set_jwst_to_hst_keywords(file, reset=True)
    
    # Subtract grism sky
    status = visit_grism_sky(grism=grism, apply=True, sky_iter=sky_iter,
                          column_average=column_average, verbose=True, ext=1,
                          iter_atol=iter_atol)

    # Run on second chip (also for UVIS/G280)
    if isACS:
        visit_grism_sky(grism=grism, apply=True, sky_iter=sky_iter,
                        column_average=column_average, verbose=True, ext=2,
                        iter_atol=iter_atol)

        # Add back in some pedestal or CR rejection fails for ACS
        for file in grism['files']:
            with pyfits.open(file, mode='update') as flt:
                h = flt[0].header
                flat_sky = h['GSKY101']*h['EXPTIME']

                # Use same pedestal for both chips for skysub
                for ext in [1, 2]:
                    flt['SCI', ext].data += flat_sky

                flt.flush()

    # Redrizzle with new background subtraction
    if isACS:
        skyfile = ''
    else:
        skyfile = '{0}.skyfile'.format(grism['product'])
        with open(skyfile, 'w') as fp:
            fp.writelines(['{0} 0.0\n'.format(f) for f in grism['files']])

    if 'par' in grism['product']:
        pixfrac = 1.0
    else:
        pixfrac = 0.8
    
    if isJWST:
        # Set keywords back to HST for drizzle
        for file in grism['files']:
            _ = jwst_utils.set_jwst_to_hst_keywords(file, reset=False)
    
    AstroDrizzle(grism['files'], output=grism['product'], clean=True,
                 context=isACS, preserve=False, skysub=True, skyfile=skyfile,
                 driz_separate=gris_cr_corr, driz_sep_wcs=gris_cr_corr,
                 median=gris_cr_corr,
                 blot=gris_cr_corr, driz_cr=gris_cr_corr, 
                 driz_cr_corr=False,
                 driz_cr_snr=driz_cr_snr, driz_cr_scale=driz_cr_scale,
                 driz_combine=True, driz_sep_bits=bits, final_bits=bits,
                 coeffs=True, resetbits=4096, final_pixfrac=pixfrac,
                 build=False,
                 gain=_gain, rdnoise=_rdnoise,
                 final_wht_type='IVM')
    
    if os.path.exists(skyfile):
        os.remove(skyfile)
        
    clean_drizzle(grism['product'])

    # Add direct filter to grism FLT headers
    set_grism_dfilter(direct, grism)
    
    if isJWST:
        # Set keywords back
        for file in grism['files']:
            _ = jwst_utils.set_jwst_to_hst_keywords(file, reset=True)
    
    return True


def set_grism_dfilter(direct, grism):
    """Set direct imaging filter for grism exposures

    Parameters
    ----------
    direct, grism : dict

    Returns
    -------
    Nothing

    """
    d_im = pyfits.open(direct['files'][0])
    direct_filter = utils.parse_filter_from_header(d_im[0].header)
    for file in grism['files']:
        if '_flc' in file:
            ext = [1, 2]
        else:
            ext = [1]

        print('DFILTER: {0} {1}'.format(file, direct_filter))
        with pyfits.open(file, mode='update') as flt:
            for e in ext:
                flt['SCI', e].header['DFILTER'] = (direct_filter,
                                                   'Direct imaging filter')
            
            flt.flush()


def tweak_align(direct_group={}, grism_group={}, max_dist=1., n_min=10, key=' ', threshold=3, drizzle=False, fit_order=-1, ref_exp=0):
    """Intra-visit shift alignment
    
    Parameters
    ----------
    direct_group : dict
        Visit info (`product`, `files`) for direct images
    
    grism_group : dict
        Visit info (`product`, `files`) for grism images
    
    max_dist, threshold : float
        Passed to `~grizli.prep.tweak_flt` 
    
    n_min : int
        Minimum number of sources for a valid fit.  
    
    drizzle : bool
        Run `AstroDrizzle` after performing the alignment
    
    fit_order : int
        If > 0, then fit a polynomial to the derived shifts rather than 
        using the shifts themselves, e.g., for DASH imaging
    
    ref_exp : int
        Index of the exposure to use as the reference for the tweak shifts
        
    Returns
    -------
    Nothing, but updates WCS of direct and (optionally) grism exposures
    
    """
    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'prep.tweak_align')

    from drizzlepac.astrodrizzle import AstroDrizzle
    from scipy import polyfit, polyval

    if len(direct_group['files']) < 2:
        logstr = '# ! {0}: Only one direct image found, can\'t compute shifts'
        logstr = logstr.format(direct_group['product'])

        utils.log_comment(utils.LOGFILE, logstr, verbose=True)
        return True

    wcs_ref, shift_dict = tweak_flt(files=direct_group['files'],
                                    max_dist=max_dist, threshold=threshold,
                                    verbose=True, ref_exp=ref_exp)

    grism_matches = find_direct_grism_pairs(direct=direct_group, grism=grism_group, check_pixel=[507, 507], toler=0.1, key=key)
    logstr = '\ngrism_matches = {0}\n'.format(grism_matches)
    utils.log_comment(utils.LOGFILE, logstr, verbose=True)

    fp = open('{0}_shifts.log'.format(direct_group['product']), 'w')
    fp.write('# flt xshift yshift rot scale N rmsx rmsy\n')
    fp.write('# fit_order: {0}\n'.format(fit_order))

    for k in grism_matches:
        d = shift_dict[k]
        fp.write('# match[\'{0}\'] = {1}\n'.format(k, grism_matches[k]))

    for k in shift_dict:
        d = shift_dict[k]
        n_i = d[4]
        if (n_i < n_min) | (np.abs(d[:2]).max() > max_dist):
            fp.write('# ! {0:s} {1:7.3f} {2:7.3f} {3:8.5f} {4:8.5f} {5:5d} {6:6.3f} {7:6.3f}\n'.format(k, d[0], d[1], d[2], d[3], d[4], d[5][0], d[5][1]))
            d[0] = d[1] = 0.

        fp.write('{0:s} {1:7.3f} {2:7.3f} {3:8.5f} {4:8.5f} {5:5d} {6:6.3f} {7:6.3f}\n'.format(k, d[0], d[1], d[2], d[3], d[4], d[5][0], d[5][1]))

    fp.close()

    # Fit a polynomial, e.g., for DASH
    if fit_order > 0:
        logstr = '# {0}: Fit polynomial order={1} to shifts.'
        logstr = logstr.format(direct_group['product'], fit_order)
        utils.log_comment(utils.LOGFILE, logstr, verbose=True)

        shifts = np.array([shift_dict[k][:2] for k in sorted(shift_dict)])
        t = np.arange(shifts.shape[0])
        cx = polyfit(t, shifts[:, 0], fit_order)
        sx = polyval(cx, t)
        cy = polyfit(t, shifts[:, 1], fit_order)
        sy = polyval(cy, t)
        fit_shift = np.array([sx, sy]).T

        for ik, k in enumerate(sorted(shift_dict)):
            shift_dict[k][:2] = fit_shift[ik, :]

    # Apply the shifts to the header WCS
    apply_tweak_shifts(wcs_ref, shift_dict, grism_matches=grism_matches,
                       verbose=False)

    if not drizzle:
        return True

    # Redrizzle
    bits = 576
    driz_cr_snr = '8.0 5.0'
    driz_cr_scale = '2.5 0.7'
    if 'par' in direct_group['product']:
        pixfrac = 1.0
    else:
        pixfrac = 0.8

    AstroDrizzle(direct_group['files'], output=direct_group['product'],
                 clean=True, final_pixfrac=pixfrac, context=False,
                 resetbits=4096, final_bits=bits, driz_sep_bits=bits,
                 preserve=False, driz_cr_snr=driz_cr_snr,
                 driz_cr_scale=driz_cr_scale, build=False,
                 final_wht_type='IVM')

    clean_drizzle(direct_group['product'])
    #cat = make_drz_catalog(root=direct_group['product'], threshold=1.6)
    cat = make_SEP_catalog(root=direct_group['product'], threshold=1.6)
    table_to_regions(cat, '{0}.cat.reg'.format(direct_group['product']))

    if (grism_group == {}) | (grism_group is None):
        return True

    # Grism
    skyfile = '{0}.skyfile'.format(grism_group['product'])
    with open(skyfile, 'w') as fp:
        fp.writelines(['{0} 0.0\n'.format(f) for f in grism_group['files']])

    AstroDrizzle(grism_group['files'], output=grism_group['product'],
                 clean=True, context=False, preserve=False, skysub=True,
                 skyfile=skyfile, driz_separate=True, driz_sep_wcs=True,
                 median=True, blot=True, driz_cr=True, driz_cr_corr=True,
                 driz_combine=True, driz_sep_bits=bits, final_bits=bits,
                 coeffs=True, resetbits=4096, final_pixfrac=pixfrac,
                 build=False, final_wht_type='IVM')
    
    if os.path.exists(skyfile):
        os.remove(skyfile)
        
    clean_drizzle(grism_group['product'])

    return True


def get_visit_radec_from_database(visit, nmax=200):
    """
    Get reference astrometry from grizli database
    
    Parameters
    ----------
    visit : dict
        Visit dictionary, with `product`, `files`, `footprint` keys
    
    nmax : 200
        Maximum number of alignment sources to return
        
    Returns
    -------
    radec_file : str
        Creates radec file from the database and returns the filename, or None
        if not overlapping reference sources found
        
    """
    from grizli.aws import db

    if 'footprint' not in visit:
        msg = f"'footprint' not found in visit dictionary {visit['product']}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        return None

    sr = utils.SRegion(visit['footprint'])

    try:
        srcs = db.SQL(f"""select src, count(src) from astrometry_reference
    where polygon('{sr.polystr()[0]}') @> point(ra, dec)
    group by src
    """)
    except:
        msg = f"Failed to query astrometry database for {visit['product']}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        return None
        
    if len(srcs) == 0:
        return None

    _src = srcs['src'][np.argmax(srcs['count'])]

    pts = db.SQL(f"""select * from astrometry_reference
    where polygon('{sr.polystr()[0]}') @> point(ra, dec)
    AND src = '{_src}'
    order by mag limit {nmax}
    """)

    msg = f"master_radec for {visit['product']}: {_src} N={len(pts)}"
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    table_to_radec(pts, _src+'.radec')
    return _src+'.radec'
    
    
def drizzle_footprint(weight_image, shrink=10, ext=0, outfile=None, label=None):
    """
    Footprint of image pixels where values > 0.  Works best with drizzled
    weight images.
    
    (not used)
    """
    from scipy.spatial import ConvexHull

    im = pyfits.open(weight_image)
    wcs = pywcs.WCS(im[ext].header, fobj=im)
    sh = np.array(im[ext].data.shape)//shrink

    yp, xp = np.indices(tuple(sh))*shrink
    nonzero = im[ext].data[yp, xp] > 0

    h = ConvexHull(np.array([xp[nonzero], yp[nonzero]]).T)
    hx = xp[nonzero][h.vertices]
    hy = yp[nonzero][h.vertices]

    hrd = wcs.all_pix2world(np.stack([hx, hy]).T, 0)

    pstr = 'polygon('+','.join(['{0:.6f}'.format(i) for i in hrd.flatten()])+')'
    if label is not None:
        pstr += ' # text={{{0}}}'.format(label)

    if outfile is None:
        return pstr

    fp = open(outfile, 'w')
    fp.write('fk5\n')

    fp.write(pstr+'\n')
    fp.close()
    im.close()


def clean_drizzle(root, context=False, fix_wcs_system=False):
    """Zero-out WHT=0 pixels in drizzle mosaics

    Parameters
    ----------
    root : str
        Rootname of the mosaics.  I.e., `{root}_drz_sci.fits`.

    Returns
    -------
    Nothing, science mosaic modified in-place
    """

    try:
        drz_file = glob.glob('{0}_dr[zc]_sci.fits'.format(root))[0]
        is_build = False
        sci_ext = 0
    except:
        drz_file = glob.glob('{0}_dr[zc].fits'.format(root))[0]
        sci_ext = 1
        is_build = True

    # Is result from build=True?

    sci = pyfits.open(drz_file, mode='update')

    if is_build:
        mask = sci['WHT'].data == 0
    else:
        wht = pyfits.open(drz_file.replace('_sci.fits', '_wht.fits'))
        mask = wht[0].data == 0

    if fix_wcs_system:
        # Force RADESYS/EQUINOX = ICRS/2000. and fix LATPOLE to CRVAL2
        sci[sci_ext].header['RADESYS'] = 'ICRS'
        sci[sci_ext].header['EQUINOX'] = 2000.0
        sci[sci_ext].header['LATPOLE'] = sci[sci_ext].header['CRVAL2']

    # Mask where context shows that mosaic comes from a single input
    ctx_file = drz_file.replace('_sci.', '_ctx.')
    if context & os.path.exists(ctx_file):
        ctx = pyfits.open(ctx_file)

        bits = np.log(ctx[0].data)/np.log(2)
        # bits = round(bits) when is a power of 2
        mask &= bits != np.round(bits)

    sci[sci_ext].data[mask] = 0

    # Rescale WFPC2 to ~WFC3 image zeropoint
    if sci[0].header['INSTRUME'] == 'WFPC2':
        #exptime = sci[0].header['EXPTIME']

        scl = sci[0].header['PHOTFLAM'] / 1.5e-20

        #sci[0].data /= exptime

        sci[sci_ext].data *= scl

        for k in ['PHOTFLAM', 'PHOTFNU']:
            if k in sci[0].header:
                sci[0].header[k] /= scl

        if is_build:
            sci['WHT'].data /= scl**2
        else:
            wht = pyfits.open(drz_file.replace('_sci.fits', '_wht.fits'),
                          mode='update')

            wht[0].data /= scl**2
            wht.flush()

    sci.flush()
    sci.close()


MATCH_KWS = dict(maxKeep=10, auto_keep=3, auto_transform=None, auto_limit=3,
                 size_limit=[5, 1800], ignore_rot=True, ignore_scale=True, 
                 ba_max=0.9)
                                
def tweak_flt(files=[], max_dist=0.4, threshold=3, verbose=True, tristars_kwargs=MATCH_KWS, ref_exp=0, use_sewpy=False):
    """Refine shifts of FLT files
    
    Parameters
    ----------
    files : list
        List of flt filenames
    
    max_dist : float
        Maximum shift distance to allow
    
    threshold : float
        Source detection threshold for `sep.extract`
    
    verbose : bool
        Status messages
    
    tristars_kwargs : dict
        Keyword arguments for `tristars.match.match_catalog_tri`
    
    ref_exp : int
        Index of the exposure to use as the reference for the tweak shifts
    
    use_sewpy : bool
        Use `sewpy` for source detection (deprecated)
    
    Returns
    -------
    ref_wcs : `~astropy.wcs.WCS`
        Reference WCS (WCS of the first file in `files`)
    
    shift_dict : dict
        Shift dictionary with keys from `files` and values like
        ``[xshift, yshift, rot, scale, N, rms]``.  Note that only shifts are 
        fit, so `rot = 0.` and `scale = 1.`.  ``N`` is the number of sources
        used for the fit.
        
    """
    import scipy.spatial

    try:
        import tristars
        from tristars.match import match_catalog_tri, match_diagnostic_plot
    except:
        print("""
    Couldn't `import tristars`.  Get it from https://github.com/gbrammer/tristars to enable improved blind astrometric matching with triangle asterisms.
    """)

    try:
        # https://github.com/megalut/sewpy
        import sewpy
    except:
        sewpy = None
        use_sewpy = False

    # Make FLT catalogs
    cats = []
    logstr = '### Tweak alignment (use_sewpy={0}) '.format(use_sewpy)
    utils.log_comment(utils.LOGFILE, logstr, verbose=True)

    for i, file in enumerate(files):
        root = file.split('.fits')[0]

        im = pyfits.open(file)
        try:
            ok = im['DQ', 1].data == 0
        except:
            ok = np.isfinite(im['SCI', 1].data)

        sci = im['SCI', 1].data*ok - np.nanmedian(im['SCI', 1].data[ok])

        header = im['SCI', 1].header.copy()

        for k in ['PHOTFNU', 'PHOTFLAM', 'PHOTPLAM', 'FILTER']:
            if k in im[0].header:
                header[k] = im[0].header[k]

        hst_filter = utils.parse_filter_from_header(im[0].header)
        header['FILTER'] = hst_filter

        pyfits.writeto('{0}_xsci.fits'.format(root), data=sci,
                       header=header,
                       overwrite=True)

        pyfits.writeto('{0}_xrms.fits'.format(root), data=im['ERR', 1].data,
                       header=im['ERR', 1].header, overwrite=True)

        if use_sewpy:
            params = ["X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD",
                                    "FLUX_RADIUS(3)", "FLAGS"]
            sew = sewpy.SEW(params=params,
                            config={"DETECT_THRESH": threshold,
                                    "DETECT_MINAREA": 8,
                                    "PHOT_FLUXFRAC": "0.3, 0.5, 0.8",
                                    "WEIGHT_TYPE": "MAP_RMS",
                                "WEIGHT_IMAGE": "{0}_xrms.fits".format(root)})

            output = sew('{0}_xsci.fits'.format(root))
            cat = output['table']
        else:
            # SEP
            wht = 1/im['ERR', 1].data**2
            wht[~(np.isfinite(wht))] = 0
            pyfits.writeto('{0}_xwht.fits'.format(root), data=wht,
                           header=im['ERR', 1].header, overwrite=True)

            pars = SEP_DETECT_PARAMS.copy()
            pars['minarea'] = 8

            cat = make_SEP_catalog(root=root,
                                   sci='{0}_xsci.fits'.format(root),
                                   wht='{0}_xwht.fits'.format(root),
                                   threshold=threshold, detection_params=pars,
                                   get_background=True, verbose=False)

        ######
        if '_flc' in file:
            wcs = pywcs.WCS(im['SCI', 1].header, fobj=im, relax=True)
        else:
            wcs = pywcs.WCS(im['SCI', 1].header, relax=True)

        cats.append([cat, wcs])

        for ext in ['_xsci', '_xrms', '_xwht', '_bkg', '_seg', '.cat']:
            file = '{0}{1}.fits'.format(root, ext)
            if os.path.exists(file):
                os.remove(file)
        
        im.close()
        
    if ref_exp < len(cats):
        c0, wcs_0 = cats[ref_exp]
    else:
        c0, wcs_0 = cats[0]
    
    not_CR = c0['FLUX_RADIUS'] > 1.5
    c0 = c0[not_CR]

    xy_0 = np.array([c0['X_IMAGE'], c0['Y_IMAGE']]).T
    tree = scipy.spatial.cKDTree(xy_0, 10)

    try:
        # Use Tristars for matching

        # First 100
        NMAX = 100
        if len(xy_0) > NMAX:
            so = np.argsort(c0['MAG_AUTO'])
            xy_0 = xy_0[so[:NMAX], :]

        shift_dict = OrderedDict()
        for i in range(0, len(files)):
            c_ii, wcs_i = cats[i]

            not_CR = c_ii['FLUX_RADIUS'] > 1.5
            c_i = c_ii[not_CR]

            # SExtractor doesn't do SIP WCS?
            rd = np.array(wcs_i.all_pix2world(c_i['X_IMAGE'], c_i['Y_IMAGE'], 1))
            xy = np.array(wcs_0.all_world2pix(rd.T, 1))

            if len(xy) > NMAX:
                so = np.argsort(c_i['MAG_AUTO'])
                xy = xy[so[:NMAX], :]

            pair_ix = match_catalog_tri(xy, xy_0, **tristars_kwargs)

            # if False:
            #     match_diagnostic_plot(xy, xy_0, pair_ix, tf=None, 
            #                           new_figure=False)

            dr = xy[pair_ix[:, 0], :] - xy_0[pair_ix[:, 1], :]
            ok = dr.max(axis=1) < 1000
            dx = np.median(dr[ok, :], axis=0)
            rms = np.std(dr[ok, :], axis=0)/np.sqrt(ok.sum())

            shift_dict[files[i]] = [dx[0], dx[1], 0.0, 1.0, ok.sum(), rms]

            lstr = "# tw {0} [{1:6.3f}, {2:6.3f}]  [{3:6.3f}, {4:6.3f}] N={5}"
            lstr = lstr.format(files[i], dx[0], dx[1], rms[0], rms[1],
                                   ok.sum())
            utils.log_comment(utils.LOGFILE, lstr, verbose=verbose)

    except:
        utils.log_exception(utils.LOGFILE, traceback)
        utils.log_comment(utils.LOGFILE, "# !! `tweak_flt` tristars failed")

        shift_dict = OrderedDict()
        for i in range(0, len(files)):
            c_i, wcs_i = cats[i]
            # SExtractor doesn't do SIP WCS?
            rd = wcs_i.all_pix2world(c_i['X_IMAGE'], c_i['Y_IMAGE'], 1)
            xy = np.array(wcs_0.all_world2pix(np.array(rd).T, 1))
            N = xy.shape[0]
            dist, ix = np.zeros(N), np.zeros(N, dtype=int)
            for j in range(N):
                dist[j], ix[j] = tree.query(xy[j, :], k=1,
                                            distance_upper_bound=np.inf)

            ok = dist < max_dist
            if ok.sum() == 0:
                shift_dict[files[i]] = [0.0, 0.0, 0.0, 1.0]
                logstr = '# tw {0}'.format(files[i], '! no match')
                utils.log_comment(utils.LOGFILE, logstr, verbose=True)

                continue

            dr = xy - xy_0[ix, :]
            dx = np.median(dr[ok, :], axis=0)
            rms = np.std(dr[ok, :], axis=0)/np.sqrt(ok.sum())

            shift_dict[files[i]] = [dx[0], dx[1], 0.0, 1.0, ok.sum(), rms]

            lstr = '# tw {0} {1} {2} N={3}'
            lstr = logstr.format(files[i], dx, rms, ok.sum())
            utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    wcs_ref = cats[0][1]
    return wcs_ref, shift_dict


def apply_tweak_shifts(wcs_ref, shift_dict, grism_matches={}, verbose=True, log=True):
    """
    Apply derived shifts to exposure WCS
    
    Parameters
    ----------
    wcs_ref : `~astropy.wcs.WCS`
        Reference WCS against where shifts were computed
    
    shift_dict : dict
        Dictionary of shift information
        
        >>> shift_dict[file] = [xshift, yshift, rot, scale]
        
    grism_matches : dict
        Dictionary defining associated grism / direct image exposures
    
    verbose : bool
        Print status information to the console
    
    log : bool
        Log arguments to `grizli.utils.LOGFILE`
    
    Returns
    -------
    Nothing, FLT image WCS are modified in place
    
    """
    from drizzlepac import updatehdr

    if log:
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'prep.apply_tweak_shifts')

    hdu = wcs_ref.to_fits(relax=True)
    file0 = list(shift_dict.keys())[0].split('.fits')[0]
    tweak_file = '{0}_tweak_wcs.fits'.format(file0)
    hdu.writeto(tweak_file, overwrite=True)
    for file in shift_dict:
        xyscale = shift_dict[file][:2]+[0., 1]
        update_wcs_fits_log(file, wcs_ref, xyscale=xyscale, initialize=True,
                            replace=('.fits', '.wcslog.fits'),
                            wcsname='SHIFT')

        updatehdr.updatewcs_with_shift(file, tweak_file,
                                        xsh=shift_dict[file][0],
                                        ysh=shift_dict[file][1],
                                        rot=0., scale=1.,
                                        wcsname='SHIFT', force=True,
                                        reusename=True, verbose=verbose,
                                        sciext='SCI')

        # Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
        # keywords
        with pyfits.open(file, mode='update') as im:
            im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
            im.flush()

        # Update paired grism exposures
        if file in grism_matches:
            for grism_file in grism_matches[file]:
                xyscale = shift_dict[file][:2]+[0., 1]
                update_wcs_fits_log(grism_file, wcs_ref, xyscale=xyscale,
                                    initialize=True,
                                    replace=('.fits', '.wcslog.fits'),
                                    wcsname='SHIFT')

                updatehdr.updatewcs_with_shift(grism_file, tweak_file,
                                              xsh=shift_dict[file][0],
                                              ysh=shift_dict[file][1],
                                              rot=0., scale=1.,
                                              wcsname='SHIFT', force=True,
                                              reusename=True, verbose=verbose,
                                              sciext='SCI')

                # Bug in astrodrizzle?
                with pyfits.open(grism_file, mode='update') as im:
                    im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
                    im.flush()

    os.remove(tweak_file)


def find_direct_grism_pairs(direct={}, grism={}, check_pixel=[507, 507],
                            toler=0.1, key='A', same_visit=True, log=True):
    """Compute pairs of direct and grism exposures
    
    For each grism exposure, check if there is a direct exposure
    that matches the WCS to within `toler` pixels.  If so, copy that WCS
    directly.
    
    Parameters
    ----------
    direct : dict
        Direct image visit dictionary (`product`, `files`)
    
    grism : dict
        Grism image visit dictionary (`product`, `files`)
    
    check_pixel : (float, float)
        Reference pixel to use for comparing WCS
    
    toler : float
        Tolerance in pixels for assigning matched exposure pairs
    
    key : str
        WCS key of the direct image WCS
    
    same_visit : bool
        Require that matches are from same program / visit as defined by the 
        first 6 characters in the image filenames
    
    log : bol
        Write function call to `grizli.utils.LOGFILE`
    
    Returns
    -------
    grism_matches : dict
        Dictionary of the matched exposures, where the keys are filenames 
        of direct images and the values are lists of the computed associated
        grism exposures
        
    """
    if log:
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'prep.find_direct_grism_pairs')

    direct_wcs = {}
    full_direct_wcs = {}
    direct_rd = {}

    grism_wcs = {}
    grism_pix = {}

    grism_matches = OrderedDict()

    for file in direct['files']:
        grism_matches[file] = []
        with pyfits.open(file) as im:
            _h1 = im[1].header
            
            if '_flc' in file:
                direct_wcs[file] = pywcs.WCS(_h1,
                                             fobj=im,
                                             relax=True,
                                             key=key)
                                             
                full_direct_wcs[file] = pywcs.WCS(_h1,
                                                  fobj=im,
                                                  relax=True)
            else:
                direct_wcs[file] = pywcs.WCS(_h1, relax=True, key=key)
                full_direct_wcs[file] = pywcs.WCS(_h1, relax=True)

        direct_rd[file] = direct_wcs[file].all_pix2world([check_pixel], 1)

    if 'files' not in grism:
        return grism_matches

    for file in grism['files']:
        with pyfits.open(file) as im:
            _h1 = im[1].header
            if '_flc' in file:
                grism_wcs[file] = pywcs.WCS(_h1, relax=True, key=key,
                                            fobj=im)
            else:
                grism_wcs[file] = pywcs.WCS(_h1, relax=True, key=key)

        # print file
        delta_min = 10
        for d in direct['files']:
            _test = (os.path.basename(d)[:6] != os.path.basename(file)[:6])
            _test &= same_visit
            
            if _test:
                continue

            pix = grism_wcs[file].all_world2pix(direct_rd[d], 1)
            dx = pix-np.array(check_pixel)
            delta = np.sqrt(np.sum(dx**2))
            # print '  %s %s, %.3f' %(d, dx, delta)
            if delta < delta_min:
                delta_min = delta
                delta_min_file = d
                if delta_min < toler:
                    grism_matches[delta_min_file].append(file)

    return grism_matches


def match_direct_grism_wcs(direct={}, grism={}, get_fresh_flt=True,
                           run_drizzle=True, xyscale=None):
    """Match WCS of grism exposures to corresponding direct images

    Parameters
    ----------
    direct : dict
        Direct image visit dictionary (`product`, `files`)
    
    grism : dict
        Grism image visit dictionary (`product`, `files`)
    
    get_fresh_flt : bool
        Get fresh versions of the grism exposures without any subsequent 
        modifications
    
    run_drizzle : bool
        Not used
    
    xyscale : None, list
        Transformation parameters ``[xshift, yshift, rot, scale]``.  If not 
        specified, then get from the `wcs.log` file associated with the 
        direct images
    
    Returns
    -------
    Nothing, WCS headers updated in the grism FLT files
        
    """
    from drizzlepac import updatehdr
    from stwcs import updatewcs
    from drizzlepac.astrodrizzle import AstroDrizzle

    if get_fresh_flt:
        for file in grism['files']:
            fresh_flt_file(file)
            try:
                updatewcs.updatewcs(file, verbose=False, use_db=False)
            except TypeError:
                updatewcs.updatewcs(file, verbose=False)

    with pyfits.open(direct['files'][0]) as _direct_flt:
        ref_catalog = _direct_flt['SCI', 1].header['WCSNAME']

    # User-defined shifts
    if xyscale is not None:
        # Use user-defined shifts
        xsh, ysh, rot, scale = xyscale

        tmp_wcs_file = '{0}_tmpwcs.fits'.format(str(direct['product']))

        try:
            # Use WCS in catalog file
            wcs_hdu = pyfits.open('{0}.cat.fits'.format(direct['product']))
            ext = 'WCS'
        except:
            wcs_hdu = pyfits.open('{0}_wcs.fits'.format(direct['product']))
            ext = len(wcs_hdu)-1

        wcs_hdu[ext].writeto(tmp_wcs_file, overwrite=True)
        tmp_wcs = pywcs.WCS(wcs_hdu[ext].header, relax=True)

        for file in grism['files']:
            xyscale = [xsh, ysh, rot, scale]
            update_wcs_fits_log(file, tmp_wcs,
                                xyscale=xyscale,
                                initialize=False,
                                replace=('.fits', '.wcslog.fits'),
                                wcsname=ref_catalog)

            updatehdr.updatewcs_with_shift(file, tmp_wcs_file,
                                      xsh=xsh,
                                      ysh=ysh,
                                      rot=rot, scale=scale,
                                      wcsname=ref_catalog, force=True,
                                      reusename=True, verbose=True,
                                      sciext='SCI')

            # Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
            # keywords
            with pyfits.open(file, mode='update') as im:
                im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
                im.flush()
        
        wcs_hdu.close()
        
        if os.path.exists(tmp_wcs_file):
            os.remove(tmp_wcs_file)
            
        return True

    # Get from WCS log file
    wcs_log = Table.read('{0}_wcs.log'.format(direct['product']),
                         format='ascii.commented_header')

    wcs_hdu = pyfits.open('{0}_wcs.fits'.format(direct['product']))

    for ext in wcs_log['ext']:
        tmp_wcs_file = '{0}_tmpwcs.fits'.format(str(direct['product']))
        wcs_hdu[ext].writeto(tmp_wcs_file, overwrite=True)
        tmp_wcs = pywcs.WCS(wcs_hdu[ext].header, relax=True)

        if 'scale' in wcs_log.colnames:
            scale = wcs_log['scale'][ext]
        else:
            scale = 1.

        for file in grism['files']:
            xyscale = [wcs_log['xshift'][ext], wcs_log['yshift'][ext],
                       wcs_log['rot'][ext], scale]

            update_wcs_fits_log(file, tmp_wcs,
                                xyscale=xyscale,
                                initialize=False,
                                replace=('.fits', '.wcslog.fits'),
                                wcsname=ref_catalog)

            updatehdr.updatewcs_with_shift(file, tmp_wcs_file,
                                      xsh=wcs_log['xshift'][ext],
                                      ysh=wcs_log['yshift'][ext],
                                      rot=wcs_log['rot'][ext], scale=scale,
                                      wcsname=ref_catalog, force=True,
                                      reusename=True, verbose=True,
                                      sciext='SCI')

            # Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
            # keywords
            with pyfits.open(file, mode='update') as im:
                im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
                im.flush()
        
        if os.path.exists(tmp_wcs_file):
            os.remove(tmp_wcs_file)
    
    wcs_hdu.close()
      
    # Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
    # keywords
    for file in grism['files']:
        with pyfits.open(file, mode='update') as im:
            im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
            im.flush()


def get_jwst_wfssbkg_file(file, valid_flat=[0.6, 1.3], make_figure=False):
    """
    Divide flat-field from NIRISS wfssbkg file
    
    Parameters
    ----------
    file : str
        Exposure filename
    
    valid_flat : [float,float]
        Range of valid files in the flat-field reference
        
    Returns
    -------
    bkg_file : str
        Path to ``wfssbkg`` file, which may have been corrected by dividing by
        the flat for the given CTX
        
    """
    from jwst.wfss_contam import WfssContamStep
    from jwst.flatfield import FlatFieldStep
    
    bkg_file = WfssContamStep().get_reference_file(file, 'wfssbkg')
    
    # Not a FITS file?  e.g., N/A for imaging exposures
    if 'fits' not in bkg_file:
        return bkg_file
    
    # Only run for NIRISS, seems like NIRCam provides wfssbkg without
    # the flat?
    _im = pyfits.open(file)
    if _im[0].header['INSTRUME'] != 'NIRISS':
        _im.close()
        return bkg_file

    flat_file = FlatFieldStep().get_reference_file(file, 'flat')
    
    wf = pyfits.open(bkg_file, mode='update')
    key = f"{wf[0].header['FILTER']} {wf[0].header['PUPIL']}"
        
    if 'FIXFLAT' in wf[0].header:
        msg = f'Flat {flat_file} already removed from wfssbkg file'
        msg += f' {bkg_file}'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        wf.close()
        return bkg_file
    else:
        msg = f'Divide flat {flat_file} from wfssbkg file {bkg_file}'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

    with pyfits.open(flat_file) as fl:
        fix = (wf[1].data - 0.) / (fl[1].data**1)
        bad = (fl[1].data < valid_flat[0]) | (fl[1].data > valid_flat[1])

    fix /= np.nanmedian(fix)
    bad |= ~np.isfinite(fix)
    fix[bad] = 1
    
    wf[0].header['FIXFLAT'] = True
    wf[0].header['FLATFILE'] = (flat_file,
                                'Flat-field file divided from CRDS file')
    wf[1].data = fix
    wf.flush()
    wf.close()
    
    if make_figure:
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        ax.imshow(fix, vmin=0.9, vmax=1.1, origin='lower')
        ax.text(0.1, 0.1, key,
            transform=ax.transAxes)

    return bkg_file


def visit_grism_sky(grism={}, apply=True, column_average=True, verbose=True, ext=1, sky_iter=10, iter_atol=1.e-4, use_spline=True, NXSPL=50):
    """Subtract sky background from grism exposures

    Implementation of the multi-component grism sky subtraction from 
    `WFC3/ISR 2015-17 <https://ui.adsabs.harvard.edu/abs/2015wfc..rept...17B>`_


    """
    import numpy.ma
    import scipy.ndimage as nd

    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'prep.visit_grism_sky')

    #from sklearn.gaussian_process import GaussianProcess
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    # # Figure out which grism    
    # isJWST = check_isJWST(grism['files'][0])
    # if isJWST:
    #     pupil = im[0].header['PUPIL']
    #     # Set headers back to original
    #     for file in grism['files']:
    #         jwst_utils.reset_jwst_header_keywords(file)
    # else:
    #     pupil = ''
    
    with pyfits.open(grism['files'][0]) as im:
        grism_element = utils.parse_filter_from_header(im[0].header, 
                                                   jwst_detector=True)
    
    isJWST = False
    isACS = False
    
    # Skip NIRCam grism
    if 'GRISM' in grism_element:
        logstr = f'# visit_grism_sky: skip for {grism_element}'
        utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)
        return False
        
    flat = 1.
    if grism_element == 'G141':
        bg_fixed = ['zodi_G141_clean.fits']
        bg_vary = ['zodi_G141_clean.fits', 'excess_lo_G141_clean.fits',
                   'G141_scattered_light.fits'][1:]

    elif grism_element == 'G102':
        bg_fixed = ['zodi_G102_clean.fits']
        bg_vary = ['excess_G102_clean.fits']

    elif grism_element == 'G280':
        bg_fixed = ['UVIS.G280.flat.fits']
        bg_vary = ['UVIS.G280.ext{0:d}.sky.fits'.format(ext)]
        isACS = True
        flat = 1.

    elif len(grism_element.split('-')) > 1:
        # JWST grism_element like F115W-GR150R
        #(grism_element == 'GR150C') & (pupil == 'F115W'):
        # from jwst.wfss_contam import WfssContamStep
        # 
        # wfss_ref =  WfssContamStep().get_reference_file(grism['files'][0], 
        #                                                 'wfssbkg')
        
        # Get the background file, perhaps after correcting it for the flat
        wfss_ref = get_jwst_wfssbkg_file(grism['files'][0])
        
        # GRISM_NIRCAM
        if 'GRISM' in grism_element:
            # Grism background files in GRISM_NIRCAM
            
            _bgpath = os.path.join(GRIZLI_PATH, 'CONF/GRISM_NIRCAM/V2')
            # 'NRCA5-F356W-GRISMR'
            gr = grism_element[-1]
            mod = grism_element[3]
            fi = grism_element.split('-')[-2]
            _bgfile = os.path.join(_bgpath, 
                                   f'{fi}_mod{mod}_{gr}_back.norm.fits.gz')
                        
            if os.path.exists(_bgfile):
                wfss_ref = _bgfile
                
            # TBD
                
        bg_fixed = [wfss_ref]
                                                        
        #bg_fixed = ['jwst_niriss_wfssbkg_0002.fits'] # bg_fixed should be normalized background with flat divided out
        bg_vary = [] 
        flat = 1.
        isJWST = True
        
    elif grism_element == 'G800L':
        bg_fixed = ['ACS.WFC.CHIP{0:d}.msky.1.smooth.fits'.format({1: 2, 2: 1}[ext])]
        bg_vary = ['ACS.WFC.flat.fits']
        #bg_fixed = ['ACS.WFC.CHIP%d.msky.1.fits' %({1:2,2:1}[ext])]
        #bg_fixed = []
        isACS = True

        flat_files = {'G800L': 'n6u12592j_pfl.fits'}  # F814W
        flat_file = flat_files[grism_element]
        flat_im = pyfits.open(os.path.join(os.getenv('jref'), flat_file))
        flat = (flat_im['SCI', ext].data*1).flatten()
        flat_im.close()
        
    logstr = '# visit_grism_sky / {0}: EXTVER={1:d} / {2} / {3}'
    logstr = logstr.format(grism['product'], ext, bg_fixed, bg_vary)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    if not isACS:
        ext = 1

    # Read sky files
    data_fixed = []
    for file in bg_fixed:
        if file.startswith('/'):
            im = pyfits.open(file)
        else:
            im = pyfits.open('{0}/CONF/{1}'.format(GRIZLI_PATH, file))
            
        if isJWST:
            sh = im[1].data.shape
            data = im[1].data.flatten()/flat
            data /= np.nanmedian(data)
        else:
            sh = im[0].data.shape
            data = im[0].data.flatten()/flat
            
        data_fixed.append(data)
        im.close()
        
    data_vary = []
    for file in bg_vary:
        im = pyfits.open('{0}/CONF/{1}'.format(GRIZLI_PATH, file))
        if isJWST:
            data_vary.append(im[1].data.flatten()*1)
            sh = im[1].data.shape
        else:
            data_vary.append(im[0].data.flatten()*1)
            sh = im[0].data.shape
        im.close()
        
    yp, xp = np.indices(sh)

    Npix = sh[0]*sh[1]
    Nexp = len(grism['files'])
    Nfix = len(data_fixed)
    Nvary = len(data_vary)
    Nimg = Nexp*Nvary + Nfix

    A = np.zeros((Npix*Nexp, Nimg), dtype=np.float32)
    data = np.zeros(Npix*Nexp, dtype=np.float32)
    wht = data*0.
    mask = data > -1
    medians = np.zeros(Nexp)
    exptime = np.ones(Nexp)

    # Build combined arrays
    if isACS:
        bits = 64+32
    elif isJWST:
        bits = 1+4+6+32768 #+16777200+1049600+26232800+9438210+9438220
    else:
        bits = 576

    for i in range(Nexp):
        flt = pyfits.open(grism['files'][i])
        dq = utils.unset_dq_bits(flt['DQ', ext].data, okbits=bits)
        dq_mask = dq == 0

        # Data
        data[i*Npix:(i+1)*Npix] = (flt['SCI', ext].data*dq_mask).flatten()
        mask[i*Npix:(i+1)*Npix] &= dq_mask.flatten()  # == 0
        wht[i*Npix:(i+1)*Npix] = 1./(flt['ERR', ext].data**2*dq_mask).flatten()
        wht[~np.isfinite(wht)] = 0.

        if isACS:
            exptime[i] = flt[0].header['EXPTIME']
            data[i*Npix:(i+1)*Npix] /= exptime[i]
            wht[i*Npix:(i+1)*Npix] *= exptime[i]**2

            medians[i] = np.median(flt['SCI', ext].data[dq_mask]/exptime[i])
        else:
            medians[i] = np.median(flt['SCI', ext].data[dq_mask])

        # Fixed arrays
        for j in range(Nfix):
            for k in range(Nexp):
                A[k*Npix:(k+1)*Npix, j] = data_fixed[j]

            mask_j = (data_fixed[j] > 0) & np.isfinite(data_fixed[j])
            mask[i*Npix:(i+1)*Npix] &= mask_j

        # Variable arrays
        for j in range(Nvary):
            k = Nfix+j+Nvary*i
            A[i*Npix:(i+1)*Npix, k] = data_vary[j]
            mask[i*Npix:(i+1)*Npix] &= np.isfinite(data_vary[j])
        
        flt.close()
        
    # Initial coeffs based on image medians
    coeffs = np.array([np.min(medians)])
    if Nvary > 0:
        coeffs = np.hstack((coeffs, np.zeros(Nexp*Nvary)))
        coeffs[1::Nvary] = medians-medians.min()

    model = np.dot(A, coeffs)

    coeffs_0 = coeffs

    for iter in range(sky_iter):
        model = np.dot(A, coeffs)
        resid = (data-model)*np.sqrt(wht)
        obj_mask = (resid < 2.5) & (resid > -3)
        for j in range(Nexp):
            obj_j = nd.minimum_filter(obj_mask[j*Npix:(j+1)*Npix], size=30)
            obj_mask[j*Npix:(j+1)*Npix] = (obj_j > 0).flatten()

        logstr = '# visit_grism_sky   {0} > Iter: {1:d}, masked: {2:2.0f}%, {3}'
        logstr = logstr.format(grism['product'], iter+1, obj_mask.sum()/Npix/Nimg*100, coeffs)
        utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

        out = np.linalg.lstsq(A[mask & obj_mask, :], data[mask & obj_mask], 
                              rcond=utils.LSTSQ_RCOND)
        coeffs = out[0]

        # Test for convergence
        if np.allclose(coeffs, coeffs_0, rtol=1.e-5, atol=iter_atol):
            break
        else:
            coeffs_0 = coeffs

    # Best-fit sky
    sky = np.dot(A, coeffs).reshape(Nexp, Npix)

    # log file
    fp = open('{0}_{1}_sky_background.info'.format(grism['product'], ext), 'w')
    fp.write('# file c1 {0}\n'.format(' '.join(['c{0:d}'.format(v+2)
                                            for v in range(Nvary)])))

    fp.write('# {0}\n'.format(grism['product']))

    fp.write('# bg1: {0}\n'.format(bg_fixed[0]))
    for v in range(Nvary):
        fp.write('# bg{0:d}: {1}\n'.format(v+2, bg_vary[v]))

    for j in range(Nexp):
        file = grism['files'][j]
        line = '{0} {1:9.4f}'.format(file, coeffs[0])
        for v in range(Nvary):
            k = Nfix + j*Nvary + v
            line = '{0} {1:9.4f}'.format(line, coeffs[k])

        fp.write(line+'\n')

    fp.close()

    if apply:
        for j in range(Nexp):
            file = grism['files'][j]

            flt = pyfits.open(file, mode='update')
            flt['SCI', ext].data -= sky[j, :].reshape(sh)*exptime[j]

            header = flt[0].header
            header['GSKYCOL{0:d}'.format(ext)] = (False,
                                                  'Subtract column average')
            header['GSKYN{0:d}'.format(ext)] = (Nfix+Nvary,
                                                'Number of sky images')
            header['GSKY{0:d}01'.format(ext)] = (coeffs[0],
                                'Sky image {0} (fixed)'.format(bg_fixed[0]))

            header['GSKY{0:d}01F'.format(ext)] = (bg_fixed[0],
                                                  'Sky image (fixed)')

            for v in range(Nvary):
                k = Nfix + j*Nvary + v
                # print coeffs[k]
                header['GSKY{0}{1:02d}'.format(ext, v+Nfix+1)] = (coeffs[k],
                                'Sky image {0} (variable)'.format(bg_vary[v]))

                header['GSKY{0}{1:02d}F'.format(ext, v+Nfix+1)] = (bg_vary[v],
                                                      'Sky image (variable)')

            flt.flush()
            flt.close()
            
    # Don't do `column_average` for ACS
    if (not column_average) | isACS:
        return isACS

    ######
    # Now fit residual column average & make diagnostic plot
    interactive_status = plt.rcParams['interactive']
    plt.ioff()

    fig = plt.figure(figsize=[6., 6.])
    ax = fig.add_subplot(111)

    if isJWST:
        im_shape = (2048, 2048)
    else:
        im_shape = (1014, 1014)

    for j in range(Nexp):

        file = grism['files'][j]

        resid = (data[j*Npix:(j+1)*Npix] - sky[j, :]).reshape(im_shape)
        m = (mask & obj_mask)[j*Npix:(j+1)*Npix].reshape(im_shape)

        rot90 = grismconf.JwstDispersionTransform(header=flt[0].header).rot90
        if rot90 % 2 == 0:
            avg_axis = 0
            col_label = 'column'
        else:
            avg_axis = 1
            col_label = 'row'
            
        # Statistics of masked arrays
        ma = np.ma.masked_array(resid, mask=(~m))
        med = np.ma.median(ma, axis=avg_axis)

        bg_sky = 0
        yrms = np.ma.std(ma, axis=avg_axis)/np.sqrt(np.sum(m, axis=avg_axis))
        xmsk = np.arange(im_shape[avg_axis])
        yres = med
        yok = (~yrms.mask) & np.isfinite(yrms) & np.isfinite(xmsk) & np.isfinite(yres)

        if yok.sum() == 0:
            print('ERROR: No valid pixels found!')
            continue

        # Fit column average with smoothed Gaussian Process model
        # if False:
        #     #### xxx old GaussianProcess implementation
        #     gp = GaussianProcess(regr='constant', corr='squared_exponential',
        #                          theta0=8, thetaL=5, thetaU=12,
        #                          nugget=(yrms/bg_sky)[yok][::1]**2,
        #                          random_start=10, verbose=True, normalize=True)
        #
        #     try:
        #         gp.fit(np.atleast_2d(xmsk[yok][::1]).T, yres[yok][::1]+bg_sky)
        #     except:
        #         warn = '# visit_grism_sky / GaussianProces failed!\n# visit_grism_sky / Check that this exposure wasn\'t fried by variable backgrounds.'
        #         print(warn)
        #         utils.log_exception(utils.LOGFILE, traceback)
        #         utils.log_comment(utils.LOGFILE, warn)
        #
        #         continue
        #
        #     y_pred, MSE = gp.predict(np.atleast_2d(xmsk).T, eval_MSE=True)
        #     gp_sigma = np.sqrt(MSE)

        if use_spline:
            # Fit with Spline basis functions

            #NXSPL = 50
            xpad = np.arange(-1*NXSPL, im_shape[avg_axis]+1*NXSPL)

            Aspl = utils.bspline_templates(xpad, degree=3,
                                           df=4+im_shape[avg_axis]//NXSPL,
                                           get_matrix=True, log=False,
                                           clip=0.0001)[1*NXSPL:-1*NXSPL, :]

            Ax = (Aspl.T/yrms).T
            cspl, _, _, _ = np.linalg.lstsq(Ax, (yres+bg_sky)/yrms, 
                                            rcond=utils.LSTSQ_RCOND)
            y_pred = Aspl.dot(cspl)

            try:
                ND = 100
                #covar = np.matrix(np.dot(Ax.T, Ax)).I.A
                covar = utils.safe_invert(np.dot(Ax.T, Ax))
                draws = np.random.multivariate_normal(cspl, covar, ND)
                gp_sigma = np.std(Aspl.dot(draws.T), axis=1)
            except:
                gp_sigma = y_pred*0.

        else:
            # Updated sklearn GaussianProcessRegressor
            nmad_y = utils.nmad(yres)

            gpscl = 100  # rough normalization
            k1 = 0.3**2 * RBF(length_scale=80)  # Background variations
            k2 = 1**2 * WhiteKernel(noise_level=(nmad_y*gpscl)**2)  # noise
            gp_kernel = k1+k2  # +outliers

            yok &= np.abs(yres-np.median(yres)) < 50*nmad_y

            gp = GaussianProcessRegressor(kernel=gp_kernel,
                                          alpha=nmad_y*gpscl/5,
                                          optimizer='fmin_l_bfgs_b',
                                          n_restarts_optimizer=0,
                                          normalize_y=False,
                                          copy_X_train=True,
                                          random_state=None)

            gp.fit(np.atleast_2d(xmsk[yok][::1]).T,
                   (yres[yok][::1]+bg_sky)*gpscl)

            y_pred, gp_sigma = gp.predict(np.atleast_2d(xmsk).T,
                                          return_std=True)
            gp_sigma /= gpscl
            y_pred /= gpscl

        # Plot Results
        pi = ax.plot(med, alpha=0.1, zorder=-100)
        ax.plot(y_pred-bg_sky, color=pi[0].get_color())
        ax.fill_between(xmsk, y_pred-bg_sky-gp_sigma, y_pred-bg_sky+gp_sigma,
                        color=pi[0].get_color(), alpha=0.3,
                        label=grism['files'][j].split('_fl')[0])

        # result
        fp = open(file.replace('_flt.fits', '_column.dat').replace('_rate.fits', '_column.dat'), 'wb')
        fp.write(b'# column obs_resid ok resid uncertainty\n')
        np.savetxt(fp, np.array([xmsk, yres, yok*1, y_pred-bg_sky, gp_sigma]).T, fmt='%.5f')
        fp.close()

        if apply:
            # Subtract the column average in 2D & log header keywords
            flt = pyfits.open(file, mode='update')
            imshape = flt[1].data.shape[0]
            gp_res = np.dot(y_pred[:, None]-bg_sky, np.ones((imshape, 1)).T).T             
            if rot90 % 2 == 1:
                gp_res = np.rot90(gp_res, 1) # test this to see if it looks correct, if not try 3
            flt['SCI', 1].data -= gp_res
            flt[0].header['GSKYCOL'] = (True, f'Subtract {col_label} average')
            flt[0].header['GSKYCAX'] = (avg_axis, 'Array axis for average')
            
            flt.flush()
            flt.close()
            
    # Finish plot
    ax.legend(loc='lower left', fontsize=10)
    ax.plot([-10, im_shape[0]+10], [0, 0], color='k')
    ax.set_xlim(-10, im_shape[0]+10)
    if avg_axis == 0:
        ax.set_xlabel(r'pixel column ($x$)')
    else:
        ax.set_xlabel(r'pixel row ($y$)')
        
    ax.set_ylabel(f'{col_label} average (e-/s)')
    ax.set_title(grism['product'])
    ax.grid()

    fig.tight_layout(pad=0.1)
    fig.savefig('{0}_column.png'.format(grism['product']))
    #fig.savefig('%s_column.pdf' %(grism['product']))
    plt.close()

    # Clean up large arrays
    del(data)
    del(A)
    del(wht)
    del(mask)
    del(model)

    if interactive_status:
        plt.ion()

    return False


def fix_star_centers(root='macs1149.6+2223-rot-ca5-22-032.0-f105w',
                     mag_lim=22, verbose=True, drizzle=False,
                     cutout_size=16):
    """Unset the CR bit (4096) in the centers of bright objects

    Parameters
    ----------
    root : str
        Root name of drizzle product (direct imaging).

    mag_lim : float
        Magnitude limit of objects to consider

    verbose : bool
        Print messages to the terminal

    drizzle : bool
        Redrizzle the output image

    cutout_size : int
        Size of the cutout to extract around the bright stars

    Returns
    -------
    Nothing, updates FLT files in place.

    """
    frame = inspect.currentframe()
    utils.log_function_arguments(utils.LOGFILE, frame,
                                 'prep.fix_star_centers')

    from drizzlepac.astrodrizzle import AstroDrizzle

    EPSF = utils.EffectivePSF()

    sci = pyfits.open('{0}_drz_sci.fits'.format(root))
    #cat = Table.read('{0}.cat'.format(root), format='ascii.commented_header')
    cat = utils.GTable.gread('{0}.cat.fits'.format(root))

    # Load FITS files
    N = sci[0].header['NDRIZIM']
    images = []
    wcs = []
    for i in range(N):
        flt = pyfits.open(sci[0].header['D{0:03d}DATA'.format(i+1)].split('[')[0], mode='update')
        # if True:
        #     flt = pyfits.open('../RAW/'+sci[0].header['D{0:03d}DATA'.format(i+1)].split('[')[0], mode='update')
        wcs.append(pywcs.WCS(flt[1], relax=True))
        images.append(flt)
    isJWST = check_isJWST('{0}_drz_sci.fits'.format(root))
    if isJWST:
        im_shape = (2048, 2048)
    else:
        im_shape = (1014, 1014)
    yp, xp = np.indices(im_shape)
    use = cat['MAG_AUTO'] < mag_lim
    so = np.argsort(cat['MAG_AUTO'][use])

    if verbose:
        print('# {0:6s} {1:12s} {2:12s} {3:7s} {4}     {5}'.format('id', 'ra',
                                                             'dec', 'mag',
                                                             'nDQ', 'nSat'))

    for line in cat[use][so]:
        rd = line['X_WORLD'], line['Y_WORLD']
        nset = []
        nsat = []
        for i in range(N):
            xi, yi = wcs[i].all_world2pix([rd[0], ], [rd[1], ], 0)
            r = np.sqrt((xp-xi[0])**2 + (yp-yi[0])**2)
            unset = (r <= 3) & ((images[i]['DQ'].data & 4096) > 0)
            nset.append(unset.sum())
            if nset[i] > 0:
                images[i]['DQ'].data[unset] -= 4096

            # Fill saturated with EPSF fit
            satpix = (r <= 10) & (((images[i]['DQ'].data & 256) > 0) | ((images[i]['DQ'].data & 2048) > 0))
            nsat.append(satpix.sum())

            if nsat[i] > 0:
                xpi = int(np.round(xi[0]))
                ypi = int(np.round(yi[0]))

                slx = slice(xpi-cutout_size, xpi+cutout_size)
                sly = slice(ypi-cutout_size, ypi+cutout_size)

                sci = images[i]['SCI'].data[sly, slx]
                dq = images[i]['DQ'].data[sly, slx]
                dqm = dq - (dq & 2048)
                err = images[i]['ERR'].data[sly, slx]
                mask = satpix[sly, slx]

                ivar = 1/err**2
                ivar[(~np.isfinite(ivar)) | (dqm > 0)] = 0

                # Fit the EPSF model
                try:
                    psf_filter = images[0][0].header['FILTER']
                    Np = 15
                    guess = [cutout_size-1, cutout_size-1]
                    #guess = None
                    tol = 1.e-3

                    psf_params = EPSF.fit_ePSF(sci, ivar=ivar, center=None,
                                               tol=tol, N=Np,
                                    origin=(ypi-cutout_size, xpi-cutout_size),
                                    filter=psf_filter, get_extended=True,
                                    method='Powell', only_centering=True,
                                    guess=guess, psf_params=None)

                    result = EPSF.fit_ePSF(sci, ivar=ivar, center=None,
                                               tol=tol, N=Np,
                                    origin=(ypi-cutout_size, xpi-cutout_size),
                                    filter=psf_filter, get_extended=True,
                                    method='Powell', only_centering=True,
                                    guess=guess, psf_params=psf_params)

                    psf, psf_bkg, psfA, psf_coeffs = result

                    # psf = EPSF.get_ePSF(psf_params,
                    #                 origin=(ypi-cutout_size, xpi-cutout_size),
                    #                 shape=sci.shape, filter=psf_filter,
                    #                 get_extended=True)

                    # if i == 0:
                    #     break
                except:
                    continue

                sci[mask] = psf[mask]
                dq[mask] -= (dq[mask] & 2048)
                #dq[mask] -= (dq[mask] & 256)
                #dq[mask] |= 512

        if verbose:
            print('{0:6d} {1:12.6f} {2:12.6f} {3:7.2f} {4} {5}'.format(
                line['NUMBER'], rd[0], rd[1], line['MAG_AUTO'], nset, nsat))

    # Overwrite image
    for i in range(N):
        images[i].flush()

    if drizzle:
        files = [flt.filename() for flt in images]

        bits = 576

        if root.startswith('par'):
            pixfrac = 1.0
        else:
            pixfrac = 0.8

        # Fix Nans:
        for flt_file in files:
            utils.fix_flt_nan(flt_file, bad_bit=4096, verbose=True)

        AstroDrizzle(files, output=root,
                     clean=True, final_pixfrac=pixfrac, context=False,
                     resetbits=0, final_bits=bits, driz_sep_bits=bits,
                     preserve=False, driz_separate=False,
                     driz_sep_wcs=False, median=False, blot=False,
                     driz_cr=False, driz_cr_corr=False, build=False,
                     final_wht_type='IVM')

        clean_drizzle(root)
        #cat = make_drz_catalog(root=root)
        cat = make_SEP_catalog(root=root)


def find_single_image_CRs(visit, simple_mask=False, with_ctx_mask=True,
                          run_lacosmic=True):
    """Use LACosmic to find CRs in parts of an ACS mosaic where only one
    exposure was available

    Parameters
    ----------
    visit : dict
        List of visit information from `~grizli.utils.parse_flt_files`.

    simple_mask : bool
        If true, set 1024 CR bit for all parts of a given FLT where it does
        not overlap with any others in the visit.  If False, then run
        LACosmic to flag CRs in this area but keep the pixels.

    run_lacosmic : bool
        Run LA Cosmic.

    Requires context (CTX) image `visit['product']+'_drc_ctx.fits`.
    """
    from drizzlepac import astrodrizzle
    from astroscrappy import detect_cosmics

    # try:
    #     import reproject
    #     HAS_REPROJECT = True
    # except:
    #     HAS_REPROJECT = False
    HAS_REPROJECT = False

    ctx_files = glob.glob(visit['product']+'_dr?_ctx.fits')
    has_ctx = len(ctx_files) > 0
    if has_ctx:
        ctx = pyfits.open(ctx_files[0])
        bits = np.log2(ctx[0].data)
        mask = ctx[0].data == 0
        #single_image = np.cast[np.float32]((np.cast[int](bits) == bits) & (~mask))
        single_image = np.cast[np.float]((np.cast[int](bits) == bits) & (~mask))
        ctx_wcs = pywcs.WCS(ctx[0].header)
        ctx_wcs.pscale = utils.get_wcs_pscale(ctx_wcs)
        ctx.close()
    else:
        simple_mask = False
        with_ctx_mask = False

    for file in visit['files']:
        flt = pyfits.open(file, mode='update')

        # WFPC2
        if '_c0' in file:
            dq_hdu = pyfits.open(file.replace('_c0', '_c1'), mode='update')
            dq_extname = 'SCI'
        else:
            dq_hdu = flt
            dq_extname = 'DQ'

        for ext in [1, 2, 3, 4]:

            if ('SCI', ext) not in flt:
                continue

            flt_wcs = pywcs.WCS(flt['SCI', ext].header, fobj=flt, relax=True)
            flt_wcs.pscale = utils.get_wcs_pscale(flt_wcs)

            if has_ctx:
                blotted = utils.blot_nearest_exact(single_image, ctx_wcs,
                                                   flt_wcs)

                ctx_mask = blotted > 0
            else:
                ctx_mask = np.zeros(flt['SCI', ext].data.shape, dtype=bool)

            sci = flt['SCI', ext].data
            dq = dq_hdu[dq_extname, ext].data

            if simple_mask:
                print('{0}: Mask image without overlaps, extension {1:d}'.format(file, ext))
                dq[ctx_mask] |= 1024
            else:
                print('{0}: Clean CRs with LACosmic, extension {1:d}'.format(file, ext))

                if with_ctx_mask:
                    inmask = blotted == 0
                else:
                    inmask = dq > 0

                if run_lacosmic:
                    crmask, clean = detect_cosmics(sci, inmask=inmask,
                             sigclip=4.5, sigfrac=0.3, objlim=5.0, gain=1.0,
                             readnoise=6.5, satlevel=65536.0, pssl=0.0,
                             niter=4, sepmed=True, cleantype='meanmask',
                             fsmode='median', psfmodel='gauss',
                             psffwhm=2.5, psfsize=7, psfk=None, psfbeta=4.765,
                             verbose=False)
                else:
                    crmask = ctx_mask

                if with_ctx_mask:
                    dq[crmask & ctx_mask] |= 1024
                else:
                    dq[crmask] |= 1024

                #sci[crmask & ctx_mask] = 0

        flt.flush()
        flt.close()


def clean_amplifier_residuals(files, extensions=[1,2], minpix=5e5, max_percentile=99, seg_hdu=None, skip=10, polynomial_degree=3, verbose=True, imsh_kwargs={'vmin':-1.e-3, 'vmax':1.e-3, 'cmap':'magma'}):
    """
    Fit and remove a 2D polynomial fit to the detector-frame UVIS/WFC images
    
    Parameters
    ----------
    files : list
        List of FLC files

    extensions : list
        List of extensions to consider (ACS and UVIS have two).  Extensions
        will be addressed as ``im['SCI',ext]``.

    minpix : int
        Minimum number of unmasked pixels required to perform the fit

    seg_hdu : `~astropy.fits.ImageHDU`
        Optional HDU defining a mask for the individual exposures.  Will 
        be blotted to the FLC frame and valid pixels are taken to be where
        seg_hdu.data == 0 (e.g., a segmentation image).

    skip : int
        Pass every ``skip`` (unmasked) pixel to the polynomial fit.

    polynomial_degree : int
        Order of the `~astropy.models.Polynomial2D` model to use.

    verbose : bool
        Print status information

    imsh_kwargs : dict
        Args to pass to `~matplotlib.pyplot.imshow` for the figure

    """
    from astropy.modeling.fitting import LinearLSQFitter
    from astropy.modeling.models import Polynomial2D
    from matplotlib.ticker import MultipleLocator
    
    if len(files) == 0:
        print('No files specified')
        return False
        
    if files[0].startswith('j'):
        # ACS WFC
        sh = (2048, 4096)
    else:
        sh = (2051, 4096)

    yp, xp = np.indices(sh, dtype=np.float32)
    yp /= sh[0]+1
    xp /= sh[1]+1

    num = np.zeros(sh, dtype=np.float32)
    den = np.zeros(sh, dtype=np.float32)

    if seg_hdu is not None:
        seg_wcs = pywcs.WCS(seg_hdu.header)

    ims = [pyfits.open(file, mode='update') for file in files]

    n_ext = len(extensions)
    fig, axes = plt.subplots(nrows=n_ext, ncols=1, figsize=(4, 2*n_ext), 
                             sharex=True)

    for ext in extensions:
        for im in ims:
            wht = 1/im['ERR',ext].data**2*(im['DQ',ext].data == 0)
            valid = np.isfinite(wht) & np.isfinite(im['SCI',ext].data)

            if verbose:
                print(f'   Process {im.filename()}[{ext}]')

            if seg_hdu is not None:
                flc_wcs = pywcs.WCS(im['SCI',ext].header, fobj=im)
                _blt = utils.blot_nearest_exact(seg_hdu.data, seg_wcs, 
                                                flc_wcs, verbose=False, 
                                                stepsize=-1, 
                                                scale_by_pixel_area=False, 
                                                wcs_mask=True, fill_value=0)

                valid &= (_blt == 0)

            wht[~valid] = 0
            if 'MDRIZSKY' in im['SCI',ext].header:
                bkg = im['SCI',ext].header['MDRIZSKY']
            else:
                bkg = np.median(im['SCI', ext].data[valid])

            _sci = (im['SCI',ext].data - bkg)/im[0].header['EXPTIME']
            num += _sci*wht
            den += wht

        avg = num/den
        avg[den == 0] = 0

        # Amps in detector middle
        poly = Polynomial2D(polynomial_degree)
        fitter = LinearLSQFitter()

        quad_model = np.zeros(sh, dtype=np.float32)

        _h = pyfits.Header()
        _h['QORDER'] = polynomial_degree, 'Quad polynomial degree'

        for q in [1,2]:
            quad = (xp >= (q-1)/2.) & (xp < q/2.) 
            clip = quad & (den > 0)
            limit = np.percentile(avg[clip], max_percentile)
            clip &= avg < limit
            
            if clip.sum() < minpix:
                print('Warning: not enough pixels found for ext:{ext} q:{q}')
                continue
                
            _fit = fitter(poly, xp[clip][::skip], yp[clip][::skip], 
                          avg[clip][::skip]) 

            quad_model[quad] = _fit(xp[quad], yp[quad])
            for name, val in zip(_fit.param_names, _fit.parameters):
                _h[f'Q{q}_{name}'] = (val, 'Quad polynomial component')

        axes[::-1][ext-1].imshow(quad_model, **imsh_kwargs)
        axes[::-1][ext-1].text(0.05, 0.95, f'Ext {ext}', ha='left', va='top', 
                               color='w', fontsize=10, 
                               transform=axes[::-1][ext-1].transAxes)

        axes[::-1][ext-1].text(0.05, 0.05, f'Q1', ha='left', va='bottom', 
                               color='w', fontsize=8, 
                               transform=axes[::-1][ext-1].transAxes)

        axes[::-1][ext-1].text(0.55, 0.05, f'Q2', ha='left', va='bottom', 
                               color='w', fontsize=8, 
                               transform=axes[::-1][ext-1].transAxes)

        for im in ims:
            expt = im[0].header['EXPTIME']
            im['SCI',ext].data -= quad_model*expt
            
            for key, comment in zip(_h.keys(), _h.comments):
                if key in im['SCI', ext].header:
                    # Already exists, add polynomial component
                    im['SCI',ext].header[key] += _h[key]
                else:
                    im['SCI',ext].header[key] = (_h[key], comment)

    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(1024))
        ax.xaxis.set_minor_locator(MultipleLocator(256))
        ax.yaxis.set_major_locator(MultipleLocator(1024))
        ax.yaxis.set_minor_locator(MultipleLocator(256))

    fig.tight_layout(pad=0.5)

    for im in ims:
        im.flush()
        im.close()
        
    return fig


def drizzle_overlaps(exposure_groups, parse_visits=False, check_overlaps=True, max_files=999, pixfrac=0.8, scale=0.06, skysub=True, skymethod='localmin', skyuser='MDRIZSKY', bits=None, build=False, final_wcs=True, final_rot=0, final_outnx=None, final_outny=None, final_ra=None, final_dec=None, final_wht_type='EXP', final_wt_scl='exptime', final_kernel='square', context=False, static=True, use_group_footprint=False, fetch_flats=True, fix_wcs_system=False, include_saturated=False, run_driz_cr=False, driz_cr_snr=None, driz_cr_scale=None, resetbits=0, driz_cr_snr_grow=1, driz_cr_scale_grow=1, log=False, **kwargs):
    """Combine overlapping visits into single output mosaics

    Parameters
    ----------
    exposure_groups : list
        Output list of visit information from `~grizli.utils.parse_flt_files`.

    parse_visits : bool
        If set, parse the `exposure_groups` list for overlaps with
        `~grizli.utils.parse_visit_overlaps`, otherwise assume that it has
        already been parsed.

    check_overlaps: bool
        Only pass exposures that overlap with the desired output mosaic to
        AstroDrizzle.

    max_files : bool
        Split output products if the number of exposures in a group is greater
        than `max_files`.  Default value of 999 appropriate for AstroDrizzle,
        which crashes because it tries to create a header keyword with only
        three digits (i.e., 0-999).

    pixfrac : float
        `~drizzlepac.astrodrizzle.AstroDrizzle` "pixfrac" value.

    scale : type
        `~drizzlepac.astrodrizzle.AstroDrizzle` "scale" value, output pixel
        scale in `~astropy.units.arcsec`.

    skysub : bool
        Run `~drizzlepac.astrodrizzle.AstroDrizzle` sky subtraction.

    bits : None or int
        Data quality bits to treat as OK.  If None, then default to 64+32 for
        ACS and 512+64 for WFC3/IR.

    final_* : Parameters passed through to AstroDrizzle to define output WCS
        Note that these are overridden if an exposure group has a 'reference'
        keyword pointing to a reference image / WCS.

    Returns
    -------
    Produces drizzled images.

    """
    if log:
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'prep.drizzle_overlaps')

    from drizzlepac.astrodrizzle import AstroDrizzle
    from shapely.geometry import Polygon

    if parse_visits:
        exposure_groups = utils.parse_visit_overlaps(exposure_groups, buffer=15.)

    # Drizzle can only handle 999 files at a time
    if check_overlaps:
        for group in exposure_groups:
            if 'reference' not in group:
                continue

            if 'footprints' in group:
                footprints = group['footprints']
            elif ('footprint' in group) & use_group_footprint:
                footprints = [group['footprint']]*len(group['files'])
            else:
                footprints = []
                files = group['files']
                for i in range(len(files)):
                    print(i, files[i])
                    im = pyfits.open(files[i])
                    p_i = None
                    for ext in [1, 2, 3, 4]:
                        if ('SCI', ext) in im:
                            wcs = pywcs.WCS(im['SCI', ext], fobj=im)
                            fp_x = wcs.calc_footprint()
                            if p_i is None:
                                p_i = Polygon(fp_x)
                            else:
                                p_i = p_i.union(fp_x)
                    
                    im.close()
                    footprints.append(p_i)

            ref = pyfits.getheader(group['reference'])
            wcs = pywcs.WCS(ref)
            ref_fp = Polygon(wcs.calc_footprint())

            files = []
            out_fp = []

            if 'awspath' in group:
                aws = []

            for j in range(len(group['files'])):
                olap = ref_fp.intersection(footprints[j])
                if olap.area > 0:
                    files.append(group['files'][j])
                    if 'awspath' in group:
                        aws.append(group['awspath'][j])

                    out_fp.append(footprints[j])

            print(group['product'], len(files), len(group['files']))
            group['files'] = files
            group['footprints'] = out_fp
            if 'awspath' in group:
                group['awspath'] = aws

            # Download the file from aws.  The 'awspath' entry
            # is a list with the same length of 'files', and starts with
            # the bucket name.
            if 'awspath' in group:
                import boto3
                session = boto3.Session()
                s3 = boto3.resource('s3')

                bkt = None
                for awspath, file in zip(group['awspath'], group['files']):
                    if os.path.exists(file):
                        continue

                    spl = awspath.split('/')
                    bucket_name = spl[0]
                    path_to_file = '/'.join(spl[1:])

                    if bkt is None:
                        bkt = s3.Bucket(bucket_name)
                    else:
                        if bkt.name != bucket_name:
                            bkt = s3.Bucket(bucket_name)

                    s3_file = (path_to_file+'/'+file).replace('//', '/')
                    print('Fetch from s3:  s3://{0}/{1}'.format(bucket_name, s3_file))
                    bkt.download_file(s3_file, file,
                                    ExtraArgs={"RequestPayer": "requester"})

    if max_files > 0:
        all_groups = []
        for group in exposure_groups:
            N = len(group['files']) // int(max_files) + 1
            if N == 1:
                all_groups.append(group)
            else:
                for k in range(N):
                    sli = slice(k*max_files, (k+1)*max_files)
                    files_list = group['files'][sli]
                    root = '{0}-{1:03d}'.format(group['product'], k)
                    g_k = OrderedDict(product=root,
                                      files=files_list,
                                      reference=group['reference'])

                    if 'footprints' in group:
                        g_k['footprints'] = group['footprints'][sli]

                    all_groups.append(g_k)

    else:
        all_groups = exposure_groups

    for group in all_groups:
        if len(group['files']) == 0:
            continue

        isACS = '_flc' in group['files'][0]
        isWFPC2 = '_c0' in group['files'][0]
        isJWST = os.path.basename(group['files'][0]).startswith('jw')
        
        if (driz_cr_snr is None) | (driz_cr_scale is None):
            if isACS:
                driz_cr_snr = '3.5 3.0'
                driz_cr_scale = '1.2 0.7'
            elif isWFPC2:
                driz_cr_snr = '3.5 3.0'
                driz_cr_scale = '1.2 0.7'
            else:
                driz_cr_snr = '8.0 5.0'
                driz_cr_scale = '2.5 0.7'
            
            if driz_cr_snr_grow != 1:
                spl = driz_cr_snr.split()
                new_snr = np.cast[float](spl)*driz_cr_snr_grow
                driz_cr_snr = ' '.join([f'{val:.2f}' for val in new_snr])
            
            if driz_cr_scale_grow != 1:
                spl = driz_cr_scale.split()
                new_scale = np.cast[float](spl)*driz_cr_scale_grow
                driz_cr_scale = ' '.join([f'{val:.2f}' for val in new_scale])
            
        if bits is None:
            if isACS | isWFPC2:
                bits = 64+32
            elif isJWST:
                bits = 1+4
            else:
                bits = 576

            if include_saturated:
                bits |= 256

        # All the same instrument?
        inst_keys = np.unique([os.path.basename(file)[0]
                               for file in group['files']])

        print('\n\n### drizzle_overlaps: {0} ({1})\n'.format(group['product'],
                                                     len(group['files'])))
        
        if isJWST:
            # Make sure HST-like keywords are set for JWST
            for file in group['files']:
                _ = jwst_utils.set_jwst_to_hst_keywords(file, reset=False)
                                                             
        if fetch_flats:
            # PFL files needed for IVM weights
            for file in group['files']:
                try:
                    if isWFPC2:
                        im = pyfits.open(file)
                        flat_file = im[0].header['FLATFILE'].strip('uref$')
                        utils.fetch_wfpc2_calib(file=flat_file,
                                                path=os.getenv('uref'),
                                                use_mast=False, verbose=True,
                                                overwrite=True)
                        im.close()
                    elif isJWST:
                        pass
                    else:
                        utils.fetch_hst_calibs(file, calib_types=['PFLTFILE'],
                                       verbose=False)
                except:
                    utils.log_exception(utils.LOGFILE, traceback)
        
        if 'gain' in kwargs:
            gain = kwargs['gain']
        elif isJWST:
            gain = '1.0'
        else:
            gain = None
            
        if 'rdnoise' in kwargs:
            rdnoise = kwargs['rdnoise']
        elif isJWST:
            rdnoise = '0.0'
        else:
            rdnoise = None
            
        # Fetch files from aws
        if 'reference' in group:
            AstroDrizzle(group['files'], output=group['product'],
                     clean=True, context=context, preserve=False,
                     skysub=skysub, skyuser=skyuser, skymethod=skymethod,
                     driz_separate=run_driz_cr, driz_sep_wcs=run_driz_cr,
                     median=run_driz_cr, blot=run_driz_cr,
                     driz_cr=run_driz_cr,
                     driz_cr_snr=driz_cr_snr, driz_cr_scale=driz_cr_scale,
                     driz_cr_corr=False, driz_combine=True,
                     final_bits=bits, coeffs=True, build=build,
                     final_wht_type=final_wht_type,
                     final_wt_scl=final_wt_scl,
                     final_pixfrac=pixfrac,
                     final_wcs=True, final_refimage=group['reference'],
                     final_kernel=final_kernel,
                     resetbits=resetbits,
                     static=(static & (len(inst_keys) == 1)), 
                     gain=gain, rdnoise=rdnoise)
        else:
            AstroDrizzle(group['files'], output=group['product'],
                     clean=True, context=context, preserve=False,
                     skysub=skysub, skyuser=skyuser, skymethod=skymethod,
                     driz_separate=run_driz_cr, driz_sep_wcs=run_driz_cr,
                     median=run_driz_cr, blot=run_driz_cr,
                     driz_cr=run_driz_cr,
                     driz_cr_snr=driz_cr_snr, driz_cr_scale=driz_cr_scale,
                     driz_cr_corr=False, driz_combine=True,
                     final_bits=bits, coeffs=True, build=build,
                     final_wht_type=final_wht_type,
                     final_wt_scl=final_wt_scl,
                     final_pixfrac=pixfrac,
                     final_wcs=final_wcs, final_rot=final_rot,
                     final_scale=scale,
                     final_ra=final_ra, final_dec=final_dec,
                     final_outnx=final_outnx, final_outny=final_outny,
                     final_kernel=final_kernel,
                     resetbits=resetbits,
                     static=(static & (len(inst_keys) == 1)),
                     gain=gain, rdnoise=rdnoise)
        
        clean_drizzle(group['product'], fix_wcs_system=fix_wcs_system)
        
        # Reset JWST headers
        if isJWST:
            for file in group['files']:
                _ = jwst_utils.set_jwst_to_hst_keywords(file, reset=True)


def manual_alignment(visit, ds9, reference=None, reference_catalogs=['SDSS', 'PS1', 'GAIA', 'WISE'], use_drz=False):
    """Manual alignment of a visit with respect to an external region file

    Parameters
    ----------
    visit : dict
        List of visit information from `~grizli.utils.parse_flt_files`.

    ds9 : `~grizli.ds9.DS9`
        DS9 instance for interaction.  Requires `~pyds9` and the extended
        methods in `~grizli.ds9.DS9`.

    reference : str
        Filename of a DS9 region file that will be used as reference.  If
        None, then tries to find a local file based on the `visit['product']`.

    reference_catalogs : list
        If no valid `reference` file provided or found, query external
        catalogs with `~grizli.prep.get_radec_catalog`.  The external
        catalogs will be queried in the order specified in this list.


    Returns
    -------
    Generates a file like `{{0}}.align_guess'.format(visit['product'])` that
    the alignment scripts know how to read.

    .. note::

    The alignment here is done interactively in the DS9 window.  The script
    prompts you to first center the frame on a source in the image itself,
    which can be done in "panning" mode.  After centering, hit <enter> in the
    command line.  The script will then prompt to center the frame on the
    corresponding region from the reference file.  After recentering, type
    enter again and the output file will be computed and stored.

    If you wish to break out of the script and not generate the output file,
    type any character in the terminal at the first pause/prompt.

    """
    import os

    ref_image = os.path.join(os.getcwd(), '../RAW/', visit['files'][0])
    files = glob.glob('{0}_dr?_sci.fits'.format(visit['product']))
    ext = 1
    if use_drz & (len(files) > 0):
        ref_image = files[0]
        ext = 0

    im = pyfits.open(ref_image)
    ra, dec = im[ext].header['CRVAL1'], im[ext].header['CRVAL2']

    if reference is None:
        reg_files = glob.glob('{0}_*reg'.format(visit['product']))
        if len(reg_files) == 0:
            get_radec_catalog(ra=ra, dec=dec, radius=3.,
                              product=visit['product'], verbose=True,
                              reference_catalogs=reference_catalogs,
                              date=im[0].header['EXPSTART'],
                              date_format='mjd')

        reg_files = glob.glob('{0}_*reg'.format(visit['product']))
        reference = os.path.join(os.getcwd(), reg_files[0])

    print(visit['product'], reference)

    #im = pyfits.open('{0}_drz_sci.fits'.format(visit['product']))
    #ds9.view(im[1].data, header=im[1].header)
    if '_c0' in im.filename():
        ds9.set('file {0}[3]'.format(im.filename()))
    else:
        ds9.set('file {0}'.format(im.filename()))

    ds9.set('regions file '+reference)
    x = input('pan to object in image: ')
    if x:
        print('Input detected ({0}).  Abort.'.format(x))
        return False

    x0 = np.cast[float](ds9.get('pan image').split())
    x = input('pan to object in region: ')
    x1 = np.cast[float](ds9.get('pan image').split())

    print('Saved {0}.align_guess'.format(visit['product']))

    dx = x0[0]-x1[0]
    dy = x0[1]-x1[1]

    if '_c0' in im.filename():
        dx *= -1
        dy * + -1

    np.savetxt('{0}.align_guess'.format(visit['product']), [[dx, dy, 0, 1].__repr__()[1:-1].replace(',', '')], fmt='%s')
    
    im.close()


def extract_fits_log(file='idk106ckq_flt.fits', get_dq=True):
    """
    not used
    """
    log = OrderedDict()
    im = pyfits.open(file)

    for k in im[0].header:
        if k in ['HISTORY', 'COMMENT', 'ORIGIN', '']:
            continue

        if k.strip().startswith('/'):
            continue

        log[k] = im[0].header[k]

    log['chips'] = []

    imshape = im[1].data.shape
    if get_dq:
        idx = np.arange(imshape[0]**2, dtype=np.int32).reshape(imshape)

    for chip in [1, 2, 3, 4]:
        key = 'SCI{0}'.format(chip)
        if ('SCI', chip) in im:
            log['chips'].append(chip)
            log[key] = OrderedDict()
            h = im['SCI', chip].header
            for k in h:
                if k in ['HISTORY', 'COMMENT', 'ORIGIN', '']:
                    continue

                if k.strip().startswith('/'):
                    continue

                log[key][k] = h[k]

            if get_dq:
                dq = im['DQ', chip].data
                mask = dq > 0
                log['DQi{0}'.format(chip)] = list(idx[mask].astype(str))
                log['DQv{0}'.format(chip)] = list(dq[mask].astype(str))
    
    im.close()
    return log


def check_isJWST(infile=''):
    """
    Check if a file or list of files is JWST image. 
    
    Parameters
    ----------
    infile : fits file
        File to test
    
    Returns
    -------
    isJWST : bool
    """
    
    with pyfits.open(infile) as hdu:
        if 'OTELESCO' in hdu[0].header:
            isJWST = hdu[0].header['OTELESCO'] == 'JWST'
        elif 'TELESCOP' in hdu[0].header:
            isJWST = hdu[0].header['TELESCOP'] == 'JWST'
        else:
            isJWST = False
    
    return isJWST
