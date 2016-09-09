"""
Align direct images & make mosaics
"""
import os
from collections import OrderedDict
import glob

import numpy as np
import matplotlib.pyplot as plt

# conda install shapely
# from shapely.geometry.polygon import Polygon

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table

from . import utils

def check_status():
    """Make sure all files and modules are in place and print some information if they're not
    """
    for ref_dir in ['iref']:
        if not os.getenv(ref_dir):
            print """
No $%s set!  Make a directory and point to it in ~/.bashrc or ~/.cshrc.
For example,

  $ mkdir $GRIZLI/%s
  $ export %s="${GRIZLI}/%s/" # put this in ~/.bashrc
""" %(ref_dir, ref_dir, ref_dir, ref_dir)
        else:
            ### WFC3
            if not os.getenv('iref').endswith('/'):
                print "Warning: $iref should end with a '/' character [%s]" %(os.getenv('iref'))
        
            test_file = 'iref$uc72113oi_pfl.fits'.replace('iref$', os.getenv('iref'))
            if not os.path.exists(test_file):
                print """
        HST calibrations not found in $iref [%s]

        To fetch them, run

           >>> import grizli.utils
           >>> grizli.utils.fetch_default_calibs()

        """ %(os.getenv('iref'))
    
    ### Sewpy
    try:
        import sewpy
    except:
        print """
`sewpy` module needed for wrapping SExtractor within python.  
Get it from https://github.com/megalut/sewpy.
"""
        
check_status()
 
def go_all():
    """TBD
    """
    from stsci.tools import asnutil
    info = Table.read('files.info', format='ascii.commented_header')
        
    # files=glob.glob('../RAW/i*flt.fits')
    # info = utils.get_flt_info(files)
    
    for col in info.colnames:
        if not col.islower():
            info.rename_column(col, col.lower())
    
    output_list, filter_list = utils.parse_flt_files(info=info, uniquename=False)
    
    for key in output_list:
        #files = [os.path.basename(file) for file in output_list[key]]
        files = output_list[key]
        asn = asnutil.ASNTable(files, output=key)
        asn.create()
        asn.write()
        
def fresh_flt_file(file, preserve_dq=False, path='../RAW/', verbose=True, extra_badpix=True, apply_grism_skysub=True):
    """Copy "fresh" unmodified version of a data file from some central location
        
    Parameters
    ----------
    preserve_dq : bool
        Preserve DQ arrays of files if they exist in './'
        
    path : str
        Path where to find the "fresh" files
        
    verbose : bool
        Print information about what's being done
        
    extra_badpix : bool
        Apply extra bad pixel mask.  Currently this is hard-coded to look for 
        a file "badpix_spars200_Nov9.fits" in the directory specified by
        the `$iref` environment variable.  The file can be downloaded from 
        
        https://github.com/gbrammer/wfc3/tree/master/data
        
    apply_grism_skysub : bool
        xx nothing now xxx
    
    Returns
    -------
    Nothing, but copies the file from `path` to `./`.
        
    """
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
    
    if dq is not None:
        orig_file['DQ'] = dq
    
    head = orig_file[0].header
    
    ### Divide grism images by imaging flats
    ### G102 -> F105W, uc72113oi_pfl.fits
    ### G141 -> F140W, uc72113oi_pfl.fits
    flat, extra_msg = 1., ''
    filter = utils.get_hst_filter(head)
    if filter in ['G102', 'G141']:
        flat_files = {'G102': 'uc72113oi_pfl.fits',
                      'G141': 'uc721143i_pfl.fits'}
                
        flat_file = flat_files[filter]
        extra_msg = ' / flat: %s' %(flat_file)

        flat_im = pyfits.open(os.path.join(os.getenv('iref'), flat_file))
        flat = flat_im['SCI'].data[5:-5, 5:-5]
        flat_dq = (flat < 0.2)
        
        ### Grism FLT from IR amplifier gain
        pfl_file = orig_file[0].header['PFLTFILE'].replace('iref$',
                                                           os.getenv('iref'))
        grism_pfl = pyfits.open(pfl_file)[1].data[5:-5,5:-5]
        
        orig_file['DQ'].data |= 4*flat_dq
        orig_file['SCI'].data *= grism_pfl/flat
        
        # if apply_grism_skysub:
        #     if 'GSKY001' in orig_file:
    
    if filter == 'G800L':
        flat_files = {'G800L':'n6u12592j_pfl.fits'} # F814W
        flat_file = flat_files[filter]
        extra_msg = ' / flat: %s' %(flat_file)
        
        flat_im = pyfits.open(os.path.join(os.getenv('jref'), flat_file))
        pfl_file = orig_file[0].header['PFLTFILE'].replace('jref$',
                                                    os.getenv('jref'))
        pfl_im = pyfits.open(pfl_file)
        for ext in [1,2]:
            flat = flat_im['SCI',ext].data
            flat_dq = (flat < 0.2)
            
            grism_pfl = pfl_im['SCI',ext].data
            
            orig_file['DQ',ext].data |= 4*flat_dq
            orig_file['SCI',ext].data *= grism_pfl/flat
        
        orig_file[0].header['NPOLFILE'] = 'jref$v971826jj_npl.fits' # F814W
        
    if (head['INSTRUME'] == 'WFC3') & (head['DETECTOR'] == 'IR')&extra_badpix: 
        bp = pyfits.open(os.path.join(os.getenv('iref'),
                                      'badpix_spars200_Nov9.fits'))    
        orig_file['DQ'].data |= bp[0].data
        extra_msg += ' / bpix: $iref/badpix_spars200_Nov9.fits'
        
    if verbose:
        print '%s -> %s %s' %(orig_file.filename(), local_file, extra_msg)
        
    orig_file.writeto(local_file, clobber=True)
    
def apply_persistence_mask(flt_file, path='../Persistence', dq_value=1024,
                           err_threshold=0.6, grow_mask=3, verbose=True):
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
        Threshold for defining affected pixels:
        
        flagged = persist > err_threshold*ERR
        
    grow_mask : int
        Factor by which to dilate the persistence mask.
        
    verbose : bool
        Print information to the terminal
    
    Returns
    -------
    Nothing, updates the DQ extension of `flt_file`.
    
    """
    import scipy.ndimage as nd
    
    flt = pyfits.open(flt_file, mode='update')
    
    pers_file = os.path.join(path,
             os.path.basename(flt_file).replace('_flt.fits', '_persist.fits'))
    
    if not os.path.exists(pers_file):
        if verbose:
            print 'Persistence file %s not found' %(pers_file)
        
        #return 0
    
    pers = pyfits.open(pers_file)
    
    pers_mask = pers['SCI'].data > err_threshold*flt['ERR'].data
    
    if grow_mask > 0:
        pers_mask = nd.maximum_filter(pers_mask*1, size=grow_mask)
    else:
        pers_mask = pers_mask * 1
    
    NPERS = pers_mask.sum()
    if verbose:
        print '%s: flagged %d pixels affected by persistence (pers/err=%0.2f)' %(pers_file, NPERS, err_threshold)
    
    if NPERS > 0:
        flt['DQ'].data[pers_mask > 0] |= dq_value
        flt.flush()

def apply_saturated_mask(flt_file, dq_value=1024):
    """Saturated pixels have some pulldown in the opposite amplifier
    
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
    
    ## Don't flag pixels in lower right corner
    sat[:80,-80:] = False
    
    ## Flag only if a number of nearby pixels also saturated
    kern = np.ones((3,3))
    sat_grow = nd.convolve(sat*1, kern)
    
    sat_mask = (sat & (sat_grow > 2))[::-1,:]*1
    
    NSAT = sat_mask.sum()
    if verbose:
        print '%s: flagged %d pixels affected by saturation pulldown' %(flt_file, NSAT)
    
    if NSAT > 0:
        flt['DQ'].data[sat_mask > 0] |= dq_value
        flt.flush()
    

def clip_lists(input, output, clip=20):
    """TBD
    
    Clip [x,y] arrays of objects that don't have a match within `clip` pixels
    in either direction
    """
    import scipy.spatial
    
    tree = scipy.spatial.cKDTree(input, 10)
    
    ### Forward
    N = output.shape[0]
    dist, ix = np.zeros(N), np.zeros(N, dtype=int)
    for j in range(N):
        dist[j], ix[j] = tree.query(output[j,:], k=1,
                                    distance_upper_bound=np.inf)
    
    ok = dist < clip
    out_arr = output[ok]
    if ok.sum() == 0:
        print 'No matches within `clip=%f`' %(clip)
        return False
        
    ### Backward
    tree = scipy.spatial.cKDTree(out_arr, 10)
    
    N = input.shape[0]
    dist, ix = np.zeros(N), np.zeros(N, dtype=int)
    for j in range(N):
        dist[j], ix[j] = tree.query(input[j,:], k=1,
                                    distance_upper_bound=np.inf)
    
    ok = dist < clip
    in_arr = input[ok]
    
    return in_arr, out_arr

def match_lists(input, output, transform=None, scl=3600., simple=True,
                outlier_threshold=5, toler=5):
    """TBD
    
    Compute matched objects and transformation between two [x,y] lists.
    
    If `transform` is None, use Similarity transform (shift, scale, rot) 
    """
    from astropy.table import Table    
    
    import skimage.transform
    from skimage.measure import ransac

    import stsci.stimage
    
    if transform is None:
        transform = skimage.transform.SimilarityTransform
        
    #print 'xyxymatch'
    match = stsci.stimage.xyxymatch(input, output, 
                                    origin=np.median(input, axis=0), 
                                    mag=(1.0, 1.0), rotation=(0.0, 0.0),
                                    ref_origin=np.median(input, axis=0), 
                                    algorithm='tolerance', tolerance=toler, 
                                    separation=0.5, nmatch=10, maxratio=10.0, 
                                    nreject=10)
                                    
    m = Table(match)

    output_ix = m['ref_idx'].data
    input_ix = m['input_idx'].data
    
    tf = transform()
    tf.estimate(input[input_ix,:], output[output_ix])
    
    if not simple:
        model, inliers = ransac((input[input_ix,:], output[output_ix]),
                                   transform, min_samples=3,
                                   residual_threshold=2, max_trials=100)
        
        outliers = ~inliers 
    else:
        model = tf
        ### Compute statistics
        if len(input_ix) > 10:
            mout = tf(input[input_ix,:])
            dx = mout - output[output_ix]
            dr = np.sqrt(np.sum(dx**2, axis=1))
            outliers = dr > outlier_threshold
        else:
            outliers = np.zeros(len(input_ix), dtype=bool)
            
    return input_ix, output_ix, outliers, model

def align_drizzled_image(root='', mag_limits=[14,23], radec=None, NITER=3, 
                         clip=20, log=True, outlier_threshold=5, 
                         verbose=True):
    """TBD
    """
    if hasattr(radec, 'upper'):
        rd_ref = np.loadtxt(radec)
    else:
        rd_ref = radec*1
        
    if not os.path.exists('%s.cat' %(root)):
        cat = make_drz_catalog(root=root)
    else:
        cat = Table.read('%s.cat' %(root), format='ascii.commented_header')
    
    ### Clip obviously distant files to speed up match
    rd_cat = np.array([cat['X_WORLD'], cat['Y_WORLD']])
    rd_cat_center = np.median(rd_cat, axis=1)
    cosdec = np.array([np.cos(rd_cat_center[1]/180*np.pi),1])
    dr_cat = np.sqrt(np.sum((rd_cat.T-rd_cat_center)**2*cosdec**2, axis=1))
    
    dr = np.sqrt(np.sum((rd_ref-rd_cat_center)**2*cosdec**2, axis=1))
    
    rd_ref = rd_ref[dr < 1.1*dr_cat.max(),:]
    
    ok = (cat['MAG_AUTO'] > mag_limits[0]) & (cat['MAG_AUTO'] < mag_limits[1])
    if ok.sum() == 0:
        print '%s.cat: no objects found in magnitude range %s' %(root,
                                                                 mag_limits)
        return False
    
    xy_drz = np.array([cat['X_IMAGE'][ok], cat['Y_IMAGE'][ok]]).T
    
    drz_file = glob.glob('%s_dr[zc]_sci.fits' %(root))[0]
    drz_im = pyfits.open(drz_file)
    sh = drz_im[0].data.shape
    
    drz_wcs = pywcs.WCS(drz_im[0].header, relax=True)
    orig_wcs = drz_wcs.copy()
    
    out_shift, out_rot, out_scale = np.zeros(2), 0., 1.
    
    for iter in range(NITER):
        xy = np.array(drz_wcs.all_world2pix(rd_ref, 0))
        pix = np.cast[int](np.round(xy)).T

        ### Find objects where drz pixels are non-zero
        okp = (pix[0,:] > 0) & (pix[1,:] > 0)
        okp &= (pix[0,:] < sh[1]) & (pix[1,:] < sh[0])
        ok2 = drz_im[0].data[pix[1,okp], pix[0,okp]] != 0

        N = ok2.sum()
        status = clip_lists(xy_drz, xy+1, clip=clip)
        if not status:
            print 'Problem xxx'
        
        input, output = status
        #print np.sum(input) + np.sum(output)
        
        toler=5
        titer=0
        while (titer < 3):
            try:
                res = match_lists(output, input, scl=1., simple=True,
                          outlier_threshold=outlier_threshold, toler=toler)
                output_ix, input_ix, outliers, tf = res
                break
            except:
                toler += 5
                titer += 1
                
        titer = 0 
        while (len(input_ix)*1./len(input) < 0.1) & (titer < 3):
            titer += 1
            toler += 5
            res = match_lists(output, input, scl=1., simple=True,
                              outlier_threshold=outlier_threshold,
                              toler=toler)
            output_ix, input_ix, outliers, tf = res
        
        #
        tf_out = tf(output[output_ix])
        dx = input[input_ix] - tf_out
        rms = utils.nmad(np.sqrt((dx**2).sum(axis=1)))
        outliers = outliers | (np.sqrt((dx**2).sum(axis=1)) > 4*rms)
                                                  
        if outliers.sum() > 0:
            res2 = match_lists(output[output_ix][~outliers],
                              input[input_ix][~outliers], scl=1., simple=True,
                              outlier_threshold=outlier_threshold,
                              toler=toler)
            
            output_ix2, input_ix2, outliers2, tf = res2
        
        if verbose:
            shift = tf.translation
            NGOOD = (~outliers).sum()
            print '%s (%d) %d: %6.2f %6.2f %7.3f %.3f' %(root,iter,NGOOD,
                                                   shift[0], shift[1], 
                                                   tf.rotation/np.pi*180, 
                                                   1./tf.scale)
        
        out_shift += tf.translation
        out_rot -= tf.rotation
        out_scale *= tf.scale
        
        drz_wcs.wcs.crpix += tf.translation
        theta = -tf.rotation
        _mat = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
        
        drz_wcs.wcs.cd = np.dot(drz_wcs.wcs.cd, _mat)/tf.scale
                
    if log:
        tf_out = tf(output[output_ix][~outliers])
        dx = input[input_ix][~outliers] - tf_out
        rms = utils.nmad(np.sqrt((dx**2).sum(axis=1)))
        
        interactive_status=plt.rcParams['interactive']
        plt.ioff()

        fig = plt.figure(figsize=[6.,6.])
        ax = fig.add_subplot(111)
        ax.scatter(dx[:,0], dx[:,1], alpha=0.5, color='b')
        ax.scatter([0],[0], marker='+', color='red', s=40)
        ax.set_xlabel(r'$dx$'); ax.set_ylabel(r'$dy$')
        ax.set_title(root)
        
        ax.set_xlim(-7*rms, 7*rms)
        ax.set_ylim(-7*rms, 7*rms)
        ax.grid()
        
        fig.tight_layout(pad=0.1)
        fig.savefig('%s_wcs.png' %(root))
        plt.close()
        
        if interactive_status:
            plt.ion()
        
        log_wcs(root, orig_wcs, out_shift, out_rot/np.pi*180, out_scale, rms,
                n=NGOOD, initialize=False)
            
    return orig_wcs, drz_wcs, out_shift, out_rot/np.pi*180, out_scale

def log_wcs(root, drz_wcs, shift, rot, scale, rms=0., n=-1, initialize=True):
    """Save WCS offset information to a file
    """
    if (not os.path.exists('%s_wcs.log' %(root))) | initialize:
        print 'Initialize %s_wcs.log' %(root)
        orig_hdul = pyfits.HDUList()
        fp = open('%s_wcs.log' %(root), 'w')
        fp.write('# ext xshift yshift rot scale rms N\n')
        fp.write('# %s\n' %(root))
        count = 0
    else:
        orig_hdul = pyfits.open('%s_wcs.fits' %(root))
        fp = open('%s_wcs.log' %(root), 'a')
        count = len(orig_hdul)
    
    hdu = drz_wcs.to_fits()[0]
    orig_hdul.append(hdu)
    orig_hdul.writeto('%s_wcs.fits' %(root), clobber=True)
    
    fp.write('%5d %13.4f %13.4f %13.4f %13.5f %13.3f %4d\n' %(count, shift[0],
                                                       shift[1], rot, scale,
                                                       rms, n))
    fp.close()
    
def table_to_regions(table, output='ds9.reg'):
    """Make a DS9 region file from a table object
    """
    fp = open(output,'w')
    fp.write('fk5\n')
    
    if 'X_WORLD' in table.colnames:
        rc, dc = 'X_WORLD', 'Y_WORLD'
    else:
        rc, dc = 'ra', 'dec'
        
    lines = ['circle(%.6f, %.6f, 0.5")\n' %(table[rc][i], table[dc][i])
                                              for i in range(len(table))]

    fp.writelines(lines)
    fp.close()
    
def make_drz_catalog(root='', threshold=2., get_background=True, 
                     verbose=True, extra_config={}):
    """Make a SExtractor catalog from drizzle products
    
    TBD
    """
    import sewpy
    
    drz_file = glob.glob('%s_dr[zc]_sci.fits' %(root))[0]
    im = pyfits.open(drz_file)
    
    if 'PHOTFNU' in im[0].header:
        ZP = -2.5*np.log10(im[0].header['PHOTFNU'])+8.90
    else:
        ZP = (-2.5*np.log10(im[0].header['PHOTFLAM']) - 21.10 -
                 5*np.log10(im[0].header['PHOTPLAM']) + 18.6921)
        
    config = OrderedDict(DETECT_THRESH=threshold, ANALYSIS_THRESH=threshold,
              DETECT_MINAREA=6,
              PHOT_FLUXFRAC="0.5", 
              WEIGHT_TYPE="MAP_WEIGHT",
              WEIGHT_IMAGE=drz_file.replace('_sci.fits', '_wht.fits'),
              CHECKIMAGE_TYPE="SEGMENTATION",
              CHECKIMAGE_NAME='%s_seg.fits' %(root),
              MAG_ZEROPOINT=ZP, 
              CLEAN="N", 
              PHOT_APERTURES="6, 8.335, 16.337, 20",
              BACK_SIZE=32)
    
    if get_background:
        config['CHECKIMAGE_TYPE'] = 'SEGMENTATION,BACKGROUND'
        config['CHECKIMAGE_NAME'] = '%s_seg.fits,%s_bkg.fits' %(root,root)
    
    for key in extra_config:
        config[key] = extra_config[key]
        
    sew = sewpy.SEW(params=["NUMBER", "X_IMAGE", "Y_IMAGE", "X_WORLD",
                    "Y_WORLD", "A_IMAGE", "B_IMAGE", "THETA_IMAGE", 
                    "MAG_AUTO", "MAGERR_AUTO", "FLUX_AUTO", "FLUXERR_AUTO",
                    "FLUX_APER", "FLUXERR_APER",
                    "FLUX_RADIUS", "BACKGROUND", "FLAGS"],
                    config=config)
    
    output = sew(drz_file)
    cat = output['table']
    cat.meta = config
    cat.write('%s.cat' %(root), format='ascii.commented_header')
            
    if verbose:
        print '%s catalog: %d objects' %(root, len(cat))
    
    return cat
    
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
    if instance(input_asn, dict):
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

def get_sdss_catalog(ra=165.86, dec=34.829694, radius=3):
    """Query for objects in the SDSS photometric catalog 
    
    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees
    
    radius : float
        Radius of the query, in arcmin
    
    Returns
    -------
    table : `~astropy.table.Table`
        Result of the query
        
    """
    from astroquery.sdss import SDSS
    
    coo = coord.SkyCoord(ra*u.deg, dec*u.deg)
    
    fields = ['ra', 'dec', 'raErr', 'decErr', 'petroMag_r', 'petroMagErr_r']
    #print fields
    fields = None
    
    table = SDSS.query_region(coo, radius=radius*u.arcmin, spectro=False, 
                              photoobj_fields = fields)
                              
    return table
    
def get_wise_catalog(ra=165.86, dec=34.829694, radius=3):
    """Query for objects in the `AllWISE <http://wise2.ipac.caltech.edu/docs/release/allwise/>`_ source catalog 
    
    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees
    
    radius : float
        Radius of the query, in arcmin
    
    Returns
    -------
    table : `~astropy.table.Table`
        Result of the query
        
    """
    from astroquery.irsa import Irsa
    
    all_wise = 'wise_allwise_p3as_psd'
    coo = coord.SkyCoord(ra*u.deg, dec*u.deg)
    
    table = Irsa.query_region(coo, catalog=all_wise, spatial="Cone",
                              radius=radius*u.arcmin, get_query_payload=False)
    
    return table

def get_radec_catalog(ra=0., dec=0., radius=3., product='cat', verbose=True):
    """Decide what reference astrometric catalog to use
    
    First search SDSS, then WISE looking for nearby matches.  
    
    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees
    
    radius : float
        Radius of the query, in arcmin
    
    product : str
        Basename of the drizzled product. If a locally-created catalog with
        filename that startswith `product` is found, use that one instead of
        the external (low precision) catalogs so that you're matching
        HST-to-HST astrometry.
    
    Returns
    -------
    radec : str
        Filename of the RA/Dec list derived from the parent catalog
    
    ref_catalog : str, {'SDSS', 'WISE', 'VISIT'}
        Provenance of the `radec` list.
    
    """
    try:
        sdss = get_sdss_catalog(ra=ra, dec=dec, radius=radius)
    except:
        print 'SDSS query failed'
        sdss = []
    
    if sdss is None:
        sdss = []
            
    if len(sdss) > 5:
        table_to_regions(sdss, output='%s_sdss.reg' %(product))
        sdss['ra','dec'].write('%s_sdss.radec' %(product), 
                                format='ascii.commented_header')
        radec = '%s_sdss.radec' %(product)
        ref_catalog = 'SDSS'
    else:
        try:
            wise = get_wise_catalog(ra=ra, dec=dec, radius=2)
            
            table_to_regions(wise, output='%s_wise.reg' %(product))
            wise['ra','dec'].write('%s_wise.radec' %(product),
                                    format='ascii.commented_header')

            radec = '%s_wise.radec' %(product)
            ref_catalog = 'WISE'
        except:
            print 'WISE query failed'
    
    #### WISP, check if a catalog already exists for a given rootname and use 
    #### that if so.
    cat_files = glob.glob('-f1'.join(product.split('-f1')[:-1]) + '-f*cat')
    if len(cat_files) > 0:
        cat = Table.read(cat_files[0], format='ascii.commented_header')
        root = cat_files[0].split('.cat')[0]
        cat['X_WORLD','Y_WORLD'].write('%s.radec' %(root),
                                format='ascii.commented_header')
        
        radec = '%s.radec' %(root)
        ref_catalog = 'VISIT'
    
    if verbose:
        print '%s - Reference RADEC: %s [%s]' %(product, radec, ref_catalog)    
    
    return radec, ref_catalog
    
def process_direct_grism_visit(direct={}, grism={}, radec=None,
                               align_tolerance=5, 
                               align_mag_limits = [14,23],
                               column_average=True, 
                               run_tweak_align=True):
    """Full processing of a direct + grism image visit.
    
    TBD
    
    """    
    from stsci.tools import asnutil
    from stwcs import updatewcs
    from drizzlepac import updatehdr
    from drizzlepac.astrodrizzle import AstroDrizzle
    
    ################# 
    ##########  Direct image processing
    #################
    
    ### Copy FLT files from ../RAW
    for file in direct['files']:
        fresh_flt_file(file)
        updatewcs.updatewcs(file, verbose=False)
    
    ### Make ASN
    asn = asnutil.ASNTable(direct['files'], output=direct['product'])
    asn.create()
    asn.write()
    
    ### Initial grism processing
    skip_grism = (grism == {}) | (grism is None) | (len(grism) == 0)
    if not skip_grism:
        for file in grism['files']:
            fresh_flt_file(file)
            updatewcs.updatewcs(file, verbose=False)
    
        ### Make ASN
        asn = asnutil.ASNTable(grism['files'], output=grism['product'])
        asn.create()
        asn.write()
            
    ACS = '_flc' in direct['files'][0]
    if ACS:
        bits = 0
        driz_cr_snr = '3.5 3.0'
        driz_cr_scale = '1.2 0.7'
    else:
        bits = 576
        driz_cr_snr = '8.0 5.0'
        driz_cr_scale = '2.5 0.7'
    
    if (not ACS) & run_tweak_align:
        tweak_align(direct_group=direct, grism_group=grism, max_dist=1.,
                    key=' ', drizzle=False, threshold=1.5)
      
    ### Get reference astrometry from SDSS or WISE
    if radec is None:
        im = pyfits.open(direct['files'][0])
        radec, ref_catalog = get_radec_catalog(ra=im[0].header['RA_TARG'],
                        dec=im[0].header['DEC_TARG'], 
                        product=direct['product'])
        
        if ref_catalog == 'VISIT':
            align_mag_limits = [16,23]
        elif ref_catalog == 'WISE':
            align_mag_limits = [16,21]
        elif ref_catalog == 'SDSS':
            align_mag_limits = [16,21]
    else:
        ref_catalog = 'USER'
    
    print '%s: First Drizzle' %(direct['product'])
    
    ### Clean up
    for ext in ['.fits', '.log']:
        file = '%s_wcs.%s' %(direct['product'], ext)
        if os.path.exists(file):
            os.remove(file)
                
    ### First drizzle
    if len(direct['files']) > 1:
        AstroDrizzle(direct['files'], output=direct['product'], clean=True,
                     context=False, preserve=False, skysub=True,
                     driz_separate=True, driz_sep_wcs=True, median=True,
                     blot=True, driz_cr=True, driz_cr_corr=False,
                     driz_combine=True, final_bits=bits, coeffs=True)        
    else:
        AstroDrizzle(direct['files'], output=direct['product'], clean=True,
                     final_scale=None, final_pixfrac=1, context=False,
                     final_bits=bits, preserve=False, driz_separate=False,
                     driz_sep_wcs=False, median=False, blot=False, 
                     driz_cr=False, driz_cr_corr=False, driz_combine=True) 
    
    ### Make catalog & segmentation image
    cat = make_drz_catalog(root=direct['product'], threshold=2)
    if radec == 'self':
        cat['X_WORLD', 'Y_WORLD'][(cat['MAG_AUTO'] > align_mag_limits[0]) & (cat['MAG_AUTO'] < align_mag_limits[1])].write('self', format='ascii.commented_header')
        
    clip=20
    logfile = '%s_wcs.log' %(direct['product'])
    if os.path.exists(logfile):
        os.remove(logfile)
        
    result = align_drizzled_image(root=direct['product'], 
                                  mag_limits=align_mag_limits,
                                  radec=radec, NITER=5, clip=clip, log=True,
                                  outlier_threshold=align_tolerance)
                                  
    orig_wcs, drz_wcs, out_shift, out_rot, out_scale = result
        
    ### Update direct FLT WCS
    for file in direct['files']:
        updatehdr.updatewcs_with_shift(file, 
                                  '%s_wcs.fits' %(direct['product']),
                                  xsh=out_shift[0], ysh=out_shift[1],
                                  rot=out_rot, scale=out_scale,
                                  wcsname=ref_catalog, force=True,
                                  reusename=True, verbose=True,
                                  sciext='SCI')
        
        ### Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
        ### keywords
        im = pyfits.open(file, mode='update')
        im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
        im.flush()
    
    ### Second drizzle with aligned wcs, refined CR-rejection params 
    ### tuned for WFC3/IR
    if len(direct['files']) == 1:
        AstroDrizzle(direct['files'], output=direct['product'], clean=True,
                     final_pixfrac=0.8, context=False, resetbits=4096, 
                     final_bits=bits, driz_sep_bits=bits, preserve=False, 
                     driz_cr_snr=driz_cr_snr, driz_cr_scale=driz_cr_scale, 
                     driz_separate=False, driz_sep_wcs=False, median=False, 
                     blot=False, driz_cr=False, driz_cr_corr=False) 
    else:
        if 'par' in direct['product']:
            pixfrac=1.0
        else:
            pixfrac=0.8
        
        AstroDrizzle(direct['files'], output=direct['product'], clean=True, 
                     final_pixfrac=pixfrac, context=False, resetbits=4096, 
                     final_bits=bits, driz_sep_bits=bits, preserve=False, 
                     driz_cr_snr=driz_cr_snr, driz_cr_scale=driz_cr_scale)
    
    ### Make DRZ catalog again with updated DRZWCS
    clean_drizzle(direct['product'])
    cat = make_drz_catalog(root=direct['product'], threshold=1.6)
    table_to_regions(cat, '%s.cat.reg' %(direct['product']))
    
    ################# 
    ##########  Grism image processing
    #################
    
    if skip_grism:       
        return True
    
    ### Match grism WCS to the direct images
    match_direct_grism_wcs(direct=direct, grism=grism, get_fresh_flt=False,
                           run_drizzle=True)
        
    ### Subtract grism sky
    status = visit_grism_sky(grism=grism, apply=True,
                          column_average=column_average, verbose=True, ext=1)
    
    ### Run on second chip (also for UVIS/G280)
    if ACS:
        visit_grism_sky(grism=grism, apply=True,
                        column_average=column_average, verbose=True, ext=2)
        
    ### Redrizzle with new background subtraction
    skyfile = '/tmp/%s.skyfile' %(grism['product'])
    fp = open(skyfile,'w')
    fp.writelines(['%s 0.0\n' %(f) for f in grism['files']])
    fp.close()
    
    if 'par' in grism['product']:
        pixfrac=1.0
    else:
        pixfrac=0.8
            
    AstroDrizzle(grism['files'], output=grism['product'], clean=True,
                 context=False, preserve=False, skysub=True, skyfile=skyfile,
                 driz_separate=True, driz_sep_wcs=True, median=True, 
                 blot=True, driz_cr=True, driz_cr_corr=True, 
                 driz_combine=True, driz_sep_bits=bits, final_bits=bits,
                 coeffs=True, resetbits=4096, final_pixfrac=pixfrac)        
    
    clean_drizzle(grism['product'])
    
    return True

def tweak_align(direct_group={}, grism_group={}, max_dist=1., key=' ', 
                threshold=3, drizzle=False):
    """
    Intra-visit shifts (WFC3/IR)
    """
    from drizzlepac.astrodrizzle import AstroDrizzle
    
    if len(direct_group['files']) < 2:
        print 'Only one direct image found, can\'t compute shifts!'
        return True
        
    wcs_ref, shift_dict = tweak_flt(files=direct_group['files'],
                                    max_dist=max_dist, threshold=threshold,
                                    verbose=True)

    grism_matches = find_direct_grism_pairs(direct=direct_group, grism=grism_group, check_pixel=[507, 507], toler=0.1, key=key)
    
    fp = open('%s_shifts.log' %(direct_group['product']), 'w')
    fp.write('# flt xshift yshift rot scale\n')
    for k in grism_matches:
        fp.write('# match[\'%s\'] = %s\n' %(k, grism_matches[k]))
    
    for k in shift_dict:
        d = shift_dict[k]
        fp.write('%s %7.3f %7.3f %8.5f %8.5f\n' %(k, d[0], d[1], d[2], d[3]))
    
    fp.close()
    
    apply_tweak_shifts(wcs_ref, shift_dict, grism_matches=grism_matches,
                       verbose=False)

    if not drizzle:
        return True
        
    ### Redrizzle
    bits = 576
    driz_cr_snr = '8.0 5.0'
    driz_cr_scale = '2.5 0.7'
    if 'par' in direct_group['product']:
        pixfrac=1.0
    else:
        pixfrac=0.8

    AstroDrizzle(direct_group['files'], output=direct_group['product'],
                 clean=True, final_pixfrac=pixfrac, context=False,
                 resetbits=4096, final_bits=bits, driz_sep_bits=bits,
                 preserve=False, driz_cr_snr=driz_cr_snr,
                 driz_cr_scale=driz_cr_scale)
    
    clean_drizzle(direct_group['product'])
    cat = make_drz_catalog(root=direct_group['product'], threshold=1.6)
    table_to_regions(cat, '%s.cat.reg' %(direct_group['product']))
    
    if (grism_group == {}) | (grism_group is None):
        return True
        
    # Grism  
    skyfile = '/tmp/%s.skyfile' %(grism_group['product'])
    fp = open(skyfile,'w')
    fp.writelines(['%s 0.0\n' %(f) for f in grism_group['files']])
    fp.close()
      
    AstroDrizzle(grism_group['files'], output=grism_group['product'],
                 clean=True, context=False, preserve=False, skysub=True,
                 skyfile=skyfile, driz_separate=True, driz_sep_wcs=True,
                 median=True, blot=True, driz_cr=True, driz_cr_corr=True,
                 driz_combine=True, driz_sep_bits=bits, final_bits=bits,
                 coeffs=True, resetbits=4096, final_pixfrac=pixfrac)
    
    clean_drizzle(grism_group['product'])
    
    return True
    
def clean_drizzle(root):
    """Zero-out WHT=0 pixels in drizzle mosaics
    
    Parameters
    ----------
    root : str
        Rootname of the mosaics.  I.e., `{root}_drz_sci.fits`.
    
    Returns
    -------
    Nothing, science mosaic modified in-place
    """
    drz_file = glob.glob('%s_dr[zc]_sci.fits' %(root))[0]
    
    sci = pyfits.open(drz_file, mode='update')
    wht = pyfits.open(drz_file.replace('_sci.fits', '_wht.fits'))
    mask = wht[0].data == 0
    sci[0].data[mask] = 0
    sci.flush()

def tweak_flt(files=[], max_dist=0.4, threshold=3, verbose=True):
    """TBD
    
    Refine shifts of FLT files
    """
    import scipy.spatial
    # https://github.com/megalut/sewpy
    import sewpy
    
    ### Make FLT catalogs
    cats = []
    for i, file in enumerate(files):
        sew = sewpy.SEW(params=["X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD",
                                "FLUX_RADIUS(3)", "FLAGS"],
                        config={"DETECT_THRESH":threshold, "DETECT_MINAREA":8,
                                "PHOT_FLUXFRAC":"0.3, 0.5, 0.8",
                                "WEIGHT_TYPE":"MAP_RMS",
                                "WEIGHT_IMAGE":"rms.fits"})
        
        im = pyfits.open(file)
        ok = im['DQ',1].data == 0
        sci = im['SCI',1].data*ok - np.median(im['SCI',1].data[ok])
        
        pyfits.writeto('sci.fits', data=sci, header=im['SCI',1].header,
                       clobber=True)
        
        pyfits.writeto('rms.fits', data=im['ERR',1].data,
                       header=im['ERR'].header, clobber=True)
        
        output = sew('sci.fits')        
        
        wcs = pywcs.WCS(im['SCI',1].header, relax=True)
        cats.append([output['table'], wcs])
    
    c0 = cats[0][0]
    wcs_0 = cats[0][1]
    xy_0 = np.array([c0['X_IMAGE'], c0['Y_IMAGE']]).T
    tree = scipy.spatial.cKDTree(xy_0, 10)
    
    d = OrderedDict()
    for i in range(0, len(files)):
        c_i, wcs_i = cats[i]
        ## SExtractor doesn't do SIP WCS?
        rd = np.array(wcs_i.all_pix2world(c_i['X_IMAGE'], c_i['Y_IMAGE'], 1))
        xy = np.array(wcs_0.all_world2pix(rd.T, 1))
        N = xy.shape[0]
        dist, ix = np.zeros(N), np.zeros(N, dtype=int)
        for j in range(N):
            dist[j], ix[j] = tree.query(xy[j,:], k=1,
                                        distance_upper_bound=np.inf)
        
        ok = dist < max_dist
        if ok.sum() == 0:
            d[files[i]] = [0.0, 0.0, 0.0, 1.0]
            if verbose:
                print files[i], '! no match'
            
            continue
            
        dr = xy - xy_0[ix,:] 
        dx = np.median(dr[ok,:], axis=0)
        rms = np.std(dr[ok,:], axis=0)/np.sqrt(ok.sum())

        d[files[i]] = [dx[0], dx[1], 0.0, 1.0]
        
        if verbose:
            print files[i], dx, rms, 'N=%d' %(ok.sum())
    
    wcs_ref = cats[0][1]
    return wcs_ref, d

def apply_tweak_shifts(wcs_ref, shift_dict, grism_matches={}, verbose=True):
    """
    
    """
    from drizzlepac import updatehdr

    hdu = wcs_ref.to_fits(relax=True)
    hdu.writeto('tweak_wcs.fits', clobber=True)
    for file in shift_dict:
        updatehdr.updatewcs_with_shift(file, 'tweak_wcs.fits',
                                        xsh=shift_dict[file][0],
                                        ysh=shift_dict[file][1],
                                        rot=0., scale=1.,
                                        wcsname='SHIFT', force=True,
                                        reusename=True, verbose=verbose,
                                        sciext='SCI')
        
        ### Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
        ### keywords
        im = pyfits.open(file, mode='update')
        im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
        im.flush()
        
        # Update paired grism exposures
        if file in grism_matches:
            for grism_file in grism_matches[file]:
                updatehdr.updatewcs_with_shift(grism_file, 'tweak_wcs.fits',
                                              xsh=shift_dict[file][0],
                                              ysh=shift_dict[file][1],
                                              rot=0., scale=1.,
                                              wcsname='SHIFT', force=True,
                                              reusename=True, verbose=verbose,
                                              sciext='SCI')
                
                ### Bug in astrodrizzle? 
                im = pyfits.open(grism_file, mode='update')
                im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
                im.flush()
                
def find_direct_grism_pairs(direct={}, grism={}, check_pixel=[507, 507],
                            toler=0.1, key='A'):
    """
    For each grism exposure, check if there is a direct exposure
    that matches the WCS to within `toler` pixels.  If so, copy that WCS 
    directly.
    """
    direct_wcs = {}
    full_direct_wcs = {}
    direct_rd = {}
    
    grism_wcs = {}
    grism_pix = {}
    
    grism_matches = OrderedDict()
    
    for file in direct['files']:
        grism_matches[file] = []
        im = pyfits.open(file)
        direct_wcs[file] = pywcs.WCS(im[1].header, relax=True, key=key)
        full_direct_wcs[file] = pywcs.WCS(im[1].header, relax=True)
        direct_rd[file] = direct_wcs[file].all_pix2world([check_pixel], 1)
    
    if 'files' not in grism:
        return grism_matches
         
    for file in grism['files']:
        im = pyfits.open(file)
        grism_wcs[file] = pywcs.WCS(im[1].header, relax=True, key=key)
        #print file
        delta_min = 10
        for d in direct['files']:
            pix = grism_wcs[file].all_world2pix(direct_rd[d], 1)
            dx = pix-np.array(check_pixel)
            delta = np.sqrt(np.sum(dx**2))
            #print '  %s %s, %.3f' %(d, dx, delta)
            if delta < delta_min:
                delta_min = delta
                delta_min_file = d
                if delta_min < toler:
                    grism_matches[delta_min_file].append(file)
    
    return grism_matches
            
        # ### Found a match, copy the header
        # if delta_min < toler:
        #     print file, delta_min_file, delta_min
        #     
        #     im = pyfits.open(file, mode='update')
        #     
        #     wcs_header = full_direct_wcs[delta_min_file].to_header(relax=True)
        #     for i in [1,2]: 
        #         for j in [1,2]:
        #             wcs_header.rename_keyword('PC%d_%d' %(i,j), 
        #                                       'CD%d_%d' %(i,j))
        #     
        #     for ext in ['SCI','ERR','DQ']:
        #         for key in wcs_header:
        #             im[ext].header[key] = wcs_header[key]
        #     
        #     im.flush()
            
def match_direct_grism_wcs(direct={}, grism={}, get_fresh_flt=True, 
                           run_drizzle=True):
    """Match WCS of grism exposures to corresponding direct images
    
    TBD
    """
    from drizzlepac import updatehdr
    from stwcs import updatewcs
    from drizzlepac.astrodrizzle import AstroDrizzle
    
    wcs_log = Table.read('%s_wcs.log' %(direct['product']),
                         format='ascii.commented_header')
                         
    wcs_hdu = pyfits.open('%s_wcs.fits' %(direct['product']))
    
    if get_fresh_flt:
        for file in grism['files']:
            fresh_flt_file(file)
            updatewcs.updatewcs(file, verbose=False)
        
    direct_flt = pyfits.open(direct['files'][0])
    ref_catalog = direct_flt['SCI',1].header['WCSNAME']
    
    for ext in wcs_log['ext']:
        tmp_wcs = '/tmp/%s_tmpwcs.fits' %(direct['product'])
        wcs_hdu[ext].writeto(tmp_wcs, clobber=True)
        if 'scale' in wcs_log.colnames:
            scale = wcs_log['scale'][ext]
        else:
            scale = 1.
            
        for file in grism['files']:
            updatehdr.updatewcs_with_shift(file, tmp_wcs,
                                      xsh=wcs_log['xshift'][ext],
                                      ysh=wcs_log['yshift'][ext],
                                      rot=wcs_log['rot'][ext], scale=scale,
                                      wcsname=ref_catalog, force=True,
                                      reusename=True, verbose=True,
                                      sciext='SCI')
            
            ### Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
            ### keywords
            im = pyfits.open(file, mode='update')
            im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
            im.flush()
            
    ### Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
    ### keywords
    for file in grism['files']:
        im = pyfits.open(file, mode='update')
        im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
        im.flush()
    
    if run_drizzle:
        AstroDrizzle(grism['files'], output=grism['product'], clean=True,
                     context=False, preserve=False, skysub=True,
                     driz_separate=True, driz_sep_wcs=True, median=True, 
                     blot=True, driz_cr=True, driz_cr_corr=True, 
                     driz_combine=True, final_bits=576, coeffs=True, 
                     resetbits=4096)        
        
def align_multiple_drizzled(mag_limits=[16,23]):
    """TBD
    """
    from stwcs import updatewcs
    from drizzlepac import updatehdr
    from drizzlepac.astrodrizzle import AstroDrizzle
    
    drz_files = ['j0800+4029-080.0-f140w_drz_sci.fits', 
                 'j0800+4029-117.0-f140w_drz_sci.fits']
    
    for drz_file in drz_files:
        cat = make_drz_catalog(root=drz_file.split('_drz')[0], threshold=2)
        
    cref = Table.read(drz_files[0].replace('_drz_sci.fits', '.cat'), 
                      format='ascii.commented_header')
    
    ok = (cref['MAG_AUTO'] > mag_limits[0]) & (cref['MAG_AUTO'] < mag_limits[1])
    rd_ref = np.array([cref['X_WORLD'][ok], cref['Y_WORLD'][ok]]).T
    
    for drz_file in drz_files[1:]:
        root = drz_file.split('_drz')[0]
        result = align_drizzled_image(root=root, mag_limits=mag_limits,
                                      radec=rd_ref,
                                      NITER=5, clip=20)

        orig_wcs, drz_wcs, out_shift, out_rot, out_scale = result
        
        im = pyfits.open(drz_file)
        files = []
        for i in range(im[0].header['NDRIZIM']):
            files.append(im[0].header['D%03dDATA' %(i+1)].split('[')[0])
        
        
        for file in files:
            updatehdr.updatewcs_with_shift(file, drz_files[0],
                                      xsh=out_shift[0], ysh=out_shift[1],
                                      rot=out_rot, scale=out_scale,
                                      wcsname=ref_catalog, force=True,
                                      reusename=True, verbose=True,
                                      sciext='SCI')

            im = pyfits.open(file, mode='update')
            im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
            im.flush()

        ### Second drizzle
        if len(files) > 1:
            AstroDrizzle(files, output=root, clean=True, context=False, preserve=False, skysub=True, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True, final_bits=576, coeffs=True, resetbits=0)        
        else:
            AstroDrizzle(files, output=root, clean=True, final_scale=None, final_pixfrac=1, context=False, final_bits=576, preserve=False, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True) 

        cat = make_drz_catalog(root=root, threshold=2)
        
    if False:
        files0 = ['icou09fvq_flt.fits', 'icou09fyq_flt.fits', 'icou09gpq_flt.fits',
               'icou09h3q_flt.fits']

        files1 = ['icou10emq_flt.fits', 'icou10eqq_flt.fits', 'icou10euq_flt.fits',
               'icou10frq_flt.fits']
        
        all_files = list(np.append(files0, files1))
        AstroDrizzle(all_files, output='total', clean=True, context=False, preserve=False, skysub=True, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True, final_bits=576, coeffs=True, resetbits=0, final_rot=0)    
        
        AstroDrizzle(files0, output='group0', clean=True, context=False, preserve=False, skysub=True, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True, final_bits=576, coeffs=True, resetbits=0, final_refimage='total_drz_sci.fits')    
        AstroDrizzle(files1, output='group1', clean=True, context=False, preserve=False, skysub=True, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True, final_bits=576, coeffs=True, resetbits=0, final_refimage='total_drz_sci.fits')    
        
        im0 = pyfits.open('group0_drz_sci.fits')
        im1 = pyfits.open('group1_drz_sci.fits')
        imt = pyfits.open('total_drz_sci.fits')

        
def visit_grism_sky(grism={}, apply=True, column_average=True, verbose=True, ext=1):
    """Subtract sky background from grism exposures
    
    Implementation of grism sky subtraction from ISR 2015-17    
    
    TBD
    
    """
    import numpy.ma
    import scipy.ndimage as nd
    
    from sklearn.gaussian_process import GaussianProcess
    
    ### Figure out which grism 
    im = pyfits.open(grism['files'][0])
    grism_element = utils.get_hst_filter(im[0].header)
    
    flat = 1.
    if grism_element == 'G141':
        bg_fixed = ['zodi_G141_clean.fits']
        bg_vary = ['zodi_G141_clean.fits', 'excess_lo_G141_clean.fits',
                   'G141_scattered_light.fits'][1:]
        ACS = False
    elif grism_element == 'G102':
        bg_fixed = ['zodi_G102_clean.fits']
        bg_vary = ['excess_G102_clean.fits']
        ACS = False
    elif grism_element == 'G800L':
        bg_fixed = ['ACS.WFC.CHIP%d.msky.1.smooth.fits' %({1:2,2:1}[ext])]
        #bg_fixed = ['ACS.WFC.CHIP%d.msky.1.fits' %({1:2,2:1}[ext])]
        bg_vary = []
        ACS = True
        
        flat_files = {'G800L':'n6u12592j_pfl.fits'} # F814W
        flat_file = flat_files[grism_element]        
        flat_im = pyfits.open(os.path.join(os.getenv('jref'), flat_file))
        flat = flat_im['SCI',ext].data.flatten()
    
    if verbose:
        print '%s: EXTVER=%d / %s / %s' %(grism['product'], ext, bg_fixed,
                                       bg_vary)
    if not ACS:
        ext = 1
        
    ### Read sky files    
    data_fixed = []
    for file in bg_fixed:
        im = pyfits.open('%s/CONF/%s' %(os.getenv('GRIZLI'), file))
        sh = im[0].data.shape
        data = im[0].data.flatten()/flat
        data_fixed.append(data)
        
    data_vary = []
    for file in bg_vary:
        im = pyfits.open('%s/CONF/%s' %(os.getenv('GRIZLI'),
                                        file))[0].data.flatten()
        data_vary.append(im)
    
    yp, xp = np.indices(sh)
    
    ### Hard-coded (1014,1014) WFC3/IR images
    Npix = sh[0]*sh[1]
    Nexp = len(grism['files'])
    Nfix = len(data_fixed)
    Nvary = len(data_vary)
    Nimg = Nexp*Nvary + Nfix
    
    A = np.zeros((Npix*Nexp, Nimg))
    data = np.zeros(Npix*Nexp)
    wht = data*0.    
    mask = data > -1
    medians = np.zeros(Nexp)
    exptime = np.ones(Nexp)
    
    ### Build combined arrays
    
    for i in range(Nexp):
        flt = pyfits.open(grism['files'][i])
        dq_mask = flt['DQ',ext].data == 0
        
        ## Data
        data[i*Npix:(i+1)*Npix] = (flt['SCI',ext].data*dq_mask).flatten()
        mask[i*Npix:(i+1)*Npix] &= dq_mask.flatten() #== 0
        wht[i*Npix:(i+1)*Npix] = 1./(flt['ERR',ext].data**2*dq_mask).flatten()
        wht[~np.isfinite(wht)] = 0.
        
        if ACS:
            exptime[i] = flt[0].header['EXPTIME']
            data[i*Npix:(i+1)*Npix] /= exptime[i]
            wht[i*Npix:(i+1)*Npix] *= exptime[i]**2

            medians[i] = np.median(flt['SCI',ext].data[dq_mask]/exptime[i])
        else:
            medians[i] = np.median(flt['SCI',ext].data[dq_mask])
            
        ## Fixed arrays      
        for j in range(Nfix):
            for k in range(Nexp):
                A[k*Npix:(k+1)*Npix,j] = data_fixed[j]
            
            mask_j = (data_fixed[j] > 0) & np.isfinite(data_fixed[j])
            mask[i*Npix:(i+1)*Npix] &= mask_j
        
        ## Variable arrays    
        for j in range(Nvary):
            k = Nfix+j+Nvary*i
            A[i*Npix:(i+1)*Npix,k] = data_vary[j]
            mask[i*Npix:(i+1)*Npix] &= np.isfinite(data_vary[j])
                
    ### Initial coeffs based on image medians
    coeffs = np.array([np.min(medians)])
    if Nvary > 0:
        coeffs = np.hstack((coeffs, np.zeros(Nexp*Nvary)))
        coeffs[1::Nvary] = medians-medians.min()
        
    model = np.dot(A, coeffs)
    
    for iter in range(10):
        model = np.dot(A, coeffs)
        resid = (data-model)*np.sqrt(wht)
        obj_mask = (resid < 2.5) & (resid > -3)
        for j in range(Nexp):
            obj_j = nd.minimum_filter(obj_mask[j*Npix:(j+1)*Npix], size=30)
            obj_mask[j*Npix:(j+1)*Npix] = (obj_j > 0).flatten()
        
        if False:
            j = 1
            mask_i = (obj_mask & mask)[j*Npix:(j+1)*Npix].reshape(sh)
            r_i = (data-model)[j*Npix:(j+1)*Npix].reshape(sh)
            ds9.view(r_i * mask_i)
        
        if verbose:
            print '   %s > Iter: %d, masked: %d, %s' %(grism['product'], iter,
                                                obj_mask.sum(), coeffs)
                                                
        out = np.linalg.lstsq(A[mask & obj_mask,:], data[mask & obj_mask])
        coeffs = out[0]
        
    ### Best-fit sky
    sky = np.dot(A, coeffs).reshape(Nexp, Npix)
        
    ## log file
    fp = open('%s_%d_sky_background.info' %(grism['product'], ext), 'w')
    fp.write('# file c1 %s\n' %(' '.join(['c%d' %(v+2) 
                                            for v in range(Nvary)])))
    
    fp.write('# %s\n' %(grism['product']))
    
    fp.write('# bg1: %s\n' %(bg_fixed[0]))
    for v in range(Nvary):
        fp.write('# bg%d: %s\n' %(v+2, bg_vary[v]))
    
    for j in range(Nexp):
        file = grism['files'][j]
        line = '%s %9.4f' %(file, coeffs[0])           
        for v in range(Nvary):
            k = Nfix + j*Nvary + v
            line = '%s %9.4f' %(line, coeffs[k])
        
        fp.write(line+'\n')
    
    fp.close()
    
    if apply:
        for j in range(Nexp):
            file = grism['files'][j]
            
            flt = pyfits.open(file, mode='update')
            flt['SCI',ext].data -= sky[j,:].reshape(sh)*exptime[j]
        
            flt[0].header['GSKYCOL%d' %(ext)] = (False, 
                                                 'Subtract column average')
            flt[0].header['GSKYN%d' %(ext)] = (Nfix+Nvary, 
                                               'Number of sky images')
            flt[0].header['GSKY%d01' %(ext)] = (coeffs[0], 
                                        'Sky image %s (fixed)' %(bg_fixed[0]))
            
            flt[0].header['GSKY%d01F' %(ext)] = (bg_fixed[0], 
                                                  'Sky image (fixed)')
            
            for v in range(Nvary):
                k = Nfix + j*Nvary + v
                #print coeffs[k]
                flt[0].header['GSKY%d%02d' %(ext, v+Nfix+1)] = (coeffs[k], 
                                     'Sky image %s (variable)' %(bg_vary[v]))
                
                flt[0].header['GSKY%d%02dF' %(ext, v+Nfix+1)] = (bg_vary[v], 
                                     'Sky image (variable)')
                
            flt.flush()
    
    ### Don't do `column_average` for ACS
    if (not column_average) | ACS:
        return ACS
        
    ######
    ### Now fit residual column average & make diagnostic plot
    interactive_status=plt.rcParams['interactive']
    plt.ioff()
    
    fig = plt.figure(figsize=[6.,6.])
    ax = fig.add_subplot(111)
    
    im_shape = (1014,1014)
    
    for j in range(Nexp):
        resid = (data[j*Npix:(j+1)*Npix] - sky[j,:]).reshape(im_shape)
        m = (mask & obj_mask)[j*Npix:(j+1)*Npix].reshape(im_shape)
        
        ## Statistics of masked arrays    
        ma = np.ma.masked_array(resid, mask=(~m))
        med = np.ma.median(ma, axis=0)
    
        bg_sky = 1
        yrms = np.ma.std(ma, axis=0)/np.sqrt(np.sum(m, axis=0))
        xmsk = np.arange(im_shape[0])
        yres = med
        yok = ~yrms.mask
        
        ### Fit column average with smoothed Gaussian Process model
        gp = GaussianProcess(regr='constant', corr='squared_exponential',
                             theta0=8, thetaL=5, thetaU=12,
                             nugget=(yrms/bg_sky)[yok][::1]**2,
                             random_start=10, verbose=True, normalize=True)
                             
        gp.fit(np.atleast_2d(xmsk[yok][::1]).T, yres[yok][::1]+bg_sky)
        y_pred, MSE = gp.predict(np.atleast_2d(xmsk).T, eval_MSE=True)
        gp_sigma = np.sqrt(MSE)
        
        ## Plot Results
        pi = ax.plot(med[0:2], alpha=0.2)
        ax.plot(y_pred-1, color=pi[0].get_color())
        ax.fill_between(xmsk, y_pred-1-gp_sigma, y_pred-1+gp_sigma,
                        color=pi[0].get_color(), alpha=0.3,
                        label=grism['files'][j])
        
        ## result
        file = grism['files'][j]
        fp = open(file.replace('_flt.fits', '_column.dat'), 'w')
        fp.write('# column resid uncertainty\n')
        np.savetxt(fp, np.array([xmsk, y_pred-1, gp_sigma]).T, fmt='%.5f')
        fp.close()
        
        if apply:
            ### Subtract the column average in 2D & log header keywords
            gp_res = np.dot(y_pred[:,None]-1, np.ones((1014,1)).T).T
            flt = pyfits.open(file, mode='update')
            flt['SCI',1].data -= gp_res 
            flt[0].header['GSKYCOL'] = (True, 'Subtract column average')
            flt.flush()
                
    ### Finish plot      
    ax.legend(loc='lower left', fontsize=10)
    ax.plot([-10,1024],[0,0], color='k')
    ax.set_xlim(-10,1024)
    ax.set_xlabel(r'pixel column ($x$)')
    ax.set_ylabel(r'column average (e-/s)')
    ax.set_title(grism['product'])
    ax.grid()
    
    fig.tight_layout(pad=0.1)
    fig.savefig('%s_column.png' %(grism['product']))
    #fig.savefig('%s_column.pdf' %(grism['product']))
    plt.close()
    
    if interactive_status:
        plt.ion()
    
    return False