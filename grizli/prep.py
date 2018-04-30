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
from . import model

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
            ### WFC3
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
    
    ### Sewpy
#     try:
#         import sewpy
#     except:
#         print("""
# `sewpy` module needed for wrapping SExtractor within python.  
# Get it from https://github.com/megalut/sewpy.
# """)
#         

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
        
def fresh_flt_file(file, preserve_dq=False, path='../RAW/', verbose=True, extra_badpix=True, apply_grism_skysub=True, crclean=False, mask_regions=True):
    """Copy "fresh" unmodified version of a data file from some central location
    
    TBD
    
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
    import shutil
    
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
    
    ### Copy calibs for ACS/UVIS files
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
        
        ### Grism FLT from IR amplifier gain
        pfl_file = orig_file[0].header['PFLTFILE'].replace('iref$',
                                                           os.getenv('iref'))
        grism_pfl = pyfits.open(pfl_file)[1].data[5:-5,5:-5]
        
        orig_file['DQ'].data |= 4*flat_dq
        orig_file['SCI'].data *= grism_pfl/flat
        
        # if apply_grism_skysub:
        #     if 'GSKY001' in orig_file:
    
    if filter == 'G280':
        ### Use F200LP flat
        flat_files = {'G280':'zcv2053ei_pfl.fits'} # F200LP
        flat_file = flat_files[filter]
        extra_msg = ' / flat: {0}'.format(flat_file)
        
        flat_im = pyfits.open(os.path.join(os.getenv('jref'), flat_file))

        for ext in [1,2]:
            flat = flat_im['SCI',ext].data
            flat_dq = (flat < 0.2)
                        
            orig_file['DQ',ext].data |= 4*flat_dq
            orig_file['SCI',ext].data *= 1./flat

    if filter == 'G800L':
        flat_files = {'G800L':'n6u12592j_pfl.fits'} # F814W
        flat_file = flat_files[filter]
        extra_msg = ' / flat: {0}'.format(flat_file)
        
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
        
        if orig_file[0].header['NPOLFILE'] == 'N/A':
            # Use an F814W file, but this should be updated
            orig_file[0].header['NPOLFILE'] = 'jref$v971826jj_npl.fits'
    
    if head['INSTRUME'] == 'WFPC2':
        head['DETECTOR'] = 'WFPC2'
            
    if (head['INSTRUME'] == 'WFC3') & (head['DETECTOR'] == 'IR')&extra_badpix: 
        bp = pyfits.open(os.path.join(os.getenv('iref'),
                                      'badpix_spars200_Nov9.fits'))    
        
        if orig_file['DQ'].data.shape == bp[0].data.shape:
            orig_file['DQ'].data |= bp[0].data
        
        extra_msg += ' / bpix: $iref/badpix_spars200_Nov9.fits'
    
    if crclean:
        import lacosmicx
        for ext in [1,2]:
            print('Clean CRs with LACosmic, extension {0:d}'.format(ext))
            
            sci = orig_file['SCI',ext].data
            dq = orig_file['DQ',ext].data
            
            crmask, clean = lacosmicx.lacosmicx(sci, inmask=None,
                         sigclip=4.5, sigfrac=0.3, objlim=5.0, gain=1.0,
                         readnoise=6.5, satlevel=65536.0, pssl=0.0, niter=4,
                         sepmed=True, cleantype='meanmask', fsmode='median',
                         psfmodel='gauss', psffwhm=2.5,psfsize=7, psfk=None,
                         psfbeta=4.765, verbose=False)
            
            dq[crmask] |= 1024
            sci[crmask] = 0
                                    
    if verbose:
        print('{0} -> {1} {2}'.format(orig_file.filename(), local_file, extra_msg))
        
    ### WFPC2            
    if '_c0m' in file:
        # point to FITS reference files
        for key in ['MASKFILE', 'ATODFILE', 'BLEVFILE', 'BLEVDFIL', 'BIASFILE', 'BIASDFIL', 'DARKFILE', 'DARKDFIL', 'FLATFILE', 'FLATDFIL', 'SHADFILE']:
            ref_file = '_'.join(head[key].split('.'))+'.fits'
            orig_file[0].header[key] = ref_file.replace('h.fits', 'f.fits')
        
        waiv = orig_file[0].header['FLATFILE']
        orig_file[0].header['FLATFILE'] = waiv.replace('.fits', '_c0h.fits')
        # 
        # ## testing
        # orig_file[0].header['FLATFILE'] = 'm341820ju_pfl.fits'
        
        # Copy WFPC2 DQ file (c1m)
        dqfile = os.path.join(path, file.replace('_c0m', '_c1m'))
        print('Copy WFPC2 DQ file: {0}'.format(dqfile))
        if os.path.exists(os.path.basename(dqfile)):
            os.remove(os.path.basename(dqfile))
           
        shutil.copy(dqfile, './')
        
        ## Add additional masking since AstroDrizzle having trouble with flats
        flat_file = orig_file[0].header['FLATFILE'].replace('uref$', os.getenv('uref')+'/')
        pfl = pyfits.open(flat_file)
        c1m = pyfits.open(os.path.basename(dqfile), mode='update')
        for ext in [1,2,3,4]:
            mask = pfl[ext].data > 1.3
            c1m[ext].data[mask] |= 2
        
        c1m.flush()
        
    orig_file.writeto(local_file, clobber=True)
    
    if mask_regions:
        apply_region_mask(local_file, dq_value=1024)
    
def apply_persistence_mask(flt_file, path='../Persistence', dq_value=1024,
                           err_threshold=0.6, grow_mask=3, subtract=True,
                           verbose=True):
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
    
    subtract : bool
        Subtract the persistence model itself from the SCI extension.
        
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
            print('Persistence file {0} not found'.format(pers_file))
        
        #return 0
    
    pers = pyfits.open(pers_file)
    
    pers_mask = pers['SCI'].data > err_threshold*flt['ERR'].data
    
    if grow_mask > 0:
        pers_mask = nd.maximum_filter(pers_mask*1, size=grow_mask)
    else:
        pers_mask = pers_mask * 1
    
    NPERS = pers_mask.sum()
    if verbose:
        print('{0}: flagged {1:d} pixels affected by persistence (pers/err={2:.2f})'.format(pers_file, NPERS, err_threshold))
    
    if NPERS > 0:
        flt['DQ'].data[pers_mask > 0] |= dq_value
        if subtract:
            dont_subtract=False
            if 'SUBPERS' in flt[0].header:
                if flt[0].header['SUBPERS']:
                    dont_subtract = True
                    
            if not dont_subtract:
                flt['SCI'].data -= pers['SCI'].data
            
            flt['ERR'].data = np.sqrt(flt['ERR'].data**2+pers['SCI'].data**2)
            flt[0].header['SUBPERS'] = (True, 'Persistence model subtracted')
            
        flt.flush()

def apply_region_mask(flt_file, dq_value=1024, verbose=True):
    """Apply DQ mask from a DS9 region file
    
    Parameters
    ----------
    flt_file : str
        Filename of the FLT exposure
    
    dq_value : int
        DQ bit to flip for affected pixels
    
    Searches for region files with filenames like 
    `flt_file.replace('_flt.fits','.[ext].mask.reg')`, where `[ext]` is an 
    integer referring to the SCI extension in the FLT file.

    """
    import pyregion
    
    mask_files = glob.glob(flt_file.replace('_flt.fits','.*.mask.reg').replace('_flc.fits','.*.mask.reg').replace('_c0m.fits','.*.mask.reg'))
    if len(mask_files) == 0:
        return True
     
    if verbose:
        print('Region mask for {0}: {1}'.format(flt_file, mask_files))
    
    flt = pyfits.open(flt_file, mode='update')
    for mask_file in mask_files:
        ext = int(mask_file.split('.')[-3])
        try:
            reg = pyregion.open(mask_file).as_imagecoord(flt['SCI',ext].header)
            mask = reg.get_mask(hdu=flt['SCI',ext])
        except:
            # Above fails for lookup-table distortion (ACS / UVIS)
            # Here just assume the region file is defined in image coords
            reg = pyregion.open(mask_file)
            mask = reg.get_mask(shape=flt['SCI',ext].data.shape)
             
        flt['DQ',ext].data[mask] |= dq_value
    
    flt.flush()
    return True
    
def apply_saturated_mask(flt_file, dq_value=1024):
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
    
    ## Don't flag pixels in lower right corner
    sat[:80,-80:] = False
    
    ## Flag only if a number of nearby pixels also saturated
    kern = np.ones((3,3))
    sat_grow = nd.convolve(sat*1, kern)
    
    sat_mask = (sat & (sat_grow > 2))[::-1,:]*1
    
    NSAT = sat_mask.sum()
    if verbose:
        print('{0}: flagged {1:d} pixels affected by saturation pulldown'.format(flt_file, NSAT))
    
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
        print('No matches within `clip={0:f}`'.format(clip))
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
    import copy
    from astropy.table import Table    
    
    import skimage.transform
    from skimage.measure import ransac

    import stsci.stimage
    
    if transform is None:
        transform = skimage.transform.SimilarityTransform
        
    #print 'xyxymatch'
    if (len(output) == 0) | (len(input) == 0):
        print('No entries!')
        return input, output, None, transform()
    
    match = stsci.stimage.xyxymatch(copy.copy(input), copy.copy(output), 
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
                         verbose=True, guess=[0., 0., 0., 1]):
    """TBD
    """
    if hasattr(radec, 'upper'):
        rd_ref = np.loadtxt(radec)
    else:
        rd_ref = radec*1
        
    if not os.path.exists('{0}.cat.fits'.format(root)):
        #cat = make_drz_catalog(root=root)
        cat = make_SEP_catalog(root=root)
    else:
        cat = Table.read('{0}.cat.fits'.format(root))
    
    ### Clip obviously distant files to speed up match
    rd_cat = np.array([cat['X_WORLD'], cat['Y_WORLD']])
    rd_cat_center = np.median(rd_cat, axis=1)
    cosdec = np.array([np.cos(rd_cat_center[1]/180*np.pi),1])
    dr_cat = np.sqrt(np.sum((rd_cat.T-rd_cat_center)**2*cosdec**2, axis=1))
    
    #print('xxx', rd_ref.shape, rd_cat_center.shape, cosdec.shape)
    
    dr = np.sqrt(np.sum((rd_ref-rd_cat_center)**2*cosdec**2, axis=1))
    
    rd_ref = rd_ref[dr < 1.1*dr_cat.max(),:]
    
    ok = (cat['MAG_AUTO'] > mag_limits[0]) & (cat['MAG_AUTO'] < mag_limits[1])
    if ok.sum() == 0:
        print('{0}.cat: no objects found in magnitude range {1}'.format(root,
                                                                 mag_limits))
        return False
    
    xy_drz = np.array([cat['X_IMAGE'][ok], cat['Y_IMAGE'][ok]]).T
    
    drz_file = glob.glob('{0}_dr[zc]_sci.fits'.format(root))[0]
    drz_im = pyfits.open(drz_file)
    sh = drz_im[0].data.shape
    
    drz_wcs = pywcs.WCS(drz_im[0].header, relax=True)
    orig_wcs = drz_wcs.copy()
    
    #out_shift, out_rot, out_scale = np.zeros(2), 0., 1.
    out_shift, out_rot, out_scale = guess[:2], guess[2], guess[3]    
    drz_wcs = utils.transform_wcs(drz_wcs, out_shift, out_rot, out_scale)
    print('{0} (guess)   : {1:6.2f} {2:6.2f} {3:7.3f} {4:7.3f}'.format(root, guess[0], guess[1], guess[2]/np.pi*180, 1./guess[3]))
        
    NGOOD, rms = 0, 0
    for iter in range(NITER):
        #print('xx iter {0} {1}'.format(iter, NITER))
        xy = np.array(drz_wcs.all_world2pix(rd_ref, 0))
        pix = np.cast[int](np.round(xy)).T

        ### Find objects where drz pixels are non-zero
        okp = (pix[0,:] > 0) & (pix[1,:] > 0)
        okp &= (pix[0,:] < sh[1]) & (pix[1,:] < sh[0])
        ok2 = drz_im[0].data[pix[1,okp], pix[0,okp]] != 0

        N = ok2.sum()
        status = clip_lists(xy_drz, xy+1, clip=clip)
        if not status:
            print('Problem xxx')
        
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
        
        #print(output.shape, output_ix.shape, output_ix.min(), output_ix.max(), titer, toler, input_ix.shape, input.shape)
              
        titer = 0 
        while (len(input_ix)*1./len(input) < 0.1) & (titer < 3):
            titer += 1
            toler += 5
            try:
                res = match_lists(output, input, scl=1., simple=True,
                              outlier_threshold=outlier_threshold,
                              toler=toler)
            except:
                pass
                
            output_ix, input_ix, outliers, tf = res
        
        #print(output.shape, output_ix.shape, output_ix.min(), output_ix.max(), titer, toler, input_ix.shape, input.shape)
        
        tf_out = tf(output[output_ix])
        dx = input[input_ix] - tf_out
        rms = utils.nmad(np.sqrt((dx**2).sum(axis=1)))
        #outliers = outliers | (np.sqrt((dx**2).sum(axis=1)) > 4*rms)
        outliers = (np.sqrt((dx**2).sum(axis=1)) > 4*rms)
                                          
        if outliers.sum() > 0:
            res2 = match_lists(output[output_ix][~outliers],
                              input[input_ix][~outliers], scl=1., simple=True,
                              outlier_threshold=outlier_threshold,
                              toler=toler)
            
            output_ix2, input_ix2, outliers2, tf = res2
        
        if verbose:
            shift = tf.translation
            NGOOD = (~outliers).sum()
            print('{0} ({1:d}) {2:d}: {3:6.2f} {4:6.2f} {5:7.3f} {6:7.3f}'.format(root,iter,NGOOD,
                                                   shift[0], shift[1], 
                                                   tf.rotation/np.pi*180, 
                                                   1./tf.scale))
        
        out_shift += tf.translation
        out_rot -= tf.rotation
        out_scale *= tf.scale
        
        drz_wcs = utils.transform_wcs(drz_wcs, tf.translation, tf.rotation, 
                                      tf.scale)
                                      
        # drz_wcs.wcs.crpix += tf.translation
        # theta = -tf.rotation
        # _mat = np.array([[np.cos(theta), -np.sin(theta)],
        #                  [np.sin(theta), np.cos(theta)]])
        # 
        # drz_wcs.wcs.cd = np.dot(drz_wcs.wcs.cd, _mat)/tf.scale
                
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
        fig.savefig('{0}_wcs.png'.format(root))
        plt.close()
        
        if interactive_status:
            plt.ion()
        
    log_wcs(root, orig_wcs, out_shift, out_rot/np.pi*180, out_scale, rms,
            n=NGOOD, initialize=False)
            
    return orig_wcs, drz_wcs, out_shift, out_rot/np.pi*180, out_scale

def log_wcs(root, drz_wcs, shift, rot, scale, rms=0., n=-1, initialize=True):
    """Save WCS offset information to a file
    """
    if (not os.path.exists('{0}_wcs.log'.format(root))) | initialize:
        print('Initialize {0}_wcs.log'.format(root))
        orig_hdul = pyfits.HDUList()
        fp = open('{0}_wcs.log'.format(root), 'w')
        fp.write('# ext xshift yshift rot scale rms N\n')
        fp.write('# {0}\n'.format(root))
        count = 0
    else:
        orig_hdul = pyfits.open('{0}_wcs.fits'.format(root))
        fp = open('{0}_wcs.log'.format(root), 'a')
        count = len(orig_hdul)
    
    hdu = drz_wcs.to_fits()[0]
    orig_hdul.append(hdu)
    orig_hdul.writeto('{0}_wcs.fits'.format(root), clobber=True)
    
    fp.write('{0:5d} {1:13.4f} {2:13.4f} {3:13.4f} {4:13.5f} {5:13.3f} {6:4d}\n'.format(
              count, shift[0], shift[1], rot, scale, rms, n))
              
    fp.close()

def table_to_radec(table, output='coords.radec'):
    """Make a DS9 region file from a table object
    """

    if 'X_WORLD' in table.colnames:
        rc, dc = 'X_WORLD', 'Y_WORLD'
    else:
        rc, dc = 'ra', 'dec'
    
    table[rc, dc].write(output, format='ascii.commented_header', 
                        overwrite=True)
    
def table_to_regions(table, output='ds9.reg', comment=None):
    """Make a DS9 region file from a table object
    """
    fp = open(output,'w')
    fp.write('fk5\n')
    
    if 'X_WORLD' in table.colnames:
        rc, dc = 'X_WORLD', 'Y_WORLD'
    else:
        rc, dc = 'ra', 'dec'
    
    ### GAIA
    if 'solution_id' in table.colnames:
        e = np.sqrt(table['ra_error']**2+table['dec_error']**2)/1000.
        e = np.maximum(e, 0.1)
    else:
        e  = np.ones(len(table))*0.5
    
    lines = ['circle({0:.7f}, {1:.7f}, {2:.3f}")\n'.format(table[rc][i],
                                                           table[dc][i], e[i])
                                              for i in range(len(table))]
    
    if comment is not None:
        for i in range(len(table)):
            lines[i] = '{0} # text={{{1}}}\n'.format(lines[i].strip(), comment[i])
                                                      
    fp.writelines(lines)
    fp.close()
    
SEXTRACTOR_DEFAULT_PARAMS = ["NUMBER", "X_IMAGE", "Y_IMAGE", "X_WORLD",
                    "Y_WORLD", "A_IMAGE", "B_IMAGE", "THETA_IMAGE", 
                    "MAG_AUTO", "MAGERR_AUTO", "FLUX_AUTO", "FLUXERR_AUTO",
                    "FLUX_RADIUS", "BACKGROUND", "FLAGS"]

SEXTRACTOR_PHOT_APERTURES = "6, 8.335, 16.337, 20"
                    
SEXTRACTOR_CONFIG_3DHST = {'DETECT_MINAREA':14, 'DEBLEND_NTHRESH':32, 'DEBLEND_MINCONT':0.005, 'FILTER_NAME':'/usr/local/share/sextractor/gauss_3.0_7x7.conv', 'FILTER':'Y'}

# /usr/local/share/sextractor/gauss_3.0_7x7.conv
GAUSS_3_7x7 = np.array(
[[ 0.004963,  0.021388,  0.051328,  0.068707,  0.051328,  0.021388,  0.004963], 
 [ 0.021388,  0.092163,  0.221178,  0.296069,  0.221178,  0.092163,  0.021388], 
 [ 0.051328,  0.221178,  0.530797,  0.710525,  0.530797,  0.221178,  0.051328], 
 [ 0.068707,  0.296069,  0.710525,  0.951108,  0.710525,  0.296069,  0.068707], 
 [ 0.051328,  0.221178,  0.530797,  0.710525,  0.530797,  0.221178,  0.051328], 
 [ 0.021388,  0.092163,  0.221178,  0.296069,  0.221178,  0.092163,  0.021388], 
 [ 0.004963,  0.021388,  0.051328,  0.068707,  0.051328,  0.021388,  0.004963]])

def make_SEP_catalog(root='',threshold=2., get_background=True, 
                      verbose=True, extra_config={}, sci=None, wht=None, 
                      phot_apertures=SEXTRACTOR_PHOT_APERTURES,
                      filter_kernel=GAUSS_3_7x7, filter_type='conv',
                      clean=True, rescale_weight=True, minarea=14,
                      uppercase_columns=True, save_to_fits=True,
                      source_xy=None,
                      **kwargs):
    """Make a catalog from drizzle products using the SEP implementation of SExtractor

    """
    import copy
    import astropy.units as u
    import sep

    if sci is not None:
        drz_file = sci
    else:
        drz_file = glob.glob('{0}_dr[zc]_sci.fits'.format(root))[0]

    im = pyfits.open(drz_file)

    ## Get AB zeropoint
    if 'PHOTFNU' in im[0].header:
        ZP = -2.5*np.log10(im[0].header['PHOTFNU'])+8.90
    elif 'PHOTFLAM' in im[0].header:
        ZP = (-2.5*np.log10(im[0].header['PHOTFLAM']) - 21.10 -
              5*np.log10(im[0].header['PHOTPLAM']) + 18.6921)
    elif 'FILTER' in im[0].header:
        fi = im[0].header['FILTER'].upper()
        if fi in model.photflam_list:
            ZP = (-2.5*np.log10(model.photflam_list[fi]) - 21.10 -
                  5*np.log10(model.photplam_list[fi]) + 18.6921)
        else:
            print('Couldn\'t find PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25') 
            ZP = 25
    else:
        print('Couldn\'t find FILTER, PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25') 
        ZP = 25

    if verbose:
        print('Image AB zeropoint: {0:.3f}'.format(ZP))

    # Scale fluxes to mico-Jy
    uJy_to_dn = 1/(3631*1e6*10**(-0.4*ZP))

    weight_file = drz_file.replace('_sci.fits', '_wht.fits').replace('_drz.fits', '_wht.fits')
    if (weight_file == drz_file) | (not os.path.exists(weight_file)):
        WEIGHT_TYPE = "NONE"
        weight_file = None
    else:
        WEIGHT_TYPE = "MAP_WEIGHT"

    drz_im = pyfits.open(drz_file)
    data = drz_im[0].data.byteswap().newbyteorder()
    wcs = pywcs.WCS(drz_im[0].header)

    if weight_file is not None:
        wht_im = pyfits.open(weight_file)
        wht_data = wht_im[0].data.byteswap().newbyteorder()

        err = 1/np.sqrt(wht_data)
        err[~np.isfinite(err)] = 0
        mask = (err == 0) 
    else:
        mask = (data == 0)
        err = None

    if get_background:
        bkg = sep.Background(data, mask=mask, bw=32, bh=32, fw=3, fh=3)
        bkg_data = bkg.back()

        pyfits.writeto('{0}_bkg.fits'.format(root), data=bkg_data,
                    header=utils.to_header(wcs), overwrite=True)

        if err is None:
         err = bkg.rms()

        ratio = bkg.rms()/err
        err_scale = np.median(ratio[(~mask) & np.isfinite(ratio)])

    else:
        bkg_data = 0.
        err_scale = 1.

    if rescale_weight:
        err *= err_scale
     
    #mask = None

    if source_xy is None:
        ### Run the detection
        objects, seg = sep.extract(data - bkg_data, 
                           thresh=threshold, err=err, mask=mask, 
                           minarea=minarea,
                           filter_kernel=filter_kernel,
                           filter_type=filter_type, deblend_nthresh=32, 
                           deblend_cont=0.005, clean=clean, clean_param=1.,
                           segmentation_map=True)

        tab = utils.GTable(objects)

        # ID
        tab['number'] = np.arange(len(tab), dtype=np.int32)+1

        ## Segmentation
        pyfits.writeto('{0}_seg.fits'.format(root), data=seg,
                       header=utils.to_header(wcs), overwrite=True)

        for c in ['a','b']:
            tab = tab[np.isfinite(tab[c])]

        # WCS coordinates
        tab['ra'], tab['dec'] = wcs.all_pix2world(tab['x'], tab['y'], 0)
        tab['ra'].unit = u.deg
        tab['dec'].unit = u.deg
        tab['x_world'], tab['y_world'] = tab['ra'], tab['dec']

        tab.meta['MINAREA'] = (minarea, 'Minimum source area in pixels')
        tab.meta['CLEAN'] = clean
        tab.meta['FILTER_TYPE'] = (filter_type, 'Type of filter applied, conv or weight')
        tab.meta['THRESHOLD'] = (threshold, 'Detection threshold')

        ## FLUX_AUTO
        # https://sep.readthedocs.io/en/v1.0.x/apertures.html#equivalent-of-flux-auto-e-g-mag-auto-in-source-extractor
        kronrad, krflag = sep.kron_radius(data - bkg_data, tab['x'], tab['y'],
                                       tab['a'], tab['b'], tab['theta'], 6.0)

        #kronrad *= 2.5
        kronrad[~np.isfinite(kronrad)] = 1.75*2
        
        kron_out = sep.sum_ellipse(data - bkg_data, tab['x'], tab['y'], 
                                tab['a'], tab['b'], tab['theta'], 
                                2.5*kronrad, subpix=5)

        kron_flux, kron_fluxerr, kron_flag = kron_out

        # Minimum radius = 3.5, PHOT_AUTOPARAMS 2.5, 3.5
        r_min = 1.75*2
        use_circle = kronrad * np.sqrt(tab['a'] * tab['b']) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(data - bkg_data,
                                             tab['x'][use_circle], 
                                             tab['y'][use_circle],
                                             r_min, subpix=5)

        kron_flux[use_circle] = cflux
        kron_fluxerr[use_circle] = cfluxerr
        kronrad[use_circle] = r_min

        tab['flux_auto'] = kron_flux/uJy_to_dn*u.uJy
        tab['fluxerr_auto'] = kron_fluxerr/uJy_to_dn*u.uJy

        if get_background:
            kron_out = sep.sum_ellipse(bkg_data, tab['x'], tab['y'], tab['a'], tab['b'],
                                    tab['theta'], 2.5*kronrad, subpix=1)

            kron_bkg, kron_bkg_fluxerr, kron_flag = kron_out
            tab['flux_bkg_auto'] = kron_bkg/uJy_to_dn*u.uJy
        else:
            tab['flux_bkg_auto'] = 0.

        tab['mag_auto'] = ZP - 2.5*np.log10(kron_flux)
        tab['magerr_auto'] = 2.5/np.log(10)*kron_fluxerr/kron_flux

        tab['kron_radius'] = kronrad*u.pixel
        tab['kron_flag'] = krflag

        ## FLUX_RADIUS
        # https://sep.readthedocs.io/en/v1.0.x/apertures.html#equivalent-of-flux-radius-in-source-extractor
        fr, fr_flag = sep.flux_radius(data - bkg_data, tab['x'], tab['y'],
                                      tab['a']*6, 0.5, normflux=kron_flux)
        tab['flux_radius'] = fr*u.pixel

        fr, fr_flag = sep.flux_radius(data - bkg_data, tab['x'], tab['y'],
                                      tab['a']*6, 0.9, normflux=kron_flux)
        tab['flux_radius_90'] = fr*u.pixel

        ## Bad DQ
        bad = (tab['flux_auto'] <= 0) | (tab['flux_radius'] <= 0)
        tab = tab[~bad]
        
        # for id in tab['number'][bad]:
        #     is_seg = seg == id
        #     seg[is_seg] = 0

        for c in ['cflux','flux','peak','cpeak']:
            tab[c] *= 1. / uJy_to_dn
            tab[c].unit = u.uJy
        
        source_x, source_y = tab['x'], tab['y']

        # Rename to look like SExtractor
        for c in ['x','y','a','b','theta','cxx','cxy','cyy','x2','y2','xy']:
            tab.rename_column(c, c+'_image')

    else:
        source_x, source_y = source_xy
        
        if hasattr(source_x, 'unit'):
            if source_x.unit == u.deg:
                ra, dec = source_xy
                source_x, source_y = wcs.all_world2pix(ra, dec, 0)
                
        tab = utils.GTable()

    # Info
    tab.meta['ZP'] = (ZP, 'AB zeropoint')
    if 'PHOTPLAM' in im[0].header:
        tab.meta['PLAM'] = (im[0].header['PHOTPLAM'], 'AB zeropoint')
        tab.meta['FNU'] = (im[0].header['PHOTFNU'], 'AB zeropoint')
        tab.meta['FLAM'] = (im[0].header['PHOTFLAM'], 'AB zeropoint')
    
    tab.meta['uJy2dn'] = (uJy_to_dn, 'Convert uJy fluxes to image DN')

    tab.meta['DRZ_FILE'] = (drz_file, 'SCI file')
    tab.meta['WHT_FILE'] = (weight_file, 'WHT file')

    tab.meta['GET_BACK'] = (get_background, 'Background computed')
    tab.meta['ERR_SCALE'] = (err_scale, 'Scale factor applied to weight image (like MAP_WEIGHT)')
    
    ## Photometry
    apertures = np.cast[float](phot_apertures.replace(',','').split())
    for iap, aper in enumerate(apertures):
        flux, fluxerr, flag = sep.sum_circle(data - bkg_data, 
                                      source_x, source_y,
                                      aper/2, err=err, 
                                      gain=2000., subpix=5)

        tab['flux_aper_{0}'.format(iap)] = flux/uJy_to_dn*u.uJy
        tab['fluxerr_aper_{0}'.format(iap)] = fluxerr/uJy_to_dn*u.uJy
        tab['flag_aper_{0}'.format(iap)] = flag

        if get_background:
            flux, fluxerr, flag = sep.sum_circle(bkg_data, 
                                          source_x, source_y,
                                          aper*2, err=err, gain=1.0)

            tab['bkg_aper_{0}'.format(iap)] = flux
        else:
            tab['bkg_aper_{0}'.format(iap)] = 0.

        tab.meta['aper_{0}'.format(iap)] = (aper, 'Aperture diameter, pix')
        
    if uppercase_columns:
        for c in tab.colnames:
            tab.rename_column(c, c.upper())
            
    if save_to_fits:
        tab.write('{0}.cat.fits'.format(root), format='fits', overwrite=True)

    if verbose:
        print('{0}.cat.fits: {1:d} objects'.format(root, len(tab)))

    return tab
     
def make_drz_catalog(root='', sexpath='sex',threshold=2., get_background=True, 
                     verbose=True, extra_config={}, sci=None, wht=None, 
                     get_sew=False, output_params=SEXTRACTOR_DEFAULT_PARAMS,
                     phot_apertures=SEXTRACTOR_PHOT_APERTURES):
    """Make a SExtractor catalog from drizzle products
    
    TBD
    """
    import copy
    import sewpy
    
    if sci is not None:
        drz_file = sci
    else:
        drz_file = glob.glob('{0}_dr[zc]_sci.fits'.format(root))[0]
    
    im = pyfits.open(drz_file)
    
    if 'PHOTFNU' in im[0].header:
        ZP = -2.5*np.log10(im[0].header['PHOTFNU'])+8.90
    elif 'PHOTFLAM' in im[0].header:
        ZP = (-2.5*np.log10(im[0].header['PHOTFLAM']) - 21.10 -
                 5*np.log10(im[0].header['PHOTPLAM']) + 18.6921)
    elif 'FILTER' in im[0].header:
        fi = im[0].header['FILTER'].upper()
        if fi in model.photflam_list:
            ZP = (-2.5*np.log10(model.photflam_list[fi]) - 21.10 -
                     5*np.log10(model.photplam_list[fi]) + 18.6921)
        else:
            print('Couldn\'t find PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25') 
            ZP = 25
    else:
        print('Couldn\'t find FILTER, PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25') 
        ZP = 25
        
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
              WEIGHT_TYPE= WEIGHT_TYPE,
              WEIGHT_IMAGE= weight_file,
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
    cat.write('{0}.cat'.format(root), format='ascii.commented_header',
              overwrite=True)
            
    if verbose:
        print('{0} catalog: {1:d} objects'.format(root, len(cat)))
    
    return cat
    
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
    
    ZP =  -2.5*np.log10(PHOTFLAM) - 21.10 - 5*np.log10(PHOTPLAM) + 18.6921
    
    wcs = pywcs.WCS(sci[0])
    pscale = utils.get_wcs_pscale(wcs)
    
    rd = wcs.all_pix2world(np.array([[sh[1]/2], [sh[0]/2]]).T, 0)[0]
    
    radius = np.sqrt(2)*np.maximum(sh[0], sh[1])/2.*pscale/60.
    
    if catalog == '2mass':
        cat = get_irsa_catalog(rd[0], rd[1], radius=radius, twomass=True)
        cat['mag'] = cat['h_m']+1.362 # AB
        table_to_regions(cat, '{0}_2mass.reg'.format(root))
    elif catalog == 'panstarrs':
        cat = get_panstarrs_catalog(rd[0], rd[1], radius=radius)
        #cat['mag'] = cat['rMeanKronMag']+0.14 # AB
        cat['mag'] = cat['iMeanKronMag']+0.35 # AB
        table_to_regions(cat, '{0}_panstarrs.reg'.format(root))
    elif catalog == 'ukidss':
        cat = get_ukidss_catalog(rd[0], rd[1], radius=radius)
        cat['mag'] = cat['HAperMag3']+1.362 # AB
        cat.rename_column('RA','ra')
        cat.rename_column('Dec','dec')
        table_to_regions(cat, '{0}_ukidss.reg'.format(root))
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
        #ds9.view(flux)
        
    clip = (wht[0].data == 0) & (flux > 1.e-6*flux.max())
    wht_val = np.percentile(wht[0].data, 95)
    wht[0].data[clip] = wht_val
    sci[0].data[clip] = flux[clip]
    
    sci.writeto(sci_file.replace('_drz', '_{0}_drz'.format(catalog)), 
                clobber=True)
    
    wht.writeto(wht_file.replace('_drz', '_{0}_drz'.format(catalog)), 
                clobber=True)  
                        
    if False:
        # Mask
        kern = (np.arange(flt.conf.conf['BEAMA'][1]) > flt.conf.conf['BEAMA'][0])*1.
        kern /= kern.sum()
        
        mask = flt.direct['REF'] == 0
        full_mask = nd.convolve(mask*1., kern.reshape((1,-1)), origin=(0,-kern.size//2+20))
            
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

def get_ukidss_catalog(ra=165., dec=34.8, radius=3, database='UKIDSSDR9PLUS',
                       programme_id='LAS'):
    """Query for objects in the UKIDSS catalogs
    
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

    from astroquery.ukidss import Ukidss
    
    coo = coord.SkyCoord(ra*u.deg, dec*u.deg)
    
    table = Ukidss.query_region(coo, radius=radius*u.arcmin,
                                database=database, programme_id=programme_id)
    
    return table
    
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

def get_irsa_catalog(ra=165.86, dec=34.829694, radius=3, catalog='allwise_p3as_psd', wise=False, twomass=False):
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
    
    #all_wise = 'wise_allwise_p3as_psd'
    #all_wise = 'allwise_p3as_psd'
    if wise:
        catalog = 'allwise_p3as_psd'
    elif twomass:
        catalog = 'fp_psc'
        
    coo = coord.SkyCoord(ra*u.deg, dec*u.deg)
    
    table = Irsa.query_region(coo, catalog=catalog, spatial="Cone",
                              radius=radius*u.arcmin, get_query_payload=False)
    
    return table

def get_gaia_radec_at_time(gaia_tbl, date=2015.5, format='decimalyear'):
    """
    Use `~astropy.coordinates.SkyCoord.apply_space_motion` to compute GAIA positions at a specific observation date
    
    Parameters
    ----------
    gaia_tbl : `~astropy.table.Table`
        GAIA table query, e.g., provided by `get_gaia_DR2_catalog`.
    
    date : e.g., float
        Observation date that can be parsed with `~astropy.time.Time`
        
    format : str
        Date format, see `~astropy.time.Time.FORMATS`.
    
    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        Projected sky coordinates.
    
    """
    import astropy.time 
    import pyia
    
    obstime = astropy.time.Time(date, format=format)
    g = pyia.GaiaData(gaia_tbl)
    coord_at_time = g.skycoord.apply_space_motion(obstime)
    return(coord_at_time)
    
def gaia_dr2_conesearch_query(ra=165.86, dec=34.829694, radius=3., max=100000):
    """
    Generate a query string for the TAP servers
    TBD
    
    Parameters
    ----------
    ra, dec : float
        RA, Dec in decimal degrees

    radius : float
        Search radius, in arc-minutes.
    
    Returns
    -------
    query : str
        Query string
        
    """
    query =  "SELECT TOP {3} * FROM gaiadr2.gaia_source  WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',{0},{1},{2:.2f}))=1".format(ra, dec, radius/60., max)
    return query
    
def get_gaia_DR2_catalog(ra=165.86, dec=34.829694, radius=3.,
                         use_mirror=False):
    """Query GAIA DR2 astrometric catalog
    
    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees
    
    radius : float
        Radius of the query, in arcmin
    
    use_mirror : bool
        If True, use the mirror at `gaia.ari.uni-heidelberg.de`.  Otherwise
        use `gea.esac.esa.int`.
        
    Returns
    -------
    table : `~astropy.table.Table`
        Result of the query
    
    """
    try:
        import httplib
        from urllib import urlencode
    except:
        # python 3
        import http.client as httplib
        from urllib.parse import urlencode
        
    #import http.client in Python 3
    #import urllib.parse in Python 3
    import time
    from xml.dom.minidom import parseString

    host = "gea.esac.esa.int"
    port = 80
    pathinfo = "/tap-server/tap/async"
    
    if use_mirror:
        host = "gaia.ari.uni-heidelberg.de"
        pathinfo = "/tap/async"
    
    #-------------------------------------
    #Create job

    query =  gaia_dr2_conesearch_query(ra=ra, dec=dec, radius=radius) #"SELECT TOP 100000 * FROM gaiadr2.gaia_source  WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',{0},{1},{2:.2f}))=1".format(ra, dec, radius/60.)
    print(query)
    
    params = urlencode({\
    	"REQUEST": "doQuery", \
    	"LANG":    "ADQL", \
    	"FORMAT":  "fits", \
    	"PHASE":  "RUN", \
    	"QUERY":  query
    	})

    headers = {\
    	"Content-type": "application/x-www-form-urlencoded", \
    	"Accept":       "text/plain" \
    	}

    connection = httplib.HTTPConnection(host, port)
    connection.request("POST",pathinfo,params,headers)

    #Status
    response = connection.getresponse()
    print("Status: " +str(response.status), "Reason: " + str(response.reason))

    #Server job location (URL)
    location = response.getheader("location")
    print("Location: " + location)

    #Jobid
    jobid = location[location.rfind('/')+1:]
    print("Job id: " + jobid)

    connection.close()

    #-------------------------------------
    #Check job status, wait until finished

    while True:
    	connection = httplib.HTTPConnection(host, port)
    	connection.request("GET",pathinfo+"/"+jobid)
    	response = connection.getresponse()
    	data = response.read()
    	#XML response: parse it to obtain the current status
    	dom = parseString(data)
    	
    	if use_mirror:
    	    phaseElement = dom.getElementsByTagName('phase')[0]
    	else:
    	    phaseElement = dom.getElementsByTagName('uws:phase')[0]
    	
    	phaseValueElement = phaseElement.firstChild
    	phase = phaseValueElement.toxml()
    	print("Status: " + phase)
    	#Check finished
    	if phase == 'COMPLETED': break
    	#wait and repeat
    	time.sleep(0.2)

    #print "Data:"
    #print data

    connection.close()

    #-------------------------------------
    #Get results
    connection = httplib.HTTPConnection(host, port)
    connection.request("GET",pathinfo+"/"+jobid+"/results/result")
    response = connection.getresponse()
    data = response.read()
    outputFileName = "gaia.fits" + (not use_mirror)*".gz"
    try:
        outputFile = open(outputFileName, "w")
        outputFile.write(data)
    except:
        # Python 3
        outputFile = open(outputFileName, "wb")
        outputFile.write(data)
        
    outputFile.close()
    connection.close()
    print("Data saved in: " + outputFileName)
    
    if not use_mirror:
        ## ESA archive returns gzipped
        try:
            os.remove('gaia.fits')
        except:
            pass
    
        os.system('gunzip gaia.fits.gz')
    
    table = Table.read('gaia.fits', format='fits')
    return table

def get_panstarrs_catalog(ra=0., dec=0., radius=3, columns='objName,objID,raStack,decStack,raStackErr,decStackErr,rMeanKronMag,rMeanKronMagErr,iMeanKronMag,iMeanKronMagErr', max_records=10000):
    """TBD
    """
    try:
        import httplib
        from urllib import urlencode
        from urllib import urlopen
    except:
        # python 3
        import http.client as httplib
        from urllib.parse import urlencode
        from urllib.request import urlopen
        
    query_url = "http://archive.stsci.edu/panstarrs/search.php?RA={ra}&DEC={dec}&radius={radius}&max_records={max_records}&outputformat=CSV&action=Search&coordformat=dec&selectedColumnsCsv={columns}&raStack%3E=0".format(ra=ra, dec=dec, radius=radius, max_records=int(max_records), columns=columns)
    
    print('Query PanSTARRS catalog ({ra},{dec})'.format(ra=ra, dec=dec))
    
    query = urlopen(query_url)
    lines = [bytes(columns+'\n', encoding='utf-8')]
    lines.extend(query.readlines()[2:])
    
    csv_file = '/tmp/ps1_{ra}_{dec}.csv'.format(ra=ra, dec=dec)
    fp = open(csv_file,'wb')
    fp.writelines(lines)
    fp.close()
    
    table = utils.GTable.read(csv_file)
    clip = (table['rMeanKronMag'] > 0) & (table['raStack'] > 0)
    table['ra'] = table['raStack']
    table['dec'] = table['decStack']
    return table[clip]
    
def get_radec_catalog(ra=0., dec=0., radius=3., product='cat', verbose=True, reference_catalogs = ['GAIA', 'PS1', 'SDSS', 'WISE'], **kwargs):
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
    
    reference_catalogs : list
        Order in which to query reference catalogs.  Options are 'GAIA',
        'PS1' (STScI PanSTARRS), 'SDSS', 'WISE'.
        
    Returns
    -------
    radec : str
        Filename of the RA/Dec list derived from the parent catalog
    
    ref_catalog : str, {'SDSS', 'WISE', 'VISIT'}
        Provenance of the `radec` list.
    
    """
    # try:
    #     sdss = get_sdss_catalog(ra=ra, dec=dec, radius=radius)
    # except:
    #     print('SDSS query failed')
    #     sdss = []
    # 
    # if sdss is None:
    #     sdss = []
        
    query_functions = {'SDSS':get_sdss_catalog, 
                       'GAIA':get_gaia_DR2_catalog,
                       'PS1':get_panstarrs_catalog,
                       'WISE':get_irsa_catalog}
      
    # if len(sdss) > 5:
    #     table_to_regions(sdss, output='{0}_sdss.reg'.format(product))
    #     sdss['ra','dec'].write('{0}_sdss.radec'.format(product), 
    #                             format='ascii.commented_header')
    #     radec = '{0}_sdss.radec'.format(product)
    #     ref_catalog = 'SDSS'
    #     has_catalog = True
    
    ### Try queries
    has_catalog = False
    ref_catalog = 'None'
    ref_cat = []
    
    for ref_src in reference_catalogs:
        try:
            ref_cat = query_functions[ref_src](ra=ra, dec=dec, radius=radius)
            if len(ref_cat) < 2:
                raise ValueError
                
            table_to_regions(ref_cat, output='{0}_{1}.reg'.format(product,
                                                         ref_src.lower()))
            ref_cat['ra','dec'].write('{0}_{1}.radec'.format(product, 
                                                         ref_src.lower()),
                                    format='ascii.commented_header',
                                    overwrite=True)

            radec = '{0}_{1}.radec'.format(product, ref_src.lower())
            ref_catalog = ref_src
            has_catalog = True
            if len(ref_cat) > 0:
                break
        except:
            print('{0} query failed'.format(ref_src))
            has_catalog = False
        
    if (ref_src == 'GAIA') & ('date' in kwargs) & has_catalog:
        
        gaia_tbl = utils.GTable.gread('gaia.fits')
        if 'date_format' in kwargs:
            date_format = kwargs['date_format']
        else:
            date_format = 'decimalyear'
        
        coo = get_gaia_radec_at_time(gaia_tbl, date=kwargs['date'],
                                     format=date_format)
        
        coo_tbl = utils.GTable()
        coo_tbl['ra'] = coo.ra
        coo_tbl['dec'] = coo.dec
        
        ok = np.isfinite(coo_tbl['ra']) & np.isfinite(coo_tbl['dec'])
        
        coo_tbl.meta['date'] = kwargs['date']
        coo_tbl.meta['datefmt'] = date_format
        
        print('Apply observation ({0},{1}) to GAIA catalog'.format(kwargs['date'], date_format))
        
        table_to_regions(coo_tbl[ok], output='{0}_{1}.reg'.format(product,
                                                     ref_src.lower()))
        
        coo_tbl['ra','dec'][ok].write('{0}_{1}.radec'.format(product, 
                                                     ref_src.lower()),
                                format='ascii.commented_header',
                                overwrite=True)
        
    
    if not has_catalog:
        return False
            
    #### WISP, check if a catalog already exists for a given rootname and use 
    #### that if so.
    cat_files = glob.glob('-f1'.join(product.split('-f1')[:-1]) + '-f*.cat*')
    if len(cat_files) > 0:
        ref_cat = utils.GTable.gread(cat_files[0])
        root = cat_files[0].split('.cat')[0]
        ref_cat['X_WORLD','Y_WORLD'].write('{0}.radec'.format(root),
                                format='ascii.commented_header',
                                overwrite=True)
        
        radec = '{0}.radec'.format(root)
        ref_catalog = 'VISIT'
    
    if verbose:
        print('{0} - Reference RADEC: {1} [{2}] N={3}'.format(product, radec, ref_catalog, len(ref_cat)))    
    
    return radec, ref_catalog
    
def process_direct_grism_visit(direct={}, grism={}, radec=None,
                               align_tolerance=5, align_clip=30,
                               align_mag_limits = [14,23],
                               column_average=True, 
                               sky_iter=10,
                               run_tweak_align=True,
                               tweak_fit_order=-1,
                               skip_direct=False,
                               fix_stars=True,
                               tweak_max_dist=1.,
                               tweak_threshold=1.5, 
                               drizzle_params = {},
                             reference_catalogs=['GAIA','PS1','SDSS','WISE']):
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
    isACS = '_flc' in direct['files'][0]
    isWFPC2 = '_c0m' in direct['files'][0]
    
    if not skip_direct:
        for file in direct['files']:
            crclean = isACS & (len(direct['files']) == 1)
            fresh_flt_file(file, crclean=crclean)
            updatewcs.updatewcs(file, verbose=False)
    
        ### Make ASN
        if not isWFPC2:
            asn = asnutil.ASNTable(inlist=direct['files'], output=direct['product'])
            asn.create()
            asn.write()
    
    ### Initial grism processing
    skip_grism = (grism == {}) | (grism is None) | (len(grism) == 0)
    if not skip_grism:
        for file in grism['files']:
            fresh_flt_file(file)
            
            # Need to force F814W filter for updatewcs
            if isACS:
                flc = pyfits.open(file, mode='update')
                if flc[0].header['INSTRUME'] == 'ACS':
                    changed_filter = True
                    flc[0].header['FILTER1'] = 'CLEAR1L'
                    flc[0].header['FILTER2'] = 'F814W'
                    flc.flush()
                    flc.close()
                else:
                    changed_filter = False
                    flc.close()
            else:
                changed_filter = False
                     
            # Run updatewcs 
            updatewcs.updatewcs(file, verbose=False)
            
            # Change back
            if changed_filter:
                flc = pyfits.open(file, mode='update')
                flc[0].header['FILTER1'] = 'CLEAR2L'
                flc[0].header['FILTER2'] = 'G800L'
                flc.flush()
                flc.close()
                
        ### Make ASN
        asn = asnutil.ASNTable(grism['files'], output=grism['product'])
        asn.create()
        asn.write()
            
    if isACS:
        bits = 64+32
        driz_cr_snr = '3.5 3.0'
        driz_cr_scale = '1.2 0.7'
    elif isWFPC2:
        bits = 64+32
        driz_cr_snr = '3.5 3.0'
        driz_cr_scale = '1.2 0.7'
    else:
        bits = 576
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
        
    if not skip_direct:
        if (not isACS) & (not isWFPC2) & run_tweak_align:
            #if run_tweak_align:
            tweak_align(direct_group=direct, grism_group=grism,
                        max_dist=tweak_max_dist, key=' ', drizzle=False,
                        threshold=tweak_threshold, fit_order=tweak_fit_order)
      
        ### Get reference astrometry from SDSS or WISE
        if radec is None:
            im = pyfits.open(direct['files'][0])
            radec, ref_catalog = get_radec_catalog(ra=im[0].header['RA_TARG'],
                            dec=im[0].header['DEC_TARG'], 
                            product=direct['product'],
                            reference_catalogs=reference_catalogs,
                            date=im[0].header['EXPSTART'],
                            date_format='mjd')
        
            if ref_catalog == 'VISIT':
                align_mag_limits = [16,23]
            elif ref_catalog == 'SDSS':
                align_mag_limits = [16,21]
            elif ref_catalog == 'PS1':
                align_mag_limits = [16,22]
            elif ref_catalog == 'WISE':
                align_mag_limits = [15,20]
        else:
            ref_catalog = 'USER'
    
        print('{0}: First Drizzle'.format(direct['product']))
    
        ### Clean up
        for ext in ['.fits', '.log']:
            file = '{0}_wcs.{1}'.format(direct['product'], ext)
            if os.path.exists(file):
                os.remove(file)
                
        ### First drizzle
        if len(direct['files']) > 1:
            AstroDrizzle(direct['files'], output=direct['product'],
                         clean=True, context=False, preserve=False,
                         skysub=True, driz_separate=True, driz_sep_wcs=True,
                         median=True, blot=True, driz_cr=True,
                         driz_cr_snr=driz_cr_snr, driz_cr_scale=driz_cr_scale,
                         driz_cr_corr=False, driz_combine=True,
                         final_bits=bits, coeffs=True, build=False, 
                         final_wht_type='IVM', **drizzle_params)
        else:
            AstroDrizzle(direct['files'], output=direct['product'], 
                         clean=True, final_scale=None, final_pixfrac=1,
                         context=False, final_bits=bits, preserve=False,
                         driz_separate=False, driz_sep_wcs=False,
                         median=False, blot=False, driz_cr=False,
                         driz_cr_corr=False, driz_combine=True,
                         build=False, final_wht_type='IVM', **drizzle_params)
        
        ## Now do tweak_align for ACS
        if (isACS | isWFPC2) & run_tweak_align:
            tweak_align(direct_group=direct, grism_group=grism,
                    max_dist=tweak_max_dist, key=' ', drizzle=False,
                    threshold=tweak_threshold)
            
            # Redrizzle with no CR rejection
            AstroDrizzle(direct['files'], output=direct['product'],
                             clean=True, context=False, preserve=False,
                             skysub=False, driz_separate=False,
                             driz_sep_wcs=False,
                             median=False, blot=False, driz_cr=False,
                             driz_cr_corr=False, driz_combine=True,
                             final_bits=bits, coeffs=True, build=False, 
                             final_wht_type='IVM', resetbits=0)
            
        ### Make catalog & segmentation image
        if isWFPC2:
            thresh = 8
        else:
            thresh = 2
        
        #cat = make_drz_catalog(root=direct['product'], threshold=thresh)
        cat = make_SEP_catalog(root=direct['product'], threshold=thresh)
        
        if radec == 'self':
            okmag = ((cat['MAG_AUTO'] > align_mag_limits[0]) & 
                    (cat['MAG_AUTO'] < align_mag_limits[1]))
                    
            cat['X_WORLD', 'Y_WORLD'][okmag].write('self',
                                        format='ascii.commented_header',
                                        overwrite=True)
        
        #clip=30
        logfile = '{0}_wcs.log'.format(direct['product'])
        if os.path.exists(logfile):
            os.remove(logfile)
        
        guess_file = '{0}.align_guess'.format(direct['product'])
        if os.path.exists(guess_file):
            guess = np.loadtxt(guess_file)
        else:
            guess = [0., 0., 0., 1]
            
        try:
            result = align_drizzled_image(root=direct['product'], 
                                      mag_limits=align_mag_limits,
                                      radec=radec, NITER=5, clip=align_clip,
                                      log=True, guess=guess,
                                      outlier_threshold=align_tolerance)
        except:
            fp = open('{0}.wcs_failed'.format(direct['product']),'w')
            fp.write(guess.__str__())
            fp.close()
            
            result = align_drizzled_image(root=direct['product'], 
                                      mag_limits=align_mag_limits,
                                      radec=radec, NITER=0, clip=align_clip,
                                      log=False, guess=guess,
                                      outlier_threshold=align_tolerance)
                                       
        orig_wcs, drz_wcs, out_shift, out_rot, out_scale = result
        
        ### Update direct FLT WCS
        for file in direct['files']:
            updatehdr.updatewcs_with_shift(file, 
                                str('{0}_wcs.fits'.format(direct['product'])),
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
            AstroDrizzle(direct['files'], output=direct['product'],
                         clean=True, final_pixfrac=0.8, context=False,
                         resetbits=4096, final_bits=bits, driz_sep_bits=bits,
                         preserve=False, driz_cr_snr=driz_cr_snr,
                         driz_cr_scale=driz_cr_scale, driz_separate=False,
                         driz_sep_wcs=False, median=False, blot=False,
                         driz_cr=False, driz_cr_corr=False,
                         build=False, final_wht_type='IVM', **drizzle_params)
        else:
            if 'par' in direct['product']:
                pixfrac=1.0
            else:
                pixfrac=0.8
        
            AstroDrizzle(direct['files'], output=direct['product'], 
                         clean=True, final_pixfrac=pixfrac, context=isACS,
                         resetbits=4096, final_bits=bits, driz_sep_bits=bits,
                         preserve=False, driz_cr_snr=driz_cr_snr,
                         driz_cr_scale=driz_cr_scale, build=False, 
                         final_wht_type='IVM', **drizzle_params)
    
        ### Make DRZ catalog again with updated DRZWCS
        clean_drizzle(direct['product'])
        
        if isWFPC2:
            thresh = 8
        else:
            thresh = 1.6
        
        #cat = make_drz_catalog(root=direct['product'], threshold=thresh)
        cat = make_SEP_catalog(root=direct['product'], threshold=thresh)
        
        table_to_regions(cat, '{0}.cat.reg'.format(direct['product']))
        table_to_radec(cat, '{0}.cat.radec'.format(direct['product']))
        
        if (fix_stars) & (not isACS) & (not isWFPC2):
            fix_star_centers(root=direct['product'], drizzle=True, mag_lim=21)
        
    ################# 
    ##########  Grism image processing
    #################
    
    if skip_grism:       
        return True
        
    ### Match grism WCS to the direct images
    match_direct_grism_wcs(direct=direct, grism=grism, get_fresh_flt=False)
    
    ### First drizzle to flag CRs
    gris_cr_corr = len(grism['files']) > 1
    
    AstroDrizzle(grism['files'], output=grism['product'], clean=True,
                 context=False, preserve=False, skysub=True,
                 driz_separate=gris_cr_corr, driz_sep_wcs=gris_cr_corr, median=gris_cr_corr, 
                 blot=gris_cr_corr, driz_cr=gris_cr_corr, driz_cr_corr=gris_cr_corr, 
                 driz_cr_snr=driz_cr_snr, driz_cr_scale=driz_cr_scale, 
                 driz_combine=True, final_bits=bits, coeffs=True, 
                 resetbits=4096, build=False, final_wht_type='IVM')        
        
    ### Subtract grism sky
    status = visit_grism_sky(grism=grism, apply=True, sky_iter=sky_iter,
                          column_average=column_average, verbose=True, ext=1)
    
    # Run on second chip (also for UVIS/G280)
    if isACS:
        visit_grism_sky(grism=grism, apply=True, sky_iter=sky_iter,
                        column_average=column_average, verbose=True, ext=2)
        
        # Add back in some pedestal or CR rejection fails for ACS
        for file in grism['files']:
            flt = pyfits.open(file, mode='update')
            h = flt[0].header
            flat_sky = h['GSKY101']*h['EXPTIME']
            
            # Use same pedestal for both chips for skysub
            for ext in [1,2]:
                flt['SCI',ext].data += flat_sky
            
            flt.flush()
            
            
    ### Redrizzle with new background subtraction
    if isACS:
        skyfile=''
    else:
        skyfile = '/tmp/{0}.skyfile'.format(grism['product'])
        fp = open(skyfile,'w')
        fp.writelines(['{0} 0.0\n'.format(f) for f in grism['files']])
        fp.close()
    
    if 'par' in grism['product']:
        pixfrac=1.0
    else:
        pixfrac=0.8
            
    AstroDrizzle(grism['files'], output=grism['product'], clean=True,
                 context=isACS, preserve=False, skysub=True, skyfile=skyfile,
                 driz_separate=gris_cr_corr, driz_sep_wcs=gris_cr_corr, median=gris_cr_corr, 
                 blot=gris_cr_corr, driz_cr=gris_cr_corr, driz_cr_corr=gris_cr_corr, 
                 driz_cr_snr=driz_cr_snr, driz_cr_scale=driz_cr_scale, 
                 driz_combine=True, driz_sep_bits=bits, final_bits=bits,
                 coeffs=True, resetbits=4096, final_pixfrac=pixfrac, 
                 build=False, final_wht_type='IVM')        
    
    clean_drizzle(grism['product'])
    
    ### Add direct filter to grism FLT headers
    set_grism_dfilter(direct, grism)
    
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
    direct_filter = utils.get_hst_filter(d_im[0].header)
    for file in grism['files']:
        if '_flc' in file:
            ext = [1,2]
        else:
            ext = [1]
            
        print('DFILTER: {0} {1}'.format(file, direct_filter))
        flt = pyfits.open(file, mode='update')
        for e in ext:
            flt['SCI',e].header['DFILTER'] = (direct_filter, 
                                              'Direct imaging filter')
        flt.flush()
    
def tweak_align(direct_group={}, grism_group={}, max_dist=1., key=' ', 
                threshold=3, drizzle=False, fit_order=-1):
    """
    Intra-visit shifts (WFC3/IR)
    """
    from drizzlepac.astrodrizzle import AstroDrizzle
    from scipy import polyfit, polyval
    
    if len(direct_group['files']) < 2:
        print('Only one direct image found, can\'t compute shifts!')
        return True
        
    wcs_ref, shift_dict = tweak_flt(files=direct_group['files'],
                                    max_dist=max_dist, threshold=threshold,
                                    verbose=True)

    grism_matches = find_direct_grism_pairs(direct=direct_group, grism=grism_group, check_pixel=[507, 507], toler=0.1, key=key)
    
    fp = open('{0}_shifts.log'.format(direct_group['product']), 'w')
    fp.write('# flt xshift yshift rot scale N rmsx rmsy\n')
    fp.write('# fit_order: {0}\n'.format(fit_order))
    
    for k in grism_matches:
        d = shift_dict[k]
        fp.write('# match[\'{0}\'] = {1}\n'.format(k, grism_matches[k]))
    
    for k in shift_dict:
        d = shift_dict[k]
        fp.write('{0:s} {1:7.3f} {2:7.3f} {3:8.5f} {4:8.5f} {5:5d} {6:6.3f} {7:6.3f}\n'.format(k, d[0], d[1], d[2], d[3], d[4], d[5][0], d[5][1]))
    
    fp.close()
    
    # Fit a polynomial, e.g., for DASH
    if fit_order > 0:
        print('Fit polynomial order={0} to shifts.'.format(fit_order))
        
        shifts = np.array([shift_dict[k][:2] for k in sorted(shift_dict)])
        t = np.arange(shifts.shape[0])
        cx = polyfit(t, shifts[:,0], fit_order)
        sx = polyval(cx, t)
        cy = polyfit(t, shifts[:,1], fit_order)
        sy = polyval(cy, t)
        fit_shift = np.array([sx, sy]).T
        
        for ik, k in enumerate(sorted(shift_dict)):
            shift_dict[k][:2] = fit_shift[ik,:]
    
    ## Apply the shifts to the header WCS
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
                 driz_cr_scale=driz_cr_scale, build=False, 
                 final_wht_type='IVM')
    
    clean_drizzle(direct_group['product'])
    #cat = make_drz_catalog(root=direct_group['product'], threshold=1.6)
    cat = make_SEP_catalog(root=direct_group['product'], threshold=1.6)
    table_to_regions(cat, '{0}.cat.reg'.format(direct_group['product']))
    
    if (grism_group == {}) | (grism_group is None):
        return True
        
    # Grism  
    skyfile = '/tmp/{0}.skyfile'.format(grism_group['product'])
    fp = open(skyfile,'w')
    fp.writelines(['{0} 0.0\n'.format(f) for f in grism_group['files']])
    fp.close()
      
    AstroDrizzle(grism_group['files'], output=grism_group['product'],
                 clean=True, context=False, preserve=False, skysub=True,
                 skyfile=skyfile, driz_separate=True, driz_sep_wcs=True,
                 median=True, blot=True, driz_cr=True, driz_cr_corr=True,
                 driz_combine=True, driz_sep_bits=bits, final_bits=bits,
                 coeffs=True, resetbits=4096, final_pixfrac=pixfrac, 
                 build=False, final_wht_type='IVM')
    
    clean_drizzle(grism_group['product'])
    
    return True
    
def clean_drizzle(root, context=False):
    """Zero-out WHT=0 pixels in drizzle mosaics
    
    Parameters
    ----------
    root : str
        Rootname of the mosaics.  I.e., `{root}_drz_sci.fits`.
    
    Returns
    -------
    Nothing, science mosaic modified in-place
    """
    drz_file = glob.glob('{0}_dr[zc]_sci.fits'.format(root))[0]
    
    sci = pyfits.open(drz_file, mode='update')
    wht = pyfits.open(drz_file.replace('_sci.fits', '_wht.fits'))
    mask = wht[0].data == 0
    
    # Mask where context shows that mosaic comes from a single input
    ctx_file = drz_file.replace('_sci.','_ctx.')
    if context & os.path.exists(ctx_file):
        ctx = pyfits.open(ctx_file)
        
        bits = np.log(ctx[0].data)/np.log(2)
        # bits = round(bits) when is a power of 2
        mask &= bits != np.round(bits) 
        
    sci[0].data[mask] = 0
    sci.flush()

def tweak_flt(files=[], max_dist=0.4, threshold=3, verbose=True, use_sewpy=False):
    """TBD
    
    Refine shifts of FLT files
    """
    import scipy.spatial
    
    try:
        # https://github.com/megalut/sewpy
        import sewpy
    except:
        use_sewpy = False
        
    ### Make FLT catalogs
    cats = []
    for i, file in enumerate(files):
        root = file.split('.fits')[0]
        
        im = pyfits.open(file)
        ok = im['DQ',1].data == 0
        sci = im['SCI',1].data*ok - np.median(im['SCI',1].data[ok])
        
        header = im['SCI',1].header.copy()
        for k in ['PHOTFNU', 'PHOTFLAM', 'PHOTPLAM', 'FILTER']:
            header[k] = im[0].header[k]
            
        pyfits.writeto('{0}_xsci.fits'.format(root), data=sci,
                       header=header,
                       clobber=True)
        
        pyfits.writeto('{0}_xrms.fits'.format(root), data=im['ERR',1].data,
                       header=im['ERR',1].header, clobber=True)
        
        if use_sewpy:
            params = ["X_IMAGE", "Y_IMAGE", "X_WORLD", "Y_WORLD",
                                    "FLUX_RADIUS(3)", "FLAGS"]
            sew = sewpy.SEW(params=params,
                            config={"DETECT_THRESH":threshold,
                                    "DETECT_MINAREA":8,
                                    "PHOT_FLUXFRAC":"0.3, 0.5, 0.8",
                                    "WEIGHT_TYPE":"MAP_RMS",
                                "WEIGHT_IMAGE":"{0}_xrms.fits".format(root)})
        
            output = sew('{0}_xsci.fits'.format(root))        
            cat = output['table']
        else:
            # SEP
            wht = 1/im['ERR',1].data**2
            wht[~np.isfinite(wht)] = 0
            pyfits.writeto('{0}_xwht.fits'.format(root), data=wht,
                           header=im['ERR',1].header, clobber=True)
        
            cat = make_SEP_catalog(root=root, 
                                   sci='{0}_xsci.fits'.format(root),
                                   wht='{0}_xwht.fits'.format(root),
                                   threshold=threshold, minarea=8, 
                                   get_background=True, verbose=False)
        
        ######
        if '_flc' in file:
            wcs = pywcs.WCS(im['SCI',1].header, fobj=im, relax=True)
        else:
            wcs = pywcs.WCS(im['SCI',1].header, relax=True)
            
        cats.append([cat, wcs])
        
        for ext in ['_xsci', '_xrms', '_xwht', '_bkg', '_seg', '.cat']:
            file='{0}{1}.fits'.format(root, ext)
            if os.path.exists(file):
                os.remove(file)
            
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
                print(files[i], '! no match')
            
            continue
            
        dr = xy - xy_0[ix,:] 
        dx = np.median(dr[ok,:], axis=0)
        rms = np.std(dr[ok,:], axis=0)/np.sqrt(ok.sum())

        d[files[i]] = [dx[0], dx[1], 0.0, 1.0, ok.sum(), rms]
        
        if verbose:
            print(files[i], dx, rms, 'N={0:d}'.format(ok.sum()))
    
    wcs_ref = cats[0][1]
    return wcs_ref, d

def apply_tweak_shifts(wcs_ref, shift_dict, grism_matches={}, verbose=True):
    """
    
    """
    from drizzlepac import updatehdr

    hdu = wcs_ref.to_fits(relax=True)
    file0 = list(shift_dict.keys())[0].split('.fits')[0]
    tweak_file = '{0}_tweak_wcs.fits'.format(file0)
    hdu.writeto(tweak_file, clobber=True)
    for file in shift_dict:
        updatehdr.updatewcs_with_shift(file, tweak_file,
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
                updatehdr.updatewcs_with_shift(grism_file, tweak_file,
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
    
    os.remove(tweak_file)
    
def find_direct_grism_pairs(direct={}, grism={}, check_pixel=[507, 507],
                            toler=0.1, key='A', same_visit=True):
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
        #direct_wcs[file] = pywcs.WCS(im[1].header, relax=True, key=key)
        #full_direct_wcs[file] = pywcs.WCS(im[1].header, relax=True)
        
        if '_flc' in file:
            direct_wcs[file] = pywcs.WCS(im[1].header, fobj=im, relax=True,
                                         key=key)
            full_direct_wcs[file] = pywcs.WCS(im[1].header, fobj=im,
                                              relax=True)
        else:
            direct_wcs[file] = pywcs.WCS(im[1].header, relax=True, key=key)
            full_direct_wcs[file] = pywcs.WCS(im[1].header, relax=True)
        
        direct_rd[file] = direct_wcs[file].all_pix2world([check_pixel], 1)
    
    if 'files' not in grism:
        return grism_matches
         
    for file in grism['files']:
        im = pyfits.open(file)
        if '_flc' in file:
            grism_wcs[file] = pywcs.WCS(im[1].header, relax=True, key=key, 
                                        fobj=im)
        else:
            grism_wcs[file] = pywcs.WCS(im[1].header, relax=True, key=key)
        
        #print file
        delta_min = 10
        for d in direct['files']:
            if (os.path.basename(d)[:6] != os.path.basename(file)[:6]) & same_visit:
                continue
                
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
                           run_drizzle=True, xyscale=None):
    """Match WCS of grism exposures to corresponding direct images
    
    TBD
    """
    from drizzlepac import updatehdr
    from stwcs import updatewcs
    from drizzlepac.astrodrizzle import AstroDrizzle
    
    wcs_log = Table.read('{0}_wcs.log'.format(direct['product']),
                         format='ascii.commented_header')
                         
    wcs_hdu = pyfits.open('{0}_wcs.fits'.format(direct['product']))
    
    if get_fresh_flt:
        for file in grism['files']:
            fresh_flt_file(file)
            updatewcs.updatewcs(file, verbose=False)
        
    direct_flt = pyfits.open(direct['files'][0])
    ref_catalog = direct_flt['SCI',1].header['WCSNAME']
    
    #### User-defined shifts
    if xyscale is not None:
        # Use user-defined shifts
        xsh, ysh, rot, scale = xyscale
        
        tmp_wcs = '/tmp/{0}_tmpwcs.fits'.format(str(direct['product']))
        ext = len(wcs_hdu)-1
        wcs_hdu[ext].writeto(tmp_wcs, clobber=True)
        
        for file in grism['files']:
            updatehdr.updatewcs_with_shift(file, tmp_wcs,
                                      xsh=xsh,
                                      ysh=ysh,
                                      rot=rot, scale=scale,
                                      wcsname=ref_catalog, force=True,
                                      reusename=True, verbose=True,
                                      sciext='SCI')
            
            ### Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
            ### keywords
            im = pyfits.open(file, mode='update')
            im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
            im.flush()
        
        return True
        
    #### Get from WCS log file
    for ext in wcs_log['ext']:
        tmp_wcs = '/tmp/{0}_tmpwcs.fits'.format(str(direct['product']))
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
            
def align_multiple_drizzled(mag_limits=[16,23]):
    """TBD
    """
    from stwcs import updatewcs
    from drizzlepac import updatehdr
    from drizzlepac.astrodrizzle import AstroDrizzle
    
    drz_files = ['j0800+4029-080.0-f140w_drz_sci.fits', 
                 'j0800+4029-117.0-f140w_drz_sci.fits']
    
    for drz_file in drz_files:
        #cat = make_drz_catalog(root=drz_file.split('_drz')[0], threshold=2)
        cat = make_SEP_catalog(root=drz_file.split('_drz')[0], threshold=2)
        
    cref = utils.GTable.gread(drz_files[0].replace('_drz_sci.fits', '.cat.fits'))
    
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
          files.append(im[0].header['D{0:03d}DATA'.format(i+1)].split('[')[0])
        
        
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
            AstroDrizzle(files, output=root, clean=True, context=False, preserve=False, skysub=True, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True, final_bits=576, coeffs=True, resetbits=0, build=False, final_wht_type='IVM')        
        else:
            AstroDrizzle(files, output=root, clean=True, final_scale=None, final_pixfrac=1, context=False, final_bits=576, preserve=False, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True, build=False, final_wht_type='IVM') 

        #cat = make_drz_catalog(root=root, threshold=2)
        cat = make_SEP_catalog(root=root, threshold=2)
        
    if False:
        files0 = ['icou09fvq_flt.fits', 'icou09fyq_flt.fits', 'icou09gpq_flt.fits',
               'icou09h3q_flt.fits']

        files1 = ['icou10emq_flt.fits', 'icou10eqq_flt.fits', 'icou10euq_flt.fits',
               'icou10frq_flt.fits']
        
        all_files = list(np.append(files0, files1))
        AstroDrizzle(all_files, output='total', clean=True, context=False, preserve=False, skysub=True, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True, final_bits=576, coeffs=True, resetbits=0, final_rot=0, build=False, final_wht_type='IVM')    
        
        AstroDrizzle(files0, output='group0', clean=True, context=False, preserve=False, skysub=True, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True, final_bits=576, coeffs=True, resetbits=0, final_refimage='total_drz_sci.fits', build=False, final_wht_type='IVM')    
        AstroDrizzle(files1, output='group1', clean=True, context=False, preserve=False, skysub=True, driz_separate=False, driz_sep_wcs=False, median=False, blot=False, driz_cr=False, driz_cr_corr=False, driz_combine=True, final_bits=576, coeffs=True, resetbits=0, final_refimage='total_drz_sci.fits', build=False, final_wht_type='IVM')    
        
        im0 = pyfits.open('group0_drz_sci.fits')
        im1 = pyfits.open('group1_drz_sci.fits')
        imt = pyfits.open('total_drz_sci.fits')

        
def visit_grism_sky(grism={}, apply=True, column_average=True, verbose=True, ext=1, sky_iter=10):
    """Subtract sky background from grism exposures
    
    Implementation of grism sky subtraction from ISR 2015-17    
    
    TBD
    
    """
    import numpy.ma
    import scipy.ndimage as nd
    
    #from sklearn.gaussian_process import GaussianProcess
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    
    ### Figure out which grism 
    im = pyfits.open(grism['files'][0])
    grism_element = utils.get_hst_filter(im[0].header)
    
    flat = 1.
    if grism_element == 'G141':
        bg_fixed = ['zodi_G141_clean.fits']
        bg_vary = ['zodi_G141_clean.fits', 'excess_lo_G141_clean.fits',
                   'G141_scattered_light.fits'][1:]
        isACS = False
    elif grism_element == 'G102':
        bg_fixed = ['zodi_G102_clean.fits']
        bg_vary = ['excess_G102_clean.fits']
        isACS = False
    
    elif grism_element == 'G280':
        bg_fixed = ['UVIS.G280.flat.fits']
        bg_vary = ['UVIS.G280.ext{0:d}.sky.fits'.format(ext)]
        isACS = True
        flat = 1.
        
    elif grism_element == 'G800L':
        bg_fixed = ['ACS.WFC.CHIP{0:d}.msky.1.smooth.fits'.format({1:2,2:1}[ext])]
        bg_vary = ['ACS.WFC.flat.fits']
        #bg_fixed = ['ACS.WFC.CHIP%d.msky.1.fits' %({1:2,2:1}[ext])]
        #bg_fixed = []
        isACS = True
        
        flat_files = {'G800L':'n6u12592j_pfl.fits'} # F814W
        flat_file = flat_files[grism_element]        
        flat_im = pyfits.open(os.path.join(os.getenv('jref'), flat_file))
        flat = flat_im['SCI',ext].data.flatten()
    
    if verbose:
        print('{0}: EXTVER={1:d} / {2} / {3}'.format(grism['product'], ext, bg_fixed, bg_vary))
    if not isACS:
        ext = 1
        
    ### Read sky files    
    data_fixed = []
    for file in bg_fixed:
        im = pyfits.open('{0}/CONF/{1}'.format(os.getenv('GRIZLI'), file))
        sh = im[0].data.shape
        data = im[0].data.flatten()/flat
        data_fixed.append(data)
        
    data_vary = []
    for file in bg_vary:
        im = pyfits.open('{0}/CONF/{1}'.format(os.getenv('GRIZLI'), file))
        data_vary.append(im[0].data.flatten()*1)
        sh = im[0].data.shape
        
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
    if isACS:
        bits = 64+32
    else:
        bits = 576
    
    for i in range(Nexp):
        flt = pyfits.open(grism['files'][i])
        dq = utils.unset_dq_bits(flt['DQ',ext].data, okbits=bits)
        dq_mask = dq == 0
        
        ## Data
        data[i*Npix:(i+1)*Npix] = (flt['SCI',ext].data*dq_mask).flatten()
        mask[i*Npix:(i+1)*Npix] &= dq_mask.flatten() #== 0
        wht[i*Npix:(i+1)*Npix] = 1./(flt['ERR',ext].data**2*dq_mask).flatten()
        wht[~np.isfinite(wht)] = 0.
        
        if isACS:
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
    
    for iter in range(sky_iter):
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
            print('   {0} > Iter: {1:d}, masked: {2:d}, {3}'.format(grism['product'], iter+1, obj_mask.sum(), coeffs))
                                                
        out = np.linalg.lstsq(A[mask & obj_mask,:], data[mask & obj_mask])
        coeffs = out[0]
            
    ### Best-fit sky
    sky = np.dot(A, coeffs).reshape(Nexp, Npix)
        
    ## log file
    fp = open('{0}_{1}_sky_background.info'.format(grism['product'],ext), 'w')
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
            flt['SCI',ext].data -= sky[j,:].reshape(sh)*exptime[j]
                
            header = flt[0].header
            header['GSKYCOL{0:d}'.format(ext)] = (False, 'Subtract column average')
            header['GSKYN{0:d}'.format(ext)] = (Nfix+Nvary, 'Number of sky images')
            header['GSKY{0:d}01'.format(ext)] = (coeffs[0], 
                                'Sky image {0} (fixed)'.format(bg_fixed[0]))
            
            header['GSKY{0:d}01F'.format(ext)] = (bg_fixed[0], 'Sky image (fixed)')
            
                
            for v in range(Nvary):
                k = Nfix + j*Nvary + v
                #print coeffs[k]
                header['GSKY{0}{1:02d}'.format(ext, v+Nfix+1)] = (coeffs[k], 
                                'Sky image {0} (variable)'.format(bg_vary[v]))
                
                header['GSKY{0}{1:02d}F'.format(ext, v+Nfix+1)] = (bg_vary[v], 
                                                      'Sky image (variable)')
                
            flt.flush()
    
    ### Don't do `column_average` for ACS
    if (not column_average) | isACS:
        return isACS
        
    ######
    ### Now fit residual column average & make diagnostic plot
    interactive_status=plt.rcParams['interactive']
    plt.ioff()
    
    fig = plt.figure(figsize=[6.,6.])
    ax = fig.add_subplot(111)
    
    im_shape = (1014,1014)
    
    for j in range(Nexp):
        
        file = grism['files'][j]
        
        resid = (data[j*Npix:(j+1)*Npix] - sky[j,:]).reshape(im_shape)
        m = (mask & obj_mask)[j*Npix:(j+1)*Npix].reshape(im_shape)
        
        ## Statistics of masked arrays    
        ma = np.ma.masked_array(resid, mask=(~m))
        med = np.ma.median(ma, axis=0)
    
        bg_sky = 1
        yrms = np.ma.std(ma, axis=0)/np.sqrt(np.sum(m, axis=0))
        xmsk = np.arange(im_shape[0])
        yres = med
        yok = (~yrms.mask) & np.isfinite(yrms) & np.isfinite(xmsk) & np.isfinite(yres)
        
        if yok.sum() == 0:
            print('ERROR: No valid pixels found!')
            continue
            
        ### Fit column average with smoothed Gaussian Process model
        if False:
            #### xxx old GaussianProcess implementation
            gp = GaussianProcess(regr='constant', corr='squared_exponential',
                                 theta0=8, thetaL=5, thetaU=12,
                                 nugget=(yrms/bg_sky)[yok][::1]**2,
                                 random_start=10, verbose=True, normalize=True)
                             
            try:
                gp.fit(np.atleast_2d(xmsk[yok][::1]).T, yres[yok][::1]+bg_sky)
            except:
                print('GaussianProces failed!  Check that this exposure wasn\'t fried by variable backgrounds.')
                continue
            
            y_pred, MSE = gp.predict(np.atleast_2d(xmsk).T, eval_MSE=True)
            gp_sigma = np.sqrt(MSE)
        
        ## Updated sklearn
        nmad_y = utils.nmad(yres)
        
        gpscl = 100 # rough normalization
        k1 = 0.3**2 * RBF(length_scale=80)  # Background variations
        k2 = 1**2 * WhiteKernel(noise_level=(nmad_y*gpscl)**2) # noise
        gp_kernel = k1+k2#+outliers
        
        yok &= np.abs(yres-np.median(yres)) < 50*nmad_y

        gp = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)

        gp.fit(np.atleast_2d(xmsk[yok][::1]).T, (yres[yok][::1]+bg_sky)*gpscl)
        y_pred, gp_sigma = gp.predict(np.atleast_2d(xmsk).T, return_std=True)
        gp_sigma /= gpscl
        y_pred /= gpscl
        
        ## Plot Results
        pi = ax.plot(med[0:2], alpha=0.2)
        ax.plot(y_pred-1, color=pi[0].get_color())
        ax.fill_between(xmsk, y_pred-1-gp_sigma, y_pred-1+gp_sigma,
                        color=pi[0].get_color(), alpha=0.3,
                        label=grism['files'][j].split('_fl')[0])
        
        ## result
        fp = open(file.replace('_flt.fits', '_column.dat'), 'wb')
        fp.write(b'# column obs_resid ok resid uncertainty\n')
        np.savetxt(fp, np.array([xmsk, yres, yok*1, y_pred-bg_sky, gp_sigma]).T, fmt='%.5f')
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
    fig.savefig('{0}_column.png'.format(grism['product']))
    #fig.savefig('%s_column.pdf' %(grism['product']))
    plt.close()
    
    ## Clean up large arrays
    del(data); del(A); del(wht); del(mask); del(model)
    
    if interactive_status:
        plt.ion()
    
    return False
    
def fix_star_centers(root='macs1149.6+2223-rot-ca5-22-032.0-f105w',
                     mag_lim=22, verbose=True, drizzle=False):
    """Unset CR bit (4096) in the centers of bright objects
    
    TBD
    
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
        
    Returns
    -------
    Nothing, updates FLT files in place.

    """
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
        wcs.append(pywcs.WCS(flt[1], relax=True))
        images.append(flt)
        
    yp, xp = np.indices((1014,1014))
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
            xi, yi = wcs[i].all_world2pix([rd[0],], [rd[1],], 0) 
            r = np.sqrt((xp-xi[0])**2 + (yp-yi[0])**2)
            unset = (r <= 3) & ((images[i]['DQ'].data & 4096) > 0)
            nset.append(unset.sum())
            if nset[i] > 0:
                images[i]['DQ'].data[unset] -= 4096
            
            # Fill saturated with EPSF fit
            satpix = (r <= 5) & (((images[i]['DQ'].data & 256) > 0) | ((images[i]['DQ'].data & 2048) > 0))
            nsat.append(satpix.sum())
            
            if nsat[i] > 0:
                xpi = int(np.round(xi[0]))
                ypi = int(np.round(yi[0]))
            
                slx = slice(xpi-12, xpi+12)
                sly = slice(ypi-12, ypi+12)
                
                sci = images[i]['SCI'].data[sly, slx]
                dq = images[i]['DQ'].data[sly, slx]
                err = images[i]['ERR'].data[sly, slx]
                ivar = 1/err**2
                ivar[(~np.isfinite(ivar)) | (dq > 0)] = 0
                
                # Fit the EPSF model
                try:
                    psf, psf_params = EPSF.fit_ePSF(sci, ivar=ivar, center=None, tol=1.e-3, N=12, origin=(ypi-12, xpi-12), filter=images[0][0].header['FILTER'])
                except:
                    continue
                    
                mask = satpix[sly, slx]
                sci[mask] = psf[mask]
                dq[mask] -= (dq[mask] & 2048)
                dq[mask] -= (dq[mask] & 256)
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
            pixfrac=1.0
        else:
            pixfrac=0.8
        
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

def find_single_image_CRs(visit, simple_mask=False, with_ctx_mask=True):
    """Use LACosmic to find CRs in parts of an ACS mosaic where only one
    exposure was available
    
    Paramters
    ---------
    visit : dict
        List of visit information from `~grizli.utils.parse_flt_files`.
    
   simple_mask : bool
        If true, set 1024 CR bit for all parts of a given FLT where it does
        not overlap with any others in the visit.  If False, then run 
        LACosmic to flag CRs in this area but keep the pixels.
        
    Requires context (CTX) image `visit['product']+'_drc_ctx.fits`.   
    """
    from drizzlepac import astrodrizzle
    import lacosmicx
    
    ctx = pyfits.open(visit['product']+'_drc_ctx.fits')
    bits = np.log2(ctx[0].data)
    mask = ctx[0].data == 0
    single_image = np.cast[np.float32]((np.cast[int](bits) == bits) & (~mask))
    ctx_wcs = pywcs.WCS(ctx[0].header)
    ctx_wcs.pscale = utils.get_wcs_pscale(ctx_wcs)
    
    for file in visit['files']:
        flt = pyfits.open(file, mode='update')
        for ext in [1,2]:
            
            flt_wcs = pywcs.WCS(flt['SCI',ext].header, fobj=flt, relax=True)
            flt_wcs.pscale = utils.get_wcs_pscale(flt_wcs)
            
            blotted = astrodrizzle.ablot.do_blot(single_image, ctx_wcs,
                            flt_wcs, 1, coeffs=True, interp='nearest',
                            sinscl=1.0, stepsize=10, wcsmap=None)
            
            ctx_mask = blotted > 0
            
            sci = flt['SCI',ext].data
            dq = flt['DQ',ext].data

            if simple_mask:
                print('{0}: Mask image without overlaps, extension {1:d}'.format(file, ext))
                dq[ctx_mask] |= 1024
            else:
                print('{0}: Clean CRs with LACosmic, extension {1:d}'.format(file, ext))

                if with_ctx_mask:
                    inmask = blotted == 0
                else:
                    inmask = dq > 0
                    
                crmask, clean = lacosmicx.lacosmicx(sci, inmask=inmask,
                             sigclip=4.5, sigfrac=0.3, objlim=5.0, gain=1.0,
                             readnoise=6.5, satlevel=65536.0, pssl=0.0,
                             niter=4, sepmed=True, cleantype='meanmask',
                             fsmode='median', psfmodel='gauss',
                             psffwhm=2.5,psfsize=7, psfk=None, psfbeta=4.765,
                             verbose=False)
            
                if with_ctx_mask:
                    dq[crmask & ctx_mask] |= 1024
                else:
                    dq[crmask] |= 1024
                    
                #sci[crmask & ctx_mask] = 0
        
        flt.flush()
        
def drizzle_overlaps(exposure_groups, parse_visits=False, check_overlaps=True, max_files=999, pixfrac=0.8, scale=0.06, skysub=True, skymethod='localmin', skyuser='MDRIZSKY', bits=None, final_wcs=True, final_rot=0, final_outnx=None, final_outny=None, final_ra=None, final_dec=None, final_wht_type='EXP', final_wt_scl='exptime', context=False):
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
    from drizzlepac.astrodrizzle import AstroDrizzle
    from shapely.geometry import Polygon
    
    if parse_visits:
        exposure_groups = utils.parse_visit_overlaps(exposure_groups, buffer=15.)
    
    ## Drizzle can only handle 999 files at a time
    if check_overlaps:
        for group in exposure_groups:
            if 'reference' not in group:
                continue
            
            if 'footprints' in group:
                footprints = group['footprints']
            else:
                footprints = []
                files=group['files']
                for i in range(len(files)):
                    print(i, files[i])
                    im = pyfits.open(files[i])
                    wcs = pywcs.WCS(im[1])
                    footprints.append(Polygon(wcs.calc_footprint()))
            
            ref = pyfits.getheader(group['reference'])
            wcs = pywcs.WCS(ref)
            ref_fp = Polygon(wcs.calc_footprint())
            
            files = []
            out_fp = []
            
            for j in range(len(group['files'])):
                olap = ref_fp.intersection(footprints[j])
                if olap.area > 0:
                    files.append(group['files'][j])
                    out_fp.append(footprints[j])
                    
            print(group['product'], len(files), len(group['files']))
            group['files'] = files
            group['footprints'] = out_fp
            
    if max_files > 0:
        all_groups = []
        for group in exposure_groups:
            N = len(group['files']) // int(max_files) +1
            if N == 1:
                all_groups.append(group)
            else:
                for k in range(N):
                    sli = slice(k*999,(k+1)*999)
                    files_list = group['files'][sli]
                    root='{0}-{1:03d}'.format(group['product'], k)
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
        if bits is None:
            if isACS:
                bits = 64+32
            else:
                bits = 576
        
        if 'reference' in group:
            AstroDrizzle(group['files'], output=group['product'],
                     clean=True, context=context, preserve=False,
                     skysub=skysub, skyuser=skyuser, skymethod=skymethod,
                     driz_separate=False, driz_sep_wcs=False,
                     median=False, blot=False, driz_cr=False,
                     driz_cr_corr=False, driz_combine=True,
                     final_bits=bits, coeffs=True, build=False, 
                     final_wht_type=final_wht_type,
                     final_wt_scl=final_wt_scl,
                     final_pixfrac=pixfrac,
                     final_wcs=True, final_refimage=group['reference'],
                     resetbits=0)
        else:
            AstroDrizzle(group['files'], output=group['product'],
                     clean=True, context=context, preserve=False,
                     skysub=skysub, skyuser=skyuser, skymethod=skymethod,
                     driz_separate=False, driz_sep_wcs=False,
                     median=False, blot=False, driz_cr=False,
                     driz_cr_corr=False, driz_combine=True,
                     final_bits=bits, coeffs=True, build=False, 
                     final_wht_type=final_wht_type,
                     final_wt_scl=final_wt_scl,
                     final_pixfrac=pixfrac,
                     final_wcs=final_wcs, final_rot=final_rot,
                     final_scale=scale, 
                     final_ra=final_ra, final_dec=final_dec,
                     final_outnx=final_outnx, final_outny=final_outny,
                     resetbits=0)
        
        clean_drizzle(group['product'])

def manual_alignment(visit, ds9, reference=None, reference_catalogs=['SDSS', 'PS1', 'GAIA', 'WISE']):
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
    
    im = pyfits.open(os.path.join(os.getcwd(), '../RAW/', visit['files'][0]))
    ra, dec = im[1].header['CRVAL1'], im[1].header['CRVAL2']
    
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
    ds9.set('file {0}'.format(im.filename()))
    ds9.set('regions file '+reference)
    x = input('pan to object in image: ')
    if x:
        print('Input detected ({0}).  Abort.'.format(x))
        return False

    x0 = np.cast[float](ds9.get('pan image').split())
    x = input('pan to object in region: ')
    x1 = np.cast[float](ds9.get('pan image').split())
    
    print ('Saved {0}.align_guess'.format(visit['product']))
    
    np.savetxt('{0}.align_guess'.format(visit['product']), [[x0[0]-x1[0], x0[1]-x1[1], 0, 1].__repr__()[1:-1].replace(',', '')], fmt='%s')
        