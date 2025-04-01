import os
import inspect
import warnings

import scipy.ndimage as nd
import numpy as np

import astropy.io.fits as pyfits
import astropy.time
import astropy.units as u

from jwst import datamodels
from jwst.ramp_fitting.ramp_fit_step import get_reference_file_subarrays
from jwst import __version__ as CAL_VER

try:
    import sep
    HAS_SEP = True
except ImportError:
    HAS_SEP = False
    
import mastquery.utils
from grizli import utils

DET1PIPE = None

DQ_ANY = 1
DQ_NAN = 8
DQ_NJUMP = 16
DQ_HOT = 32
DQ_RESID = 64
DQ_ERR = 128
DQ_FIRST_SATURATED = 256

BKG_KWARGS = dict(bw=128, bh=128, fw=3, fh=3)

SCALE_RNOISE = {'xxxNRSIRS2RAPID': 2.}

PIXELDQ_BITS = {'NRS1': [10], 
                'NRS2': [10],
                'NIS': [10,11,13,14],
                }
                
for d in [1,2,3,4,'LONG']:
    for m in 'AB':
        PIXELDQ_BITS[f'NRC{m}{d}'] = [10,11,13,14]

HOT_THRESHOLDS = {'NRS1': [1000,20000],
                  'NRS2': [1000,20000],
                   'NIS': [40000,40000]
                }
                
for d in ['LONG']:
    for m in 'AB':
        HOT_THRESHOLDS[f'NRC{m}{d}'] = [40000,40000]

for d in [1,2,3,4]:
    for m in 'AB':
        HOT_THRESHOLDS[f'NRC{m}{d}'] = [1000,20000]

# Global CRDS_CONTEXT

if 'CRDS_CONTEXT' in os.environ:
    CRDS_CONTEXT = os.getenv('CRDS_CONTEXT')
else:
    CRDS_CONTEXT = 'jwst_1069.pmap'
    # CRDS_CONTEXT = 'jwst_1123.pmap'


def get_pipeline_object():
    """
    Initialize `jwst.pipeline.Detector1Pipeline`
    """
    from jwst.pipeline import Detector1Pipeline

    global DET1PIPE
    
    if DET1PIPE is None:
        DET1PIPE = Detector1Pipeline()
    
    return DET1PIPE


def download_files(file):
    """
    Download uncal and rate files
    """

    if not os.path.exists(file):
        _ = mastquery.utils.download_from_mast([file])
    if not os.path.exists(file.replace('_uncal','_rate')):
        _ = mastquery.utils.download_from_mast([file.replace('_uncal','_rate')])
    

def process_level1_to_dark(file):
    """
    Run `jwst.pipeline.Detector1Pipeline` up through `dark_current`
    
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html
    
    """
    
    pipe = get_pipeline_object()
    
    download_files(file)
    
    if CRDS_CONTEXT:
        os.environ["CRDS_CONTEXT"] = CRDS_CONTEXT
        
    grp = pipe.group_scale.process(file)
    dqi = pipe.dq_init.process(grp)
    sat = pipe.saturation.process(dqi)
    bias = pipe.superbias.process(sat)
    refpix = pipe.refpix.process(bias)
    lin = pipe.linearity.process(refpix)
    pers = pipe.persistence.process(lin)
    dark = pipe.dark_current.process(pers)
    
    return dark


def process_uncal_level1(file='jw01208048001_03101_00001_nrs1_uncal.fits', output_extension='_xrate', jump_threshold=4, jump_ndilate=1, erode_snowballs=5, grow_snowballs=5, resid_thresh=4, hot_thresh='auto', hot_type='diff', max_njump=6, groups_for_rnoise=np.inf, flag_for_persistence=True, outlier_min_nints=3, integration_sigmas=[5,4,3], rescale_uncertainty=True, rescale_with_background=True, bkg_kwargs=BKG_KWARGS, dark=None, verbose=True, debug=None, **kwargs):
    """
    Custom ramp-fit scripts for JWST uncal images.

    All calculations are done with `numpy` arrays to be somewhat faster than 
    computing least-squares fits to individual pixel ramps.  One shortcut here
    is that only ramp sequences with two or fewer identified jumps are 
    considered for the ramp fit, i.e., the reads before the first jump and 
    after and including the last jump.
    
    Parameters
    ----------
    file : str
        JWST raw "uncal" filename
    
    output_extension : str
        The script puts the derived ramp and uncertainty into a ``rate``-like 
        file with filename ``file.replace('_uncal', output_extension)``.
        
    jump_threshold : float
        Jump identification threshold, sigma
    
    jump_ndilate : int
        Number of `scipy.ndimage.binary_dilation` iterations run on the initial
        jump mask
    
    erode_snowballs : int
        Number of `scipy.ndimage.binary_erosion` iterations run on the jump
        mask of each read before identifying snowballs.  This removes small
        clumps of pixels that probably aren't big snowballs
    
    grow_snowballs : float
        Once snowballs are identified as big clumps of jump pixels, compute
        the pixel area of each smowball clump and make a circular mask with
        radius ``grow_snowballs * np.sqrt(area/pi)``
    
    resid_thresh : float
        Compute a ramp model and flag pixels where
        ``(ramp - model) > resid_thresh*sigma``
    
    hot_thresh : [float, float]
        Thresholds for hot pixels.  The first number is the DN threshold
        in the first ramp read.  The second number is the threshold in the 
        first ramp *diff*.
    
    hot_type : ('diff', 'raw')
        If ``diff``, then first hot pixel threshold is on the first ramp 
        *difference*, otherwise threshold is on the raw values of the first
        read
    
    max_njump : int
        Unset jump mask where more than ``max_jump`` jumps are found in a ramp
    
    groups_for_rnoise : int
        Minimum `ngroups` for estimating read noise directy from the ramp fits
    
    flag_for_persistence : bool
        While stepping through integrations and ramps, make a mask for flagged
        saturated pixels and set the flag for *all* subsequent reads of that
        pixel, i.e., to catch saturated pixels that will likely cause 
        persistence in subsequent integrations.
    
    outlier_min_nints : int
        Run outlier rejection across integrations when at least
        ``outlier_min_nints`` integrations are available.
    
    integration_sigmas : list of float
        Iterative thresholds of integration outlier rejection
    
    rescale_uncertainty : bool
        Rescale final uncertainties based on statistics of the final rate
        image
    
    dark : `jwst.datamodels.ramp.RampModel`
        Precomputed ramp product.  If not provided, run `calwebb_detector1 <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html>`_ 
        up to the `dark_current <https://jwst-pipeline.readthedocs.io/en/latest/jwst/dark_current/index.html#dark-current-step>`_ step.
    
    verbose : bool
        Message verbosity
    
    Returns
    -------
    debug : dict
        Dictionary with a bunch of stuff computed by the script
        
    """
    
    frame = inspect.currentframe()
    _LOGFILE = utils.LOGFILE
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    if 'grow_snowball' in kwargs:
        grow_snowballs = kwargs['grow_snowball']
        
    utils.LOGFILE = file.split('.fits')[0] + '.log'
    utils.log_function_arguments(utils.LOGFILE, frame, 'process_uncal',
                                 ignore=['dark','debug','_LOGFILE',
                                         'CRDS_CONTEXT'],
                                 )
    
    pipe = get_pipeline_object()
    
    if debug is not None:
        if debug['file'] == file:
            print('debug: use dark from debug data')
            dark = debug['dark']
            
    if dark is None:
        # Detector1 through dark_current
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html
        # grp = pipe.group_scale.process(file)
        # dqi = pipe.dq_init.process(grp)
        # sat = pipe.saturation.process(dqi)
        # bias = pipe.superbias.process(sat)
        # refpix = pipe.refpix.process(bias)
        # lin = pipe.linearity.process(refpix)
        # pers = pipe.persistence.process(lin)
        # dark = pipe.dark_current.process(pers)
        
        dark = process_level1_to_dark(file)
    
    NINTS, NGROUPS, NYPIX, NXPIX = dark.data.shape
    
    ## readnoise, gain reference files
    readnoise_filename = pipe.get_reference_file(dark, 'readnoise')
    gain_filename = pipe.get_reference_file(dark, 'gain')
    # mask_filename = pipe.get_reference_file(dark, 'mask')
    
    msg =  f'\nprocess_uncal: {file}   readnoise file {readnoise_filename}'
    msg += f'\nprocess_uncal: {file}   gain file      {gain_filename}'
    utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
    
    ## From jwst.ramp_fitting.ramp_fit_step
    with datamodels.ReadnoiseModel(readnoise_filename) as readnoise_model, \
            datamodels.GainModel(gain_filename) as gain_model:

        # Try to retrieve the gain factor from the gain reference file.
        # If found, store it in the science model meta data, so that it's
        # available later in the gain_scale step, which avoids having to
        # load the gain ref file again in that step.
        if gain_model.meta.exposure.gain_factor is not None:
            gain_factor = gain_model.meta.exposure.gain_factor
            dark.meta.exposure.gain_factor = gain_factor

        # Get gain arrays, subarrays if desired.
        frames_per_group = dark.meta.exposure.nframes
        readnoise_2d, gain_2d = get_reference_file_subarrays(
            dark, readnoise_model, gain_model, frames_per_group)
            
    ### Now comes the good stuff
    
    if dark.meta.exposure.readpatt in SCALE_RNOISE:
        msg = f'process_uncal: {file} scale readnoise_2d for '
        msg += f'READPATT = {dark.meta.exposure.readpatt}: '
        msg += f'{SCALE_RNOISE[dark.meta.exposure.readpatt]:.2f}'
        
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        readnoise_2d *= SCALE_RNOISE[dark.meta.exposure.readpatt]
        
    ## Read times, correct??
     
    int1 = dark.group['integration_number'] == 1
    grp = dark.group[int1]
    
    t = astropy.time.Time(grp['helio_end_time'], format='mjd')
    
    try:
        #if len(t) == len(grp):
        #     print('\n debug !! len(t) == len(dark.group) !! (NIRSPEC?)')
        #     dt = t[1:] - (t[0] - dark.meta.exposure.frame_time*u.second)
        #     times = [dark.meta.exposure.frame_time]
        #     for i in range(dark.meta.exposure.ngroups-1):
        #         times.append(dt[i].to(u.second).value)
        # else:
        #     dt = t[1:] - (t[0] - dark.meta.exposure.frame_time*u.second)
        #     times = []
        #     for i in range(dark.meta.exposure.ngroups):
        #         times.append(dt[i].to(u.second).value)
        dt = t[1:] - (t[0]) # - dark.meta.exposure.frame_time*u.second)
        times = []
        for i in range(dark.meta.exposure.ngroups-1):
            times.append(dt[i].to(u.second).value)
        
        times.append(times[-1] + np.diff(times)[-1])
        
    except IndexError:
        #print('xxx', len(t), dark.meta.exposure.ngroups)
        
        print('!!! Times failed')
        return {'file':file, 'dark':dark}
        
    times = np.array(times)
    
    msg = f'process_uncal: {file}  group_time = {dark.meta.exposure.group_time:.1f}'
    msg += f'\nprocess_uncal: {file}  frame_time = {dark.meta.exposure.frame_time:.1f}'
    for i in range(NGROUPS):
        msg += f'\nprocess_uncal: {file}  sample[{i:>2}] = {times[i]:>6.1f}'

    msg += f'\nprocess_uncal: {file}  integration_time = {dark.meta.exposure.integration_time:6.1f}'
        
    utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
    
    # print('xxx', times, np.diff(times))
    
    # Copy dark dq array
    groupdq = dark.groupdq*1
        
    jump_saturated = np.zeros((NYPIX, NXPIX), dtype=groupdq.dtype)
    
    # Containers across integrations

    sci_list = []
    err_list = []
    dq_list = []
    slope_list = []
    slope_err_list = []
    nsamp_list = []
    rn_list = []
    jump_list = []
    
    for integ in range(NINTS):
        msg = f'\nprocess_uncal: {file} run for int {integ+1} / {NINTS}'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        
        ## Array copies
        dd = dark.data[integ,:,:,:] #/gain_2d
        #dd[groupdq[integ,:,:,:] > 0] = np.nan
        
        ## Diffs for jump indentification
        ds = np.diff(dd, axis=0)
        #ds[groupdq[integ,1:,:,:] > 0] = np.nan
        
        ## times and index arrays in full cube like the ramp
        t2d = np.zeros_like(dd)
        i2d = np.zeros((NGROUPS, NYPIX, NXPIX), dtype=int)
        for i, t in enumerate(times):
            t2d[i,:,:] = t
            i2d[i,:,:] = i
        
        ## global "background" from the first read
        bkg0 = np.nanmedian(dark.data[integ,0,:,:])

        ## Crude uncertainty arrays
        rn_det = readnoise_2d / np.sqrt(dark.meta.exposure.nframes)
        raw_var = ((rn_det)**2 + np.maximum(dd - bkg0, 0)/gain_2d)
        raw_err = np.sqrt(raw_var[:-1,:,:] + raw_var[1:,:,:])
        
        ## Identify jumps
        msg = f'process_uncal: {file} {integ+1} find jumps'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        
        jump = ds > (np.nanmedian(ds, axis=0)+jump_threshold*raw_err)
        njump = jump.sum(axis=0)
        
        ## Jumps only if first read not flagged
        jump[0,:,:] &= (groupdq[integ,0,:,:] == 0)
    
        ## Dilate jump
        msg = f'process_uncal: {file} {integ+1} dilate jumps'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        
        for i in range(NGROUPS-1):
            ji = jump[i,:,:] & (njump < max_njump)
            jump[i,:,:] |= nd.binary_dilation(ji,
                                              structure=np.ones((3,3)),
                                              iterations=jump_ndilate)

        ## Identify snowballs and grow masks around them
        for ij in range(NGROUPS-1):
            ji = jump[ij,:,:]

            js = nd.binary_erosion(ji, iterations=erode_snowballs)

            label, num_labels = nd.label(js)
            if num_labels == 0:
                ## No snowballs identified
                msg = f'process_uncal: {integ+1} snowball read {ij+1:>2} '
                msg += f' n={num_labels:>2}  '
                msg += f'nsat={int(jump_saturated.sum())//4:>6}'
                utils.log_comment(utils.LOGFILE, msg, show_date=False, 
                                  verbose=verbose)
                
                continue
        
            xy_label = nd.center_of_mass(label, labels=label, 
                                         index=np.unique(label)[1:])
            area = nd.sum(label > 0, labels=label, index=np.unique(label)[1:])

            R_label = np.sqrt(area/np.pi)
            yp, xp = np.indices((NYPIX, NXPIX))

            lmask = np.zeros_like(js)
            smask = np.zeros_like(js)

            for (yyi, xxi), ri in zip(xy_label, R_label):
                R = np.sqrt((xp-xxi)**2 + (yp-yyi)**2)
                lmask |= R < grow_snowballs*np.maximum(ri, 2)
                smask |= R < grow_snowballs/2*np.maximum(ri, 2)
    
            jump[ij,:,:] |= lmask
            
            ## Mask for saturated pixels
            blob_sat = (groupdq[integ,ij,:,:] > 0) | smask
            # blob_sat = nd.binary_erosion(blob_sat, iterations=1)
            blob_sat = nd.binary_dilation(blob_sat, iterations=2)*2
            #blob_sat = smask*2
            
            jump_saturated |= blob_sat.astype(jump_saturated.dtype)
            
            msg = f'process_uncal: {integ+1} snowball read {ij+1:>2} '
            msg += f' n={num_labels:>2}  '
            msg += f'nsat={int(jump_saturated.sum())//4:>6}'
            utils.log_comment(utils.LOGFILE, msg, show_date=False, 
                              verbose=verbose)

            if flag_for_persistence:
                groupdq[integ,ij,:,:] |= jump_saturated
        
        ## Number of jumps along the ramp
        njump = jump.sum(axis=0)
            
        ## Fit slopes when just zero or one jump found
        
        # Array index of first jump
        t2x = t2d[1:,:,:]*0+times[-1]*2
        t2x[jump > 0] = t2d[1:,:,:][jump > 0]
        ix = np.nanargmin(t2x, axis=0)
        
        # No jumps found
        ix[njump == 0] = NGROUPS
        
        # Array index after jump
        t2x = t2d[1:,:,:]*0
        t2x[jump > 0] = t2d[1:,:,:][jump > 0]
        ix2 = np.nanargmax(t2x, axis=0)
    
        n2d = np.zeros_like(i2d)
        ix2d = np.zeros_like(i2d)
        ix2d2 = np.zeros_like(i2d)
        for i, t in enumerate(times):
            n2d[i,:,:] = njump
            ix2d[i,:,:] = ix
            ix2d2[i,:,:] = ix2
    
        dvalid = np.zeros((NGROUPS, NYPIX, NXPIX), dtype=bool)
        
        # Masks for before / after jumps
        before = (i2d <= ix) & (ix2d > 1) & (groupdq[integ,:,:,:] == 0)
        after = (i2d > ix2) & (ix2d2 < NGROUPS-2) #& (n2d == 1)
        after &= (groupdq[integ,:,:,:] == 0)

        tmx = np.zeros_like(dd) #*np.nan
        dmx = np.zeros_like(dd) #*np.nan
        xme = np.zeros_like(dd) #*np.nan
        yme = np.zeros_like(dd) #*np.nan

        for msk, mask_type in zip([before, after],
                                  ['before first jump', 'after last jump']):
            
            msg = f'process_uncal: {file} {integ+1} slope mask {mask_type}'
            utils.log_comment(utils.LOGFILE, msg, show_date=False, 
                              verbose=verbose)
        
            ddm = dd*1
            ddm[~msk] = np.nan
            tm = t2d*1
            tm[~msk] = np.nan
            ymean = np.nanmean(ddm, axis=0)
            xmean = np.nanmean(tm, axis=0)
    
            dvalid[msk] = True
    
            tmx[msk] = (tm-xmean)[msk]
            dmx[msk] = (ddm-ymean)[msk]
    
            xme[msk] = (tm*0 + xmean)[msk]
            yme[msk] = (ddm*0 + ymean)[msk]
        
        msg = f'process_uncal: {file} {integ+1} OLS slope model'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)

        ## Simple ordinary least squares fit
        slope_num = np.nansum(tmx*dmx, axis=0)
        slope_den = np.nansum(tmx**2, axis=0)
        slope = slope_num/slope_den
        #slope_err = 1./np.sqrt(slope_den)
        
        ## Intercept
        ahat = yme - slope*xme
        # Model ramp
        smodel = ahat + slope*t2d
        
        ## Recompute with weights after removing the intercept
        ddm = (dd - ahat)
        ddm[~dvalid] = np.nan
        tm = t2d*1
        tm[~dvalid] = np.nan
        
        slope_num = np.nansum(ddm*tm/raw_var, axis=0)
        slope_den = np.nansum(tm**2/raw_var, axis=0)
        slope = slope_num/slope_den
        slope_err = 1/np.sqrt(slope_den)
        
        ## Output arrays
        sci = slope*1
        dq = (~np.isfinite(sci))*DQ_NAN | (njump > 4)*DQ_NJUMP
        dq |= (groupdq[integ,0,:,:] > 0)*DQ_FIRST_SATURATED
    
        sci[njump > 4] = np.nan
        sci[~np.isfinite(sci)] = 0.

        nsamp = dvalid.sum(axis=0)
        
        ## Residuals w.r.t model
        resid = (dd - smodel)
    
        xerr = np.sqrt((readnoise_2d)**2 + slope*t2d/gain_2d)
        nbad = ((np.abs(resid) > resid_thresh*xerr) & (xme > 0)).sum(axis=0)
        
        msg = f'process_uncal: {file} {integ+1} residuals > {resid_thresh}: '
        msg += f' {(nbad > 1).sum()}'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        
        if hot_thresh in ['auto']:
            if dark.meta.instrument.detector in HOT_THRESHOLDS:
                hot_thresh = HOT_THRESHOLDS[dark.meta.instrument.detector]
                
        ## Hot pixels
        if hot_type == 'raw':
            hot = (dark.data[integ,0,:,:] > hot_thresh[0]) #& (njump > 0)
            hot &= (dark.data[integ,1,:,:] > hot_thresh[0]*2)
        else:
            med_rate = np.nanmedian(ds[:2,:,:])
            hot = (ds[0,:,:]-med_rate > hot_thresh[0])
            hot = (ds[1,:,:]-med_rate > hot_thresh[0])
        
        hmask = nd.binary_erosion(hot, iterations=4)
        hmask = nd.binary_dilation(hmask, iterations=4)
        hot &= ~hmask
        
        msg = f'process_uncal: {file} {integ+1} {hot_type} hot pixels > '
        msg += f' {hot_thresh[0]}: {hot.sum()}'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        
        plus = np.array([[0,1,0],[1,1,1],[0,1,0]])
        hot_plus = nd.binary_dilation(hot, structure=np.ones((3,3)))

        hot = (ds[0,:,:] > hot_thresh[1]) & (njump > 0)
        
        hmask = nd.binary_erosion(hot, iterations=4)
        hmask = nd.binary_dilation(hmask, iterations=4)
        hot &= ~hmask
        
        msg = f'process_uncal: {file} {integ+1} very hot pixels > '
        msg += f' {hot_thresh[1]}: {hot.sum()}'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        
        hot_plus |= nd.binary_dilation(hot, structure=np.ones((3,3)))
    
        hot_plus |= nd.binary_dilation((njump > 4)) #, structure=np.ones((3,3)))
        
        # Final slope image
        # sci[njump > 1] = np.nan
        dq |= hot_plus*DQ_HOT + (nbad > 1)*DQ_RESID
        sci[~np.isfinite(sci) | (nbad > 1) | (hot_plus)] = 0.
    
        # Uncertainty
        valid = (np.isfinite(sci) & (sci != 0))
        rmask = resid * valid
        rmask[rmask == 0] = np.nan
        
        # Effective integration time used for uncertainty estimates
        itime = dark.meta.exposure.group_time #*dark.meta.exposure.ngroups
        
        # Read noise derived from the ramp residuals
        if dark.meta.exposure.ngroups > groups_for_rnoise:
            rn_i = utils.nmad(rmask[np.isfinite(rmask) & valid])
            msg = f'process_uncal: {file} {integ+1} computed RNOISE: {rn_i:.2f}'
            utils.log_comment(utils.LOGFILE, msg, show_date=False, 
                              verbose=verbose)
            
            rn_i = np.ones_like(readnoise_2d)*rn_i
        else:
            rn_i = readnoise_2d
        
        rn_var = 12 * rn_i**2 / np.maximum(nsamp**3-nsamp, 6)
        p_var = np.maximum(slope*itime,0)/np.sqrt(nsamp)/gain_2d
        err = np.sqrt(rn_var + p_var)/itime
        
        dq[nsamp <= 2] |= DQ_ERR
        dq |= (~np.isfinite(err))*DQ_ERR
        dq |= (dq > 0)*DQ_ANY
    
        sci[dq > 0] = 0
        err[dq > 0] = 0
        
        slope_list.append(slope)
        slope_err_list.append(slope_err)
        
        sci_list.append(sci)
        err_list.append(err)
        dq_list.append(dq)
        nsamp_list.append(nsamp)
        rn_list.append(rn_i)
        jump_list.append(jump)
        
        # Set DQ=4 jump flag in groupdq
        for ij in range(NGROUPS-1):
            groupdq[integ,ij+1,:,:] |= (jump[ij,:,:]*4).astype(groupdq.dtype)
            
    ## Combine iterations
    for iter in range(3):
        #print(f'iter combine {iter}')
        sci = np.array(sci_list)
        err = np.array(err_list)
        dqi = np.array(dq_list)
        ivar = 1/err**2
        ivar[(dqi > 0) | (err == 0)] = 0
    
        num = (sci*ivar).sum(axis=0)
        den = ivar.sum(axis=0)
    
        sci = num/den
        err = 1/np.sqrt(den)

        for integ in range(NINTS):
            nsamp = nsamp_list[integ]
            rn_var = 12*rn_list[integ]**2/np.maximum(nsamp**3-nsamp, 6)
            _var = rn_var + np.maximum(sci*itime,0)/gain_2d/np.sqrt(nsamp)
            err_list[integ] = np.sqrt(_var)/itime
            err_list[integ][dq_list[integ] > 0] = 0
    
    dq = ((~np.isfinite(sci+err)) | (den <= 0))*1 | np.min(dqi, axis=0)
    dq *= 1
    
    if dark.meta.instrument.detector in PIXELDQ_BITS:
        pixeldq_bits = PIXELDQ_BITS[dark.meta.instrument.detector]
        
        msg = f'process_uncal: {file} set pixeldq bits {pixeldq_bits}'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        dqp = np.zeros_like(dark.pixeldq)
        for b in pixeldq_bits:
            dqp |= dark.pixeldq & b
    
        dq |= dqp
    
    valid = dq == 0
    sci[~valid] = 0
    err[~valid] = 0
        
    ## Outlier rejection across integrations
    if NINTS >= outlier_min_nints:
        if rescale_uncertainty:
            sci_med = np.nanmedian(sci[valid])
            evalid = valid & (np.abs((sci - sci_med)/err) < 30)
            xerr_scale = utils.nmad(((sci - sci_med)/err)[evalid])
        else:
            xerr_scale = 1.
        
        msg = f'\nprocess_uncal: {file} find integration outliers '
        msg += f' sigmas={integration_sigmas}  err_scale={xerr_scale:.2f}'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        
        for isig, sn_limit in enumerate(integration_sigmas):
            xsci = np.array(sci_list)
            xerr = np.array(err_list)
            xdqi = np.array(dq_list)
            
            resid = (xsci - sci)/xerr #/xerr_scale #xerr/xerr_scale
            xivar = 1/xerr**2
            if isig > 0:
                bad = np.abs(resid) > sn_limit
            else:
                bad = resid > sn_limit
            
            xivar[(dqi > 0) | (err == 0) | bad] = 0
    
            num = (xsci*xivar).sum(axis=0)
            den = xivar.sum(axis=0)
    
            sci = num/den
            err = 1/np.sqrt(den)
    
        dq = ((~np.isfinite(sci+err)) | (den <= 0))*1 | np.min(dqi, axis=0)
        valid = dq == 0
        sci[~valid] = 0
        err[~valid] = 0
    
    rn_meas = np.nanmedian(rn_list)
    
    nsamp = np.array(nsamp_list).sum(axis=0)
    rn_tot = np.nanmean(np.array(rn_list), axis=0)
    rn_var = 12*rn_tot**2/np.maximum(nsamp**3-nsamp, 6)/itime**2
    p_var = np.maximum(sci*itime,0)/gain_2d/itime**2
    
    if rescale_uncertainty:
        
        err = np.sqrt(rn_var + p_var)
        
        if rescale_with_background & HAS_SEP:
            bkg = sep.Background(sci.astype(np.float32), mask=(~valid), 
                                 **bkg_kwargs)
            sci_med = bkg.back()
        else:
            sci_med = np.nanmedian(sci[valid])
            
        evalid = valid & (np.abs((sci - sci_med)/err) < 30)
        evalid &= (err > 0) & np.isfinite(err)
        
        err_scale = utils.nmad(((sci - sci_med)/err)[evalid])
        msg = f'\nprocess_uncal: {file} uncertainty scale {err_scale:.2f}'
        utils.log_comment(utils.LOGFILE, msg, show_date=False, verbose=verbose)
        
        rn_var *= err_scale**2
        p_var *= err_scale**2
        err = np.sqrt(rn_var + p_var)
        
        #print('xxx', utils.nmad(((sci - sci_med)/err)[evalid]))
        #import matplotlib.pyplot as plt
        #plt.hist(((sci-sci_med)/err)[evalid],  bins=np.linspace(-10,10,128))
        
    else:
        err_scale = 1.
        evalid = valid
        
    with pyfits.open(file.replace('_uncal','_rate')) as im:
        imsci = im['SCI'].data*1
        imerr = im['ERR'].data*1

        im['SCI'].header['RNMEAS'] = (rn_meas,
                                      'Estimated read noise from ramps')
        # im['SCI'].header['RNDET'] = (rn_det, 'Assumed read noise')
        
        im['SCI'].data = sci.astype(im['SCI'].data.dtype)
        im['ERR'].data = err.astype(im['ERR'].data.dtype)
        im['ERR'].header['ESCALE'] = (err_scale, 'Uncertainty scale factor')
        im['ERR'].header['ESCLNPIX'] = (evalid.sum(),
                                        'Number of pixels used for scaling')
        
        im['DQ'].data = dq.astype(im['DQ'].data.dtype)
        im['VAR_RNOISE'].data = rn_var.astype(im['VAR_RNOISE'].data.dtype)
        im['VAR_POISSON'].data = p_var.astype(im['VAR_POISSON'].data.dtype)
        
        im.writeto(file.replace('_uncal', output_extension), overwrite=True)
        
        
    debug = {'file':file,
             'sci':sci, 'err':err, 'dq':dq,
             'nsamp':nsamp, 'rn':rn_meas,
             'times':times, 'dark':dark,
             'jump':jump,
             'jump_list':jump_list,
             'groupdq':groupdq,
             'njump':njump, 'resid':resid,
             'smodel':smodel, 'slope':slope, 'ahat':ahat,
             'slope_err':slope_err,
             'raw_err':raw_err,
             'hot_plus':hot_plus,
             'slope_list':slope_list,
             'slope_err_list':slope_err_list,
             'sci_list':sci_list,
             'err_list':err_list,
             'dq_list':dq_list,
             'jump_saturated':jump_saturated,
             'gain_2d':gain_2d,
             'readnoise_2d':readnoise_2d,
             'imsci':imsci,'imerr':imerr,
        }
    
    # Reset LOGFILE
    utils.LOGFILE = _LOGFILE
    warnings.filterwarnings('default', category=RuntimeWarning)
    
    return debug


def grow_snowball_jumps(dark, jump, erode_snowballs=3, grow_snowballs=5, verbose=True, **kwargs):
    """
    Grow masks around large groups of pixels identified in the jump step that
    are likely "snowballs"
    
    Parameters
    ----------
    dark : 
    
    jump : 
    
    erode_snowballs : int
        Number of `scipy.ndimage.binary_erosion` iterations run on the jump
        mask of each read before identifying snowballs.  This removes small
        clumps of pixels that probably aren't big snowballs
    
    grow_snowballs : float
        Once snowballs are identified as big clumps of jump pixels, compute
        the pixel area of each smowball clump and make a circular mask with
        radius ``grow_snowballs * np.sqrt(area/pi)``
    
    Returns
    -------
    group_dq :array-like
        DQ mask array with dimensions ``NINT, NGROUP, NY, NX`` with the ``4``
        bit set for identified jumps
    
    """
    nint, ngroup, NYPIX, NXPIX = jump.groupdq.shape
    
    groupdq = (jump.groupdq*1).astype(jump.groupdq.dtype)
    
    jump_saturated = np.zeros((NYPIX, NXPIX), dtype=bool)
    
    for ii in range(nint):
        for ig in range(1,ngroup):
            ji = groupdq[ii,ig,:,:] & 4 > 0
            sat = dark.groupdq[ii,ig:,:,:].max(axis=0) > 0
            ji |= sat
            
            js = nd.binary_erosion(ji, iterations=erode_snowballs)
            
            jump_saturated |= js # | sat
            
            groupdq[ii,ig,:,:] |= (jump_saturated*2).astype(groupdq.dtype)
            
            label, num_labels = nd.label(js)
            if num_labels > 0:
                
                msg = f'process_uncal: {ii+1} snowball read {ig:>2} '
                msg += f' n={num_labels:>2}  '
                msg += f'nsat={int(jump_saturated.sum())//4:>6}'
                utils.log_comment(utils.LOGFILE, msg, show_date=False, 
                                  verbose=verbose)
                
                xy_label = nd.center_of_mass(label, labels=label, 
                                             index=np.unique(label)[1:])
                area = nd.sum(label > 0, labels=label, 
                              index=np.unique(label)[1:])

                R_label = np.sqrt(area/np.pi)
                yp, xp = np.indices((NYPIX, NXPIX))

                blob = np.zeros_like(js)
                #smask = np.zeros_like(js)

                for (yyi, xxi), ri in zip(xy_label, R_label):
                    R = np.sqrt((xp-xxi)**2 + (yp-yyi)**2)
                    blob |= R < grow_snowballs*np.maximum(ri, 2)
                    #smask |= R < grow_snowballs/2*np.maximum(ri, 2)
                
                groupdq[ii,ig,:,:] |= (blob*4).astype(groupdq.dtype)
    
    return groupdq


def get_level1_references(model):
    """
    Get CRDS metadata for reference files
    
    Parameters
    ----------
    model : `jwst.datamodels.ImageModel`
        Datamodel created by the pipeline steps, e.g., ``gain_step``
    
    Returns
    -------
    ref_file : list
        List of tuples of ``(step, reference_filename)``
    
    """    
    pipe = get_pipeline_object()
    
    steps = [pipe.group_scale,
             pipe.dq_init ,
             pipe.saturation,
             pipe.superbias,
             pipe.refpix,
             pipe.linearity,
             pipe.persistence,
             pipe.dark_current,
             pipe.jump,
             pipe.ramp_fit,
             pipe.gain_scale]
    
    ref_files = []
    keys = []
    for step in steps:
        for reftype in step.reference_file_types:
            _file = pipe.get_reference_file(model, reftype)
            #print(reftype, files)
            if _file == 'N/A':
                continue
            else:
                row = (reftype, 'crds://' + os.path.basename(_file))
                if reftype not in keys:
                    ref_files.append(row)
                    keys.append(reftype)
    
    return ref_files


def pipeline_level1(file, output_extension='_rate', maximum_cores='half', use_pipe_jumps=True, unset_jump_dq=True, force_uncal=True, debug=None, **kwargs):
    """
    Use the pipeline for the ramp fits with manual jump detections
    
    Parameters
    ----------
    file : str
        JWST raw "uncal" filename
    
    output_extension : str
        The script puts the derived ramp and uncertainty into a ``rate``-like 
        file with filename ``file.replace('_uncal', output_extension)``.
    
    maximum_cores : 'none', 'quarter', 'half', 'all'
        Number of processes for `jwst` jump and ramp pipeline steps
    
    unset_jump_dq : bool
        Unset the ``DQ=4`` pixels from the final product
    
    force_uncal : bool
        Require that ``file`` endswith "_uncal.fits"
    
    debug : dict, bool
        Previous output
    
    kwargs : dict
        Keyword arguments passed to `grizli.jwst_level1.process_uncal_level1`
    
    Returns
    -------
    debug : dict
        Full output from `grizli.jwst_level1.process_uncal_level1`
    
    gain : `jwst.datamodels.ImageModel`
        Result of the final `gain_scale` level 1 step, after the others
        have been run (dark, jump, etc.).
    
    """    
    #from jwst.pipeline import Detector1Pipeline
    frame = inspect.currentframe()
    _LOGFILE = utils.LOGFILE
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    utils.LOGFILE = file.split('.fits')[0] + '.log'
    utils.log_function_arguments(utils.LOGFILE, frame, 'process_uncal',
                                 ignore=['debug','_LOGFILE'],
                                 )
    
    if (not file.endswith('_uncal.fits')) & (force_uncal):
        utils.LOGFILE = _LOGFILE
        warnings.filterwarnings('default', category=RuntimeWarning)
        
        raise ValueError(f'{file} does not have _uncal extension')
        
        
    #pipe = Detector1Pipeline()
    pipe = get_pipeline_object()
    
    dark = process_level1_to_dark(file)
    pipe.jump.maximum_cores = maximum_cores
    jump = pipe.jump.process(dark)
    
    jumpdq = grow_snowball_jumps(dark, jump, **kwargs)
    jump.groupdq |= jumpdq
    
    # Unset some DQ flags    
    orig_dq = jump.pixeldq*1
    
    dq = np.zeros_like(jump.pixeldq)
    for i in range(1,20):
        #for i in PIXELDQ_BITS[dark.meta.instrument.detector]:
        if i in [15,]:
            continue
            
        dqi = 2**i & jump.pixeldq
        dq |= dqi
        
    jump.pixeldq = dq
    
    # Run Ramp_fit
    pipe.ramp_fit.maximum_cores = maximum_cores
    myramp = pipe.ramp_fit.process(jump)
    
    # Run Gain
    gain = pipe.gain_scale.process(myramp[0])
    
    bad = ~np.isfinite(gain.data + gain.err)
    bad |= dq > 0
    #bad |= (gain.dq & 1) == 1
    gain.dq[bad] |= 1
    gain.data[bad] = 0.
    gain.err[bad] = 0
    
    if unset_jump_dq:
        gain.dq -= (gain.dq & 4)
    
    # Update metadata with reference files since this doesn't seem to
    # be run automatically?
    ref_files = get_level1_references(gain)
    pipe.gain_scale.finalize_result(gain, ref_files)
    
    gain.write(file.replace('_uncal', output_extension),
               overwrite=True)
    
    
    utils.LOGFILE = _LOGFILE
    warnings.filterwarnings('default', category=RuntimeWarning)
    
    return debug, gain