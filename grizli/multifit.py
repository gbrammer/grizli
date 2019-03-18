"""Functionality for manipulating multiple grism exposures simultaneously
"""

import os
import time
import glob
from collections import OrderedDict
import multiprocessing as mp

import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

import astropy.units as u

## local imports
from . import utils
from . import model
#from . import stack
from .fitting import GroupFitter
from .utils_c import disperse
from .utils_c import interp

from .utils import GRISM_COLORS, GRISM_MAJOR, GRISM_LIMITS, DEFAULT_LINE_LIST


def test():
    
    import glob
    from grizlidev import utils
    import grizlidev.multifit
    
    reload(utils)
    reload(grizlidev.model)
    reload(grizlidev.multifit)
    
    files=glob.glob('i*flt.fits')
    output_list, filter_list = utils.parse_flt_files(files, uniquename=False)
    
    # grism_files = filter_list['G141'][164]
    # #grism_files.extend(filter_list['G141'][247])
    # 
    # direct_files = filter_list['F140W'][164][:4]
    #direct_files.extend(filter_list['F140W'][247][:4])
    
    # grp = grizlidev.multifit.GroupFLT(grism_files=grism_files, direct_files=direct_files)
    # 
    # 
    # grp = grizlidev.multifit.GroupFLT(grism_files=grism_files, direct_files=direct_files, ref_file=ref)

    # ref = 'MACS0416-F140W_drz_sci_filled.fits'
    # seg = 'hff_m0416_v0.1_bkg_detection_seg_grow.fits'
    # catalog = 'hff_m0416_v0.1_f140w.cat'
    # 
    # key = 'cl1301-11.3-122.5-g102'
    # seg = 'cl1301-11.3-14-122-f105w_seg.fits'
    # catalog = 'cl1301-11.3-14-122-f105w.cat'
    # #ref = 'cl1301-11.3-14-122-f105w_drz_sci.fits'
    # grism_files = output_list[key]
    # direct_files = output_list[key.replace('f105w','g102')]
    
    grism_files = filter_list['G141'][1]
    grism_files.extend(filter_list['G141'][33])
    
    grism_files = glob.glob('*cmb.fits')
    
    ref = 'F160W_mosaic.fits'
    seg = 'F160W_seg_blot.fits'
    catalog = '/Users/brammer/3DHST/Spectra/Work/3DHST_Detection/GOODS-N_IR.cat'
    
    direct_files = []

    reload(utils)
    reload(grizlidev.model)
    reload(grizlidev.multifit)

    grp = grizlidev.multifit.GroupFLT(grism_files=grism_files[:8], direct_files=direct_files, ref_file=ref, seg_file=seg, catalog=catalog)
    
    self = grp
    
    fit_info = {3286: {'mag':-99, 'spec': None},
                3279: {'mag':-99, 'spec': None}}
    
    fit_info = OrderedDict()
    
    bright = self.catalog['MAG_AUTO'] < 25
    ids = self.catalog['NUMBER'][bright]
    mags = self.catalog['MAG_AUTO'][bright]
    for id, mag in zip(ids, mags):
        fit_info[id] = {'mag':mag, 'spec': None}
    
    # Fast?
    #fit_info = {3212: {'mag':-99, 'spec': None}}
    
    #self.compute_single_model(3212)
    
    ### parallel
    self.compute_full_model(fit_info, store=False)
    
    ## Refine
    bright = (self.catalog['MAG_AUTO'] < 22) & (self.catalog['MAG_AUTO'] > 16)
    ids = self.catalog['NUMBER'][bright]*1
    mags = self.catalog['MAG_AUTO'][bright]*1
    so = np.argsort(mags)
    
    ids, mags = ids[so], mags[so]
    
    self.refine_list(ids, mags, ds9=ds9, poly_order=1)

    # bright = (self.catalog['MAG_AUTO'] < 22) & (self.catalog['MAG_AUTO'] > 16)
    # ids = self.catalog['NUMBER'][bright]*1
    # mags = self.catalog['MAG_AUTO'][bright]*1
    # so = np.argsort(mags)
    # 
    # self.refine_list(ids, mags, ds9=ds9, poly_order=5)
    
    beams = self.get_beams(3212)
    
    ### serial
    t0 = time.time()
    out = _compute_model(0, self.FLTs[i], fit_info, False, False)
    t1 = time.time()
    #print t1-t0
    
    id = 3219
    fwhm = 1200
    zr = [0.58,2.4]
    
    beams = grp.get_beams(id, size=30)
    mb = grizlidev.multifit.MultiBeam(beams)
    fit, fig = mb.fit_redshift(fwhm=fwhm, zr=zr, poly_order=3, dz=[0.003, 0.003])
    
    A, out_coeffs, chi2, modelf = mb.fit_at_z(poly_order=1)
    m2d = mb.reshape_flat(modelf)
    
def _loadFLT(grism_file, sci_extn, direct_file, pad, ref_file, 
               ref_ext, seg_file, verbose, catalog, ix):
    """Helper function for loading `.model.GrismFLT` objects with `multiprocessing`.
    
    TBD
    """
    import time
    try:
        import cPickle as pickle
    except:
        # Python 3
        import pickle
        
    ## slight random delay to avoid synchronization problems
    # np.random.seed(ix)
    # sleeptime = ix*1
    # print '%s sleep %.3f %d' %(grism_file, sleeptime, ix)
    # time.sleep(sleeptime)
    
    #print grism_file, direct_file
    
    new_root = '.{0:02d}.GrismFLT.fits'.format(sci_extn)
    save_file = grism_file.replace('_flt.fits', new_root)
    save_file = save_file.replace('_flc.fits', new_root)
    save_file = save_file.replace('_cmb.fits', new_root)
    save_file = save_file.replace('_rate.fits', new_root)
    
    if (grism_file.find('_') < 0) & ('GrismFLT' not in grism_file):
        save_file = 'xxxxxxxxxxxxxxxxxxx'
        
    if os.path.exists(save_file):
        print('Load {0}!'.format(save_file))
        
        fp = open(save_file.replace('GrismFLT.fits', 'GrismFLT.pkl'), 'rb')
        flt = pickle.load(fp)
        fp.close()
        
        status = flt.load_from_fits(save_file)
                
    else:    
        flt = model.GrismFLT(grism_file=grism_file, sci_extn=sci_extn,
                         direct_file=direct_file, pad=pad, 
                         ref_file=ref_file, ref_ext=ref_ext, 
                         seg_file=seg_file, shrink_segimage=True, 
                         verbose=verbose)
    
    if flt.direct.wcs.wcs.has_pc():
        for obj in [flt.grism, flt.direct]:
            obj.get_wcs()
    
    if catalog is not None:
        flt.catalog = flt.blot_catalog(catalog, 
                                   sextractor=('X_WORLD' in catalog.colnames))
        flt.catalog_file = catalog
                   
    else:
        flt.catalog = None 

    if flt.grism.instrument in ['NIRISS', 'NIRCAM']:
        flt.transform_NIRISS()
        
    return flt #, out_cat
    
def _fit_at_z(self, zgrid, i, templates, fitter, fit_background, poly_order):
    """
    For parallel processing
    """
    # self, z=0., templates={}, fitter='nnls',
    #              fit_background=True, poly_order=0
    print(i, zgrid[i])
    out = self.fit_at_z(z=zgrid[i], templates=templates,
                        fitter=fitter, poly_order=poly_order,
                        fit_background=fit_background)
    
    data = {'out':out, 'i':i}
    return data
    #A, coeffs[i,:], chi2[i], model_2d = out
    
def test_parallel():
    
    zgrid = np.linspace(1.1,1.3,10)
    templates = mb.load_templates(fwhm=800)
    fitter = 'nnls'
    fit_background = True
    poly_order = 0
    
    self.FLTs = []
    t0_pool = time.time()

    pool = mp.Pool(processes=4)
    results = [pool.apply_async(_fit_at_z, (mb, zgrid, i, templates, fitter, fit_background, poly_order)) for i in range(len(zgrid))]

    pool.close()
    pool.join()
    
    chi = zgrid*0.
    
    for res in results:
        data = res.get(timeout=1)
        A, coeffs, chi[data['i']], model_2d = data['out']
        #flt_i.catalog = cat_i

    t1_pool = time.time()
    
    
def _compute_model(i, flt, fit_info, is_cgs, store):
    """Helper function for computing model orders.
    """
    for id in fit_info:
        try:
            status = flt.compute_model_orders(id=id, compute_size=True,
                          mag=fit_info[id]['mag'], in_place=True, store=store,
                          spectrum_1d = fit_info[id]['spec'], is_cgs=is_cgs, 
                          verbose=False)
        except:
            print('Failed: {0} {1}'.format(flt.grism.parent_file, id))
            continue
            
    print('{0}: _compute_model Done'.format(flt.grism.parent_file))
        
    return i, flt.model, flt.object_dispersers
    
class GroupFLT():
    def __init__(self, grism_files=[], sci_extn=1, direct_files=[],
                 pad=200, group_name='group', 
                 ref_file=None, ref_ext=0, seg_file=None,
                 shrink_segimage=True, verbose=True, cpu_count=0,
                 catalog='', polyx=[0.3, 2.35],
                 MW_EBV=0.):
        """Main container for handling multiple grism exposures together
        
        Parameters
        ----------
        grism_files : list
            List of grism exposures (typically WFC3/IR "FLT" or ACS/UVIS "FLC"
            files). These can be from different grisms and/or orients.
            
        sci_extn : int
            Science extension to extract from the files in `grism_files`.  For 
            WFC3/IR this can only be 1, though for the two-chip instruments
            WFC3/UVIS and ACS/WFC3 this can be 1 or 2.
            
        direct_files : list
            List of direct exposures (typically WFC3/IR "FLT" or ACS/UVIS
            "FLC" files). This list should either be empty or should 
            correspond one-to-one with entries in the `grism_files` list, 
            i.e., from an undithered pair of direct and grism exposures.  If 
            such pairs weren't obtained or if you simply wish to ignore them
            and just use the `ref_file` reference image, set to an empty list
            (`[]`).
            
        pad : int
            Padding in pixels to apply around the edge of the detector to 
            allow modeling of sources that fall off of the nominal FOV.  For 
            this to work requires using a `ref_file` reference image that 
            covers this extra area.
            
        group_name : str
            Name to apply to products produced by this group.
            
        ref_file : `None` or str
            Undistorted reference image filename, e.g., a drizzled mosaic
            covering the area around a given grism exposure.
            
        ref_ext : 0
            FITS extension of the reference file where to find the image 
            itself.  
            
        seg_file : `None` or str
            Segmentation image filename.
            
        shrink_segimage : bool
            Do some preprocessing on the segmentation image to speed up the
            blotting to the distorted frame of the grism exposures.  
            
        verbose : bool
            Print verbose information.
            
        cpu_count : int
            Use parallelization if > 0.  If equal to zero, then use the 
            maximum number of available cores.
            
        catalog : str
            Catalog filename assocated with `seg_file`.  These are typically
            generated with "SExtractor", but the source of the files 
            themselves isn't critical.
            
        Attributes
        ----------
        catalog : `~astropy.table.Table`
            The table read in with from the above file specified in `catalog`.
        
        FLTs : list
            List of `~grizli.model.GrismFLT` objects generated from each of 
            the files in the `grism_files` list.
        
        grp.N : int
            Number of grism files (i.e., `len(FLTs)`.)
        
        """
        self.N = len(grism_files)
        if len(direct_files) != len(grism_files):
            direct_files = ['']*self.N
        
        self.grism_files = grism_files
        self.direct_files = direct_files
        self.group_name = group_name
        
        # Wavelengths for polynomial fits
        self.polyx = polyx
        
        ### Read catalog
        if catalog:
            if isinstance(catalog, str):
                self.catalog = utils.GTable.gread(catalog)                
            else:
                self.catalog = catalog
            
            # necessary columns from SExtractor / photutils
            pairs = [['NUMBER','id'], 
                     ['MAG_AUTO', 'mag'], 
                     ['MAGERR_AUTO', 'mag_err']]
                     
            cols = self.catalog.colnames
            for pair in pairs:
                if (pair[0] not in cols) & (pair[1] in cols):
                    self.catalog[pair[0]] = self.catalog[pair[1]]
                    
        else:
            self.catalog = None
                 
        if cpu_count == 0:
            cpu_count = mp.cpu_count()
        
        if cpu_count < 0:
            ### serial
            self.FLTs = []
            t0_pool = time.time()
            for i in range(self.N):
                flt = _loadFLT(self.grism_files[i], sci_extn, self.direct_files[i], pad, ref_file, ref_ext, seg_file, verbose, self.catalog, i)
                self.FLTs.append(flt)
                
            t1_pool = time.time()
        else:
            ### Read files in parallel
            self.FLTs = []
            t0_pool = time.time()
        
            pool = mp.Pool(processes=cpu_count)
            results = [pool.apply_async(_loadFLT, (self.grism_files[i], sci_extn, self.direct_files[i], pad, ref_file, ref_ext, seg_file, verbose, self.catalog, i)) for i in range(self.N)]
        
            pool.close()
            pool.join()
    
            for res in results:
                flt_i = res.get(timeout=1)
                #flt_i.catalog = cat_i
                
                # somehow WCS getting flipped from cd to pc in res.get()???
                if flt_i.direct.wcs.wcs.has_pc():
                    for obj in [flt_i.grism, flt_i.direct]:
                        obj.get_wcs()
                
                self.FLTs.append(flt_i)
                
                
            t1_pool = time.time()
        
        # Parse grisms & PAs
        self.Ngrism = {}
        for i in range(self.N):
            if self.FLTs[i].grism.instrument == 'NIRISS':
                grism = self.FLTs[i].grism.pupil
            else:
                grism = self.FLTs[i].grism.filter

            if grism in self.Ngrism:
                self.Ngrism[grism] += 1
            else:
                self.Ngrism[grism] = 1

        self.grisms = list(self.Ngrism.keys())

        self.PA = {}
        for g in self.Ngrism:
            self.PA[g] = {}

        for i in range(self.N):
            if self.FLTs[i].grism.instrument == 'NIRISS':
                grism = self.FLTs[i].grism.pupil
            else:
                grism = self.FLTs[i].grism.filter

            PA = self.FLTs[i].get_dispersion_PA(decimals=0)
            if PA in self.PA[grism]:
                self.PA[grism][PA].append(i)
            else:
                self.PA[grism][PA] = [i]
        
        if verbose:
            print('Files loaded - {0:.2f} sec.'.format(t1_pool - t0_pool))
    
    def save_full_data(self, warn=True):
        """Save models and data files for fast regeneration.
        
        The filenames of the outputs are generated from the input grism 
        exposure filenames with the following:
        
            >>> file = 'ib3701ryq_flt.fits'
            >>> sci_extn = 1
            >>> new_root = '.{0:02d}.GrismFLT.fits'.format(sci_extn)
            >>> 
            >>> save_file = file.replace('_flt.fits', new_root)
            >>> save_file = save_file.replace('_flc.fits', new_root)
            >>> save_file = save_file.replace('_cmb.fits', new_root)
            >>> save_file = save_file.replace('_rate.fits', new_root)
                
        It will also save data to a `~pickle` file:
        
            >>> pkl_file = save_file.replace('.fits', '.pkl')
        
        Parameters
        ----------
        warn : bool
            Print a warning and skip if an output file is already found to
            exist.
                
        Notes
        -----
        The save filename format was changed May 9, 2017 to the format like 
        `ib3701ryq.01.GrismFLT.fits` from `ib3701ryq_GrismFLT.fits` to both
        allow easier filename parsing and also to allow for instruments that 
        have multiple `SCI` extensions in a single calibrated file
        (e.g., ACS and WFC3/UVIS).
        """      
        for i in range(self.N):
            file = self.FLTs[i].grism_file
            if self.FLTs[i].grism.data is None:
                if warn:
                    print('{0}: Looks like data already saved!'.format(file))
                    continue
            
            new_root = '.{0:02d}.GrismFLT.fits'.format(self.FLTs[i].grism.sci_extn)
            
            save_file = file.replace('_flt.fits', new_root)
            save_file = save_file.replace('_flc.fits', new_root)
            save_file = save_file.replace('_cmb.fits', new_root)
            save_file = save_file.replace('_rate.fits', new_root)
            print('Save {0}'.format(save_file))
            self.FLTs[i].save_full_pickle()
            
            ### Reload initialized data
            self.FLTs[i].load_from_fits(save_file)
            
    def extend(self, new, verbose=True):
        """Add another `GroupFLT` instance to `self`
        
        This function appends the exposures if a separate `GroupFLT` instance
        to the current instance.  You might do this, for example, if you 
        generate separate `GroupFLT` instances for different grisms and 
        reference images with different filters.
        """
        import copy
        self.FLTs.extend(new.FLTs)
        self.N = len(self.FLTs)
        
        direct_files = copy.copy(self.direct_files)
        direct_files.extend(new.direct_files)
        self.direct_files = direct_files

        grism_files = copy.copy(self.grism_files)
        grism_files.extend(new.grism_files)
        self.grism_files = grism_files
                        
        # self.direct_files.extend(new.direct_files)
        # self.grism_files.extend(new.grism_files)
        
        if verbose:
            print('Now we have {0:d} FLTs'.format(self.N))
            
    def compute_single_model(self, id, center_rd=None, mag=-99, size=-1, store=False, spectrum_1d=None, is_cgs=False, get_beams=None, in_place=True, psf_param_dict={}):
        """Compute model spectrum in all exposures
        TBD
        
        Parameters
        ----------
        id : type
        
        center_rd : None
        
        mag : type
        
        size : type
        
        store : type
        
        spectrum_1d : type
        
        get_beams : type
        
        in_place : type
        
        
        Returns
        -------
        TBD
               
        """
        out_beams = []
        for flt in self.FLTs:                
            if flt.grism.parent_file in psf_param_dict:
                psf_params = psf_param_dict[flt.grism.parent_file]
            else:
                psf_params = None
            
            if center_rd is None:
                x = y = None
            else:
                x, y = flt.direct.wcs.all_world2pix(np.array(center_rd)[None,:], 0).flatten()
                
            status = flt.compute_model_orders(id=id, x=x, y=y, verbose=False,
                          size=size, compute_size=(size < 0),
                          mag=mag, in_place=in_place, store=store,
                          spectrum_1d=spectrum_1d, is_cgs=is_cgs,
                          get_beams=get_beams, psf_params=psf_params)
            
            out_beams.append(status)
        
        if get_beams:
            return out_beams
        else:
            return True
            
    def compute_full_model(self, fit_info=None, verbose=True, store=False, 
                           mag_limit=25, coeffs=[1.2, -0.5], cpu_count=0,
                           is_cgs=False):
        """TBD
        """
        if cpu_count == 0:
            cpu_count = mp.cpu_count()
        
        if fit_info is None:
            bright = self.catalog['MAG_AUTO'] < mag_limit
            ids = self.catalog['NUMBER'][bright]
            mags = self.catalog['MAG_AUTO'][bright]

            # Polynomial component
            #xspec = np.arange(0.3, 5.35, 0.05)-1
            xspec = np.arange(self.polyx[0], self.polyx[1], 0.05)-1
            
            yspec = [xspec**o*coeffs[o] for o in range(len(coeffs))]
            xspec = (xspec+1)*1.e4
            yspec = np.sum(yspec, axis=0)
            
            fit_info = OrderedDict()
            for id, mag in zip(ids, mags):
                fit_info[id] = {'mag':mag, 'spec': [xspec, yspec]}
            
            is_cgs = False
            
        t0_pool = time.time()
        
        pool = mp.Pool(processes=cpu_count)
        results = [pool.apply_async(_compute_model, (i, self.FLTs[i], fit_info, is_cgs, store)) for i in range(self.N)]

        pool.close()
        pool.join()
                
        for res in results:
            i, model, dispersers = res.get(timeout=1)
            self.FLTs[i].object_dispersers = dispersers
            self.FLTs[i].model = model
            
        t1_pool = time.time()
        if verbose:
            print('Models computed - {0:.2f} sec.'.format(t1_pool - t0_pool))
        
    def get_beams(self, id, size=10, center_rd=None, beam_id='A',
                  min_overlap=0.1, min_valid_pix=10, min_mask=0.01, 
                  min_sens=0.08, get_slice_header=True):
        """Extract 2D spectra "beams" from the GroupFLT exposures.
        
        Parameters
        ----------
        id : int
            Catalog ID of the object to extract.
            
        size : int
            Half-size of the 2D spectrum to extract, along cross-dispersion
            axis.
            
        center_rd : optional, (float, float)
            Extract based on RA/Dec rather than catalog ID.
            
        beam_id : type
            Name of the order to extract.  
            
        min_overlap : float
            Fraction of the spectrum along wavelength axis that has one 
            or more valid pixels.
            
        min_valid_pix : int
            Minimum number of valid pixels (`beam.fit_mask == True`) in 2D
            spectrum.
        
        min_mask : float
            Minimum factor relative to the maximum pixel value of the flat
            f-lambda model where the 2D cutout data are considered good.  
            Passed through to `~grizli.model.BeamCutout`.
        
        min_sens : float
            See `~grizli.model.BeamCutout`.
            
        get_slice_header : bool
            Passed to `~grizli.model.BeamCutout`.
            
        Returns
        -------
        beams : list
            List of `~grizli.model.BeamCutout` objects.
        
        """
        beams = self.compute_single_model(id, center_rd=center_rd, size=size, store=False, get_beams=[beam_id])
        
        out_beams = []
        for flt, beam in zip(self.FLTs, beams):
            try:
                out_beam = model.BeamCutout(flt=flt, beam=beam[beam_id],
                                        conf=flt.conf, min_mask=min_mask,
                                        min_sens=min_sens,
                                        get_slice_header=get_slice_header)
            except:
                #print('Except: get_beams')
                continue
            
            valid =  (out_beam.grism['SCI'] != 0) 
            valid &= out_beam.fit_mask.reshape(out_beam.sh)               
            hasdata = (valid.sum(axis=0) > 0).sum()
            if hasdata*1./out_beam.model.shape[1] < min_overlap:
                continue
            
            # Empty direct image?
            if out_beam.beam.total_flux == 0:
                continue
                
            if out_beam.fit_mask.sum() < min_valid_pix:    
                continue
                
            out_beams.append(out_beam)
            
        return out_beams
    
    def refine_list(self, ids=[], mags=[], poly_order=3, mag_limits=[16,24], 
                    max_coeff=5, ds9=None, verbose=True, fcontam=0.5,
                    wave=np.linspace(0.2, 2.5e4, 100)):
        """Refine contamination model for list of objects.  Loops over `refine`.
        
        Parameters
        ----------
        ids : list
            List of object IDs
        
        mags : list
            Magnitudes to to along with IDs.  If `ids` and `mags` not 
            specified, then get the ID list from `self.catalog['MAG_AUTO']`.
        
        poly_order : int
            Order of the polynomial fit to the spectra.
        
        mag_limits : [float, float]
            Magnitude limits of objects to fit from `self.catalog['MAG_AUTO']`
            when `ids` and `mags` not set.
        
        max_coeff : float
            Fit is considered bad when one of the coefficients is greater
            than this value.  See `refine`.
        
        ds9 : `~grizli.ds9.DS9`, optional
            Display the refined models to DS9 as they are computed.
        
        verbose : bool
            Print fit coefficients.
        
        fcontam : float
            Contamination weighting parameter.
                
        wave : `~numpy.array`
            Wavelength array for the polynomial fit.  
        
        Returns
        -------
        Updates `self.model` in place.
        
        """
        if (len(ids) == 0) | (len(ids) != len(mags)):
            bright = ((self.catalog['MAG_AUTO'] < mag_limits[1]) &
                      (self.catalog['MAG_AUTO'] > mag_limits[0]))
                      
            ids = self.catalog['NUMBER'][bright]*1
            mags = self.catalog['MAG_AUTO'][bright]*1
            
            so = np.argsort(mags)
            ids, mags = ids[so], mags[so]
        
        #wave = np.linspace(0.2,5.4e4,100)
        poly_templates = utils.polynomial_templates(wave, order=poly_order, line=False)
            
        for id, mag in zip(ids, mags):
            self.refine(id, mag=mag, poly_order=poly_order,
                        max_coeff=max_coeff, size=30, ds9=ds9,
                        verbose=verbose, fcontam=fcontam, 
                        templates=poly_templates)
    
    def refine(self, id, mag=-99, poly_order=3, size=30, ds9=None, verbose=True, max_coeff=2.5, fcontam=0.5, templates=None):
        """Fit polynomial to extracted spectrum of single object to use for contamination model.
        
        Parameters
        ----------
        id : int
            Object ID to extract.
        
        mag : float
            Object magnitude.  Determines which orders to extract; see
            `~grizli.model.GrismFLT.compute_model_orders`.
        
        poly_order : int
            Order of the polynomial to fit.
        
        size : int
            Size of cutout to extract.
        
        ds9 : `~grizli.ds9.DS9`, optional
            Display the refined models to DS9 as they are computed.
        
        verbose : bool
            Print information about the fit
        
        max_coeff : float
            The script computes the implied flux of the polynomial template
            at the pivot wavelength of the direct image filters.  If this
            flux is greater than `max_coeff` times the *observed* flux in the
            direct image, then the polynomal fit is considered bad.

        fcontam : float
            Contamination weighting parameter.
            
        templates : dict, optional
            Precomputed template dictionary.  If `None` then compute 
            polynomial templates with order `poly_order`.
        
        Returns
        -------
        Updates `self.model` in place.
        
        """
        beams = self.get_beams(id, size=size, min_overlap=0.5, get_slice_header=False)
        if len(beams) == 0:
            return True
        
        mb = MultiBeam(beams, fcontam=fcontam)
        
        if templates is None:
            wave = np.linspace(0.9*mb.wavef.min(),1.1*mb.wavef.max(),100)
            templates = utils.polynomial_templates(wave, order=poly_order,
                                                   line=False)
        
        try:
            tfit = mb.template_at_z(z=0, templates=templates, fit_background=True, fitter='lstsq', get_uncertainties=2)
        except:
            return False
            
        scale_coeffs = [tfit['cfit']['poly {0}'.format(i)][0] for i in range(1+poly_order)]
        xspec, ypoly = tfit['cont1d'].wave, tfit['cont1d'].flux
        
        # Check where templates inconsistent with broad-band fluxes
        xb = [beam.direct.ref_photplam if beam.direct['REF'] is not None else beam.direct.photplam for beam in beams]
        obs_flux = np.array([beam.beam.total_flux for beam in beams])
        mod_flux = np.polyval(scale_coeffs[::-1], np.array(xb)/1.e4-1)
        nonz = obs_flux != 0
        
        if (np.abs(mod_flux/obs_flux)[nonz].max() > max_coeff) | ((~np.isfinite(mod_flux/obs_flux)[nonz]).sum() > 0) | (np.min(mod_flux[nonz]) < 0) | ((~np.isfinite(ypoly)).sum() > 0):
            if verbose:
                cstr = ' '.join(['{0:9.2e}'.format(c) for c in scale_coeffs])
                print('{0:>5d} mag={1:6.2f} {2} xx'.format(id, mag, cstr))

            return True
        
        # Put the refined model into the full-field model    
        self.compute_single_model(id, mag=mag, size=-1, store=False, spectrum_1d=[xspec, ypoly], is_cgs=True, get_beams=None, in_place=True)
        
        # Display the result?
        if ds9:
            flt = self.FLTs[0]
            mask = flt.grism['SCI'] != 0
            ds9.view((flt.grism['SCI'] - flt.model)*mask,
                      header=flt.grism.header)
        
        if verbose:
            cstr = ' '.join(['{0:9.2e}'.format(c) for c in scale_coeffs])
            print('{0:>5d} mag={1:6.2f} {2}'.format(id, mag, cstr))
            
        return True
        #m2d = mb.reshape_flat(modelf)
        
    ############
    def old_refine(self, id, mag=-99, poly_order=1, size=30, ds9=None, verbose=True, max_coeff=2.5):
        """TBD
        """
        # Extract and fit beam spectra
        beams = self.get_beams(id, size=size, min_overlap=0.5, get_slice_header=False)
        if len(beams) == 0:
            return True
        
        mb = MultiBeam(beams)
        try:
            A, out_coeffs, chi2, modelf = mb.fit_at_z(poly_order=poly_order, fit_background=True, fitter='lstsq')
        except:
            return False
            
        # Poly template
        scale_coeffs = out_coeffs[mb.N*mb.fit_bg:mb.N*mb.fit_bg+mb.n_poly]
        xspec, yfull = mb.eval_poly_spec(out_coeffs)
        
        # Check where templates inconsistent with broad-band fluxes
        xb = [beam.direct.ref_photplam if beam.direct['REF'] is not None else beam.direct.photplam for beam in beams]
        fb = [beam.beam.total_flux for beam in beams]
        mb = np.polyval(scale_coeffs[::-1], np.array(xb)/1.e4-1)
        
        if (np.abs(mb/fb).max() > max_coeff) | (~np.isfinite(mb/fb).sum() > 0) | (np.min(mb) < 0):
            if verbose:
                print('{0} mag={1:6.2f} {2} xx'.format(id, mag, scale_coeffs))

            return True
        
        # Put the refined model into the full-field model    
        self.compute_single_model(id, mag=mag, size=-1, store=False, spectrum_1d=[(xspec+1)*1.e4, yfull], is_cgs=True, get_beams=None, in_place=True)
        
        # Display the result?
        if ds9:
            flt = self.FLTs[0]
            mask = flt.grism['SCI'] != 0
            ds9.view((flt.grism['SCI'] - flt.model)*mask,
                      header=flt.grism.header)
        
        if verbose:
            print('{0} mag={1:6.2f} {2}'.format(id, mag, scale_coeffs))
            
        return True
        #m2d = mb.reshape_flat(modelf)
    
    def make_stack(self, id, size=20, target='grism', skip=True, fcontam=1., scale=1, save=True, kernel='point', pixfrac=1, diff=True):
        """Make drizzled 2D stack for a given object
        
        Parameters
        ----------
        id : int
            Object ID number.
        
        target : str
            Rootname for output files.
            
        skip : bool
            If True and the stack PNG file already exists, don't proceed.
            
        fcontam : float
            Contamination weighting parameter.
        
        save : bool
            Save the figure and FITS HDU to files with names like
            
                >>> img_file = '{0}_{1:05d}.stack.png'.format(target, id)
                >>> fits_file = '{0}_{1:05d}.stack.fits'.format(target, id)
        
        diff : bool
             Plot residual in final stack panel.
             
        Returns
        -------
        hdu : `~astropy.io.fits.HDUList`
            FITS HDU of the stacked spectra.
        
        fig : `~matplotlib.figure.Figure`
            Stack figure object.  
                                
        """
        print(target, id)
        if os.path.exists('{0}_{1:05d}.stack.png'.format(target, id)) & skip:
            return True
        
        beams = self.get_beams(id, size=size, beam_id='A')
        if len(beams) == 0:
            print('id = {0}: No beam cutouts available.'.format(id))
            return None
            
        mb = MultiBeam(beams, fcontam=fcontam, group_name=target)

        hdu, fig = mb.drizzle_grisms_and_PAs(fcontam=fcontam, flambda=False,
                                             size=size, scale=scale, 
                                             kernel=kernel, pixfrac=pixfrac,
                                             diff=diff)
                                             
        if save:
            fig.savefig('{0}_{1:05d}.stack.png'.format(target, id))
            hdu.writeto('{0}_{1:05d}.stack.fits'.format(target, id),
                        clobber=True)
        
        return hdu, fig
    
    def drizzle_grism_models(self, root='grism_model', kernel='square', scale=0.1, pixfrac=1):
        """
        Make model-subtracted drizzled images of each grism / PA
        
        Parameters
        ----------
        root : str
            Rootname of the output files.
            
        kernel : str
            Drizzle kernel e.g., ('square', 'point').
            
        scale : float
            Drizzle `scale` parameter, pixel scale in arcsec.
            
        pixfrac : float
            Drizzle "pixfrac".
        
        """
        try:
            from .utils import drizzle_array_groups
        except:
            from grizli.utils import drizzle_array_groups
            
        # Loop through grisms and PAs
        for g in self.PA:
            for pa in self.PA[g]:
                idx = self.PA[g][pa]

                N = len(idx)
                sci_list = [self.FLTs[i].grism['SCI'] for i in idx]
                clean_list = [self.FLTs[i].grism['SCI']-self.FLTs[i].model 
                                 for i in idx]

                wht_list = [1/self.FLTs[i].grism['ERR']**2 for i in idx]
                for i in range(N):
                    mask = ~np.isfinite(wht_list[i])
                    wht_list[i][mask] = 0

                wcs_list = [self.FLTs[i].grism.wcs for i in idx]
                for i, ix in enumerate(idx):
                    if wcs_list[i]._naxis[0] == 0:
                        wcs_list[i]._naxis = self.FLTs[ix].grism.sh
                        
                # Science array
                outfile='{0}-{1}-{2}_grism_sci.fits'.format(root, g.lower(),
                                                            pa)
                print(outfile)
                out = drizzle_array_groups(sci_list, wht_list, wcs_list,
                                           scale=scale, kernel=kernel, 
                                           pixfrac=pixfrac)
                                           
                outsci, _, _, header, outputwcs = out
                header['FILTER'] = g
                header['PA'] = pa
                pyfits.writeto(outfile, data=outsci, header=header, 
                               overwrite=True, output_verify='fix')

                # Model-subtracted
                outfile='{0}-{1}-{2}_grism_clean.fits'.format(root, g.lower(), 
                                                              pa)
                print(outfile) 
                out = drizzle_array_groups(clean_list, wht_list, wcs_list,
                                           scale=scale, kernel=kernel, 
                                           pixfrac=pixfrac)
                                           
                outsci, _, _, header, outputwcs = out
                header['FILTER'] = g
                header['PA'] = pa
                pyfits.writeto(outfile, data=outsci, header=header, 
                               overwrite=True, output_verify='fix')
          
    def drizzle_full_wavelength(self, wave=1.4e4, ref_header=None,
                     kernel='point', pixfrac=1., verbose=True, 
                     offset=[0,0], fcontam=0.):
        """Drizzle FLT frames recentered at a specified wavelength
        
        Script computes polynomial coefficients that define the dx and dy
        offsets to a specific dispersed wavelengh relative to the reference
        position and adds these to the SIP distortion keywords before
        drizzling the input exposures to the output frame.
                
        Parameters
        ----------
        wave : float
            Reference wavelength to center the output products
            
        ref_header : `~astropy.io.fits.Header`
            Reference header for setting the output WCS and image dimensions.
            
        kernel : str, ('square' or 'point')
            Drizzle kernel to use
        
        pixfrac : float
            Drizzle PIXFRAC (for `kernel` = 'point')
        
        verbose : bool
            Print information to terminal
            
        Returns
        -------
        sci, wht : `~np.ndarray`
            Drizzle science and weight arrays with dimensions set in
            `ref_header`.
        """
        from astropy.modeling import models, fitting
        import astropy.wcs as pywcs
        
        # try:
        #     import drizzle
        #     if drizzle.__version__ != '1.12.99':
        #         # Not the fork that works for all input/output arrays
        #         raise(ImportError)
        #     
        #     #print('drizzle!!')
        #     from drizzle.dodrizzle import dodrizzle
        #     drizzler = dodrizzle
        #     dfillval = '0'
        # except:
        from drizzlepac import adrizzle
        adrizzle.log.setLevel('ERROR')
        drizzler = adrizzle.do_driz
        dfillval = 0
            
        ## Quick check now for which grism exposures we should use
        if wave < 1.1e4:
            use_grism = 'G102'
        else:
            use_grism = 'G141'
        
        # Get the configuration file
        conf = None
        for i in range(self.N):
            if self.FLTs[i].grism.filter == use_grism:
                conf = self.FLTs[i].conf
        
        # Grism not found in list
        if conf is None:
            return False
        
        # Compute field-dependent dispersion parameters
        dydx_0_p = conf.conf['DYDX_A_0']
        dydx_1_p = conf.conf['DYDX_A_1']

        dldp_0_p = conf.conf['DLDP_A_0']
        dldp_1_p = conf.conf['DLDP_A_1']

        yp, xp = np.indices((1014,1014)) # hardcoded for WFC3/IR
        sk = 10 # don't need to evaluate at every pixel

        dydx_0 = conf.field_dependent(xp[::sk,::sk], yp[::sk,::sk], dydx_0_p)
        dydx_1 = conf.field_dependent(xp[::sk,::sk], yp[::sk,::sk], dydx_1_p)

        dldp_0 = conf.field_dependent(xp[::sk,::sk], yp[::sk,::sk], dldp_0_p)
        dldp_1 = conf.field_dependent(xp[::sk,::sk], yp[::sk,::sk], dldp_1_p)
        
        # Inverse pixel offsets from the specified wavelength
        dp = (wave - dldp_0)/dldp_1
        i_x, i_y = 1, 0 # indexing offsets
        dx = dp/np.sqrt(1+dydx_1) + i_x
        dy = dydx_0 + dydx_1*dx + i_y
        
        dx += offset[0]
        dy += offset[1]
        
        # Compute polynomial coefficients
        p_init = models.Polynomial2D(degree=4)
        #fit_p = fitting.LevMarLSQFitter()
        fit_p = fitting.LinearLSQFitter()
        p_dx = fit_p(p_init, xp[::sk,::sk]-507, yp[::sk,::sk]-507, -dx)
        p_dy = fit_p(p_init, xp[::sk,::sk]-507, yp[::sk,::sk]-507, -dy)

        # Output WCS
        out_wcs = pywcs.WCS(ref_header, relax=True)
        out_wcs.pscale = utils.get_wcs_pscale(out_wcs)
        
        # Initialize outputs
        shape = (ref_header['NAXIS2'], ref_header['NAXIS1'])
        outsci = np.zeros(shape, dtype=np.float32)
        outwht = np.zeros(shape, dtype=np.float32)
        outctx = np.zeros(shape, dtype=np.int32)
        
        # Loop through exposures
        for i in range(self.N):
            flt = self.FLTs[i]
            if flt.grism.filter != use_grism:
                continue
                
            h = flt.grism.header.copy()

            # Update SIP coefficients
            for j, p in enumerate(p_dx.param_names):
                key = 'A_'+p[1:]
                if key in h:
                    h[key] += p_dx.parameters[j]
                else:
                    h[key] = p_dx.parameters[j]

            for j, p in enumerate(p_dy.param_names):
                key = 'B_'+p[1:]
                if key in h:
                    h[key] += p_dy.parameters[j]
                else:
                    h[key] = p_dy.parameters[j]
            
            line_wcs = pywcs.WCS(h, relax=True)
            line_wcs.pscale = utils.get_wcs_pscale(line_wcs)
            if not hasattr(line_wcs, 'pixel_shape'):
                line_wcs.pixel_shape = line_wcs._naxis1, line_wcs._naxis2

            # Science and wht arrays
            sci = flt.grism['SCI'] - flt.model
            wht = 1/(flt.grism['ERR']**2)
            scl = np.exp(-(fcontam*np.abs(flt.model)/flt.grism['ERR']))
            wht *= scl
            
            wht[~np.isfinite(wht)] = 0
            
            # Drizzle it
            if verbose:
                print('Drizzle {0} to wavelength {1:.2f}'.format(flt.grism.parent_file, wave))
                                                        
            drizzler(sci, line_wcs, wht, out_wcs, 
                             outsci, outwht, outctx, 1., 'cps', 1,
                             wcslin_pscale=line_wcs.pscale, uniqid=1, 
                             pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
        
        # Done!
        return outsci, outwht
            
    
class MultiBeam(GroupFitter):
    def __init__(self, beams, group_name='group', fcontam=0., psf=False, polyx=[0.3, 2.5], MW_EBV=0., min_mask=0.01, min_sens=0.08, sys_err=0.0, verbose=True):
        """Tools for dealing with multiple `~.model.BeamCutout` instances 
        
        Parameters
        ----------
        beams : list
            List of `~.model.BeamCutout` objects.
        
        group_name : str
            Rootname to use for saved products
            
        fcontam : float
            Factor to use to downweight contaminated pixels.  The pixel 
            inverse variances are scaled by the following weight factor when 
            evaluating chi-squared of a 2D fit,
            
            `weight = np.exp(-(fcontam*np.abs(contam)*np.sqrt(ivar)))`
            
            where `contam` is the contaminating flux and `ivar` is the initial
            pixel inverse variance.
            
        psf : bool
            Fit an ePSF model to the direct image to use as the morphological
            reference.
            
        MW_EBV : float
            Milky way foreground extinction.
        
        min_mask : float
            Minimum factor relative to the maximum pixel value of the flat
            f-lambda model where the 2D cutout data are considered good.  
            Passed through to `~grizli.model.BeamCutout`.

        min_sens : float
            See `~grizli.model.BeamCutout`.
            
        sys_err : float
            Systematic error added in quadrature to the pixel variances: 
                
                `var_total = var_initial + (beam.sci*sys_err)**2`
                
        Attributes
        ----------
        TBD : type
        
        """     
        self.group_name = group_name
        self.fcontam = fcontam
        self.polyx = polyx
        self.min_mask = min_mask
        self.min_sens = min_sens
        
        if isinstance(beams, str):
            self.load_master_fits(beams, verbose=verbose)            
        else:
            if isinstance(beams[0], str):
                ### `beams` is list of strings
                if 'beams.fits' in beams[0]:
                    # Master beam files
                    self.load_master_fits(beams[0], verbose=verbose)            
                    for i in range(1, len(beams)):
                        b_i = MultiBeam(beams[i], group_name=group_name, fcontam=fcontam, psf=psf, polyx=polyx, MW_EBV=np.maximum(MW_EBV, 0), sys_err=sys_err, verbose=verbose)
                        self.extend(b_i)
                        
                else:
                    # List of individual beam.fits files
                    self.load_beam_fits(beams)            
            else:
                self.beams = beams
        
        # minimum error
        self.sys_err = sys_err
        for beam in self.beams:
            beam.ivarf = 1./(1/beam.ivarf + (sys_err*beam.scif)**2)
            beam.ivarf[~np.isfinite(beam.ivarf)] = 0
            beam.ivar = beam.ivarf.reshape(beam.sh)
        
        self.ra, self.dec = self.beams[0].get_sky_coords()
        
        if MW_EBV < 0:
            ### Try to get MW_EBV from hsaquery.utils
            try:
                import hsaquery.utils
                MW_EBV = hsaquery.utils.get_irsa_dust(self.ra, self.dec)
            except:
                MW_EBV = 0
        
        self.MW_EBV = MW_EBV
        
        self._set_MW_EBV(MW_EBV)
        
        self._parse_beams(psf=psf)

        self.apply_trace_shift()

        self.Nphot = 0
        self.is_spec = 1
        
    def _set_MW_EBV(self, MW_EBV, R_V=utils.MW_RV):
        """
        Initialize Galactic extinction
        
        Parameters
        ----------
        MW_EBV : float
            Local E(B-V)
        
        R_V : float
            Relation between specific and total extinction, 
            ``a_v = r_v * ebv``.
        
        """
        for b in self.beams:
            beam = b.beam
            if beam.MW_EBV != MW_EBV:
                beam.MW_EBV = MW_EBV
                beam.init_galactic_extinction(MW_EBV, R_V=R_V)
                beam.process_config()
                b.flat_flam = b.compute_model(in_place=False, is_cgs=True)
                
    def _parse_beams(self, psf=False):
        
        self.N = len(self.beams)
        self.Ngrism = {}
        for i in range(self.N):
            if self.beams[i].grism.instrument == 'NIRISS':
                grism = self.beams[i].grism.pupil
            else:
                grism = self.beams[i].grism.filter
                
            if grism in self.Ngrism:
                self.Ngrism[grism] += 1
            else:
                self.Ngrism[grism] = 1
        
        self.grisms = list(self.Ngrism.keys())
         
        self.PA = {}
        for g in self.Ngrism:
            self.PA[g] = {}
            
        for i in range(self.N):
            if self.beams[i].grism.instrument == 'NIRISS':
                grism = self.beams[i].grism.pupil
            else:
                grism = self.beams[i].grism.filter

            PA = self.beams[i].get_dispersion_PA(decimals=0)
            if PA in self.PA[grism]:
                self.PA[grism][PA].append(i)
            else:
                self.PA[grism][PA] = [i]
            
        self.id = self.beams[0].id
        
        # Use WFC3 ePSF for the fit
        self.psf_param_dict = None
        if (psf > 0) & (self.beams[i].grism.instrument in ['WFC3', 'ACS']):

            self.psf_param_dict = OrderedDict()
            for ib, beam in enumerate(self.beams):
                if (beam.direct.data['REF'] is not None):
                    # Use REF extension.  scale factors might be wrong
                    beam.direct.data['SCI'] = beam.direct.data['REF'] 
                    new_err = np.ones_like(beam.direct.data['ERR'])
                    new_err *= utils.nmad(beam.direct.data['SCI'])
                    beam.direct.data['ERR'] = new_err
                    beam.direct.filter = beam.direct.ref_filter #'F160W'
                    beam.direct.photflam = beam.direct.ref_photflam
                
                beam.init_epsf(yoff=0.0, skip=psf*1, N=4, get_extended=True)
                #beam.compute_model = beam.compute_model_psf
                #beam.beam.compute_model = beam.beam.compute_model_psf
                beam.compute_model(use_psf=True)
                m = beam.compute_model(in_place=False)
                #beam.modelf = beam.model.flatten()
                #beam.model = beam.modelf.reshape(beam.beam.sh_beam)
                
                beam.flat_flam = beam.compute_model(in_place=False, is_cgs=True) #/self.beam.total_flux
                
                
                self.psf_param_dict[beam.grism.parent_file] = beam.beam.psf_params
        
        self._parse_beam_arrays()
        
    def _parse_beam_arrays(self):
        """
        """        
        self.poly_order = None
        
        self.shapes = [beam.model.shape for beam in self.beams]
        self.Nflat = [np.product(shape) for shape in self.shapes]
        self.Ntot = np.sum(self.Nflat)
        
        ### Big array of normalized wavelengths (wave / 1.e4 - 1)
        self.xpf = np.hstack([np.dot(np.ones((b.beam.sh_beam[0],1)),
                                    b.beam.lam[None,:]).flatten()/1.e4 
                              for b in self.beams]) - 1
        
        ### Flat-flambda model spectra
        self.flat_flam = np.hstack([b.flat_flam for b in self.beams])
        self.fit_mask = np.hstack([b.fit_mask*b.contam_mask 
                                      for b in self.beams])
                                               
        self.DoF = self.fit_mask.sum()
        self.ivarf = np.hstack([b.ivarf for b in self.beams])
        
        self.fit_mask &= (self.ivarf >= 0) 
        
        self.scif = np.hstack([b.scif for b in self.beams])

        #self.ivarf = 1./(1/self.ivarf + (self.sys_err*self.scif)**2)
        self.ivarf[~np.isfinite(self.ivarf)] = 0
        self.sivarf = np.sqrt(self.ivarf)

        self.wavef = np.hstack([b.wavef for b in self.beams])
        self.contamf = np.hstack([b.contam.flatten() for b in self.beams])
        
        self.weightf = np.exp(-(self.fcontam*np.abs(self.contamf)*np.sqrt(self.ivarf)))
        self.weightf[~np.isfinite(self.weightf)] = 0
        self.fit_mask *= self.weightf > 0
        
        self.DoF = int((self.weightf*self.fit_mask).sum())
        self.Nmask = np.sum([b.fit_mask.sum() for b in self.beams])
                
        ### Initialize background fit array
        # self.A_bg = np.zeros((self.N, self.Ntot))
        # i0 = 0
        # for i in range(self.N):
        #     self.A_bg[i, i0:i0+self.Nflat[i]] = 1.
        #     i0 += self.Nflat[i]
        
        self.slices = self._get_slices(masked=False)
        self.A_bg = self._init_background(masked=False)
        
        self._update_beam_mask()
        self.A_bgm = self._init_background(masked=True)
        
        self.init_poly_coeffs(poly_order=1)
        
        self.ra, self.dec = self.beams[0].get_sky_coords()
    
    def compute_exptime(self):
        exptime = {}
        for beam in self.beams:
            if beam.grism.instrument == 'NIRISS':
                grism = beam.grism.pupil
            else:
                grism = beam.grism.filter
            
            if grism in exptime:
                exptime[grism] += beam.grism.exptime
            else:
                exptime[grism] = beam.grism.exptime
        
        return exptime
        
    def extend(self, new, verbose=True):
        """Concatenate `~grizli.multifit.MultiBeam` objects
        
        Parameters
        ----------
        new : `~grizli.multifit.MultiBeam`
            Beam object containing new beams to add.
            
        verbose : bool
            Print summary of the change.
        
        """
        self.beams.extend(new.beams)
        self._parse_beams()
        if verbose:
            print('Add beams: {0}\n      Now: {1}'.format(new.Ngrism, self.Ngrism))
        
    def write_master_fits(self, verbose=True, get_hdu=False):
        """Store all beams in a single HDU
        TBD
        """ 
        hdu = pyfits.HDUList([pyfits.PrimaryHDU()])
        rd = self.beams[0].get_sky_coords()
        hdu[0].header['ID'] = (self.id, 'Object ID')
        hdu[0].header['RA'] = (rd[0], 'Right Ascension')
        hdu[0].header['DEC'] = (rd[1], 'Declination')
        
        exptime = {}
        for g in self.Ngrism:
            exptime[g] = 0.
         
        count = []
        for ib, beam in enumerate(self.beams):
            hdu_i = beam.write_fits(get_hdu=True, strip=True)
            hdu.extend(hdu_i[1:])
            count.append(len(hdu_i)-1)
            hdu[0].header['FILE{0:04d}'.format(ib)] = (beam.grism.parent_file, 'Grism parent file')
            hdu[0].header['GRIS{0:04d}'.format(ib)] = (beam.grism.filter, 'Grism element')
            hdu[0].header['NEXT{0:04d}'.format(ib)] = (count[-1], 'Number of extensions')
            
            try:
                exptime[beam.grism.filter] += beam.grism.header['EXPTIME']
            except:
                exptime[beam.grism.pupil] += beam.grism.header['EXPTIME']
                
        hdu[0].header['COUNT'] = (self.N, ' '.join(['{0}'.format(c) for c in count]))
        for g in self.Ngrism:
            hdu[0].header['T_{0}'.format(g)] = (exptime[g], 'Exposure time in grism {0}'.format(g))
            
        if get_hdu:
            return hdu
        
        outfile = '{0}_{1:05d}.beams.fits'.format(self.group_name, self.id)
        if verbose:
            print(outfile)
        
        hdu.writeto(outfile, clobber=True)
    
    def load_master_fits(self, beam_file, verbose=True):
        import copy
        
        try:
            utils.fetch_acs_wcs_files(beam_file)
        except:
            pass
            
        hdu = pyfits.open(beam_file, lazy_load_hdus=False)
        N = hdu[0].header['COUNT']
        Next = np.cast[int](hdu[0].header.comments['COUNT'].split())
        
        i0 = 1
        self.beams = []
        for i in range(N):
            key = 'NEXT{0:04d}'.format(i)
            if key in hdu[0].header:
                Next_i = hdu[0].header[key]
            else:
                Next_i = 6 # Assume doesn't have direct SCI/ERR cutouts
            
            # Testing for multiprocessing
            if True:
                hducopy = hdu[i0:i0+Next_i]
            else:
                #print('Copy!')
                hducopy = pyfits.HDUList([hdu[i].__class__(data=hdu[i].data*1, header=copy.deepcopy(hdu[i].header), name=hdu[i].name) for i in range(i0, i0+Next_i)])
            
            beam = model.BeamCutout(fits_file=hducopy, min_mask=self.min_mask,
                                    min_sens=self.min_sens) 
            
            self.beams.append(beam)
            if verbose:
                print('{0} {1} {2}'.format(i+1, beam.grism.parent_file, beam.grism.filter))
                
            i0 += Next_i #6#Next[i]
        
        hdu.close()
        
    def write_beam_fits(self, verbose=True):
        """TBD
        """
        outfiles = []
        for beam in self.beams:
            root = beam.grism.parent_file.split('.fits')[0]
            outfile = beam.write_fits(root)
            if verbose:
                print('Wrote {0}'.format(outfile))
            
            outfiles.append(outfile)
            
        return outfiles
    
    def load_beam_fits(self, beam_list, conf=None, verbose=True):
        """TBD
        """
        self.beams = []
        for file in beam_list:
            if verbose:
                print(file)
            
            beam = model.BeamCutout(fits_file=file, conf=conf, 
                                    min_mask=self.min_mask, 
                                    min_sens=self.min_sens)
            self.beams.append(beam)

    def reshape_flat(self, flat_array):
        """TBD
        """
        out = []
        i0 = 0
        for ib in range(self.N):
            im2d = flat_array[i0:i0+self.Nflat[ib]].reshape(self.shapes[ib])
            out.append(im2d)
            i0 += self.Nflat[ib]
        
        return out
                
    def init_poly_coeffs(self, flat=None, poly_order=1):
        """TBD
        """
        ### Already done?
        if poly_order < 0:
            ok_poly = False
            poly_order = 0
        else:
            ok_poly = True
            
        if poly_order == self.poly_order:
            return None
        
        self.poly_order = poly_order
        if flat is None:
            flat = self.flat_flam
                       
        ### Polynomial continuum arrays        
        self.A_poly = np.array([self.xpf**order*flat
                                      for order in range(poly_order+1)])
        
        self.A_poly *= ok_poly
        
        self.n_poly = poly_order + 1
        self.x_poly = np.array([(self.beams[0].beam.lam/1.e4-1)**order
                                      for order in range(poly_order+1)])
    
    def eval_poly_spec(self, coeffs_full):
        """Evaluate polynomial spectrum
        """
        xspec = np.arange(self.polyx[0], self.polyx[1], 0.05)-1
        i0 = self.N*self.fit_bg
        scale_coeffs = coeffs_full[i0:i0+self.n_poly]

        #yspec = [xspec**o*scale_coeffs[o] for o in range(self.poly_order+1)]
        yfull = np.polyval(scale_coeffs[::-1], xspec)
        return xspec, yfull
                                          
    def compute_model(self, id=None, spectrum_1d=None, is_cgs=False):
        """TBD
        """
        for beam in self.beams:
            beam.beam.compute_model(id=id, spectrum_1d=spectrum_1d, 
                                    is_cgs=is_cgs)
            
            beam.modelf = beam.beam.modelf 
            beam.model = beam.beam.modelf.reshape(beam.beam.sh_beam)
            
    def compute_model_psf(self, id=None, spectrum_1d=None, is_cgs=False):
        """TBD
        """
        for beam in self.beams:
            beam.beam.compute_model_psf(id=id, spectrum_1d=spectrum_1d, 
                                    is_cgs=is_cgs)
            
            beam.modelf = beam.beam.modelf 
            beam.model = beam.beam.modelf.reshape(beam.beam.sh_beam)
                                            
    def fit_at_z(self, z=0., templates={}, fitter='nnls',
                 fit_background=True, poly_order=0):
        """TBD
        """
        import sklearn.linear_model
        import numpy.linalg
        import scipy.optimize
        
        #print 'xxx Init poly'
        self.init_poly_coeffs(poly_order=poly_order)
        
        #print 'xxx Init bg'
        if fit_background:
            self.fit_bg = True
            A = np.vstack((self.A_bg, self.A_poly))
        else:
            self.fit_bg = False
            A = self.A_poly*1
        
        NTEMP = len(templates)
        A_temp = np.zeros((NTEMP, self.Ntot))
                  
        #print 'xxx Load templates'
        for i, key in enumerate(templates.keys()):
            NTEMP += 1
            temp = templates[key]#.zscale(z, 1.)
            spectrum_1d = [temp.wave*(1+z), temp.flux/(1+z)]
            
            if z > 4:
                try:
                    import eazy.igm
                    igm = eazy.igm.Inoue14()
                    igmz = igm.full_IGM(z, spectrum_1d[0])
                    spectrum_1d[1]*=igmz    
                    #print('IGM')            
                except:
                    # No IGM
                    pass
                  
            i0 = 0            
            for ib in range(self.N):
                beam = self.beams[ib]
                lam_beam = beam.beam.lam_beam
                if ((temp.wave.min()*(1+z) > lam_beam.max()) | 
                    (temp.wave.max()*(1+z) < lam_beam.min())):
                    tmodel = 0.
                else:
                    tmodel = beam.compute_model(spectrum_1d=spectrum_1d, 
                                                in_place=False, is_cgs=True) #/beam.beam.total_flux
                
                A_temp[i, i0:i0+self.Nflat[ib]] = tmodel#.flatten()
                i0 += self.Nflat[ib]
                        
        if NTEMP > 0:
            A = np.vstack((A, A_temp))
        
        ok_temp = np.sum(A, axis=1) > 0  
        out_coeffs = np.zeros(A.shape[0])
        
        ### LSTSQ coefficients
        #print 'xxx Fitter'
        fit_functions = {'lstsq':np.linalg.lstsq, 'nnls':scipy.optimize.nnls}
        
        if fitter in fit_functions:
            #'lstsq':
            
            Ax = A[:, self.fit_mask][ok_temp,:].T
            ### Weight by ivar
            Ax *= np.sqrt(self.ivarf[self.fit_mask][:, np.newaxis])
            
            #print 'xxx lstsq'
            #out = numpy.linalg.lstsq(Ax,y)
            if fitter == 'lstsq':
                y = self.scif[self.fit_mask]
                ### Weight by ivar
                y *= np.sqrt(self.ivarf[self.fit_mask])

                try:
                    out = np.linalg.lstsq(Ax,y, rcond=None)                      
                except:
                    print(A.min(), Ax.min(), self.fit_mask.sum(), y.min())
                    raise ValueError
                    
                lstsq_coeff, residuals, rank, s = out
                coeffs = lstsq_coeff
            
            if fitter == 'nnls':
                if fit_background:
                    off = 0.04
                    y = self.scif[self.fit_mask]+off
                    y *= np.sqrt(self.ivarf[self.fit_mask])

                    coeffs, rnorm = scipy.optimize.nnls(Ax, y+off)
                    coeffs[:self.N] -= 0.04
                else:
                    y = self.scif[self.fit_mask]
                    y *= np.sqrt(self.ivarf[self.fit_mask])
                    
                    coeffs, rnorm = scipy.optimize.nnls(Ax, y)  
            
            # if fitter == 'bounded':
            #     if fit_background:
            #         off = 0.04
            #         y = self.scif[self.fit_mask]+off
            #         y *= self.ivarf[self.fit_mask]
            # 
            #         coeffs, rnorm = scipy.optimize.nnls(Ax, y+off)
            #         coeffs[:self.N] -= 0.04
            #     else:
            #         y = self.scif[self.fit_mask]
            #         y *= np.sqrt(self.ivarf[self.fit_mask])
            #         
            #         coeffs, rnorm = scipy.optimize.nnls(Ax, y)  
            #     
            #     out = scipy.optimize.minimize(self.eval_trace_shift, shifts, bounds=bounds, args=args, method='Powell', tol=tol)
                                  
        else:
            Ax = A[:, self.fit_mask][ok_temp,:].T
            y = self.scif[self.fit_mask]
            
            ### Wieght by ivar
            Ax *= np.sqrt(self.ivarf[self.fit_mask][:, np.newaxis])
            y *= np.sqrt(self.ivarf[self.fit_mask])
            
            clf = sklearn.linear_model.LinearRegression()
            status = clf.fit(Ax, y)
            coeffs = clf.coef_
                
        out_coeffs[ok_temp] = coeffs
        modelf = np.dot(out_coeffs, A)
        chi2 = np.sum((self.weightf*(self.scif - modelf)**2*self.ivarf)[self.fit_mask])
        
        if fit_background:
            poly_coeffs = out_coeffs[self.N:self.N+self.n_poly]
        else:
            poly_coeffs = out_coeffs[:self.n_poly]
            
        self.y_poly = np.dot(poly_coeffs, self.x_poly)
        # x_poly = self.x_poly[1,:]+1 = self.beams[0].beam.lam/1.e4
        
        return A, out_coeffs, chi2, modelf
    
    def parse_fit_outputs(self, z, templates, coeffs_full, A):
        """Parse output from `fit_at_z`.

        Parameters
        ----------
        z : float
            Redshift at which to evaluate the fits.
        
        templates : list of `~grizli.utils.SpectrumTemplate` objects
            Generated with, e.g., `~grizli.utils.load_templates`.
        
        coeffs_full : `~np.ndarray`
            Template fit coefficients
        
        A : `~np.ndarray`
            Matrix generated for fits and used for computing model 2D spectra:
                
                >>> model_flat = np.dot(coeffs_full, A)
                >>> # mb = MultiBeam(...)
                >>> all_models = mb.reshape_flat(model_flat)
                >>> m0 = all_models[0] # model for mb.beams[0]
        
        Returns
        -------        
        line_flux : dict
            Line fluxes and uncertainties, in cgs units (erg/s/cm2)
        
        covar : `~np.ndarray`
            Covariance matrix for the fit coefficients
        
        cont1d, line1d, model1d : `~grizli.utils.SpectrumTemplate`
            Best-fit continuum, line, and full (continuum + line) templates
            
        model_continuum : `~np.ndarray`
            Flat array of the best fit 2D continuum
        
        """
        from collections import OrderedDict

        ## Covariance matrix for line flux uncertainties
        Ax = A[:,self.fit_mask]
        ok_temp = (np.sum(Ax, axis=1) > 0) & (coeffs_full != 0)
        Ax = Ax[ok_temp,:].T*1 #A[:, self.fit_mask][ok_temp,:].T
        Ax *= np.sqrt(self.ivarf[self.fit_mask][:, np.newaxis])
        try:
            covar = np.matrix(np.dot(Ax.T, Ax)).I
            covard = np.sqrt(covar.diagonal())
        except:
            N = ok_temp.sum()
            covar = np.zeros((N,N))
            covard = np.zeros(N)#-1.
        
        covar_full = utils.fill_masked_covar(covar, ok_temp)
        
        ## Random draws from covariance matrix
        # draws = np.random.multivariate_normal(coeffs_full[ok_temp], covar, size=500)
                    
        line_flux_err = coeffs_full*0.
        line_flux_err[ok_temp] = covard

        ## Continuum fit
        mask = np.isfinite(coeffs_full)
        for i, key in enumerate(templates.keys()):
            if key.startswith('line'):
                mask[self.N*self.fit_bg+self.n_poly+i] = False

        model_continuum = np.dot(coeffs_full*mask, A)
        self.model_continuum = self.reshape_flat(model_continuum)
        #model_continuum.reshape(self.beam.sh_beam)

        ### 1D spectrum
        
        # Polynomial component
        xspec, yspec = self.eval_poly_spec(coeffs_full)
        
        model1d = utils.SpectrumTemplate((xspec+1)*1.e4, yspec)

        cont1d = model1d*1

        i0 = self.fit_bg*self.N + self.n_poly

        line_flux = OrderedDict()
        fscl = 1. #self.beams[0].beam.total_flux/1.e-17
        line1d = OrderedDict()
        for i, key in enumerate(templates.keys()):
            temp_i = templates[key].zscale(z, coeffs_full[i0+i])
            model1d += temp_i
            if not key.startswith('line'):
                cont1d += temp_i
            else:
                line1d[key.split()[1]] = temp_i
                line_flux[key.split()[1]] = np.array([coeffs_full[i0+i]*fscl, 
                                             line_flux_err[i0+i]*fscl])

        return line_flux, covar_full, cont1d, line1d, model1d, model_continuum
        
    def fit_stars(self, poly_order=1, fitter='nnls', fit_background=True, 
                  verbose=True, make_figure=True, zoom=None,
                  delta_chi2_threshold=0.004, zr=0, dz=0, fwhm=0, 
                  prior=None, templates={}, figsize=[8,5],
                  fsps_templates=False):
        """TBD
        """
        
        ## Polynomial fit
        out = self.fit_at_z(z=0., templates={}, fitter='lstsq',
                            poly_order=3,
                            fit_background=fit_background)
        
        A, coeffs, chi2_poly, model_2d = out
        
        ### Star templates
        templates = utils.load_templates(fwhm=fwhm, stars=True)
        NTEMP = len(templates)

        key = list(templates)[0]
        temp_i = {key:templates[key]}
        out = self.fit_at_z(z=0., templates=temp_i, fitter=fitter,
                            poly_order=poly_order,
                            fit_background=fit_background)
                            
        A, coeffs, chi2, model_2d = out
        
        chi2 = np.zeros(NTEMP)
        coeffs = np.zeros((NTEMP, coeffs.shape[0]))
        
        chi2min = 1e30
        iz = 0
        best = key
        for i, key in enumerate(list(templates)):
            temp_i = {key:templates[key]}
            out = self.fit_at_z(z=0., templates=temp_i,
                                fitter=fitter, poly_order=poly_order,
                                fit_background=fit_background)
            
            A, coeffs[i,:], chi2[i], model_2d = out
            if chi2[i] < chi2min:
                iz = i
                chi2min = chi2[i]
                best = key

            if verbose:                    
                print(utils.NO_NEWLINE + '  {0} {1:9.1f} ({2})'.format(key, chi2[i], best))
        
        ## Best-fit
        temp_i = {best:templates[best]}
        out = self.fit_at_z(z=0., templates=temp_i,
                            fitter=fitter, poly_order=poly_order,
                            fit_background=fit_background)
        
        A, coeffs_full, chi2_best, model_full = out
        
        ## Continuum fit
        mask = np.isfinite(coeffs_full)
        for i, key in enumerate(templates.keys()):
            if key.startswith('line'):
                mask[self.N*self.fit_bg+self.n_poly+i] = False
            
        model_continuum = np.dot(coeffs_full*mask, A)
        self.model_continuum = self.reshape_flat(model_continuum)
        #model_continuum.reshape(self.beam.sh_beam)
                
        ### 1D spectrum
        # xspec = np.arange(0.3, 2.35, 0.05)-1
        # scale_coeffs = coeffs_full[self.N*self.fit_bg:  
        #                           self.N*self.fit_bg+self.n_poly]
        #                           
        # yspec = [xspec**o*scale_coeffs[o] for o in range(self.poly_order+1)]
        xspec, yspec = self.eval_poly_spec(coeffs_full)
        model1d = utils.SpectrumTemplate((xspec+1)*1.e4, yspec)

        cont1d = model1d*1
        
        i0 = self.fit_bg*self.N + self.n_poly
        
        line_flux = OrderedDict()
        fscl = 1. #self.beams[0].beam.total_flux/1.e-17

        temp_i = templates[best].zscale(0, coeffs_full[i0])
        model1d += temp_i
        cont1d += temp_i
        
        fit_data = OrderedDict()
        fit_data['poly_order'] = poly_order
        fit_data['fwhm'] = 0
        fit_data['zbest'] = np.argmin(chi2)
        fit_data['chibest'] = chi2_best
        fit_data['chi_poly'] = chi2_poly
        fit_data['zgrid'] = np.arange(NTEMP)
        fit_data['prior'] = 1
        fit_data['A'] = A
        fit_data['coeffs'] = coeffs
        fit_data['chi2'] = chi2
        fit_data['DoF'] = self.DoF
        fit_data['model_full'] = model_full
        fit_data['coeffs_full'] = coeffs_full
        fit_data['line_flux'] = {}
        #fit_data['templates_full'] = templates
        fit_data['model_cont'] = model_continuum
        fit_data['model1d'] = model1d
        fit_data['cont1d'] = cont1d
        
        #return fit_data
        
        fig = None   
        if make_figure:
            fig = self.show_redshift_fit(fit_data)
            #fig.savefig('fit.pdf')
            
        return fit_data, fig
        
        
    def fit_redshift(self, prior=None, poly_order=1, fwhm=1200,
                     make_figure=True, zr=None, dz=None, verbose=True,
                     fit_background=True, fitter='nnls', 
                     delta_chi2_threshold=0.004, zoom=True, 
                     line_complexes=True, templates={}, figsize=[8,5],
                     fsps_templates=False):
        """TBD
        """
        from scipy import polyfit, polyval
        
        if zr is None:
            zr = [0.65, 1.6]
        
        if dz is None:
            dz = [0.005, 0.0004]
        
        # if True:
        #     beams = grp.get_beams(id, size=30)
        #     mb = grizlidev.multifit.MultiBeam(beams)
        #     self = mb
                    
        if zr is 0:
            stars = True
            zr = [0, 0.01]
            fitter='nnls'
        else:
            stars = False
            
        zgrid = utils.log_zgrid(zr, dz=dz[0])
        NZ = len(zgrid)

        ## Polynomial fit
        out = self.fit_at_z(z=0., templates={}, fitter='lstsq',
                            poly_order=3,
                            fit_background=fit_background)
        
        A, coeffs, chi2_poly, model_2d = out
        
        ### Set up for template fit
        if templates == {}:
            templates = utils.load_templates(fwhm=fwhm, stars=stars, line_complexes=line_complexes, fsps_templates=fsps_templates)
        else:
            if verbose:
                print('User templates! N={0} \n'.format(len(templates)))
            
        NTEMP = len(templates)
        
        out = self.fit_at_z(z=0., templates=templates, fitter=fitter,
                            poly_order=poly_order,
                            fit_background=fit_background)
                            
        A, coeffs, chi2, model_2d = out
        
        chi2 = np.zeros(NZ)
        coeffs = np.zeros((NZ, coeffs.shape[0]))
        
        chi2min = 1e30
        iz = 0
        for i in range(NZ):
            out = self.fit_at_z(z=zgrid[i], templates=templates,
                                fitter=fitter, poly_order=poly_order,
                                fit_background=fit_background)
            
            A, coeffs[i,:], chi2[i], model_2d = out
            if chi2[i] < chi2min:
                iz = i
                chi2min = chi2[i]

            if verbose:                    
                print(utils.NO_NEWLINE + '  {0:.4f} {1:9.1f} ({2:.4f})'.format(zgrid[i], chi2[i], zgrid[iz]))
        
        print('First iteration: z_best={0:.4f}\n'.format(zgrid[iz]))
            
        # peaks
        import peakutils
        # chi2nu = (chi2.min()-chi2)/self.DoF
        # indexes = peakutils.indexes((chi2nu+delta_chi2_threshold)*(chi2nu > -delta_chi2_threshold), thres=0.3, min_dist=20)
        
        chi2_rev = (chi2_poly - chi2)/self.DoF
        if chi2_poly < (chi2.min() + 9):
            chi2_rev = (chi2.min() + 16 - chi2)/self.DoF

        chi2_rev[chi2_rev < 0] = 0
        indexes = peakutils.indexes(chi2_rev, thres=0.4, min_dist=8)
        num_peaks = len(indexes)
        
        if False:
            plt.plot(zgrid, (chi2-chi2.min())/ self.DoF)
            plt.scatter(zgrid[indexes], (chi2-chi2.min())[indexes]/ self.DoF, color='r')
        
        # delta_chi2 = (chi2.max()-chi2.min())/self.DoF
        # if delta_chi2 > delta_chi2_threshold:      
        if (num_peaks > 0) & (not stars) & zoom:
            zgrid_zoom = []
            for ix in indexes:
                if (ix > 0) & (ix < len(chi2)-1):
                    c = polyfit(zgrid[ix-1:ix+2], chi2[ix-1:ix+2], 2)
                    zi = -c[1]/(2*c[0])
                    chi_i = polyval(c, zi)
                    zgrid_zoom.extend(np.arange(zi-2*dz[0], 
                                      zi+2*dz[0]+dz[1]/10., dz[1]))
                    
            # zgrid_zoom = utils.zoom_zgrid(zgrid, chi2/self.DoF,
            #                               threshold=delta_chi2_threshold,
            #                               factor=dz[0]/dz[1])
            NZOOM = len(zgrid_zoom)
        
            chi2_zoom = np.zeros(NZOOM)
            coeffs_zoom = np.zeros((NZOOM, coeffs.shape[1]))

            iz = 0
            chi2min = 1.e30
            for i in range(NZOOM):
                out = self.fit_at_z(z=zgrid_zoom[i], templates=templates,
                                    fitter=fitter, poly_order=poly_order,
                                    fit_background=fit_background)

                A, coeffs_zoom[i,:], chi2_zoom[i], model_2d = out
                if chi2_zoom[i] < chi2min:
                    chi2min = chi2_zoom[i]
                    iz = i
                
                if verbose:
                    print(utils.NO_NEWLINE+'- {0:.4f} {1:9.1f} ({2:.4f}) {3:d}/{4:d}'.format(zgrid_zoom[i], chi2_zoom[i], zgrid_zoom[iz], i+1, NZOOM))
        
            zgrid = np.append(zgrid, zgrid_zoom)
            chi2 = np.append(chi2, chi2_zoom)
            coeffs = np.append(coeffs, coeffs_zoom, axis=0)
        
        so = np.argsort(zgrid)
        zgrid = zgrid[so]
        chi2 = chi2[so]
        coeffs=coeffs[so,:]

        if prior is not None:
            #print('\n\nPrior!\n\n', chi2.min(), prior[1].min())
            interp_prior = np.interp(zgrid, prior[0], prior[1])
            chi2 += interp_prior
        else:
            interp_prior = None
            
        print(' Zoom iteration: z_best={0:.4f}\n'.format(zgrid[np.argmin(chi2)]))
        
        ### Best redshift
        if not stars:
            templates = utils.load_templates(line_complexes=False, fwhm=fwhm, fsps_templates=fsps_templates)
        
        zbest = zgrid[np.argmin(chi2)]
        ix = np.argmin(chi2)
        chibest = chi2.min()
        
        ## Fit parabola
        if (ix > 0) & (ix < len(chi2)-1):
            c = polyfit(zgrid[ix-1:ix+2], chi2[ix-1:ix+2], 2)
            zbest = -c[1]/(2*c[0])
            chibest = polyval(c, zbest)
        
        out = self.fit_at_z(z=zbest, templates=templates,
                            fitter=fitter, poly_order=poly_order, 
                            fit_background=fit_background)
        
        A, coeffs_full, chi2_best, model_full = out
        
        # Parse results
        out2 = self.parse_fit_outputs(zbest, templates, coeffs_full, A)
        line_flux, covar, cont1d, line1d, model1d, model_continuum = out2
        
        # Output dictionary with fit parameters
        fit_data = OrderedDict()
        fit_data['poly_order'] = poly_order
        fit_data['fwhm'] = fwhm
        fit_data['zbest'] = zbest
        fit_data['chibest'] = chibest
        fit_data['chi_poly'] = chi2_poly
        fit_data['zgrid'] = zgrid
        fit_data['prior'] = interp_prior
        fit_data['A'] = A
        fit_data['coeffs'] = coeffs
        fit_data['chi2'] = chi2
        fit_data['DoF'] = self.DoF
        fit_data['model_full'] = model_full
        fit_data['coeffs_full'] = coeffs_full
        fit_data['covar'] = covar
        fit_data['line_flux'] = line_flux
        #fit_data['templates_full'] = templates
        fit_data['model_cont'] = model_continuum
        fit_data['model1d'] = model1d
        fit_data['cont1d'] = cont1d
        fit_data['line1d'] = line1d
        
        #return fit_data
        
        fig = None   
        if make_figure:
            fig = self.show_redshift_fit(fit_data, figsize=figsize)
            #fig.savefig('fit.pdf')
            
        return fit_data, fig
    
    def run_individual_fits(self, z=0, templates={}):
        """Run template fits on each *exposure* individually to evaluate
        variance in line and continuum fits.

        Parameters
        ----------
        z : float
            Redshift at which to evaluate the fit

        templates : list of `~grizli.utils.SpectrumTemplate` objects
            Generated with, e.g., `load_templates`.

        Returns
        -------        
        line_flux, line_err : dict
            Dictionaries with the measured line fluxes and uncertainties for
            each exposure fit.

        coeffs_list : `~np.ndarray` [Nbeam x Ntemplate]
            Raw fit coefficients

        chi2_list, DoF_list : `~np.ndarray` [Nbeam]
            Chi-squared and effective degrees of freedom for each separate fit

        """  
        
        # Fit on the full set of beams      
        out = self.fit_at_z(z=z, templates=templates,
                            fitter='nnls', poly_order=self.poly_order, 
                            fit_background=self.fit_bg)

        A, coeffs_full, chi2_best, model_full = out

        out2 = self.parse_fit_outputs(z, templates, coeffs_full, A)
        line, covar, cont1d, line1d, model1d, model_continuum = out2

        NB, NTEMP = len(self.beams), len(templates)
        
        # Outputs
        coeffs_list = np.zeros((NB, NTEMP))
        chi2_list = np.zeros(NB)
        DoF_list = np.zeros(NB)

        line_flux = OrderedDict()
        line_err = OrderedDict()
        line_keys = list(line.keys())

        for k in line_keys:
            line_flux[k] = np.zeros(NB)
            line_err[k] = np.zeros(NB)
        
        # Generate separate MultiBeam objects for each individual beam
        for i, b in enumerate(self.beams):
            b_i = MultiBeam([b], fcontam=self.fcontam,
                            group_name=self.group_name)

            out_i = b_i.fit_at_z(z=z, templates=templates,
                                fitter='nnls', poly_order=self.poly_order, 
                                fit_background=self.fit_bg)

            A_i, coeffs_i, chi2_i, model_full_i = out_i
            
            # Parse fit information from individual fits
            out2 = b_i.parse_fit_outputs(z, templates, coeffs_i, A_i)
            line_i, covar_i, cont1d_i, line1d_i, model1d_i, model_continuum_i = out2

            for k in line_keys:
                line_flux[k][i] = line_i[k][0]
                line_err[k][i] = line_i[k][1]

            coeffs_list[i,:] = coeffs_i[-NTEMP:]
            chi2_list[i] = chi2_i
            DoF_list[i] = b_i.DoF

        return line_flux, line_err, coeffs_list, chi2_list, DoF_list
    
    def show_redshift_fit(self, fit_data, plot_flambda=True, figsize=[8,5]):
        """TBD
        """
        import matplotlib.gridspec
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[0.6,1])
        
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(gs[0])
        
        c2min = fit_data['chi2'].min()
               
        scale_pz = True
        if scale_pz:
            scale_nu = c2min/self.DoF
            scl_label = '_s'
        else:
            scale_nu = 1.
            scl_label = ''
            
        #axz.plot(z, (chi2-chi2.min())/scale_nu, color='k')
        #ax.plot(fit_data['zgrid'], fit_data['chi2']/self.DoF)
        ax.plot(fit_data['zgrid'], (fit_data['chi2']-c2min)/scale_nu)
        
        ax.set_xlabel('z')
        ax.set_ylabel(r'$\chi^2_\nu$, $\nu$={0:d}'.format(self.DoF))
        
        ax.set_ylim(-4,27)
        ax.set_ylabel(r'$\Delta\chi^2{2}$ ({0:.0f}/$\nu$={1:d})'.format(c2min, self.DoF, scl_label))
        ax.set_yticks([1,4,9,16,25])
        
        # for delta in [1,4,9]:
        #     ax.plot(fit_data['zgrid'],
        #             fit_data['zgrid']*0.+(c2min+delta)/self.DoF, 
        #             color='{0:.2f}'.format(1-delta*1./10))
        
        ax.plot(fit_data['zgrid'], (fit_data['chi2']*0+fit_data['chi_poly']-c2min)/scale_nu, color='b', linestyle='--', alpha=0.8)
        
        ax.set_xlim(fit_data['zgrid'].min(), fit_data['zgrid'].max())
        ax.grid()        
        ax.set_title(r'ID = {0:d}, $z_\mathrm{{grism}}$={1:.4f}'.format(self.beams[0].id, fit_data['zbest']))
                                               
        ax = fig.add_subplot(gs[1])
        
        ymax = 0
        ymin = 1e10
        
        continuum_fit = self.reshape_flat(fit_data['model_cont'])
        line_fit = self.reshape_flat(fit_data['model_full'])
        
        grisms = self.Ngrism.keys()
        wfull = {}
        ffull = {}
        efull = {}
        for grism in grisms:
            wfull[grism] = []
            ffull[grism] = []
            efull[grism] = []
        
        for ib in range(self.N):
            beam = self.beams[ib]
            
            clean = beam.grism['SCI'] - beam.contam 
            if self.fit_bg:
                bg_i = fit_data['coeffs_full'][ib]
                clean -= bg_i # background
            else:
                bg_i = 0.
            
            #ivar = 1./(1./beam.ivar + self.fcontam*beam.contam)
            #ivar[~np.isfinite(ivar)] = 0
            ## New weight scheme
            ivar = beam.ivar
            weight = np.exp(-(self.fcontam*np.abs(beam.contam)*np.sqrt(ivar)))
            
            wave, flux, err = beam.beam.optimal_extract(clean, 
                                                        ivar=ivar,
                                                        weight=weight)
            
            mwave, mflux, merr = beam.beam.optimal_extract(line_fit[ib]-bg_i, 
                                                        ivar=ivar,
                                                        weight=weight)
            
            flat = beam.flat_flam.reshape(beam.beam.sh_beam)
            wave, fflux, ferr = beam.beam.optimal_extract(flat, ivar=ivar,
                                                          weight=weight)
                                         
            if plot_flambda:
                ok = beam.beam.sensitivity > 0.1*beam.beam.sensitivity.max()

                wave = wave[ok]
                fscl = 1./1.e-19 #beam.beam.total_flux/1.e-17
                flux  = (flux*fscl/fflux)[ok]*beam.beam.scale
                err   = (err*fscl/fflux)[ok]
                mflux = (mflux*fscl/fflux)[ok]*beam.beam.scale
                
                ylabel = r'$f_\lambda\,/\,10^{-19}\,\mathrm{cgs}$'
            else:
                ylabel = 'flux (e-/s)'
            
            scl_region = np.isfinite(mflux) 
            if scl_region.sum() == 0:
                continue
                
            # try:
            #     okerr = np.isfinite(err) #& (np.abs(flux/err) > 0.2) & (err != 0)
            #     med_err = np.median(err[okerr])
            #     
            #     ymax = np.maximum(ymax, 
            #                 (mflux[scl_region][2:-2] + med_err).max())
            #     ymin = np.minimum(ymin, 
            #                 (mflux[scl_region][2:-2] - med_err).min())
            # except:
            #     continue
            
            #okerr = (err != 0) & (np.abs(flux/err) > 0.2)
            okerr = np.isfinite(err)
            ax.errorbar(wave[okerr]/1.e4, flux[okerr], err[okerr], alpha=0.15+0.2*(self.N <= 2), linestyle='None', marker='.', color='{0:.2f}'.format(ib*0.5/self.N), zorder=1)
            ax.plot(wave[okerr]/1.e4, mflux[okerr], color='r', alpha=0.5, zorder=3)
            
            if beam.grism.instrument == 'NIRISS':
                grism = beam.grism.pupil
            else:
                grism = beam.grism.filter
                
            #for grism in grisms:
            wfull[grism] = np.append(wfull[grism], wave[okerr])
            ffull[grism] = np.append(ffull[grism], flux[okerr])
            efull[grism] = np.append(efull[grism], err[okerr])
            
            ## Scatter direct image flux
            if beam.direct.ref_photplam is None:
                ax.scatter(beam.direct.photplam/1.e4, beam.beam.total_flux/1.e-19, marker='s', edgecolor='k', color=GRISM_COLORS[grism], alpha=0.2, zorder=100, s=100)
            else:
                ax.scatter(beam.direct.ref_photplam/1.e4, beam.beam.total_flux/1.e-19, marker='s', edgecolor='k', color=GRISM_COLORS[grism], alpha=0.2, zorder=100, s=100)
                
        for grism in grisms:                        
            if self.Ngrism[grism] > 1:
                ## binned
                okb = (np.isfinite(wfull[grism]) & np.isfinite(ffull[grism]) &
                                   np.isfinite(efull[grism]))
                                   
                so = np.argsort(wfull[grism][okb])
                var = efull[grism]**2
            
                N = int(np.ceil(self.Ngrism[grism]/2)*2)*2
                kernel = np.ones(N, dtype=float)/N
                wht = 1/var[okb][so]
                fbin = nd.convolve(ffull[grism][okb][so]*wht, kernel)[N//2::N]
                wbin = nd.convolve(wfull[grism][okb][so]*wht, kernel)[N//2::N]
                #vbin = nd.convolve(var[okb][so], kernel**2)[N//2::N]
                wht_bin = nd.convolve(wht, kernel)[N//2::N]
                vbin = nd.convolve(wht, kernel**2)[N//2::N]/wht_bin**2
                
                fbin /= wht_bin
                wbin /= wht_bin
                #vbin = 1./wht_bin
                
                ax.errorbar(wbin/1.e4, fbin, np.sqrt(vbin), alpha=0.8,
                            linestyle='None', marker='.', 
                            color=GRISM_COLORS[grism], zorder=2)
                
                med_err = np.median(np.sqrt(vbin))
                ymin = np.minimum(ymin, (fbin-2*med_err).min())
                ymax = np.maximum(ymax, (fbin+2*med_err).max())
        
        ymin = np.maximum(0, ymin)        
        ax.set_ylim(ymin - 0.2*np.abs(ymax), 1.3*ymax)
        
        xmin, xmax = 1.e5, 0        
        for g in GRISM_LIMITS:
            if g in grisms:
                xmin = np.minimum(xmin, GRISM_LIMITS[g][0])
                xmax = np.maximum(xmax, GRISM_LIMITS[g][1])
                #print g, xmin, xmax
                
        ax.set_xlim(xmin, xmax)
        ax.semilogx(subsx=[xmax])
        #axc.set_xticklabels([])
        #axc.set_xlabel(r'$\lambda$')
        #axc.set_ylabel(r'$f_\lambda \times 10^{-19}$')
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        
        labels = np.arange(np.ceil(xmin*10), np.ceil(xmax*10))/10.
        ax.set_xticks(labels)
        ax.set_xticklabels(labels)
        
        ax.grid()
        
        ### Label
        ax.text(0.03, 1.03, ('{0}'.format(self.Ngrism)).replace('\'','').replace('{','').replace('}',''), ha='left', va='bottom', transform=ax.transAxes, fontsize=10)
        
        #ax.plot(wave/1.e4, wave/1.e4*0., linestyle='--', color='k')
        ax.hlines(0, xmin, xmax, linestyle='--', color='k')
        
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(ylabel)
        
        gs.tight_layout(fig, pad=0.1)
        return fig
    
    def redshift_fit_twod_figure(self, fit, spatial_scale=1, dlam=46., NY=10,
                                 figsize=[8,3.5], **kwargs):
        """Make figure of 2D spectrum
        
        TBD
        """        
        ### xlimits        
        xmin, xmax = 1.e5, 0
                   
        for g in GRISM_LIMITS:
            if g in self.Ngrism:
                xmin = np.minimum(xmin, GRISM_LIMITS[g][0])
                xmax = np.maximum(xmax, GRISM_LIMITS[g][1])
        
        hdu_sci = drizzle_2d_spectrum(self.beams, ds9=None, NY=NY,
                                      spatial_scale=spatial_scale, dlam=dlam, 
                                      kernel='point', pixfrac=0.6,
                                      wlimit=[xmin, xmax], 
                                      fcontam=self.fcontam)
                                  
        ### Continuum model
        cont = self.reshape_flat(fit['model_cont'])        
        hdu_con = drizzle_2d_spectrum(self.beams, data=cont, ds9=None, NY=NY,
                                      spatial_scale=spatial_scale, dlam=dlam, 
                                      kernel='point', pixfrac=0.6,
                                      wlimit=[xmin, xmax], 
                                      fcontam=self.fcontam)
        
        full = self.reshape_flat(fit['model_full'])        
        hdu_full = drizzle_2d_spectrum(self.beams, data=full, ds9=None, NY=NY,
                                      spatial_scale=spatial_scale, dlam=dlam, 
                                      kernel='point', pixfrac=0.6,
                                      wlimit=[xmin, xmax],
                                      fcontam=self.fcontam)
        
        clip = hdu_full['WHT'].data > np.percentile(hdu_full['WHT'].data, 30)
        #vmax = np.maximum(1.1*np.percentile(hdu_full['SCI'].data[clip], 98), 0.04)
        avg_rms = 1/np.median(np.sqrt(hdu_full['WHT'].data[clip]))
        vmax = np.maximum(1.1*np.percentile(hdu_full['SCI'].data[clip], 98), 5*avg_rms)
        
        #print 'VMAX: %f\n\n' %vmax
        
        sh = hdu_full[1].data.shape
        extent = [hdu_full[0].header['WMIN'], hdu_full[0].header['WMAX'],
                  0, sh[0]]
                  
        fig = plt.figure(figsize=figsize)
        show = [hdu_sci[1].data, hdu_full[1].data,
                hdu_sci[1].data-hdu_con[1].data]
        
        desc = [r'$Contam$'+'\n'+r'$Cleaned$', r'$Model$', r'$Line$'+'\n'+r'$Residual$']
        
        i=0
        for data_i, desc_i in zip(show, desc):
            ax = fig.add_subplot(11+i+100*len(show))        
            ax.imshow(data_i, origin='lower',
                      interpolation='Nearest', vmin=-0.1*vmax, vmax=vmax, 
                      extent=extent, cmap = plt.cm.viridis_r, 
                      aspect='auto')
            
            ax.set_yticklabels([])
            ax.set_ylabel(desc_i)
            
            i+=1
            
        for ax in fig.axes[:-1]:
            ax.set_xticklabels([])
            
        fig.axes[-1].set_xlabel(r'$\lambda$')
        
        fig.tight_layout(pad=0.2)

        ## Label
        label = 'ID={0:6d}, z={1:.4f}'.format(self.beams[0].id, fit['zbest'])
        fig.axes[-1].text(0.97, -0.27, label, ha='right', va='top',
                          transform=fig.axes[-1].transAxes, fontsize=10)
        
        label2 = ('{0}'.format(self.Ngrism)).replace('\'', '').replace('{', '').replace('}', '')
        fig.axes[-1].text(0.03, -0.27, label2, ha='left', va='top',
                          transform=fig.axes[-1].transAxes, fontsize=10)
                
        hdu_sci.append(hdu_con[1])
        hdu_sci[-1].name = 'CONTINUUM'
        hdu_sci.append(hdu_full[1])
        hdu_sci[-1].name = 'FULL'
        
        return fig, hdu_sci
    
    def drizzle_segmentation(self, wcsobj=None, kernel='square', pixfrac=1, verbose=False):
        """
        Drizzle segmentation image from individual `MultiBeam.beams`.
        
        Parameters
        ----------
        wcsobj: `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
            Output WCS.
        
        kernel: e.g., 'square', 'point', 'gaussian'
            Drizzle kernel, see `~drizzlepac.adrizzle.drizzle`.
        
        pixfrac: float
            Drizzle 'pixfrac', see `~drizzlepac.adrizzle.drizzle`.
        
        verbose: bool
            Print status messages.
                        
        Returns
        ----------
        drizzled_segm: `~numpy.ndarray`, type `~numpy.int64`.
            Drizzled segmentation image, with image dimensions and 
            WCS defined in `wcsobj`.
            
        """
        import numpy as np

        import astropy.wcs as pywcs
        import astropy.io.fits as pyfits
        
        try:
            from . import utils
        except:
            from grizli import multifit, utils
        
        all_ids = [np.unique(beam.beam.seg) for beam in self.beams]
        all_ids = np.unique(np.hstack(all_ids))[1:]
        
        if isinstance(wcsobj, pyfits.Header):
            wcs = pywcs.WCS(wcsobj)
            wcs.pscale = utils.get_wcs_pscale(wcs)
        else:
            wcs = wcsobj
        
        if not hasattr(wcs, 'pscale'):
            wcs.pscale = utils.get_wcs_pscale(wcs)
        
        if verbose:
            print('Drizzle ID={0:.0f} (primary)'.format(self.id))
                
        drizzled_segm = self.drizzle_segmentation_id(id=self.id, wcsobj=wcsobj, kernel=kernel, pixfrac=pixfrac, verbose=verbose)
        
        for id in all_ids:
            if int(id) == self.id:
                continue

            if verbose:
                print('Drizzle ID={0:.0f}'.format(id))
            
            dseg_i = self.drizzle_segmentation_id(id=id, wcsobj=wcsobj, kernel=kernel, pixfrac=pixfrac, verbose=False)
            new_seg = drizzled_segm == 0
            drizzled_segm[new_seg] = dseg_i[new_seg]
        
        return drizzled_segm
        
    def drizzle_segmentation_id(self, id=None, wcsobj=None, kernel='square', pixfrac=1, verbose=True):
        """
        Drizzle segmentation image for a single ID
        """
        import numpy as np

        import astropy.wcs as pywcs
        import astropy.io.fits as pyfits
        
        try:
            from . import utils
        except:
            from grizli import multifit, utils

        # Can be either a header or WCS object
        if isinstance(wcsobj, pyfits.Header):
            wcs = pywcs.WCS(wcsobj)
            wcs.pscale = utils.get_wcs_pscale(wcs)
        else:
            wcs = wcsobj
        
        if not hasattr(wcs, 'pscale'):
            wcs.pscale = utils.get_wcs_pscale(wcs)
        
        if id is None:
            id = self.id

        sci_list = [(beam.beam.seg == id)*1. for beam in self.beams]        
        wht_list = [np.isfinite(beam.beam.seg)*1. for beam in self.beams]        
        wcs_list = [beam.direct.wcs for beam in self.beams]        

        out = utils.drizzle_array_groups(sci_list, wht_list, wcs_list, outputwcs=wcs, scale=0.1, kernel=kernel, pixfrac=pixfrac, verbose=verbose)
        drizzled_segm = (out[0] > 0)*id

        return drizzled_segm
        
    def drizzle_fit_lines(self, fit, pline, force_line=['Ha+NII', 'Ha', 'OIII', 'Hb', 'OII'], save_fits=True, mask_lines=True, mask_sn_limit=3, mask_4959=True, verbose=True, include_segmentation=True, get_ir_psfs=True):
        """
        TBD
        """
        line_wavelengths, line_ratios = utils.get_line_wavelengths()
        hdu_full = []
        saved_lines = []
        
        if ('cfit' in fit) & mask_4959:
            if 'line OIII' in fit['templates']:
                t_o3 = utils.load_templates(fwhm=fit['templates']['line OIII'].fwhm, line_complexes=False, stars=False, full_line_list=['OIII-4959'], continuum_list=[], fsps_templates=False)
        
        if 'zbest' in fit:
            z_driz = fit['zbest']
        else:
            z_driz = fit['z']
            
        if 'line_flux' in fit:
            line_flux_dict = fit['line_flux']
        else:
            line_flux_dict = OrderedDict()
            for key in fit['cfit']:
                if key.startswith('line'):
                    line_flux_dict[key.replace('line ','')] = fit['cfit'][key]
        
        # Compute continuum model
        if 'cfit' in fit:
            if 'bg {0:03d}'.format(self.N-1) in fit['cfit']:
                for ib, beam in enumerate(self.beams):
                    key = 'bg {0:03d}'.format(ib)
                    self.beams[ib].background = fit['cfit'][key][0]
                    
        cont = fit['cont1d']
        for beam in self.beams:
            beam.compute_model(spectrum_1d=[cont.wave, cont.flux],
                               is_cgs=True)
        
            if hasattr(self, 'pscale'):
                if (self.pscale is not None):
                    scale = self.compute_scale_array(self.pscale, beam.wavef) 
                    beam.beam.pscale_array = scale.reshape(beam.sh)
                else:
                    beam.beam.pscale_array = 1.
            else:
                beam.beam.pscale_array = 1.
                
        for line in line_flux_dict:
            line_flux, line_err = line_flux_dict[line]
            if line_err == 0:
                continue

            if (line_flux/line_err > 4) | (line in force_line):
                if verbose:
                    print('Drizzle line -> {0:4s} ({1:.2f} {2:.2f})'.format(line, line_flux/1.e-17, line_err/1.e-17))

                line_wave_obs = line_wavelengths[line][0]*(1+z_driz)
                
                if mask_lines:
                    for beam in self.beams:
                        
                        beam.oivar = beam.ivar*1
                        lam = beam.beam.lam_beam
                        
                        if hasattr(beam.beam, 'pscale_array'):
                            pscale_array = beam.beam.pscale_array
                        else:
                            pscale_array = 1.
                        
                        ### another idea, compute a model for the line itself
                        ### and mask relatively "contaminated" pixels from 
                        ### other lines
                        try:
                            lm = fit['line1d'][line]
                            sp = [lm.wave, lm.flux]
                        except:
                            key = 'line '+ line
                            lm = fit['templates'][key]
                            line_flux = fit['cfit'][key][0]
                            scl = line_flux/(1+z_driz)
                            sp = [lm.wave*(1+z_driz), lm.flux*scl]
                            
                        #lm = fit['line1d'][line]
                        if ((lm.wave.max() < lam.min()) | 
                            (lm.wave.min() > lam.max())):
                            continue
                        
                        #sp = [lm.wave, lm.flux]
                        if line_flux > 0:
                            m = beam.compute_model(spectrum_1d=sp, 
                                               in_place=False, is_cgs=True)
                            lmodel = m.reshape(beam.beam.sh_beam)*pscale_array
                        else:
                            lmodel = np.zeros(beam.beam.sh_beam)
                            
                        # if lmodel.max() == 0:
                        #     continue
                        
                        if 'cfit' in fit:
                            keys = fit['cfit']
                        else:
                            keys = fit['line1d']
                        
                        beam.extra_lines = beam.contam*0.
                                
                        for lkey in keys:
                            if not lkey.startswith('line'):
                                continue
                            
                            key = lkey.replace('line ', '')
                            lf, le = line_flux_dict[key]
                            ### Don't mask if the line missing or undetected
                            if (lf <= 0):# | (lf < mask_sn_limit*le):
                                continue
                                
                            if key != line:
                                try:
                                    lm = fit['line1d'][lkey]
                                    sp = [lm.wave, lm.flux]
                                except:
                                    lm = fit['templates'][lkey]
                                    scl = fit['cfit'][lkey][0]/(1+z_driz)
                                    sp = [lm.wave*(1+z_driz), lm.flux*scl]
                                    
                                if ((lm.wave.max() < lam.min()) | 
                                    (lm.wave.min() > lam.max())):
                                    continue
                                    
                                m = beam.compute_model(spectrum_1d=sp, 
                                                       in_place=False,
                                                       is_cgs=True) 
                                                       
                                lcontam = m.reshape(beam.beam.sh_beam)
                                lcontam *= pscale_array
                                if lcontam.max() == 0:
                                    #print beam.grism.parent_file, lkey
                                    continue

                                beam.extra_lines += lcontam
                                
                                # Only mask if line flux > 0
                                if line_flux > 0:
                                    extra_msk = lcontam > mask_sn_limit*lmodel
                                    extra_msk &= (lcontam > 0)
                                    extra_msk &= (lmodel > 0)
                                
                                    beam.ivar[extra_msk] *= 0
                        
                        # Subtract 4959
                        if (line == 'OIII') & ('cfit' in fit) & mask_4959:
                            lm = t_o3['line OIII-4959']
                            scl = fit['cfit']['line OIII'][0]/(1+z_driz)
                            scl *= 1./(2.98+1)
                            sp = [lm.wave*(1+z_driz), lm.flux*scl]
                            
                            if ((lm.wave.max() < lam.min()) | 
                                (lm.wave.min() > lam.max())):
                                continue

                            m = beam.compute_model(spectrum_1d=sp, 
                                                   in_place=False,
                                                   is_cgs=True) 
                                                   
                            lcontam = m.reshape(beam.beam.sh_beam)
                            lcontam *= pscale_array
                            if lcontam.max() == 0:
                                continue

                            #print('Mask 4959!')
                            beam.extra_lines += lcontam
                            
                hdu = drizzle_to_wavelength(self.beams, ra=self.ra, 
                                            dec=self.dec, wave=line_wave_obs,
                                            fcontam=self.fcontam,
                                            **pline)
                
                if mask_lines:
                    for beam in self.beams:
                        beam.ivar = beam.oivar*1
                        delattr(beam, 'oivar')
                        
                hdu[0].header['REDSHIFT'] = (z_driz, 'Redshift used')
                #for e in [3,4,5,6]:
                for e in [-4,-3,-2,-1]:
                    hdu[e].header['EXTVER'] = line
                    hdu[e].header['REDSHIFT'] = (z_driz, 'Redshift used')
                    hdu[e].header['RESTWAVE'] = (line_wavelengths[line][0], 
                                                 'Line rest wavelength')

                saved_lines.append(line)

                if len(hdu_full) == 0:
                    hdu_full = hdu
                    hdu_full[0].header['NUMLINES'] = (1, 
                                               "Number of lines in this file")
                else:
                    hdu_full.extend(hdu[-4:])
                    hdu_full[0].header['NUMLINES'] += 1 
                
                    # Make sure DSCI extension is filled.  Can be empty for 
                    # lines at the edge of the grism throughput
                    for f_i in range(hdu[0].header['NDFILT']):
                        filt_i = hdu[0].header['DFILT{0:02d}'.format(f_i+1)]
                        if hdu['DWHT',filt_i].data.max() != 0:
                            hdu_full['DSCI',filt_i] = hdu['DSCI',filt_i]
                            hdu_full['DWHT',filt_i] = hdu['DWHT',filt_i]
                    
                li = hdu_full[0].header['NUMLINES']
                hdu_full[0].header['LINE{0:03d}'.format(li)] = line
                hdu_full[0].header['FLUX{0:03d}'.format(li)] = (line_flux, 
                                        'Line flux, 1e-17 erg/s/cm2')
                hdu_full[0].header['ERR{0:03d}'.format(li)] = (line_err, 
                                        'Line flux err, 1e-17 erg/s/cm2')

        if len(hdu_full) > 0:
            hdu_full[0].header['HASLINES'] = (' '.join(saved_lines), 
                                              'Lines in this file')
        else:
            hdu = drizzle_to_wavelength(self.beams, ra=self.ra, 
                                        dec=self.dec, 
                                        wave=np.median(self.beams[0].wave),
                                        fcontam=self.fcontam,
                                        **pline)
            hdu_full = hdu[:-4]
            hdu_full[0].header['REDSHIFT'] = (z_driz, 'Redshift used')
            hdu_full[0].header['NUMLINES'] = 0
            hdu_full[0].header['HASLINES'] = ' '
        
        if include_segmentation:
            line_wcs = pywcs.WCS(hdu_full[1].header)
            segm = self.drizzle_segmentation(wcsobj=line_wcs)
            seg_hdu = pyfits.ImageHDU(data=segm.astype(np.int32), name='SEG')
            hdu_full.insert(1, seg_hdu)
        
        if get_ir_psfs:
            import grizli.galfit.psf
            ir_beams = []
            gr_filters = {'G102':['F105W'], 'G141':['F105W','F125W','F140W','F160W']}
            show_filters = []
            
            for gr in ['G102','G141']:
                if gr in self.PA:
                    show_filters.extend(gr_filters[gr])
                    for pa in self.PA[gr]:
                        for i in self.PA[gr][pa]:
                            ir_beams.append(self.beams[i])
            
            if len(ir_beams) > 0:             
                dp = grizli.galfit.psf.DrizzlePSF(driz_hdu=hdu_full['DSCI'], 
                                    beams=self.beams)
                
                for filt in np.unique(show_filters):
                    if verbose:
                        print('Get linemap PSF: {0}'.format(filt))
                    
                    psf = dp.get_psf(ra=dp.driz_wcs.wcs.crval[0],
                                     dec=dp.driz_wcs.wcs.crval[1], 
                                     filter=filt, 
                                     pixfrac=dp.driz_header['PIXFRAC'], 
                                     kernel=dp.driz_header['DRIZKRNL'], 
                                     wcs_slice=dp.driz_wcs, get_extended=True, 
                                     verbose=False, get_weight=False)
                                     
                    psf[1].header['EXTNAME'] = 'DPSF'
                    psf[1].header['EXTVER'] = filt
                    hdu_full.append(psf[1])
                    
        if save_fits:
            hdu_full.writeto('{0}_{1:05d}.line.fits'.format(self.group_name, self.id), clobber=True, output_verify='silentfix')
        
        return hdu_full
        
    def run_full_diagnostics(self, pzfit={}, pspec2={}, pline={}, 
                      force_line=['Ha+NII', 'Ha', 'OIII', 'Hb', 'OII'],
                      GroupFLT=None,
                      prior=None, zoom=True, verbose=True):
        """TBD
        
        size=20, pixscale=0.1,
        pixfrac=0.2, kernel='square'
        
        """        
        import copy
        
        ## Defaults
        pzfit_def, pspec2_def, pline_def = get_redshift_fit_defaults()
            
        if pzfit == {}:
            pzfit = pzfit_def
            
        if pspec2 == {}: 
            pspec2 = pspec2_def
        
        if pline == {}:
            pline = pline_def
        
        ### Check that keywords allowed
        for d, default in zip([pzfit, pspec2, pline], 
                              [pzfit_def, pspec2_def, pline_def]):
            for key in d:
                if key not in default:
                    p = d.pop(key)
                            
        ### Auto generate FWHM (in km/s) to use for line fits
        if 'fwhm' in pzfit:
            fwhm = pzfit['fwhm']
            if pzfit['fwhm'] == 0:
                fwhm = 700
                if 'G141' in self.Ngrism:
                    fwhm = 1200
                if 'G800L' in self.Ngrism:
                    fwhm = 1400
                #
                if 'G280' in self.Ngrism:
                    fwhm = 1500
                # WFIRST
                if 'GRISM' in self.Ngrism:
                    fwhm = 350
                
        ### Auto generate delta-wavelength of 2D spectrum
        if 'dlam' in pspec2:
            dlam = pspec2['dlam']
            if dlam == 0:
                dlam = 25
                if 'G141' in self.Ngrism:
                    dlam = 45
                
                if 'G800L' in self.Ngrism:
                    dlam = 40
                
                if 'G280' in self.Ngrism:
                    dlam = 18
                
                if 'GRISM' in self.Ngrism:
                    dlam = 11
                
        ### Redshift fit
        zfit_in = copy.copy(pzfit)
        zfit_in['fwhm'] = fwhm
        zfit_in['prior'] = prior
        zfit_in['zoom'] = zoom
        zfit_in['verbose'] = verbose
        
        if zfit_in['zr'] is 0:
            fit, fig = self.fit_stars(**zfit_in)
        else:
            fit, fig = self.fit_redshift(**zfit_in)
        
        ### Make sure model attributes are set to the continuum model
        models = self.reshape_flat(fit['model_cont'])
        for j in range(self.N):
            self.beams[j].model = models[j]*1
        
        ### 2D spectrum
        spec_in = copy.copy(pspec2)
        spec_in['fit'] = fit
        spec_in['dlam'] = dlam
        
        #fig2, hdu2 = self.redshift_fit_twod_figure(**spec_in)#, kwargs=spec2) #dlam=dlam, spatial_scale=spatial_scale, NY=NY)
        fig2 = hdu2 = None
        
        ### Update master model
        if GroupFLT is not None:
            try:
                ix = GroupFLT.catalog['NUMBER'] == self.beams[0].id
                mag = GroupFLT.catalog['MAG_AUTO'][ix].data[0]
            except:
                mag = 22
                
            sp = fit['cont1d']
            GroupFLT.compute_single_model(id, mag=mag, size=-1, store=False,
                                          spectrum_1d=[sp.wave, sp.flux],
                                          is_cgs=True,
                                          get_beams=None, in_place=True)
        
        ## 2D lines to drizzle
        hdu_full = self.drizzle_fit_lines(fit, pline, force_line=force_line, 
                                          save_fits=True)
        
        
        fit['id'] = self.id
        fit['fit_bg'] = self.fit_bg
        fit['grism_files'] = [b.grism.parent_file for b in self.beams]
        for item in ['A','coeffs','model_full','model_cont']:
            if item in fit:
                p = fit.pop(item)
            
        #p = fit.pop('coeffs')
        
        np.save('{0}_{1:05d}.zfit.npy'.format(self.group_name, self.id), [fit])
            
        fig.savefig('{0}_{1:05d}.zfit.png'.format(self.group_name, self.id))
        
        #fig2.savefig('{0}_{1:05d}.zfit.2D.png'.format(self.group_name, self.id))
        #hdu2.writeto('{0}_{1:05d}.zfit.2D.fits'.format(self.group_name, self.id), clobber=True, output_verify='silentfix')
        
        label = '# id ra dec zbest '
        data = '{0:7d} {1:.6f} {2:.6f} {3:.5f}'.format(self.id, self.ra, self.dec,
                                      fit['zbest'])
        
        for grism in ['G800L', 'G280', 'G102', 'G141', 'GRISM']:
            label += ' N{0}'.format(grism)
            if grism in self.Ngrism:
                data += ' {0:2d}'.format(self.Ngrism[grism])
            else:
                data += ' {0:2d}'.format(0)
                
        label += ' chi2 DoF ' 
        data += ' {0:14.1f} {1:d} '.format(fit['chibest'], self.DoF)
        
        for line in ['SII', 'Ha', 'OIII', 'Hb', 'Hg', 'OII']:
            label += ' {0} {0}_err'.format(line)
            if line in fit['line_flux']:
                flux = fit['line_flux'][line][0]
                err =  fit['line_flux'][line][1]
                data += ' {0:10.3e} {1:10.3e}'.format(flux, err)
        
        fp = open('{0}_{1:05d}.zfit.dat'.format(self.group_name, self.id),'w')
        fp.write(label+'\n')
        fp.write(data+'\n')
        fp.close()
        
        fp = open('{0}_{1:05d}.zfit.beams.dat'.format(self.group_name, self.id),'w')
        fp.write('# file filter origin_x origin_y size pad bg\n')
        for ib, beam in enumerate(self.beams):
            data = '{0:40s} {1:s} {2:5d} {3:5d} {4:5d} {5:5d}'.format(beam.grism.parent_file, beam.grism.filter,
                                          beam.direct.origin[0],
                                          beam.direct.origin[1],
                                          beam.direct.sh[0], 
                                          beam.direct.pad)
            if self.fit_bg:
                data += ' {0:8.4f}'.format(fit['coeffs_full'][ib])
            else:
                data += ' {0:8.4f}'.format(0.0)
            
            fp.write(data + '\n')
        
        fp.close()
                                      
        ## Save figures
        plt_status = plt.rcParams['interactive']
        # if not plt_status:
        #     plt.close(fig)
        #     plt.close(fig2)
            
        return fit, fig, fig2, hdu2, hdu_full
    
    def apply_trace_shift(self, set_to_zero=False):
        """
        Set beam.yoffset back to zero
        """
        indices = [[i] for i in range(self.N)]
        if set_to_zero:
            s0 = np.zeros(len(indices))
        else:
            s0 = [beam.beam.yoffset for beam in self.beams] 
            
        args = (self, indices, 0, False, False, True)
        self.eval_trace_shift(s0, *args)
        
        ### Reset model profile for optimal extractions
        for b in self.beams:
            #b._parse_from_data()
            b._parse_from_data(contam_sn_mask=b.contam_sn_mask,
                                  min_mask=b.min_mask, min_sens=b.min_sens)
        
        self._parse_beam_arrays()
                
    def fit_trace_shift(self, split_groups=True, max_shift=5, tol=1.e-2, 
                        verbose=True, lm=False, fit_with_psf=False,
                        reset=False):
        """TBD
        """
        from scipy.optimize import leastsq, minimize
        
        if split_groups:
            indices = []
            for g in self.PA:
                for p in self.PA[g]:
                    indices.append(self.PA[g][p])
        else:
            indices = [[i] for i in range(self.N)]
        
        s0 = np.zeros(len(indices))
        bounds = np.array([[-max_shift,max_shift]]*len(indices))
        
        args = (self, indices, 0, lm, verbose, fit_with_psf)
        if reset:
            shifts = np.zeros(len(indices))
            out = None
        elif lm:
            out = leastsq(self.eval_trace_shift, s0, args=args, Dfun=None, full_output=0, col_deriv=0, ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None)
            shifts = out[0]
        else:
            out = minimize(self.eval_trace_shift, s0, bounds=bounds, args=args, method='Powell', tol=tol)
            if out.x.shape == ():
                shifts = [float(out.x)]
            else:
                shifts = out.x
        
        # Apply to PSF if necessary
        args = (self, indices, 0, lm, verbose, True)
        self.eval_trace_shift(shifts, *args)
        
        ### Reset model profile for optimal extractions
        for b in self.beams:
            #b._parse_from_data()
            b._parse_from_data(contam_sn_mask=b.contam_sn_mask,
                                  min_mask=b.min_mask, min_sens=b.min_sens)
            
            # Needed for background modeling
            if hasattr(b, 'xp'):
                delattr(b, 'xp')
                
        self._parse_beam_arrays()
        self.initialize_masked_arrays()
           
        return shifts, out
        
    @staticmethod
    def eval_trace_shift(shifts, self, indices, poly_order, lm, verbose, fit_with_psf):
        """TBD
        """
        import scipy.ndimage as nd

        for il, l in enumerate(indices):
            for i in l:
                beam = self.beams[i]
                beam.beam.add_ytrace_offset(shifts[il])

                if hasattr(self.beams[i].beam, 'psf') & fit_with_psf:
                    #beam.model = nd.shift(beam.modelf.reshape(beam.sh_beam), (shifts[il], 0))
                    # This is slow, so run with fit_with_psf=False if possible
                    beam.init_epsf(yoff=0, #shifts[il],
                                   psf_params=beam.beam.psf_params)

                    beam.compute_model(use_psf=True)
                    m = beam.compute_model(in_place=False)
                    #beam.modelf = beam.model.flatten()
                    #beam.model = beam.modelf.reshape(beam.beam.sh_beam)

                    beam.flat_flam = beam.compute_model(in_place=False, is_cgs=True) 
                                                         
                else:
                    #self.beams[i].beam.add_ytrace_offset(shifts[il])
                    #self.beams[i].compute_model(is_cgs=True)
                    beam.compute_model(use_psf=False)
                    
                if __name__ == '__main__':
                    print(self.beams[i].beam.yoffset, shifts[il])
                    ds9.view(self.beams[i].model)
                    
        self.flat_flam = np.hstack([b.beam.model.flatten() for b in self.beams])
        self.poly_order=-1
        self.init_poly_coeffs(poly_order=poly_order)

        self.fit_bg = False
        A = self.A_poly*1
        ok_temp = np.sum(A, axis=1) != 0  
        out_coeffs = np.zeros(A.shape[0])

        y = self.scif
        out = np.linalg.lstsq(A.T, y, rcond=None)                       
        lstsq_coeff, residuals, rank, s = out
        coeffs = lstsq_coeff

        out_coeffs = np.zeros(A.shape[0])
        out_coeffs[ok_temp] = coeffs
        modelf = np.dot(out_coeffs, A)
        
        if lm:
            # L-M, return residuals
            if verbose:
                print('{0} [{1}]'.format(utils.NO_NEWLINE, ' '.join(['{0:5.2f}'.format(s) for s in shifts])))
                return ((self.scif-modelf)*self.sivarf)[self.fit_mask]
                
        chi2 = np.sum(((self.scif - modelf)**2*self.ivarf)[self.fit_mask])

        if verbose:
            print('{0} [{1}] {2:6.2f}'.format(utils.NO_NEWLINE, ' '.join(['{0:5.2f}'.format(s) for s in shifts]), chi2/self.DoF))
        
        return chi2/self.DoF    
    
    def drizzle_grisms_and_PAs(self, size=10, fcontam=0, flambda=False, scale=1, pixfrac=0.5, kernel='square', make_figure=True, usewcs=False, zfit=None, diff=True, grism_list=['G800L','G102','G141','F090W','F115W','F150W','F200W','F356W','F410M','F444W']):
        """Make figure showing spectra at different orients/grisms
        
        TBD
        """
        from matplotlib.ticker import MultipleLocator
        #import pysynphot as S
        
        if usewcs:
            drizzle_function = drizzle_2d_spectrum_wcs
        else:
            drizzle_function = drizzle_2d_spectrum
            
        NX = len(self.PA)
        NY = 0
        for g in self.PA:
            NY = np.maximum(NY, len(self.PA[g]))
        
        NY += 1
                
        # keys = list(self.PA)
        # keys.sort()
        
        keys = []
        for key in grism_list:
            if key in self.PA:
                keys.append(key)
                
        if zfit is not None:
            if 'coeffs_full' in zfit:
                bg = zfit['coeffs_full'][:self.N]
                z_cont = zfit['zbest']
            else:
                # fitting.GroupFitter
                z_cont = zfit['z']
                bg = []
                for k in zfit['cfit']:
                    if k.startswith('bg '):
                        bg.append(zfit['cfit'][k][0])
                
                bg = np.array(bg)
        else:
            # Fit background
            try:
                out = self.xfit_at_z(z=0, templates={}, fitter='lstsq',
                                    poly_order=3, fit_background=True)
                bg = out[-3][:self.N]
            except:
                bg = [0]*self.N
            
        for ib, beam in enumerate(self.beams):
            beam.bg = bg[ib]
        
        prim = pyfits.PrimaryHDU()
        h0 = prim.header
        h0['ID'] = (self.id, 'Object ID')
        h0['RA'] = (self.ra, 'Right ascension')
        h0['DEC'] = (self.dec, 'Declination')
        h0['ISFLAM'] = (flambda, 'Pixels in f-lam units')
        h0['FCONTAM'] = (fcontam, 'Contamination parameter')
        h0['NGRISM'] = (len(keys), 'Number of grisms')
            
        all_hdus = []
        for ig, g in enumerate(keys):
            all_beams = []
            hdus = []
            
            pas = list(self.PA[g].keys())
            pas.sort()
            
            h0['GRISM{0:03d}'.format(ig+1)] = (g, 'Grism name')
            
            h0['N'+g] = (len(pas), 'Number of PAs for grism '+g)
            
            for ipa, pa in enumerate(pas):
                h0[g+'{0:02d}'.format(ipa+1)] = (pa, 'PA')
                
                beams = [self.beams[i] for i in self.PA[g][pa]]
                all_beams.extend(beams)
                #dlam = np.ceil(np.diff(beams[0].beam.lam)[0])*scale
                dlam = GRISM_LIMITS[g][2]*scale
                
                data = [beam.grism['SCI']-beam.contam-beam.bg
                           for beam in beams]
                
                hdu = drizzle_function(beams, data=data, 
                                          wlimit=GRISM_LIMITS[g], dlam=dlam, 
                                          spatial_scale=scale, NY=size,
                                          pixfrac=pixfrac,
                                          kernel=kernel,
                                          convert_to_flambda=flambda,
                                          fcontam=0, ds9=None)
                
                hdu[0].header['RA'] = (self.ra, 'Right ascension')
                hdu[0].header['DEC'] = (self.dec, 'Declination')
                hdu[0].header['GRISM'] = (g, 'Grism')
                hdu[0].header['PA'] = (pa, 'Dispersion PA')
                hdu[0].header['ISFLAM'] = (flambda, 'Pixels in f-lam units')
                hdu[0].header['CONF'] = (beams[0].beam.conf.conf_file,
                                         'Configuration file')
                                         
                hdu[0].header['DLAM0'] = (np.median(np.diff(beams[0].wave)),
                                         'Native dispersion per pix')
                
                ## Contam
                data = [beam.contam for beam in beams]
                
                hdu_contam = drizzle_function(beams, data=data, 
                                          wlimit=GRISM_LIMITS[g], dlam=dlam, 
                                          spatial_scale=scale, NY=size,
                                          pixfrac=pixfrac,
                                          kernel=kernel,
                                          convert_to_flambda=flambda,
                                          fcontam=0, ds9=None)
                
                hdu_contam[1].header['EXTNAME'] = 'CONTAM'
                hdu.append(hdu_contam[1])
                
                ## Continuum model
                if zfit is not None:
                    m = zfit['cont1d']
                    for beam in beams:
                        beam.compute_model(spectrum_1d=[m.wave, m.flux],
                                           is_cgs=True)
                else:
                    # simple flat spectrum
                    for beam in beams:
                        beam.compute_model()

                data = [beam.beam.model for beam in beams]
                    
                hdu_model = drizzle_function(beams, data=data, 
                                          wlimit=GRISM_LIMITS[g], dlam=dlam, 
                                          spatial_scale=scale, NY=size,
                                          pixfrac=pixfrac,
                                          kernel=kernel,
                                          convert_to_flambda=flambda,
                                          fcontam=0, ds9=None)
                
                hdu_model[1].header['EXTNAME'] = 'MODEL'
                if zfit is not None:
                    hdu_model[1].header['CONTIN1D'] = (True, 'Model is fit continuum')
                    hdu_model[1].header['REDSHIFT'] = (z_cont, 'Redshift of the continuum spectrum')
                else:
                    hdu_model[1].header['CONTIN1D'] = (False, 'Model is fit continuum')
                hdu.append(hdu_model[1])
                
                # Line kernel
                if not usewcs:
                    h = hdu[1].header
                    #gau = S.GaussianSource(1.e-17, h['CRVAL1'], h['CD1_1']*1)
                    
                    # header keywords scaled to um
                    toA = 1.e4
                    #toA = 1.
                    
                    #gau = S.GaussianSource(1., h['CRVAL1']*toA, h['CD1_1']*toA)
                    gau = utils.SpectrumTemplate(central_wave=h['CRVAL1']*toA, fwhm=h['CD1_1']*toA)
                    
                    #print('XXX', h['CRVAL1'], h['CD1_1'], h['CRPIX1'], toA, gau.wave[np.argmax(gau.flux)])
                    for beam in beams:
                        beam.compute_model(spectrum_1d=[gau.wave, gau.flux],
                                           is_cgs=True)
                
                    data = [beam.beam.model for beam in beams]
                
                    h_kern = drizzle_function(beams, data=data, 
                                              wlimit=GRISM_LIMITS[g],
                                              dlam=dlam, 
                                              spatial_scale=scale, NY=size,
                                              pixfrac=pixfrac,
                                              kernel=kernel,
                                              convert_to_flambda=flambda,
                                              fcontam=0, fill_wht=True,
                                              ds9=None)
                
                    kern = h_kern[1].data[:,h['CRPIX1']-1-size:h['CRPIX1']-1+size]
                    #print('XXX', kern.max(), h_kern[1].data.max())
                    hdu_kern = pyfits.ImageHDU(data=kern, header=h_kern[1].header, name='KERNEL')
                    hdu.append(hdu_kern)
                else:
                    hdu['DSCI'].header['EXTNAME'] = 'KERNEL'
                    
                ## Pull out zeroth extension
                for k in hdu[0].header:
                    hdu[1].header[k] = hdu[0].header[k]
                
                for e in hdu[1:]:
                    e.header['EXTVER'] = '{0},{1}'.format(g, pa)
                
                hdus.append(hdu[1:])
            
            
            ### Stack of each grism    
            data = [beam.grism['SCI']-beam.contam-beam.bg 
                        for beam in all_beams]
            
            hdu = drizzle_function(all_beams, data=data, 
                                      wlimit=GRISM_LIMITS[g], dlam=dlam, 
                                      spatial_scale=scale, NY=size,
                                      pixfrac=pixfrac,
                                      kernel=kernel,
                                      convert_to_flambda=flambda,
                                      fcontam=fcontam, ds9=None)
            
            hdu[0].header['RA'] = (self.ra, 'Right ascension')
            hdu[0].header['DEC'] = (self.dec, 'Declination')
            hdu[0].header['GRISM'] = (g, 'Grism')
            hdu[0].header['ISFLAM'] = (flambda, 'Pixels in f-lam units')
            hdu[0].header['CONF'] = (beams[0].beam.conf.conf_file,
                                     'Configuration file')
            hdu[0].header['DLAM0'] = (np.median(np.diff(beams[0].wave)),
                                     'Native dispersion per pix')
                                     
            ## Full continuum model
            if zfit is not None:
                m = zfit['cont1d']
                for beam in all_beams:
                    beam.compute_model(spectrum_1d=[m.wave, m.flux],
                                       is_cgs=True)
            else:
                for beam in all_beams:
                    beam.compute_model()

            data = [beam.beam.model for beam in all_beams]
                
            hdu_model = drizzle_function(all_beams, data=data, 
                                      wlimit=GRISM_LIMITS[g], dlam=dlam, 
                                      spatial_scale=scale, NY=size,
                                      pixfrac=pixfrac,
                                      kernel=kernel,
                                      convert_to_flambda=flambda,
                                      fcontam=fcontam, ds9=None)
            
            hdu_model[1].header['EXTNAME'] = 'MODEL'
            if zfit is not None:
                hdu_model[1].header['CONTIN1D'] = (True, 'Model is fit continuum')
                hdu_model[1].header['REDSHIFT'] = (z_cont, 'Redshift of the continuum spectrum')
            else:
                hdu_model[1].header['CONTIN1D'] = (False, 'Model is fit continuum')
                
            hdu.append(hdu_model[1])
            
            ## Full kernel
            h = hdu[1].header
            #gau = S.GaussianSource(1.e-17, h['CRVAL1'], h['CD1_1']*1)
            toA = 1.e4
            #gau = S.GaussianSource(1., h['CRVAL1']*toA, h['CD1_1']*toA)
            gau = utils.SpectrumTemplate(central_wave=h['CRVAL1']*toA, fwhm=h['CD1_1']*toA)
            
            for beam in all_beams:
                beam.compute_model(spectrum_1d=[gau.wave, gau.flux],
                                   is_cgs=True)
            
            data = [beam.beam.model for beam in all_beams]
            
            h_kern = drizzle_function(all_beams, data=data, 
                                      wlimit=GRISM_LIMITS[g], dlam=dlam, 
                                      spatial_scale=scale, NY=size,
                                      pixfrac=pixfrac,
                                      kernel=kernel,
                                      convert_to_flambda=flambda,
                                      fcontam=0, fill_wht=True, ds9=None)
            
            kern = h_kern[1].data[:,h['CRPIX1']-1-size:h['CRPIX1']-1+size]
            hdu_kern = pyfits.ImageHDU(data=kern, header=h_kern[1].header, name='KERNEL')
            hdu.append(hdu_kern)
            
            ## Pull out zeroth extension
            for k in hdu[0].header:
                hdu[1].header[k] = hdu[0].header[k]
            
            for e in hdu[1:]:
                e.header['EXTVER'] = '{0}'.format(g)
            
            hdus.append(hdu[1:])
            all_hdus.extend(hdus)
                    
        output_hdu = pyfits.HDUList([prim])
        for hdu in all_hdus:
            output_hdu.extend(hdu)
            
        if make_figure:
            fig = show_drizzle_HDU(output_hdu, diff=diff)
            return output_hdu, fig
        else:
            return output_hdu #all_hdus
    
    def flag_with_drizzled(self, hdul, sigma=4, update=True, interp='nearest', verbose=True):
        """
        Update `MultiBeam` masks based on the blotted drizzled combined image
        
        [in progress ... xxx]
        
        Parameters
        ----------
        hdul : `~astropy.io.fits.HDUList`
            FITS HDU list output from `drizzle_grisms_and_PAs` or read from a
            `stack.fits` file.
        
        sigma : float
            Residual threshold to flag.
        
        update : bool
            Update the mask.
        
        interp : str
            Interpolation method for `~drizzlepac.ablot`.
            
        Returns
        -------
        Updates the individual `fit_mask` attributes of the individual beams
        if `update==True`.
        
        """
        try:
            from drizzle.doblot import doblot
            blotter = doblot
        except:
            from drizzlepac import ablot
            blotter = ablot.do_blot
        
        # Read the drizzled arrays
        Ng = hdul[0].header['NGRISM']
        ref_wcs = {}
        ref_data = {}
        flag_grism = {}
        
        for i in range(Ng):
            g = hdul[0].header['GRISM{0:03d}'.format(i+1)]
            ref_wcs[g] = pywcs.WCS(hdul['SCI',g].header)
            ref_wcs[g].pscale = utils.get_wcs_pscale(ref_wcs[g])
            ref_data[g] = hdul['SCI',g].data
            flag_grism[g] = hdul[0].header['N{0}'.format(g)] > 1
        
        # Do the masking
        for i, beam in enumerate(self.beams):
            g = beam.grism.filter
            if not flag_grism[g]:
                continue
            
            beam_header, flt_wcs = beam.full_2d_wcs()
            blotted = blotter(ref_data[g], ref_wcs[g],
                                flt_wcs, 1,
                                coeffs=True, interp=interp, sinscl=1.0,
                                stepsize=10, wcsmap=None)
            
            resid = (beam.grism['SCI'] - beam.contam - blotted) 
            resid *= np.sqrt(beam.ivar)
            blot_mask = (blotted != 0) & (np.abs(resid) < sigma)
            if verbose:
                print('Beam {0:>3d}: {1:>4d} new masked pixels'.format(i, beam.fit_mask.sum() - (beam.fit_mask & blot_mask.flatten()).sum()))
                
            if update:
                beam.fit_mask &= blot_mask.flatten()
        
        if update:
            self._parse_beams()
            self.initialize_masked_arrays()
            
    def oned_spectrum(self, tfit=None, **kwargs):
        """Compute full 1D spectrum with optional best-fit model
        
        Parameters
        ----------
        bin : float / int
            Bin factor relative to the size of the native spectral bins of a
            given grism.
            
        tfit : dict
            Output of `~grizli.fitting.mb.template_at_z`.
            
        Returns
        -------
        sp : dict
            Dictionary of the extracted 1D spectra.  Keys are the grism 
            names and the values are `~astropy.table.Table` objects.
        
        """
        import astropy.units as u
        
        # "Flat" spectrum to perform flux calibration
        if self.Nphot > 0:
            flat_data = self.flat_flam[self.fit_mask[:-self.Nphotbands]]
        else:
            flat_data = self.flat_flam[self.fit_mask]

        sp_flat = self.optimal_extract(flat_data, **kwargs)

        # Best-fit line and continuum models, with background fit 
        if tfit is not None:
            bg_model = self.get_flat_background(tfit['coeffs'],
                                                apply_mask=True)

            line_model = self.get_flat_model([tfit['line1d'].wave,
                                              tfit['line1d'].flux])
            cont_model = self.get_flat_model([tfit['line1d'].wave,
                                              tfit['cont1d'].flux])
                                              
            sp_line = self.optimal_extract(line_model, **kwargs)
            sp_cont = self.optimal_extract(cont_model, **kwargs)
        else:
            bg_model = 0.
        
        # Optimal spectral extraction
        sp = self.optimal_extract(self.scif_mask[:self.Nspec]-bg_model, **kwargs)
        
        # Loop through grisms, change units and add fit columns
        # NB: setting units to "count / s" to comply with FITS standard, 
        #     where count / s = electron / s
        for k in sp:
            sp[k]['flat'] = sp_flat[k]['flux']
            flat_unit = (u.count / u.s) / (u.erg / u.s / u.cm**2 / u.AA)
            sp[k]['flat'].unit = flat_unit
            
            sp[k]['flux'].unit = u.count / u.s
            sp[k]['err'].unit = u.count / u.s
            
            if tfit is not None:
                sp[k]['line'] = sp_line[k]['flux']
                sp[k]['line'].unit = u.count / u.s
                sp[k]['cont'] = sp_cont[k]['flux']
                sp[k]['cont'].unit = u.count / u.s
            
            sp[k].meta['GRISM'] = (k, 'Grism name')
            
            # Metadata
            exptime = count = 0            
            for pa in self.PA[k]:
                for i in self.PA[k][pa]:
                    exptime += self.beams[i].grism.header['EXPTIME']
                    count += 1
                    parent = (self.beams[i].grism.parent_file, 'Parent file')
                    sp[k].meta['FILE{0:04d}'.format(count)] = parent
            
            sp[k].meta['NEXP'] = (count, 'Number of exposures')
            sp[k].meta['EXPTIME'] = (exptime, 'Total exposure time')
            sp[k].meta['NPA'] = (len(self.PA[k]), 'Number of PAs')
            
        return sp
    
    def oned_spectrum_to_hdu(self, sp=None, outputfile=None, units=None, **kwargs):
        """Generate 1D spectra fits HDUList
        
        Parameters
        ----------
        sp : optional, dict
            Output of `~grizli.multifit.MultiBeam.oned_spectrum`.  If None, 
            then run that function with `**kwargs`.
            
        outputfile : None, str
            If a string supplied, then write the `~astropy.io.fits.HDUList` to
            a file.
        
        Returns
        -------
        hdul : `~astropy.io.fits.HDUList`
            FITS version of the 1D spectrum tables.
        
        """
        from astropy.io.fits.convenience import table_to_hdu
        
        # Generate the spectrum if necessary
        if sp is None:
            sp = self.oned_spectrum(**kwargs)
        
        # Metadata in PrimaryHDU
        prim = pyfits.PrimaryHDU()
        prim.header['ID'] = (self.id, 'Object ID')
        prim.header['RA'] = (self.ra, 'Right Ascension')
        prim.header['DEC'] = (self.dec, 'Declination')
        prim.header['TARGET'] = (self.group_name, 'Target Name')
        prim.header['MW_EBV'] = (self.MW_EBV, 'Galactic extinction E(B-V)')
        
        for g in ['G102', 'G141', 'G800L']:
            if g in sp:
                prim.header['N_{0}'.format(g)] = sp[g].meta['NEXP']
                prim.header['T_{0}'.format(g)] = sp[g].meta['EXPTIME']
                prim.header['PA_{0}'.format(g)] = sp[g].meta['NPA']
            else:
                prim.header['N_{0}'.format(g)] = (0, 'Number of exposures')
                prim.header['T_{0}'.format(g)] = (0, 'Total exposure time')
                prim.header['PA_{0}'.format(g)] = (0, 'Number of PAs')
                    
        for i, k in enumerate(sp):
            prim.header['GRISM{0:03d}'.format(i+1)] = (k, 'Grism name')
        
        # Generate HDUList
        hdul = [prim]
        for k in sp:
            hdu = table_to_hdu(sp[k])
            hdu.header['EXTNAME'] = k
            hdul.append(hdu)
        
        # Outputs
        hdul = pyfits.HDUList(hdul)
        if outputfile is None:
            return hdul
        else:
            hdul.writeto(outputfile, overwrite=True)
            return hdul
    
    def check_for_bad_PAs(self, poly_order=1, chi2_threshold=1.5, fit_background=True, reinit=True):
        """
        """

        wave = np.linspace(2000,2.5e4,100)
        poly_templates = utils.polynomial_templates(wave, order=poly_order)
        
        fit_log = OrderedDict()
        keep_dict = {}
        has_bad = False
        
        keep_beams = []
        
        for g in self.PA:
            fit_log[g] = OrderedDict()
            keep_dict[g] = []
                            
            for pa in self.PA[g]:
                beams = [self.beams[i] for i in self.PA[g][pa]]
                mb_i = MultiBeam(beams, fcontam=self.fcontam,
                                 sys_err=self.sys_err)
                              
                try:
                    chi2, _, _, _ = mb_i.xfit_at_z(z=0,
                                                   templates=poly_templates,
                                                fit_background=fit_background)
                except:
                    chi2 = 1e30
                    
                if False:
                    p_i = mb_i.template_at_z(z=0, templates=poly_templates, fit_background=fit_background, fitter='lstsq', fwhm=1400, get_uncertainties=2)
                
                fit_log[g][pa] = {'chi2': chi2, 'DoF': mb_i.DoF, 
                                  'chi_nu': chi2/np.maximum(mb_i.DoF, 1)}
                
            
            min_chinu = 1e30
            for pa in self.PA[g]:
                min_chinu = np.minimum(min_chinu, fit_log[g][pa]['chi_nu'])
            
            fit_log[g]['min_chinu'] = min_chinu
            
            for pa in self.PA[g]:
                fit_log[g][pa]['chinu_ratio'] = fit_log[g][pa]['chi_nu']/min_chinu
                
                if fit_log[g][pa]['chinu_ratio'] < chi2_threshold:
                    keep_dict[g].append(pa)
                    keep_beams.extend([self.beams[i] for i in self.PA[g][pa]])
                else:
                    has_bad = True
        
        if reinit:
            self.beams = keep_beams
            self._parse_beams(psf=self.psf_param_dict is not None)
            
        return fit_log, keep_dict, has_bad
            
                
def get_redshift_fit_defaults():
    """TBD
    """
    pzfit_def = dict(zr=[0.5, 1.6], dz=[0.005, 0.0004], fwhm=0,
                 poly_order=0, fit_background=True,
                 delta_chi2_threshold=0.004, fitter='nnls', 
                 prior=None, templates={}, figsize=[8,5],
                 fsps_templates=False)
    
    pspec2_def = dict(dlam=0, spatial_scale=1, NY=20, figsize=[8,3.5])
    pline_def = dict(size=20, pixscale=0.1, pixfrac=0.2, kernel='square', 
                     wcs=None)

    return pzfit_def, pspec2_def, pline_def
                
def drizzle_2d_spectrum(beams, data=None, wlimit=[1.05, 1.75], dlam=50, 
                        spatial_scale=1, NY=10, pixfrac=0.6, kernel='square',
                        convert_to_flambda=True, fcontam=0.2, fill_wht=False,
                        ds9=None):
    """Drizzle 2D spectrum from a list of beams
    
    Parameters
    ----------
    beams : list of `~.model.BeamCutout` objects
    
    data : None or list
        optionally, drizzle data specified in this list rather than the 
        contamination-subtracted arrays from each beam.
    
    wlimit : [float, float]
        Limits on the wavelength array to drizzle ([wlim, wmax])
    
    dlam : float
        Delta wavelength per pixel
    
    spatial_scale : float
        Relative scaling of the spatial axis (1 = native pixels)
    
    NY : int
        Size of the cutout in the spatial dimension, in output pixels
    
    pixfrac : float
        Drizzle PIXFRAC (for `kernel` = 'point')

    kernel : str, ('square' or 'point')
        Drizzle kernel to use
    
    convert_to_flambda : bool, float
        Convert the 2D spectrum to physical units using the sensitivity curves
        and if float provided, scale the flux densities by that value
    
    fcontam: float
        Factor by which to scale the contamination arrays and add to the 
        pixel variances.
    
    fill_wht: bool
        Fill `wht==0` pixels of the beam weights with the median nonzero 
        value.
        
    ds9: `~grizli.ds9.DS9`
        Show intermediate steps of the drizzling
    
    Returns
    -------
    hdu : `~astropy.io.fits.HDUList`
        FITS HDUList with the drizzled 2D spectrum and weight arrays
        
    """
    from astropy import log
    # try:
    #     import drizzle
    #     if drizzle.__version__ != '1.12.99':
    #         # Not the fork that works for all input/output arrays
    #         raise(ImportError)
    #     
    #     #print('drizzle!!')
    #     from drizzle.dodrizzle import dodrizzle
    #     drizzler = dodrizzle
    #     dfillval = '0'
    # except:
    from drizzlepac import adrizzle
    adrizzle.log.setLevel('ERROR')
    drizzler = adrizzle.do_driz
    dfillval = 0
    
    log.setLevel('ERROR')
    #log.disable_warnings_logging()
    
    NX = int(np.round(np.diff(wlimit)[0]*1.e4/dlam)) // 2
    center = np.mean(wlimit[:2])*1.e4
    out_header, output_wcs = utils.full_spectrum_wcsheader(center_wave=center,
                                 dlam=dlam, NX=NX, 
                                 spatial_scale=spatial_scale, NY=NY)
                                 
    sh = (out_header['NAXIS2'], out_header['NAXIS1'])
    
    outsci = np.zeros(sh, dtype=np.float32)
    outwht = np.zeros(sh, dtype=np.float32)
    outctx = np.zeros(sh, dtype=np.int32)
    
    outvar = np.zeros(sh, dtype=np.float32)
    outwv = np.zeros(sh, dtype=np.float32)
    outcv = np.zeros(sh, dtype=np.int32)
    
    if data is None:
        data = []
        for i, beam in enumerate(beams):
            ### Contamination-subtracted
            beam_data = beam.grism.data['SCI'] - beam.contam 
            data.append(beam_data)
            
    for i, beam in enumerate(beams):
        ## Get specific WCS for each beam
        beam_header, beam_wcs = beam.full_2d_wcs()
        if not hasattr(beam_wcs, 'pixel_shape'):
            beam_wcs.pixel_shape = beam_wcs._naxis1, beam_wcs._naxis2
        
        # Downweight contamination
        # wht = 1/beam.ivar + (fcontam*beam.contam)**2
        # wht = np.cast[np.float32](1/wht)
        # wht[~np.isfinite(wht)] = 0.
        
        contam_weight = np.exp(-(fcontam*np.abs(beam.contam)*np.sqrt(beam.ivar)))
        wht = beam.ivar*contam_weight
        
        wht[~np.isfinite(wht)] = 0.
        contam_weight[beam.ivar == 0] = 0
        
        if fill_wht:
            wht_mask = wht == 0
            med_wht = np.median(wht[~wht_mask])
            wht[wht_mask] = med_wht
            #print('xx Fill weight: {0}'.format(med_wht))
            
        data_i = data[i]*1.
        scl = 1.
        if convert_to_flambda:
            #data_i *= convert_to_flambda/beam.beam.sensitivity
            #wht *= (beam.beam.sensitivity/convert_to_flambda)**2
            
            scl = convert_to_flambda#/1.e-17
            scl *= 1./beam.flat_flam.reshape(beam.beam.sh_beam).sum(axis=0)
            #scl = convert_to_flambda/beam.beam.sensitivity
            
            data_i *= scl
            wht *= (1/scl)**2            
            #contam_weight *= scl
            
            wht[~np.isfinite(data_i+scl)] = 0
            contam_weight[~np.isfinite(data_i+scl)] = 0
            data_i[~np.isfinite(data_i+scl)] = 0
        
        ###### Go drizzle
        
        ### Contamination-cleaned
        drizzler(data_i, beam_wcs, wht, output_wcs, 
                         outsci, outwht, outctx, 1., 'cps', 1,
                         wcslin_pscale=1., uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
        
        # For variance
        drizzler(contam_weight, beam_wcs, wht, output_wcs, 
                         outvar, outwv, outcv, 1., 'cps', 1,
                         wcslin_pscale=1., uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
        
        if ds9 is not None:
            ds9.view(outsci/output_wcs.pscale**2, header=out_header)
        
        # if False:
        #     # Plot the spectra for testing
        #     w, f, e = beam.beam.trace_extract(data_i, ivar=wht, r=3)
        #     clip = (f/e > 0.5) 
        #     clip &= (e < 2*np.median(e[clip]))
        #     plt.errorbar(w[clip], f[clip], e[clip], marker='.', color='k', alpha=0.5, ecolor='0.8', linestyle='None')
        #     dw = np.median(np.diff(w))
        
    ### Correct for drizzle scaling    
    area_ratio = 1./output_wcs.pscale**2
    
    ### Preserve flux (has to preserve aperture flux along spatial axis but
    ### average in spectral axis).
    #area_ratio *= spatial_scale
    
    # preserve flux density
    flux_density_scale = spatial_scale**2
    
    # science
    outsci *= area_ratio*flux_density_scale
    
    # variance
    outvar *= area_ratio/outwv*flux_density_scale**2
    outwht = 1/outvar
    outwht[(outvar == 0) | (~np.isfinite(outwht))] = 0
    
    # if True:
    #     # Plot for testing....
    #     yp, xp = np.indices(outsci.shape)
    #     mask = np.abs(yp-NY) <= 3/spatial_scale
    #     fl = (outsci*mask).sum(axis=0)
    #     flv = (1/outwht*mask).sum(axis=0)
    #     
    #     wi = grizli.stack.StackedSpectrum.get_wavelength_from_header(out_header) 
    #     
    #     plt.errorbar(wi[:-1], fl[1:], np.sqrt(flv)[1:], alpha=0.8) #*area_ratio)
            
    #return outwht, outsci, outvar, outwv, output_wcs.pscale
        
    p = pyfits.PrimaryHDU()
    p.header['ID'] = (beams[0].id, 'Object ID')
    p.header['WMIN'] = (wlimit[0], 'Minimum wavelength')
    p.header['WMAX'] = (wlimit[1], 'Maximum wavelength')
    p.header['DLAM'] = (dlam, 'Delta wavelength')
    
    p.header['SSCALE'] = (spatial_scale, 'Spatial scale factor w.r.t native')
    p.header['FCONTAM'] = (fcontam, 'Contamination weight')
    p.header['PIXFRAC'] = (pixfrac, 'Drizzle PIXFRAC')
    p.header['DRIZKRNL'] = (kernel, 'Drizzle kernel')
    p.header['BEAM'] = (beams[0].beam.beam, 'Grism order')
    
    p.header['NINPUT'] = (len(beams), 'Number of drizzled beams')
    exptime = 0.
    for i, beam in enumerate(beams):
        p.header['FILE{0:04d}'.format(i+1)] = (beam.grism.parent_file, 
                                             'Parent filename')
        p.header['GRIS{0:04d}'.format(i+1)] = (beam.grism.filter, 
                                             'Beam grism element')
        p.header['PA{0:04d}'.format(i+1)] = (beam.get_dispersion_PA(), 
                                             'PA of dispersion axis')
        exptime += beam.grism.exptime
    
    p.header['EXPTIME'] = (exptime, 'Total exposure time [s]')
        
    h = out_header.copy()
        
    grism_sci = pyfits.ImageHDU(data=outsci, header=h, name='SCI')
    grism_wht = pyfits.ImageHDU(data=outwht, header=h, name='WHT')
    
    hdul = pyfits.HDUList([p, grism_sci, grism_wht])
    
    return hdul
    
def drizzle_to_wavelength(beams, wcs=None, ra=0., dec=0., wave=1.e4, size=5,
                          pixscale=0.1, pixfrac=0.6, kernel='square',
                          direct_extension='REF', fcontam=0.2, ds9=None):
    """Drizzle a cutout at a specific wavelength from a list of `BeamCutout`s
    
    Parameters
    ----------
    beams : list of `~.model.BeamCutout` objects.
    
    wcs : `~astropy.wcs.WCS` or None
        Pre-determined WCS.  If not specified, generate one based on `ra`, 
        `dec`, `pixscale` and `pixscale`
        
    ra, dec, wave : float
        Sky coordinates and central wavelength

    size : float
        Size of the output thumbnail, in arcsec
        
    pixscale : float
        Pixel scale of the output thumbnail, in arcsec
        
    pixfrac : float
        Drizzle PIXFRAC (for `kernel` = 'point')
        
    kernel : str, ('square' or 'point')
        Drizzle kernel to use
    
    direct_extension : str, ('SCI' or 'REF')
        Extension of `self.direct.data` do drizzle for the thumbnail
    
    fcontam: float
        Factor by which to scale the contamination arrays and add to the 
        pixel variances.
        
    ds9 : `~grizli.ds9.DS9`, optional
        Display each step of the drizzling to an open DS9 window
    
    Returns
    -------
    hdu : `~astropy.io.fits.HDUList`
        FITS HDUList with the drizzled thumbnail, line and continuum 
        cutouts.
    """
    # try:
    #     import drizzle
    #     if drizzle.__version__ != '1.12.99':
    #         # Not the fork that works for all input/output arrays
    #         raise(ImportError)
    #     
    #     #print('drizzle!!')
    #     from drizzle.dodrizzle import dodrizzle
    #     drizzler = dodrizzle
    #     dfillval = '0'
    # except:
    from drizzlepac import adrizzle
    adrizzle.log.setLevel('ERROR')
    drizzler = adrizzle.do_driz
    dfillval = 0
        
    # Nothing to do
    if len(beams) == 0:
        return False
        
    ### Get output header and WCS
    if wcs is None:
        header, output_wcs = utils.make_wcsheader(ra=ra, dec=dec, size=size, pixscale=pixscale, get_hdu=False)
    else:
        output_wcs = wcs.copy()
        if not hasattr(output_wcs, 'pscale'):
            output_wcs.pscale = utils.get_wcs_pscale(output_wcs)
            
        header = utils.to_header(output_wcs, relax=True)
        
    ### Initialize data
    sh = (header['NAXIS2'], header['NAXIS1'])
    
    outsci = np.zeros(sh, dtype=np.float32)
    outwht = np.zeros(sh, dtype=np.float32)
    outctx = np.zeros(sh, dtype=np.int32)

    coutsci = np.zeros(sh, dtype=np.float32)
    coutwht = np.zeros(sh, dtype=np.float32)
    coutctx = np.zeros(sh, dtype=np.int32)

    xoutsci = np.zeros(sh, dtype=np.float32)
    xoutwht = np.zeros(sh, dtype=np.float32)
    xoutctx = np.zeros(sh, dtype=np.int32)
    
    #direct_filters = np.unique([b.direct.filter for b in self.beams])
    all_direct_filters = []
    for beam in beams:
        if direct_extension == 'REF':
            if beam.direct['REF'] is None:
                filt_i = beam.direct.ref_filter
                direct_extension = 'SCI'
            else:
                filt_i = beam.direct.filter
        
        all_direct_filters.append(filt_i)
    
    direct_filters = np.unique(all_direct_filters)
    
    doutsci, doutwht, doutctx = {}, {}, {}
    for f in direct_filters:
        doutsci[f] = np.zeros(sh, dtype=np.float32)
        doutwht[f] = np.zeros(sh, dtype=np.float32)
        doutctx[f] = np.zeros(sh, dtype=np.int32)

    # doutsci = np.zeros(sh, dtype=np.float32)
    # doutwht = np.zeros(sh, dtype=np.float32)
    # doutctx = np.zeros(sh, dtype=np.int32)
    
    ## Loop through beams and run drizzle
    for i, beam in enumerate(beams):
        ## Get specific wavelength WCS for each beam
        beam_header, beam_wcs = beam.get_wavelength_wcs(wave)
        
        if not hasattr(beam_wcs, 'pixel_shape'):
            beam_wcs.pixel_shape = beam_wcs._naxis1, beam_wcs._naxis2
        
        ## Make sure CRPIX set correctly for the SIP header
        for j in [0,1]: 
            # if beam_wcs.sip is not None:
            #     beam_wcs.sip.crpix[j] = beam_wcs.wcs.crpix[j]
            if beam.direct.wcs.sip is not None:
                beam.direct.wcs.sip.crpix[j] = beam.direct.wcs.wcs.crpix[j]
            
            for wcs_ext in [beam_wcs.sip]: 
                if wcs_ext is not None:
                    wcs_ext.crpix[j] = beam_wcs.wcs.crpix[j]
                
        # ACS requires additional wcs attributes
        ACS_CRPIX = [4096/2,2048/2] 
        dx_crpix = beam_wcs.wcs.crpix[0] - ACS_CRPIX[0]
        dy_crpix = beam_wcs.wcs.crpix[1] - ACS_CRPIX[1]
        for wcs_ext in [beam_wcs.cpdis1, beam_wcs.cpdis2, beam_wcs.det2im1, beam_wcs.det2im2]:
            if wcs_ext is not None:
                wcs_ext.crval[0] += dx_crpix
                wcs_ext.crval[1] += dy_crpix
                        
        beam_data = beam.grism.data['SCI'] - beam.contam 
        if hasattr(beam, 'background'):
            beam_data -= beam.background
        
        if hasattr(beam, 'extra_lines'):
            beam_data -= beam.extra_lines    
        
        beam_continuum = beam.beam.model*1
        if hasattr(beam.beam, 'pscale_array'):
            beam_continuum *= beam.beam.pscale_array
                
        # Downweight contamination
        if fcontam > 0:
            # wht = 1/beam.ivar + (fcontam*beam.contam)**2
            # wht = np.cast[np.float32](1/wht)
            # wht[~np.isfinite(wht)] = 0.
            
            contam_weight = np.exp(-(fcontam*np.abs(beam.contam)*np.sqrt(beam.ivar)))
            wht = beam.ivar*contam_weight
            wht[~np.isfinite(wht)] = 0.
            
        else:
            wht = beam.ivar*1
        
        ### Convert to f_lambda integrated line fluxes: 
        ###     (Inverse of the aXe sensitivity) x (size of pixel in \AA)
        sens = np.interp(wave, beam.beam.lam, beam.beam.sensitivity, 
                         left=0, right=0)
        
        dlam = np.interp(wave, beam.beam.lam[1:], np.diff(beam.beam.lam))
        # 1e-17 erg/s/cm2 #, scaling closer to e-/s
        sens *= 1.e-17
        sens *= 1./dlam
        
        if sens == 0:
            continue
        else:
            wht *= sens**2
            beam_data /= sens
            beam_continuum /= sens
        
        ###### Go drizzle
                
        ### Contamination-cleaned
        drizzler(beam_data, beam_wcs, wht, output_wcs, 
                         outsci, outwht, outctx, 1., 'cps', 1,
                         wcslin_pscale=beam.grism.wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
        
        ### Continuum
        drizzler(beam_continuum, beam_wcs, wht, output_wcs, 
                         coutsci, coutwht, coutctx, 1., 'cps', 1, 
                         wcslin_pscale=beam.grism.wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
        
        ### Contamination
        drizzler(beam.contam, beam_wcs, wht, output_wcs, 
                         xoutsci, xoutwht, xoutctx, 1., 'cps', 1, 
                         wcslin_pscale=beam.grism.wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
        
        ### Direct thumbnail
        filt_i = all_direct_filters[i]
        
        if direct_extension == 'REF':
            thumb = beam.direct['REF']
            thumb_wht = np.cast[np.float32]((thumb != 0)*1)
        else:
            thumb = beam.direct[direct_extension]#/beam.direct.photflam
            thumb_wht = 1./(beam.direct.data['ERR']/beam.direct.photflam)**2
            thumb_wht[~np.isfinite(thumb_wht)] = 0
        
        if not hasattr(beam.direct.wcs, 'pixel_shape'):
            beam.direct.wcs.pixel_shape = beam.direct.wcs._naxis1, beam.direct.wcs._naxis2
           
        drizzler(thumb, beam.direct.wcs, thumb_wht, output_wcs, 
                         doutsci[filt_i], doutwht[filt_i], doutctx[filt_i], 
                         1., 'cps', 1, 
                         wcslin_pscale=beam.direct.wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
        
        ## Show in ds9
        if ds9 is not None:
            ds9.view((outsci-coutsci), header=header)
    
    ## Scaling of drizzled outputs     
    #print 'Pscale: ', output_wcs.pscale    
    #outsci /= (output_wcs.pscale)**2
    #coutsci /= (output_wcs.pscale)**2
    # doutsci /= (output_wcs.pscale)**2
    
    outwht  *= (beams[0].grism.wcs.pscale/output_wcs.pscale)**4
    coutwht *= (beams[0].grism.wcs.pscale/output_wcs.pscale)**4
    xoutwht *= (beams[0].grism.wcs.pscale/output_wcs.pscale)**4
    
    for filt_i in all_direct_filters:
        doutwht[filt_i] *= (beams[0].direct.wcs.pscale/output_wcs.pscale)**4
    
    ### Make output FITS products
    p = pyfits.PrimaryHDU()
    p.header['ID'] = (beams[0].id, 'Object ID')
    p.header['RA'] = (ra, 'Central R.A.')
    p.header['DEC'] = (dec, 'Central Decl.')
    p.header['PIXFRAC'] = (pixfrac, 'Drizzle PIXFRAC')
    p.header['DRIZKRNL'] = (kernel, 'Drizzle kernel')
    
    p.header['NINPUT'] = (len(beams), 'Number of drizzled beams')
    for i, beam in enumerate(beams):
        p.header['FILE{0:04d}'.format(i+1)] = (beam.grism.parent_file, 
                                             'Parent filename')
        p.header['GRIS{0:04d}'.format(i+1)] = (beam.grism.filter, 
                                             'Beam grism element')
        p.header['PA{0:04d}'.format(i+1)] = (beam.get_dispersion_PA(), 
                                             'PA of dispersion axis')
        
    h = header.copy()
    h['ID'] = (beam.id, 'Object ID')
    h['PIXFRAC'] = (pixfrac, 'Drizzle PIXFRAC')
    h['DRIZKRNL'] = (kernel, 'Drizzle kernel')
    
    p.header['NDFILT'] = len(direct_filters), 'Number of direct image filters'
    for i, filt_i in enumerate(direct_filters):
        p.header['DFILT{0:02d}'.format(i+1)] = filt_i
        p.header['NFILT{0:02d}'.format(i+1)] = all_direct_filters.count(filt_i), 'Number of beams with this direct filter'
    
    HDUL = [p]
    for i, filt_i in enumerate(direct_filters):
        h['FILTER'] = (filt_i, 'Direct image filter')
        
        thumb_sci = pyfits.ImageHDU(data=doutsci[filt_i], header=h,
                                    name='DSCI')
        thumb_wht = pyfits.ImageHDU(data=doutwht[filt_i], header=h,
                                    name='DWHT')

        thumb_sci.header['EXTVER'] = filt_i                         
        thumb_wht.header['EXTVER'] = filt_i                         
        
        HDUL += [thumb_sci, thumb_wht]
                                    
    #thumb_seg = pyfits.ImageHDU(data=seg_slice, header=h, name='DSEG')
        
    h['FILTER'] = (beam.grism.filter, 'Grism filter')
    h['WAVELEN'] = (wave, 'Central wavelength')
    
    grism_sci = pyfits.ImageHDU(data=outsci-coutsci, header=h, name='LINE')
    grism_cont = pyfits.ImageHDU(data=coutsci, header=h, name='CONTINUUM')
    grism_contam = pyfits.ImageHDU(data=xoutsci, header=h, name='CONTAM')
    grism_wht = pyfits.ImageHDU(data=outwht, header=h, name='LINEWHT')
    
    #HDUL = [p, thumb_sci, thumb_wht, grism_sci, grism_cont, grism_contam, grism_wht]        
    HDUL += [grism_sci, grism_cont, grism_contam, grism_wht]        

    return pyfits.HDUList(HDUL)

def show_drizzle_HDU(hdu, diff=True):
    """Make a figure from the multiple extensions in the drizzled grism file.
    
    Parameters
    ----------
    hdu : `~astropy.io.fits.HDUList`
        HDU list output by `drizzle_grisms_and_PAs`.
    
    diff : bool
        If True, then plot the stacked spectrum minus the model.
        
    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The figure.

    """
    from collections import OrderedDict
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MultipleLocator
    
    h0 = hdu[0].header
    NX = h0['NGRISM']
    NY = 0
    
    grisms = OrderedDict()
    
    for ig in range(NX):
        g = h0['GRISM{0:03d}'.format(ig+1)]
        NY = np.maximum(NY, h0['N'+g])
        grisms[g] = h0['N'+g]
        
    NY += 1
        
    fig = plt.figure(figsize=(5*NX, 1*NY))
    
    widths = []
    for i in range(NX):
        widths.extend([0.2, 1])
        
    gs = GridSpec(NY, NX*2, height_ratios=[1]*NY, width_ratios=widths)
    
    for ig, g in enumerate(grisms):
        
        sci_i = hdu['SCI',g]
        wht_i = hdu['WHT',g]
        model_i = hdu['MODEL',g]
        kern_i = hdu['KERNEL',g]
        h_i = sci_i.header
        
        clip = wht_i.data > 0 
        if clip.sum() == 0:
            clip = np.isfinite(wht_i.data)
        
        avg_rms = 1/np.median(np.sqrt(wht_i.data[clip]))
        vmax = np.maximum(1.1*np.percentile(sci_i.data[clip],98),
                         5*avg_rms)
        
        vmax_kern = 1.1*np.percentile(kern_i.data,99.5)
        
        # Kernel
        ax = fig.add_subplot(gs[NY-1, ig*2+0])
        sh = kern_i.data.shape
        extent = [0, sh[1], 0, sh[0]]
        
        ax.imshow(kern_i.data, origin='lower', interpolation='Nearest', 
                  vmin=-0.1*vmax_kern, vmax=vmax_kern, cmap=plt.cm.viridis_r,
                  extent=extent, aspect='auto')
        
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.xaxis.set_tick_params(length=0)
        ax.yaxis.set_tick_params(length=0)
        
        # Spectrum
        sh = sci_i.data.shape
        extent = [h_i['WMIN'], h_i['WMAX'], 0, sh[0]]
        
        ax = fig.add_subplot(gs[NY-1, ig*2+1])
        
        if diff:
            #print('xx DIFF!')
            m = model_i.data
        else:
            m = 0
            
        ax.imshow(sci_i.data-m, origin='lower',
                  interpolation='Nearest', vmin=-0.1*vmax, vmax=vmax, 
                  extent=extent, cmap = plt.cm.viridis_r, 
                  aspect='auto')
        
        ax.set_yticklabels([])
        ax.set_xlabel(r'$\lambda$ ($\mu$m) - '+g)
        ax.xaxis.set_major_locator(MultipleLocator(GRISM_MAJOR[g]))
        
        for ip in range(grisms[g]):
            #print(ip, ig)
            pa = h0['{0}{1:02d}'.format(g, ip+1)]

            sci_i = hdu['SCI','{0},{1}'.format(g, pa)]
            wht_i = hdu['WHT','{0},{1}'.format(g, pa)]
            kern_i = hdu['KERNEL','{0},{1}'.format(g, pa)]
            h_i = sci_i.header

            # Kernel
            ax = fig.add_subplot(gs[ip, ig*2+0])
            sh = kern_i.data.shape
            extent = [0, sh[1], 0, sh[0]]
            
            ax.imshow(kern_i.data, origin='lower', interpolation='Nearest', 
                      vmin=-0.1*vmax_kern, vmax=vmax_kern, extent=extent,
                      cmap=plt.cm.viridis_r, aspect='auto')

            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.xaxis.set_tick_params(length=0)
            ax.yaxis.set_tick_params(length=0)
            
            # Spectrum
            sh = sci_i.data.shape
            extent = [h_i['WMIN'], h_i['WMAX'], 0, sh[0]]

            ax = fig.add_subplot(gs[ip, ig*2+1])

            ax.imshow(sci_i.data, origin='lower',
                      interpolation='Nearest', vmin=-0.1*vmax, vmax=vmax, 
                      extent=extent, cmap = plt.cm.viridis_r, 
                      aspect='auto')
           
            ax.set_yticklabels([]); ax.set_xticklabels([])
            ax.xaxis.set_major_locator(MultipleLocator(GRISM_MAJOR[g]))
            ax.text(0.015, 0.94, '{0:3.0f}'.format(pa), ha='left',
                    va='top',
                    transform=ax.transAxes, fontsize=8, 
                    backgroundcolor='w')
            
            if (ig == (NX-1)) & (ip == 0):
                ax.text(0.98, 0.94, 'ID = {0}'.format(h0['ID']), 
                        ha='right', va='top', transform=ax.transAxes,
                        fontsize=8, backgroundcolor='w')
            
    gs.tight_layout(fig, pad=0.1)
    return fig

def drizzle_2d_spectrum_wcs(beams, data=None, wlimit=[1.05, 1.75], dlam=50, 
                        spatial_scale=1, NY=10, pixfrac=0.6, kernel='square',
                        convert_to_flambda=True, fcontam=0.2, fill_wht=False,
                        ds9=None):
    """Drizzle 2D spectrum from a list of beams
    
    Parameters
    ----------
    beams : list of `~.model.BeamCutout` objects
    
    data : None or list
        optionally, drizzle data specified in this list rather than the 
        contamination-subtracted arrays from each beam.
    
    wlimit : [float, float]
        Limits on the wavelength array to drizzle ([wlim, wmax])
    
    dlam : float
        Delta wavelength per pixel
    
    spatial_scale : float
        Relative scaling of the spatial axis (1 = native pixels)
    
    NY : int
        Size of the cutout in the spatial dimension, in output pixels
    
    pixfrac : float
        Drizzle PIXFRAC (for `kernel` = 'point')

    kernel : str, ('square' or 'point')
        Drizzle kernel to use
    
    convert_to_flambda : bool, float
        Convert the 2D spectrum to physical units using the sensitivity curves
        and if float provided, scale the flux densities by that value
    
    fcontam: float
        Factor by which to scale the contamination arrays and add to the 
        pixel variances.
    
    ds9: `~grizli.ds9.DS9`
        Show intermediate steps of the drizzling
    
    Returns
    -------
    hdu : `~astropy.io.fits.HDUList`
        FITS HDUList with the drizzled 2D spectrum and weight arrays
        
    """
    # try:
    #     import drizzle
    #     if drizzle.__version__ != '1.12.99':
    #         # Not the fork that works for all input/output arrays
    #         raise(ImportError)
    #     
    #     #print('drizzle!!')
    #     from drizzle.dodrizzle import dodrizzle
    #     drizzler = dodrizzle
    #     dfillval = '0'
    # except:
    from drizzlepac import adrizzle
    adrizzle.log.setLevel('ERROR')
    drizzler = adrizzle.do_driz
    dfillval = 0

    from stwcs import distortion
    from astropy import log
    
    log.setLevel('ERROR')
    #log.disable_warnings_logging()
    adrizzle.log.setLevel('ERROR')
    
    NX = int(np.round(np.diff(wlimit)[0]*1.e4/dlam)) // 2
    center = np.mean(wlimit[:2])*1.e4
    out_header, output_wcs = utils.make_spectrum_wcsheader(center_wave=center,
                                 dlam=dlam, NX=NX, 
                                 spatial_scale=spatial_scale, NY=NY)
    
    pixscale = 0.128*spatial_scale
   
    # # Get central RA, reference pixel of beam[0]
    # #rd = beams[0].get_sky_coords()
    # x0 = beams[0].beam.x0.reshape((1,2))
    # #x0[0,1] += beam.direct.origin[1]-beam.grism.origin[1]
    # rd = beam.grism.wcs.all_pix2world(x0,1)[0]
    # theta = 270-beams[0].get_dispersion_PA()
    
    #out_header, output_wcs = utils.make_wcsheader(ra=rd[0], dec=rd[1], size=[50,10], pixscale=pixscale, get_hdu=False, theta=theta)
    
    if True:
        theta = -np.arctan2(np.diff(beams[0].beam.ytrace)[0], 1)
    
        undist_wcs = distortion.utils.output_wcs([beams[0].grism.wcs],undistort=True)    
        undist_wcs = utils.transform_wcs(undist_wcs, rotation=theta, scale=undist_wcs.pscale/pixscale)
    
        output_wcs = undist_wcs.copy()
        out_header = utils.to_header(output_wcs)
        
        # Direct image
        d_undist_wcs = distortion.utils.output_wcs([beams[0].direct.wcs],undistort=True)    
        d_undist_wcs = utils.transform_wcs(d_undist_wcs, rotation=0., scale=d_undist_wcs.pscale/pixscale)
    
        d_output_wcs = d_undist_wcs.copy()
        # Make square
        dx = d_output_wcs._naxis1-d_output_wcs._naxis2
        d_output_wcs._naxis1 = d_output_wcs._naxis2
        d_output_wcs.wcs.crpix[0] -= dx/2.
        d_out_header = utils.to_header(d_output_wcs)
        
    #delattr(output_wcs, 'orientat')
    
    #beam_header = utils.to_header(beam_wcs)
    #output_wcs = beam_wcs
    #output_wcs = pywcs.WCS(beam_header, relax=True)
    #output_wcs.pscale = utils.get_wcs_pscale(output_wcs)
    
    # shift CRPIX to reference position of beam[0]
                              
    sh = (out_header['NAXIS2'], out_header['NAXIS1'])
    
    sh_d = (d_out_header['NAXIS2'], d_out_header['NAXIS1'])
    
    outsci = np.zeros(sh, dtype=np.float32)
    outwht = np.zeros(sh, dtype=np.float32)
    outctx = np.zeros(sh, dtype=np.int32)
    
    doutsci = np.zeros(sh_d, dtype=np.float32)
    doutwht = np.zeros(sh_d, dtype=np.float32)
    doutctx = np.zeros(sh_d, dtype=np.int32)
    
    outvar = np.zeros(sh, dtype=np.float32)
    outwv = np.zeros(sh, dtype=np.float32)
    outcv = np.zeros(sh, dtype=np.int32)
    
    outls = np.zeros(sh, dtype=np.float32)
    outlw = np.zeros(sh, dtype=np.float32)
    outlc = np.zeros(sh, dtype=np.int32)
    
    if data is None:
        data = []
        for i, beam in enumerate(beams):
            ### Contamination-subtracted
            beam_data = beam.grism.data['SCI'] - beam.contam 
            data.append(beam_data)
            
    for i, beam in enumerate(beams):
        ## Get specific WCS for each beam
        beam_header, beam_wcs = beam.get_2d_wcs()
        beam_wcs = beam.grism.wcs.deepcopy()
        
        # Shift SIP reference
        dx_sip = beam.grism.origin[1] - beam.direct.origin[1]
        #beam_wcs.sip.crpix[0] += dx_sip
        for wcs_ext in [beam_wcs.sip]:
            if wcs_ext is not None:
                wcs_ext.crpix[0] += dx_sip

        for wcs_ext in [beam_wcs.cpdis1, beam_wcs.cpdis2, beam_wcs.det2im1, beam_wcs.det2im2]:
            if wcs_ext is not None:
                wcs_ext.crval[0] += dx_sip
                        
        # Shift y for trace
        xy0 = beam.grism.wcs.all_world2pix(output_wcs.wcs.crval.reshape((1,2)),0)[0]
        dy = np.interp(xy0[0], np.arange(beam.beam.sh_beam[1]), beam.beam.ytrace)
        #beam_wcs.sip.crpix[1] += dy
        beam_wcs.wcs.crpix[1] += dy
        
        for wcs_ext in [beam_wcs.sip]:
            if wcs_ext is not None:
                wcs_ext.crpix[1] += dy
        
        for wcs_ext in [beam_wcs.cpdis1, beam_wcs.cpdis2, beam_wcs.det2im1, beam_wcs.det2im2]:
            if wcs_ext is not None:
                wcs_ext.crval[1] += dy
        
        if not hasattr(beam_wcs, 'pixel_shape'):
            beam_wcs.pixel_shape = beam_wcs._naxis1, beam_wcs._naxis2
                
        d_beam_wcs = beam.direct.wcs
        if beam.direct['REF'] is None:
            d_wht = 1./beam.direct['ERR']**2
            d_wht[~np.isfinite(d_wht)] = 0
            d_sci = beam.direct['SCI']*1
        else:
            d_sci = beam.direct['REF']*1
            d_wht = d_sci*0.+1
            
        d_sci *= (beam.beam.seg == beam.id)
        
        # Downweight contamination
        # wht = 1/beam.ivar + (fcontam*beam.contam)**2
        # wht = np.cast[np.float32](1/wht)
        # wht[~np.isfinite(wht)] = 0.
        
        contam_weight = np.exp(-(fcontam*np.abs(beam.contam)*np.sqrt(beam.ivar)))
        wht = beam.ivar*contam_weight
        
        wht[~np.isfinite(wht)] = 0.
        contam_weight[beam.ivar == 0] = 0
        
        data_i = data[i]*1.
        scl = 1.
        if convert_to_flambda:
            #data_i *= convert_to_flambda/beam.beam.sensitivity
            #wht *= (beam.beam.sensitivity/convert_to_flambda)**2
            
            scl = convert_to_flambda#/1.e-17
            scl *= 1./beam.flat_flam.reshape(beam.beam.sh_beam).sum(axis=0)
            #scl = convert_to_flambda/beam.beam.sensitivity
            
            data_i *= scl
            wht *= (1/scl)**2            
            #contam_weight *= scl
            
            wht[~np.isfinite(data_i+scl)] = 0
            contam_weight[~np.isfinite(data_i+scl)] = 0
            data_i[~np.isfinite(data_i+scl)] = 0
        
        ###### Go drizzle
        
        data_wave = np.dot(np.ones(beam.beam.sh_beam[0])[:,None], beam.beam.lam[None,:])
        drizzler(data_wave, beam_wcs, wht*0.+1, output_wcs, 
                         outls, outlw, outlc, 1., 'cps', 1,
                         wcslin_pscale=1., uniqid=1, 
                         pixfrac=1, kernel='square', fillval=dfillval)
        
        ### Direct image
        drizzler(d_sci, d_beam_wcs, d_wht, d_output_wcs, 
                         doutsci, doutwht, doutctx, 1., 'cps', 1,
                         wcslin_pscale=d_beam_wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
                           
        ### Contamination-cleaned
        drizzler(data_i, beam_wcs, wht, output_wcs, 
                         outsci, outwht, outctx, 1., 'cps', 1,
                         wcslin_pscale=beam_wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
        
        # For variance
        drizzler(contam_weight, beam_wcs, wht, output_wcs, 
                         outvar, outwv, outcv, 1., 'cps', 1,
                         wcslin_pscale=beam_wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=dfillval)
        
        if ds9 is not None:
            ds9.view(outsci, header=out_header)
        
        # if True:
        #     w, f, e = beam.beam.optimal_extract(data_i, ivar=beam.ivar)
        #     plt.scatter(w, f, marker='.', color='k', alpha=0.5)
            
    ### Correct for drizzle scaling
    #outsci /= output_wcs.pscale**2
    outls /= output_wcs.pscale**2
    wave = np.median(outls, axis=0)
    
    # # Testing
    # fl = (sp[1].data*mask).sum(axis=0)
    
    # variance
    outvar /= outwv#*output_wcs.pscale**2
    outwht = 1/outvar
    outwht[(outvar == 0) | (~np.isfinite(outwht))] = 0
    
    #return outwht, outsci, outvar, outwv, output_wcs.pscale
        
    p = pyfits.PrimaryHDU()
    p.header['ID'] = (beams[0].id, 'Object ID')
    p.header['WMIN'] = (wave[0], 'Minimum wavelength')
    p.header['WMAX'] = (wave[-1], 'Maximum wavelength')
    p.header['DLAM'] = ((wave[-1]-wave[0])/wave.size, 'Delta wavelength')
    
    p.header['FCONTAM'] = (fcontam, 'Contamination weight')
    p.header['PIXFRAC'] = (pixfrac, 'Drizzle PIXFRAC')
    p.header['DRIZKRNL'] = (kernel, 'Drizzle kernel')
    
    p.header['NINPUT'] = (len(beams), 'Number of drizzled beams')
    for i, beam in enumerate(beams):
        p.header['FILE{0:04d}'.format(i+1)] = (beam.grism.parent_file, 
                                             'Parent filename')
        p.header['GRIS{0:04d}'.format(i+1)] = (beam.grism.filter, 
                                             'Beam grism element')
        
    h = out_header.copy()
    for k in p.header:
        h[k] = p.header[k]
    
    direct_sci = pyfits.ImageHDU(data=doutsci, header=d_out_header, name='DSCI')
    grism_sci = pyfits.ImageHDU(data=outsci, header=h, name='SCI')
    grism_wht = pyfits.ImageHDU(data=outwht, header=h, name='WHT')
    
    hdul = pyfits.HDUList([p, grism_sci, grism_wht, direct_sci])
    
    return hdul
