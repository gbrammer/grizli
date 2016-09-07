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

## local imports
from . import grismconf
from . import utils
from . import model
from .utils_c import disperse
from .utils_c import interp

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
    out = _compute_model(0, self.FLTs[i], fit_info, False)
    t1 = time.time()
    print t1-t0
    
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
    import cPickle as pickle
    
    ## slight random delay to avoid synchronization problems
    # np.random.seed(ix)
    # sleeptime = ix*1
    # print '%s sleep %.3f %d' %(grism_file, sleeptime, ix)
    # time.sleep(sleeptime)
    
    #print grism_file, direct_file
    
    save_file = grism_file.replace('_flt.fits', '_GrismFLT.fits')
    save_file = save_file.replace('_flc.fits', '_GrismFLT.fits')
    save_file = save_file.replace('_cmb.fits', '_GrismFLT.fits')
    if grism_file.find('_') < 0:
        save_file = 'xxxxxxxxxxxxxxxxxxx'
        
    if os.path.exists(save_file):
        print 'Load %s!' %(save_file)
        
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
    
    if catalog is not None:
        flt.catalog = flt.blot_catalog(catalog, 
                                   sextractor=('X_WORLD' in catalog.colnames))
    else:
        flt.catalog = None 
    
    return flt #, out_cat
    

def _compute_model(i, flt, fit_info, store):
    """TBD
    """
    for id in fit_info:
        status = flt.compute_model_orders(id=id, compute_size=True,
                          mag=fit_info[id]['mag'], in_place=True, store=store,
                          spectrum_1d = fit_info[id]['spec'], 
                          verbose=False)
    
    print '%s: _compute_model Done' %(flt.grism.parent_file)
        
    return i, flt.model, flt.object_dispersers
    
class GroupFLT():
    def __init__(self, grism_files=[], sci_extn=1, direct_files=[],
                 pad=200, group_name='group', 
                 ref_file=None, ref_ext=0, seg_file=None,
                 shrink_segimage=True, verbose=True, cpu_count=0,
                 catalog=''):
        """TBD
        
        Parameters
        ----------
        grism_files : list
        
        sci_extn : int
        
        direct_files : list
        
        pad : int
        
        group_name : str
        
        ref_file : `None` or str
        
        ref_ext : 0
        
        seg_file : `None` or str
        
        shrink_segimage : bool
        
        verbose : bool
        
        cpu_count : int
        
        catalog : str
        
        Attributes
        ----------
        TBD : type
        
        """
        self.N = len(grism_files)
        if len(direct_files) != len(grism_files):
            direct_files = ['']*self.N
        
        self.grism_files = grism_files
        self.direct_files = direct_files
        self.group_name = group_name
        
        ### Read catalog
        if catalog:
            if isinstance(catalog, str):
                try:
                    self.catalog = Table.read(catalog,
                                              format='ascii.sextractor')
                except:
                    self.catalog = Table.read(catalog,
                                              format='ascii.commented_header')
                    
            else:
                self.catalog = catalog
        else:
            self.catalog = None
                 
        if cpu_count == 0:
            cpu_count = mp.cpu_count()
        
        if cpu_count < 0:
            ### serial
            self.FLTs = []
            t0_pool = time.time()
            for i in xrange(self.N):
                flt = _loadFLT(self.grism_files[i], sci_extn, self.direct_files[i], pad, ref_file, ref_ext, seg_file, verbose, self.catalog, i)
                self.FLTs.append(flt)
                
            t1_pool = time.time()
        else:
            ### Read files in parallel
            self.FLTs = []
            t0_pool = time.time()
        
            pool = mp.Pool(processes=cpu_count)
            results = [pool.apply_async(_loadFLT, (self.grism_files[i], sci_extn, self.direct_files[i], pad, ref_file, ref_ext, seg_file, verbose, self.catalog, i)) for i in xrange(self.N)]
        
            pool.close()
            pool.join()
    
            for res in results:
                flt_i = res.get(timeout=1)
                #flt_i.catalog = cat_i
                self.FLTs.append(flt_i)
        
            t1_pool = time.time()
        
        if verbose:
            print 'Files loaded - %.2f sec.' %(t1_pool - t0_pool)
    
    def save_full_data(self, warn=True):
        """TBD
        """      
        for i in range(self.N):
            file = self.FLTs[i].grism_file
            if self.FLTs[i].grism.data is None:
                if warn:
                    print '%s: Looks like data already saved!' %(file)
                    continue
                    
            save_file = file.replace('_flt.fits', '_GrismFLT.fits')
            save_file = save_file.replace('_flc.fits', '_GrismFLT.fits')
            save_file = save_file.replace('_cmb.fits', '_GrismFLT.fits')
            print 'Save %s' %(save_file)
            self.FLTs[i].save_full_pickle()
            
            ### Reload initialized data
            self.FLTs[i].load_from_fits(save_file)
            
    def extend(self, new, verbose=True):
        """Add another GroupFLT instance to `self`
        
        TBD
        """
        self.FLTs.extend(new.FLTs)
        self.N = len(self.FLTs)
        self.direct_files.extend(new.direct_files)
        self.grism_files.extend(new.grism_files)
        
        if verbose:
            print 'Now we have %d FLTs' %(self.N)
            
    def compute_single_model(self, id, mag=-99, size=None, store=False, spectrum_1d=None, get_beams=None, in_place=True):
        """TBD
        """
        out_beams = []
        for flt in self.FLTs:
            status = flt.compute_model_orders(id=id, verbose=False,
                          size=size, compute_size=(size < 0),
                          mag=mag, in_place=in_place, store=store,
                          spectrum_1d = spectrum_1d, get_beams=get_beams)
            
            out_beams.append(status)
        
        if get_beams:
            return out_beams
        else:
            return True
            
    def compute_full_model(self, fit_info=None, verbose=True, store=False, 
                           mag_limit=25, coeffs=[1.2, -0.5], cpu_count=0):
        """TBD
        """
        if cpu_count == 0:
            cpu_count = mp.cpu_count()
        
        if fit_info is None:
            bright = self.catalog['MAG_AUTO'] < mag_limit
            ids = self.catalog['NUMBER'][bright]
            mags = self.catalog['MAG_AUTO'][bright]

            xspec = np.arange(0.6, 2.1, 0.05)-1

            yspec = [xspec**o*coeffs[o] for o in range(len(coeffs))]
            xspec = (xspec+1)*1.e4
            yspec = np.sum(yspec, axis=0)

            fit_info = OrderedDict()
            for id, mag in zip(ids, mags):
                fit_info[id] = {'mag':mag, 'spec': [xspec, yspec]}
            
        t0_pool = time.time()
        
        pool = mp.Pool(processes=cpu_count)
        results = [pool.apply_async(_compute_model, (i, self.FLTs[i], fit_info, store)) for i in xrange(self.N)]

        pool.close()
        pool.join()
                
        for res in results:
            i, model, dispersers = res.get(timeout=1)
            self.FLTs[i].object_dispersers = dispersers
            self.FLTs[i].model = model
            
        t1_pool = time.time()
        if verbose:
            print 'Models computed - %.2f sec.' %(t1_pool - t0_pool)
        
    def get_beams(self, id, size=10, beam_id='A', min_overlap=0.2, 
                  get_slice_header=True):
        """TBD
        """
        beams = self.compute_single_model(id, size=size, store=False, get_beams=[beam_id])
        
        out_beams = []
        for flt, beam in zip(self.FLTs, beams):
            try:
                out_beam = model.BeamCutout(flt=flt, beam=beam[beam_id],
                                        conf=flt.conf, 
                                        get_slice_header=get_slice_header)
            except:
                continue
            
            hasdata = ((out_beam.grism['SCI'] != 0).sum(axis=0) > 0).sum()
            if hasdata*1./out_beam.model.shape[1] < min_overlap:
                continue
            
            out_beams.append(out_beam)
            
        return out_beams
    
    def refine_list(self, ids=[], mags=[], poly_order=2, mag_limits=[16,24], 
                    max_coeff=5, ds9=None, verbose=True):
        """TBD
        
        bright = self.catalog['MAG_AUTO'] < 24
        ids = self.catalog['NUMBER'][bright]*1
        mags = self.catalog['MAG_AUTO'][bright]*1
        so = np.argsort(mags)
        
        ids, mags = ids[so], mags[so]
        
        self.refine_list(ids, mags, ds9=ds9)
        
        """
        if (len(ids) == 0) | (len(ids) != len(mags)):
            bright = ((self.catalog['MAG_AUTO'] < mag_limits[1]) &
                      (self.catalog['MAG_AUTO'] > mag_limits[0]))
                      
            ids = self.catalog['NUMBER'][bright]*1
            mags = self.catalog['MAG_AUTO'][bright]*1
            
            so = np.argsort(mags)
            ids, mags = ids[so], mags[so]
            
        for id, mag in zip(ids, mags):
            self.refine(id, mag=mag, poly_order=poly_order,
                        max_coeff=max_coeff, size=30, ds9=ds9,
                        verbose=verbose)
            
    def refine(self, id, mag=-99, poly_order=1, size=30, ds9=None, verbose=True, max_coeff=2.5):
        """TBD
        """
        beams = self.get_beams(id, size=size, min_overlap=0.5, get_slice_header=False)
        if len(beams) == 0:
            return True
        
        mb = MultiBeam(beams)
        try:
            A, out_coeffs, chi2, modelf = mb.fit_at_z(poly_order=poly_order, fit_background=True, fitter='lstsq')
        except:
            return False
            
        xspec = np.arange(0.3, 2.1, 0.05)-1
        scale_coeffs = out_coeffs[mb.N*mb.fit_bg:mb.N*mb.fit_bg+mb.n_poly]
        yspec = [xspec**o*scale_coeffs[o] for o in range(mb.poly_order+1)]
        if np.abs(scale_coeffs).max() > max_coeff:
            return True
            
        self.compute_single_model(id, mag=mag, size=None, store=False, spectrum_1d=[(xspec+1)*1.e4, np.sum(yspec, axis=0)], get_beams=None, in_place=True)
        
        if ds9:
            flt = self.FLTs[0]
            mask = flt.grism['SCI'] != 0
            ds9.view((flt.grism['SCI'] - flt.model)*mask,
                      header=flt.grism.header)
        
        if verbose:
            print '%d mag=%.2f %s' %(id, mag, scale_coeffs)
            
        return True
        #m2d = mb.reshape_flat(modelf)
    
    def drizzle_full_wavelength(self, wave=1.4e4, ref_header=None,
                     kernel='point', pixfrac=1., verbose=True, 
                     offset=[0,0]):
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
        from drizzlepac.astrodrizzle import adrizzle
        import astropy.wcs as pywcs
        
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
        fit_p = fitting.LevMarLSQFitter()
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

            # Science and wht arrays
            sci = flt.grism['SCI'] - flt.model
            wht = 1/(flt.grism['ERR']**2)
            wht[~np.isfinite(wht)] = 0
            
            # Drizzle it
            if verbose:
                print 'Drizzle %s to wavelength %.2f' %(flt.grism.parent_file, 
                                                        wave)
                                                        
            adrizzle.do_driz(sci, line_wcs, wht, out_wcs, 
                             outsci, outwht, outctx, 1., 'cps', 1,
                             wcslin_pscale=line_wcs.pscale, uniqid=1, 
                             pixfrac=pixfrac, kernel=kernel, fillval=0, 
                             stepsize=10, wcsmap=None)
        
        # Done!
        return outsci, outwht
        
class MultiBeam():
    def __init__(self, beams, group_name='group', fcontam=0.):
        """Tools for dealing with multiple `~.model.BeamCutout` instances 
        
        Parameters
        ----------
        beams : list
            List of `~.model.BeamCutout` objects.
        
        group_name : type
            Rootname to use for saved products
            
        fcontam : type
            Factor to use to downweight contaminated pixels.
        
        Attributes
        ----------
        TBD : type
        
        """     
        self.N = len(beams)
        self.group_name = group_name

        self.Ngrism = {}
        for i in range(self.N):
            grism = beams[i].grism.filter
            if grism in self.Ngrism:
                self.Ngrism[grism] += 1
            else:
                self.Ngrism[grism] = 1
                
        if hasattr(beams[0], 'lower'):
            ### `beams` is list of strings
            self.load_beam_fits(beams)            
        else:
            self.beams = beams
        
        self.id = self.beams[0].id
        
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
        self.fit_mask &= self.ivarf >= 0
        
        self.scif = np.hstack([b.scif for b in self.beams])
        self.contamf = np.hstack([b.contam.flatten() for b in self.beams])
        
        self.fcontam = fcontam
        if fcontam > 0:
            self.ivarf = 1./(1./self.ivarf + (fcontam*self.contamf)**2)
            self.ivarf[~np.isfinite(self.ivarf)] = 0
            self.ivarf[self.ivarf < 0] = 0
            
            #mask = (self.contamf*np.sqrt(self.ivarf) > fcontam) & (self.contamf > fcontam*self.flat_flam)
            #self.ivarf[mask] = 0
            
        ### Initialize background fit array
        self.A_bg = np.zeros((self.N, self.Ntot))
        i0 = 0
        for i in range(self.N):
            self.A_bg[i, i0:i0+self.Nflat[i]] = 1.
            i0 += self.Nflat[i]
        
        self.init_poly_coeffs(poly_order=1)
        
        self.ra, self.dec = self.beams[0].get_sky_coords()
        
    def write_beam_fits(self, verbose=True):
        """TBD
        """
        outfiles = []
        for beam in self.beams:
            root = beam.grism.parent_file.split('.fits')[0]
            outfile = beam.write_fits(root)
            if verbose:
                print 'Wrote %s' %(outfile)
            
            outfiles.append(outfile)
            
        return outfiles
        
    def load_beam_fits(self, beam_list, conf=None):
        """TBD
        """
        self.beams = []
        for file in beam_list:
            beam = model.BeamCutout(fits_file=file, conf=conf)
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
        if poly_order == self.poly_order:
            return None
        
        self.poly_order = poly_order
        if flat is None:
            flat = self.flat_flam
                    
        ### Polynomial continuum arrays        
        self.A_poly = np.array([self.xpf**order*flat
                                      for order in range(poly_order+1)])
        
        self.n_poly = poly_order + 1
    
    def compute_model(self, id=None, spectrum_1d=None):
        """TBD
        """
        for beam in self.beams:
            beam.beam.compute_model(id=id, spectrum_1d=spectrum_1d)
            
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
            temp = templates[key].zscale(z, 1.)
            spectrum_1d = [temp.wave, temp.flux]
                
            i0 = 0            
            for ib in range(self.N):
                beam = self.beams[ib]
                lam_beam = beam.beam.lam_beam
                if ((temp.wave[0] > lam_beam[-1]) | 
                    (temp.wave[-1] < lam_beam[0])):
                    tmodel = 0.
                else:
                    tmodel = beam.compute_model(spectrum_1d=spectrum_1d, 
                                                in_place=False)
                
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
                    out = np.linalg.lstsq(Ax,y)                         
                except:
                    print A.min(), Ax.min(), self.fit_mask.sum(), y.min()
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
        chi2 = np.sum(((self.scif - modelf)**2*self.ivarf)[self.fit_mask])
        
        return A, out_coeffs, chi2, modelf
    
    def load_templates(self, fwhm=400, line_complexes=True, stars=False):
        """TBD
        """
        
        if stars:
            #templates = glob.glob('%s/templates/Pickles_stars/ext/*dat' %(os.getenv('GRIZLI')))
            # templates = []
            # for t in 'obafgkmrw':
            #     templates.extend( glob.glob('%s/templates/Pickles_stars/ext/uk%s*dat' %(os.getenv('GRIZLI'), t)))
            # templates.extend(glob.glob('%s/templates/SPEX/spex-prism-M*txt' %(os.getenv('GRIZLI'))))
            # templates.extend(glob.glob('%s/templates/SPEX/spex-prism-[LT]*txt' %(os.getenv('GRIZLI'))))
            # 
            # #templates = glob.glob('/Users/brammer/Downloads/templates/spex*txt')
            # templates = glob.glob('bpgs/*ascii')
            # info = catIO.Table('bpgs/bpgs.info')
            # type = np.array([t[:2] for t in info['type']])
            # templates = []
            # for t in 'OBAFGKM':
            #     test = type == '-%s' %(t)
            #     so = np.argsort(info['type'][test])
            #     templates.extend(info['file'][test][so])
            #             
            # temp_list = OrderedDict()
            # for temp in templates:
            #     data = np.loadtxt('bpgs/'+temp, unpack=True)
            #     #data[0] *= 1.e4 # spex
            #     scl = np.interp(5500., data[0], data[1])
            #     name = os.path.basename(temp)
            #     ix = info['file'] == temp
            #     name='%5s %s' %(info['type'][ix][0][1:], temp.split('.as')[0])
            #     temp_list[name] = utils.SpectrumTemplate(wave=data[0],
            #                                              flux=data[1]/scl)
            # 
            # np.save('stars_bpgs.npy', [temp_list])
            
            temp_list = np.load(os.path.join(os.getenv('GRIZLI'), 
                                             'templates/stars.npy'))[0]
            return temp_list
            
        ## Intermediate and very old
        # templates = ['templates/EAZY_v1.0_lines/eazy_v1.0_sed3_nolines.dat',  
        #              'templates/cvd12_t11_solar_Chabrier.extend.skip10.dat']     
        templates = ['templates/eazy_intermediate.dat', 
                     'templates/cvd12_t11_solar_Chabrier.dat']
                     
        ## Post starburst
        #templates.append('templates/UltraVISTA/eazy_v1.1_sed9.dat')
        templates.append('templates/post_starburst.dat')
        
        ## Very blue continuum
        #templates.append('templates/YoungSB/erb2010_continuum.dat')
        templates.append('templates/erb2010_continuum.dat')
        
        temp_list = OrderedDict()
        for temp in templates:
            data = np.loadtxt(os.getenv('GRIZLI') + '/' + temp, unpack=True)
            scl = np.interp(5500., data[0], data[1])
            name = os.path.basename(temp)
            temp_list[name] = utils.SpectrumTemplate(wave=data[0],
                                                     flux=data[1]/scl)
        
        ### Emission lines:
        line_wavelengths, line_ratios = utils.get_line_wavelengths()
         
        if line_complexes:
            #line_list = ['Ha+SII', 'OIII+Hb+Ha', 'OII']
            #line_list = ['Ha+SII', 'OIII+Hb', 'OII']
            line_list = ['Ha+NII+SII+SIII+He', 'OIII+Hb', 'OII+Ne']
        else:
            line_list = ['SIII', 'SII', 'Ha', 'NII', 'OI', 'OIII', 'Hb', 'OIIIx', 'Hg', 'Hd', 'NeIII', 'OII']
            #line_list = ['Ha', 'SII']
            
        for li in line_list:
            scl = line_ratios[li]/np.sum(line_ratios[li])
            for i in range(len(scl)):
                line_i = utils.SpectrumTemplate(wave=line_wavelengths[li][i], 
                                          flux=None, fwhm=fwhm, velocity=True)
                                          
                if i == 0:
                    line_temp = line_i*scl[i]
                else:
                    line_temp = line_temp + line_i*scl[i]
            
            temp_list['line %s' %(li)] = line_temp
                                     
        return temp_list
    
    def fit_stars(self, poly_order=1, fitter='nnls', fit_background=True, 
                  verbose=True, make_figure=True, zoom=None,
                  delta_chi2_threshold=0.004, zr=0, dz=0, fwhm=0, prior=None):
        """TBD
        """
        
        ## Polynomial fit
        out = self.fit_at_z(z=0., templates={}, fitter='lstsq',
                            poly_order=3,
                            fit_background=fit_background)
        
        A, coeffs, chi2_poly, model_2d = out
        
        ### Star templates
        templates = self.load_templates(fwhm=fwhm, stars=True)
        NTEMP = len(templates)

        key = templates.keys()[0]
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
        for i, key in enumerate(templates):
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
                print utils.no_newline + '  %s %9.1f (%s)' %(key, chi2[i], best)
        
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
        xspec = np.arange(0.3, 1.8, 0.05)-1
        scale_coeffs = coeffs_full[self.N*self.fit_bg:  
                                  self.N*self.fit_bg+self.n_poly]
                                  
        yspec = [xspec**o*scale_coeffs[o] for o in range(self.poly_order+1)]
        model1d = utils.SpectrumTemplate((xspec+1)*1.e4, 
                                         np.sum(yspec, axis=0))

        cont1d = model1d*1
        
        i0 = self.fit_bg*self.N + self.n_poly
        
        line_flux = OrderedDict()
        fscl = self.beams[0].beam.total_flux/1.e-17

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
                     delta_chi2_threshold=0.004, zoom=True):
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
        templates = self.load_templates(fwhm=fwhm, stars=stars)
        NTEMP = len(templates)
        
        out = self.fit_at_z(z=0., templates=templates, fitter=fitter,
                            poly_order=poly_order,
                            fit_background=fit_background)
                            
        A, coeffs, chi2, model_2d = out
        
        chi2 = np.zeros(NZ)
        coeffs = np.zeros((NZ, coeffs.shape[0]))
        
        chi2min = 1e30
        iz = 0
        for i in xrange(NZ):
            out = self.fit_at_z(z=zgrid[i], templates=templates,
                                fitter=fitter, poly_order=poly_order,
                                fit_background=fit_background)
            
            A, coeffs[i,:], chi2[i], model_2d = out
            if chi2[i] < chi2min:
                iz = i
                chi2min = chi2[i]

            if verbose:                    
                print utils.no_newline + '  %.4f %9.1f (%.4f)' %(zgrid[i], chi2[i], zgrid[iz])
        
        print 'First iteration: z_best=%.4f\n' %(zgrid[iz])
            
        # peaks
        import peakutils
        # chi2nu = (chi2.min()-chi2)/self.DoF
        # indexes = peakutils.indexes((chi2nu+delta_chi2_threshold)*(chi2nu > -delta_chi2_threshold), thres=0.3, min_dist=20)
        
        chi2_rev = (chi2_poly - chi2)/self.DoF
        if chi2_poly < chi2.min():
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
            for i in xrange(NZOOM):
                out = self.fit_at_z(z=zgrid_zoom[i], templates=templates,
                                    fitter=fitter, poly_order=poly_order,
                                    fit_background=fit_background)

                A, coeffs_zoom[i,:], chi2_zoom[i], model_2d = out
                if chi2_zoom[i] < chi2min:
                    chi2min = chi2_zoom[i]
                    iz = i
                
                if verbose:
                    print utils.no_newline+'- %.4f %9.1f (%.4f) %d/%d' %(zgrid_zoom[i], chi2_zoom[i], zgrid_zoom[iz], i+1, NZOOM)
        
            zgrid = np.append(zgrid, zgrid_zoom)
            chi2 = np.append(chi2, chi2_zoom)
            coeffs = np.append(coeffs, coeffs_zoom, axis=0)
        
        so = np.argsort(zgrid)
        zgrid = zgrid[so]
        chi2 = chi2[so]
        coeffs=coeffs[so,:]

        if prior is not None:
            interp_prior = np.interp(zgrid, prior[0], prior[1])
            chi2 += interp_prior
        else:
            interp_prior = None
            
        print ' Zoom iteration: z_best=%.4f\n' %(zgrid[np.argmin(chi2)])
        
        ### Best redshift
        if not stars:
            templates = self.load_templates(line_complexes=False, fwhm=fwhm)
        
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
        
        ## Covariance matrix for line flux uncertainties
        ok_temp = (np.sum(A, axis=1) > 0) & (coeffs_full != 0)
        Ax = A[:, self.fit_mask][ok_temp,:].T
        Ax *= np.sqrt(self.ivarf[self.fit_mask][:, np.newaxis])
        try:
            covar = np.matrix(np.dot(Ax.T, Ax)).I
            covard = np.sqrt(covar.diagonal())
        except:
            covard = np.zeros(ok_temp.sum())#-1.
            
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
        xspec = np.arange(0.3, 1.8, 0.05)-1
        scale_coeffs = coeffs_full[self.N*self.fit_bg:  
                                  self.N*self.fit_bg+self.n_poly]
                                  
        yspec = [xspec**o*scale_coeffs[o] for o in range(self.poly_order+1)]
        model1d = utils.SpectrumTemplate((xspec+1)*1.e4, 
                                         np.sum(yspec, axis=0))
        
        # model1d = SpectrumTemplate(wave=self.beam.lam, 
        #                 flux=np.dot(self.y_poly.T, 
        #                       coeffs_full[self.n_bg:self.n_poly+self.n_bg]))
        
        cont1d = model1d*1
        
        i0 = self.fit_bg*self.N + self.n_poly
        
        line_flux = OrderedDict()
        fscl = self.beams[0].beam.total_flux/1.e-17
        line1d = OrderedDict()
        for i, key in enumerate(templates.keys()):
            temp_i = templates[key].zscale(zbest, coeffs_full[i0+i])
            model1d += temp_i
            if not key.startswith('line'):
                cont1d += temp_i
            else:
                line1d[key.split()[1]] = temp_i
                line_flux[key.split()[1]] = np.array([coeffs_full[i0+i]*fscl, 
                                             line_flux_err[i0+i]*fscl])
                
                        
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
        fit_data['line_flux'] = line_flux
        #fit_data['templates_full'] = templates
        fit_data['model_cont'] = model_continuum
        fit_data['model1d'] = model1d
        fit_data['cont1d'] = cont1d
        fit_data['line1d'] = line1d
        
        #return fit_data
        
        fig = None   
        if make_figure:
            fig = self.show_redshift_fit(fit_data)
            #fig.savefig('fit.pdf')
            
        return fit_data, fig
    
    def show_redshift_fit(self, fit_data, plot_flambda=True):
        """TBD
        """
        fig = plt.figure(figsize=[8,5])
        ax = fig.add_subplot(211)
        
        ax.plot(fit_data['zgrid'], fit_data['chi2']/self.DoF)
        ax.set_xlabel('z')
        ax.set_ylabel(r'$\chi^2_\nu$, $\nu$=%d' %(self.DoF))
        
        c2min = fit_data['chi2'].min()
        for delta in [1,4,9]:
            ax.plot(fit_data['zgrid'],
                    fit_data['zgrid']*0.+(c2min+delta)/self.DoF, 
                    color='%.2f' %(1-delta*1./10))
        
        ax.plot(fit_data['zgrid'], (fit_data['chi2']*0+fit_data['chi_poly'])/self.DoF, color='b', linestyle='--', alpha=0.8)
        
        ax.set_xlim(fit_data['zgrid'].min(), fit_data['zgrid'].max())
        ax.grid()        
        ax.set_title(r'ID = %d, $z_\mathrm{grism}$=%.4f' %(self.beams[0].id, 
                                               fit_data['zbest']))
                                               
        ax = fig.add_subplot(212)
        
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
            ivar = beam.ivar
               
            wave, flux, err = beam.beam.optimal_extract(clean, 
                                                        ivar=ivar)
            
            mwave, mflux, merr = beam.beam.optimal_extract(line_fit[ib]-bg_i, 
                                                        ivar=ivar)
            
            wave, fflux, ferr = beam.beam.optimal_extract(beam.flat_flam.reshape(beam.beam.sh_beam), ivar=ivar)
                
            # wave, flux, err = beam.beam.trace_extract(clean, 
            #                                             ivar=ivar, r=10)
            # 
            # mwave, mflux, merr = beam.beam.trace_extract(line_fit[ib]-bg_i, 
            #                                             ivar=ivar, r=10)
            
            if plot_flambda:
                ok = beam.beam.sensitivity > 0.1*beam.beam.sensitivity.max()
                #wave = wave[ok]
                #flux = (flux/beam.beam.sensitivity)[ok]
                #err = (err/beam.beam.sensitivity)[ok]
                #mflux = (mflux/beam.beam.sensitivity)[ok]
                
                wave = wave[ok]
                flux  = (flux*beam.beam.total_flux/1.e-17/fflux)[ok]*beam.beam.scale
                err   = (err*beam.beam.total_flux/1.e-17/fflux)[ok]
                mflux = (mflux*beam.beam.total_flux/1.e-17/fflux)[ok]*beam.beam.scale
                
                ylabel = r'$f_\lambda$'
            else:
                ylabel = 'flux (e-/s)'
            
            scl_region = np.isfinite(mflux) 
            if scl_region.sum() == 0:
                continue
                
            try:
                okerr = np.isfinite(err)
                med_err = np.median(err[okerr])
                
                ymax = np.maximum(ymax, 
                            (mflux[scl_region][2:-2] + med_err).max())
                ymin = np.minimum(ymin, 
                            (mflux[scl_region][2:-2] - med_err).min())
            except:
                continue
                
            ax.errorbar(wave/1.e4, flux, err, alpha=0.15+0.2*(self.N <= 2), linestyle='None', marker='.', color='%.2f' %(ib*0.5/self.N), zorder=1)
            ax.plot(wave/1.e4, mflux, color='r', alpha=0.5, zorder=3)
            
            grism = beam.grism.filter
            #for grism in grisms:
            wfull[grism] = np.append(wfull[grism], wave)
            ffull[grism] = np.append(ffull[grism], flux)
            efull[grism] = np.append(efull[grism], err)
        
        cp = {'G800L':(0.0, 0.4470588235294118, 0.6980392156862745),
              'G102':(0.0, 0.6196078431372549, 0.45098039215686275),
              'G141':(0.8352941176470589, 0.3686274509803922, 0.0),
              'none':(0.8, 0.4745098039215686, 0.6549019607843137),
              'GRISM':'k'}
        
        for grism in grisms:                        
            if self.Ngrism[grism] > 1:
                ## binned
                okb = (np.isfinite(wfull[grism]) & np.isfinite(ffull[grism]) &
                                   np.isfinite(efull[grism]))
                                   
                so = np.argsort(wfull[grism][okb])
                var = efull[grism]**2
            
                N = int(np.ceil(self.Ngrism[grism]/2)*2)*2
                kernel = np.ones(N, dtype=float)/N
                fbin = nd.convolve(ffull[grism][okb][so], kernel)[N/2::N]
                wbin = nd.convolve(wfull[grism][okb][so], kernel)[N/2::N]
                vbin = nd.convolve(var[okb][so], kernel**2)[N/2::N]
                ax.errorbar(wbin/1.e4, fbin, np.sqrt(vbin), alpha=0.8,
                            linestyle='None', marker='.', color=cp[grism], zorder=2)
                
        ax.set_ylim(ymin - 0.1*np.abs(ymax), 1.1*ymax)
        
        xmin, xmax = 1.e5, 0
        limits = {'G800L':[0.545, 1.02],
                   'G102':[0.77, 1.18],
                   'G141':[1.06, 1.73],
                   'GRISM':[0.98, 1.98]}
        
        for g in limits:
            if g in grisms:
                xmin = np.minimum(xmin, limits[g][0])
                xmax = np.maximum(xmin, limits[g][1])
                #print g, xmin, xmax
                
        ax.set_xlim(xmin, xmax)
        
        ### Label
        ax.text(0.03, 1.03, ('%s' %(self.Ngrism)).replace('\'','').replace('{','').replace('}',''), ha='left', va='bottom', transform=ax.transAxes, fontsize=10)
        
        ax.plot(wave/1.e4, wave/1.e4*0., linestyle='--', color='k')
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(ylabel)
        
        fig.tight_layout(pad=0.1)
        return fig
    
    def redshift_fit_twod_figure(self, fit, spatial_scale=1, dlam=46., NY=10,
                                 **kwargs):
        """Make figure of 2D spectrum
        
        TBD
        """        
        ### xlimits        
        xmin, xmax = 1.e5, 0
        limits = {'G800L':[0.545, 1.02],
                   'G102':[0.77, 1.18],
                   'G141':[1.06, 1.73],
                   'GRISM':[0.98, 1.98]}
        
        for g in limits:
            if g in self.Ngrism:
                xmin = np.minimum(xmin, limits[g][0])
                xmax = np.maximum(xmin, limits[g][1])
        
        hdu_sci = drizzle_2d_spectrum(self.beams, ds9=None, NY=NY,
                                      spatial_scale=spatial_scale, dlam=dlam, 
                                      kernel='point', pixfrac=0.6,
                                      wlimit=[xmin, xmax])
                                  
        ### Continuum model
        cont = self.reshape_flat(fit['model_cont'])        
        hdu_con = drizzle_2d_spectrum(self.beams, data=cont, ds9=None, NY=NY,
                                      spatial_scale=spatial_scale, dlam=dlam, 
                                      kernel='point', pixfrac=0.6,
                                      wlimit=[xmin, xmax])
        
        full = self.reshape_flat(fit['model_full'])        
        hdu_full = drizzle_2d_spectrum(self.beams, data=full, ds9=None, NY=NY,
                                      spatial_scale=spatial_scale, dlam=dlam, 
                                      kernel='point', pixfrac=0.6,
                                      wlimit=[xmin, xmax])
        
        
        vmax = np.maximum(1.1*np.percentile(hdu_full[1].data, 98), 0.04)
        #print 'VMAX: %f\n\n' %vmax
        
        sh = hdu_full[1].data.shape
        extent = [hdu_full[0].header['WMIN'], hdu_full[0].header['WMAX'],
                  0, sh[0]]
                  
        fig = plt.figure(figsize=[8,3.5])
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
        label = 'ID=%6d, z=%.4f' %(self.beams[0].id, fit['zbest'])
        fig.axes[-1].text(0.97, -0.27, label, ha='right', va='top',
                          transform=fig.axes[-1].transAxes, fontsize=10)
        
        label2 = ('%s' %(self.Ngrism)).replace('\'', '').replace('{', '').replace('}', '')
        fig.axes[-1].text(0.03, -0.27, label2, ha='left', va='top',
                          transform=fig.axes[-1].transAxes, fontsize=10)
                
        hdu_sci.append(hdu_con[1])
        hdu_sci[-1].name = 'CONTINUUM'
        hdu_sci.append(hdu_full[1])
        hdu_sci[-1].name = 'FULL'
        
        return fig, hdu_sci
    
    def drizzle_fit_lines(self, fit, pline, force_line=['Ha', 'OIII', 'Hb', 'OII'], save_fits=True, mask_lines=True, mask_sn_limit=3, wcs=None):
        """
        TBD
        """
        line_wavelengths, line_ratios = utils.get_line_wavelengths()
        hdu_full = []
        saved_lines = []
        for line in fit['line_flux']:
            line_flux, line_err = fit['line_flux'][line]
            if line_flux == 0:
                continue

            if (line_flux/line_err > 7) | (line in force_line):
                print 'Drizzle line -> %4s (%.2f %.2f)' %(line, line_flux,
                                                         line_err)

                line_wave_obs = line_wavelengths[line][0]*(1+fit['zbest'])
                
                if mask_lines:
                    for beam in self.beams:
                        cont = fit['cont1d']
                        beam.compute_model(spectrum_1d=[cont.wave, cont.flux])
                        
                        beam.oivar = beam.ivar*1
                        lam = beam.beam.lam_beam
                        
                        ### another idea, compute a model for the line itself
                        ### and mask relatively "contaminated" pixels from 
                        ### other lines
                        lm = fit['line1d'][line]
                        if ((lm.wave.max() < lam.min()) | 
                            (lm.wave.min() > lam.max())):
                            continue
                        
                        sp = [lm.wave, lm.flux]
                        m = beam.compute_model(spectrum_1d=sp, 
                                               in_place=False)
                        lmodel = m.reshape(beam.beam.sh_beam)
                        if lmodel.max() == 0:
                            continue
                            
                        for l in fit['line1d']:
                            lf, le = fit['line_flux'][l]
                            ### Don't mask if the line missing or undetected
                            if (lf == 0) | (lf < mask_sn_limit*le):
                                continue
                                
                            if l != line:
                                lm = fit['line1d'][l]
                                sp = [lm.wave, lm.flux]
                                if ((lm.wave.max() < lam.min()) | 
                                    (lm.wave.min() > lam.max())):
                                    continue
                                    
                                m = beam.compute_model(spectrum_1d=sp, 
                                                       in_place=False)
                                lcontam = m.reshape(beam.beam.sh_beam)
                                if lcontam.max() == 0:
                                    #print beam.grism.parent_file, l
                                    continue
                                    
                                beam.ivar[lcontam > mask_sn_limit*lmodel] *= 0

                hdu = drizzle_to_wavelength(self.beams, wcs=wcs, ra=self.ra, 
                                            dec=self.dec, wave=line_wave_obs,
                                            **pline)
                
                if mask_lines:
                    for beam in self.beams:
                        beam.ivar = beam.oivar*1
                        delattr(beam, 'oivar')
                        
                hdu[0].header['REDSHIFT'] = (fit['zbest'], 'Redshift used')
                for e in [3,4,5,6]:
                    hdu[e].header['EXTVER'] = line
                    hdu[e].header['REDSHIFT'] = (fit['zbest'], 
                                                 'Redshift used')
                    hdu[e].header['RESTWAVE'] = (line_wavelengths[line][0], 
                                                 'Line rest wavelength')

                saved_lines.append(line)

                if len(hdu_full) == 0:
                    hdu_full = hdu
                    hdu_full[0].header['NUMLINES'] = (1, 
                                               "Number of lines in this file")
                else:
                    hdu_full.extend(hdu[3:])
                    hdu_full[0].header['NUMLINES'] += 1 

                li = hdu_full[0].header['NUMLINES']
                hdu_full[0].header['LINE%03d' %(li)] = line
                hdu_full[0].header['FLUX%03d' %(li)] = (line_flux, 
                                                'Line flux, 1e-17 erg/s/cm2')
                hdu_full[0].header['ERR%03d' %(li)] = (line_err, 
                                        'Line flux err, 1e-17 erg/s/cm2')

        if len(hdu_full) > 0:
            hdu_full[0].header['HASLINES'] = (' '.join(saved_lines), 
                                              'Lines in this file')

            if save_fits:
                hdu_full.writeto('%s_zfit_%05d.line.fits' %(self.group_name,
                                                            self.id),
                                 clobber=True, output_verify='fix')
        
        return hdu_full
        
    def run_full_diagnostics(self, pzfit={}, pspec2={}, pline={}, 
                      force_line=['Ha', 'OIII', 'Hb'], GroupFLT=None,
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
        fig2, hdu2 = self.redshift_fit_twod_figure(**spec_in)#, kwargs=spec2) #dlam=dlam, spatial_scale=spatial_scale, NY=NY)
        
        ### Update master model
        if GroupFLT is not None:
            try:
                ix = GroupFLT.catalog['NUMBER'] == self.beams[0].id
                mag = GroupFLT.catalog['MAG_AUTO'][ix].data[0]
            except:
                mag = 22
                
            sp = fit['cont1d']
            GroupFLT.compute_single_model(id, mag=mag, size=None, store=False,
                                          spectrum_1d=[sp.wave, sp.flux],
                                          get_beams=None, in_place=True)
        
        ## 2D lines to drizzle
        hdu_full = self.drizzle_fit_lines(fit, pline, force_line=force_line, 
                                     save_fits=True)
        
        # ra, dec = self.beams[0].get_sky_coords()
        # 
        # line_wavelengths, line_ratios = utils.get_line_wavelengths()
        # hdu_full = []
        # saved_lines = []
        # for line in fit['line_flux']:
        #     line_flux, line_err = fit['line_flux'][line]
        #     if line_flux == 0:
        #         continue
        #         
        #     if (line_flux/line_err > 7) | (line in force_line):
        #         print 'Drizzle line -> %s (%.2f %.2f)' %(line, line_flux,
        #                                                  line_err)
        #                                                  
        #         line_wave_obs = line_wavelengths[line][0]*(1+fit['zbest'])
        #         
        #         hdu = drizzle_to_wavelength(self.beams, ra=ra, dec=dec, 
        #                                   wave=line_wave_obs, **pline)
        #         
        #         hdu[0].header['REDSHIFT'] = (fit['zbest'], 'Redshift used')
        #         for e in [3,4,5]:
        #             hdu[e].header['EXTVER'] = line
        #             hdu[e].header['REDSHIFT'] = (fit['zbest'], 
        #                                          'Redshift used')
        #             hdu[e].header['RESTWAVE'] = (line_wavelengths[line][0], 
        #                                          'Line rest wavelength')
        #         
        #         saved_lines.append(line)
        #             
        #         if len(hdu_full) == 0:
        #             hdu_full = hdu
        #             hdu_full[0].header['NUMLINES'] = (1, 
        #                                        "Number of lines in this file")
        #         else:
        #             hdu_full.extend(hdu[3:])
        #             hdu_full[0].header['NUMLINES'] += 1 
        #         
        #         li = hdu_full[0].header['NUMLINES']
        #         hdu_full[0].header['LINE%03d' %(li)] = line
        #         hdu_full[0].header['FLUX%03d' %(li)] = (line_flux, 
        #                                         'Line flux, 1e-17 erg/s/cm2')
        #         hdu_full[0].header['ERR%03d' %(li)] = (line_err, 
        #                                 'Line flux err, 1e-17 erg/s/cm2')
        
        # if len(hdu_full) > 0:        
        #     hdu_full.writeto('%s_zfit_%05d.line.fits' %(self.group_name,
        #                                                 self.id),
        #                      clobber=True, output_verify='fix')
        
        fit['id'] = self.id
        fit['fit_bg'] = self.fit_bg
        fit['grism_files'] = [b.grism.parent_file for b in self.beams]
        for item in ['A','coeffs','model_full','model_cont']:
            if item in fit:
                p = fit.pop(item)
            
        #p = fit.pop('coeffs')
        
        np.save('%s_zfit_%05d.fit.npy' %(self.group_name, self.id), [fit])
            
        fig.savefig('%s_zfit_%05d.png' %(self.group_name, self.id))
        
        fig2.savefig('%s_zfit_%05d.2D.png' %(self.group_name, self.id))
        hdu2.writeto('%s_zfit_%05d.2D.fits' %(self.group_name, self.id), clobber=True, output_verify='fix')
        
        label = '# id ra dec zbest '
        data = '%7d %.6f %.6f %.5f' %(self.id, self.ra, self.dec,
                                      fit['zbest'])
        
        for grism in ['G800L', 'G102', 'G141', 'GRISM']:
            label += ' N%s' %(grism)
            if grism in self.Ngrism:
                data += ' %2d' %(self.Ngrism[grism])
            else:
                data += ' %2d' %(0)
                
        label += ' chi2 DoF ' 
        data += ' %14.1f %d ' %(fit['chibest'], self.DoF)
        
        for line in ['SII', 'Ha', 'OIII', 'Hb', 'Hg', 'OII']:
            label += ' %s %s_err' %(line, line)
            if line in fit['line_flux']:
                flux = fit['line_flux'][line][0]
                err =  fit['line_flux'][line][1]
                data += ' %10.3e %10.3e' %(flux, err)
        
        fp = open('%s_zfit_%05d.fit.dat' %(self.group_name, self.id),'w')
        fp.write(label+'\n')
        fp.write(data+'\n')
        fp.close()
        
        fp = open('%s_zfit_%05d.beams.dat' %(self.group_name, self.id),'w')
        fp.write('# file filter origin_x origin_y size pad bg\n')
        for ib, beam in enumerate(self.beams):
            data = '%40s %s %5d %5d %5d %5d' %(beam.grism.parent_file, 
                                          beam.grism.filter,
                                          beam.direct.origin[0],
                                          beam.direct.origin[1],
                                          beam.direct.sh[0], 
                                          beam.direct.pad)
            if self.fit_bg:
                data += ' %8.4f' %(fit['coeffs_full'][ib])
            else:
                data += ' %8.4f' %(0.0)
            
            fp.write(data + '\n')
        
        fp.close()
                                      
        ## Save figures
        plt_status = plt.rcParams['interactive']
        if not plt_status:
            plt.close(fig)
            plt.close(fig2)
            
        return fit, fig, fig2, hdu2, hdu_full
    
    def fit_trace_shift(self, split_groups=True, max_shift=5, tol=1.e-2):
        """TBD
        """
        import scipy.optimize
        
        if split_groups:
            roots = np.unique([b.grism.parent_file.split('_')[0] for b in self.beams])
            indices = []
            for root in roots:
                idx = [i for i in range(self.N) if self.beams[i].grism.parent_file.startswith(root)]
                indices.append(idx)
            
        else:
            indices = [[i] for i in range(self.N)]
        
        shifts = np.zeros(len(indices))
        bounds = np.array([[-max_shift,max_shift]]*len(indices))
        
        args = (self, indices, 0)
        out = scipy.optimize.minimize(self.eval_trace_shift, shifts, bounds=bounds, args=args, method='Powell', tol=tol)
        
        self.eval_trace_shift(out.x, *args)
        
        ### Rest model profile for optimal extractions
        for b in self.beams:
            if hasattr(b.beam, 'optimal_profile'):
                delattr(b.beam, 'optimal_profile')
            
        return out.x
        
    @staticmethod
    def eval_trace_shift(shifts, self, indices, poly_order):
        """TBD
        """
        for il, l in enumerate(indices):
            for i in l:
                self.beams[i].beam.add_ytrace_offset(shifts[il])
                self.beams[i].compute_model()

        self.flat_flam = np.hstack([b.beam.model.flatten() for b in self.beams])
        self.poly_order=-1
        self.init_poly_coeffs(poly_order=poly_order)

        self.fit_bg = False
        A = self.A_poly*1
        ok_temp = np.sum(A, axis=1) != 0  
        out_coeffs = np.zeros(A.shape[0])

        y = self.scif
        out = np.linalg.lstsq(A.T,y)                         
        lstsq_coeff, residuals, rank, s = out
        coeffs = lstsq_coeff

        out_coeffs = np.zeros(A.shape[0])
        out_coeffs[ok_temp] = coeffs
        modelf = np.dot(out_coeffs, A)
        chi2 = np.sum(((self.scif - modelf)**2*self.ivarf)[self.fit_mask])

        print shifts, chi2/self.DoF
        return chi2/self.DoF    
            
def get_redshift_fit_defaults():
    """TBD
    """
    pzfit_def = dict(zr=[0.5, 1.6], dz=[0.005, 0.0004], fwhm=0,
                 poly_order=0, fit_background=True,
                 delta_chi2_threshold=0.004, fitter='nnls')
    
    pspec2_def = dict(dlam=0, spatial_scale=1, NY=20)
    pline_def = dict(size=20, pixscale=0.1, pixfrac=0.2, kernel='square', 
                     fcontam=0.05)

    return pzfit_def, pspec2_def, pline_def
                
def drizzle_2d_spectrum(beams, data=None, wlimit=[1.05, 1.75], dlam=50, 
                        spatial_scale=1, NY=10, pixfrac=0.6, kernel='square',
                        convert_to_flambda=True, fcontam=0.2,
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
    
    ds9: `pyds9.DS9`
        Show intermediate steps of the drizzling
    
    Returns
    -------
    hdu : `~astropy.io.fits.HDUList`
        FITS HDUList with the drizzled 2D spectrum and weight arrays
        
    """
    from drizzlepac.astrodrizzle import adrizzle
    
    NX = int(np.round(np.diff(wlimit)[0]*1.e4/dlam))/2
    center = np.mean(wlimit)*1.e4
    out_header, output_wcs = utils.make_spectrum_wcsheader(center_wave=center,
                                 dlam=dlam, NX=NX, 
                                 spatial_scale=spatial_scale, NY=NY)
                                 
    sh = (out_header['NAXIS2'], out_header['NAXIS1'])
    
    outsci = np.zeros(sh, dtype=np.float32)
    outwht = np.zeros(sh, dtype=np.float32)
    outctx = np.zeros(sh, dtype=np.int32)
    
    if data is None:
        data = []
        for i, beam in enumerate(beams):
            ### Contamination-subtracted
            beam_data = beam.grism.data['SCI'] - beam.contam 
            data.append(beam_data)
            
    for i, beam in enumerate(beams):
        ## Get specific WCS for each beam
        beam_header, beam_wcs = beam.get_2d_wcs()
        
        # Downweight contamination
        wht = 1/beam.ivar + fcontam*beam.contam
        wht = np.cast[np.float32](1/wht)
        wht[~np.isfinite(wht)] = 0.
        
        data_i = data[i]*1.
        if convert_to_flambda:
            #data_i *= convert_to_flambda/beam.beam.sensitivity
            #wht *= (beam.beam.sensitivity/convert_to_flambda)**2
            
            scl = convert_to_flambda*beam.beam.total_flux/1.e-17
            scl *= 1./beam.flat_flam.reshape(beam.beam.sh_beam).sum(axis=0)
            #scl = convert_to_flambda/beam.beam.sensitivity
            
            data_i *= scl
            wht *= (1/scl)**2
            
            wht[~np.isfinite(data_i+scl)] = 0
            data_i[~np.isfinite(data_i+scl)] = 0
        
        ###### Go drizzle
        
        ### Contamination-cleaned
        adrizzle.do_driz(data_i, beam_wcs, wht, output_wcs, 
                         outsci, outwht, outctx, 1., 'cps', 1,
                         wcslin_pscale=1.0, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=None)
        
        if ds9 is not None:
            ds9.view(outsci/output_wcs.pscale**2, header=out_header)
    
    ### Correct for drizzle scaling
    outsci /= output_wcs.pscale**2
    
    p = pyfits.PrimaryHDU()
    p.header['ID'] = (beams[0].id, 'Object ID')
    p.header['WMIN'] = (wlimit[0], 'Minimum wavelength')
    p.header['WMAX'] = (wlimit[1], 'Maximum wavelength')
    p.header['DLAM'] = (dlam, 'Delta wavelength')
    
    p.header['PIXFRAC'] = (pixfrac, 'Drizzle PIXFRAC')
    p.header['DRIZKRNL'] = (kernel, 'Drizzle kernel')
    
    p.header['NINPUT'] = (len(beams), 'Number of drizzled beams')
    for i, beam in enumerate(beams):
        p.header['FILE%04d' %(i+1)] = (beam.grism.parent_file, 
                                             'Parent filename')
        p.header['GRIS%04d' %(i+1)] = (beam.grism.filter, 
                                             'Beam grism element')
        
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
        
    ds9 : `pyds9.DS9`, optional
        Display each step of the drizzling to an open DS9 window
    
    Returns
    -------
    hdu : `~astropy.io.fits.HDUList`
        FITS HDUList with the drizzled thumbnail, line and continuum 
        cutouts.
    """
    from drizzlepac.astrodrizzle import adrizzle
    
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
    sh = (header['NAXIS1'], header['NAXIS2'])
    
    outsci = np.zeros(sh, dtype=np.float32)
    outwht = np.zeros(sh, dtype=np.float32)
    outctx = np.zeros(sh, dtype=np.int32)

    coutsci = np.zeros(sh, dtype=np.float32)
    coutwht = np.zeros(sh, dtype=np.float32)
    coutctx = np.zeros(sh, dtype=np.int32)

    xoutsci = np.zeros(sh, dtype=np.float32)
    xoutwht = np.zeros(sh, dtype=np.float32)
    xoutctx = np.zeros(sh, dtype=np.int32)

    doutsci = np.zeros(sh, dtype=np.float32)
    doutwht = np.zeros(sh, dtype=np.float32)
    doutctx = np.zeros(sh, dtype=np.int32)
    
    ## Loop through beams and run drizzle
    for i, beam in enumerate(beams):
        ## Get specific wavelength WCS for each beam
        beam_header, beam_wcs = beam.get_wavelength_wcs(wave)
        ## Make sure CRPIX set correctly for the SIP header
        for j in [0,1]: 
            if beam_wcs.sip is not None:
                beam_wcs.sip.crpix[j] = beam_wcs.wcs.crpix[j]
            if beam.direct.wcs.sip is not None:
                beam.direct.wcs.sip.crpix[j] = beam.direct.wcs.wcs.crpix[j]
        
        beam_data = beam.grism.data['SCI'] - beam.contam 
        beam_continuum = beam.model*1
        
        # Downweight contamination
        if fcontam > 0:
            wht = 1/beam.ivar + fcontam*beam.contam
            wht = np.cast[np.float32](1/wht)
            wht[~np.isfinite(wht)] = 0.
        else:
            wht = beam.ivar*1
        
        ### Convert to f_lambda integrated line fluxes: 
        ###     (Inverse of the aXe sensitivity) x (size of pixel in \AA)
        sens = np.interp(wave, beam.beam.lam, beam.beam.sensitivity, 
                         left=0, right=0)
        
        dlam = np.interp(wave, beam.beam.lam[1:], np.diff(beam.beam.lam))
        # 1e-17 erg/s/cm2 #, scaling closer to e-/s
        sens *= 1./dlam
        
        if sens == 0:
            continue
        else:
            wht *= sens**2
            beam_data /= sens
            beam_continuum /= sens
        
        ###### Go drizzle
        
        ### Contamination-cleaned
        adrizzle.do_driz(beam_data, beam_wcs, wht, output_wcs, 
                         outsci, outwht, outctx, 1., 'cps', 1,
                         wcslin_pscale=beam.grism.wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=None)
        
        ### Continuum
        adrizzle.do_driz(beam_continuum, beam_wcs, wht, output_wcs, 
                         coutsci, coutwht, coutctx, 1., 'cps', 1, 
                         wcslin_pscale=beam.grism.wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=None)
        
        ### Contamination
        adrizzle.do_driz(beam.contam, beam_wcs, wht, output_wcs, 
                         xoutsci, xoutwht, xoutctx, 1., 'cps', 1, 
                         wcslin_pscale=beam.grism.wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=None)
        
        ### Direct thumbnail
        if direct_extension == 'REF':
            if beam.direct['REF'] is None:
                direct_extension = 'SCI'
                
        if direct_extension == 'REF':
            thumb = beam.direct['REF']
            thumb_wht = np.cast[np.float32]((thumb != 0)*1)
        else:
            thumb = beam.direct[direct_extension]#/beam.direct.photflam
            thumb_wht = 1./(beam.direct.data['ERR']/beam.direct.photflam)**2
            thumb_wht[~np.isfinite(thumb_wht)] = 0
            
        
        adrizzle.do_driz(thumb, beam.direct.wcs, thumb_wht, output_wcs, 
                         doutsci, doutwht, doutctx, 1., 'cps', 1, 
                         wcslin_pscale=beam.direct.wcs.pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=None)
        
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
    doutwht *= (beams[0].direct.wcs.pscale/output_wcs.pscale)**4
    xoutwht *= (beams[0].direct.wcs.pscale/output_wcs.pscale)**4
    
    ### Make output FITS products
    p = pyfits.PrimaryHDU()
    p.header['ID'] = (beams[0].id, 'Object ID')
    p.header['RA'] = (ra, 'Central R.A.')
    p.header['DEC'] = (dec, 'Central Decl.')
    p.header['PIXFRAC'] = (pixfrac, 'Drizzle PIXFRAC')
    p.header['DRIZKRNL'] = (kernel, 'Drizzle kernel')
    
    p.header['NINPUT'] = (len(beams), 'Number of drizzled beams')
    for i, beam in enumerate(beams):
        p.header['FILE%04d' %(i+1)] = (beam.grism.parent_file, 
                                             'Parent filename')
        p.header['GRIS%04d' %(i+1)] = (beam.grism.filter, 
                                             'Beam grism element')
        
    h = header.copy()
    h['ID'] = (beam.id, 'Object ID')
    h['FILTER'] = (beam.direct.filter, 'Direct image filter')
    thumb_sci = pyfits.ImageHDU(data=doutsci, header=h, name='DSCI')
    thumb_wht = pyfits.ImageHDU(data=doutwht, header=h, name='DWHT')
    #thumb_seg = pyfits.ImageHDU(data=seg_slice, header=h, name='DSEG')
        
    h['FILTER'] = (beam.grism.filter, 'Grism filter')
    h['WAVELEN'] = (wave, 'Central wavelength')
    
    grism_sci = pyfits.ImageHDU(data=outsci-coutsci, header=h, name='LINE')
    grism_cont = pyfits.ImageHDU(data=coutsci, header=h, name='CONTINUUM')
    grism_contam = pyfits.ImageHDU(data=xoutsci, header=h, name='CONTAM')
    grism_wht = pyfits.ImageHDU(data=outwht, header=h, name='LINEWHT')
    
    return pyfits.HDUList([p, thumb_sci, thumb_wht, grism_sci, grism_cont, 
                           grism_contam, grism_wht])
