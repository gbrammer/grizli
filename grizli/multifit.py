import os
import time
import collections
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
    
    fit_info = collections.OrderedDict()
    
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
    bright = (self.catalog['MAG_AUTO'] < 23) & (self.catalog['MAG_AUTO'] > 16)
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
    """TBD
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
                 pad=200, ref_file=None, ref_ext=0, seg_file=None,
                 shrink_segimage=True, verbose=True, cpu_count=0,
                 catalog=''):
        """TBD
        """
        self.N = len(grism_files)
        if len(direct_files) != len(grism_files):
            direct_files = ['']*self.N
        
        self.grism_files = grism_files
        self.direct_files = direct_files
        
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
            print 'Save %s' %(save_file)
            self.FLTs[i].save_full_pickle()
            
            ### Reload initialized data
            self.FLTs[i].load_from_fits(save_file)
            
            
    def extend(self, new, verbose=True):
        """Add another GroupFLT instance to self
        
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
            
    def compute_full_model(self, fit_info, verbose=True, store=False, 
                           cpu_count=0):
        """TBD
        """
        if cpu_count == 0:
            cpu_count = mp.cpu_count()
        
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
    
    def refine_list(self, ids, mags, poly_order=1, ds9=None, verbose=True):
        """TBD
        
        bright = self.catalog['MAG_AUTO'] < 24
        ids = self.catalog['NUMBER'][bright]*1
        mags = self.catalog['MAG_AUTO'][bright]*1
        so = np.argsort(mags)
        
        ids, mags = ids[so], mags[so]
        
        self.refine_list(ids, mags, ds9=ds9)
        
        """
        for id, mag in zip(ids, mags):
            #print id, mag
            self.refine(id, mag=mag, poly_order=poly_order, size=30, ds9=ds9, verbose=verbose)
            
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
            
        xspec = np.arange(0.7, 1.8, 0.05)-1
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
        
class MultiBeam():
    """TBD
    """     
    def __init__(self, beams, fcontam=0.):
        self.N = len(beams)
        
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
        self.scif = np.hstack([b.scif for b in self.beams])
        self.contamf = np.hstack([b.contam.flatten() for b in self.beams])
        
        self.fcontam = fcontam
        if fcontam > 0:
            self.ivarf = 1./(1./self.ivarf + fcontam*self.contamf)
            self.ivarf[~np.isfinite(self.ivarf)] = 0
            
        ### Initialize background fit array
        self.A_bg = np.zeros((self.N, self.Ntot))
        i0 = 0
        for i in range(self.N):
            self.A_bg[i, i0:i0+self.Nflat[i]] = 1.
            i0 += self.Nflat[i]
        
        self.init_poly_coeffs(poly_order=1)
    
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
            
    def init_poly_coeffs(self, poly_order=1):
        """TBD
        """
        ### Already done?
        if poly_order == self.poly_order:
            return None
        
        self.poly_order = poly_order
                
        ### Polynomial continuum arrays        
        self.A_poly = np.array([self.xpf**order*self.flat_flam 
                                      for order in range(poly_order+1)])
        
        self.n_poly = poly_order + 1
    
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
            Ax *= self.ivarf[self.fit_mask][:, np.newaxis]
            
            #print 'xxx lstsq'
            #out = numpy.linalg.lstsq(Ax,y)
            if fitter == 'lstsq':
                y = self.scif[self.fit_mask]
                ### Weight by ivar
                y *= self.ivarf[self.fit_mask]

                out = np.linalg.lstsq(Ax,y)                         
                lstsq_coeff, residuals, rank, s = out
                coeffs = lstsq_coeff
            
            if fitter == 'nnls':
                if fit_background:
                    off = 0.04
                    y = self.scif[self.fit_mask]+off
                    y *= self.ivarf[self.fit_mask]

                    coeffs, rnorm = scipy.optimize.nnls(Ax, y+off)
                    coeffs[:self.N] -= 0.04
                else:
                    y = self.scif[self.fit_mask]
                    y *= self.ivarf[self.fit_mask]
                    
                    coeffs, rnorm = scipy.optimize.nnls(Ax, y)                    
        else:
            Ax = A[:, self.fit_mask][ok_temp,:].T
            y = self.scif[self.fit_mask]
            
            ### Wieght by ivar
            Ax *= self.ivarf[self.fit_mask][:, np.newaxis]
            y *= self.ivarf[self.fit_mask]
            
            clf = sklearn.linear_model.LinearRegression()
            status = clf.fit(Ax, y)
            coeffs = clf.coef_
                
        out_coeffs[ok_temp] = coeffs
        modelf = np.dot(out_coeffs, A)
        chi2 = np.sum(((self.scif - modelf)**2*self.ivarf)[self.fit_mask])
        
        return A, out_coeffs, chi2, modelf
    
    def load_templates(self, fwhm=400, line_complexes=True):
        """TBD
        """
        # templates = ['templates/EAZY_v1.0_lines/eazy_v1.0_sed1_nolines.dat',
        # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed2_nolines.dat',  
        # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed3_nolines.dat',     
        # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed4_nolines.dat',     
        # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed5_nolines.dat',     
        # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed6_nolines.dat',     
        # 'templates/cvd12_t11_solar_Chabrier.extend.dat',     
        # 'templates/dobos11/bc03_pr_ch_z02_ltau07.0_age09.2_av2.5.dat']

        templates = ['templates/EAZY_v1.0_lines/eazy_v1.0_sed3_nolines.dat',  
                     'templates/cvd12_t11_solar_Chabrier.extend.dat']     
        
        templates.append('templates/YoungSB/erb2010_continuum.dat')
        #templates = templates[-1:]
        
        #print 'XXX! templates', templates, '\n'
        
        # templates.extend(['templates/dobos11/SF0_0.emline.hiOIII.txt', 
        #                   'templates/dobos11/SF0_0.emline.loOIII.txt'])
                                  
        temp_list = collections.OrderedDict()
        for temp in templates:
            data = np.loadtxt(os.getenv('GRIZLI') + '/' + temp, unpack=True)
            scl = np.interp(5500., data[0], data[1])
            name = os.path.basename(temp)
            temp_list[name] = utils.SpectrumTemplate(wave=data[0],
                                                             flux=data[1]/scl)
            #plt.plot(temp_list[-1].wave, temp_list[-1].flux, label=temp, alpha=0.5)
            
        line_wavelengths = {} ; line_ratios = {}
        line_wavelengths['Ha'] = [6564.61]; line_ratios['Ha'] = [1.]
        line_wavelengths['Hb'] = [4862.68]; line_ratios['Hb'] = [1.]
        line_wavelengths['Hg'] = [4341.68]; line_ratios['Hg'] = [1.]
        line_wavelengths['Hd'] = [4102.892]; line_ratios['Hd'] = [1.]
        line_wavelengths['OIIIx'] = [4364.436]; line_ratios['OIIIx'] = [1.]
        line_wavelengths['OIII'] = [5008.240, 4960.295]; line_ratios['OIII'] = [2.98, 1]
        line_wavelengths['OIII+Hb'] = [5008.240, 4960.295, 4862.68]; line_ratios['OIII+Hb'] = [2.98, 1, 3.98/8.]
        
        line_wavelengths['OIII+Hb+Ha'] = [5008.240, 4960.295, 4862.68, 6564.61]; line_ratios['OIII+Hb+Ha'] = [2.98, 1, 3.98/10., 3.98/10.*2.86]

        line_wavelengths['OIII+Hb+Ha+SII'] = [5008.240, 4960.295, 4862.68, 6564.61, 6718.29, 6732.67]
        line_ratios['OIII+Hb+Ha+SII'] = [2.98, 1, 3.98/10., 3.98/10.*2.86*4, 3.98/10.*2.86/10.*4, 3.98/10.*2.86/10.*4]

        line_wavelengths['OIII+OII'] = [5008.240, 4960.295, 3729.875]; line_ratios['OIII+OII'] = [2.98, 1, 3.98/4.]

        line_wavelengths['OII'] = [3729.875]; line_ratios['OII'] = [1]
        line_wavelengths['OI'] = [6302.046]; line_ratios['OI'] = [1]

        line_wavelengths['NeIII'] = [3869]; line_ratios['NeIII'] = [1.]
        line_wavelengths['NeV'] = [3346.8]; line_ratios['NeV'] = [1.]
        line_wavelengths['NeVI'] = [3426.85]; line_ratios['NeVI'] = [1.]
        line_wavelengths['SIII'] = [9068.6, 9530.6]; line_ratios['SIII'] = [1, 2.44]
        line_wavelengths['HeII'] = [4687.5]; line_ratios['HeII'] = [1.]
        line_wavelengths['HeI'] = [5877.2]; line_ratios['HeI'] = [1.]
        line_wavelengths['HeIb'] = [3889.5]; line_ratios['HeIb'] = [1.]
        #### Test line
        #line_wavelengths['HeI'] = fakeLine; line_ratios['HeI'] = [1. for line in fakeLine]
        
        line_wavelengths['MgII'] = [2799.117]; line_ratios['MgII'] = [1.]
        line_wavelengths['CIV'] = [1549.480]; line_ratios['CIV'] = [1.]
        line_wavelengths['Lya'] = [1215.4]; line_ratios['Lya'] = [1.]

        line_wavelengths['Ha+SII'] = [6564.61, 6718.29, 6732.67]; line_ratios['Ha+SII'] = [1., 1./10, 1./10]
        line_wavelengths['SII'] = [6718.29, 6732.67]; line_ratios['SII'] = [1., 1.]
        
        if line_complexes:
            #line_list = ['Ha+SII', 'OIII+Hb+Ha', 'OII']
            line_list = ['Ha+SII', 'OIII+Hb', 'OII']
        else:
            line_list = ['SIII', 'SII', 'Ha', 'OI', 'OIII', 'Hb', 'Hg', 'Hd', 'NeIII', 'OII']
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
    
    def fit_redshift(self, prior=None, poly_order=1, fwhm=1200,
                     make_figure=True, zr=None, dz=None, verbose=True,
                     fit_background=True, fitter='nnls'):
        """TBD
        """
        # if False:
        #     reload(grizlidev.utils); utils = grizlidev.utils
        #     reload(grizlidev.utils_c); reload(grizlidev.model); 
        #     reload(grizlidev.grismconf); reload(grizlidev.utils); reload(grizlidev.multifit); reload(grizlidev); reload(grizli)
        # 
        #     beams = []
        #     if id in flt.object_dispersers:
        #         b = flt.object_dispersers[id]['A']
        #         beam = grizli.model.BeamCutout(flt, b, conf=flt.conf)
        #         #print beam.grism.pad, beam.beam.grow
        #         beams.append(beam)
        #     else:
        #         print flt.grism.parent_file, 'ID %d not found' %(id)
        # 
        #     #plt.imshow(beam.beam.direct*(beam.beam.seg == id), interpolation='Nearest', origin='lower', cmap='viridis_r')
        #     self = beam
        # 
        #     #poly_order = 3
        
        # if self.grism.filter == 'G102':
        #     if zr is None:
        #         zr = [0.78e4/6563.-1, 1.2e4/5007.-1]
        #     if dz is None:
        #         dz = [0.001, 0.0005]
        # 
        # if self.grism.filter == 'G141':
        #     if zr is None:
        #         zr = [1.1e4/6563.-1, 1.65e4/5007.-1]
        #     if dz is None:
        #         dz = [0.003, 0.0005]
        
        if zr is None:
            zr = [0.65, 1.6]
        
        if dz is None:
            dz = [0.005, 0.003]
        
        # if True:
        #     beams = grp.get_beams(id, size=30)
        #     mb = grizlidev.multifit.MultiBeam(beams)
        #     self = mb
                    
        zgrid = utils.log_zgrid(zr, dz=dz[0])
        NZ = len(zgrid)
        
        templates = self.load_templates(fwhm=fwhm)
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
            if verbose:
                if chi2[i] < chi2min:
                    iz = i
                    chi2min = chi2[i]
                    
                print utils.no_newline + '  %.4f %9.1f (%.4f)' %(zgrid[i], chi2[i], zgrid[iz])
        
        if verbose:
            print 'First iteration: z_best=%.4f\n' %(zgrid[iz])
            
        # peaks
        import peakutils
        chi2nu = (chi2.min()-chi2)/self.DoF
        indexes = peakutils.indexes((chi2nu+0.01)*(chi2nu > -0.004), thres=0.003, min_dist=20)
        num_peaks = len(indexes)
        # plt.plot(zgrid, (chi2-chi2.min())/ self.DoF)
        # plt.scatter(zgrid[indexes], (chi2-chi2.min())[indexes]/ self.DoF, color='r')
              
        ### zoom
        if ((chi2.max()-chi2.min())/self.DoF > 0.01) & (num_peaks < 5):
            threshold = 0.005
        else:
            threshold = 0.003
        
        zgrid_zoom = utils.zoom_zgrid(zgrid, chi2/self.DoF, threshold=threshold, factor=dz[0]/0.0004)
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
            if verbose:
                if chi2_zoom[i] < chi2min:
                    chi2min = chi2_zoom[i]
                    iz = i
                    
                print utils.no_newline + '- %.4f %9.1f (%.4f) %d/%d' %(zgrid_zoom[i],
                                                          chi2_zoom[i],
                                                          zgrid_zoom[iz],
                                                          i+1, NZOOM)
        
        zgrid = np.append(zgrid, zgrid_zoom)
        chi2 = np.append(chi2, chi2_zoom)
        coeffs = np.append(coeffs, coeffs_zoom, axis=0)
    
        so = np.argsort(zgrid)
        zgrid = zgrid[so]
        chi2 = chi2[so]
        coeffs=coeffs[so,:]
        
        if verbose:
            print ' Zoom iteration: z_best=%.4f' %(zgrid[np.argmin(chi2)])
        
        ### Best redshift
        templates = self.load_templates(line_complexes=False, fwhm=fwhm)
        zbest = zgrid[np.argmin(chi2)]
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
        xspec = np.arange(0.7, 1.8, 0.05)-1
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
        
        line_flux = collections.OrderedDict()
        fscl = self.beams[0].beam.total_flux/1.e-17
        for i, key in enumerate(templates.keys()):
            temp_i = templates[key].zscale(zbest, coeffs_full[i0+i])
            model1d += temp_i
            if not key.startswith('line'):
                cont1d += temp_i
            else:
                line_flux[key.split()[1]] = np.array([coeffs_full[i0+i]*fscl, 
                                             line_flux_err[i0+i]*fscl])
                
                        
        fit_data = collections.OrderedDict()
        fit_data['poly_order'] = poly_order
        fit_data['fwhm'] = fwhm
        fit_data['zbest'] = zbest
        fit_data['zgrid'] = zgrid
        fit_data['A'] = A
        fit_data['coeffs'] = coeffs
        fit_data['chi2'] = chi2
        fit_data['model_full'] = model_full
        fit_data['coeffs_full'] = coeffs_full
        fit_data['line_flux'] = line_flux
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
    
    def show_redshift_fit(self, fit_data, plot_flambda=True):
        
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
            
            if plot_flambda:
                ok = beam.beam.sensitivity > 0.1*beam.beam.sensitivity.max()
                wave = wave[ok]
                flux = (flux/beam.beam.sensitivity)[ok]
                err = (err/beam.beam.sensitivity)[ok]
                mflux = (mflux/beam.beam.sensitivity)[ok]
                ylabel = r'$f_\lambda$'
            else:
                ylabel = 'flux (e-/s)'
            
            scl_region = np.isfinite(mflux) 
            if scl_region.sum() == 0:
                continue
                
            ymax = np.maximum(ymax, mflux[scl_region].max())
            ymin = np.minimum(ymin, mflux[scl_region].min())
            
            ax.errorbar(wave/1.e4, flux, err, alpha=0.15+0.2*(self.N <= 2), linestyle='None', marker='.', color='%.2f' %(ib*0.5/self.N), zorder=1)
            ax.plot(wave/1.e4, mflux, color='r', alpha=0.5, zorder=3)
            
            grism = beam.grism.filter
            for grism in grisms:
                wfull[grism] = np.append(wfull[grism], wave)
                ffull[grism] = np.append(ffull[grism], flux)
                efull[grism] = np.append(efull[grism], err)
        
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
                            linestyle='None', marker='.', color='k', zorder=2)
                
        ax.set_ylim(ymin - 0.1*np.abs(ymax), 1.1*ymax)
        
        xmin, xmax = 1.e5, 0
        limits = {'G102':[0.77, 1.18],
                  'G141':[1.06, 1.73]}
        
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
    
    def redshift_fit_twod_figure(self, fit):
        """Make figure of 2D spectrum
        TBD
        """        
        ### xlimits        
        xmin, xmax = 1.e5, 0
        limits = {'G102':[0.77, 1.18],
                  'G141':[1.06, 1.73]}
        
        for g in limits:
            if g in self.Ngrism:
                xmin = np.minimum(xmin, limits[g][0])
                xmax = np.maximum(xmin, limits[g][1])
        
        hdu_sci = drizzle_2d_spectrum(self.beams, ds9=None, NY=10,
                                      spatial_scale=1, dlam=46., 
                                      kernel='point', pixfrac=0.6,
                                      wlimit=[xmin, xmax])
                                  
        ### Continuum model
        cont = self.reshape_flat(fit['model_cont'])        
        hdu_con = drizzle_2d_spectrum(self.beams, data=cont, ds9=None, NY=10,
                                      spatial_scale=1, dlam=46., 
                                      kernel='point', pixfrac=0.6,
                                      wlimit=[xmin, xmax])
        
        full = self.reshape_flat(fit['model_full'])        
        hdu_full = drizzle_2d_spectrum(self.beams, data=full, ds9=None, NY=10,
                                      spatial_scale=1, dlam=46., 
                                      kernel='point', pixfrac=0.6,
                                      wlimit=[xmin, xmax])
        
        
        vmax = 1.1*hdu_full[1].data.max()
        sh = hdu_full[1].data.shape
        extent = [hdu_full[0].header['WMIN'], hdu_full[0].header['WMAX'],
                  0, sh[0]]
                  
        fig = plt.figure(figsize=[8,3.5])
        show = [hdu_sci[1].data, hdu_full[1].data,
                hdu_sci[1].data-hdu_con[1].data]
        
        desc = [r'$Cleaned$', r'$Model$', r'$Residual$']
        
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
        
        hdu_sci.append(hdu_con[1])
        hdu_sci[-1].name = 'CONTINUUM'
        hdu_sci.append(hdu_full[1])
        hdu_sci[-1].name = 'FULL'
        
        return fig, hdu_sci
        
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

def drizzle_2d_spectrum(beams, data=None, wlimit=[1.05, 1.75], dlam=50, 
                        spatial_scale=1, NY=10, pixfrac=0.6, kernel='square',
                        convert_to_flambda=True,
                        ds9=None):
    """Drizzle 2D spectrum from a list of beams
    
    Parameters
    ----------
    beams : list of `BeamCutout` objects
    
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

    kernel : {'square', 'point'}
        Drizzle kernel to use
    
    convert_to_flambda : bool, float
        Convert the 2D spectrum to physical units using the sensitivity curves
        and if float provided, scale the flux densities by that value
    
    ds9: `pyds9.DS9`
        Show intermediate steps of the drizzling
    
    Returns
    -------
    hdu : `astropy.io.fits.HDUList`
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
        wht = 1/beam.ivar + 0.2*beam.contam
        wht = np.cast[np.float32](1/wht)
        wht[~np.isfinite(wht)] = 0.
        
        data_i = data[i]*1.
        if convert_to_flambda:
            data_i *= convert_to_flambda/beam.beam.sensitivity
            wht *= (beam.beam.sensitivity/convert_to_flambda)**2
            wht[~np.isfinite(data_i)] = 0
            data_i[~np.isfinite(data_i)] = 0
        
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
    
def drizzle_to_wavelength(beams, ra=0., dec=0., wave=1.e4, size=5,
                          pixscale=0.1, pixfrac=0.6, kernel='square',
                          direct_extension='REF', ds9=None):
    """Drizzle a cutout at a specific wavelength from a list of `BeamCutout`s
    
    Parameters
    ----------
    beams : list of `model.BeamCutout` objects.
    
    ra, dec, wave : float
        Sky coordinates and central wavelength

    size : float
        Size of the output thumbnail, in arcsec
        
    pixscale : float
        Pixel scale of the output thumbnail, in arcsec
        
    pixfrac : float
        Drizzle PIXFRAC (for `kernel` = 'point')
        
    kernel : {'square', 'point'}
        Drizzle kernel to use
    
    direct_extension : {'SCI', 'REF'}
        Extension of `self.direct.data` do drizzle for the thumbnail
        
    ds9 : `pyds9.DS9`, optional
        Display each step of the drizzling to an open DS9 window
    
    Returns
    -------
    hdu : `astropy.io.fits.HDUList`
        FITS HDUList with the drizzled thumbnail, line and continuum 
        cutouts.
    """
    from drizzlepac.astrodrizzle import adrizzle
    
    # Nothing to do
    if len(beams) == 0:
        return False
        
    ### Get output header and WCS
    header, output_wcs = utils.make_wcsheader(ra=ra, dec=dec, size=size, pixscale=pixscale, get_hdu=False)

    ### Initialize data
    sh = (header['NAXIS1'], header['NAXIS2'])
    
    outsci = np.zeros(sh, dtype=np.float32)
    outwht = np.zeros(sh, dtype=np.float32)
    outctx = np.zeros(sh, dtype=np.int32)

    coutsci = np.zeros(sh, dtype=np.float32)
    coutwht = np.zeros(sh, dtype=np.float32)
    coutctx = np.zeros(sh, dtype=np.int32)

    doutsci = np.zeros(sh, dtype=np.float32)
    doutwht = np.zeros(sh, dtype=np.float32)
    doutctx = np.zeros(sh, dtype=np.int32)
    
    ## Loop through beams and run drizzle
    for i, beam in enumerate(beams):
        ## Get specific wavelength WCS for each beam
        beam_header, beam_wcs = beam.get_wavelength_wcs(wave)
        ## Make sure CRPIX set correctly for the SIP header
        for i in [0,1]: 
            beam_wcs.sip.crpix[i] = beam_wcs.wcs.crpix[i]
        
        beam_data = beam.grism.data['SCI'] - beam.contam 
        beam_continuum = beam.model*1
        
        # Downweight contamination
        wht = 1/beam.ivar + 0.2*beam.contam
        wht = np.cast[np.float32](1/wht)
        wht[~np.isfinite(wht)] = 0.
        
        ### Convert to f_lambda integrated line fluxes: 
        ###     (Inverse of the aXe sensitivity) x (size of pixel in \AA)
        sens = np.interp(wave, beam.beam.lam, beam.beam.sensitivity, 
                         left=0, right=0)
        
        dlam = np.interp(wave, beam.beam.lam[1:], np.diff(beam.beam.lam))
        # 1e-18 erg/s/cm2, scaling closer to e-/s
        sens *= 10./dlam
        
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
                         wcslin_pscale=1.0, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=None)
        
        ### Continuum
        adrizzle.do_driz(beam_continuum, beam_wcs, wht, output_wcs, 
                         coutsci, coutwht, coutctx, 1., 'cps', 1, 
                         wcslin_pscale=1.0, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=None)
        
        ### Direct thumbnail
        thumb = beam.direct.data[direct_extension]/beam.direct.photflam
        thumb_wht = 1./(beam.direct.data['ERR']/beam.direct.photflam)**2
        thumb_wht[~np.isfinite(thumb_wht)] = 0
        
        adrizzle.do_driz(thumb, beam.direct.wcs, thumb_wht, output_wcs, 
                         doutsci, doutwht, doutctx, 1., 'cps', 1, 
                         wcslin_pscale=1.0, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=None)
        
        ## Show in ds9
        if ds9 is not None:
            ds9.view((outsci-coutsci)/output_wcs.pscale**2, header=header)
    
    ## Scaling of drizzled outputs        
    outsci /= output_wcs.pscale**2
    coutsci /= output_wcs.pscale**2
    doutsci /= output_wcs.pscale**2
    
    ### Make output FITS products
    p = pyfits.PrimaryHDU()
    p.header['ID'] = (beams[0].id, 'Object ID')
    p.header['RA'] = (ra, 'Central R.A.')
    p.header['DEC'] = (dec, 'Central Decl.')
    p.header['WAVELEN'] = (wave, 'Central wavelength')
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
    
    grism_sci = pyfits.ImageHDU(data=outsci-coutsci, header=h, name='GSCI')
    grism_cont = pyfits.ImageHDU(data=coutsci, header=h, name='GCONTINUUM')
    grism_wht = pyfits.ImageHDU(data=outwht, header=h, name='GWHT')
    
    return pyfits.HDUList([p, thumb_sci, thumb_wht, grism_sci, grism_cont, grism_wht])
