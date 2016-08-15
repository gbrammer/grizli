import os
import time
import collections
import multiprocessing as mp

import scipy.ndimage as nd
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

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
    
    ## slight random delay to avoid synchronization problems
    # np.random.seed(ix)
    # sleeptime = ix*1
    # print '%s sleep %.3f %d' %(grism_file, sleeptime, ix)
    # time.sleep(sleeptime)
    
    #print grism_file, direct_file
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
            
    def refine(self, id, mag=-99, poly_order=1, size=30, ds9=None, verbose=True, max_coeff=2):
        """TBD
        """
        beams = self.get_beams(id, size=size, min_overlap=0.5, get_slice_header=False)
        if len(beams) == 0:
            return True
        
        mb = MultiBeam(beams)
        try:
            A, out_coeffs, chi2, modelf = mb.fit_at_z(poly_order=poly_order)
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
            ds9.view(flt.grism['SCI'] - flt.model, header=flt.grism.header)
        
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
                self.Ngrism[grism] = 0
                
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
    
    @staticmethod
    def func_chi2(xdata, params, ivar):
        pass
        
    def fit_at_z(self, z=0., templates={}, fitter='lstsq',
                 fit_background=True, poly_order=3):
        """TBD
        """
        import sklearn.linear_model
        import numpy.linalg
        
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
        if fitter == 'lstsq':
            
            Ax = A[:, self.fit_mask][ok_temp,:].T
            y = self.scif[self.fit_mask]
            
            ### Wieght by ivar
            Ax *= self.ivarf[self.fit_mask][:, np.newaxis]
            y *= self.ivarf[self.fit_mask]
            
            #print 'xxx lstsq'
            out = numpy.linalg.lstsq(Ax,y)
                                     
            lstsq_coeff, residuals, rank, s = out
            coeffs = lstsq_coeff
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
                     fit_background=True):
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
        
        out = self.fit_at_z(z=0., templates=templates, fitter='lstsq',
                            poly_order=poly_order,
                            fit_background=fit_background)
                            
        A, coeffs, chi2, model_2d = out
        
        chi2 = np.zeros(NZ)
        coeffs = np.zeros((NZ, coeffs.shape[0]))
        
        chi2min = 1e30
        iz = 0
        for i in xrange(NZ):
            out = self.fit_at_z(z=zgrid[i], templates=templates,
                                fitter='lstsq', poly_order=poly_order,
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
            threshold = 0.01
        else:
            threshold = 0.001
        
        zgrid_zoom = utils.zoom_zgrid(zgrid, chi2/self.DoF, threshold=threshold, factor=dz[0]/0.0004)
        NZOOM = len(zgrid_zoom)
        
        chi2_zoom = np.zeros(NZOOM)
        coeffs_zoom = np.zeros((NZOOM, coeffs.shape[1]))

        iz = 0
        chi2min = 1.e30
        for i in xrange(NZOOM):
            out = self.fit_at_z(z=zgrid_zoom[i], templates=templates,
                                fitter='lstsq', poly_order=poly_order,
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
                            fitter='lstsq', poly_order=poly_order, 
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
        for i, key in enumerate(templates.keys()):
            temp_i = templates[key].zscale(zbest, coeffs_full[i0+i])
            model1d += temp_i
            if not key.startswith('line'):
                cont1d += temp_i
            else:
                line_flux[key.split()[1]] = (coeffs_full[i0+i] * 
                                             self.beams[0].beam.total_flux/1.e-17)
                
                        
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
        ax.set_title('ID = %d, z_grism=%.4f' %(self.beams[0].id, 
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
            
            ax.errorbar(wave/1.e4, flux, err, alpha=0.05+0.3*(self.N <= 2), linestyle='None', marker='.', color='%.2f' %(ib*0.5/self.N), zorder=1)
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
            
                N = int(np.ceil(self.Ngrism[grism]/2)*2)*4
                kernel = np.ones(N, dtype=float)/N
                fbin = nd.convolve(ffull[grism][okb][so], kernel)[N/2::N]
                wbin = nd.convolve(wfull[grism][okb][so], kernel)[N/2::N]
                vbin = nd.convolve(var[okb][so], kernel**2)[N/2::N]
                ax.errorbar(wbin/1.e4, fbin, np.sqrt(vbin), alpha=0.8,
                            linestyle='None', marker='.', color='k', zorder=2)
                
        ax.set_ylim(ymin - 0.1*np.abs(ymax), 1.1*ymax)
        ax.plot(wave/1.e4, wave/1.e4*0., linestyle='--', color='k')
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(ylabel)
        
        fig.tight_layout(pad=0.1)
        return fig
        
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
    
    def drizzle(self, wcs):
        """TBD
        
        id=19258
        
        id = 21610
           
        zsp = c['z_spec'] > 0.7
        
        idx = np.arange(len(c))[zsp]
        
        lines = ['circle(%.6f,%.6f,0.2") # text={%.3f}\n' %(c['ra'][i], c['dec'][i], c['z_spec'][i]) for i in idx]
        fp = open('zsp.reg','w')
        fp.write('fk5\n')
        fp.writelines(lines)
        fp.close()
        
        """
        
        c, z, f = unicorn.analysis.read_catalogs('goodsn')
        
        l0 = 6563.
        N = 80
        size=30
        
        beams = self.get_beams(id, size=size)
        mb = grizlidev.multifit.MultiBeam(beams)
        
        img = pyfits.open(ref)
        ref_seg = pyfits.open(seg)
        ref_wcs = astropy.wcs.WCS(img[0].header)
        
        ix = c['id'] == id
        ra, dec = c['ra'][ix][0], c['dec'][ix][0]
        
        wave = l0*(1+c['z_spec'][ix][0])
        
        ref_pix = np.array(ref_wcs.all_world2pix([ra], [dec], 0))
        xr, yr = np.cast[int](np.round(ref_pix[:,0].flatten()))

        slx, sly = slice(xr-N, xr+N), slice(yr-N, yr+N)
        #print slx, sly
        ref_wcs_slice = ref_wcs.slice((sly, slx))

        ref_wcs_slice.pscale = np.sqrt(ref_wcs_slice.wcs.cd[0,0]**2 + ref_wcs_slice.wcs.cd[1,0]**2)*3600.

        ref_slice = img[0].data[sly, slx]
        seg_slice = ref_seg[0].data[sly, slx]

        ### Recenter slice
        sh = ref_slice.shape
        rd0 = ref_wcs_slice.all_pix2world(np.array([sh])/2., 1)
        #print rd0[0]
        #print ref_wcs_slice
        ref_wcs_slice.wcs.crval = rd0[0]
        ref_wcs_slice.wcs.crpix = np.array(sh)[::-1]/2.

        ### Scale slice
        #ref_wcs_slice.wcs.cd *= 2
        #ref_wcs_slice.wcs.cd = np.array([[1,0],[0,1]])*0.1/3600.
        #print ref_wcs_slice

        ref_header_slice = ref_wcs_slice.to_header(relax=True)
        for key in ref_header_slice:
            if key.startswith('PC'):
                ref_header_slice.rename_keyword(key, key.replace('PC', 'CD'))
        
        #
        from drizzlepac import astrodrizzle

        sh = ref_slice.shape
        outsci = np.zeros(sh, dtype=np.float32)
        outwht = np.zeros(sh, dtype=np.float32)
        outctx = np.zeros(sh, dtype=np.int32)

        coutsci = np.zeros(sh, dtype=np.float32)
        coutwht = np.zeros(sh, dtype=np.float32)
        coutctx = np.zeros(sh, dtype=np.int32)

        doutsci = np.zeros(sh, dtype=np.float32)
        doutwht = np.zeros(sh, dtype=np.float32)
        doutctx = np.zeros(sh, dtype=np.int32)

        out_wcs = ref_wcs_slice.deepcopy()

        for i, beam in enumerate(beams):
            beam_header, beam_wcs = beam.get_wavelength_wcs(wave)
            beam_data = beam.grism.data['SCI'] - beam.contam 

            wht = 1/beam.ivar + 0.2*beam.contam
            wht = np.cast[np.float32](1/wht)
            wht[~np.isfinite(wht)] = 0.

            pf = 0.6
            kernel = 'square'

            astrodrizzle.adrizzle.do_driz(beam_data, beam_wcs, wht, 
                                          out_wcs, outsci, outwht, outctx,
                                          1., 'cps', 1, wcslin_pscale=1.0,
                                          uniqid=1, pixfrac=pf, kernel=kernel, 
                                          fillval=0, stepsize=10, wcsmap=None)   

            astrodrizzle.adrizzle.do_driz(self.model_continuum[i], beam_wcs, wht, 
                                          out_wcs, coutsci, coutwht, coutctx,
                                          1., 'cps', 1, wcslin_pscale=1.0,
                                          uniqid=1, pixfrac=pf, kernel=kernel, 
                                          fillval=0, stepsize=10, wcsmap=None)   

            thumb = beam.direct.data['SCI']/beam.direct.photflam
            thumb_wht = 1./(beam.direct.data['ERR']/beam.direct.photflam)**2
            thumb_wht[~np.isfinite(thumb_wht)] = 0

            astrodrizzle.adrizzle.do_driz(thumb, beam.direct.wcs, thumb_wht, 
                                          out_wcs, doutsci, doutwht, doutctx,
                                          1., 'cps', 1, wcslin_pscale=1.0,
                                          uniqid=1, pixfrac=pf, kernel=kernel, 
                                          fillval=0, stepsize=10, wcsmap=None)   
            #
            ds9.view((outsci-coutsci)/out_wcs.pscale**2, header=ref_header_slice)
            
        outsci /= out_wcs.pscale**2
        coutsci /= out_wcs.pscale**2
        doutsci /= out_wcs.pscale**2
        
class SpectrumTemplate(object):
    """TBD
    """
    def __init__(self, wave=None, flux=None, fwhm=None, velocity=False):
        """TBD
        """
        self.wave = wave
        self.flux = flux
        
        ### Gaussian
        if (wave is not None) & (fwhm is not None):
            rms = fwhm/2.35
            if velocity:
                rms *= wave/3.e5
                
            xgauss = np.arange(-5,5.01,0.1)*rms+wave
            gaussian = np.exp(-(xgauss-wave)**2/2/rms**2)
            gaussian /= np.sqrt(2*np.pi*rms**2)
            
            self.wave = xgauss
            self.flux = gaussian
    
    def __add__(self, spectrum):
        """TBD
        """
        new_wave = np.unique(np.append(self.wave, spectrum.wave))
        new_wave.sort()
        
        new_flux = np.interp(new_wave, self.wave, self.flux)
        new_flux += np.interp(new_wave, spectrum.wave, spectrum.flux)
        return SpectrumTemplate(wave=new_wave, flux=new_flux)
    
    def __mul__(self, scalar):
        return SpectrumTemplate(wave=self.wave, flux=self.flux*scalar)
        
    def zscale(self, z, scalar=1):
        return SpectrumTemplate(wave=self.wave*(1+z),
                                flux=self.flux*scalar/(1+z))
