"""
Utilities for fitting stacked (drizzled) spectra
"""
from collections import OrderedDict
from imp import reload

import astropy.io.fits as pyfits
import astropy.units as u

import numpy as np

from . import utils
from .utils import GRISM_COLORS, GRISM_MAJOR, GRISM_LIMITS, DEFAULT_LINE_LIST

from .fitting import GroupFitter

def make_templates(grism='G141', return_lists=False, fsps_templates=False,
                   line_list=DEFAULT_LINE_LIST):
    """Generate template savefile
    
    This script generates the template sets with the emission line 
    complexes and with individual lines.
    
    Parameters
    ----------
    grism : str
        Grism of interest, which defines what FWHM to use for the line
        templates.
    
    return_lists : bool
        Return the templates rather than saving them to a file
        
    Returns
    -------
    t_complexes, t_lines : list
        If `return` then return two lists of templates.  Otherwise, 
        store them to a `~numpy` save file "templates_{fwhm}.npy".
        
    """
    
    from .multifit import MultiBeam
    
    if grism == 'G141':    # WFC3/IR
        fwhm = 1100
    elif grism == 'G800L': # ACS/UVIS
        fwhm = 1400
    elif grism == 'G280':  # WFC3/UVIS
        fwhm = 1500
    elif grism == 'GRISM': # WFIRST
        fwhm = 350
    else:
        fwhm = 700 # G102
        
    # Line complex templates
    t_complexes = utils.load_templates(fwhm=fwhm, line_complexes=True,
                                           fsps_templates=fsps_templates)
    
    # Individual lines
    # line_list = ['SIII', 'SII', 'Ha', 'OI-6302', 'OIII', 'Hb', 
    #              'OIII-4363', 'Hg', 'Hd', 'NeIII', 'OII', 'MgII']
                 
    t_lines = utils.load_templates(fwhm=fwhm, line_complexes=False,
                                       full_line_list=line_list,
                                       fsps_templates=fsps_templates)
    
    if return_lists:
        return t_complexes, t_lines
    else:
        # Save them to a file
        np.save('templates_{0}.npy'.format(fwhm), [t_complexes, t_lines])
        print('Wrote `templates_{0}.npy`'.format(fwhm))

class StackFitter(GroupFitter):
    def __init__(self, files='gnt_18197.stack.fits', group_name=None, sys_err=0.02, mask_min=0.1, fit_stacks=True, fcontam=1, PAs=None, extensions=None, min_ivar=0.01, overlap_threshold=3, verbose=True, eazyp=None, eazy_ix=0, MW_EBV=0., chi2_threshold=1.5, min_DoF=200):
        """Object for fitting stacked spectra.
        
        Parameters
        ----------
        files : str or list of str
            Stack FITS filename.  If a list is supplied, e.g., the product
            of a `~glob` command, then append all specified files.
        
        group_name : str
            Rootname to associate with the object.  If none, then default to 
            `files`.
            
        sys_err : float
            Minimum systematic error, interpreted as a fractional error.  
            The adjusted variance is taken to be
            
                >>> var = var0 + (sys_err*flux)**2
                
        mask_min : float
            Only fit 2D pixels where the flat-flambda model has pixel values
            greater than `mask_min` times the maximum of the model.
        
        fit_stacks : bool
            Fit the stacks of each grism combined from all available PAs.  If
            False, then fit the PAs individually.
        
        fcontam : float
            Parameter to control weighting of contaminated pixels for 
            `fit_stacks=False`.  
            
        """
        if isinstance(files, list):
            file=files[0]
        else:
            file=files
        
        self.files = [file]
        if group_name is not None:
            self.group_name = group_name
        else:
            self.group_name = file
            
        if verbose:
            print('Load file {0}'.format(file))
            
        self.file = file
        self.hdulist = pyfits.open(file)
        self.min_ivar = min_ivar
        self.sys_err = sys_err
        self.fcontam = fcontam
        
        self.MW_EBV = MW_EBV
        
        self.h0 = self.hdulist[0].header.copy()
        #self.Ngrism = self.h0['NGRISM']
        self.grisms = []
        self.ext = []
        for i in range(self.h0['NGRISM']):
            g = self.h0['GRISM{0:03d}'.format(i+1)]
            self.grisms.append(g)
            if fit_stacks:
                if extensions is not None:
                    if g not in extensions:
                        continue
                        
                self.ext.append(g)
            else:
                ng = self.h0['N{0}'.format(g)]
                for j in range(ng):
                    pa = self.h0['{0}{1:02d}'.format(g, j+1)]
                    
                    if PAs is not None:
                        if pa not in PAs:
                            continue
                    
                    ext = '{0},{1}'.format(g,pa)
                    if extensions is not None:
                        if ext not in extensions:
                            continue
                            
                    self.ext.append(ext)
                        
        self.N = len(self.ext)
        self.beams = []
        pop = []
        for i in range(self.N):
            E_i = StackedSpectrum(file=self.file, sys_err=sys_err,
                                  mask_min=mask_min, extver=self.ext[i], 
                                  mask_threshold=-1, fcontam=fcontam, 
                                  min_ivar=min_ivar, MW_EBV=MW_EBV)
            E_i.compute_model()
            
            if np.isfinite(E_i.kernel.sum()) & (E_i.DoF >= min_DoF):
                self.beams.append(E_i)
            else:
                pop.append(i)
            
            
        for i in pop[::-1]:
            self.N -= 1
            p = self.ext.pop(i)
                                    
        # Get some parameters from the beams
        self.id = self.h0['ID']
        self.ra = self.h0['RA']
        self.dec = self.h0['DEC']
        
        ## Photometry
        self.is_spec = 1
        self.Nphot = 0
        
        ## Parse the beam data
        self._parse_beams_list()
        
        if not fit_stacks:
            # self.mask_drizzle_overlaps(threshold=overlap_threshold,
            #                           verbose=verbose)
            if chi2_threshold > 0:
                orig_ext = [e for e in self.ext]
                fit_log, keep_dict, has_bad = self.check_for_bad_PAs(poly_order=3, chi2_threshold=chi2_threshold, fit_background=True, reinit=True, verbose=False)
                if has_bad & verbose:
                    print('Found bad PA.  New list: {0}'.format(keep_dict))
        
        if verbose:
            print('  {0}'.format(' '.join(self.ext)))
            
        # Read multiple
        if isinstance(files, list):
            if len(files) > 1:
                for file in files[1:]:
                    extra = StackFitter(files=file, sys_err=sys_err, mask_min=mask_min, fit_stacks=fit_stacks, fcontam=fcontam, pas=pas, extensions=extensions, min_ivar=min_ivar, overlap_threshold=overlap_threshold, eazyp=eazyp, eazy_ix=eazy_ix, chi2_threshold=chi2_threshold, verbose=verbose)
                    self.extend(extra)
                    
                    
        # if eazyp is not None:
        #     self.eazyp = eazyp
        #     
        #     # TBD: do matching to eazyp.cat directly?
        #     self.eazy_ix = eazy_ix
        #     
        #     ok_phot = (eazyp.efnu[eazy_ix,:] > 0) & (eazyp.fnu[eazy_ix,:] > eazyp.param['NOT_OBS_THRESHOLD']) & np.isfinite(eazyp.fnu[eazy_ix,:]) & np.isfinite(eazyp.efnu[eazy_ix,:])
        #     ok_phot = np.squeeze(ok_phot)
        #     self.ok_phot = ok_phot
        # 
        #     self.Nphot = ok_phot.sum()
        #     if self.Nphot > 0:
        #         
        #         # F-lambda photometry, 1e-19 erg/s/cm2/A
        #         self.photom_eflam = (eazyp.efnu[eazy_ix,:]*eazyp.to_flam*eazyp.zp*eazyp.ext_corr/100.)[ok_phot]
        #         self.photom_flam = (eazyp.fnu[eazy_ix,:]*eazyp.to_flam*eazyp.zp*eazyp.ext_corr/100.)[ok_phot]
        #         self.photom_lc = eazyp.lc[ok_phot]
        #     
        #         self.scif = np.hstack((self.scif, self.photom_flam))
        #         self.ivarf = np.hstack((self.ivarf, 1/self.photom_eflam**2))
        #         self.sivarf = np.hstack((self.sivarf, 1/self.photom_eflam))
        #         self.wavef = np.hstack((self.wavef, self.photom_lc))
        # 
        #         self.weightf = np.hstack((self.weightf, np.ones(self.Nphot)))
        #         self.fit_mask = np.hstack((self.fit_mask, np.ones(self.Nphot, dtype=bool)))
        #         self.DoF += self.Nphot
        #         self.phot_scale = np.array([10.])
    
    def _parse_beams_list(self):
        """
        """                    
        # Parse from self.beams list
        self.N = len(self.beams)
        self.ext = [E.extver for E in self.beams]
        
        self.Ngrism = OrderedDict()
        for beam in self.beams:
            if beam.grism in self.Ngrism:
                self.Ngrism[beam.grism] += 1
            else:
                self.Ngrism[beam.grism] = 1
                
        # Make "PA" attribute
        self.PA = OrderedDict()
        for g in self.Ngrism:
            self.PA[g] = OrderedDict()

        for i in range(self.N):
            grism = self.ext[i].split(',')[0]
            if ',' in self.ext[i]:
                PA = float(self.ext[i].split(',')[1])
            else:
                PA = 0
                
            if PA in self.PA[grism]:
                self.PA[grism][PA].append(i)
            else:
                self.PA[grism][PA] = [i]
        
        self.grisms = list(self.PA.keys())
                    
        self.Ntot = np.sum([E.size for E in self.beams])
        self.scif = np.hstack([E.scif for E in self.beams])
        self.ivarf = np.hstack([E.ivarf for E in self.beams])
        self.wavef = np.hstack([E.wavef for E in self.beams])

        self.weightf = np.hstack([E.weightf for E in self.beams])
        #self.ivarf *= self.weightf

        self.sivarf = np.sqrt(self.ivarf)

        self.fit_mask = np.hstack([E.fit_mask for E in self.beams])
        self.fit_mask &= self.ivarf > self.min_ivar*self.ivarf.max()
        
        # Dummy parameter.  Implemented for MultiBeam
        self.sens_mask = 1. 
        
        self.DoF = int((self.fit_mask*self.weightf).sum())
        #self.Nmask = self.fit_mask.sum()
        self.Nmask = np.sum([E.fit_mask.sum() for E in self.beams])
        
        self.slices = self._get_slices(masked=False)
        self.A_bg = self._init_background(masked=False)

        self._update_beam_mask()
        self.A_bgm = self._init_background(masked=True)
        
        self.flat_flam = np.hstack([E.flat_flam for E in self.beams])
        
        self.initialize_masked_arrays()
        
    def extend(self, st):
        """
        Append second StackFitter objects to `self`.
        """
        self.beams.extend(st.beams)
        self._parse_beams_list()
        
        # self.grisms.extend(st.grisms)
        # self.grisms = list(np.unique(self.grisms))
        # 
        # self.Ngrism = {}
        # for grism in self.grisms:
        #     self.Ngrism[grism] = 0
        # 
        # for beam in self.beams:
        #     self.Ngrism[beam.grism] += 1
        #     
        # #self.Ngrism = len(self.grisms)
        # 
        # self.N += st.N
        # self.ext.extend(st.ext)
        # self.files.extend(st.files)
        # 
        # # Re-init
        # self.Ntot = np.sum([E.size for E in self.beams])
        # self.scif = np.hstack([E.scif for E in self.beams])
        # self.ivarf = np.hstack([E.ivarf for E in self.beams])
        # self.wavef = np.hstack([E.wavef for E in self.beams])
        # 
        # self.weightf = np.hstack([E.weightf for E in self.beams])
        # #self.ivarf *= self.weightf
        # 
        # self.sivarf = np.sqrt(self.ivarf)
        # 
        # self.fit_mask = np.hstack([E.fit_mask for E in self.beams])
        # self.fit_mask &= self.ivarf > self.min_ivar*self.ivarf.max()
        # 
        # self.slices = self._get_slices(masked=False)
        # self.A_bg = self._init_background(masked=False)
        # 
        # self._update_beam_mask()
        # self.A_bgm = self._init_background(masked=True)
        #         
        # self.Nmask = self.fit_mask.sum()        
        # self.DoF = int((self.fit_mask*self.weightf).sum())
        # 
        # self.flat_flam = np.hstack([E.flat_flam for E in self.beams])
    
    def check_for_bad_PAs(self, poly_order=1, chi2_threshold=1.5, fit_background=True, reinit=True, verbose=False):
        """
        """

        wave = np.linspace(2000,2.5e4,100)
        poly_templates = utils.polynomial_templates(wave, order=poly_order)
        
        fit_log = OrderedDict()
        keep_dict = OrderedDict()
        has_bad = False
        
        keep_beams = []
        
        for g in self.PA:
            fit_log[g] = OrderedDict()
            keep_dict[g] = []
                            
            for pa in self.PA[g]:
                extensions = [self.ext[i] for i in self.PA[g][pa]]
                mb_i = StackFitter(self.file, fcontam=self.fcontam,
                                   sys_err=self.sys_err,
                                   extensions=extensions, fit_stacks=False,
                                   verbose=verbose, chi2_threshold=-1)
                              
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
            self._parse_beams_list()
            #self._parse_beams(psf=self.psf_param_dict is not None)
            
        return fit_log, keep_dict, has_bad
        
    def compute_model(self, spectrum_1d=None):
        """
        TBD
        """
        return False

    def get_sky_coords(self):
        """Get WCS coordinates of the center of the direct image
        
        Returns
        -------
        ra, dec : float
            Center coordinates of the beam thumbnail in decimal degrees
        """
        ra = self.beams[0].h0['RA']
        dec = self.beams[0].h0['DEC']
        return ra, dec
    
    def fit_combined_at_z(self, z=0, fitter='nnls', get_uncertainties=False, eazyp=None, ix=0, order=1, scale_fit=None, method='BFGS'):
        """Fit the 2D spectra with a set of templates at a specified redshift.
        TBD
        Parameters
        ----------
        z : float
            Redshift.
        
        templates : list
            List of templates to fit.
        
        fitter : str
            Minimization algorithm to compute template coefficients.
            The default 'nnls' uses non-negative least squares.  
            The other option is standard 'leastsq'.
        
        get_uncertainties : bool
            Compute coefficient uncertainties from the covariance matrix
        
        
        Returns
        -------
        chi2 : float
            Chi-squared of the fit
        
        background : `~np.ndarray`
            Background model
        
        full_model : `~np.ndarray`
            Best-fit 2D model.
        
        coeffs, err : `~np.ndarray`
            Template coefficients and uncertainties.
        
        """
        import scipy.optimize
        
        # Photometry
        ok_phot = (eazyp.efnu[ix,:] > 0) & (eazyp.fnu[ix,:] > eazyp.param['NOT_OBS_THRESHOLD']) & np.isfinite(eazyp.fnu[ix,:]) & np.isfinite(eazyp.efnu[ix,:])
        ok_phot = np.squeeze(ok_phot)
        self.ok_phot = ok_phot
        
        Nphot = ok_phot.sum()
        photom_eflam = (eazyp.efnu[ix,:]*eazyp.to_flam*eazyp.ext_corr/100.)[ok_phot]
        photom_flam = (eazyp.fnu[ix,:]*eazyp.to_flam*eazyp.ext_corr/100.)[ok_phot]
        
        templates = eazyp.templates
        
        NTEMP = len(templates)
        A = np.zeros((self.N+NTEMP, self.Ntot+Nphot))
        A[:self.N,:-Nphot] += self.A_bg
        
        pedestal = 0.04
        
        sivarf = np.hstack((self.sivarf, 1/photom_eflam))
        dataf = np.hstack((self.scif+pedestal, photom_flam))
        fit_mask = np.hstack((self.fit_mask, np.ones(Nphot, dtype=bool)))
        #sivarf *= fit_mask
        
        # Photometry
        Aphot = (eazyp.tempfilt(z)*3.e18/eazyp.lc**2*(1+z))[:,ok_phot]
        A[self.N:,-Nphot:] += Aphot
        
        for i, ti in enumerate(templates):
            #ti = templates[t]
            s = [ti.wave*(1+z), ti.flux/(1+z)]
            
            for j, E in enumerate(self.beams):
                clip = E.ivar.sum(axis=0) > 0                    
                if (s[0][0] > E.wave[clip].max()) | (s[0][-1] < E.wave[clip].min()):
                    continue

                sl = self.slices[j]
                A[self.N+i, sl] = E.compute_model(spectrum_1d=s)
                    
        oktemp = (A*fit_mask).sum(axis=1) != 0
        
        # ATA = np.dot(A[oktemp,:]*(sivarf*fit_mask)**2, A[oktemp,:].T)
        # ATy = np.dot(A[oktemp,:]*(sivarf*fit_mask)**2, dataf)
        
        Ax = A[oktemp,:]*sivarf
        #AxT = Ax[:,fit_mask].T
        #data = (dataf*sivarf)[fit_mask]
        
        # Run the optimizer
        #method = 'Powell'
        #method = 'BFGS'
        tol = 1.e-4
        init = np.zeros(order+1)
        init[0] = 10.
        
        if scale_fit is None:
            scale_fit = scipy.optimize.minimize(self.objective_scale, init, args=(Ax, dataf*sivarf, self.wavef, fit_mask, sivarf, Nphot, self.N, 0), method=method, jac=None, hess=None, hessp=None, bounds=(), constraints=(), tol=tol, callback=None, options=None)
        
            if order == 0:
                scale_fit.x = np.array([np.float(scale_fit.x)])
            
        coeffs, full, resid, chi2, AxT = self.objective_scale(scale_fit.x, Ax, dataf*sivarf, self.wavef, fit_mask, sivarf, Nphot, self.N, True)
        
        background = np.dot(coeffs[:self.N], A[:self.N,:])
        full -= background
        
        background -= pedestal
        background[-Nphot:] = 0
                 
        # Uncertainties from covariance matrix
        if get_uncertainties:
            try:
                covar = np.matrix(np.dot(AxT.T, AxT)).I
                covard = np.sqrt(covar.diagonal()).A.flatten()
                
                # covarf = np.matrix(np.dot(ATA.T, ATA)).I
                # covardf = np.sqrt(covarf.diagonal()).A.flatten()
                
            except:
                print('Except!')
                covard = np.zeros(oktemp.sum())#-1.
        else:
            covard = np.zeros(oktemp.sum())#-1.
        
        full_coeffs = np.zeros(NTEMP)
        full_coeffs[oktemp[self.N:]] = coeffs[self.N:]

        full_coeffs_err = np.zeros(NTEMP)
        full_coeffs_err[oktemp[self.N:]] = covard[self.N:]
        
        return chi2, background, full, full_coeffs, full_coeffs_err, scale_fit
    
    @staticmethod 
    def scale_AxT(p, Ax, spec_wave, Nphot, Next):
        """
        Scale spectrum templates by polynomial function
        """
        from scipy import polyval
        
        scale = np.ones(Ax.shape[1])
        scale[:-Nphot] = polyval(p[::-1]/10., (spec_wave-1.e4)/1000.)
        AxT = Ax*scale
        for i in range(Next):
            AxT[i,:] /= scale
        
        return AxT
        
    @staticmethod
    def objective_scale(p, Ax, data, spec_wave, fit_mask, sivarf, Nphot, Next, return_coeffs):
        """
        Objective function for fitting for a scale term between photometry and 
        spectra
        """
        import scipy.optimize
        from scipy import polyval
        
        scale = np.ones(Ax.shape[1])
        scale[:-Nphot] = polyval(p[::-1]/10., (spec_wave-1.e4)/1000.)
        AxT = Ax*scale
        
        # Remove scaling from background component
        for i in range(Next):
            AxT[i,:] /= scale
        
        #AxT = AxT[:,fit_mask].T
        #(Ax*scale)[:,fit_mask].T
        #AxT[:,:Next] = 1.
        
        coeffs, rnorm = scipy.optimize.nnls(AxT[:,fit_mask].T, data[fit_mask])  
            
        full = np.dot(coeffs, AxT/sivarf)
        resid = data/sivarf - full# - background
        chi2 = np.sum(resid[fit_mask]**2*sivarf[fit_mask]**2)
        
        #print('{0} {1}'.format(p, chi2))

        if return_coeffs:
            return coeffs, full, resid, chi2, AxT
        else:
            return chi2
        
        # Testing
        # method = 'COBYLA'
        # out = scipy.optimize.minimize(objective_scale, [10.], args=(Ax, dataf*sivarf, fit_mask, sivarf, Nphot, 0), method=method, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
        # print(method, out.nfev, out.x)
        # out = scipy.optimize.minimize(objective_scale, [10.], args=(Ax, dataf*sivarf, fit_mask, sivarf, Nphot, 0), method='COBYLA', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
        
    def fit_at_z(self, z=0, templates=[], fitter='nnls', get_uncertainties=False):
        """Fit the 2D spectra with a set of templates at a specified redshift.
        
        Parameters
        ----------
        z : float
            Redshift.
        
        templates : list
            List of templates to fit.
        
        fitter : str
            Minimization algorithm to compute template coefficients.
            The default 'nnls' uses non-negative least squares.  
            The other option is standard 'leastsq'.
        
        get_uncertainties : bool
            Compute coefficient uncertainties from the covariance matrix
        
        
        Returns
        -------
        chi2 : float
            Chi-squared of the fit
        
        background : `~np.ndarray`
            Background model
        
        full_model : `~np.ndarray`
            Best-fit 2D model.
        
        coeffs, err : `~np.ndarray`
            Template coefficients and uncertainties.
        
        """
        import scipy.optimize
        
        NTEMP = len(templates)
        A = np.zeros((self.N+NTEMP, self.Ntot))
        A[:self.N,:] += self.A_bg
                        
        for i, t in enumerate(templates):
            ti = templates[t]
            try:
                import eazy.igm
                if z > 7:
                    igm = eazy.igm.Inoue14()
                    igmz = igm.full_IGM(z, ti.wave*(1+z))         
                else:
                    igmz = 1.
                    
            except:
                igmz = 1.

            
            s = [ti.wave*(1+z), ti.flux/(1+z)*igmz]
            
            for j, E in enumerate(self.beams):
                clip = E.ivar.sum(axis=0) > 0                    
                if (s[0][0] > E.wave[clip].max()) | (s[0][-1] < E.wave[clip].min()):
                    continue

                sl = self.slices[j]
                A[self.N+i, sl] = E.compute_model(spectrum_1d=s)
                    
        oktemp = (A*self.fit_mask).sum(axis=1) != 0
        
        Ax = A[oktemp,:]*self.sivarf
        
        pedestal = 0.04
        
        AxT = Ax[:,self.fit_mask].T
        data = ((self.scif+pedestal)*self.sivarf)[self.fit_mask]
        
        if fitter == 'nnls':
            coeffs, rnorm = scipy.optimize.nnls(AxT, data)            
        else:
            coeffs, residuals, rank, s = np.linalg.lstsq(AxT, data)
                
        background = np.dot(coeffs[:self.N], A[:self.N,:]) - pedestal
        full = np.dot(coeffs[self.N:], Ax[self.N:,:]/self.sivarf)
        
        resid = self.scif - full - background
        chi2 = np.sum(resid[self.fit_mask]**2*self.sivarf[self.fit_mask]**2)
        
        # Uncertainties from covariance matrix
        if get_uncertainties:
            try:
                covar = np.matrix(np.dot(AxT.T, AxT)).I
                covard = np.sqrt(covar.diagonal()).A.flatten()
            except:
                print('Except: covar!')
                covard = np.zeros(oktemp.sum())#-1.
        else:
            covard = np.zeros(oktemp.sum())#-1.
        
        full_coeffs = np.zeros(NTEMP)
        full_coeffs[oktemp[self.N:]] = coeffs[self.N:]

        full_coeffs_err = np.zeros(NTEMP)
        full_coeffs_err[oktemp[self.N:]] = covard[self.N:]
        
        return chi2, background, full, full_coeffs, full_coeffs_err
    
    def fit_zgrid(self, dz0=0.005, zr=[0.4, 3.4], fitter='nnls', make_plot=True, save_data=True, prior=None, templates_file='templates.npy', verbose=True, outlier_threshold=1e30, eazyp=None, ix=0, order=0, scale_fit=None):
        """Fit templates on a redshift grid.
        
        Parameters
        ----------
        dz0 : float
            Initial step size of the redshift grid (dz/1+z).
        
        zr : list
            Redshift range to consider.
        
        fitter : str
            Minimization algorithm.  Default is non-negative least-squares.
        
        make_plot : bool
            Make the diagnostic plot.
        
        prior : list
            Naive prior to add to the nominal chi-squared(z) of the template
            fits.  The example below is a simple Gaussian prior centered
            at z=1.5. 
            
                >>> z_prior = np.arange(0,3,0.001)
                >>> chi_prior = (z_prior-1.5)**2/2/0.1**2
                >>> prior = [z_prior, chi_prior]
        
        templates_file : str
            Filename of the `~numpy` save file containing the templates.  Use 
            the `make_templates` script to generate this.
            
        verbose : bool
            Print the redshift grid steps.
        
        Returns
        -------
        hdu : `~astropy.io.fits.HDUList`
            Multi-extension FITS file with the result of the redshift fits.
        
        """
        import os
        #import grizli
        
        import matplotlib.gridspec
        import matplotlib.pyplot as plt
        import numpy as np
        
        from . import utils
        
        t_complex, t_i = np.load(templates_file)
        
        z = utils.log_zgrid(zr=zr, dz=dz0)
        chi2 = z*0.
        for i in range(len(z)):
            if eazyp:
                out = self.fit_combined_at_z(z=z[i], eazyp=eazyp, ix=ix, order=order, scale_fit=scale_fit)
                chi2[i], bg, full, coeffs, err, scale_fit = out            
            else:
                out = self.fit_at_z(z=z[i], templates=t_complex)
                chi2[i], bg, full, coeffs, err = out
            
            if verbose:
                print('{0:.4f} - {1:10.1f}'.format(z[i], chi2[i]))
        
        # Zoom in on the chi-sq minimum.
        ci = chi2
        zi = z
        for iter in range(1,7):
            if prior is not None:
                pz = np.interp(zi, prior[0], prior[1])
                cp = ci+pz
            else:
                cp = ci
                
            iz = np.argmin(cp)
            z0 = zi[iz]
            dz = dz0/2.02**iter
            zi = utils.log_zgrid(zr=[z0-dz*4, z0+dz*4], dz=dz)
            ci = zi*0.
            for i in range(len(zi)):
                
                if eazyp:
                    out = self.fit_combined_at_z(z=zi[i], eazyp=eazyp, ix=ix, order=order, scale_fit=scale_fit)
                    ci[i], bg, full, coeffs, err, scale_fit = out            
                else:
                    out = self.fit_at_z(z=zi[i], templates=t_complex, fitter=fitter)
                    ci[i], bg, full, coeffs, err = out
                
                # out = self.fit_at_z(z=zi[i], templates=t_complex,
                #                     fitter=fitter)
                # 
                # ci[i], bg, full, coeffs, err = out
                
                if verbose:
                    print('{0:.4f} - {1:10.1f}'.format(zi[i], ci[i]))
            
            z = np.append(z, zi)
            chi2 = np.append(chi2, ci)
        
        so = np.argsort(z)
        z = z[so]
        chi2 = chi2[so]
        
        # Apply the prior
        if prior is not None:
            pz = np.interp(z, prior[0], prior[1])
            chi2 += pz
        
        # Get the fit with the individual line templates at the best redshift
        chi2x, bgz, fullz, coeffs, err = self.fit_at_z(z=z[np.argmin(chi2)], templates=t_i, fitter=fitter, get_uncertainties=True)
        
        # Mask outliers
        if outlier_threshold > 0:
            resid = self.scif - fullz - bgz
            outlier_mask = (resid*self.sivarf < outlier_threshold)
            #outlier_mask &= self.fit_mask
            #self.sivarf[outlier_mask] = 1/resid[outlier_mask]
            #print('Mask {0} pixels with resid > {1} sigma'.format((outlier_mask).sum(), outlier_threshold))
            
            print('Mask {0} pixels with resid > {1} sigma'.format((~outlier_mask & self.fit_mask).sum(), outlier_threshold))
            self.fit_mask &= outlier_mask
            #self.DoF = self.fit_mask.sum() #(self.ivar > 0).sum()
            self.DoF = int((self.fit_mask*self.weightf).sum())
            
        # Table with z, chi-squared
        t = utils.GTable()
        t['z'] = z
        t['chi2'] = chi2
        
        if prior is not None:
            t['prior'] = pz
            
        # "area" parameter for redshift quality.
        num = np.trapz(np.clip(chi2-chi2.min(), 0, 25), z)
        denom = np.trapz(z*0+25, z)
        area25 = 1-num/denom
        
        # "best" redshift
        zbest = z[np.argmin(chi2)]
        
        # Metadata will be stored as header keywords in the FITS table
        t.meta = OrderedDict()
        t.meta['ID'] = (self.h0['ID'], 'Object ID')
        t.meta['RA'] = (self.h0['RA'], 'Right Ascension')
        t.meta['DEC'] = (self.h0['DEC'], 'Declination')
        t.meta['Z'] = (zbest, 'Best-fit redshift')
        t.meta['CHIMIN'] = (chi2.min(), 'Min Chi2')
        t.meta['CHIMAX'] = (chi2.max(), 'Max Chi2')
        t.meta['DOF'] = (self.DoF, 'Degrees of freedom')
        t.meta['AREA25'] = (area25, 'Area under CHIMIN+25')
        t.meta['FITTER'] = (fitter, 'Minimization algorithm')
        t.meta['HASPRIOR'] = (prior is not None, 'Was prior specified?')
        
        # Best-fit templates
        tc, tl = self.generate_1D_templates(coeffs,
                                            templates_file=templates_file)
        # for i, te in enumerate(t_i):
        #     if i == 0:
        #         tc = t_i[te].zscale(0, scalar=coeffs[i])
        #         tl = t_i[te].zscale(0, scalar=coeffs[i])
        #     else:
        #         if te.startswith('line'):
        #             tc += t_i[te].zscale(0, scalar=0.)
        #         else:
        #             tc += t_i[te].zscale(0, scalar=coeffs[i])
        #            
        #         tl += t_i[te].zscale(0, scalar=coeffs[i])
            
        # Get line fluxes, uncertainties and EWs
        il = 0
        for i, te in enumerate(t_i):
            if te.startswith('line'):
                il+=1
                
                if coeffs[i] == 0:
                    EW = 0.
                else:
                    tn = (t_i[te].zscale(0, scalar=coeffs[i]) +
                              tc.zscale(0, scalar=1))
                              
                    td = (t_i[te].zscale(0, scalar=0) + 
                             tc.zscale(0, scalar=1))
                             
                    clip = (td.wave <= t_i[te].wave.max())
                    clip &= (td.wave >= t_i[te].wave.min())
                    
                    EW = np.trapz((tn.flux/td.flux)[clip]-1, td.wave[clip])
                    if not np.isfinite(EW):
                        EW = -1000.
                        
                t.meta['LINE{0:03d}F'.format(il)] = (coeffs[i], 
                                       '{0} line flux'.format(te[5:]))
                
                t.meta['LINE{0:03d}E'.format(il)] = (err[i], 
                            '{0} line flux uncertainty'.format(te[5:]))
                
                #print('xxx EW', EW)
                t.meta['LINE{0:03d}W'.format(il)] = (EW, 
                            '{0} line rest EQW'.format(te[5:]))
                
        tfile = self.file.replace('.fits', '.zfit.fits')
        if os.path.exists(tfile):
            os.remove(tfile)

        t.write(tfile)
        
        ### Add image HDU and templates
        hdu = pyfits.open(tfile, mode='update') 
        hdu[1].header['EXTNAME'] = 'ZFIT'
        
        # oned_templates = np.array([tc.wave*(1+zbest), tc.flux/(1+zbest),
        #                            tl.flux/(1+zbest)])
        header = pyfits.Header()
        header['TEMPFILE'] = (templates_file, 'File with stored templates')
        hdu.append(pyfits.ImageHDU(data=coeffs, name='COEFFS'))
        
        for i in range(self.N):
            E = self.beams[i]
            model_i = fullz[self.slices[i]].reshape(E.sh)
            bg_i = bgz[self.slices[i]].reshape(E.sh)
            
            model_i[~np.isfinite(model_i)] = 0
            bg_i[~np.isfinite(bg_i)] = 0
            
            hdu.append(pyfits.ImageHDU(data=model_i, header=E.header,
                                       name='MODEL'))
        
            hdu.append(pyfits.ImageHDU(data=bg_i, header=E.header,
                                       name='BACKGROUND'))
        
        hdu.flush()
                                       
        if make_plot:
            self.make_fit_plot(hdu)

        return hdu
    
    @classmethod
    def generate_1D_templates(self, coeffs, templates_file='templates.npy'):
        """
        TBD
        """
        t_complex, t_i = np.load(templates_file)
        
        # Best-fit templates
        for i, te in enumerate(t_i):
            if i == 0:
                tc = t_i[te].zscale(0, scalar=coeffs[i])
                tl = t_i[te].zscale(0, scalar=coeffs[i])
            else:
                if te.startswith('line'):
                    tc += t_i[te].zscale(0, scalar=0.)
                else:
                    tc += t_i[te].zscale(0, scalar=coeffs[i])
                   
                tl += t_i[te].zscale(0, scalar=coeffs[i])
        
        return tc, tl
    
    def mask_drizzle_overlaps(self, threshold=3, verbose=True):
        """
        TBD
        
        mask pixels in beams of a given grism that are greater than 
        `threshold` sigma times the minimum of all beams of that 
        grism
        """
        min_grism = {}
        for grism in self.grisms:
            for E in self.beams:
                if not E.extver.startswith(grism):
                    continue
                
                if grism not in min_grism:
                    min_grism[grism] = E.scif*E.fit_mask
                else:
                    empty = (min_grism[grism] == 0) & E.fit_mask
                    min_grism[grism][E.fit_mask] = np.minimum(min_grism[grism][E.fit_mask], E.scif[E.fit_mask])
                    min_grism[grism][empty] = E.scif[empty]
                    
        for grism in self.grisms:
            for E in self.beams:
                if not E.extver.startswith(grism):
                    continue
                
                new_mask = (E.scif - min_grism[grism]) < threshold/E.sivarf                
                if verbose:
                    print('Mask {0} additional pixels for ext {1}'.format((~new_mask & E.fit_mask).sum(), E.extver))
                
                E.fit_mask &= new_mask
                            
    def make_fit_plot(self, hdu, scale_pz=True, show_2d=False):
        """Make a plot showing the fit
        
        Parameters
        ----------
        hdu : `~astropy.io.fits.HDUList`
            Fit results from `fit_zgrid`.
        
        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            The figure object.

        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec
        from matplotlib.ticker import MultipleLocator
        
        #import grizli
        from . import utils
        
        zfit = utils.GTable.read(hdu[1])
        z = zfit['z']
        chi2 = zfit['chi2']
        
        # Initialize plot window
        Ng = len(self.grisms)
        if show_2d:
            height_ratios = [0.25]*self.N
            height_ratios.append(1)
            gs = matplotlib.gridspec.GridSpec(self.N+1,2, 
                                width_ratios=[1,1.5+0.5*(Ng>1)],
                                height_ratios=height_ratios, hspace=0.)
                
            fig = plt.figure(figsize=[8+4*(Ng>1), 3.5+0.5*self.N])
        else:
            gs = matplotlib.gridspec.GridSpec(1,2, 
                            width_ratios=[1,1.5+0.5*(Ng>1)],
                            hspace=0.)
                
            fig = plt.figure(figsize=[8+4*(Ng>1), 3.5])
            
        # Chi-squared
        axz = fig.add_subplot(gs[-1,0]) #121)
        
        axz.text(0.95, 0.04, self.file + '\n'+'z={0:.4f}'.format(z[np.argmin(chi2)]), ha='right', va='bottom', transform=axz.transAxes, fontsize=9)
        
        # Scale p(z) to DoF=1.
        if scale_pz:
            scale_nu = chi2.min()/self.DoF
            scl_label = '_s'
        else:
            scale_nu = 1.
            scl_label = ''
            
        axz.plot(z, (chi2-chi2.min())/scale_nu, color='k')
        axz.fill_between(z, (chi2-chi2.min())/scale_nu, 27, color='k', alpha=0.5)
        axz.set_ylim(-4,27)
        axz.set_xlabel(r'$z$')
        axz.set_ylabel(r'$\Delta\chi^2{2}$ ({0:.0f}/$\nu$={1:d})'.format(chi2.min(), self.DoF, scl_label))
        axz.set_yticks([1,4,9,16,25])
        
        axz.set_xlim(z.min(), z.max())
        axz.grid()
        
        # 2D spectra
        if show_2d:
            twod_axes = []
            for i in range(self.N):
                ax_i = fig.add_subplot(gs[i,1])

                model = hdu['MODEL', self.ext[i]].data
                ymax = model[np.isfinite(model)].max()
                #print('MAX', ymax)
            
                cmap = 'viridis_r'
                cmap = 'cubehelix_r'
            
                clean = self.beams[i].sci - hdu['BACKGROUND', self.ext[i]].data
                clean *= self.beams[i].fit_mask.reshape(self.beams[i].sh)
            
                w = self.beams[i].wave/1.e4
            
                ax_i.imshow(clean, vmin=-0.02*ymax, vmax=1.1*ymax, origin='lower',
                            extent=[w[0], w[-1], 0., 1.], aspect='auto',
                            cmap=cmap)
            
                ax_i.text(0.04, 0.92, self.ext[i], ha='left', va='top', transform=ax_i.transAxes, fontsize=8)
                        
                ax_i.set_xticklabels([]); ax_i.set_yticklabels([])
                twod_axes.append(ax_i)
                    
        axc = fig.add_subplot(gs[-1,1]) #224)
        
        # 1D Spectra + fit
        ymin = 1.e30
        ymax = -1.e30
        wmin = 1.e30
        wmax = -1.e30
        
        for i in range(self.N):
            
            E = self.beams[i]
            
            clean = E.sci - hdu['BACKGROUND', self.ext[i]].data
            w, fl, er = E.optimal_extract(clean)            
            w, flm, erm = E.optimal_extract(hdu['MODEL', self.ext[i]].data)
            w = w/1.e4
            
            # Do we need to convert to F-lambda units?
            if E.is_flambda:
                unit_corr = 1.
                clip = (er > 0) & np.isfinite(er) & np.isfinite(flm)
                clip[:10] = False
                clip[-10:] = False
                
                if clip.sum() == 0:
                    clip = (er > -1)
            
            else:
                unit_corr = 1./E.sens
                clip = (E.sens > 0.1*E.sens.max()) 
                clip &= (np.isfinite(flm)) & (er > 0)
                
            
            if clip.sum() == 0:
                continue
                
            # fl *= 100*unit_corr
            # er *= 100*unit_corr
            # flm *= 100*unit_corr

            fl *= unit_corr/1.e-19
            er *= unit_corr/1.e-19
            flm *= unit_corr/1.e-19
            
            f_alpha = 1./self.Ngrism[E.grism]**0.5 #self.h0['N{0}'.format(E.header['GRISM'])]**0.5
            
            axc.errorbar(w[clip], fl[clip], er[clip], color=GRISM_COLORS[E.grism], alpha=0.3*f_alpha, marker='.', linestyle='None')
            #axc.fill_between(w[clip], (fl+er)[clip], (fl-er)[clip], color='k', alpha=0.2)
            axc.plot(w[clip], flm[clip], color='r', alpha=0.6*f_alpha, linewidth=2) 
              
            # Plot limits              
            # ymax = np.maximum(ymax, (flm+np.median(er))[clip].max())
            # ymin = np.minimum(ymin, (flm-er*0.)[clip].min())
            
            ymax = np.maximum(ymax,
                              np.percentile((flm+np.median(er))[clip], 98))
            
            ymin = np.minimum(ymin, np.percentile((flm-er*0.)[clip], 2))
            
            wmax = np.maximum(wmax, w.max())
            wmin = np.minimum(wmin, w.min())
                    
        axc.set_xlim(wmin, wmax)
        axc.semilogx(subsx=[wmax])
        #axc.set_xticklabels([])
        axc.set_xlabel(r'$\lambda$')
        axc.set_ylabel(r'$f_\lambda \times 10^{-19}$')
        #axc.xaxis.set_major_locator(MultipleLocator(0.1))
        
        axc.set_ylim(ymin-0.2*ymax, 1.2*ymax)
        axc.grid()
                
        for ax in [axc]: #[axa, axb, axc]:
            
            labels = np.arange(np.ceil(wmin*10), np.ceil(wmax*10))/10.
            ax.set_xticks(labels)
            ax.set_xticklabels(labels)
            #ax.set_xticklabels([])
            #print(labels, wmin, wmax)
            
        if show_2d:
            for ax in twod_axes:
                ax.set_xlim(wmin, wmax)
            
        gs.tight_layout(fig, pad=0.1, w_pad=0.1)
        fig.savefig(self.file.replace('.fits', '.zfit.png'))        
        return fig
                
class StackedSpectrum(object):
    def __init__(self, file='gnt_18197.stack.G141.285.fits', sys_err=0.02, mask_min=0.1, extver='G141', mask_threshold=7, fcontam=1., min_ivar=0.001, MW_EBV=0.):
        import os
        #import grizli
        from . import GRIZLI_PATH
        from . import grismconf
        
        self.sys_err = sys_err
        self.mask_min = mask_min
        self.extver = extver
        self.grism = self.extver.split(',')[0]
        self.mask_threshold=mask_threshold
        
        self.MW_EBV = MW_EBV
        self.init_galactic_extinction(MW_EBV)
        
        self.file = file
        self.hdulist = pyfits.open(file)
        
        self.h0 = self.hdulist[0].header.copy()
        self.header = self.hdulist['SCI',extver].header.copy()
        self.sh = (self.header['NAXIS2'], self.header['NAXIS1'])
        self.wave = self.get_wavelength_from_header(self.header)
        self.wavef = np.dot(np.ones((self.sh[0],1)), self.wave[None,:]).flatten()
        
        if 'BEAM' in self.header:
            self.beam_name = self.header['BEAM']
        else:
            self.beam_name = 'A'
            
        # Configuration file
        self.is_flambda = self.header['ISFLAM']
        self.conf_file = self.header['CONF']
        try:
            self.conf = grismconf.aXeConf(self.conf_file)
        except:
            # Try global path 
            base = os.path.basename(self.conf_file)
            localfile = os.path.join(GRIZLI_PATH, 'CONF', base)
            self.conf = grismconf.aXeConf(localfile)
            
        self.conf.get_beams()
        
        self.sci = self.hdulist['SCI',extver].data*1.
        self.ivar0 = self.hdulist['WHT',extver].data*1
        self.size = self.sci.size
        self.thumbs = {}
        
        self.scif = self.sci.flatten()
        self.ivarf0 = self.ivar0.flatten()
        
        self.ivarf = 1/(1/self.ivarf0 + (sys_err*self.scif)**2)
        self.ivar = self.ivarf.reshape(self.sh)
        
        self.sivarf = np.sqrt(self.ivar).flatten()
        
        self.fit_mask = (self.ivarf > min_ivar*self.ivarf.max()) 
        self.fit_mask &= np.isfinite(self.scif) & np.isfinite(self.ivarf)
        
        # Dummy parameter.  Implemented for MultiBeam
        self.sens_mask = 1.
        
        # Contamination weighting
        self.fcontam = fcontam
        if ('CONTAM',extver) in self.hdulist:
            self.contam = np.abs(self.hdulist['CONTAM',extver].data*1.)
            self.weight = np.exp(-fcontam*self.contam*np.sqrt(self.ivar0))            
            self.contamf = self.contam.flatten()
            self.weightf = self.weight.flatten()
        else:
            self.contam = self.sci*0.
            self.contamf = self.scif*0.
            
            self.weight = self.sci*0.+1
            self.weightf = self.scif*0.+1
        
        # Spatial kernel
        self.kernel = self.hdulist['KERNEL',extver].data*1
        self.kernel /= self.kernel.sum()
        
        self._build_model()
        
        self.flat = self.compute_model()
        
        if self.is_flambda:
            elec = self.flat*self.ivarf0/self.ivarf0.max()
        else:
            elec = self.flat
            
        self.fit_mask &= elec > mask_min*elec.max()
        
        self.DoF = self.fit_mask.sum() #(self.ivar > 0).sum()
        
        if mask_threshold > 0:
            self.drizzle_mask(mask_threshold=mask_threshold)
        
        self.flat_flam = self.compute_model(in_place=False, is_cgs=True)
        
    def init_galactic_extinction(self, MW_EBV=0., R_V=utils.MW_RV):
        """
        Initialize Fitzpatrick 99 Galactic extinction
        
        Parameters
        ----------
        MW_EBV : float
            Local E(B-V)
        
        R_V : float
            Relation between specific and total extinction, 
            ``a_v = r_v * ebv``.
        
        Returns
        -------
        Sets `self.MW_F99` attribute, which is a callable function that 
        returns the extinction for a supplied array of wavelengths.
        
        If MW_EBV <= 0, then sets `self.MW_F99 = None`.
        
        """
        self.MW_F99 = None
        if MW_EBV > 0:
            try:
                from specutils.extinction import ExtinctionF99
                self.MW_F99 = ExtinctionF99(MW_EBV*R_V, r_v=R_V)
            except(ImportError):
                print('Couldn\'t import `specutils.extinction`, MW extinction not implemented')
                
    @classmethod    
    def get_wavelength_from_header(self, h):
        """
        Generate wavelength array from WCS keywords
        """
        w = (np.arange(h['NAXIS1'])+1-h['CRPIX1'])*h['CD1_1'] + h['CRVAL1']
        
        # Now header keywords scaled to microns
        if w.max() < 3:
            w *= 1.e4
        
        return w
    
    def init_optimal_profile(self):
        """Initilize optimal extraction profile
        """
        m = self.compute_model(in_place=False)
        m = m.reshape(self.sh)
        m[m < 0] = 0
        self.optimal_profile = m/m.sum(axis=0)
               
    def optimal_extract(self, data, bin=0, ivar=None, weight=None, loglam=False):
        """
        Optimally-weighted 1D extraction
        
        xx Dummy parameters ivar, weight, loglam
        """
        import scipy.ndimage as nd
        
        # flatf = self.flat.reshape(self.sh).sum(axis=0)
        # prof = self.flat.reshape(self.sh)/flatf
        # 
        if not hasattr(self, 'optimal_profile'):
            self.init_optimal_profile()
        
        prof = self.optimal_profile
        
        if ivar is None:
            ivar = self.ivar
            
        num = prof*data*ivar*self.weight
        den = prof**2*ivar*self.weight
        opt_flux = num.sum(axis=0)/den.sum(axis=0)
        opt_var = 1./den.sum(axis=0)
        
        opt_rms = np.sqrt(opt_var)
        clip = (opt_var == 0) | ~np.isfinite(opt_var)
        opt_rms[clip] = 0
        opt_flux[clip] = 0
        
        if bin > 1:
            kern = np.ones(bin, dtype=float)/bin
            opt_flux = nd.convolve(opt_flux, kern)[bin // 2::bin]
            opt_var = nd.convolve(opt_var, kern**2)[bin // 2::bin]
            opt_rms = np.sqrt(opt_var)            
            wave = self.wave[bin // 2::bin]
        else:
            wave = self.wave
        
        return wave, opt_flux, opt_rms
        
    def _build_model(self):
        """
        Initiazize components for generating 2D model
        """
        from .utils_c.interp import interp_conserve_c
        
        NY = self.sh[0]
        data = np.zeros((self.header['NAXIS1'], self.header['NAXIS2'], self.header['NAXIS1']))
                                            
        for j in range(NY//2):
            data[j,:,:j+NY//2] += self.kernel[:, -NY//2-j:]
        
        for j in range(self.sh[1]-NY//2, self.sh[1]):
            data[j,:,-NY//2+j:] += self.kernel[:, :self.sh[1]-j+NY//2]
        
        for j in range(NY//2, self.sh[1]-NY//2):
            #print(j)
            data[j,:,j-NY//2:j+NY//2] += self.kernel
                
        self.fit_data = data.reshape(self.sh[1],-1)
        
        if not self.is_flambda:
            conf_sens = self.conf.sens[self.beam_name]
            if self.MW_F99 is not None:
                MWext = 10**(-0.4*(self.MW_F99(conf_sens['WAVELENGTH']*u.AA)))
            else:
                MWext = 1.
            
            sens = interp_conserve_c(self.wave, conf_sens['WAVELENGTH'],
                                     conf_sens['SENSITIVITY']*MWext)
            
            if 'DLAM0' in self.header:
                #print('xx Header')
                dlam = self.header['DLAM0']
            else:
                dlam = np.median(np.diff(self.wave))
            
            #dlam = np.median(np.diff(self.wave))
            dlam = np.gradient(self.wave)
            
            # G800 variable dispersion, which isn't coming out right 
            # in the 2D drizzled spectrum
            if self.grism == 'G800L':
                ## Linear fit to differential G800L dispersion
                # y = np.diff(beam.wave)/np.diff(beam.wave)[0]
                # x = beam.wave[1:]
                # c_i = np.polyfit(x/1.e4, y, 1)
                c_i = np.array([ 0.07498747,  0.98928126])
                scale = np.polyval(c_i, self.wave/1.e4)
                sens *= scale
                
            self.sens = sens*dlam #*1.e-17
            self.fit_data = (self.fit_data.T*self.sens).T
            
    def compute_model(self, spectrum_1d=None, is_cgs=None, in_place=False):
        """
        Generate the model spectrum
        
        xxx is_cgs and in_place are dummy parameters to match `MultiBeam.compute_model`.
        """
        from .utils_c.interp import interp_conserve_c
        
        if spectrum_1d is None:
            fl = self.wave*0+1
        else:
            fl = interp_conserve_c(self.wave, spectrum_1d[0], spectrum_1d[1])
            
        model = np.dot(fl, self.fit_data)#.reshape(self.sh)
        #self.model = model
        return model
        
        