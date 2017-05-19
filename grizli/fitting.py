"""
fitting.py

Created by Gabriel Brammer on 2017-05-19.

"""
import os
import glob

from collections import OrderedDict

import numpy as np

import astropy.io.fits as pyfits

from . import utils
from .model import BeamCutout
from .utils import GRISM_COLORS

PLINE = {'kernel': 'point', 'pixfrac': 0.2, 'pixscale': 0.1, 'size': 10, 'wcs': None}

def run_all(id, t0=None, t1=None, fwhm=1200, zr=[0.65, 1.6], dz=[0.005, 0.0005], group_name='grism', fit_stacks=True, prior=None, fcontam=0.2, pline=PLINE):
    """Run the full procedure
    
    1) Load MultiBeam and stack files 
    2) 
    """
    import glob
    import grizli.multifit
    from grizli.stack import StackFitter
    from grizli.multifit import MultiBeam
    
    mb_files = glob.glob('*{0:05d}.beams.fits'.format(id))
    st_files = glob.glob('*{0:05d}.stack.fits'.format(id))
    
    st = StackFitter(st_files, fit_stacks=fit_stacks, group_name=group_name, fcontam=fcontam)
    
    mb = MultiBeam(mb_files[0], fcontam=fcontam, group_name=group_name)
    if len(mb_files) > 1:
        for file in mb_files[1:]:
            mb.extend(MultiBeam(file, fcontam=fcontam, group_name=group_name))
            
    if t0 is None:
        t0 = grizli.utils.load_templates(line_complexes=True, fsps_templates=True, fwhm=fwhm)
    
    if t1 is None:
        t1 = grizli.utils.load_templates(line_complexes=False, fsps_templates=True, fwhm=fwhm)
        
    # Fit on stacked spectra
    fit = st.xfit_redshift(templates=t0, zr=zr, dz=dz, prior=prior) 
    fit_hdu = pyfits.table_to_hdu(fit)
    fit_hdu.header['EXTNAME'] = 'ZFIT_STACK'
    
    # Zoom-in fit with individual beams
    mb_zr = fit.meta['Z50'][0] + 10*fit.meta['ZWIDTH1'][0]*np.array([-1,1])
    mb_fit = mb.xfit_redshift(templates=t0, zr=mb_zr, dz=[0.0005, 0.0002], prior=prior) 
    mb_fit_hdu = pyfits.table_to_hdu(mb_fit)
    mb_fit_hdu.header['EXTNAME'] = 'ZFIT_BEAM'
    
    # Get best-fit template of beams
    tfit = mb.template_at_z(z=mb_fit.meta['z_risk'][0], templates=t1, fit_background=True)
    tfit_sp = grizli.utils.GTable()
    for ik, key in enumerate(tfit['cfit']):
        tfit_sp.meta['CVAL{0:03d}'.format(ik)] = tfit['cfit'][key][0], 'Coefficient for {0}'.format(key)
        tfit_sp.meta['CERR{0:03d}'.format(ik)] = tfit['cfit'][key][1], 'Uncertainty for {0}'.format(key)
        
    tfit_sp['wave'] = tfit['cont1d'].wave
    tfit_sp['cont'] = tfit['cont1d'].flux
    tfit_sp['full'] = tfit['line1d'].flux
    
    tfit_sp['wave'].unit = tfit['cont1d'].waveunits
    tfit_sp['cont'].unit = tfit['cont1d'].fluxunits
    tfit_sp['full'].unit = tfit['line1d'].fluxunits
    
    tfit_hdu = pyfits.table_to_hdu(tfit_sp)
    tfit_hdu.header['EXTNAME'] = 'TEMPL'
     
    # Make the plot
    fig = mb.xmake_fit_plot(mb_fit, tfit)
    fig.axes[0].plot(fit['zgrid'], np.log10(fit['pdf']), color='0.5', alpha=0.5)
    fig.axes[0].set_xlim(fit['zgrid'].min(), fit['zgrid'].max())
    fig.savefig('{0}_{1:05d}.full.png'.format(group_name, id))
    
    # Make the line maps
    if pline is None:
         pzfit, pspec2, pline = grizli.multifit.get_redshift_fit_defaults()
    
    line_hdu = mb.drizzle_fit_lines(tfit, pline, force_line=['SIII','SII','Ha', 'OIII', 'Hb', 'OII'], save_fits=False, mask_lines=True, mask_sn_limit=3)
    
    line_hdu.insert(1, fit_hdu)
    line_hdu.insert(2, mb_fit_hdu)
    line_hdu.insert(3, tfit_hdu)
    
    line_hdu.writeto('{0}_{1:05d}.full.fits'.format(group_name, id), clobber=True, output_verify='fix')
    
def _loss(dz, gamma=0.15):
    """Risk / Loss function, Tanaka et al. (https://arxiv.org/abs/1704.05988)
    
    Parameters
    ----------
    gamma : float
    
    Returns
    -------
    loss : float
    """
    return 1-1/(1+(dz/gamma)**2)
    
class GroupFitter(object):
    """Combine stack.StackFitter and MultiBeam fitting into a single object
    
    Will have to match the attributes between the different objects, which 
    is already close.
    """
    def _test(self):
        print(self.Ngrism)
    
    def _get_slices(self):
        """Precompute array slices for how the individual components map into the single combined arrays.
        
        Parameters
        ----------
        None 
        
        Returns
        -------
        slices : list
            List of slices.
        """
        x = 0
        slices = []
        for i in range(self.N):
            slices.append(slice(x+0, x+self.beams[i].size))
            x += self.beams[i].size
        
        return slices    
    
    def _init_background(self):
        """Initialize the (flat) background model components
        
        Parameters
        ----------
        None :
        
        Returns
        -------
        A_bg : `~np.ndarray`
            
        """
        A_bg = np.zeros((self.N, self.Ntot))
        for i in range(self.N):
            A_bg[i, self.slices[i]] = 1.
        
        return A_bg
    
    def set_photometry(self, flam=[], eflam=[], filters=[], force=False, tempfilt=None):
        """
        Add photometry
        """
        if (self.Nphot > 0) & (not force):
            print('Photometry already set (Nphot={0})'.format(self.Nphot))
            return True
            
        self.Nphot = len(flam)
        if self.Nphot == 0:
            return True
        
        if (len(flam) != len(eflam)) | (len(flam) != len(filters)):
            print('flam/eflam/filters dimensions don\'t match')
            return False
            
        self.photom_flam = flam
        self.photom_eflam = eflam
        self.photom_filters = filters
        
        self.sivarf = np.hstack((self.sivarf, 1/self.photom_eflam))
        self.fit_mask = np.hstack((self.fit_mask, eflam > 0))
        self.scif = np.hstack((self.scif, flam))
        self.DoF = self.fit_mask.sum()
        
        self.is_spec = np.isfinite(self.scif)
        self.is_spec[-self.Nphot:] = False
        
        self.photom_pivot = np.array([filter.pivot() for filter in filters])
        
        # eazypy tempfilt for faster interpolation
        self.tempfilt = tempfilt
        
    def unset_photometry(self):
        if self.Nphot == 0:
            return True
            
        self.sivarf = self.sivarf[:-self.Nphot]
        self.fit_mask = self.fit_mask[:-self.Nphot]
        self.scif = self.scif[:-self.Nphot]
        self.DoF = self.fit_mask.sum()
        
        self.is_spec = 1
        self.Nphot = 0
        
    def _interpolate_photometry(self, z=0., templates=[]):
        """
        Interpolate templates through photometric filters
        
        xx: TBD better handling of emission line templates and use eazpy tempfilt
        object for huge speedup
        
        """
        NTEMP = len(templates)
        A_phot = np.zeros((NTEMP+self.N, self.Nphot))
        
        if (self.tempfilt is not None):
            if (self.tempfilt.NTEMP == NTEMP):
                #A_spl = self.tempfilt(z)
                A_phot[self.N:,:] = self.tempfilt(z)
                A_phot *= 3.e18/self.photom_pivot**2*(1+z)
                A_phot[~np.isfinite(A_phot)] = 0
                return A_phot

        for it, key in enumerate(templates):
            tz = templates[key].zscale(z, scalar=1)
            for ifilt, filt in enumerate(self.photom_filters):
                A_phot[self.N+it, ifilt] = tz.integrate_filter(filt)*3.e18/self.photom_pivot[ifilt]**2#*(1+z)
        
        return A_phot
        
    def xfit_at_z(self, z=0, templates=[], fitter='nnls', fit_background=True, get_uncertainties=False):
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
        
        fit_background : bool
            Fit additive pedestal background offset.
            
        get_uncertainties : bool
            Compute coefficient uncertainties from the covariance matrix
        
        
        Returns
        -------
        chi2 : float
            Chi-squared of the fit
        
        background : `~np.ndarray`
            Background model
        
        model2d : `~np.ndarray`
            Best-fit 2D model.
        
        coeffs, coeffs_err : `~np.ndarray`
            Template coefficients and uncertainties.
        
        """
        import scipy.optimize
        
        NTEMP = len(templates)
        A = np.zeros((self.N+NTEMP, self.Ntot))
        A[:self.N,:] += self.A_bg*fit_background
                        
        for i, t in enumerate(templates):
            ti = templates[t]
            try:
                import eazy.igm
                if z > 5:
                    igm = eazy.igm.Inoue14()
                    igmz = igm.full_IGM(z, ti.wave*(1+z))         
                else:
                    igmz = 1.
            except:
                igmz = 1.

            s = [ti.wave*(1+z), ti.flux/(1+z)*igmz]
            
            for j, beam in enumerate(self.beams):
                mask_i = beam.fit_mask.reshape(beam.sh)
                clip = mask_i.sum(axis=0) > 0        
                if clip.sum() == 0:
                    continue
                                
                lam_beam = beam.wave[clip]
                if ((s[0].min() > lam_beam.max()) | 
                    (s[0].max() < lam_beam.min())):
                    continue

                sl = self.slices[j]
                A[self.N+i, sl] = beam.compute_model(spectrum_1d=s, in_place=False, is_cgs=True)
                            
        if fit_background:
            if fitter == 'nnls':
                pedestal = 0.04
            else:
                pedestal = 0.
        else:
            pedestal = 0
                    
        # Photometry
        if self.Nphot > 0:
            A_phot = self._interpolate_photometry(z=z, templates=templates)
            A = np.hstack((A, A_phot))
            
        oktemp = (A*self.fit_mask).sum(axis=1) != 0
        
        # Weight design matrix and data by 1/sigma
        Ax = A[oktemp,:]*self.sivarf        
        AxT = Ax[:,self.fit_mask].T
        data = ((self.scif+pedestal*self.is_spec)*self.sivarf)[self.fit_mask]
        
        # Run the minimization
        if fitter == 'nnls':
            coeffs_i, rnorm = scipy.optimize.nnls(AxT, data)            
        else:
            coeffs_i, residuals, rank, s = np.linalg.lstsq(AxT, data)
       
        # Compute background array         
        if fit_background:
            background = np.dot(coeffs_i[:self.N], A[:self.N,:]) - pedestal
            background[-self.Nphot:] = 0.
            coeffs_i[:self.N] -= pedestal
        else:
            background = self.scif*0.
        
        # Full model
        model2d = np.dot(coeffs_i[self.N:], Ax[self.N:,:]/self.sivarf)
        
        # Residuals and Chi-squared
        resid = self.scif - model2d - background
        chi2 = np.sum(resid[self.fit_mask]**2*self.sivarf[self.fit_mask]**2)
        
        # Uncertainties from covariance matrix
        if get_uncertainties:
            try:
                # Covariance is inverse of AT.A
                covar_i = np.matrix(np.dot(AxT.T, AxT)).I.A
                covar = utils.fill_masked_covar(covar_i, oktemp)
                covard = np.sqrt(covar.diagonal())
            except:
                print('Except: covar!')
                covar = np.zeros((self.N+NTEMP, self.N+NTEMP))
                covard = np.zeros(self.N+NTEMP)#-1.
        else:
            covar = np.zeros((self.N+NTEMP, self.N+NTEMP))
            covard = np.zeros(self.N+NTEMP)#-1.
        
        coeffs = np.zeros(self.N+NTEMP)
        coeffs[oktemp] = coeffs_i #[self.N:]] = coeffs[self.N:]

        coeffs_err = covard #np.zeros(NTEMP)
        #full_coeffs_err[oktemp[self.N:]] = covard[self.N:]
        
        return chi2, background, model2d, coeffs, coeffs_err, covar
    
    def xfit_redshift(self, prior=None, fwhm=1200,
                     make_figure=True, zr=[0.65, 1.6], dz=[0.005, 0.0004],
                     verbose=True, fit_background=True, fitter='nnls', 
                     delta_chi2_threshold=0.004, zoom=True, 
                     line_complexes=True, templates={}, figsize=[8,5],
                     fsps_templates=False):
        """TBD
        """
        from scipy import polyfit, polyval
        
        if zr is 0:
            stars = True
            zr = [0, 0.01]
            fitter='nnls'
        else:
            stars = False
            
        zgrid = utils.log_zgrid(zr, dz=dz[0])
        NZ = len(zgrid)
        
        #### Polynomial SED fit
        wpoly = np.arange(1000,5.e4,1000)
        tpoly = utils.polynomial_templates(wpoly, line=True)
        out = self.xfit_at_z(z=0., templates=tpoly, fitter='nnls',
                            fit_background=True)
        
        chi2_poly, b, m2, coeffs_poly, c, cov = out

        # tpoly = utils.polynomial_templates(wpoly, order=3)
        # out = self.xfit_at_z(z=0., templates=tpoly, fitter='lstsq',
        #                     fit_background=True)          
        # chi2_poly, b, m2, coeffs_poly, c, cov = out

        # if True:
        #     cp, lp = utils.dot_templates(coeffs_poly[self.N:], tpoly)
            
        ### Set up for template fit
        if templates == {}:
            templates = utils.load_templates(fwhm=fwhm, stars=stars, line_complexes=line_complexes, fsps_templates=fsps_templates)
        else:
            if verbose:
                print('User templates! N={0} \n'.format(len(templates)))
            
        NTEMP = len(templates)
        
        out = self.xfit_at_z(z=0., templates=templates, fitter=fitter,
                            fit_background=fit_background, 
                            get_uncertainties=True)
                            
        chi2, background, model2d, coeffs, coeffs_err, covar = out
        
        chi2 = np.zeros(NZ)
        coeffs = np.zeros((NZ, coeffs.shape[0]))
        covar = np.zeros((NZ, covar.shape[0], covar.shape[1]))
        
        chi2min = 1e30
        iz = 0
        for i in range(NZ):
            out = self.xfit_at_z(z=zgrid[i], templates=templates,
                                fitter=fitter, fit_background=fit_background,
                                get_uncertainties=True)
            
            chi2[i], background, model2d, coeffs[i,:], coeffs_err, covar[i,:,:] = out
            if chi2[i] < chi2min:
                iz = i
                chi2min = chi2[i]

            if verbose:                    
                print(utils.NO_NEWLINE + '  {0:.4f} {1:9.1f} ({2:.4f})'.format(zgrid[i], chi2[i], zgrid[iz]))
        
        if verbose:
            print('First iteration: z_best={0:.4f}\n'.format(zgrid[iz]))
            
        ## Find peaks
        import peakutils
        
        # Make "negative" chi2 for peak-finding
        if chi2_poly > (chi2.min()+100):
            chi2_rev = (chi2.min() + 100 - chi2)/self.DoF
        elif chi2_poly < (chi2.min() + 9):
            chi2_rev = (chi2.min() + 16 - chi2)/self.DoF
        else:
            chi2_rev = (chi2_poly - chi2)/self.DoF
            
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
            covar_zoom = np.zeros((NZOOM, coeffs.shape[1], covar.shape[2]))

            iz = 0
            chi2min = 1.e30
            for i in range(NZOOM):
                out = self.xfit_at_z(z=zgrid_zoom[i], templates=templates,
                                    fitter=fitter,
                                    fit_background=fit_background,
                                    get_uncertainties=True)

                chi2_zoom[i], b, m2, coeffs_zoom[i,:], e, covar_zoom[i,:,:] = out
                #A, coeffs_zoom[i,:], chi2_zoom[i], model_2d = out
                if chi2_zoom[i] < chi2min:
                    chi2min = chi2_zoom[i]
                    iz = i
                
                if verbose:
                    print(utils.NO_NEWLINE+'- {0:.4f} {1:9.1f} ({2:.4f}) {3:d}/{4:d}'.format(zgrid_zoom[i], chi2_zoom[i], zgrid_zoom[iz], i+1, NZOOM))
        
            zgrid = np.append(zgrid, zgrid_zoom)
            chi2 = np.append(chi2, chi2_zoom)
            coeffs = np.append(coeffs, coeffs_zoom, axis=0)
            covar = np.vstack((covar, covar_zoom))
            
        so = np.argsort(zgrid)
        zgrid = zgrid[so]
        chi2 = chi2[so]
        coeffs = coeffs[so,:]
        covar = covar[so,:,:]
        
        fit = utils.GTable()
        fit.meta['chi2poly'] = (chi2_poly, 'Chi^2 of polynomial fit')
        fit.meta['DoF'] = (self.DoF, 'Degrees of freedom (number of pixels)')
        fit.meta['chimin'] = (chi2.min(), 'Minimum chi2')
        fit.meta['chimax'] = (chi2.max(), 'Maximum chi2')
        fit.meta['fitter'] = (fitter, 'Minimization algorithm')
        
        fit.meta['NTEMP'] = (len(templates), 'Number of fitting templates')
        
        for i, tname in enumerate(templates):
            fit.meta['T{0:03d}NAME'.format(i+1)] = (templates[tname].name, 'Template name')
            fit.meta['T{0:03d}FWHM'.format(i+1)] = (templates[tname].fwhm, 'FWHM, if emission line')
        
        dtype = np.float64
        
        fit['zgrid'] = np.cast[dtype](zgrid)
        fit['chi2'] = np.cast[dtype](chi2)
        #fit['chi2poly'] = chi2_poly
        fit['coeffs'] = np.cast[dtype](coeffs)
        fit['covar'] = np.cast[dtype](covar)
        
        fit = self._parse_zfit_output(fit, prior=prior)
        
        return fit
    
    def _parse_zfit_output(self, fit, prior=None):
        """Parse best-fit redshift, etc.
        TBD
        """
        import scipy.interpolate
        
        # Normalize to min(chi2)/DoF = 1.
        scl_nu = fit['chi2'].min()/self.DoF
        
        # PDF
        pdf = np.exp(-0.5*(fit['chi2']-fit['chi2'].min())/scl_nu)
        
        if prior is not None:
            interp_prior = np.interp(zgrid, prior[0], prior[1])
            pdf *= interp_prior
            fit.meta['hasprior'] = True, 'Prior applied to PDF'
            fit['prior'] = interp_prior
        else:
            interp_prior = None
            fit.meta['hasprior'] = False, 'Prior applied to PDF'

        # Normalize PDF
        pdf /= np.trapz(pdf, fit['zgrid'])
        
        # Interpolate pdf for more continuous measurement
        spl = scipy.interpolate.Akima1DInterpolator(fit['zgrid'], np.log(pdf), axis=1)
        zfine = utils.log_zgrid(zr=[0.01,3.4], dz=0.0001)
        ok = np.isfinite(spl(zfine))
        norm = np.trapz(np.exp(spl(zfine[ok])), zfine[ok])
        
        # Compute CDF and probability intervals
        dz = np.gradient(zfine[ok])
        cdf = np.cumsum(np.exp(spl(zfine[ok]))*dz/norm)
        pz_percentiles = np.interp(np.array([2.5, 16, 50, 84, 97.5])/100., cdf, zfine[ok])

        # Random draws, testing
        #rnd = np.interp(np.random.rand(1000), cdf, fit['zgrid']+dz/2.)
        
        dz = np.gradient(fit['zgrid'])
        
        zsq = np.dot(fit['zgrid'][:,None], np.ones_like(fit['zgrid'])[None,:])
        L = _loss((zsq-fit['zgrid'])/(1+fit['zgrid']), gamma=0.01)
        
        risk = np.dot(pdf*L, dz)
        zi = np.argmin(risk)
        c = np.polyfit(fit['zgrid'][zi-1:zi+2], risk[zi-1:zi+2], 2)
        z_best = -c[1]/(2*c[0])
        risk_best = np.trapz(pdf*_loss((z_best-fit['zgrid'])/(1+fit['zgrid']), gamma=0.01), fit['zgrid'])
        
        # Store data in the fit table
        fit['pdf'] = pdf
        fit['risk'] = risk
        fit.meta['Z02'] = pz_percentiles[0], 'Integrated p(z) = 0.025'
        fit.meta['Z16'] = pz_percentiles[1], 'Integrated p(z) = 0.16'
        fit.meta['Z50'] = pz_percentiles[2], 'Integrated p(z) = 0.5'
        fit.meta['Z84'] = pz_percentiles[3], 'Integrated p(z) = 0.84'
        fit.meta['Z97'] = pz_percentiles[4], 'Integrated p(z) = 0.975'
        fit.meta['ZWIDTH1'] = pz_percentiles[3]-pz_percentiles[1], 'Width between the 16th and 84th p(z) percentiles'
        fit.meta['ZWIDTH2'] = pz_percentiles[4]-pz_percentiles[0], 'Width between the 2.5th and 97.5th p(z) percentiles'
        
        fit.meta['z_risk'] = z_best, 'Redshift at minimum risk'
        fit.meta['min_risk'] = risk_best, 'Minimum risk'
        fit.meta['gam_loss'] = 0.01, 'Gamma factor of the risk/loss function'
        return fit
                        
    def template_at_z(self, z=0, templates=None, fit_background=True, fitter='nnls', fwhm=1400):
        """TBD
        """
        if templates is None:
            templates = utils.load_templates(line_complexes=False, fsps_templates=True, fwhm=fwhm)
        
        out = self.xfit_at_z(z=z, templates=templates, fitter=fitter, 
                             fit_background=fit_background,
                             get_uncertainties=True)

        chi2, modelBG, model2D, coeffs, coeffs_err, covar = out
        cont1d, line1d = utils.dot_templates(coeffs[self.N:], templates, z=z)

        # Parse template coeffs
        cfit = OrderedDict()
        
        for i in range(self.N):
            cfit['bg {0:03d}'.format(i)] = coeffs[i], coeffs_err[i]
        
        for j, key in enumerate(templates):
            i = j+self.N
            cfit[key] = coeffs[i], coeffs_err[i]
        
        if False:
            # Compare drizzled and beam fits (very close)
            for j, key in enumerate(templates):
                print('{key:<16s} {0:.2e} {1:.2e}  {2:.2e} {3:.2e}'.format(mb_cfit[key][0], mb_cfit[key][1], st_cfit[key][0], st_cfit[key][1], key=key))
        
        tfit = OrderedDict()
        tfit['cont1d'] = cont1d
        tfit['line1d'] = line1d
        tfit['cfit'] = cfit
        tfit['covar'] = covar
        tfit['z'] = z
        tfit['templates'] = templates
        
        return tfit #cont1d, line1d, cfit, covar
        
        ### Random draws
        # Unique wavelengths
        wfull = np.hstack([templates[key].wave for key in templates])
        w = np.unique(wfull)
        so = np.argsort(w)
        w = w[so]
        
        xclip = (w*(1+z) > 7000) & (w*(1+z) < 1.8e4)
        temp = np.array([grizli.utils_c.interp.interp_conserve_c(w[xclip], templates[key].wave, templates[key].flux) for key in templates])
        
        clip = coeffs_err[self.N:] > 0
        covar_clip = covar[self.N:,self.N:][clip,:][:,clip]
        draws = np.random.multivariate_normal(coeffs[self.N:][clip], covar_clip, size=100)
        
        tdraw = np.dot(draws, temp[clip,:])/(1+z)
            
        for ib, beam in enumerate(self.beams):
            ww, ff, ee = beam.optimal_extract(beam.sci - beam.contam - coeffs[ib])
            plt.errorbar(ww, ff/beam.sens, ee/beam.sens, color='k', marker='.', linestyle='None', alpha=0.5)
            
            for i in range(tdraw.shape[0]):
                sp = [w[xclip]*(1+z), tdraw[i,:]]
                m = beam.compute_model(spectrum_1d=sp, is_cgs=True, in_place=False).reshape(beam.sh)
                ww, ff, ee = beam.optimal_extract(m)
                plt.plot(ww, ff/beam.sens, color='r', alpha=0.05)
                
            plt.plot(w[xclip]*(1+z), tdraw.T, alpha=0.05, color='r')
    
    def xmake_fit_plot(self, fit, tfit):
        """TBD
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec
        from matplotlib.ticker import MultipleLocator
        
        import grizli
        
        # Initialize plot window
        Ng = len(self.grisms)
        gs = matplotlib.gridspec.GridSpec(1,2, 
                        width_ratios=[1,1.5+0.5*(Ng>1)],
                        hspace=0.)
            
        fig = plt.figure(figsize=[8+4*(Ng>1), 3.5])
        
        # p(z)
        axz = fig.add_subplot(gs[-1,0]) #121)
        
        axz.text(0.95, 0.96, self.group_name + '\n'+'ID={0:<5d}  z={1:.4f}'.format(self.id, fit.meta['z_risk'][0]), ha='right', va='top', transform=axz.transAxes, fontsize=9)
                 
        axz.plot(fit['zgrid'], np.log10(fit['pdf']), color='k')
        #axz.fill_between(z, (chi2-chi2.min())/scale_nu, 27, color='k', alpha=0.5)
        
        axz.set_xlabel(r'$z$')
        axz.set_ylabel(r'$\log\ p(z)$'+' / '+ r'$\chi^2=\frac{{{0:.0f}}}{{{1:d}}}={2:.2f}$'.format(fit.meta['chimin'][0], fit.meta['DoF'][0], fit.meta['chimin'][0]/fit.meta['DoF'][0]))
        #axz.set_yticks([1,4,9,16,25])
        
        axz.set_xlim(fit['zgrid'].min(), fit['zgrid'].max())
        pzmax = np.log10(fit['pdf'].max())
        axz.set_ylim(pzmax-3, pzmax+0.2)
        axz.grid()
        axz.yaxis.set_major_locator(MultipleLocator(base=1))
        
        #### Spectra
        axc = fig.add_subplot(gs[-1,1]) #224)
        ymin = 1.e30
        ymax = -1.e30
        wmin = 1.e30
        wmax = -1.e30
        
        # 1D Model
        sp = tfit['line1d'].wave, tfit['line1d'].flux
        
        for i in range(self.N):
            beam = self.beams[i]
            m_i = beam.compute_model(spectrum_1d=sp, is_cgs=True, in_place=False).reshape(beam.sh)
            
            if isinstance(beam, BeamCutout):
                grism = beam.grism.filter
                clean = beam.grism['SCI'] - beam.contam - tfit['cfit']['bg {0:03d}'.format(i)][0]
                
                w, fl, er = beam.beam.optimal_extract(clean, ivar=beam.ivar)            
                w, flm, erm = beam.beam.optimal_extract(m_i, ivar=beam.ivar)
                sens = beam.beam.sensitivity                
            else:
                grism = beam.grism
                clean = beam.sci - beam.contam - tfit['cfit']['bg {0:03d}'.format(i)][0]
                w, fl, er = beam.optimal_extract(clean, ivar=beam.ivar)            
                w, flm, erm = beam.optimal_extract(m_i, ivar=beam.ivar)
                
                sens = beam.sens

            w = w/1.e4
                 
            unit_corr = 1./sens
            clip = (sens > 0.1*sens.max()) 
            clip &= (np.isfinite(flm)) & (er > 0)
            if clip.sum() == 0:
                continue
            
            fl *= unit_corr/1.e-19
            er *= unit_corr/1.e-19
            flm *= unit_corr/1.e-19
            
            f_alpha = 1./(self.Ngrism[grism.upper()])*0.8 #**0.5
            
            # Plot
            axc.errorbar(w[clip], fl[clip], er[clip], color=GRISM_COLORS[grism], alpha=f_alpha, marker='.', linestyle='None', zorder=1)
            axc.plot(w[clip], flm[clip], color='r', alpha=f_alpha, linewidth=2, zorder=10) 
              
            # Plot limits         
            ymax = np.maximum(ymax,
                        np.percentile((flm+np.median(er[clip]))[clip], 98))
            
            ymin = np.minimum(ymin, np.percentile((flm-er*0.)[clip], 2))
            
            wmax = np.maximum(wmax, w[clip].max())
            wmin = np.minimum(wmin, w[clip].min())
        
        # Cleanup
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

        gs.tight_layout(fig, pad=0.1, w_pad=0.1)
        return fig
        
    def process_zfit(self, zgrid, chi2, prior=None):
        """Parse redshift fit"""
        
        zbest = zgrid[np.argmin(chi2)]
        
        ###############
        
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

