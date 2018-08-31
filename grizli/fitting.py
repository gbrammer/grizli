"""
fitting.py

Created by Gabriel Brammer on 2017-05-19.

"""
import os
import glob
import inspect

from collections import OrderedDict

import numpy as np

import astropy.io.fits as pyfits
import astropy.units as u
from astropy.cosmology import Planck15
import astropy.constants as const

from . import utils
#from .model import BeamCutout
from .utils import GRISM_COLORS

# Minimum redshift where IGM is applied
IGM_MINZ = 3.4 # blue edge of G800L

# Default parameters for drizzled line map
PLINE = {'kernel': 'point', 'pixfrac': 0.2, 'pixscale': 0.1, 'size': 8, 'wcs': None}

# IGM from eazy-py
try:
    import eazy.igm
    IGM = eazy.igm.Inoue14()
except:
    IGM = None

def run_all_parallel(id, get_output_data=False, **kwargs):
    import numpy as np
    from grizli.fitting import run_all
    from grizli import multifit
    import time
    import traceback
    
    t0 = time.time()

    print('Run {0}'.format(id))
    args = np.load('fit_args.npy')[0]
    args['verbose'] = False
    for k in kwargs:
        args[k] = kwargs[k]
        
    fp = open('{0}_{1:05d}.log_par'.format(args['group_name'], id),'w')
    fp.write('{0}_{1:05d}: {2}\n'.format(args['group_name'], id, time.ctime()))
    fp.close()
    
    try:
        #args['zr'] = [0.7, 1.0]
        #mb = multifit.MultiBeam('j100025+021651_{0:05d}.beams.fits'.format(id))
        out = run_all(id, **args)
        if get_output_data:
            return out
        status=1
    except:
        status=-1
        trace = traceback.format_exc(limit=2)#, file=fp)
        if args['verbose']:
            print(trace)
            
    t1 = time.time()
    
    return id, status, t1-t0
    
def run_all(id, t0=None, t1=None, fwhm=1200, zr=[0.65, 1.6], dz=[0.004, 0.0002], fitter='nnls', group_name='grism', fit_stacks=True, only_stacks=False, prior=None, fcontam=0.2, pline=PLINE, mask_sn_limit=3, fit_only_beams=False, fit_beams=True, root='*', fit_trace_shift=False, phot=None, phot_obj=None, verbose=True, scale_photometry=False, show_beams=True, scale_on_stacked_1d=True, overlap_threshold=5, MW_EBV=0., sys_err=0.03, get_dict=False, bad_pa_threshold=1.6, units1d='flam', redshift_only=False, line_size=1.6, use_psf=False, get_line_width=False, sed_args={'bin':1, 'xlim':[0.3, 9]}, **kwargs):
    """Run the full procedure
    
    1) Load MultiBeam and stack files 
    2) ... tbd
    
    fwhm=1200; zr=[0.65, 1.6]; dz=[0.004, 0.0002]; group_name='grism'; fit_stacks=True; prior=None; fcontam=0.2; mask_sn_limit=3; fit_beams=True; root=''
    
    """
    import glob
    import grizli.multifit
    from grizli.stack import StackFitter
    from grizli.multifit import MultiBeam
    
    if get_dict:
        frame = inspect.currentframe()
        args = inspect.getargvalues(frame).locals
        for k in ['id', 'get_dict', 'frame', 'glob', 'grizli', 'StackFitter', 'MultiBeam']:
            if k in args:
                args.pop(k)
        
        return args 
        
    mb_files = glob.glob('{0}_{1:05d}.beams.fits'.format(root, id))
    st_files = glob.glob('{0}_{1:05d}.stack.fits'.format(root, id))
    
    if not only_stacks:
        mb = MultiBeam(mb_files, fcontam=fcontam, group_name=group_name, MW_EBV=MW_EBV, sys_err=sys_err, verbose=verbose, psf=use_psf)
        # Check for PAs with unflagged contamination or otherwise discrepant
        # fit
        out = mb.check_for_bad_PAs(chi2_threshold=bad_pa_threshold,
                                                   poly_order=1, reinit=True, 
                                                  fit_background=True)
    
        fit_log, keep_dict, has_bad = out
    
        if has_bad:
            if verbose:
                print('\nHas bad PA!  Final list: {0}\n{1}'.format(keep_dict,
                                                                   fit_log))
                
            hdu, fig = mb.drizzle_grisms_and_PAs(fcontam=0.5, flambda=False, kernel='point', size=32)
            fig.savefig('{0}_{1:05d}.fix.stack.png'.format(group_name, id))
            good_PAs = []
            for k in keep_dict:
                good_PAs.extend(keep_dict[k])
        else:
            good_PAs = None # All good
    else:
        good_PAs = None # All good
        redshift_only=True # can't drizzle line maps from stacks
        
    if fit_only_beams:
        st = None
    else:
        st = StackFitter(st_files, fit_stacks=fit_stacks, group_name=group_name, fcontam=fcontam, overlap_threshold=overlap_threshold, MW_EBV=MW_EBV, verbose=verbose, sys_err=sys_err, PAs=good_PAs, chi2_threshold=bad_pa_threshold)
        st.initialize_masked_arrays()
    
    if only_stacks:
        mb = st
        
    if not only_stacks:
        if fit_trace_shift:
            b = mb.beams[0]
            b.compute_model()
            sn_lim = fit_trace_shift*1
            if (np.max((b.model/b.grism['ERR'])[b.fit_mask.reshape(b.sh)]) > sn_lim) | (sn_lim > 100):
                shift, _ = mb.fit_trace_shift(tol=1.e-3, verbose=verbose, 
                                           split_groups=True)
            
        mb.initialize_masked_arrays()
    
    ## Get photometry from phot_obj
    if (phot is None) & (phot_obj is not None):
        phot_i, ii, dd = phot_obj.get_phot_dict(mb.ra, mb.dec)
        if dd < 0.5*u.arcsec:
            phot = phot_i
            
    if phot is not None:
        if st is not None:
            st.set_photometry(**phot)
        
        mb.set_photometry(**phot)
            
    if t0 is None:
        t0 = grizli.utils.load_templates(line_complexes=True, fsps_templates=True, fwhm=fwhm)
    
    if t1 is None:
        t1 = grizli.utils.load_templates(line_complexes=False, fsps_templates=True, fwhm=fwhm)
        
    # Fit on stacked spectra or individual beams
    if fit_only_beams:
        fit_obj = mb
    else:
        fit_obj = st
    
    ### Do scaling now with direct spectrum function
    if (scale_photometry > 0) & (phot is not None):
        try:
            scl = mb.scale_to_photometry(z=0, method='lm', templates=t0, order=scale_photometry*1-1)
        except:
            scl = [10.]
            
        if hasattr(scl,'status'):
            if scl.status > 0:
                print('scale_to_photometry: [{0}]'.format(', '.join(['{0:.2f}'.format(x_i) for x_i in scl.x])))
                mb.pscale = scl.x
                if st is not None:
                    st.pscale = scl.x
    
    # First pass    
    fit = fit_obj.xfit_redshift(templates=t0, zr=zr, dz=dz, prior=prior, fitter=fitter, verbose=verbose) 
    fit_hdu = pyfits.table_to_hdu(fit)
    fit_hdu.header['EXTNAME'] = 'ZFIT_STACK'
    
    if hasattr(fit_obj, 'pscale'):
        fit_hdu.header['PSCALEN'] = (len(fit_obj.pscale)-1, 'PSCALE order')
        for i, p in enumerate(fit_obj.pscale):
            fit_hdu.header['PSCALE{0}'.format(i)] = (p, 'PSCALE parameter {0}'.format(i))
    
    # Add photometry information
    if (fit_obj.Nphot > 0) & hasattr(fit_obj, 'photom_filters'):
        h = fit_hdu.header
        h['NPHOT'] = fit_obj.Nphot, 'Number of photometry filters'
        for i in range(len(fit_obj.photom_filters)):
            h['PHOTN{0:03d}'.format(i)] = fit_obj.photom_filters[i].name.split()[0], 'Filter {0} name'.format(i)
            h['PHOTL{0:03d}'.format(i)] = fit_obj.photom_pivot[i], 'Filter {0} pivot wavelength'.format(i)
            h['PHOTF{0:03d}'.format(i)] = fit_obj.photom_flam[i], 'Filter {0} flux flam'.format(i)
            h['PHOTE{0:03d}'.format(i)] = fit_obj.photom_eflam[i], 'Filter {0} err flam'.format(i)
            
    # # Second pass if rescaling spectrum to photometry
    # if scale_photometry:
    #     scl = mb.scale_to_photometry(z=fit.meta['z_map'][0], method='lm', templates=t0, order=scale_photometry*1-1)
    #     if scl.status > 0:
    #         mb.pscale = scl.x
    #         if st is not None:
    #             st.pscale = scl.x
    #         
    #         fit = fit_obj.xfit_redshift(templates=t0, zr=zr, dz=dz, prior=prior, fitter=fitter, verbose=verbose) 
    #         fit_hdu = pyfits.table_to_hdu(fit)
    #         fit_hdu.header['EXTNAME'] = 'ZFIT_STACK'
            
    # Zoom-in fit with individual beams
    if fit_beams:
        #z0 = fit.meta['Z50'][0]
        z0 = fit.meta['z_map'][0]
        
        #width = np.maximum(3*fit.meta['ZWIDTH1'][0], 3*0.001*(1+z0))
        width = 20*0.001*(1+z0)
        
        mb_zr = z0 + width*np.array([-1,1])
        mb_fit = mb.xfit_redshift(templates=t0, zr=mb_zr, dz=[0.001, 0.0002], prior=prior, fitter=fitter, verbose=verbose) 
        mb_fit_hdu = pyfits.table_to_hdu(mb_fit)
        mb_fit_hdu.header['EXTNAME'] = 'ZFIT_BEAM'
    else:
        mb_fit = fit
           
    #### Get best-fit template 
    tfit = mb.template_at_z(z=mb_fit.meta['z_map'][0], templates=t1, fit_background=True, fitter=fitter)
    
    # Redrizzle? ... testing
    if False:
        hdu, fig = mb.drizzle_grisms_and_PAs(fcontam=fcontam,
                                         flambda=False,
                                         size=48, scale=1., 
                                         kernel='point', pixfrac=0.1,
                                         zfit=tfit)
                
    # Fit covariance
    cov_hdu = pyfits.ImageHDU(data=tfit['covar'], name='COVAR')
    Next = mb_fit.meta['N']
    cov_hdu.header['N'] = Next
    
    # Line EWs & fluxes
    coeffs_clip = tfit['coeffs'][mb.N:]
    covar_clip = tfit['covar'][mb.N:,mb.N:]
    lineEW = utils.compute_equivalent_widths(t1, coeffs_clip, covar_clip, max_R=5000, Ndraw=1000, z=tfit['z'])
    
    for ik, key in enumerate(lineEW):
        for j in range(3):
            if not np.isfinite(lineEW[key][j]):
                lineEW[key][j] = -1.e30
                
        cov_hdu.header['FLUX_{0:03d}'.format(ik)] = tfit['cfit'][key][0], '{0} line flux; erg / (s cm2)'.format(key.strip('line '))
        cov_hdu.header['ERR_{0:03d}'.format(ik)] = tfit['cfit'][key][1], '{0} line uncertainty; erg / (s cm2)'.format(key.strip('line '))
        
        cov_hdu.header['EW16_{0:03d}'.format(ik)] = lineEW[key][0], 'Rest-frame {0} EW, 16th percentile; Angstrom'.format(key.strip('line '))
        cov_hdu.header['EW50_{0:03d}'.format(ik)] = lineEW[key][1], 'Rest-frame {0} EW, 50th percentile; Angstrom'.format(key.strip('line '))
        cov_hdu.header['EW84_{0:03d}'.format(ik)] = lineEW[key][2], 'Rest-frame {0} EW, 84th percentile; Angstrom'.format(key.strip('line '))
        cov_hdu.header['EWHW_{0:03d}'.format(ik)] = (lineEW[key][2]-lineEW[key][0])/2, 'Rest-frame {0} EW, 1-sigma half-width; Angstrom'.format(key.strip('line '))
    
    # Velocity width
    if get_line_width:
        if phot is not None:
            mb.unset_photometry()
        
        vel_width_res = mb.fit_line_width(z0=tfit['z'], bl=1.2, nl=1.2)
        if verbose:
            print('Velocity width: BL/NL = {0:.0f}/{1:.0f}, z={2:.4f}'.format(vel_width_res[0]*1000, vel_width_res[1]*1000, vel_width_res[2]))
        
        fit_hdu.header['VEL_BL'] = vel_width_res[0]*1000, 'Broad line FWHM'
        fit_hdu.header['VEL_NL'] = vel_width_res[1]*1000, 'Narrow line FWHM'
        fit_hdu.header['VEL_Z'] = vel_width_res[2], 'Line width, best redshift'
        fit_hdu.header['VEL_NFEV'] = vel_width_res[3], 'Line width, NFEV'
        fit_hdu.header['VEL_FLAG'] = vel_width_res[4], 'Line width, NFEV'
    
        if phot is not None:
            mb.set_photometry(**phot)
        
    # Best-fit template itself
    tfit_sp = grizli.utils.GTable()
    for ik, key in enumerate(tfit['cfit']):
        for save in [tfit_sp.meta]:
            save['CVAL{0:03d}'.format(ik)] = tfit['cfit'][key][0], 'Coefficient for {0}'.format(key)
            save['CERR{0:03d}'.format(ik)] = tfit['cfit'][key][1], 'Uncertainty for {0}'.format(key)
            save['CNAME{0:03d}'.format(ik)] = key, 'Template name'
                
    tfit_sp['wave'] = tfit['cont1d'].wave
    tfit_sp['continuum'] = tfit['cont1d'].flux
    tfit_sp['full'] = tfit['line1d'].flux
    
    tfit_sp['wave'].unit = tfit['cont1d'].waveunits
    tfit_sp['continuum'].unit = tfit['cont1d'].fluxunits
    tfit_sp['full'].unit = tfit['line1d'].fluxunits
    
    tfit_hdu = pyfits.table_to_hdu(tfit_sp)
    tfit_hdu.header['EXTNAME'] = 'TEMPL'
     
    # Make the plot
    fig = mb.xmake_fit_plot(mb_fit, tfit, show_beams=show_beams, scale_on_stacked_1d=scale_on_stacked_1d)
    
    # Add prior
    if prior is not None:
        fig.axes[0].plot(prior[0], np.log10(prior[1]), color='#1f77b4', alpha=0.5)
        
    # Add stack fit to the existing plot
    fig.axes[0].plot(fit['zgrid'], np.log10(fit['pdf']), color='0.5', alpha=0.5)
    fig.axes[0].set_xlim(fit['zgrid'].min(), fit['zgrid'].max())
    
    if phot is not None:
        fig.axes[1].errorbar(mb.photom_pivot/1.e4, mb.photom_flam/1.e-19, mb.photom_eflam/1.e-19, marker='s', alpha=0.5, color='k', linestyle='None')
        #fig.axes[1].plot(tfit['line1d'].wave/1.e4, tfit['line1d'].flux/1.e-19, color='k', alpha=0.2, zorder=100)
         
    # Save the figure
    fig.savefig('{0}_{1:05d}.full.png'.format(group_name, id))
    
    if redshift_only:
        return mb, st, fit, tfit, None
        
    # Make the line maps
    if pline is None:
         pzfit, pspec2, pline = grizli.multifit.get_redshift_fit_defaults()
    
    line_hdu = mb.drizzle_fit_lines(tfit, pline, force_line=['SIII','SII','Ha', 'OIII', 'Hb', 'OII', 'Lya'], save_fits=False, mask_lines=True, mask_sn_limit=mask_sn_limit, verbose=verbose)
    
    # Add beam exposure times
    exptime = mb.compute_exptime()
    for k in exptime:
        line_hdu[0].header['T_{0}'.format(k)] = (exptime[k], 'Total exposure time [s]')
         
    line_hdu.insert(1, fit_hdu)
    line_hdu.insert(2, cov_hdu)
    if fit_beams:
        line_hdu.insert(2, mb_fit_hdu)
    line_hdu.insert(3, tfit_hdu)
    
    line_hdu.writeto('{0}_{1:05d}.full.fits'.format(group_name, id), clobber=True, output_verify='fix')
    
    # 1D spectrum
    oned_hdul = mb.oned_spectrum_to_hdu(tfit=tfit, bin=1, outputfile='{0}_{1:05d}.1D.fits'.format(group_name, id))#, units=units1d)
    
    ######
    # Show the drizzled lines and direct image cutout, which are
    # extensions `DSCI`, `LINE`, etc.
    s, si = 1, line_size
    
    s = 4.e-19/np.max([beam.beam.total_flux for beam in mb.beams]) 
    s = np.clip(s, 0.25, 4)

    full_line_list = ['Lya', 'OII', 'Hb', 'OIII', 'Ha', 'SII', 'SIII']
    fig = show_drizzled_lines(line_hdu, size_arcsec=si, cmap='plasma_r', scale=s, dscale=s, full_line_list=full_line_list)
    fig.savefig('{0}_{1:05d}.line.png'.format(group_name, id))
    
    if phot is not None:
        out = mb, st, fit, tfit, line_hdu
        if 'pz' in phot:
            full_sed_plot(mb, tfit, zfit=fit, photometry_pz=phot['pz'], **sed_args)
        else:
            full_sed_plot(mb, tfit, zfit=fit, **sed_args)
        
    return mb, st, fit, tfit, line_hdu

###################################
def full_sed_plot(mb, tfit, zfit=None, bin=1, minor=0.1, save='png', sed_resolution=180, photometry_pz=None, zspec=None, spectrum_steps=False, xlim=[0.3, 9], **kwargs):
    """
    Make a separate plot showing photometry and the spectrum
    """
    #import seaborn as sns
    import prospect.utils.smoothing
    
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    import matplotlib.gridspec as gridspec
    
    #mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mpl_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # sns_colors = colors = sns.color_palette("cubehelix", 8)
    ### seaborn cubehelix colors
    sns_colors = colors = [(0.1036, 0.094, 0.206),
                           (0.0825, 0.272, 0.307),
                           (0.1700, 0.436, 0.223),
                           (0.4587, 0.480, 0.199),
                           (0.7576, 0.476, 0.437),
                           (0.8299, 0.563, 0.776),
                           (0.7638, 0.757, 0.949),
                           (0.8106, 0.921, 0.937)]
     
    # Best-fit
    #mb = out[0]
    #zfit = out[2]
    #tfit = out[3]
    t1 = tfit['templates']
    
    best_model = mb.get_flat_model([tfit['line1d'].wave, tfit['line1d'].flux])
    flat_model = mb.get_flat_model([tfit['line1d'].wave, tfit['line1d'].flux*0+1])
    bg = mb.get_flat_background(tfit['coeffs'])
    
    sp = mb.optimal_extract(mb.scif[mb.fit_mask][:-mb.Nphot] - bg, bin=bin)#['G141']
    spm = mb.optimal_extract(best_model, bin=bin)#['G141']
    spf = mb.optimal_extract(flat_model, bin=bin)#['G141']
    
    # Photometry 
    A_phot = mb._interpolate_photometry(z=tfit['z'], templates=t1)
    A_model = A_phot.T.dot(tfit['coeffs'])
    photom_mask = mb.photom_eflam > -98
    
    ##########
    # Figure

    if True:
        if zfit is not None:
            fig = plt.figure(figsize=[11, 9./3])
            gs = gridspec.GridSpec(1,3, width_ratios=[1,1.5,1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
        else:
            fig = plt.figure(figsize=[9, 9./3])
            gs = gridspec.GridSpec(1,2, width_ratios=[1,1.5])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
    else:
        gs = None
        fig = plt.figure(figsize=[9, 9./3])
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
    
    # Photometry SED
    ax1.errorbar(np.log10(mb.photom_pivot[photom_mask]/1.e4), mb.photom_flam[photom_mask]/1.e-19, mb.photom_eflam[photom_mask]/1.e-19, color='k', alpha=0.6, marker='s', linestyle='None', zorder=30)

    sm = prospect.utils.smoothing.smoothspec(tfit['line1d'].wave, tfit['line1d'].flux, resolution=sed_resolution, smoothtype='R') #nsigma=10, inres=10)

    ax1.scatter(np.log10(mb.photom_pivot[photom_mask]/1.e4), A_model/1.e-19, color='w', marker='s', s=80, zorder=10)
    ax1.scatter(np.log10(mb.photom_pivot[photom_mask]/1.e4), A_model/1.e-19, color=sns_colors[4], marker='s', s=20, zorder=11)
    yl1 = ax1.get_ylim()

    ax1.plot(np.log10(tfit['line1d'].wave/1.e4), sm/1.e-19, color=sns_colors[4], linewidth=1, zorder=0)

    if xlim is None:
        xlim = [0.7*mb.photom_pivot.min()/1.e4, mb.photom_pivot.max()/1.e4/0.7]
        
    ax1.set_xlim(np.log10(xlim[0]), np.log10(xlim[1]))
    
    ticks = np.array([0.5, 1, 2, 4, 8])
    ticks = ticks[(ticks >= xlim[0]) & (ticks <= xlim[1])]
    
    ax1.set_xticks(np.log10(ticks))
    ax1.set_xticklabels(ticks)
    
    #ax1.grid()
    ax1.set_xlabel(r'$\lambda$ / $\mu$m')
    ax2.set_xlabel(r'$\lambda$ / $\mu$m')
    
    # Spectrum 
    ymax, ymin = -1e30, 1e30
    for g in sp:
        sn = sp[g]['flux']/sp[g]['err']
        clip = sn > 3
        clip = spf[g]['flux'] > 0.2*spf[g]['flux'].max()
        
        try:
            scale = mb.compute_scale_array(mb.pscale, sp[g]['wave']) 
        except:
            scale = 1
            
        ax2.errorbar(sp[g]['wave'][clip]/1.e4, (sp[g]['flux']/spf[g]['flux']/scale)[clip]/1.e-19, (sp[g]['err']/spf[g]['flux']/scale)[clip]/1.e-19, marker='.', color='k', alpha=0.5, linestyle='None', elinewidth=0.5, zorder=11)
        
        if spectrum_steps:
            ax2.plot(sp[g]['wave']/1.e4, spm[g]['flux']/spf[g]['flux']/1.e-19, color=sns_colors[4], linewidth=2, alpha=0.8, zorder=10, linestyle='steps-mid')
        else:
            ax2.plot(sp[g]['wave']/1.e4, spm[g]['flux']/spf[g]['flux']/1.e-19, color=sns_colors[4], linewidth=2, alpha=0.8, zorder=10, marker='.')
            
        ymax = np.maximum(ymax, (spm[g]['flux']/spf[g]['flux']/1.e-19)[clip].max())
        ymin = np.minimum(ymin, (spm[g]['flux']/spf[g]['flux']/1.e-19)[clip].min())
        
        ax1.errorbar(np.log10(sp[g]['wave'][clip]/1.e4), (sp[g]['flux']/spf[g]['flux']/scale)[clip]/1.e-19, (sp[g]['err']/spf[g]['flux']/scale)[clip]/1.e-19, marker='.', color='k', alpha=0.2, linestyle='None', elinewidth=0.5, zorder=-100)
        
        
    xl, yl = ax2.get_xlim(), ax2.get_ylim()
    yl = (ymin-0.3*ymax, 1.3*ymax)
    
    ax2.scatter((mb.photom_pivot[photom_mask]/1.e4), A_model/1.e-19, color='w', marker='s', s=80, zorder=11)
    ax2.scatter((mb.photom_pivot[photom_mask]/1.e4), A_model/1.e-19, color=sns_colors[4], marker='s', s=20, zorder=12)
    
    ax2.errorbar(mb.photom_pivot[photom_mask]/1.e4, mb.photom_flam[photom_mask]/1.e-19, mb.photom_eflam[photom_mask]/1.e-19, color='k', alpha=0.6, marker='s', linestyle='None', zorder=20)
    ax2.set_xlim(xl); ax2.set_ylim(yl)

    ax2.set_yticklabels([])
    #ax2.set_xticks(np.arange(1.1, 1.8, 0.1))
    #ax2.set_xticklabels([1.1, '', 1.3, '', 1.5, '', 1.7])
    ax2.xaxis.set_minor_locator(MultipleLocator(minor))
    ax2.xaxis.set_major_locator(MultipleLocator(minor*2))
    
    # Show spectrum range on SED panel
    xb, yb = np.array([0, 1, 1, 0, 0]), np.array([0, 0, 1, 1, 0])
    ax1.plot(np.log10(xl[0]+xb*(xl[1]-xl[0])), yl[0]+yb*(yl[1]-yl[0]), linestyle=':', color='k', alpha=0.4)
    ymax = np.maximum(yl1[1], yl[1]+0.02*(yl[1]-yl[0]))
    ax1.set_ylim(-0.1*ymax, ymax)
    
    tick_diff = np.diff(ax1.get_yticks())[0]
    ax2.yaxis.set_major_locator(MultipleLocator(tick_diff))
    #ax2.set_yticklabels([])
    
    ##########
    # P(z)
    if zfit is not None:

        if photometry_pz is not None:
            ax3.plot(photometry_pz[0], np.log10(photometry_pz[1]), color=mpl_colors[0])

        ax3.plot(zfit['zgrid'], np.log10(zfit['pdf']), color=sns_colors[0])
        ax3.fill_between(zfit['zgrid'], np.log10(zfit['pdf']), np.log10(zfit['pdf'])*0-100, color=sns_colors[0], alpha=0.3)
        ax3.set_xlim(zfit['zgrid'].min(), zfit['zgrid'].max())

        ax3.set_ylim(-3, 2.9) #np.log10(zfit['pdf']).max())
        ax3.set_ylabel(r'log $p(z)$')
        ax3.set_xlabel(r'$z$')

    ax1.set_ylabel(r'$f_\lambda$ / $10^{-19}$')

    axt = ax2
    axt.text(0.95, 0.95, r'$z_\mathrm{grism}$='+'{0:.3f}'.format(tfit['z']), ha='right', va='top', transform=axt.transAxes, color=sns_colors[0], size=10)#, backgroundcolor='w')
    if zspec is not None:
        axt.text(0.95, 0.89, r'$z_\mathrm{spec}$='+'{0:.3f}'.format(zspec), ha='right', va='top', transform=axt.transAxes, color='r', size=10)
        
        if zfit is not None:
            ax3.scatter(zspec, 2.7, color='r', marker='v', zorder=100)
        
    axt.text(0.05, 0.95, '{0}: {1:>6d}'.format(mb.group_name, mb.id), ha='left', va='top', transform=axt.transAxes, color='k', size=10)#, backgroundcolor='w')
    #axt.text(0.05, 0.89, '{0:>6d}'.format(mb.id), ha='left', va='top', transform=axt.transAxes, color='k', size=10)#, backgroundcolor='w')

    if gs is None:
        gs.tight_layout(pad=0.1)
    else:
        if zfit is not None:
            fig.tight_layout(pad=0.1)
        else:
            fig.tight_layout(pad=0.5)
            
    if save:
        fig.savefig('{0}_{1:05d}.sed.{2}'.format(mb.group_name, mb.id, save))
        
    return fig
    
def make_summary_catalog(target='pg0117+213', sextractor='pg0117+213-f140w.cat', verbose=True, filter_bandpasses=[]):
    
    import glob
    import os
    from collections import OrderedDict
    import matplotlib.pyplot as plt
    
    import astropy.units as u
    import astropy.io.fits as pyfits
    import numpy as np
    import grizli
    from grizli import utils
    
    keys = OrderedDict()
    keys['PRIMARY'] = ['ID','RA','DEC','NINPUT','REDSHIFT','T_G102', 'T_G141', 'T_G800L', 'NUMLINES','HASLINES']
    
    keys['ZFIT_STACK'] = ['CHI2POLY','CHI2SPL','SPLF01','SPLE01','SPLF02','SPLE02','SPLF03','SPLE03','SPLF04','SPLE04', 'DOF','CHIMIN','CHIMAX','BIC_POLY','BIC_SPL','BIC_TEMP','Z02', 'Z16', 'Z50', 'Z84', 'Z97', 'ZWIDTH1', 'ZWIDTH2', 'Z_MAP', 'Z_RISK', 'MIN_RISK', 'VEL_BL','VEL_NL','VEL_Z','VEL_NFEV','VEL_FLAG']
    
    keys['ZFIT_BEAM'] = ['CHI2POLY','CHI2SPL','SPLF01','SPLE01','SPLF02','SPLE02','SPLF03','SPLE03','SPLF04','SPLE04', 'DOF','CHIMIN','CHIMAX','BIC_POLY','BIC_SPL','BIC_TEMP','Z02', 'Z16', 'Z50', 'Z84', 'Z97', 'ZWIDTH1', 'ZWIDTH2', 'Z_MAP', 'Z_RISK', 'MIN_RISK', 'VEL_BL','VEL_NL','VEL_Z','VEL_NFEV','VEL_FLAG']
    
    keys['COVAR'] = ' '.join(['FLUX_{0:03d} ERR_{0:03d} EW50_{0:03d} EWHW_{0:03d}'.format(i) for i in range(24)]).split()
    
    lines = []
    pdf_max = []
    files=glob.glob('{0}*full.fits'.format(target))
    files.sort()
    
    template_mags = []
    sps_params = []
    
    for file in files:
        print(utils.NO_NEWLINE+file)
        line = []
        full = pyfits.open(file)
        
        if 'DSCI' not in full:
            continue
            
        tab = utils.GTable.read(full['ZFIT_STACK'])
        pdf_max.append(tab['pdf'].max())
        
        for ext in keys:
            if ext not in full:
                for k in keys[ext]:
                    line.append(np.nan)
                
                continue
                
            h = full[ext].header
            for k in keys[ext]:
                if k in h:
                    line.append(h[k])
                else:
                    line.append(np.nan)
        
        # SPS
        try:
            sps = compute_sps_params(full)
        except:
            sps = {'Lv':-1*u.solLum, 'MLv':-1*u.solMass/u.solLum, 'MLv_rms':-1*u.solMass/u.solLum, 'SFRv':-1*u.solMass/u.year, 'SFRv_rms':-1*u.solMass/u.year, 'templ':-1}
        sps_params.append(sps)
        
        lines.append(line)
        
        # Integrate best-fit template through filter bandpasses
        if filter_bandpasses:
            tfit = utils.GTable.gread(full['TEMPL'])
            sp = utils.SpectrumTemplate(wave=tfit['wave'], flux=tfit['full'])
            mags = [sp.integrate_filter(bp, abmag=True) 
                        for bp in filter_bandpasses]
            
            template_mags.append(mags)
            
    columns = []
    for ext in keys:
        if ext == 'ZFIT_BEAM':
            columns.extend(['beam_{0}'.format(k) for k in keys[ext]])
        else:
            columns.extend(keys[ext])
    
    info = utils.GTable(rows=lines, names=columns)
    info['PDF_MAX'] = pdf_max
    
    root_col = utils.GTable.Column(name='root', data=[target]*len(info))
    info.add_column(root_col, index=0)
    
    for k in ['Lv','MLv','MLv_rms','SFRv','SFRv_rms']:
        datak = [sps[k].value for sps in sps_params]
        info[k] = datak
        info[k].unit = sps[k].unit
    
    info['sSFR'] = info['SFRv']/info['MLv']
    info['stellar_mass'] = info['Lv']*info['MLv']
    
    info['Lv'].format = '.1e'
    info['MLv'].format = '.2f'
    info['MLv_rms'].format = '.2f'
    info['SFRv'].format = '.1f'
    info['SFRv_rms'].format = '.1f'
    info['sSFR'].format = '.1e'
    info['stellar_mass'].format = '.1e'
    
    
    if filter_bandpasses:
        arr = np.array(template_mags)
        for i, bp in enumerate(filter_bandpasses):
            info['mag_{0}'.format(bp.name)] = arr[:,i]
            info['mag_{0}'.format(bp.name)].format = '.3f'
            
    for c in info.colnames:
        info.rename_column(c, c.lower())
    
    # Emission line names
    files=glob.glob('{0}*full.fits'.format(target))
    im = pyfits.open(files[0])
    h = im['COVAR'].header
    for i in range(24):
        key = 'FLUX_{0:03d}'.format(i)
        if key not in h:
            continue
        
        line = h.comments[key].split()[0]
        
        for root in ['flux','err','ew50','ewhw']:
            col = '{0}_{1}'.format(root, line)
            info.rename_column('{0}_{1:03d}'.format(root, i), col)
            if root.startswith('ew'):
                info[col].format = '.1f'
            else:
                info[col].format = '.1f'
        
        info['sn_{0}'.format(line)] = info['flux_'+line]/info['err_'+line]
        info['sn_{0}'.format(line)][info['err_'+line] == 0] = -99
        #info['sn_{0}'.format(line)].format = '.1f'
           
    info['chinu'] = info['chimin']/info['dof']
    info['chinu'].format = '.2f'
    
    info['bic_diff'] = info['bic_poly'] - info['bic_temp']
    info['bic_diff'].format = '.1f'
    
    info['log_risk'] = np.log10(info['min_risk'])
    info['log_risk'].format = '.2f'
        
    info['log_pdf_max'] = np.log10(info['pdf_max'])
    info['log_pdf_max'].format = '.2f'
    
    info['zq'] = info['log_risk'] - info['log_pdf_max']
    info['zq'].format = '.2f'
    
    info['beam_chinu'] = info['beam_chimin']/info['beam_dof']
    info['beam_chinu'].format = '.2f'

    info['beam_bic_diff'] = info['beam_bic_poly'] - info['beam_bic_temp']
    info['beam_bic_diff'].format = '.1f'

    info['beam_log_risk'] = np.log10(info['beam_min_risk'])
    info['beam_log_risk'].format = '.2f'
    
    # ID with link to CDS
    idx = ['<a href="http://vizier.u-strasbg.fr/viz-bin/VizieR?-c={0:.6f}+{1:.6f}&-c.rs=2">{2}</a>'.format(info['ra'][i], info['dec'][i], info['id'][i]) for i in range(len(info))]
    info['idx'] = idx
    
    ### PNG columns    
    for ext in ['stack','full','line']:
        png = ['{0}_{1:05d}.{2}.png'.format(target, id, ext) for id in info['id']]
        info['png_{0}'.format(ext)] = ['<a href={0}><img src={0} height=200></a>'.format(p) for p in png]
    
    ### Column formats
    for col in info.colnames:
        if col.strip('beam_').startswith('z'):
            info[col].format = '.4f'
        
        if col in ['ra','dec']:
            info[col].format = '.6f'
            
    ### Sextractor catalog
    if sextractor is None:
        info.write('{0}.info.fits'.format(target), overwrite=True)
        return info
        
    #sextractor = glob.glob('{0}-f*cat'.format(target))[0]
    try:
        hcat = grizli.utils.GTable.gread(sextractor) #, format='ascii.sextractor')
    except:
        hcat = grizli.utils.GTable.gread(sextractor, sextractor=True)
    
    for c in hcat.colnames:
        hcat.rename_column(c, c.lower())
    
    idx, dr = hcat.match_to_catalog_sky(info, self_radec=('x_world', 'y_world'), other_radec=None)
    for c in hcat.colnames:
        info.add_column(hcat[c][idx])
        
    info.write('{0}.info.fits'.format(target), overwrite=True)
    return info

def compute_sps_params(full='j021820-051015_01276.full.fits', cosmology=Planck15):
    import numpy as np

    from astropy.io import fits as pyfits
    from astropy.table import Table
    import astropy.units as u
    
    from grizli import utils
        
    import pysynphot as S
    
    if isinstance(full, str):
        im = pyfits.open(full)
    else:
        im = full
        
    h = im['TEMPL'].header
    templ = Table(im['TEMPL'].data)
    z = im['ZFIT_STACK'].header['Z_MAP']
    
    # Get coefffs
    coeffs, keys, ix = [], [], []
    count=0
    for k in h:
        if k.startswith('CNAME'):
            if h[k].startswith('fsps'):
                ix.append(count)
                keys.append(h[k])
                coeffs.append(h[k.replace('CNAME','CVAL')])

            count += 1
    
    cov = im['COVAR'].data[np.array(ix),:][:,np.array(ix)]
    covd = cov.diagonal()
    
    # Normalize to V band, fsps_QSF_12_v3
    normV = np.array([3.75473763e-15, 2.73797790e-15, 1.89469588e-15,
    1.32683449e-15, 9.16760812e-16, 2.43922395e-16, 4.76835746e-15,
    3.55616962e-15, 2.43745972e-15, 1.61394625e-15, 1.05358710e-15,
    5.23733297e-16])
           
    coeffsV = np.array(coeffs)*normV
    rmsV = np.sqrt(covd)*normV
    rms_norm = rmsV/coeffsV.sum()
    coeffs_norm = coeffsV/coeffsV.sum()
    
    param_file = os.path.join(os.path.dirname(__file__), 'data/templates/fsps/fsps_QSF_12_v3.param.fits')
    tab_temp = Table.read(param_file)
    temp_MLv = tab_temp['mass']/tab_temp['Lv']
    temp_SFRv = tab_temp['sfr']
    
    mass_norm = (coeffs_norm*tab_temp['mass']).sum()*u.solMass
    Lv_norm = (coeffs_norm*tab_temp['Lv']).sum()*u.solLum
    MLv = mass_norm / Lv_norm
        
    SFR_norm = (coeffs_norm*tab_temp['sfr']).sum()*u.solMass/u.yr
    SFRv = SFR_norm / Lv_norm
    
    mass_var = ((rms_norm*tab_temp['mass'])**2).sum()
    Lv_var = ((rms_norm*tab_temp['Lv'])**2).sum()
    SFR_var = ((rms_norm*tab_temp['sfr'])**2).sum()
    
    MLv_var = MLv**2 * (mass_var/mass_norm.value**2 + Lv_var/Lv_norm.value**2)
    MLv_rms = np.sqrt(MLv_var)

    SFRv_var = SFRv**2 * (SFR_var/SFR_norm.value**2 + Lv_var/Lv_norm.value**2)
    SFRv_rms = np.sqrt(SFRv_var)
    
    vband = S.ObsBandpass('v')
    vbandz = S.ArrayBandpass(vband.wave*(1+z), vband.throughput)
    
    best_templ = utils.SpectrumTemplate(templ['wave'], templ['full'])
    fnu = best_templ.integrate_filter(vbandz)*(u.erg/u.s/u.cm**2/u.Hz)
    
    dL = cosmology.luminosity_distance(z).to(u.cm)
    Lnu = fnu*4*np.pi*dL**2
    pivotV = vbandz.pivot()*u.Angstrom
    nuV = (const.c/pivotV).to(u.Hz) 
    Lv = (nuV*Lnu).to(u.L_sun)
            
    mass = MLv*Lv
    SFR = SFRv*Lv
    
    sps = {'Lv':Lv, 'MLv':MLv, 'MLv_rms':MLv_rms, 'SFRv':SFRv, 'SFRv_rms':SFRv_rms, 'templ':best_templ}
    
    return sps
  
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

def refit_beams(root='j012017+213343', append='x', id=708, keep_dict={'G141':[201, 291]}, poly_order=3, make_products=True, run_fit=True, **kwargs):
    """
    Regenerate a MultiBeam object selecting only certiain PAs
    
    Parameters
    ----------
    root : str
        Root of the "beams.fits" file to load.
        
    append : str
        String to append to the rootname of the updated products.
        
    id : int
        Object ID.  The input filename is built like 
        
           >>> beams_file = '{0}_{1:05d}.beams.fits'.format(root, id)
           
    keep_dict : bool
        Dictionary of the PAs/grisms to keep.  (See the
        `~grizli.multifit.MultiBeam.PA` attribute.)
    
    poly_order : int
        Order of the polynomial to fit.
    
    make_products : bool
        Make stacked spectra and diagnostic figures.
        
    run_fit : bool
        Run the redshift fit on the new products
    
    kwargs : dict
        Optional keywords passed to `~grizli.fitting.run_all_parallel`.
        
    Returns
    -------

    mb : `~grizli.multifit.MultiBeam`
        New beam object.

    """
    import numpy as np
    
    try:
        from grizli import utils, fitting
    except:
        from . import utils, fitting
    
    mb = MultiBeam('{0}_{1:05d}.beams.fits'.format(root, id), group_name=root)
    
    keep_beams = []
    for g in keep_dict:
        if g not in mb.PA:
            continue
            
        for pa in keep_dict[g]:
            if float(pa) in mb.PA[g]:
                keep_beams.extend([mb.beams[i] for i in mb.PA[g][float(pa)]])
    
    mb = MultiBeam(keep_beams, group_name=root+append)
    mb.write_master_fits()
    
    if not make_products:
        return mb
            
    wave = np.linspace(2000,2.5e4,100)
    poly_templates = utils.polynomial_templates(wave, order=poly_order)
    
    pfit = mb.template_at_z(z=0, templates=poly_templates, fit_background=True, fitter='lstsq', get_uncertainties=2)
    
    try:
        fig1 = mb.oned_figure(figsize=[5,3], tfit=pfit)
        fig1.savefig('{0}_{1:05d}.1D.png'.format(root+append, id))
    except:
        pass
    
    hdu, fig = mb.drizzle_grisms_and_PAs(fcontam=0.5, flambda=False, kernel='point', size=32, zfit=pfit)
    fig.savefig('{0}_{1:05d}.stack.png'.format(root+append, id))
    
    if run_fit:
        fitting.run_all_parallel(id, group_name=root+append, root=root+'x', verbose=True, **kwargs)
    
    return mb
    
class GroupFitter(object):
    """Combine stack.StackFitter and MultiBeam fitting into a single object
    
    Will have to match the attributes between the different objects, which 
    is already close.
    """
    def _test(self):
        print(self.Ngrism)
    
    def _get_slices(self, masked=False):
        """Precompute array slices for how the individual components map into the single combined arrays.
        
        Parameters
        ----------
        masked : bool
            Return indices of masked arrays rather than simple slices of the 
            full beams.
        
        Returns
        -------
        slices : list
            List of slices.
        """
        x = 0
        slices = []
        # use masked index arrays rather than slices
        if masked:
            for i in range(self.N):
                beam = self.beams[i]
                if beam.fit_mask.sum() == 0:
                    slices.append(None)
                    continue
                    
                idx = np.arange(beam.fit_mask.sum())+x
                slices.append(idx) #[slice(x+0, x+beam.size)][beam.fit_mask])
                x = idx[-1]+1
        else:    
            for i in range(self.N):
                slices.append(slice(x+0, x+self.beams[i].size))
                x += self.beams[i].size
        
        return slices    
    
    def _update_beam_mask(self):
        """
        Compute versions of the masked arrays
        """
        for ib, b in enumerate(self.beams):
            b.fit_mask &= self.fit_mask[self.slices[ib]]
            
        self.mslices = self._get_slices(masked=True)
        self.Nmask = self.fit_mask.sum()       
        if hasattr(self, 'Nphot'):
            self.Nspec = self.Nmask - self.Nphot
        else:
            self.Nspec = self.Nmask
        
    def _init_background(self, masked=True):
        """Initialize the (flat) background model components
        
        Parameters
        ----------
        None :
        
        Returns
        -------
        A_bg : `~np.ndarray`
            
        """
        if masked:
            A_bg = np.zeros((self.N, self.Nmask))
            for i in range(self.N):
                A_bg[i, self.mslices[i]] = 1.
        else:
            A_bg = np.zeros((self.N, self.Ntot))
            for i in range(self.N):
                A_bg[i, self.slices[i]] = 1. 
                           
        return A_bg
    
    def get_SDSS_photometry(self, bands='ugriz', templ=None, radius=2, SDSS_CATALOG='V/147/sdss12', get_panstarrs=False):
        #from astroquery.sdss import SDSS
        #from astropy import coordinates as coords
        import astropy.units as u
        
        from astroquery.vizier import Vizier
        import astropy.coordinates as coord
        
        import pysynphot as S
        
        from eazy.templates import Template
        from eazy.filters import FilterFile
        from eazy.photoz import TemplateGrid
        from eazy.filters import FilterDefinition
        
        if get_panstarrs:
            SDSS_CATALOG = 'II/349'
            bands = 'grizy'
        
        # pos = coords.SkyCoord(self.ra*u.deg, self.dec*u.deg, frame='icrs')
        # fields = ['ra','dec','modelMag_r', 'modelMagErr_r']
        # for b in bands:
        #     fields.extend(['modelFlux_'+b, 'modelFluxIvar_'+b])
        #     
        # xid = SDSS.query_region(pos, photoobj_fields=fields, spectro=False, radius=radius*u.arcsec)
        
        from astroquery.vizier import Vizier
        import astropy.units as u
        import astropy.coordinates as coord
        
        coo = coord.SkyCoord(ra=self.ra, dec=self.dec, unit=(u.deg, u.deg),
                             frame='icrs')
                                                          
        v = Vizier(catalog=SDSS_CATALOG, columns=['+_r','*'])
        try:
            tab = v.query_region(coo, radius="{0}s".format(radius),
                              catalog=SDSS_CATALOG)[0]
            
            ix = np.argmin(tab['rmag'])
            tab = tab[ix]
        except:
            return None
                               
        filters = [FilterDefinition(bp=S.ObsBandpass('sdss,{0}'.format(b))) for b in bands]
        pivot = {}
        for ib, b in enumerate(bands):
            pivot[b] = filters[ib].pivot()
            
        #to_flam = 10**(-0.4*(22.5+48.6))*3.e18 # / pivot(Ang)**2
        #flam = np.array([xid['modelFlux_{0}'.format(b)][0]*to_flam/pivot[b]**2 for b in bands])
        #eflam = np.array([np.sqrt(1/xid['modelFluxIvar_{0}'.format(b)][0])*to_flam/pivot[b]**2 for b in bands])
        
        to_flam = 10**(-0.4*(48.6))*3.e18 # / pivot(Ang)**2
        flam = np.array([10**(-0.4*(tab[b+'mag']))*to_flam/pivot[b]**2 for ib, b in enumerate(bands)])
        eflam = np.array([tab['e_{0}mag'.format(b)]*np.log(10)/2.5*flam[ib] for ib, b in enumerate(bands)])
        
        phot = {'flam':flam, 'eflam':eflam, 'filters':filters, 'tempfilt':None}
        
        if templ is None:
            return phot
        
        # Make fast SDSS template grid
        templates = [Template(arrays=[templ[t].wave, templ[t].flux], name=t) for t in templ]
        zgrid = utils.log_zgrid(zr=[0.01, 3.4], dz=0.005)
        
        tempfilt = TemplateGrid(zgrid, templates, filters=filters, add_igm=True, galactic_ebv=0, Eb=0, n_proc=0)
        
        #filters = [all_filters.filters[f-1] for f in [156,157,158,159,160]]
        phot = {'flam':flam, 'eflam':eflam, 'filters':filters, 'tempfilt':tempfilt}
        return phot
        
        ### Vizier
               

        
    def set_photometry(self, flam=[], eflam=[], filters=[], lc=None, force=False, tempfilt=None, min_err=0.02, TEF=None, pz=None):
        """
        Add photometry
        """
        if (self.Nphot > 0) & (not force):
            print('Photometry already set (Nphot={0})'.format(self.Nphot))
            return True
        
        self.Nphot = (eflam > 0).sum() #len(flam)
        self.Nphotbands = len(eflam)
        
        if self.Nphot == 0:
            return True
        
        if (len(flam) != len(eflam)) | (len(flam) != len(filters)):
            print('flam/eflam/filters dimensions don\'t match')
            return False
        
        self.photom_flam = flam
        self.photom_eflam = np.sqrt(eflam**2+(min_err*flam)**2)
        self.photom_flam[eflam < 0] = -99
        self.photom_eflam[eflam < 0] = -99
        
        self.photom_filters = filters
        
        self.sivarf = np.hstack((self.sivarf, 1/self.photom_eflam))
        self.weightf = np.hstack((self.weightf, self.photom_eflam*0+1))
        self.fit_mask = np.hstack((self.fit_mask, eflam > 0))
        #self.flat_flam = np.hstack((self.flat_flam, self.photom_eflam*0.))
        
        # Mask for just spectra
        self.fit_mask_spec = self.fit_mask & True
        self.fit_mask_spec[-self.Nphotbands:] = False
        
        self.Nmask = self.fit_mask.sum()       
        self.Nspec = self.Nmask - self.Nphot
        
        self.scif = np.hstack((self.scif, flam))
        
        self.DoF = int((self.weightf*self.fit_mask).sum())
        
        self.is_spec = np.isfinite(self.scif)
        self.is_spec[-len(flam):] = False
        
        self.photom_pivot = np.array([filter.pivot() for filter in filters])
        self.wavef = np.hstack((self.wavef, self.photom_pivot))
        
        # eazypy tempfilt for faster interpolation
        self.tempfilt = tempfilt
        
        self.TEF = TEF
        
    def unset_photometry(self):
        if self.Nphot == 0:
            return True
        
        Nbands = self.Nphotbands
        self.sivarf = self.sivarf[:-Nbands]
        self.weightf = self.weightf[:-Nbands]

        #self.flat_flam = self.flat_flam[:-Nbands]        
        self.fit_mask = self.fit_mask[:-Nbands]
        self.fit_mask_spec = self.fit_mask & True
        
        self.scif = self.scif[:-Nbands]
        self.wavef = self.wavef[:-Nbands]
                        
        self.DoF = int((self.weightf*self.fit_mask).sum())
        
        self.is_spec = 1
        self.Nphot = 0
        self.Nphotbands = 0
        self.Nmask = self.fit_mask.sum()
        self.Nspec = self.Nmask - self.Nphot

        self.tempfilt = None
        
    def _interpolate_photometry(self, z=0., templates=[]):
        """
        Interpolate templates through photometric filters
        
        xx: TBD better handling of emission line templates and use eazpy tempfilt
        object for huge speedup
        
        """
        NTEMP = len(templates)
        A_phot = np.zeros((NTEMP+self.N, len(self.photom_flam))) #self.Nphot))
        mask = self.photom_eflam > 0
        
        if (self.tempfilt is not None):
            if (self.tempfilt.NTEMP == NTEMP):
                #A_spl = self.tempfilt(z)
                A_phot[self.N:,:] = self.tempfilt(z)
                A_phot *= 3.e18/self.photom_pivot**2*(1+z)
                A_phot[~np.isfinite(A_phot)] = 0
                return A_phot[:,mask]

        for it, key in enumerate(templates):
            #print(key)
            tz = templates[key].zscale(z, scalar=1)
            for ifilt, filt in enumerate(self.photom_filters):
                A_phot[self.N+it, ifilt] = tz.integrate_filter(filt)*3.e18/self.photom_pivot[ifilt]**2#*(1+z)
            
            # pl = plt.plot(tz.wave, tz.flux)
            # plt.scatter(self.photom_pivot, A_phot[self.N+it,:], color=pl[0].get_color())
            
        return A_phot[:,mask]
        
    def xfit_at_z(self, z=0, templates=[], fitter='nnls', fit_background=True, get_uncertainties=False, get_design_matrix=False, pscale=None, COEFF_SCALE=1.e-19, get_components=False, huber_delta=4, get_residuals=False, include_photometry=True):
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
        
        get_design_matrix : bool
            Return design matrix and data, rather than nominal outputs.
        
        huber_delta : float
            Use the Huber loss function (`~scipy.special.huber`) rather than
            direct chi-squared.  If `huber_delta` < 0, then fall back to chi2.
            
        Returns
        -------
        chi2 : float
            Chi-squared of the fit
        
        coeffs, coeffs_err : `~np.ndarray`
            Template coefficients and uncertainties.
        
        covariance : `~np.ndarray`
            Full covariance
            
        """
        import scipy.optimize
        #import scipy.sparse
        from scipy.special import huber
        
        NTEMP = len(templates)
        if (self.Nphot > 0) & include_photometry:
            A = np.zeros((self.N+NTEMP, self.Nmask))
        else:
            A = np.zeros((self.N+NTEMP, self.Nspec))
            
        if fit_background:
            A[:self.N,:self.Nspec] = self.A_bgm
        
        lower_bound = np.zeros(self.N+NTEMP)
        lower_bound[:self.N] = -0.05
        upper_bound = np.ones(self.N+NTEMP)*np.inf
        upper_bound[:self.N] = 0.05
        
        # A = scipy.sparse.csr_matrix((self.N+NTEMP, self.Ntot))
        # bg_sp = scipy.sparse.csc_matrix(self.A_bg)
                
        for i, t in enumerate(templates):
            if t.startswith('line'):
                lower_bound[self.N+i] = -np.inf
                
            ti = templates[t]
            if z > IGM_MINZ:
                if IGM is None:
                    igmz = 1.
                else:
                    lylim = ti.wave < 1250
                    igmz = np.ones_like(ti.wave)
                    igmz[lylim] = IGM.full_IGM(z, ti.wave[lylim]*(1+z))         
            else:
                igmz = 1.
            
            # Don't redshift spline templates
            if ti.name.startswith('bspl'):
                s = [ti.wave, ti.flux*igmz]            
            else:
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

                sl = self.mslices[j]
                if t in beam.thumbs:
                    #print('Use thumbnail!', t)
                    A[self.N+i, sl] = beam.compute_model(thumb=beam.thumbs[t], spectrum_1d=s, in_place=False, is_cgs=True)[beam.fit_mask]*COEFF_SCALE
                else:
                    A[self.N+i, sl] = beam.compute_model(spectrum_1d=s, in_place=False, is_cgs=True)[beam.fit_mask]*COEFF_SCALE
                    
                # if j == 0:
                #     m = beam.compute_model(spectrum_1d=s, in_place=False, is_cgs=True)
                #     ds9.frame(i)
                #     ds9.view(m.reshape(beam.sh))
                        
        if fit_background:
            if fitter in ['nnls', 'lstsq']:
                pedestal = 0.04
            else:
                pedestal = 0.
        else:
            pedestal = 0
        
        #oktemp = (A*self.fit_mask).sum(axis=1) != 0
        oktemp = A.sum(axis=1) != 0
        
        # Photometry
        if (self.Nphot > 0):
            if include_photometry:
                A_phot = self._interpolate_photometry(z=z,
                                                      templates=templates)
                A[:,-self.Nphot:] = A_phot*COEFF_SCALE #np.hstack((A, A_phot))
                full_fit_mask = self.fit_mask
            else:
                full_fit_mask = self.fit_mask_spec
        else:
            full_fit_mask = self.fit_mask
            
        # Weight design matrix and data by 1/sigma
        Ax = A[oktemp,:]*self.sivarf[full_fit_mask]        
        #AxT = Ax[:,full_fit_mask].T
        
        # Scale photometry
        if hasattr(self, 'pscale'):
            if (self.pscale is not None):
                scale = self.compute_scale_array(self.pscale, self.wavef[full_fit_mask]) 
                if self.Nphot > 0:
                    scale[-self.Nphot:] = 1.
                
                Ax *= scale
                if fit_background:
                    for i in range(self.N):
                        Ax[i,:] /= scale
        
        # Need transpose
        AxT = Ax.T
        
        # Masked data array, including background pedestal
        data = ((self.scif+pedestal*self.is_spec)*self.sivarf)[full_fit_mask]
        
        if get_design_matrix:
            return AxT, data
            
        # Run the minimization
        if fitter == 'nnls':
            coeffs_i, rnorm = scipy.optimize.nnls(AxT, data)            
        elif fitter == 'lstsq':
            coeffs_i, residuals, rank, s = np.linalg.lstsq(AxT, data, rcond=None)
        else:
            # Bounded Least Squares
            lsq_out = scipy.optimize.lsq_linear(AxT, data, bounds=(lower_bound[oktemp], upper_bound[oktemp]), method='bvls', tol=1.e-8)
            coeffs_i = lsq_out.x
            
        # Compute background array         
        if fit_background:
            background = np.dot(coeffs_i[:self.N], A[:self.N,:]) - pedestal
            if self.Nphot > 0:
                background[-self.Nphot:] = 0.
            coeffs_i[:self.N] -= pedestal
        else:
            background = self.scif[full_fit_mask]*0.
            
        # Full model
        if fit_background:
            model = np.dot(coeffs_i[self.N:], Ax[self.N:,:]/self.sivarf[full_fit_mask])
        else:
            model = np.dot(coeffs_i, Ax/self.sivarf[full_fit_mask])
            
        # Residuals and Chi-squared
        resid = self.scif[full_fit_mask] - model - background
        
        if get_components:
            return model, background
            
        #chi2 = np.sum(resid[full_fit_mask]**2*self.sivarf[full_fit_mask]**2)
        norm_resid = resid*(self.sivarf*np.sqrt(self.weightf))[full_fit_mask]
        
        # Use Huber loss function rather than direct chi2
        if huber_delta > 0:
            chi2 = huber(huber_delta, norm_resid)*2.
        else:
            chi2 = norm_resid**2
        
        if get_residuals:
            chi2 = norm_resid
        else:
            chi2 = np.sum(chi2)
            
        if self.Nphot > 0:
            self.photom_model = model[-self.Nphot:]*1
            
        # Uncertainties from covariance matrix
        if get_uncertainties:
            try:
                # Covariance is inverse of AT.A
                covar_i = np.matrix(np.dot(AxT.T, AxT)).I.A
                covar = utils.fill_masked_covar(covar_i, oktemp)
                covard = np.sqrt(covar.diagonal())
                
                # Compute covariances after masking templates with coeffs = 0
                if get_uncertainties == 2:
                    nonzero = coeffs_i != 0
                    if nonzero.sum() > 0:
                        AxTm = AxT[:,nonzero]
                        #mcoeffs_i, rnorm = scipy.optimize.nnls(AxTm, data)            
                        #mcoeffs_i[:self.N] -= pedestal

                        mcovar_i = np.matrix(np.dot(AxTm.T, AxTm)).I.A
                        mcovar = utils.fill_masked_covar(mcovar_i, nonzero)
                        mcovar = utils.fill_masked_covar(mcovar, oktemp)
                        mcovard = np.sqrt(mcovar.diagonal())
                        
                        covar = mcovar
                        covard = mcovard
            except:
                print('Except: covar!')
                covar = np.zeros((self.N+NTEMP, self.N+NTEMP))
                covard = np.zeros(self.N+NTEMP)#-1.
                mcovard = covard
        else:
            covar = np.zeros((self.N+NTEMP, self.N+NTEMP))
            covard = np.zeros(self.N+NTEMP)#-1.
            
        coeffs = np.zeros(self.N+NTEMP)
        coeffs[oktemp] = coeffs_i #[self.N:]] = coeffs[self.N:]

        coeffs_err = covard #np.zeros(NTEMP)
        #full_coeffs_err[oktemp[self.N:]] = covard[self.N:]
        del(A); del(Ax); del(AxT)
        
        #if fit_background:
        coeffs[self.N:] *= COEFF_SCALE
        coeffs_err[self.N:] *= COEFF_SCALE
        #covar[self.N:,self.N:] *= COEFF_SCALE**2
        covar[self.N:,:] *= COEFF_SCALE
        covar[:,self.N:] *= COEFF_SCALE
           
        return chi2, coeffs, coeffs_err, covar
    
    def xfit_redshift(self, prior=None, fwhm=1200,
                     make_figure=True, zr=[0.65, 1.6], dz=[0.005, 0.0004],
                     verbose=True, fit_background=True, fitter='nnls', 
                     delta_chi2_threshold=0.004, poly_order=3, zoom=True, 
                     line_complexes=True, templates={}, figsize=[8,5],
                     fsps_templates=False, get_uncertainties=True,
                     Rspline=30):
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
        wpoly = np.linspace(1000,5.e4,1000)
        # tpoly = utils.polynomial_templates(wpoly, line=True)
        # out = self.xfit_at_z(z=0., templates=tpoly, fitter='nnls',
        #                     fit_background=True, get_uncertainties=False)
        tpoly = utils.polynomial_templates(wpoly, order=poly_order,
                                           line=False)
        out = self.xfit_at_z(z=0., templates=tpoly, fitter='lstsq',
                            fit_background=True, get_uncertainties=False,
                            include_photometry=False)
        
        chi2_poly, coeffs_poly, err_poly, cov = out
        
        #### Spline SED fit
        wspline = np.arange(4200, 2.5e4)
        #Rspline = 30
        df_spl = len(utils.log_zgrid(zr=[wspline[0], wspline[-1]], dz=1./Rspline))
        tspline = utils.bspline_templates(wspline, df=df_spl+2, log=True, clip=0.0001)
        
        out = self.xfit_at_z(z=0., templates=tspline, fitter='lstsq',
                            fit_background=True, get_uncertainties=True,
                            include_photometry=False)
        
        chi2_spline, coeffs_spline, err_spline, cov = out
                
        #poly1d, xxx = utils.dot_templates(coeffs_poly[self.N:], tpoly, z=0)

        # tpoly = utils.polynomial_templates(wpoly, order=3)
        # out = self.xfit_at_z(z=0., templates=tpoly, fitter='lstsq',
        #                     fit_background=True)          
        # chi2_poly, coeffs_poly, c, cov = out

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
                            get_uncertainties=get_uncertainties)
                            
        chi2, coeffs, coeffs_err, covar = out
        
        chi2 = np.zeros(NZ)
        coeffs = np.zeros((NZ, coeffs.shape[0]))
        covar = np.zeros((NZ, covar.shape[0], covar.shape[1]))
        
        chi2min = 1e30
        iz = 0
        for i in range(NZ):
            out = self.xfit_at_z(z=zgrid[i], templates=templates,
                                fitter=fitter, fit_background=fit_background,
                                get_uncertainties=get_uncertainties)
            
            chi2[i], coeffs[i,:], coeffs_err, covar[i,:,:] = out
            if chi2[i] < chi2min:
                iz = i
                chi2min = chi2[i]

            if verbose:                    
                print(utils.NO_NEWLINE + '  {0:.4f} {1:9.1f} ({2:.4f}) {3:d}/{4:d}'.format(zgrid[i], chi2[i], zgrid[iz], i+1, NZ))
        
        if verbose:
            print('First iteration: z_best={0:.4f}\n'.format(zgrid[iz]))
            
        ## Find peaks
        import peakutils
        
        # Make "negative" chi2 for peak-finding
        #chi2_test = chi2_poly
        chi2_test = chi2_spline
        
        if chi2_test > (chi2.min()+100):
            chi2_rev = (chi2.min() + 100 - chi2)/self.DoF
        elif chi2_test < (chi2.min() + 9):
            chi2_rev = (chi2.min() + 16 - chi2)/self.DoF
        else:
            chi2_rev = (chi2_test - chi2)/self.DoF
            
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
                                    get_uncertainties=get_uncertainties)

                chi2_zoom[i], coeffs_zoom[i,:], e, covar_zoom[i,:,:] = out
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
        fit.meta['N'] = (self.N, 'Number of spectrum extensions')
        fit.meta['polyord'] = (poly_order, 'Order polynomial fit')
        fit.meta['chi2poly'] = (chi2_poly, 'Chi^2 of polynomial fit')
        
        kspl = (coeffs_spline != 0).sum()
        fit.meta['chi2spl'] = (chi2_spline, 'Chi^2 of spline fit')
        fit.meta['kspl'] = (kspl, 'Parameters, k, of spline fit')
        
        # Evaluate spline at wavelengths for stars
        xspline = np.array([8100, 9000, 1.27e4, 1.4e4])
        flux_spline = utils.eval_bspline_templates(xspline, tspline, coeffs_spline[self.N:])
        fluxerr_spline = utils.eval_bspline_templates(xspline, tspline, err_spline[self.N:])
        
        for i in range(len(xspline)):
            fit.meta['splf{0:02d}'.format(i+1)] = flux_spline[i], 'Spline flux at {0:.2f} um'.format(xspline[i]/1.e4)
            fit.meta['sple{0:02d}'.format(i+1)] = fluxerr_spline[i], 'Spline flux err at {0:.2f} um'.format(xspline[i]/1.e4)
            
        izbest = np.argmin(chi2)
        clip = coeffs[izbest,:] != 0
        ktempl = clip.sum()
                
        fit.meta['NTEMP'] = (len(templates), 'Number of fitting templates')
        
        fit.meta['DoF'] = (self.DoF, 'Degrees of freedom (number of pixels)')
        
        fit.meta['ktempl'] = (ktempl, 'Parameters, k, of template fit')
        fit.meta['chimin'] = (chi2.min(), 'Minimum chi2 of template fit')
        fit.meta['chimax'] = (chi2.max(), 'Maximum chi2 of template fit')
        fit.meta['fitter'] = (fitter, 'Minimization algorithm')
        
        # Bayesian information criteria, normalized to template min_chi2
        # BIC = log(number of data points)*(number of params) + min(chi2) + C
        # https://en.wikipedia.org/wiki/Bayesian_information_criterion
        
        scale_chinu = self.DoF/chi2.min()
        scale_chinu = 1 # Don't rescale
        
        fit.meta['bic_poly'] = np.log(self.DoF)*(poly_order+1+self.N) + (chi2_poly-chi2.min())*scale_chinu, 'BIC of polynomial fit'

        fit.meta['bic_spl'] = np.log(self.DoF)*kspl + (chi2_spline-chi2.min())*scale_chinu, 'BIC of spline fit'
        
        fit.meta['bic_temp'] = np.log(self.DoF)*ktempl, 'BIC of template fit'
        
        
        for i, tname in enumerate(templates):
            fit.meta['T{0:03d}NAME'.format(i+1)] = (templates[tname].name, 'Template name')
            if tname.startswith('line '):
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
            interp_prior = np.interp(fit['zgrid'], prior[0], prior[1])
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
        zfine = utils.log_zgrid(zr=[fit['zgrid'].min(), fit['zgrid'].max()], dz=0.0001)
        ok = np.isfinite(spl(zfine))
        norm = np.trapz(np.exp(spl(zfine[ok])), zfine[ok])
        
        # Compute CDF and probability intervals
        dz = np.gradient(zfine[ok])
        cdf = np.cumsum(np.exp(spl(zfine[ok]))*dz/norm)
        pz_percentiles = np.interp(np.array([2.5, 16, 50, 84, 97.5])/100., cdf, zfine[ok])

        # Random draws, testing
        #rnd = np.interp(np.random.rand(1000), cdf, fit['zgrid']+dz/2.)
        
        dz = np.gradient(fit['zgrid'])
        
        gamma = 0.15
        zsq = np.dot(fit['zgrid'][:,None], np.ones_like(fit['zgrid'])[None,:])
        L = _loss((zsq-fit['zgrid'])/(1+fit['zgrid']), gamma=gamma)
        
        risk = np.dot(pdf*L, dz)
        zi = np.argmin(risk)
        
        #print('xxx', zi, len(risk))
        
        if (zi < len(risk)-1) & (zi > 0):
            c = np.polyfit(fit['zgrid'][zi-1:zi+2], risk[zi-1:zi+2], 2)
            z_risk = -c[1]/(2*c[0])
        else:
            z_risk = fit['zgrid'][zi]
            
        min_risk = np.trapz(pdf*_loss((z_risk-fit['zgrid'])/(1+fit['zgrid']), gamma=gamma), fit['zgrid'])
        
        # MAP, maximum p(z)
        zi = np.argmax(pdf)
        if (zi < len(pdf)-1) & (zi > 0):
            c = np.polyfit(fit['zgrid'][zi-1:zi+2], pdf[zi-1:zi+2], 2)
            z_map = -c[1]/(2*c[0])
        else:
            z_map = fit['zgrid'][zi]
            
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
        
        fit.meta['z_map'] = z_map, 'Redshift at MAX(PDF)'
        
        fit.meta['z_risk'] = z_risk, 'Redshift at minimum risk'
        fit.meta['min_risk'] = min_risk, 'Minimum risk'
        fit.meta['gam_loss'] = gamma, 'Gamma factor of the risk/loss function'
        return fit
                        
    def template_at_z(self, z=0, templates=None, fit_background=True, fitter='nnls', fwhm=1400, get_uncertainties=2, get_residuals=False, include_photometry=True, draws=0):
        """TBD
        """
        if templates is None:
            templates = utils.load_templates(line_complexes=False, fsps_templates=True, fwhm=fwhm)
        
        out = self.xfit_at_z(z=z, templates=templates, fitter=fitter, 
                             fit_background=fit_background,
                             get_uncertainties=get_uncertainties,
                             get_residuals=get_residuals,
                             include_photometry=include_photometry)

        chi2, coeffs, coeffs_err, covar = out
        cont1d, line1d = utils.dot_templates(coeffs[self.N:], templates, z=z,
                                             apply_igm=(z > IGM_MINZ))
        
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
        tfit['coeffs'] = coeffs
        tfit['chi2'] = chi2
        tfit['covar'] = covar
        tfit['z'] = z
        tfit['templates'] = templates
        
        if draws > 0:
            xte, yte, lte = utils.array_templates(templates, max_R=5000, z=z)
            err = np.sqrt(covar.diagonal())
            nonzero = err > 0
            cov_norm = ((covar/err).T/err)[nonzero,:][:,nonzero]
            draw_coeff = np.zeros((draws, len(err)))
            draw_coeff[:,nonzero] = np.random.multivariate_normal((coeffs/err)[nonzero], cov_norm, draws)*err[nonzero]
            draw_spec = draw_coeff[:,self.N:].dot(yte)
            err_spec = np.diff(np.percentile(draw_spec, [16,84], axis=0), axis=0).flatten()/2.
            tfit['line1d_err'] = err_spec
            
        return tfit #cont1d, line1d, cfit, covar
        
        ### Random draws
        # Unique wavelengths
        wfull = np.hstack([templates[key].wave for key in templates])
        w = np.unique(wfull)
        so = np.argsort(w)
        w = w[so]
        
        xclip = (w*(1+z) > 7000) & (w*(1+z) < 1.8e4)
        temp = []
        for key in templates:
            if key.startswith('bspl'):
                temp.append(grizli.utils_c.interp.interp_conserve_c(w[xclip]/(1+z), templates[key].wave, templates[key].flux))
            else:
                temp.append(grizli.utils_c.interp.interp_conserve_c(w[xclip], templates[key].wave, templates[key].flux))
                
        temp = np.vstack(temp)
        #array([) for key in templates])
        
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
    
    def xmake_fit_plot(self, fit, tfit, show_beams=True, bin=1, minor=0.1,
                       scale_on_stacked_1d=True):
        """TBD
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec
        from matplotlib.ticker import MultipleLocator
        
        import grizli.model
        
        # Initialize plot window
        Ng = len(self.grisms)
        gs = matplotlib.gridspec.GridSpec(1,2, 
                        width_ratios=[1,1.5+0.5*(Ng>1)],
                        hspace=0.)
            
        fig = plt.figure(figsize=[8+4*(Ng>1), 3.5])
        
        # p(z)
        axz = fig.add_subplot(gs[-1,0]) #121)
        
        axz.text(0.95, 0.96, self.group_name + '\n'+'ID={0:<5d}  z={1:.4f}'.format(self.id, fit.meta['z_map'][0]), ha='right', va='top', transform=axz.transAxes, fontsize=9)
                 
        axz.plot(fit['zgrid'], np.log10(fit['pdf']), color='k')
        #axz.fill_between(z, (chi2-chi2.min())/scale_nu, 27, color='k', alpha=0.5)
        
        axz.set_xlabel(r'$z$')
        axz.set_ylabel(r'$\log\ p(z)$'+' / '+ r'$\chi^2=\frac{{{0:.0f}}}{{{1:d}}}={2:.2f}$'.format(fit.meta['chimin'][0], fit.meta['DoF'][0], fit.meta['chimin'][0]/fit.meta['DoF'][0]))
        #axz.set_yticks([1,4,9,16,25])
        
        axz.set_xlim(fit['zgrid'].min(), fit['zgrid'].max())
        pzmax = np.log10(fit['pdf'].max())
        axz.set_ylim(pzmax-6, pzmax+0.8)
        axz.grid()
        axz.yaxis.set_major_locator(MultipleLocator(base=1))
        
        #### Spectra
        axc = fig.add_subplot(gs[-1,1]) #224)
        
        self.oned_figure(bin=bin, show_beams=show_beams, minor=minor, tfit=tfit, axc=axc, scale_on_stacked=scale_on_stacked_1d)
        
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
    
    def scale_to_photometry(self, tfit=None, tol=1.e-4, order=0, init=None, fit_background=True, Rspline=50, use_fit=True, **kwargs):
        """Compute scale factor between spectra and photometry
        
        method : 'Powell' or 'BFGS' work well, latter a bit faster but less robust
        
        New implementation of Levenberg-Markwardt minimization
        
        TBD
        """
        from scipy.optimize import minimize, least_squares
        
        if self.Nphot == 0:
            return np.array([10.])
        
        if (tfit is None) & (fit_background):
            wspline = np.arange(4200, 2.5e4)
            #Rspline = 50
            df_spl = len(utils.log_zgrid(zr=[wspline[0], wspline[-1]], dz=1./Rspline))
            tspline = utils.bspline_templates(wspline, df=df_spl+2, log=True, clip=0.0001)
            tfit = self.template_at_z(z=0, templates=tspline, include_photometry=False, fit_background=fit_background, draws=1000)
        
        if use_fit:
            oned = self.oned_spectrum(tfit=tfit)
            wmi = np.min([oned[k]['wave'].min() for k in oned])
            wma = np.max([oned[k]['wave'].max() for k in oned])
            
            clip = (tfit['line1d'].wave > wmi) & (tfit['line1d'].wave < wma) & (tfit['line1d_err'] > 0)
            spl_temp = utils.SpectrumTemplate(wave=tfit['line1d'].wave[clip], flux=tfit['line1d'].flux[clip], err=tfit['line1d_err'][clip])
            args = (self, {'spl':spl_temp})
        else:
            oned = self.oned_spectrum(tfit=tfit)
            args = (self, oned)
            
        if init is None:
            init = np.zeros(order+1)
            init[0] = 10.
            
        scale_fit = least_squares(self._objective_scale_direct, init, jac='2-point', method='lm', ftol=tol, xtol=tol, gtol=tol, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=args, kwargs={})
            
        # pscale = scale_fit.x
        return scale_fit
    
    def _old_scale_to_photometry(self, z=0, templates={}, tol=1.e-4, order=0, init=None, method='lm', fit_background=True):
        """Compute scale factor between spectra and photometry
        
        method : 'Powell' or 'BFGS' work well, latter a bit faster but less robust
        
        New implementation of Levenberg-Markwardt minimization
        
        TBD
        """
        from scipy.optimize import minimize, least_squares
        
        if self.Nphot == 0:
            return np.array([10.])
        
        AxT, data = self.xfit_at_z(z=z, templates=templates, fitter='nnls',
                                   fit_background=fit_background,
                                   get_uncertainties=False,
                                   get_design_matrix=True)
        
        if init is None:
            init = np.zeros(order+1)
            init[0] = 10.
            
        if method == 'lm':
            scale_fit = least_squares(self.objfun_scale, init, jac='2-point', method='lm', ftol=tol, xtol=tol, gtol=tol, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(AxT, data, self, 'resid'), kwargs={})
        else:
            scale_fit = minimize(self.objfun_scale, init, args=(AxT, data, self, 'chi2'), method=method, jac=None, hess=None, hessp=None, tol=tol, callback=None, options=None)
            
        # pscale = scale_fit.x
        return scale_fit
    
    @staticmethod
    def compute_scale_array(pscale, wave):
        """Return the scale array given the input coefficients
        TBD
        """
        N = len(pscale)
        rescale = 10**(np.arange(N)+1)
        return np.polyval((pscale/rescale)[::-1], (wave-1.e4)/1000.)
        
    @staticmethod
    def objfun_scale(pscale, AxT, data, self, retval):
        """
        Objective function for fitting for a scale term between photometry and 
        spectra
        """
        import scipy.optimize
        from scipy import polyval

        scale = self.compute_scale_array(pscale, self.wavef[self.fit_mask])
        scale[-self.Nphot:] = 1.
        Ax = (AxT.T*scale)

        # Remove scaling from background component
        for i in range(self.N):
            Ax[i,:] /= scale

        coeffs, rnorm = scipy.optimize.nnls(Ax.T, data)  
        #coeffs, rnorm, rank, s = np.linalg.lstsq(Ax.T, data)  
            
        full = np.dot(coeffs, Ax)
        resid = data - full# - background
        chi2 = np.sum(resid**2*self.weightf[self.fit_mask])

        print('{0} {1:.1f}'.format(' '.join(['{0:6.2f}'.format(p) for p in pscale]), chi2))
        if retval == 'resid':
            return resid*np.sqrt(self.weightf[self.fit_mask])
            
        if retval == 'coeffs':
            return coeffs, full, resid, chi2, AxT
        else:
            return chi2
        
    @staticmethod
    def _objective_scale_direct(pscale, self, oned):

        from eazy.filters import FilterDefinition
        
        flam = []
        eflam = []
        spec_flux = []
        
        filters = []
        for filt in self.photom_filters:
            clip = filt.throughput > 0.001*filt.throughput.max()
            filters.append(FilterDefinition(name=filt.name,
                                            wave=filt.wave[clip],
                                            throughput=filt.throughput[clip]))
        
        filters = np.array(filters)
                 
        lc = self.photom_pivot
        
        for k in oned:
            #spec, okfilt, lc = spec1d[k]

            # Covered filters
            if isinstance(oned[k], utils.SpectrumTemplate):
                spec1 = utils.SpectrumTemplate(wave=oned[k].wave, flux=3.e18/oned[k].wave**2)
            else:
                spec1 = utils.SpectrumTemplate(wave=oned[k]['wave'], flux=3.e18/oned[k]['wave']**2)
            
            flux1 = np.array([spec1.integrate_filter(filt, use_wave='filter') for filt in filters]) 
            okfilt = flux1 > 0.98
            
            if okfilt.sum() == 0:
                print('scale_to_photometry: no filters overlap '+k)
                continue

            if isinstance(oned[k], utils.SpectrumTemplate):
                scale = 1./self.compute_scale_array(pscale, oned[k].wave)
                spec = utils.SpectrumTemplate(wave=oned[k].wave, flux=oned[k].flux*scale, err=oned[k].err*scale)
            else:
                scale = 1./self.compute_scale_array(pscale, oned[k]['wave'])
                spec = utils.SpectrumTemplate(wave=oned[k]['wave'], flux=oned[k]['flux']*scale/np.maximum(oned[k]['flat'], 1), err=oned[k]['err']*scale/np.maximum(oned[k]['flat'], 1))

            spec_flux.append((np.array([spec.integrate_filter(filt, use_wave='templ') for filt in filters[okfilt]]).T*3.e18/lc[okfilt]**2).T)

            flam.append(self.photom_flam[okfilt])
            eflam.append(self.photom_eflam[okfilt])

        if not flam:
            return [0]

        spec_flux = np.vstack(spec_flux)
        flam = np.hstack(flam)
        eflam = np.hstack(eflam)
        chi2 = (flam-spec_flux[:,0])**2/(eflam**2+spec_flux[:,1]**2)

        #print(pscale, chi2.sum())
        return chi2
                
    def xfit_star(self, tstar=None, spline_correction=True, fitter='nnls', fit_background=True, spline_args={'Rspline':5}, oned_args={}):
        """Fit stellar templates
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec
        from matplotlib.ticker import MultipleLocator
        
        #self = grizli.multifit.MultiBeam('ers-grism_{0:05d}.beams.fits'.format(id), fcontam=0.2, psf=True)
        #self.extend(grizli.multifit.MultiBeam('/Volumes/Pegasus/Grizli/ACS/goodss/Prep/ers-grism-pears_{0:05d}.beams.fits'.format(id), fcontam=0.2))

        if tstar is None:
            tstar = utils.load_templates(fwhm=1200, line_complexes=True, fsps_templates=True, stars=True)

        NTEMP = len(tstar)
        #covar = np.zeros((NTEMP, self.N+1, self.N+1))
        #coeffs = np.zeros((NTEMP, self.N+1))
        chi2 = np.zeros(NTEMP)

        types = np.array(list(tstar.keys()))
        
        split_templates = []
        split_fits = []
        
        for ik, k in enumerate(tstar):
            if spline_correction:
                ts = utils.split_spline_template(tstar[k], **spline_args)
            else:
                ts = {k:tstar[k]}
                
            split_templates.append(ts)
            
            print(k)
            #chi2[ik], coeffs[ik,:], coeffs_err, covar[ik,:,:] = self.xfit_at_z(z=0, templates=ts, fitter='nnls', fit_background=True, get_uncertainties=True)
            sfit = self.template_at_z(z=0, templates=ts, fit_background=fit_background, fitter=fitter, get_uncertainties=2)
            
            split_fits.append(sfit)
        
        chi2 = np.array([sfit['chi2'] for sfit in split_fits])
        ixbest = np.argmin(chi2)
        
        # Initialize plot window
        Ng = len(self.grisms)
        gs = matplotlib.gridspec.GridSpec(1,2, 
                        width_ratios=[1,1.5+0.5*(Ng>1)],
                        hspace=0.)

        fig = plt.figure(figsize=[8+4*(Ng>1), 3.5])

        # p(z)
        axz = fig.add_subplot(gs[-1,0]) #121)

        if ('_g' in k) & ('_t' in k):
            hast = True
            teff = np.array([float(k.split('_')[1][1:]) for k in tstar])
            logg = np.array([float(k.split('_')[2][1:]) for k in tstar])
            for g in np.unique(logg):
                ig = logg == g
                so = np.argsort(teff[ig])
                axz.plot(teff[ig][so], chi2[ig][so]-chi2.min(), label='g{0:.1f}'.format(g))
            
            
            label = '{0} t{1:.0f} g{2:.1f}'.format(k.split('_')[0], teff[ixbest], logg[ixbest])
        else:    
            hast = False
            axz.plot(chi2-chi2.min(), marker='.', color='k')
            label = types[np.argmin(chi2)].strip('stars/').strip('.txt')
            
        #axz.fill_between(z, (chi2-chi2.min())/scale_nu, 27, color='k', alpha=0.5)

        axz.text(0.95, 0.96, self.group_name + '\n'+'ID={0:<5d} {1:s}'.format(self.id, label), ha='right', va='top', transform=axz.transAxes, fontsize=9, bbox=dict(facecolor='w', alpha=0.8))

        if hast:
            axz.set_xlabel(r'Teff')
            axz.legend(fontsize=7, loc='lower right')
        else:
            axz.set_xlabel(r'Sp. Type')
            
        axz.set_ylabel(r'$\chi^2_\nu$'+' ; '+ r'$\chi^2_\mathrm{{min}}=\frac{{{0:.0f}}}{{{1:d}}}={2:.2f}$'.format(chi2.min(), self.DoF, chi2.min()/self.DoF))
        #axz.set_yticks([1,4,9,16,25])

        if len(tstar) < 30:
            tx = [t.strip('stars/').strip('.txt') for t in types]
            axz.set_xticks(np.arange(len(tx)))
            tl = axz.set_xticklabels(tx)
            for ti in tl:
                ti.set_size(8)

        axz.set_ylim(-2, 27)
        axz.set_yticks([1,4,9,16,25])
        axz.grid()
        #axz.yaxis.set_major_locator(MultipleLocator(base=1))

        #### Spectra
        axc = fig.add_subplot(gs[-1,1]) #224)
        self.oned_figure(tfit=split_fits[ixbest], axc=axc, **oned_args)
        
        if spline_correction:
            sfit = split_fits[ixbest]
            cspl = np.array([sfit['cfit'][t] for t in sfit['cfit']])
            spline_func = sfit['templates'].tspline.dot(cspl[self.N:,0])
            
            yl = axc.get_ylim()
            xl = axc.get_xlim()
            
            y0 = np.interp(np.mean(xl), sfit['templates'].wspline/1.e4, spline_func)
            spl,  = axc.plot(sfit['templates'].wspline/1.e4, spline_func/y0*yl[1]*0.8, color='k', linestyle='--', alpha=0.5, label='Spline correction')
            axc.legend([spl], ['Spline correction'], loc='upper right', fontsize=8)
            
            
        # ymin = 1.e30
        # ymax = -1.e30
        # wmin = 1.e30
        # wmax = -1.e30
        # 
        # # 1D Model
        # ix = np.argmin(chi2)
        # tbest = types[ix]
        # sp = tstar[tbest].wave, tstar[tbest].flux*coeffs[ix,-1]
        # 
        # for i in range(self.N):
        #     beam = self.beams[i]
        #     m_i = beam.compute_model(spectrum_1d=sp, is_cgs=True, in_place=False).reshape(beam.sh)
        # 
        #     #if isinstance(beam, grizli.model.BeamCutout):
        #     if hasattr(beam, 'init_epsf'): # grizli.model.BeamCutout
        #         grism = beam.grism.filter
        #         clean = beam.grism['SCI'] - beam.contam - coeffs[ix,i]
        # 
        #         w, fl, er = beam.beam.optimal_extract(clean, ivar=beam.ivar)            
        #         w, flm, erm = beam.beam.optimal_extract(m_i, ivar=beam.ivar)
        #         sens = beam.beam.sensitivity                
        #     else:
        #         grism = beam.grism
        #         clean = beam.sci - beam.contam - coeffs[ix,i]
        #         w, fl, er = beam.optimal_extract(clean, ivar=beam.ivar)            
        #         w, flm, erm = beam.optimal_extract(m_i, ivar=beam.ivar)
        # 
        #         sens = beam.sens
        # 
        #     w = w/1.e4
        # 
        #     unit_corr = 1./sens
        #     clip = (sens > 0.1*sens.max()) 
        #     clip &= (np.isfinite(flm)) & (er > 0)
        #     if clip.sum() == 0:
        #         continue
        # 
        #     fl *= unit_corr/1.e-19
        #     er *= unit_corr/1.e-19
        #     flm *= unit_corr/1.e-19
        # 
        #     f_alpha = 1./(self.Ngrism[grism.upper()])*0.8 #**0.5
        # 
        #     # Plot
        #     pscale = 1.
        #     if hasattr(self, 'pscale'):
        #         if (self.pscale is not None):
        #             pscale = self.compute_scale_array(self.pscale, w[clip])
        #             
        #     axc.errorbar(w[clip], fl[clip]/pscale, er[clip]/pscale, color=GRISM_COLORS[grism], alpha=f_alpha, marker='.', linestyle='None', zorder=1)
        #     axc.plot(w[clip], flm[clip], color='r', alpha=f_alpha, linewidth=2, zorder=10) 
        # 
        #     # Plot limits         
        #     ymax = np.maximum(ymax,
        #                 np.percentile((flm+np.median(er[clip]))[clip], 98))
        # 
        #     ymin = np.minimum(ymin, np.percentile((flm-er*0.)[clip], 2))
        # 
        #     wmax = np.maximum(wmax, w[clip].max())
        #     wmin = np.minimum(wmin, w[clip].min())
        # 
        # sp_flat = self.optimal_extract(self.flat_flam[self.fit_mask], bin=1)
        # bg_model = self.get_flat_background(coeffs[ix,:], apply_mask=True)
        # sp_data = self.optimal_extract(self.scif_mask-bg_model, bin=1)
        # 
        # for g in sp_data:
        # 
        #     clip = sp_flat[g]['flux'] != 0
        # 
        #     pscale = 1.
        #     if hasattr(self, 'pscale'):
        #         if (self.pscale is not None):
        #             pscale = self.compute_scale_array(self.pscale, sp_data[g]['wave'])
        # 
        #     axc.errorbar(sp_data[g]['wave'][clip]/1.e4, (sp_data[g]['flux']/sp_flat[g]['flux']/1.e-19/pscale)[clip], (sp_data[g]['err']/sp_flat[g]['flux']/1.e-19/pscale)[clip], color=GRISM_COLORS[g], alpha=0.8, marker='.', linestyle='None', zorder=1)
        # 
        # # Cleanup
        # axc.set_xlim(wmin, wmax)
        # #axc.semilogx(subsx=[wmax])
        # #axc.set_xticklabels([])
        # axc.set_xlabel(r'$\lambda$')
        # axc.set_ylabel(r'$f_\lambda \times 10^{-19}$')
        # #axc.xaxis.set_major_locator(MultipleLocator(0.1))
        # 
        # axc.set_ylim(ymin-0.2*ymax, 1.2*ymax)
        # axc.grid()
        # 
        # for ax in [axc]: #[axa, axb, axc]:
        # 
        #     labels = np.arange(np.ceil(wmin*10), np.ceil(wmax*10))/10.
        #     ax.set_xticks(labels)
        #     ax.set_xticklabels(labels)
        #     #ax.set_xticklabels([])
        #     #print(labels, wmin, wmax)

        gs.tight_layout(fig, pad=0.1, w_pad=0.1)
        
        line = '# root id chi2 dof best_template\n'
        line += '{0:16} {1:>5d} {2:10.3f} {3:>10d} {4:20s}'.format(self.group_name, self.id, chi2[ixbest], self.DoF, list(tstar.keys())[ixbest])
        
        
        return fig, line, split_fits[ixbest]
        
        # # Output TBD
        # if False:
        #     sfit = OrderedDict()
        # 
        #     k = list(tstar.keys())[ix]
        #     ts = {k:tstar[k]}
        #     cont1d, line1d = utils.dot_templates(coeffs[ix,self.N:], ts, z=0.)
        # 
        #     sfit['cfit'] = {}
        #     sfit['coeffs'] = coeffs[ix,:]
        #     sfit['covar'] = covar[ix,:,:]
        #     sfit['z'] = 0.
        #     sfit['templates'] = ts
        #     sfit['cont1d'] = cont1d
        #     sfit['line1d'] = line1d
        # 
        #     return fig, sfit
    
    def oned_figure(self, bin=1, show_beams=True, minor=0.1, tfit=None, axc=None, figsize=[6,4], fill=False, units='flam', min_sens=0.1, ylim_percentile=2, scale_on_stacked=False, show_individual_templates=False):
        """
        1D figure
        1D figure
        
        Parameters
        ----------
        bin : type
        
        show_beams : type
        
        minor : type
        
        tfit : type
        
        acx : type
        
        figsize : type
        
        fill : type
        
        units : 'flam', 'nJy', 'mJy', 'eps', 'meps'
            Plot units. 
            
        min_sens : type
        
        ylim_percentile : float
        
        Returns
        -------
        fig : type
        
        
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import UnivariateSpline
        
        if (tfit is None) & (units in ['resid', 'spline']):
            print('`tfit` not specified.  Can\'t plot units=\'{0}\'.'.format(units))
            return False
            
        #### Spectra
        if axc is None:
            fig = plt.figure(figsize=figsize)
            axc = fig.add_subplot(111)
            newfigure=True
        else:
            newfigure=False
            
        ymin = 1.e30
        ymax = -1.e30
        wmin = 1.e30
        wmax = -1.e30
        
        if not show_beams:
            scale_on_stacked=True
            
        if units == 'spline':
            ran = ((tfit['cont1d'].wave >= self.wavef.min()) & 
                   (tfit['cont1d'].wave <= self.wavef.max()))
            
            if ran.sum() == 0:
                print('No overlap with template')
                return False
                
            spl = UnivariateSpline(tfit['cont1d'].wave[ran], 
                                   tfit['cont1d'].flux[ran], ext=1)
            mspl = (tfit['cont1d'].wave, spl(tfit['cont1d'].wave))
        else:
            mspl = None
                                   
        # 1D Model
        if tfit is not None:
            sp = tfit['line1d'].wave, tfit['line1d'].flux
            w = sp[0]
        else:
            sp = None
            w = np.arange(self.wavef.min()-201, self.wavef.max()+201, 100)

        spf = w, w*0+1
        
        for i in range(self.N):
            beam = self.beams[i]
            if tfit is not None:
                m_i = beam.compute_model(spectrum_1d=sp, is_cgs=True, in_place=False).reshape(beam.sh)
            
            if mspl is not None:
                mspl_i = beam.compute_model(spectrum_1d=mspl, is_cgs=True, in_place=False).reshape(beam.sh)
                
            f_i = beam.compute_model(spectrum_1d=spf, is_cgs=True, in_place=False).reshape(beam.sh)
            
            if hasattr(beam, 'init_epsf'): # grizli.model.BeamCutout
                if beam.grism.instrument == 'NIRISS':
                    grism = beam.grism.pupil
                else:
                    grism = beam.grism.filter
                
                clean = beam.grism['SCI'] - beam.contam 
                if tfit is not None:
                    clean -= tfit['cfit']['bg {0:03d}'.format(i)][0]
                    w, flm, erm = beam.beam.optimal_extract(m_i, bin=bin, ivar=beam.ivar)
                
                if mspl is not None:
                    w, flspl, erm = beam.beam.optimal_extract(mspl_i, bin=bin, ivar=beam.ivar)
                        
                w, fl, er = beam.beam.optimal_extract(clean, bin=bin, ivar=beam.ivar)            
                w, sens, ers = beam.beam.optimal_extract(f_i, bin=bin, ivar=beam.ivar)
                #sens = beam.beam.sensitivity                
            else:
                grism = beam.grism
                clean = beam.sci - beam.contam
                if tfit is not None:
                    clean -= - tfit['cfit']['bg {0:03d}'.format(i)][0]
                    w, flm, erm = beam.optimal_extract(m_i, bin=bin, ivar=beam.ivar)
                    
                if mspl is not None:
                    w, flspl, erm = beam.beam.optimal_extract(mspl_i, bin=bin, ivar=beam.ivar)
                    
                w, fl, er = beam.optimal_extract(clean, bin=bin, ivar=beam.ivar)            
                w, sens, ers = beam.optimal_extract(f_i, bin=bin, ivar=beam.ivar)
                
                #sens = beam.sens
            
            sens[~np.isfinite(sens)] = 1
            
            pscale = 1.
            if hasattr(self, 'pscale'):
                if (self.pscale is not None):
                    pscale = self.compute_scale_array(self.pscale, w)
                                                 
            if units == 'nJy':
                unit_corr = 1./sens*w**2/2.99e18/1.e-23/1.e-9#/pscale
                unit_label = r'$f_\nu$ (nJy)'
            elif units == 'uJy':
                unit_corr = 1./sens*w**2/2.99e18/1.e-23/1.e-6#/pscale
                unit_label = r'$f_\nu$ ($\mu$Jy)'
            elif units == 'meps':
                unit_corr = 1000.
                unit_label = 'milli-e/s'
            elif units == 'eps':
                unit_corr = 1.
                unit_label = 'e/s'
            elif units == 'resid':
                unit_corr = 1./flm
                unit_label = 'resid'
            elif units == 'spline':
                unit_corr = 1./flspl
                unit_label = 'spline resid'
            else: # 'flam
                unit_corr = 1./sens/1.e-19#/pscale
                unit_label = r'$f_\lambda \times 10^{-19}$'
            
            w = w/1.e4
            
            clip = (sens > min_sens*sens.max()) 
            clip &= (er > 0)
            if clip.sum() == 0:
                continue
            
            fl *= unit_corr/pscale#/1.e-19
            er *= unit_corr/pscale#/1.e-19
            if tfit is not None:
                flm *= unit_corr#/1.e-19
            
            f_alpha = 1./(self.Ngrism[grism.upper()])*0.8 #**0.5
            
            # Plot
            # pscale = 1.
            # if hasattr(self, 'pscale'):
            #     if (self.pscale is not None):
            #         pscale = self.compute_scale_array(self.pscale, w[clip]*1.e4)
                    
            if show_beams:
                
                if (show_beams == 1) & (f_alpha < 0.09):
                    axc.errorbar(w[clip], fl[clip], er[clip], color='k', alpha=f_alpha, marker='.', linestyle='None', zorder=1)
                else:
                    axc.errorbar(w[clip], fl[clip], er[clip], color=GRISM_COLORS[grism], alpha=f_alpha, marker='.', linestyle='None', zorder=1)
            if tfit is not None:
                axc.plot(w[clip], flm[clip], color='r', alpha=f_alpha, linewidth=2, zorder=10) 

                # Plot limits         
                ymax = np.maximum(ymax,
                            np.percentile((flm+np.median(er[clip]))[clip],
                                           100-ylim_percentile))
            
                ymin = np.minimum(ymin, np.percentile((flm-er*0.)[clip],
                                                       ylim_percentile))
            else:
                
                # Plot limits         
                ymax = np.maximum(ymax,
                            np.percentile((fl+np.median(er[clip]))[clip], 95))
            
                ymin = np.minimum(ymin, np.percentile((fl-er*0.)[clip], 5))
            
            wmax = np.maximum(wmax, w[clip].max())
            wmin = np.minimum(wmin, w[clip].min())
        
        # Cleanup
        axc.set_xlim(wmin, wmax)
        axc.semilogx(subsx=[wmax])
        #axc.set_xticklabels([])
        axc.set_xlabel(r'$\lambda$')
        axc.set_ylabel(unit_label)
        #axc.xaxis.set_major_locator(MultipleLocator(0.1))
                                          
        for ax in [axc]: #[axa, axb, axc]:
            
            labels = np.arange(np.ceil(wmin/minor), np.ceil(wmax/minor))*minor
            ax.set_xticks(labels)
            ax.set_xticklabels(['{0:.1f}'.format(l) for l in labels])    
        
        ### Binned spectrum by grism
        if (tfit is None) | (scale_on_stacked) | (not show_beams):
            ymin = 1.e30
            ymax = -1.e30
        
        if self.Nphot > 0:
            sp_flat = self.optimal_extract(self.flat_flam[self.fit_mask[:-self.Nphotbands]], bin=bin)
        else:
            sp_flat = self.optimal_extract(self.flat_flam[self.fit_mask], bin=bin)
        
        if tfit is not None:
            bg_model = self.get_flat_background(tfit['coeffs'], apply_mask=True)
            m2d = self.get_flat_model(sp, apply_mask=True, is_cgs=True)
            sp_model = self.optimal_extract(m2d, bin=bin)
        else:
            bg_model = 0.
            sp_model = 1.
        
        if mspl is not None:
            m2d = self.get_flat_model(mspl, apply_mask=True, is_cgs=True)
            sp_spline = self.optimal_extract(m2d, bin=bin)
            
        sp_data = self.optimal_extract(self.scif_mask[:self.Nspec]-bg_model, bin=bin)

        for g in sp_data:

            clip = sp_flat[g]['flux'] != 0

            pscale = 1.
            if hasattr(self, 'pscale'):
                if (self.pscale is not None):
                    pscale = self.compute_scale_array(self.pscale, sp_data[g]['wave'])
            
            if units == 'nJy':
                unit_corr = sp_data[g]['wave']**2/sp_flat[g]['flux']
                unit_corr *= 1/2.99e18/1.e-23/1.e-9#/pscale
            elif units == 'uJy':
                unit_corr = sp_data[g]['wave']**2/sp_flat[g]['flux']
                unit_corr *= 1/2.99e18/1.e-23/1.e-6#/pscale
            elif units == 'meps':
                unit_corr = 1000.
            elif units == 'eps':
                unit_corr = 1.
            elif units == 'resid':
                unit_corr = 1./sp_model[g]['flux']
            elif units == 'spline':
                unit_corr = 1./sp_spline[g]['flux']
            else: # 'flam
                unit_corr = 1./sp_flat[g]['flux']/1.e-19#/pscale
            
            flux = (sp_data[g]['flux']*unit_corr/pscale)[clip]
            err = (sp_data[g]['err']*unit_corr/pscale)[clip]
            
            if fill:
                axc.fill_between(sp_data[g]['wave'][clip]/1.e4, flux-err, flux+err, color=GRISM_COLORS[g], alpha=0.8, zorder=1, label=g) 
            else:
                axc.errorbar(sp_data[g]['wave'][clip]/1.e4, flux, err, color=GRISM_COLORS[g], alpha=0.8, marker='.', linestyle='None', zorder=1, label=g) 
                
            if ((tfit is None) & (clip.sum() > 0)) | (scale_on_stacked):
                # Plot limits         
                ymax = np.maximum(ymax, np.percentile((flux+err),
                                                     100-ylim_percentile))
                ymin = np.minimum(ymin, np.percentile((flux-err),
                                                      ylim_percentile))
                       
        if (ymin < 0) & (ymax > 0):
            ymin = -0.1*ymax
        
        axc.set_ylim(ymin-0.2*ymax, 1.2*ymax)            
        axc.grid()
        
        if (ymin-0.2*ymax < 0) & (1.2*ymax > 0):
            axc.plot([wmin, wmax], [0,0], color='k', linestyle=':', alpha=0.8)
        
        ### Individual templates
        if (tfit is not None) & (show_individual_templates > 0) & (units in ['flam', 'nJy','uJy']):
            
            xt, yt, mt = utils.array_templates(tfit['templates'], z=tfit['z'])
            cfit = np.array([tfit['cfit'][t][0] for t in tfit['cfit']])
            
            xt *= (1+tfit['z'])
            
            if units == 'nJy':
                unit_corr = xt**2/2.99e18/1.e-23/1.e-9#/pscale
            elif units == 'uJy':
                unit_corr = xt**2/2.99e18/1.e-23/1.e-6#/pscale
            else: # 'flam
                unit_corr = 1./1.e-19#/pscale
            
            tscl = (yt.T*cfit[self.N:])/(1+tfit['z'])*unit_corr
            t_names = np.array(list(tfit['cfit'].keys()))[self.N:]
            is_spline = np.array([t.startswith('bspl') for t in tfit['cfit']][self.N:])
            
            if is_spline.sum() > 0:
                spline_templ = tscl[:,is_spline].sum(axis=1)
                axc.plot(xt/1.e4, spline_templ, color='k', alpha=0.5)
                for ti in tscl[:,is_spline].T:
                    axc.plot(xt/1.e4, ti, color='k', alpha=0.1)
            
            for ci, ti, tn in zip(cfit[self.N:][~is_spline], tscl[:,~is_spline].T, t_names[~is_spline]):
                if ci == 0:
                    continue
                
                if show_individual_templates > 1:
                    axc.plot(xt/1.e4, ti, alpha=0.6, label=tn.strip('line '))
                else:
                    axc.plot(xt/1.e4, ti, alpha=0.6)
            
            if show_individual_templates > 1:
                axc.legend(fontsize=6)
        
        # Photometry?
        
        if newfigure:
            axc.text(0.95, 0.95, '{0}  {1:>5d}'.format(self.group_name, self.id), ha='right', va='top', transform=axc.transAxes)
            fig.tight_layout(pad=0.2)
            return fig
        
        else:
            return True
            
    ### 
    ### Generic functions for generating flat model and background arrays
    ###
    def optimal_extract(self, data=None, bin=1, wave=None, ivar=None, trace_limits=None):
        """
        TBD: split by grism
        """
        import astropy.units as u
        
        if not hasattr(self, 'optimal_profile_mask'):
            self.initialize_masked_arrays()
        
        if data is None:
            data = self.scif_mask
            
        if data.size not in [self.Nmask, self.Nspec]:
            print('`data` has to be sized like masked arrays (self.fit_mask)')
            return False
        
        if ivar is None:
            #ivar = 1./self.sigma2_mask
            ivar = 1./self.weighted_sigma2_mask
        
        if trace_limits is None:
            prof = self.optimal_profile_mask
            
            num = prof[:self.Nspec]*data[:self.Nspec]*ivar[:self.Nspec]
            den = prof[:self.Nspec]**2*ivar[:self.Nspec]
            
        else:
            prof = np.isfinite(self.optimal_profile_mask)
            trace_mask = ((self.yp_trace_mask > trace_limits[0]) & (self.yp_trace_mask < trace_limits[1]))[:self.Nspec]
            
            num = data[:self.Nspec]*trace_mask
            den = ivar[:self.Nspec]*trace_mask
            
        out = {}
        for grism in self.Ngrism:
            lim = utils.GRISM_LIMITS[grism]
            if wave is None:
                wave_bin = np.arange(lim[0]*1.e4, lim[1]*1.e4, lim[2]*bin)
            else:
                wave_bin = wave
                
            flux_bin = wave_bin*0.
            var_bin = wave_bin*0.
        
            for j in range(len(wave_bin)):
                ix = np.abs(self.wave_mask-wave_bin[j]) < lim[2]*bin/2.
                ix &= self.grism_name_mask == grism
                if ix.sum() > 0:
                    if trace_limits is None:
                        var_bin[j] = 1./den[ix].sum()
                        flux_bin[j] = num[ix].sum()*var_bin[j]
                    else:
                        var_bin[j] = 1./den[ix].sum()
                        flux_bin[j] = num[ix].sum()
                        
            binned_spectrum = utils.GTable()
            binned_spectrum['wave'] = wave_bin*u.Angstrom
            binned_spectrum['flux'] = flux_bin*(u.electron/u.second)
            binned_spectrum['err'] = np.sqrt(var_bin)*(u.electron/u.second)
            
            binned_spectrum.meta['BIN'] = (bin, 'Spectrum binning')
            
            out[grism] = binned_spectrum
            
        return out

            
    def initialize_masked_arrays(self):
        """
        Initialize flat masked arrays for fast likelihood calculation
        """
        try:
            # MultiBeam
            if self.Nphot > 0:
                self.contamf_mask = self.contamf[self.fit_mask[:-self.Nphotbands]]
            else:
                self.contamf_mask = self.contamf[self.fit_mask]
                
            p = []
            for beam in self.beams:
                beam.beam.init_optimal_profile()
                p.append(beam.beam.optimal_profile.flatten()[beam.fit_mask])
            
            self.optimal_profile_mask = np.hstack(p)
            
            # trace offset
            p = []
            for beam in self.beams:
                # w.r.t trace
                yp, xp = np.indices(beam.sh)
                ypt = (yp + 1 - (beam.sh[0]/2.+beam.beam.ytrace))
                beam.ypt = ypt
                p.append(ypt.flatten()[beam.fit_mask])
            
            self.yp_trace_mask = np.hstack(p)
                
            # Inverse sensitivity
            self.sens_mask = np.hstack([np.dot(np.ones(beam.sh[0])[:,None], beam.beam.sensitivity[None,:]).flatten()[beam.fit_mask] for beam in self.beams])
            
            self.grism_name_mask = np.hstack([[beam.grism.pupil]*beam.fit_mask.sum() if beam.grism.instrument == 'NIRISS' else [beam.grism.filter]*beam.fit_mask.sum() for beam in self.beams])
        except:
            # StackFitter
            self.contamf_mask = np.hstack([beam.contamf[beam.fit_mask] 
                                           for beam in self.beams])

            p = []
            for beam in self.beams:
                beam.init_optimal_profile()
                p.append(beam.optimal_profile.flatten()[beam.fit_mask])
            
            self.optimal_profile_mask = np.hstack(p)
            
            # Inverse sensitivity
            self.sens_mask = np.hstack([np.dot(np.ones(beam.sh[0])[:,None], beam.sens[None,:]).flatten()[beam.fit_mask] for beam in self.beams])
            
            self.grism_name_mask = np.hstack([[beam.grism]*beam.fit_mask.sum() for beam in self.beams])
            
        self.wave_mask = np.hstack([np.dot(np.ones(beam.sh[0])[:,None], beam.wave[None,:]).flatten()[beam.fit_mask] for beam in self.beams])
            
        # (scif attribute is already contam subtracted)
        self.scif_mask = self.scif[self.fit_mask] 
        # sigma
        self.sigma_mask = 1/self.sivarf[self.fit_mask]
        # sigma-squared 
        self.sigma2_mask = self.sigma_mask**2
        #self.sigma2_mask = 1/self.ivarf[self.fit_mask] 
        
        # weighted sigma-squared 
        #self.weighted_sigma2_mask = 1/(self.weightf*self.ivarf)[self.fit_mask] 
        self.weighted_sigma2_mask = 1/(self.weightf*self.sivarf**2)[self.fit_mask] 
        
        self.Nmask = self.fit_mask.sum()
        if hasattr(self, 'Nphot'):
            self.Nspec = self.Nmask - self.Nphot
        else:
            self.Nspec = self.Nmask
            
    def get_flat_model(self, spectrum_1d, apply_mask=True, is_cgs=True):
        """
        Generate model array based on the model 1D spectrum in `spectrum_1d`

        Parameters
        ----------

        spectrum_1d : list
            List of 1D arrays [wavelength, flux].

        is_cgs : bool
            `spectrum_1d` flux array has CGS f-lambda flux density units.
            
        Returns
        -------

        model : Array with dimensions `(self.fit_mask.sum(),)`
            Flattened, masked model array.

        """
        mfull = []
        for ib, beam in enumerate(self.beams):
            model_i = beam.compute_model(spectrum_1d=spectrum_1d, 
                                         is_cgs=is_cgs, in_place=False)

            if apply_mask:
                mfull.append(model_i.flatten()[beam.fit_mask])
            else:
                mfull.append(model_i.flatten())
                
        return np.hstack(mfull)

    def get_flat_background(self, bg_params, apply_mask=True):
        """
        Generate background array the same size as the flattened total 
        science array.

        Parameters
        ----------
        bg_params : array with shape (self.N) or (self.N, M)
        
            Background parameters for each beam, where the `M` axis is
            polynomial cofficients in the order expected by
            `~astropy.modeling.models.Polynomial2D`.  If the array is 1D,
            then provide a simple pedestal background.

        Returns
        -------

        bg_model : Array with dimensions `(self.fit_mask.sum(),)`
        
            Flattened, masked background array.

        """
        from astropy.modeling.models import Polynomial2D
        
        # Initialize beam pixel coordinates
        for beam in self.beams:
            needs_init = not hasattr(beam, 'xp')
            if hasattr(beam, 'xp_mask'):
                needs_init |= apply_mask is not beam.xp_mask
                
            if needs_init:
                #print('Initialize xp/yp')
                yp, xp = np.indices(beam.sh)
                xp = (xp - beam.sh[1]/2.)/(beam.sh[1]/2.)
                
                # normalized to center
                yp = (yp - beam.sh[0]/2.)/(beam.sh[0]/2.)
                
                if apply_mask:
                    beam.xp = xp.flatten()[beam.fit_mask]
                    beam.yp = yp.flatten()[beam.fit_mask]
                else:
                    beam.xp = xp.flatten()
                    beam.yp = yp.flatten()
                    
                beam.xp_mask = apply_mask
                
            if (not hasattr(beam, 'ones')) | needs_init:
                if apply_mask:
                    beam.ones = np.ones(beam.fit_mask.sum())
                else:
                    beam.ones = np.ones(beam.fit_mask.size)
                    
        # Initialize 2D polynomial
        poly = None
        if bg_params.ndim > 1:
            if bg_params.shape[1] > 1:
                M = bg_params.shape[1]
                order = {3:1,6:2,10:3}
                poly = Polynomial2D(order[M])

        #mfull = self.scif[self.fit_mask]
        bg_full = []

        for ib, beam in enumerate(self.beams):        
            if poly is not None:
                poly.parameters = bg_params[ib, :]
                bg_i = poly(beam.xp, beam.yp)
            else:
                # Order = 0, pedestal offset
                bg_i = beam.ones*bg_params[ib]

            bg_full.append(bg_i)

        return np.hstack(bg_full)
    
    @staticmethod
    def _objective_line_width(params, self, verbose):
        """
        Objective function for emission line velocity widths
        """
        bl, nl, z = params

        t0, t1 = utils.load_quasar_templates(uv_line_complex=False, broad_fwhm=bl*1000, narrow_fwhm=nl*1000, t1_only=True)
        tfit = self.template_at_z(z=z, templates=t1, fitter='nnls', fit_background=True, get_residuals=True)

        if verbose:
            print(params, tfit['chi2'].sum())

        return tfit['chi2']
    
    def fit_line_width(self, bl=2.5, nl=1.1, z0=1.9367, max_nfev=100, tol=1.e-3, verbose=False):
        """
        Fit for emisson line width

        Returns:

            width/(1000 km/s), z, nfev, (nfev==max_nfev)

        """
        from scipy.optimize import least_squares

        init = [bl, nl, z0]
        args = (self, verbose)

        out = least_squares(self._objective_line_width, init, jac='2-point', method='lm', ftol=tol, xtol=tol, gtol=tol, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=max_nfev, verbose=0, args=args, kwargs={})

        params = out.x
        res = [out.x[0], out.x[1], out.x[2], out.nfev, out.nfev == max_nfev]
        return res
    
def show_drizzled_lines(line_hdu, full_line_list=['OII', 'Hb', 'OIII', 'Ha', 'SII', 'SIII'], size_arcsec=2, cmap='cubehelix_r', scale=1., dscale=1):
    """TBD
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    
    show_lines = []
    for line in full_line_list:
        if line in line_hdu[0].header['HASLINES'].split():
            show_lines.append(line)
    
    #print(line_hdu[0].header['HASLINES'], show_lines)
    
    NL = len(show_lines)
    
    fig = plt.figure(figsize=[3*(NL+1),3.4])
    
    # Direct
    ax = fig.add_subplot(1,NL+1,1)
    ax.imshow(line_hdu['DSCI'].data*dscale, vmin=-0.02, vmax=0.6, cmap=cmap, origin='lower')
    ax.set_title('Direct   {0}    z={1:.3f}'.format(line_hdu[0].header['ID'], line_hdu[0].header['REDSHIFT']))
    
    ax.set_xlabel('RA'); ax.set_ylabel('Decl.')

    # 1" ticks
    pix_size = np.abs(line_hdu['DSCI'].header['CD1_1']*3600)
    majorLocator = MultipleLocator(1./pix_size)
    N = line_hdu['DSCI'].data.shape[0]/2
    ax.errorbar([N-0.5/pix_size], N-0.9*size_arcsec/pix_size, yerr=0, xerr=0.5/pix_size, color='k')
    ax.text(N-0.5/pix_size, N-0.9*size_arcsec/pix_size, r'$1^{\prime\prime}$', ha='center', va='bottom', color='k')

    # Line maps
    for i, line in enumerate(show_lines):
        ax = fig.add_subplot(1,NL+1,2+i)
        ax.imshow(line_hdu['LINE',line].data*scale, vmin=-0.02, vmax=0.6, cmap=cmap, origin='lower')
        ax.set_title(r'%s %.3f $\mu$m' %(line, line_hdu['LINE', line].header['WAVELEN']/1.e4))

    # End things
    for ax in fig.axes:
        ax.set_yticklabels([]); ax.set_xticklabels([])
        ax.set_xlim(N+np.array([-1,1])*size_arcsec/pix_size)
        ax.set_ylim(N+np.array([-1,1])*size_arcsec/pix_size)
        
        x0 = np.mean(ax.get_xlim())
        y0 = np.mean(ax.get_xlim())
        ax.scatter(N, N, marker='+', color='k', zorder=100, alpha=0.5)
        ax.scatter(N, N, marker='+', color='w', zorder=101, alpha=0.5)
        
        ax.xaxis.set_major_locator(majorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    fig.tight_layout(pad=0.1, w_pad=0.5)
    return fig

