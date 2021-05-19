"""
Simple tests for grizli
"""

def test_flt():
    """
    Disperse a direct FLT
    """
    import matplotlib as mpl
    mpl.use('Agg')
    
    import matplotlib.pyplot as plt    
    import matplotlib.gridspec
    import numpy as np
    
    import grizli
        
    ### (demo on aligned, background-subtracted FLT images)
    
    #########
    ### Initialize the GrismFLT object
    flt = grizli.model.GrismFLT(flt_file='ibhj34h8q_flt.fits', direct_image='ibhj34h6q_flt.fits', refimage=None, segimage=None, verbose=True, pad=0)
    
    ## Make a catalog/segmetnation image from the direct FLT and make a full
    ## grism model for those detected objects
    flt.photutils_detection(detect_thresh=2, grow_seg=5, gauss_fwhm=2., 
                            compute_beams=['A','B', 'C','D'],
                            verbose=True, save_detection=False, wcs=None)
    
    ## Find object near (x,y) = (495, 749)
    #dr = np.sqrt((flt.catalog['xcentroid']-330)**2+(flt.catalog['ycentroid']-744)**2)
    dr = np.sqrt((flt.catalog['xcentroid']-495)**2+(flt.catalog['ycentroid']-749)**2)
    dr = np.sqrt((flt.catalog['xcentroid']-712)**2+(flt.catalog['ycentroid']-52)**2)
    
    ix = np.argmin(dr)
    id, x0, y0 = flt.catalog['id'][ix], flt.catalog['xcentroid'][ix]+1, flt.catalog['ycentroid'][ix]+1
    
    ## Get basic trace parameters, `conf` is a grizli.grism.aXeConf object, here for G141 & F140W
    # x pixels from the center of the direct image
    dx = np.arange(220)
    # ytrace and wavelength at x=dx
    dy, lam = flt.conf.get_beam_trace(x=x0, y=y0, dx=dx, beam='A')
    
    fig = plt.figure(figsize=[5,1.5])
    #fig = plt.Figure(figsize=[5,1.5])
    ax = fig.add_subplot(111)
    ax.imshow(flt.im_data['SCI'], cmap='gray_r', vmin=-0.05, vmax=0.2, interpolation='Nearest', aspect='auto')
    ax.set_xlim(x0-10, x0+230)
    ax.set_ylim(y0-10, y0+10)
    
    ax.plot(x0+dx-1, y0+dy-1, color='red', linewidth=3, alpha=0.7)
    ## 0.1 micron tick marks along the trace as in the next figure
    xint = np.interp(np.arange(1,1.81,0.1), lam/1.e4, dx)
    yint = np.interp(np.arange(1,1.81,0.1), lam/1.e4, dy)
    ax.scatter(x0+xint-1, y0+yint-1, marker='o', color='red', alpha=0.8)
    ax.set_xlabel(r'$x$ (FLT)'); ax.set_ylabel(r'$y$ (FLT)')
    
    fig.tight_layout(pad=0.1)
    fig.savefig('grizli_demo_0.pdf')
        
    #########
    ### Spectrum cutout
    beam = grizli.model.BeamCutout(id=id, x=x0, y=y0, cutout_dimensions=[18,18], conf=flt.conf, GrismFLT=flt)
    
    # (mask bad pixel)
    beam.cutout_seg[(beam.thumb/beam.photflam > 100) | (beam.thumb < 0)] = 0
    beam.total_flux = np.sum(beam.thumb[beam.cutout_seg == beam.id])
    
    ### Compute the model in the FLT frame for a single object
    model_id = flt.compute_model(id=id, x=x0, y=y0, sh=[80,80], in_place=False).reshape(flt.sh_pad)
    beam.contam = beam.get_cutout(flt.model-model_id)
    
    ## 1D optimal extraction (Horne 1986)
    xspec, yspec, yerr = beam.optimal_extract(beam.cutout_sci, bin=0)
    ## Simple flat flambda continuum model, normalized at F140W
    beam.compute_model(beam.thumb, id=beam.id)
    cmodel = beam.model*1.
    xspecm, yspecm, yerrm = beam.optimal_extract(cmodel, bin=0)
    
    ## Fit simple line + continuum slope
    line_centers, coeffs, chi2, lmask, lmodel, l0, lflux = beam.simple_line_fit(fwhm=5., grid=[1.12e4, 1.65e4, 1, 20])
    xspecl, yspecl, yerrl = beam.optimal_extract(lmodel, bin=0)
    
    ### Make a figure
    fig = plt.figure(figsize=(8,4))
    #fig = plt.Figure(figsize=(8,4))
    
    ## 1D plots
    gsb = matplotlib.gridspec.GridSpec(3,1)  
    
    ax = fig.add_subplot(gsb[-2:,:])
    ax.errorbar(xspec/1.e4, yspec, yerr, linestyle='None', marker='o', markersize=3, color='black', alpha=0.5, label='Data (id=%d)' %(beam.id))
    ax.plot(xspecm/1.e4, yspecm, color='red', linewidth=2, alpha=0.8, label=r'Flat $f_\lambda$ (%s)' %(beam.filter))
    ax.plot(xspecl/1.e4, yspecl, color='orange', linewidth=2, alpha=0.8, label='Cont+line (%.3f, %.2e)' %(l0/1.e4, lflux*1.e-17))
    ax.legend(fontsize=8, loc='lower center', scatterpoints=1)
    
    ax.set_xlabel(r'$\lambda$'); ax.set_ylabel('flux (e-/s)')
    
    ax = fig.add_subplot(gsb[-3,:])
    ax.plot(line_centers/1.e4, chi2/lmask.sum())
    ax.set_xticklabels([])
    ax.set_ylabel(r'$\chi^2/(\nu=%d)$' %(lmask.sum()))
    
    xt = np.arange(1.,1.82,0.1)
    for ax in fig.axes:
        ax.set_xlim(1., 1.8)
        ax.set_xticks(xt)
        
    axt = ax.twiny()
    axt.set_xlim(np.array(ax.get_xlim())*1.e4/6563.-1)
    axt.set_xlabel(r'$z_\mathrm{H\alpha}$')

    ## 2D spectra
    gst = matplotlib.gridspec.GridSpec(3,1)  
    if 'viridis_r' in plt.colormaps():
        cmap = 'viridis_r'
    else:
        cmap = 'cubehelix_r'
    
    ax = fig.add_subplot(gst[0,:])
    ax.imshow(beam.cutout_sci, vmin=-0.05, vmax=0.2, cmap=cmap, interpolation='Nearest', origin='lower', aspect='auto')
    ax.set_ylabel('Observed')
    
    ax = fig.add_subplot(gst[1,:])
    ax.imshow(lmodel+beam.contam, vmin=-0.05, vmax=0.2, cmap=cmap, interpolation='Nearest', origin='lower', aspect='auto')
    ax.set_ylabel('Model')

    ax = fig.add_subplot(gst[2,:])
    ax.imshow(beam.cutout_sci-lmodel-beam.contam, vmin=-0.05, vmax=0.2, cmap=cmap, interpolation='Nearest', origin='lower', aspect='auto')
    ax.set_ylabel('Resid.')
    
    for ax in fig.axes[-3:]:
        ax.set_yticklabels([])
        xi = np.interp(xt, beam.wave/1.e4, np.arange(beam.shg[1]))
        xl = np.interp([1,1.8], beam.wave/1.e4, np.arange(beam.shg[1]))
        ax.set_xlim(xl)
        ax.set_xticks(xi)
        ax.set_xticklabels([])
    
    gsb.tight_layout(fig, pad=0.1,h_pad=0.01, rect=(0,0,0.5,1))
    gst.tight_layout(fig, pad=0.1,h_pad=0.01, rect=(0.5,0.1,1,0.9))
    
    fig.savefig('grizli_demo_1.pdf')
    
    # ## Emission line with calibrated flux
    # lpar = lflux*1.e-17,l0, 3 # line parameters: flux, wave, rms width
    # wave_array = np.arange(lpar[1]-100, lpar[1]+100, lpar[2]/10.)
    # gauss_line = lpar[0]/np.sqrt(2*np.pi*lpar[2]**2)*np.exp(-(wave_array - lpar[1])**2/2/lpar[2]**2)
    # line_model = beam.compute_model(beam.thumb/beam.total_flux, xspec=wave_array, yspec=gauss_line, id=beam.id, in_place=False)
    # xspecl, yspecl, yerrl = beam.optimal_extract(line_model.reshape(beam.shg), bin=0)
    # plt.plot(xspecl, yspecl+yspecm, color='green', linewidth=2, alpha=0.8, linestyle='steps-mid')
    # 
    # ## Check line flux with pysynphot
    # import pysynphot as S
    # g = S.ArraySpectrum(wave_array, gauss_line, fluxunits='flam')
    # g = S.GaussianSource(lflux*1.e-17, l0, lpar[2]*2.35)
    # bp = S.ObsBandpass('wfc3,ir,g141')
    # obs = S.Observation(g, bp) #, force='extrap')
    # print 'Model countrate: %.2f, pySynphot countrate: %.2f' %(line_model.sum(), obs.countrate())
    
    #######################################
    
        
    
    