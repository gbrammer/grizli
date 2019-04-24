
def apply_catalog_corrections(root, total_flux='flux_auto', auto_corr=True, get_external_photometry=False, aperture_indices='all', suffix='_apcorr', verbose=True, apply_background=True):
    """
    Aperture and background corrections to photometric catalog
    """
    import os
    import eazy
    import numpy as np
    
    from grizli import utils
    import mastquery.utils
    
    cat = utils.read_catalog('{0}_phot.fits'.format(root))
    filters = []
    for c in cat.meta:
        if c.endswith('_ZP'):
            filters.append(c.split('_ZP')[0].lower())
    
    if get_external_photometry:
        print('Get external photometry from Vizier')
        try:
            ext = get_external_catalog(cat, external_limits=external_limits,
                                       timeout=external_timeout,
                                       sys_err=external_sys_err)
            for c in ext.colnames:
                if c not in cat.colnames:
                    cat[c] = ext[c]
            
            for k in ext.meta:
                cat.meta[k] = ext.meta[k]
        except:
            print(' - External catalog FAILED')
            pass
        
    # Fix: Take flux_auto when flag==0, flux otherwise
    if (total_flux == 'flux_auto_fix') & (total_flux not in cat.colnames):
        flux = cat['flux_auto']*1.
        flagged = (cat['flag'] > 0)
        flux[flagged] = cat['flux'][flagged]
        cat['flux_auto_fix'] = flux*1.
    
    # Additional auto correction

    cat.meta['TOTALCOL'] = total_flux, 'Column for total flux'
    #cat.meta['HASTOT'] = (auto_corr &  ('tot_corr' in cat.colnames), 'Catalog has full total flux')
    
    apcorr = {}
    for NAPER in range(100):
        if 'APER_{0}'.format(NAPER) not in cat.meta:
            break
    
    if aperture_indices == 'all':
        aperture_indices = range(NAPER)
        
    for i in aperture_indices:
        
        if verbose:
            print('Compute aperture corrections: i={0}, D={1:.2f}" aperture'.format(i, cat.meta['ASEC_{0}'.format(i)]))
            
        if 'flux_aper_{0}'.format(i) in cat.colnames:
            cat['apcorr_{0}'.format(i)] = cat[total_flux]/cat['flux_aper_{0}'.format(i)]
            for f in filters:
                bkgc = '{0}_bkg_aper_{1}'.format(f, i)
                if (bkgc in cat.colnames) & apply_background:
                    bkg = cat[bkgc]
                else:
                    bkg = 0.
                    
                cat['{0}_corr_{1}'.format(f, i)] = (cat['{0}_flux_aper_{1}'.format(f, i)]-bkg)*cat['apcorr_{0}'.format(i)]
                cat['{0}_ecorr_{1}'.format(f, i)] = cat['{0}_fluxerr_aper_{1}'.format(f, i)]*cat['apcorr_{0}'.format(i)]
                
                # mask_thresh = np.percentile(cat['{0}_mask_aper_{1}'.format(f, i)], 95)
                aper_area = np.pi*(cat.meta['APER_{0}'.format(i)]/2)**2
                mask_thresh = aper_area
                
                bad = cat['{0}_mask_aper_{1}'.format(f, i)] > 0.2*mask_thresh
                cat['{0}_corr_{1}'.format(f, i)][bad] = -99
                cat['{0}_ecorr_{1}'.format(f, i)][bad] = -99
                
                tot_col = '{0}_tot_corr'.format(f.lower())
                
                if auto_corr and (tot_col in cat.colnames):   
                    cat['{0}_tot_{1}'.format(f, i)] = cat['{0}_corr_{1}'.format(f, i)]*cat[tot_col]
                    cat['{0}_etot_{1}'.format(f, i)] = cat['{0}_ecorr_{1}'.format(f, i)]*cat[tot_col]

                    cat['{0}_tot_{1}'.format(f, i)][bad] = -99
                    cat['{0}_etot_{1}'.format(f, i)][bad] = -99
                                    
    cat.rename_column('number','id')
    cat['z_spec'] = cat['id']*0.-1
    
    # Spurious sources, sklearn SVM model trained for a single field
    morph_model = os.path.join(os.path.dirname(utils.__file__),
                               'data/sep_catalog_junk.pkl')
                               
    if os.path.exists(morph_model):
        if verbose:
            print('Apply morphological validity class')
        
        from sklearn.externals import joblib
        clf = joblib.load(morph_model)
        X = np.hstack([[cat['peak']/cat['flux'], 
                        cat['cpeak']/cat['peak']]]).T
        
        # Predict labels, which where generated for 
        #    bad_bright, bad_faint, stars, big_galaxies, small_galaxies
        pred = clf.predict_proba(X)
        
        # Should be >~ 0.9 for valid sources, stars & galaxies in "ir" image
        cat['class_valid'] = pred[:,-3:].sum(axis=1) 
        cat['class_valid'].format = '.2f'
    
    cat['dummy_err'] =  10**(-0.4*(8-23.9))
    cat['dummy_flux'] = cat[total_flux] # detection band
    
    if suffix:
        if verbose:
            print('Write {0}_phot{1}.fits'.format(root, suffix))
            
        cat.write('{0}_phot{1}.fits'.format(root, suffix), overwrite=True)
    
    return cat
    
def eazy_photoz(root, force=False, object_only=True, apply_background=True, aper_ix=1, apply_prior=False, beta_prior=True, get_external_photometry=False, external_limits=3, external_sys_err=0.3, external_timeout=300, sys_err=0.05, z_step=0.01, z_min=0.01, z_max=12, total_flux='flux_auto', auto_corr=True, compute_residuals=False, dummy_prior=False, extra_rf_filters=[]):
    
    import os
    import eazy
    import numpy as np
    
    from grizli import utils
    import mastquery.utils
    
    if (os.path.exists('{0}.eazypy.self.npy'.format(root))) & (not force):
        self = np.load('{0}.eazypy.self.npy'.format(root))[0]
        zout = utils.read_catalog('{0}.eazypy.zout.fits'.format(root))
        cat = utils.read_catalog('{0}_phot_apcorr.fits'.format(root))
        return self, cat, zout
        
    trans = {'f098m':201, 'f105w':202, 'f110w':241, 'f125w':203, 'f140w':204, 'f160w':205, 'f435w':233, 'f438w':211, 'f606w':236, 'f625w':237, 'f814w':239, 'f702w':15, 'f555w':235, 'f350lp':339, 'f475w':212, 'f775w':238, 'f850lp':240}
    #trans.pop('f814w')
    
    print('Apply catalog corrections')
    apply_catalog_corrections(root, suffix='_apcorr')
    
    cat = utils.read_catalog('{0}_phot_apcorr.fits'.format(root))
    filters = []
    for c in cat.meta:
        if c.endswith('_ZP'):
            filters.append(c.split('_ZP')[0].lower())
        
    # Translate
    fp = open('zphot.translate','w')
    for f in filters:
        if f in trans:
            fp.write('{0}_tot_{1} F{2}\n'.format(f, aper_ix, trans[f]))
            fp.write('{0}_etot_{1} E{2}\n'.format(f, aper_ix, trans[f]))
    
    fp.write('irac_ch1_flux F18\n')
    fp.write('irac_ch1_err  E18\n')
    
    # For zeropoint
    if dummy_prior:
        fp.write('dummy_flux F205x\n')
        fp.write('dummy_err  E205x\n')
            
    fp.close()
    
    params = {}
    params['CATALOG_FILE'] = '{0}_phot_apcorr.fits'.format(root)
    params['Z_STEP'] = z_step
    params['MAIN_OUTPUT_FILE'] = '{0}.eazypy'.format(root)
    
    params['Z_MAX'] = z_max
    params['MW_EBV'] = mastquery.utils.get_irsa_dust(cat['ra'].mean(), cat['dec'].mean())
    params['PRIOR_ABZP'] = 23.9
    
    params['SYS_ERR'] = sys_err
    params['CAT_HAS_EXTCORR'] = False
    
    # Pick prior filter, starting from reddest
    for f in ['f435w', 'f606w', 'f814w', 'f105w', 'f110w', 'f125w', 'f140w', 'f160w'][::-1]:
        if f in filters:
            if dummy_prior:
                params['PRIOR_FILTER'] = 'dummy_flux'
            else:
                params['PRIOR_FILTER'] = trans[f]
            
            mag = 23.9-2.5*np.log10(cat['{0}_corr_{1}'.format(f, aper_ix)])
            break
    #
    if os.path.exists('templates/fsps_full/tweak_fsps_QSF_11_v3_noRed.param.fits'):
        params['TEMPLATES_FILE'] = 'templates/fsps_full/tweak_fsps_QSF_11_v3_noRed.param'
    else:
        params['TEMPLATES_FILE'] = 'templates/fsps_full/tweak_fsps_QSF_12_v3.param'

    zpfile = None
    load_products = False

    if (not os.path.exists('FILTER.RES.latest') or 
        not os.path.exists('templates')):
        try:
            # should work with eazy-py >= 0.2.0
            eazy.symlink_eazy_inputs(path=None)
        except:
            print("""
The filter file `FILTER.RES.latest` and `templates` directory were not
found in the working directory and the automatic command to retrieve them 
failed: 
    
    >>> import eazy; eazy.symlink_eazy_inputs(path=None)  

Run it with `path` pointing to the location of the `eazy-photoz` repository.""")
            return False
            
    self = eazy.photoz.PhotoZ(param_file=None, translate_file='zphot.translate', zeropoint_file=zpfile, params=params, load_prior=True, load_products=load_products)
    
    if object_only:
        return self
        
    idx = np.arange(self.NOBJ)
    
    #sample = (mag < 27) #& (self.cat['star_flag'] != 1)
    #sample |= (self.cat['z_spec'] > 0)
    sample = np.isfinite(self.cat['id']) #mag)
    
    for iter in range(1+(get_external_photometry & compute_residuals)*1):
        self.fit_parallel(idx[sample], n_proc=10)
        if compute_residuals:
            self.error_residuals()
    
    self.standard_output(prior=apply_prior, beta_prior=beta_prior, extra_rf_filters=extra_rf_filters)
    
    zout = utils.read_catalog('{0}.eazypy.zout.fits'.format(root))
    
    np.save('{0}.eazypy.self.npy'.format(root), [self])
    
    return self, cat, zout
    
def show_from_ds9(ds9, self, zout, **kwargs):
    
    import numpy as np
    
    xy = np.cast[float](ds9.get('pan image').split())
    r = np.sqrt((self.cat['x_image']-xy[0])**2 + (self.cat['y_image']-xy[1])**2)
    
    ix = np.argmin(r)
    print('ID: {0}, r={1:.1f} pix'.format(self.cat['id'][ix], r[ix]))
    print('  z={0:.2f} logM={1:.2f}'.format(zout['z_phot'][ix], np.log10(zout['mass'][ix])))
    
    fig = self.show_fit(self.cat['id'][ix], **kwargs)
    return fig, self.cat['id'][ix], zout['z_phot'][ix]
     
class EazyPhot(object):
    def __init__(self, photoz, grizli_templates=None, zgrid=None, apcorr=None):
        """
        photoz : `~eazypy.photoz.PhotoZ`
        
        apcorr : array
            Aperture correction applied to the photometry to match the 
            grism spectra.  For the internal grizli catalogs, this should 
            generally be something like 
            
            >>> apcorr = 'flux_iso' / 'flux_auto'
            
            
        """
        try:
            from .. import utils
        except:
            from grizli import utils
            
        not_obs_mask =  (photoz.fnu < -90) | (photoz.efnu < 0)
        
        self.zgrid = photoz.zgrid
        
        if apcorr is None:
            self.apcorr = np.ones(photoz.NOBJ)
        else:
            self.apcorr = apcorr
            
        self.flam = photoz.fnu*photoz.to_flam*photoz.zp*photoz.ext_corr
        self.flam[not_obs_mask] = -99
        
        self.eflam = photoz.efnu*photoz.to_flam*photoz.zp**photoz.ext_corr
        self.eflam[not_obs_mask] = -99
        
        #self.rdcat = utils.GTable(photoz.cat['ra','dec'])
        self.ra_cat = photoz.cat['ra'].data
        self.dec_cat = photoz.cat['dec'].data
        
        self.filters = photoz.filters
        self.f_numbers = photoz.f_numbers
        self.param = photoz.param
        
        if grizli_templates is None:
            self.tempfilt = None
        else:
            self.tempfilt = self.initialize_templates(grizli_templates,
                                                      zgrid=zgrid)
            
    def initialize_templates(self, grizli_templates, zgrid=None):
        
        from eazy import templates as templates_module 
        from eazy.photoz import TemplateGrid
        
        if zgrid is None:
            zgrid = self.zgrid
            
        template_list = [templates_module.Template(arrays=(grizli_templates[k].wave, grizli_templates[k].flux), name=k) for k in grizli_templates]
        
        tempfilt = TemplateGrid(zgrid, template_list, RES=self.param['FILTERS_RES'], f_numbers=self.f_numbers, add_igm=True, galactic_ebv=self.param['MW_EBV'], Eb=self.param['SCALE_2175_BUMP'])
        
        return tempfilt
    
    def get_phot_dict(self, ra, dec):
        """
        
        """
        from collections import OrderedDict
        try:
            from .. import utils
        except:
            from grizli import utils
            
        icat = utils.GTable()
        icat['ra'] = [ra]
        icat['dec'] = [dec]
        
        rdcat = utils.GTable()
        rdcat['ra'] = self.ra_cat
        rdcat['dec'] = self.dec_cat
        
        ix, dr = rdcat.match_to_catalog_sky(icat)
        
        phot = OrderedDict()
        apcorr_i = self.apcorr[ix[0]]
        
        phot['flam'] = self.flam[ix[0],:]*1.e-19*apcorr_i
        phot['eflam'] = self.eflam[ix[0],:]*1.e-19*apcorr_i
        phot['filters'] = self.filters
        phot['tempfilt'] = self.tempfilt
        return phot, ix[0], dr[0]
        
def get_external_catalog(phot, filter_file='/usr/local/share/eazy-photoz/filters/FILTER.RES.latest', ZP=23.9, sys_err=0.3, verbose=True, external_limits=3, timeout=300):
    """
    Fetch photometry from vizier
    """
    import numpy as np
    
    import astropy.units as u
    
    try:
        from .. import utils
    except:
        from grizli import utils
    
    from eazy.filters import FilterFile
    res = FilterFile(filter_file)
        
    vizier_catalog = list(utils.VIZIER_VEGA.keys())

    ra = np.median(phot['ra'])
    dec = np.median(phot['dec'])
    dr = np.sqrt((phot['ra']-ra)**2*np.cos(phot['dec']/180*np.pi)**2 + (phot['dec']-dec)**2)*60
    
    radius = 1.5*dr.max() # arcmin
    tabs = utils.get_Vizier_photometry(ra, dec, templates=None, radius=radius*60, vizier_catalog=vizier_catalog, filter_file=filter_file, MW_EBV=0, convert_vega=False, raw_query=True, verbose=True, timeout=timeout)

    extern_phot = utils.GTable()
    N = len(phot)
    
    for t_i in tabs:

        # Match
        if 'RAJ2000' in t_i.colnames:
            other_radec = ('RAJ2000','DEJ2000')
        elif 'RA_ICRS' in t_i.colnames:
            other_radec = ('RA_ICRS','DE_ICRS')
        else:
            other_radec = ('ra','dec')
            
        idx, dr = phot.match_to_catalog_sky(t_i, other_radec=other_radec)
        if (dr < 2*u.arcsec).sum() == 0:
            continue
        
        tab = t_i[dr < 2*u.arcsec]
        idx = idx[dr < 2*u.arcsec]
            
        # Downweight PS1 if have SDSS
        # if (tab.meta['name'] == PS1_VIZIER) & (SDSS_DR12_VIZIER in viz_tables):
        #     continue
        #     err_scale = 1
        # else:
        #     err_scale = 1
        err_scale = 1
        
        if (tab.meta['name'] == utils.UKIDSS_LAS_VIZIER):
            flux_scale = 1.33
        else:
            flux_scale = 1.

        convert_vega = utils.VIZIER_VEGA[tab.meta['name']]
        bands = utils.VIZIER_BANDS[tab.meta['name']]

        #if verbose:
        #    print(tab.colnames)

        #filters += [res.filters[res.search(b, verbose=False)[0]] for b in bands]

        to_flam = 10**(-0.4*(48.6))*3.e18 # / pivot(Ang)**2
        
        for ib, b in enumerate(bands):
            f_number = res.search(b, verbose=False)[0]+1
            filt = res.filters[f_number-1]
            #filters.append(filt)

            if convert_vega:
                to_ab = filt.ABVega()
            else:
                to_ab = 0.
            
            fcol, ecol = bands[b]
            #pivot.append(filt.pivot())
            fnu = 10**(-0.4*(tab[fcol]+to_ab-ZP))
            efnu = tab[ecol]*np.log(10)/2.5*fnu*err_scale
            
            efnu = np.sqrt(efnu**2+(sys_err*fnu)**2)
            
            fnu.fill_value = -99
            efnu.fill_value = -99
            
            comment = 'Filter {0} from {1} (N={2})'.format(bands[b][0], t_i.meta['name'], len(idx))
            if verbose:
                print(comment)
            
            if ((~efnu.mask).sum() > 4) & (external_limits > 0):
                fill = np.percentile(efnu.data[~efnu.mask], [90])
                efill = external_limits * fill
            else:
                fill = -99
                efill = -99
                
            extern_phot.meta['F{0}'.format(f_number)] = b, comment
            extern_phot['F{0}'.format(f_number)] = fill*np.ones(N)
            extern_phot['F{0}'.format(f_number)][idx] = fnu.filled()
            
            extern_phot['E{0}'.format(f_number)] = efill*np.ones(N)
            extern_phot['E{0}'.format(f_number)][idx] = efnu.filled()
            
    return extern_phot
    
########### Selecting objects
def select_objects():
    from grizli.pipeline import photoz
    import numpy as np
    
    total_flux = 'flux_auto_fix'
    total_flux = 'flux_auto' # new segmentation masked SEP catalogs
    
    self, cat, zout = photoz.eazy_photoz(root, object_only=False, apply_prior=False, beta_prior=True, aper_ix=1, force=True, get_external_photometry=False, compute_residuals=False, total_flux=total_flux)

    flux = self.cat[total_flux]*1.
    hmag = 23.9-2.5*np.log10(cat['f160w_tot_2'])

    # Reddest HST band
    lc_clip = self.lc*1
    lc_clip[self.lc > 1.55e4] = 0 # in case ground-based / WISE red band
    ixb = np.argmax(lc_clip)
    sn_red = self.cat[self.flux_columns[ixb]]/self.cat[self.err_columns[ixb]]

    grad = np.gradient(self.zgrid)
    cumpz = np.cumsum(self.pz*grad, axis=1); cumpz = (cumpz.T/cumpz[:,-1]).T

    chi2 = (self.chi_best / self.nusefilt)

    iz6 = np.where(self.zgrid > 6.0)[0][0]
    iz7 = np.where(self.zgrid > 7)[0][0]
    iz8 = np.where(self.zgrid > 8)[0][0]
    iz9 = np.where(self.zgrid > 9)[0][0]

    highz_sel = (hmag < 27.5) & (self.cat['class_valid'] > 0.8)
    #highz_sel |= (cumpz[:,iz6] < 0.3) & (self.cat['flux_radius'] > 2.5) 
    #highz_sel &= (self.cat['flux_radius'] > 2.5) 
    highz_sel &= chi2 < 3
    highz_sel &= (sn_red > 5)
    highz_sel &= self.nusefilt >= 3
    
    flux_ratio = (cat['f160w_flux_aper_3'] - cat['f160w_bkg_aper_3'])/(cat['f160w_flux_aper_0'] - cat['f160w_bkg_aper_0'])
    flux_ratio /= cat.meta['APER_3']**2/cat.meta['APER_0']**2
    
    if False:
        sel = highz_sel
        so = np.argsort(cumpz[sel,iz7]); ids = self.cat['id'][sel][so]; i=-1
        so = np.argsort(flux_ratio[sel])[::-1]; ids = self.cat['id'][sel][so]; i=-1
        
    highz_sel &= (cumpz[:,iz6] > 0) & (flux_ratio < 0.45) & ((cumpz[:,iz6] < 0.3) | (cumpz[:,iz7] < 0.3) | (((cumpz[:,iz8] < 0.4) |  (cumpz[:,iz9] < 0.5)) & (flux_ratio < 0.5))) 

    # Big objects likely diffraction spikes
    # big = (self.cat['flux_radius'] > 10)
    # highz_sel &= ~big
    #flux_ratio = self.cat['flux_aper_0']/self.cat['flux_aper_2']

    sel = highz_sel
    so = np.argsort(hmag[sel]); ids = self.cat['id'][sel][so]; i=-1

    # Red 
    uv = -2.5*np.log10(zout['restU']/zout['restV'])
    red_sel = ((zout['z160'] > 1.) & (uv > 1.5)) | ((zout['z160'] > 1.5) & (uv > 1.1))
    red_sel &= (self.zbest < 4) & (hmag < 22)# & (~hmag.mask) 
    red_sel &= (zout['mass'] > 10**10.5) #& (self.cat['class_valid'] > 0.8)
    red_sel &= (self.cat['flux_radius'] > 2.5)
    red_sel &= (zout['restV']/zout['restV_err'] > 3)
    red_sel &= (chi2 < 3)
    #red_sel &= (sn_red > 20)

    sel = red_sel

    so = np.argsort(hmag[sel]); ids = self.cat['id'][sel][so]; i=-1
    
    ds9 = None
    
    for j in self.idx[sel][so]:
        id_j, ra, dec = self.cat['id', 'ra', 'dec'][j]

        # Photo-z
        fig, data = self.show_fit(id_j, ds9=ds9, show_fnu=True) #highz_sel[j])    
        lab = '{0} {1}\n'.format(root, id_j)
        lab += 'H={0:.1f} z={1:.1f}\n'.format(hmag[j], self.zbest[j])
        lab += 'U-V={0:.1f}, logM={1:4.1f}'.format(uv[j], np.log10(zout['mass'][j]))

        ax = fig.axes[0]
        ax.text(0.95, 0.95, lab, ha='right', va='top', transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='w', edgecolor='None', alpha=0.5))
        yl = ax.get_ylim()
        ax.set_ylim(yl[0], yl[1]*1.1)

        fig.savefig('{0}_{1:05d}.eazy.png'.format(root, id_j), dpi=70)
        plt.close()

        # Cutout
        #from grizli_aws.aws_drizzle import drizzle_images
        
        #rgb_params = {'output_format': 'png', 'output_dpi': 75, 'add_labels': False, 'show_ir': False, 'suffix':'.rgb'}
        rgb_params = None

        #aws_bucket = 's3://grizli/SelectedObjects/'
        aws_bucket = None

        label = '{0}_{1:05d}'.format(root, id_j)
        if not os.path.exists('{0}.rgb.png'.format(label)):
            drizzle_images(label=label, ra=ra, dec=dec, pixscale=0.06, size=8, pixfrac=0.8, theta=0, half_optical_pixscale=False, filters=['f160w', 'f814w', 'f140w', 'f125w', 'f105w', 'f110w', 'f098m', 'f850lp', 'f775w', 'f606w', 'f475w'], remove=False, rgb_params=rgb_params, master='grizli-jan2019', aws_bucket=aws_bucket)

        show_all_thumbnails(label=label, filters=['f775w','f814w','f098m','f105w','f110w','f125w','f140w','f160w'], scale_ab=np.clip(hmag[j],19,22), close=True)

############
def show_all_thumbnails(label='j022708p4901_00273', filters=['f775w','f814w','f098m','f105w','f110w','f125w','f140w','f160w'], scale_ab=21, close=True):
    """
    Show individual filter and RGB thumbnails
    """
    from astropy.visualization import make_lupton_rgb
    from PIL import Image
    
    ims = {}
    for filter in filters:
        drz_files = glob.glob('{0}-{1}*_dr*sci.fits'.format(label, filter))
        if len(drz_files) > 0:
            im = pyfits.open(drz_files[0])
            ims[filter] = im
            
    slx, sly, filts, fig = auto_script.field_rgb(root=label, xsize=4, output_dpi=None, HOME_PATH=None, show_ir=False, pl=1, pf=1, scl=1, rgb_scl=[1, 1, 1], ds9=None, force_ir=False, filters=ims.keys(), add_labels=False, output_format='png', rgb_min=-0.01, xyslice=None, pure_sort=False, verbose=True, force_rgb=None, suffix='.rgb', scale_ab=scale_ab)
    if close:
        plt.close()
    
    rgb = np.array(Image.open('{0}.rgb.png'.format(label)))
    
    NX = (len(filters)+1)
    fig = plt.figure(figsize=[1.5*NX,1.5])
    ax = fig.add_subplot(1,NX,NX)
    ax.imshow(rgb, origin='upper', interpolation='nearest')
    ax.text(0.5, 0.95, label, ha='center', va='top', transform=ax.transAxes, fontsize=7, bbox=dict(facecolor='w', edgecolor='None', alpha=0.5))
    
    for i, filter in enumerate(filters):
        if filter in ims:
            zp_i = utils.calc_header_zeropoint(ims[filter], ext=0)
            scl = 10**(-0.4*(zp_i-5-scale_ab))
            img = ims[filter][0].data*scl
            image = make_lupton_rgb(img, img, img, stretch=0.1, minimum=-0.01)
            
            ax = fig.add_subplot(1,NX,i+1)
            ax.imshow(255-image, origin='lower', interpolation='nearest')
            
            ax.text(0.5, 0.95, filter, ha='center', va='top', transform=ax.transAxes, fontsize=7, bbox=dict(facecolor='w', edgecolor='None', alpha=0.5))
    
    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.tight_layout(pad=0.1)
    
    fig.savefig('{0}.thumb.png'.format(label))
    if close:
        plt.close()
        
def iyj():

    flux = 'corr'
    err = 'ecorr'
    ap = 0
    
    iflux = np.maximum(self.cat['f814w_{0}_{1}'.format(flux, ap)], self.cat['f814w_{0}_{1}'.format(err, ap)])

    iSN = self.cat['f814w_{0}_{1}'.format(flux, ap)]/self.cat['f814w_{0}_{1}'.format(flux, ap)]

    yflux = self.cat['f105w_{0}_{1}'.format(flux, ap)]
    jflux = self.cat['f125w_{0}_{1}'.format(flux, ap)]
    
    iy = -2.5*np.log10(iflux/yflux)
    yj = -2.5*np.log10(yflux/jflux)
    
    sel = (hmag > 22) & (self.cat['f814w_{0}_{1}'.format(flux, ap)] > -90)
    qso = self.cat['id'] == 1072
    plt.scatter(yj[sel], iy[sel], marker='.', color='k', alpha=0.5)
    plt.scatter(yj[highz_sel], iy[highz_sel], marker='s', color='r', alpha=0.5)
    plt.scatter(yj[qso], iy[qso], marker='*', color='r')
    
    red = (yj > 0.5) & (iy > 1.2) & (self.cat['f814w_{0}_{1}'.format(flux, ap)] > -90) & (iSN < 2)

    red = (yj > -0.2) & (iy > 1.9) & (self.cat['f814w_{0}_{1}'.format(flux, ap)] > -90) & (iSN < 2)

    plt.scatter(yj[red], iy[red], marker='.', color='pink')
    
    tf = self.tempfilt.tempfilt
    tiy = -2.5*np.log10(tf[:,:,2]/tf[:,:,0])
    tyj = -2.5*np.log10(tf[:,:,0]/tf[:,:,1])
    lowz = self.zgrid < 6
    plt.plot(tyj[lowz,:], tiy[lowz,:], alpha=0.3, color='orange')
    plt.plot(tyj[~lowz,:], tiy[~lowz,:], alpha=0.4, color='r')
    plt.xlim(-1,3); plt.ylim(-1,10); plt.grid()
    