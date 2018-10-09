def eazy_photoz(root, force=False, object_only=True, apply_background=True, aper_ix=1, apply_prior=True, beta_prior=True, get_external_photometry=True, external_limits=3, external_sys_err=0.3, external_timeout=300, sys_err=0.05, z_step=0.01, z_min=0.01, z_max=12, total_flux='flux'):
    
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
        
    trans = {'f098m':201, 'f105w':202, 'f110w':241, 'f125w':203, 'f140w':204, 'f160w':205, 'f435w':233, 'f438w':211, 'f606w':236, 'f625w':237, 'f814w':239, 'f702w':15, 'f555w':235, 'f350lp':339}
    
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
        
    # Total flux
    cat.meta['TOTALCOL'] = total_flux, 'Column for total flux'
    
    apcorr = {}
    for i in range(5):
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
                
    cat.rename_column('number','id')
    cat['z_spec'] = cat['id']*0.-1
    
    # Spurious sources, sklearn SVM model trained for a single field
    morph_model = os.path.join(os.path.dirname(utils.__file__),
                               'data/sep_catalog_junk.pkl')
                               
    if os.path.exists(morph_model):
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
        
    cat.write('{0}_phot_apcorr.fits'.format(root), overwrite=True)
    
    # Translate
    fp = open('zphot.translate','w')
    for f in filters:
        if f in trans:
            fp.write('{0}_corr_{1} F{2}\n'.format(f, aper_ix, trans[f]))
            fp.write('{0}_ecorr_{1} E{2}\n'.format(f, aper_ix, trans[f]))
            
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
            params['PRIOR_FILTER'] = trans[f]
            mag = 23.9-2.5*np.log10(cat['{0}_corr_{1}'.format(f, aper_ix)])
            break
    #
    params['TEMPLATES_FILE'] = 'templates/fsps_full/tweak_fsps_QSF_12_v3.param'

    zpfile = None
    load_products = False

    eazy.symlink_eazy_inputs(path='/usr/local/share/python/eazy-py/eazy-photoz', path_is_env=False)
    
    self = eazy.photoz.PhotoZ(param_file=None, translate_file='zphot.translate', zeropoint_file=zpfile, params=params, load_prior=True, load_products=load_products)
    
    if object_only:
        return self
        
    idx = np.arange(self.NOBJ)
    
    #sample = (mag < 27) #& (self.cat['star_flag'] != 1)
    #sample |= (self.cat['z_spec'] > 0)
    sample = np.isfinite(mag)
    
    for iter in range(2):
        self.fit_parallel(idx[sample], n_proc=10)
        self.error_residuals()
    
    self.standard_output(prior=apply_prior, beta_prior=beta_prior)
    
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
    def __init__(self, photoz, grizli_templates=None, zgrid=None):
        """
        photoz : `~eazypy.photoz.PhotoZ`
        """
        try:
            from .. import utils
        except:
            from grizli import utils
            
        not_obs_mask =  (photoz.fnu < -90) | (photoz.efnu < 0)
        
        self.zgrid = photoz.zgrid
        
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
        phot['flam'] = self.flam[ix[0],:]*1.e-19
        phot['eflam'] = self.eflam[ix[0],:]*1.e-19
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
    
        