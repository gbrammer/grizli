def eazy_photoz(root, force=False, object_only=True):
    
    import os
    import eazy
    import numpy as np
    from grizli import utils
    
    if (os.path.exists('{0}.eazypy.self.npy'.format(root))) & (not force):
        self = np.load('{0}.eazypy.self.npy'.format(root))[0]
        zout = utils.read_catalog('{0}.eazypy.zout.fits'.format(root))
        cat = utils.read_catalog('{0}_phot_apcorr.fits'.format(root))
        return self, cat, zout
        
    trans = {'f105w':202, 'f110w':241, 'f125w':203, 'f140w':204, 'f160w':205, 'f435w':233, 'f606w':236, 'f814w':239, 'f702w':15, 'f555w':235}
    
    cat = utils.read_catalog('{0}_phot.fits'.format(root))
    filters = []
    for c in cat.meta:
        if c.endswith('_ZP'):
            filters.append(c.split('_ZP')[0].lower())
    
    # Total flux
    apcorr = {}
    for i in range(5):
        if 'flux_aper_{0}'.format(i) in cat.colnames:
            cat['apcorr_{0}'.format(i)] = cat['flux_auto']/cat['flux_aper_{0}'.format(i)]
            for f in filters:
                cat['{0}_corr_{1}'.format(f, i)] = cat['{0}_flux_aper_{1}'.format(f, i)]*cat['apcorr_{0}'.format(i)]
                cat['{0}_ecorr_{1}'.format(f, i)] = cat['{0}_fluxerr_aper_{1}'.format(f, i)]*cat['apcorr_{0}'.format(i)]

                bad = cat['{0}_mask_aper_{1}'.format(f, i)] > 0.2*np.percentile(cat['{0}_mask_aper_{1}'.format(f, i)], 95)
                cat['{0}_corr_{1}'.format(f, i)][bad] = -99
                cat['{0}_ecorr_{1}'.format(f, i)][bad] = -99
                
    cat.rename_column('number','id')
    cat['z_spec'] = cat['id']*0.-1
    cat.write('{0}_phot_apcorr.fits'.format(root), overwrite=True)
    
    # Translate
    ix = 0
    fp = open('zphot.translate','w')
    for f in filters:
        if f in trans:
            fp.write('{0}_corr_{1} F{2}\n'.format(f, ix, trans[f]))
            fp.write('{0}_ecorr_{1} E{2}\n'.format(f, ix, trans[f]))
    fp.close()
    
    params = {}
    params['CATALOG_FILE'] = '{0}_phot_apcorr.fits'.format(root)
    params['Z_STEP'] = 0.01
    params['MAIN_OUTPUT_FILE'] = '{0}.eazypy'.format(root)
    
    params['Z_MAX'] = 12
    params['MW_EBV'] = 0.
    params['PRIOR_ABZP'] = 23.9
    
    # Pick prior filter, starting from reddest
    for f in ['f435w', 'f606w', 'f814w', 'f105w', 'f110w', 'f125w', 'f140w', 'f160w'][::-1]:
        if f in filters:
            params['PRIOR_FILTER'] = trans[f]
            mag = 23.9-2.5*np.log10(cat['{0}_corr_{1}'.format(f, ix)])
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
    
    self.standard_output()
    zout = utils.read_catalog('{0}.eazypy.zout.fits'.format(root))
    
    np.save('{0}.eazypy.self.npy'.format(root), [self])
    
    return self, cat, zout
    
def show_from_ds9(ds9, self, zout, **kwargs):
    
    xy = np.cast[float](ds9.get('pan image').split())
    r = np.sqrt((self.cat['x_image']-xy[0])**2 + (self.cat['y_image']-xy[1])**2)
    
    ix = np.argmin(r)
    print('ID: {0}, r={1:.1f} pix'.format(self.cat['id'][ix], r[ix]))
    print('  z={0:.2f} logM={1:.2f}'.format(zout['z_phot'][ix], np.log10(zout['mass'][ix])))
    
    self.show_fit(self.cat['id'][ix], **kwargs)
    
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
        
        