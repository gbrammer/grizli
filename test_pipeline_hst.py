import unittest

import os
import glob
import yaml

import numpy as np
import matplotlib.pyplot as plt

import mastquery

from grizli import utils, prep, multifit, GRIZLI_PATH
from grizli.pipeline import auto_script

TEST_HOME = os.getcwd()

HOME_PATH = f'{os.getcwd()}/PipelineTest'
if not os.path.exists(HOME_PATH):
    os.mkdir(HOME_PATH)
    
os.chdir(HOME_PATH)

root = ''
kwargs = ''
visits = None
groups = None
info = None
visit_prep_args = None
grp = None

def test_config():
    """
    Fetch config files if CONF not found
    """
    new = []
    for subd in ['iref','jref','CONF']:
        conf_path = os.path.join(GRIZLI_PATH, subd)
        if not os.path.exists(conf_path):
            new.append(subd)
            os.mkdir(conf_path)
    
    if 'CONF' in new:
        print(f'Download config and calib files to {conf_path}')
        utils.fetch_default_calibs(get_acs=False)
        utils.fetch_config_files(get_epsf=True)
        files = glob.glob(f'{conf_path}/*')
        print('Files: ', '\n'.join(files))

    assert(os.path.exists(os.path.join(conf_path,
                          'G141.F140W.V4.32.conf')))


def test_query():
    """
    """
    from mastquery import query, overlaps
    global root
    
    # "parent" query is grism exposures in GO-11359.  Can also query the archive on position with
    # box=[ra, dec, radius_in_arcmin]
    parent = query.run_query(box=None, proposal_id=[11359], 
                             instruments=['WFC3/IR', 'ACS/WFC'], 
                             filters=['G102','G141'])

    # ### "overlap" query finds anything that overlaps with the exposures 
    # ### in the parent query
    # extra = query.DEFAULT_EXTRA # ignore calibrations, etc.

    # ## To match *just* the grism visits, add, e.g., the following:
    # extra += ["TARGET.TARGET_NAME LIKE 'WFC3-ERSII-G01'"]

    tabs = overlaps.find_overlaps(parent, buffer_arcmin=0.01,
                                  filters=['F140W','G141'], 
                                  proposal_id=[11359], 
                           instruments=['WFC3/IR','WFC3/UVIS','ACS/WFC'],
                                  extra={'target_name':'WFC3-ERSII-G01'},
                                  close=False)
    
    root = tabs[0].meta['NAME']


def test_set_kwargs():
    """
    """
    global kwargs
    
    from grizli.pipeline.auto_script import get_yml_parameters
    kwargs = get_yml_parameters()
    kwargs['is_parallel_field'] = False
    kwargs['fetch_files_args']['reprocess_clean_darks'] = False 
    kwargs['parse_visits_args']['combine_same_pa'] = False


def test_fetch_files():
    """
    """
    auto_script.fetch_files(field_root=root, HOME_PATH=HOME_PATH, 
                            **kwargs['fetch_files_args'])
    
    assert(len(glob.glob(f'{HOME_PATH}/{root}/RAW/*raw.fits')) == 8)
    assert(len(glob.glob(f'{HOME_PATH}/{root}/RAW/*flt.fits')) == 8)


def test_parse_visits():
    """
    """
    global visits, groups, info
    os.chdir(os.path.join(HOME_PATH, root, 'Prep'))
    visits, groups, info = auto_script.parse_visits(field_root=root,
                                                **kwargs['parse_visits_args'])
    
    assert(len(visits) == 2)


def test_preprocess():
    """
    """
    global kwargs, visit_prep_args
    
    visit_prep_args = kwargs['visit_prep_args']
    preprocess_args = kwargs['preprocess_args']

    # Maximum shift for "tweakshifts" relative alignment
    tweak_max_dist = 1.
    if 'tweak_max_dist' not in visit_prep_args:
        visit_prep_args['tweak_max_dist'] = tweak_max_dist

    # Fit and subtract a SExtractor-like background to each visit
    visit_prep_args['imaging_bkg_params']  = {'bh': 256, 'bw': 256,
                                              'fh': 3, 'fw': 3, 
                                              'pixel_scale': 0.1,
                                              'get_median': False}

    # Alignment reference catalogs, searched in this order
    visit_prep_args['reference_catalogs'] = ['LS_DR9', 'PS1', 'GAIA',
                                             'SDSS','WISE']

    os.chdir(os.path.join(HOME_PATH, root, 'Prep'))
    auto_script.preprocess(field_root=root, HOME_PATH=HOME_PATH,
                           visit_prep_args=visit_prep_args, **preprocess_args)
    
    assert(os.path.exists('wfc3-ersii-g01-b6o-23-119.0-f140w_drz_sci.fits'))
    assert(os.path.exists('wfc3-ersii-g01-b6o-23-119.0-f140w_shifts.log'))
    assert(os.path.exists('wfc3-ersii-g01-b6o-23-119.0-f140w_wcs.log'))


# def test_fine_alignment():
#     """
#     """
#     global kwargs
#     fine_alignment_args = kwargs['fine_alignment_args']
#     
#     # Align to GAIA with proper motions evaluated at 
#     # each separate visit execution epoch
#     fine_alignment_args['catalogs'] = ['GAIA']
#     fine_alignment_args['gaia_by_date'] = True
#     
#     os.chdir(os.path.join(HOME_PATH, root, 'Prep'))
#     
#     out = auto_script.fine_alignment(field_root=root, HOME_PATH=HOME_PATH, 
#                                      **fine_alignment_args)
#     
#     visit_file = auto_script.find_visit_file(root=root)
#     print('Update exposure footprints in {0}'.format(visit_file))
#     res = auto_script.get_visit_exposure_footprints(root=root, 
#                                                 check_paths=['./', '../RAW'])


def test_make_mosaics():
    """
    """
    global visits, groups, info
    
    # Drizzle mosaics in each filter and combine all IR filters
    
    mosaic_args = kwargs['mosaic_args']
    mosaic_pixfrac = mosaic_args['mosaic_pixfrac']

    os.chdir(os.path.join(HOME_PATH, root, 'Prep'))

    preprocess_args = kwargs['preprocess_args']

    combine_all_filters=True
    
    if len(glob.glob('{0}-ir_dr?_sci.fits'.format(root))) == 0:

        ## Mosaic WCS
        wcs_ref_file = '{0}_wcs-ref.fits'.format(root)
        if not os.path.exists(wcs_ref_file):
            auto_script.make_reference_wcs(info, output=wcs_ref_file,
                                           get_hdu=True, 
                                           **mosaic_args['wcs_params'])


        if combine_all_filters:
            all_filters = mosaic_args['ir_filters']
            all_filters += mosaic_args['optical_filters']
            auto_script.drizzle_overlaps(root, 
                                     filters=all_filters, 
                                     min_nexp=1, pixfrac=mosaic_pixfrac,
                                     make_combined=True,
                                     ref_image=wcs_ref_file,
                                     drizzle_filters=False) 


        ## IR filters
        if 'fix_stars' in visit_prep_args:
            fix_stars = visit_prep_args['fix_stars']
        else:
            fix_stars = False

        auto_script.drizzle_overlaps(root, filters=mosaic_args['ir_filters'], 
                                     min_nexp=1, pixfrac=mosaic_pixfrac,
                                     make_combined=(not combine_all_filters),
                                     ref_image=wcs_ref_file,
                                     include_saturated=fix_stars) 

        ## Mask diffraction spikes
        mask_spikes=True

        ir_mosaics = glob.glob('{0}-f*drz_sci.fits'.format(root))
        if (len(ir_mosaics) > 0) & (mask_spikes):
            cat = prep.make_SEP_catalog('{0}-ir'.format(root), threshold=4, 
                                        save_fits=False, 
                                        column_case=str.lower)

            selection = (cat['mag_auto'] < 17) & (cat['flux_radius'] < 4.5)
            for visit in visits:
                filt = visit['product'].split('-')[-1]
                if filt[:2] in ['f0','f1']:
                    auto_script.mask_IR_psf_spikes(visit=visit, 
                                       selection=selection,
                                       cat=cat, minR=5, dy=5)

            ## Remake mosaics
            auto_script.drizzle_overlaps(root, 
                                         filters=mosaic_args['ir_filters'], 
                                         min_nexp=1, pixfrac=mosaic_pixfrac,
                                    make_combined=(not combine_all_filters),
                                         ref_image=wcs_ref_file,
                                         include_saturated=True) 

        # Fill IR filter mosaics with scaled combined data so they can be used 
        # as grism reference
        fill_mosaics = mosaic_args['fill_mosaics']
        if fill_mosaics:
            if fill_mosaics == 'grism':
                # Only fill mosaics if grism filters exist
                has_grism = utils.column_string_operation(info['FILTER'], 
                                         ['G141','G102','G800L'],
                                         'count', 'or').sum() > 0
                if has_grism:
                    auto_script.fill_filter_mosaics(root)                                             
            else:
                auto_script.fill_filter_mosaics(root)

        mosaics = glob.glob('{0}-ir_dr?_sci.fits'.format(root))
        wcs_ref_optical = wcs_ref_file

        auto_script.drizzle_overlaps(root, 
                filters=mosaic_args['optical_filters'], 
                pixfrac=mosaic_pixfrac,
                make_combined=(len(mosaics) == 0), ref_image=wcs_ref_optical,
                min_nexp=1+preprocess_args['skip_single_optical_visits']*1)
    
    assert(os.path.exists('j033216m2743-ir_drz_sci.fits'))
    assert(os.path.exists('j033216m2743-f140w_drz_sci.fits'))
    
    if not os.path.exists('{0}.field.jpg'.format(root)):
        slx, sly, rgb_filts, fig = auto_script.field_rgb(root=root, scl=3, 
                                                         HOME_PATH=None)  
        plt.close(fig)


def test_make_phot():
    """
    """
    os.chdir(os.path.join(HOME_PATH, root, 'Prep'))
    
    if not os.path.exists(f'{root}_phot.fits'):
        multiband_catalog_args=kwargs['multiband_catalog_args']
        tab = auto_script.multiband_catalog(field_root=root,
                                        **multiband_catalog_args)
                     
    assert(os.path.exists(f'{root}_phot.fits'))
    assert(os.path.exists(f'{root}-ir.cat.fits'))


def test_make_contam_model():
    """
    """
    global grp

    os.chdir(os.path.join(HOME_PATH, root, 'Prep'))
    files = glob.glob('*GrismFLT.fits')

    if len(files) == 0:
        ### Grism contamination model

        # Which filter to use as direct image
        # Will try in order of the list until a match is found.
        grism_prep_args = kwargs['grism_prep_args']
        grism_prep_args['gris_ref_filters'] = {'G141': ['F140W', 'F160W'], 
                                        'G102': ['F105W', 'F098M', 'F110W']}

        grp = auto_script.grism_prep(field_root=root, **grism_prep_args)

        grp = multifit.GroupFLT(grism_files=glob.glob('*GrismFLT.fits'), 
                                catalog=f'{root}-ir.cat.fits', 
                                cpu_count=-1, sci_extn=1, pad=256)

    else:
        grp = multifit.GroupFLT(grism_files=glob.glob('*GrismFLT.fits'), 
                                catalog=f'{root}-ir.cat.fits', 
                                cpu_count=-1, sci_extn=1, pad=256)


def test_make_field_psf():
    """
    """
    # Make PSF file
    os.chdir(os.path.join(HOME_PATH, root, 'Prep'))

    if not os.path.exists('{0}-f140w_psf.fits'.format(root)):
        auto_script.field_psf(root=root)


def test_extract_and_fit():
    """
    """
    global grp
    
    import astropy.units as u
    from grizli import fitting
    
    pline = auto_script.DITHERED_PLINE

    os.chdir(os.path.join(HOME_PATH, root, 'Extractions'))

    # Generate the parameter dictionary
    args = auto_script.generate_fit_params(field_root=root, 
                                           prior=None, 
                                           MW_EBV=0.005, 
                                           pline=pline, 
                                           fit_only_beams=True, 
                                           run_fit=True, 
                                           poly_order=7, 
                                           fsps=True, 
                                           sys_err = 0.03, 
                                           fcontam=0.2, 
                                           zr=[0.05, 3.4], 
                                           save_file='fit_args.npy')
    
    tab = utils.GTable()
    tab['ra'] = [53.0657456, 53.0624459]
    tab['dec'] = [-27.720518, -27.707018]

    idx, dr = grp.catalog.match_to_catalog_sky(tab)
    assert(np.allclose(dr.value, 0, atol=0.2))
    source_ids = grp.catalog['NUMBER'][idx]
    
    # Cutouts
    drizzler_args = kwargs['drizzler_args']
    print(yaml.dump(drizzler_args))

    os.chdir('../Prep')
    auto_script.make_rgb_thumbnails(root=root, ids=source_ids,
                                    drizzler_args=drizzler_args)
    
    os.chdir('../Extractions')
    
    id=source_ids[0]
    auto_script.extract(field_root=root, ids=[id], MW_EBV=0.005, 
                        pline=pline, run_fit=False, grp=grp, diff=True)
    
    # Redshift fit
    _res = fitting.run_all_parallel(id)
    
    assert(os.path.exists(f'{root}_{id:05d}.row.fits'))
    row = utils.read_catalog(f'{root}_{id:05d}.row.fits')
    assert(np.allclose(row['z_map'], 1.7397, rtol=1.e-2))


#
# def test_run_full(self):
#     """
#     All in one go
#     """
#     auto_script.go(root=root, HOME_PATH=HOME_PATH, **kwargs)

def test_cleanup():
    """
    """
    pass
        