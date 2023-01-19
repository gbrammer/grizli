"""
Automatic processing scripts for grizli
"""
import os
import inspect
import traceback
import glob
import time
import warnings

import yaml

import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from .. import prep, utils
from .default_params import UV_N_FILTERS, UV_M_FILTERS, UV_W_FILTERS
from .default_params import OPT_N_FILTERS, OPT_M_FILTERS, OPT_W_FILTERS
from .default_params import IR_N_FILTERS, IR_M_FILTERS, IR_W_FILTERS
from .default_params import ALL_IMAGING_FILTERS, VALID_FILTERS
from .default_params import UV_GRISMS, OPT_GRISMS, IR_GRISMS, GRIS_REF_FILTERS

from .default_params import get_yml_parameters, write_params_to_yml

# needed for function definitions
args = get_yml_parameters()

if False:
    np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')

# Only fetch F814W optical data for now
#ONLY_F814W = True
ONLY_F814W = False


def get_extra_data(root='j114936+222414', HOME_PATH='/Volumes/Pegasus/Grizli/Automatic', PERSIST_PATH=None, instruments=['WFC3'], filters=['F160W', 'F140W', 'F098M', 'F105W'], radius=2, run_fetch=True, from_mast=True, reprocess_parallel=True, s3_sync=False):

    import os
    import glob

    import numpy as np

    from hsaquery import query, fetch, fetch_mast
    from hsaquery.fetch import DEFAULT_PRODUCTS

    if PERSIST_PATH is None:
        PERSIST_PATH = os.path.join(HOME_PATH, root, 'Persistence')

    tab = utils.GTable.gread(os.path.join(HOME_PATH, 
                             f'{root}_footprint.fits'))

    # Fix CLEAR filter names
    for i, filt_i in enumerate(tab['filter']):
        if 'clear' in filt_i.lower():
            spl = filt_i.lower().split(';')
            if len(spl) > 1:
                for s in spl:
                    if 'clear' not in s:
                        #print(filt_i, s)
                        filt_i = s.upper()
                        break

            tab['filter'][i] = filt_i.upper()

    ra, dec = tab.meta['RA'], tab.meta['DEC']

    fp = np.load(os.path.join(HOME_PATH, '{0}_footprint.npy'.format(root)),
                 allow_pickle=True)[0]

    radius = np.sqrt(fp.area*np.cos(dec/180*np.pi))*60/np.pi

    xy = np.array(fp.boundary.convex_hull.boundary.xy)
    dims = np.array([(xy[0].max()-xy[0].min())*np.cos(dec/180*np.pi),
                      xy[1].max()-xy[1].min()])*60

    extra = query.run_query(box=[ra, dec, radius],
                            proposid=[],
                            instruments=instruments,
                            extensions=['FLT'], 
                            filters=filters,
                            extra=query.DEFAULT_EXTRA)

    # Fix CLEAR filter names
    for i, filt_i in enumerate(extra['filter']):
        if 'clear' in filt_i.lower():
            spl = filt_i.lower().split(';')
            if len(spl) > 1:
                for s in spl:
                    if 'clear' not in s:
                        #print(filt_i, s)
                        filt_i = s.upper()
                        break

            extra['filter'][i] = filt_i.upper()

    for k in tab.meta:
        extra.meta[k] = tab.meta[k]

    extra.write(os.path.join(HOME_PATH, root, 'extra_data.fits'), 
                format='fits', overwrite=True)

    CWD = os.getcwd()
    os.chdir(os.path.join(HOME_PATH, root, 'RAW'))

    if run_fetch:
        if from_mast:
            out = fetch_mast.get_from_MAST(extra, 
                            inst_products=DEFAULT_PRODUCTS, 
                            direct=True, 
                            path=os.path.join(HOME_PATH, root, 'RAW'), 
                            skip_existing=True)
        else:

            curl = fetch.make_curl_script(extra,
                            level=None, 
                            script_name='extra.sh',
                            inst_products={'WFC3/UVIS': ['FLC'], 
                                           'WFPC2/PC': ['C0M', 'C1M'],
                                           'WFC3/IR': ['RAW'],
                                           'ACS/WFC': ['FLC']}, 
                            skip_existing=True,
                            output_path=os.path.join(HOME_PATH, root, 'RAW'),
                            s3_sync=s3_sync)

            os.system('sh extra.sh')
            files = glob.glob('*raw.fits.gz')
            files.extend(glob.glob('*fl?.fits.gz'))
            for file in files:
                print('gunzip '+file)
                os.system('gunzip {0}'.format(file))

    else:
        return extra

    remove_bad_expflag(field_root=root, HOME_PATH=HOME_PATH, min_bad=2)

    # Reprocess the RAWs into FLTs
    status = os.system("python -c 'from grizli.pipeline import reprocess; reprocess.reprocess_wfc3ir(parallel={0})'".format(reprocess_parallel))
    if status != 0:
        from grizli.pipeline import reprocess
        reprocess.reprocess_wfc3ir(parallel=False)

    # Persistence products
    os.chdir(PERSIST_PATH)
    persist_files = fetch.persistence_products(extra)
    for file in persist_files:
        if not os.path.exists(os.path.basename(file)):
            print(file)
            os.system('curl -O {0}'.format(file))

    for file in persist_files:
        root = os.path.basename(file).split('.tar.gz')[0]
        if os.path.exists(root):
            print('Skip', root)
            continue
        
        if not os.path.exists(file):
            print('Persistence tar file {0} not found'.format(file))
            continue
            
        # Ugly callout to shell
        os.system('tar xzvf {0}.tar.gz'.format(root))
        
        # Clean unneeded files
        clean_files = glob.glob('{0}/*extper.fits'.format(root))
        clean_files += glob.glob('{0}/*flt_cor.fits'.format(root))
        for f in clean_files:
            os.remove(f)
        
        # Symlink to ./
        pfiles = glob.glob('{0}/*persist.fits'.format(root))
        if len(pfiles) > 0:
            for f in pfiles:
                if not os.path.exists(os.path.basename(f)):
                    os.system('ln -sf {0} ./'.format(f))

    os.chdir(CWD)


def create_path_dict(root='j142724+334246', home='$PWD', raw=None, prep=None, extract=None, persist=None, thumbs=None, paths={}):
    """
    Generate path dict.
    
    Default:
        {home}
        {home}/{root}
        {home}/{root}/RAW
        {home}/{root}/Prep
        {home}/{root}/Persistence
        {home}/{root}/Extractions
        {home}/{root}/Thumbnails
    
    If ``home`` specified as '$PWD', then will be calculated from 
    `os.getcwd`.
    
    Only generates values for keys not already specified in `paths`.
    
    """
    import copy
    
    if home == '$PWD':
        home = os.getcwd()
    
    base = os.path.join(home, root)
    
    if raw is None:
        raw = os.path.join(home, root, 'RAW')

    if prep is None:
        prep = os.path.join(home, root, 'Prep')
    
    if persist is None:
        persist = os.path.join(home, root, 'Persistence')

    if extract is None:
        extract = os.path.join(home, root, 'Extractions')

    if thumbs is None:
        thumbs = os.path.join(home, root, 'Thumbnaails')
    
    xpaths = copy.deepcopy(paths)
    for k in xpaths:
        if xpaths[k] is None:
            _ = xpaths.pop(k)
            
    if 'home' not in xpaths:
        xpaths['home'] = home
    
    if 'base' not in xpaths:
        xpaths['base'] = base
        
    if 'raw' not in xpaths:
        xpaths['raw'] = raw

    if 'prep' not in xpaths:
        xpaths['prep'] = prep

    if 'persist' not in xpaths:
        xpaths['persist'] = persist

    if 'extract' not in xpaths:
        xpaths['extract'] = extract
    
    if 'thumbs' not in xpaths:
        xpaths['thumbs'] = extract
        
    return xpaths


def go(root='j010311+131615', 
       HOME_PATH='$PWD',
       RAW_PATH=None, PREP_PATH=None, PERSIST_PATH=None, EXTRACT_PATH=None,
       CRDS_CONTEXT=None,
       filters=args['filters'],
       fetch_files_args=args['fetch_files_args'],
       inspect_ramps=False,
       is_dash=False, run_prepare_dash=True,
       run_parse_visits=True,
       is_parallel_field=False,
       parse_visits_args=args['parse_visits_args'],
       manual_alignment=False,
       manual_alignment_args=args['manual_alignment_args'],
       preprocess_args=args['preprocess_args'],
       visit_prep_args=args['visit_prep_args'],
       persistence_args=args['persistence_args'],
       redo_persistence_mask=False,
       run_fine_alignment=True,
       fine_backup=True,
       fine_alignment_args=args['fine_alignment_args'],
       make_mosaics=True,
       mosaic_args=args['mosaic_args'],
       mosaic_drizzle_args=args['mosaic_drizzle_args'],
       mask_spikes=False,
       mosaic_driz_cr_type=0,
       make_phot=True,
       multiband_catalog_args=args['multiband_catalog_args'],
       only_preprocess=False,
       overwrite_fit_params=False,
       grism_prep_args=args['grism_prep_args'],
       refine_with_fits=True,
       run_extractions=False,
       include_photometry_in_fit=False,
       extract_args=args['extract_args'],
       make_thumbnails=True,
       thumbnail_args=args['thumbnail_args'],
       make_final_report=True,
       get_dict=False,
       kill='',
       **kwargs
       ):
    """
    Run the full pipeline for a given target

    Parameters
    ----------
    root : str
        Rootname of the `mastquery` file.

    extract_maglim : [min, max]
        Magnitude limits of objects to extract and fit.

    """
    #isJWST = visit_prep_args['isJWST']
    # Function defaults
    if get_dict:
        if get_dict <= 2:
            # Default function arguments (different value to avoid recursion)
            default_args = go(get_dict=10)

        frame = inspect.currentframe()
        args = inspect.getargvalues(frame).locals
        for k in ['root', 'HOME_PATH', 'frame', 'get_dict']:
            if k in args:
                args.pop(k)

        if get_dict == 2:
            # Print keywords summary
            if len(kwargs) > 0:
                print('\n*** Extra args ***\n')
                for k in kwargs:
                    if k not in default_args:
                        print('\'{0}\':{1},'.format(k, kwargs[k]))

            print('\n*** User args ***\n')
            for k in args:
                if k in default_args:
                    if args[k] != default_args[k]:
                        print('\'{0}\':{1},'.format(k, args[k]))

            print('\n*** Default args ***\n')
            for k in args:
                if k in default_args:
                    print('\'{0}\':{1},'.format(k, args[k]))
            return args
        else:
            return args

    # import os
    # import glob
    # import traceback
    #
    #
    try:
        from .. import multifit
        from . import auto_script
    except:
        from grizli import multifit
        from grizli.pipeline import auto_script

    # #import grizli.utils
    import matplotlib.pyplot as plt

    # Silence numpy and astropy warnings
    utils.set_warnings()
    
    PATHS = create_path_dict(root=root, home=HOME_PATH, 
                             raw=RAW_PATH, prep=PREP_PATH, 
                             persist=PERSIST_PATH, extract=EXTRACT_PATH)
        
    fpfile = os.path.join(PATHS['home'], '{0}_footprint.fits'.format(root))
    exptab = utils.GTable.gread(fpfile)
    # Fix CLEAR filter names
    for i, filt_i in enumerate(exptab['filter']):
        if 'clear' in filt_i.lower():
            spl = filt_i.lower().split(';')
            if len(spl) > 1:
                for s in spl:
                    if 'clear' not in s:
                        #print(filt_i, s)
                        filt_i = s.upper()
                        break

            exptab['filter'][i] = filt_i.upper()

    utils.LOGFILE = os.path.join(PATHS['home'], f'{root}.auto_script.log.txt')

    utils.log_comment(utils.LOGFILE, '### Pipeline start', show_date=True)
    
    if CRDS_CONTEXT is not None:
        utils.log_comment(utils.LOGFILE,
                          f"# export CRDS_CONTEXT='{CRDS_CONTEXT}'",
                          show_date=True)
                          
        os.environ['CRDS_CONTEXT'] = CRDS_CONTEXT
        
    ######################
    # Download data
    os.chdir(PATHS['home'])
    if fetch_files_args is not None:
        fetch_files_args['reprocess_clean_darks'] &= (not is_dash)
        auto_script.fetch_files(field_root=root, HOME_PATH=HOME_PATH,
                                paths=PATHS, 
                                filters=filters, **fetch_files_args)
    else:
        os.chdir(PATHS['prep'])

    if is_dash & run_prepare_dash:
        from wfc3dash import process_raw
        os.chdir(PATHS['raw'])
        process_raw.run_all()

    files = glob.glob(os.path.join(PATHS['raw'], '*_fl*fits'))
    files += glob.glob(os.path.join(PATHS['raw'], '*_c[01]m.fits'))
    files += glob.glob(os.path.join(PATHS['raw'], '*_rate.fits'))
    files += glob.glob(os.path.join(PATHS['raw'], '*_cal.fits'))
    
    if len(files) == 0:
        print('No FL[TC] files found!')
        utils.LOGFILE = '/tmp/grizli.log'
        return False

    if kill == 'fetch_files':
        print('kill=\'fetch_files\'')
        return True

    if inspect_ramps:
        # Inspect for CR trails
        os.chdir(PATHS['raw'])
        status = os.system("python -c 'from grizli.pipeline.reprocess import inspect; inspect()'")

    ######################
    # Parse visit associations
    os.chdir(PATHS['prep'])
    visit_file = auto_script.find_visit_file(root=root)
    
    if (visit_file is None) | run_parse_visits:
        # Parsing for parallel fields, where time-adjacent exposures
        # may have different visit IDs and should be combined
        if 'combine_same_pa' in parse_visits_args:
            if (parse_visits_args['combine_same_pa'] == -1):
                if is_parallel_field:
                    parse_visits_args['combine_same_pa'] = True
                    parse_visits_args['max_dt'] = 4./24
                else:
                    parse_visits_args['combine_same_pa'] = False
                    parse_visits_args['max_dt'] = 1.

        else:
            parse_visits_args['combine_same_pa'] = is_parallel_field

        parsed = auto_script.parse_visits(field_root=root,
                                          RAW_PATH=PATHS['raw'],
                                          filters=filters, is_dash=is_dash,
                                          **parse_visits_args)
    else:
        #parsed = np.load(f'{root}_visits.npy', allow_pickle=True)
        parsed = load_visit_info(root, verbose=False)

    if kill == 'parse_visits':
        print('kill=\'parse_visits\'')
        return True

    visits, all_groups, info = parsed
    run_has_grism = np.in1d(info['FILTER'], ['G141', 'G102', 'G800L', 'GR150C', 'GR150R']).sum() > 0
    # is PUPIL in info?
    run_has_grism |= np.in1d(info['PUPIL'], ['GRISMR', 'GRISMC']).sum() > 0

    # Alignment catalogs
    #catalogs = ['PS1','SDSS','GAIA','WISE']

    #######################
    # Manual alignment
    if manual_alignment:
        os.chdir(PATHS['prep'])
        auto_script.manual_alignment(field_root=root, HOME_PATH=PATHS['home'],
                                     **manual_alignment_args)

    if kill == 'manual_alignment':
        print('kill=\'manual_alignment\'')
        return True

    #####################
    # Alignment & mosaics
    os.chdir(PATHS['prep'])

    tweak_max_dist = (5 if is_parallel_field else 1)
    if 'tweak_max_dist' not in visit_prep_args:
        visit_prep_args['tweak_max_dist'] = tweak_max_dist

    if 'use_self_catalog' not in visit_prep_args:
        visit_prep_args['use_self_catalog'] = is_parallel_field

    auto_script.preprocess(field_root=root, HOME_PATH=PATHS['home'], 
                           PERSIST_PATH=PATHS['persist'],
                           visit_prep_args=visit_prep_args,
                           persistence_args=persistence_args, 
                           **preprocess_args)

    if kill == 'preprocess':
        print('kill=\'preprocess\'')
        
        visit_file = find_visit_file(root=root)
        print(f'Update exposure footprints in {visit_file}')
        check_paths = ['./', PATHS['raw'], '../RAW']
        get_visit_exposure_footprints(root=root, check_paths=check_paths)
        
        return True

    if redo_persistence_mask:
        comment = '# Redo persistence masking: {0}'.format(persistence_args)
        print(comment)
        utils.log_comment(utils.LOGFILE, comment)

        all_flt_files = glob.glob('*_flt.fits')
        all_flt_files.sort()

        for file in all_flt_files:
            print(file)
            pfile = os.path.join(PATHS['persist'], 
                                 file.replace('_flt', '_persist'))
            if os.path.exists(pfile):
                prep.apply_persistence_mask(file, path=PATHS['persist'],
                                            **persistence_args)

    ##########
    # Fine alignment

    fine_files = glob.glob('{0}*fine.png'.format(root))
    if (run_fine_alignment == 2) & (len(fine_files) > 0) & (len(visits) > 1):

        msg = '\n\n### Redo visit-level mosaics and catalogs for fine alignment\n\n'
        utils.log_comment(utils.LOGFILE, msg, show_date=True, verbose=True)

        keep_visits = []

        for visit in visits:

            visit_files = glob.glob(visit['product']+'*.cat.*')
            visit_files += glob.glob(visit['product']+'_dr*')
            visit_files += glob.glob(visit['product']+'*seg.fits*')

            if len(visit_files) > 0:
                keep_visits.append(visit)
                for file in visit_files:
                    os.remove(file)

        # Redrizzle visit-level mosaics and remake catalogs
        prep.drizzle_overlaps(keep_visits, check_overlaps=False, skysub=False,
                              static=False, pixfrac=0.5, scale=None,
                              final_wcs=False, fetch_flats=False,
                              final_rot=None,
                              include_saturated=True)

        # Make new catalogs
        for visit in keep_visits:
            if len(visit['files']) == 0:
                continue

            visit_filter = visit['product'].split('-')[-1]
            is_single = len(visit['files']) == 1
            isACS = '_flc' in visit['files'][0]
            isWFPC2 = '_c0' in visit['files'][0]

            if visit_filter in ['g102', 'g141', 'g800l', 'g280']:
                print('# Skip grism visit: {0}'.format(visit['product']))
                continue

            # New catalog
            if visit_prep_args['align_thresh'] is None:
                thresh = 2.5
            else:
                thresh = visit_prep_args['align_thresh']

            cat = prep.make_SEP_catalog(root=visit['product'],
                                        threshold=thresh)

            # New region file
            prep.table_to_regions(cat, '{0}.cat.reg'.format(visit['product']))

            # New radec
            if not ((isACS | isWFPC2) & is_single):
                # 140 brightest or mag range
                clip = (cat['MAG_AUTO'] > 18) & (cat['MAG_AUTO'] < 23)
                clip &= cat['MAGERR_AUTO'] < 0.05
                clip &= utils.catalog_mask(cat,
                    max_err_percentile=visit_prep_args['max_err_percentile'],
                         pad=visit_prep_args['catalog_mask_pad'],
                         pad_is_absolute=False, min_flux_radius=1.)

                NMAX = 140
                so = np.argsort(cat['MAG_AUTO'][clip])
                if clip.sum() > NMAX:
                    so = so[:NMAX]

                prep.table_to_radec(cat[clip][so],
                                    '{0}.cat.radec'.format(visit['product']))

        for file in fine_files:
            print('rm {0}'.format(file))
            os.remove(file)

        fine_files = []

    if (len(fine_files) == 0) & (run_fine_alignment > 0) & (len(visits) > 1):
        fine_catalogs = ['GAIA', 'PS1', 'DES', 'SDSS', 'WISE']
        try:
            out = auto_script.fine_alignment(field_root=root, 
                                             HOME_PATH=PATHS['home'], 
                                             **fine_alignment_args)

            plt.close()

            # Update WCS headers with fine alignment
            auto_script.update_wcs_headers_with_fine(root, backup=fine_backup)

        except:
            utils.log_exception(utils.LOGFILE, traceback)
            utils.log_comment(utils.LOGFILE, "# !! Fine alignment failed")

    # Update the visits file with the new exposure footprints
    visit_file = auto_script.find_visit_file(root=root)
    print('Update exposure footprints in {0}'.format(visit_file))
    check_paths = ['./', PATHS['raw'], '../RAW']
    get_visit_exposure_footprints(root=root, check_paths=check_paths)

    # Make combined mosaics
    no_mosaics_found = len(glob.glob(f'{root}-ir_dr?_sci.fits')) == 0
    if no_mosaics_found & make_mosaics:

        skip_single = preprocess_args['skip_single_optical_visits']

        if 'fix_stars' in visit_prep_args:
            fix_stars = visit_prep_args['fix_stars']
        else:
            fix_stars = False

        # For running at the command line
        # if False:
        #     mos_args = {'mosaic_args': kwargs['mosaic_args'],
        #                 'fix_stars': kwargs['visit_prep_args']['fix_stars'],
        #                 'mask_spikes': kwargs['mask_spikes'], 'skip_single_optical_visits': kwargs['preprocess_args']['skip_single_optical_visits']}
        #     auto_script.make_combined_mosaics(root, **mos_args)

        make_combined_mosaics(root, mosaic_args=mosaic_args,
                        fix_stars=fix_stars, mask_spikes=mask_spikes,
                        skip_single_optical_visits=skip_single,
                        mosaic_driz_cr_type=mosaic_driz_cr_type,
                        mosaic_drizzle_args=mosaic_drizzle_args)

        # Make PSFs.  Always set get_line_maps=False since PSFs now
        # provided for each object.
        mosaic_files = glob.glob('{0}-f*sci.fits'.format(root))

        if (not is_dash) & (len(mosaic_files) > 0):
            print('Make field PSFs')
            auto_script.field_psf(root=root, PREP_PATH=PATHS['prep'],
                                  RAW_PATH=PATHS['raw'], 
                                  EXTRACT_PATH=PATHS['extract'], 
                                  get_line_maps=False, skip=False)

    # Are there full-field mosaics?
    mosaic_files = glob.glob(f'{root}-f*sci.fits')

    # Photometric catalog
    has_phot_file = os.path.exists(f'{root}_phot.fits')
    if (not has_phot_file) & make_phot & (len(mosaic_files) > 0):
        try:
            tab = auto_script.multiband_catalog(field_root=root,
                                                **multiband_catalog_args)

            try:
                # Add columns indicating objects that fall in grism exposures
                phot = utils.read_catalog(f'{root}_phot.fits')
                out = count_grism_exposures(phot, all_groups,
                                      grisms=['g800l', 'g102', 'g141', 'gr150c', 'gr150r'],
                                      verbose=True)
                phot.write(f'{root}_phot.fits', overwrite=True)
            except:
                pass

        except:
            utils.log_exception(utils.LOGFILE, traceback)
            utils.log_comment(utils.LOGFILE,
               '# Run `multiband_catalog` with `detection_background=True`')

            multiband_catalog_args['detection_background'] = True
            tab = auto_script.multiband_catalog(field_root=root,
                                                **multiband_catalog_args)
            #tab = auto_script.multiband_catalog(field_root=root, threshold=threshold, detection_background=True, photometry_background=True, get_all_filters=False)

    # Make exposure json / html report
    auto_script.exposure_report(root, log=True)

    # Stop if only want to run pre-processing
    if (only_preprocess | (len(all_groups) == 0)):
        if make_thumbnails:
            print('#####\n# Make RGB thumbnails\n#####')

            if thumbnail_args['drizzler_args'] is None:
                thumbnail_args['drizzler_args'] = DRIZZLER_ARGS.copy()

            os.chdir(PATHS['prep'])

            #print('XXX ', thumbnail_args)

            auto_script.make_rgb_thumbnails(root=root, **thumbnail_args)

            if not os.path.exists(PATHS['thumbs']):
                os.mkdir(PATHS['thumbs'])

            os.system('mv {0}_[0-9]*.png {0}_[0-9]*.fits {1}'.format(root, 
                                                           PATHS['thumbs']))

        if make_final_report:
            make_report(root, make_rgb=True)
        
        utils.LOGFILE = '/tmp/grizli.log'
        return True

    ######################
    # Grism prep
    files = glob.glob(os.path.join(PATHS['prep'], '*GrismFLT.fits'))
    files += glob.glob(os.path.join(PATHS['extract'], '*GrismFLT.fits'))
    if len(files) == 0:
        os.chdir(PATHS['prep'])
        grp = auto_script.grism_prep(field_root=root, PREP_PATH=PATHS['prep'], 
                                     EXTRACT_PATH=PATHS['extract'], 
                                     **grism_prep_args)
        del(grp)

    ######################
    # Grism extractions
    os.chdir(PATHS['extract'])

    #####################
    # Update the contam model with the "full.fits"
    # files in the working directory
    if (len(glob.glob('*full.fits')) > 0) & (refine_with_fits):
        auto_script.refine_model_with_fits(field_root=root, clean=True,
                                           grp=None, master_files=None,
                                           spectrum='continuum', max_chinu=5)

    # Drizzled grp objects
    # All files
    if len(glob.glob(f'{root}*_grism*fits*')) == 0:
        grism_files = glob.glob('*GrismFLT.fits')
        grism_files.sort()

        catalog = glob.glob(f'{root}-*.cat.fits')[0]
        try:
            seg_file = glob.glob(f'{root}-*_seg.fits')[0]
        except:
            seg_file = None

        grp = multifit.GroupFLT(grism_files=grism_files, direct_files=[], 
                                ref_file=None, seg_file=seg_file, 
                                catalog=catalog, cpu_count=-1, sci_extn=1, 
                                pad=(64,256))

        # Make drizzle model images
        grp.drizzle_grism_models(root=root, kernel='point', scale=0.15)

        # Free grp object
        del(grp)

    if is_parallel_field:
        pline = auto_script.PARALLEL_PLINE.copy()
    else:
        pline = auto_script.DITHERED_PLINE.copy()

    # Make script for parallel processing
    args_file = f'{root}_fit_args.npy'

    if (not os.path.exists(args_file)) | (overwrite_fit_params):
        msg = '# generate_fit_params: ' + args_file
        utils.log_comment(utils.LOGFILE, msg, verbose=True, show_date=True)

        pline['pixscale'] = mosaic_args['wcs_params']['pixel_scale']
        pline['pixfrac'] = mosaic_args['mosaic_pixfrac']
        if pline['pixfrac'] > 0:
            pline['kernel'] = 'square'
        else:
            pline['kernel'] = 'point'

        has_g800l = utils.column_string_operation(info['FILTER'], ['G800L'],
                                                      'count', 'or').sum()

        if has_g800l > 0:
            min_sens = 0.
            fit_trace_shift = True
        else:
            min_sens = 0.001
            fit_trace_shift = True

        try:
            auto_script.generate_fit_params(field_root=root, prior=None, MW_EBV=exptab.meta['MW_EBV'], pline=pline, fit_only_beams=True, run_fit=True, poly_order=7, fsps=True, min_sens=min_sens, sys_err=0.03, fcontam=0.2, zr=[0.05, 3.4], save_file=args_file, fit_trace_shift=fit_trace_shift, include_photometry=True, use_phot_obj=include_photometry_in_fit)
        except:
            # include_photometry failed?
            auto_script.generate_fit_params(field_root=root, prior=None, MW_EBV=exptab.meta['MW_EBV'], pline=pline, fit_only_beams=True, run_fit=True, poly_order=7, fsps=True, min_sens=min_sens, sys_err=0.03, fcontam=0.2, zr=[0.05, 3.4], save_file=args_file, fit_trace_shift=fit_trace_shift, include_photometry=False, use_phot_obj=False)

        # Copy for now
        os.system(f'cp {args_file} fit_args.npy')

    # Done?
    if (not run_extractions) | (run_has_grism == 0):
        # Make RGB thumbnails
        if make_thumbnails:
            print('#####\n# Make RGB thumbnails\n#####')

            if thumbnail_args['drizzler_args'] is None:
                thumbnail_args['drizzler_args'] = DRIZZLER_ARGS.copy()

            os.chdir(PATHS['prep'])

            auto_script.make_rgb_thumbnails(root=root, **thumbnail_args)

            if not os.path.exists(PATHS['thumbs']):
                os.mkdir(PATHS['thumbs'])

            os.system('mv {0}_[0-9]*.png {0}_[0-9]*.fits {1}'.format(root,
                                                            PATHS['thumbs']))

        utils.LOGFILE = '/tmp/grizli.log'
        return True

    # Run extractions (and fits)
    auto_script.extract(field_root=root, **extract_args)

    # Make RGB thumbnails
    if make_thumbnails:
        print('#####\n# Make RGB thumbnails\n#####')

        if thumbnail_args['drizzler_args'] is None:
            thumbnail_args['drizzler_args'] = DRIZZLER_ARGS.copy()

        os.chdir(PATHS['prep'])

        auto_script.make_rgb_thumbnails(root=root, **thumbnail_args)

        if not os.path.exists(PATHS['thumbs']):
            os.mkdir(PATHS['thumbs'])

        os.system('mv {0}_[0-9]*.png {0}_[0-9]*.fits {1}'.format(root, 
                                                        PATHS['thumbs']))

    if extract_args['run_fit']:
        os.chdir(PATHS['extract'])

        # Redrizzle grism models
        grism_files = glob.glob('*GrismFLT.fits')
        grism_files.sort()

        seg_file = glob.glob(f'{root}-[fi]*_seg.fits')[0]
        #catalog = glob.glob(f'{root}-*.cat.fits')[0]
        catalog = seg_file.replace('_seg.fits','.cat.fits')
        
        grp = multifit.GroupFLT(grism_files=grism_files, direct_files=[], 
                                ref_file=None, seg_file=seg_file, 
                                catalog=catalog, cpu_count=-1, sci_extn=1, 
                                pad=(64,256))

        # Make drizzle model images
        grp.drizzle_grism_models(root=root, kernel='point', scale=0.15)

        # Free grp object
        del(grp)

        ######################
        # Summary catalog & webpage
        auto_script.summary_catalog(field_root=root, dzbin=0.01,
                                    use_localhost=False,
                                    filter_bandpasses=None)

    if make_final_report:
        make_report(root, make_rgb=True)


def make_directories(root='j142724+334246', HOME_PATH='$PWD', paths={}):
    """
    Make RAW, Prep, Persistence, Extractions directories
    """
    import os
    
    paths = create_path_dict(root=root, home=HOME_PATH, paths=paths)
    
    for k in paths:
        if k in ['thumbs']:
            continue
            
        dir = paths[k]
        if not os.path.exists(dir):
            print(f'mkdir {dir}')
            os.mkdir(dir)
            os.system(f'chmod ugoa+rwx {dir}')
        else:
            print(f'directory {dir} exists')
            
    return paths


def fetch_files(field_root='j142724+334246', HOME_PATH='$PWD', paths={}, inst_products={'WFPC2/WFC': ['C0M', 'C1M'], 'WFPC2/PC': ['C0M', 'C1M'], 'ACS/WFC': ['FLC'], 'WFC3/IR': ['RAW'], 'WFC3/UVIS': ['FLC']}, remove_bad=True, reprocess_parallel=False, reprocess_clean_darks=True, s3_sync=False, fetch_flt_calibs=['IDCTAB', 'PFLTFILE', 'NPOLFILE'], filters=VALID_FILTERS, min_bad_expflag=2, fetch_only=False, force_rate=True):
    """
    Fully automatic script
    """
    import os
    import glob

    try:
        from .. import utils
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'auto_script.fetch_files')
    except:
        from grizli import utils
    
    try:
        from .. import jwst_utils
    except ImportError:
        jwst_utils = None
        
    try:
        try:
            from mastquery import query, fetch
            import mastquery.utils
            
            MAST_QUERY = True
            instdet_key = 'instrument_name'
        except:
            from hsaquery import query, fetch
            MAST_QUERY = False
            instdet_key = 'instdet'
            
    except ImportError as ERR:
        warn = """{0}

    Get one of the query scripts from
        https://github.com/gbrammer/esa-hsaquery
        https://github.com/gbrammer/mastquery

    """.format(ERR)

        raise(ImportError(warn))
    
    paths = create_path_dict(root=field_root, home=HOME_PATH, paths=paths)
    print('paths: ', paths)
    
    if not os.path.exists(paths['raw']):
        make_directories(root=field_root, HOME_PATH=HOME_PATH, 
                         paths=paths)

    
    tab = utils.read_catalog(os.path.join(paths['home'], 
                                          f'{field_root}_footprint.fits'))

    # Fix CLEAR filter names
    for i, filt_i in enumerate(tab['filter']):
        if 'clear' in filt_i.lower():
            spl = filt_i.lower().split(';')
            if len(spl) > 1:
                for s in spl:
                    if 'clear' not in s:
                        #print(filt_i, s)
                        filt_i = s.upper()
                        break

            tab['filter'][i] = filt_i.upper()

    use_filters = utils.column_string_operation(tab['filter'], filters,
                                            method='startswith', logical='or')
    
    # Allow all JWST for now xxx
    instr = [str(inst) for inst in tab['instrument_name']]
    jw = np.in1d(instr, ['NIRISS','NIRCAM','MIRI','NIRSPEC','FGS'])
    use_filters |= jw
    
    tab = tab[use_filters]
    
    # JWST
    if jw.sum() > 0:
        print(f'Fetch {jw.sum()} JWST files!')
        
        os.chdir(paths['raw'])
            
        if 'dummy' in field_root:
            # Get dummy files copied to AWS
            for f in tab['dataURL'][jw]:
                _file = os.path.basename(f).replace('nis_cal','nis_rate')
                if not os.path.exists(_file):
                    os.system(f'aws s3 cp s3://grizli-v2/JwstDummy/{_file} .')
            
        else:
            ## Download from MAST API when data available xxx            
            _resp = mastquery.utils.download_from_mast(tab[jw], 
                                                       force_rate=force_rate)
            # update targname
            for i, _file in enumerate(_resp):
                _test = os.path.exists(_file) & (jwst_utils is not None)
                _jwst_grism = ('_nis_rate' in _file)
                if 'filter' in tab.colnames:
                    _jwst_grism |= 'GRISM' in tab[jw]['filter'][i]
                    
                _test &= _jwst_grism
                
                if _test:
                    # Need initialization here for JWST grism exposures
                    jwst_utils.initialize_jwst_image(_file,
                                                     oneoverf_correction=False,
                    )
                    jwst_utils.set_jwst_to_hst_keywords(_file, reset=True)
                    
        tab = tab[~jw]
        
    if len(tab) > 0:
        if MAST_QUERY:
            tab = query.get_products_table(tab, extensions=['RAW', 'C1M'])

        tab = tab[(tab['filter'] != 'F218W')]
        if ONLY_F814W:
            tab = tab[(tab['filter'] == 'F814W') |
                      (tab[instdet_key] == 'WFC3/IR')]

        # Fetch and preprocess IR backgrounds
        os.chdir(paths['raw'])

        # Ignore files already moved to RAW/Expflag
        bad_files = glob.glob('./Expflag/*')
        badexp = np.zeros(len(tab), dtype=bool)
        for file in bad_files:
            root = os.path.basename(file).split('_')[0]
            badexp |= tab['observation_id'] == root.lower()

        is_wfpc2 = utils.column_string_operation(tab['instrument_name'],
                                  'WFPC2', method='startswith', logical='or')

        use_filters = utils.column_string_operation(tab['filter'],
                                  filters, method='startswith', logical='or')

        fetch_selection = (~badexp) & (~is_wfpc2) & use_filters
        curl = fetch.make_curl_script(tab[fetch_selection], level=None,
                        script_name='fetch_{0}.sh'.format(field_root),
                        inst_products=inst_products, skip_existing=True,
                        output_path='./', s3_sync=s3_sync)

        msg = 'Fetch {0} files (s3_sync={1})'.format(fetch_selection.sum(),
                                                     s3_sync)
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        # Ugly callout to shell
        os.system('sh fetch_{0}.sh'.format(field_root))

        if (is_wfpc2 & use_filters).sum() > 0:
            # Have to get WFPC2 from ESA
            wfpc2_files = (~badexp) & (is_wfpc2) & use_filters
            curl = fetch.make_curl_script(tab[wfpc2_files], level=None,
                          script_name='fetch_wfpc2_{0}.sh'.format(field_root),
                          inst_products=inst_products, skip_existing=True,
                          output_path='./', s3_sync=False)

            os.system('sh fetch_wfpc2_{0}.sh'.format(field_root))

    else:
        msg = 'Warning: no files to fetch for filters={0}.'.format(filters)
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

    # Gunzip if necessary
    files = glob.glob('*raw.fits.gz')
    files.extend(glob.glob('*fl?.fits.gz'))
    files.extend(glob.glob('*c[01]?.fits.gz'))  # WFPC2
    files.sort()
    
    for file in files:
        status = os.system('gunzip {0}'.format(file))
        print('gunzip '+file+'  # status="{0}"'.format(status))
        if status == 256:
            os.system('mv {0} {1}'.format(file, file.split('.gz')[0]))
                
    if fetch_only:
        files = glob.glob('*raw.fits')
        files.sort()
        
        return files
        
    # Remove exposures with bad EXPFLAG
    if remove_bad:
        remove_bad_expflag(field_root=field_root, HOME_PATH=paths['home'],
                           min_bad=min_bad_expflag)
    
    # EXPTIME = 0
    files = glob.glob('*raw.fits')
    files.extend(glob.glob('*fl?.fits'))
    files.extend(glob.glob('*c[01]?.fits'))  # WFPC2
    files.sort()
    for file in files:
        try:
            im = pyfits.open(file)
            badexp = (im[0].header['EXPTIME'] < 0.1)*1
        except:
            badexp = 2
        
        if badexp > 0:
            if not os.path.exists('Expflag'):
                os.mkdir('Expflag')
            
            if badexp == 2:
                msg = f'# fetch_files : Failed to get EXPTIME from {file}[0]'
            else:
                msg = f'# fetch_files : EXPTIME = 0 for {file}'
                
            utils.log_comment(utils.LOGFILE, msg, verbose=True)
            
            os.system(f'mv {file} Expflag/')
    
    # Reprocess the RAWs into FLTs
    if reprocess_parallel:
        rep = "python -c 'from grizli.pipeline import reprocess; "
        rep += "reprocess.reprocess_wfc3ir(parallel={0},clean_dark_refs={1})'"
        os.system(rep.format(reprocess_parallel, reprocess_clean_darks))
    else:
        from grizli.pipeline import reprocess
        reprocess.reprocess_wfc3ir(parallel=False,
                                   clean_dark_refs=reprocess_clean_darks)

    # Fetch PFLAT reference files needed for optimal drizzled weight images
    if fetch_flt_calibs:
        flt_files = glob.glob('*_fl?.fits')
        flt_files.sort()
        
        #calib_paths = []
        for file in flt_files:
            cpaths = utils.fetch_hst_calibs(file, 
                                            calib_types=fetch_flt_calibs)
            # calib_paths.extend(paths)

    # Copy mask files generated from preprocessing
    os.system('cp *mask.reg {0}'.format(paths['prep']))

    # Persistence products
    os.chdir(paths['persist'])
    persist_files = fetch.persistence_products(tab)
    for file in persist_files:
        if not os.path.exists(os.path.basename(file)):
            print(file)
            os.system('curl -O {0}'.format(file))

    for file in persist_files:
        root = os.path.basename(file).split('.tar.gz')[0]
        if os.path.exists(root):
            print('Skip', root)
            continue

        # Ugly callout to shell
        os.system('tar xzvf {0}.tar.gz'.format(root))
        os.system('rm {0}/*extper.fits {0}/*flt_cor.fits'.format(root))
        os.system('ln -sf {0}/*persist.fits ./'.format(root))


def remove_bad_expflag(field_root='', HOME_PATH='./', min_bad=2):
    """
    Remove FLT files in RAW directory with bad EXPFLAG values, which
    usually corresponds to failed guide stars.

    The script moves files associated with an affected visit to a subdirectory

        >>> bad_dir = os.path.join(HOME_PATH, field_root, 'RAW', 'Expflag')

    Parameters
    ----------
    field_root : str
        Field name, i.e., 'j123654+621608'

    HOME_PATH : str
        Base path where files are found.

    min_bad : int
        Minimum number of exposures of a visit where
        `EXPFLAG == 'INDETERMINATE'`.  Occasionally the first exposure of a
        visit has this value set even though guiding is OK, so set to 2
        to try to flag more problematic visits.

    """
    import os
    import glob
    import numpy as np

    try:
        from .. import prep, utils
    except:
        from grizli import prep, utils

    os.chdir(os.path.join(HOME_PATH, field_root, 'RAW'))

    files = glob.glob('*raw.fits')+glob.glob('*flc.fits')
    files.sort()

    if len(files) == 0:
        return False

    expf = utils.header_keys_from_filelist(files, keywords=['EXPFLAG'],
                                           ext=0, colname_case=str.upper)
    expf.write('{0}_expflag.txt'.format(field_root),
               format='csv', overwrite=True)

    visit_name = np.array([file[:6] for file in expf['FILE']])
    visits = np.unique(visit_name)

    for visit in visits:
        bad = (visit_name == visit) & (expf['EXPFLAG'] != 'NORMAL')
        if bad.sum() >= min_bad:
            logstr = '# Found bad visit: {0}, N={1}\n'
            logstr = logstr.format(visit, bad.sum())
            utils.log_comment(utils.LOGFILE, logstr, verbose=True)

            if not os.path.exists('Expflag'):
                os.mkdir('Expflag')

            os.system('mv {0}* Expflag/'.format(visit))


def visit_dict_to_strings(v):
    """
    Make visit dictionary cleaner for writing to YAML
    """
    newv = {}

    if 'files' in v:
        newv['files'] = [str(f) for f in v['files']]
        
    if 'footprint' in v:
        newv['footprint'] = utils.SRegion(v['footprint']).s_region
    
    if 'footprints' in v:
        newv['footprints'] = []
        for fp in v['footprints']:
            try:
                newv['footprints'].append(utils.SRegion(fp).s_region)
            except:
                print(fp)
                print(v['footprints'])
                raise ValueError
    
    for k in v:
        if k not in newv:
            newv[k] = v[k]
            
    return newv


def visit_dict_from_strings(v):
    """
    Make visit dictionary cleaner for writing to YAML
    """
    newv = {}
    
    if 'files' in v:
        newv['files'] = [str(f) for f in v['files']]
        
    if 'footprint' in v:
        newv['footprint'] = utils.SRegion(v['footprint']).shapely[0]
    
    if 'footprints' in v:
        newv['footprints'] = [utils.SRegion(fp).shapely[0]
                              for fp in v['footprints']]
    
    for k in v:
        if k not in newv:
            newv[k] = v[k]
            
    return newv


def write_visit_info(visits, groups, info, root='j033216m2743', path='./'):
    """
    Write visit association files
    """
    
    new_visits = [visit_dict_to_strings(v) for v in visits]
    
    new_groups = []
    for g in groups:
        gn = {}
        for k in g:
            gn[k] = visit_dict_to_strings(g[k])
        
        new_groups.append(gn)
        
    data = {'visits':new_visits, 'groups': new_groups}
    data['info'] = {}
    for c in info.colnames:
        data['info'][c] = info[c].tolist()
        
    with open(os.path.join(path, f'{root}_visits.yaml'), 'w') as fp:
        yaml.dump(data, stream=fp, Dumper=yaml.Dumper)


def convert_visits_npy_to_yaml(npy_file):
    """
    """
    root = npy_file.split('_visits.npy')[0]
    visits, groups, info = np.load(npy_file, allow_pickle=True)
    write_visit_info(visits, groups, info, root=root)


def find_visit_file(root='j033216m2743', path='./'):
    """
    Find the yaml or npy visits file, return None if neither found
    """
    yaml_file = os.path.join(path, f'{root}_visits.yaml')
    npy_file = os.path.join(path, f'{root}_visits.npy')
    
    if os.path.exists(yaml_file):
        return yaml_file
    elif os.path.exists(npy_file):
        return npy_file
    else:
        return None


def load_visits_yaml(file):
    """
    Load a {root}_visits.yaml file
    
    Returns
    -------
    visits, groups, info
    
    """
    with open(file) as fp:
        data = yaml.load(fp, Loader=yaml.Loader)
    
    visits = [visit_dict_from_strings(v) for v in data['visits']]

    groups = []
    for g in data['groups']:
        gn = {}
        for k in g:
            gn[k] = visit_dict_from_strings(g[k])

        groups.append(gn)
    
    info = utils.GTable(data['info'])
    return visits, groups, info


def load_visit_info(root='j033216m2743', path='./', verbose=True):
    """
    Load visit info from a visits.npy or visits.yaml file
    """
    
    yaml_file = os.path.join(path, f'{root}_visits.yaml')
    npy_file = os.path.join(path, f'{root}_visits.npy')

    if os.path.exists(yaml_file):
        msg = f'Load visit information from {yaml_file}'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        visits, groups, info = load_visits_yaml(yaml_file)
        
    elif os.path.exists(npy_file):
        msg = f'Load visit information from {npy_file}'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        visits, groups, info = np.load(npy_file, allow_pickle=True)
    
    else:
        msg = f'Could not find {root}_visits.yaml/npy in {path}'
        raise IOError(msg)
    
    return visits, groups, info


def parse_visits(files=[], field_root='', RAW_PATH='../RAW', use_visit=True, combine_same_pa=True, combine_minexp=2, is_dash=False, filters=VALID_FILTERS, max_dt=1e9, visit_split_shift=1.5, file_query='*'):

    """
    Organize exposures into "visits" by filter / position / PA / epoch
    
    Parameters
    ----------
    field_root : str
        Rootname of the ``{field_root}_visits.yaml`` file to create.
    
    RAW_PATH : str
        Path to raw exposures, relative to working directory
        
    use_visit, max_dt, visit_split_shift : bool, float, float
        See `~grizli.utils.parse_flt_files`.
    
    combine_same_pa : bool
        Combine exposures taken at same PA/orient + filter across visits
    
    combine_minexp : int
        Try to concatenate visits with fewer than this number of exposures
    
    filters : list
        Filters to consider
    
    Returns
    -------
    visits : list
        List of "visit" dicts with keys ``product``, ``files``, ``footprint``, 
        etc.
    
    groups : list
        Visit groups for direct / grism
    
    info : `~astropy.table.Table`
        Exposure summary table
        
    """
    import copy

    #import grizli.prep
    try:
        from .. import prep, utils
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'auto_script.parse_visits')
    except:
        from grizli import prep, utils

    from shapely.geometry import Polygon
    from scipy.spatial import ConvexHull

    if len(files) == 0:
        files = glob.glob(os.path.join(RAW_PATH, file_query+'fl[tc].fits*'))
        files += glob.glob(os.path.join(RAW_PATH, file_query+'c0m.fits*'))
        files += glob.glob(os.path.join(RAW_PATH, file_query+'c0f.fits*'))
        files += glob.glob(os.path.join(RAW_PATH, file_query+'rate.fits*'))
        files += glob.glob(os.path.join(RAW_PATH, file_query+'cal.fits*'))
        
    if len(files) == 0:
        raise ValueError(f'No exposure files found in {RAW_PATH}')
    
    # Remove rate files where cal exist
    rate_files = []
    for file in files:
        if '_rate.fits' in file:
            rate_files.append(file)
    
    for file in rate_files:
        if file.replace('_rate.fits', '_cal.fits') in files:
            print(f'cal found for {file}')
            _ = files.pop(files.index(file))
            
    files.sort()

    # for file in files:
    #     # JWST files have slightly different naming conventions for the header
    #     if os.path.basename(file).startswith('jw'):
    #         hdu = pyfits.open(file, mode='update')
    #         hdu[0].header['RA_TARG'] = hdu[0].header['targ_ra']
    #         hdu[0].header['DEC_TARG'] = hdu[0].header['targ_dec']
    #         hdu[0].header['EXPTIME'] = hdu[0].header['effexptm']
    #         hdu[0].header['PA_V3'] = hdu[1].header['pa_v3']
    # 
    #         if hdu[0].header['FILTER'] != 'CLEAR':
    #             hdu[0].header['OFILTER'] = hdu[0].header['FILTER']
    #             hdu[0].header['OEXPTYP'] = hdu[0].header['EXP_TYPE']
    # 
    #         if 'NRIMDTPT' in hdu[0].header:
    #             hdu[0].header['NRIMDTPT'] = int(hdu[0].header['NRIMDTPT'])
    # 
    #         if 'NDITHPTS' in hdu[0].header:
    #             hdu[0].header['NRIMDTPT'] = int(hdu[0].header['NDITHPTS'])
    # 
    #         hdu.flush()
    #         hdu.close()
            
    info = utils.get_flt_info(files)

    # Only F814W on ACS
    if ONLY_F814W:
        info = info[((info['INSTRUME'] == 'WFC3') & (info['DETECTOR'] == 'IR')) | (info['FILTER'] == 'F814W')]
    elif filters is not None:
        sel = utils.column_string_operation(info['FILTER'], filters, 
                                            method='count', logical='OR')
        info = info[sel]

    if is_dash:
        # DASH visits split by exposure
        ima_files = glob.glob(os.path.join(RAW_PATH, '*ima.fits'))
        ima_files.sort()

        visits = []
        for file in ima_files:
            # Build from IMA filename
            root = os.path.basename(file).split("_ima")[0][:-1]
            im = pyfits.open(file)
            filt = utils.parse_filter_from_header(im[0].header).lower()
            wcs = pywcs.WCS(im['SCI'].header)
            fp = Polygon(wcs.calc_footprint())

            # q_flt.fits is the pipeline product.  will always be
            # fewer DASH-split files
            files = glob.glob(os.path.join(RAW_PATH, 
                                           f'{root}*[a-o]_flt.fits'))
            files.sort()

            if len(files) == 0:
                continue

            files = [os.path.basename(file) for file in files]
            direct = {'product': '{0}-{1}'.format(root, filt),
                      'files': files, 'footprint': fp}

            visits.append(direct)

        all_groups = utils.parse_grism_associations(visits, info)

        write_visit_info(visits, all_groups, info, root=field_root, path='./')
                
        return visits, all_groups, info

    visits, filters = utils.parse_flt_files(info=info, path=RAW_PATH,
                                  uniquename=True, get_footprint=True,
                                  use_visit=use_visit, max_dt=max_dt,
                                  visit_split_shift=visit_split_shift)

    # Don't run combine_minexp if have grism exposures
    grisms = ['G141', 'G102', 'G800L', 'G280', 'GR150C', 'GR150R']
    has_grism = np.in1d(info['FILTER'], grisms).sum() > 0
    # is PUPIL in info?
    has_grism |= np.in1d(info['PUPIL'], ['GRISMR', 'GRISMC']).sum() > 0
    info.meta['HAS_GRISM'] = has_grism

    if combine_same_pa:
        combined = {}
        for visit in visits:
            vspl = visit['product'].split('-')
            if vspl[-2][0].upper() in 'FG':
                # pupil-filter
                filter_pa = '-'.join(vspl[-3:])
                prog = '-'.join(visit['product'].split('-')[-5:-4])
                key = 'jw{0}-{1}'.format(prog, filter_pa)
            else:
                filter_pa = '-'.join(vspl[-2:])
                prog = '-'.join(visit['product'].split('-')[-4:-3])
                key = 'i{0}-{1}'.format(prog, filter_pa)
                
            if key not in combined:
                combined[key] = {'product': key, 'files': [],
                                 'footprint': visit['footprint']}

            combined[key]['files'].extend(visit['files'])

        visits = [combined[k] for k in combined]

        # Account for timing to combine only exposures taken at an
        # epoch defined by `max_dt` days.
        msg = 'parse_visits(combine_same_pa={0}),'.format(combine_same_pa)
        msg += ' max_dt={1:.1f}: {0} {2:>3} visits'
        utils.log_comment(utils.LOGFILE,
                          msg.format('BEFORE', max_dt, len(visits)),
                          verbose=True, show_date=True)

        split_list = []
        for v in visits:
            split_list.extend(utils.split_visit(v, max_dt=max_dt,
                                          visit_split_shift=visit_split_shift, 
                                          path=RAW_PATH))

        visits = split_list
        utils.log_comment(utils.LOGFILE,
                          msg.format(' AFTER', max_dt, len(visits)),
                          verbose=True, show_date=True)

        get_visit_exposure_footprints(visits, 
                                      check_paths=['./', '../RAW', RAW_PATH])

        print('** Combine same PA: **')
        for i, visit in enumerate(visits):
            print('{0} {1} {2}'.format(i, visit['product'], len(visit['files'])))

    elif (combine_minexp > 0) & (not has_grism):
        combined = []
        for visit in visits:
            if len(visit['files']) >= combine_minexp*1:
                combined.append(copy.deepcopy(visit))
            else:
                filter_pa = '-'.join(visit['product'].split('-')[-2:])
                has_match = False
                fp = visit['footprint']
                for ic, cvisit in enumerate(combined):
                    ckey = '-'.join(cvisit['product'].split('-')[-2:])
                    if ckey == filter_pa:
                        cfp = cvisit['footprint']

                        if cfp.intersection(fp).area > 0.2*fp.area:
                            has_match = True
                            cvisit['files'].extend(visit['files'])
                            if 'footprints' in visit.keys():
                                cvisit['footprints'].extend(visit['footprints'])
                            cvisit['footprint'] = cfp.union(fp)

                # No match, add the singleton visit
                if not has_match:
                    combined.append(copy.deepcopy(visit))

        visits = combined
        print('** Combine Singles: **')
        for i, visit in enumerate(visits):
            print('{0} {1} {2}'.format(i, visit['product'], len(visit['files'])))
    all_groups = utils.parse_grism_associations(visits, info)
    print('\n == Grism groups ==\n')
    valid_groups = []
    for g in all_groups:
        try:
            print(g['direct']['product'], len(g['direct']['files']), g['grism']['product'], len(g['grism']['files']))
            valid_groups.append(g)
        except:
            pass

    all_groups = valid_groups

    #np.save('{0}_visits.npy'.format(field_root), [visits, all_groups, info])
    write_visit_info(visits, all_groups, info, root=field_root, path='./')

    return visits, all_groups, info


def get_visit_exposure_footprints(root='j1000p0210', check_paths=['./', '../RAW'], simplify=1.e-6):
    """
    Add exposure-level footprints to the visit dictionary

    Parameters
    ----------
    visit_file : str, list
        File produced by `parse_visits` (`visits`, `all_groups`, `info`).
        If a list, just parse a list of visits and don't save the file.

    check_paths : list
        Look for the individual exposures in `visits[i]['files']` in these
        paths.

    simplify : float
        Shapely `simplify` parameter the visit footprint polygon.

    Returns
    -------
    visits : dict

    """

    if isinstance(root, str):
        #visits, all_groups, info = np.load(visit_file, allow_pickle=True)
        visits, all_groups, info = load_visit_info(root, verbose=False)
    else:
        visits = root

    fps = {}

    for visit in visits:
        visit['footprints'] = []
        visit_fp = None
        for file in visit['files']:
            fp_i = None
            for path in check_paths:
                pfile = os.path.join(path, file)
                if not os.path.exists(pfile):
                    pfile = os.path.join(path, file + '.gz')
                    
                if os.path.exists(pfile):
                    fp_i = utils.get_flt_footprint(flt_file=pfile)

                    if visit_fp is None:
                        visit_fp = fp_i.buffer(1./3600)
                    else:
                        visit_fp = visit_fp.union(fp_i.buffer(1./3600))
                    break

            visit['footprints'].append(fp_i)
            if visit_fp is not None:
                if simplify > 0:
                    visit['footprint'] = visit_fp.simplify(simplify)
                else:
                    visit['footprint'] = visit_fp

            fps[file] = fp_i

    # ToDo: also update visits in all_groups with `fps`

    # Resave the file
    if isinstance(root, str):
        write_visit_info(visits, all_groups, info, root=root, path='./')
        #np.save(visit_file, [visits, all_groups, info])

    return visits


def manual_alignment(field_root='j151850-813028', HOME_PATH='/Volumes/Pegasus/Grizli/Automatic/', skip=True, radius=5., catalogs=['PS1', 'DES', 'SDSS', 'GAIA', 'WISE'], visit_list=None, radec=None):

    #import pyds9
    import glob
    import os
    import numpy as np

    #import grizli
    from ..prep import get_radec_catalog
    from .. import utils, prep, ds9

    files = glob.glob('*guess')

    tab = utils.read_catalog(os.path.join(HOME_PATH, 
                                          f'{field_root}_footprint.fits'))

    #visits, all_groups, info = np.load('{0}_visits.npy'.format(field_root),
    #                                   allow_pickle=True)
    visits, all_groups, info = load_visit_info(field_root, verbose=False)
    
    use_visits = []

    for visit in visits:
        if visit_list is not None:
            if visit['product'] not in visit_list:
                continue

        filt = visit['product'].split('-')[-1]
        if (not filt.startswith('g')):
            hasg = os.path.exists('{0}.align_guess'.format(visit['product']))
            if hasg & skip:
                continue

            use_visits.append(visit)

    print(len(use_visits), len(visits))
    if len(use_visits) == 0:
        return True

    if radec is None:
        radec, ref_catalog = get_radec_catalog(ra=np.mean(tab['ra']),
                    dec=np.median(tab['dec']),
                    product=field_root,
                    reference_catalogs=catalogs, radius=radius)
    else:
        ref_catalog = catalogs[0]

    reference = '{0}/{1}_{2}.reg'.format(os.getcwd(), field_root,
                                         ref_catalog.lower())

    ds9 = ds9.DS9()
    ds9.set('mode pan')
    ds9.set('scale zscale')
    ds9.set('scale log')

    for visit in use_visits:

        filt = visit['product'].split('-')[-1]
        if (not filt.startswith('g')):
            prep.manual_alignment(visit, reference=reference, ds9=ds9)

    ds9.set('quit')


def clean_prep(field_root='j142724+334246'):
    """
    Clean unneeded files after the field preparation
    """
    import glob
    import os

    #visits, all_groups, info = np.load('{0}_visits.npy'.format(field_root),
    #                                   allow_pickle=True)
    visits, all_groups, info = load_visit_info(field_root, verbose=False)

    for visit in visits:
        for ext in ['_drz_wht', '_seg', '_bkg']:
            file = visit['product']+ext+'.fits'
            if os.path.exists(file):
                print('remove '+file)
                os.remove(file)

    clean_files = glob.glob('*crclean.fits')
    for file in clean_files:
        print('remove '+file)
        os.remove(file)

    # Do this in preprocess to avoid doing it over and over
    # Fix NaNs
    # flt_files = glob.glob('*_fl?.fits')
    # for flt_file in flt_files:
    #     utils.fix_flt_nan(flt_file, verbose=True)


def preprocess(field_root='j142724+334246',  HOME_PATH='/Volumes/Pegasus/Grizli/Automatic/', PERSIST_PATH=None, min_overlap=0.2, make_combined=True, catalogs=['PS1', 'DES', 'NSC', 'SDSS', 'GAIA', 'WISE'], use_visit=True, master_radec=None, parent_radec=None, use_first_radec=False, skip_imaging=False, clean=True, skip_single_optical_visits=True, visit_prep_args=args['visit_prep_args'], persistence_args=args['persistence_args']):
    """
    master_radec: force use this radec file

    parent_radec: use this file if overlap < min_overlap

    """

    try:
        from .. import prep, utils
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'auto_script.preprocess')
    except:
        from grizli import prep, utils

    import os
    import glob
    import numpy as np
    import grizli

    from shapely.geometry import Polygon
    from scipy.spatial import ConvexHull
    import copy
    
    if PERSIST_PATH is None:
        PERSIST_PATH = os.path.join(HOME_PATH, field_root, 'Persistence')
    
    #visits, all_groups, info = np.load(f'{field_root}_visits.npy',
    #                                   allow_pickle=True)
    visits, all_groups, info = load_visit_info(field_root, verbose=False)

    # check if isJWST
    #isJWST = prep.check_isJWST('../RAW/' + all_groups[0]['direct']['files'][0])

    # Grism visits
    master_footprint = None
    radec = None

    # Master table
    # visit_table = os.path.join(os.path.dirname(grizli.__file__), 'data/visit_alignment.txt')
    # if os.path.exists(visit_table):
    #     visit_table = utils.GTable.gread(visit_table)
    # else:
    #     visit_table = None

    for i in range(len(all_groups)):
        direct = all_groups[i]['direct']
        grism = all_groups[i]['grism']

        print(i, direct['product'], len(direct['files']), grism['product'], len(grism['files']))

        if len(glob.glob(grism['product']+'_dr?_sci.fits')) > 0:
            print('Skip grism', direct['product'], grism['product'])
            continue

        # Do all ACS G800L files exist?
        if 'g800l' in grism['product']:
            test_flc = True
            for file in grism['files']:
                test_flc &= os.path.exists(file)

            if test_flc:
                print('Skip grism (all FLC exist)', direct['product'],
                      grism['product'])
                continue

        # Make guess file
        # if visit_table is not None:
        #     ix = ((visit_table['visit'] == direct['product']) &
        #           (visit_table['field'] == field_root))
        #
        #     if ix.sum() > 0:
        #         guess = visit_table['xshift', 'yshift', 'rot', 'scale'][ix]
        #         guess['rot'] = 0.
        #         guess['scale'] = 1.
        #         print('\nWCS: '+direct['product']+'\n', guess)
        #         guess.write('{0}.align_guess'.format(direct['product']),
        #                     format='ascii.commented_header')

        radec = None
        
        #print('xxxxxx', master_radec, radec)
        
        if master_radec is not None:
            radec = master_radec
            best_overlap = 0.
            
            if master_radec == 'astrometry_db':
                radec = prep.get_visit_radec_from_database(direct)
                # if radec is None:
                #     master_radec = None
        
        #print('yyyyyy', master_radec, radec)
              
        if radec is None:
            radec_files = glob.glob('*cat.radec')
            radec = parent_radec
            best_overlap = 0
            fp = direct['footprint']
            for rdfile in radec_files:
                if os.path.exists(rdfile.replace('cat.radec', 'wcs_failed')):
                    continue

                points = np.loadtxt(rdfile)
                try:
                    hull = ConvexHull(points)
                except:
                    continue

                rd_fp = Polygon(points[hull.vertices, :])
                olap = rd_fp.intersection(fp)
                if (olap.area > min_overlap*fp.area) & (olap.area > best_overlap):
                    radec = rdfile
                    best_overlap = olap.area

        if use_first_radec:
            master_radec = radec

        print('\n\n\n{0} radec: {1}\n\n\n'.format(direct['product'], radec))
        
        ###########################
        # Preprocessing script, background subtraction, etc.
        status = prep.process_direct_grism_visit(direct=direct, grism=grism,
                            radec=radec, skip_direct=False, **visit_prep_args)

        ###################################
        # Persistence Masking
        for file in direct['files']+grism['files']:
            print(file)
            pfile = os.path.join(PERSIST_PATH, 
                                 file.replace('_flt', '_persist'))
            if os.path.exists(pfile):
                prep.apply_persistence_mask(file, path=PERSIST_PATH,
                                            **persistence_args)

            # Fix NaNs
            utils.fix_flt_nan(file, verbose=True)

    # From here, `radec` will be the radec file from the first grism visit
    #master_radec = radec

    if skip_imaging:
        return True

    # Ancillary visits
    imaging_visits = []
    for visit in visits:
        filt = visit['product'].split('-')[-1]
        if (len(glob.glob(visit['product']+'_dr?_sci.fits')) == 0) & (not filt.startswith('g1')):
            imaging_visits.append(visit)
    
    # Run preprocessing in order of decreasing filter wavelength
    filters = []
    for v in visits:
        vspl = v['product'].split('-')
        if (vspl[-1] == 'clear') | (vspl[-1][:-1] in ['grism','gr150']):
            fi = vspl[-2]
            if fi.endswith('w2'):
                fi = fi[:-1]
                
            if fi[1] in '0':
                filt = 'f0' + fi[1:]
            elif fi[1] > '1':
                filt = fi[:-1] + '0w'
            else:
                filt = fi        
            
            filters.append(filt)
            
        else:
            filters.append(vspl[-1])

    fwave = np.cast[float]([f.replace('f1', 'f10'). \
                              replace('f098m', 'f0980m'). \
                              replace('lp', 'w'). \
                              replace('gr','g'). \
                              replace('grism','g410'). \
                              replace('fq', 'f')[1:-1] 
                            for f in filters])
    
    if len(np.unique(fwave)) > 1:
        sort_idx = np.argsort(fwave)[::-1]
    else:
        sort_idx = np.arange(len(fwave), dtype=int)

    for i in sort_idx:
        direct = visits[i]
        if 'g800l' in direct['product']:
            continue

        # Skip singleton optical visits
        if (fwave[i] < 900) & (len(direct['files']) == 1):
            if skip_single_optical_visits:
                print('Only one exposure, skip', direct['product'])
                continue

        if len(glob.glob(direct['product']+'_dr?_sci.fits')) > 0:
            print('Skip', direct['product'])
            continue
        else:
            print(direct['product'])

        # if master_radec is not None:
        #     radec = master_radec
        #     best_overlap = 0
        #     fp = direct['footprint']
        radec = None
        
        if master_radec is not None:
            radec = master_radec
            best_overlap = 0.
            
            if master_radec == 'astrometry_db':
                radec = prep.get_visit_radec_from_database(direct)
                # if radec is None:
                #     master_radec = None
                if radec is not None:
                    fp = direct['footprint']
            else:
                if radec is not None:
                    fp = direct['footprint']
                
        if radec is None:
            radec_files = glob.glob('*cat.radec')
            radec = parent_radec
            best_overlap = 0
            radec_n = 0
            fp = direct['footprint']
            for rdfile in radec_files:
                points = np.loadtxt(rdfile)
                hull = ConvexHull(points)
                rd_fp = Polygon(points[hull.vertices, :])
                olap = rd_fp.intersection(fp)
                if (olap.area > min_overlap*fp.area) & (olap.area > best_overlap) & (len(points) > 0.2*radec_n):
                    radec = rdfile
                    best_overlap = olap.area
                    radec_n = len(points)

        print('\n\n\n{0} radec: {1} ({2:.2f})\n\n\n'.format(direct['product'], radec, best_overlap/fp.area))

        try:
            try:
                status = prep.process_direct_grism_visit(direct=direct,
                                        grism={}, radec=radec,
                                        skip_direct=False, **visit_prep_args)
            except:
                utils.log_exception(utils.LOGFILE, traceback)
                utils.log_comment(utils.LOGFILE, "# !! First `prep` run failed with `run_tweak_align`. Try again")

                if 'run_tweak_align' in visit_prep_args:
                    visit_prep_args['run_tweak_align'] = False

                status = prep.process_direct_grism_visit(direct=direct,
                                        grism={}, radec=radec,
                                        skip_direct=False, **visit_prep_args)

            failed_file = '%s.failed' % (direct['product'])
            if os.path.exists(failed_file):
                os.remove(failed_file)

            ###################################
            # Persistence Masking
            for file in direct['files']:
                print(file)
                pfile = os.path.join(PERSIST_PATH, 
                                     file.replace('_flt', '_persist'))
                if os.path.exists(pfile):
                    prep.apply_persistence_mask(file, path=PERSIST_PATH,
                                                **persistence_args)

                # Fix NaNs
                utils.fix_flt_nan(file, verbose=True)
        except:
            fp = open('%s.failed' % (direct['product']), 'w')
            fp.write('\n')
            fp.close()

    ###################################
    # WFC3/IR Satellite trails
    if False:
        from mywfc3.satdet import _detsat_one
        wfc3 = (info['INSTRUME'] == 'WFC3') & (info['DETECTOR'] == 'IR')
        for file in info['FILE'][wfc3]:
            print(file)
            mask = _detsat_one(file, update=False, ds9=None, plot=False, verbose=True)

    ###################################
    # Clean up
    if clean:
        clean_prep(field_root=field_root)
    
    ###################################
    # Drizzle by filter
    # failed = [f.split('.failed')[0] for f in glob.glob('*failed')]
    # keep_visits = []
    # for visit in visits:
    #     if visit['product'] not in failed:
    #         keep_visits.append(visit)
    #
    # overlaps = utils.parse_visit_overlaps(keep_visits, buffer=15.0)
    # np.save('{0}_overlaps.npy'.format(field_root), [overlaps])
    #
    # keep = []
    # wfc3ir = {'product':'{0}-ir'.format(field_root), 'files':[]}

    # if not make_combined:
    #     return True
    #
    # for overlap in overlaps:
    #     filt = overlap['product'].split('-')[-1]
    #     overlap['product'] = '{0}-{1}'.format(field_root, filt)
    #
    #     overlap['reference'] = '{0}-ir_drz_sci.fits'.format(field_root)
    #
    #     if False:
    #         if 'g1' not in filt:
    #             keep.append(overlap)
    #     else:
    #         keep.append(overlap)
    #
    #     if filt.upper() in ['F098M','F105W','F110W', 'F125W','F140W','F160W']:
    #         wfc3ir['files'].extend(overlap['files'])
    #
    # prep.drizzle_overlaps([wfc3ir], parse_visits=False, pixfrac=0.6, scale=0.06, skysub=False, bits=None, final_wcs=True, final_rot=0, final_outnx=None, final_outny=None, final_ra=None, final_dec=None, final_wht_type='IVM', final_wt_scl='exptime', check_overlaps=False)
    #
    # prep.drizzle_overlaps(keep, parse_visits=False, pixfrac=0.6, scale=0.06, skysub=False, bits=None, final_wcs=True, final_rot=0, final_outnx=None, final_outny=None, final_ra=None, final_dec=None, final_wht_type='IVM', final_wt_scl='exptime', check_overlaps=False)


def mask_IR_psf_spikes(visit={},
mag_lim=17, cat=None, cols=['mag_auto', 'ra', 'dec'], minR=8, dy=5, selection=None, length_scale=1, dq_bit=2048):
    """
    Mask 45-degree diffraction spikes around bright stars

    minR: float
        Mask spike pixels > minR from the star centers

    dy : int
        Mask spike pixels +/- `dy` pixels from the computed center of a spike.

    selection : bool array
        If None, then compute `mag < mag_auto` from `cat`.  Otherwise if
        supplied, use as the selection mask.

    length_scale : float
        Scale length of the spike mask by this factor.  The default spike mask
        length in pixels is

        >>> # m = star AB magnitude
        >>> mask_len = 4*np.sqrt(10**(-0.4*(np.minimum(m,17)-17)))/0.06


    """
    from scipy.interpolate import griddata

    if cat is None:
        cat = utils.read_catalog('{0}.cat.fits'.format(visit['product']))

    try:
        mag, ra, dec = cat[cols[0]], cat[cols[1]], cat[cols[2]]
    except:
        mag, ra, dec = cat['MAG_AUTO'], cat['X_WORLD'], cat['Y_WORLD']

    if selection is None:
        selection = mag < 17

    for file in visit['files']:
        if not os.path.exists(file):
            print('Mask diffraction spikes (skip file {0})'.format(file))
            continue

        im = pyfits.open(file, mode='update')
        print('Mask diffraction spikes ({0}), N={1} objects'.format(file, selection.sum()))

        for ext in [1, 2, 3, 4]:
            if ('SCI', ext) not in im:
                break

            wcs = pywcs.WCS(im['SCI', ext].header, fobj=im)
            try:
                cd = wcs.wcs.cd
            except:
                cd = wcs.wcs.pc

            footp = utils.WCSFootprint(wcs)
            points = np.array([ra, dec]).T
            selection &= footp.path.contains_points(points)
            if selection.sum() == 0:
                continue

            sh = im['SCI', ext].data.shape
            mask = np.zeros(sh, dtype=int)
            iy, ix = np.indices(sh)

            # Spider angles, by hand!
            thetas = np.array([[1.07000000e+02, 1.07000000e+02, -8.48089636e-01,  8.46172810e-01],
             [3.07000000e+02, 1.07000000e+02, -8.48252315e-01,  8.40896646e-01],
             [5.07000000e+02, 1.07000000e+02, -8.42360089e-01,  8.38631568e-01],
             [7.07000000e+02, 1.07000000e+02, -8.43990233e-01,  8.36766818e-01],
             [9.07000000e+02, 1.07000000e+02, -8.37264191e-01,  8.31481992e-01],
             [1.07000000e+02, 3.07000000e+02, -8.49196752e-01,  8.47137753e-01],
             [3.07000000e+02, 3.07000000e+02, -8.46919396e-01,  8.43697746e-01],
             [5.07000000e+02, 3.07000000e+02, -8.43849045e-01,  8.39136104e-01],
             [7.07000000e+02, 3.07000000e+02, -8.40070025e-01,  8.36362299e-01],
             [9.07000000e+02, 3.07000000e+02, -8.35218388e-01,  8.34258999e-01],
             [1.07000000e+02, 5.07000000e+02, -8.48708154e-01,  8.48377823e-01],
             [3.07000000e+02, 5.07000000e+02, -8.45874787e-01,  8.38512574e-01],
             [5.07000000e+02, 5.07000000e+02, -8.37238493e-01,  8.42544142e-01],
             [7.07000000e+02, 5.07000000e+02, -8.26696970e-01,  8.37981214e-01],
             [9.07000000e+02, 5.07000000e+02, -8.29422567e-01,  8.32182726e-01],
             [1.07000000e+02, 7.07000000e+02, -8.42331487e-01,  8.43417815e-01],
             [3.07000000e+02, 7.07000000e+02, -8.40006233e-01,  8.48355643e-01],
             [5.07000000e+02, 7.07000000e+02, -8.39776844e-01,  8.48106508e-01],
             [7.07000000e+02, 7.07000000e+02, -8.38620315e-01,  8.40031240e-01],
             [9.07000000e+02, 7.07000000e+02, -8.28351652e-01,  8.31933185e-01],
             [1.07000000e+02, 9.07000000e+02, -8.40726238e-01,  8.51621083e-01],
             [3.07000000e+02, 9.07000000e+02, -8.36006159e-01,  8.46746171e-01],
             [5.07000000e+02, 9.07000000e+02, -8.35987878e-01,  8.48932633e-01],
             [7.07000000e+02, 9.07000000e+02, -8.34104095e-01,  8.46009851e-01],
             [9.07000000e+02, 9.07000000e+02, -8.32700159e-01,  8.38512715e-01]])

            thetas[thetas == 107] = 0
            thetas[thetas == 907] = 1014

            xy = np.array(wcs.all_world2pix(ra[selection], dec[selection], 0)).T

            t0 = griddata(thetas[:, :2], thetas[:, 2], xy, method='linear',
                          fill_value=np.mean(thetas[:, 2]))
            t1 = griddata(thetas[:, :2], thetas[:, 3], xy, method='linear',
                          fill_value=np.mean(thetas[:, 3]))

            for i, m in enumerate(mag[selection]):

                # Size that depends on magnitude
                xlen = 4*np.sqrt(10**(-0.4*(np.minimum(m, 17)-17)))/0.06
                xlen *= length_scale

                x = np.arange(-xlen, xlen, 0.05)
                xx = np.array([x, x*0.])

                for t in [t0[i], t1[i]]:
                    _mat = np.array([[np.cos(t), -np.sin(t)],
                                     [np.sin(t), np.cos(t)]])

                    xr = _mat.dot(xx).T

                    x = xr+xy[i, :]
                    xp = np.cast[int](np.round(x))
                    #plt.plot(xp[:,0], xp[:,1], color='pink', alpha=0.3, linewidth=5)

                    for j in range(-dy, dy+1):
                        ok = (xp[:, 1]+j >= 0) & (xp[:, 1]+j < sh[0])
                        ok &= (xp[:, 0] >= 0) & (xp[:, 0] < sh[1])
                        ok &= np.abs(xp[:, 1]+j - xy[i, 1]) > minR
                        ok &= np.abs(xp[:, 0] - xy[i, 0]) > minR

                        mask[xp[ok, 1]+j, xp[ok, 0]] = 1

            im['DQ', ext].data |= mask*dq_bit
            im.flush()


def multiband_catalog(field_root='j142724+334246', threshold=1.8, detection_background=True, photometry_background=True, get_all_filters=False, filters=None, det_err_scale=-np.inf, phot_err_scale=-np.inf, rescale_weight=True, run_detection=True, detection_filter='ir', detection_root=None, output_root=None, use_psf_filter=True, detection_params=prep.SEP_DETECT_PARAMS,  phot_apertures=prep.SEXTRACTOR_PHOT_APERTURES_ARCSEC, master_catalog=None, bkg_mask=None, bkg_params={'bw': 64, 'bh': 64, 'fw': 3, 'fh': 3, 'pixel_scale': 0.06}, use_bkg_err=False, aper_segmask=True, sci_image=None):
    """
    Make a detection catalog and run aperture photometry on all available
    filter images with the SourceExtractor clone `~sep`.
    
    Parameters
    ----------
    field_root : str
        Rootname of detection images and individual filter images (and 
        weights).
        
    threshold : float
        Detection threshold,  see `~grizli.prep.make_SEP_catalog`.
        
    detection_background : bool
        Background subtraction on detection image, see `get_background` on  
        `~grizli.prep.make_SEP_catalog`.
        
    photometry_background : bool
        Background subtraction when doing photometry on filter images, 
        see `get_background` on `~grizli.prep.make_SEP_catalog`.
        
    get_all_filters : bool
        Find all filter images available for `field_root`
        
    filters : list, None
        Explicit list of filters to include, rather than all available
        
    det_err_scale : float
        Uncertainty scaling for detection image, see `err_scale` on 
        `~grizli.prep.make_SEP_catalog`.
        
    phot_err_scale : float
        Uncertainty scaling for filter images, see `err_scale` on 
        `~grizli.prep.make_SEP_catalog`.
        
    rescale_weight : bool
        Rescale the weight images based on `sep.Background.rms` for both 
        detection and filter images, see `grizli.prep.make_SEP_catalog`.
        
    run_detection : bool
        Run the source detection.  Can be False if the detection catalog file
        (`master_catalog`) and segmentation image 
        (``{field_root}-{detection_filter}_seg.fits``) already exist, i.e., 
        from a separate call to `~grizli.prep.make_SEP_catalog`.
        
    detection_filter : str
        Filter image to use for the source detection.  The default ``ir`` is
        the product of `grizli.pipeline.auto_script.make_filter_combinations`.
        The detection image filename will be 
        ``{field_root}-{detection_filter}_drz_sci.fits`` and with associated 
        weight image ``{field_root}-{detection_filter}_drz_wht.fits``.
        
    detection_root : str, None
        Alternative rootname to use for the detection (and weight) image,
        i.e., ``{detection_root}_drz_sci.fits``.
        Note that the ``_drz_sci.fits`` suffixes are currently required by 
        `~grizli.prep.make_SEP_catalog`.
        
    output_root : str, None
        Rootname of the output catalog file to use, if desired other than 
        `field_root`.
        
    use_psf_filter : bool
        For HST, try to use the PSF as the convolution filter for source 
        detection
        
    detection_params : dict
        Source detection parameters, see `~grizli.prep.make_SEP_catalog`.  
        Many of these are analogous to SourceExtractor parameters.
        
    phot_apertures : list
        Aperture *diameters*. If provided as a string, then apertures assumed
        to be in pixel units. Can also provide a list of elements with
        astropy.unit attributes, which are converted to pixels given the image
        WCS/pixel size.  See `~grizli.prep.make_SEP_catalog`.
    
    master_catalog : str, None
        Filename of the detection catalog, if None then build as 
        ``{field_root}-{detection_filter}.cat.fits``
        
    bkg_mask : array-like, None
        Mask to use for the detection and photometry background determination,
        see `~grizli.prep.make_SEP_catalog`.  This has to be the same 
        dimensions as the images themselves.
        
    bkg_params : dict
        Background parameters, analogous to SourceExtractor, see
        `~grizli.prep.make_SEP_catalog`.
        
    use_bkg_err : bool
        Use the background rms array determined by `sep` for the uncertainties
        (see `sep.Background.rms`).
        
    aper_segmask : bool
        Use segmentation masking for the aperture photometry, see
        `~grizli.prep.make_SEP_catalog`.
        
    sci_image : array-like, None
        Array itself to use for source detection, see
        `~grizli.prep.make_SEP_catalog`.
        
    Returns
    -------
    tab : `~astropy.table.Table`
        Catalog with detection parameters and aperture photometry.  This 
        is essentially the same as the output for 
        `~grizli.prep.make_SEP_catalog` but with separate photometry columns 
        for each multi-wavelength filter image found.
    
    """
    try:
        from .. import prep, utils
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'auto_script.multiband_catalog')
    except:
        from grizli import prep, utils

    # Make catalog
    if master_catalog is None:
        master_catalog = '{0}-{1}.cat.fits'.format(field_root, 
                                                   detection_filter)
    else:
        if not os.path.exists(master_catalog):
            print('Master catalog {0} not found'.format(master_catalog))
            return False

    if not os.path.exists(master_catalog):
        run_detection = True

    if detection_root is None:
        detection_root = '{0}-{1}'.format(field_root, detection_filter)

    if output_root is None:
        output_root = field_root

    if run_detection:
        if use_psf_filter:
            psf_files = glob.glob('{0}*psf.fits'.format(field_root))
            if len(psf_files) > 0:
                psf_files.sort()
                psf_im = pyfits.open(psf_files[-1])

                msg = '# Generate PSF kernel from {0}\n'.format(psf_files[-1])
                utils.log_comment(utils.LOGFILE, msg, verbose=True)

                sh = psf_im['PSF', 'DRIZ1'].data.shape
                # Cut out center of PSF
                skip = (sh[0]-1-11)//2
                psf = psf_im['PSF', 'DRIZ1'].data[skip:-1-skip, skip:-1-skip]*1

                # Optimal filter is reversed PSF (i.e., PSF cross-correlation)
                # https://arxiv.org/pdf/1512.06872.pdf
                psf_kernel = psf[::-1, :][:, ::-1]
                psf_kernel /= psf_kernel.sum()

                detection_params['filter_kernel'] = psf_kernel

        tab = prep.make_SEP_catalog(sci=sci_image, 
                                    root=detection_root,
                                    threshold=threshold,
                                    get_background=detection_background,
                                    save_to_fits=True,
                                    rescale_weight=rescale_weight,
                                    err_scale=det_err_scale,
                                    phot_apertures=phot_apertures,
                                    detection_params=detection_params,
                                    bkg_mask=bkg_mask,
                                    bkg_params=bkg_params,
                                    use_bkg_err=use_bkg_err,
                                    aper_segmask=aper_segmask)
        
        cat_pixel_scale = tab.meta['asec_0'][0]/tab.meta['aper_0'][0]
        
    else:
        tab = utils.GTable.gread(master_catalog)
        cat_pixel_scale = tab.meta['ASEC_0']/tab.meta['APER_0']

    # Source positions
    #source_xy = tab['X_IMAGE'], tab['Y_IMAGE']
    if aper_segmask:
        seg_data = pyfits.open('{0}_seg.fits'.format(detection_root))[0].data
        seg_data = np.cast[np.int32](seg_data)

        aseg, aseg_id = seg_data, tab['NUMBER']

        source_xy = tab['X_WORLD'], tab['Y_WORLD'], aseg, aseg_id
        aseg_half = None
    else:
        source_xy = tab['X_WORLD'], tab['Y_WORLD']

    if filters is None:
        #visits_file = '{0}_visits.yaml'.format(field_root)
        visits_file = find_visit_file(root=field_root)
        if visits_file is None:
            get_all_filters = True

        if get_all_filters:
            mq = '{0}-f*dr?_sci.fits*'
            mq = mq.format(field_root.replace('-100mas','-*mas'))
            mosaic_files = glob.glob(mq)
            
            mq = '{0}-clear*dr?_sci.fits*'
            mq = mq.format(field_root.replace('-100mas','-*mas'))
            mosaic_files += glob.glob(mq)
            
            mosaic_files.sort()
            
            filters = [file.split('_')[-3][len(field_root)+1:] 
                       for file in mosaic_files]
        else:
            #vfile = '{0}_visits.npy'.format(field_root)
            #visits, all_groups, info = np.load(vfile, allow_pickle=True)
            visits, all_groups, info = load_visit_info(field_root, 
                                                       verbose=False)

            if ONLY_F814W:
                info = info[((info['INSTRUME'] == 'WFC3') & 
                             (info['DETECTOR'] == 'IR')) | 
                            (info['FILTER'] == 'F814W')]

            # UVIS
            info_filters = [f for f in info['FILTER']]
            for i in range(len(info)):
                file_i = info['FILE'][i]
                if file_i.startswith('i') & ('_flc' in file_i):
                    info_filters[i] += 'U'

            info['FILTER'] = info_filters

            filters = [f.lower() for f in np.unique(info['FILTER'])]

    #filters.insert(0, 'ir')

    #segment_img = pyfits.open('{0}-ir_seg.fits'.format(field_root))[0].data

    fq = '{0}-{1}_dr?_sci.fits*'
    
    for ii, filt in enumerate(filters):
        print(filt)
        if filt.startswith('g'):
            continue

        if filt not in ['g102', 'g141', 'g800l']:
            _fstr = fq.format(field_root.replace('-100mas','-*mas'), filt)
            sci_files = glob.glob(_fstr)
            if len(sci_files) == 0:
                continue

            root = sci_files[0].split('{0}_dr'.format(filt))[0]+filt
            # root = '{0}-{1}'.format(field_root, filt)

            # Check for half-pixel optical images if using segmask
            if aper_segmask:
                sci = pyfits.open(sci_files[0])
                sci_shape = sci[0].data.shape
                sci.close()
                del(sci)

                if sci_shape[0] != aseg.shape[0]:
                    msg = '# filt={0}, need half-size segmentation image'
                    msg += ', shapes sci:{1} seg:{2}'
                    print(msg.format(filt, sci_shape, aseg.shape))
                    
                    if aseg_half is None:
                        aseg_half = np.zeros(sci_shape, dtype=aseg.dtype)
                        for i in [0, 1]:
                            for j in [0, 1]:
                                aseg_half[i::2, j::2] += aseg

                    source_xy = (tab['X_WORLD'], tab['Y_WORLD'],
                                 aseg_half, aseg_id)
                else:
                    source_xy = (tab['X_WORLD'], tab['Y_WORLD'],
                                 aseg, aseg_id)

            filter_tab = prep.make_SEP_catalog(root=root,
                                    threshold=threshold,
                                    rescale_weight=rescale_weight,
                                    err_scale=phot_err_scale,
                                    get_background=photometry_background,
                                    save_to_fits=False,
                                    source_xy=source_xy,
                                    phot_apertures=phot_apertures,
                                    bkg_mask=bkg_mask,
                                    bkg_params=bkg_params,
                                    use_bkg_err=use_bkg_err,
                                    sci=sci_image)

            for k in filter_tab.meta:
                
                newk = '{0}_{1}'.format(filt.upper(), k)
                newk = newk.replace('-CLEAR','')
                tab.meta[newk] = filter_tab.meta[k]

            for c in filter_tab.colnames:
                newc = '{0}_{1}'.format(filt.upper(), c)
                tab[newc] = filter_tab[c]

            # Kron total correction from EE
            newk = '{0}_PLAM'.format(filt.upper())
            newk = newk.replace('-CLEAR','')
            filt_plam = tab.meta[newk]

            tot_corr = prep.get_kron_tot_corr(tab, filt.lower(), 
                                                pixel_scale=cat_pixel_scale, 
                                                photplam=filt_plam)

            #ee_corr = prep.get_kron_tot_corr(tab, filter=filt.lower())
            tab['{0}_tot_corr'.format(filt.upper())] = tot_corr

        else:
            continue

    for c in tab.colnames:
        tab.rename_column(c, c.lower())

    idcol = utils.GTable.Column(data=tab['number'], name='id')
    tab.add_column(idcol, index=0)

    tab.write('{0}_phot.fits'.format(output_root),
              format='fits',
              overwrite=True)

    return tab


def count_grism_exposures(phot, groups, grisms=['g800l', 'g102', 'g141', 'gr150c', 'gr150r'], reset=True, verbose=False):
    """
    Count number of grism exposures that contain objects in a catalog
    """
    from matplotlib.path import Path

    points = np.array([phot['ra'], phot['dec']]).T

    for g in grisms:
        if ('nexp_'+g not in phot.colnames) | reset:
            phot['nexp_'+g] = np.zeros(len(phot), dtype=np.int32)

    for ig, g in enumerate(groups):
        gri = g['grism']['product'].split('-')[-1]
        if gri not in grisms:
            continue

        if verbose:
            print('{0:<4} {1:48} {2}'.format(ig, g['grism']['product'], gri))

        for fp in g['grism']['footprints']:
            hull = Path(np.array(fp.convex_hull.boundary.xy).T)
            phot['nexp_'+gri] += hull.contains_points(points)*1

    phot['has_grism'] = (phot['nexp_'+grisms[0]] > 0).astype(np.uint8)

    if len(grisms) > 1:
        for ig, g in enumerate(grisms):
            phot['has_grism'] |= (phot['nexp_'+g] > 0).astype(np.uint8)*2**ig
            phot.meta[g+'bit'] = 2**ig

    return phot


def photutils_catalog(field_root='j142724+334246', threshold=1.8, subtract_bkg=True):
    """
    Make a detection catalog with SourceExtractor and then measure
    photometry with `~photutils`.
    
    [deprecated, use `~grizli.pipeline.auto_script.multiband_catalog`.]
    
    """
    from photutils import segmentation, background
    import photutils.utils
    import warnings
    warnings.warn('photutils_catalog is deprecated, use ``sep`` catalog '
                  'in multiband_catalog')
                  
    try:
        from .. import prep, utils
    except:
        from grizli import prep, utils

    # Photutils catalog

    #overlaps = np.load('{0}_overlaps.npy'.format(field_root))[0]

    # Make catalog
    sexcat = prep.make_drz_catalog(root='{0}-ir'.format(field_root), threshold=threshold, extra_config=prep.SEXTRACTOR_CONFIG_3DHST)
    #sexcat = prep.make_SEP_catalog(root='{0}-ir'.format(field_root), threshold=threshold, extra_config=prep.SEXTRACTOR_CONFIG_3DHST)

    for c in sexcat.colnames:
        sexcat.rename_column(c, c.lower())

    sexcat = sexcat['number', 'mag_auto', 'flux_radius']

    files = glob.glob('../RAW/*fl[tc].fits')
    info = utils.get_flt_info(files)
    if ONLY_F814W:
        info = info[((info['INSTRUME'] == 'WFC3') & (info['DETECTOR'] == 'IR')) | (info['FILTER'] == 'F814W')]

    filters = [f.lower() for f in np.unique(info['FILTER'])]

    filters.insert(0, 'ir')

    segment_img = pyfits.open('{0}-ir_seg.fits'.format(field_root))[0].data

    for ii, filt in enumerate(filters):
        print(filt)
        if filt.startswith('g'):
            continue

        if filt not in ['g102', 'g141']:
            sci_files = glob.glob(('{0}-{1}_dr?_sci.fits'.format(field_root, filt)))
            if len(sci_files) == 0:
                continue
            else:
                sci_file = sci_files[0]

            sci = pyfits.open(sci_file)
            wht = pyfits.open(sci_file.replace('_sci', '_wht'))
        else:
            continue

        photflam = sci[0].header['PHOTFLAM']
        ABZP = (-2.5*np.log10(sci[0].header['PHOTFLAM']) - 21.10 -
                   5*np.log10(sci[0].header['PHOTPLAM']) + 18.6921)

        bkg_err = 1/np.sqrt(wht[0].data)
        bkg_err[~np.isfinite(bkg_err)] = 0  # 1e30
        total_error = photutils.utils.calc_total_error(sci[0].data, bkg_err, sci[0].header['EXPTIME'])

        wht_mask = (wht[0].data == 0) | (sci[0].data == 0)
        sci[0].data[wht[0].data == 0] = 0

        mask = None  # bkg_err > 1.e29

        ok = wht[0].data > 0
        if ok.sum() == 0:
            print(' No valid pixels')
            continue

        if subtract_bkg:
            try:
                bkg = background.Background2D(sci[0].data, 100, mask=wht_mask | (segment_img > 0), filter_size=(3, 3), filter_threshold=None, edge_method='pad')
                bkg_obj = bkg.background
            except:
                bkg_obj = None

                utils.log_exception(utils.LOGFILE, traceback)
                utils.log_comment(utils.LOGFILE, "# !! Couldn't make bkg_obj")
        else:
            bkg_obj = None

        cat = segmentation.source_properties(sci[0].data, segment_img, error=total_error, mask=mask, background=bkg_obj, filter_kernel=None, wcs=pywcs.WCS(sci[0].header), labels=None)

        if filt == 'ir':
            cols = ['id', 'xcentroid', 'ycentroid', 'sky_centroid', 'sky_centroid_icrs', 'source_sum', 'source_sum_err', 'xmin', 'xmax', 'ymin', 'ymax', 'min_value', 'max_value', 'minval_xpos', 'minval_ypos', 'maxval_xpos', 'maxval_ypos', 'area', 'equivalent_radius', 'perimeter', 'semimajor_axis_sigma', 'semiminor_axis_sigma', 'eccentricity', 'orientation', 'ellipticity', 'elongation', 'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'cxx', 'cxy', 'cyy']
            tab = utils.GTable(cat.to_table(columns=cols))
            cols = ['source_sum', 'source_sum_err']
            for c in cols:
                tab[c.replace('sum', 'flam')] = tab[c]*photflam
        else:
            cols = ['source_sum', 'source_sum_err']
            t_i = cat.to_table(columns=cols)

            mask = (np.isfinite(t_i['source_sum_err']))
            for c in cols:
                tab['{0}_{1}'.format(filt, c)] = t_i[c]
                tab['{0}_{1}'.format(filt, c)][~mask] = np.nan
                cflam = c.replace('sum', 'flam')
                tab['{0}_{1}'.format(filt, cflam)] = t_i[c]*photflam
                tab['{0}_{1}'.format(filt, cflam)][~mask] = np.nan

        tab.meta['PW{0}'.format(filt.upper())] = sci[0].header['PHOTPLAM']
        tab.meta['ZP{0}'.format(filt.upper())] = ABZP
        tab.meta['FL{0}'.format(filt.upper())] = sci[0].header['PHOTFLAM']

    icrs = [(coo.ra.value, coo.dec.value) for coo in tab['sky_centroid_icrs']]
    tab['ra'] = [coo[0] for coo in icrs]
    tab['dec'] = [coo[1] for coo in icrs]

    tab.remove_column('sky_centroid_icrs')
    tab.remove_column('sky_centroid')

    tab.write('{0}_phot.fits'.format(field_root), format='fits', overwrite=True)

    return tab


def load_GroupFLT(field_root='j142724+334246', PREP_PATH='../Prep', force_ref=None, force_seg=None, force_cat=None, galfit=False, pad=(64,256), files=None, gris_ref_filters=GRIS_REF_FILTERS, split_by_grism=False):
    """
    Initialize a GroupFLT object from exposures in a working directory.  The 
    script tries to match mosaic images with grism exposures based on the 
    filter combinations listed in `gris_ref_filters`
    
    Parameters
    ----------
    field_root : str
        Rootname of catalog and imaging mosaics, e.g., 
        ``'{field_root}-ir.cat.fits'``, ``'{field_root}-f160w_drz_sci.fits'``.
    
    PREP_PATH : str
        Path to search for the files
    
    force_ref : str, 'clean', 'model'
        Force filename of the reference image.
    
    force_seg : str
        Force filename of segmentation image
    
    galfit : bool
        Not used recently
        
    Returns
    -------
    grp : `~grizli.multifit.GroupFLT` or a list of them
        `~grizli.multifit.GroupFLT` object[s]
        
    """
    import glob
    import os
    import numpy as np

    from .. import prep, utils, multifit

    if files is None:
        files = glob.glob(os.path.join(PREP_PATH, '*fl[tc].fits'))
        files += glob.glob(os.path.join(PREP_PATH, '*rate.fits'))
        files.sort()

    info = utils.get_flt_info(files)
    for idx, filter in enumerate(info['FILTER']):
        if '-' in filter:
            spl = filter.split('-')
            info['FILTER'][idx] = spl[-1]
            if len(spl) in [2,3]:
                info['PUPIL'][idx] = filter.split('-')[-2]

    masks = {}
    for gr in ['G141', 'G102', 'G800L']:
        masks[gr.lower()] = [info['FILTER'] == gr, gr, '']
    
    ## NIRISS
    for gr in ['GR150R', 'GR150C']:
        for filt in ['F090W', 'F115W', 'F150W', 'F200W']:
            #key = f'{gr.lower()}-{filt.lower()}'
            # key = filt.lower()
            key = filt.lower() + 'n-clear'

            if key in masks:
                masks[key][0] |= ((info['PUPIL'] == filt) & 
                                  (info['FILTER'] == gr))
            else:
                masks[key] = [(info['PUPIL'] == filt) & 
                              (info['FILTER'] == gr), 
                              gr, filt]
    
    # for gr in ['GR150R', 'GR150C']:
    #     for filt in ['F090W', 'F115W', 'F150W', 'F200W']:
    #         key = f'{gr.lower()}-{filt.lower()}'
    #         masks[key] = [(info['PUPIL'] == filt) & (info['FILTER'] == gr), 
    #                       gr, filt]
    
    # NIRCam
    for ig, gr in enumerate(['GRISMR','GRISMC']):
        for filt in ['F277W', 'F356W', 'F410M', 'F444W']:
            #key = f'{gr.lower()}-{filt.lower()}'
            key = filt.lower() + '-clear'
            if key in masks:
                masks[key][0] |= ((info['PUPIL'] == filt) &
                                  (info['FILTER'] == gr))
            else:
                masks[key] = [((info['PUPIL'] == filt) & 
                               (info['FILTER'] == gr)),
                               gr, filt]
                                      
    # has_nircam = False
    # for gr in ['GRISMR','GRISMC']:
    #     for filt in ['F277W', 'F356W', 'F410M', 'F444W']:
    #         key = f'{gr.lower()}-{filt.lower()}'
    #         masks[key] = [(info['PUPIL'] == filt) & (info['FILTER'] == gr), 
    #                       gr, filt]
    #         has_nircam |= masks[key][0].sum() > 0
            
    if force_cat is None:
        #catalog = '{0}-ir.cat.fits'.format(field_root)
        catalog = glob.glob('{0}-ir.cat.fits'.format(field_root))[0]
    else:
        catalog = force_cat

    grp_objects = []

    grp = None

    for key in masks:
        if (masks[key][0].sum() > 0) & (masks[key][1] in gris_ref_filters):
            if masks[key][2] != '':
                ref = key #masks[mask][2]
            else:
                for f in gris_ref_filters[masks[key][1]]:
                    _fstr = '{0}-{1}_drz_sci.fits'
                    if os.path.exists(_fstr.format(field_root, f.lower())):
                        ref = f
                        break
        else:
            continue
        
        # Segmentation image
        if force_seg is None:
            if force_seg is None:
                if galfit == 'clean':
                    _fstr = '{0}-{1}_galfit_orig_seg.fits'
                    seg_file = _fstr.format(field_root,ref.lower())
                elif galfit == 'model':
                    _fstr = '{0}-{1}_galfit_seg.fits'
                    seg_file = _fstr.format(field_root, ref.lower())
                else:
                    _fstr = '{0}-*_seg.fits'
                    seg_file = glob.glob(_fstr.format(field_root))[0]
        else:
            seg_file = force_seg
            
        # Reference image
        if force_ref is None:
            if galfit == 'clean':
                _fstr = '{0}-{1}_galfit_clean.fits'
                ref_file = _fstr.format(field_root, ref.lower())
            elif galfit == 'model':
                _fstr = '{0}-{1}_galfit.fits'
                ref_file = _fstr.format(field_root, ref.lower())
            else:
                _fstr = '{0}-{1}_dr*_sci.fits*'
                ref_file = _fstr.format(field_root, ref.lower())                
                ref_file = glob.glob(ref_file)[0]
        else:
            ref_file = force_ref
        
        _grism_files=list(info['FILE'][masks[key][0]])
        if masks[key][1].startswith('GRISM'):
            # NIRCAM
            if 'F444W' in masks[key][2].upper():
                polyx = [2.1, 5.5, 4.4]
            elif 'F410M' in masks[key][2].upper():
                polyx = [2.1, 5.5, 4.1]
            elif 'F356W' in masks[key][2].upper():
                polyx = [2.1, 5.5, 3.6]
            elif 'F277W' in masks[key][2].upper():
                polyx = [2.1, 5.5, 2.8]
            else:
                polyx = [2.1, 5.5, 3.6]
            
        elif masks[key][1].startswith('GR150'):
            # NIRISS
            polyx = [0.6, 2.5]
            
        else:
            # HST
            polyx = [0.3, 1.8]
        
        msg = f"auto_script.grism_prep: group = {key}"
        msg += f"\nauto_script.grism_prep: N = {len(_grism_files)}"
        for _i, _f in enumerate(_grism_files):
            msg += f'\nauto_script.grism_prep: file {_i} = {_f}'
            
        msg += f'\nauto_script.grism_prep: ref_file = {ref_file}' 
        msg += f'\nauto_script.grism_prep: seg_file = {seg_file}' 
        msg += f'\nauto_script.grism_prep: catalog  = {catalog}' 
        msg += f'\nauto_script.grism_prep: polyx  = {polyx}' 
        msg += f'\nauto_script.grism_prep: pad  = {pad}' 
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        
        grp_i = multifit.GroupFLT(grism_files=_grism_files,
                                  direct_files=[],
                                  ref_file=ref_file,
                                  seg_file=seg_file,
                                  catalog=catalog,
                                  cpu_count=-1,
                                  sci_extn=1,
                                  pad=pad, 
                                  polyx=polyx)
                                  
        grp_objects.append(grp_i)

    if split_by_grism:
        return grp_objects
    else:
        grp = grp_objects[0]
        if len(grp_objects) > 0:
            for i in range(1, len(grp_objects)):
                grp.extend(grp_objects[i])
                del(grp_objects[i])

        return [grp]


def grism_prep(field_root='j142724+334246', PREP_PATH='../Prep', EXTRACT_PATH='../Extractions', ds9=None, refine_niter=3, gris_ref_filters=GRIS_REF_FILTERS, force_ref=None, files=None, split_by_grism=True, refine_poly_order=1, refine_fcontam=0.5, cpu_count=0, mask_mosaic_edges=True, prelim_mag_limit=25, refine_mag_limits=[18, 24], init_coeffs=[1.1, -0.5], grisms_to_process=None, pad=(64, 256), model_kwargs={'compute_size': True}, sep_background_kwargs=None, subtract_median_filter=False, median_filter_size=71, median_filter_central=10):
    """
    Contamination model for grism exposures
    """
    import glob
    import os
    import numpy as np
    import scipy.stats

    try:
        from .. import prep, utils, multifit
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'auto_script.grism_prep')
    except:
        from grizli import prep, utils, multifit

    if grisms_to_process is not None:
        for g in gris_ref_filters.copy():
            if g not in grisms_to_process:
                pg = gris_ref_filters.pop(g)

    grp_objects = load_GroupFLT(field_root=field_root,
                                PREP_PATH=PREP_PATH, 
                                gris_ref_filters=gris_ref_filters,
                                files=files,
                                split_by_grism=split_by_grism, 
                                force_ref=force_ref,
                                pad=pad)

    for grp in grp_objects:
        
        if subtract_median_filter:
            
            # Subtract a median along the dispersion direction
            # and don't do any contamination modeling
            refine_niter = 0
            grp.subtract_median_filter(filter_size=median_filter_size, 
                                       filter_central=median_filter_central)
            
        else:
            ################
            # Compute preliminary model
            grp.compute_full_model(fit_info=None, verbose=True, store=False, 
                                   mag_limit=prelim_mag_limit, 
                                   coeffs=init_coeffs, 
                                   cpu_count=cpu_count,
                                   model_kwargs=model_kwargs)
        
        ##############
        # Save model to avoid having to recompute it again
        grp.save_full_data()

        #############
        # Mask edges of the exposures not covered by reference image
        if mask_mosaic_edges:
            try:
                # Read footprint file created ealier
                fp_file = '{0}-ir.yaml'.format(field_root)
                if os.path.exists(fp_file):
                    with open(fp_file) as fp:
                        fp_data = yaml.load(fp, Loader=yaml.Loader)
                    
                    det_poly = utils.SRegion(fp_data[0]['footprint'])
                    det_poly = det_poly.shapely[0]
                    for flt in grp.FLTs:
                        flt.mask_mosaic_edges(sky_poly=det_poly, verbose=True,
                                              dq_mask=False, dq_value=1024,
                                              err_scale=10, resid_sn=-1)
                else:
                    msg = 'auto_script.grism_prep: '
                    msg += f'Couldn\'t find file {fp_file}'
                    msg += 'for mask_mosaic_edges'
                    utils.log_comment(utils.LOGFILE, msg, verbose=True)
            except:
                utils.log_exception(utils.LOGFILE, traceback)
                pass
        
        
        # Sep background
        if sep_background_kwargs is not None:
            grp.subtract_sep_background(**sep_background_kwargs)
            
        elif not subtract_median_filter:
            ################
            # Remove constant modal background
            for i in range(grp.N):
                mask = (grp.FLTs[i].model < grp.FLTs[i].grism['ERR']*0.6) 
                mask &= (grp.FLTs[i].grism['SCI'] != 0)

                # Fit Gaussian to the masked pixel distribution
                clip = np.ones(mask.sum(), dtype=bool)
                for iter in range(3):
                    clip_data = grp.FLTs[i].grism.data['SCI'][mask][clip]
                    n = scipy.stats.norm.fit(clip_data)
                    clip = np.abs(grp.FLTs[i].grism.data['SCI'][mask]) < 3*n[1]
            
                del(clip_data)
                mode = n[0]

                logstr = '# grism_mode_bg {0} {1} {2:.4f}'
                logstr = logstr.format(grp.FLTs[i].grism.parent_file, 
                                       grp.FLTs[i].grism.filter, mode)
                utils.log_comment(utils.LOGFILE, logstr, verbose=True)

                try:
                    ds9.view(grp.FLTs[i].grism['SCI'] - grp.FLTs[i].model)
                except:
                    pass

                # Subtract
                grp.FLTs[i].grism.data['SCI'] -= mode

        #############
        # Refine the model
        i = 0
        if ds9:
            ds9.view(grp.FLTs[i].grism['SCI'] - grp.FLTs[i].model)
            fr = ds9.get('frame')

        if refine_niter > 0:
            utils.log_comment(utils.LOGFILE, '# Refine contamination', 
                          verbose=True, show_date=True)

            for iter in range(refine_niter):
                print(f'\nRefine contamination model, iter # {iter}\n')
                if ds9:
                    ds9.set('frame {0}'.format(int(fr)+iter+1))

                if (iter == 0) & (refine_niter > 0):
                    refine_i = 1
                else:
                    refine_i = refine_fcontam

                grp.refine_list(poly_order=refine_poly_order, 
                                mag_limits=refine_mag_limits,
                                max_coeff=5, ds9=ds9, verbose=True,
                                fcontam=refine_i,
                                wave=np.linspace(*grp.polyx[:2], 100)*1.e4)
        
            # Do background again
            if sep_background_kwargs is not None:
                grp.subtract_sep_background(**sep_background_kwargs)
        
        ##############
        # Save model to avoid having to recompute it again
        grp.save_full_data()

    # Link minimal files to Extractions directory
    os.chdir(EXTRACT_PATH)
    os.system(f'ln -s {PREP_PATH}/*GrismFLT* .')
    os.system(f'ln -s {PREP_PATH}/*.0?.wcs.fits .')
    os.system(f'ln -s {PREP_PATH}/{field_root}-*.cat.fits .')
    os.system(f'ln -s {PREP_PATH}/{field_root}-*seg.fits .')
    os.system(f'ln -s {PREP_PATH}/*_phot.fits .')

    return grp


DITHERED_PLINE = {'kernel': 'point', 'pixfrac': 0.2, 'pixscale': 0.1, 'size': 8, 'wcs': None}
PARALLEL_PLINE = {'kernel': 'square', 'pixfrac': 1.0, 'pixscale': 0.1, 'size': 8, 'wcs': None}


def refine_model_with_fits(field_root='j142724+334246', grp=None, master_files=None, spectrum='continuum', clean=True, max_chinu=5):
    """
    Refine the full-field grism models with the best fit spectra from 
    individual extractions.
    """
    import glob
    import traceback

    try:
        from .. import multifit
    except:
        from grizli import multifit

    if grp is None:
        if master_files is None:
            master_files = glob.glob('*GrismFLT.fits')
            master_files.sort()

        catalog = glob.glob(f'{field_root}-*.cat.fits')[0]
        try:
            seg_file = glob.glob(f'{field_root}-*_seg.fits')[0]
        except:
            seg_file = None

        grp = multifit.GroupFLT(grism_files=master_files,
                                direct_files=[],
                                ref_file=None,
                                seg_file=seg_file,
                                catalog=catalog,
                                cpu_count=-1,
                                sci_extn=1,
                                pad=(64,256))

    fit_files = glob.glob('*full.fits')
    fit_files.sort()
    N = len(fit_files)
    if N == 0:
        return False
    
    msg = 'Refine model ({0}/{1}): {2} / skip (chinu={3:.1f}, dof={4})'
    
    for i, file in enumerate(fit_files):
        try:
            hdu = pyfits.open(file)
            id = hdu[0].header['ID']

            fith = hdu['ZFIT_STACK'].header
            chinu = fith['CHIMIN']/fith['DOF']
            if (chinu > max_chinu) | (fith['DOF'] < 10):
                print(msg.format(i, N, file, chinu, fith['DOF']))
                continue

            sp = utils.GTable(hdu['TEMPL'].data)

            dt = np.float
            wave = np.cast[dt](sp['wave'])  # .byteswap()
            flux = np.cast[dt](sp[spectrum])  # .byteswap()
            grp.compute_single_model(int(id), mag=19, size=-1, store=False,
                                     spectrum_1d=[wave, flux], is_cgs=True,
                                     get_beams=None, in_place=True)
            print('Refine model ({0}/{1}): {2}'.format(i, N, file))
        except:
            print('Refine model ({0}/{1}): {2} / failed'.format(i, N, file))

    grp.save_full_data()

    if clean:
        print('# refine_model_with_fits: cleanup')
        files = glob.glob('*_grism_*fits')
        files += glob.glob('*beams.fits')
        files += glob.glob('*stack.fits')
        files += glob.glob('*stack.png')
        files += glob.glob('*full.fits')
        for file in files:
            os.remove(file)

    del(grp)


def extract(field_root='j142724+334246', maglim=[13, 24], prior=None, MW_EBV=0.00, ids=[], pline=DITHERED_PLINE, fit_only_beams=True, run_fit=True, poly_order=7, oned_R=30, master_files=None, grp=None, bad_pa_threshold=None, fit_trace_shift=False, size=32, diff=True, min_sens=0.02, fcontam=0.2, min_mask=0.01, sys_err=0.03, skip_complete=True, fit_args={}, args_file='fit_args.npy', get_only_beams=False):
    import glob
    import os

    import numpy as np
    import matplotlib.pyplot as plt

    #import grizli
    try:
        from .. import multifit, prep, utils, fitting
    except:
        from grizli import multifit, prep, utils, fitting

    if master_files is None:
        master_files = glob.glob('*GrismFLT.fits')
        master_files.sort()
    
    isJWST = prep.check_isJWST(master_files[0])

    if grp is None:
        init_grp = True
        catalog = glob.glob('{0}-*.cat.fits'.format(field_root))[0]
        try:
            seg_file = glob.glob('{0}-*_seg.fits'.format(field_root))[0]
        except:
            seg_file = None

        grp = multifit.GroupFLT(grism_files=master_files,
                                direct_files=[],
                                ref_file=None,
                                seg_file=seg_file,
                                catalog=catalog,
                                cpu_count=-1,
                                sci_extn=1,
                                pad=(64,256)
                                )
    else:
        init_grp = False

    ###############
    # PHotometry

    target = field_root

    try:
        file_args = np.load(args_file, allow_pickle=True)[0]
        MW_EBV = file_args['MW_EBV']
        min_sens = file_args['min_sens']
        min_mask = file_args['min_mask']
        fcontam = file_args['fcontam']
        sys_err = file_args['sys_err']
        pline = file_args['pline']
        fit_args = file_args
        fit_args.pop('kwargs')
    except:
        pass

    if get_only_beams:
        beams = grp.get_beams(ids, size=size, beam_id='A', min_sens=min_sens)
        if init_grp:
            del(grp)

        return(beams)

    ###########
    # IDs to extract
    # ids=[1096]

    if ids == []:
        clip = (grp.catalog['MAG_AUTO'] > maglim[0]) & (grp.catalog['MAG_AUTO'] < maglim[1])
        so = np.argsort(grp.catalog['MAG_AUTO'][clip])
        ids = grp.catalog['NUMBER'][clip][so]
    else:
        ids = [int(id) for id in ids]

    # Stack the different beans

    # Use "binning" templates for standardized extraction
    if oned_R:
        bin_steps, step_templ = utils.step_templates(wlim=[5000, 18000.0],
                                                     R=oned_R, round=10)
        init_templates = step_templ
    else:
        # Polynomial templates
        wave = np.linspace(2000, 2.5e4, 100)
        poly_templ = utils.polynomial_templates(wave, order=poly_order)
        init_templates = poly_templ

    #size = 32
    close = True
    show_beams = True

    if __name__ == '__main__':  # Interactive
        size = 32
        close = Skip = False
        pline = {'kernel': 'point', 'pixfrac': 0.2, 'pixscale': 0.1, 'size': 8, 'wcs': None}
        prior = None

        skip_complete = True
        fit_trace_shift = False
        bad_pa_threshold = 1.6
        MW_EBV = 0

    ###############
    # Stacked spectra
    for ii, id in enumerate(ids):
        if skip_complete:
            if os.path.exists('{0}_{1:05d}.stack.png'.format(target, id)):
                continue

        beams = grp.get_beams(id, size=size, beam_id='A', min_sens=min_sens)
        for i in range(len(beams))[::-1]:
            if beams[i].fit_mask.sum() < 10:
                beams.pop(i)

        print('{0}/{1}: {2} {3}'.format(ii, len(ids), id, len(beams)))
        if len(beams) < 1:
            continue

        #mb = multifit.MultiBeam(beams, fcontam=fcontam, group_name=target, psf=False, MW_EBV=MW_EBV, min_sens=min_sens)

        mb = multifit.MultiBeam(beams, fcontam=fcontam, group_name=target, psf=False, isJWST=isJWST, MW_EBV=MW_EBV, sys_err=sys_err, min_mask=min_mask, min_sens=min_sens)

        if bad_pa_threshold is not None:
            out = mb.check_for_bad_PAs(chi2_threshold=bad_pa_threshold,
                                                   poly_order=1, reinit=True,
                                                  fit_background=True)

            fit_log, keep_dict, has_bad = out

            if has_bad:
                print('\n  Has bad PA!  Final list: {0}\n{1}'.format(keep_dict, fit_log))

        ixi = grp.catalog['NUMBER'] == id
        if (fit_trace_shift > 0) & (grp.catalog['MAG_AUTO'][ixi][0] < 24.5):
            b = mb.beams[0]
            b.compute_model()
            sn_lim = fit_trace_shift*1
            if (np.max((b.model/b.grism['ERR'])[b.fit_mask.reshape(b.sh)]) > sn_lim) | (sn_lim > 100):
                print(' Fit trace shift: \n')
                try:
                    shift = mb.fit_trace_shift(tol=1.e-3, verbose=True, split_groups=True, lm=True)
                except:
                    pass

        try:
            tfit = mb.template_at_z(z=0, templates=init_templates, fit_background=True, fitter='lstsq', get_uncertainties=2)
        except:
            tfit = None

        try:
            fig1 = mb.oned_figure(figsize=[5, 3], tfit=tfit, show_beams=show_beams, scale_on_stacked=True, ylim_percentile=5)
            if oned_R:
                outroot = '{0}_{1:05d}.R{2:.0f}'.format(target, id, oned_R)
                hdu = mb.oned_spectrum_to_hdu(outputfile=outroot+'.fits',
                                              tfit=tfit, wave=bin_steps)
            else:
                outroot = '{0}_{1:05d}.1D'.format(target, id)
                hdu = mb.oned_spectrum_to_hdu(outputfile=outroot+'.fits',
                                              tfit=tfit)

            fig1.savefig(outroot+'.png')

        except:
            continue

        hdu, fig = mb.drizzle_grisms_and_PAs(fcontam=0.5, flambda=False, kernel='point', size=32, tfit=tfit, diff=diff)
        fig.savefig('{0}_{1:05d}.stack.png'.format(target, id))

        hdu.writeto('{0}_{1:05d}.stack.fits'.format(target, id),
                    overwrite=True)
        mb.write_master_fits()

        if False:
            # Fit here for AWS...
            fitting.run_all_parallel(id, verbose=True)

        if close:
            plt.close(fig)
            plt.close(fig1)
            del(hdu)
            del(mb)
            for k in range(100000):
                plt.close()

    if not run_fit:
        if init_grp:
            return grp
        else:
            return True

    for ii, id in enumerate(ids):
        print('{0}/{1}: {2}'.format(ii, len(ids), id))

        if not os.path.exists('{0}_{1:05d}.beams.fits'.format(target, id)):
            continue

        if skip_complete:
            if os.path.exists('{0}_{1:05d}.line.png'.format(target, id)):
                continue

        try:
            out = fitting.run_all_parallel(id, get_output_data=True, **fit_args, args_file=args_file)
            mb, st, fit, tfit, line_hdu = out

            spectrum_1d = [tfit['cont1d'].wave, tfit['cont1d'].flux]
            grp.compute_single_model(id, mag=-99, size=-1, store=False, spectrum_1d=spectrum_1d, get_beams=None, in_place=True, is_cgs=True)

            if close:
                for k in range(1000):
                    plt.close()

            del(out)
        except:
            pass

    # Re-save data with updated models
    if init_grp:
        grp.save_full_data()
        return grp
    else:
        return True


def generate_fit_params(field_root='j142724+334246', fitter=['nnls', 'bounded'], prior=None, MW_EBV=0.00, pline=DITHERED_PLINE, fit_only_beams=True, run_fit=True, poly_order=7, fsps=True, min_sens=0.01, sys_err=0.03, fcontam=0.2, zr=[0.05, 3.6], dz=[0.004, 0.0004], fwhm=1000, lorentz=False, include_photometry=True, use_phot_obj=False, save_file='fit_args.npy', fit_trace_shift=False, full_line_list=['Lya', 'OII', 'Hb', 'OIII', 'Ha', 'Ha+NII', 'SII', 'SIII','PaB','He-1083','PaA'],  **kwargs):
    """
    Generate a parameter dictionary for passing to the fitting script
    """
    import numpy as np
    from grizli import utils, fitting
    from . import photoz

    phot = None

    t0 = utils.load_templates(fwhm=fwhm, line_complexes=True, stars=False, full_line_list=None, continuum_list=None, fsps_templates=fsps, alf_template=True, lorentz=lorentz)
    t1 = utils.load_templates(fwhm=fwhm, line_complexes=False, stars=False, full_line_list=None, continuum_list=None, fsps_templates=fsps, alf_template=True, lorentz=lorentz)

    args = fitting.run_all(0, t0=t0, t1=t1, fwhm=1200, zr=zr, dz=dz, fitter=fitter, group_name=field_root, fit_stacks=False, prior=prior,  fcontam=fcontam, pline=pline, min_sens=min_sens, mask_sn_limit=np.inf, fit_beams=False,  root=field_root, fit_trace_shift=fit_trace_shift, phot=phot, use_phot_obj=use_phot_obj, verbose=True, scale_photometry=False, show_beams=True, overlap_threshold=10, get_ir_psfs=True, fit_only_beams=fit_only_beams, MW_EBV=MW_EBV, sys_err=sys_err, get_dict=True, full_line_list=full_line_list)
    
    for k in kwargs:
        if k in args:
            args[k] = kwargs[k]
            
    # EAZY-py photometry object from HST photometry
    try:
        import eazy.photoz
        HAS_EAZY = True
    except:
        HAS_EAZY = False

    if include_photometry & HAS_EAZY:
        aper_ix = include_photometry*1
        utils.set_warnings()

        total_flux = 'flux_auto'
        obj = photoz.eazy_photoz(field_root, object_only=True,
                  apply_prior=False, beta_prior=True, aper_ix=aper_ix-1,
                  force=True,
                  get_external_photometry=False, compute_residuals=False,
                  total_flux=total_flux)

        cat = obj.cat

        #apcorr = cat['flux_iso']/(cat['flux_auto']*cat['tot_corr'])
        apcorr = None

        phot_obj = photoz.EazyPhot(obj, grizli_templates=t0,
                                   source_text='grizli_HST_photometry',
                                   apcorr=apcorr,
                                   include_photometry=True, include_pz=False)

        args['phot_obj'] = phot_obj
        args['scale_photometry'] = True

    np.save(save_file, [args])
    print('Saved arguments to {0}.'.format(save_file))
    return args


def summary_catalog(**kwargs):
    from . import summary
    res = summary.summary_catalog(**kwargs)
    return res


def fine_alignment(field_root='j142724+334246', HOME_PATH='/Volumes/Pegasus/Grizli/Automatic/', min_overlap=0.2, stopme=False, ref_err=1.e-3, radec=None, redrizzle=True, shift_only=True, maglim=[17, 24], NITER=1, catalogs=['GAIA', 'PS1', 'NSC', 'SDSS', 'WISE'], method='Powell', radius=5., program_str=None, match_str=[], all_visits=None, date=None, gaia_by_date=False, tol=None, fit_options=None, print_options={'precision': 3, 'sign': ' '}, include_internal_matches=True, master_gaia_catalog=None):
    """
    Try fine alignment from visit-based SourceExtractor catalogs
    
    """
    import os
    import glob
    import time

    try:
        from .. import prep, utils
        from ..prep import get_radec_catalog, get_gaia_radec_at_time
        from ..utils import transform_wcs

        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'auto_script.fine_alignment')

    except:
        from grizli import prep, utils
        from grizli.prep import get_radec_catalog
        from grizli.utils import transform_wcs

    import numpy as np
    np.set_printoptions(**print_options)

    import matplotlib.pyplot as plt

    from shapely.geometry import Polygon
    from scipy.spatial import ConvexHull
    from drizzlepac import updatehdr

    import astropy.units as u
    from scipy.optimize import minimize, fmin_powell

    import copy

    if all_visits is None:
        #_ = np.load(f'{field_root}_visits.npy', allow_pickle=True)
        all_visits, all_groups, info = load_visit_info(field_root, 
                                                       verbose=False)
        #all_visits, all_groups, info = _

    failed_list = glob.glob('*failed')

    visits = []
    files = []
    for visit in all_visits:
        file = '{0}.cat.fits'.format(visit['product'])

        if visit['product']+'.failed' in failed_list:
            continue

        if os.path.exists(file):
            if program_str is not None:
                prog = visit['product'].split('-')[-4]
                if prog != program_str:
                    continue

            if len(match_str) > 0:
                has_match = False
                for m in match_str:
                    has_match |= m in visit['product']

                if not has_match:
                    continue

            visits.append(visit)
            files.append(file)

    if radec is None:
        ra_i, dec_i = np.median(info['RA_TARG']), np.median(info['DEC_TARG'])
        print('Center coordinate: ', ra_i, dec_i)
        if date is not None:
            radec, ref_catalog = get_radec_catalog(ra=ra_i, dec=dec_i,
                    product=field_root, date=date,
                    reference_catalogs=catalogs, radius=radius)
        else:
            radec, ref_catalog = get_radec_catalog(ra=ra_i, dec=dec_i,
                    product=field_root,
                    reference_catalogs=catalogs, radius=radius)

    #ref = 'j152643+164738_sdss.radec'
    ref_tab = utils.GTable(np.loadtxt(radec, unpack=True).T, 
                           names=['ra', 'dec'])
    ridx = np.arange(len(ref_tab))
    
    # Global GAIA DR2 catalog
    if gaia_by_date:
        if master_gaia_catalog is not None:
            msg = (f'Master gaia catalog: {master_gaia_catalog}')                   
            utils.log_comment(utils.LOGFILE, msg, show_date=True,
                              verbose=True)

            gaia_tab = utils.read_catalog(master_gaia_catalog)
            
        else:
            
            dra = np.max(np.abs(info['RA_TARG']-ra_i))
            dde = np.max(np.abs(info['DEC_TARG']-dec_i))
            drad = np.sqrt((dra*np.cos(dec_i/180*np.pi))**2+(dde**2))*60+2
        
            msg = (f'Get field GAIA catalog ({ra_i:.6f}, {dec_i:.6f})' +
                   f' r={drad:.1f}arcmin')
                   
            utils.log_comment(utils.LOGFILE, msg, show_date=True,
                              verbose=True)
                          
            gaia_tab = prep.get_gaia_DR2_vizier(ra=ra_i, dec=dec_i, 
                                                radius=drad)
        
        # Done
        msg = f'GAIA catalog: {len(gaia_tab)} objects'
        utils.log_comment(utils.LOGFILE, msg, show_date=True,
                          verbose=True)
        
        if len(gaia_tab) == 0:
            msg = f'!No GAIA objects found, will run without absolute frame'
            utils.log_comment(utils.LOGFILE, msg, show_date=True,
                              verbose=True)
            
            gaia_tab = None
    else:
        gaia_tab = None
    
    # Find matches
    tab = {}
    for i, file in enumerate(files):
        tab[i] = {}
        t_i = utils.GTable.gread(file)
        mclip = (t_i['MAG_AUTO'] > maglim[0]) & (t_i['MAG_AUTO'] < maglim[1])
        if mclip.sum() == 0:
            continue

        tab[i]['cat'] = t_i[mclip]

        try:
            sci_file = glob.glob(file.replace('.cat', '_dr?_sci'))[0]
        except:
            sci_file = glob.glob(file.replace('.cat', '_wcs'))[0]

        im = pyfits.open(sci_file)
        tab[i]['wcs'] = pywcs.WCS(im[0].header)

        tab[i]['transform'] = [0, 0, 0, 1]
        tab[i]['xy'] = np.array([tab[i]['cat']['X_IMAGE'], tab[i]['cat']['Y_IMAGE']]).T

        tab[i]['match_idx'] = {}

        if gaia_by_date & (gaia_tab is not None):
            drz_file = glob.glob(file.replace('.cat.fits',
                                              '*dr?_sci.fits'))[0]
            drz_im = pyfits.open(drz_file)
            
            coo = get_gaia_radec_at_time(gaia_tab, 
                                         date=drz_im[0].header['EXPSTART'],
                                         format='mjd')
            
            ok = np.isfinite(coo.ra+coo.dec)
            ref_tab = utils.GTable()
            if ok.sum() == 0:
                ref_tab['ra'] = [0.]
                ref_tab['dec'] = [-89.]
            else:
                ref_tab['ra'] = coo.ra[ok].value
                ref_tab['dec'] = coo.dec[ok].value
            
            prod = '-'.join(file.split('-')[:-1])
            prep.table_to_radec(ref_tab, f'{prod}_gaia.radec')
            
            # radec, ref_catalog = get_radec_catalog(ra=drz_im[0].header['CRVAL1'],
            #         dec=drz_im[0].header['CRVAL2'],
            #         product='-'.join(file.split('-')[:-1]),  date=drz_im[0].header['EXPSTART'], date_format='mjd',
            #         reference_catalogs=['GAIA'], radius=radius)
            # 
            # ref_tab = utils.GTable(np.loadtxt(radec, unpack=True).T, names=['ra', 'dec'])
            ridx = np.arange(len(ref_tab))

        tab[i]['ref_tab'] = ref_tab
        idx, dr = tab[i]['cat'].match_to_catalog_sky(ref_tab)
        clip = dr < 0.6*u.arcsec
        if clip.sum() > 1:
            tab[i]['match_idx'][-1] = [idx[clip], ridx[clip]]
        
        msg = '{0} Ncat={1} Nref={2}'
        utils.log_comment(utils.LOGFILE, 
                          msg.format(sci_file, mclip.sum(), clip.sum()), 
                          show_date=False,
                          verbose=True)
                          
        # ix, jx = tab[i]['match_idx'][-1]
        # ci = tab[i]['cat']#[ix]
        # cj = ref_tab#[jx]

    if include_internal_matches:
        for i, file in enumerate(files):
            for j in range(i+1, len(files)):
                sidx = np.arange(len(tab[j]['cat']))
                idx, dr = tab[i]['cat'].match_to_catalog_sky(tab[j]['cat'])
                clip = dr < 0.3*u.arcsec
                print(file, files[j], clip.sum())

                if clip.sum() < 5:
                    continue

                if clip.sum() > 0:
                    tab[i]['match_idx'][j] = [idx[clip], sidx[clip]]

    #ref_err = 0.01

    # shift_only=True
    if shift_only > 0:
        # Shift only
        p0 = np.vstack([[0, 0] for i in tab])
        pscl = np.array([10., 10.])
    elif shift_only < 0:
        # Shift + rot + scale
        p0 = np.vstack([[0, 0, 0, 1] for i in tab])
        pscl = np.array([10., 10., 100., 100.])
    else:
        # Shift + rot
        p0 = np.vstack([[0, 0, 0] for i in tab])
        pscl = np.array([10., 10., 100.])

    #ref_err = 0.06

    if False:
        field_args = (tab, ref_tab, ref_err, shift_only, 'field')
        _objfun_align(p0*10., *field_args)

    fit_args = (tab, ref_tab, ref_err, shift_only, 'huber')
    plot_args = (tab, ref_tab, ref_err, shift_only, 'plot')
    plotx_args = (tab, ref_tab, ref_err, shift_only, 'plotx')

    pi = p0*1.  # *10.
    for iter in range(NITER):
        fit = minimize(_objfun_align, pi*pscl, args=fit_args, method=method, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=tol, callback=None, options=fit_options)
        pi = fit.x.reshape((-1, len(pscl)))/pscl

    ########
    # Show the result
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot(221)
    _objfun_align(p0*pscl, *plot_args)
    ax.set_xticklabels([])
    ax.set_ylabel('dDec')

    ax = fig.add_subplot(223)
    _objfun_align(p0*pscl, *plotx_args)
    ax.set_ylabel('dDec')
    ax.set_xlabel('dRA')

    ax = fig.add_subplot(222)
    _objfun_align(fit.x, *plot_args)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = fig.add_subplot(224)
    _objfun_align(fit.x, *plotx_args)
    ax.set_yticklabels([])
    ax.set_xlabel('dRA')

    for ax in fig.axes:
        ax.grid()
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)

    fig.tight_layout(pad=0.5)

    extra_str = ''
    if program_str:
        extra_str += '.{0}'.format(program_str)

    if match_str:
        extra_str += '.{0}'.format('.'.join(match_str))

    fig.text(0.97, 0.02, time.ctime(), ha='right', va='bottom', fontsize=5, transform=fig.transFigure)

    fig.savefig('{0}{1}_fine.png'.format(field_root, extra_str))
    fine_file = '{0}{1}_fine.yaml'.format(field_root, extra_str)
    save_fine_yaml(visits, fit, yaml_file=fine_file)
    #np.save('{0}{1}_fine.npy'.format(field_root, extra_str), [visits, fit])

    return tab, fit, visits


def save_fine_yaml(visits, fit, yaml_file='fit_fine.yaml'):
    """
    Save results from fine_alignment
    """
    vis = [visit_dict_to_strings(v) for v in visits]
    
    fit_res = {}
    for k in fit:
        if k in ['direc','x']:
            fit_res[k] = fit[k].tolist()
        else:
            fit_res[k] = fit[k]
    
    fit_res['fun'] = float(fit['fun'])
    
    with open(yaml_file, 'w') as fp:
        yaml.dump([vis, fit_res], stream=fp, Dumper=yaml.Dumper)


def load_fine_yaml(yaml_file='fit_fine.yaml'):
    """
    Load fine_alignment result
    """
    
    with open(yaml_file) as fp:
        vis, fit = yaml.load(fp, Loader=yaml.Loader)
    
    visits = [visit_dict_from_strings(v) for v in vis]
    for k in ['direc','x']:
        fit[k] = np.array(fit[k])
    
    return visits, fit


def update_wcs_headers_with_fine(field_root, backup=True):
    """
    Update grism headers with the fine shifts
    """
    import os
    import numpy as np
    import glob

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    from drizzlepac import updatehdr

    #import grizli.prep
    try:
        from .. import prep
    except:
        from grizli import prep

    if backup:
        if not os.path.exists('FineBkup'):
            os.mkdir('FineBkup')

    #visits, all_groups, info = np.load(f'{field_root}_visits.npy',
    #                                   allow_pickle=True)
    visits, all_groups, info = load_visit_info(field_root, verbose=False)

    fit_files = glob.glob('{0}*fine.yaml'.format(field_root))
    
    for fit_file in fit_files:
        #fine_visits, fine_fit = np.load(fit_file, allow_pickle=True)
        fine_visits, fine_fit = load_fine_yaml(yaml_file=fit_file)
        
        N = len(fine_visits)

        if backup:
            for i in range(N):
                direct = fine_visits[i]
                for file in direct['files']:
                    os.system(f'cp {file} FineBkup/')
                    print(file)

        trans = np.reshape(fine_fit['x'], (N, -1))  # /10.
        sh = trans.shape
        if sh[1] == 2:
            pscl = np.array([10., 10.])
            trans = np.hstack([trans/pscl, np.zeros((N, 1)), np.ones((N, 1))])
        elif sh[1] == 3:
            pscl = np.array([10., 10., 100])
            trans = np.hstack([trans/pscl, np.ones((N, 1))])
        elif sh[1] == 4:
            pscl = np.array([10., 10., 100, 100])
            trans = trans/pscl

        # Update direct WCS
        for ix, direct in enumerate(fine_visits):
            #direct = visits[ix]
            out_shift, out_rot = trans[ix, :2], trans[ix, 2]
            out_scale = trans[ix, 3]

            xyscale = trans[ix, :4]

            xyscale[2] *= -1
            out_rot *= -1

            try:
                wcs_ref_file = str('{0}.cat.fits'.format(direct['product']))
                wcs_ref = pywcs.WCS(pyfits.open(wcs_ref_file)['WCS'].header,
                                relax=True)
            except:
                wcs_ref_file = str('{0}_wcs.fits'.format(direct['product']))
                wcs_ref = pywcs.WCS(pyfits.open(wcs_ref_file)[0].header,
                                relax=True)

            for file in direct['files']:
                prep.update_wcs_fits_log(file, wcs_ref,
                                    xyscale=xyscale,
                                    initialize=False,
                                    replace=('.fits', '.wcslog.fits'),
                                    wcsname='FINE')

                updatehdr.updatewcs_with_shift(file,
                                      wcs_ref_file,
                                      xsh=out_shift[0], ysh=out_shift[1],
                                      rot=out_rot, scale=out_scale,
                                      wcsname='FINE', force=True,
                                      reusename=True, verbose=True,
                                      sciext='SCI')

            # Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
            # keywords
            im = pyfits.open(file, mode='update')
            im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
            im.flush()

        # Update grism WCS
        for i in range(len(all_groups)):
            direct = all_groups[i]['direct']
            grism = all_groups[i]['grism']
            for j in range(N):
                if fine_visits[j]['product'] == direct['product']:
                    print(direct['product'], grism['product'], trans[j, :])

                    if backup:
                        for file in grism['files']:
                            os.system(f'cp {file} FineBkup/')
                            print(file)

                    prep.match_direct_grism_wcs(direct=direct, grism=grism,
                                                get_fresh_flt=False,
                                                xyscale=trans[j, :])


def make_reference_wcs(info, files=None, output='mosaic_wcs-ref.fits', filters=['G800L', 'G102', 'G141','GR150C', 'GR150R'], force_wcs=None, pad_reference=90, pixel_scale=None, get_hdu=True):
    """
    Make a reference image WCS based on the grism exposures

    Parameters
    ----------
    info : `~astropy.table.Table`
        Exposure information table with columns 'FILE' and 'FILTER'.

    output : str, None
        Filename for output wcs reference image.

    filters : list or None
        List of filters to consider for the output mosaic.  If None, then
        use all exposures in the `info` list.
    
    force_wcs : `~astropy.wcs.WCS` or None
        Explicit WCS to use, rather than deriving from a list of files
        
    pad_reference : float
        Image padding, in `~astropy.units.arcsec`.

    pixel_scale : None or float
        Pixel scale in in `~astropy.units.arcsec`.  If None, then the script
        computes automatically

    get_hdu : bool
        If True, then generate an `~astropy.io.fits.ImageHDU` object and
        save to a file if `output` is defined.  If False, return just the
        computed `~astropy.wcs.WCS`.

    Returns
    -------
    `~astropy.io.fits.ImageHDU` or `~astropy.wcs.WCS`, see `get_hdu`.


    """
    
    if isinstance(force_wcs, pywcs.WCS):
        _h = utils.to_header(force_wcs)
        _data = np.zeros((_h['NAXIS2'], _h['NAXIS1']), dtype=np.int16)
        ref_hdu = pyfits.PrimaryHDU(header=_h, data=_data)
        if output is not None:
            ref_hdu.writeto(output, overwrite=True, output_verify='fix')
        
        return ref_hdu
        
    if filters is not None:
        use = utils.column_values_in_list(info['FILTER'], filters)
        if use.sum() == 0:
            # All files
            files = info['FILE']
        else:
            files = info['FILE'][use]

    else:
        files = info['FILE']

    # Just ACS, pixel scale 0.03
    if pixel_scale is None:
        # Auto determine pixel size, 0.03" pixels if only ACS, otherwise 0.06
        any_grism = utils.column_values_in_list(info['FILTER'],
                            ['G800L', 'G102', 'G141', 'GR150C', 'GR150R'])
        acs_grism = (info['FILTER'] == 'G800L')
        only_acs = list(np.unique(info['INSTRUME'])) == ['ACS']
        
        if ((acs_grism.sum() == any_grism.sum()) & (any_grism.sum() > 0)) | (only_acs):
            pixel_scale = 0.03
        else:
            pixel_scale = 0.06

    ref_hdu = utils.make_maximal_wcs(files, pixel_scale=pixel_scale, 
                                     get_hdu=get_hdu, pad=pad_reference, 
                                     verbose=True)

    if get_hdu:
        ref_hdu.data = ref_hdu.data.astype(np.int16)

        if output is not None:
            ref_hdu.writeto(output, overwrite=True, output_verify='fix')

        return ref_hdu
    else:
        return ref_hdu[1]


def drizzle_overlaps(field_root, filters=['F098M', 'F105W', 'F110W', 'F115W', 'F125W', 'F140W', 'F150W', 'F160W', 'F200W'], ref_image=None, ref_wcs=None, bits=None, pixfrac=0.75, scale=0.06, make_combined=False, drizzle_filters=True, skysub=False, skymethod='localmin', match_str=[], context=False, pad_reference=60, min_nexp=2, static=True, skip_products=[], include_saturated=False, multi_driz_cr=False, filter_driz_cr=False, **kwargs):
    """
    Drizzle filter groups based on precomputed image associations
    """

    import numpy as np
    import glob

    try:
        from .. import prep, utils
    except:
        from grizli import prep

    ##############
    # Redrizzle

    #visits, all_groups, info = np.load('{0}_visits.npy'.format(field_root),
    #                                   allow_pickle=True)
    visits, all_groups, info = load_visit_info(field_root, verbose=False)

    failed_list = glob.glob('*failed')

    #overlaps = np.load('{0}_overlaps.npy'.format(field_root))[0]
    #keep = []

    if make_combined:
        if isinstance(make_combined, str):
            label = make_combined
        else:
            label = 'ir'
    else:
        label = 'ir'

    wfc3ir = {'product': '{0}-{1}'.format(field_root, label), 'files': []}

    if ref_image is not None:
        wfc3ir['reference'] = ref_image

    if ref_wcs is not None:
        wfc3ir['reference_wcs'] = ref_wcs

    filter_groups = {}

    for visit in visits:
        msg = (visit)                   
        utils.log_comment(utils.LOGFILE, msg, show_date=True,
                              verbose=True)
        
        # Visit failed for some reason
        _failed = (visit['product']+'.wcs_failed' in failed_list)
        _failed |= (visit['product']+'.failed' in failed_list)
        _failed |= (visit['product'] in skip_products)
        
        if _failed:
            msg = ('visit failed')                   
            utils.log_comment(utils.LOGFILE, msg, show_date=True,
                              verbose=True)
            continue

        # Too few exposures (i.e., one with unreliable CR flags)
        if len(visit['files']) < min_nexp:
            msg = ('visit has too few exposures')                   
            utils.log_comment(utils.LOGFILE, msg, show_date=True,
                              verbose=True)
            continue

        # Not one of the desired filters
        filt = visit['product'].split('-')[-1]
        if filt == 'clear':
            filt = visit['product'].split('-')[-2]
            
        if filt.upper() not in filters:
            msg = ('filt.upper not in filters')                   
            utils.log_comment(utils.LOGFILE, msg, show_date=True,
                              verbose=True)
            continue

        # Are all of the exposures in ./?
        has_exposures = True
        for file in visit['files']:
            has_exposures &= os.path.exists('../Prep/'+file)

        if not has_exposures:
            print('Visit {0} missing exposures, skip'.format(visit['product']))
            continue

        # IS UVIS?
        if visit['files'][0].startswith('i') & ('_flc' in visit['files'][0]):
            filt += 'u'
            is_uvis = True
        else:
            is_uvis = False

        if len(match_str) > 0:
            has_match = False
            for m in match_str:
                has_match |= m in visit['product']

            if not has_match:
                continue

        if filt not in filter_groups:
            filter_groups[filt] = {'product': f'{field_root}-{filt}',
                                   'files': [],
                                   'reference': ref_image,
                                   'reference_wcs': ref_wcs}

        filter_groups[filt]['files'].extend(visit['files'])

        # Add polygon
        if 'footprints' in visit:
            for fp in visit['footprints']:
                if 'footprint' in filter_groups[filt]:
                    fpun = filter_groups[filt]['footprint'].union(fp)
                    filter_groups[filt]['footprint'] = fpun
                else:
                    filter_groups[filt]['footprint'] = fp.buffer(0)

        if (filt.upper() in filters) | (is_uvis & (filt.upper()[:-1] in filters)):
            wfc3ir['files'].extend(visit['files'])
            if 'footprint' in filter_groups[filt]:
                fp_i = filter_groups[filt]['footprint']
                if 'footprint' in wfc3ir:
                    wfc3ir['footprint'] = wfc3ir['footprint'].union(fp_i)
                else:
                    wfc3ir['footprint'] = fp_i.buffer(0)

    if len(filter_groups) == 0:
        print('No filters found ({0})'.format(filters))
        return None

    keep = [filter_groups[k] for k in filter_groups]

    if (ref_image is None) & (ref_wcs is None):
        print('\nCompute mosaic WCS: {0}_wcs-ref.fits\n'.format(field_root))

        ref_hdu = utils.make_maximal_wcs(wfc3ir['files'],
                                         pixel_scale=scale,
                                         get_hdu=True,
                                         pad=pad_reference,
                                         verbose=True)

        ref_hdu.writeto('{0}_wcs-ref.fits'.format(field_root), overwrite=True,
                        output_verify='fix')

        wfc3ir['reference'] = '{0}_wcs-ref.fits'.format(field_root)
        for i in range(len(keep)):
            keep[i]['reference'] = '{0}_wcs-ref.fits'.format(field_root)

    if ref_wcs is not None:
        pass

    if make_combined:

        # Figure out if we have more than one instrument
        inst_keys = np.unique([os.path.basename(file)[0]
                               for file in wfc3ir['files']])

        prep.drizzle_overlaps([wfc3ir], parse_visits=False, 
                              pixfrac=pixfrac, scale=scale, skysub=False,
                              bits=bits, final_wcs=True, final_rot=0,
                              final_outnx=None, final_outny=None,
                              final_ra=None, final_dec=None,
                              final_wht_type='IVM', final_wt_scl='exptime',
                              check_overlaps=False, context=context,
                              static=(static & (len(inst_keys) == 1)),
                              include_saturated=include_saturated,
                              run_driz_cr=multi_driz_cr, **kwargs)

        # np.save('{0}.npy'.format(wfc3ir['product']), [wfc3ir])
        wout = visit_dict_to_strings(wfc3ir)
        with open('{0}.yaml'.format(wfc3ir['product']), 'w') as fp:
            yaml.dump([wout], stream=fp, Dumper=yaml.Dumper)

    if drizzle_filters:
        print('Drizzle mosaics in filters: {0}'.format(filter_groups.keys()))
        prep.drizzle_overlaps(keep, parse_visits=False,
                              pixfrac=pixfrac, scale=scale, skysub=skysub,
                              skymethod=skymethod, bits=bits, final_wcs=True,
                              final_rot=0, final_outnx=None, final_outny=None,
                              final_ra=None, final_dec=None,
                              final_wht_type='IVM', final_wt_scl='exptime',
                              check_overlaps=False, context=context,
                              static=static,
                              include_saturated=include_saturated,
                              run_driz_cr=filter_driz_cr, **kwargs)


FILTER_COMBINATIONS = {'ir': IR_M_FILTERS+IR_W_FILTERS,
                       'opt': OPT_M_FILTERS+OPT_W_FILTERS}


def make_filter_combinations(root, weight_fnu=2, filter_combinations=FILTER_COMBINATIONS, force_photfnu=1.e-8, min_count=1, block_filters=[]):
    """
    Combine ir/opt mosaics manually scaling a specific zeropoint
    """
    from astropy.nddata import block_reduce

    # Output normalization os F814W/F140W
    ref_h = {}
    ref_h['opt'] = {'INSTRUME': 'ACS', 'DETECTOR': 'WFC',
                    'PHOTFLAM': 7.0178627203125e-20,
                    'PHOTBW': 653.24393453125, 'PHOTZPT': -21.1,
                    'PHOTMODE': 'ACS WFC1 F814W MJD#56438.5725',
                    'PHOTPLAM': 8045.415190625002,
                    'FILTER1': 'CLEAR1L', 'FILTER2': 'F814W'}

    ref_h['ir'] = {'INSTRUME': 'WFC3', 'DETECTOR': 'IR',
                   'PHOTFNU': 9.5291135e-08,
                   'PHOTFLAM': 1.4737148e-20,
                   'PHOTBW': 1132.39, 'PHOTZPT': -21.1,
                   'PHOTMODE': 'WFC3 IR F140W',
                   'PHOTPLAM': 13922.907, 'FILTER': 'F140W'}
                   
    if force_photfnu is not None:
        for k in ref_h:
            ref_h[k]['PHOTFNU'] = force_photfnu
            
    ####
    count = {}
    num = {}
    den = {}
    for f in filter_combinations:
        num[f] = None
        den[f] = None
        count[f] = 0

    output_sci = {}
    head = {}

    sci_files = glob.glob('{0}-[cf]*sci.fits*'.format(root))
    sci_files.sort()
    
    for _isci, sci_file in enumerate(sci_files):
        #filt_i = sci_file.split('_dr')[0].split('-')[-1]
        #filt_ix = sci_file.split('_dr')[0].split('-')[-1]
        filt_i = sci_file.split(root+'-')[1].split('_dr')[0]
        filt_ix = sci_file.split(root+'-')[1].split('_dr')[0]
        
        # UVIS
        if filt_i.startswith('f') & filt_i.endswith('u'):
            filt_i = filt_i[:-1]

        band = None
        for f in filter_combinations:
            if filt_i.upper() in filter_combinations[f]:
                band = f
                break

        if band is None:
            continue

        # Which reference parameters to use?
        if filt_i.upper() in OPT_W_FILTERS + OPT_M_FILTERS:
            ref_h_i = ref_h['opt']
        elif 'clear' in filt_i:
            ref_h_i = ref_h['ir']      
        else:
            ref_h_i = ref_h['ir']

        output_sci[band] = sci_file.replace(filt_ix, band)

        im_i = pyfits.open(sci_file)
        wht_i = pyfits.open(sci_file.replace('_sci', '_wht'))
        photflam = im_i[0].header['PHOTFLAM']
        ref_photflam = ref_h_i['PHOTFLAM']

        photplam = im_i[0].header['PHOTPLAM']
        ref_photplam = ref_h_i['PHOTPLAM']
        
        if weight_fnu == 2:
            photflam = im_i[0].header['PHOTFNU']
            ref_photflam = ref_h_i['PHOTFNU']

            photplam = 1.0
            ref_photplam = 1.0
        
        if band not in head:
            head[band] = im_i[0].header.copy()
        else:
            for k in im_i[0].header:
                head[band][k] = im_i[0].header[k]
                
            for k in ref_h_i:
                head[band][k] = ref_h_i[k]
        
        scl = photflam/ref_photflam
        if weight_fnu:
            scl_weight = photplam**2/ref_photplam**2
        else:
            scl_weight = 1.
        
        if 'NCOMP' in head[band]:
            head[band]['NCOMP'] += 1
        else:
            head[band]['NCOMP'] = (1, 'Number of combined images')
        
        _sci = im_i[0].data.astype(np.float32)
        _wht = wht_i[0].data.astype(np.float32)
        
        if filt_i.lower() in [b.lower() for b in block_filters]:
            _blocked = True
            blocked_wht = block_reduce(_wht, 2) / 4**2
            blocked_sci = block_reduce(_sci*_wht, 2) / blocked_wht / 4
            _sci = blocked_sci
            _sci[blocked_wht <= 0] = 0
            _wht = blocked_wht
            
        else:
            _blocked = False
                  
        msg = f'{sci_file} {filt_i} {band} block_reduce={_blocked}'
        msg += f' scl={scl} scl_wht={scl_weight}'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        
        if num[band] is None:
            num[band] = np.zeros_like(_sci)
            den[band] = np.zeros_like(_sci)
        
        den_i = _wht/scl**2*scl_weight
        num[band] += _sci*scl*den_i
        den[band] += den_i
        count[band] += 1

        head[band][f'CFILE{count[band]}'] = (os.path.basename(sci_file), 
                                         'Component file')
        head[band][f'CSCAL{count[band]}'] = (scl,                                                                     
                                        'Scale factor applied in combination')
        head[band][f'CFILT{count[band]}'] = (filt_i,                                                                     
                                         'Filter derived from the filename')

    # Done, make outputs
    for band in filter_combinations:
        if (num[band] is not None) & (count[band] >= min_count):
            sci = num[band]/den[band]
            wht = den[band]

            mask = (~np.isfinite(sci)) | (den == 0)
            sci[mask] = 0
            wht[mask] = 0

            print('Write {0}'.format(output_sci[band]))

            pyfits.PrimaryHDU(data=sci, header=head[band]).writeto(output_sci[band], overwrite=True, output_verify='fix')
            pyfits.PrimaryHDU(data=wht, header=head[band]).writeto(output_sci[band].replace('_sci', '_wht'), overwrite=True, output_verify='fix')


def make_combined_mosaics(root, fix_stars=False, mask_spikes=False, skip_single_optical_visits=True, ir_wcsref_file=None, ir_wcs=None, mosaic_args=args['mosaic_args'], mosaic_driz_cr_type=0, mosaic_drizzle_args=args['mosaic_drizzle_args'], **kwargs):
    """
    Drizzle combined mosaics

    mosaic_driz_cr_type : int
        (mosaic_driz_cr_type & 1) : flag CRs on all IR combined
        (mosaic_driz_cr_type & 2) : flag CRs on IR filter combinations
        (mosaic_driz_cr_type & 4) : flag CRs on all OPT combined
        (mosaic_driz_cr_type & 8) : flag CRs on OPT filter combinations
    """

    # Load visits info
    visits, groups, info = load_visit_info(root, verbose=False)

    # Mosaic WCS
    if ir_wcsref_file is None:
        ir_wcsref_file = f'{root}_wcs-ref.fits'
        
    if (not os.path.exists(ir_wcsref_file)):
        make_reference_wcs(info, output=ir_wcsref_file, get_hdu=True,
                           force_wcs=ir_wcs,
                           **mosaic_args['wcs_params'])

    mosaic_pixfrac = mosaic_args['mosaic_pixfrac']
    combine_all_filters = mosaic_args['combine_all_filters']

    drizzle_overlaps(root,
                     filters=mosaic_args['ir_filters'],
                     min_nexp=1,
                     pixfrac=mosaic_pixfrac,
                     make_combined=False,
                     ref_image=ir_wcsref_file,
                     ref_wcs=None,
                     include_saturated=fix_stars,
                     multi_driz_cr=(mosaic_driz_cr_type & 1) > 0,
                     filter_driz_cr=(mosaic_driz_cr_type & 2) > 0, 
                     **mosaic_drizzle_args)

    make_filter_combinations(root, weight_fnu=True, min_count=1,
                        filter_combinations={'ir': IR_M_FILTERS+IR_W_FILTERS})

    # Mask diffraction spikes
    ir_mosaics = glob.glob('{0}-f*drz_sci.fits'.format(root))
    if (len(ir_mosaics) > 0) & (mask_spikes):
        cat = prep.make_SEP_catalog('{0}-ir'.format(root), threshold=4,
                                    save_fits=False,
                                    column_case=str.lower)

        selection = (cat['mag_auto'] < 18) & (cat['flux_radius'] < 4.5)
        selection |= (cat['mag_auto'] < 15.2) & (cat['flux_radius'] < 20)

        # Bright GAIA stars to catch things with bad photometry
        if True:
            print('## Include GAIA stars in spike mask')

            ra_center = np.median(cat['ra'])
            dec_center = np.median(cat['dec'])
            rad_arcmin = np.sqrt((cat['ra']-ra_center)**2*np.cos(cat['dec']/180*np.pi)**2+(cat['dec']-dec_center)**2)*60

            try:
                gaia_tmp = prep.get_gaia_DR2_catalog(ra_center, dec_center,
                               radius=rad_arcmin.max()*1.1, use_mirror=False)
                idx, dr = utils.GTable(gaia_tmp).match_to_catalog_sky(cat)

                gaia_match = (dr.value < 0.5)
                gaia_match &= (gaia_tmp['phot_g_mean_mag'][idx] < 20)
                gaia_match &= (cat['mag_auto'] < 17.5)

                selection |= gaia_match

            except:
                print('## Include GAIA stars in spike mask - failed')
                pass

        # Note: very bright stars could still be saturated and the spikes
        # might not be big enough given their catalog mag

        msg = '\n### mask_spikes: {0} stars\n\n'.format(selection.sum())
        utils.log_comment(utils.LOGFILE, msg, show_date=True,
                          verbose=True)

        if selection.sum() > 0:
            for visit in visits:
                filt = visit['product'].split('-')[-1]
                if filt[:2] in ['f0', 'f1']:
                    mask_IR_psf_spikes(visit=visit, selection=selection,
                                       cat=cat, minR=8, dy=5)

            # Remake mosaics
            drizzle_overlaps(root, filters=mosaic_args['ir_filters'],
                             min_nexp=1,
                             pixfrac=mosaic_pixfrac,
                             make_combined=False,
                             ref_image=ir_wcsref_file,
                             ref_wcs=None,
                             include_saturated=fix_stars, 
                             **mosaic_drizzle_args)

            make_filter_combinations(root, weight_fnu=True, min_count=1,
                        filter_combinations={'ir':IR_M_FILTERS+IR_W_FILTERS})

    # More IR filter combinations for mosaics
    if False:
        extra_combinations = {'h': ['F140W', 'F160W'],
                              'yj': ['F098M', 'F105W', 'F110W', 'F125W']}

        make_filter_combinations(root, weight_fnu=True, min_count=2,
                            filter_combinations=extra_combinations)

    # Optical filters
    mosaics = glob.glob('{0}-ir_dr?_sci.fits'.format(root))

    if (mosaic_args['half_optical_pixscale']):  # & (len(mosaics) > 0):
        # Drizzle optical images to half the pixel scale determined for
        # the IR mosaics.  The optical mosaics can be 2x2 block averaged
        # to match the IR images.

        ref = pyfits.open(ir_wcsref_file)
        if len(ref) > 1:
            _ir_wcs = pywcs.WCS(ref[1].header)
        else:
            _ir_wcs = pywcs.WCS(ref[0].header)
        
        opt_wcs = utils.half_pixel_scale(_ir_wcs)
        h = utils.to_header(opt_wcs)
        
        opt_wcsref_file = f'{root}-opt_wcs-ref.fits'
        data = np.zeros((h['NAXIS2'], h['NAXIS1']), dtype=np.int16)
        pyfits.writeto(opt_wcsref_file, header=h, data=data, overwrite=True)
    else:
        opt_wcsref_file = ir_wcsref_file
            
    if len(mosaics) == 0:
        # Call a single combined mosaic "ir" for detection catalogs, etc.
        make_combined_label = 'ir'
    else:
        # Make a separate optical combined image
        make_combined_label = 'opt'

    drizzle_overlaps(root,
                     filters=mosaic_args['optical_filters'],
                     pixfrac=mosaic_pixfrac,
                     make_combined=False,
                     ref_image=opt_wcsref_file,
                     ref_wcs=None,
                     min_nexp=1+skip_single_optical_visits*1,
                     multi_driz_cr=(mosaic_driz_cr_type & 4) > 0,
                     filter_driz_cr=(mosaic_driz_cr_type & 8) > 0,
                     **mosaic_drizzle_args)

    make_filter_combinations(root, weight_fnu=True, min_count=1,
                            filter_combinations={make_combined_label:
                                                 OPT_M_FILTERS+OPT_W_FILTERS})

    # Fill IR filter mosaics with scaled combined data so they can be used
    # as grism reference
    fill_mosaics = mosaic_args['fill_mosaics']
    if fill_mosaics:
        if fill_mosaics == 'grism':
            # Only fill mosaics if grism filters exist
            has_grism = info.meta['HAS_GRISM']
            if has_grism:
                fill_filter_mosaics(root)
        else:
            fill_filter_mosaics(root)

    # Remove the WCS reference files
    for file in [opt_wcsref_file, ir_wcsref_file]:
        if file is not None:
            if os.path.exists(file):
                os.remove(file)


def make_mosaic_footprints(field_root):
    """
    Make region files where wht images nonzero
    """
    import matplotlib.pyplot as plt
    
    files = glob.glob('{0}-f*dr?_wht.fits'.format(field_root))
    files.sort()

    fp = open('{0}_mosaic.reg'.format(field_root), 'w')
    fp.write('fk5\n')
    fp.close()

    for weight_image in files:
        filt = weight_image.split('_dr')[0].split('-')[-1]

        wave = filt[1:4]
        if wave[0] in '01':
            w = float(wave)*10
        else:
            w = float(wave)

        wint = np.clip(np.interp(np.log10(w/800), [-0.3, 0.3], [0, 1]), 0, 1)

        print(filt, w, wint)

        clr = utils.RGBtoHex(plt.cm.Spectral_r(wint))
        #plt.scatter([0],[0], color=clr, label=filt)

        reg = prep.drizzle_footprint(weight_image, shrink=10, ext=0, outfile=None, label=filt) + ' color={0}\n'.format(clr)

        fp = open('{0}_mosaic.reg'.format(field_root), 'a')
        fp.write(reg)
        fp.close()


def fill_filter_mosaics(field_root):
    """
    Fill field mosaics with the average value taken from other filters so that all images have the same coverage

    Parameters
    ----------
    field_root : str

    """
    import glob
    import os
    import scipy.ndimage as nd

    import astropy.io.fits as pyfits

    mosaic_files = glob.glob('{0}-ir_dr?_sci.fits'.format(field_root))
    mosaic_files += glob.glob('{0}-opt_dr?_sci.fits'.format(field_root))

    if len(mosaic_files) == 0:
        return False

    ir = pyfits.open(mosaic_files[0])

    filter_files = glob.glob('{0}-f[01]*sci.fits'.format(field_root))
    # If not IR filters, try optical
    if len(filter_files) == 0:
        filter_files = glob.glob('{0}-f[5-8]*sci.fits'.format(field_root))

    for file in filter_files:
        print(file)
        sci = pyfits.open(file, mode='update')
        wht = pyfits.open(file.replace('sci', 'wht'))
        mask = wht[0].data == 0
        scale = ir[0].header['PHOTFLAM']/sci[0].header['PHOTFLAM']
        sci[0].data[mask] = ir[0].data[mask]*scale
        sci.flush()

    # Fill empty parts of IR mosaic with optical if both available
    if len(mosaic_files) == 2:
        print('Fill -ir- mosaic with -opt-')
        ir_sci = pyfits.open(mosaic_files[0], mode='update')
        ir_wht = pyfits.open(mosaic_files[0].replace('sci', 'wht'),
                             mode='update')

        opt_sci = pyfits.open(mosaic_files[1])
        opt_wht = pyfits.open(mosaic_files[1].replace('sci', 'wht'))
        opt_sci_data = opt_sci[0].data
        opt_wht_data = opt_wht[0].data

        if opt_sci_data.shape[0] == 2*ir_wht[0].data.shape[0]:
            # Half pixel scale
            kern = np.ones((2, 2))
            num = nd.convolve(opt_sci_data*opt_wht_data, kern)[::2, ::2]
            den = nd.convolve(opt_wht_data, kern)[::2, ::2]
            opt_sci_data = num/den
            opt_sci_data[den <= 0] = 0
            opt_wht_data = den

        mask = ir_wht[0].data == 0
        scale = opt_sci[0].header['PHOTFLAM']/ir_sci[0].header['PHOTFLAM']
        ir_sci[0].data[mask] = opt_sci_data[mask]*scale
        ir_wht[0].data[mask] = opt_wht_data[mask]/scale**2

        ir_sci.flush()
        ir_wht.flush()

    return True

######################
# Objective function for catalog shifts


def _objfun_align(p0, tab, ref_tab, ref_err, shift_only, ret):
    #from grizli.utils import transform_wcs
    from scipy.special import huber
    from scipy.stats import t as student
    from scipy.stats import norm

    import numpy as np
    import matplotlib.pyplot as plt

    from ..utils import transform_wcs

    N = len(tab)

    trans = np.reshape(p0, (N, -1))  # /10.
    #trans[0,:] = [0,0,0,1]
    sh = trans.shape
    if sh[1] == 2:
        # Shift only
        pscl = np.array([10., 10.])
        trans = np.hstack([trans/pscl, np.zeros((N, 1)), np.ones((N, 1))])
    elif sh[1] == 3:
        # Shift + rot
        pscl = np.array([10., 10., 100.])
        trans = np.hstack([trans/pscl, np.ones((N, 1))])
    elif sh[1] == 4:
        # Shift + rot + scale
        pscl = np.array([10., 10., 100., 100])
        trans = trans/pscl

    print(trans)

    #N = trans.shape[0]
    trans_wcs = {}
    trans_rd = {}
    for ix, i in enumerate(tab):
        if (ref_err > 0.1) & (ix == 0):
            trans_wcs[i] = transform_wcs(tab[i]['wcs'], translation=[0, 0], rotation=0., scale=1.)
            trans_rd[i] = trans_wcs[i].all_pix2world(tab[i]['xy'], 1)
        else:
            trans_wcs[i] = transform_wcs(tab[i]['wcs'], translation=list(trans[ix, :2]), rotation=trans[ix, 2]/180*np.pi, scale=trans[ix, 3])
            trans_rd[i] = trans_wcs[i].all_pix2world(tab[i]['xy'], 1)

    # Cosine declination factor
    cosd = np.cos(np.median(trans_rd[i][:, 1]/180*np.pi))

    if ret == 'field':
        for ix, i in enumerate(tab):
            print(tab[i]['wcs'])
            plt.scatter(trans_rd[i][:, 0], trans_rd[i][:, 1], alpha=0.8, marker='x')
            continue
            for m in tab[i]['match_idx']:
                ix, jx = tab[i]['match_idx'][m]
                if m < 0:
                    continue
                else:
                    # continue
                    dx_i = (trans_rd[i][ix, 0] - trans_rd[m][jx, 0])*3600.*cosd
                    dy_i = (trans_rd[i][ix, 1] - trans_rd[m][jx, 1])*3600.
                    for j in range(len(ix)):
                        if j == 0:
                            p = plt.plot(trans_rd[i][j, 0]+np.array([0, dx_i[j]/60.]), trans_rd[i][j, 1]+np.array([0, dy_i[j]/60.]), alpha=0.8)
                            c = p[0].get_color()
                        else:
                            p = plt.plot(trans_rd[i][j, 0]+np.array([0, dx_i[j]/60.]), trans_rd[i][j, 1]+np.array([0, dy_i[j]/60.]), alpha=0.8, color=c)

        return True

    trans_wcs = {}
    trans_rd = {}
    for ix, i in enumerate(tab):
        trans_wcs[i] = transform_wcs(tab[i]['wcs'],
                                     translation=list(trans[ix, :2]),
                                     rotation=trans[ix, 2]/180*np.pi,
                                     scale=trans[ix, 3])
        trans_rd[i] = trans_wcs[i].all_pix2world(tab[i]['xy'], 1)

    dx, dy = [], []
    for i in tab:
        mcount = 0
        for m in tab[i]['match_idx']:
            ix, jx = tab[i]['match_idx'][m]
            if m < 0:
                continue
            else:
                # continue
                dx_i = (trans_rd[i][ix, 0] - trans_rd[m][jx, 0])*3600.*cosd
                dy_i = (trans_rd[i][ix, 1] - trans_rd[m][jx, 1])*3600.
                mcount += len(dx_i)
                dx.append(dx_i/0.01)
                dy.append(dy_i/0.01)

                if ret == 'plot':
                    plt.gca().scatter(dx_i, dy_i, marker='.', alpha=0.1)

        # Reference sources
        if -1 in tab[i]['match_idx']:
            m = -1
            ix, jx = tab[i]['match_idx'][m]

            dx_i = (trans_rd[i][ix, 0] - tab[i]['ref_tab']['ra'][jx])*3600.*cosd
            dy_i = (trans_rd[i][ix, 1] - tab[i]['ref_tab']['dec'][jx])*3600.
            rcount = len(dx_i)
            mcount = np.maximum(mcount, 1)
            rcount = np.maximum(rcount, 1)
            dx.append(dx_i/(ref_err/np.clip(mcount/rcount, 1, 1000)))
            dy.append(dy_i/(ref_err/np.clip(mcount/rcount, 1, 1000)))

            if ret.startswith('plotx') & (ref_err < 0.1):
                plt.gca().scatter(dx_i, dy_i, marker='+', color='k', alpha=0.3, zorder=1000)

    # Residuals
    dr = np.sqrt(np.hstack(dx)**2+np.hstack(dy)**2)

    if ret == 'huber':  # Minimize Huber loss function
        loss = huber(1, dr).sum()*2
        return loss
    elif ret == 'student':  # student-t log prob (maximize)
        df = 2.5  # more power in wings than normal
        lnp = student.logpdf(dr, df, loc=0, scale=1).sum()
        return lnp
    else:  # Normal log prob (maximize)
        lnp = norm.logpdf(dr, loc=0, scale=1).sum()
        return lnp


def get_rgb_filters(filter_list, force_ir=False, pure_sort=False):
    """
    Compute which filters to use to make an RGB cutout

    Parameters
    ----------
    filter_list : list
        All available filters

    force_ir : bool
        Only use IR filters.

    pure_sort : bool
        Don't use preference for red filters, just use order they appear

    Returns
    -------
    rgb_filt : [r, g, b]
        List of filters to use
    """
    from collections import OrderedDict

    # Sort by wavelength
    for_sort = OrderedDict()
    use_filters = []
    ir_filters = []

    # Preferred combinations
    filter_list_lower = [f.lower() for f in filter_list]
    rpref = ['h', 'f160w', 'f140w','f444w-clear','f200w-clear','f410m-clear']
    gpref = ['j', 'yj', 'f125w', 'f110w', 'f105w', 'f098m','f356w-clear',
             'f150w-clear','f300m-clear']
    bpref = ['opt', 'visr', 'visb', 'f814w', 'f814wu', 'f606w', 'f606wu',
             'f775w', 'f850lp', 'f435w',
             'f090w-clear','f070w-clear','f115w-clear',
             'f277w-clear',
             ]
             
    pref_list = [None, None, None]
    has_pref = 0
    for i, pref in enumerate([rpref, gpref, bpref]):
        for f in pref:
            if f in filter_list_lower:
                pref_list[i] = f
                has_pref += 1
                break

    if has_pref == 3:
        print('Use preferred r/g/b combination: {0}'.format(pref_list))
        return pref_list

    for f in filter_list:
        if f == 'ir':
            continue
        elif f == 'opt':
            continue

        if f == 'uv':
            val = 'f00300'
        elif f == 'visb':
            val = 'f00435'
        elif f == 'visr':
            val = 'f00814'
        elif f == 'y':
            val = 'f01000'
        elif f == 'yj':
            val = 'f01100'
        elif f == 'j':
            val = 'f01250'
        elif f == 'h':
            val = 'f01500'
        elif 'clear' in f:
            val = 'f0'+f[1:4]+'0'
        elif f in ['f560w','f770w']:
            val = 'f0'+f[1:4]
            
        elif f in ['f1000w','f1130w','f1280w','f1500w','f1800w',
                   'f2100w','f2550w']:
            val = f[:5]+'0'
        elif f[1] in '01':
            val = 'f0'+f[1:4]+'0'
        else:
            val = 'f00'+f[1:4]

        # Red filters (>6000)
        if val > 'f07':
            if (val >= 'v09') & (force_ir):
                ir_filters.append(f)

        use_filters.append(f)
        for_sort[f] = val

    pop_indices = []

    joined = {'uv': '23', 'visb': '45', 'visr': '678',
               'y': ['f098m', 'f105w', 'f070w','f090w'],
               'j': ['f110w', 'f125w','f115w'],
               'h': ['f140w', 'f160w','f150w']}

    for j in joined:
        if j in use_filters:
            indices = []
            for f in use_filters:
                if f in joined:
                    continue

                if j in 'yjh':
                    if f in joined[j]:
                        indices.append(use_filters.index(f))
                else:
                    if f[1] in joined[j]:
                        indices.append(use_filters.index(f))

            if len(indices) == len(use_filters)-1:
                # All filters are in a given group so pop the group
                pop_indices.append(use_filters.index(j))
            else:
                pop_indices.extend(indices)

    pop_indices.sort()
    for i in pop_indices[::-1]:
        filt_i = use_filters.pop(i)
        for_sort.pop(filt_i)

    # Only one filter
    if len(use_filters) == 1:
        f = use_filters[0]
        return [f, f, f]

    if len(filter_list) == 1:
        f = filter_list[0]
        return [f, f, f]

    if (len(use_filters) == 0) & (len(filter_list) > 0):
        so = np.argsort(filter_list)
        f = filter_list[so[-1]]
        return [f, f, f]

    # Preference for red filters
    if (len(ir_filters) >= 3) & (not pure_sort):
        use_filters = ir_filters
        for k in list(for_sort.keys()):
            if k not in ir_filters:
                p = for_sort.pop(k)

    so = np.argsort(list(for_sort.values()))
    waves = np.cast[float]([for_sort[f][1:] for f in for_sort])

    # Reddest
    rfilt = use_filters[so[-1]]

    # Bluest
    bfilt = use_filters[so[0]]

    if len(use_filters) == 2:
        return [rfilt, 'sum', bfilt]
    elif len(use_filters) == 3:
        gfilt = use_filters[so[1]]
        return [rfilt, gfilt, bfilt]
    else:
        # Closest to average wavelength
        mean = np.mean([waves.max(), waves.min()])
        ix_g = np.argmin(np.abs(waves-mean))
        gfilt = use_filters[ix_g]
        return [rfilt, gfilt, bfilt]

TICKPARAMS = dict(axis='both', colors='w', which='both')

def field_rgb(root='j010514+021532', xsize=8, output_dpi=None, HOME_PATH='./', show_ir=True, pl=1, pf=1, scl=1, scale_ab=None, rgb_scl=[1, 1, 1], ds9=None, force_ir=False, filters=None, add_labels=True, output_format='jpg', rgb_min=-0.01, xyslice=None, pure_sort=False, verbose=True, force_rgb=None, suffix='.field', mask_empty=False, tick_interval=60, timestamp=False, mw_ebv=0, use_background=False, tickparams=TICKPARAMS, fill_black=False, ref_spectrum=None, gzext='', full_dimensions=False, invert=False, get_rgb_array=False, get_images=False):
    """
    RGB image of the field mosaics
    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    #import montage_wrapper
    from astropy.visualization import make_lupton_rgb

    try:
        from .. import utils
    except:
        from grizli import utils

    if HOME_PATH is not None:
        phot_file = '{0}/{1}/Prep/{1}_phot.fits'.format(HOME_PATH, root)
        if not os.path.exists(phot_file):
            print('Photometry file {0} not found.'.format(phot_file))
            return False

        phot = utils.GTable.gread(phot_file)
        sci_files = glob.glob('{0}/{1}/Prep/{1}-[ofuvyjh]*sci.fits{2}'.format(HOME_PATH, root, gzext))

        PATH_TO = '{0}/{1}/Prep'.format(HOME_PATH, root)
    else:
        PATH_TO = './'
        sci_files = glob.glob('./{1}-[fuvyjho]*sci.fits{2}'.format(PATH_TO, root, gzext))

    print('PATH: {0}, files:{1}'.format(PATH_TO, sci_files))

    if filters is None:
        filters = [file.split('_')[-3].split('-')[-1] for file in sci_files]
        if show_ir:
            filters += ['ir']

    #mag_auto = 23.9-2.5*np.log10(phot['flux_auto'])

    ims = {}
    for f in filters:
        try:
            img = glob.glob('{0}/{1}-{2}_dr?_sci.fits{3}'.format(PATH_TO, root, f, gzext))[0]
        except:
            print('Failed: {0}/{1}-{2}_dr?_sci.fits{3}'.format(PATH_TO, root, f, gzext))

        try:
            ims[f] = pyfits.open(img)
            if 'IMGMED' in ims[f][0].header:
                imgmed = ims[f][0].header['IMGMED']
                ims[f][0].data -= imgmed
            else:
                imgmed = 0
                
            bkg_file = img.split('_dr')[0]+'_bkg.fits'
            if use_background & os.path.exists(bkg_file):
                print('Subtract background: '+bkg_file)
                bkg = pyfits.open(bkg_file)
                ims[f][0].data -= bkg[0].data - imgmed
                
        except:
            continue

    filters = list(ims.keys())

    wcs = pywcs.WCS(ims[filters[-1]][0].header)
    pscale = utils.get_wcs_pscale(wcs)
    minor = MultipleLocator(tick_interval/pscale)

    if force_rgb is None:
        rf, gf, bf = get_rgb_filters(filters, force_ir=force_ir, pure_sort=pure_sort)
    else:
        rf, gf, bf = force_rgb
    
    _rgbf = [rf, gf, bf]
    _rgb_auto = False
    if rgb_scl == [1, 1, 1]:
        if (_rgbf == ['f444w-clear', 'f356w-clear', 'f277w-clear']):
            rgb_scl = [2.7, 2.0, 1.0]
        elif _rgbf == ['f200w-clear', 'f150w-clear', 'f115w-clear']:
            rgb_scl = [1.1, 1., 1.0]
        elif _rgbf == ['f444w-clear', 'f356w-clear', 'f115w-clear']:
            rgb_scl = [1.4, 1., 0.35]
        elif _rgbf == ['f444w-clear', 'f356w-clear', 'f090w-clear']:
            rgb_scl = [1.4, 1., 0.35]
        elif _rgbf == ['f444w-clear', 'f356w-clear', 'f150w-clear']:
            rgb_scl = [1.4, 1., 0.35]
        elif _rgbf == ['f444w-clear', 'f356w-clear', 'f200w-clear']:
            rgb_scl = [1.4, 1., 0.35]
            
    logstr = '# field_rgb {0}: r {1} / g {2} / b {3}\n'
    logstr += '# field_rgb scl={7:.2f} / r {4:.2f} / g {5:.2f} / b {6:.2f}'
    logstr = logstr.format(root, *_rgbf, *rgb_scl, scl)
    utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

    #pf = 1
    #pl = 1

    if scale_ab is not None:
        zp_r = utils.calc_header_zeropoint(ims[rf], ext=0)
        scl = 10**(-0.4*(zp_r-5-scale_ab))

    scl *= (0.06/pscale)**2

    if mw_ebv > 0:
        MW_F99 = utils.MW_F99(mw_ebv*utils.MW_RV, r_v=utils.MW_RV)
    else:
        MW_F99 = None

    rimg = ims[rf][0].data * (ims[rf][0].header['PHOTFLAM']/5.e-20)**pf * (ims[rf][0].header['PHOTPLAM']/1.e4)**pl*scl*rgb_scl[0]

    if MW_F99 is not None:
        rmw = 10**(0.4*(MW_F99(np.array([ims[rf][0].header['PHOTPLAM']]))))[0]
        print('MW_EBV={0:.3f}, {1}: {2:.2f}'.format(mw_ebv, rf, rmw))
        rimg *= rmw

    if bf == 'sum':
        bimg = rimg
    elif bf == rf:
        bimg = rimg
    else:
        bimg = ims[bf][0].data * (ims[bf][0].header['PHOTFLAM']/5.e-20)**pf * (ims[bf][0].header['PHOTPLAM']/1.e4)**pl*scl*rgb_scl[2]
        if MW_F99 is not None:
            bmw = 10**(0.4*(MW_F99(np.array([ims[bf][0].header['PHOTPLAM']]))))[0]
            print('MW_EBV={0:.3f}, {1}: {2:.2f}'.format(mw_ebv, bf, bmw))
            bimg *= bmw
    
    # Double-acs
    if bimg.shape != rimg.shape:
        import scipy.ndimage as nd
        kern = np.ones((2, 2))
        bimg = nd.convolve(bimg, kern)[::2, ::2]

    if gf == 'sum':
        gimg = (rimg+bimg)/2.
    elif gf == rf:
        gimg = rimg
    elif gf == bf:
        gimg = bimg
    else:
        gscl = (ims[gf][0].header['PHOTFLAM']/5.e-20)**pf
        gscl *= (ims[gf][0].header['PHOTPLAM']/1.e4)**pl
                    
        gimg = ims[gf][0].data * gscl * scl * rgb_scl[1]  # * 1.5
        if MW_F99 is not None:
            gmw = 10**(0.4*(MW_F99(np.array([ims[gf][0].header['PHOTPLAM']]))))[0]
            print('MW_EBV={0:.3f}, {1}: {2:.2f}'.format(mw_ebv, gf, gmw))
            gimg *= gmw
    
    rmsk = rimg == 0
    gmsk = gimg == 0
    bmsk = bimg == 0
    
    if gimg.shape != rimg.shape:
        import scipy.ndimage as nd
        kern = np.ones((2, 2))
        gimg = nd.convolve(gimg, kern)[::2, ::2]
        gmsk = gmsk[::2,::2]
        
    # Scale by reference synphot spectrum
    if ref_spectrum is not None:
        import pysynphot as S
        try:
            _obsm = [utils.get_filter_obsmode(filter=_f) for _f in [rf, gf, bf]]
            _bp = [S.ObsBandpass(_m) for _m in _obsm]
            _bpf = [ref_spectrum.integrate_filter(_b)/_b.pivot()**2 for _b in _bp]
            gimg *= _bpf[0]/_bpf[1]
            bimg *= _bpf[0]/_bpf[2]
            print('ref_spectrum supplied: {0}*{1:.2f} {2}*{3:.2f}'.format(gf, _bpf[0]/_bpf[1], bf, _bpf[0]/_bpf[2]))
        except:
            pass
            
    if mask_empty:
        mask = rmsk | gmsk | bmsk
        print('Mask empty pixels in any channel: {0}'.format(mask.sum()))

        rimg[mask] = 0
        gimg[mask] = 0
        bimg[mask] = 0

    if ds9:

        ds9.set('rgb')
        ds9.set('rgb channel red')
        wcs_header = utils.to_header(pywcs.WCS(ims[rf][0].header))
        ds9.view(rimg, header=wcs_header)
        ds9.set_defaults()
        ds9.set('cmap value 9.75 0.8455')

        ds9.set('rgb channel green')
        ds9.view(gimg, wcs_header)
        ds9.set_defaults()
        ds9.set('cmap value 9.75 0.8455')

        ds9.set('rgb channel blue')
        ds9.view(bimg, wcs_header)
        ds9.set_defaults()
        ds9.set('cmap value 9.75 0.8455')

        ds9.set('rgb channel red')
        ds9.set('rgb lock colorbar')

        return False

    xsl = ysl = None

    if show_ir & (not full_dimensions):
        # Show only area where IR is available
        yp, xp = np.indices(ims[rf][0].data.shape)
        wht = pyfits.open(ims[rf].filename().replace('_sci', '_wht'))
        mask = wht[0].data > 0
        xsl = slice(xp[mask].min(), xp[mask].max())
        ysl = slice(yp[mask].min(), yp[mask].max())

        rimg = rimg[ysl, xsl]
        bimg = bimg[ysl, xsl]
        gimg = gimg[ysl, xsl]
        
        if fill_black:
            rmsk = rmsk[ysl, xsl]
            gmsk = gmsk[ysl, xsl]
            bmsk = bmsk[ysl, xsl]
        
    else:
        if xyslice is not None:
            xsl, ysl = xyslice
            rimg = rimg[ysl, xsl]
            bimg = bimg[ysl, xsl]
            gimg = gimg[ysl, xsl]

            if fill_black:
                rmsk = rmsk[ysl, xsl]
                gmsk = gmsk[ysl, xsl]
                bmsk = bmsk[ysl, xsl]
    
    if get_images:
        return (rimg, gimg, bimg), (rf, gf, bf)
                    
    image = make_lupton_rgb(rimg, gimg, bimg, stretch=0.1, minimum=rgb_min)
    if invert:
        image = 255-image
        
    if fill_black:
        image[rmsk,0] = 0
        image[gmsk,1] = 0
        image[bmsk,2] = 0
        
    if get_rgb_array:
        return image
    
    sh = image.shape
    ny, nx, _ = sh
            
    if full_dimensions:
        dpi = int(nx/xsize)
        xsize = nx/dpi
        #print('xsize: ', xsize, ny, nx, dpi)
        
    elif (output_dpi is not None):
        xsize = nx/output_dpi

    dim = [xsize, xsize/nx*ny]

    fig, ax = plt.subplots(1,1,figsize=dim)

    ax.imshow(image, origin='lower', extent=(-nx/2, nx/2, -ny/2, ny/2))

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.xaxis.set_major_locator(minor)
    ax.yaxis.set_major_locator(minor)

    #ax.tick_params(axis='x', colors='w', which='both')
    #ax.tick_params(axis='y', colors='w', which='both')
    if tickparams:
        ax.tick_params(**tickparams)

    if add_labels:
        ax.text(0.03, 0.97, root, bbox=dict(facecolor='w', alpha=0.8), size=10, ha='left', va='top', transform=ax.transAxes)

        ax.text(0.06+0.08*2, 0.02, rf, color='r', bbox=dict(facecolor='w', alpha=1), size=8, ha='center', va='bottom', transform=ax.transAxes)
        ax.text(0.06+0.08, 0.02, gf, color='g', bbox=dict(facecolor='w', alpha=1), size=8, ha='center', va='bottom', transform=ax.transAxes)
        ax.text(0.06, 0.02, bf, color='b', bbox=dict(facecolor='w', alpha=1), size=8, ha='center', va='bottom', transform=ax.transAxes)
    
    if timestamp:
        fig.text(0.97, 0.03, time.ctime(), ha='right', va='bottom', fontsize=5, transform=fig.transFigure, color='w')

    if full_dimensions:
        ax.axis('off')
        fig.tight_layout(pad=0)
        dpi = int(nx/xsize/full_dimensions)
        fig.savefig('{0}{1}.{2}'.format(root, suffix, output_format), dpi=dpi)
    else:
        fig.tight_layout(pad=0.1)
        fig.savefig('{0}{1}.{2}'.format(root, suffix, output_format))
    
    return xsl, ysl, (rf, gf, bf), fig

#########


THUMB_RGB_PARAMS = {'xsize': 4,
                    'output_dpi': None,
                    'rgb_min': -0.01,
                    'add_labels': False,
                    'output_format': 'png',
                    'show_ir': False,
                    'scl': 2,
                    'suffix': '.rgb',
                    'mask_empty': False,
                    'tick_interval': 1,
                    'pl': 1,  # 1 for f_lambda, 2 for f_nu
                    }

DRIZZLER_ARGS = {'aws_bucket': False,
                 'scale_ab': 21.5,
                 'subtract_median': False,
                 'theta': 0.,
                 'pixscale': 0.1,
                 'pixfrac': 0.33,
                 'kernel': 'square',
                 'half_optical_pixscale': True,
                 'filters': ['f160w', 'f814w', 'f140w', 'f125w', 'f105w',
                            'f110w', 'f098m', 'f850lp', 'f775w', 'f606w',
                            'f475w', 'f555w', 'f600lp', 'f390w', 'f350lp'],
                 'size': 3,
                 'thumb_height': 1.5,
                 'rgb_params': THUMB_RGB_PARAMS,
                 'remove': False,
                 'include_ir_psf': True,
                 'combine_similar_filters': False,
                 'single_output': True}


def make_rgb_thumbnails(root='j140814+565638', ids=None, maglim=21,
                        drizzler_args=DRIZZLER_ARGS, use_line_wcs=False,
                        remove_fits=False, skip=True, min_filters=2,
                        auto_size=False, size_limits=[4, 15], mag=None,
                        make_segmentation_figure=True):
    """
    Make RGB thumbnails in working directory
    """
    import matplotlib.pyplot as plt
    import astropy.wcs as pywcs
    from grizli.aws import aws_drizzler

    phot_cat = glob.glob('../Prep/{0}_phot.fits'.format(root))[0]
    cat = utils.read_catalog(phot_cat)

    if make_segmentation_figure:
        plt.ioff()

        seg_files = glob.glob('../*/{0}*seg.fits*'.format(root))
        if len(seg_files) == 0:
            make_segmentation_figure = False
        else:
            seg = pyfits.open(seg_files[0])
            seg_data = seg[0].data
            seg_wcs = pywcs.WCS(seg[0].header)

            # Randomize seg to get dispersion between neighboring objects
            np.random.seed(hash(root) % (10 ** 8))
            rnd_ids = np.append([0], np.argsort(np.random.rand(len(cat)))+1)
            #rnd_seg = rnd_ids[seg[0].data]
            #phot_xy = seg_wcs.all_world2pix(cat['ra'], cat['dec'], 0)

    # Count filters
    num_filters = 0
    for k in cat.meta:
        if k.startswith('F') & k.endswith('uJy2dn'):
            num_filters += 1

    if min_filters > num_filters:
        print('# make_rgb_thumbnails: only {0} filters found'.format(num_filters))
        return False

    if mag is None:
        auto_mag = 23.9-2.5*np.log10(cat['flux_auto']*cat['tot_corr'])
        # More like surface brightness
        try:
            mag = 23.9-2.5*np.log10(cat['flux_aper_2'])
            mag[~np.isfinite(mag)] = auto_mag[~np.isfinite(mag)]
        except:
            mag = auto_mag

    pixel_scale = cat.meta['ASEC_0']/cat.meta['APER_0']
    sx = (cat['xmax']-cat['xmin'])*pixel_scale
    sy = (cat['ymax']-cat['ymin'])*pixel_scale

    #lim_mag  = 23.9-2.5*np.log10(200*np.percentile(cat['fluxerr_aper_4'], 50))
    #print('limiting mag: ', lim_mag)
    lim_mag = 22.8

    extracted_ids = False

    if ids is None:
        ids = cat['id'][mag < maglim]

    elif ids == 'extracted':
        extracted_ids = True
        # Make thumbnails for extracted objects
        beams_files = glob.glob('../Extractions/*beams.fits')
        if len(beams_files) == 0:
            return False

        beams_files.sort()
        ids = [int(os.path.basename(file).split('_')[-1].split('.beams')[0]) for file in beams_files]

    for id_column in ['id', 'number']:
        if id_column in cat.colnames:
            break

    args = drizzler_args.copy()

    N = len(ids)
    for i, id in enumerate(ids):
        ix = cat[id_column] == id
        label = '{0}_{1:05d}'.format(root, id)

        thumb_files = glob.glob('../*/{0}.thumb.png'.format(label))
        if (skip) & (len(thumb_files) > 0):
            print('\n##\n## RGB thumbnail {0}  ({1}/{2})\n##'.format(label, i+1, N))
            continue

        args['scale_ab'] = np.clip(mag[ix][0]-1, 17, lim_mag)

        # Use drizzled line image for WCS?
        if use_line_wcs:
            line_file = glob.glob('../Extractions/{0}.full.fits'.format(label))

            # Reset
            if 'wcs' in args:
                args.pop('wcs')

            for k in ['pixfrac', 'kernel']:
                if k in drizzler_args:
                    args[k] = drizzler_args[k]

            # Find line extrension
            msg = '\n# Use WCS from {0}[{1},{2}] (pixfrac={3:.2f}, kernel={4})'
            if len(line_file) > 0:
                full = pyfits.open(line_file[0])
                for ext in full:
                    if 'EXTNAME' in ext.header:
                        if ext.header['EXTNAME'] == 'LINE':
                            try:
                                wcs = pywcs.WCS(ext.header)
                                args['wcs'] = wcs
                                args['pixfrac'] = ext.header['PIXFRAC']
                                args['kernel'] = ext.header['DRIZKRNL']

                                print(msg.format(line_file[0],
                                      ext.header['EXTNAME'],
                                      ext.header['EXTVER'], args['pixfrac'],
                                      args['kernel']))
                            except:
                                pass

                            break

        if (auto_size) & ('wcs' not in args):
            s_i = np.maximum(sx[ix][0], sy[ix][0])
            args['size'] = np.ceil(np.clip(s_i,
                                           size_limits[0], size_limits[1]))

            print('\n##\n## RGB thumbnail {0} *size={3}* ({1}/{2})\n##'.format(label, i+1, N, args['size']))
        else:
            print('\n##\n## RGB thumbnail {0}  ({1}/{2})\n##'.format(label, i+1, N))

        aws_drizzler.drizzle_images(label=label,
                         ra=cat['ra'][ix][0], dec=cat['dec'][ix][0],
                         master='local', single_output=True,
                         make_segmentation_figure=False, **args)

        files = glob.glob('{0}.thumb.fits'.format(label))
        blot_seg = None
        if (make_segmentation_figure) & (len(files) > 0):
            th = pyfits.open(files[0], mode='update')
            th_wcs = pywcs.WCS(th[0].header)
            blot_seg = utils.blot_nearest_exact(seg_data, seg_wcs, th_wcs,
                                       stepsize=-1, scale_by_pixel_area=False)

            rnd_seg = rnd_ids[np.cast[int](blot_seg)]*1.
            th_ids = np.unique(blot_seg)

            sh = th[0].data.shape
            yp, xp = np.indices(sh)

            thumb_height = 2.
            fig = plt.figure(figsize=[thumb_height*sh[1]/sh[0], thumb_height])
            ax = fig.add_subplot(111)
            rnd_seg[rnd_seg == 0] = np.nan

            ax.imshow(rnd_seg, aspect='equal', cmap='terrain_r',
                      vmin=-0.05*len(cat), vmax=1.05*len(cat))
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ix = utils.column_values_in_list(cat['number'], th_ids)
            xc, yc = th_wcs.all_world2pix(cat['ra'][ix], cat['dec'][ix], 0)
            xc = np.clip(xc, 0.09*sh[1], 0.91*sh[1])
            yc = np.clip(yc, 0.08*sh[0], 0.92*sh[0])

            for th_id, x_i, y_i in zip(cat['number'][ix], xc, yc):
                if th_id == 0:
                    continue

                ax.text(x_i, y_i, '{0:.0f}'.format(th_id), ha='center', va='center', fontsize=8,  color='w')
                ax.text(x_i, y_i, '{0:.0f}'.format(th_id), ha='center', va='center', fontsize=8,  color='k', alpha=0.95)

            ax.set_xlim(0, sh[1]-1)
            ax.set_ylim(0, sh[0]-1)
            ax.set_axis_off()

            fig.tight_layout(pad=0.01)
            fig.savefig('{0}.seg.png'.format(label))
            plt.close(fig)

            # Append to thumbs file
            seg_hdu = pyfits.ImageHDU(data=np.cast[int](blot_seg), name='SEG')
            th.append(seg_hdu)
            th.writeto('{0}.thumb.fits'.format(label), overwrite=True,
                         output_verify='fix')
            th.close()

        if remove_fits > 0:
            files = glob.glob('{0}*_dr[cz]*fits'.format(label))
            for file in files:
                os.remove(file)


def field_psf(root='j020924-044344', PREP_PATH='../Prep', RAW_PATH='../RAW', EXTRACT_PATH='../Extractions', factors=[1, 2, 4], get_drizzle_scale=True, subsample=256, size=6, get_line_maps=False, raise_fault=False, verbose=True, psf_filters=['F098M', 'F110W', 'F105W', 'F125W', 'F140W', 'F160W'], skip=False, make_fits=True, **kwargs):
    """
    Generate PSFs for the available filters in a given field
    """
    import os
    import glob

    import astropy.wcs as pywcs
    import astropy.io.fits as pyfits

    try:
        from .. import utils
        from ..galfit import psf as gpsf
    except:
        from grizli import utils
        from grizli.galfit import psf as gpsf

    os.chdir(PREP_PATH)

    drz_str = '{0}-ir_dr?_sci.fits'.format(root)
    drz_file = glob.glob(drz_str)
    if len(drz_file) == 0:
        err = f'Reference file {drz_str} not found.'
        if raise_fault:
            raise FileNotFoundError(err)
        else:
            print(err)
            return False
    else:
        drz_file = drz_file[0]

    scale = []
    pixfrac = []
    kernel = []
    labels = []

    # For the line maps
    if get_line_maps:
        args_file = os.path.join(EXTRACT_PATH, f'{root}_fit_args.npy')
        if not os.path.exists(args_file):
            err = 'fit_args.npy not found.'
            if raise_fault:
                raise FileNotFoundError(err)
            else:
                print(err)
                return False

        default = DITHERED_PLINE

        # Parameters of the line maps
        args = np.load(args_file, allow_pickle=True)[0]

        # Line images
        pline = args['pline']
        for factor in factors:
            if 'pixscale' in pline:
                scale.append(pline['pixscale']/factor)
            else:
                scale.append(default['pixscale']/factor)

            if 'pixfrac' in pline:
                pixfrac.append(pline['pixfrac'])
            else:
                pixfrac.append(default['pixfrac'])

            if 'kernel' in pline:
                kernel.append(pline['kernel'])
            else:
                kernel.append(default['kernel'])

            labels.append('LINE{0}'.format(factor))

    # Mosaic
    im = pyfits.open(drz_file)
    drz_wcs = pywcs.WCS(im[0].header)
    pscale = utils.get_wcs_pscale(drz_wcs)
    sh = im[0].data.shape

    if get_drizzle_scale:
        #rounded = int(np.round(im[0].header['D001SCAL']*1000))/1000.
        rounded = int(np.round(pscale*1000))/1000.
        if 'D001KERN' in im[0].header:
            kern = im[0].header['D001KERN']
            pixf = im[0].header['D001PIXF']
        else:
            kern = im[0].header['KERNEL']
            pixf = im[0].header['PIXFRAC']
            
        for factor in factors:
            scale.append(rounded/factor)
            labels.append('DRIZ{0}'.format(factor))
            kernel.append(kern)
            pixfrac.append(pixf)

    # FITS info
    #visits_file = '{0}_visits.npy'.format(root)
    visits_file = find_visit_file(root=root)
    if visits_file is None:
        parse_visits(field_root=root, RAW_PATH=RAW_PATH)
    
    visits, groups, info = load_visit_info(root, verbose=False)
    #visits, groups, info = np.load(visits_file, allow_pickle=True)

    # Append "U" to UVIS filters in info
    if 'DETECTOR' in info.colnames:
        uvis = np.where(info['DETECTOR'] == 'UVIS')[0]
        filters = [f for f in info['FILTER']]
        for i in uvis:
            filters[i] += 'U'

        info['FILTER'] = filters

    # Average PSF
    xp, yp = np.meshgrid(np.arange(0, sh[1], subsample), 
                         np.arange(0, sh[0], subsample))
    ra, dec = drz_wcs.all_pix2world(xp, yp, 0)

    # Ref images
    files = glob.glob('{0}-f[0-9]*sci.fits'.format(root))

    if verbose:
        print(' ')

    hdus = []

    for file in files:
        filter = file.split(root+'-')[1].split('_')[0]
        if filter.upper() not in psf_filters:
            continue

        if (os.path.exists('{0}-{1}_psf.fits'.format(root, filter))) & skip:
            continue

        flt_files = list(info['FILE'][info['FILTER'] == filter.upper()])
        if len(flt_files) == 0:
            # Try to use HDRTAB in drizzled image
            flt_files = None
            driz_image = file
        else:
            driz_image = drz_file

        driz_hdu = pyfits.open(file)
        GP = gpsf.DrizzlePSF(flt_files=flt_files, info=None,
                             driz_image=driz_image)

        hdu = pyfits.HDUList([pyfits.PrimaryHDU()])
        hdu[0].header['ROOT'] = root

        for scl, pf, kern_i, label in zip(scale, pixfrac, kernel, labels):
            ix = 0
            psf_f = None

            if pf == 0:
                kern = 'point'
            else:
                kern = kern_i

            logstr = '# psf {0} {5:6} / {1:.3f}" / pixf: {2} / {3:8} / {4}'
            logstr = logstr.format(root, scl, pf, kern, filter, label)
            utils.log_comment(utils.LOGFILE, logstr, verbose=verbose)

            for ri, di in zip(ra.flatten(), dec.flatten()):
                slice_h, wcs_slice = utils.make_wcsheader(ra=ri, dec=di, 
                                              size=size, pixscale=scl, 
                                              get_hdu=False, theta=0)
                
                # Filters with extended profiles
                irext = ['F098M', 'F110W', 'F105W', 'F125W', 'F140W', 'F160W']
                get_extended = filter.upper() in irext
                try:
                    psf_i = GP.get_psf(ra=ri, dec=di, filter=filter.upper(),
                                       pixfrac=pf, kernel=kern, verbose=False,
                                       wcs_slice=wcs_slice,
                                       get_extended=get_extended,
                                       get_weight=True)
                except:
                    continue

                msk_i = (psf_i[1].data != 0)
                msk_i &= np.isfinite(psf_i[1].data)
                if msk_i.sum() == 0:
                    continue

                if ix == 0:
                    # Initialize
                    msk_f = msk_i*1
                    psf_f = psf_i
                    psf_f[1].data[msk_f == 0] = 0
                    ix += 1
                else:
                    # Add to existing
                    msk_f += msk_i*1
                    psf_f[1].data[msk_i > 0] += psf_i[1].data[msk_i > 0]
                    ix += 1

            if psf_f is None:
                msg = 'PSF for {0} (filter={1}) is empty'
                print(msg.format(file, filter))
                continue

            # Average
            psf_f[1].data /= np.maximum(msk_f, 1)

            psf_f[1].header['FILTER'] = filter, 'Filter'
            psf_f[1].header['PSCALE'] = scl, 'Pixel scale, arcsec'
            psf_f[1].header['PIXFRAC'] = pf, 'Pixfrac'
            psf_f[1].header['KERNEL'] = kern, 'Kernel'
            psf_f[1].header['EXTNAME'] = 'PSF'
            psf_f[1].header['EXTVER'] = label

            hdu.append(psf_f[1])

        if make_fits:
            psf_file = '{0}-{1}_psf.fits'.format(root, filter)
            hdu.writeto(psf_file, overwrite=True)

        hdus.append(hdu)

    return hdus


def make_report(root, gzipped_links=True, xsize=18, output_dpi=None, make_rgb=True, mw_ebv=0):
    """
    Make HTML report of the imaging and grism data products
    """
    import glob
    import matplotlib.pyplot as plt
    import astropy.time

    now = astropy.time.Time.now().iso

    plt.ioff()

    os.chdir('../Prep/')

    bfilters = glob.glob('{0}-f[2-8]*sci.fits'.format(root))
    bfilters.sort()

    rfilters = glob.glob('{0}-f[01]*sci.fits'.format(root))
    rfilters.sort()
    filters = [f.split('-')[-1].split('_dr')[0] for f in bfilters + rfilters]

    if len(filters) == 0:
        has_mosaics = False
        #visits, groups, info = np.load('{0}_visits.npy'.format(root),
        #                               allow_pickle=True)
        visits, groups, info = load_visit_info(root, verbose=False)
        
        filters = np.unique([v['product'].split('-')[-1] for v in visits])
    else:
        has_mosaics = True

    if make_rgb & has_mosaics:
        field_rgb(root, HOME_PATH=None, xsize=xsize, output_dpi=output_dpi, ds9=None, scl=2, suffix='.rgb', timestamp=True, mw_ebv=mw_ebv)
        for filter in filters:
            field_rgb(root, HOME_PATH=None, xsize=18, ds9=None, scl=2, force_rgb=[filter, 'sum', 'sum'], suffix='.'+filter, timestamp=True)
    
    ##
    ## Mosaic table
    ##
    rows = []
    line = 'grep -e " 0 " -e "radec" *{0}*wcs.log > /tmp/{1}.log'
    for filter in filters:
        os.system(line.format(filter.strip('u'), root))
        wcs_files = glob.glob('*{0}*wcs.log'.format(filter))

        wcs = '<pre>'+''.join(open('/tmp/{0}.log'.format(root)).readlines())+'</pre>'
        for file in wcs_files:
            png_url = '<a href={1}>{0}</a>'.format(file, file.replace('.log', '.png').replace('+', '%2B'))
            wcs = wcs.replace(file, png_url)

        try:
            im = pyfits.open(glob.glob('{0}-{1}*sci.fits'.format(root, filter))[0])
            h = im[0].header

            url = '<a href="./{0}">sci</a>'.format(im.filename())
            url += '  '+url.replace('_sci', '_wht').replace('>sci', '>wht')

            if gzipped_links:
                url = url.replace('.fits', '.fits.gz')

            psf_file = '{0}-{1}_psf.fits'.format(root, filter)
            if os.path.exists(psf_file):
                url += ' '+'<a href="./{0}">psf</a>'.format(psf_file)

            row = [filter, url, '{0} {1}'.format(h['NAXIS1'], h['NAXIS2']), '{0:.5f} {1:.5f}'.format(h['CRVAL1'], h['CRVAL2']), h['EXPTIME'], h['NDRIZIM'], wcs, '<a href={0}.{1}.jpg><img src={0}.{1}.jpg height=200px></a>'.format(root, filter)]
        except:
            row = [filter, '--', '--', '--', 0., 0, wcs, '--']

        rows.append(row)
        
    tab = utils.GTable(rows=rows, names=['filter', 'FITS', 'naxis', 'crval', 'exptime', 'ndrizim', 'wcs_log', 'img'], dtype=[str, str, str, str,  float, int, str, str])
    tab['exptime'].format = '.1f'

    tab.write_sortable_html('{0}.summary.html'.format(root), replace_braces=True, localhost=False, max_lines=500, table_id=None, table_class='display compact', css=None, filter_columns=[], buttons=['csv'], toggle=False, use_json=False)

    ## Grism figures
    column_files = glob.glob('*column.png')
    if len(column_files) > 0:
        column_files.sort()
        column_url = '<div>' + ' '.join(['<a href="./{0}"><img src="./{0}" height=100px title="{1}"></a>'.format(f.replace('+', '%2B'), f) for f in column_files]) + '</div>'
    else:
        column_url = ''

    grism_files = glob.glob('../Extractions/*grism*fits*')
    if len(grism_files) > 0:
        grism_files.sort()
        grism_pngs = glob.glob('../Extractions/*grism*png')
        if len(grism_pngs) > 0:
            grism_pngs.sort()
            grism_url = '<div>' + ' '.join(['<a href="./{0}"><img src="./{0}" width=400px title="{1}"></a>'.format(f.replace('+', '%2B'), f) for f in grism_pngs]) + '</div>\n'

        else:
            grism_url = ''

        grism_url += '<pre>'
        grism_url += '\n'.join(['<a href="./{0}">{1}</a>'.format(f.replace('+', '%2B'), f) for f in grism_files])
        grism_url += '\n <a href=../Extractions/{0}-fit.html> {0}-fit.html </a>'.format(root)
        grism_url += '\n <a href="../Extractions/{0}_zhist.png"><img src="../Extractions/{0}_zhist.png" width=400px title="{0}_zhist.png"> </a>'.format(root)
        grism_url += '\n</pre>'
        if gzipped_links:
            grism_url = grism_url.replace('.fits', '.fits.gz')

    else:
        grism_url = ''

    try:
        catalog = glob.glob('{0}-*.cat.fits'.format(root))[0]
    except:
        catalog = 'xxx'

    catroot = catalog.split('.cat.fits')[0]

    root_files = glob.glob('{0}-[ioyh]*fits*'.format(root))
    root_files.sort()

    if gzipped_links:
        gzext = '.gz'
    else:
        gzext = ''

    root_urls = '\n    '.join(['<a href={0}{1}>{0}{1}</a>'.format(f, gzext) for f in root_files])

    body = """

    <h4>{root} </h4>

    {now}<br>

    <a href={root}.exposures.html>Exposure report</a>
    / <a href={root}_expflag.txt>{root}_expflag.txt</a>
    / <a href={root}.auto_script.log.txt>{root}.auto_script.log.txt</a>
    / <a href={root}.auto_script.yml>{root}.auto_script.yml</a>

    <pre>
    {root_urls}
    <a href="{root}_visits.yaml">{root}_visits.yaml</a>
    </pre>

    {column}
    {grism}

    <a href="./{root}.rgb.jpg"><img src="./{root}.rgb.jpg" height=300px></a>
    <a href="https://s3.amazonaws.com/grizli-v1/Master/{root}_footprint.png"><img src="https://s3.amazonaws.com/grizli-v1/Master/{root}_footprint.png" height=300px></a>
    <a href="./{root}_fine.png"><img src="./{root}_fine.png" height=200px></a>
    <br>

    """.format(root=root, column=column_url, grism=grism_url, gz='.gz'*(gzipped_links), now=now, catroot=catroot, root_urls=root_urls)

    lines = open('{0}.summary.html'.format(root)).readlines()
    for i in range(len(lines)):
        if '<body>' in lines[i]:
            break

    lines.insert(i+1, body)
    fp = open('{0}.summary.html'.format(root), 'w')
    fp.writelines(lines)
    fp.close()


def exposure_report(root, log=True):
    """
    Save exposure info to webpage & json file
    """

    if log:
        frame = inspect.currentframe()
        utils.log_function_arguments(utils.LOGFILE, frame,
                                     'auto_script.exposure_report')

    from collections import OrderedDict
    import json

    # Exposures
    #visits, all_groups, info = np.load('{0}_visits.npy'.format(root),
    #                                   allow_pickle=True)
    visits, all_groups, info = load_visit_info(root, verbose=False)

    tab = utils.GTable(info)
    tab.add_index('FILE')

    visit_product = ['']*len(info)
    ramp = ['']*len(info)
    trails = ['']*len(info)
    persnpix = [-1]*len(info)

    tab['complete'] = False

    flt_dict = OrderedDict()

    for visit in visits:
        failed = len(glob.glob('{0}*fail*'.format(visit['product']))) > 0

        for file in visit['files']:

            ix = tab.loc_indices[file]

            if os.path.exists(file):
                fobj = pyfits.open(file)
                fd = utils.flt_to_dict(fobj)
                fd['complete'] = not failed
                flt_dict[file] = fd
                flt_dict['visit'] = visit['product']

                if 'PERSNPIX' in fobj[0].header:
                    persnpix[ix] = fobj[0].header['PERSNPIX']

            visit_product[ix] = visit['product']
            tab['complete'][ix] = not failed

            base = file.split('_')[0]
            ramp_file = '../RAW/{0}_ramp.png'.format(base)

            has_mask = glob.glob('{0}*mask.reg'.format(base))
            if has_mask:
                extra = ' style="border:5px solid red;"'
            else:
                extra = ''

            if os.path.exists(ramp_file):
                ramp[ix] = '<a href="{0}"><img src="{0}" height=180 {1}></a>'.format(ramp_file, extra)

            trails_file = '../RAW/{0}_trails.png'.format(base)
            if os.path.exists(trails_file):
                trails[ix] = '<a href="{0}"><img src="{0}" height=180 {1}></a>'.format(trails_file, extra)

    tab['persnpix'] = persnpix

    tab['product'] = visit_product
    tab['ramp'] = ramp
    tab['trails'] = trails

    tab['EXPSTART'].format = '.3f'
    tab['EXPTIME'].format = '.1f'
    tab['PA_V3'].format = '.1f'

    tab['RA_TARG'].format = '.6f'
    tab['DEC_TARG'].format = '.6f'

    # Turn fileinto a URL
    file_urls = ['<a href="./{0}">{0}</a>'.format(f) for f in tab['FILE']]
    tab['FLT'] = file_urls

    cols = ['FLT']+tab.colnames[1:-1]

    fp = open('{0}_exposures.json'.format(root), 'w')
    json.dump(flt_dict, fp)
    fp.close()

    tab[cols].write_sortable_html('{0}.exposures.html'.format(root), replace_braces=True, localhost=False, max_lines=1e5, table_id=None, table_class='display compact', css=None, filter_columns=[], buttons=['csv'], toggle=True, use_json=False)
