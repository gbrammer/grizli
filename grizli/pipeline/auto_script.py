"""
Automatic processing scripts for grizli
"""

import numpy as np

if False:
    np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')
 
# Only fetch F814W optical data for now
#ONLY_F814W = True
ONLY_F814W = False

def demo():
    """
    Test full pipeline, program #12471
    """
    
    import numpy as np
    import os
    
    from hsaquery import query, overlaps
    from grizli.pipeline import auto_script
    
    # "Parent" query is grism exposures from the WFC3 ERS program (11359)
    parent = query.run_query(box=None, proposid=[11359], instruments=['WFC3', 'ACS'], extensions=['FLT'], filters=['G102','G141'], extra=[])
    
    # Demo: match everything nearby, includes tons of things from GOODS-S
    #extra = query.DEFAULT_EXTRA
    #tabs = overlaps.find_overlaps(parent, buffer_arcmin=0.1, filters=[], proposid=[], instruments=['WFC3','ACS'], extra=extra, close=False)
    
    # Match only the grism visit
    extra = query.DEFAULT_EXTRA+["TARGET.TARGET_NAME LIKE 'WFC3-ERSII-G01'"]
    tabs = overlaps.find_overlaps(parent, buffer_arcmin=0.1, filters=['F098M', 'F140W', 'G102', 'G141'], proposid=[11359], instruments=['WFC3'], extra=extra, close=False)
    
    #HOME_PATH = '/Volumes/Pegasus/Grizli/DemoERS/'
    import os
    HOME_PATH = os.getcwd()
    
    root = 'j033217-274236'
    from grizli.pipeline import auto_script
    auto_script.go(root=root, maglim=[19,20], HOME_PATH=HOME_PATH, reprocess_parallel=True, s3_sync=True, run_fit=False)
    
    # Interactive session
    from grizli.pipeline import auto_script
    HOME_PATH = '/Volumes/Pegasus/Grizli/Automatic'
    maglim = [19,21]
    inspect_ramps=True
    manual_alignment=True
    reprocess_parallel=True
    is_parallel_field=False
    
def get_extra_data(root='j114936+222414', HOME_PATH='/Volumes/Pegasus/Grizli/Automatic', instruments=['WFC3'], filters=['F160W','F140W','F098M','F105W'], radius=2, run_fetch=True, from_mast=True, reprocess_parallel=True, s3_sync=False):
    
    import os
    import glob
    
    import numpy as np
    try:
        from .. import utils
    except:
        from grizli import utils
        
    from hsaquery import query, fetch, fetch_mast
    from hsaquery.fetch import DEFAULT_PRODUCTS
    
    tab = utils.GTable.gread(os.path.join(HOME_PATH, '{0}_footprint.fits'.format(root)))
    
    ra, dec = tab.meta['RA'], tab.meta['DEC']

    fp = np.load(os.path.join(HOME_PATH, '{0}_footprint.npy'.format(root)))[0]
    radius = np.sqrt(fp.area*np.cos(dec/180*np.pi))*60/np.pi
    
    xy = np.array(fp.boundary.convex_hull.boundary.xy)
    dims = np.array([(xy[0].max()-xy[0].min())*np.cos(dec/180*np.pi), xy[1].max()-xy[1].min()])*60
        
    extra = query.run_query(box=[ra, dec, radius], proposid=[], instruments=instruments, extensions=['FLT'], filters=filters, extra=query.DEFAULT_EXTRA)
    
    for k in tab.meta:
        extra.meta[k] = tab.meta[k]
    
    extra.write(os.path.join(HOME_PATH, root, 'extra_data.fits'), format='fits', overwrite=True)
    
    CWD = os.getcwd()
    os.chdir(os.path.join(HOME_PATH, root, 'RAW'))
    
    if run_fetch:
        if from_mast:
            out = fetch_mast.get_from_MAST(extra, inst_products=DEFAULT_PRODUCTS, direct=True, path=os.path.join(HOME_PATH, root, 'RAW'), skip_existing=True)
        else:
                        
            curl = fetch.make_curl_script(extra, level=None, script_name='extra.sh', inst_products={'WFC3/UVIS': ['FLC'], 'WFPC2/WFPC2': ['C0M', 'C1M'], 'WFC3/IR': ['RAW'], 'ACS/WFC': ['FLC']}, skip_existing=True, output_path=os.path.join(HOME_PATH, root, 'RAW'), s3_sync=s3_sync)
    
            os.system('sh extra.sh')
            files = glob.glob('*raw.fits.gz')
            files.extend(glob.glob('*fl?.fits.gz'))
            for file in files:
                print('gunzip '+file)
                os.system('gunzip {0}'.format(file))
                        
    else:
        return extra
        
    remove_bad_expflag(field_root=root, HOME_PATH=HOME_PATH, min_bad=2)

    #### Reprocess the RAWs into FLTs    
    os.system("python -c 'from grizli.pipeline import reprocess; reprocess.reprocess_wfc3ir(parallel={0})'".format(reprocess_parallel))
    
    # Persistence products
    os.chdir(os.path.join(HOME_PATH, root, 'Persistence'))
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
            
        # Ugly callout to shell
        os.system('tar xzvf {0}.tar.gz'.format(root))        
        os.system('rm {0}/*extper.fits {0}/*flt_cor.fits'.format(root))
        os.system('ln -sf {0}/*persist.fits ./'.format(root))

    os.chdir(CWD)
    
def go(root='j010311+131615', maglim=[17,26], HOME_PATH='/Volumes/Pegasus/Grizli/Automatic', inspect_ramps=False, manual_alignment=False, is_parallel_field=False, reprocess_parallel=False, only_preprocess=False, run_fit=True, s3_sync=False):
    """
    Run the full pipeline for a given target
        
    Parameters
    ----------
    root : str
        Rootname of the `~hsaquery` file.
    
    maglim : [min, max]
        Magnitude limits of objects to extract and fit.
    
    """
    import os
    import glob
    import matplotlib.pyplot as plt

    try:
        from .. import prep, utils
        from . import auto_script
    except:
        from grizli import prep, utils
        from grizli.pipeline import auto_script
        
    #import grizli.utils
    
    # Silence numpy and astropy warnings
    utils.set_warnings()
    
    roots = [f.split('_info')[0] for f in glob.glob('*dat')]
    
    exptab = utils.GTable.gread(os.path.join(HOME_PATH, '{0}_footprint.fits'.format(root)))
    
    if False:
        is_parallel_field = 'MALKAN' in [name.split()[0] for name in np.unique(exptab['pi_name'])]
        
    ######################
    ### Download data
    os.chdir(HOME_PATH)
    auto_script.fetch_files(field_root=root, HOME_PATH=HOME_PATH, remove_bad=True, reprocess_parallel=reprocess_parallel, s3_sync=s3_sync)
    
    files=glob.glob('../RAW/*_fl*fits')
    if len(files) == 0:
        print('No FL[TC] files found!')
        return False
        
    if inspect_ramps:
        # Inspect for CR trails
        os.chdir(os.path.join(HOME_PATH, root, 'RAW'))
        os.system("python -c 'from grizli.pipeline.reprocess import inspect; inspect()'")
    
    ######################
    ### Parse visit associations
    os.chdir(os.path.join(HOME_PATH, root, 'Prep'))
    visits, all_groups, info = auto_script.parse_visits(field_root=root, HOME_PATH=HOME_PATH, use_visit=True, combine_same_pa=is_parallel_field)
    
    # Alignment catalogs
    catalogs = ['PS1','SDSS','GAIA','WISE']
    
    #######################
    ### Manual alignment
    if manual_alignment:
        os.chdir(os.path.join(HOME_PATH, root, 'Prep'))
        auto_script.manual_alignment(field_root=root, HOME_PATH=HOME_PATH, skip=True, catalogs=catalogs, radius=15, visit_list=None)

    #####################
    ### Alignment & mosaics    
    os.chdir(os.path.join(HOME_PATH, root, 'Prep'))
    auto_script.preprocess(field_root=root, HOME_PATH=HOME_PATH, make_combined=False, catalogs=catalogs, use_visit=True)
        
    # Fine alignment
    fine_catalogs = ['GAIA','PS1','SDSS','WISE']
    if len(glob.glob('{0}*fine.png'.format(root))) == 0:
        try:
            out = auto_script.fine_alignment(field_root=root, HOME_PATH=HOME_PATH, min_overlap=0.2, stopme=False, ref_err=0.08, catalogs=fine_catalogs, NITER=1, maglim=[17,23], shift_only=True, method='Powell', redrizzle=False, radius=30, program_str=None, match_str=[])
            plt.close()

            # Update WCS headers with fine alignment
            auto_script.update_wcs_headers_with_fine(root)

        except:
            pass
            
    if not os.path.exists('{0}-ir_drz_sci.fits'.format(root)):
        
        ## Make mosaics
        IR_filters = ['F105W', 'F110W', 'F125W', 'F140W', 'F160W', 
                      'F098M', 'F139M', 'F127M', 'F153M']
        auto_script.drizzle_overlaps(root, filters=IR_filters) 
    
        # Fill image mosaics with scaled data so they can be used as
        # grism reference
        auto_script.fill_filter_mosaics(root)
        
        # optical images
        optical_filters = ['F814W', 'F606W', 'F435W', 'F850LP']
        auto_script.drizzle_overlaps(root, filters=optical_filters,
            make_combined=False, ref_image='{0}-ir_drz_sci.fits'.format(root)) 
        
    # Photometric catalog
    if not os.path.exists('{0}_phot.fits'.format(root)):
        tab = auto_script.multiband_catalog(field_root=root)
    
    # Stop if only want to run pre-processing
    if only_preprocess | (len(all_groups) == 0):
        return True
                
    ######################
    ### Grism prep
    files = glob.glob('*GrismFLT.fits')
    if len(files) == 0:
        os.chdir(os.path.join(HOME_PATH, root, 'Prep'))
        grp = auto_script.grism_prep(field_root=root, refine_niter=3)
        
        # Make drizzle model images
        grp.drizzle_grism_models(root=root, kernel='point')
        
        # Free grp object
        del(grp)
    
    ######################
    ### Grism extractions
    os.chdir(os.path.join(HOME_PATH, root, 'Extractions'))
    try:
        test = maglim
    except:
        maglim = [17,23]
    
    if is_parallel_field:
        pline = auto_script.PARALLEL_PLINE
    else:
        pline = auto_script.DITHERED_PLINE
    
    # Make script for parallel processing
    auto_script.generate_fit_params(field_root=root, prior=None, MW_EBV=exptab.meta['MW_EBV'], pline=pline, fit_only_beams=True, run_fit=True, poly_order=7, fsps=True, sys_err = 0.03, fcontam=0.2, zr=[0.1, 3.4], save_file='fit_args.npy')
    
    # Run extractions (and fits)
    auto_script.extract(field_root=root, maglim=maglim, MW_EBV=exptab.meta['MW_EBV'], pline=pline, run_fit=run_fit)
    
    ######################
    ### Summary catalog & webpage
    os.chdir(os.path.join(HOME_PATH, root, 'Extractions'))
    if run_fit:
        auto_script.summary_catalog(field_root=root)

def fetch_files(field_root='j142724+334246', HOME_PATH='/Volumes/Pegasus/Grizli/Automatic/', inst_products={'WFPC2/WFPC2': ['C0M', 'C1M'], 'ACS/WFC': ['FLC'], 'WFC3/IR': ['RAW'], 'WFC3/UVIS': ['FLC']}, remove_bad=True, reprocess_parallel=False, s3_sync=False):
    """
    Fully automatic script
    """
    import os
    import glob
    
    try:
        from hsaquery import query, fetch
    except ImportError as ERR:
        warn = """{0}

    Get it from https://github.com/gbrammer/esa-hsaquery""".format(ERR)

        raise(ImportError(warn))
        
    #import grizli
    try:
        from .. import utils
    except:
        from grizli import utils
        
    for dir in [os.path.join(HOME_PATH, field_root), 
                os.path.join(HOME_PATH, field_root, 'RAW'),
                os.path.join(HOME_PATH, field_root, 'Prep'),
                os.path.join(HOME_PATH, field_root, 'Persistence'),
                os.path.join(HOME_PATH, field_root, 'Extractions')]:
        
        if not os.path.exists(dir):
            os.mkdir(dir)
            
    
    tab = utils.GTable.gread('{0}_footprint.fits'.format(field_root))
    tab = tab[(tab['filter'] != 'F218W')]
    if ONLY_F814W:
        tab = tab[(tab['filter'] == 'F814W') | (tab['instdet'] == 'WFC3/IR')]
    
    # Fetch and preprocess IR backgrounds
    os.chdir(os.path.join(HOME_PATH, field_root, 'RAW'))
    
    curl = fetch.make_curl_script(tab, level=None, script_name='fetch_{0}.sh'.format(field_root), inst_products=inst_products, skip_existing=True, output_path='./', s3_sync=s3_sync)
        
    # Ugly callout to shell
    os.system('sh fetch_{0}.sh'.format(field_root))
    files = glob.glob('*raw.fits.gz')
    files.extend(glob.glob('*fl?.fits.gz'))
    for file in files:
        print('gunzip '+file)
        os.system('gunzip {0}'.format(file))
    
    if remove_bad:
        remove_bad_expflag(field_root=field_root, HOME_PATH=HOME_PATH, min_bad=2)
    
    #### Reprocess the RAWs into FLTs    
    if reprocess_parallel:
        os.system("python -c 'from grizli.pipeline import reprocess; reprocess.reprocess_wfc3ir(parallel={0})'".format(reprocess_parallel))
    else:
        from grizli.pipeline import reprocess
        reprocess.reprocess_wfc3ir(parallel=False)
        
    # Persistence products
    os.chdir(os.path.join(HOME_PATH, field_root, 'Persistence'))
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
    if len(files) == 0:
        return False
    
    expf = utils.header_keys_from_filelist(files, keywords=['EXPFLAG'], 
                                           ext=0, colname_case=str.upper)
    expf.write('expflag.info', format='csv', overwrite=True)
    
    # os.system('dfits *raw.fits *flc.fits | fitsort EXPFLAG | sed "s/\t/ , /"> expflag.info')
    # expf = utils.GTable.gread('expflag.info', format='csv')
    
    visit_name = np.array([file[:6] for file in expf['FILE']])
    visits = np.unique(visit_name)
    
    for visit in visits:
        bad = (visit_name == visit) & (expf['EXPFLAG'] != 'NORMAL')
        if bad.sum() > min_bad:
            print('Found bad visit: {0}, N={1}\n'.format(visit, bad.sum()))
            if not os.path.exists('Expflag'):
                os.mkdir('Expflag')
            
            os.system('mv {0}* Expflag/'.format(visit))
            

def parse_visits(field_root='', HOME_PATH='./', use_visit=True, combine_same_pa=True, is_dash=False):
    import os
    import glob
    import copy

    import numpy as np
    import astropy.io.fits as pyfits
    
    from shapely.geometry import Polygon
    from scipy.spatial import ConvexHull

    #import grizli.prep
    try:
        from .. import prep, utils
    except:
        from grizli import prep, utils
            
    files=glob.glob('../RAW/*fl[tc].fits')
    info = utils.get_flt_info(files)
    #info = info[(info['FILTER'] != 'G141') & (info['FILTER'] != 'G102')]
    
    # Only F814W on ACS
    if ONLY_F814W:
        info = info[((info['INSTRUME'] == 'WFC3') & (info['DETECTOR'] == 'IR')) | (info['FILTER'] == 'F814W')]
    
    if is_dash:
        # DASH visits split by exposure
        ima_files=glob.glob('../RAW/*ima.fits')
        ima_files.sort()
        
        visits = []
        for file in ima_files:
            # Build from IMA filename
            root=os.path.basename(file).split("_ima")[0][:-1]
            im = pyfits.open(file)
            filt = utils.get_hst_filter(im[0].header).lower()
            
            # q_flt.fits is the pipeline product.  will always be 
            # fewer DASH-split files
            files=glob.glob('../RAW/%s*[a-o]_flt.fits' %(root))
            if len(files) == 0:
                continue

            files = [os.path.basename(file) for file in files]
            direct = {'product': '{0}-{1}'.format(root, filt), 
                      'files':files}
            visits.append(direct)
        
        all_groups = utils.parse_grism_associations(visits)
        np.save('{0}_visits.npy'.format(field_root), [visits, all_groups, info])
        return True
        
    visits, filters = utils.parse_flt_files(info=info, uniquename=True, get_footprint=True, use_visit=use_visit)
    
    if combine_same_pa:
        combined = {}
        for visit in visits:
            filter_pa = '-'.join(visit['product'].split('-')[-2:])
            prog = '-'.join(visit['product'].split('-')[-4:-3])
            key = 'i{0}-{1}'.format(prog, filter_pa)
            if key not in combined:
                combined[key] = {'product':key, 'files':[], 'footprint':visit['footprint']}
            
            combined[key]['files'].extend(visit['files'])
        
        visits = [combined[k] for k in combined]
        
    all_groups = utils.parse_grism_associations(visits)
    
    print('\n == Grism groups ==\n')
    for g in all_groups:
        print(g['direct']['product'], len(g['direct']['files']), g['grism']['product'], len(g['grism']['files']))
        
    np.save('{0}_visits.npy'.format(field_root), [visits, all_groups, info])
    
    return visits, all_groups, info
def manual_alignment(field_root='j151850-813028', HOME_PATH='/Volumes/Pegasus/Grizli/Automatic/', skip=True, radius=5., catalogs=['PS1','SDSS','GAIA','WISE'], visit_list=None, radec=None):
    
    #import pyds9
    import glob
    import os
    import numpy as np
    
    #import grizli
    from ..prep import get_radec_catalog
    from .. import utils, prep, ds9
        
    files = glob.glob('*guess')
    if (len(files) > 0) & skip:
        return True
        
    tab = utils.GTable.gread('{0}/{1}_footprint.fits'.format(HOME_PATH, field_root))
    
    visits, all_groups, info = np.load('{0}_visits.npy'.format(field_root))
    
    if radec is None:
        radec, ref_catalog = get_radec_catalog(ra=np.mean(tab['ra']),
                    dec=np.median(tab['dec']), 
                    product=field_root,
                    reference_catalogs=catalogs, radius=radius)
    
    ds9 = ds9.DS9()
    ds9.set('mode pan')
    ds9.set('scale zscale')
    ds9.set('scale log')
    
    for visit in visits:
        if visit_list is not None:
            if visit['product'] not in visit_list:
                continue
                
        filt = visit['product'].split('-')[-1]
        if (not filt.startswith('g')):
            prep.manual_alignment(visit, reference='{0}/{1}_{2}.reg'.format(os.getcwd(), field_root, ref_catalog.lower()), ds9=ds9)
        
    ds9.set('quit')

def clean_prep(field_root='j142724+334246'):
    """
    Clean unneeded files after the field preparation
    """
    import glob
    import os

    visits, all_groups, info = np.load('{0}_visits.npy'.format(field_root))

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
                
def preprocess(field_root='j142724+334246', HOME_PATH='/Volumes/Pegasus/Grizli/Automatic/', min_overlap=0.2, make_combined=True, catalogs=['PS1','SDSS','GAIA','WISE'], use_visit=True, master_radec=None, use_first_radec=False, skip_imaging=False, clean=True):
    
    import os
    import glob
    import numpy as np
    import grizli

    from shapely.geometry import Polygon
    from scipy.spatial import ConvexHull
    import copy

    #import grizli.prep
    try:
        from .. import prep, utils
    except:
        from grizli import prep, utils
        
    files=glob.glob('../RAW/*fl[tc].fits')
    visits, all_groups, info = np.load('{0}_visits.npy'.format(field_root))
    
    # Grism visits
    master_footprint = None
    radec = None
    
    for i in range(len(all_groups)):
        direct = all_groups[i]['direct']
        grism = all_groups[i]['grism']

        print(i, direct['product'], len(direct['files']), grism['product'], len(grism['files']))
        
        if len(glob.glob(grism['product']+'_dr?_sci.fits')) > 0:
            continue
        
        if master_radec is not None:
            radec = master_radec
            best_overlap = 0.
        else:
            radec = None
            best_overlap = 0
            radec_files = glob.glob('*cat.radec')
            fp = direct['footprint']
            for rdfile in radec_files:
                points = np.loadtxt(rdfile)
                hull = ConvexHull(points)
                rd_fp = Polygon(points[hull.vertices,:])                
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
                            radec=radec, skip_direct=False,
                            align_mag_limits=[14,22],
                            reference_catalogs=catalogs)
        
        ###################################
        # Persistence Masking
        for file in direct['files']+grism['files']:
            print(file)
            pfile = '../Persistence/'+file.replace('_flt', '_persist')
            if os.path.exists(pfile):
                prep.apply_persistence_mask(file, path='../Persistence',
                                     dq_value=1024, err_threshold=0.6,
                                     grow_mask=3, verbose=True)
        
    # From here, `radec` will be the radec file from the first grism visit
    #master_radec = radec
    
    if skip_imaging:
        return True
        
    ### Ancillary visits
    imaging_visits = []
    for visit in visits:
        filt = visit['product'].split('-')[-1]
        if (len(glob.glob(visit['product']+'_dr?_sci.fits')) == 0) & (not filt.startswith('g1')):
            imaging_visits.append(visit)
    
    filters = [v['product'].split('-')[-1] for v in visits]
    fwave = np.cast[float]([f.replace('f1','f10').replace('f0','f00').replace('lp','w')[1:-1] for f in filters])
    sort_idx = np.argsort(fwave)[::-1]
    
    for i in sort_idx:
        direct = visits[i]
        if 'g800l' in direct['product']:
            continue
        
        # Skip singleton optical visits
        if (fwave[i] < 900) & (len(direct['files']) == 1):
            print('Only one exposure, skip', direct['product'])
            continue
            
        if len(glob.glob(direct['product']+'_dr?_sci.fits')) > 0:
            print('Skip', direct['product'])
            continue
        else:
            print(direct['product'])
        
        if master_radec is not None:
            radec = master_radec
            best_overlap = 0
            fp = direct['footprint']
        else:
            radec = None
            best_overlap = 0
            radec_n = 0
            radec_files = glob.glob('*cat.radec')
            fp = direct['footprint']
            for rdfile in radec_files:
                points = np.loadtxt(rdfile)
                hull = ConvexHull(points)
                rd_fp = Polygon(points[hull.vertices,:])                
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
                                        skip_direct=False,
                                        run_tweak_align=True,
                                        align_mag_limits=[14,24],
                                        reference_catalogs=catalogs,
                                        align_tolerance=8)
            except:
                status = prep.process_direct_grism_visit(direct=direct,
                                            grism={}, radec=radec,
                                            skip_direct=False,
                                            run_tweak_align=False,
                                            align_mag_limits=[14,24],
                                            reference_catalogs=catalogs,
                                            align_tolerance=8)
                
            failed_file = '%s.failed' %(direct['product'])
            if os.path.exists(failed_file):
                os.remove(failed_file)
            
            ###################################
            # Persistence Masking
            for file in direct['files']:
                print(file)
                pfile = '../Persistence/'+file.replace('_flt', '_persist')
                if os.path.exists(pfile):
                    prep.apply_persistence_mask(file, path='../Persistence',
                                         dq_value=1024, err_threshold=0.6,
                                         grow_mask=3, verbose=True)
            
        except:
            fp = open('%s.failed' %(direct['product']), 'w')
            fp.write('\n')
            fp.close()
    
    # ###################################
    # # Persistence Masking
    # for visit in visits:
    #     for file in visit['files']:
    #         print(file)
    #         if os.path.exists('../Persistence/'+file.replace('_flt', '_persist')):
    #             prep.apply_persistence_mask(file, path='../Persistence',
    #                                  dq_value=1024, err_threshold=0.6,
    #                                  grow_mask=3, verbose=True)
    
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
    # prep.drizzle_overlaps([wfc3ir], parse_visits=False, pixfrac=0.6, scale=0.06, skysub=False, bits=Nonoe, final_wcs=True, final_rot=0, final_outnx=None, final_outny=None, final_ra=None, final_dec=None, final_wht_type='IVM', final_wt_scl='exptime', check_overlaps=False)
    #             
    # prep.drizzle_overlaps(keep, parse_visits=False, pixfrac=0.6, scale=0.06, skysub=False, bits=None, final_wcs=True, final_rot=0, final_outnx=None, final_outny=None, final_ra=None, final_dec=None, final_wht_type='IVM', final_wt_scl='exptime', check_overlaps=False)

def multiband_catalog(field_root='j142724+334246', threshold=1.8, get_background=True):
    """
    Make a detection catalog with SExtractor and then measure
    photometry with `~photutils`.
    """
    import glob
    import numpy as np
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from photutils import segmentation, background
    import photutils.utils
    
    #import grizli
    #import grizli.prep
    try:
        from .. import prep, utils
    except:
        from grizli import prep, utils
            
    # Make catalog
    tab = prep.make_SEP_catalog(root='{0}-ir'.format(field_root), threshold=threshold, get_background=get_background, save_to_fits=True)
        
    # Source positions
    source_xy = tab['X_IMAGE'], tab['Y_IMAGE']
    
    files=glob.glob('../RAW/*fl[tc].fits')
    info = utils.get_flt_info(files)
    
    if ONLY_F814W:
        info = info[((info['INSTRUME'] == 'WFC3') & (info['DETECTOR'] == 'IR')) | (info['FILTER'] == 'F814W')]
    
    filters = [f.lower() for f in np.unique(info['FILTER'])]
    
    #filters.insert(0, 'ir')
    
    segment_img = pyfits.open('{0}-ir_seg.fits'.format(field_root))[0].data
    
    for ii, filt in enumerate(filters):
        print(filt)
        if filt.startswith('g'):
            continue
                
        if filt not in ['g102','g141','g800l']:
            sci_files = glob.glob(('{0}-{1}_dr?_sci.fits'.format(field_root, filt)))
            if len(sci_files) == 0:
                continue
            
            root = '{0}-{1}'.format(field_root, filt)
                
            filter_tab = prep.make_SEP_catalog(root=root,
                      threshold=threshold, get_background=get_background,
                      save_to_fits=False, source_xy=source_xy)
            
            for k in filter_tab.meta:
                newk = '{0}_{1}'.format(filt.upper(), k)
                tab.meta[newk] = filter_tab.meta[k]
            
            for c in filter_tab.colnames:
                newc = '{0}_{1}'.format(filt.upper(), c)         
                tab[newc] = filter_tab[c]

        else:
            continue
        
    
    for c in tab.colnames:
        tab.rename_column(c, c.lower())
    
    tab.write('{0}_phot.fits'.format(field_root), format='fits', overwrite=True)
    
    return tab   
          
def photutils_catalog(field_root='j142724+334246', threshold=1.8, subtract_bkg=True):
    """
    Make a detection catalog with SExtractor and then measure
    photometry with `~photutils`.
    """
    import glob
    import numpy as np
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from photutils import segmentation, background
    import photutils.utils
    
    #import grizli
    #import grizli.prep
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
    
    sexcat = sexcat['number','mag_auto','flux_radius']

    files=glob.glob('../RAW/*fl[tc].fits')
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
        
        if filt not in ['g102','g141']:
            sci_files = glob.glob(('{0}-{1}_dr?_sci.fits'.format(field_root, filt)))
            if len(sci_files) == 0:
                continue
            else:
                sci_file=sci_files[0]
                
            sci = pyfits.open(sci_file)
            wht = pyfits.open(sci_file.replace('_sci','_wht'))
        else:
            continue
        
        photflam = sci[0].header['PHOTFLAM']
        ABZP = (-2.5*np.log10(sci[0].header['PHOTFLAM']) - 21.10 -
                   5*np.log10(sci[0].header['PHOTPLAM']) + 18.6921)
                 
        bkg_err = 1/np.sqrt(wht[0].data)
        bkg_err[~np.isfinite(bkg_err)] = 0#1e30        
        total_error = photutils.utils.calc_total_error(sci[0].data, bkg_err, sci[0].header['EXPTIME'])
        
        wht_mask = (wht[0].data == 0) | (sci[0].data == 0)    
        sci[0].data[wht[0].data == 0] = 0
            
        mask = None #bkg_err > 1.e29
        
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
        else:
            bkg_obj = None
            
        cat = segmentation.source_properties(sci[0].data, segment_img, error=total_error, mask=mask, background=bkg_obj, filter_kernel=None, wcs=pywcs.WCS(sci[0].header), labels=None)
        
        if False:
            obj = cat[0]
            seg_cutout = obj.make_cutout(segment_img)
            morph = statmorph.source_morphology(obj.data_cutout, segmap=(seg_cutout == obj.id)*1, variance=obj.error_cutout_ma**2)[0]#, psf=psf)
            
        if filt == 'ir':
            cols = ['id', 'xcentroid', 'ycentroid', 'sky_centroid', 'sky_centroid_icrs', 'source_sum', 'source_sum_err', 'xmin', 'xmax', 'ymin', 'ymax', 'min_value', 'max_value', 'minval_xpos', 'minval_ypos', 'maxval_xpos', 'maxval_ypos', 'area', 'equivalent_radius', 'perimeter', 'semimajor_axis_sigma', 'semiminor_axis_sigma', 'eccentricity', 'orientation', 'ellipticity', 'elongation', 'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'cxx', 'cxy', 'cyy']
            tab = utils.GTable(cat.to_table(columns=cols))
            cols = ['source_sum', 'source_sum_err']
            for c in cols:
                tab[c.replace('sum','flam')] = tab[c]*photflam            
        else:
            cols = ['source_sum', 'source_sum_err']
            t_i = cat.to_table(columns=cols)
            
            mask = (np.isfinite(t_i['source_sum_err']))
            for c in cols:
                tab['{0}_{1}'.format(filt, c)] = t_i[c]
                tab['{0}_{1}'.format(filt, c)][~mask] = np.nan
                cflam = c.replace('sum','flam')
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
    
def load_GroupFLT(field_root='j142724+334246', force_ref=None, force_seg=None, force_cat=None, galfit=False, pad=256, files=None):
    """
    Initialize a GroupFLT object
    """
    import glob
    import os
    import numpy as np
    
    from .. import prep, utils, multifit
    
    if files is None:
        files=glob.glob('../RAW/*fl[tc].fits')
    
    info = utils.get_flt_info(files)
    
    g141 = info['FILTER'] == 'G141'
    g102 = info['FILTER'] == 'G102'
    g800l = info['FILTER'] == 'G800L'
                
    if force_cat is None:
        catalog = '{0}-ir.cat.fits'.format(field_root)
    else:
        catalog = force_cat
    
    grp = None
    if g141.sum() > 0:
        for f in ['F140W', 'F160W', 'F125W', 'F105W', 'F110W', 'F098M', 'F127M', 'F139M', 'F153M', 'F132N', 'F130N', 'F128N', 'F126N', 'F164N', 'F167N']:
            
            if os.path.exists('{0}-{1}_drz_sci.fits'.format(field_root, f.lower())):
                g141_ref = f
                break
            # if f in info['FILTER']:
            #     g141_ref = f
            #     break
        
        ## Segmentation image
        if force_seg is None:
            if galfit == 'clean':
                seg_file = '{0}-{1}_galfit_orig_seg.fits'.format(field_root, g141_ref.lower())
            elif galfit == 'model':
                seg_file = '{0}-{1}_galfit_seg.fits'.format(field_root, g141_ref.lower())
            else:
                seg_file = '{0}-ir_seg.fits'.format(field_root)
        else:
            seg_file = force_seg
            
        ## Reference image
        if force_ref is None:
            if galfit == 'clean':
                ref_file = '{0}-{1}_galfit_clean.fits'.format(field_root, g141_ref.lower())
            elif galfit == 'model':
                ref_file = '{0}-{1}_galfit.fits'.format(field_root, g141_ref.lower())
            else:
                ref_file = '{0}-{1}_drz_sci.fits'.format(field_root, g141_ref.lower())
            
        else:
            ref_file = force_ref
        
        grp = multifit.GroupFLT(grism_files=list(info['FILE'][g141]), direct_files=[], ref_file=ref_file, seg_file=seg_file, catalog=catalog, cpu_count=-1, sci_extn=1, pad=pad)
    
    if g102.sum() > 0:
        for f in ['F105W', 'F098M', 'F110W', 'F125W', 'F140W', 'F160W', 'F127M', 'F139M', 'F153M', 'F132N', 'F130N', 'F128N', 'F126N', 'F164N', 'F167N']:
            if os.path.exists('{0}-{1}_drz_sci.fits'.format(field_root, f.lower())):
                g102_ref = f
                break
        
        ## Segmentation image
        if force_seg is None:
            if galfit == 'clean':
                seg_file = '{0}-{1}_galfit_orig_seg.fits'.format(field_root, g102_ref.lower())
            elif galfit == 'model':
                seg_file = '{0}-{1}_galfit_seg.fits'.format(field_root, g102_ref.lower())
            else:
                seg_file = '{0}-ir_seg.fits'.format(field_root)
        else:
            seg_file = force_seg
        
        ## Reference image
        if force_ref is None:
            if galfit == 'clean':
                ref_file = '{0}-{1}_galfit_clean.fits'.format(field_root, g102_ref.lower())
            elif galfit == 'model':
                ref_file = '{0}-{1}_galfit.fits'.format(field_root, g102_ref.lower())
            else:
                ref_file = '{0}-{1}_drz_sci.fits'.format(field_root, g102_ref.lower())
            
        else:
            ref_file = force_ref
                    
        grp_i = multifit.GroupFLT(grism_files=list(info['FILE'][g102]), direct_files=[], ref_file=ref_file, seg_file=seg_file, catalog=catalog, cpu_count=-1, sci_extn=1, pad=pad)
        if g141.sum() > 0:
            grp.extend(grp_i)
        else:
            grp = grp_i
            
        del(grp_i)
    
    # ACS
    if g800l.sum() > 0:
        for f in ['F814W', 'F606W', 'F850LP', 'F435W']:
            if os.path.exists('{0}-{1}_drc_sci.fits'.format(field_root, f.lower())):
                g800l_ref = f
                break
        
        ## Segmentation image
        if force_seg is None:
            if galfit == 'clean':
                seg_file = '{0}-{1}_galfit_orig_seg.fits'.format(field_root, g800l_ref.lower())
            elif galfit == 'model':
                seg_file = '{0}-{1}_galfit_seg.fits'.format(field_root, g800l_ref.lower())
            else:
                seg_file = '{0}-ir_seg.fits'.format(field_root)
        else:
            seg_file = force_seg
        
        ## Reference image
        if force_ref is None:
            if galfit == 'clean':
                ref_file = '{0}-{1}_galfit_clean.fits'.format(field_root, g800l_ref.lower())
            elif galfit == 'model':
                ref_file = '{0}-{1}_galfit.fits'.format(field_root, g800l_ref.lower())
            else:
                ref_file = '{0}-{1}_drc_sci.fits'.format(field_root, g800l_ref.lower())
            
        else:
            ref_file = force_ref
        
        for sci_extn in [1,2]:        
            grp_i = multifit.GroupFLT(grism_files=list(info['FILE'][g800l]), direct_files=[], ref_file=ref_file, seg_file=seg_file, catalog=catalog, cpu_count=-1, sci_extn=sci_extn, pad=pad)
        
            if grp is not None:
                grp.extend(grp_i)
            else:
                grp = grp_i
            
            del(grp_i)
    
    return grp
        
def grism_prep(field_root='j142724+334246', ds9=None, refine_niter=3):
    import glob
    import os
    import numpy as np
    
    from .. import prep, utils, multifit

    grp = load_GroupFLT(field_root=field_root)
    
    ################
    # Compute preliminary model
    grp.compute_full_model(fit_info=None, verbose=True, store=False, mag_limit=25, coeffs=[1.1, -0.5], cpu_count=4)
        
    ################
    # Remove constant modal background 
    import scipy.stats
    
    for i in range(grp.N):
        mask = (grp.FLTs[i].model < grp.FLTs[i].grism['ERR']*0.6) & (grp.FLTs[i].grism['SCI'] != 0)
        
        # Fit Gaussian to the masked pixel distribution
        clip = np.ones(mask.sum(), dtype=bool)
        for iter in range(3):
            n = scipy.stats.norm.fit(grp.FLTs[i].grism.data['SCI'][mask][clip])
            clip = np.abs(grp.FLTs[i].grism.data['SCI'][mask]) < 3*n[1]
        
        mode = n[0]
        
        print(grp.FLTs[i].grism.parent_file, grp.FLTs[i].grism.filter, mode)
        try:
            ds9.view(grp.FLTs[i].grism['SCI'] - grp.FLTs[i].model)
        except:
            pass
        
        ## Subtract
        grp.FLTs[i].grism.data['SCI'] -= mode
    
    #############
    # Refine the model
    i=0
    if ds9:
        ds9.view(grp.FLTs[i].grism['SCI'] - grp.FLTs[i].model)
        fr = ds9.get('frame')
    
    for iter in range(refine_niter):
        if ds9:
            ds9.set('frame {0}'.format(int(fr)+iter+1))
        
        grp.refine_list(poly_order=3, mag_limits=[19, 24], max_coeff=5, ds9=ds9, verbose=True)

    ##############
    # Save model to avoid having to recompute it again
    grp.save_full_data()
    
    # Link minimal files to Extractions directory
    os.chdir('../Extractions/')
    os.system('ln -s ../Prep/*GrismFLT* .')
    os.system('ln -s ../Prep/*-ir.cat.fits .')
    os.system('ln -s ../Prep/*_phot.fits .')
   
    return grp
    
DITHERED_PLINE = {'kernel': 'point', 'pixfrac': 0.2, 'pixscale': 0.1, 'size': 8, 'wcs': None}
PARALLEL_PLINE = {'kernel': 'square', 'pixfrac': 0.8, 'pixscale': 0.1, 'size': 8, 'wcs': None}
  
def extract(field_root='j142724+334246', maglim=[13,24], prior=None, MW_EBV=0.00, ids=None, pline=DITHERED_PLINE, fit_only_beams=True, run_fit=True, poly_order=7):
    import glob
    import os
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    #import grizli
    try:
        from .. import multifit, prep, utils, fitting
    except:
        from grizli import multifit, prep, utils, fitting
        
    grp = multifit.GroupFLT(grism_files=glob.glob('*GrismFLT.fits'), direct_files=[], ref_file=None, seg_file='{0}-ir_seg.fits'.format(field_root), catalog='{0}-ir.cat.fits'.format(field_root), cpu_count=-1, sci_extn=1, pad=256)
    
    ###############
    # PHotometry
    
    target = field_root
    
    if os.path.exists('{0}_phot.fits'.format(target)):
        photom = utils.GTable.gread('{0}_phot.fits'.format(target))
        photom_filters = []
        for c in photom.colnames:
            if c.endswith('_flux_aper_0'):
                photom_filters.append(c.split('_flux_aper_0')[0])
    
        photom_flux = np.vstack([photom['{0}_flux_aper_0'.format(f)].data for f in photom_filters])
        photom_err = np.vstack([photom['{0}_fluxerr_aper_0'.format(f)].data for f in photom_filters])
        photom_pivot = np.array([photom.meta['{0}_PLAM'.format(f.upper())] for f in photom_filters])
    else:
        photom = None
        
    ###########
    # IDs to extract
    #ids=[1096]
    
    if ids is None:
        clip = (grp.catalog['MAG_AUTO'] > maglim[0]) & (grp.catalog['MAG_AUTO'] < maglim[1])
        so = np.argsort(grp.catalog['MAG_AUTO'][clip])
        ids = grp.catalog['NUMBER'][clip][so]
    
    # Stack the different beans
    wave = np.linspace(2000,2.5e4,100)
    poly_templates = utils.polynomial_templates(wave, order=poly_order)
        
    fsps = True
    t0 = utils.load_templates(fwhm=1000, line_complexes=True, stars=False, full_line_list=None, continuum_list=None, fsps_templates=fsps, alf_template=True)
    t1 = utils.load_templates(fwhm=1000, line_complexes=False, stars=False, full_line_list=None, continuum_list=None, fsps_templates=fsps, alf_template=True)
    
    size = 32
    close = True
    Skip = True
    
    if __name__ == '__main__': # Interactive
        size=32
        close = Skip = False
        pline = {'kernel': 'point', 'pixfrac': 0.2, 'pixscale': 0.1, 'size': 8, 'wcs': None}
        prior=None
        
    ###############
    # Stacked spectra
    for id in ids:
        if Skip:
            if os.path.exists('{0}_{1:05d}.stack.png'.format(target, id)):
                continue
            
        beams = grp.get_beams(id, size=size, beam_id='A')
        for i in range(len(beams))[::-1]:
            if beams[i].fit_mask.sum() < 10:
                beams.pop(i)
                
        print(id, len(beams))
        if len(beams) < 1:
            continue
            
        mb = multifit.MultiBeam(beams, fcontam=0.5, group_name=target, psf=False, MW_EBV=MW_EBV)
        
        try:
            pfit = mb.template_at_z(z=0, templates=poly_templates, fit_background=True, fitter='lstsq', get_uncertainties=2)
        except:
            pfit = None
    
        try:
            fig1 = mb.oned_figure(figsize=[5,3], tfit=pfit)
            fig1.savefig('{0}_{1:05d}.1D.png'.format(target, id))
        except:
            continue

            
        hdu, fig = mb.drizzle_grisms_and_PAs(fcontam=0.5, flambda=False, kernel='point', size=32, zfit=pfit)
        fig.savefig('{0}_{1:05d}.stack.png'.format(target, id))

        hdu.writeto('{0}_{1:05d}.stack.fits'.format(target, id), clobber=True)
        mb.write_master_fits()
        
        if close:
            plt.close(fig); plt.close(fig1); del(hdu); del(mb)
    
    if not run_fit:
       return True
        
    ###############
    # Redshift Fit    
    phot = None
    scale_photometry = False
    fit_beams = True
    zr = [0.1, 3.3]
    sys_err = 0.03
    prior=None
    pline = {'kernel': 'point', 'pixfrac': 0.2, 'pixscale': 0.1, 'size': 8, 'wcs': None}
        
    for id in ids:
        if Skip:
            if os.path.exists('{0}_{1:05d}.line.png'.format(target, id)):
                continue
        
        try:
            out = fitting.run_all(id, t0=t0, t1=t1, fwhm=1200, zr=zr, dz=[0.004, 0.0005], fitter='nnls', group_name=target, fit_stacks=False, prior=prior,  fcontam=0.2, pline=pline, mask_sn_limit=10, fit_beams=(not fit_only_beams),  root=target+'*', fit_trace_shift=False, phot=phot, verbose=True, scale_photometry=(phot is not None) & (scale_photometry), show_beams=True, overlap_threshold=10, fit_only_beams=fit_only_beams, MW_EBV=MW_EBV, sys_err=sys_err)
            mb, st, fit, tfit, line_hdu = out
            
            spectrum_1d = [tfit['cont1d'].wave, tfit['cont1d'].flux]
            grp.compute_single_model(id, mag=-99, size=-1, store=False, spectrum_1d=spectrum_1d, get_beams=None, in_place=True, is_cgs=True)
            
            if close:
                for k in range(1000): plt.close()
                
            del(out)
        except:
            pass
    
    # Re-save data with updated models
    grp.save_full_data()
    return grp

def generate_fit_params(field_root='j142724+334246', prior=None, MW_EBV=0.00, pline=DITHERED_PLINE, fit_only_beams=True, run_fit=True, poly_order=7, fsps=True, sys_err = 0.03, fcontam=0.2, zr=[0.1, 3.4], save_file='fit_args.npy'):
    """
    Generate a parameter dictionary for passing to the fitting script
    """
    import numpy as np
    from grizli import utils, fitting
    phot = None
    
    t0 = utils.load_templates(fwhm=1000, line_complexes=True, stars=False, full_line_list=None, continuum_list=None, fsps_templates=fsps, alf_template=True)
    t1 = utils.load_templates(fwhm=1000, line_complexes=False, stars=False, full_line_list=None, continuum_list=None, fsps_templates=fsps, alf_template=True)

    args = fitting.run_all(0, t0=t0, t1=t1, fwhm=1200, zr=zr, dz=[0.004, 0.0005], fitter='nnls', group_name=field_root, fit_stacks=False, prior=prior,  fcontam=fcontam, pline=pline, mask_sn_limit=10, fit_beams=False,  root=field_root, fit_trace_shift=False, phot=phot, verbose=True, scale_photometry=False, show_beams=True, overlap_threshold=10, fit_only_beams=fit_only_beams, MW_EBV=MW_EBV, sys_err=sys_err, get_dict=True)
    
    np.save(save_file, [args])
    print('Saved arguments to {0}.'.format(save_file))
    return args

def set_column_formats(fit):
    """
    Set print formats for the master catalog columns
    """
    fit['ellipticity'].format = '.2f'
    
    for c in ['ra', 'dec']:
        fit[c].format = '.5f'
    
    for col in ['MAG_AUTO', 'FLUX_RADIUS', 'A_IMAGE']:
        fit[col.lower()].format = '.2f'
            
    for c in ['log_risk', 'log_pdf_max', 'zq','chinu', 'bic_diff']:
        fit[c].format = '.2f'
    
    for c in ['z_risk', 'z_map', 'z02', 'z16', 'z50', 'z84', 'z97']: 
        fit[c].format = '.4f'
    
    for c in ['t_g102', 't_g141']: 
        fit[c].format = '.0f'
    
    for c in fit.colnames:
        if c.startswith('flux_'):
            fit[c].format = '.1e'
        
        if c.startswith('err_'):
            fit[c].format = '.1e'
        
        if c.startswith('ew50_'):
            fit[c].format = '.1e'
        
        if c.startswith('ewhw_'):
            fit[c].format = '.1e'
            
        if c.startswith('sn_'):
            fit[c].format = '.1f'
        
        if c.startswith('zwidth'):
            fit[c].format = '.3f'
        
    return fit
    
def summary_catalog(field_root='', dzbin=0.01, use_localhost=True, filter_bandpasses=None):
    """
    Make redshift histogram and summary catalog / HTML table
    """
    import os
    import numpy as np
    from matplotlib.ticker import FixedLocator
    import matplotlib.pyplot as plt
    
    from grizli import utils, fitting
    
    if filter_bandpasses is None:
        import pysynphot as S
        filter_bandpasses = [S.ObsBandpass(bpstr) for bpstr in ['wfc3,ir,f105w', 'wfc3,ir,f110w', 'wfc3,ir,f125w', 'wfc3,ir,f140w', 'wfc3,ir,f160w']]
        
    ### SUmmary catalog
    fit = fitting.make_summary_catalog(target=field_root, sextractor=None,
                                       filter_bandpasses=filter_bandpasses)
                                       
    fit.meta['root'] = field_root
    
    ## Add photometric catalog
    sex = utils.GTable.gread('../Prep/{0}-ir.cat.fits'.format(field_root))
    # try:
    # except:
    #     sex = utils.GTable.gread('../Prep/{0}-ir.cat.fits'.format(field_root), sextractor=True)
        
    idx = np.arange(len(sex))
    sex_idx = np.array([idx[sex['NUMBER'] == id][0] for id in fit['id']])
    
    fit['ellipticity'] = (sex['B_IMAGE']/sex['A_IMAGE'])[sex_idx]
        
    for col in ['MAG_AUTO', 'FLUX_RADIUS', 'A_IMAGE']:
        fit[col.lower()] = sex[col][sex_idx]

    fit = set_column_formats(fit)
    
    # Overwrite with additional sextractor keywords
    fit.write('{0}.info.fits'.format(field_root), overwrite=True)
    
    clip = (fit['chinu'] < 2.0) & (fit['log_risk'] < -1)
    clip = (fit['chinu'] < 2.0) & (fit['zq'] < -3) & (fit['zwidth1']/(1+fit['z_map']) < 0.005)
    clip &= fit['bic_diff'] > -40
    
    bins = utils.log_zgrid(zr=[0.1, 3.5], dz=dzbin)
    
    fig = plt.figure(figsize=[6,4])
    ax = fig.add_subplot(111)
    
    ax.hist(np.log10(1+fit['z_map']), bins=np.log10(1+bins), alpha=0.2, color='k')
    ax.hist(np.log10(1+fit['z_map'][clip]), bins=np.log10(1+bins), alpha=0.8)
    
    xt = np.array(np.arange(0.25, 3.55, 0.25))
    ax.xaxis.set_minor_locator(FixedLocator(np.log10(1+xt)))
    xt = np.array([1,2,3])
    ax.set_xticks(np.log10(1+xt))
    ax.set_xticklabels(xt)
    
    ax.set_xlabel('z')
    ax.set_ylabel(r'$N$')
    
    ax.grid()
    ax.text(0.05, 0.95, field_root, ha='left', va='top', transform=ax.transAxes)
    
    fig.tight_layout(pad=0.2)
    fig.savefig('{0}_zhist.png'.format(field_root))

    cols = ['id','ra', 'dec', 'mag_auto', 't_g102', 't_g141', 'z_map', 'log_risk', 'log_pdf_max', 'zq', 'chinu', 'bic_diff', 'zwidth1', 'png_stack', 'png_full', 'png_line']
    
    fit[cols].write_sortable_html(field_root+'-fit.html', replace_braces=True, localhost=use_localhost, max_lines=50000, table_id=None, table_class='display compact', css=None)
        
    fit[cols][clip].write_sortable_html(field_root+'-fit.zq.html', replace_braces=True, localhost=use_localhost, max_lines=50000, table_id=None, table_class='display compact', css=None)

    if False:
        
        fit = utils.GTable.gread('{0}.info.fits'.format(root))
        fit = auto_script.set_column_formats(fit)
        
        cols = ['id','ra', 'dec', 'mag_auto', 't_g102', 't_g141', 'sn_Ha', 'sn_OIII', 'sn_Hb', 'z_map','log_risk', 'log_pdf_max', 'zq', 'chinu', 'bic_diff', 'zwidth1', 'png_stack', 'png_full', 'png_line']
        
        #clip = ((fit['sn_Ha'] > 5) | (fit['sn_OIII'] > 5)) & (fit['bic_diff'] > 50) & (fit['chinu'] < 2)
        #clip = (fit['sn_OIII'] > 5) & (fit['bic_diff'] > 100) & (fit['chinu'] < 3)
        
        test_line = {}
        for l in ['Ha','OIII','OII']:
            test_line[l] = (fit['sn_'+l] > 5) & (fit['err_'+l] < 1.e-16)
            
        clip = (test_line['Ha'] | test_line['OIII'] | test_line['OII']) & (fit['bic_diff'] > 50) & (fit['chinu'] < 2)
        
        star = fit['flux_radius'] < 2.3
        clip &= ~star
        
        jh = fit['mag_wfc3,ir,f125w'] - fit['mag_wfc3,ir,f160w']
        clip = (fit['chinu'] < 2) & (jh > 0.9) & (fit['mag_wfc3,ir,f160w'] < 23)
        fit['jh'] = jh
        fit['jh'].format = '.1f'
        
        fit['dmag'] = fit['mag_wfc3,ir,f140w'] - fit['mag_auto']
        fit['dmag'].format = '.1f'
        
        cols = ['idx','ra', 'dec', 'mag_auto', 'jh', 'dmag', 't_g141', 'sn_Ha', 'sn_OIII', 'sn_Hb', 'z_map','log_risk', 'log_pdf_max', 'zq', 'chinu', 'bic_diff', 'zwidth1', 'png_stack', 'png_full', 'png_line']
        
        fit[cols][clip].write_sortable_html(root+'-fit.lines.html', replace_braces=True, localhost=False, max_lines=50000, table_id=None, table_class='display compact', css=None)
        
        
        
def fine_alignment(field_root='j142724+334246', HOME_PATH='/Volumes/Pegasus/Grizli/Automatic/', min_overlap=0.2, stopme=False, ref_err = 1.e-3, radec=None, redrizzle=True, shift_only=True, maglim=[17,24], NITER=1, catalogs = ['PS1','SDSS','GAIA','WISE'], method='Powell', radius=5., program_str=None, match_str=[], all_visits=None):
    """
    Try fine alignment from visit-based SExtractor catalogs
    """    
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt

    from shapely.geometry import Polygon
    from scipy.spatial import ConvexHull
    from drizzlepac import updatehdr
        
    try:
        from .. import prep, utils
        from ..prep import get_radec_catalog
        from ..utils import transform_wcs
    except:
        from grizli import prep, utils
        from grizli.prep import get_radec_catalog
        from grizli.utils import transform_wcs
            
    import astropy.units as u
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from scipy.optimize import minimize, fmin_powell
    
    import copy
        
    if all_visits is None:
        all_visits, all_groups, info = np.load('{0}_visits.npy'.format(field_root))
    #all_visits, filters = utils.parse_flt_files(info=info, uniquename=True, get_footprint=False)
    
    visits = []
    files = []
    for visit in all_visits:
        file = '{0}.cat.fits'.format(visit['product'])
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
        radec, ref_catalog = get_radec_catalog(ra=np.mean(info['RA_TARG']),
                    dec=np.median(info['DEC_TARG']), 
                    product=field_root,
                    reference_catalogs=catalogs, radius=radius)
                                 
    #ref = 'j152643+164738_sdss.radec'
    ref_tab = utils.GTable(np.loadtxt(radec, unpack=True).T, names=['ra','dec'])
    ridx = np.arange(len(ref_tab))
            
    # Find matches
    tab = {}
    for i, file in enumerate(files):
        tab[i] = {}
        t_i = utils.GTable.gread(file)
        mclip = (t_i['MAG_AUTO'] > maglim[0]) & (t_i['MAG_AUTO'] < maglim[1])
        if mclip.sum() == 0:
            continue
            
        tab[i]['cat'] = t_i[mclip]
        
        sci_file = glob.glob(file.replace('.cat','_dr?_sci'))[0]        
        im = pyfits.open(sci_file)
        tab[i]['wcs'] = pywcs.WCS(im[0].header)
        
        print(sci_file, mclip.sum())
        
        tab[i]['transform'] = [0, 0, 0, 1]
        tab[i]['xy'] = np.array([tab[i]['cat']['X_IMAGE'], tab[i]['cat']['Y_IMAGE']]).T
        
        tab[i]['match_idx'] = {}
        idx, dr = tab[i]['cat'].match_to_catalog_sky(ref_tab)
        clip = dr < 0.6*u.arcsec
        if clip.sum() > 2:
            tab[i]['match_idx'][-1] = [idx[clip], ridx[clip]]
        
        # ix, jx = tab[i]['match_idx'][-1]
        # ci = tab[i]['cat']#[ix]
        # cj = ref_tab#[jx]
            
    for i, file in enumerate(files):
        for j in range(i+1,len(files)):
            sidx = np.arange(len(tab[j]['cat']))
            idx, dr = tab[i]['cat'].match_to_catalog_sky(tab[j]['cat'])
            clip = dr < 0.3*u.arcsec
            print(file, files[j], clip.sum())

            if clip.sum() < 5:
                continue
            
            if clip.sum() > 0:
                tab[i]['match_idx'][j] = [idx[clip], sidx[clip]]
    
    #ref_err = 0.01
        
    #shift_only=True
    if shift_only: 
        p0 = np.vstack([[0,0] for i in tab])
    else:
        p0 = np.vstack([[0,0,0,1] for i in tab])
        
    #ref_err = 0.06

    if False:
        field_args = (tab, ref_tab, ref_err, shift_only, 'field')
        _objfun_align(p0*10., *field_args)
    
    fit_args = (tab, ref_tab, ref_err, shift_only, 'huber')
    plot_args = (tab, ref_tab, ref_err, shift_only, 'plot')
    plotx_args = (tab, ref_tab, ref_err, shift_only, 'plotx')
    
    pi = p0*10.
    for iter in range(NITER):
        fit = minimize(_objfun_align, pi*10, args=fit_args, method=method, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
        pi = fit.x/10.
        
    ########
    # Show the result
    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot(221)
    _objfun_align(p0*10., *plot_args)
    ax.set_xticklabels([])
    ax.set_ylabel('dDec')
    
    ax = fig.add_subplot(223)
    _objfun_align(p0*10., *plotx_args)
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
    if program_str is not None:
        extra_str += '.{0}'.format(program_str)
    
    if match_str is not []:
        extra_str += '.{0}'.format('.'.join(match_str))
        
    fig.savefig('{0}{1}_fine.png'.format(field_root, extra_str))
    np.save('{0}{1}_fine.npy'.format(field_root, extra_str), [visits, fit])

    return tab, fit, visits
    
    # ######### 
    # ## Update FLT files
    # N = len(visits)
    # 
    # trans = np.reshape(fit.x, (N,-1))/10.
    # #trans[0,:] = [0,0,0,1]
    # sh = trans.shape
    # if sh[1] == 2:
    #     trans = np.hstack([trans, np.zeros((N,1)), np.ones((N,1))])
    # elif sh[1] == 3:
    #     trans = np.hstack([trans, np.ones((N,1))])
    # 
    # if ref_err > 0.1:
    #     trans[0,:] = [0,0,0,1]
    #     
    # if not os.path.exists('FineBkup'):
    #     os.mkdir('FineBkup')
    # 
    # for i in range(N):
    #     direct = visits[i]
    #     for file in direct['files']:
    #         os.system('cp {0} FineBkup/'.format(file))
    #         print(file)
    #         
    # for ix, direct in enumerate(visits):
    #     #direct = visits[ix]
    #     out_shift, out_rot, out_scale = trans[ix,:2], trans[ix,2], trans[ix,3]
    #     for file in direct['files']:
    #         updatehdr.updatewcs_with_shift(file, 
    #                         str('{0}_wcs.fits'.format(direct['product'])),
    #                               xsh=out_shift[0], ysh=out_shift[1],
    #                               rot=out_rot, scale=out_scale,
    #                               wcsname='FINE', force=True,
    #                               reusename=True, verbose=True,
    #                               sciext='SCI')
    # 
    #     ### Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
    #     ### keywords
    #     im = pyfits.open(file, mode='update')
    #     im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
    #     im.flush()
    # 
    # if redrizzle:
    #     drizzle_overlaps(field_root)

def update_wcs_headers_with_fine(field_root, backup=True):
    """
    Update grism headers with the fine shifts
    """    
    import os
    import numpy as np
    import glob
    
    import astropy.io.fits as pyfits
    from drizzlepac import updatehdr
    
    #import grizli.prep
    try:
        from .. import prep
    except:
        from grizli import prep
        
    if backup:
        if not os.path.exists('FineBkup'):
            os.mkdir('FineBkup')

    visits, all_groups, info = np.load('{0}_visits.npy'.format(field_root))
    
    fit_files = glob.glob('{0}*fine.npy'.format(field_root))
    for fit_file in fit_files:
        fine_visits, fine_fit = np.load(fit_file)
    
        N = len(fine_visits)
    
        if backup:
            for i in range(N):
                direct = fine_visits[i]
                for file in direct['files']:
                    os.system('cp {0} FineBkup/'.format(file))
                    print(file)
    
        trans = np.reshape(fine_fit.x, (N,-1))/10.
        sh = trans.shape
        if sh[1] == 2:
            trans = np.hstack([trans, np.zeros((N,1)), np.ones((N,1))])
        elif sh[1] == 3:
            trans = np.hstack([trans, np.ones((N,1))])
        
        # Update direct WCS
        for ix, direct in enumerate(fine_visits):
            #direct = visits[ix]
            out_shift, out_rot = trans[ix,:2], trans[ix,2]
            out_scale = trans[ix,3]
            
            for file in direct['files']:
                updatehdr.updatewcs_with_shift(file, 
                                str('{0}_wcs.fits'.format(direct['product'])),
                                      xsh=out_shift[0], ysh=out_shift[1],
                                      rot=out_rot, scale=out_scale,
                                      wcsname='FINE', force=True,
                                      reusename=True, verbose=True,
                                      sciext='SCI')

            ### Bug in astrodrizzle? Dies if the FLT files don't have MJD-OBS
            ### keywords
            im = pyfits.open(file, mode='update')
            im[0].header['MJD-OBS'] = im[0].header['EXPSTART']
            im.flush()
        
        # Update grism WCS
        for i in range(len(all_groups)):
            direct = all_groups[i]['direct']
            grism = all_groups[i]['grism']
            for j in range(N):
                if fine_visits[j]['product'] == direct['product']:
                    print(direct['product'], grism['product'], trans[j,:])
                    
                    if backup:
                        for file in grism['files']:
                            os.system('cp {0} FineBkup/'.format(file))
                            print(file)
                        
                    prep.match_direct_grism_wcs(direct=direct, grism=grism, 
                                                get_fresh_flt=False, 
                                                xyscale=trans[j,:])
                
        
def drizzle_overlaps(field_root, filters=['F098M','F105W','F110W', 'F125W','F140W','F160W'], ref_image=None, bits=None, pixfrac=0.6, scale=0.06, make_combined=True, skysub=False, skymethod='localmin', match_str=[], context=False):
    import numpy as np

    try:
        from .. import prep
    except:
        from grizli import prep
        
    ##############
    ## Redrizzle
    
    visits, all_groups, info = np.load('{0}_visits.npy'.format(field_root))
    
    #overlaps = np.load('{0}_overlaps.npy'.format(field_root))[0]
    #keep = []
    
    if ref_image is None:
        ref_image = '{0}-ir_drz_sci.fits'.format(field_root)
        wfc3ir = {'product':'{0}-ir'.format(field_root), 'files':[]}
    else:
        wfc3ir = {'product':'{0}-ir'.format(field_root), 'files':[], 'reference':ref_image}
        
    filter_groups = {}
    for visit in visits:
        filt = visit['product'].split('-')[-1]
        if filt.upper() not in filters:
            continue
        
        if len(match_str) > 0:
            has_match = False
            for m in match_str:
                has_match |= m in visit['product']
            
            if not has_match:
                continue
                    
        if filt not in filter_groups:
            filter_groups[filt] = {'product':'{0}-{1}'.format(field_root, filt), 'files':[], 'reference':ref_image}
        
        filter_groups[filt]['files'].extend(visit['files'])
        
        if filt.upper() in filters:
            wfc3ir['files'].extend(visit['files'])
    
    if len(filter_groups) == 0:
        print('No filters found ({0})'.format(filters))
        return None
        
    keep = [filter_groups[k] for k in filter_groups]
                        
    if make_combined:
        prep.drizzle_overlaps([wfc3ir], parse_visits=False, pixfrac=pixfrac, scale=scale, skysub=False, bits=bits, final_wcs=True, final_rot=0, final_outnx=None, final_outny=None, final_ra=None, final_dec=None, final_wht_type='IVM', final_wt_scl='exptime', check_overlaps=False, context=context)
    
    prep.drizzle_overlaps(keep, parse_visits=False, pixfrac=pixfrac, scale=scale, skysub=skysub, skymethod=skymethod, bits=bits, final_wcs=True, final_rot=0, final_outnx=None, final_outny=None, final_ra=None, final_dec=None, final_wht_type='IVM', final_wt_scl='exptime', check_overlaps=False, context=context)
    
def fill_filter_mosaics(field_root):
    """
    Fill field mosaics with the average value taken from other filters so that all images have the same coverage
    
    Parameters
    ----------
    field_root : str

    """ 
    import glob
    import os

    import astropy.io.fits as pyfits
     
    ir_drz = '{0}-ir_drz_sci.fits'.format(field_root)
    if not os.path.exists(ir_drz):
        return False
        
    ir = pyfits.open(ir_drz)
    filter_files = glob.glob('{0}-f[01]*sci.fits'.format(field_root))
    
    for file in filter_files:
        print(file)
        sci = pyfits.open(file, mode='update')
        wht = pyfits.open(file.replace('sci','wht'))
        mask = wht[0].data == 0
        scale = ir[0].header['PHOTFLAM']/sci[0].header['PHOTFLAM']
        sci[0].data[mask] = ir[0].data[mask]*scale
        sci.flush()
        
    return True
    
######################
## Objective function for catalog shifts      
def _objfun_align(p0, tab, ref_tab, ref_err, shift_only, ret):
    #from grizli.utils import transform_wcs
    from scipy.special import huber
    from scipy.stats import t as student
    from scipy.stats import norm
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    from ..utils import transform_wcs
    
    N = len(tab)
    
    trans = np.reshape(p0, (N,-1))/10.
    #trans[0,:] = [0,0,0,1]
    sh = trans.shape
    if sh[1] == 2:
        trans = np.hstack([trans, np.zeros((N,1)), np.ones((N,1))])
    elif sh[1] == 3:
        trans = np.hstack([trans, np.ones((N,1))])
        
    print(trans)
    
    #N = trans.shape[0]
    trans_wcs = {}
    trans_rd = {}
    for ix, i in enumerate(tab):
        if (ref_err > 0.1) & (ix == 0):
            trans_wcs[i] = transform_wcs(tab[i]['wcs'], translation=[0,0], rotation=0., scale=1.) 
            trans_rd[i] = trans_wcs[i].all_pix2world(tab[i]['xy'], 1)
        else:
            trans_wcs[i] = transform_wcs(tab[i]['wcs'], translation=list(trans[ix,:2]), rotation=trans[ix,2], scale=trans[ix,3]) 
            trans_rd[i] = trans_wcs[i].all_pix2world(tab[i]['xy'], 1) 
    
    # Cosine declination factor
    cosd = np.cos(np.median(trans_rd[i][:,1]/180*np.pi))
    
    if ret == 'field':
        for ix, i in enumerate(tab):
            print(tab[i]['wcs'])
            plt.scatter(trans_rd[i][:,0], trans_rd[i][:,1], alpha=0.8, marker='x')
            continue
            for m in tab[i]['match_idx']:
                ix, jx = tab[i]['match_idx'][m]
                if m < 0:
                    continue
                else:
                    #continue
                    dx_i = (trans_rd[i][ix,0] - trans_rd[m][jx,0])*3600.*cosd
                    dy_i = (trans_rd[i][ix,1] - trans_rd[m][jx,1])*3600.
                    for j in range(len(ix)):
                        if j == 0:
                            p = plt.plot(trans_rd[i][j,0]+np.array([0,dx_i[j]/60.]), trans_rd[i][j,1]+np.array([0,dy_i[j]/60.]), alpha=0.8)
                            c = p[0].get_color()
                        else:
                            p = plt.plot(trans_rd[i][j,0]+np.array([0,dx_i[j]/60.]), trans_rd[i][j,1]+np.array([0,dy_i[j]/60.]), alpha=0.8, color=c)

        return True
    
    trans_wcs = {}
    trans_rd = {}
    for ix, i in enumerate(tab):
        trans_wcs[i] = transform_wcs(tab[i]['wcs'], translation=list(trans[ix,:2]), rotation=trans[ix,2], scale=trans[ix,3]) 
        trans_rd[i] = trans_wcs[i].all_pix2world(tab[i]['xy'], 1)
                    
    dx, dy = [], []    
    for i in tab:
        mcount = 0
        for m in tab[i]['match_idx']:
            ix, jx = tab[i]['match_idx'][m]
            if m < 0:
                continue
            else:
                #continue
                dx_i = (trans_rd[i][ix,0] - trans_rd[m][jx,0])*3600.*cosd
                dy_i = (trans_rd[i][ix,1] - trans_rd[m][jx,1])*3600.
                mcount += len(dx_i)
                dx.append(dx_i/0.01)
                dy.append(dy_i/0.01)                
            
                if ret == 'plot':
                    plt.gca().scatter(dx_i, dy_i, marker='.', alpha=0.1)
        
        # Reference sources
        if -1 in tab[i]['match_idx']:
            m = -1
            ix, jx = tab[i]['match_idx'][m]
        
            dx_i = (trans_rd[i][ix,0] - ref_tab['ra'][jx])*3600.*cosd
            dy_i = (trans_rd[i][ix,1] - ref_tab['dec'][jx])*3600.
            rcount = len(dx_i)
            mcount = np.maximum(mcount, 1)
            rcount = np.maximum(rcount, 1)
            dx.append(dx_i/(ref_err/np.clip(mcount/rcount, 1, 1000)))
            dy.append(dy_i/(ref_err/np.clip(mcount/rcount, 1, 1000)))

            if ret.startswith('plotx') & (ref_err < 0.1):
                plt.gca().scatter(dx_i, dy_i, marker='+', color='k', alpha=0.3, zorder=1000)
    
    # Residuals        
    dr = np.sqrt(np.hstack(dx)**2+np.hstack(dy)**2)
    
    if ret == 'huber': # Minimize Huber loss function
        loss = huber(1, dr).sum()*2
        return loss
    elif ret == 'student': #student-t log prob (maximize)
        df = 2.5 # more power in wings than normal
        lnp = student.logpdf(dr, df, loc=0, scale=1).sum()
        return lnp
    else: # Normal log prob (maximize)
        lnp = norm.logpdf(dr, loc=0, scale=1).sum()
        return lnp