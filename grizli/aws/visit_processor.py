#!/usr/bin/env python

"""
Process space telescope visits

2022 version with associations defined in the database in `assoc_table`

"""
import os
import glob
import numpy as np

from . import db

def s3_object_path(dataset, product='raw', ext='fits', base_path='hst/public/'):
    """
    S3 path for an HST ``dataset``
    """
    if dataset.startswith('jw'):
        prog = dataset[:len('jwppppp')].lower()
    else:
        prog = dataset[:4].lower()
        
    s3_obj = '{0}/{1}/{2}/{2}_{3}.{4}'.format(base_path, prog, dataset.lower(), product, ext)
    return s3_obj


def setup_log_table():
    """
    Set up exposure_files table
    """
    
    if 0:
        db.execute('DROP TABLE exposure_files')
    
    SQL = f"""CREATE TABLE IF NOT EXISTS exposure_files (
        file varchar,
        extension varchar,
        dataset varchar,
        assoc varchar,
        parent varchar,
        awspath varchar,
        filter varchar,
        pupil varchar,
        mdrizsky real,
        chipsky real,
        exptime real,
        expstart real,
        sciext int,
        instrume varchar,
        detector varchar,
        ndq int,
        expflag varchar,
        sunangle real,
        gsky101 real,
        gsky102 real,
        gsky103 real,
        persnpix int,
        perslevl real,
        naxis1 int, 
        naxis2 int, 
        crpix1 int,
        crpix2 int,
        crval1 real,
        crval2 real,
        cd11 real,
        cd12 real,
        cd21 real,
        cd22 real,
        ra1 real,
        dec1 real,
        ra2 real,
        dec2 real,
        ra3 real,
        dec3 real,        
        ra4 real,
        dec4 real,
        footprint path,
        modtime real
    );
    """
    db.execute(SQL)


def all_visit_exp_info(all_visits):
    """
    Run `exposure_info_from_visit` for a list of visits
    """
    for i, v in enumerate(all_visits):
        assoc = v['files'][0].split('/')[0]
        print(f'=======================\n     {i+1} / {len(all_visits)}')
        print(v['files'][0], assoc)
        print('========================')
        exposure_info_from_visit(v, assoc=assoc)


def exposure_info_from_visit(visit, assoc=''):
    """
    Run `s3_put_exposure` for each file in visit['files']
    """

    for file in visit['files']:
        s3_put_exposure(file, visit['product'], assoc, remove_old=True)


def s3_put_exposure(flt_file, product, assoc, remove_old=True, verbose=True):
    """
    """
    import os
    from tqdm import tqdm
    import pandas as pd
    import astropy.time
    from grizli.aws import db
    from grizli import utils
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    
    from .tile_mosaic import add_exposure_to_tile_db
    
    hdul = pyfits.open(flt_file)
    modtime = astropy.time.Time(os.path.getmtime(flt_file), format='unix').mjd
    
    filename = os.path.basename(flt_file)
    file = '_'.join(filename.split('_')[:-1])
    extension = filename.split('_')[-1].split('.')[0]
    
    s3_obj = s3_object_path(file, product=extension, ext='fits', 
                            base_path='Exposures')
    
    h0 = hdul[0].header
    
    if flt_file.startswith('jw'):
        filt = h0['FILTER']
        pupil = h0['PUPIL']
        exptime = h0['EFFEXPTM']
        expflag, sunangle = None, None
        
    else:
        filt = utils.get_hst_filter(h0)
        pupil = ''
        exptime = h0['EXPTIME']
        expflag = h0['EXPFLAG']
        sunangle = h0['SUNANGLE']
        
    names = ['file','extension','dataset','assoc','parent','awspath','filter', 
             'pupil','exptime','expstart','sciext','instrume','detector',
             'ndq','expflag','sunangle','mdrizsky','chipsky',
             'gsky101','gsky102','gsky103','persnpix','perslevl',
             'naxis1','naxis2','crpix1','crpix2','crval1','crval2',
             'cd11','cd12','cd21','cd22',
             'ra1','dec1','ra2','dec2','ra3','dec3','ra4','dec4','modtime']
             
    gsky = []
    for i in range(3):
        if f'GSKY10{i+1}' in h0:
            gsky.append(h0[f'GSKY10{i+1}'])
        else:
            gsky.append(None)
            
    rows = []
    for ext in range(8):
        if ('SCI',ext+1) not in hdul:
            continue
        
        hi = hdul['SCI',ext+1].header
        
        if 'MDRIZSKY' in hi:
            mdrizsky = hi['MDRIZSKY']
        else:
            mdrizsky = 0
        
        if 'CHIPSKY' in hi:
            chipsky = hi['CHIPSKY']
        else:
            chipsky = 0
            
        row = [file, extension, file, assoc, product, os.path.dirname(s3_obj)] 
        
        row += [filt, pupil]
        
        row += [exptime, h0['EXPSTART'], ext+1, h0['INSTRUME'], 
                h0['DETECTOR'], (hdul['DQ',ext+1].data > 0).sum(), 
                expflag, sunangle, mdrizsky, chipsky] 
        
        row += gsky
        
        row += [None, None] # pernpix, persnlevl
        
        for k in ['NAXIS1','NAXIS2','CRPIX1','CRPIX2','CRVAL1','CRVAL2',
                  'CD1_1','CD1_2','CD2_1','CD2_2']:
            if k in hi:
                row.append(hi[k])
            else:
                row.append(None)
        
        wi = pywcs.WCS(hdul['SCI',ext+1].header, fobj=hdul)
        co = wi.calc_footprint()
        
        row += wi.calc_footprint().flatten().tolist()
        
        row.append(modtime)
        
        rows.append(row)
    
    if remove_old:
        db.execute(f"""DELETE FROM mosaic_tiles_exposures t
                             USING exposure_files e 
                             WHERE t.expid = e.eid
                             AND file='{file}'
                             AND extension='{extension}'""")

        db.execute(f"""DELETE FROM exposure_files
                       WHERE file='{file}' AND extension='{extension}'""")
    
    #df = pd.DataFrame(data=rows, columns=names)
    t = utils.GTable(rows=rows, names=names)
    df = t.to_pandas()
    df.to_sql('exposure_files', db._ENGINE, index=False, if_exists='append', 
              method='multi')
         
    # Set footprint
    # ('(' || latitude || ', ' || longitude || ')')::point
    
    db.execute(f"""UPDATE exposure_files
                   SET footprint= (
                    '((' || ra1 || ', ' || dec1 || '),
                      (' || ra2 || ', ' || dec2 || '),
                      (' || ra3 || ', ' || dec3 || '),
                      (' || ra4 || ', ' || dec4 || '))')::path
                   WHERE file='{file}' AND extension='{extension}'""")
    
    # Update tile_db
    tiles = db.SQL('select * from mosaic_tiles')
        
    exp = db.SQL(f"""SELECT eid, assoc, dataset, extension, filter, 
                          sciext, crval1 as ra, crval2 as dec, footprint
                          FROM exposure_files
                     WHERE file='{file}' AND extension='{extension}'""")
    
    res = [add_exposure_to_tile_db(row=exp[i:i+1], tiles=tiles)
           for i in tqdm(range(len(exp)))]
    
    for j in range(len(res))[::-1]:
        if res[j] is None:
            res.pop(j)
            
    if len(res) > 0:
        tab = astropy.table.vstack(res)
        db.execute(f"""DELETE from mosaic_tiles_exposures t
                       USING exposure_files e
                       WHERE t.expid = e.eid
                       AND file='{file}' AND extension='{extension}'""")
         
        df = tab.to_pandas()
        df.to_sql('mosaic_tiles_exposures', db._ENGINE, index=False, 
                  if_exists='append', method='multi')
                             
    if verbose:
        print(f'Add {file}_{extension} ({len(rows)}) to exposure_files table')


def setup_astrometry_tables():
    """
    Initialize shifts_log and wcs_log tables
    """
    from grizli.aws import db
    
    db.execute('DROP TABLE shifts_log')
    
    SQL = f"""CREATE TABLE IF NOT EXISTS shifts_log (
        shift_dataset varchar,
        shift_parent varchar,
        shift_assoc varchar,
        shift_dx real,
        shift_dy real, 
        shift_n int,
        shift_xrms real,
        shift_yrms real,
        shift_modtime real);
    """
    db.execute(SQL)
    
    db.execute('CREATE INDEX on shifts_log (shift_dataset)')
    db.execute('ALTER TABLE shifts_log ADD COLUMN shift_parent VARCHAR;')
    db.execute('CREATE INDEX on shifts_log (shift_parent)')
    db.execute('ALTER TABLE shifts_log ADD COLUMN shift_assoc VARCHAR;')
    
    db.execute('DROP TABLE wcs_log')

    SQL = f"""CREATE TABLE IF NOT EXISTS wcs_log (
        wcs_assoc varchar, 
        wcs_parent varchar,
        wcs_radec varchar,
        wcs_iter int,
        wcs_dx real,
        wcs_dy real, 
        wcs_rot real, 
        wcs_scale real,
        wcs_rms real, 
        wcs_n int, 
        wcs_modtime real);
    """
    db.execute(SQL)

    db.execute('CREATE INDEX on wcs_log (wcs_parent)')
    db.execute('CREATE INDEX on exposure_files (dataset)')
    db.execute('CREATE INDEX on exposure_files (parent)')
    
    # Add assoc to shifts, wcs
    db.execute('ALTER TABLE wcs_log ADD COLUMN wcs_assoc VARCHAR')
    
    ###### Run this to update wcs_log.wcs_assoc column and pop out 
    ### reprocessed visits
    db.execute("""UPDATE wcs_log SET wcs_assoc = NULL""")
    db.execute("""UPDATE wcs_log
                      SET wcs_assoc = exposure_files.assoc
                      FROM exposure_files
                      WHERE wcs_log.wcs_parent = exposure_files.parent;""")
                      
    db.execute('DELETE FROM wcs_log where wcs_assoc IS NULL')


def add_shifts_log(files=None, assoc=None, remove_old=True, verbose=True):
    """
    """
    import glob
    import pandas as pd
    import astropy.time

    if files is None:
        files = glob.glob('*shifts.log')
        
    for ifile, file in enumerate(files):
        if not file.endswith('_shifts.log'):
            continue
        
        parent = os.path.basename(file).split('_shifts.log')[0]
        
        with open(file) as fp:
            lines = fp.readlines()
        
        modtime = astropy.time.Time(os.path.getmtime(file), format='unix').mjd
        
        rows = []
        names = ['shift_dataset','shift_parent', 'shift_dx', 'shift_dy',
                 'shift_n', 'shift_xrms','shift_yrms','shift_modtime']
        
        if assoc is not None:
            names.append('shift_assoc')
            
        for line in lines:
            if line.startswith('#'):
                continue
            
            spl = line.strip().split()
            dataset = '_'.join(spl[0].split('_')[:-1])
            
            if remove_old:
                db.execute(f"""DELETE FROM shifts_log
                               WHERE shift_dataset='{dataset}'""")
            
            row = [dataset, parent, float(spl[1]), float(spl[2]), int(spl[5]), 
                   float(spl[6]), float(spl[7]), modtime]
            if assoc is not None:
                row.append(assoc)
                
            rows.append(row)
        
        if len(rows) > 0:
            df = pd.DataFrame(data=rows, columns=names)
            if verbose:
                print(f'{ifile+1} / {len(files)}: Send {file} > `shifts_log` table')
                
            df.to_sql('shifts_log', db._ENGINE, index=False, 
                      if_exists='append', method='multi')


def add_wcs_log(files=None, assoc=None, remove_old=True, verbose=True):
    """
    """
    import glob
    import pandas as pd
    import astropy.time

    if files is None:
        files = glob.glob('*wcs.log')
        
    for ifile, file in enumerate(files):
        if not file.endswith('_wcs.log'):
            continue
        
        with open(file) as fp:
            lines = fp.readlines()
        
        modtime = astropy.time.Time(os.path.getmtime(file), format='unix').mjd
        
        rows = []
        names = ['wcs_parent', 'wcs_radec', 'wcs_iter', 'wcs_dx', 'wcs_dy', 
                 'wcs_rot', 'wcs_scale', 'wcs_rms', 'wcs_n', 'wcs_modtime']
        
        if assoc is not None:
            names.append('wcs_assoc')
            
        parent = os.path.basename(file).split('_wcs.log')[0]
        if remove_old:
            db.execute(f"DELETE FROM wcs_log WHERE wcs_parent='{parent}'")
            
        rows = []
        
        for line in lines:
            if line.startswith('#'):
                if 'radec:' in line:
                    radec = line.strip().split()[-1]
                
                continue
            
            spl = line.strip().split()
                        
            row = [parent, radec, int(spl[0]), float(spl[1]), float(spl[2]),
                   float(spl[3]), float(spl[4]), float(spl[5]), int(spl[6]), 
                   modtime]
            if assoc is not None:
                row.append(assoc)
                
            rows.append(row)
            
        if len(rows) > 0:
            df = pd.DataFrame(data=rows, columns=names)
            if verbose:
                print(f'{ifile+1} / {len(files)}: Send {file} > `wcs_log` table')
                
            df.to_sql('wcs_log', db._ENGINE, index=False, if_exists='append', 
                      method='multi')


def get_random_visit(extra=''):
    """
    Find a visit that needs processing
    """

    all_assocs = db.SQL(f"""SELECT DISTINCT(assoc_name) 
                           FROM assoc_table
                           WHERE status=0 {extra}""")
    
    if len(all_assocs) == 0:
        return None
    
    random_assoc = all_assocs[np.random.randint(0, len(all_assocs))][0]
    return random_assoc


def update_assoc_status(assoc, status=1, verbose=True):
    """
    Update `status` for `assoc_name` = assoc in `assoc_table`
    """
    import astropy.time

    NOW = astropy.time.Time.now().mjd
    
    table = 'assoc_table'
    
    sqlstr = """UPDATE {0}
        SET status = {1}, modtime = '{2}'
        WHERE (assoc_name = '{3}');
        """.format(table, status, NOW, assoc)

    if verbose:
        msg = 'Update status = {1} for assoc={0} on `{2}` ({3})'
        print(msg.format(assoc, status, table, NOW))

    db.execute(sqlstr)


def delete_all_assoc_data(assoc):
    """
    Remove files from S3 and database
    """
    import os
    from grizli.aws import db
    
    vis = db.SQL(f"""SELECT obsid, assoc_name, obs_id
                     FROM assoc_table
                     WHERE assoc_name = '{assoc}'""")
                     
    vis_root = np.unique([v[:6] for v in vis['obs_id']])
    for r in vis_root:
        print(f'Remove {r} from shifts_log')
        db._ENGINE.execute(f"""DELETE from shifts_log
                               WHERE shift_dataset like '{r}%%'""")
    
    os.system(f'aws s3 rm --recursive s3://grizli-v2/HST/Pipeline/{assoc}')


def clear_failed():
    """
    Reset status for assoc with 'failed' files
    """
    import glob
    import os

    files = glob.glob('*/Prep/*fail*')
    
    failed_assoc = np.unique([file.split('/')[0] for file in files])
    
    for assoc in failed_assoc:
        update_assoc_status(assoc, status=0, verbose=True)
        os.system(f'rm -rf {assoc}*')


def reset_failed_assoc(failed_status='status != 2', reset=True, remove_files=True):
    """
    Reset failed visits
    """
    import astropy.time
    import numpy as np
    
    from grizli.aws import db
    
    failed_assoc = db.SQL(f"""SELECT * FROM assoc_table 
                              WHERE {failed_status}""")
    
    if reset:
        for assoc in np.unique(failed_assoc['assoc_name']):
            update_assoc_status(assoc, status=0, verbose=True)
            if remove_files:
                os.system(f'rm -rf {assoc}*')
            
    return failed_assoc
    
    
def reset_old():
    """
    Reset status on old runs
    """
    import astropy.time

    now = astropy.time.Time.now().mjd
    
    old_assoc = db.SQL(f"""SELECT distinct(assoc_name),
                                  (modtime - {now})*24 AS dt 
                           FROM assoc_table
                           WHERE modtime < {now-0.2} AND status > 0
                           AND assoc_name LIKE '%%f435%%'""")
    
    old_assoc = db.SQL(f"""SELECT distinct(assoc_name),
                                  (modtime - {now})*24 AS dt
                           FROM assoc_table
                           WHERE assoc_name NOT LIKE '%%f435%%'""")
    
    for assoc in old_assoc['assoc_name']:
        update_assoc_status(assoc, status=0, verbose=True)


def show_recent_assoc(limit=50):
    """
    Show recently-processed associations
    """
    import astropy.time

    last = db.SQL(f"""SELECT assoc_name, status, modtime 
                      FROM assoc_table ORDER BY modtime DESC LIMIT {limit}""")
    
    time = astropy.time.Time(last['modtime'], format='mjd')
    last['iso'] = time.iso
    return last


def launch_ec2_instances(nmax=50):
    """
    Launch EC2 instances from a launch template that run through all 
    status=0 associations and then terminate
    """

    assoc = db.SQL("""SELECT distinct(assoc_name)
                      FROM assoc_table
                      WHERE status = 0""")
    
    count = np.minimum(nmax, len(assoc))
    
    templ = 'lt-0e8c2b8611c9029eb,Version=9'
    
    if count == 0:
        print('No associations to run, abort.')
        return True
    else:
        print(f'# Launch {count} instances with LaunchTemplateId={templ}: ')
        cmd = f'aws ec2 run-instances --count {count}'
        cmd += f' --launch-template LaunchTemplateId={templ}'
        print(cmd)
        os.system(cmd)


def update_visit_results():
    """
    Idea: make a table which is all of the (bright) sources detected 
    in each visit to use for alignment of next
    
    exposure info (alignment, background)
    
    copy files (gzipped)
    
    wcs alignment information
    
    """
    pass


def get_assoc_yaml_from_s3(assoc, s_region=None, bucket='grizli-v2', prefix='HST/Pipeline/Input', write_file=True):
    """
    Get presets from yaml file on s3, if found
    """
    import boto3
    from grizli.pipeline import auto_script
    from grizli import utils
    
    LOGFILE = f'/GrizliImaging/{assoc}.auto_script.log.txt'
    
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(bucket)
    
    s3_prefix = os.path.join(prefix, assoc + '.yaml')
    
    files = [obj.key for obj in bkt.objects.filter(Prefix=s3_prefix)]
    if len(files) > 0:
        local_file = os.path.basename(s3_prefix)
        bkt.download_file(s3_prefix, local_file,
                          ExtraArgs={"RequestPayer": "requester"})
        
        utils.log_comment(LOGFILE,
                          f'Fetch params from s3://{bucket}/{s3_prefix}', 
                          verbose=True)
    else:
        local_file = None
        
    kws = auto_script.get_yml_parameters(local_file=local_file)
    
    if ((s_region is not None) & 
        (kws['preprocess_args']['master_radec'] is None)):
        kws['preprocess_args']['master_radec'] = get_master_radec(s_region)

    # Download radec file if has s3 path
    for k in ['master_radec', 'parent_radec']:
        kw = kws['preprocess_args'][k]
        if not isinstance(kw, str):
            continue
        
        if 's3' in kw:
            path_split = kw.split('s3://')[1].split('/')
            file_bucket = path_split[0]
            file_bkt = s3.Bucket(file_bucket)
            file_prefix = '/'.join(path_split[1:])
            files = [obj.key for obj in
                     file_bkt.objects.filter(Prefix=file_prefix)]
            if len(files) > 0:
                local_file = os.path.join('/GrizliImaging/', 
                                          os.path.basename(file_prefix))
                
                if not os.path.exists(local_file):
                    file_bkt.download_file(file_prefix, local_file,
                                  ExtraArgs={"RequestPayer": "requester"})
                
                kws['preprocess_args'][k] = local_file
                utils.log_comment(LOGFILE, 
                                  f'preprocess_args.{k}: {kw} > {local_file}',
                                  verbose=True)
            else:
                utils.log_comment(LOGFILE, 
                                  f'Warning: preprocess_args.{k} = {kw} but '
                                  'file not found on s3', verbose=True)

                kws['preprocess_args'][k] = None

    kws['visit_prep_args']['fix_stars'] = False
    kws['mask_spikes'] = False
    
    # params for DASH processing
    if ('_cxe_cos' in assoc) | ('_edw_cos' in assoc):
        utils.log_comment(LOGFILE, f'Process {assoc} as DASH', verbose=True)
        
        kws['is_dash'] = True

        kws['visit_prep_args']['align_mag_limits'] = [14,24,0.8]
        kws['visit_prep_args']['align_ref_border'] = 600
        kws['visit_prep_args']['match_catalog_density'] = False
        kws['visit_prep_args']['tweak_threshold'] = 1.3
        
        kws['visit_prep_args']['tweak_fit_order'] = -1
        kws['visit_prep_args']['tweak_max_dist'] = 200
        kws['visit_prep_args']['tweak_n_min'] = 4
        
    if write_file:
        auto_script.write_params_to_yml(kws, output_file=f'{assoc}.run.yaml')
        
    return kws


def get_master_radec(s_region, bucket='grizli-v2', prefix='HST/Pipeline/Astrometry'):
    """
    See if an assoc footprint overlaps with some precomputed astrometric 
    reference
    
    Parameters
    ----------
    s_region : str
        Footprint ``s_region``, e.g., `POLYGON ra1 dec1 ra2 dec2 ...`
    
    Returns
    -------
    radec_file : str
        S3 path of an astrometry reference file, if appropriate.  `None` if no
        overlaps found
    
    """
    from grizli import utils
    
    precomputed_radec = {}
    precomputed_radec['uds_hsc_hst_21457.i24.radec'] = np.array([
                                     [34.06742, -5.36379],
                                     [34.20736, -5.3638 ],
                                     [34.63754, -5.36371],
                                     [34.65164, -5.36304],
                                     [34.65545, -5.35716],
                                     [34.65743, -5.34032],
                                     [34.65748, -4.96357],
                                     [34.65707, -4.96307],
                                     [34.64854, -4.96285],
                                     [34.58495, -4.96281],
                                     [34.36087, -4.9628 ],
                                     [34.10944, -4.96283],
                                     [34.09694, -4.96298],
                                     [34.06705, -4.96413],
                                     [34.06521, -4.97184],
                                     [34.06498, -5.21497],
                                     [34.0653 , -5.32854],
                                     [34.06643, -5.36036]])
    
    precomputed_radec['macs1423.leg_dr9'] = np.array([
                                     [216.08698,  23.96965],
                                     [216.08727,  24.18285],
                                     [216.08692,  24.19107],
                                     [216.08584,  24.20342],
                                     [216.08348,  24.2056 ],
                                     [216.04448,  24.20611],
                                     [215.88471,  24.20637],
                                     [215.82872,  24.20619],
                                     [215.82035,  24.20381],
                                     [215.81864,  24.19812],
                                     [215.815  ,  24.17694],
                                     [215.81388,  24.13725],
                                     [215.81356,  23.99988],
                                     [215.81394,  23.9621 ],
                                     [215.82814,  23.95759],
                                     [215.83219,  23.95646],
                                     [215.97568,  23.95678],
                                     [216.06191,  23.95742],
                                     [216.07548,  23.95938]])
    
    precomputed_radec['cosmos_hsc_full_21467_i_17_24.radec'] = np.array([
                                     [151.84607046,   1.49990913],
                                     [151.83524951,   3.1462682 ],
                                     [151.67890143,   3.49829784],
                                     [151.3974854 ,   3.73217542],
                                     [151.08078764,   3.82451372],
                                     [148.95188266,   3.82225799],
                                     [148.55670267,   3.45314942],
                                     [148.38322832,   2.96712681],
                                     [148.38496107,   1.34437438],
                                     [148.51674431,   0.97117464],
                                     [148.79819847,   0.71573734],
                                     [149.1507641 ,   0.64608533],
                                     [151.32535471,   0.67056198],
                                     [151.70028664,   1.01865704],
                                     [151.84607046,   1.49990913]])
    
    precomputed_radec['egs_i24_221366.radec'] = np.array([                                
                                     [214.76998667,  52.60002448],
                                     [215.39474476,  52.60208779],
                                     [215.39792003,  53.01921821],
                                     [215.20241417,  53.13964003],
                                     [214.6017278 ,  53.14890739],
                                     [214.60297439,  52.60618651],
                                     [214.76998667,  52.60002448]])
                                                              
    sr = utils.SRegion(s_region)
    if sr.centroid[0][0] < 0:
        sr.xy[0][:,0] += 360
        sr = utils.SRegion(sr.xy[0])
        
    radec_match = None
    for k in precomputed_radec:
        srd = utils.SRegion(precomputed_radec[k])
        if srd.shapely[0].intersects(sr.shapely[0]):
            radec_match = k
            break
    
    if radec_match:
        return f's3://{bucket}/{prefix}/{radec_match}'
    else:
        return None


blue_align_params={}
blue_align_params['align_mag_limits'] = [20,25,0.12]
blue_align_params['align_simple'] = False
blue_align_params['max_err_percentile'] = 80
blue_align_params['catalog_mask_pad'] = 0.05
blue_align_params['match_catalog_density'] = False
blue_align_params['align_ref_border'] = 8
blue_align_params['align_min_flux_radius'] = 1.7
blue_align_params['tweak_n_min'] = 5


ALL_FILTERS = ['F410M', 'F467M', 'F547M', 'F550M', 'F621M', 'F689M', 'F763M', 'F845M', 'F200LP', 'F350LP', 'F435W', 'F438W', 'F439W', 'F450W', 'F475W', 'F475X', 'F555W', 'F569W', 'F600LP', 'F606W', 'F622W', 'F625W', 'F675W', 'F702W', 'F775W', 'F791W', 'F814W', 'F850LP', 'G800L', 'F098M', 'F127M', 'F139M', 'F153M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W', 'G102', 'G141']


def process_visit(assoc, clean=True, sync=True, max_dt=4, visit_split_shift=1.2, blue_align_params=blue_align_params, ref_catalogs=['LS_DR9', 'PS1', 'DES', 'GAIA'], filters=None, **kwargs):
    """
    `assoc_table.status`
    
     1 = start
     2 = finished
     9 = has failed files
    10 = no wcs.log files found, so probably nothing was done?
    
    """
    import os
    import glob
    
    from grizli.pipeline import auto_script
    from grizli import utils, prep

    os.chdir('/GrizliImaging/')
    
    if os.path.exists(assoc) & (clean > 0):
        os.system(f'rm -rf {assoc}*')
        
    os.environ['orig_iref'] = os.environ.get('iref')
    os.environ['orig_jref'] = os.environ.get('jref')
    set_private_iref(assoc)

    update_assoc_status(assoc, status=1)
        
    tab = db.SQL(f"""SELECT * FROM assoc_table
                     WHERE assoc_name='{assoc}'""")
    
    if len(tab) == 0:
        print(f"assoc_name='{assoc}' not found in assoc_table")
        return False
        
    keep = []
    for c in tab.colnames:
        if tab[c].dtype not in [object]:
            keep.append(c)
    
    tab[keep].write(f'{assoc}_footprint.fits', overwrite=True)
    
    # Send params to yml files on S3
    
    # Handle 's3://' paths in 'master_radec'
    
    #kws = auto_script.get_yml_parameters()
    kws = get_assoc_yaml_from_s3(assoc, bucket='grizli-v2', 
                                 prefix='HST/Pipeline/Input', 
                                 s_region=tab['footprint'][0])
            
    kws['visit_prep_args']['reference_catalogs'] = ref_catalogs
    if filters is None:
        kws['filters'] = np.unique(tab['filter']).tolist()
    else:
        kws['filters'] = filters
    
    if '_':
        kws['parse_visits_args']['max_dt'] = max_dt
        kws['parse_visits_args']['visit_split_shift'] = visit_split_shift
        
    if ('_f4' in assoc) | ('_f3' in assoc) | ('_f2' in assoc) & (blue_align_params is not None):        
        for k in blue_align_params:
            kws['visit_prep_args'][k] = blue_align_params[k]
            
    kws['kill'] = 'preprocess'
        
    ######## 
    auto_script.go(assoc, **kws)
    ########
    
    failed = glob.glob('*failed')
    
    if len(failed) > 0:
        update_assoc_status(assoc, status=9)
    elif len(glob.glob('*wcs.log')) == 0:
        update_assoc_status(assoc, status=10)
    else:
        update_assoc_status(assoc, status=2)
        
        # Exposure info database table
        visits_file = glob.glob('*visits.npy')
        if len(visits_file) > 0:
            visits, groups, info = np.load(visits_file[0], allow_pickle=True)
        
            for i, v in enumerate(visits):
                print('File exposure info: ', v['files'][0], assoc)
                exposure_info_from_visit(v, assoc=assoc)
    
    add_shifts_log(assoc=assoc, remove_old=True, verbose=True)
    add_wcs_log(assoc=assoc, remove_old=True, verbose=True)
    
    os.environ['iref'] = os.environ['orig_iref']
    os.environ['jref'] = os.environ['orig_jref']
    
    os.chdir('/GrizliImaging/')
    
    if sync:
        os.system(f'aws s3 rm --recursive ' + 
                  f' s3://grizli-v2/HST/Pipeline/{assoc}')
        
        if os.path.exists(assoc):
            os.chdir(assoc)
            cmd = f"""aws s3 sync ./ s3://grizli-v2/HST/Pipeline/{assoc}/ \
                  --exclude "*" \
                  --include "Prep/*_fl*fits" \
                  --include "Prep/*s.log" \
                  --include "Prep/*npy" \
                  --include "*fail*" \
                  --include "RAW/*[nx][tg]" """
        
            os.system(cmd)
            print('\n# Sync\n' + cmd.replace('     ', '') + '\n')
            os.chdir('/GrizliImaging')
            
        files = glob.glob(f'{assoc}*.*')
        for file in files:
            os.system(f'aws s3 cp {file} ' + 
                      f' s3://grizli-v2/HST/Pipeline/{assoc}/')
    
    if (clean & 2) > 0:
        print(f'rm -rf {assoc}*')
        os.system(f'rm -rf {assoc}*')


def set_private_iref(assoc):
    """
    Set reference file directories within a particular association to 
    avoid file conflicts
    """
    import os
    from grizli import utils
    
    if not os.path.exists(assoc):
        os.mkdir(assoc)
    
    if not os.path.exists(assoc+'/iref'):
        os.mkdir(assoc+'/iref')
    
    if not os.path.exists(assoc+'/jref'):
        os.mkdir(assoc+'/jref')
    
    os.environ['iref'] = f'{os.getcwd()}/{assoc}/iref/'
    os.environ['jref'] = f'{os.getcwd()}/{assoc}/jref/'
    
    utils.fetch_default_calibs()


def make_parent_mosaic(parent='j191436m5928', **kwargs):
    """
    Get the full footprint of all exposure in the database for a given 'parent'
    """
    from grizli import utils
    from grizli.aws.visit_processor import cutout_mosaic

    fps = db.SQL(f"""SELECT  a.parent, e.filter, e.footprint
                     FROM exposure_files e, assoc_table a
                     WHERE e.assoc = a.assoc_name
                       AND a.parent = '{parent}'""")

    fp = None
    for f in fps['footprint']:
        fp_i = utils.SRegion(f)
        for fs in fp_i.shapely:
            if fp is None:
                fp = fs
            else:
                fp = fp.union(fs)
    
    ra, dec = np.squeeze(fp.centroid.xy)
    bx, by = fp.convex_hull.boundary.xy
    cosd = np.cos(dec/180*np.pi)
    size = ((np.max(bx) - np.min(bx))*cosd*3600+10., 
            (np.max(by) - np.min(by))*3600+10.)
    
    cutout_mosaic(rootname=parent, ra=ra, dec=dec, size=size, **kwargs)
    
def cutout_mosaic(rootname='gds', product='{rootname}-{f}', ra=53.1615666, dec=-27.7910651, size=5*60, filters=['F160W'], ir_scale=0.1, ir_wcs=None, res=None, half_optical=True, kernel='point', pixfrac=0.33, make_figure=True, skip_existing=True, clean_flt=True, gzip_output=True, s3output='s3://grizli-v2/HST/Pipeline/Mosaic/', **kwargs):
    """
    Make mosaic from exposures defined in the exposure database
    
        
    """
    import os
    import glob
    import matplotlib.pyplot as plt
    import astropy.io.fits as pyfits
    
    import boto3
    
    from grizli import utils
    from mastquery import overlaps

    if ir_wcs is None:
        out_h, ir_wcs = utils.make_wcsheader(ra=ra, dec=dec, size=size, 
                                          pixscale=ir_scale, get_hdu=False)
    
    fp = ir_wcs.calc_footprint()
    opt_wcs = ir_wcs  
    if half_optical:
        opt_wcs = utils.half_pixel_scale(ir_wcs)
        
    x1, y1 = fp[0]
    x2, y2 = fp[2]
    
    # Database query
    SQL = f"""
    SELECT dataset, extension, sciext, assoc, 
           e.filter, e.exptime, e.footprint
    FROM exposure_files e, assoc_table a
    WHERE e.assoc = a.assoc_name
    AND a.status = 2
    AND polygon(e.footprint) && polygon(box '(({x1},{y1}),({x2},{y2}))')    
    """
    
    if filters is not None:
        filter_sql = ' OR '.join([f"a.filter = '{f}'" for f in filters])
        SQL += f'AND ({filter_sql})'
    
    SQL += ' ORDER BY e.filter'
    if res is None:
        res = db.SQL(SQL)
    
    for f in np.unique(res['filter']):
        
        visit = {'product':product.format(rootname=rootname, f=f).lower()}
        
        if (len(glob.glob(visit['product'] + '*fits*')) > 0) & skip_existing:
            print('Skip ' + visit['product'])
            continue
            
        print('============', visit['product'], '============')
        
        if f[1] > '1':
            visit['reference'] = opt_wcs
            is_optical = True
        else:
            visit['reference'] = ir_wcs
            is_optical = False
            
        fi = res['filter'] == f
        un = utils.Unique(res['dataset'][fi], verbose=False)
        
        ds = res[fi]['dataset']
        ex = res[fi]['extension']
        assoc = res[fi]['assoc']
        ffp = res[fi]['footprint']
        expt = res[fi]['exptime'].sum()
        
        if make_figure:
            fig, ax = plt.subplots(1,1,figsize=(6,6))
            ax.scatter(*fp.T, marker='.', color='r')
            sr = utils.SRegion(fp)
            for p in sr.get_patch(alpha=0.8, ec='r', fc='None',zorder=100):
                ax.add_patch(p)

            for f in ffp:
                sr = utils.SRegion(f)
                for p in sr.get_patch(alpha=0.03, ec='k', zorder=-100):
                    ax.add_patch(p)

            ax.grid()
            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_title(visit['product'])
            overlaps.draw_axis_labels(ax=ax)
            ax.text(0.95, 0.95, f'N={fi.sum()}\nexpt = {expt:.1f}', 
                    ha='right', va='top', transform=ax.transAxes)
            
            fig.savefig(visit['product']+'_fp.png')
            plt.close('all')
            
        visit['files'] = [f"{ds[un[v]][0]}_{ex[un[v]][0]}.fits"
                          for v in un.values]
        
        visit['awspath'] = [f"grizli-v2/HST/Pipeline/{assoc[un[v]][0]}/Prep"
                          for v in un.values]
        
        visit['footprints'] = []
        for v in un.values:
            fps = ffp[un[v]]
            fp_i = None
            for fpi in fps:
                sr = utils.SRegion(fpi)
                for p in sr.shapely:
                    if fp_i is None:
                        fp_i = p
                    else:
                        fp_i = fp_i.union(p)
            
            visit['footprints'].append(fp_i)
                           
        _ = utils.drizzle_from_visit(visit, visit['reference'], 
                                     pixfrac=pixfrac, 
                                     kernel=kernel, clean=clean_flt)
                             
        outsci, outwht, header, flist = _
        
        if is_optical:
            drz = 'drc'
        else:
            drz = 'drz'
            
        pyfits.writeto('{0}_{1}_sci.fits'.format(visit['product'], drz),
                       data=outsci, header=header, 
                       overwrite=True)
    
        pyfits.writeto('{0}_{1}_wht.fits'.format(visit['product'], drz),
                      data=outwht, header=header, 
                      overwrite=True)

    if s3output:
        files = []
        for f in np.unique(res['filter']):
            prod = product.format(rootname=rootname, f=f).lower()
            
            if gzip_output:
                print(f'gzip --force {prod}*fits')
                os.system(f'gzip --force {prod}*fits')

            files += glob.glob(f'{prod}*fits*')
            files += glob.glob(f'{prod}*_fp.png')
        
        
        for file in files:
            #os.system(f'aws s3 cp {file} {s3output}')
            bucket = s3output.split('s3://')[-1].split('/')[0]
            path = '/'.join(s3output.split('s3://')[-1].strip('/').split('/')[1:])
            object_name = f'{path}/{file}'
            print(f'{file} > s3://{bucket}/{object_name}')
            db.upload_file(file, bucket, object_name=object_name)


def make_mosaic(jname='', ds9=None, skip_existing=True, ir_scale=0.1, half_optical=False, pad=16, kernel='point', pixfrac=0.33, sync=True, ir_wcs=None):
    """
    Make mosaics from all exposures in a group of associations
    """
    import os
    import glob
    from grizli.pipeline import auto_script
    
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from grizli import utils
    import numpy as np
    
    if jname:
        dirs = glob.glob(f'{jname}*/Prep')
        if len(dirs) == 0:
            os.system(f'aws s3 sync s3://grizli-v2/HST/Pipeline/ ./ --exclude "*" --include "{jname}*/Prep/*"')
            
    all_visits = []
    products = []
    visit_files = glob.glob(f'{jname}*/Prep/*visits.npy')
    
    for i, file in enumerate(visit_files):
        visits, groups, info = np.load(file, allow_pickle=True)
        print(file, len(visits))
        for v in visits:
            has_fp = ('footprints' in v)
            if not has_fp:
                print('No footprint: {0}'.format(v['product']))
            
            if file.startswith('j'):
                vprod = v['product'] + '-' + v['files'][0]
            else:
                vprod = v['product']
                
            if has_fp & (vprod not in products):
                v['parent'] = file.split("_visits")[0].split('-')[-1]
                v['first'] = v['files'][0]
                v['parent_file'] = file
                v['xproduct'] = vprod
                #v['awspath'] = ['']*len(v['files'])
                v['files'] = [os.path.dirname(v['parent_file']) + '/' + f
                              for f in v['files']]
                              
                all_visits.append(v)
                products.append(vprod)

    for v in all_visits:
        v['filter'] = v['product'].split('-')[-1]
        is_uvis = ('_flc' in v['files'][0])     
        is_uvis &= os.path.basename(v['files'][0]).startswith('i')
        
        if is_uvis:
            v['filter'] += 'u'
            
        v['first'] = v['files'][0]
        
    all_files = []
    file_products = []

    for v in all_visits:
        all_files.extend(v['files'])
        file_products.extend([v['xproduct']]*len(v['files']))
    
    #########    
    tab = utils.GTable()

    for k in ['parent', 'product', 'filter', 'first']:
        tab[k] = [visit[k] for visit in all_visits]

    coo = np.array([np.array(visit['footprint'].centroid.xy).flatten() for visit in all_visits])
    tab['ra'] = coo[:, 0]
    tab['dec'] = coo[:, 1]
    tab['nexp'] = [len(visit['files']) for visit in all_visits]
    tab['bounds'] = [np.array(v['footprint'].bounds) for v in all_visits]

    if jname:
        master = jname
    else:
        master = 'local'
    
    tab.write(master+'_visits.fits', overwrite=True)
    np.save(master+'_visits.npy', [all_visits])
    
    dash = tab['ra'] < 0
    
    #
    groups = {}
    fpstr = {}

    for filt in np.unique(tab['filter']):
        mat = (tab['filter'] == filt) & (~dash)
        groups[filt] = {'filter': filt, 'files': [], 'footprints': []}
        fpstr[filt] = 'fk5\n'

        for ix in np.where(mat)[0]:
            fp = all_visits[ix]['footprint']
            if hasattr(fp, '__len__'):
                fps = fp
            else:
                fps = [fp]
                
            for fp in fps:
                xy = fp.boundary.xy
                pstr = 'polygon('+','.join(['{0:.6f}'.format(i) for i in np.array([xy[0].tolist(), xy[1].tolist()]).T.flatten()])+') # text={{{0}}}\n'.format(all_visits[ix]['product'])

                fpstr[filt] += pstr

            for k in ['files', 'footprints']:
                groups[filt][k].extend(all_visits[ix][k])

        fp = open('{0}-pointings-{1}.reg'.format(master, filt), 'w')
        fp.write(fpstr[filt])
        fp.close()

        print('{0:6} {1:>3d} {2:>4d} ({3:>4d})'.format(filt, mat.sum(), len(groups[filt]['files']), len(np.unique(groups[filt]['files']))))

    np.save('{0}_filter_groups.npy'.format(master), [groups])
    
    # Drizzle mosaic    
    base = 'uds22'
    base = 'macs1423'
    if jname:
        base = jname
        
    if base == 'uds22':
        url = 'https://erda.ku.dk/vgrid/Gabriel%20Brammer/HST/Mosaic/CANDELS/'
        files = ['uds-100mas-f140w_drz_sci.fits.gz', 'uds-050mas-f606w_drz_sci.fits.gz']
        for file in files:
            if not os.path.exists(file):
                os.system(f'wget "{url}/{file}"')
            
        ir_im = pyfits.open('uds-100mas-f140w_drz_sci.fits.gz')
        ir_wcs = pywcs.WCS(ir_im[0].header, relax=True)

        opt_im = pyfits.open('uds-050mas-f606w_drz_sci.fits.gz')
        opt_wcs = pywcs.WCS(opt_im[0].header, relax=True)
    
        del(ir_im)
        del(opt_im)
    
    else:
        if ir_wcs is None:
            h, ir_wcs = utils.make_maximal_wcs(all_files, 
                                               pixel_scale=ir_scale, 
                                               get_hdu=False, pad=pad)
        
        opt_wcs = ir_wcs  
        if half_optical:
            opt_wcs = utils.half_pixel_scale(ir_wcs)
            
            # h, opt_wcs = utils.make_maximal_wcs(all_files, 
            #                              pixel_scale=ir_scale/2, 
            #                              get_hdu=False, pad=pad)
    
    #skip = True
    
    for f in groups:
        
        groups[f]['product'] = f'{base}-{f}'
        if (len(glob.glob(groups[f]['product'] + '*fits*')) > 0) & skip_existing:
            print('Skip ' + groups[f]['product'])
            continue
            
        print('============', groups[f]['product'], '============')
        
        if f[1] > '1':
            groups[f]['reference'] = opt_wcs
        else:
            groups[f]['reference'] = ir_wcs
            
        _ = utils.drizzle_from_visit(groups[f], groups[f]['reference'], 
                                     pixfrac=pixfrac, 
                                     kernel=kernel, clean=False)
                             
        outsci, outwht, header, flist = _
    
        pyfits.writeto(groups[f]['product']+'_drz_sci.fits',
                       data=outsci, header=header, 
                       overwrite=True)
    
        pyfits.writeto(groups[f]['product']+'_drz_wht.fits',
                      data=outwht, header=header, 
                      overwrite=True)
    
    if ds9 is not None:
        auto_script.field_rgb(base, HOME_PATH=None, xsize=12, output_dpi=300, 
              ds9=ds9, scl=2, suffix='.rgb', timestamp=True, mw_ebv=0)
    
    auto_script.field_rgb(base, HOME_PATH=None, xsize=12, output_dpi=72, 
          ds9=None, scl=1, suffix='.rgb', timestamp=True, mw_ebv=0, 
          show_ir=False)
    
    if (len(jname) > 0) & (sync):
        os.system(f'gzip --force {jname}-f*fits')
        
        os.system(f'aws s3 sync ./ s3://grizli-v2/HST/Pipeline/Mosaic/ --exclude "*" --include "{jname}-f*fits.gz" --include "{jname}*jpg"')
        
    x = """
    aws s3 sync ./ s3://grizli-v2/HST/Pipeline/ --exclude "*" --include "*/Prep/*_fl*fits" --include "*yml" --include "*/Prep/*s.log" --include "*log.txt" --include "*/Prep/*npy" --include "local*reg" --include "*fail*"
    
    aws ec2 stop-instances --instance-ids i-0a0048f42851000e9
    
    ###### UDS
    aws s3 sync ./ s3://grizli-v2/HST/UDS/ --exclude "*" --include "*/Prep/*_fl*fits" --include "*/Prep/*s.log" --include "*log.txt" --include "*/Prep/*npy" --include "local*reg"
    """


def run_all():
    """
    Process all assoc with status=0
    """
    import os
    import time
    from grizli.aws import db

    nassoc = db.SQL('select count(distinct(assoc_name)) '
                         ' from assoc_table')['count'][0]
                         
    assoc = -1
    j = 0
    while (assoc is not None) & (j < nassoc):
        j += 1
        assoc = get_random_visit()
        if assoc is not None:
            print(f'============  Run association  ==============')
            print(f'{j}: {assoc}')
            print(f'========= {time.ctime()} ==========')
            
            process_visit(assoc, clean=2)


def run_one(clean=2, sync=True):
    """
    Run a single random visit
    """
    import os
    import time

    nassoc = db.SQL("""SELECT count(distinct(assoc_name))
                       FROM assoc_table""")['count'][0] 
    
    assoc = get_random_visit()
    if assoc is None:
        with open('/GrizliImaging/finished.txt','w') as fp:
            fp.write(time.ctime() + '\n')
    else:
        print(f'============  Run association  ==============')
        print(f'{assoc}')
        print(f'========= {time.ctime()} ==========')
        
        with open('/GrizliImaging/visit_history.txt','a') as fp:
            fp.write(f'{time.ctime()} {assoc}\n')
        
        process_visit(assoc, clean=clean, sync=sync)
    
    
if __name__ == '__main__':
    run_all()