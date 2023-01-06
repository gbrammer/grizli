#!/usr/bin/env python

"""
Process space telescope visits

2022 version with associations defined in the database in `assoc_table`

"""
import os
import glob
import numpy as np
import astropy.units as u

from . import db
from .. import utils

ROOT_PATH = '/GrizliImaging'
if not os.path.exists(ROOT_PATH):
    ROOT_PATH = os.getcwd()
    print(f'Set ROOT_PATH={ROOT_PATH}')


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
        filt = utils.parse_filter_from_header(h0) #h0['FILTER']
        if 'PUPIL' in h0:
            pupil = h0['PUPIL']
        else:
            pupil = ''
            
        exptime = h0['EFFEXPTM']
        expflag, sunangle = None, None
        
    else:
        filt = utils.parse_filter_from_header(h0)
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
                     WHERE file='{file}' 
                     AND extension='{extension}'
                     AND exptime > 0.0""")
    
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


def check_missing_files():
    """
    check for files that are in exposure_files but not on S3
    """
    from grizli.aws import db
    import boto3
    from tqdm import tqdm
    from grizli import utils
    
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket('grizli-v2')
    
    files = db.SQL("select assoc, file,extension from exposure_files group by assoc, file,extension order by assoc")
    
    exists = []
    for a, f, e in tqdm(zip(files['assoc'], files['file'], files['extension'])):
        s3_prefix = f'HST/Pipeline/{a}/Prep/{f}_{e}.fits'
        xfiles = [obj.key for obj in bkt.objects.filter(Prefix=s3_prefix)]
        exists.append(len(xfiles))
        if len(xfiles) == 0:
            print(a, f, e)
            
    exists = np.array(exists)
    
    miss = exists == 0
    
    m = utils.GTable()
    m['miss_file'] = files['file'][miss]
    m['miss_assoc'] = files['assoc'][miss]
    
    db.send_to_database('missing_files', m)
        
    db.execute("""UPDATE exposure_files e
                  set modtime = 99999
                  from missing_files m
                  where assoc = m.miss_assoc
    """)
    
    db.execute("""UPDATE assoc_table e
                  set status = 0
                  from missing_files m
                  where assoc_name = m.miss_assoc
    """)

    db.SQL(f"""SELECT count(*) 
         FROM mosaic_tiles_exposures t, exposure_files e
                WHERE t.expid = e.eid AND e.modtime > 90000""")
    
    db.execute(f"""DELETE FROM mosaic_tiles_exposures t
                USING exposure_files e
                WHERE t.expid = e.eid AND e.modtime > 90000""")
    
    db.execute(f"""DELETE FROM exposure_files
                WHERE modtime > 90000""")
    
    db.execute(f"""DELETE FROM shifts_log 
                 USING missing_files m
                WHERE shift_dataset = m.miss_file""")
    
    db.execute(f"""DELETE FROM wcs_log 
                 USING missing_files m
                WHERE wcs_assoc = m.miss_assoc""")
    
    db.execute('DROP TABLE missing_files')
    
    # EXPTIME= 0
    db.execute(f"""DELETE FROM mosaic_tiles_exposures t
                USING exposure_files e
                WHERE t.expid = e.eid AND e.exptime <= 0.""")
    
    db.execute(f"""DELETE FROM exposure_files
                WHERE exptime <= 0""")
    
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
        WHERE (assoc_name = '{3}' AND status != 99);
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
    from grizli.aws import db, tile_mosaic
    
    vis = db.SQL(f"""SELECT obsid, assoc_name, obs_id
                     FROM assoc_table
                     WHERE assoc_name = '{assoc}'""")
                     
    obs_id = []
    for v in vis['obs_id']:
        if v.startswith('jw'):
            obs_id.append(v)
        else:
            obs_id.append(v[:6])
            
    vis_root = np.unique(obs_id)
    for r in vis_root:
        print(f'Remove {r} from shifts_log')
        db.execute(f"""DELETE from shifts_log
                               WHERE shift_dataset like '{r}%%'""")
    
    print('Remove from wcs_log')
    db.execute(f"DELETE from wcs_log where wcs_assoc = '{assoc}'")
    
    print('Remove from tile_exposures')
    
    # Reset tiles of deleted exposures potentially from other exposures
    tile_mosaic.reset_tiles_in_assoc(assoc)
    
    # Remove deleted exposure from mosaic_tiles_exposures
    res = db.execute(f"""DELETE from mosaic_tiles_exposures t
                 USING exposure_files e
                WHERE t.expid = e.eid AND e.assoc = '{assoc}'""")
    
    res = db.execute(f"""DELETE from exposure_files
                    WHERE assoc = '{assoc}'""")
                    
    os.system(f'aws s3 rm --recursive s3://grizli-v2/HST/Pipeline/{assoc}')

    res = db.execute(f"""UPDATE assoc_table
                    SET status=12
                    WHERE assoc_name = '{assoc}' AND status != 99""")


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


def launch_ec2_instances(nmax=50, count=None, templ='lt-0e8c2b8611c9029eb,Version=24'):
    """
    Launch EC2 instances from a launch template that run through all 
    status=0 associations/tiles and then terminate
    
    Version 19 is the latest run_all_visits.sh
    Version 20 is the latest run_all_tiles.sh
    Version 24 is run_all_visits with a new python39 environment
    """

    if count is None:
        assoc = db.SQL("""SELECT distinct(assoc_name)
                      FROM assoc_table
                      WHERE status = 0""")
    
        count = int(np.minimum(nmax, len(assoc)/2))

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
    
    LOGFILE = f'{ROOT_PATH}/{assoc}.auto_script.log.txt'
    
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
                local_file = os.path.join(f'{ROOT_PATH}/', 
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
    
    kws['fetch_files_args']['reprocess_clean_darks'] = False
    
    # params for DASH processing
    if ('_cxe_cos' in assoc) | ('_edw_cos' in assoc):
        utils.log_comment(LOGFILE, f'Process {assoc} as DASH', verbose=True)
        
        kws['is_dash'] = True

        kws['visit_prep_args']['align_mag_limits'] = [14,23,0.1]
        kws['visit_prep_args']['align_ref_border'] = 600
        kws['visit_prep_args']['match_catalog_density'] = False
        kws['visit_prep_args']['tweak_threshold'] = 1.3
        
        kws['visit_prep_args']['tweak_fit_order'] = -1
        kws['visit_prep_args']['tweak_max_dist'] = 200
        kws['visit_prep_args']['tweak_n_min'] = 4
        
    if write_file:
        auto_script.write_params_to_yml(kws, output_file=f'{assoc}.run.yaml')
        
    return kws


def cdfs_hsc_catalog():
    """
    """
    from grizli import utils
    from grizli.aws import db
    from grizli import prep
    
    import numpy as np
    
    os.system('wget https://zenodo.org/record/2225161/files/WCDFS_CATALOGS.tar.gz?download=1')
    os.system('tar xzvf WCDFS_CATALOGS.tar.gz?download=1 WCDFS_CATALOGS/wcdfs_hsc_forcedsrc_v01.fits')
    
    hsc = utils.read_catalog('WCDFS_CATALOGS/wcdfs_hsc_forcedsrc_v01.fits')
    
    pat = db.SQL("select * from sky_patches where parent like 'j033%%m27%%' and nassoc > 100")[0]
    
    dx = 10/60
    sel = (hsc['coord_ra'] > pat['ramin']-dx) & (hsc['coord_ra'] < pat['ramax']+dx)
    sel &= (hsc['coord_dec'] > pat['demin']-dx) & (hsc['coord_dec'] < pat['demax']+dx)
    
    imag = 23.9-2.5*np.log10(hsc['i_modelfit_CModel_flux'])
    sel &= (~hsc['i_modelfit_CModel_flux'].mask) & (imag < 24)
    sel &= hsc['i_modelfit_CModel_flag'] == 'False'
    #sel &= hsc['base_Blendedness_flag'] == 'False'
    #sel &= hsc['base_SdssCentroid_flag'] == 'False'
    sel &= hsc['flag_clean'] == 'True'
    sel &= hsc['detect_isPatchInner'] == 'True'
    sel &= hsc['detect_isTractInner'] == 'True'
    
    hsc = hsc[sel]
    cosd = np.cos(hsc['coord_dec']/180*np.pi)
    
    hsc['ra'] = hsc['coord_ra'] + 0.013/3600/cosd
    hsc['dec'] = hsc['coord_dec'] - 0.002/3600.
    
    prep.table_to_radec(hsc, 'Ni2009_WCDFS_i24.radec')
    
    os.system('aws s3 cp Ni2009_WCDFS_i24.radec s3://grizli-v2/HST/Pipeline/Astrometry/')


def get_master_radec(s_region, nmax=200):
    """
    Get overlapping reference sources from the `astrometry_reference` table
    in the grizli database
    """
    # Do it in prep
    return None
    
    
    # from grizli.aws import db
    # 
    # sr = utils.SRegion(s_region)
    # 
    # srcs = db.SQL(f"""select src, count(src) from astrometry_reference
    # where polygon('{sr.polystr()}') @> point(ra, dec)
    # order by src
    # """)
    # if len(srcs) == 0:
    #     return None
    # 
    # _src = src['src'][np.argmax(src['count'])]
    # 
    # pts = db.SQL(f"""select * from astrometry_reference
    # where polygon('{sr.polystr()}') @> point(ra, dec)
    # AND src = '{_src}'
    # order by mag limit {nmax}
    # """)
    
    
def old_get_master_radec(s_region, bucket='grizli-v2', prefix='HST/Pipeline/Astrometry'):
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
    
    precomputed_radec['Ni2009_WCDFS_i24.radec'] = np.array([                                
                                     [52.58105818, -28.26843991],
                                     [52.58105818, -27.35729025],
                                     [53.61045392, -27.35729025],
                                     [53.61045392, -28.26843991],
                                     [52.58105818 , -28.26843991]])
    
    precomputed_radec['j041732m1154.ls_dr9.radec'] = np.array([
                                     [64.27799225, -11.97970169],
                                     [64.27799225, -11.78052877],
                                     [64.48217567, -11.78052877],
                                     [64.48217567, -11.97970169],
                                     [64.27799225, -11.97970169]])
    
    precomputed_radec['smacs0723-f444w.radec'] = np.array([
                                     [110.668751, -73.507313],
                                     [110.674946, -73.507179],
                                     [110.690501, -73.504213],
                                     [110.84973 , -73.473404],
                                     [110.901457, -73.462825],
                                     [110.914218, -73.460128],
                                     [110.916415, -73.458724],
                                     [110.915898, -73.458051],
                                     [110.908464, -73.453869],
                                     [110.88691 , -73.444296],
                                     [110.877191, -73.440097],
                                     [110.854772, -73.4309  ],
                                     [110.846844, -73.427759],
                                     [110.843476, -73.42792 ],
                                     [110.82299 , -73.431392],
                                     [110.777001, -73.439989],
                                     [110.700783, -73.454502],
                                     [110.68813 , -73.456941],
                                     [110.605675, -73.473557],
                                     [110.595953, -73.475866],
                                     [110.593894, -73.47821 ],
                                     [110.598514, -73.480156],
                                     [110.609213, -73.484384],
                                     [110.64198 , -73.497214],
                                     [110.646436, -73.498917]])
    
    precomputed_radec['abell2744_ip_2008_20220620_g3sw.radec'] = np.array([
                                      [3.382855, -30.158365],
                                      [3.287974, -30.160039],
                                      [3.283219, -30.163239],
                                      [3.274886, -30.169153],
                                      [3.25321 , -30.186363],
                                      [3.242435, -30.205177],
                                      [3.237953, -30.217099],
                                      [3.237638, -30.23572 ],
                                      [3.236802, -30.343877],
                                      [3.236131, -30.431126],
                                      [3.236052, -30.444164],
                                      [3.236112, -30.476929],
                                      [3.236585, -30.55128 ],
                                      [3.238121, -30.567037],
                                      [3.271421, -30.625737],
                                      [3.280073, -30.632062],
                                      [3.289894, -30.637195],
                                      [3.303426, -30.639061],
                                      [3.314717, -30.640257],
                                      [3.556417, -30.641503],
                                      [3.615938, -30.64173 ],
                                      [3.771419, -30.640564],
                                      [3.834617, -30.639879],
                                      [3.911254, -30.638888],
                                      [3.916556, -30.638332],
                                      [3.920868, -30.63479 ],
                                      [3.927549, -30.628326],
                                      [3.929028, -30.616425],
                                      [3.929714, -30.528361],
                                      [3.92942 , -30.341083],
                                      [3.928447, -30.27237 ],
                                      [3.894966, -30.172395],
                                      [3.881937, -30.163145],
                                      [3.850611, -30.159172],
                                      [3.777518, -30.158458],
                                      [3.685884, -30.157784],
                                      [3.550749, -30.157794]])
    
                                     
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
blue_align_params['align_mag_limits'] = [18,25,0.15]
blue_align_params['align_simple'] = False
blue_align_params['max_err_percentile'] = 80
blue_align_params['catalog_mask_pad'] = 0.05
blue_align_params['match_catalog_density'] = False
blue_align_params['align_ref_border'] = 8
blue_align_params['align_min_flux_radius'] = 1.7
blue_align_params['tweak_n_min'] = 5


def check_jwst_assoc_guiding(assoc):
    """
    Check the guidestar logs that accompany a given visit
    """
    import os
    import glob
    
    import matplotlib.pyplot as plt
    
    import astropy.table
    
    import mastquery.utils
    import mastquery.jwst
    
    os.chdir(f'{ROOT_PATH}/')
    
    atab = db.SQL(f"""SELECT t_min, t_max, filter, proposal_id, 
                             "dataURL", status
                     FROM assoc_table
                     WHERE assoc_name='{assoc}'
                     """)
    so = np.argsort(atab['t_min'])
    atab = atab[so]
    
    gs = mastquery.jwst.query_guidestar_log(
             mjd=(atab['t_min'].min()-0.1, atab['t_max'].max()+0.1),
             program=atab['proposal_id'][0],
             exp_type=['FGS_FINEGUIDE'],
         )
    
    keep = (gs['expstart'] < atab['t_max'].max())
    gs = gs[keep]
    
    for _iter in range(3):
        res = mastquery.utils.download_from_mast(gs)
    
    tabs = []
    for f in res:
        if not os.path.exists(f):
            continue
            
        tabs.append(utils.read_catalog(f))
    
    tab = astropy.table.vstack(tabs)
    so = np.argsort(tab['time'])
    tab = tab[so]
    
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    t0 = atab['t_min'].min()
    ax.scatter((tab['time'] - t0)*24, tab['jitter'],
               alpha=.02, marker='.', color='k')
    
    atab['jitter16'] = -1.
    atab['jitter50'] = -1.
    atab['jitter84'] = -1.
    
    ymax = 500
    
    for i, row in enumerate(atab):
        gsx = (tab['time'] > row['t_min']) & (tab['time'] < row['t_max'])
        
        if gsx.sum() == 0:
            gs_stats = [-1,-1,-1]
        else:
            gs_stats = np.percentile(tab['jitter'][gsx], [16, 50, 84])

        atab['jitter16'][i] = gs_stats[0]
        atab['jitter50'][i] = gs_stats[1]
        atab['jitter84'][i] = gs_stats[2]
        
        print(f"{i} {os.path.basename(atab['dataURL'][i])} {gs_stats[1]:.2f}")
        
        ax.fill_between([(row['t_min']-t0)*24, (row['t_max']-t0)*24], 
                        np.ones(2)*gs_stats[0], np.ones(2)*gs_stats[2], 
                        color='r', alpha=0.2)
        ax.hlines(gs_stats[1], (row['t_min']-t0)*24, (row['t_max']-t0)*24, 
                  color='r')
        ax.fill_between([(row['t_min']-t0)*24, (row['t_max']-t0)*24], 
                        [0.2,0.2], np.ones(2)*(0.12/0.2*ymax), 
                        color='0.8', alpha=0.1, zorder=-1)
                        
    ax.set_ylim(0.12, ymax)
    ax.semilogy()
    
    ax.set_xlabel(r'$\Delta t$, hours since ' + f'{t0:.2f}')
    ax.set_ylabel('GuideStar jitter, mas')
    ax.set_yticklabels([1,10,100])
    ax.set_yticks([1,10,100])
    ax.set_title(assoc)
    
    ax.set_xlim(-0.2, (atab['t_max'].max() - t0)*24 +0.2)
    ax.grid()
    
    fig.tight_layout(pad=0.5)
    fig.savefig(f'{assoc}.guide.png')
    
    atab.write(f'{assoc}.guide.csv', overwrite=True)
    
    return fig, atab
    

ALL_FILTERS = ['F410M', 'F467M', 'F547M', 'F550M', 'F621M', 'F689M', 'F763M', 'F845M', 'F200LP', 'F350LP', 'F435W', 'F438W', 'F439W', 'F450W', 'F475W', 'F475X', 'F555W', 'F569W', 'F600LP', 'F606W', 'F622W', 'F625W', 'F675W', 'F702W', 'F775W', 'F791W', 'F814W', 'F850LP', 'G800L', 'F098M', 'F127M', 'F139M', 'F153M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W', 'G102', 'G141']


def process_visit(assoc, clean=True, sync=True, max_dt=4, combine_same_pa=False, visit_split_shift=1.2, blue_align_params=blue_align_params, ref_catalogs=['LS_DR9', 'PS1', 'DES', 'NSC', 'GAIA'], filters=None, prep_args={}, get_wcs_guess_from_table=True, master_radec='astrometry_db', align_guess=None, **kwargs):
    """
    `assoc_table.status`
    
     1 = start
     2 = finished
     9 = has failed files
    10 = no wcs.log files found, so probably nothing was done?
         This is most often the case when visits were ignored due to 
         abnormal EXPFLAG keywords
    12 = Don't redo ever 
    
    
    """
    import os
    import glob
    
    from grizli.pipeline import auto_script
    from grizli import utils, prep

    os.chdir(f'{ROOT_PATH}/')
    
    tab = db.SQL(f"""SELECT * FROM assoc_table
                     WHERE assoc_name='{assoc}'
                     AND status != 99
                     """)
    
    if len(tab) == 0:
        print(f"assoc_name='{assoc}' not found in assoc_table")
        return False
        
    if os.path.exists(assoc) & (clean > 0):
        os.system(f'rm -rf {assoc}*')
        
    os.environ['orig_iref'] = os.environ.get('iref')
    os.environ['orig_jref'] = os.environ.get('jref')
    set_private_iref(assoc)

    if sync:
        update_assoc_status(assoc, status=1)
    
    if get_wcs_guess_from_table:
        get_wcs_guess(assoc)
    
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
        kws['filters'] = [f.split('-')[0] 
                          for f in np.unique(tab['filter']).tolist()]            
    else:
        kws['filters'] = filters
    
    if True:
        kws['parse_visits_args']['max_dt'] = max_dt
        kws['parse_visits_args']['visit_split_shift'] = visit_split_shift
        kws['parse_visits_args']['combine_same_pa'] = combine_same_pa
        
    if ('_f4' in assoc) | ('_f3' in assoc) | ('_f2' in assoc) & (blue_align_params is not None):        
        for k in blue_align_params:
            kws['visit_prep_args'][k] = blue_align_params[k]
    
    for k in prep_args:
        kws['visit_prep_args'][k] = prep_args[k]
                
    kws['kill'] = 'preprocess'
    
    if master_radec is not None:
        kws['preprocess_args']['master_radec'] = master_radec
    
    for fi in ['f560w','f770w','f1000w','f1130w','f1280w',
               'f1500w','f1800w','f2100w','f2550w']:
        
        if fi in assoc:
            # Don't run tweak align for MIRI long filters
            miri_prep = dict(run_tweak_align=False,
                             align_mag_limits=[14,28,0.2],
                             align_clip=20, 
                             align_simple=True)
                           
            for k in miri_prep:
                if k not in prep_args:
                    kws['visit_prep_args'][k] = miri_prep[k]
                    
            break
               
    ######## 
    auto_script.go(assoc, **kws)
    ########
    
    failed = glob.glob('*failed')
    
    if len(failed) > 0:
        if sync:
            update_assoc_status(assoc, status=9)
    elif len(glob.glob('*wcs.log')) == 0:
        if sync:
            update_assoc_status(assoc, status=10)
    else:
        if sync:
            update_assoc_status(assoc, status=2)
        
        # Exposure info database table
        vis_files = glob.glob('*visits.yaml')
        if len(vis_files) > 0:
            #visits, groups, info = np.load(visits_file[0], allow_pickle=True)
            visits, groups, info = auto_script.load_visits_yaml(vis_files[0])
            
            for i, v in enumerate(visits):
                if sync:
                    print('File exposure info: ', v['files'][0], assoc)
                    exposure_info_from_visit(v, assoc=assoc)
    
    if sync:
        add_shifts_log(assoc=assoc, remove_old=True, verbose=True)
        add_wcs_log(assoc=assoc, remove_old=True, verbose=True)
    
    os.environ['iref'] = os.environ['orig_iref']
    os.environ['jref'] = os.environ['orig_jref']
    
    os.chdir(f'{ROOT_PATH}/')
    
    if sync:
        os.system(f'aws s3 rm --recursive ' + 
                  f' s3://grizli-v2/HST/Pipeline/{assoc}')
        
        if os.path.exists(assoc):
            os.chdir(assoc)
            cmd = f"""aws s3 sync ./ s3://grizli-v2/HST/Pipeline/{assoc}/ \
                  --exclude "*" \
                  --include "Prep/*_fl*fits" \
                  --include "Prep/*_cal.fits" \
                  --include "Prep/*_rate.fits" \
                  --include "Prep/*s.log" \
                  --include "Prep/*visits.*" \
                  --include "Prep/*skyflat.*" \
                  --include "*fail*" \
                  --include "RAW/*[nx][tg]" """
        
            os.system(cmd)
            print('\n# Sync\n' + cmd.replace('     ', '') + '\n')
            os.chdir(f'{ROOT_PATH}')
            
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
    
    is_jwst = False
    for f in ['f090w','f115w','f150w','f200w','f277w','f356w','f444w',
              'f182m','f140m','f210m','f410m','f430m','f460m','f300m',
              'f250m','f480m']:
        if f in assoc:
            is_jwst = True
            break
    
    if not is_jwst:        
        utils.fetch_default_calibs()


def get_wcs_guess(assoc, guess=None, verbose=True):
    """
    Get visit wcs alignment from the database and use it as a "guess"
    """
    from grizli import utils
    
    wcs = db.from_sql(f"select * from wcs_log where wcs_assoc = '{assoc}'")
    
    LOGFILE = f'{ROOT_PATH}/{assoc}.auto_script.log.txt'
    
    if len(wcs) == 0:
        msg = f"# get_wcs_guess : No entries found in wcs_log for wcs_assoc='{assoc}'"
        utils.log_comment(LOGFILE, msg, verbose=verbose, show_date=True)
        return True
        
    if not os.path.exists(assoc+'/Prep'):
        os.mkdir(assoc+'/Prep')
    
    msg = f"# get_wcs_guess : '{assoc}'"
    utils.log_comment(LOGFILE, msg, verbose=verbose, show_date=True)
        
    for i in range(len(wcs)):
        p = wcs['wcs_parent'][i]
        dx = wcs['wcs_dx'][i]
        dy = wcs['wcs_dy'][i]
        guess_file = f'{assoc}/Prep/{p}.align_guess'
        guess = f"{dx:.2f} {dy:.2f} 0.0 1.0\n"
        
        msg = f"{guess_file} : {guess}"
        utils.log_comment(LOGFILE, msg, verbose=verbose, show_date=False)
        
        with open(guess_file,'w') as fp:
            fp.write(guess)        
        
            
def make_parent_mosaic(parent='j191436m5928', root=None, **kwargs):
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
    
    if root is None:
        root = parent
        
    cutout_mosaic(rootname=root, ra=ra, dec=dec, size=size, **kwargs)


def res_query_from_local(files=None, filters=None):
    """
    Immitate a query from exposure_files for files in a local working 
    directory that can be passed to `grizli.aws.visit_processor.cuotut_mosaic`
    
    Parameters
    ----------
    files : list
        File list.  If `None`, then finds all ``flt``, ``flc``, ``rate``, 
        ``cal`` files in the current working directory
    
    filters : list
        Subset of filters to consider
    
    Returns
    -------
    res : `astropy.table.Table`
        Table of exposure info
        
    """
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    
    if files is None:
        files = glob.glob('*_flt.fits')
        files += glob.glob('*_flc.fits')
        files += glob.glob('*_rate.fits')
        files += glob.glob('*_cal.fits')
        files.sort()

    rows = []
    for file in files:
        im = pyfits.open(file)
        
        filt = utils.parse_filter_from_header(im[0].header)
        if filters is not None:
            if filt not in filters:
                continue
        
        ds = '_'.join(os.path.basename(file).split('_')[:-1])
        ext = file.split('_')[-1].split('.fits')[0]
        if 'EXPTIME' in im[0].header:
            expt = im[0].header['EXPTIME']
        elif 'EFFEXPTM' in im[0].header:
            expt = im[0].header['EFFEXPTM']
        else:
            expt = 0.
        
        assoc = 'manual'

        if 'PUPIL' in im[0].header:
            pup = im[0].header['PUPIL']
        else:
            pup = '-'
                        
        det = im[0].header['DETECTOR']
        wcs = pywcs.WCS(im['SCI',1].header)
        sr = utils.SRegion(wcs)
        fp = sr.polystr()[0]
        
        rows.append([ds, ext, 1, assoc, filt, pup, expt, fp, det])
    
    res = utils.GTable(names=['dataset','extension','sciext','assoc',
                              'filter','pupil','exptime','footprint',
                              'detector'],
                       rows=rows)
    return res


def query_exposures(ra=53.16, dec=-27.79, size=1., pixel_scale=0.1, theta=0,  filters=['F160W'], wcs=None, extra_columns=[], extra_query='', res=None, max_wcs=False):
    """
    Query exposures that overlap a WCS, either specified explicitly or 
    generated from position parameters
    
    Parameters
    ----------
    ra : float
        RA center, decimal degrees
    
    dec : float
        Dec center, decimal degrees
    
    size : float
        Point query, arcsec
    
    pixel_scale : float
        Pixel scale, arcsec
    
    theta : float
        WCS position angle
        
    filters : list, None
        Filter list
    
    wcs : `~astropy.wcs.WCS`, None
        WCS to supersede ``ra``, ``dec``, ``size``, ``pixel_scale``
    
    extra_columns : string
        Additional columns to query, start with a ','
        
    extra_query : str
        Extra criteria appended to SQL query
    
    Returns
    -------
    header : `~astropy.io.fits.Header`
        WCS header
        
    wcs : `~astropy.wcs.WCS`
        Query WCS
        
    SQL : str
        Query string
        
    res : `~grizli.utils.GTable`
        Query result
    
    """
    import astropy.wcs as pywcs
    
    if wcs is None:
        _hdu = utils.make_wcsheader(ra=ra, dec=dec, size=size,
                                    pixscale=pixel_scale,
                                    theta=theta,
                                    get_hdu=True)
        header = _hdu.header
        wcs = pywcs.WCS(header)
    else:
        header = utils.to_header(wcs)
        
    fp = wcs.calc_footprint()
        
    x1, y1 = fp.min(axis=0)
    x2, y2 = fp.max(axis=0)
    
    # Database query
    SQL = f"""
    SELECT dataset, extension, sciext, assoc, 
           e.filter, e.pupil, e.exptime, e.footprint, e.detector
           {extra_columns}
    FROM exposure_files e, assoc_table a
    WHERE e.assoc = a.assoc_name
    AND a.status = 2
    AND e.exptime > 0
    AND polygon(e.footprint) && polygon(box '(({x1},{y1}),({x2},{y2}))') 
    {extra_query}   
    """
    
    if filters is not None:
        filter_sql = ' OR '.join([f"e.filter = '{f}'" for f in filters])
        SQL += f'AND ({filter_sql})'
    
    SQL += ' ORDER BY e.filter'
    if res is None:
        res = db.SQL(SQL)
    
    if max_wcs:
        fp = None
        for f in res['footprint']:
            fp_i = utils.SRegion(f)

            for fs in fp_i.shapely:
                if fp is None:
                    fp = fs
                else:
                    fp = fp.union(fs)

        bx, by = fp.convex_hull.boundary.xy
        ra = (np.max(bx) + np.min(bx))/2
        dec = (np.max(by) + np.min(by))/2

        cosd = np.cos(dec/180*np.pi)

        size = (np.max(np.abs(bx-ra))*2*cosd*3600 + max_wcs*1., 
                np.max(np.abs(by-dec))*2*3600 + max_wcs*1.)
        
        _hdu = utils.make_wcsheader(ra=ra, dec=dec, size=size,
                                    pixscale=pixel_scale,
                                    theta=theta,
                                    get_hdu=True)
        header = _hdu.header
        wcs = pywcs.WCS(header)
        
    return header, wcs, SQL, res


def cutout_mosaic(rootname='gds', product='{rootname}-{f}', ra=53.1615666, dec=-27.7910651, size=5*60, theta=0., filters=['F160W'], ir_scale=0.1, ir_wcs=None, res=None, half_optical=True, kernel='point', pixfrac=0.33, make_figure=True, skip_existing=True, clean_flt=True, gzip_output=True, s3output='s3://grizli-v2/HST/Pipeline/Mosaic/', split_uvis=True, extra_query='', extra_wfc3ir_badpix=True, fix_niriss=True, scale_nanojy=10, verbose=True, niriss_ghost_kwargs={}, scale_photom=True, **kwargs):
    """
    Make mosaic from exposures defined in the exposure database
    
    Parameters
    ----------
    
    Returns
    -------
    """
    import os
    import glob
    import matplotlib.pyplot as plt
    import astropy.io.fits as pyfits
    
    import boto3
    
    from grizli import utils
    from mastquery import overlaps
    
    utils.set_warnings()
    
    # out_h, ir_wcs, res = query_exposures(ra=ra, dec=dec,
    #                                      size=size,
    #                                      pixel_scale=pixel_scale,
    #                                      theta=theta,
    #                                      filters=filters,
    #                                      wcs=ir_wcs,
    #                                      extra_query=extra_query)
    if ir_wcs is None:
        out_h, ir_wcs = utils.make_wcsheader(ra=ra, dec=dec, size=size,
                                          pixscale=ir_scale, get_hdu=False)
    
    fp = ir_wcs.calc_footprint()
    opt_wcs = ir_wcs  
    if half_optical:
        opt_wcs = utils.half_pixel_scale(ir_wcs)
        
    x1, y1 = fp.min(axis=0)
    x2, y2 = fp.max(axis=0)
    
    # Database query
    SQL = f"""
    SELECT dataset, extension, sciext, assoc, 
           e.filter, e.pupil, e.exptime, e.footprint, e.detector
    FROM exposure_files e, assoc_table a
    WHERE e.assoc = a.assoc_name
    AND a.status = 2
    AND e.exptime > 0
    AND polygon(e.footprint) && polygon(box '(({x1},{y1}),({x2},{y2}))') 
    {extra_query}   
    """
    
    if filters is not None:
        filter_sql = ' OR '.join([f"e.filter = '{f}'" for f in filters])
        SQL += f'AND ({filter_sql})'
    
    SQL += ' ORDER BY e.filter'
    if res is None:
        res = db.SQL(SQL)
    
    if len(res) == 0:
        print('No exposures found overlapping output wcs with '+
              f'corners = ({x1:.6f},{y1:.6f}), ({x2:.6f},{y2:.6f})')
        return None
        
    file_filters = [f for f in res['filter']]
    if split_uvis:
        # UVIS filters
        uvis = res['detector'] == 'UVIS'
        if uvis.sum() > 0:
            for j in np.where(uvis)[0]:
                file_filters[j] += 'u'
    
    if fix_niriss:
        nis = res['detector'] == 'NIS'
        if nis.sum() > 0:
            for j in np.where(nis)[0]:
                file_filters[j] = file_filters[j].replace('-CLEAR','N-CLEAR')
            
            msg = 'Fix NIRISS: filters = '
            msg += ' '.join(np.unique(file_filters).tolist())
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            
    uniq_filts = utils.Unique(file_filters, verbose=False)
    
    for f in uniq_filts.values:
        
        visit = {'product':product.format(rootname=rootname, f=f).lower()}
        
        if (len(glob.glob(visit['product'] + '*fits*')) > 0) & skip_existing:
            print('Skip ' + visit['product'])
            continue
            
        print('============', visit['product'], '============')
        
        if 'clear' in f:
            ftest = f.lower().replace('clear-','').replace('-clear','')
        else:
            ftest = f.lower()
        
        if ftest in ['f098m','f105w','f110w','f125w',
                       'f128n','f130n','f132n','f127m','f139m','f153m',
                       'f140w','f160w','g102','g141']:
            # WFC3/IR
            is_optical = False
        elif ftest in ['f0560w','f0770w','f560w','f770w','f1000w',
                       'f1130w','f1280w','f1500w','f1800w','f2100w','f2550w']:
            # MIRI
            is_optical = False
        else:
            # Drizzle ACS, NIRCam to smaller pixels
            is_optical = True
                
        if is_optical:
            visit['reference'] = opt_wcs
            #is_optical = True
        else:
            visit['reference'] = ir_wcs
            #is_optical = False
        
        fi = uniq_filts[f] # res['filter'] == f
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
            ax.text(0.95, 0.95, f'N={un.N}\nexpt = {expt:.1f}', 
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
                                     kernel=kernel, clean=clean_flt, 
                                     extra_wfc3ir_badpix=extra_wfc3ir_badpix,
                                     verbose=verbose,
                                     scale_photom=scale_photom,
                                     niriss_ghost_kwargs=niriss_ghost_kwargs)
                             
        outsci, outwht, header, flist, wcs_tab = _
        
        wcs_tab.write('{0}_wcs.csv'.format(visit['product']), overwrite=True)
        
        if is_optical:
            drz = 'drc'
        else:
            drz = 'drz'
        
        # if ('PHOTFNU' not in header) & ('PHOTFLAM' in header):
        #     photfnu = utils.photfnu_from_photflam(header['photflam'],
        #                                           header['photplam'])
        #     header['PHOTFNU'] = (photfnu,
        #                          'Inverse sensitivity from PHOTFLAM, Jy/DN')
            
        if scale_nanojy is not None:
            to_njy = scale_nanojy/(header['PHOTFNU']*1.e9)
            msg = f'Scale PHOTFNU x {to_njy:.3f} to {scale_nanojy:.1f} nJy'
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            
            header['PHOTFNU'] *= to_njy
            header['PHOTFLAM'] *= to_njy
            header['BUNIT'] = f'{scale_nanojy:.1f}*nanoJansky'
            outsci *= 1./to_njy
            outwht *= to_njy**2
        else:
            #pass
            to_njy = header['PHOTFNU']*1.e9
            header['BUNIT'] = f'{to_njy:.3f}*nanoJansky'
            
        pyfits.writeto('{0}_{1}_sci.fits'.format(visit['product'], drz),
                       data=outsci, header=header, 
                       overwrite=True)
    
        pyfits.writeto('{0}_{1}_wht.fits'.format(visit['product'], drz),
                      data=outwht, header=header, 
                      overwrite=True)

    if s3output:
        files = []
        for f in uniq_filts.values:
            prod = product.format(rootname=rootname, f=f).lower()
            
            if gzip_output:
                print(f'gzip --force {prod}*fits')
                os.system(f'gzip --force {prod}*fits')

            files += glob.glob(f'{prod}*fits*')
            files += glob.glob(f'{prod}*_fp.png')
            files += glob.glob(f'{prod}*wcs.csv')
        
        
        for file in files:
            #os.system(f'aws s3 cp {file} {s3output}')
            bucket = s3output.split('s3://')[-1].split('/')[0]
            path = '/'.join(s3output.split('s3://')[-1].strip('/').split('/')[1:])
            object_name = f'{path}/{file}'
            print(f'{file} > s3://{bucket}/{object_name}')
            db.upload_file(file, bucket, object_name=object_name)
    
    return res
    

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
        
        if 'clear' in f:
            groups[f]['reference'] = opt_wcs
        elif f in ['f560w','f770w','f1000w','f1500w','f1800w']:
            groups[f]['reference'] = opt_wcs            
        elif f[1] > '1':
            groups[f]['reference'] = opt_wcs
        else:
            groups[f]['reference'] = ir_wcs
            
        _ = utils.drizzle_from_visit(groups[f], groups[f]['reference'], 
                                     pixfrac=pixfrac, 
                                     kernel=kernel, clean=False)
                             
        outsci, outwht, header, flist, wcs_tab = _
    
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
        with open(f'{ROOT_PATH}/finished.txt','w') as fp:
            fp.write(time.ctime() + '\n')
    else:
        print(f'============  Run association  ==============')
        print(f'{assoc}')
        print(f'========= {time.ctime()} ==========')
        
        with open(f'{ROOT_PATH}/visit_history.txt','a') as fp:
            fp.write(f'{time.ctime()} {assoc}\n')
        
        process_visit(assoc, clean=clean, sync=sync)
    
    
if __name__ == '__main__':
    run_all()