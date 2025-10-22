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
from ..constants import JWST_DQ_FLAGS

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


def exposure_info_from_visit(visit, assoc='', **kwargs):
    """
    Run `s3_put_exposure` for each file in visit['files']
    """

    for file in visit['files']:
        if os.path.exists(file):
            s3_put_exposure(file, visit['product'], assoc, remove_old=True, **kwargs)


def send_saturated_log(flt_file, sat_kwargs={}, remove_old=True, verbose=True, **kwargs):
    """
    Get saturated pixels from DQ extension and send to exposure_saturated DB table
    """
    from .. import jwst_utils

    #  if False:
    #      # Initialize table
    #      db.execute("""CREATE TABLE exposure_saturated (
    # eid int references exposure_files(eid),
    # i smallint,
    # j smallint
    #      )""")
    #
    #      db.execute("CREATE INDEX ON exposure_saturated (eid)")
    #      db.execute("CREATE INDEX ON exposure_saturated (eid, i, j)")
    #      db.execute("CREATE INDEX ON exposure_files (eid, detector, expstart)")
    #      db.execute("CREATE INDEX ON exposure_files (eid, file, expstart, detector)")
    #      db.execute("CREATE INDEX ON exposure_files (eid)")

    if not flt_file.endswith("_rate.fits"):
        return False

    froot = flt_file.split("_rate.fits")[0]
    row = db.SQL(f"""select eid from exposure_files where file = '{froot}'""")

    if len(row) == 0:
        msg = f"send_saturated_log: {froot} not found in exposure_files table"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        return False

    sat_file, df = jwst_utils.get_saturated_pixel_table(
        file=flt_file,
        output="file",
        **sat_kwargs,
    )

    if len(df) == 0:
        msg = f"send_saturated_log: {froot} no flagged pixels found"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        return False

    _eid = row["eid"][0]
    df.insert(0, "eid", _eid)

    if remove_old:
        db.execute(f"delete from exposure_saturated where eid = {_eid}")

    msg = 'send_saturated_log: '
    msg += f'Add {flt_file} (eid={_eid}) N={len(df)} to exposure_saturated table'
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    df.to_sql(
        'exposure_saturated',
        db._ENGINE,
        index=False,
        if_exists='append',
        method='multi'
    )

    return df


def s3_put_exposure(flt_file, product, assoc, remove_old=True, verbose=True, get_saturated=True, **kwargs):
    """
    Send exposure information to S3 and DB
    """
    import os
    from tqdm import tqdm
    import pandas as pd
    import astropy.time
    from . import db
    from .. import utils
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

        db.execute(f"""DELETE FROM exposure_saturated sat
                       USING exposure_files e
                       WHERE sat.eid = e.eid
                       AND e.file='{file}' AND e.extension='{extension}'""")

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

    msg = f'Add {file}_{extension} ({len(rows)}) to exposure_files table'
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    # Saturated pixels for persistence masking
    if get_saturated:
        _status = send_saturated_log(
            flt_file,
            remove_old=remove_old,
            verbose=verbose,
            **kwargs
        )


snowblind_kwargs = dict(require_prefix='jw', max_fraction=0.3, new_jump_flag=1024, min_radius=4, growth_factor=1.5, unset_first=True, verbose=True)

def make_visit_mosaic(assoc, base_path=ROOT_PATH, version='v7.0', pixscale=0.08, vmax=0.5, skip_existing=True, sync=True, clean=False, verbose=True, snowblind_kwargs=snowblind_kwargs, sat_kwargs={}, max_pixel_dim=20000, **kwargs):
    """
    Make a mosaic of the exposures from a visit with a tangent point selected
    from the sky tile grid
    
    Parameters
    ----------
    assoc : str
        grizli association name
    
    base_path : str
        Base working directory.  Working directory will be
        ``{base_path}/{assoc}/Prep/``
    
    version : str
        version string
    
    pixscale : float
        Reference pixel scale in arcsec (used for MIRI).  HST, LW, NIRISS will have 
        ``pixscale/2`` and SW will have ``pixscale/4``
    
    skip_existing : bool
        Don't overwrite existing assoc mosaics
    
    sync : bool
        Update results in grizli database and copy to S3
    
    clean : bool
        Remove working directory and all files in it
    
    """
    import skimage.io
    
    import astropy.wcs as pywcs
    import astropy.io.fits as pyfits
    from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                       ImageNormalize, LogStretch
                                   )

    if sync:
        xtab = utils.GTable()
        xtab['assoc_name'] = [assoc]
        xtab['version'] = version
        xtab['status'] = 1
        
        msg = f'make_visit_mosaic: {assoc} set status = 1'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        db.execute(f"delete from assoc_mosaic where assoc_name = '{assoc}' and version = '{version}'")
        db.send_to_database('assoc_mosaic', xtab, index=False, if_exists='append')

    if base_path is None:
        path = './'
    else:
        path = os.path.join(base_path, assoc, 'Prep')
    
    msg = f"make_visit_mosaic: working directory {path}\n"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    if not os.path.exists(path):
        os.makedirs(path)
        fresh_path = True
    else:
        fresh_path = False
        
    os.chdir(path)

    if fresh_path:
        msg = f"make_visit_mosaic: sync from "
        msg += f"s3://grizli-v2/HST/Pipeline/{assoc}/Prep/"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        cmd = f'aws s3 sync s3://grizli-v2/HST/Pipeline/{assoc}/Prep/ ./ --exclude "*"'
        cmd += ' --include "*rate.fits" --include "*flt.fits" --include "*flc.fits"'
        os.system(cmd)
        
    files = glob.glob('*rate.fits')
    files += glob.glob('*flt.fits')
    files += glob.glob('*flc.fits')

    # Check output pixel dimensions
    _htile, _wtile = utils.make_maximal_wcs(
        files,
        pixel_scale=pixscale,
        pad=1,
        verbose=True,
        get_hdu=False,
    )

    if np.maximum(_htile['NAXIS1'], _htile['NAXIS2']) > max_pixel_dim:
        msg = (
            "make_visit_mosaic: calculated WCS too large "
            + f"({_htile['NAXIS1']}, {_htile['NAXIS2']}) max_size={max_pixel_dim}"
        )
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        return False

    # info = utils.get_flt_info(files)
    res = res_query_from_local(files=files, extensions=[1,2])

    # Get closest sky tile
    sr = [utils.SRegion(fp) for fp in res['footprint']]
    crv = np.mean([s.centroid[0] for s in sr], axis=0)
    
    tile = db.SQL(f"""select * from mosaic_tiles
    order by SQRT(POWER(crval1 - {crv[0]},2)+POWER(crval2 - {crv[1]},2)) 
    limit 1""")[0]
    
    htile, wtile = utils.make_wcsheader(tile['crval1'],
                                        tile['crval2'],
                                        size=tile['npix']*0.1*1.1,
                                        pixscale=pixscale,
                                        get_hdu=False)

    corners = np.hstack([wtile.all_world2pix(*s.xy[0].T, 0) for s in sr])
    
    # Cutout from sky tile
    mi = corners.min(axis=1).astype(int) - 64
    ma = corners.max(axis=1).astype(int) + 64
    ma -= (ma-mi) % 4

    slx, sly = slice(mi[0], ma[0]), slice(mi[1], ma[1])
    
    wsl = pywcs.WCS(utils.get_wcs_slice_header(wtile, slx, sly))

    det = res['detector'][0]
    if det.startswith('NRC') | ('WFC' in det) | ('UVIS' in det):
        wsl = utils.half_pixel_scale(wsl)
        skip = 8
        if ('NRC' in det) & ('LONG' not in det):
            wsl = utils.half_pixel_scale(wsl)
            skip = 16
    elif det in ['NIS','IR']:
        wsl = utils.half_pixel_scale(wsl)
        skip = 8
    else:
        skip = 4
        # MIRI at nominal pixscale
        pass

    # if det in ['WFC','UVIS','IR']:
    #     weight_type = 'median_err'
    # else:
    #     weight_type = 'jwst'
    weight_type = "jwst_var"

    pixscale_mas = int(np.round(utils.get_wcs_pscale(wsl)*1000))

    msg = f"make_visit_mosaic: {assoc} {version} {det} {pixscale_mas}"
    msg += f"  tile {tile['tile']} "
    msg += f" [{slx.start}:{slx.stop}, {sly.start}:{sly.stop}]\n"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    
    res = res[res['sciext'] == 1]
    
    cutout_mosaic(
        assoc,
        ir_wcs=wsl,
        half_optical=False,
        clean_flt=False,
        s3output=False,
        gzip_output=True,
        make_exptime_map=True,
        skip_existing=skip_existing,
        kernel='square',
        pixfrac=0.8,
        res=res,
        weight_type=weight_type,
        snowblind_kwargs=snowblind_kwargs,
        saturated_lookback=1e4,
        write_sat_file=True,
        sat_kwargs=sat_kwargs,
    )

    files = glob.glob(f'{assoc}*_sci.fits*')
    files.sort()
    
    # Make a figure
    for file in files:
        imgroot = file.split('.fits')[0]
        if os.path.exists(f'{imgroot}.jpg'):
            continue
            
        with pyfits.open(file) as im:
            imsk = im[0].data[::-skip,::skip]
            msk = imsk != 0
    
            vmax = 0.8/0.0625*2**(-(np.log(skip/4)/np.log(2)*2))
            vmin = -0.1*vmax
    
            imsk[~msk] = np.nan
            norm = ImageNormalize(imsk, vmin=vmin, vmax=vmax,
                                  clip=True, stretch=LogStretch(a=1e1))
            inorm = 1-norm(imsk)
            inorm[~np.isfinite(inorm)] = 1
            
            skimage.io.imsave(f'{imgroot}.jpg', inorm,
                              plugin='pil', quality=95, check_contrast=False)
    
    os.system(f'gzip --force {assoc}*_dr*fits')

    files = glob.glob(f'{assoc}*_sci.fits.gz')
    files.sort()

    tab = utils.GTable()
    tab['file'] = files
    tab['filter'] = [f.split(assoc)[-1].split('_d')[0][1:].upper() for f in files]
    tab['assoc_name'] = assoc
    
    tab['modtime'] = [os.path.getmtime(f) for f in files]
    
    tab['footprint'] = utils.SRegion(wsl.calc_footprint()).polystr()[0]
    for c in ['tile','strip','nx','crpix1','crpix2']:
        tab[c] = tile[c]

    tab['xstart'], tab['xstop'] = slx.start, slx.stop
    tab['ystart'], tab['ystop'] = sly.start, sly.stop

    tab['input_pixscale'] = pixscale
    tab['pixscale_mas'] = pixscale_mas

    tab['detector'] = det
    
    tab['version'] = version
    tab['status'] = 2

    if sync:
        db.execute(f"delete from assoc_mosaic where assoc_name = '{assoc}' and version = '{version}'")
        db.send_to_database('assoc_mosaic', tab, index=False, if_exists='append')
        os.system(f"""aws s3 sync ./ s3://grizli-v2/assoc_mosaic/{version}/ --exclude "*" --include "{assoc}-*" --acl public-read""")
    
    if clean & (base_path is not None):
        msg = f"make_visit_mosaic: Remove {assoc} from {base_path}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        
        os.chdir(base_path)
        os.system(f'rm -rf {assoc}')
    
    return tab


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
            cmd = f"DELETE FROM wcs_log WHERE wcs_parent='{parent}'"
            if assoc is not None:
                cmd += f" AND wcs_assoc='{assoc}'"
                
            db.execute(cmd)
            
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
            
    # vis_root = np.unique(obs_id)
    # for r in vis_root:
    #     print(f'Remove {r} from shifts_log')
    #     db.execute(f"""DELETE from shifts_log
    #                            WHERE shift_dataset like '{r}%%'""")

    print('Remove from shifts_log')
    db.execute(f"DELETE from shifts_log where shift_assoc = '{assoc}'")
    
    print('Remove from wcs_log')
    db.execute(f"DELETE from wcs_log where wcs_assoc = '{assoc}'")
    
    print('Remove from tile_exposures')
    
    # Reset tiles of deleted exposures potentially from other exposures
    tile_mosaic.reset_tiles_in_assoc(assoc)
    
    # Remove deleted exposure from mosaic_tiles_exposures
    res = db.execute(f"""DELETE from mosaic_tiles_exposures t
                 USING exposure_files e
                WHERE t.expid = e.eid AND e.assoc = '{assoc}'""")
    
    res = db.execute(f"""DELETE FROM exposure_saturated sat
                   USING exposure_files e
                   WHERE sat.eid = e.eid
                   AND e.assoc='{assoc}'""")
    
    res = db.execute(f"""DELETE from exposure_files
                    WHERE assoc = '{assoc}'""")
    
    res = db.execute(f"""DELETE from assoc_mosaic
                     WHERE assoc_name = '{assoc}'
                     """)

    os.system(f'aws s3 rm s3://grizli-v2/assoc_mosaic/v7.0/ --recursive --exclude "*" --include "{assoc}*"')
                     
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


def launch_ec2_instances(nmax=50, count=None, templ='lt-0e8c2b8611c9029eb,Version=33'):
    """
    Launch EC2 instances from a launch template that run through all 
    status=0 associations/tiles and then terminate
    
    Version 19 is the latest run_all_visits.sh
    Version 20 is the latest run_all_tiles.sh
    Version 24 is run_all_visits with a new python39 environment
    Version 33 includes libglvnd-glx dependency for libGL
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


def get_assoc_yaml_from_s3(assoc, s_region=None, bucket='grizli-v2', prefix='HST/Pipeline/Input', write_file=True, fetch=True):
    """
    Get presets from yaml file on s3, if found
    """
    
    from grizli.pipeline import auto_script
    from grizli import utils
    
    LOGFILE = f'{ROOT_PATH}/{assoc}.auto_script.log.txt'
        
    s3_prefix = os.path.join(prefix, assoc + '.yaml')
    
    if fetch:
        import boto3
        s3 = boto3.resource('s3')
        s3_client = boto3.client('s3')
        bkt = s3.Bucket(bucket)
        files = [obj.key
                 for obj in bkt.objects.filter(Prefix=s3_prefix)]
    else:
        files = []
        
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
    
    MAG_LIMIT = 24
    MAG_LIMIT = 24.5
    
    sel &= (~hsc['i_modelfit_CModel_flux'].mask) & (imag < MAG_LIMIT)
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
    
    prep.table_to_radec(hsc, f'Ni2009_WCDFS_i{MAG_LIMIT:.1f}.radec')
    
    os.system(f'aws s3 cp Ni2009_WCDFS_i{MAG_LIMIT:.1f}.radec s3://grizli-v2/HST/Pipeline/Astrometry/')
    
    # os.system('aws s3 cp Ni2009_WCDFS_i24.radec s3://grizli-v2/HST/Pipeline/Astrometry/')
    
    hsc['src'] = f'Ni2009_WCDFS_i{MAG_LIMIT:.1f}'
    hsc['mag'] = 23.9-2.5*np.log10(hsc['i_modelfit_CModel_flux'])
    
    db.send_to_database('astrometry_reference',
                        hsc['ra','dec','src','mag'],
                        if_exists='append')
    
    # Compare Cosmic-Dawn catalog
    hsc = db.SQL("""select * from astrometry_reference
    where src like 'Ni2009%%'
    """)
    
    edf = utils.read_catalog('H20_EDFF_v1.2_release.fits')
    sub = (edf['HSC_i_MAG'] < 24.5)
    idx, dr, dx, dy = edf[sub].match_to_catalog_sky(hsc, get_2d_offset=True)
    
    idx, dr, dx, dy = hsc.match_to_catalog_sky(edf, get_2d_offset=True)
    ecdfs = dr.value < 30
    
    hasm = dr.value < 0.6


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
    
    # New HSC UDS query
    _query = """
    --
    -- put your SQL here
    -- uds_hsc_dr3_i24_428556
    --
    SELECT object_id,ra,dec,g_cmodel_flux,g_cmodel_fluxerr,r_cmodel_flux,r_cmodel_fluxerr,i_cmodel_mag,i_cmodel_magerr,i_cmodel_fluxerr,
    z_cmodel_flux,z_cmodel_fluxerr,y_cmodel_flux,y_cmodel_fluxerr,i_extendedness_value,i_extendedness_flag,nchild
    FROM pdr3_dud.forced
    WHERE isprimary
    AND ra > 33 AND ra < 36 AND dec > -7 AND dec < -3
    AND i_cmodel_mag < 24
    AND NOT i_cmodel_flag_badcentroid 
    AND NOT i_cmodel_flag
    
    """
    
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
blue_align_params['align_mag_limits'] = [18,26,0.15]
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
    
    dt = 0.1
    gs = mastquery.jwst.query_guidestar_log(
             mjd=(atab['t_min'].min()-dt, atab['t_max'].max()+dt),
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


def process_visit(assoc, clean=True, sync=True, s3_acl="public-read", max_dt=4, combine_same_pa=False, visit_split_shift=1.2, blue_align_params=blue_align_params, ref_catalogs=['LS_DR9', 'PS1', 'DES', 'NSC', 'GAIA'], filters=None, prep_args={}, fetch_args={}, get_wcs_guess_from_table=True, master_radec='astrometry_db', align_guess=None, with_db=True, global_miri_skyflat=None, miri_tweak_align=False, tab=None, other_args={}, do_make_visit_mosaic=True, visit_mosaic_kwargs={}, expinfo_kwargs={}, **kwargs):
    """
    Run the `grizli.pipeline.auto_script.go` pipeline on an association defined
    in the `grizli` database.
    
    Parameters
    ----------
    assoc : str
        File association name
    
    clean : bool
        Remove all intermediate products
    
    sync : bool
        Sync results to `grizli` S3 buckets and database (requires write 
        permissions).

    s3_acl : str
        S3 access control list (ACL) type for synced files (e.g., "public-read",
        "private")

    prep_args : dict
        Keyword arguments to pass in `visit_prep_args`
    
    fetch_args : dict
        Keyword arguments to pass in `fetch_files_args`
    
    get_wcs_guess_from_table : bool
        Query astrometry alignment guess from the archive, if available
    
    master_radec : str
        Reference astrometric catalog, either a filename, ``astrometry_db``, or
        `None`.  If ``astrometry_db``, will try to query astrometry reference
        sources from the `grizli` database
    
    align_guess : [float, float]
        Shift guess (not used)
    
    with_db : bool
        Try to use the `grizli` archive.  Otherwise, get the assoc information
        from the web API
    
    tab : `~astropy.table.Table`
        Manually specify the datasets to process as an association.  Needs to be some 
        result like ``SELECT * FROM assoc_table WHERE ...``.
    
    `assoc_table.status`
    
     1 = start
     2 = finished
     9 = has failed files
    10 = no wcs.log files found, so probably nothing was done?
         This is most often the case when HST visits were ignored due to 
         abnormal EXPFLAG keywords or if data couldn't be downloaded due to
         proprietary restrictions
    12 = Don't redo ever 
    
    
    """
    import os
    import glob
    
    from grizli.pipeline import auto_script
    from grizli import utils, prep

    os.chdir(f'{ROOT_PATH}/')
    
    if tab is None:
        if with_db:
            tab = db.SQL(f"""SELECT * FROM assoc_table
                         WHERE assoc_name='{assoc}'
                         AND status != 99
                         """)
        else:
            _url = f'http://grizli-cutout.herokuapp.com/assoc_json?name={assoc}'
            tab = utils.read_catalog(_url, format='pandas.json')
            sync = False
            get_wcs_guess_from_table = False
    else:
        sync = False
        get_wcs_guess_from_table = False
        
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
                                 s_region=tab['footprint'][0],
                                 fetch=with_db)
           
    kws['visit_prep_args']['reference_catalogs'] = ref_catalogs
    if filters is None:
        kws['filters'] = [f.split('-')[0] 
                          for f in np.unique(tab['filter']).tolist()]            
    else:
        kws['filters'] = filters
    
    # Negative values of visit_split_shift override keyword args
    if kws['parse_visits_args']['visit_split_shift'] > 0:
        kws['parse_visits_args']['max_dt'] = max_dt
        kws['parse_visits_args']['visit_split_shift'] = visit_split_shift
        kws['parse_visits_args']['combine_same_pa'] = combine_same_pa
    else:
        kws['parse_visits_args']['visit_split_shift'] *= -1
    
    if len(kws['visit_prep_args']['align_mag_limits']) != 3:
        align_kws_are_default = False
    else:
        align_kws_are_default = np.allclose(kws['visit_prep_args']['align_mag_limits'],
                                        [14, 24, 0.05])
    
    is_blue_optical = False
    for filt in ['f218w','f225w','f275w','f336w','f390w','f435w','f475w']:
        if filt in assoc:
            is_blue_optical = True
            break
            
    if (is_blue_optical) & (blue_align_params is not None) & (align_kws_are_default):
        for k in blue_align_params:
            kws['visit_prep_args'][k] = blue_align_params[k]
    
    for k in prep_args:
        kws['visit_prep_args'][k] = prep_args[k]
    
    for k in fetch_args:
        kws['fetch_files_args'][k] = fetch_args[k]
    
    kws['kill'] = 'preprocess'
    
    if global_miri_skyflat is not None:
        kws['global_miri_skyflat'] = global_miri_skyflat
    
    if master_radec is not None:
        kws['preprocess_args']['master_radec'] = master_radec
    
    if 'miri_tweak_align' in kws:
        miri_tweak_align = kws.pop('miri_tweak_align')
        
    for fi in ['f560w','f770w','f1000w','f1130w','f1280w',
               'f1500w','f1800w','f2100w','f2550w']:
        
        if fi in assoc:
            # Don't run tweak align for MIRI long filters
            miri_prep = dict(run_tweak_align=miri_tweak_align,
                             align_mag_limits=[14,28,0.2],
                             align_clip=20, 
                             align_simple=True)
                           
            for k in miri_prep:
                if k not in prep_args:
                    kws['visit_prep_args'][k] = miri_prep[k]
                    
            break
    
    for k in other_args:
        kws[k] = other_args[k]
    
    ## Full parameter file as executed
    auto_script.write_params_to_yml(kws, output_file=f'{assoc}.run.yaml')
    
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
                    exposure_info_from_visit(v, assoc=assoc, **expinfo_kwargs)
    
    if sync:
        add_shifts_log(assoc=assoc, remove_old=True, verbose=True)
        add_wcs_log(assoc=assoc, remove_old=True, verbose=True)
        
    if (do_make_visit_mosaic) & (with_db):
        make_visit_mosaic(assoc, sync=sync,
                          base_path=ROOT_PATH,
                          clean=False,
                          **visit_mosaic_kwargs)
        
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
                  --include "Prep/*sat.csv.gz" \
                  --include "Prep/*s.log" \
                  --include "Prep/*visits.*" \
                  --include "Prep/*skyflat.*" \
                  --include "Prep/*angles.*" \
                  --include "*fail*" \
                  --include "RAW/*[nx][tg]" --acl {s3_acl}"""
        
            os.system(cmd)
            print('\n# Sync\n' + cmd.replace('     ', '') + '\n')
            os.chdir(f'{ROOT_PATH}')
            
        files = glob.glob(f'{assoc}*.*')
        for file in files:
            os.system(f'aws s3 cp {file} ' + 
                      f' s3://grizli-v2/HST/Pipeline/{assoc}/ --acl {s3_acl}')
    
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
              'f250m','f480m'
              'f560w','f770w','f1000w','f1280w',
              'f1500w','f1800w','f2100w',
              ]:
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


def res_query_from_local(files=None, filters=None, extensions=[1], extra_keywords=[]):
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
        with pyfits.open(file) as im:
        
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
            
            extra_data = []
            for k in extra_keywords:
                if k in im[0].header:
                    extra_data.append(im[0].header[k])
                else:
                    extra_data.append(np.nan)
            
            for ex in extensions:
                if ('SCI', ex) not in im:
                    continue
                    
                wcs = pywcs.WCS(im['SCI', ex].header, fobj=im)
                sr = utils.SRegion(wcs)
                fp = sr.polystr()[0]
            
                rows.append([ds, ext, ex, assoc, filt, pup, expt, fp, det] + extra_data)
    
    extra_names = [k.lower() for k in extra_keywords]
    
    res = utils.GTable(names=['dataset','extension','sciext','assoc',
                              'filter','pupil','exptime','footprint',
                              'detector'] + extra_names,
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


def cutout_mosaic(rootname='gds', product='{rootname}-{f}', ra=53.1615666, dec=-27.7910651, size=5*60, theta=0., filters=['F160W'], ir_scale=0.1, ir_wcs=None, res=None, half_optical=True, kernel='point', pixfrac=0.33, make_figure=True, skip_existing=True, clean_flt=True, gzip_output=True, s3output='s3://grizli-v2/HST/Pipeline/Mosaic/', split_uvis=True, extra_query='', query_order=' ORDER BY e.filter, e.dataset', just_query=False, extra_wfc3ir_badpix=True, fix_niriss=True, scale_nanojy=10, verbose=True, weight_type='jwst_var', rnoise_percentile=99, get_dbmask=True, niriss_ghost_kwargs={}, snowblind_kwargs=None, scale_photom=True, context=None, calc_wcsmap=False, make_exptime_map=False, expmap_sample_factor=4, keep_expmap_small=True, write_ctx=False, **kwargs):
    """
    Make mosaic from exposures defined in the exposure database
    
    Parameters
    ----------
    rootname : str
        Output file rootname
    
    product : str
        File name structure that will be parsed with ``product.format(rootname, f)``,
        where ``f`` is the filter name
    
    ra, dec : float
        Cutout center in decimal degrees
    
    size : float
        Cutout size in arcsec
    
    theta : float
        Position angle of the output WCS
    
    filters : list
        List of filter names to use for the mosaic.
    
    ir_scale : float
        Pixel scale in arcsec
    
    ir_wcs : `~astropy.wcs.WCS`
        Specify the output WCS explicitly rather than deriving from
        ``ra, dec, size, theta``.
    
    res : None, `~astropy.table.Table`
        Explicit list of exposure file metadata, e.g., from 
        `~grizli.aws.visit_processor.res_query_from_local`.  If not specified, the 
        grizli-processed exposures that overlap with the requested cutout WCS will be
        determined by querying the `grizli` database (`~grizli.aws.db`).
    
    half_optical : bool
        Use WCS with pixels 2x smaller than defined in `ir_wcs` for ACS/WFC, WFC3/UVIS,
        NIRCam and NIRISS filters.
    
    kernel, pixfrac : str, float
        Drizzle parameters
    
    skip_existing : bool
        Skip `product` combinations already found in the working directory
    
    clean_flt : bool
        Remove exposures one-by-one after they have been added to the cutout / mosaic.
    
    gzip_output : bool
        Compress the output files after completion
    
    s3output : str
        Path name of an AWS S3 bucket where products will be sent upon completion
    
    split_uvis : bool
        Make separate mosaics for ACS/WFC and WFC3/UVIS filters with the same name
    
    extra_query : str
        Extra SQL query criteria added to the DB query if ``res`` not provided
    
    query_order : str
        Query sorting in SQL syntax

    just_query : bool
        If True, return the result of the DB query

    extra_wfc3ir_badpix : str
        Add additional bad pixel mask for WFC3/IR and NIRCam from the bad pixel files 
        provided in `grizli.data`.
    
    fix_niriss : bool
        Separate filter names for NIRISS
    
    scale_nanojy : bool
        Photometric scale of the output image, i.e., a pixel value of one corresponds
        to a flux density of `scale_nanojy` nJy per pix.
    
    verbose : bool
        Verbose messaging
    
    weight_type : 'jwst', 'median_err', 'err', 'median_variance', 'jwst_var'
        Drizzle weight strategy (see `~grizli.utils.drizzle_from_visit`)
    
    rnoise_percentile : float
        See `~grizli.utils.drizzle_from_visit`
    
    niriss_ghost_kwargs : dict
        Parameters of the NIRISS ghost flagging
    
    snowblind_kwargs : dict
        Arguments to pass to `~grizli.utils.jwst_snowblind_mask` if `snowblind` hasn't
        already been run on JWST exposures
    
    scale_photom : bool
        See `~grizli.utils.drizzle_from_visit`
    
    calc_wcsmap : bool
        See `~grizli.utils.drizzle_from_visit`

    context : str, None
        ``CRDS_CONTEXT`` to use for optionally controlling JWST zeropoints.  If ``None``
        then first try to get from ``os.getenv('CRDS_CONTEXT')`` and then fall back to
        ``grizli.jwst_utils.CRDS_CONTEXT``.

    write_ctx : bool
        Optionally output context map
    
    Returns
    -------
    Image mosaic files with the requested WCS and filters
    
    """
    import os
    import glob
    import matplotlib.pyplot as plt
    import astropy.io.fits as pyfits
    
    from mastquery import overlaps
    try:
        from .. import jwst_utils
    except ImportError:
        jwst_utils = None
    
    ORIG_LOGFILE = utils.LOGFILE

    if context is None:
        if jwst_utils is not None:
            _env_context = os.getenv("CRDS_CONTEXT")
            context = jwst_utils.CRDS_CONTEXT if _env_context is None else _env_context
        else:
            context = "jwst_0989.pmap"

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
    
    SQL += query_order

    if res is None:
        import boto3
        res = db.SQL(SQL)

    if just_query:
        return res

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
        
        utils.LOGFILE = f"{visit['product']}.log.txt"
        
        if (len(glob.glob(visit['product'] + '*fits*')) > 0) & skip_existing:
            print('Skip ' + visit['product'])
            continue
            
        msg = '============ ' + visit['product'] + '============'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        
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
        # un = utils.Unique(res['dataset'][fi], verbose=False)
        unique_idx = []
        unique_files = []
        for i, dataset_i in enumerate(res['dataset'][fi]):
            if dataset_i in unique_files:
                continue
            else:
                unique_files.append(dataset_i)
                unique_idx.append(i)
        
        unique_idx = np.array(unique_idx)
        
        # ds = res[fi]['dataset'][unique_idx]
        # ex = res[fi]['extension'][unique_idx]
        # assoc = res[fi]['assoc'][unique_idx]
        ffp = res[fi]['footprint'][unique_idx]
        expt = res[fi]['exptime'][unique_idx].sum()

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
            ax.text(
                0.95, 0.95,
                f'N={len(unique_idx)}\nexpt = {expt:.1f}',
                ha='right', va='top', transform=ax.transAxes
            )

            fig.savefig(visit['product']+'_fp.png')
            plt.close('all')

        visit['files'] = [
            "{dataset}_{extension}.fits".format(**row)
            for row in res['dataset','extension','assoc'][fi][unique_idx]
        ]

        visit['awspath'] = [
            "grizli-v2/HST/Pipeline/{assoc}/Prep".format(**row)
            for row in res['dataset','extension','assoc'][fi][unique_idx]
        ]

        # visit['files'] = [f"{ds[un[v]][0]}_{ex[un[v]][0]}.fits"
        #                   for v in un.values]
        #
        # visit['awspath'] = [f"grizli-v2/HST/Pipeline/{assoc[un[v]][0]}/Prep"
        #                   for v in un.values]
        
        visit['footprints'] = []
        # for v in un.values:
        #     fps = ffp[un[v]]
        #     fp_i = None
        #     for fpi in fps:
        #         sr = utils.SRegion(fpi)
        #         for p in sr.shapely:
        #             if fp_i is None:
        #                 fp_i = p
        #             else:
        #                 fp_i = fp_i.union(p)
        for fpi in ffp:
            fp_i = None
            sr = utils.SRegion(fpi)
            for p in sr.shapely:
                if fp_i is None:
                    fp_i = p
                else:
                    fp_i = fp_i.union(p)
            
            visit['footprints'].append(fp_i)
                           
        _ = utils.drizzle_from_visit(
            visit,
            visit['reference'],
            pixfrac=pixfrac,
            kernel=kernel,
            clean=clean_flt,
            extra_wfc3ir_badpix=extra_wfc3ir_badpix,
            verbose=verbose,
            scale_photom=scale_photom,
            context=context,
            get_dbmask=get_dbmask,
            niriss_ghost_kwargs=niriss_ghost_kwargs,
            snowblind_kwargs=snowblind_kwargs,
            weight_type=weight_type,
            rnoise_percentile=rnoise_percentile,
            calc_wcsmap=calc_wcsmap,
            **kwargs,
        )

        outsci, outwht, outvar, outctx, header, flist, wcs_tab = _
        
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
            
            header['OPHOTFNU'] = header['PHOTFNU'], 'Original PHOTFNU before scaling'
            header['PHOTFNU'] *= to_njy
            header['PHOTFLAM'] *= to_njy
            header['BUNIT'] = f'{scale_nanojy:.1f}*nanoJansky'
            outsci *= 1./to_njy
            outwht *= to_njy**2
            if outvar is not None:
                outvar *= 1./to_njy**2
        else:
            #pass
            to_njy = header['PHOTFNU']*1.e9
            header['BUNIT'] = f'{to_njy:.3f}*nanoJansky'

        sci_file = '{0}_{1}_sci.fits'.format(visit['product'], drz)

        pyfits.writeto(
            sci_file,
            data=outsci,
            header=header,
            overwrite=True
        )

        # Weight
        pyfits.writeto(
            '{0}_{1}_wht.fits'.format(visit['product'], drz),
            data=outwht,
            header=header,
            overwrite=True
        )

        # Variance
        if outvar is not None:
            pyfits.writeto(
                '{0}_{1}_var.fits'.format(visit['product'], drz),
                data=outvar,
                header=header,
                overwrite=True
            )

        if write_ctx:
            pyfits.writeto('{0}_{1}_ctx.fits'.format(visit['product'], drz),
                        data=outctx, header=header, 
                        overwrite=True)

        if make_exptime_map:
            matched_exptime_map(
                sci_file,
                sample_factor=expmap_sample_factor,
                keep_small=keep_expmap_small,
                output_type='file',
                verbose=True
            )

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
            files += glob.glob(f'{prod}*log.txt')
        
        
        for file in files:
            #os.system(f'aws s3 cp {file} {s3output}')
            bucket = s3output.split('s3://')[-1].split('/')[0]
            path = '/'.join(s3output.split('s3://')[-1].strip('/').split('/')[1:])
            object_name = f'{path}/{file}'
            print(f'{file} > s3://{bucket}/{object_name}')
            db.upload_file(file, bucket, object_name=object_name)
    
    return res


def matched_exptime_map(ref_file, sample_factor=4, keep_small=True, output_type='file', verbose=True):
    """
    Make an exposure time map by querying footprints from the exposure database
    
    Parameters
    ----------
    ref_file : str
        Reference mosaic file.  Needs a valid WCS and header keywords that can be
        parsed with `~grizli.utils.parse_filter_from_header`.
    
    sample_factor : int
        Undersampling factor
    
    keep_small : bool
        Store the output on the undersampled grid
    
    output_type : str
        Output behavior:
        
        - ``file`` write a file `ref_file.replace('sci.fits','exp.fits')` if 
          `ref_file.endswith('sci.fits*')`
        
        - ``hdu``: get a `~astropy.io.fits.ImageHDU` object
        
        - ``array``: return a 2D array
    
    Returns
    -------
    output : see ``output_type``
    
    """
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    import astropy.units as u
    
    from tqdm import tqdm
    import scipy.ndimage as nd

    with pyfits.open(ref_file) as im:

        filter_key = utils.parse_filter_from_header(im[0].header)
        wcs = pywcs.WCS(im[0].header)
        header = im[0].header.copy()
        
        if ('FLT00001' in header) & ('NDRIZIM' in header):
            added_files = []
            for i in range(header['NDRIZIM']):
                k = f'FLT{i+1:05d}'
                if k in header:
                    added_files.append('_'.join(header[k].split('_')[:-1]))
                    
            if len(added_files) == 0:
                added_files = None
                
    wsr = utils.SRegion(wcs.calc_footprint())

    exp = db.SQL(f"""SELECT dataset, footprint, exptime, filter, sciext
    FROM exposure_files
    WHERE filter = '{filter_key}'
    AND polygon(footprint) && polygon('{wsr.polystr()[0]}')
    """)
        
    NX = header['NAXIS1']
    NY = header['NAXIS2']
    yp, xp = np.indices((NY//sample_factor, NX//sample_factor))*sample_factor
    sh = xp.shape

    msg = f'matched_exptime_map: {ref_file}, sample_factor={sample_factor}\n'
    msg += f'matched_exptime_map: ({NY},{NX}) > ({sh[0]}, {sh[1]})\n'
    msg += f'matched_exptime_map: {len(exp)} exposures in {filter_key}'
    
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    rr, dd = wcs.all_pix2world(xp.flatten(), yp.flatten(), 0)
    coo = np.array([rr, dd]).T

    expm = np.zeros(coo.shape[0], dtype=np.uint32)
    
    added = []
    
    for footprint, exptime, ds, ext in tqdm(zip(exp['footprint'], exp['exptime'], 
                                                exp['dataset'], exp['sciext'])):
        if added_files is not None:
            if (ds not in added_files) | ((ds,ext) in added):
                continue
        
        added.append((ds,ext))
        
        sr = utils.SRegion(footprint)
        in_fp = sr.path[0].contains_points(coo) #.reshape(sh)

            
        expm[in_fp] += np.uint32(exptime) # *in_fp
        
    ### Put back in original scale
    if keep_small:
        full_expm = expm.reshape(sh)
        
        header['CRPIX1'] /= sample_factor
        header['CRPIX2'] /= sample_factor
        for i in [1,2]:
            for j in [1,2]:
                k = f'CD{i}_{j}'
                if k in header:
                    header[k] *= sample_factor
        
    else:
        
        full_expm = np.zeros((NY, NX), dtype=np.uint32)

        for i, (xi, yi) in enumerate(zip(xp.flatten(), yp.flatten())):
            full_expm[yi, xi] = expm[i]

        if sample_factor > 1:
            full_expm = nd.maximum_filter(full_expm, sample_factor)
    
    header['BUNIT'] = 'second'
    header['SAMPLE'] = sample_factor, 'Sampling factor'
    header['NXORIG'] = NX
    header['NYORIG'] = NY
        
    mosaic_pscale = utils.get_wcs_pscale(wcs)
    
    if 'PIXAR_SR' in header:
        orig_pscale = np.sqrt((header['PIXAR_SR']*u.steradian).to(u.arcsec**2)).value
    else:
        # ToDo: HST pixel scales
        if 'DETECTOR' in header:
            if header['DETECTOR'] == 'IR':
                orig_pscale = 0.128
            elif header['DETECTOR'] == 'WFC':
                orig_pscale = 0.05
            elif header['DETECTOR'] == 'UVIS':
                orig_pscale = 0.04
            else:
                orig_pscale = 0.128
        else:
            orig_pscale = 0.128
            
    #pscale_ratio = orig_pscale/mosaic_pscale
    
    header['MOSPSCL'] = mosaic_pscale, 'Mosaic pixel scale arcsec'
    header['ORIGPSCL'] = orig_pscale, 'Original detector pixel scale arcsec'
    
    phot_scale = 1.
    
    if 'PHOTMJSR' in header:
        phot_scale /= header['PHOTMJSR']
    
    if 'PHOTSCAL' in header:
        phot_scale /= header['PHOTSCAL']
    
    if 'OPHOTFNU' in header:
        phot_scale *= header['OPHOTFNU'] / header['PHOTFNU']
        
    header['DNTOEPS'] = ((orig_pscale/mosaic_pscale)**2 * phot_scale,
                         'Inverse flux conversion back to e per second')
    
    if output_type == 'array':
        return full_expm
    
    hdu = pyfits.ImageHDU(header=header, data=full_expm)
    
    if '_sci.fits' in ref_file:
        output_file = ref_file.replace('_sci.fits','_exp.fits')
    else:
        output_type = 'hdu'
        
    if output_type == 'hdu':
        return hdu
    else:
        msg = f'matched_exptime_map: write to {output_file}'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        
        pyfits.writeto(output_file, data=full_expm, header=header, overwrite=True)
        return output_file


def show_epochs_filter(ra=150.1, dec=2.2, outroot=None, size=4, pixel_scale=0.04, filter='F444W-CLEAR', kernel='point', pixfrac=1., use_cutout=True, scale_native=1.0, dq_flags=1+1024+4096, round_days=4, jwst_dq_flags=JWST_DQ_FLAGS, cleanup=True, vmax=1.0, cmap='magma_r', panel_size=2.5, **kwargs):
    """
    Make a figure showing cutouts around a particular position for all exposures 
    covering that point
    """
    import glob
    import secrets

    import numpy as np
    
    import matplotlib.pyplot as plt
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    
    from ..jwst_utils import get_jwst_dq_bit
    
    tmp_hash = secrets.token_urlsafe(16)[:5]
    tmp_name = f'epoch-{tmp_hash}'

    cut = cutout_mosaic(
        tmp_name,
        ra=ra,
        dec=dec,
        ir_scale=pixel_scale,
        kernel=kernel,
        pixfrac=pixfrac,
        half_optical=False,
        size=size,
        s3output=None,
        gzip_output=False,
        filters=[filter],
        clean_flt=False,
        skip_existing=False,
        **kwargs,
    )

    cut_file = glob.glob(f'{tmp_name}*{filter}*sci.fits'.lower())
    with pyfits.open(cut_file[0]) as im:
        outh = im[0].header
        full_sci = im[0].data*1

    with pyfits.open(cut_file[0].replace('_sci', '_wht')) as im:
        full_wht = im[0].data*1

    files = []
    for d in cut['dataset']:
        files += glob.glob(f'{d}*fits')

    files = np.unique(files).tolist()
    
    info = utils.get_flt_info(files=files)
    
    so = np.argsort(info['EXPSTART'])
    info = info[so]
    N = len(info)

    res = res_query_from_local(files=info['FILE'])
    
    # Loop over files
    thumbs = []
    
    for i in range(N):
        cut = cutout_mosaic(
            tmp_name,
            ra=ra,
            dec=dec,
            res=res[i:i+1],
            ir_scale=pixel_scale,
            kernel=kernel,
            pixfrac=pixfrac,
            half_optical=False,
            size=size,
            s3output=None,
            gzip_output=False,
            filters=[filter],
            clean_flt=False,
            skip_existing=False,
            **kwargs,
        )
        
        with pyfits.open(cut_file[0]) as im:
            thumb_sci = im[0].data*1

        with pyfits.open(cut_file[0].replace('_sci', '_wht')) as im:
            thumb_wht = im[0].data*1
        
        thumbs.append([thumb_sci, thumb_wht])

    if round_days is not None:
        info['DAY'] = (np.round(info['EXPSTART'] / round_days) * round_days).astype(int)
        une = utils.Unique(info['DAY'], verbose=True)
        
        stack_thumbs = []
        labels = []
        
        outh['NEPOCH'] = (une.N, 'Number of epochs')
        
        for k, d in enumerate(une.values):
            ix = une[d]
            ni = ix.sum()

            cut = cutout_mosaic(
                tmp_name,
                ra=ra,
                dec=dec,
                res=res[ix],
                ir_scale=pixel_scale,
                kernel=kernel,
                pixfrac=pixfrac,
                half_optical=False,
                size=size,
                s3output=None,
                gzip_output=False,
                filters=[filter],
                clean_flt=False,
                skip_existing=False,
                **kwargs,
            )
        
            with pyfits.open(cut_file[0]) as im:
                thumb_sci = im[0].data*1

            with pyfits.open(cut_file[0].replace('_sci', '_wht')) as im:
                thumb_wht = im[0].data*1
        
            stack_thumbs.append([thumb_sci, thumb_wht])

            outh[f'EPOCH{k}'] = (info['EXPSTART'][ix].mean(), 'Combined epoch')
            outh[f'EPCHN{k}'] = (ni, 'Files combined at this epoch')
            
            ii = np.where(ix)[0]
            label = f"{1:<3}  {info['DATE-OBS'][ii[0]]}  {info['TIME-OBS'][ii[0]][:-4]}"
            label += f"\n{ni:<3}"
            label += f"  {info['DATE-OBS'][ii[-1]]}  {info['TIME-OBS'][ii[-1]][:-4]}"
            labels.append(label)

    else:
        labels = []
        for i in range(len(info)):
            label = f"{info['DATE-OBS'][i]}   {info['TIME-OBS'][i][:-4]}"
            label += '\n' + f"{files[i].split('_rate')[0]}"
            labels.append(label)
        
        outh['NEPOCH'] = (len(info), 'Number of files')
        for k in range(len(info)):
            outh[f'EPOCH{k}'] = (info['EXPSTART'][k], 'File epoch')
        
    # Build FITS HDU
    outh['FILTER'] = filter
    outh['KERNEL'] = (kernel, 'Drizzle kernel')
    outh['PIXFRAC'] = (pixfrac, 'Drizzle pixfrac')
    outh['NFILES'] = len(info)
    for i, row in enumerate(info):
        outh[f'FILE{i}'] = row['FILE']
        outh[f'START{i}'] = row['EXPSTART'], 'EXPSTART MJD'
        outh[f'EXPT{i}'] = row['EXPTIME'], 'EXPTIME s'
        
    cube_sci = np.array([thumb[0] for thumb in thumbs])
    cube_wht = np.array([thumb[1] for thumb in thumbs])
    
    hdul = pyfits.HDUList([
        pyfits.PrimaryHDU(header=outh),
        pyfits.ImageHDU(data=full_sci, name='SCI', header=outh),
        pyfits.ImageHDU(data=full_wht, name='WHT', header=outh),
        pyfits.ImageHDU(data=cube_sci, name='EXP_SCI', header=outh),
        pyfits.ImageHDU(data=cube_wht, name='EXP_WHT', header=outh),
    ])

    if round_days is not None:
        stack_sci = np.array([thumb[0] for thumb in stack_thumbs])
        stack_wht = np.array([thumb[1] for thumb in stack_thumbs])
        hdul.append(pyfits.ImageHDU(data=stack_sci, name='STACK_SCI', header=outh))
        hdul.append(pyfits.ImageHDU(data=stack_wht, name='STACK_WHT', header=outh))
        
    # Make the figure
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    if round_days is not None:
        NX = len(stack_thumbs)
    else:
        NX = len(thumbs)

    fig, axes = plt.subplots(1, NX+1, figsize=(panel_size * (NX+1), panel_size),
                             sharex=True, sharey=True)
    
    kws = dict(vmin=-0.1*vmax, vmax=vmax, cmap=cmap, origin='lower')
    
    fs = 7
    
    for i in range(len(labels)):
        if round_days is not None:
            axes[i].imshow(stack_thumbs[i][0], **kws)
        else:
            axes[i].imshow(thumbs[i][0], **kws)

        axes[i].text(0.5, 0.05, labels[i],
                     ha='center', va='bottom', transform=axes[i].transAxes,
                     fontsize=fs, color='k', bbox=dict(fc='w', alpha=0.8, ec='None'),
                    )
        
    axes[-1].imshow(full_sci, **kws)
    axes[-1].text(0.5, 0.95, f'{ra:.6f}, {dec:.6f}',
        ha='center', va='bottom', transform=axes[-1].transAxes,
        fontsize=fs, color='k', bbox=dict(fc='w', alpha=0.8, ec='None'),
    )

    axes[-1].text(0.5, 0.05, filter,
                     ha='center', va='bottom', transform=axes[-1].transAxes,
                     fontsize=fs, color='k', bbox=dict(fc='w', alpha=0.8, ec='None'),
                )
    
    sh = full_sci.shape
    for ax in axes:
        ax.set_xticks((-0.5, sh[0]-0.5))
        ax.set_yticks(ax.get_xticks())
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    fig.tight_layout(pad=0)
    fig.tight_layout(pad=0.5)

    if cleanup:
        for file in files:
            if os.path.exists(file):
                os.remove(file)

        xfiles = glob.glob(f'{tmp_name}*')
        for file in xfiles:
            os.remove(file)

    if outroot is not None:
        fig.savefig(f'{outroot}.png')
        hdul.writeto(f'{outroot}.fits', overwrite=True, output_verify='fix')

    if round_days is not None:
        anim = plot_epochs_differences(
            hdul, outroot=outroot, vmax=vmax, cmap=cmap, panel_size=panel_size
        )
    else:
        anim = None
    
    return info, fig, hdul, anim


def plot_epochs_differences(hdul, outroot=None, vmax=1, cmap='magma_r', panel_size=2.5):
    """
    Make a figure plotting the epoch difference images
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    import astropy.time

    thumbs = hdul['STACK_SCI'].data
    
    NX = thumbs.shape[0]

    fig, axes = plt.subplots(2, NX+1, figsize=(panel_size * (NX+1), panel_size * 2),
                             sharex=True, sharey=True)
    
    kws = dict(vmin=-0.1*vmax, vmax=vmax, cmap=cmap, origin='lower')
    
    fs = 7
    
    diffs = []
    text_labels = []
    
    for i in range(NX):
        axes[0][i].imshow(thumbs[i,:,:], **kws)

        ep = astropy.time.Time(hdul[0].header[f'EPOCH{i}'], format='mjd').iso
        txt = axes[0][i].text(
            0.5, 0.05, ep[:-4],
            ha='center', va='bottom', transform=axes[0][i].transAxes,
            fontsize=fs, color='k',
            bbox=dict(fc='w', alpha=0.8, ec='None'),
            animated=True,
        )
        
        whts = np.ones(NX, dtype=bool)
        whts[i] = False
        num_ = (hdul['STACK_SCI'].data[whts,:,:] * hdul['STACK_WHT'].data[whts,:,:])
        den_ = hdul['STACK_WHT'].data[whts,:,:].sum(axis=0)
        other = num_.sum(axis=0) / den_
        diff = thumbs[i,:,:] - other
        
        diffs.append(diff)
        text_labels.append(txt)

        axes[1][i].imshow(diff, **kws)

    axes[0][-1].imshow(hdul['SCI'].data, **kws)
    axes[0][-1].text(
        0.5, 0.95,
        '{CRVAL1:.6f}, {CRVAL2:.6f}'.format(**hdul['SCI'].header), 
        ha='center', va='top', transform=axes[0][-1].transAxes,
        fontsize=fs, color='k', bbox=dict(fc='w', alpha=0.8, ec='None'),
    )

    axes[0][-1].text(0.5, 0.05, hdul['SCI'].header['FILTER'],
                     ha='center', va='bottom', transform=axes[0][-1].transAxes,
                     fontsize=fs, color='k', bbox=dict(fc='w', alpha=0.8, ec='None'),
                )
    
    sh = hdul['SCI'].data.shape
    for i in [0,1]:
        for j in range(NX):
            ax = axes[i,j]
            ax.set_xticks((-0.5, sh[0]-0.5))
            ax.set_yticks(ax.get_xticks())
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
    fig.tight_layout(pad=0)
    fig.tight_layout(pad=0.5)
    # fig.savefig('epoch_diffs.png')
    fig.savefig(f'{outroot}_diffs.png')

    # panel for animated diffs
    im = axes[1][-1].imshow(diffs[0], animated=True, **kws)
    def update(i):
        im.set_array(diffs[i])
        for j in range(len(diffs)):
            if i == j:
                text_labels[j].set_bbox(dict(fc='w', alpha=0.8, ec='magenta'))
            else:
                text_labels[j].set_bbox(dict(fc='w', alpha=0.8, ec='None'))
                
        return [im] + text_labels

    animation_fig = animation.FuncAnimation(
        fig,
        update,
        frames=len(diffs),
        interval=400,
        blit=True,
        repeat_delay=10,
    )

    if outroot is not None:
        animation_fig.save(f'{outroot}_diffs.gif')

    return fig, animation_fig


def make_mosaic(jname='', ds9=None, skip_existing=True, ir_scale=0.1, half_optical=False, pad=16, kernel='point', pixfrac=0.33, sync=True, ir_wcs=None, weight_type='jwst_var', write_ctx=False, **kwargs):
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
            
        _ = utils.drizzle_from_visit(
            groups[f],
            groups[f]['reference'],
            pixfrac=pixfrac,
            kernel=kernel,
            clean=False,
            weight_type=weight_type,
            **kwargs
        )
                             
        outsci, outwht, outvar, outctx, header, flist, wcs_tab = _
    
        pyfits.writeto(
            groups[f]['product']+'_drz_sci.fits',
            data=outsci,
            header=header,
            overwrite=True
        )
    
        pyfits.writeto(
            groups[f]['product']+'_drz_wht.fits',
            data=outwht,
            header=header,
            overwrite=True
        )

        if outvar is not None:
            pyfits.writeto(
                groups[f]['product']+'_drz_var.fits',
                data=outvar,
                header=header,
                overwrite=True
            )

        if write_ctx:
            pyfits.writeto(groups[f]['product']+'_drz_ctx.fits',
                        data=outctx, header=header, 
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

    # nassoc = db.SQL('select count(distinct(assoc_name)) '
    #                      ' from assoc_table')['count'][0]
    nassoc = db.SQL("""
        SELECT count(distinct(assoc_name))
        FROM assoc_table WHERE status = 0
    """)['count'][0]

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


def snowblind_exposure_mask(assoc, file, verbose=True, clean=True, new_jump_flag=1024, min_radius=4, growth_factor=1.5, unset_first=True, run_again=False, force=False):
    """
    Run snowblind on a previously processed exposures
    """
    import time
    import boto3
    import astropy.io.fits as pyfits
    
    if 0:
        db.execute("drop table exposure_snowblind")
        db.execute("""create table exposure_snowblind as 
        select assoc, dataset || '_' || extension || '.fits' as file, instrume 
        from exposure_files 
        where instrume in ('NIRCAM','NIRISS')
        group by assoc, dataset, extension, instrume
        """)
        db.execute("alter table exposure_snowblind add column status int default 70")
        db.execute("alter table exposure_snowblind add column maskfrac real default 0.")
        db.execute("alter table exposure_snowblind add column npix int default 0")
        db.execute("alter table exposure_snowblind add column utime double precision default 0.")
    
    status = db.SQL(f"""select * from exposure_snowblind
    where assoc = '{assoc}' and file = '{file}'
    """)
    if len(status) == 0:
        msg = f"File {assoc} {file} not found"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return False
        
    if (status['status'].max() != 0) & (~force):
        msg = "{assoc}  {file}  status={status} locked".format(**status[0])
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return False
    
    # set lock status
    db.execute(f"""update exposure_snowblind set status = 1 
    where assoc = '{assoc}' and file = '{file}'
    """)
    
    # Run the snowblind mask    
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket('grizli-v2')

    local_file = os.path.basename(file)
    s3_prefix = f'HST/Pipeline/{assoc}/Prep/{file}'
    
    if os.path.exists(local_file):
        clean = False
    else:
        msg = f'Fetch {s3_prefix}'
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        bkt.download_file(s3_prefix, local_file,
                          ExtraArgs={"RequestPayer": "requester"})
    
    do_run = True
    with pyfits.open(local_file) as im:
        dq = im['DQ'].data & new_jump_flag
        maskfrac = (dq > 0).sum() / dq.size
        if 'SNOWBLND' in im['SCI'].header:
            do_run = run_again

    if do_run:        
        dq, maskfrac = utils.jwst_snowblind_mask(local_file,
                                       new_jump_flag=new_jump_flag,
                                       min_radius=min_radius,
                                       growth_factor=growth_factor,
                                       unset_first=unset_first)
    
    with pyfits.open(local_file, mode='update') as im:
        if unset_first:
            im['DQ'].data -= im['DQ'].data & new_jump_flag
        
        im['DQ'].data |= dq.astype(im['DQ'].data.dtype)
        im['SCI'].header['SNOWMASK'] = (True, 'Snowball mask applied')
        im['SCI'].header['SNOWBLND'] = (True, 'Mask with snowblind')
        im['SCI'].header['SNOWBALF'] = (maskfrac,
                             'Fraction of masked pixels in snowball mask')
        
        im.flush()
        npix = ((im['DQ'].data & new_jump_flag) > 0).sum()
        
    # Upload back to s3
    msg = f'Send {file} > s3://grizli-v2/{s3_prefix}'
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    bkt.upload_file(local_file, s3_prefix, ExtraArgs={'ACL': 'public-read'},
                    Callback=None, Config=None)
    
    # Update status
    db.execute(f"""update exposure_snowblind
    set status = 2, maskfrac = {maskfrac}, npix = {npix}, utime = {time.time()}
    where assoc = '{assoc}' and file = '{file}'
    """)
    
    if clean:
        os.remove(local_file)
        
    return True


def snowblind_batch():
    import os
    os.chdir('/GrizliImaging')
    
    db.execute("""update exposure_snowblind set status = 70 
    where (file not like 'jw01895%%' and assoc not like 'j03%%m27%%') and status = 0
    """)
    
    db.execute("""update exposure_snowblind set status = 0 
    where (file like 'jw01895%%' and (assoc like 'j03%%m27%%' or assoc like 'j12%%p62%%')) and status = 70
    """)
    
    files = db.SQL("""select * from exposure_snowblind
    where (file like '%%_rate.fits') and status = 0
    order by random()
    """)

    files = db.SQL("""select * from exposure_snowblind
    where (file like '%%_rate.fits') and status = 0 and assoc like '%%f444w%%'
    order by random()
    """)
    
    print(f'Run {len(files)} files ....\n')
    for row in files:
        snowblind_exposure_mask(row['assoc'], row['file'], verbose=True, clean=True, 
                                new_jump_flag=1024,
                                min_radius=4,
                                growth_factor=1.5, unset_first=True, run_again=False)
        # break


if __name__ == '__main__':
    run_all()