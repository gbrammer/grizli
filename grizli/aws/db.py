"""
Interact with the grizli AWS database
"""
import os
import glob
import numpy as np

import astropy.time
import astropy.units as u

try:
    import pandas as pd
except:
    pd = None

from .. import utils
    
FLAGS = {'init_lambda': 1,
         'start_beams': 2,
         'done_beams': 3,
         'no_run_fit': 4,
         'start_redshift_fit': 5,
         'fit_complete': 6}

COLUMNS = ['root', 'id', 'status', 'ra', 'dec', 'ninput', 'redshift', 'as_epsf', 't_g102', 'n_g102', 'p_g102', 't_g141', 'n_g141', 'p_g141', 't_g800l', 'n_g800l', 'p_g800l', 'numlines', 'haslines', 'chi2poly', 'chi2spl', 'splf01', 'sple01', 'splf02', 'sple02', 'splf03', 'sple03', 'splf04', 'sple04', 'huberdel', 'st_df', 'st_loc', 'st_scl', 'dof', 'chimin', 'chimax', 'bic_poly', 'bic_spl', 'bic_temp', 'z02', 'z16', 'z50', 'z84', 'z97', 'zwidth1', 'zwidth2', 'z_map', 'zrmin', 'zrmax', 'z_risk', 'min_risk', 'd4000', 'd4000_e', 'dn4000', 'dn4000_e', 'dlineid', 'dlinesn', 'flux_pab', 'err_pab', 'ew50_pab', 'ewhw_pab', 'flux_hei_1083', 'err_hei_1083', 'ew50_hei_1083', 'ewhw_hei_1083', 'flux_siii', 'err_siii', 'ew50_siii', 'ewhw_siii', 'flux_oii_7325', 'err_oii_7325', 'ew50_oii_7325', 'ewhw_oii_7325', 'flux_ariii_7138', 'err_ariii_7138', 'ew50_ariii_7138', 'ewhw_ariii_7138', 'flux_sii', 'err_sii', 'ew50_sii', 'ewhw_sii', 'flux_ha', 'err_ha', 'ew50_ha', 'ewhw_ha', 'flux_oi_6302', 'err_oi_6302', 'ew50_oi_6302', 'ewhw_oi_6302', 'flux_hei_5877', 'err_hei_5877', 'ew50_hei_5877', 'ewhw_hei_5877', 'flux_oiii', 'err_oiii', 'ew50_oiii', 'ewhw_oiii', 'flux_hb', 'err_hb', 'ew50_hb', 'ewhw_hb', 'flux_oiii_4363', 'err_oiii_4363', 'ew50_oiii_4363', 'ewhw_oiii_4363', 'flux_hg', 'err_hg', 'ew50_hg', 'ewhw_hg', 'flux_hd', 'err_hd', 'ew50_hd', 'ewhw_hd', 'flux_h7', 'err_h7', 'ew50_h7', 'ewhw_h7', 'flux_h8', 'err_h8', 'ew50_h8', 'ewhw_h8', 'flux_h9', 'err_h9', 'ew50_h9', 'ewhw_h9', 'flux_h10', 'err_h10', 'ew50_h10', 'ewhw_h10', 'flux_neiii_3867', 'err_neiii_3867', 'ew50_neiii_3867', 'ewhw_neiii_3867', 'flux_oii', 'err_oii', 'ew50_oii', 'ewhw_oii', 'flux_nevi_3426', 'err_nevi_3426', 'ew50_nevi_3426', 'ewhw_nevi_3426', 'flux_nev_3346', 'err_nev_3346', 'ew50_nev_3346', 'ewhw_nev_3346', 'flux_mgii', 'err_mgii', 'ew50_mgii', 'ewhw_mgii', 'flux_civ_1549', 'err_civ_1549', 'ew50_civ_1549', 'ewhw_civ_1549', 'flux_ciii_1908', 'err_ciii_1908', 'ew50_ciii_1908', 'ewhw_ciii_1908', 'flux_oiii_1663', 'err_oiii_1663', 'ew50_oiii_1663', 'ewhw_oiii_1663', 'flux_heii_1640', 'err_heii_1640', 'ew50_heii_1640', 'ewhw_heii_1640', 'flux_niii_1750', 'err_niii_1750', 'ew50_niii_1750', 'ewhw_niii_1750', 'flux_niv_1487', 'err_niv_1487', 'ew50_niv_1487', 'ewhw_niv_1487', 'flux_nv_1240', 'err_nv_1240', 'ew50_nv_1240', 'ewhw_nv_1240', 'flux_lya', 'err_lya', 'ew50_lya', 'ewhw_lya', 'pdf_max', 'cdf_z', 'sn_pab', 'sn_hei_1083', 'sn_siii', 'sn_oii_7325', 'sn_ariii_7138', 'sn_sii', 'sn_ha', 'sn_oi_6302', 'sn_hei_5877', 'sn_oiii', 'sn_hb', 'sn_oiii_4363', 'sn_hg', 'sn_hd', 'sn_h7', 'sn_h8', 'sn_h9', 'sn_h10', 'sn_neiii_3867', 'sn_oii', 'sn_nevi_3426', 'sn_nev_3346', 'sn_mgii', 'sn_civ_1549', 'sn_ciii_1908', 'sn_oiii_1663', 'sn_heii_1640', 'sn_niii_1750', 'sn_niv_1487', 'sn_nv_1240', 'sn_lya', 'chinu', 'bic_diff', 'log_risk', 'log_pdf_max', 'zq', 'mtime', 'vel_bl', 'vel_nl', 'vel_z', 'vel_nfev', 'vel_flag', 'grizli_version']

# New columns in redshift_fit_v2
COLUMNS += ['flux_bra', 'err_bra', 'ew50_bra', 'ewhw_bra',
            'flux_brb', 'err_brb', 'ew50_brb', 'ewhw_brb',
            'flux_brg', 'err_brg', 'ew50_brg', 'ewhw_brg',
            'flux_pfg', 'err_pfg', 'ew50_pfg', 'ewhw_pfg',
            'flux_pfd', 'err_pfd', 'ew50_pfd', 'ewhw_pfd',
            'flux_paa', 'err_paa', 'ew50_paa', 'ewhw_paa',
            'flux_pag', 'err_pag', 'ew50_pag', 'ewhw_pag',
            'flux_pad', 'err_pad', 'ew50_pad', 'ewhw_pad',
            'flux_nii', 'err_nii', 'ew50_nii', 'ewhw_nii',
            'flux_oiii_4959', 'err_oiii_4959',
            'ew50_oiii_4959', 'ewhw_oiii_4959',
            'flux_oiii_5007', 'err_oiii_5007',
            'ew50_oiii_5007', 'ewhw_oiii_5007',
            'flux_ciii_1906', 'err_ciii_1906',
            'ew50_ciii_1906', 'ewhw_ciii_1906',
            'sn_bra', 'sn_brb', 'sn_brg', 'sn_pfg', 'sn_pfd',
            'sn_paa', 'sn_pag', 'sn_pad', 'sn_nii',
            'sn_oiii_4959', 'sn_oiii_5007',
            'sn_ciii_1906']

engine = None

def get_connection_info(config_file=None):
    """
    Read the database connection info
    """
    import yaml
    
    if 'DB_USER' in os.environ:
        db_info = {'username': os.environ['DB_USER'], 
                   'password': os.environ['DB_PASS'], 
                   'hostname': os.environ['DB_HOST'], 
                   'database': os.environ['DB_NAME'], 
                   'port': 5432}
        
        if 'DB_PORT' in os.environ:
            db_info['port'] = os.environ['DB_PORT']
        
        return db_info
        
    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__),
                                   '../data/db.yml')

        try:
            local_file = os.path.join(os.getenv('HOME'), 'db.local.yml')

            if os.path.exists(local_file):
                # print('Use ~/db.local.yml')
                config_file = local_file
        except:
            pass

    fp = open(config_file)
    try:
        db_info = yaml.load(fp, Loader=yaml.FullLoader)
    except:
        db_info = yaml.load(fp)

    fp.close()

    return db_info


# DB connection engine
_ENGINE = None

def get_db_engine(config=None, echo=False, iam_file='/home/ec2-user/db.iam.yaml'):
    """
    Generate an SQLAlchemy engine for the grizli database
    """
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    
    global _ENGINE
    
    import psycopg2
    import boto3
    
    # With IAM auth on EC2
    if os.path.exists(iam_file) & (config is None):
        config = get_connection_info(config_file=iam_file)
        session = boto3.Session()
        client = session.client('rds', region_name=config['region'])
        
        token = client.generate_db_auth_token(DBHostname=config['hostname'],
                                              Port=config['port'], 
                                              DBUsername=config['username'], 
                                              Region=config['region'])
        
        connect_args = dict(host=config['hostname'],
                            port=config['port'], 
                            database=config['database'],
                            user=config['username'],
                            password=token,
                            sslrootcert="SSLCERTIFICATE")
        
        args = ('postgresql+psycopg2://',)
        kws = dict(connect_args=connect_args, poolclass=NullPool)
        
        engine = create_engine(*args, **kws)
        
        engine._init_time = astropy.time.Time.now()
        engine._init_args = args
        engine._init_kws = kws
        
        _ENGINE = engine
        return engine
     
    if config is None:
        config = get_connection_info()
    
    db_string = "postgresql://{0}:{1}@{2}:{3}/{4}"
    db_string = db_string.format(config['username'], config['password'], 
                                 config['hostname'], config['port'], 
                                 config['database'])
    
    args = (db_string,)
    kws = dict(echo=echo, poolclass=NullPool)
                               
    engine = create_engine(*args, **kws)
    engine._init_time = astropy.time.Time.now()
    engine._init_args = args
    engine._init_kws = kws
    
    _ENGINE = engine
    return engine


ENGINE_REFRESH_DT = 10*u.minute

def refresh_engine():
    """
    Refresh the DB engine
    """
    from sqlalchemy import create_engine
    
    global _ENGINE
            
    now = astropy.time.Time.now()
    if _ENGINE is None:
        get_db_engine()
    elif 'connect_args' in _ENGINE._init_kws:
        get_db_engine()
    elif (now - _ENGINE._init_time) > ENGINE_REFRESH_DT:
        args, kws = _ENGINE._init_args, _ENGINE._init_kws
        _ENGINE = create_engine(*args, **kws)
        _ENGINE._init_time = astropy.time.Time.now()
        _ENGINE._init_args = args
        _ENGINE._init_kws = kws


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    import logging
    import boto3
    from botocore.exceptions import ClientError
    import os

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_s3_file(path='s3://grizli-v2/HST/Pipeline/Tiles/2529/tile.2529.010.492.f160w_drz_sci.fits', output_dir='./', ExtraArgs={"RequestPayer": "requester"}, overwrite=True, verbose=True):
    """
    """
    import boto3
    from botocore.exceptions import ClientError
    
    s3 = boto3.resource('s3')
    
    path_split = path.split('s3://')[1].split('/')
    file_bucket = path_split[0]
    file_bkt = s3.Bucket(file_bucket)
    file_prefix = '/'.join(path_split[1:])
    
    local_file = os.path.join(output_dir, os.path.basename(file_prefix))
    if os.path.exists(local_file) & (not overwrite):
        if verbose:
            print(f'{local_file} exists')
        
        return local_file
    
    try:
        if verbose:
            print(f'{path} > {local_file}')
            
        file_bkt.download_file(file_prefix, local_file,
                      ExtraArgs=ExtraArgs)
        return local_file
    except ClientError:
        print(f'{path} not found')
        return None
        
    # files = [obj.key for obj in
    #          file_bkt.objects.filter(Prefix=file_prefix)]
    # 
    # if len(files) > 0:                
    #     if verbose:
    #         print(f'{path} > {local_file}')
    #         
    #     file_bkt.download_file(file_prefix, local_file,
    #                   ExtraArgs=ExtraArgs)
    #     return local_file
    # else:
    #     print(f'{path} not found')
    #     return None


def get_redshift_fit_status(root, id, table='redshift_fit_v2', engine=None):
    """
    Get status value from the database for root_id object
    """
    import pandas as pd

    if engine is None:
        engine = get_db_engine(echo=False)

    res = pd.read_sql_query("SELECT status FROM {2} WHERE (root = '{0}' AND id = {1})".format(root, id, table), engine)

    if len(res) == 0:
        return -1
    else:
        return res['status'][0]


def update_jname():

    from grizli import utils
    from grizli.aws import db as grizli_db
    
    res = grizli_db.from_sql("select p_root, p_id, p_ra, p_dec from photometry_apcorr", engine)

    jn = [utils.radec_to_targname(ra=ra, dec=dec, round_arcsec=(0.001, 0.001), precision=2, targstr='j{rah}{ram}{ras}.{rass}{sign}{ded}{dem}{des}.{dess}') for ra, dec in zip(res['p_ra'], res['p_dec'])]

    for c in res.colnames:
        res.rename_column(c, c.replace('p_', 'j_'))

    zres = grizli_db.from_sql("select root, phot_root, id, ra, dec, z_map,"
                              "q_z, t_g800l, t_g102, t_g141, status from "
                              "redshift_fit_v2 where ra is not null and "
                              "status > 5", engine)

    # Find duplicates
    from scipy.spatial import cKDTree
    data = np.array([zres['ra'], zres['dec']]).T

    ok = zres['q_z'].filled(-100) > -0.7

    tree = cKDTree(data[ok])
    dr, ix = tree.query(data[ok], k=2)
    cosd = np.cos(data[:, 1]/180*np.pi)
    dup = (dr[:, 1] < 0.01/3600)  # & (zres['phot_root'][ix[:,0]] != zres['phot_root'][ix[:,1]])

    ix0 = ix[:, 0]
    ix1 = ix[:, 1]

    dup = (dr[:, 1] < 0.01/3600)
    dup &= (zres['phot_root'][ok][ix0] == zres['phot_root'][ok][ix1])
    dup &= (zres['id'][ok][ix0] == zres['id'][ok][ix1])

    # second is G800L
    dup &= zres['t_g800l'].filled(0)[ok][ix1] > 10

    # plt.scatter(zres['z_map'][ok][ix0[dup]], zres['z_map'][ok][ix1[dup]],
    #            marker='.', alpha=0.1)


def update_redshift_fit_status(root, id, status=0, table='redshift_fit_v2', engine=None, verbose=True):
    """
    Set the status flag in the table
    """
    import time

    import pandas as pd
    from astropy.table import Table
    from astropy.time import Time
    NOW = Time.now().iso

    if engine is None:
        engine = get_db_engine(echo=False)

    old_status = get_redshift_fit_status(root, id, table=table, engine=engine)
    if old_status < 0:
        # Need to add an empty row
        tab = Table()
        tab['root'] = [root]
        tab['id'] = [id]
        tab['status'] = [status]
        tab['mtime'] = [NOW]

        row_df = tab.to_pandas()

        add_redshift_fit_row(row_df, engine=engine, table=table,
                             verbose=verbose)

    else:
        sqlstr = """UPDATE {0}
            SET status = {1}, mtime = '{2}'
            WHERE (root = '{3}' AND id = {4});
            """.format(table, status, NOW, root, id)

        if verbose:
            msg = 'Update status for {0} {1}: {2} -> {3} on `{4}` ({5})'
            print(msg.format(root, id, old_status, status, table, NOW))

        if hasattr(engine, 'cursor'):
            with engine.cursor() as cur:
                cur.execute(sqlstr)
        else:
            engine.execute(sqlstr)


def execute_helper(sqlstr, engine):
    """
    Different behaviour for psycopg2.connection and sqlalchemy.engine
    """
    if hasattr(engine, 'cursor'):
        with engine.cursor() as cur:
            cur.execute(sqlstr)
    else:
        engine.execute(sqlstr)


def get_row_data(rowfile='gds-g800l-j033236m2748_21181.row.fits', status_flag=FLAGS['fit_complete']):
    """
    Convert table from a row file to a pandas DataFrame
    """
    import pandas as pd
    from astropy.table import Table
    from astropy.time import Time
    NOW = Time.now().iso

    if isinstance(rowfile, str):
        if rowfile.endswith('.fits'):
            tab = Table.read(rowfile, character_as_bytes=False)
            allowed_columns = COLUMNS
        else:
            # Output of stellar fits
            tab = Table.read(rowfile, format='ascii.commented_header')

            tab['chinu'] = tab['chi2']/tab['dof']
            tab['phot_root'] = tab['root']
            tab.rename_column('best_template', 'stellar_template')

            try:
                tab['chinu'] = tab['chi2']/tab['dof']
                tab['phot_root'] = tab['root']

                # BIC of spline-only and template fits
                bic_spl = np.log(tab['dof'])*(tab['nk']-1) + tab['chi2_flat']
                bic_star = np.log(tab['dof'])*(tab['nk']) + tab['chi2']
                tab['bic_diff_star'] = bic_spl - bic_star

            except:
                print('Parse {0} failed'.format(rowfile))
                pass

            allowed_columns = ['root', 'id', 'ra', 'dec', 'chi2', 'nk', 'dof',
                               'chinu', 'chi2_flat', 'bic_diff_star', 'mtime',
                               'stellar_template', 'status', 'phot_root',
                               'as_epsf']
    else:
        tab = rowfile

    if 'cdf_z' in tab.colnames:
        cdf_z = tab['cdf_z'].data
        tab.remove_column('cdf_z')
    else:
        cdf_z = None

    tab['mtime'] = NOW
    tab['status'] = status_flag
    remove_cols = []
    for c in tab.colnames:
        if '-' in c:
            tab.rename_column(c, c.replace('-', '_'))

    for c in tab.colnames:
        tab.rename_column(c, c.lower())

    # Remove columns not in the database
    remove_cols = []
    for c in tab.colnames:
        if c not in allowed_columns:
            #print('Remove column: ', c)
            remove_cols.append(c)

    if len(remove_cols) > 0:
        tab.remove_columns(remove_cols)

    row_df = tab.to_pandas()
    if cdf_z is not None:
        row_df['cdf_z'] = cdf_z.tolist()

    return row_df


def delete_redshift_fit_row(root, id, table='redshift_fit_v2', engine=None):
    """
    Delete a row from the redshift fit table
    """
    if engine is None:
        engine = get_db_engine(echo=False)

    res = engine.execute("DELETE from {2} WHERE (root = '{0}' AND id = {1})".format(root, id, table))


def add_redshift_fit_row(row_df, table='redshift_fit_v2', engine=None, verbose=True):
    """
    Update the row in the redshift_fit table
    """
    if engine is None:
        engine = get_db_engine(echo=False)

    if isinstance(row_df, str):
        row_df = get_row_data(row_df)

    if ('root' not in row_df.columns) | ('id' not in row_df.columns):
        print('Need at least "root" and "id" columns in the row data')
        return False

    root = row_df['root'][0]
    id = row_df['id'][0]
    status = get_redshift_fit_status(root, id, table=table, engine=engine)
    # Delete the old row?
    if status >= 0:
        print('Delete and update row for {0}/{1} on `{2}`'.format(root, id,
                                                                  table))
        delete_redshift_fit_row(root, id, table=table, engine=engine)
    else:
        print('Add row for {0}/{1} on `{2}`'.format(root, id, table))

    # Add the new data
    row_df.to_sql(table, engine, index=False, if_exists='append', method='multi')

###########


def add_missing_rows(root='j004404m2034', engine=None):
    """
    Add rows that were completed but that aren't in the table
    """
    import glob
    from astropy.table import vstack, Table
    import pandas as pd
    
    from grizli.aws import db as grizli_db

    if engine is None:
        engine = grizli_db.get_db_engine(echo=False)

    os.system('aws s3 sync s3://grizli-v2/HST/Pipeline/{0}/Extractions/ ./ --exclude "*" --include "*row.fits"'.format(root))

    row_files = glob.glob('{0}*row.fits'.format(root))
    row_files.sort()

    res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit_v2 WHERE root = '{0}' AND status=6".format(root), engine)

    res_ids = res['id'].to_list()
    tabs = []

    print('\n\n NROWS={0}, NRES={1}\n\n'.format(len(row_files), len(res)))

    for row_file in row_files:
        id_i = int(row_file.split('.row.fits')[0][-5:])
        if id_i not in res_ids:
            grizli_db.add_redshift_fit_row(row_file, engine=engine, verbose=True)


def convert_1D_to_lists(file='j234420m4245_00615.1D.fits'):
    """
    Convert 1D spectral data to lists suitable for putting into dataframes
    and sending to the databases.
    """
    from collections import OrderedDict
    import astropy.io.fits as pyfits

    if not os.path.exists(file):
        print('Spectrum file not found')
        return False

    im = pyfits.open(file)
    obj_id = im[0].header['ID']
    obj_root = im[0].header['TARGET']

    if '.R30.' in file:
        skip_columns = ['line', 'cont']
        pref = 'spec1d_r30'
    else:
        skip_columns = []
        pref = 'spec1d'

    spectra = OrderedDict()
    has_spectra = False
    for gr in ['G102', 'G141', 'G800L']:
        if gr in im:
            has_spectra = True
            sp = utils.GTable.read(file, hdu=gr)
            prefix = '{0}_{1}_'.format(pref, gr.lower())

            spd = {prefix+'id': obj_id, prefix+'root': obj_root}
            for c in sp.colnames:
                if c in skip_columns:
                    continue

                spd[prefix+c] = sp[c].tolist()

            spectra[gr.lower()] = spd

    if has_spectra:
        return spectra
    else:
        return False


def make_v2_tables():
    
    from grizli.aws import db
    
    db.execute("""CREATE TABLE redshift_fit_v2 AS TABLE redshift_fit WITH NO DATA;""")
    
    db.execute("""CREATE TABLE redshift_fit_quasar_v2 AS TABLE redshift_fit_quasar WITH NO DATA;""")
    
    db.execute("""CREATE TABLE stellar_fit_v2 AS TABLE stellar_fit_v2 WITH NO DATA;""")
    db.execute("""CREATE TABLE stellar_fit_v2 AS TABLE stellar_fit WITH NO DATA;""")
    db.execute("""CREATE TABLE multibeam_v2 AS TABLE multibeam WITH NO DATA;""")
    db.execute("""CREATE TABLE beam_geometry_v2 AS TABLE beam_geometry WITH NO DATA;""")

    for t in ['spec1d_r30_g141', 'spec1d_r30_g102', 'spec1d_g141', 'spec1d_g102']:
        db.execute(f"""CREATE TABLE {t}_v2_wave AS TABLE {t}_wave;""")
    

def send_1D_to_database(files=[], engine=None):
    """
    Send a list of 1D spectra to the spectra databases

    ToDo: check for existing lines

    """
    from collections import OrderedDict
    import pandas as pd

    if engine is None:
        engine = get_db_engine()

    tables = OrderedDict()
    for file in files:
        sp_i = convert_1D_to_lists(file=file)
        print('Read spec1d file: {0}'.format(file))
        for gr in sp_i:

            # Initialize the columns
            if gr not in tables:
                tables[gr] = OrderedDict()
                for c in sp_i[gr]:
                    tables[gr][c] = []

            # Add the data
            for c in sp_i[gr]:
                tables[gr][c].append(sp_i[gr][c])

    prefix = 'spec1d_r30' if '.R30.' in files[0] else 'spec1d'

    for gr in tables:
        tablepref = '{0}_{1}'.format(prefix, gr)
        tablename = '{0}_v2'.format(tablepref)
        df = pd.DataFrame(tables[gr])

        # Put wavelengths in their own tables to avoid massive duplication
        wave_table = tablename+'_wave'
        if wave_table not in engine.table_names():
            print('Create wave table: '+wave_table)
            wdf = pd.DataFrame(data=tables[gr][wave_table][0],
                               columns=[wave_table])

            wdf.to_sql(wave_table, engine, if_exists='replace',
                       index=True, index_label=tablename+'_idx')

        # drop wave from spectra tables
        df.drop('{0}_{1}_wave'.format(prefix, gr), axis=1, inplace=True)

        # Create table
        if tablename not in engine.table_names():
            print('Initialize table {0}'.format(tablename))

            SQL = "CREATE TABLE {0} (\n".format(tablename)
            SQL += '    {0}_root   text,\n'.format(tablename)
            SQL += '    {0}_id  integer,\n'.format(tablename)
            for c in df.columns:
                item = df[c][0]
                if isinstance(item, list):
                    SQL += '    {0} real[{1}],\n'.format(c, len(item))

            engine.execute(SQL[:-2]+')')

            try:
                engine.execute("CREATE INDEX {0}_idx ON {0} ({0}_root, {0}_id);".format(tablename))
            except:
                pass

        # Delete existing duplicates
        if tablename in engine.table_names():
            SQL = """DELETE from {0} WHERE """.format(tablename)
            cmd = "({0}_root = '{1}' AND {0}_id = {2})"
            mat = [cmd.format(tablepref, r, i) 
                   for r, i in zip(df[tablepref+'_root'], 
                                   df[tablepref+'_id'])]
            SQL += 'OR '.join(mat)
            rsp = engine.execute(SQL)

        # Send the table
        print('Send {0} rows to {1}'.format(len(df), tablename))
        df.to_sql(tablename, engine, index=False, if_exists='append',
                  method='multi')


def add_all_spectra():

    from grizli.aws import db as grizli_db

    roots = grizli_db.from_sql("select root,count(root) as n from redshift_fit_v2 group BY root order by n DESC", engine)

    o = 1

    for root in roots['root'][::o]:
        existing = open('log').readlines()
        if root+'\n' in existing:
            print('Skip', root)
            continue

        fp = open('log', 'a')
        fp.write(root+'\n')
        fp.close()
        try:
            grizli_db.add_oned_spectra(root=root, engine=engine)
        except:
            pass


def add_oned_spectra(root='j214224m4420gr01', bucket='grizli-v2', engine=None):
    import os
    import glob
    from collections import OrderedDict
    
    if engine is None:
        engine = get_db_engine()

    # import boto3
    # s3 = boto3.resource('s3')
    # bkt = s3.Bucket(bucket)
    #
    # files = [obj.key for obj in bkt.objects.filter(Prefix='Pipeline/{0}/Extractions/'.format(root))]
    #
    # for file in files:
    #     if (('.R30.fits' in file) | ('.1D.fits' in file)) & (not os.path.exists(file)):
    #         local_file = os.path.basename(file)
    #         print(local_file)
    #         bkt.download_file(file, local_file,
    #                           ExtraArgs={"RequestPayer": "requester"})
    os.system('aws s3 sync s3://{0}/HST/Pipeline/{1}/Extractions/ ./ --exclude "*" --include "*R30.fits" --include "*1D.fits"'.format(bucket, root))

    nmax = 500
    # 1D.fits
    files = glob.glob('{0}_*1D.fits'.format(root))
    files.sort()
    for i in range(len(files)//nmax+1):
        send_1D_to_database(files=files[i*nmax:(i+1)*nmax], engine=engine)

    files = glob.glob('{0}_*R30.fits'.format(root))
    files.sort()
    for i in range(len(files)//nmax+1):
        send_1D_to_database(files=files[i*nmax:(i+1)*nmax], engine=engine)

    os.system('rm {0}_*.1D.fits {0}_*.R30.fits'.format(root))

    if False:
        import scipy.ndimage as nd
        import matplotlib.pyplot as plt
        
        tablename = 'spec1d_g141'
        #tablename = 'spec1d_g102'

        #tablename = 'spec1d_r30_g141'

        if 1:
            # by root
            resp = pd.read_sql_query("SELECT root, id, z_map, q_z, sp.* from redshift_fit_v2, {1} as sp WHERE {1}_root = root AND {1}_id = id AND root = '{0}' AND q_z > -0.7 ORDER BY z_map".format(root, tablename), engine)
        else:
            # everything
            resp = pd.read_sql_query("SELECT root, id, z_map, q_z, sp.* from redshift_fit_v2, {1} as sp WHERE {1}_root = root AND {1}_id = id AND q_z > -0.7 ORDER BY z_map".format(root, tablename), engine)

            # Halpha EW
            resp = pd.read_sql_query("SELECT root, id, z_map, q_z, ew50_ha, flux_ha, err_ha, t_g141, sp.* from redshift_fit_v2, {1} as sp WHERE {1}_root = root AND {1}_id = id AND q_z > -0.3 AND err_ha > 0 ORDER BY ew50_ha".format(root, tablename), engine)

            # Everything
            fresp = pd.read_sql_query("SELECT root, id, z_map, q_z, ew50_ha, flux_ha, err_ha, ew50_oiii, ew50_hb, ew50_oii, d4000, d4000_e, t_g141, t_g102, t_g800l, sp.* from redshift_fit_v2, {1} as sp WHERE {1}_root = root AND {1}_id = id AND q_z > -0.7 AND chinu < 2 ORDER BY z_map".format(root, tablename), engine)

        wave = pd.read_sql_query("SELECT * from {0}_wave".format(tablename),
                                 engine)[tablename+'_wave'].values

        resp = fresp

        sort_column = 'z_map'
        bin_factor = 1

        wnorm = 6400
        zref = 1.3e4/wnorm-1

        sel = np.isfinite(fresp[sort_column]) & (fresp[sort_column] != -99)

        norm_ix = np.interp(wnorm*(1+fresp['z_map']), wave, np.arange(len(wave)), left=np.nan, right=np.nan)
        sel &= np.isfinite(norm_ix)

        resp = fresp[sel]

        norm_ix = np.cast[int](np.round(np.interp(wnorm*(1+resp['z_map']), wave, np.arange(len(wave)), left=np.nan, right=np.nan)))

        resp.sort_values(sort_column, inplace=True)

        if tablename == 'spec1d_g141':
            exptime = resp['t_g141'].values
            wlim = [1.1e4, 1.65e4]
        else:
            exptime = resp['t_g102'].values
            wlim = [8000, 1.1e4, 1.65e4]

        data = OrderedDict()
        for c in resp.columns:
            if c.startswith(tablename):
                c_i = c.split(tablename+'_')[1]
                try:
                    data[c_i] = np.array(resp[c].values.tolist())
                except:
                    pass

        #plt.imshow((data['flux'] - data['cont'])/data['flat']/1.e-19, vmin=-0.1, vmax=10)

        # Rest-frame
        dz = np.diff(wave)[10]/wave[10]
        max_zshift = np.cast[int](np.log(1+resp['z_map'].max())/dz)
        zshift = np.cast[int]((np.log(1+resp['z_map']) - np.log(1+zref))/dz)

        err_max = 5

        # Continuum normalized
        #norm = data['cont'][:,100]/data['flat'][:,100]
        norm = np.zeros(len(resp))
        for i, ix in enumerate(norm_ix):
            norm[i] = data['line'][i, ix]/data['flat'][i, ix]

        #norm = np.mean(data['cont'][:,50:120]/data['flat'][:,50:120], axis=1)

        # 2D arrays
        normed = ((data['flux']/data['flat']).T/norm).T
        cnormed = ((data['cont']/data['flat']).T/norm).T
        lnormed = (((data['line']-data['cont'])/data['flat']).T/norm).T

        err = ((data['err']/data['flat']).T/norm).T

        mask = np.isfinite(norm) & (norm > 0) & np.isfinite(norm_ix)
        normed = normed[mask, :]
        cnormed = cnormed[mask, :]
        lnormed = lnormed[mask, :]
        err = err[mask, :]
        ivar = 1/err**2
        ivar[err <= 0] = 0

        # Weight by exposure time
        ivar = (ivar.T*0+(exptime[mask]/4000.)*norm[mask]).T

        zshift = zshift[mask]

        # Clip edges
        wclip = (wave > wlim[0]) & (wave < wlim[1])

        mask_val = 1e10

        normed[:, ~wclip] = -mask_val
        cnormed[:, ~wclip] = -mask_val
        lnormed[:, ~wclip] = -mask_val
        sh = normed.shape
        rest = np.zeros((sh[0], sh[1]+zshift.max()-zshift.min())) - mask_val
        crest = np.zeros((sh[0], sh[1]+zshift.max()-zshift.min())) - mask_val
        lrest = np.zeros((sh[0], sh[1]+zshift.max()-zshift.min())) - mask_val

        rest[:, zshift.max():zshift.max()+sh[1]] = normed*1
        crest[:, zshift.max():zshift.max()+sh[1]] = cnormed*1
        lrest[:, zshift.max():zshift.max()+sh[1]] = lnormed*1

        rest_ivar = np.zeros((sh[0], sh[1]+zshift.max()-zshift.min()))
        rest_ivar[:, zshift.max():zshift.max()+sh[1]] = ivar*1

        for i in range(sh[0]):
            rest[i, :] = np.roll(rest[i, :], -zshift[i])
            crest[i, :] = np.roll(crest[i, :], -zshift[i])
            lrest[i, :] = np.roll(lrest[i, :], -zshift[i])
            rest_ivar[i, :] = np.roll(rest_ivar[i, :], -zshift[i])

        ok = np.isfinite(rest) & np.isfinite(rest_ivar) & (rest > -0.8*mask_val)
        rest_ivar[~ok] = 0
        rest[~ok] = -mask_val
        crest[~ok] = -mask_val
        lrest[~ok] = -mask_val

        shr = rest.shape
        nbin = int((shr[0]//shr[1])//2*bin_factor)*2
        kernel = np.ones((1, nbin)).T
        # npix = np.maximum(nd.convolve((rest > -0.8*mask_val)*1, kernel), 1)
        # srest = nd.convolve(rest*(rest > -0.8*mask_val), kernel)
        # sbin = (srest/npix)[::nbin,:]
        # plt.imshow(sbin, vmin=0, vmax=5)

        num = nd.convolve(rest*rest_ivar, kernel)
        cnum = nd.convolve(crest*rest_ivar, kernel)
        lnum = nd.convolve(lrest*rest_ivar, kernel)
        den = nd.convolve(rest_ivar, kernel)
        wbin = (num/den)[::nbin, :]
        wbin[~np.isfinite(wbin)] = 0

        cwbin = (cnum/den)[::nbin, :]
        cwbin[~np.isfinite(cwbin)] = 0

        lwbin = (lnum/den)[::nbin, :]
        lwbin[~np.isfinite(lwbin)] = 0

        plt.imshow(wbin, vmin=0, vmax=5)

        plt.imshow((data['line'] - data['cont'])/data['flat']/1.e-19, vmin=-0.1, vmax=10)


def run_lambda_fits(root='j004404m2034', phot_root=None, mag_limits=[15, 26], sn_limit=7, min_status=None, engine=None, zr=[0.01, 3.4], bucket='grizli-v2', verbose=True, extra={'bad_pa_threshold': 10}, ids=None):
    """
    Run redshift fits on lambda for a given root
    """
    from grizli.aws import fit_redshift_lambda
    from grizli import utils
    from grizli.aws import db as grizli_db
    if engine is None:
        engine = grizli_db.get_db_engine()

    import pandas as pd
    import numpy as np
    import glob
    import os

    print('Sync phot catalog')

    if phot_root is None:
        root = root

    os.system('aws s3 sync s3://{1}/Pipeline/{0}/Extractions/ ./ --exclude "*" --include "*_phot*.fits"'.format(phot_root, bucket))

    print('Sync wcs.fits')

    os.system('aws s3 sync s3://{1}/Pipeline/{0}/Extractions/ ./ --exclude "*" --include "*_phot*.fits" --include "*wcs.fits"'.format(root, bucket))

    phot = utils.read_catalog('{0}_phot_apcorr.fits'.format(phot_root))
    phot['has_grism'] = 0
    wcs_files = glob.glob('*wcs.fits')
    for f in wcs_files:
        w = utils.WCSFootprint(f, ext=0)
        has = w.path.contains_points(np.array([phot['ra'], phot['dec']]).T)
        print(f, has.sum())
        phot['has_grism'] += has

    mag = phot['mag_auto']*np.nan
    mag_filt = np.array(['     ']*len(phot))
    sn = phot['mag_auto']*np.nan

    for filt in ['f160w', 'f140w', 'f125w', 'f105w', 'f110w', 'f098m', 'f814w', 'f850lp', 'f606w', 'f775w']:
        if '{0}_tot_1'.format(filt) in phot.colnames:
            mag_i = 23.9-2.5*np.log10(phot['{0}_tot_1'.format(filt)])
            fill = (~np.isfinite(mag)) & np.isfinite(mag_i)
            mag[fill] = mag_i[fill]
            mag_filt[fill] = filt
            sn_i = phot['{0}_tot_1'.format(filt)]/phot['{0}_etot_1'.format(filt)]
            sn[fill] = sn_i[fill]

    sel = np.isfinite(mag) & (mag >= mag_limits[0]) & (mag <= mag_limits[1]) & (phot['has_grism'] > 0)
    sel &= phot['flux_radius'] > 1
    sel &= sn > sn_limit

    if min_status is not None:
        res = pd.read_sql_query("SELECT root, id, status, mtime FROM redshift_fit_v2 WHERE root = '{0}'".format(root, min_status), engine)
        if len(res) > 0:
            status = phot['id']*0-100
            status[res['id']-1] = res['status']
            sel &= status < min_status

    if ids is None:
        ids = phot['id'][sel]

    # Select just on min_status
    if min_status > 1000:
        if min_status > 10000:
            # Include mag constraints
            res = pd.read_sql_query("SELECT root, id, status, mtime, mag_auto FROM redshift_fit_v2,photometry_apcorr WHERE root = '{0}' AND status = {1}/10000 AND mag_auto > {2} AND mag_auto < {3} AND p_root = root AND p_id = id".format(root, min_status, mag_limits[0], mag_limits[1]), engine)
        else:
            # just select on status
            res = pd.read_sql_query("SELECT root, id, status, mtime FROM redshift_fit_v2 WHERE root = '{0}' AND status = {1}/1000".format(root, min_status, mag_limits[0], mag_limits[1]), engine)

        ids = res['id'].tolist()

    if len(ids) == 0:
        return False

    fit_redshift_lambda.fit_lambda(root=root, beams=[], ids=ids, newfunc=False, bucket_name=bucket, skip_existing=False, sleep=False, skip_started=False, show_event=False, zr=zr, force_args=True, quasar_fit=False, output_path=None, save_figures='png', verbose=verbose, **extra)

    print('Add photometry: {0}'.format(root))
    grizli_db.add_phot_to_db(phot_root, delete=False, engine=engine)

    res = grizli_db.wait_on_db_update(root, dt=15, n_iter=120, engine=engine)

    grizli_db.set_phot_root(root, phot_root, engine)

    res = pd.read_sql_query("SELECT root, id, flux_radius, mag_auto, z_map, status, bic_diff, zwidth1, log_pdf_max, chinu FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit_v2 WHERE z_map > 0 AND root = '{0}') z ON (p.p_root = z.root AND p.p_id = z.id)".format(root), engine)

    return res

    if False:
        res = pd.read_sql_query("SELECT root, id, status, redshift, bic_diff, mtime FROM redshift_fit_v2 WHERE (root = '{0}')".format(root), engine)

        # Get arguments
        args = fit_redshift_lambda.fit_lambda(root=root, beams=[], ids=ids, newfunc=False, bucket_name='grizli-v2', skip_existing=False, sleep=False, skip_started=False, quasar_fit=False, output_path=None, show_event=2, zr=[0.01, 3.4], force_args=True)


def set_phot_root(root, phot_root, engine):
    """
    """
    print(f'Set phot_root = {root} > {phot_root}')

    SQL = f"""UPDATE redshift_fit_v2
     SET phot_root = '{phot_root}'
    WHERE (root = '{root}');
    """

    engine.execute(SQL)

    if False:
        # Check where phot_root not equal to root
        res = pd.read_sql_query("SELECT root, id, status, phot_root FROM redshift_fit_v2 WHERE (phot_root != root)".format(root), engine)

        # update the one pointing where it should change in photometry_apcorr
        engine.execute("UPDATE photometry_apcorr SET p_root = 'j214224m4420' WHERE root = 'j214224m4420gr01';")
        engine.execute("UPDATE redshift_fit_v2 SET phot_root = 'j214224m4420' WHERE root LIKE 'j214224m4420g%%';")
        engine.execute("UPDATE redshift_fit_v2_quasar SET phot_root = 'j214224m4420' WHERE root LIKE 'j214224m4420g%%';")

    if False:
        # Replace in-place
        from grizli.aws import db as grizli_db
        
        engine.execute("update redshift_fit_v2 set phot_root = replace(root, 'g800l', 'grism') WHERE root not like 'j214224m4420%%' AND root LIKE '%%-grism%%")

        engine.execute("update redshift_fit_v2 set phot_root = replace(root, 'g800l', 'grism') WHERE root not like 'j214224m4420%%'")
        engine.execute("update redshift_fit_v2 set phot_root = 'j214224m4420' WHERE root like 'j214224m4420gr%%'")

        engine.execute("update redshift_fit_v2_quasar set phot_root = replace(root, 'g800l', 'grism') where root like '%%g800l%%'")

        # Set 3D-HST fields
        res = grizli_db.from_sql("select distinct root from redshift_fit_v2 where root like '%%-grism%%'", engine)
        for root in res['root']:
            grizli_db.set_phot_root(root, root, engine)
            grizli_db.set_phot_root(root.replace('-grism', '-g800l'), root, engine)
            xres = grizli_db.from_sql("select root, count(root) from redshift_fit_v2 where root like '{0}-%%' group by root".format(root.split('-')[0]), engine)
            print(xres)

        # Update OBJID for natural join
        # for tab in ['redshift_fit_v2', 'redshift_fit_v2_quasar', 'multibeam']
        SQL = """
WITH sub AS (
    SELECT objid as p_objid, p_root, p_id
    FROM photometry_apcorr
)
UPDATE redshift_fit_v2
SET objid = p_objid
FROM sub
WHERE phot_root = p_root AND id = p_id;
        """
        grizli_db.from_sql(SQL, engine)

        engine.execute(SQL)


def wait_on_db_update(root, t0=60, dt=30, n_iter=60, engine=None):
    """
    Wait for db to stop updating on root
    """
    import pandas as pd
    from astropy.table import Table
    from grizli.aws import db as grizli_db
    import numpy as np
    import time

    if engine is None:
        engine = grizli_db.get_db_engine(echo=False)

    n_i, n6_i, checksum_i = -1, -1, -1

    for i in range(n_iter):
        res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit_v2 WHERE root = '{0}'".format(root), engine)
        checksum = (2**res['status']).sum()
        n = len(res)
        n6 = (res['status'] == 6).sum()
        n5 = (res['status'] == 5).sum()
        if (n == n_i) & (checksum == checksum_i) & (n6 == n6_i):
            break

        now = utils.nowtime()
        print('{0}, {1}: n={2:<5d} n5={5:<5d} n6={3:<5d} checksum={4}'.format(root, now, n, n6, checksum, n5))
        n_i, n6_i, checksum_i = n, n6, checksum
        if i == 0:
            time.sleep(t0)
        else:
            time.sleep(dt)

    return res

##


def fit_timeouts(root='j004404m2034', mag_limits=[15, 26], sn_limit=7, min_status=None, engine=None):
    """
    Run redshift fits on lambda for a given root
    """
    from grizli.aws import fit_redshift_lambda
    from grizli import utils
    from grizli.aws import db as grizli_db
    if engine is None:
        engine = grizli_db.get_db_engine()

    import pandas as pd
    import numpy as np
    import glob
    import os

    res = pd.read_sql_query("SELECT id, status FROM redshift_fit_v2 WHERE root = '{0}' AND status = 5".format(root), engine)
    if len(res) == 0:
        return True

    ids = res['id'].tolist()

    fit_redshift_lambda.fit_lambda(root=root, beams=[], ids=ids, newfunc=False, bucket_name='grizli-v2', skip_existing=False, sleep=False, skip_started=False, quasar_fit=False, output_path=None, show_event=False, zr=[0.01, 2.4], force_args=True)

    res = grizli_db.wait_on_db_update(root, dt=15, n_iter=120, engine=engine)
    return res

    # All timeouts
    events = fit_redshift_lambda.fit_lambda(root='egs-g800l-j141956p5255', beams=[], ids=[20667], newfunc=False, bucket_name='grizli-v2', skip_existing=False, sleep=False, skip_started=False, quasar_fit=False, output_path=None, show_event=2, zr=[0.01, 2.4], force_args=True)

    res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit_v2 WHERE status = 5 AND root NOT LIKE 'cos-grism%%' ORDER BY root".format(root), engine)

    base = {'bucket': 'grizli-v2', 'skip_started': False, 'quasar_fit': False, 'zr': '0.01,2.4', 'force_args': True, 'bad_pa_threshold': 10, 'use_phot_obj': False, 'save_figures': 'png'}

    all_events = fit_redshift_lambda.generate_events(res['root'], res['id'], base=base, send_to_lambda=True)

    #################
    # Fit locally on EC2

    i0 = 0

    import os

    import pandas as pd
    import numpy as np
    from grizli.aws import db as grizli_db
    from grizli.aws import fit_redshift_lambda, lambda_handler

    engine = grizli_db.get_db_engine(echo=False)

    res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit_v2 WHERE status = 5 AND root NOT LIKE 'cos-grism%%' AND root LIKE '%%-grism%%' ORDER BY root", engine)

    res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit_v2 WHERE status = 5 AND root NOT LIKE 'cos-grism%%' AND root NOT LIKE '%%-grism%%' AND root NOT LIKE '%%g800l%%' ORDER BY root", engine)
    bucket = 'grizli-v2'

    res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit_v2 WHERE status = 5 AND root LIKE 'j114936p2222' ORDER BY id", engine)
    bucket = 'grizli-v2'

    # res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit_v2 WHERE status = 5 AND root LIKE 'cos-grism%%' order by id", engine)
    # bucket = 'grizli-cosmos-v2'

    N = len(res)
    np.random.seed(1)
    so = np.argsort(np.random.normal(size=N))

    base = {'bucket': bucket, 'skip_started': False, 'quasar_fit': False, 'zr': '0.01,3.4', 'force_args': True, 'bad_pa_threshold': 10, 'use_phot_obj': False, 'save_figures': 'png', 'verbose': True, 'working_directory': os.getcwd()}

    events = fit_redshift_lambda.generate_events(res['root'], res['id'], base=base, send_to_lambda=False)

    for event in events[i0::2]:
        lambda_handler.handler(event, {})

    ########

    xres = pd.read_sql_query("SELECT root, p_ra as ra, p_dec as dec, id, status FROM redshift_fit_v2 WHERE status = 5 AND root LIKE 'gds-grism%%' ORDER BY root".format(root), engine)

    print(len(res), len(xres))

    # show points
    xres = pd.read_sql_query("SELECT root, p_ra as ra, p_dec as dec, id, status FROM redshift_fit_v2 WHERE status = 5 AND root LIKE 'gds-grism%%' ORDER BY root".format(root), engine)


# Photometry table
def set_filter_bits(phot):
    """
    Set bits indicating available filters
    """
    import numpy as np
    filters = ['f160w', 'f140w', 'f125w', 'f110w', 'f105w', 'f098m', 'f850lp', 'f814w', 'f775w', 'f625w', 'f606w', 'f475w', 'f438w', 'f435w', 'f555w', 'f350lp', 'f390w', 'f336w', 'f275w', 'f225w']
    bits = [np.uint32(2**i) for i in range(len(filters))]

    phot['filter_bit'] = np.zeros(len(phot), dtype=np.uint32)
    phot['red_bit'] = np.zeros(len(phot), dtype=np.uint32)

    for i, filt in enumerate(filters):
        col = '{0}_flux_aper_0'.format(filt)
        if col in phot.colnames:
            red = bits[i] * np.isfinite(phot[col]) * (phot['filter_bit'] == 0)
            phot['filter_bit'] |= bits[i] * np.isfinite(phot[col])
            phot['red_bit'] |= red
            print(filt, i, bits[i], red.max())


def phot_to_dataframe(phot, root):
    """
    Convert phot_apcorr.fits table to a pandas DataFrame

    - Add 'root' column
    - remove "dummy" columns
    - rename 'xmin', 'xmax', 'ymin', 'ymax' to 'image_xmin', ...

    """
    phot['root'] = root

    set_filter_bits(phot)

    for c in ['dummy_flux', 'dummy_err']:
        if c in phot.colnames:
            phot.remove_column(c)

    for c in ['xmin', 'xmax', 'ymin', 'ymax']:
        phot.rename_column(c, 'image_'+c)

    for c in ['root', 'id', 'ra', 'dec']:
        phot.rename_column(c, 'p_'+c)

    df = phot.to_pandas()
    return df


def add_phot_to_db(root, delete=False, engine=None, nmax=500, add_missing_columns=False):
    """
    Read the table {root}_phot_apcorr.fits and append it to the grizli_db `photometry_apcorr` table
    """
    import pandas as pd
    from astropy.table import Table
    from grizli.aws import db as grizli_db
    import numpy as np

    if engine is None:
        engine = grizli_db.get_db_engine(echo=False)

    res = pd.read_sql_query("SELECT p_root, p_id FROM photometry_apcorr WHERE p_root = '{0}'".format(root), engine)
    if len(res) > 0:
        if delete:
            print('Delete rows where root={0}'.format(root))
            res = engine.execute("DELETE from photometry_apcorr WHERE (p_root = '{0}')".format(root))

            if False:
                res = engine.execute("DELETE from redshift_fit_v2 WHERE (root = '{0}')".format(root))

        else:
            print('Data found for root={0}, delete them if necessary'.format(root))
            return False

    # Read the catalog
    phot = Table.read('{0}_phot_apcorr.fits'.format(root), character_as_bytes=False)
    
    empty = SQL("SELECT * FROM photometry_apcorr WHERE false")
    
    # Add new filter columns if necessary

    df = phot_to_dataframe(phot, root)
    
    # remove columns
    remove = []
    for c in df.columns:
        if ('_corr_' in c) | ('_ecorr_' in c) | (c[-5:] in ['tot_4', 'tot_5', 'tot_6']) | ('dummy' in c):

            remove.append(c)
        
        if (not add_missing_columns) & (c not in empty.colnames):
            remove.append(c)
            
    df.drop(remove, axis=1, inplace=True)
    
    new_cols = []
    for c in df.columns:
        if c not in empty.columns:
            new_cols.append(c)

    if len(new_cols) > 0:
        for c in new_cols:
            print('Add column {0} to `photometry_apcorr` table'.format(c))
            sql = "ALTER TABLE photometry_apcorr ADD COLUMN {0} real;".format(c)
            res = engine.execute(sql)

    # Add new table
    print('Send {0}_phot_apcorr.fits to `photometry_apcorr`.'.format(root))
    if nmax > 0:
        # Split
        N = len(phot) // nmax
        for i in range(N+1):
            print('   add rows {0:>5}-{1:>5} ({2}/{3})'.format(i*nmax, (i+1)*nmax, i+1, N+1))
            df[i*nmax:(i+1)*nmax].to_sql('photometry_apcorr', engine, index=False, if_exists='append', method='multi')
    else:
        df.to_sql('photometry_apcorr', engine, index=False, if_exists='append', method='multi')


def multibeam_to_database(beams_file, engine=None, Rspline=15, force=False, **kwargs):
    """
    Send statistics of the beams.fits file to the database
    """
    import numpy as np
    import pandas as pd
    from astropy.time import Time

    from .. import multifit

    if engine is None:
        engine = get_db_engine(echo=False)

    mtime = Time(os.stat(beams_file).st_mtime, format='unix').iso
    root = '_'.join(beams_file.split('_')[:-1])
    id = int(beams_file.split('_')[-1].split('.')[0])

    res = pd.read_sql_query("SELECT mtime from multibeam_v2 WHERE (root = '{0}' AND id = {1})".format(root, id), engine)
    if len(res) == 1:
        if (res['mtime'][0] == mtime) & (not force):
            print('{0} already in multibeam_v2 table'.format(beams_file))
            return True

    mb = multifit.MultiBeam(beams_file, **kwargs)

    print('Update `multibeam_v2` and `beam_geometry_v2` tables for {0}.'.format(beams_file))

    # Dummy for loading the templates the same way as for the quasars
    # for generating the spline fit
    templ_args = {'uv_line_complex': True,
                  'broad_fwhm': 2800,
                  'narrow_fwhm': 1000,
                  'fixed_narrow_lines': True,
                  'Rspline': Rspline,
                  'include_reddened_balmer_lines': False}

    q0, q1 = utils.load_quasar_templates(**templ_args)
    for t in list(q0.keys()):
        if 'bspl' not in t:
            q0.pop(t)

    tfit = mb.template_at_z(0, templates=q0, fitter='lstsq')

    sp = tfit['line1d'].wave, tfit['line1d'].flux
    m2d = mb.get_flat_model(sp, apply_mask=True, is_cgs=True)

    mb.initialize_masked_arrays()
    chi0 = (mb.scif_mask**2*mb.ivarf[mb.fit_mask]).sum()

    # Percentiles of masked contam, sci, err and contam/sci
    pvals = np.arange(5, 96, 5)
    mpos = m2d > 0
    contam_percentiles = np.percentile(mb.contamf_mask, pvals)
    sci_percentiles = np.percentile(mb.scif_mask, pvals)
    err_percentiles = np.percentile(1/mb.sivarf[mb.fit_mask], pvals)
    sn_percentiles = np.percentile(mb.scif_mask*mb.sivarf[mb.fit_mask], pvals)
    fcontam_percentiles = np.percentile(mb.contamf_mask/mb.scif_mask, pvals)

    # multibeam dataframe
    df = pd.DataFrame()
    float_type = np.float

    df['root'] = [root]
    df['id'] = [id]
    df['objid'] = [-1]
    df['mtime'] = [mtime]
    df['status'] = [6]
    df['scip'] = [list(sci_percentiles.astype(float_type))]
    df['errp'] = [list(err_percentiles.astype(float_type))]
    df['snp'] = [list(sn_percentiles.astype(float_type))]
    df['snmax'] = [float_type((mb.scif_mask*mb.sivarf[mb.fit_mask]).max())]
    df['contamp'] = [list(contam_percentiles.astype(float_type))]
    df['fcontamp'] = [list(fcontam_percentiles.astype(float_type))]
    df['chi0'] = [np.int32(chi0)]
    df['rspline'] = [Rspline]
    df['chispl'] = [np.int32(tfit['chi2'])]
    df['mb_dof'] = [mb.DoF]
    df['wmin'] = [np.int32(mb.wave_mask.min())]
    df['wmax'] = [np.int32(mb.wave_mask.max())]

    # Input args
    for a in ['fcontam', 'sys_err', 'min_sens', 'min_mask']:
        df[a] = [getattr(mb, a)]

    # Send to DB
    res = engine.execute("DELETE from multibeam_v2 WHERE (root = '{0}' AND id = {1})".format(mb.group_name, mb.id), engine)
    df.to_sql('multibeam_v2', engine, index=False, if_exists='append', method='multi')

    # beams dataframe
    d = {}
    for k in ['root', 'id', 'objid', 'filter', 'pupil', 'pa', 'instrument', 'fwcpos', 'order', 'parent', 'parent_ext', 'ccdchip', 'sci_extn', 'exptime', 'origin_x', 'origin_y', 'pad', 'nx', 'ny', 'sregion']:
        d[k] = []

    for beam in mb.beams:
        d['root'].append(root)
        d['id'].append(id)
        d['objid'].append(-1)

        for a in ['filter', 'pupil', 'instrument','pad',
                  'fwcpos', 'ccdchip', 'sci_extn', 'exptime']:
            if a == 'pad':
                d[a].append(int(getattr(beam.grism, a)[0]))
            else:
                d[a].append(getattr(beam.grism, a))
                
        d['order'].append(beam.beam.beam)

        parent = beam.grism.parent_file.replace('.fits', '').split('_')
        d['parent'].append(parent[0])
        d['parent_ext'].append(parent[1])

        d['origin_x'].append(beam.grism.origin[1])
        d['origin_y'].append(beam.grism.origin[0])

        d['nx'].append(beam.sh[1])
        d['ny'].append(beam.sh[0])

        f = beam.grism.wcs.calc_footprint().flatten()
        fs = ','.join(['{0:.6f}'.format(c) for c in f])
        d['sregion'].append('POLYGON({0})'.format(fs))
        d['pa'].append(int(np.round(beam.get_dispersion_PA())))

    df = pd.DataFrame.from_dict(d)

    # Send to database
    res = engine.execute("DELETE from beam_geometry_v2 WHERE (root = '{0}' AND id = {1})".format(mb.group_name, mb.id), engine)
    df.to_sql('beam_geometry_v2', engine, index=False, if_exists='append', method='multi')

    if False:
        # Fix multibeam arrays
        import pandas as pd
        import numpy as np
        from sqlalchemy import types
        from grizli.aws import db as grizli_db
        engine = grizli_db.get_db_engine()

        df = pd.read_sql_query('select id, root, scip, errp, snp, contamp, fcontamp from multibeam_v2 mb', engine)
        c = 'snp'

        data = pd.DataFrame()
        data['id'] = df['id']
        data['root'] = df['root']
        dtype = {'root': types.String, 'id': types.Integer}

        for c in df.columns:
            if c.endswith('p'):
                print(c)

                dtype[c[:-1]+'_p'] = types.ARRAY(types.FLOAT)

                data[c[:-1]+'_p'] = [list(np.cast[float](line.strip()[1:-1].split(','))) for line in df[c]]

        data.to_sql('multibeam_tmp', engine, index=False, if_exists='append', method='multi')

        from sqlalchemy import types
        for c in df.columns:
            if c.endswith('p'):
                pass

        for c in df.columns:
            if c.endswith('p'):
                sql = "ALTER TABLE multibeam_v2 ADD COLUMN {0} real[];".format(c[:-1]+'_p')
                print(sql)
                sql = "UPDATE multibeam_v2 mb SET {new} = tmp.{new} FROM multibeam_tmp tmp WHERE tmp.id = mb.id AND tmp.root = mb.root;".format(new=c[:-1]+'_p')
                print(sql)

        x = grizli_db.from_sql('select id, scip, errp, snp, contamp, fcontamp from multibeam_v2 mb', engine)


def test_join():
    import pandas as pd
    
    res = pd.read_sql_query("SELECT root, id, flux_radius, mag_auto, z_map, status, bic_diff, zwidth1, log_pdf_max, chinu FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit_v2 WHERE z_map > 0) z ON (p.p_root = z.root AND p.p_id = z.id)", engine)

    res = pd.read_sql_query("SELECT * FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit_v2 WHERE z_map > 0) z ON (p.p_root = z.root AND p.p_id = z.id)", engine)

    # on root
    root = 'xxx'
    
    res = pd.read_sql_query("SELECT p.root, p.id, mag_auto, z_map, status FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit_v2 WHERE root='{0}') z ON (p.p_root = z.root AND p.p_id = z.id)".format(root), engine)


def column_comments():

    from collections import OrderedDict
    import yaml

    tablename = 'redshift_fit_v2'

    cols = pd.read_sql_query('select * from {0} where false'.format(tablename), engine)

    d = {}  # OrderedDict{}
    for c in cols.columns:
        d[c] = '---'

    if not os.path.exists('{0}_comments.yml'.format(tablename)):
        print('Init {0}_comments.yml'.format(tablename))
        fp = open('{0}_comments.yml'.format(tablename), 'w')
        yaml.dump(d, stream=fp, default_flow_style=False)
        fp.close()

    # Edit file
    comments = yaml.load(open('{0}_comments.yml'.format(tablename)))
    SQL = ""
    upd = "COMMENT ON COLUMN {0}.{1} IS '{2}';\n"
    for col in comments:
        if comments[col] != '---':
            SQL += upd.format(tablename, col, comments[col])
        else:
            print('Skip ', col)


def add_spectroscopic_redshifts(xtab, rmatch=1, engine=None, db=None):
    """
    Add spectroscopic redshifts to the photometry_apcorr table

    Input table needs (at least) columns:
       ['ra', 'dec', 'z_spec', 'z_spec_src', 'z_spec_qual_raw', 'z_spec_qual']

    """
    import glob
    import pandas as pd
    from astropy.table import vstack

    from grizli.aws import db as grizli_db
    from grizli import utils

    for c in ['ra', 'dec', 'z_spec', 'z_spec_src', 'z_spec_qual_raw', 'z_spec_qual']:
        if c not in xtab.colnames:
            print('Column {0} not found in input table'.format(c))
            return False

    if engine is None:
        engine = grizli_db.get_db_engine(echo=False)

    # Force data types
    tab = xtab[xtab['z_spec'] >= 0]
    if hasattr(tab['ra'], 'mask'):
        tab = tab[~tab['ra'].mask]

    tab['z_spec_qual'] = tab['z_spec_qual']*1
    tab['z_spec_qual_raw'] = tab['z_spec_qual_raw']*1

    if False:
        # duplicates
        fit = grizli_db.from_sql("select root, ra, dec from redshift_fit_v2", engine)

        fit = grizli_db.from_sql("select root, ra, dec from redshift_fit_v2 where ra is null", engine)

    # Select master table
    if db is None:
        res = pd.read_sql_query("SELECT p_root, p_id, p_ra, p_dec, z_spec from photometry_apcorr", engine)
        db = utils.GTable.from_pandas(res)
        for c in ['p_root', 'p_id', 'p_ra', 'p_dec']:
            db.rename_column(c, c[2:])

    idx, dr = db.match_to_catalog_sky(tab)
    hasm = (dr.value < rmatch) & (tab['z_spec'] >= 0)
    tab['z_spec_dr'] = dr.value
    tab['z_spec_ra'] = tab['ra']
    tab['z_spec_dec'] = tab['dec']

    tab['db_root'] = db['root'][idx]
    tab['db_id'] = db['id'][idx]

    tabm = tab[hasm]['db_root', 'db_id', 'z_spec', 'z_spec_src', 'z_spec_dr', 'z_spec_ra', 'z_spec_dec', 'z_spec_qual_raw', 'z_spec_qual']

    print('Send zspec to photometry_apcorr (N={0})'.format(hasm.sum()))

    df = tabm.to_pandas()
    df.to_sql('z_spec_tmp', engine, index=False, if_exists='replace', method='multi')

    SQL = """UPDATE photometry_apcorr
       SET z_spec = zt.z_spec,
       z_spec_src = zt.z_spec_src,
        z_spec_dr = zt.z_spec_dr,
        z_spec_ra = zt.z_spec_ra,
       z_spec_dec = zt.z_spec_dec,
  z_spec_qual_raw = zt.z_spec_qual_raw,
      z_spec_qual = zt.z_spec_qual
     FROM z_spec_tmp as zt
     WHERE (zt.db_root = p_root AND zt.db_id = p_id);
     """

    engine.execute(SQL)

    if False:
        # Update redshift_fit_v2 ra/dec with photometry_table double prec.
        SQL = """UPDATE redshift_fit_v2
         SET ra = p_ra
             dec = p_dec
        FROM photometry_apcorr
        WHERE (phot_root = p_root AND id = p_id AND root = 'j123556p6221');
        """


def mtime_to_iso(ct):
    """
    Convert mtime values to ISO format suitable for sorting, etc.
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    spl = ct.split()
    iso = '{yr}-{mo:02d}-{dy:02d} {time}'.format(dy=int(spl[2]), mo=int(months.index(spl[1])+1), yr=spl[-1], time=spl[-2])
    return iso


def various_selections():
    from grizli.aws import db as grizli_db
    
    # sdss z_spec
    res = make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max'], where="AND status > 5 AND z_spec > 0 AND z_spec_qual = 1 AND z_spec_src ~ '^sdss-dr15'", table_root='sdss_zspec', sync='s3://grizli-v2/tables/')

    # objects with carla redshifts (radio loud)
    res = make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max'], where="AND status > 5 AND z_spec > 0 AND z_spec_qual = 1 AND z_spec_src ~ '^carla'", table_root='carla_zspec', sync='s3://grizli-v2/tables/')

    # Bright galaxies with q_z flag
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 't_g800l', 't_g102', 't_g141', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'zwidth1/(1+z_map) as zw1', 'q_z', 'q_z > -0.69 as q_z_TPR90', 'dlinesn'], where="AND status > 4 AND mag_auto < 22 AND z_map > 1.3", table_root='bright', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'], show_hist=True)

    # High-z compiliation
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 't_g800l', 't_g102', 't_g141', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'q_z', 'h_zphot', 'h_src', 'h_dr'], where="AND status > 4 AND phot_root = h_root AND id = h_id AND h_dr < 1", tables=['highz_2015'], table_root='highz', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'], show_hist=True)

    # z_spec with dz
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_spec', 'z_map', 'z_spec_src', 'bic_diff', 'chinu', 'log_pdf_max', 'zwidth1/(1+z_map) as zw1', '(z_map-z_spec)/(1+z_spec) as dz', 'dlinesn'], where="AND status > 4 AND z_spec > 0 AND z_spec_qual = 1", table_root='zspec_delta', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'])

    # Point sources
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'id', 'red_bit', 'status', 'p_ra', 'p_dec', 't_g800l', 't_g102', 't_g141', 'mag_auto', 'flux_radius', 'z_map', 'z_spec', 'z_spec_src', 'z_spec_dr', 'bic_diff', 'chinu', 'log_pdf_max', 'q_z', 'zwidth1/(1+z_map) as zw1', 'dlinesn'], where="AND status > 4 AND mag_auto < 24 AND flux_radius < 1.9 AND ((flux_radius < 1.5 AND flux_radius > 0.75 AND red_bit > 32) OR (flux_radius < 1.9 AND flux_radius > 1.0 AND red_bit < 32))", table_root='point_sources', sync='s3://grizli-v2/tables/', png_ext=['stack', 'line', 'full', 'qso.full', 'star'], get_sql=False)

    # Reliable redshifts
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'id', 'status', 'p_ra', 'p_dec', 't_g800l', 't_g102', 't_g141', 'mag_auto', 'flux_radius', '(flux_radius < 1.7 AND ((flux_radius < 1.4 AND flux_radius > 0.75 AND red_bit > 32) OR (flux_radius < 1.7 AND flux_radius > 1.0 AND red_bit < 32)))::int as is_point', 'z_map', 'z_spec', 'z_spec_src', 'z_spec_dr', 'sn_siii', 'sn_ha', 'sn_oiii', 'sn_oii',  'ew50_ha', 'd4000', 'd4000_e', 'bic_diff', 'chinu', 'log_pdf_max', 'q_z', 'zwidth1/(1+z_map) as zw1', 'dlinesn'], where="AND status > 4 AND chinu < 30 AND q_z > -0.7 order by q_z", table_root='reliable_redshifts', sync='s3://grizli-v2/tables/', png_ext=['stack', 'line', 'full'], get_sql=False, sort_column=('q_z', -1))

    # stellar classification?
#     sql = """SELECT root, id, ra, dec, status, z_map, q_z_map, bic_diff,
#        bic_diff_star,
#        chinu as t_chinu, s_chinu, q_chinu,
#        chinu - q_chinu as tq_chinu, q_chinu - s_chinu as qs_chinu,
#        chinu - s_chinu as ts_chinu, stellar_template
# FROM redshift_fit_v2,
#      (SELECT root as s_root, id as s_id, chinu as s_chinu, bic_diff_star,
#              stellar_template
#         FROM stellar_fit
#         WHERE status = 6
#       ) as s,
#       (SELECT root as q_root, id as q_id, chinu as q_chinu,
#               bic_diff as q_bic_diff, z_map as q_z_map
#          FROM redshift_fit_v2_quasar
#          WHERE status = 6
#        ) as q
# WHERE (root = s_root AND id = s_id) AND (root = q_root AND id = q_id)
#     """
    #res = grizli_db.make_html_table(engine=engine, res=cstar, table_root='carbon_stars', sync='s3://grizli-v2/tables/', png_ext=['stack','line', 'full', 'qso.full', 'star'], sort_column=('bic_diff_star', -1), get_sql=False)

    sql = """SELECT root, id, status, ra, dec, t_g800l, t_g102, t_g141,
       z_map, q_z_map, bic_diff,
       bic_diff_star, (bic_diff_star > 10 AND q_chinu < 20 AND chinu - q_chinu > 0.05 AND q_chinu-s_chinu > 0 AND chinu-s_chinu > 0.1)::int as is_star,
       chinu as t_chinu, s_chinu, q_chinu,
       bic_qso-bic_gal as bic_gq,
       bic_gal-bic_star as bic_gs,
       bic_qso-bic_star as bic_qs,
       (bic_spl+chimin)-bic_gal as bic_gx,
       bic_spl_qso-bic_qso as bic_qx,
       q_vel_bl, qso_q_z, qso_zw1, stellar_template
FROM (SELECT *, bic_temp+chimin as bic_gal FROM redshift_fit_v2 z,
      (SELECT root as q_root, id as q_id, chinu as q_chinu,
              bic_diff as q_bic_diff, bic_temp+chimin as bic_qso,
              bic_spl+chimin as bic_spl_qso,
              z_map as qso_z_map,
              zwidth1/(1+z_map) as qso_zw1, vel_bl as q_vel_bl,
              q_z as qso_q_z
         FROM redshift_fit_v2_quasar
         WHERE status = 6
       ) q
WHERE (root = q_root AND id = q_id)) c
    LEFT JOIN
     (SELECT root as s_root, id as s_id, chinu as s_chinu,
             LN(dof)*nk+chi2 as bic_star,
             LN(dof)*(nk-1)+chi2_flat as bic_spline,
             bic_diff_star,
             stellar_template
        FROM stellar_fit
        WHERE status = 6
      ) s ON (root = s_root AND id = s_id) WHERE chinu-q_chinu > 0.5
    """
    cstar = grizli_db.from_sql(sql, engine)
    cstar['is_star'] = cstar['is_star'].filled(-1)
    print('N={0}'.format(len(cstar)))

    res = grizli_db.make_html_table(engine=engine, res=cstar, table_root='quasars_and_stars', sync='s3://grizli-v2/tables/', png_ext=['stack', 'line', 'full', 'qso.full', 'star'], sort_column=('bic_diff_star', -1), get_sql=False)

    # best-fit as quasar
    sql = """SELECT root, id, ra, dec, status, z_map, q_z_map,
       q_z, bic_diff, q_bic_diff,
       chinu as t_chinu, q_chinu,
       chinu - q_chinu as tq_chinu,
       (q_bic_temp + q_chimin) - (bic_temp + chimin) as bic_diff_quasar,
       q_vel_bl
    FROM redshift_fit_v2 z JOIN
      (SELECT root as q_root, id as q_id, chinu as q_chinu,
              bic_diff as q_bic_diff, z_map as q_z_map, vel_bl,
              chimin as q_chimin, bic_temp as q_bic_temp, vel_bl as q_vel_bl
         FROM redshift_fit_v2_quasar
         WHERE status = 6
       ) as q
    WHERE (root = q_root AND id = q_id) AND status = 6 AND q_z > -1
    """
    qq = grizli_db.from_sql(sql, engine)
    res = grizli_db.make_html_table(engine=engine, res=qq, table_root='quasar_fit', sync='s3://grizli-v2/tables/', png_ext=['stack', 'line', 'full', 'qso.full', 'star'], get_sql=False)

    # Strong lines
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'id', 'red_bit', 'status', 'p_ra', 'p_dec', 't_g800l', 't_g102', 't_g141', 'mag_auto', 'flux_radius', 'z_map', 'z_spec', 'z_spec_src', 'z_spec_dr', 'bic_diff', 'chinu', 'log_pdf_max', 'q_z', 'zwidth1/(1+z_map) as zw1', 'dlinesn', 'sn_ha', 'sn_oiii', 'sn_oii'], where="AND status > 4 AND mag_auto < 24 AND (sn_ha > 10 OR sn_oiii > 10 OR sn_oii > 10) AND flux_radius >= 1.6", table_root='strong_lines', sync='s3://grizli-v2/tables/', png_ext=['stack', 'full', 'qso.full', 'star'])

    # brown dwarf?
    tablename = 'spec1d_r30_g141'
    wave = pd.read_sql_query("SELECT * from {0}_wave".format(tablename),
                             engine)[tablename+'_wave'].values

    # 1.15, 1.25, 1.4
    i0 = 25, 28, 29, 32
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_map', 'z_spec', 'z_spec_src', 'bic_diff', 'chinu', 'log_pdf_max', 'q_z', 'zwidth1/(1+z_map) as zw1', 'dlinesn', '{0}_flux[25]/{0}_flux[28] as c1'.format(tablename), '{0}_flux[32]/{0}_flux[28] as c2'.format(tablename)], where="AND status > 4 AND flux_radius < 2 AND flux_radius > 1 AND mag_auto < 25 AND {0}_root = root AND {0}_id = id AND {0}_flux[28] > 0 AND {0}_flux[28]/{0}_err[28] > 5 AND {0}_flux[32] > 0 AND {0}_flux[25] > 0 AND {0}_flux[32]/{0}_flux[28] < 0.5".format(tablename), tables=[tablename], table_root='point_sources_colors', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'])

    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_map', 'z_spec', 'z_spec_src', 'bic_diff', 'chinu', 'log_pdf_max', 'q_z', 'zwidth1/(1+z_map) as zw1', 'dlinesn', '{0}_flux[25] as c25'.format(tablename), '{0}_flux[32] as c32'.format(tablename)], where="AND status > 4 AND z_spec = 0".format(tablename), tables=[tablename], table_root='point_sources_colors', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'])

    # with line ratios
    lstr = 'err_{0} > 0 AND err_{0} < 5e-17'
    err_lines = ' AND '.join(lstr.format(li) for li in
                             ['hb', 'oiii', 'ha', 'sii'])

    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_spec', 'z_map', 'z_spec_src', 'bic_diff', 'chinu', 'log_pdf_max', 'zwidth1/(1+z_map) as zw1', '(z_map-z_spec)/(1+z_spec) as dz', 'dlinesn', 'flux_hb/flux_ha as HbHa', 'flux_hb/flux_oiii as HbO3', 'flux_oiii/flux_ha as O3Ha'], where="AND status > 4 AND z_spec > 0 AND z_spec_qual = 1 AND sn_oiii > 3 AND sn_ha > 2 AND  {0}".format(err_lines), table_root='zspec_lines', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'])

    if False:
        from matplotlib.ticker import FixedLocator, AutoLocator, MaxNLocator
        import matplotlib.pyplot as plt
        
        xti = xt = np.arange(0, 3.6, 0.5)
        loc = np.arange(0, 3.6, 0.1)
        bins = utils.log_zgrid([0.03, 3.5], 0.01)
        fig = plt.figure(figsize=[7, 6])
        ax = fig.add_subplot(111)
        ax.scatter(np.log(1+res['z_spec']), np.log(1+res['z_map']), alpha=0.2, c=np.log10(res['zw1']), marker='.', vmin=-3.5, vmax=-0.5, cmap='plasma')

        sc = ax.scatter(np.log([1]), np.log([1]), alpha=0.8, c=[0], marker='.', vmin=-3.5, vmax=-0.5, cmap='plasma')

        cb = plt.colorbar(sc, shrink=0.6)
        cb.set_label(r'$(z_{84}-z_{16})/(1+z_{50})$')
        cb.set_ticks([-3, -2, -1])
        cb.set_ticklabels([0.001, 0.01, 0.1])

        xts = ax.set_xticks(np.log(1+xt))
        xtl = ax.set_xticklabels(xti)
        xts = ax.set_yticks(np.log(1+xt))
        xtl = ax.set_yticklabels(xti)

        ax.set_xlim(0, np.log(1+3.5))
        ax.set_ylim(0, np.log(1+3.5))

        ax.xaxis.set_minor_locator(FixedLocator(np.log(1+loc)))
        ax.yaxis.set_minor_locator(FixedLocator(np.log(1+loc)))
        ax.set_xlabel('z_spec')
        ax.set_ylabel('z_MAP')
        ax.set_aspect(1)
        ax.grid()
        ax.text(0.95, 0.05, r'$N={0}$'.format(len(res)), ha='right', va='bottom', transform=ax.transAxes)
        ax.plot(ax.get_xlim(), ax.get_xlim(), color='k', alpha=0.2, linewidth=1, zorder=-10)
        fig.tight_layout(pad=0.1)
        fig.savefig('grizli_v1_literature_zspec.pdf')

    # COSMOS test
    root = 'cos-grism-j100012p0210'
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 't_g800l', 't_g102', 't_g141', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'zwidth1/(1+z_map) as zw1', 'dlinesn'], where="AND status > 4 AND bic_diff > 100 AND root = '{0}'".format(root), table_root=root, sync='s3://grizli-v2/Pipeline/{0}/Extractions/'.format(root), png_ext=['R30', 'stack', 'full', 'line'], show_hist=True)

    # high bic_diff = unambiguous
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', 'd4000', 'd4000_e', '-(bic_temp-bic_spl) as bic_diff_spl'], where="AND status > 5 AND (((bic_diff > 50 OR  zwidth1/(1+z_map) < 0.01) AND chinu < 2))", table_root='unamb', sync='s3://grizli-v2/tables/')

    # with d4000
    res = make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', 'd4000', 'd4000_e'], where="AND status > 5 AND chinu < 3 AND d4000 > 1 AND d4000 < 5 AND d4000_e > 0 AND d4000_e < 0.25 AND bic_diff > 5", table_root='d4000', sync='s3://grizli-v2/tables/')

    # LBG?
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 't_g800l', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', '-(bic_temp-bic_spl) as bic_diff_spl', 'splf01/splf02 as r12', 'splf02/splf03 as r23', 'splf02/sple02 as sn02'], where="AND status > 5 AND mag_auto > 23 AND bic_diff > -50 AND splf01/splf02 < 0.3 AND splf02/sple02 > 2 AND splf01 != 0 AND splf02 != 0 AND splf03 != 0 ".format(root), table_root='lbg_g800l', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'])

    # stars?
    res = make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max'], where="AND status > 5 AND bic_diff > 100 AND chinu < 1.5 AND mag_auto < 24 AND sn_Ha > 20", table_root='star', sync='s3://grizli-v2/tables/')

    # By root
    root = 'j001420m3030'
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 't_g800l', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max'], where="AND status > 5 AND root = '{0}' AND bic_diff > 5".format(root), table_root=root+'-fit', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'])

    # G800L spec-zs
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 't_g800l', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', '(z_map-z_spec)/(1+z_spec) as delta_z'], where="AND status > 5 AND z_spec > 0 AND z_spec_qual = 1 AND t_g800l > 0", table_root='zspec_g800l', sync='s3://grizli-v2/tables/')

    # Large G800L likely mismatch [OIII]/Ha
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'a_image', 'flux_radius', 't_g800l', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', 'err_ha', 'ew50_oiii/(1+z_map) as ew_oiii_rest', 'sn_oiii'], where="AND status > 5 AND t_g800l > 0 AND sn_oiii > 3 AND mag_auto < 23 AND bic_diff > 5", table_root='g800l_oiii_mismatch', sync='s3://grizli-v2/tables/')

    # Potential Ly-a?
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'a_image', 'flux_radius', 't_g800l', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', 'err_ha', 'ew50_oiii/(1+z_map) as ew_oiii_rest', 'sn_oiii'], where="AND status > 5 AND t_g800l > 0 AND sn_oiii > 5 AND sn_ha > 0 AND flux_oiii/flux_ha > 1.8", table_root='g800l_oiii_mismatch', sync='s3://grizli-v2/tables/')

    # Continuum resid
    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 't_g800l', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', '23.9-2.5*log(splf01*8140*8140/3.e18*1.e29)-mag_auto as dmag'], where="AND status > 5 AND bic_diff > 5 AND splf01 > 0 AND bic_diff > 50".format(root), table_root='xxx', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'])

    res = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'a_image', 'flux_radius', 't_g800l', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', 'err_ha', 'sn_oiii', 'f814w_tot_1*3.e18/8140/8140/splf01*1.e-29 as fresid', 'splf01/sple01 as sn01', '23.9-2.5*log(splf01*8140*8140/3.e18*1.e29)-mag_auto as dmag'], where="AND status > 5 AND t_g800l > 0 AND f814w_tot_1 > 0 AND splf01 != 0 AND splf01/sple01 > 1 AND f814w_tot_1*3.e18/8140/8140/splf01*1.e-29 > 0 AND (f814w_tot_1*3.e18/8140/8140/splf01*1.e-29 < 0.3 OR f814w_tot_1*3.e18/8140/8140/splf01*1.e-29 > 4)", table_root='g800l_oiii_mismatch', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'])

    sql = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'a_image', 'flux_radius', 't_g800l', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', 'err_ha', 'sn_oiii', 'splf01', 'sple01', 'f814w_tot_1', 'f850lp_tot_1', 'flux_auto/flux_iso as flux_aper_corr', '23.9-2.5*log(splf01*8140*8140/3.e18*1.e29)-mag_auto as dmag'], where="AND status > 5 AND t_g800l > 0 AND splf01 > 0", table_root='g800l_oiii_mismatch', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'], get_sql=True)
    res = pd.read_sql_query(sql, engine)

    splmag = 23.9-2.5*np.log10(np.maximum(res['splf01'], 1.e-22)*8140**2/3.e18*1.e29)

    sql = grizli_db.make_html_table(engine=engine, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'a_image', 'flux_radius', 't_g800l', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', 'err_ha', 'sn_oiii', 'splf03', 'sple03', 'f140w_tot_1', 'f160w_tot_1', 'flux_auto/flux_iso as flux_aper_corr'], where="AND status > 5 AND t_g141 > 0 AND sple03 > 0", table_root='g800l_oiii_mismatch', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'], get_sql=True)
    res = pd.read_sql_query(sql, engine)
    splmag = 23.9-2.5*np.log10(np.maximum(res['splf03'], 1.e-22)*1.2e4**2/3.e18*1.e29)

    # Number of matches per field
    counts = pd.read_sql_query("select root, COUNT(root) as n from redshift_fit_v2, photometry_apcorr where phot_root = p_root AND id = p_id AND bic_diff > 50 AND mag_auto < 24 group by root;", engine)


def from_sql(query_text, engine=None, **kwargs):
    """
    Deprecated, see `grizli.aws.db.SQL`.
    """
    tab = SQL(query_text, engine=engine, **kwargs)
    return tab


def SQL(query_text, engine=None, pd_kwargs={}, **kwargs):
    """
    Send a query to a database through `pandas.read_sql_query`
    
    Parameters
    ----------
    query_text : str
        Query text, e.g., 'SELECT count(*) FROM table;'.
    
    engine : `sqlalchemy.engine.Engine`
        DB connection engine initialized with `~grizli.aws.db.get_db_engine`.  
        The query is sent through 
        ``pandas.read_sql_query(query_text, engine)``.
    
    kwargs : dict
        Column formatting keywords passed to 
        `~grizli.aws.db.set_column_formats`
    
    Returns
    -------
    tab : `~grizli.utils.GTable`
        Table object
        
    """
    import pandas as pd
    
    global _ENGINE
    if engine is not None:
        _ENGINE = engine
    
    if _ENGINE is None:
        _ENGINE = get_db_engine()
    
    refresh_engine()
    
    res = pd.read_sql_query(query_text, _ENGINE)
    _ENGINE.dispose()
    
    tab = utils.GTable.from_pandas(res)
    tab.meta['query'] = (query_text, 'SQL query text')
    tab.meta['qtime'] = (utils.nowtime(), 'Query timestamp')
    
    set_column_formats(tab, **kwargs)

    return tab


def execute(sql_text):
    """
    Wrapper around engine.execute()
    """
    global _ENGINE
    
    if _ENGINE is None:
        _ENGINE = get_db_engine()
    else:
        refresh_engine()
    
    resp = _ENGINE.execute(sql_text)
    return resp


def send_to_database(table_name, tab, index=False, if_exists='append', method='multi', **kwargs):
    """
    Send table object to remote database with `pandas.DataFrame.to_sql`
    
    Parameters
    ----------
    table_name : str
        Name of table to create/append
        
    tab : `~astropy.table.Table` or `~pandas.DataFrame`
        Input table.  If an `~astropy.table.Table`, then convert to 
        `~pandas.DataFrame` with `tab.to_pandas()`
    
    index : bool
        Add index column (`pandas.DataFrame.to_sql` keyword)
    
    if_exists : 'fail', 'replace', 'append'
        Action to take if `table_name` already exists in the database
        (`pandas.DataFrame.to_sql` keyword). 
        
    kwargs : dict
        See `pandas.DataFrame.to_sql`
        
    Returns
    -------
    resp : object
        Response from `pandas.DataFrame.to_sql`
        
    """
    import pandas as pd
    
    global _ENGINE
    
    if _ENGINE is None:
        _ENGINE = get_db_engine()
    else:
        refresh_engine()
    
    if isinstance(tab, pd.DataFrame):
        df = tab
    else:
        df = tab.to_pandas()
        
    resp = df.to_sql(table_name, _ENGINE, index=index, 
                     if_exists=if_exists, method=method)
    
    return resp


def render_for_notebook(tab, image_extensions=['stack', 'full', 'line'], bucket='grizli-v2', max_rows=20, link_root=True, link_type='grism', get_table=False):
    """
    Render images for inline display in a notebook

    In [1]: from IPython.display import HTML

    In [2]: HTML(tab)

    """
    import pandas as pd
    from eazy import utils as eu
    
    pd.set_option('display.max_colwidth', -1)

    rows = tab[:max_rows].copy()
    buckets = [bucket]*len(rows)
    for i, r in enumerate(rows['root']):
        if r.startswith('cos-g'):
            buckets[i] = 'grizli-cosmos-v2'

    rows['bucket'] = buckets

    rows['ext'] = 'longstring'  # longer than the longest extension

    s3url = 'https://s3.amazonaws.com/{bucket}/HST/Pipeline/{root}/Extractions/{root}_{id:05d}.{ext}.png'

    def href_root(root):
        if root.startswith('cos-g'):
            bucket_i = 'grizli-cosmos-v2'
        else:
            bucket_i = bucket

        s3 = 'https://s3.amazonaws.com/'+bucket_i+'/HST/Pipeline/{0}/Extractions/{0}.html'
        return '<a href={0}>{1}</a>'.format(s3.format(root), root)
    
    def path_to_image_html(path):
        return '<a href={0}><img src="{0}"/></a>'.format(path)

    # link for root

    fmt = {}
    cols = list(rows.colnames)
    
    if link_root:
        if link_type == 'grism':
            fmt = {'root': href_root}
        elif (link_type in ['cds','eso','alma','mast']) & ('ra' in cols):
            funcs = {'cds':eu.cds_query, 
                     'eso':eu.eso_query,
                     'alma':eu.alma_query,
                     'mast':eu.mast_query}
            
            urls = [funcs[link_type](ra, dec) 
                      for ra, dec in zip(tab['ra'], tab['dec'])]
            href = [f'<a href="{u}"> {r} {i} </a>'
                    for u, r, i in zip(urls, tab['root'], tab['id'])]
            
            rows['xroot'] = href
            cols = ['xroot'] + cols
            for c in ['root','id','ra','dec']:
                cols.pop(cols.index(c))
                
    for ext in image_extensions:
        rows['ext'] = ext
        urls = [s3url.format(**row) for row in rows.to_pandas().to_dict(orient='records')]
        rows[ext] = urls
        fmt[ext] = path_to_image_html
        cols.append(ext)
        
    rows.remove_columns(['bucket', 'ext'])
    for c in ['bucket','ext']:
        cols.pop(cols.index(c))
    
    if get_table:
        return rows, cols, fmt
        
    out = rows[cols].to_pandas().to_html(escape=False, formatters=fmt)
    return out


def add_to_charge():
    from grizli.aws import db
    engine = db.get_db_engine()

    p = db.from_sql('select distinct p_root from photometry_apcorr', engine)
    f = db.from_sql('select distinct field_root from charge_fields', engine)

    new_fields = []
    for root in p['p_root']:
        if root not in f['field_root']:
            print(root)
            new_fields.append(root)

    df = pd.DataFrame()
    df['field_root'] = new_fields
    df['comment'] = 'CANDELS'
    ix = df['field_root'] == 'j214224m4420'
    df['comment'][ix] = 'Rafelski UltraDeep'
    
    df.to_sql('charge_fields', engine, index=False, if_exists='append', method='multi')
    
    df = pd.DataFrame()
    df['field_root'] = ['fresco-gds-med']
    df['comment'] = ['FRESCO-v5.1']

    
    df.to_sql('charge_fields', engine, index=False, if_exists='append', method='multi')
    
    
def add_by_footprint(footprint_file='j141156p3415_footprint.fits', engine=None):
    
    import pandas as pd
    from grizli.aws import db
    
    ## By footprint
    if engine is None:
        engine = db.get_db_engine()

    #ch = pd.read_sql_query('select * from charge_fields', engine)
    
    f = pd.read_sql_query('select distinct field_root from charge_fields', engine)
    
    fp = utils.read_catalog(footprint_file)
    
    root = fp.meta['NAME']
    if root in f['field_root'].tolist():
        print(f'Field found: {root}')
        return False
        
    df = pd.DataFrame()
    df['field_root'] = [root]
    df['comment'] = 'manual'
    df['field_xmin'] = fp.meta['XMIN']
    df['field_xmax'] = fp.meta['XMAX']
    df['field_ymin'] = fp.meta['YMIN']
    df['field_ymax'] = fp.meta['YMAX']
    df['field_ra'] = np.mean(fp['ra'])
    df['field_dec'] = np.mean(fp['dec'])
    df['mw_ebv'] = fp.meta['MW_EBV']
    
    fp.rename_column('filter','filters')
    
    for k in ['filters','target','proposal_id']:
        df[k] = ' '.join([t for t in np.unique(fp[k])])
    
    #df['proposal_id'] = ' '.join([t for t in np.unique(fp['target'])])
    print(f'Send {root} to db.charge_fields')
    df.to_sql('charge_fields', engine, index=False, if_exists='append', method='multi')
    
def update_charge_fields():
    """
    """
    from grizli.aws import db
    
    files = [f.replace('.png','.fits') for f in glob.glob('j*footprint.png')]
    files.sort()
    
    for file in files:
        db.add_by_footprint(file, engine=engine)
    
    orig = db.from_sql('select field_root, log from charge_fields', engine)
    gtab = db.from_sql('select field_root, log from charge_fields', engine)
    
    bucket = 'grizli-v2'
    
    for st, dir in enumerate(['Start','Failed','Finished']):
        print(dir)
        os.system('aws s3 ls s3://{0}/HST/Pipeline/Log/{1}/ | sed "s/.log//" > /tmp/{1}'.format(bucket, dir))
        fin = utils.read_catalog(f'/tmp/{dir}', format='ascii')
        print('{0} {1}'.format(dir, len(fin)))
        for i, r in enumerate(fin['col4']):
            ix = gtab['field_root'] == r
            if ix.sum() > 0:
                gtab['log'][ix] = '{0} {1}-{2}'.format(dir, fin['col1'][i], fin['col2'][i])
    
    # update the table
    df = gtab[~gtab['log'].mask].to_pandas()
    df.to_sql('log_tmp', engine, index=False, if_exists='replace', method='multi')
    
    sql = "UPDATE charge_fields ch SET log = tmp.log FROM log_tmp tmp WHERE tmp.field_root = ch.field_root"

    engine.execute(sql)


def overview_table():
    """
    Generate a new overview table with the redshift histograms
    """
    from grizli.aws import db as grizli_db
    import pandas as pd
    from grizli import utils

    engine = grizli_db.get_db_engine()

    ch = from_sql("select * from charge_fields", engine)

    by_mag = from_sql("select p_root as root, COUNT(p_root) as nmag from photometry_apcorr where mag_auto < 24 group by p_root;", engine)
    by_nz = from_sql("select root, COUNT(root) as nz from redshift_fit_v2 where bic_diff > 30 group by root;", engine)

    for count in [by_mag, by_nz]:
        new_col = count.colnames[1]
        ch[new_col] = -1
        for r, n in zip(count['root'], count[new_col]):
            ix = ch['field_root'] == r
            ch[new_col][ix] = n

    zhist = ['https://s3.amazonaws.com/grizli-v2/HST/Pipeline/{0}/Extractions/{0}_zhist.png'.format(r) for r in ch['field_root']]
    ch['zhist'] = ['<a href="{1}"><img src={0} height=300px></a>'.format(zh, zh.replace('_zhist.png', '.html')) for zh in zhist]

    cols = ['field_root', 'field_ra', 'field_dec', 'mw_ebv', 'gaia5', 'nassoc', 'nfilt', 'filter', 'target', 'comment', 'proposal_id', 'proposal_pi', 'field_t_g800l', 'field_t_g102', 'field_t_g141', 'mast', 'footprint', 'rgb', 'nmag', 'nz', 'zhist', 'summary', 'log']

    sortable = []
    for c in cols:
        if not hasattr(ch[c][0], 'upper'):
            sortable.append(c)

    # https://s3.amazonaws.com/grizli-v2/Master/CHArGE-July2019.html

    table_root = 'CHArGE-July2019.zhist'

    ch[cols].write_sortable_html('{0}.html'.format(table_root), replace_braces=True, localhost=False, max_lines=1e5, table_id=None, table_class='display compact', css=None, filter_columns=sortable, buttons=['csv'], toggle=True, use_json=True)

    os.system('aws s3 sync ./ s3://grizli-v2/Master/ --exclude "*" --include "{1}.html" --include "{1}.json" --acl public-read'.format('', table_root))


def run_all_redshift_fits():
    ##############
    # Run all
    from grizli.aws import db as grizli_db
    import pandas as pd
    engine = grizli_db.get_db_engine()

    # By grism
    res = pd.read_sql_query("select field_root, field_t_g800l, field_t_g102, field_t_g141, proposal_pi from charge_fields where (nassoc < 200 AND (field_t_g800l > 0 OR field_t_g141 > 0 OR  field_t_g102 > 0) AND log LIKE '%%inish%%');", engine)
    orig_roots = pd.read_sql_query('select distinct root from redshift_fit_v2', engine)['root'].tolist()

    count = 0
    for i, (root, ta, tb, tr, pi) in enumerate(zip(res['field_root'], res['field_t_g800l'], res['field_t_g102'], res['field_t_g141'], res['proposal_pi'])):
        if root in orig_roots:
            continue

        count += 1

        zmax = 1.6
        if tb > 0:
            zmax = 2.2

        if tr > 0:
            zmax = 3.2

        print('\n\n', i, count, root, ta, tb, tr, pi, zmax, '\n\n')

        phot_root = None

        try:
            grizli_db.run_lambda_fits(root, phot_root=phot_root,
                                      min_status=6, zr=[0.01, zmax])
        except:
            pass

    ####
    # Redo fits on reprocessed fields

    # for i in range(2,11):
    #     root = 'j214224m4420gr{0:02d}'.format(i)
    #     print(root)
    #
    res = engine.execute("DELETE from redshift_fit_v2 WHERE (root = '{0}')".format(root), engine)
    res = engine.execute("DELETE from redshift_fit_v2_quasar WHERE (root = '{0}')".format(root), engine)
    res = engine.execute("DELETE from stellar_fit WHERE (root = '{0}')".format(root), engine)
    res = engine.execute("DELETE from photometry_apcorr WHERE (p_root = '{0}')".format(root), engine)

    if False:
        # Remove the whole thing
        res = engine.execute("DELETE from exposure_log WHERE (parent = '{0}')".format(root), engine)
        res = engine.execute("DELETE from charge_fields WHERE (field_root = '{0}')".format(root), engine)

    grizli_db.run_lambda_fits(root, phot_root=root, min_status=2, zr=[0.01, zmax], mag_limits=[15, 26], engine=engine)

    # for root in "j233844m5528 j105732p3620 j112416p1132 j113812m1134 j113848m1134 j122852p1046 j143200p0959 j152504p0423 j122056m0205 j122816m1132 j131452p2612".split():
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 't_g800l', 't_g102', 't_g141', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'zwidth1/(1+z_map) as zw1', 'q_z', 'q_z > -0.69 as q_z_TPR90', 'dlinesn'], where="AND status > 4 AND root = '{0}'".format(root), table_root=root, sync='s3://grizli-v2/HST/Pipeline/{0}/Extractions/'.format(root), png_ext=['R30', 'stack', 'full', 'rgb', 'line'], show_hist=True)

    grizli_db.aws_rgb_thumbnails(root, engine=engine)

    os.system('aws s3 cp s3://grizli-v2/HST/Pipeline/{0}/Extractions/{0}_zhist.png s3://grizli-v2/tables/'.format(root))


def aws_rgb_thumbnails(root, bucket='grizli-v2', engine=None, thumb_args={}, ids=None, verbose=True, res=None):
    """
    Make thumbnails for everything that has an entry in the redshift_fit_v2 table
    """
    from grizli.aws import aws_drizzler, fit_redshift_lambda

    if engine is None:
        engine = get_db_engine(echo=False)

    if res is None:
        res = from_sql("SELECT root, id, ra, dec FROM redshift_fit_v2 WHERE root = '{0}' AND ra > 0".format(root), engine)

    aws_prep_dir = 's3://{0}/HST/Pipeline/{1}/Prep/'.format(bucket, root)
    aws_bucket = 's3://{0}/HST/Pipeline/{1}/Thumbnails/'.format(bucket, root)

    event = {'make_segmentation_figure': True,
             'aws_prep_dir': aws_prep_dir,
             'single_output': True,
             'combine_similar_filters': True,
             'show_filters': ['visb', 'visr', 'y', 'j', 'h'],
             'include_ir_psf': False,
             'include_saturated': True,
             'subtract_median': True,
             'sync_fits': True,
             'thumb_height': 2.0,
             'scale_ab': 21,
             'aws_bucket': aws_bucket,
             'master': None,
             'rgb_params': {'xsize': 4, 'output_dpi': None,
                            'rgb_min': -0.01, 'add_labels': False,
                            'output_format': 'png', 'show_ir': False,
                            'scl': 2, 'suffix': '.rgb', 'mask_empty': False,
                            'tick_interval': 1, 'pl': 1},
             'remove': True,
             'filters': ['f160w', 'f140w', 'f125w', 'f105w', 'f110w', 'f098m',
                         'f850lp', 'f814w', 'f775w', 'f606w', 'f475w',
                         'f555w', 'f600lp', 'f390w', 'f350lp'],
             'half_optical_pixscale': True,
             'theta': 0,
             'kernel': 'square',
             'pixfrac': 0.33,
             'wcs': None,
             'size': 6,
             'pixscale': 0.1}

    for k in thumb_args:
        event[k] = thumb_args[k]

    N = len(res)
    for i in range(N):

        id = res['id'][i]
        ra = res['ra'][i]
        dec = res['dec'][i]
        root_i = res['root'][i]

        if ids is not None:
            if id not in ids:
                continue

        event['ra'] = ra
        event['dec'] = dec
        event['label'] = '{0}_{1:05d}'.format(root_i, id)

        fit_redshift_lambda.send_event_lambda(event, verbose=verbose)


def count_sources_for_bad_persistence():
    """
    Count the number of extracted objects for each id and look for fields
    with few objects, which are usually problems with the persistence mask
    """

    import pandas as pd
    from grizli.aws import db as grizli_db
    from grizli import utils
    engine = grizli_db.get_db_engine(echo=False)

    # Number of matches per field
    counts = pd.read_sql_query("select root, COUNT(root) as n from redshift_fit_v2, photometry_apcorr where phot_root = p_root AND id = p_id AND bic_diff > 5 AND mag_auto < 24 group by root;", engine)

    counts = utils.GTable.from_pandas(counts)
    so = np.argsort(counts['n'])

    sh = """
    BUCKET=grizli-v
    root=j113812m1134

    aws s3 rm --recursive s3://grizli-v2/HST/Pipeline/${root}/ --include "*"
    grism_run_single.sh ${root} --run_fine_alignment=True --extra_filters=g800l --bucket=grizli-v2 --preprocess_args.skip_single_optical_visits=True --mask_spikes=True --persistence_args.err_threshold=1
    """


def add_missing_photometry():

    # Add missing photometry
    import os
    import pandas as pd
    from grizli.aws import db as grizli_db
    from grizli.pipeline import photoz
    from grizli import utils

    engine = grizli_db.get_db_engine(echo=False)

    res = pd.read_sql_query("select distinct root from redshift_fit_v2 where root like 'j%%'", engine)['root'].tolist()
    orig_roots = pd.read_sql_query('select distinct p_root as root from photometry_apcorr', engine)['root'].tolist()

    # Missing grism fields?
    res = pd.read_sql_query("select field_root as root, field_t_g800l, field_t_g102, field_t_g141, proposal_pi from charge_fields where (field_t_g800l > 0 OR field_t_g141 > 0 OR  field_t_g102 > 0) AND log LIKE '%%inish%%';", engine)['root'].tolist()
    orig_roots = pd.read_sql_query('select distinct root from redshift_fit_v2', engine)['root'].tolist()

    # All photometry
    res = pd.read_sql_query("select field_root as root, field_t_g800l, field_t_g102, field_t_g141, proposal_pi from charge_fields where nassoc < 200 AND log LIKE '%%inish%%' AND field_root LIKE 'j%%';", engine)['root'].tolist()
    orig_roots = pd.read_sql_query('select distinct p_root as root from photometry_apcorr', engine)['root'].tolist()

    count = 0
    for root in res:
        if root not in orig_roots:
            #break
            count += 1
            print(count, root)
            os.system('aws s3 cp s3://grizli-v2/HST/Pipeline/{0}/Extractions/{0}_phot_apcorr.fits .'.format(root))
            os.system('aws s3 cp s3://grizli-v2/HST/Pipeline/{0}/Extractions/{0}_phot.fits .'.format(root))

            if not os.path.exists('{0}_phot_apcorr.fits'.format(root)):
                os.system('aws s3 cp s3://grizli-v2/HST/Pipeline/{0}/Prep/{0}_phot_apcorr.fits .'.format(root))
                os.system('aws s3 cp s3://grizli-v2/HST/Pipeline/{0}/Prep/{0}_phot.fits .'.format(root))

            if os.path.exists('{0}_phot_apcorr.fits'.format(root)):
                grizli_db.add_phot_to_db(root, delete=False, engine=engine)
            else:
                if os.path.exists('{0}_phot.fits'.format(root)):
                    # Make the apcorr file
                    utils.set_warnings()

                    total_flux = 'flux_auto'
                    try:
                        obj = photoz.eazy_photoz(root, object_only=True,
                              apply_prior=False, beta_prior=True,
                              aper_ix=1,
                              force=True,
                              get_external_photometry=False,
                              compute_residuals=False,
                              total_flux=total_flux)
                    except:
                        continue

                    grizli_db.add_phot_to_db(root, delete=False,
                                             engine=engine, nmax=500)

    # 3D-HST
    copy = """
    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/egs-mosaic_phot_apcorr.fits s3://grizli-v1/Pipeline/egs-grism-j141956p5255/Extractions/egs-grism-j141956p5255_phot_apcorr.fits --acl public-read
    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/egs-mosaic_phot.fits s3://grizli-v1/Pipeline/egs-grism-j141956p5255/Extractions/egs-grism-j141956p5255_phot.fits --acl public-read
    """
    grizli_db.run_lambda_fits('egs-grism-j141956p5255', min_status=6, zr=[0.01, 3.2])

    copy = """
    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/uds-mosaic_phot_apcorr.fits s3://grizli-v1/Pipeline/uds-grism-j021732m0512/Extractions/uds-grism-j021732m0512_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('uds-grism-j021732m0512', min_status=6, zr=[0.01, 3.2])

    # GDS
    copy = """
    aws s3 rm s3://grizli-v1/Pipeline/gds-grism-j033236m2748/Extractions/ --recursive --exclude "*" --include "gds-grism-j033236m2748_[0-9]*"
    aws s3 rm s3://grizli-v1/Pipeline/gds-g800l-j033236m2748/Extractions/ --recursive --exclude "*" --include "gds-g800l-j033236m2748_[0-9]*"

    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/gds-mosaic_phot_apcorr.fits s3://grizli-v1/Pipeline/gds-grism-j033236m2748/Extractions/gds-grism-j033236m2748_phot_apcorr.fits --acl public-read

    aws s3 cp s3://grizli-v1/Pipeline/gds-grism-j033236m2748/Extractions/gds-grism-j033236m2748_phot_apcorr.fits s3://grizli-v1/Pipeline/gds-g800l-j033236m2748/Extractions/gds-g800l-j033236m2748_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('gds-grism-j033236m2748', phot_root='gds-grism-j033236m2748', min_status=6, zr=[0.01, 3.2], extra={'bad_pa_threshold': 10, 'use_phot_obj': False})
    grizli_db.run_lambda_fits('gds-g800l-j033236m2748', phot_root='gds-grism-j033236m2748', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold': 10, 'use_phot_obj': False})

    # GDN
    copy = """
    #aws s3 rm s3://grizli-v1/Pipeline/gds-g800l-j033236m2748/Extractions/ --recursive --exclude "*" --include "gds-g800l-j033236m2748_[0-9]*"
    aws s3 rm s3://grizli-v1/Pipeline/gdn-grism-j123656p6215/Extractions/ --recursive --exclude "*" --include "gdn-grism-j123656p6215_[0-9]*"
    aws s3 rm s3://grizli-v1/Pipeline/gdn-g800l-j123656p6215/Extractions/ --recursive --exclude "*" --include "gdn-g800l-j123656p6215_[0-9]*"

    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/gdn-mosaic_phot_apcorr.fits s3://grizli-v1/Pipeline/gdn-grism-j123656p6215/Extractions/gdn-grism-j123656p6215_phot_apcorr.fits --acl public-read

    aws s3 cp s3://grizli-v1/Pipeline/gdn-grism-j123656p6215/Extractions/gdn-grism-j123656p6215_phot_apcorr.fits s3://grizli-v1/Pipeline/gdn-g800l-j123656p6215/Extractions/gdn-g800l-j123656p6215_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('gdn-grism-j123656p6215', phot_root='gdn-grism-j123656p6215', min_status=6, zr=[0.01, 3.2], extra={'bad_pa_threshold': 10, 'use_phot_obj': False})
    grizli_db.run_lambda_fits('gdn-g800l-j123656p6215', phot_root='gdn-grism-j123656p6215', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold': 10, 'use_phot_obj': False})

    # 3D-HST G800L
    copy = """
    aws s3 rm s3://grizli-v1/Pipeline/egs-g800l-j141956p5255/Extractions/ --recursive --exclude "*" --include "egs-g800l-j141956p5255_[0-9]*"

    aws s3 cp s3://grizli-v1/Pipeline/egs-grism-j141956p5255/Extractions/egs-grism-j141956p5255_phot_apcorr.fits s3://grizli-v1/Pipeline/egs-g800l-j141956p5255/Extractions/egs-g800l-j141956p5255_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('egs-g800l-j141956p5255', phot_root='egs-grism-j141956p5255', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold': 10, 'use_phot_obj': False})

    res = grizli_db.wait_on_db_update('egs-g800l-j141956p5255', dt=15, n_iter=120, engine=engine)
    res = grizli_db.wait_on_db_update('uds-g800l-j021732m0512', dt=15, n_iter=120, engine=engine)

    # UDS
    copy = """
    aws s3 rm s3://grizli-v1/Pipeline/uds-g800l-j021732m0512/Extractions/ --recursive --exclude "*" --include "uds-g800l-j021732m0512_[0-9]*"

    aws s3 cp s3://grizli-v1/Pipeline/uds-grism-j021732m0512/Extractions/uds-grism-j021732m0512_phot_apcorr.fits s3://grizli-v1/Pipeline/uds-g800l-j021732m0512/Extractions/uds-g800l-j021732m0512_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('uds-g800l-j021732m0512', phot_root='uds-grism-j021732m0512', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold': 10, 'use_phot_obj': False})

    grizli_db.run_lambda_fits('egs-g800l-j141956p5255', phot_root='egs-grism-j141956p5255', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold': 10, 'use_phot_obj': False})

    # Cosmos on oliveraws
    copy = """

    aws s3 rm s3://grizli-cosmos-v2/Pipeline/cos-grism-j100012p0210/Extractions/ --recursive --exclude "*" --include "cos-grism-j100012p0210_[0-9]*"

    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/Cosmos/cos-cnd-mosaic_phot_apcorr.fits s3://grizli-cosmos-v2/Pipeline/cos-grism-j100012p0210/Extractions/cos-grism-j100012p0210_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('cos-grism-j100012p0210', min_status=6, zr=[0.01, 3.2], mag_limits=[17, 17.1], bucket='grizli-cosmos-v2')
    os.system('sudo halt')


def set_column_formats(info, extra={}, convert_mtime=True, **kwargs):
    """
    Set predefined format strings of table columns

    Parameters
    ----------
    info : `astropy.table.Table`
        Data table, updated in place

    extra : dict
        Dictionary with extra format codes as values and column names as keys

    convert_mtime : bool
        If ``'mtime'`` column found in `info`, convert all time strings to 
        sortable ISO format with `~grizli.utils.ctime_to_iso`.

    """
    # Print formats
    formats = {}
    formats['ra'] = formats['dec'] = '.8f'
    formats['mag_auto'] = formats['delta_z'] = '.2f'
    formats['chinu'] = formats['chimin'] = formats['chimax'] = '.1f'
    formats['bic_diff'] = formats['bic_temp'] = formats['bic_spl'] = '.1f'
    formats['bic_poly'] = '.1f'
    formats['dlinesn'] = formats['bic_spl'] = '.1f'

    formats['flux_radius'] = formats['flux_radius_20'] = '.1f'
    formats['flux_radius_90'] = '.1f'

    formats['log_pdf_max'] = formats['log_risk'] = '.1f'
    formats['d4000'] = formats['d4000_e'] = '.2f'
    formats['dn4000'] = formats['dn4000_e'] = '.2f'
    formats['z_spec'] = formats['z_map'] = formats['reshift'] = '.3f'
    formats['z_spec_dr'] = '.1f'

    formats['t_g141'] = formats['t_g102'] = formats['t_g800l'] = '.0f'
    formats['zwidth1'] = formats['zw1'] = '.3f'
    formats['zwidth2'] = formats['zw2'] = '.3f'

    formats['q_z'] = '.2f'
    formats['dz'] = '.3f'

    for k in extra:
        formats[k] = extra[k]

    for c in info.colnames:
        if c in formats:
            info[c].format = formats[c]
        elif c.startswith('sn_'):
            info[c].format = '.1f'
        elif c.startswith('mag_'):
            info[c].format = '.2f'
        elif '_ujy' in c:
            info[c].format = '.2f'
        elif c.startswith('ew_'):
            info[c].format = '.1f'
        elif ('q_z' in c):
            info[c].format = '.2f'
        elif ('zw' in c) | ('z_map' in c):
            info[c].format = '.3f'
        elif ('chinu' in c):
            info[c].format = '.1f'
        elif c.startswith('bic_'):
            info[c].format = '.1f'
        elif c in ['z02', 'z16', 'z50', 'z84', 'z97']:
            info[c].format = '.3f'
        elif c[:4] in ['splf', 'sple']:
            info[c].format = '.1e'
        elif c.startswith('flux_') | c.startswith('err_'):
            info[c].format = '.1e'
    
    if convert_mtime & ('mtime' in info.colnames):
        iso_times = [utils.ctime_to_iso(m, verbose=False, strip_decimal=True) 
                     for m in info['mtime']]

        info['mtime'] = iso_times


def query_from_ds9(ds9, radius=5, engine=None, extra_cols=['mag_auto', 'z_map', 'bic_diff', 't_g800l', 't_g102', 't_g141'], extra_query='', table_root='/tmp/ds9_query'):
    """
    Make a table by running a query for objects based on a DS9 pan position
    """
    from grizli import utils, prep

    if engine is None:
        engine = get_db_engine(echo=False)

    ra, dec = np.cast[float](ds9.get('pan fk5').split())
    dd = radius/3600.
    dr = dd/np.cos(dec/180*np.pi)

    min_cols = ['root', 'id', 'status', 'ra', 'dec']
    colstr = ','.join(min_cols + extra_cols)

    q = from_sql(f'select {colstr} '
                      f'from redshift_fit_v2 natural join photometry_apcorr '
                      f'where ra > {ra-dr} AND ra < {ra+dr}'
                      f' AND dec > {dec-dd} and dec < {dec+dd}' + extra_query,
                      engine)

    tt = utils.GTable()
    tt['ra'] = [ra]
    tt['dec'] = [dec]

    _idx, _dr = tt.match_to_catalog_sky(q)
    q['_dr'] = _dr
    q['_dr'].format = '.2f'
    so = np.argsort(q['_dr'])

    make_html_table(sync=None, res=q[so], use_json=False, table_root=table_root, sort_column=('_dr', 1))

    comment = [f'{id}' for id in q['id'][so]]

    prep.table_to_regions(q[so], table_root+'.reg', comment=comment)

    return q[so]


def make_html_table(engine=None, columns=['root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'log_pdf_max', 'd4000', 'd4000_e'], where="AND status >= 5 AND root='j163852p4039'", tables=[], table_root='query', sync='s3://grizli-v2/tables/', png_ext=['R30', 'stack', 'full', 'line'], sort_column=('bic_diff', -1), fit_table='redshift_fit_v2', verbose=True, get_sql=False, res=None, show_hist=False, extra_formats={}, use_json=True, use_join=False):
    """
    """
    import time

    import numpy as np
    import matplotlib.pyplot as plt

    import pandas as pd
    from grizli import utils
    from grizli.aws import db as grizli_db

    if engine is None:
        engine = get_db_engine(echo=False)

    if len(tables) > 0:
        extra_tables = ','+','.join(tables)
    else:
        extra_tables = ''

    if use_join:
        query = "SELECT {0} FROM {1} NATURAL JOIN photometry_apcorr WHERE {2};".format(','.join(columns), fit_table, where)
        query = query.replace('WHERE AND', 'AND')
    else:
        query = "SELECT {0} FROM photometry_apcorr, {3}{1} WHERE phot_root = p_root AND id = p_id {2};".format(','.join(columns), extra_tables, where, fit_table)

    if get_sql:
        return query

    if res is not None:
        info = res
    else:
        res = pd.read_sql_query(query, engine)
        info = utils.GTable.from_pandas(res)

        if verbose:
            print('Query: {0}\n Results N={1}'.format(query, len(res)))
        if 'cdf_z' in info.colnames:
            info.remove_column('cdf_z')

        for c in info.colnames:
            if c.startswith('p_'):
                try:
                    info.rename_column(c, c[2:])
                except:
                    pass

    all_columns = info.colnames.copy()

    if 'idx' not in info.colnames:
        idx = ['<a href="http://vizier.u-strasbg.fr/viz-bin/VizieR?-c={0:.6f}+{1:.6f}&-c.rs=2">#{2:05d}</a>'.format(info['ra'][i], info['dec'][i], info['id'][i]) for i in range(len(info))]
        info['idx'] = idx

    all_columns.insert(0, 'idx')
    all_columns.pop(all_columns.index('id'))

    set_column_formats(info, extra=extra_formats)

    print('Sort: ', sort_column, sort_column[0] in all_columns)
    if sort_column[0] in all_columns:
        scol = info[sort_column[0]]
        if hasattr(scol, 'mask'):
            sdata = scol.filled(fill_value=-np.inf).data
        else:
            sdata = scol

        so = np.argsort(sdata)[::sort_column[1]]
        #info = info[so[::sort_column[1]]]

    # PNG columns
    AWS = 'https://s3.amazonaws.com/grizli-v2/HST/Pipeline'
    #bucket = ['grizli-cosmos-v2' if r.startswith('cos-') else 'grizli-v1' for r in info['root']]
    bucket = 'grizli-v2'
    
    for ext in png_ext:
        if ext == 'thumb':
            subdir = 'Thumbnails'
            print(ext, subdir)
        elif ext == 'rgb':
            subdir = 'Thumbnails'
        else:
            subdir = 'Extractions'

        if 'png_{0}'.format(ext) not in info.colnames:
            png = ['{0}_{1:05d}.{2}.png'.format(root, id, ext) for root, id in zip(info['root'], info['id'])]

            if ext == 'rgb':
                js = '<a href={0}/{2}><img src={0}/{1} onmouseover="this.src = this.src.replace(\'rgb.pn\', \'seg.pn\')" onmouseout="this.src = this.src.replace(\'seg.pn\', \'rgb.pn\')" height=200></a>'

                paths = ['{0}/{1}/{2}'.format(AWS.replace('grizli-v2', buck),
                                              root, subdir)
                         for buck, root in zip(bucket, info['root'])]

                png_url = [js.format(path, p,
                                     p.replace('.rgb.png', '.thumb.png'))
                           for path, p in zip(paths, png)]

                info['png_{0}'.format('rgb')] = png_url

            else:
                info['png_{0}'.format(ext)] = ['<a href="{0}/{1}/{2}/{3}"><img src={0}/{1}/{2}/{3} height=200></a>'.format(AWS.replace('grizli-v2', buck), root, subdir, p) for buck, root, p in zip(bucket, info['root'], png)]

            all_columns.append('png_{0}'.format(ext))

    sortable = []
    for c in all_columns:
        if not hasattr(info[c][0], 'upper'):
            sortable.append(c)

    info[all_columns][so].write_sortable_html('{0}.html'.format(table_root), replace_braces=True, localhost=False, max_lines=1e5, table_id=None, table_class='display compact', css=None, filter_columns=sortable, buttons=['csv'], toggle=True, use_json=use_json)

    if show_hist:
        from matplotlib.ticker import FixedLocator, AutoLocator, MaxNLocator
        xti = xt = np.arange(0, 3.6, 0.5)
        loc = np.arange(0, 3.6, 0.1)
        bins = utils.log_zgrid([0.03, 3.5], 0.01)
        fig = plt.figure(figsize=[8, 4])
        ax = fig.add_subplot(111)
        ax.hist(np.log(1+res['z_map']), bins=np.log(1+bins), color='k',
                alpha=0.2, label=table_root, normed=False)

        clip = res['bic_diff'].values > 30

        ax.hist(np.log(1+res['z_map'].values[clip]), bins=np.log(1+bins),
                color='r', alpha=0.3, normed=False)

        xts = ax.set_xticks(np.log(1+xt))
        xtl = ax.set_xticklabels(xti)
        ax.xaxis.set_minor_locator(FixedLocator(np.log(1+loc)))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('z_map')
        ax.set_ylabel(r'$N$')

        # Label to show line mis-id
        dz_wrong = (6563.-5007)/5007
        ax.plot(np.arange(5)*dz_wrong, np.ones(5)*ax.get_ylim()[1], marker='.', markerfacecolor='w', markeredgecolor='w', color='r', markersize=10)
        ax.set_xlim(0, np.log(1+3.7))

        ax.grid()
        ax.legend(loc='upper right')

        fig.tight_layout(pad=0.1)
        fig.text(1-0.02, 0.02, utils.nowtime(), ha='right', va='bottom', 
                 transform=fig.transFigure, fontsize=5)

        fig.savefig('{0}_zhist.png'.format(table_root))

    if sync:
        os.system('aws s3 sync ./ {0} --exclude "*" --include "{1}.html" --include "{1}.json" --include "{1}_zhist.png" --acl public-read'.format(sync, table_root))

    return res


def get_exposure_info():
    """
    Get exposure information from the MAST databases
    """
    import mastquery.query

    master = 'grizli-v1-19.12.04'
    master = 'grizli-v1-19.12.05'
    master = 'grizli-v1-20.10.12'

    tab = utils.read_catalog('{0}_visits.fits'.format(master))
    all_visits = np.load('{0}_visits.npy'.format(master), allow_pickle=True)[0]

    all_files = []
    for v in all_visits:
        all_files.extend(v['files'])

    prog = [f[1:4] for f in all_files]
    _res = np.unique(np.array(prog), return_counts=True)
    t = utils.GTable()
    t['prog'] = _res[0]
    t['count'] = _res[1]
    so = np.argsort(t['count'])
    t = t[so[::-1]]
    for pr in t['prog']:
        if os.path.exists('{0}_query.fits'.format(pr)):
            #print('Skip ', pr)
            continue

        print(pr)

        try:
            _q = mastquery.query.run_query(obs_id='[ij]{0}*'.format(pr))
            _p = mastquery.query.get_products_table(_q)
        except:
            continue

        _q.write('{0}_query.fits'.format(pr))
        _p.write('{0}_prod.fits'.format(pr))

    # Send to AWS
    from grizli.aws import db
    import pandas as pd
    from astropy.table import Table

    engine = db.get_db_engine()

    files = glob.glob('*query.fits')
    files.sort()

    cols = ['obs_id', 'target', 'target_ra', 'target_dec', 't_min', 't_max', 'exptime', 'wavelength_region', 'filter', 'em_min', 'em_max', 'target_classification', 'obs_title', 't_obs_release', 'instrument_name', 'proposal_pi', 'proposal_id', 'proposal_type', 'sregion', 'dataRights', 'mtFlag', 'obsid', 'objID', 'visit']

    for i, file in enumerate(files):
        print(file)
        _q = Table.read(file, character_as_bytes=False)

        _q['proposal_id'] = np.cast[np.int16](_q['proposal_id'])
        _q['obsid'] = np.cast[np.int64](_q['obsid'])
        _q['objID'] = np.cast[np.int64](_q['objID'])
        _q.rename_column('ra','target_ra')
        _q.rename_column('dec','target_dec')
        _q.rename_column('footprint', 'sregion')
        
        df = _q[cols].to_pandas()
        df.to_sql('mast_query', engine, index=False, if_exists='append', method='multi')

    files = glob.glob('*_prod.fits')
    files.sort()

    cols = ['obsid', 'dataset']

    for i, file in enumerate(files):
        print(i, file)
        _p = Table.read(file, character_as_bytes=False)

        _p['obsid'] = np.cast[np.int64](_p['obsid'])
        _p['dataset'] = [d[:-1] for d in _p['observation_id']]

        df = _p[cols].to_pandas()
        df.to_sql('mast_products', engine, index=False, if_exists='append', method='multi')

    ##########
    # Exposure log

    # Initialize, adding an array column manually for the footprints
    v = all_visits[0]
    N = len(v['files'])
    fps = [np.array(fp.convex_hull.boundary.xy)[:, :-1].tolist() for fp in v['footprints']]
    df = pd.DataFrame()
    df['file'] = [f.split('_')[0] for f in v['files']]
    df['dataset'] = [f.split('_')[0][:-1] for f in v['files']]
    df['extension'] = [f.split('_')[1][:3] for f in v['files']]
    df['filter'] = v['filter']
    df['parent'] = v['parent']
    df['awspath'] = v['awspath']
    df['product'] = v['product']
    df['filter'] = v['product'].split('-')[-1]

    df['ra'] = [fp.centroid.xy[0][0] for fp in v['footprints']]
    df['dec'] = [fp.centroid.xy[1][0] for fp in v['footprints']]
    df['area'] = [fp.area*np.cos(df['dec'][i]/180*np.pi)*3600 for i, fp in enumerate(v['footprints'])]

    # Make table
    engine.execute('drop table exposure_log;')
    df.to_sql('exposure_log', engine, index=False, if_exists='append', method='multi')
    engine.execute('alter table exposure_log add column footprint float [];')
    engine.execute('delete from exposure_log where True;')
    
    engine.execute('ALTER TABLE exposure_log ADD COLUMN mdrizsky float;')
    engine.execute('ALTER TABLE exposure_log ADD COLUMN exptime float;')
    engine.execute('ALTER TABLE exposure_log ADD COLUMN expstart float;')
    engine.execute('ALTER TABLE exposure_log ADD COLUMN ndq int;')
    engine.execute('ALTER TABLE exposure_log ADD COLUMN expflag VARCHAR;')
    engine.execute('ALTER TABLE exposure_log ADD COLUMN sunangle float;')

    engine.execute('ALTER TABLE exposure_log ADD COLUMN gsky101 real;')
    engine.execute('ALTER TABLE exposure_log ADD COLUMN gsky102 real;')
    engine.execute('ALTER TABLE exposure_log ADD COLUMN gsky103 real;')
    engine.execute('ALTER TABLE exposure_log ADD COLUMN persnpix integer;')
    engine.execute('ALTER TABLE exposure_log ADD COLUMN perslevl real;')
    
    _exp = db.from_sql("select distinct(file) from exposure_log", engine)
    db_files = np.unique(_exp['file'])
    charge = db.from_sql("select * from charge_fields", engine)
    
    SKIP = 1000
    df0 = None
    
    for i, v in enumerate(all_visits):
        _count = np.sum([f.split('_')[0] in db_files for f in v['files']])
        
        if _count == len(v['files']):
            continue
    
        if v['parent'] not in charge['field_root']:
            print('Warning: {0} not in charge["field_root"]'.format(v['parent']))
            continue
            
        print(i, v['parent'], v['product'], _count, len(v['files']))

        N = len(v['files'])

        fps = [np.array(fp.convex_hull.boundary.xy)[:, :-1].tolist() for fp in v['footprints']]

        df = pd.DataFrame()
        df['file'] = [f.split('_')[0] for f in v['files']]
        df['dataset'] = [f.split('_')[0][:-1] for f in v['files']]
        df['extension'] = [f.split('_')[1][:3] for f in v['files']]
        df['filter'] = v['filter']
        df['parent'] = v['parent']
        df['awspath'] = v['awspath']
        df['product'] = v['product']
        df['filter'] = v['product'].split('-')[-1]

        df['ra'] = [fp.centroid.xy[0][0] for fp in v['footprints']]
        df['dec'] = [fp.centroid.xy[1][0] for fp in v['footprints']]
        df['area'] = [fp.area*np.cos(df['dec'][i]/180*np.pi)*3600 for i, fp in enumerate(v['footprints'])]
        df['footprint'] = fps
        
        if df0 is None:
            df0 = df        
        else:
            df0 = df0.append(df)
         
        if len(df0) > SKIP:
            # Send to DB and reset append table
            print('>>> to DB >>> ({0}, {1})'.format(i, len(df0)))
            df0.to_sql('exposure_log', engine, index=False, if_exists='append', method='multi')
            df0 = df[:0]


def update_all_exposure_log():
    """
    Run all
    """
    import glob
    import numpy as np
    from grizli.aws import db
    
    from importlib import reload
    reload(db)
    
    config = db.get_connection_info(config_file='/home/ec2-user/db_readonly.yml')
    engine = db.get_db_engine(config=config)
    
    # DASH
    #_files = db.from_sql("SELECT file from exposure_log WHERE mdrizsky is null AND file like 'icxe%%'", engine)
    
    # COSMOS F160W
    _files = db.from_sql("SELECT file, filter, awspath from exposure_log WHERE mdrizsky is null AND awspath like 'grizli-cosmos%%' AND filter like 'f160w'", engine)
    
    # COSMOS F814W
    #_files = db.from_sql("SELECT file, filter, awspath from exposure_log WHERE mdrizsky is null AND ABS(ra-150.1) < 0.6 AND ABS(dec-2.2) < 0.6 AND filter like 'f814w'", engine)
    #_files = db.from_sql("SELECT file, filter, awspath from exposure_log WHERE mdrizsky is null AND ABS(ra-150.1) < 0.6 AND ABS(dec-2.2) < 0.6", engine)
    
    ### COSMOS grism
    #_files = db.from_sql("SELECT file, filter, awspath from exposure_log WHERE mdrizsky is null AND ABS(ra-150.1) < 0.6 AND ABS(dec-2.2) < 0.6 AND filter like 'g1%%'", engine)
    ### All grism
    #_files = db.from_sql("SELECT file, filter, awspath, mdrizsky from exposure_log WHERE mdrizsky is null AND filter like 'g1%%'", engine)

    #db.update_exposure_log({'file':_files['file'][0], 'engine':engine, 'skip':False}, {})
    
    # All IR
    # _files = db.from_sql("SELECT file, filter from exposure_log WHERE mdrizsky is null AND filter like 'f0%%'", engine)
    # 
    _files = db.from_sql("SELECT file, filter, parent from exposure_log WHERE mdrizsky is null AND filter like 'f%%'", engine)
    _files = db.from_sql("SELECT file, filter, parent, product from exposure_log WHERE mdrizsky is null AND gsky101 is null", engine)

    # Skip 3DHST
    keep = _files['parent'] != 'xx'
    for p in ['j123656p6215','j141952p5255','j033236m2748','j021740m0512']:
        keep &= _files['parent'] != p
        
    #_files = db.from_sql("SELECT file, filter from exposure_log WHERE mdrizsky is null AND awspath like 'grizli-cosmos%%' AND filter like 'f814w' LIMIT 10", engine)
    
    # latest cosmos
    _files = db.from_sql("SELECT file, filter, awspath from exposure_log WHERE mdrizsky is null AND awspath like 'cosmos-dash%%' AND filter like 'f160w'", engine)
    
    N = len(_files)
    idx = np.argsort(np.random.normal(size=N))
    for i, file in enumerate(_files['file'][idx]):
        print(f'\n {i+1} / {N}\n')
        _ = glob.glob(f'{file}*')
        if len(_) == 0:
            db.update_exposure_log({'file':file, 'engine':engine}, {})


def update_exposure_log(event, context):
    """
    Get exposure info from FITS file and put in database
    
    Recognized `event` keywords (default):
        
        'file' : file rootname in exposure_log, *required*
        'keywords' : list of keywords to take from the Primary header
                     (['EXPFLAG','EXPTIME','EXPSTART','SUNANGLE'])
        'dump_dq' : generate a compact DQ file and upload to S3 (True)
        'remove': Remove the downloaded exposure file (True)
        'skip': Don't do anything if 'mdrizsky' populated in database
        
    """
    import os
    import boto3
    import astropy.io.fits as pyfits
    from grizli import utils
    
    if 'file' not in event:
        print("'file' keyword not found in `event`")
        return False

    if 'keywords' in event:
        keywords = event['keywords']
    else:
        keywords = ['EXPFLAG','EXPTIME','EXPSTART','SUNANGLE']
        keywords += ['GSKY101', 'GSKY102', 'GSKY103', 'PERSNPIX', 'PERSLEVL']
        
    kwvals = {}
    
    if 'engine' in event:
        engine = event['engine']
    else:
        engine = get_db_engine(echo=False)

    _q = from_sql("SELECT * from exposure_log where file LIKE '{0}'".format(event['file']), engine)
    if len(_q) == 0:
        print('File {0} not found in `exposure_log`'.format(event['file']))
        return False

    if 'skip' in event:
        skip = event['skip']
    else:
        skip = True

    if (not hasattr(_q['mdrizsky'], 'mask')) & skip:
        print('Info for {0} found in `exposure_log`'.format(event['file']))
        return True
    #
    local_file = '{0}_{1}.fits'.format(_q['file'][0], _q['extension'][0])
    
    s3 = boto3.resource('s3')

    bucket = _q['awspath'][0].split('/')[0]
    bkt = s3.Bucket(bucket)

    awsfile = '/'.join(_q['awspath'][0].split('/')[1:]).strip('/')
    awsfile += '/'+local_file
                       
    print(f'{bucket}:{awsfile} > {local_file}')

    if not os.path.exists(local_file):
        try:
            bkt.download_file(awsfile, local_file, 
                      ExtraArgs={"RequestPayer": "requester"})
        except:
            print(f'Failed to download s3://{bucket}/{awsfile}')
            
            # Try other bucket path
            if 'Exposures' in awsfile:
                bucket = 'grizli-v2'
                bkt = s3.Bucket(bucket)
                awsfile = 'Pipeline/{0}/Prep/'.format(_q['parent'][0])
                awsfile += local_file

                try:
                    bkt.download_file(awsfile, local_file, 
                              ExtraArgs={"RequestPayer": "requester"})
                except:
                    print(f'Failed to download s3://{bucket}/{awsfile}')
                    return False

                kwvals['awspath'] = f'{bucket}/{os.path.dirname(awsfile)}'

            else:
                return False

    ######### Update exposure_log table
    im = pyfits.open(local_file)
    kwvals['ndq'] = (im['DQ',1].data == 0).sum()

    if 'MDRIZSKY' in im['SCI',1].header:
        kwvals['mdrizsky'] = im['SCI',1].header['MDRIZSKY']

    for k in keywords:
        if (k in im[0].header) & (k.lower() in _q.colnames):
            kwvals[k.lower()] = im[0].header[k]

    set_keys = []
    for k in kwvals:
        if isinstance(kwvals[k], str):
            _set = 'x = \'{x}\''
        else:
            _set = 'x = {x}'
        set_keys.append(_set.replace('x', k))
        
    sqlstr = ('UPDATE exposure_log SET ' + ', '.join(set_keys) + 
              " WHERE file LIKE '{0}'".format(event['file']))

    print(sqlstr.format(**kwvals))
    engine.execute(sqlstr.format(**kwvals))
    im.close()

    ######### Compact DQ file
    if 'dump_dq' in event:
        dump_dq = event['dump_dq']
    else:
        dump_dq = True
        
    if dump_dq:
        utils.dump_flt_dq(local_file)
        repl = ('.fits', '.dq.fits.gz')
        print(f'{local_file} > {bucket}:{awsfile}'.replace(*repl))

        try:
            bkt.upload_file(local_file.replace(*repl), 
                            awsfile.replace(*repl), 
                            ExtraArgs={'ACL':'public-read'})
        except:
            print(f'Failed to upload s3://{bucket}:{awsfile}'.replace(*repl))

    remove = True
    if 'remove' in event:
        remove = event['remove']    

    if remove:
        print('Remove '+local_file)
        if os.path.exists(local_file):
            os.remove(local_file)
        if dump_dq:
            print('Remove '+local_file.replace(*repl))
            if os.path.exists(local_file.replace(*repl)):
                os.remove(local_file.replace(*repl))

    return kwvals

def run_shrink_ramps():
    from grizli.aws import db
    
    _q = db.from_sql("select file, awspath, parent from exposure_log where extension LIKE 'flt' AND parent LIKE 'j002836m3311' limit 5", engine)
    for i, (file, awspath, parent) in enumerate(zip(_q['file'], _q['awspath'], _q['parent'])):
        shrink_ramp_file(file, awspath, parent, engine=engine, MAX_SIZE=2*1024**2, convert_args='-scale 35% -quality 90', remove=True)
        
def shrink_ramp_file(file, awspath, parent, engine=None, MAX_SIZE=2*1024**2, convert_args='-scale 35% -quality 90', remove=True):
    """
    Make ramp.png files smaller with ImageMagick
    """
    import os
    import subprocess
    import shutil
    
    import boto3
    import astropy.io.fits as pyfits
    from grizli import utils
    
    if engine is None:
        engine = get_db_engine(echo=False)

    local_file = '{0}_ramp.png'.format(file)
    
    s3 = boto3.resource('s3')

    bucket = awspath.split('/')[0]
    bkt = s3.Bucket(bucket)

    awsfile = '/'.join(awspath.split('/')[1:])
    awsfile += '/'+local_file
    awsfile = awsfile.replace('/Prep','/RAW')
                
    print(f'{bucket}/{awsfile} > {local_file}')

    if not os.path.exists(local_file):
        try:
            bkt.download_file(awsfile, local_file, 
                      ExtraArgs={"RequestPayer": "requester"})
        except:
            print(f'Failed to download s3://{bucket}/{awsfile}')
            
            # Try other bucket path
            if 'Exposures' in awsfile:
                bucket = 'grizli-v2'
                bkt = s3.Bucket(bucket)
                awsfile = 'Pipeline/{0}/RAW/'.format(parent)
                awsfile += local_file

                try:
                    bkt.download_file(awsfile, local_file, 
                              ExtraArgs={"RequestPayer": "requester"})
                except:
                    print(f'Failed to download s3://{bucket}/{awsfile}')
                    return False

            else:
                return False

    print(f'{local_file:>25} {os.stat(local_file).st_size/1024**2:.2f}') 
    bw_file = local_file.replace('.png', '.sm.png')
    
    if os.stat(local_file).st_size > MAX_SIZE:
        
        subprocess.call(f"convert {convert_args}  {local_file} {bw_file}",
                        shell=True)
        
        print(f'{bw_file:>25} {os.stat(bw_file).st_size/1024**2:.2f}') 
        
        try:
            bkt.upload_file(bw_file, awsfile, ExtraArgs={'ACL':'public-read'})
        except:
            print(f'Failed to upload s3://{bucket}/{awsfile}')
    else:
        print('skip')
        
    if remove:
        print('Remove '+local_file)
        if os.path.exists(local_file):
            os.remove(local_file)
        
        if os.path.exists(bw_file):
            os.remove(bw_file)


def get_exposures_at_position(ra, dec, engine, dr=10):
    cosdec = np.cos(dec/180*np.pi)
    res = from_sql('select * from exposure_log where (ABS(ra - {0}) < {1}) AND (ABS(dec-{2}) < {3})'.format(ra, dr/cosdec, dec, dr), engine)
    return res


def add_irac_table():

    from scipy.spatial import ConvexHull

    os.chdir('/Users/gbrammer/Research/HST/CHArGE/FieldsSummary')
    files = glob.glob('*ipac.fits')
    files.sort()

    bands = ['IRAC 3.6um', 'IRAC 4.5um', 'IRAC 5.8um', 'IRAC 8.0um', 'MIPS 24um']
    bkey = {}
    for b in bands:
        key = b.replace(' ', '').replace('.', '')[:-2].lower()
        bkey[key] = b

    N = 0
    data = {'field_root': []}
    aor_data = {'field_root': [], 'reqkey': []}
    for k in bkey:
        data['exp_'+k] = []
        data['n_'+k] = []
        data['fp_'+k] = []

    for i, file in enumerate(files):
        tab = utils.read_catalog(file)
        field = file.split('_ipac')[0]
        if 'x' in tab.colnames:
            data['field_root'].append(field)
            for k in bkey:
                data['exp_'+k].append(0)
                data['n_'+k].append(0)
                data['fp_'+k].append([])

            continue

        N += len(tab)
        print(i, file, N)

        data['field_root'].append(field)
        for k in bkey:
            sel = tab['with_hst'] & (tab['wavelength'] == bkey[k])
            data['exp_'+k].append(tab['exposuretime'][sel].sum()/3600)
            data['n_'+k].append(sel.sum())

            if sel.sum() == 0:
                data['fp_'+k].append([])
                continue

            r, d = [], []
            for j in range(4):
                r.extend(tab['ra{0}'.format(j+1)][sel].data)
                d.extend(tab['dec{0}'.format(j+1)][sel].data)

            pts = np.array([r, d]).T
            vert = ConvexHull(pts).vertices
            fp = pts[vert, :]
            data['fp_'+k].append(fp.T.tolist())

        aors = np.unique(tab['reqkey'])
        aor_data['field_root'].extend([field]*len(aors))
        aor_data['reqkey'].extend(list(aors))

    #
    import pandas as pd
    df = pd.DataFrame(aor_data)
    df.to_sql('spitzer_aors', engine, index=False, if_exists='append', method='multi')

    df = pd.DataFrame(data)

    # First row to initialize table
    first = df[0:1]
    for k in bkey:
        first.pop('fp_'+k)

    engine.execute('drop table spitzer_log;')
    first.to_sql('spitzer_log', engine, index=False, if_exists='append', method='multi')
    for k in bkey:
        cmd = 'alter table spitzer_log add column fp_{0} float [];'.format(k)
        engine.execute(cmd)

    engine.execute('delete from spitzer_log where True;')
    df.to_sql('spitzer_log', engine, index=False, if_exists='append', method='multi')


def show_all_fields():
    from grizli.aws import db as grizli_db
    import matplotlib.pyplot as plt
    
    plt.ioff()
    res = pd.read_sql_query("select distinct root from redshift_fit_v2 order by root;", engine)
    roots = res['root'].tolist()

    for root in roots:
        print('\n\n', root, '\n\n')
        if os.path.exists('{0}_zhist.png'.format(root)):
            continue

        try:
            res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root', 'status', 'id', 'p_ra', 'p_dec', 'mag_auto', 'flux_radius', 't_g800l', 't_g102', 't_g141', 'z_spec', 'z_map', 'bic_diff', 'chinu', 'zwidth1/(1+z_map) as zw1', 'dlinesn', 'q_z'], where="AND status > 4 AND root = '{0}'".format(root), table_root=root, sync='s3://grizli-v2/HST/Pipeline/{0}/Extractions/'.format(root), png_ext=['R30', 'stack', 'full', 'line'], show_hist=True)
        except:
            continue

        os.system('aws s3 cp s3://grizli-v2/HST/Pipeline/{0}/Extractions/{0}_zhist.png s3://grizli-v2/tables/'.format(root))
