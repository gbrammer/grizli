"""
Interact with the grizli AWS database
"""
import os

FLAGS = {'init_lambda':1,
         'start_beams':2,
         'done_beams': 3,
         'no_run_fit': 4,
         'start_redshift_fit': 5, 
         'fit_complete': 6}

COLUMNS = ['root', 'id', 'status', 'ra', 'dec', 'ninput', 'redshift', 't_g102', 'n_g102', 'p_g102', 't_g141', 'n_g141', 'p_g141', 't_g800l', 'n_g800l', 'p_g800l', 'numlines', 'haslines', 'chi2poly', 'chi2spl', 'splf01', 'sple01', 'splf02', 'sple02', 'splf03', 'sple03', 'splf04', 'sple04', 'huberdel', 'st_df', 'st_loc', 'st_scl', 'dof', 'chimin', 'chimax', 'bic_poly', 'bic_spl', 'bic_temp', 'z02', 'z16', 'z50', 'z84', 'z97', 'zwidth1', 'zwidth2', 'z_map', 'zrmin', 'zrmax', 'z_risk', 'min_risk', 'd4000', 'd4000_e', 'dn4000', 'dn4000_e', 'dlineid', 'dlinesn', 'flux_pab', 'err_pab', 'ew50_pab', 'ewhw_pab', 'flux_hei_1083', 'err_hei_1083', 'ew50_hei_1083', 'ewhw_hei_1083', 'flux_siii', 'err_siii', 'ew50_siii', 'ewhw_siii', 'flux_oii_7325', 'err_oii_7325', 'ew50_oii_7325', 'ewhw_oii_7325', 'flux_ariii_7138', 'err_ariii_7138', 'ew50_ariii_7138', 'ewhw_ariii_7138', 'flux_sii', 'err_sii', 'ew50_sii', 'ewhw_sii', 'flux_ha', 'err_ha', 'ew50_ha', 'ewhw_ha', 'flux_oi_6302', 'err_oi_6302', 'ew50_oi_6302', 'ewhw_oi_6302', 'flux_hei_5877', 'err_hei_5877', 'ew50_hei_5877', 'ewhw_hei_5877', 'flux_oiii', 'err_oiii', 'ew50_oiii', 'ewhw_oiii', 'flux_hb', 'err_hb', 'ew50_hb', 'ewhw_hb', 'flux_oiii_4363', 'err_oiii_4363', 'ew50_oiii_4363', 'ewhw_oiii_4363', 'flux_hg', 'err_hg', 'ew50_hg', 'ewhw_hg', 'flux_hd', 'err_hd', 'ew50_hd', 'ewhw_hd', 'flux_h7', 'err_h7', 'ew50_h7', 'ewhw_h7', 'flux_h8', 'err_h8', 'ew50_h8', 'ewhw_h8', 'flux_h9', 'err_h9', 'ew50_h9', 'ewhw_h9', 'flux_h10', 'err_h10', 'ew50_h10', 'ewhw_h10', 'flux_neiii_3867', 'err_neiii_3867', 'ew50_neiii_3867', 'ewhw_neiii_3867', 'flux_oii', 'err_oii', 'ew50_oii', 'ewhw_oii', 'flux_nevi_3426', 'err_nevi_3426', 'ew50_nevi_3426', 'ewhw_nevi_3426', 'flux_nev_3346', 'err_nev_3346', 'ew50_nev_3346', 'ewhw_nev_3346', 'flux_mgii', 'err_mgii', 'ew50_mgii', 'ewhw_mgii', 'flux_civ_1549', 'err_civ_1549', 'ew50_civ_1549', 'ewhw_civ_1549', 'flux_ciii_1908', 'err_ciii_1908', 'ew50_ciii_1908', 'ewhw_ciii_1908', 'flux_oiii_1663', 'err_oiii_1663', 'ew50_oiii_1663', 'ewhw_oiii_1663', 'flux_heii_1640', 'err_heii_1640', 'ew50_heii_1640', 'ewhw_heii_1640', 'flux_niii_1750', 'err_niii_1750', 'ew50_niii_1750', 'ewhw_niii_1750', 'flux_niv_1487', 'err_niv_1487', 'ew50_niv_1487', 'ewhw_niv_1487', 'flux_nv_1240', 'err_nv_1240', 'ew50_nv_1240', 'ewhw_nv_1240', 'flux_lya', 'err_lya', 'ew50_lya', 'ewhw_lya', 'pdf_max', 'cdf_z', 'sn_pab', 'sn_hei_1083', 'sn_siii', 'sn_oii_7325', 'sn_ariii_7138', 'sn_sii', 'sn_ha', 'sn_oi_6302', 'sn_hei_5877', 'sn_oiii', 'sn_hb', 'sn_oiii_4363', 'sn_hg', 'sn_hd', 'sn_h7', 'sn_h8', 'sn_h9', 'sn_h10', 'sn_neiii_3867', 'sn_oii', 'sn_nevi_3426', 'sn_nev_3346', 'sn_mgii', 'sn_civ_1549', 'sn_ciii_1908', 'sn_oiii_1663', 'sn_heii_1640', 'sn_niii_1750', 'sn_niv_1487', 'sn_nv_1240', 'sn_lya', 'chinu', 'bic_diff', 'log_risk', 'log_pdf_max', 'zq', 'mtime', 'vel_bl', 'vel_nl', 'vel_z', 'vel_nfev', 'vel_flag', 'grizli_version']
         
def get_connection_info(config_file=None):
    """
    Read the database connection info
    """
    import yaml
    
    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__),
                                   '../data/db.yml')
        
    fp = open(config_file)
    try:
        db_info = yaml.load(fp, Loader=yaml.FullLoader)
    except:
        db_info = yaml.load(fp)
        
    fp.close()
    
    return db_info

def get_db_engine(config=None, echo=False):
    """
    Generate an SQLAlchemy engine for the grizli database
    """
    from sqlalchemy import create_engine
    if config is None:
        config = get_connection_info()
    
    db_string = "postgresql://{0}:{1}@{2}:{3}/{4}".format(config['username'], config['password'], config['hostname'], config['port'], config['database'])
    engine = create_engine(db_string, echo=echo)
    return engine
    
def get_redshift_fit_status(root, id, engine=None):
    """
    Get status value from the database for root_id object
    """
    import pandas as pd
    
    if engine is None:
        engine = get_db_engine(echo=False)
    
    res = pd.read_sql_query("SELECT status FROM redshift_fit WHERE (root = '{0}' AND id = {1})".format(root, id), engine)
    
    if len(res) == 0:
        return -1
    else:
        return res['status'][0]

def update_redshift_fit_status(root, id, status=0, engine=None, verbose=True):
    """
    Set the status flag in the table
    """
    import time
    
    import pandas as pd
    from astropy.table import Table
    
    if engine is None:
        engine = get_db_engine(echo=False)
    
    old_status = get_redshift_fit_status(root, id, engine=engine)
    if old_status < 0:
        # Need to add an empty row
        tab = Table()
        tab['root'] = [root]
        tab['id'] = [id]
        tab['status'] = [status]
        tab['mtime'] = [time.ctime()]
        
        row_df = tab.to_pandas()
        
        add_redshift_fit_row(row_df, engine=engine, verbose=verbose)
        
    else:
        sqlstr = """UPDATE redshift_fit
            SET status = {0}
            WHERE (root = '{1}' AND id = {2});""".format(status, root, id)
        
        if verbose:
            print('Update status for {0} {1}: {2} -> {3}'.format(root, id, old_status, status))
            
        engine.execute(sqlstr)
    
def get_row_data(rowfile='gds-g800l-j033236m2748_21181.row.fits', status_flag=FLAGS['fit_complete']):
    """
    Convert table from a row file to a pandas DataFrame
    """
    import pandas as pd
    from astropy.table import Table
    
    if isinstance(rowfile, str):
        tab = Table.read(rowfile, character_as_bytes=False)
    else:
        tab = rowfile
        
    cdf_z = tab['cdf_z'].data
    tab.remove_column('cdf_z')
    tab['status'] = status_flag
    remove_cols = []
    for c in tab.colnames:
        if '-' in c:
            tab.rename_column(c, c.replace('-','_'))

    for c in tab.colnames:
        tab.rename_column(c, c.lower())
    
    # Remove columns not in the database
    remove_cols = []
    for c in tab.colnames:
        if c not in COLUMNS:
            remove_cols.append(c)
        
    if len(remove_cols) > 0:
        tab.remove_columns(remove_cols)        

    row_df = tab.to_pandas()
    row_df['cdf_z'] = cdf_z.tolist()
    return row_df

def delete_redshift_fit_row(root, id, engine=None):
    """
    Delete a row from the redshift fit table
    """
    if engine is None:
        engine = get_db_engine(echo=False)

    res = engine.execute("DELETE from redshift_fit WHERE (root = '{0}' AND id = {1})".format(root, id))
        
def add_redshift_fit_row(row_df, engine=None, verbose=True):
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
    status = get_redshift_fit_status(root, id, engine=engine)
    # Delete the old row?
    if status >= 0:
        print('Delete and update row for root={0}, id={1}'.format(root, id))
        delete_redshift_fit_row(root, id, engine=engine)
    else:
        print('Add row for root={0}, id={1}'.format(root, id))
        
    # Add the new data
    row_df.to_sql('redshift_fit', engine, index=False, if_exists='append', method='multi')

###########
def add_missing_rows(root='j004404m2034', engine=None):
    """
    Add rows that were completed but that aren't in the table
    """
    import glob
    from astropy.table import vstack, Table
    
    from grizli.aws import db as grizli_db
    
    if engine is None:
        engine = grizli_db.get_db_engine(echo=False)
    
    os.system('aws s3 sync s3://grizli-v1/Pipeline/{0}/Extractions/ ./ --exclude "*" --include "*row.fits"'.format(root))
    
    row_files = glob.glob('{0}*row.fits'.format(root))
    row_files.sort()
    
    res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit WHERE root = '{0}' AND status=6".format(root), engine)
    
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
    from .. import utils
    
    if not os.path.exists(file):
        print('Spectrum file not found')
        return False
    
    im = pyfits.open(file)
    obj_id = im[0].header['ID']
    obj_root = im[0].header['TARGET']
    
    if '.R30.' in file:
        skip_columns = ['line','cont']
        pref = 'spec1d_r30'
    else:
        skip_columns = []
        pref = 'spec1d'
        
    spectra = OrderedDict()
    has_spectra = False
    for gr in ['G102','G141','G800L']:
        if gr in im:
            has_spectra = True
            sp = utils.GTable.read(file, hdu=gr)
            prefix = '{0}_{1}_'.format(pref, gr.lower())
            
            spd = {prefix+'id':obj_id, prefix+'root':obj_root}
            for c in sp.colnames:
                if c in skip_columns:
                    continue
                    
                spd[prefix+c] = sp[c].tolist()
            
            spectra[gr.lower()] = spd
    
    if has_spectra:
        return spectra
    else:
        return False
        
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
        tablename = '{0}_{1}'.format(prefix, gr)
        df = pd.DataFrame(tables[gr])
                        
        # Put wavelengths in their own tables to avoid massive duplication
        wave_table = tablename+'_wave'
        if wave_table not in engine.table_names():
            print('Create wave table: '+wave_table)
            wdf = pd.DataFrame(data=tables[gr][wave_table][0], 
                               columns=[wave_table])

            wdf.to_sql(wave_table, engine, if_exists='replace', 
                       index=True, index_label = tablename+'_idx')
            
        # drop wave from spectra tables
        df.drop('{0}_wave'.format(tablename), axis=1, inplace=True) 
        
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
            mat = ["({0}_root = '{1}' AND {0}_id = {2})".format(tablename, r, i) for r, i in zip(df[tablename+'_root'], df[tablename+'_id'])]
            SQL += 'OR '.join(mat)
            rsp = engine.execute(SQL)
        
        # Send the table
        print('Send {0} rows to {1}'.format(len(df), tablename))
        df.to_sql(tablename, engine, index=False, if_exists='append',
                  method='multi')

def add_oned_spectra(root='j214224m4420gr01', bucket='grizli-v1'):
    import os
    import boto3
   
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
    os.system('aws s3 sync s3://{0}/Pipeline/{1}/Extractions/ ./ --exclude "*" --include "*R30.fits" --include "*1D.fits"'.format(bucket, root))
    
    # 1D.fits
    files = glob.glob('{0}_*1D.fits'.format(root)); files.sort()
    grizli_db.send_1D_to_database(files=files, engine=engine)

    files = glob.glob('{0}_*R30.fits'.format(root)); files.sort()
    grizli_db.send_1D_to_database(files=files, engine=engine)
    
    os.system('rm {0}_*.1D.fits {0}_*.R30.fits'.format(root))
    
    if False:
        tablename = 'spec1d_g141'

        #tablename = 'spec1d_r30_g141'
        
        resp = pd.read_sql_query("SELECT root, id, z_map, q_z, sp.* from redshift_fit, {1} as sp WHERE {1}_root = root AND {1}_id = id AND root = '{0}' AND q_z > -0.7 ORDER BY z_map".format(root, tablename), engine)
        
        wave = pd.read_sql_query("SELECT * from {0}_wave".format(tablename),
                                 engine)[tablename+'_wave'].values
        
        data = OrderedDict()
        for c in resp.columns:
            if c.startswith(tablename):
                c_i = c.split(tablename+'_')[1]
                try:
                    data[c_i] = np.array(resp[c].values.tolist())  
                except:
                    pass
                    
        plt.imshow((data['flux'] - data['cont'])/data['flat']/1.e-19, vmin=-0.1, vmax=10)

        plt.imshow((data['line'] - data['cont'])/data['flat']/1.e-19, vmin=-0.1, vmax=10)
        
def run_lambda_fits(root='j004404m2034', phot_root=None, mag_limits=[15, 26], sn_limit=7, min_status=None, engine=None, zr=[0.01,3.4], bucket='grizli-v1', verbose=True, extra={'bad_pa_threshold':10}):
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
        root=root
        
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

    for filt in ['f160w','f140w','f125w','f105w','f110w','f098m','f814w','f850lp','f606w','f775w']:
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
        res = pd.read_sql_query("SELECT root, id, status, mtime FROM redshift_fit WHERE root = '{0}'".format(root, min_status), engine)
        if len(res) > 0:
            status = phot['id']*0-100
            status[res['id']-1] = res['status']
            sel &= status < min_status
        
    ids = phot['id'][sel]
    
    # Select just on min_status
    if min_status > 1000:
        if min_status > 10000:
            # Include mag constraints
            res = pd.read_sql_query("SELECT root, id, status, mtime, mag_auto FROM redshift_fit,photometry_apcorr WHERE root = '{0}' AND status = {1}/10000 AND mag_auto > {2} AND mag_auto < {3} AND p_root = root AND p_id = id".format(root, min_status, mag_limits[0], mag_limits[1]), engine)
        else:
            # just select on status
            res = pd.read_sql_query("SELECT root, id, status, mtime FROM redshift_fit WHERE root = '{0}' AND status = {1}/1000".format(root, min_status, mag_limits[0], mag_limits[1]), engine)
            
        ids = res['id'].tolist()
        
    if len(ids) == 0:
        return False
        
    fit_redshift_lambda.fit_lambda(root=root, beams=[], ids=ids, newfunc=False, bucket_name=bucket, skip_existing=False, sleep=False, skip_started=False, show_event=False, zr=zr, force_args=True, quasar_fit=False, output_path=None, save_figures='png', verbose=verbose, **extra)
    
    print('Add photometry: {0}'.format(root))
    grizli_db.add_phot_to_db(phot_root, delete=False, engine=engine)
    
    res = grizli_db.wait_on_db_update(root, dt=15, n_iter=120, engine=engine)
    
    grizli_db.set_phot_root(root, phot_root, engine)
    
    res = pd.read_sql_query("SELECT root, id, flux_radius, mag_auto, z_map, status, bic_diff, zwidth1, log_pdf_max, chinu FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit WHERE z_map > 0 AND root = '{0}') z ON (p.p_root = z.root AND p.p_id = z.id)".format(root), engine)
    return res
    
    if False:
        res = pd.read_sql_query("SELECT root, id, status, redshift, bic_diff, mtime FROM redshift_fit WHERE (root = '{0}')".format(root), engine)
        
        # Get arguments
        args = fit_redshift_lambda.fit_lambda(root=root, beams=[], ids=ids, newfunc=False, bucket_name='grizli-v1', skip_existing=False, sleep=False, skip_started=False, quasar_fit=False, output_path=None, show_event=2, zr=[0.01,3.4], force_args=True)

def set_phot_root(root, phot_root, engine):
    """
    """
    print('Set phot_root = {0} > {1}'.format(root, phot_root))
    
    
    SQL = """UPDATE redshift_fit
     SET phot_root = '{phot_root}'
    WHERE (root = '{root}');
    """.format(phot_root=phot_root, root=root)

    engine.execute(SQL)

    if False:
        # Check where phot_root not equal to root
        res = pd.read_sql_query("SELECT root, id, status, phot_root FROM redshift_fit WHERE (phot_root != root)".format(root), engine)
        
        # update the one pointing where it should change in photometry_apcorr
        engine.execute("UPDATE photometry_apcorr SET root = 'j214224m4420' WHERE root = 'j214224m4420gr01';")
        
        
    if False:
        # Set 3D-HST fields
        res = grizli_db.from_sql("select distinct root from redshift_fit where root like '%%-grism%%'", engine)
        for root in res['root']:
            grizli_db.set_phot_root(root, root, engine)
            grizli_db.set_phot_root(root.replace('-grism', '-g800l'), root, engine)
            xres = grizli_db.from_sql("select root, count(root) from redshift_fit where root like '{0}-%%' group by root".format(root.split('-')[0]), engine)
            print(xres)
        
        
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
        res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit WHERE root = '{0}'".format(root), engine)
        checksum = (2**res['status']).sum()
        n = len(res)
        n6 = (res['status'] == 6).sum()
        n5 = (res['status'] == 5).sum()
        if (n == n_i) & (checksum == checksum_i) & (n6 == n6_i):
            break
        
        now = time.ctime()
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
    
    res = pd.read_sql_query("SELECT id, status FROM redshift_fit WHERE root = '{0}' AND status = 5".format(root), engine)
    if len(res) == 0:
        return True
        
    ids = res['id'].tolist()
    
    fit_redshift_lambda.fit_lambda(root=root, beams=[], ids=ids, newfunc=False, bucket_name='grizli-v1', skip_existing=False, sleep=False, skip_started=False, quasar_fit=False, output_path=None, show_event=False, zr=[0.01,2.4], force_args=True)
    
    res = grizli_db.wait_on_db_update(root, dt=15, n_iter=120, engine=engine)
    return res
    
    # All timeouts
    events = fit_redshift_lambda.fit_lambda(root='egs-g800l-j141956p5255', beams=[], ids=[20667], newfunc=False, bucket_name='grizli-v1', skip_existing=False, sleep=False, skip_started=False, quasar_fit=False, output_path=None, show_event=2, zr=[0.01,2.4], force_args=True)
    
    res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit WHERE status = 5 AND root NOT LIKE 'cos-grism%%' ORDER BY root".format(root), engine)
    
    base = {'bucket': 'grizli-v1', 'skip_started': False, 'quasar_fit': False, 'zr': '0.01,2.4', 'force_args': True, 'bad_pa_threshold': 10, 'use_phot_obj': False, 'save_figures':'png'}
    
    all_events = fit_redshift_lambda.generate_events(res['root'], res['id'], base=base, send_to_lambda=True)
    
    xres = pd.read_sql_query("SELECT root, p_ra as ra, p_dec as dec, id, status FROM redshift_fit WHERE status = 5 AND root LIKE 'gds-grism%%' ORDER BY root".format(root), engine)
    
    print(len(res), len(xres))

    # show points
    xres = pd.read_sql_query("SELECT root, p_ra as ra, p_dec as dec, id, status FROM redshift_fit WHERE status = 5 AND root LIKE 'gds-grism%%' ORDER BY root".format(root), engine)
    
    
########### Photometry table
def set_filter_bits(phot):
    """
    Set bits indicating available filters
    """
    import numpy as np
    filters = ['f160w','f140w','f125w','f110w','f105w','f098m','f850lp','f814w','f775w','f625w','f606w','f475w','f438w','f435w','f555w','f350lp', 'f390w','f336w','f275w','f225w'] 
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
    
    for c in ['dummy_flux','dummy_err']:
        if c in phot.colnames:
            phot.remove_column(c)
    
    for c in ['xmin','xmax','ymin','ymax']:
        phot.rename_column(c, 'image_'+c)
    
    for c in ['root', 'id', 'ra', 'dec']:
        phot.rename_column(c, 'p_'+c)
         
    df = phot.to_pandas()
    return df
    
def add_phot_to_db(root, delete=False, engine=None, nmax=500):
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
        else:
            print('Data found for root={0}, delete them if necessary'.format(root))
            return False
    
    # Read the catalog
    phot = Table.read('{0}_phot_apcorr.fits'.format(root), character_as_bytes=False)
    
    # remove columns
    remove = []
    for c in phot.colnames: 
        if ('_corr_' in c) | ('_ecorr_' in c) | (c[-5:] in ['tot_4','tot_5', 'tot_6']): 
            remove.append(c) 
    phot.remove_columns(remove)
    
    # Add new filter columns if necessary
    empty = pd.read_sql_query("SELECT * FROM photometry_apcorr WHERE false", engine)        
    
    df = phot_to_dataframe(phot, root)
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
    
def test_join():
    import pandas as pd
    
    res = pd.read_sql_query("SELECT root, id, flux_radius, mag_auto, z_map, status, bic_diff, zwidth1, log_pdf_max, chinu FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit WHERE z_map > 0) z ON (p.p_root = z.root AND p.p_id = z.id)".format(root), engine)        

    res = pd.read_sql_query("SELECT * FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit WHERE z_map > 0) z ON (p.p_root = z.root AND p.p_id = z.id)".format(root), engine)        
    
    # on root
    res = pd.read_sql_query("SELECT p.root, p.id, mag_auto, z_map, status FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit WHERE root='{0}') z ON (p.p_root = z.root AND p.p_id = z.id)".format(root), engine)        
     
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
        fit = grizli_db.from_sql("select root, ra, dec from redshift_fit", engine)

        fit = grizli_db.from_sql("select root, ra, dec from redshift_fit where ra is null", engine)

        
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
        # Update redshift_fit ra/dec with photometry_table double prec.
          SQL = """UPDATE redshift_fit
             SET ra = p_ra
                 dec = p_dec
           FROM photometry_apcorr
           WHERE (phot_root = p_root AND id = p_id AND root = 'j123556p6221');
           """        
def mtime_to_iso(ct):
    """
    Convert mtime values to ISO format suitable for sorting, etc.
    """
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    spl = ct.split()
    iso = '{yr}-{mo:02d}-{dy:02d} {time}'.format(dy=int(spl[2]), mo=int(months.index(spl[1])+1), yr=spl[-1], time=spl[-2])
    return iso
    
def various_selections():
     
    # sdss z_spec
    res = make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','z_spec','z_map','bic_diff','chinu','log_pdf_max'], where="AND status > 5 AND z_spec > 0 AND z_spec_qual = 1 AND z_spec_src ~ '^sdss-dr15'", table_root='sdss_zspec', sync='s3://grizli-v1/tables/')

    # objects with carla redshifts (radio loud)
    res = make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','z_spec','z_map','bic_diff','chinu','log_pdf_max'], where="AND status > 5 AND z_spec > 0 AND z_spec_qual = 1 AND z_spec_src ~ '^carla'", table_root='carla_zspec', sync='s3://grizli-v1/tables/')
    
    # Bright galaxies with q_z flag
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','t_g102', 't_g141', 'z_spec','z_map','bic_diff','chinu','zwidth1/(1+z_map) as zw1','q_z','q_z > -0.69 as q_z_TPR90','dlinesn'], where="AND status > 4 AND mag_auto < 22 AND z_map > 1.3", table_root='bright', sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'], show_hist=True)

    # High-z compiliation
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','t_g102', 't_g141', 'z_spec','z_map','bic_diff','chinu','q_z','h_zphot', 'h_src'], where="AND status > 4 AND phot_root = h_root AND id = h_id", tables=['highz_2015'], table_root='bright', sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'], show_hist=True)
    
    # z_spec with dz
    res = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','z_spec','z_map','z_spec_src','bic_diff','chinu','log_pdf_max', 'zwidth1/(1+z_map) as zw1','(z_map-z_spec)/(1+z_spec) as dz', 'dlinesn'], where="AND status > 4 AND z_spec > 0 AND z_spec_qual = 1", table_root='zspec_delta', sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'])

    if False:
        from matplotlib.ticker import FixedLocator, AutoLocator, MaxNLocator
        xti = xt = np.arange(0,3.6,0.5)
        loc = np.arange(0, 3.6, 0.1)
        bins = utils.log_zgrid([0.03, 3.5], 0.01)
        fig = plt.figure(figsize=[7,6])
        ax = fig.add_subplot(111)
        ax.scatter(np.log(1+res['z_spec']), np.log(1+res['z_map']), alpha=0.2, c=np.log10(res['zw1']), marker='.', vmin=-3.5, vmax=-0.5, cmap='plasma')

        sc = ax.scatter(np.log([1]), np.log([1]), alpha=0.8, c=[0], marker='.', vmin=-3.5, vmax=-0.5, cmap='plasma')
         
        cb = plt.colorbar(sc, shrink=0.6)
        cb.set_label(r'$(z_{84}-z_{16})/(1+z_{50})$')
        cb.set_ticks([-3,-2,-1])
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
    root='cos-grism-j100012p0210'
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','t_g102', 't_g141', 'z_spec','z_map','bic_diff','chinu','zwidth1/(1+z_map) as zw1','dlinesn'], where="AND status > 4 AND bic_diff > 100 AND root = '{0}'".format(root), table_root=root, sync='s3://grizli-v1/Pipeline/{0}/Extractions/'.format(root), png_ext=['R30', 'stack','full','line'], show_hist=True)
    
    # high bic_diff = unambiguous
    res = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','z_spec','z_map','bic_diff','chinu','log_pdf_max','d4000','d4000_e', '-(bic_temp-bic_spl) as bic_diff_spl'], where="AND status > 5 AND (((bic_diff > 50 OR  zwidth1/(1+z_map) < 0.01) AND chinu < 2))", table_root='unamb', sync='s3://grizli-v1/tables/')

    # with d4000
    res = make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','z_spec','z_map','bic_diff','chinu','log_pdf_max','d4000','d4000_e'], where="AND status > 5 AND chinu < 3 AND d4000 > 1 AND d4000 < 5 AND d4000_e > 0 AND d4000_e < 0.25 AND bic_diff > 5", table_root='d4000', sync='s3://grizli-v1/tables/')

    # LBG?
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','z_spec','z_map','bic_diff','chinu','log_pdf_max','-(bic_temp-bic_spl) as bic_diff_spl', 'splf01/splf02 as r12', 'splf02/splf03 as r23', 'splf02/sple02 as sn02'], where="AND status > 5 AND mag_auto > 23 AND bic_diff > -50 AND splf01/splf02 < 0.3 AND splf02/sple02 > 2 AND splf01 != 0 AND splf02 != 0 AND splf03 != 0 ".format(root), table_root='lbg_g800l', sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'])

    # stars?
    res = make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','z_spec','z_map','bic_diff','chinu','log_pdf_max'], where="AND status > 5 AND bic_diff > 100 AND chinu < 1.5 AND mag_auto < 24 AND sn_Ha > 20", table_root='star', sync='s3://grizli-v1/tables/')

    # By root
    root='j001420m3030'
    res = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','z_spec','z_map','bic_diff','chinu','log_pdf_max'], where="AND status > 5 AND root = '{0}' AND bic_diff > 5".format(root), table_root=root+'-fit', sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'])

    # G800L spec-zs
    res = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','z_spec','z_map','bic_diff','chinu','log_pdf_max','(z_map-z_spec)/(1+z_spec) as delta_z'], where="AND status > 5 AND z_spec > 0 AND z_spec_qual = 1 AND t_g800l > 0", table_root='zspec_g800l', sync='s3://grizli-v1/tables/')

    # Large G800L likely mismatch [OIII]/Ha
    res = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','a_image','flux_radius','t_g800l','z_spec','z_map','bic_diff','chinu','log_pdf_max','err_ha', 'ew50_oiii/(1+z_map) as ew_oiii_rest','sn_oiii'], where="AND status > 5 AND t_g800l > 0 AND sn_oiii > 3 AND mag_auto < 23 AND bic_diff > 5", table_root='g800l_oiii_mismatch', sync='s3://grizli-v1/tables/')

    # Potential Ly-a?
    res = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','a_image','flux_radius','t_g800l','z_spec','z_map','bic_diff','chinu','log_pdf_max','err_ha', 'ew50_oiii/(1+z_map) as ew_oiii_rest','sn_oiii'], where="AND status > 5 AND t_g800l > 0 AND sn_oiii > 5 AND sn_ha > 0 AND flux_oiii/flux_ha > 1.8", table_root='g800l_oiii_mismatch', sync='s3://grizli-v1/tables/')

    # Continuum resid
    res = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','z_spec','z_map','bic_diff','chinu','log_pdf_max', '23.9-2.5*log(splf01*8140*8140/3.e18*1.e29)-mag_auto as dmag'], where="AND status > 5 AND bic_diff > 5 AND splf01 > 0 AND bic_diff > 50".format(root), table_root='xxx', sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'])
    
    res = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','a_image','flux_radius','t_g800l','z_spec','z_map','bic_diff','chinu','log_pdf_max','err_ha','sn_oiii', 'f814w_tot_1*3.e18/8140/8140/splf01*1.e-29 as fresid', 'splf01/sple01 as sn01', '23.9-2.5*log(splf01*8140*8140/3.e18*1.e29)-mag_auto as dmag'], where="AND status > 5 AND t_g800l > 0 AND f814w_tot_1 > 0 AND splf01 != 0 AND splf01/sple01 > 1 AND f814w_tot_1*3.e18/8140/8140/splf01*1.e-29 > 0 AND (f814w_tot_1*3.e18/8140/8140/splf01*1.e-29 < 0.3 OR f814w_tot_1*3.e18/8140/8140/splf01*1.e-29 > 4)", table_root='g800l_oiii_mismatch', sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'])

    sql = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','a_image','flux_radius','t_g800l','z_spec','z_map','bic_diff','chinu','log_pdf_max','err_ha','sn_oiii', 'splf01', 'sple01', 'f814w_tot_1', 'f850lp_tot_1', 'flux_auto/flux_iso as flux_aper_corr', '23.9-2.5*log(splf01*8140*8140/3.e18*1.e29)-mag_auto as dmag'], where="AND status > 5 AND t_g800l > 0 AND splf01 > 0", table_root='g800l_oiii_mismatch', sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'], get_sql=True)
    res = pd.read_sql_query(sql, engine) 

    splmag = 23.9-2.5*np.log10(np.maximum(res['splf01'], 1.e-22)*8140**2/3.e18*1.e29)

    sql = grizli_db.make_html_table(engine=engine, columns=['root','status','id','p_ra','p_dec','mag_auto','a_image','flux_radius','t_g800l','z_spec','z_map','bic_diff','chinu','log_pdf_max','err_ha','sn_oiii', 'splf03', 'sple03', 'f140w_tot_1', 'f160w_tot_1', 'flux_auto/flux_iso as flux_aper_corr'], where="AND status > 5 AND t_g141 > 0 AND sple03 > 0", table_root='g800l_oiii_mismatch', sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'], get_sql=True)
    res = pd.read_sql_query(sql, engine) 
    splmag = 23.9-2.5*np.log10(np.maximum(res['splf03'], 1.e-22)*1.2e4**2/3.e18*1.e29)

    # Number of matches per field
    counts = pd.read_sql_query("select root, COUNT(root) as n from redshift_fit, photometry_apcorr where phot_root = p_root AND id = p_id AND bic_diff > 50 AND mag_auto < 24 group by root;", engine)
  
def from_sql(query, engine):
    import pandas as pd
    from grizli import utils
    res = pd.read_sql_query(query, engine) 
    
    tab = utils.GTable.from_pandas(res)
    set_column_formats(tab)
    
    return tab
    
def render_for_notebook(tab, image_extensions=['stack','full','line'], bucket='grizli-v1', max_rows=20):
    """
    Render images for inline display in a notebook
    
    In [1]: from IPython.display import HTML
    
    In [2]: HTML(tab)
    
    """
    rows = tab[:max_rows].copy()
    rows['bucket'] = bucket
    rows['ext'] = 'longstring' # longer than the longest extension
    
    s3url = 'https://s3.amazonaws.com/{bucket}/Pipeline/{root}/Extractions/{root}_{id:05d}.{ext}.png'

    def href_root(root):
        s3 = 'https://s3.amazonaws.com/'+bucket+'/Pipeline/{0}/Extractions/{0}.html'
        return '<a href={0}>{1}</a>'.format(s3.format(root), root)
        
    def path_to_image_html(path):
        return '<a href={0}><img src="{0}"/></a>'.format(path)
    
    # link for root
    
    fmt = {'root':href_root}
    for ext in image_extensions:
        rows['ext'] = ext
        urls = [s3url.format(**row) for row in rows.to_pandas().to_dict(orient='records')]
        rows[ext] = urls
        fmt[ext] = path_to_image_html

    rows.remove_columns(['bucket','ext'])
    out = rows.to_pandas().to_html(escape=False, formatters=fmt)
    return out
    
    
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
    by_nz = from_sql("select root, COUNT(root) as nz from redshift_fit where bic_diff > 30 group by root;", engine)
    
    for count in [by_mag, by_nz]:
        new_col = count.colnames[1]
        ch[new_col] = -1
        for r, n in zip(count['root'], count[new_col]):
            ix = ch['field_root'] == r
            ch[new_col][ix] = n
    
    zhist = ['https://s3.amazonaws.com/grizli-v1/Pipeline/{0}/Extractions/{0}_zhist.png'.format(r) for r in ch['field_root']]
    ch['zhist'] = ['<a href="{1}"><img src={0} height=300px></a>'.format(zh, zh.replace('_zhist.png','.html')) for zh in zhist]
            
    cols = ['field_root', 'field_ra', 'field_dec', 'mw_ebv', 'gaia5', 'nassoc', 'nfilt', 'filter', 'target', 'comment', 'proposal_id', 'proposal_pi', 'field_t_g800l', 'field_t_g102', 'field_t_g141', 'mast', 'footprint', 'rgb', 'nmag', 'nz', 'zhist', 'summary', 'log']
    
    sortable = []
    for c in cols:
        if not hasattr(ch[c][0], 'upper'):
            sortable.append(c)
    
    #https://s3.amazonaws.com/grizli-v1/Master/CHArGE-July2019.html
    
    table_root = 'CHArGE-July2019.zhist'
    
    ch[cols].write_sortable_html('{0}.html'.format(table_root), replace_braces=True, localhost=False, max_lines=1e5, table_id=None, table_class='display compact', css=None, filter_columns=sortable, buttons=['csv'], toggle=True, use_json=True)
    
    os.system('aws s3 sync ./ s3://grizli-v1/Master/ --exclude "*" --include "{1}.html" --include "{1}.json" --acl public-read'.format('', table_root))
    
    
def run_all_redshift_fits(): 
    ##############
    # Run all
    from grizli.aws import db as grizli_db
    import pandas as pd
    engine = grizli_db.get_db_engine()
    
    # By grism
    res = pd.read_sql_query("select field_root, field_t_g800l, field_t_g102, field_t_g141, proposal_pi from charge_fields where (nassoc < 200 AND (field_t_g800l > 0 OR field_t_g141 > 0 OR  field_t_g102 > 0) AND log LIKE '%%inish%%');", engine)
    orig_roots = pd.read_sql_query('select distinct root from redshift_fit', engine)['root'].tolist()
    
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
        
        phot_root=None
         
        try: 
            grizli_db.run_lambda_fits(root, phot_root=phot_root, 
                                      min_status=6, zr=[0.01,zmax])  
        except:
            pass

    ####
    # Redo fits on reprocessed fields

    # for i in range(2,11):
    #     root = 'j214224m4420gr{0:02d}'.format(i)
    #     print(root)
    # 
    res = engine.execute("DELETE from redshift_fit WHERE (root = '{0}')".format(root), engine)
    res = engine.execute("DELETE from photometry_apcorr WHERE (p_root = '{0}')".format(root), engine)
    grizli_db.run_lambda_fits(root, min_status=6, zr=[0.01,zmax])  
    
    #for root in "j233844m5528 j105732p3620 j112416p1132 j113812m1134 j113848m1134 j122852p1046 j143200p0959 j152504p0423 j122056m0205 j122816m1132 j131452p2612".split():
    res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','t_g102', 't_g141', 'z_spec','z_map','bic_diff','chinu','zwidth1/(1+z_map) as zw1','q_z','q_z > -0.69 as q_z_TPR90','dlinesn'], where="AND status > 4 AND root = '{0}'".format(root), table_root=root, sync='s3://grizli-v1/Pipeline/{0}/Extractions/'.format(root), png_ext=['R30', 'stack','full','line'], show_hist=True)
            
    os.system('aws s3 cp s3://grizli-v1/Pipeline/{0}/Extractions/{0}_zhist.png s3://grizli-v1/tables/'.format(root))
    
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
    counts = pd.read_sql_query("select root, COUNT(root) as n from redshift_fit, photometry_apcorr where phot_root = p_root AND id = p_id AND bic_diff > 5 AND mag_auto < 24 group by root;", engine)
    
    counts = utils.GTable.from_pandas(counts)
    so = np.argsort(counts['n'])
    
    
    sh = """
    BUCKET=grizli-v
    root=j113812m1134
    
    aws s3 rm --recursive s3://grizli-v1/Pipeline/${root}/ --include "*"
    grism_run_single.sh ${root} --run_fine_alignment=True --extra_filters=g800l --bucket=grizli-v1 --preprocess_args.skip_single_optical_visits=True --mask_spikes=True --persistence_args.err_threshold=1
    """
def add_missing_photometry():
    
    # Add missing photometry
    import os
    import pandas as pd
    from grizli.aws import db as grizli_db
    from grizli.pipeline import photoz
    
    engine = grizli_db.get_db_engine(echo=False)
    
    res = pd.read_sql_query("select distinct root from redshift_fit where root like 'j%%'", engine)['root'].tolist()
    orig_roots = pd.read_sql_query('select distinct p_root as root from photometry_apcorr', engine)['root'].tolist()
    
    # Missing grism fields?
    res = pd.read_sql_query("select field_root as root, field_t_g800l, field_t_g102, field_t_g141, proposal_pi from charge_fields where (field_t_g800l > 0 OR field_t_g141 > 0 OR  field_t_g102 > 0) AND log LIKE '%%inish%%';", engine)['root'].tolist()
    orig_roots = pd.read_sql_query('select distinct root from redshift_fit', engine)['root'].tolist()
    
    # All photometry
    res = pd.read_sql_query("select field_root as root, field_t_g800l, field_t_g102, field_t_g141, proposal_pi from charge_fields where nassoc < 200 AND log LIKE '%%inish%%' AND field_root LIKE 'j%%';", engine)['root'].tolist()
    orig_roots = pd.read_sql_query('select distinct p_root as root from photometry_apcorr', engine)['root'].tolist()
    
    count=0
    for root in res:
        if root not in orig_roots:
            count+=1
            print(count, root)
            os.system('aws s3 cp s3://grizli-v1/Pipeline/{0}/Extractions/{0}_phot_apcorr.fits .'.format(root))
            os.system('aws s3 cp s3://grizli-v1/Pipeline/{0}/Extractions/{0}_phot.fits .'.format(root))
            
            if not os.path.exists('{0}_phot_apcorr.fits'.format(root)):
                os.system('aws s3 cp s3://grizli-v1/Pipeline/{0}/Prep/{0}_phot_apcorr.fits .'.format(root))
                os.system('aws s3 cp s3://grizli-v1/Pipeline/{0}/Prep/{0}_phot.fits .'.format(root))
                
            if os.path.exists('{0}_phot_apcorr.fits'.format(root)):
                grizli_db.add_phot_to_db(root, delete=False, engine=engine)
            else:
                if os.path.exists('{0}_phot.fits'.format(root)):
                    # Make the apcorr file
                    utils.set_warnings()

                    total_flux = 'flux_auto' 
                    obj = photoz.eazy_photoz(root, object_only=True,
                              apply_prior=False, beta_prior=True, 
                              aper_ix=1, 
                              force=True,
                              get_external_photometry=False, 
                              compute_residuals=False, 
                              total_flux=total_flux)
                    
                    grizli_db.add_phot_to_db(root, delete=False, 
                                             engine=engine)
                    
    # 3D-HST
    copy = """
    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/egs-mosaic_phot_apcorr.fits s3://grizli-v1/Pipeline/egs-grism-j141956p5255/Extractions/egs-grism-j141956p5255_phot_apcorr.fits --acl public-read
    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/egs-mosaic_phot.fits s3://grizli-v1/Pipeline/egs-grism-j141956p5255/Extractions/egs-grism-j141956p5255_phot.fits --acl public-read
    """
    grizli_db.run_lambda_fits('egs-grism-j141956p5255', min_status=6, zr=[0.01,3.2])

    copy = """
    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/uds-mosaic_phot_apcorr.fits s3://grizli-v1/Pipeline/uds-grism-j021732m0512/Extractions/uds-grism-j021732m0512_phot_apcorr.fits --acl public-read    
    """
    grizli_db.run_lambda_fits('uds-grism-j021732m0512', min_status=6, zr=[0.01,3.2])

    # GDS
    copy = """
    aws s3 rm s3://grizli-v1/Pipeline/gds-grism-j033236m2748/Extractions/ --recursive --exclude "*" --include "gds-grism-j033236m2748_[0-9]*"
    aws s3 rm s3://grizli-v1/Pipeline/gds-g800l-j033236m2748/Extractions/ --recursive --exclude "*" --include "gds-g800l-j033236m2748_[0-9]*"
    
    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/gds-mosaic_phot_apcorr.fits s3://grizli-v1/Pipeline/gds-grism-j033236m2748/Extractions/gds-grism-j033236m2748_phot_apcorr.fits --acl public-read
    
    aws s3 cp s3://grizli-v1/Pipeline/gds-grism-j033236m2748/Extractions/gds-grism-j033236m2748_phot_apcorr.fits s3://grizli-v1/Pipeline/gds-g800l-j033236m2748/Extractions/gds-g800l-j033236m2748_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('gds-grism-j033236m2748', phot_root='gds-grism-j033236m2748', min_status=6, zr=[0.01, 3.2], extra={'bad_pa_threshold':10, 'use_phot_obj':False})
    grizli_db.run_lambda_fits('gds-g800l-j033236m2748', phot_root='gds-grism-j033236m2748', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold':10, 'use_phot_obj':False})
    
    # GDN
    copy = """
    #aws s3 rm s3://grizli-v1/Pipeline/gds-g800l-j033236m2748/Extractions/ --recursive --exclude "*" --include "gds-g800l-j033236m2748_[0-9]*"
    aws s3 rm s3://grizli-v1/Pipeline/gdn-grism-j123656p6215/Extractions/ --recursive --exclude "*" --include "gdn-grism-j123656p6215_[0-9]*"
    aws s3 rm s3://grizli-v1/Pipeline/gdn-g800l-j123656p6215/Extractions/ --recursive --exclude "*" --include "gdn-g800l-j123656p6215_[0-9]*"
    
    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/gdn-mosaic_phot_apcorr.fits s3://grizli-v1/Pipeline/gdn-grism-j123656p6215/Extractions/gdn-grism-j123656p6215_phot_apcorr.fits --acl public-read
    
    aws s3 cp s3://grizli-v1/Pipeline/gdn-grism-j123656p6215/Extractions/gdn-grism-j123656p6215_phot_apcorr.fits s3://grizli-v1/Pipeline/gdn-g800l-j123656p6215/Extractions/gdn-g800l-j123656p6215_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('gdn-grism-j123656p6215', phot_root='gdn-grism-j123656p6215', min_status=6, zr=[0.01, 3.2], extra={'bad_pa_threshold':10, 'use_phot_obj':False})
    grizli_db.run_lambda_fits('gdn-g800l-j123656p6215', phot_root='gdn-grism-j123656p6215', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold':10, 'use_phot_obj':False})
    
    # 3D-HST G800L
    copy = """
    aws s3 rm s3://grizli-v1/Pipeline/egs-g800l-j141956p5255/Extractions/ --recursive --exclude "*" --include "egs-g800l-j141956p5255_[0-9]*"
    
    aws s3 cp s3://grizli-v1/Pipeline/egs-grism-j141956p5255/Extractions/egs-grism-j141956p5255_phot_apcorr.fits s3://grizli-v1/Pipeline/egs-g800l-j141956p5255/Extractions/egs-g800l-j141956p5255_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('egs-g800l-j141956p5255', phot_root='egs-grism-j141956p5255', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold':10, 'use_phot_obj':False})
    
    res = grizli_db.wait_on_db_update('egs-g800l-j141956p5255', dt=15, n_iter=120, engine=engine)
    res = grizli_db.wait_on_db_update('uds-g800l-j021732m0512', dt=15, n_iter=120, engine=engine)
    
    
    # UDS
    copy = """
    aws s3 rm s3://grizli-v1/Pipeline/uds-g800l-j021732m0512/Extractions/ --recursive --exclude "*" --include "uds-g800l-j021732m0512_[0-9]*"
    
    aws s3 cp s3://grizli-v1/Pipeline/uds-grism-j021732m0512/Extractions/uds-grism-j021732m0512_phot_apcorr.fits s3://grizli-v1/Pipeline/uds-g800l-j021732m0512/Extractions/uds-g800l-j021732m0512_phot_apcorr.fits --acl public-read
    """
    grizli_db.run_lambda_fits('uds-g800l-j021732m0512', phot_root='uds-grism-j021732m0512', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold':10, 'use_phot_obj':False})
    
    grizli_db.run_lambda_fits('egs-g800l-j141956p5255', phot_root='egs-grism-j141956p5255', min_status=6, zr=[0.01, 1.6], extra={'bad_pa_threshold':10, 'use_phot_obj':False})
    
    # Cosmos on oliveraws
    copy = """
    
    aws s3 rm s3://grizli-cosmos-v2/Pipeline/cos-grism-j100012p0210/Extractions/ --recursive --exclude "*" --include "cos-grism-j100012p0210_[0-9]*"
    
    aws s3 cp /Users/gbrammer/Research/HST/Mosaics/Cosmos/cos-cnd-mosaic_phot_apcorr.fits s3://grizli-cosmos-v2/Pipeline/cos-grism-j100012p0210/Extractions/cos-grism-j100012p0210_phot_apcorr.fits --acl public-read    
    """
    grizli_db.run_lambda_fits('cos-grism-j100012p0210', min_status=6, zr=[0.01,3.2], mag_limits=[17,17.1], bucket='grizli-cosmos-v2')
    os.system('sudo halt')
    
def set_column_formats(info):
    # Print formats
    formats = {}
    formats['ra'] = formats['dec'] = '.5f'
    formats['mag_auto'] = formats['delta_z'] = '.2f'
    formats['chinu'] =  formats['chimin'] = formats['chimax'] =  '.1f'
    formats['bic_diff'] = formats['bic_temp'] = formats['bic_spl']  = '.1f'
    formats['bic_poly'] = '.1f'
    formats['dlinesn'] = formats['bic_spl'] = '.1f'

    formats['flux_radius'] = formats['flux_radius_20'] = '.1f'      
    formats['flux_radius_90'] = '.1f'
    
    formats['log_pdf_max'] = formats['log_risk'] = '.1f'
    formats['d4000'] = formats['d4000_e'] = '.2f'
    formats['dn4000'] = formats['dn4000_e'] = '.2f'
    formats['z_spec'] = formats['z_map'] = formats['reshift'] = '.3f'
    formats['t_g141'] = formats['t_g102'] = formats['t_g800l'] = '.0f'
    formats['zwidth1'] = formats['zw1'] = '.3f'
    formats['zwidth2'] = formats['zw2'] = '.3f'
    
    formats['q_z'] = '.2f'
    
    for c in info.colnames:
        if c in formats:
            info[c].format = formats[c]
        elif c.startswith('sn_'):
            info[c].format = '.2f'
        elif c.startswith('mag_'):
            info[c].format = '.2f'
        elif c.startswith('ew_'):
            info[c].format = '.1f'
        elif c.startswith('bic_'):
            info[c].format = '.1f'
        elif c in ['z02', 'z16', 'z50', 'z84', 'z97']:
            info[c].format = '.3f'
        elif c[:4] in ['splf','sple']:
            info[c].format = '.1e'
        elif c.startswith('flux_') | c.startswith('err_'):
            info[c].format = '.1e'
        
def make_html_table(engine=None, columns=['root','status','id','p_ra','p_dec','mag_auto','flux_radius','z_spec','z_map','bic_diff','chinu','log_pdf_max', 'd4000', 'd4000_e'], where="AND status >= 5 AND root='j163852p4039'", tables=[], table_root='query', sync='s3://grizli-v1/tables/', png_ext=['R30','stack','full','line'], sort_column=('bic_diff',-1), verbose=True, get_sql=False, show_hist=False):
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
        
    query = "SELECT {0} FROM photometry_apcorr, redshift_fit{1} WHERE phot_root = p_root AND id = p_id {2};".format(','.join(columns), extra_tables, where)
    
    if get_sql:
        return query
        
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
        
    idx = ['<a href="http://vizier.u-strasbg.fr/viz-bin/VizieR?-c={0:.6f}+{1:.6f}&-c.rs=2">#{2:05d}</a>'.format(info['ra'][i], info['dec'][i], info['id'][i]) for i in range(len(info))]
    info['idx'] = idx
    all_columns.insert(0, 'idx')
    all_columns.pop(all_columns.index('id'))
    
    set_column_formats(info)
    
    print('Sort: ', sort_column, sort_column[0] in all_columns)
    if sort_column[0] in all_columns:
        so = np.argsort(info[sort_column[0]])
        info = info[so[::sort_column[1]]]
        
    ### PNG columns  
    AWS = 'https://s3.amazonaws.com/grizli-v1/Pipeline'  
    bucket = ['grizli-cosmos-v2' if r.startswith('cos-') else 'grizli-v1' for r in info['root']]
    
    for ext in png_ext:
        png = ['{0}_{1:05d}.{2}.png'.format(root, id, ext) for root, id in zip(info['root'], info['id'])]
        info['png_{0}'.format(ext)] = ['<a href="{0}/{1}/Extractions/{2}"><img src={0}/{1}/Extractions/{2} height=200></a>'.format(AWS.replace('grizli-v1',buck), root, p) for buck, root, p in zip(bucket, info['root'], png)]
        all_columns.append('png_{0}'.format(ext))
    
    sortable = []
    for c in all_columns:
        if not hasattr(info[c][0], 'upper'):
            sortable.append(c)
    
    info[all_columns].write_sortable_html('{0}.html'.format(table_root), replace_braces=True, localhost=False, max_lines=1e5, table_id=None, table_class='display compact', css=None, filter_columns=sortable, buttons=['csv'], toggle=True, use_json=True)
    
    if show_hist:
        from matplotlib.ticker import FixedLocator, AutoLocator, MaxNLocator
        xti = xt = np.arange(0,3.6,0.5)
        loc = np.arange(0, 3.6, 0.1)
        bins = utils.log_zgrid([0.03, 3.5], 0.01)
        fig = plt.figure(figsize=[8,4])
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
        ax.set_xlim(0,np.log(1+3.7))

        ax.grid()
        ax.legend(loc='upper right')

        fig.tight_layout(pad=0.1)
        fig.text(1-0.02, 0.02, time.ctime(), ha='right', va='bottom', transform=fig.transFigure, fontsize=5)
        
        fig.savefig('{0}_zhist.png'.format(table_root))
           
    if sync:
        os.system('aws s3 sync ./ {0} --exclude "*" --include "{1}.html" --include "{1}.json" --include "{1}_zhist.png" --acl public-read'.format(sync, table_root))
    
    return res

def show_all_fields():
    plt.ioff()
    res = pd.read_sql_query("select distinct root from redshift_fit order by root;", engine)
    roots = res['root'].tolist()
    
    for root in roots:
        print('\n\n', root, '\n\n')
        if os.path.exists('{0}_zhist.png'.format(root)):
            continue
            
        try:
            if False:
                res = pd.read_sql_query("select root,id,status from redshift_fit where root = '{0}';".format(root), engine)
                res = pd.read_sql_query("select status, count(status) as n from redshift_fit where root = '{0}' group by status;".format(root), engine)
            
            res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','t_g102', 't_g141', 'z_spec','z_map','bic_diff','chinu','zwidth1/(1+z_map) as zw1','dlinesn'], where="AND status > 4 AND root = '{0}'".format(root), table_root=root, sync='s3://grizli-v1/Pipeline/{0}/Extractions/'.format(root), png_ext=['R30', 'stack','full','line'], show_hist=True)
            
            if False:
                grizli_db.set_phot_root(root, phot_root, engine)
                
                res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','t_g102', 't_g141', 'z_spec','z_map','bic_diff','chinu','zwidth1/(1+z_map) as zw1','dlinesn'], where="AND status > 4 AND root = '{0}' AND (bic_diff > 20 OR zwidth1/(1+z_map) < 0.01)".format(root), table_root=root, sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line','sed'], show_hist=False)

                res = grizli_db.make_html_table(engine=engine, columns=['mtime', 'root','status','id','p_ra','p_dec','mag_auto','flux_radius','t_g800l','t_g102', 't_g141', 'z_spec','z_map','bic_diff','chinu','zwidth1/(1+z_map) as zw1','dlinesn'], where="AND status > 4 AND phot_root = '{0}' AND bic_diff > 20".format(phot_root), table_root=phot_root, sync='s3://grizli-v1/tables/', png_ext=['R30', 'stack','full','line'], show_hist=False)
                
        except:
            continue
            
        os.system('aws s3 cp s3://grizli-v1/Pipeline/{0}/Extractions/{0}_zhist.png s3://grizli-v1/tables/'.format(root))
        