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

COLUMNS = ['root', 'id', 'status', 'ra', 'dec', 'ninput', 'redshift', 't_g102', 'n_g102', 'p_g102', 't_g141', 'n_g141', 'p_g141', 't_g800l', 'n_g800l', 'p_g800l', 'numlines', 'haslines', 'chi2poly', 'chi2spl', 'splf01', 'sple01', 'splf02', 'sple02', 'splf03', 'sple03', 'splf04', 'sple04', 'huberdel', 'st_df', 'st_loc', 'st_scl', 'dof', 'chimin', 'chimax', 'bic_poly', 'bic_spl', 'bic_temp', 'z02', 'z16', 'z50', 'z84', 'z97', 'zwidth1', 'zwidth2', 'z_map', 'z_risk', 'min_risk', 'd4000', 'd4000_e', 'dn4000', 'dn4000_e', 'dlineid', 'dlinesn', 'flux_pab', 'err_pab', 'ew50_pab', 'ewhw_pab', 'flux_hei_1083', 'err_hei_1083', 'ew50_hei_1083', 'ewhw_hei_1083', 'flux_siii', 'err_siii', 'ew50_siii', 'ewhw_siii', 'flux_oii_7325', 'err_oii_7325', 'ew50_oii_7325', 'ewhw_oii_7325', 'flux_ariii_7138', 'err_ariii_7138', 'ew50_ariii_7138', 'ewhw_ariii_7138', 'flux_sii', 'err_sii', 'ew50_sii', 'ewhw_sii', 'flux_ha', 'err_ha', 'ew50_ha', 'ewhw_ha', 'flux_oi_6302', 'err_oi_6302', 'ew50_oi_6302', 'ewhw_oi_6302', 'flux_hei_5877', 'err_hei_5877', 'ew50_hei_5877', 'ewhw_hei_5877', 'flux_oiii', 'err_oiii', 'ew50_oiii', 'ewhw_oiii', 'flux_hb', 'err_hb', 'ew50_hb', 'ewhw_hb', 'flux_oiii_4363', 'err_oiii_4363', 'ew50_oiii_4363', 'ewhw_oiii_4363', 'flux_hg', 'err_hg', 'ew50_hg', 'ewhw_hg', 'flux_hd', 'err_hd', 'ew50_hd', 'ewhw_hd', 'flux_h7', 'err_h7', 'ew50_h7', 'ewhw_h7', 'flux_h8', 'err_h8', 'ew50_h8', 'ewhw_h8', 'flux_h9', 'err_h9', 'ew50_h9', 'ewhw_h9', 'flux_h10', 'err_h10', 'ew50_h10', 'ewhw_h10', 'flux_neiii_3867', 'err_neiii_3867', 'ew50_neiii_3867', 'ewhw_neiii_3867', 'flux_oii', 'err_oii', 'ew50_oii', 'ewhw_oii', 'flux_nevi_3426', 'err_nevi_3426', 'ew50_nevi_3426', 'ewhw_nevi_3426', 'flux_nev_3346', 'err_nev_3346', 'ew50_nev_3346', 'ewhw_nev_3346', 'flux_mgii', 'err_mgii', 'ew50_mgii', 'ewhw_mgii', 'flux_civ_1549', 'err_civ_1549', 'ew50_civ_1549', 'ewhw_civ_1549', 'flux_ciii_1908', 'err_ciii_1908', 'ew50_ciii_1908', 'ewhw_ciii_1908', 'flux_oiii_1663', 'err_oiii_1663', 'ew50_oiii_1663', 'ewhw_oiii_1663', 'flux_heii_1640', 'err_heii_1640', 'ew50_heii_1640', 'ewhw_heii_1640', 'flux_niii_1750', 'err_niii_1750', 'ew50_niii_1750', 'ewhw_niii_1750', 'flux_niv_1487', 'err_niv_1487', 'ew50_niv_1487', 'ewhw_niv_1487', 'flux_nv_1240', 'err_nv_1240', 'ew50_nv_1240', 'ewhw_nv_1240', 'flux_lya', 'err_lya', 'ew50_lya', 'ewhw_lya', 'pdf_max', 'cdf_z', 'sn_pab', 'sn_hei_1083', 'sn_siii', 'sn_oii_7325', 'sn_ariii_7138', 'sn_sii', 'sn_ha', 'sn_oi_6302', 'sn_hei_5877', 'sn_oiii', 'sn_hb', 'sn_oiii_4363', 'sn_hg', 'sn_hd', 'sn_h7', 'sn_h8', 'sn_h9', 'sn_h10', 'sn_neiii_3867', 'sn_oii', 'sn_nevi_3426', 'sn_nev_3346', 'sn_mgii', 'sn_civ_1549', 'sn_ciii_1908', 'sn_oiii_1663', 'sn_heii_1640', 'sn_niii_1750', 'sn_niv_1487', 'sn_nv_1240', 'sn_lya', 'chinu', 'bic_diff', 'log_risk', 'log_pdf_max', 'zq', 'mtime', 'vel_bl', 'vel_nl', 'vel_z', 'vel_nfev', 'vel_flag', 'grizli_version']
         
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
        
def run_lambda_fits(root='j004404m2034', mag_limits=[15, 26], sn_limit=7, min_status=None):
    """
    Run redshift fits on lambda for a given root
    """
    from grizli.aws import fit_redshift_lambda
    from grizli import utils
    from grizli.aws import db as grizli_db
    engine = grizli_db.get_db_engine()
    
    import pandas as pd
    import numpy as np
    import glob
    import os
    
    os.system('aws s3 sync s3://grizli-v1/Pipeline/{0}/Extractions/ ./ --exclude "*" --include "*_phot*.fits"'.format(root))

    os.system('aws s3 sync s3://grizli-v1/Pipeline/{0}/Extractions/ ./ --exclude "*" --include "*_phot*.fits" --include "*wcs.fits"'.format(root))
    
    phot = utils.read_catalog('{0}_phot_apcorr.fits'.format(root))
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
        res = pd.read_sql_query("SELECT root, id, status FROM redshift_fit WHERE root = '{0}' AND status < {1}".format(root, min_status), engine)
        if len(res) > 0:
            status = phot['id']*0-100
            status[res['id']-1] = res['status']
            sel &= status < min_status
        
    ids = phot['id'][sel]
    
    fit_redshift_lambda.fit_lambda(root=root, beams=[], ids=ids, newfunc=False, bucket_name='grizli-v1', skip_existing=False, sleep=False, skip_started=False, quasar_fit=False, output_path=None, show_event=False, zr=[0.01,3.4], force_args=True)
    
    grizli_db.add_phot_to_db(root, delete=False, engine=engine)
    
    if False:
        res = pd.read_sql_query("SELECT root, id, status, redshift, bic_diff, mtime FROM redshift_fit WHERE (root = '{0}')".format(root), engine)
        
        # Get arguments
        args = fit_redshift_lambda.fit_lambda(root=root, beams=[], ids=ids, newfunc=False, bucket_name='grizli-v1', skip_existing=False, sleep=False, skip_started=False, quasar_fit=False, output_path=None, show_event=2, zr=[0.01,3.4], force_args=True)

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
    
    df = phot.to_pandas()
    return df
    
def add_phot_to_db(root, delete=True, engine=None):
    """
    Read the table {root}_phot_apcorr.fits and append it to the grizli_db `photometry_apcorr` table
    """
    import pandas as pd
    from astropy.table import Table
    from grizli.aws import db as grizli_db
    import numpy as np
    
    if engine is None:
        engine = grizli_db.get_db_engine(echo=False)
        
    res = pd.read_sql_query("SELECT root, id FROM photometry_apcorr WHERE root = '{0}'".format(root), engine)
    if len(res) > 0:
        if delete:
            print('Delete rows where root={0}'.format(root))
            res = engine.execute("DELETE from photometry_apcorr WHERE (root = '{0}')".format(root))
        else:
            print('Data found for root={0}, delete them if necessary'.format(root))
            return False
    
    # Read the catalog
    phot = Table.read('{0}_phot_apcorr.fits'.format(root), character_as_bytes=False)
    
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
    df.to_sql('photometry_apcorr', engine, index=False, if_exists='append', method='multi')
    
def test_join():
    import pandas as pd
    
    res = pd.read_sql_query("SELECT p.root, p.id, flux_radius, mag_auto, z_map, status, bic_diff, zwidth1, log_pdf_max, chinu FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit WHERE z_map > 0) z ON (p.root = z.root AND p.id = z.id)".format(root), engine)        

    res = pd.read_sql_query("SELECT * FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit WHERE z_map > 0) z ON (p.root = z.root AND p.id = z.id)".format(root), engine)        
    
    # on root
    res = pd.read_sql_query("SELECT p.root, p.id, mag_auto, z_map, status FROM photometry_apcorr AS p JOIN (SELECT * FROM redshift_fit WHERE root='{0}') z ON (p.root = z.root AND p.id = z.id)".format(root), engine)        

        