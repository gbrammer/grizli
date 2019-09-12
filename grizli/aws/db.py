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

COLUMNS = ['root', 'id', 'status', 'ra', 'dec', 'ninput', 'redshift', 't_g102', 'n_g102', 'p_g102', 't_g141', 'n_g141', 'p_g141', 't_g800l', 'n_g800l', 'p_g800l', 'numlines', 'haslines', 'chi2poly', 'chi2spl', 'splf01', 'sple01', 'splf02', 'sple02', 'splf03', 'sple03', 'splf04', 'sple04', 'huberdel', 'st_df', 'st_loc', 'st_scl', 'dof', 'chimin', 'chimax', 'bic_poly', 'bic_spl', 'bic_temp', 'z02', 'z16', 'z50', 'z84', 'z97', 'zwidth1', 'zwidth2', 'z_map', 'z_risk', 'min_risk', 'd4000', 'd4000_e', 'dn4000', 'dn4000_e', 'dlineid', 'dlinesn', 'flux_pab', 'err_pab', 'ew50_pab', 'ewhw_pab', 'flux_hei_1083', 'err_hei_1083', 'ew50_hei_1083', 'ewhw_hei_1083', 'flux_siii', 'err_siii', 'ew50_siii', 'ewhw_siii', 'flux_oii_7325', 'err_oii_7325', 'ew50_oii_7325', 'ewhw_oii_7325', 'flux_ariii_7138', 'err_ariii_7138', 'ew50_ariii_7138', 'ewhw_ariii_7138', 'flux_sii', 'err_sii', 'ew50_sii', 'ewhw_sii', 'flux_ha', 'err_ha', 'ew50_ha', 'ewhw_ha', 'flux_oi_6302', 'err_oi_6302', 'ew50_oi_6302', 'ewhw_oi_6302', 'flux_hei_5877', 'err_hei_5877', 'ew50_hei_5877', 'ewhw_hei_5877', 'flux_oiii', 'err_oiii', 'ew50_oiii', 'ewhw_oiii', 'flux_hb', 'err_hb', 'ew50_hb', 'ewhw_hb', 'flux_oiii_4363', 'err_oiii_4363', 'ew50_oiii_4363', 'ewhw_oiii_4363', 'flux_hg', 'err_hg', 'ew50_hg', 'ewhw_hg', 'flux_hd', 'err_hd', 'ew50_hd', 'ewhw_hd', 'flux_h7', 'err_h7', 'ew50_h7', 'ewhw_h7', 'flux_h8', 'err_h8', 'ew50_h8', 'ewhw_h8', 'flux_h9', 'err_h9', 'ew50_h9', 'ewhw_h9', 'flux_h10', 'err_h10', 'ew50_h10', 'ewhw_h10', 'flux_neiii_3867', 'err_neiii_3867', 'ew50_neiii_3867', 'ewhw_neiii_3867', 'flux_oii', 'err_oii', 'ew50_oii', 'ewhw_oii', 'flux_nevi_3426', 'err_nevi_3426', 'ew50_nevi_3426', 'ewhw_nevi_3426', 'flux_nev_3346', 'err_nev_3346', 'ew50_nev_3346', 'ewhw_nev_3346', 'flux_mgii', 'err_mgii', 'ew50_mgii', 'ewhw_mgii', 'flux_civ_1549', 'err_civ_1549', 'ew50_civ_1549', 'ewhw_civ_1549', 'flux_ciii_1908', 'err_ciii_1908', 'ew50_ciii_1908', 'ewhw_ciii_1908', 'flux_oiii_1663', 'err_oiii_1663', 'ew50_oiii_1663', 'ewhw_oiii_1663', 'flux_heii_1640', 'err_heii_1640', 'ew50_heii_1640', 'ewhw_heii_1640', 'flux_niii_1750', 'err_niii_1750', 'ew50_niii_1750', 'ewhw_niii_1750', 'flux_niv_1487', 'err_niv_1487', 'ew50_niv_1487', 'ewhw_niv_1487', 'flux_nv_1240', 'err_nv_1240', 'ew50_nv_1240', 'ewhw_nv_1240', 'flux_lya', 'err_lya', 'ew50_lya', 'ewhw_lya', 'pdf_max', 'cdf_z', 'sn_pab', 'sn_hei_1083', 'sn_siii', 'sn_oii_7325', 'sn_ariii_7138', 'sn_sii', 'sn_ha', 'sn_oi_6302', 'sn_hei_5877', 'sn_oiii', 'sn_hb', 'sn_oiii_4363', 'sn_hg', 'sn_hd', 'sn_h7', 'sn_h8', 'sn_h9', 'sn_h10', 'sn_neiii_3867', 'sn_oii', 'sn_nevi_3426', 'sn_nev_3346', 'sn_mgii', 'sn_civ_1549', 'sn_ciii_1908', 'sn_oiii_1663', 'sn_heii_1640', 'sn_niii_1750', 'sn_niv_1487', 'sn_nv_1240', 'sn_lya', 'chinu', 'bic_diff', 'log_risk', 'log_pdf_max', 'zq', 'idx', 'png_stack', 'png_full', 'png_line', 'png_rgb', 'mtime', 'vel_bl', 'vel_nl', 'vel_z', 'vel_nfev', 'vel_flag', 'grizli_version']
         
def get_connection_info():
    """
    Read the database connection info
    """
    import yaml
    
    fp = open(os.path.join(os.path.dirname(__file__), '../data/db.yml'))
    try:
        db_info = yaml.load(fp, Loader=yaml.FullLoader)
    except:
        db_info = yaml.load(fp)
        
    fp.close()
    
    return db_info

def get_db_engine(echo=False):
    """
    Generate an SQLAlchemy engine for the grizli database
    """
    from sqlalchemy import create_engine
    db_info = get_connection_info()
    
    db_string = "postgresql://{0}:{1}@{2}:{3}/{4}".format(db_info['username'], db_info['password'], db_info['hostname'], db_info['port'], db_info['database'])
    engine = create_engine(db_string, echo=echo)
    return engine
    
def get_redshift_fit_status(root, id, engine=None):
    import pandas as pd
    
    if engine is None:
        engine = get_db_engine(echo=False)
    
    res = pd.read_sql_query("SELECT * FROM redshift_fit", engine)
    
    res = pd.read_sql_query("SELECT status FROM redshift_fit WHERE (root = '{0}' AND id = {1})".format(root, id), engine)
    
    if len(res) == 0:
        return -1
    else:
        return res['status'][0]

def update_redshift_fit_status(root, id, status=0, engine=None, verbose=True):
    """
    Set the status flag in the table
    """
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
    
    tab = Table.read(rowfile, character_as_bytes=False)
    
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
    
        