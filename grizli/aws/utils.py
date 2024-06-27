"""
AWS utilities
"""
import os
import numpy as np
from .. import utils

def create_s3_index(path, output_file="index.html", extra_html="", upload=True, as_table=True, with_dja_css=True, show_images=True):
    """
    Create HTML listing of an S3 path
    """
    import time
    from .. import utils as grizli_utils
    
    url = path.replace('s3://', 'https://s3.amazonaws.com/')

    lsfile = output_file.replace('.html','.ls')

    # Sort by filename
    sort_str = 'sort -k 4'

    os.system(f'aws s3 ls {path} | {sort_str} |grep -v PRE |grep -v {output_file} > {lsfile}')
    
    now = time.ctime()
    html=f"<h3>{now}</h3>\n"

    html += extra_html

    html += "\n<pre>\n"

    lines=open(lsfile).readlines()

    rows = []

    for line in lines:
        lsp = line.split()
        if len(lsp) == 4:
            html += "{0} {1} {2:>10} <a href={4}{3} > {3} </a>\n".format(*lsp, url)
            rows.append(lsp)
            
    html += '</pre>\n'
    
    if as_table:
        names = ['date', 'time', 'size', 'file']
        tab = grizli_utils.GTable(rows=rows, names=names)
        size = np.asarray(tab['size'],dtype=float)
        tab['size'] = [f'{s/1.e6:.2f}' for s in size]
        tab['date'] = ['{date} {time}'.format(**row) for row in tab]
        tab.remove_column('time')
        
        tab['size'] = np.asarray(tab['size'],dtype=float)
        
        dl = []
        for file in tab['file']:
            _uf = f'{url}{file}'
            if (file.split('.')[-1] in ['png','jpg']) & show_images:
                dl.append(f'<a href="{_uf}" /> <img src="{_uf}" height=200px /> </a> ')
            else:
                dl.append(f'<a href="{_uf}" > {_uf} </a>')
        
        tab['download'] = dl
        
        tab.write_sortable_html(output_file,
                                use_json=False,
                                localhost=False,
                                max_lines=1e6,
                                with_dja_css=with_dja_css,
                                toggle=False,
                                #filter_columns=['size'],
                                buttons=['csv'],
                                )
        
        tab.write(output_file.replace('.html', '.csv'), overwrite=True)
        
        if extra_html:
            with open(output_file) as fp:
                lines = fp.readlines()
            
            for i, line in enumerate(lines):
                if '<table' in line:
                    lines.insert(i, extra_html)
                    break
            
            with open(output_file,'w') as fp:
                fp.writelines(lines)
                
    else:

        fp = open(output_file,'w')
        fp.write(html)
        fp.close()
    
    if upload:
        os.system(f'aws s3 cp {output_file} {path} --acl public-read')
        os.system(f'aws s3 cp {output_file.replace(".html", ".csv")} {path} ' + 
                  ' --acl public-read')
        print(path.replace('s3://', 'https://s3.amazonaws.com/') + output_file)
    else:
        print(output_file)


def generate_program_codes(write=True):
    """
    Generate program codes tables from DB query
    """
    from . import db
    
    hst = db.SQL("""select proposal_id, max(proposal_pi) as PI, count(proposal_id), max(substr("dataURL", 19, 3)) as prog_code
    from assoc_table
    where instrument_name in ('WFC3/IR','ACS/WFC','WFC3/UVIS')
    group by proposal_id
    order by CAST(proposal_id as INT)
    """)

    jw = db.SQL("""select proposal_id, min(proposal_pi) as PI, count(proposal_id)
    from assoc_table
    where instrument_name in ('NIRISS','NIRCAM','MIRI')
    group by proposal_id
    order by CAST(proposal_id as INT)
    """)
    
    jw['pi'][jw['proposal_id'] == '2514'] = 'Williams, Christina'
    
    if write:
        hst.write('/tmp/hst_program_codes.csv', overwrite=True)
        jw.write('/tmp/jwst_program_codes.csv', overwrite=True)
    
    return hst, jw

def program_ids_in_mosaic(wcs_file='https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/gds-grizli-v7.0-f850lp_wcs.csv', progs_list=None, return_raw_list=False):
    """
    Generate a summary of observing programs that contribute to a particular grizli mosaic
    
    Parameters
    ----------
    wcs_file : str
        Filename or URL of a wcs summary file produced by `~grizli.aws.visit_processor.cutout_mosaic`
    
    progs_list : list
        List of parsed programs from ``return_raw_list=True``, e.g., for concatenating
        multiple lists into a single table
    
    return_raw_list : bool
        Just return the list of parsed programs for each exposure
    
    Returns
    -------
    tab : list, `~astropy.table.Table`
        List of parsed programs or a full table with program summary
    
    """
    # Read summary tables
    #--------------------
    data_path = os.path.join(os.path.dirname(__file__), '../data')
    
    # HST
    #-------------
    hst_file = os.path.join(data_path, 'hst_program_codes.csv')
    if not os.path.exists(hst_file):
        hst_file = 'https://dawn-cph.github.io/dja/data/hst_program_codes.csv'
        
    hst = utils.read_catalog(hst_file)
    hst_programs = {}
    hst_pi = {}
    for row in hst:
        hst_programs[row['prog_code']] = row['proposal_id']
        hst_pi[str(row['proposal_id'])] = row['pi']
    
    # JWST
    #--------------
    jw_file = os.path.join(data_path, 'jwst_program_codes.csv')
    if not os.path.exists(jw_file):
        jw_file = 'https://dawn-cph.github.io/dja/data/jwst_program_codes.csv'
    
    jw = utils.read_catalog(jw_file)
    jwst_pi = {}
    for row in jw:
        jwst_pi[str(row['proposal_id'])] = row['pi']

    # Loop through file names and calculate program IDs
    #--------------------------------------------------
    if progs_list is None:
        
        # grizli WCS summary file, remote or local
        #------------------------------------------
        exp = utils.read_catalog(wcs_file)
        
        progs_list = []
        for file in np.unique(exp['file']):
            if file.startswith('jw'):
                progs_list.append(f'JWST-{file[3:7]}')
            else:
                progs_list.append(f'HST-{hst_programs[file[1:4]]}')
    
    if return_raw_list:
        return progs_list
        
    # Generate output table
    #----------------------
    un = utils.Unique(progs_list, verbose=False)
    
    url = 'https://www.stsci.edu/cgi-bin/get-proposal-info?id={0}&observatory={1}'
    rows = []
    for v in un.values:
        obs, prog = v.split('-')
        pi = hst_pi[prog] if obs == 'HST' else jwst_pi[prog]
        rows.append([int(prog), obs, un[v].sum(), pi, url.format(prog, obs)])
    
    tab = utils.GTable(rows=rows,
                       names=['program_id', 'observatory', 'count', 'pi', 'url'])
    
    so = np.argsort(tab['program_id'])
    tab = tab[so]
    tab.meta['wcs_file'] = wcs_file
    
    return tab