"""
AWS utilities
"""
import os
import numpy as np

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
        size = np.cast[float](tab['size'])
        tab['size'] = [f'{s/1.e6:.2f}' for s in size]
        tab['date'] = ['{date} {time}'.format(**row) for row in tab]
        tab.remove_column('time')
        
        tab['size'] = np.cast[float](tab['size'])
        
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
        print(path.replace('s3://', 'https://s3.amazonaws.com/') + output_file)
    else:
        print(output_file)