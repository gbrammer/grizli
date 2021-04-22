"""
Summary catalog from fitting outputs
"""

import os
import glob

from .. import prep, utils


def set_column_formats(fit):
    """
    Set print formats for the master catalog columns
    """

    for c in ['ra', 'dec']:
        fit[c].format = '.5f'

    for col in ['MAG_AUTO', 'FLUX_RADIUS', 'A_IMAGE', 'ellipticity']:
        if col.lower() in fit.colnames:
            fit[col.lower()].format = '.2f'

    for c in ['log_risk', 'log_pdf_max', 'zq', 'chinu', 'bic_diff']:
        fit[c].format = '.2f'

    for c in ['z_risk', 'z_map', 'z02', 'z16', 'z50', 'z84', 'z97']:
        fit[c].format = '.4f'

    for c in ['t_g102', 't_g141', 't_g800l']:
        if c in fit.colnames:
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


def summary_catalog(field_root='', dzbin=0.01, use_localhost=True, filter_bandpasses=None, files=None, cdf_sigmas=None, strip_empty_columns=False, **kwargs):
    """
    Make redshift histogram and summary catalog / HTML table
    """
    import os
    import time

    import numpy as np
    from matplotlib.ticker import FixedLocator
    import matplotlib.pyplot as plt

    import astropy.table

    try:
        from .. import fitting, prep, utils
        from . import auto_script
    except:
        from grizli import prep, utils, fitting
        from grizli.pipeline import auto_script

    if filter_bandpasses is None:
        import pysynphot as S
        filter_bandpasses = [S.ObsBandpass(bpstr) for bpstr in ['acs,wfc1,f814w', 'wfc3,ir,f105w', 'wfc3,ir,f110w', 'wfc3,ir,f125w', 'wfc3,ir,f140w', 'wfc3,ir,f160w']]

    if os.path.exists('{0}.info.fits'.format(field_root)):
        orig = utils.read_catalog('{0}.info.fits'.format(field_root))
        all_files = glob.glob('{0}*full.fits'.format(field_root))
        all_files.sort()

        print('{0}.info.fits: {1} objects.  Found {2} full.fits files, checking modify dates.'.format(field_root, len(orig), len(all_files)))
        info_mtime = os.stat('{0}.info.fits'.format(field_root)).st_mtime
        keep = np.ones(len(orig), dtype=bool)

        files = []
        for file in all_files:
            id = int(file.split('_')[1].split('.full')[0])
            if id not in orig['id']:
                files.append(file)
            else:
                full_mtime = os.stat(file).st_mtime
                if full_mtime > info_mtime:
                    files.append(file)
                    keep[orig['id'] == id] = False

        orig = orig[keep]

        if len(files) == 0:
            print('Found {0}.info.fits, and {1} new objects.\n'.format(field_root, len(files)))
            return False
        else:
            print('Found {0}.info.fits.  Adding {1} new objects.\n'.format(field_root, len(files)))

    else:
        orig = None

    # SUmmary catalog
    fit = fitting.make_summary_catalog(target=field_root, sextractor=None,
                                       filter_bandpasses=filter_bandpasses,
                                       files=files, cdf_sigmas=cdf_sigmas,
                                       write_table=(dzbin is not None))
    fit.meta['root'] = field_root

    if orig is not None:
        if len(fit) > 0:
            fit = astropy.table.vstack([orig, fit])
            if dzbin is not None:
                fit.write('{0}.info.fits'.format(field_root), overwrite=True)

    mtime = []
    for i in range(len(fit)):
        full_file = '{0}_{1:05d}.full.fits'.format(fit['root'][i], fit['id'][i])
        if os.path.exists(full_file):
            mtime.append(time.ctime(os.stat(full_file).st_mtime))
        else:
            mtime.append('-')

    fit['mtime'] = mtime

    # Add photometric catalog
    try:
        catalog = glob.glob('{0}-*.cat.fits'.format(field_root))[0]
        sex = utils.GTable.gread(catalog)
        # try:
        # except:
        #     sex = utils.GTable.gread('../Prep/{0}-ir.cat.fits'.format(field_root), sextractor=True)

        idx = np.arange(len(sex))
        sex_idx = np.array([idx[sex['NUMBER'] == id][0] for id in fit['id']])

        fit['ellipticity'] = (sex['B_IMAGE']/sex['A_IMAGE'])[sex_idx]

        for col in ['MAG_AUTO', 'FLUX_RADIUS', 'A_IMAGE']:
            fit[col.lower()] = sex[col][sex_idx]
    except:
        pass

    fit = set_column_formats(fit)

    if strip_empty_columns:
        # Remove float columns with only NaN values
        #print('Strip empty columns')
        empty_cols = []
        for c in fit.colnames:
            try:
                isfin = np.isfinite(fit[c])
                if isfin.sum() == 0:
                    empty_cols.append(c)
            except:
                pass

        for c in empty_cols:
            fit.remove_column(c)

    # Just return the table if dzbin parameter not specified
    if dzbin is None:
        return fit

    # Overwrite with additional sextractor keywords
    fit.write('{0}.info.fits'.format(field_root), overwrite=True)

    clip = (fit['chinu'] < 2.0) & (fit['log_risk'] < -1)
    clip = (fit['chinu'] < 2.0) & (fit['zq'] < -3) & (fit['zwidth1']/(1+fit['z_map']) < 0.005)
    clip &= fit['bic_diff'] > 30  # -40

    bins = utils.log_zgrid(zr=[0.1, 3.5], dz=dzbin)

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)

    ax.hist(np.log10(1+fit['z_map']), bins=np.log10(1+bins), alpha=0.2, color='k')
    ax.hist(np.log10(1+fit['z_map'][clip]), bins=np.log10(1+bins), alpha=0.8)

    xt = np.array(np.arange(0.25, 3.55, 0.25))
    ax.xaxis.set_minor_locator(FixedLocator(np.log10(1+xt)))
    xt = np.array([1, 2, 3])
    ax.set_xticks(np.log10(1+xt))
    ax.set_xticklabels(xt)

    ax.set_xlabel('z')
    ax.set_ylabel(r'$N$')

    ax.grid()
    ax.text(0.05, 0.95, field_root, ha='left', va='top', transform=ax.transAxes)

    fig.tight_layout(pad=0.2)
    fig.savefig('{0}_zhist.png'.format(field_root))

    cols = ['root', 'mtime', 'idx', 'ra', 'dec', 'mag_auto', 't_g800l', 't_g102', 't_g141', 'z_map', 'chinu', 'bic_diff', 'zwidth1', 'd4000', 'd4000_e', 'png_stack', 'png_full', 'png_rgb', 'png_line']

    for i in range(len(cols))[::-1]:
        if cols[i] not in fit.colnames:
            cols.pop(i)

    filter_columns = ['ra', 'dec', 'mag_auto', 't_g800l', 't_g102', 't_g141', 'z_map', 'chinu', 'bic_diff', 'zwidth1', 'd4000', 'd4000_e']

    fit[cols].write_sortable_html(field_root+'-fit.html', replace_braces=True, localhost=use_localhost, max_lines=50000, table_id=None, table_class='display compact', css=None, filter_columns=filter_columns, use_json=(not use_localhost))

    fit[cols][clip].write_sortable_html(field_root+'-fit.zq.html', replace_braces=True, localhost=use_localhost, max_lines=50000, table_id=None, table_class='display compact', css=None, filter_columns=filter_columns, use_json=(not use_localhost))

    zstr = ['{0:.3f}'.format(z) for z in fit['z_map'][clip]]
    prep.table_to_regions(fit[clip], output=field_root+'-fit.zq.reg', comment=zstr)

    # if False:
    # 
    #     fit = utils.GTable.gread('{0}.info.fits'.format(root))
    #     fit = auto_script.set_column_formats(fit)
    # 
    #     cols = ['id', 'ra', 'dec', 'mag_auto', 't_g102', 't_g141', 'sn_Ha', 'sn_OIII', 'sn_Hb', 'z_map', 'log_risk', 'log_pdf_max', 'zq', 'chinu', 'bic_diff', 'zwidth1', 'png_stack', 'png_full', 'png_line']
    # 
    #     #clip = ((fit['sn_Ha'] > 5) | (fit['sn_OIII'] > 5)) & (fit['bic_diff'] > 50) & (fit['chinu'] < 2)
    #     #clip = (fit['sn_OIII'] > 5) & (fit['bic_diff'] > 100) & (fit['chinu'] < 3)
    # 
    #     test_line = {}
    #     for li in ['Ha', 'OIII', 'OII']:
    #         test_line[l] = (fit['sn_'+li] > 5) & (fit['err_'+li] < 1.e-16)
    # 
    #     clip = (test_line['Ha'] | test_line['OIII'] | test_line['OII']) & (fit['bic_diff'] > 50) & (fit['chinu'] < 2)
    # 
    #     star = fit['flux_radius'] < 2.3
    #     clip &= ~star
    # 
    #     jh = fit['mag_wfc3,ir,f125w'] - fit['mag_wfc3,ir,f160w']
    #     clip = (fit['chinu'] < 2) & (jh > 0.9) & (fit['mag_wfc3,ir,f160w'] < 23)
    #     fit['jh'] = jh
    #     fit['jh'].format = '.1f'
    # 
    #     fit['dmag'] = fit['mag_wfc3,ir,f140w'] - fit['mag_auto']
    #     fit['dmag'].format = '.1f'
    # 
    #     cols = ['idx', 'ra', 'dec', 'mag_auto', 'jh', 'dmag', 't_g141', 'sn_Ha', 'sn_OIII', 'sn_Hb', 'z_map', 'log_risk', 'log_pdf_max', 'zq', 'chinu', 'bic_diff', 'zwidth1', 'png_stack', 'png_full', 'png_line']
    # 
    #     fit[cols][clip].write_sortable_html(root+'-fit.lines.html', replace_braces=True, localhost=False, max_lines=50000, table_id=None, table_class='display compact', css=None)
