#!/bin/env python
import inspect
import os

import yaml

from collections import OrderedDict

"""
# shell
aws s3 ls s3://grizli-v1/Pipeline/j000200m5558/Prep/j000200m5558_visits.npy --request-payer requester

# Python
import boto3
s3 = boto3.resource('s3')
bkt = s3.Bucket('grizli-v1')
field = 'j000200m5558'
s3_file = '{0}_visits.npy'.format(field)
s3_path = 'Pipeline/{0}/Prep'.format(field)
bkt.download_file(s3_path+'/'+s3_file, s3_file,
              ExtraArgs={"RequestPayer": "requester"})
"""


class FilterDict(OrderedDict):
    meta = OrderedDict()

    @property
    def nfiles(self):
        """
        Count number of exposures
        """
        n = 0
        for k in self:
            n += len(self[k])
        return n

    @property
    def valid_filters(self):
        """
        Return a list of filters with N >= 1 files
        """
        valid = []
        for k in self:
            if len(self[k]) > 0:
                valid.append(k)
        return valid


def get_visit_files():
    import boto3
    from grizli.aws import db

    engine = db.get_db_engine()
    fields = db.from_sql("select field_root from charge_fields where log LIKE 'Finished%%'", engine=engine)

    s3 = boto3.resource('s3')
    bkt = s3.Bucket('grizli-v1')

    for i, field in enumerate(fields['field_root']):
        s3_file = '{0}_visits.npy'.format(field)
        if not os.path.exists(s3_file):
            s3_path = f'Pipeline/{field}/Prep'
            try:
                bkt.download_file(s3_path+'/'+s3_file, s3_file,
                              ExtraArgs={"RequestPayer": "requester"})
                print(i, s3_file)
            except:
                print(i, f'Download failed: {field}')
        else:
            print(f'Skip {field}')
    

def make_visit_fits():
    import glob
    import numpy as np
    from grizli import utils
    
    visit_files = glob.glob('[egu]*visits.npy')
    visit_files.sort()

    indiv_files = glob.glob('j*visits.npy')
    indiv_files.sort()

    visit_files += indiv_files

    for p in ['grizli-v1-19.12.04_visits.npy', 'grizli-v1-19.12.05_visits.npy', 'grizli-v1-20.10.12_visits.npy', 'grizli-cosmos-v2_visits.npy','grizli-v1-21.05.20_visits.npy']:
        if p in visit_files:
            visit_files.pop(visit_files.index(p))

    all_visits = []
    products = []

    extra_visits = ['candels-july2019_visits.npy', 'grizli-cosmos-v2_visits.npy']
    #extra_visits = ['candels-july2019_visits.npy', 'cosmos-dash-apr20_visits.npy']
    extra_visits = ['candels-july2019_visits.npy', 'cosmos-dash-dec06_visits.npy']
    
    for extra in extra_visits:

        extra_visits = np.load(extra, allow_pickle=True)[0]
        
        if 'cosmos-dash' in extra:
            extra_products = [v['product'] + '-'+v['files'][0][:6] for v in extra_visits]
        else:
            extra_products = [v['product'] for v in extra_visits]
        
        for i, p in enumerate(extra_products):
            if p not in products:
                parent = p.split('_')[0]
                #print(parent, p)
                v = extra_visits[i]
                v['parent'] = parent
                v['xproduct'] = v['product']
                v['parent_file'] = extra  # 'candels-july2019_visits.npy'
                all_visits.append(v)
                products.append(p)
            else:
                print('Skip: ', p, v['parent'])
                
    # COSMOS footprint
    cosmos_fp = None
    for i, v in enumerate(extra_visits):
        if v['product'].endswith('f814w'):
            print(v['product'])
            if cosmos_fp is None:
                cosmos_fp = v['footprint'].buffer(1.e-6)
            else:
                cosmos_fp = cosmos_fp.union(v['footprint'])

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
                all_visits.append(v)
                products.append(vprod)

    for v in all_visits:
        v['filter'] = v['product'].split('-')[-1]
        v['first'] = v['files'][0]

    # File dictionary
    all_files = []
    file_products = []

    for v in all_visits:
        all_files.extend(v['files'])
        file_products.extend([v['xproduct']]*len(v['files']))

    # duplicates?? seem to be in GOODS-S.
    # Exclude them in all but the first product that contains them for now
    if True:
        _un = np.unique(all_files, return_counts=True, return_index=True, return_inverse=True)
        un_file, un_index, un_inv, un_count = _un
        dup = un_count > 1
        dup_files = un_file[dup]
        for file in dup_files:
            prods = list(np.array(file_products)[np.array(all_files) == file])
            for prod in prods[1:]:
                i = products.index(prod)
                v = all_visits[i]
                j = v['files'].index(file)
                print(file, v['parent'], prod, i, j)
                pj = all_visits[i]['files'].pop(j)
                pj = all_visits[i]['footprints'].pop(j)
                if 'awspath' in all_visits[i]:
                    pj = all_visits[i]['awspath'].pop(j)

            #print(file, prods[-1])

    # WFC3/IR copied to "Exposures" paths in CANDELS fields
    for v in all_visits:
        if v['parent_file'] in ['grizli-cosmos-v2_visits.npy', 'cosmos-dash-apr20_visits.npy', 'cosmos-dash-dec06_visits.npy']:
            continue

        if v['parent_file'].startswith('j'):
            v['awspath'] = ['grizli-v1/Pipeline/{0}/Prep'.format(v['parent']) for f in v['files']]
        else:
            #print(v['parent_file'], v['awspath'][0])
            
            if v['filter'].startswith('f0') | v['filter'].startswith('f1'):
                # print(v['product'])
                v['awspath'] = ['grizli-v1/Exposures/{0}/{1}'.format(f[:4], f.split('_')[0]) for f in v['files']]

    # Empty visits, seems to be from duplicates above and mostly in CANDELS
    nexp = np.array([len(visit['files']) for visit in all_visits])
    for i in np.where(nexp == 0)[0][::-1]:
        v_i = all_visits.pop(i)
        print(i, v_i['product'])
        products.pop(i)

    tab = utils.GTable()

    for k in ['parent', 'product', 'filter', 'first']:
        tab[k] = [visit[k] for visit in all_visits]

    coo = np.array([np.array(visit['footprint'].centroid.xy).flatten() for visit in all_visits])
    tab['ra'] = coo[:, 0]
    tab['dec'] = coo[:, 1]
    tab['nexp'] = [len(visit['files']) for visit in all_visits]
    tab['bounds'] = [np.array(v['footprint'].bounds) for v in all_visits]

    root = 'candels-july2019'
    root = 'candels-sep2019'
    root = 'grizli-v1-19.12.04'
    root = 'grizli-v1-19.12.05'
    root = 'grizli-v1-20.10.12'
    root = 'grizli-v1-21.05.20'
    root = 'grizli-v1-21.12.18'

    tab.write(root+'_visits.fits', overwrite=True)
    np.save(root+'_visits.npy', [all_visits])

    # os.system('echo "# In https://s3.amazonaws.com/grizli-v1/Mosaics/" > candels-july2019.files.txt; ls candels-july2019* |grep -v files.txt >>  candels-july2019.files.txt')
    #os.system('aws s3 sync --exclude "*" --include "candels-july2019*" --include "grizli-v1-19.12.04*" ./ s3://grizli-v1/Mosaics/ --acl public-read')
    os.system('echo "# In https://s3.amazonaws.com/grizli-v1/Mosaics/" > {0}.files.txt; ls {0}* |grep -v files.txt >>  {0}.files.txt'.format(root))

    os.system('aws s3 sync --exclude "*" --include "{0}*" ./ s3://grizli-v1/Mosaics/ --acl public-read'.format(root))

    if False:
        from shapely.geometry import Point
        from grizli.aws import db
        engine = db.get_db_engine() 
        
        fields = db.from_sql("select field_root, a_wfc3 from charge_fields where log LIKE 'Finished%%'", engine=engine)
        
        candels = utils.column_values_in_list(tab['parent'], ['j141956p5255', 'j123656p6215', 'j033236m2748', 'j021732m0512', 'j100012p0210'])
        cosmos = np.array([v['footprint'].intersection(cosmos_fp).area > 0 for v in all_visits])

        extra = candels
        extra = ~(candels | cosmos)

        # Area
        filter_polys = {}

        filt = 'f160w'
        for filt in np.unique(tab['filter']):
            print(filt)
            if filt in filter_polys:
                print(filt)
                continue

            poly = None
            count = 0

            # Dec strips
            di = np.arange(-90, 91, 5)
            strips = []
            for i in range(len(di)-1):
                strip = (tab['dec'] > di[i]) & (tab['dec'] <= di[i+1]) & (tab['filter'] == filt)
                strip &= extra

                if strip.sum() == 0:
                    continue

                indices = np.arange(len(tab))[strip]

                poly = None
                for j in indices:
                    v = all_visits[j]
                    if v['filter'] != filt:
                        continue

                    # for fp in v['footprints']:
                    for fp in [v['footprint']]:
                        count += 1
                        #print(i, v['product'], count)
                        if poly is None:
                            poly = fp.buffer(1.e-6)
                        else:
                            poly = poly.union(fp.buffer(1.e-6))

                poly.dec = di[i]+2.5
                strips.append(poly)

            if len(strips) == 0:
                filter_polys[filt] = Point(0, 0).buffer(1.e-6)
                continue

            full = strips[0].buffer(1.e-6)
            for strip in strips[1:]:
                full = full.union(strip.buffer(1.e-6))

            filter_polys[filt] = full

        optical = filter_polys['f606w'].union(filter_polys['f814w'])
        optical = optical.union(filter_polys['f850lp'])
        optical = optical.union(filter_polys['f775w'])

        yband = filter_polys['f098m'].union(filter_polys['f105w'])

        visy = optical.union(yband)

        jband = filter_polys['f125w']
        jband = jband.union(filter_polys['f110w'])

        hband = filter_polys['f140w'].union(filter_polys['f160w'])

        filter_polys[r'$\mathrm{opt} = i_{775} | i_{814} | z_{850}$'] = optical
        filter_polys[r'$\mathrm{opty} = \mathrm{opt} | Y$'] = visy
        filter_polys[r'$Y = y_{098 } | y_{105}$'] = yband
        filter_polys[r'$J = j_{110} | j_{125}$'] = jband
        filter_polys[r'$H = h_{140} | h_{160}$'] = hband

        ydrop = visy.intersection(jband)
        ydrop = ydrop.intersection(hband)
        filter_polys[r'$Y-\mathrm{drop} = (\mathrm{opt} | Y) + J + H$'] = ydrop

        yj = yband.union(jband)
        jdrop = yj.intersection(hband)
        filter_polys[r'$J-\mathrm{drop} = (Y | J) + H$'] = jdrop

        for filt in filter_polys:
            full = filter_polys[filt]
            try:
                areas = [f.area*np.cos(np.array(f.centroid.xy).flatten()[1]/180*np.pi) for f in full]
            except:
                try:
                    areas = [f.area*np.cos(np.array(f.centroid.xy).flatten()[1]/180*np.pi) for f in [full]]
                except:
                    areas = [0]

            full.total_area = np.sum(areas)
            print(filt, filter_polys[filt].total_area)

        ta = utils.GTable()
        ta['filter'] = [f.upper() for f in filter_polys]
        ta['area'] = [filter_polys[f].total_area*3600 for f in filter_polys]
        ta['area'].format = '.0f'

        # Compare areas
        h = fields['a_wfc3_ir_f160w'] > 0
        for root, aa in zip(fields['field_root'][h], fields['a_wfc3_ir_f160w'][h]):
            sel = (tab['filter'] == 'f160w') & (tab['parent'] == root)
            if sel.sum() > 0:
                indices = np.where(sel)[0]
                a = all_visits[indices[0]]['footprint'].buffer(1.e-6)
                for i in indices:
                    a = a.union(all_visits[i]['footprint'])

                a_i = a.area*3600*np.cos(tab['dec'][indices[0]]/180*np.pi)
                print(root, aa, a_i, a_i/aa)

def update_s3_paths():
    """
    Update paths to COSMOS data, eventually everything else
    """
    import numpy as np
    import os
    from tqdm import tqdm
    
    os.chdir(os.path.join(os.getenv('HOME'),
                     'Research/HST/CHArGE/Cutouts/VisitFiles'))
    
    groups = np.load('grizli-v1-21.12.18_filter_groups.npy', allow_pickle=True)
    
    for f in groups[0]:
        print(f)
        for j, aws in tqdm(enumerate(groups[0][f]['awspath'])):
            groups[0][f]['awspath'][j] = aws.replace('cosmos-dash',
                                                     'cosmos-dash-2021')
    
    np.save('grizli-v2-22.02.02_filter_groups.npy', groups)
                                     
def group_by_filter():
    """
    aws s3 sync --exclude "*" --include "cosmos_visits*" s3://grizli-preprocess/CosmosMosaic/ ./

    """
    from grizli import prep, utils
    import numpy as np

    master = 'cosmos'
    master = 'grizli-cosmos-v2'
    master = 'grizli-jan2019'
    master = 'grizli-v1-19.12.04'
    master = 'grizli-v1-19.12.05'
    master = 'grizli-v1-20.10.12'
    master = 'grizli-v1-21.05.20'
    master = 'grizli-v1-21.12.18'
    
    tab = utils.read_catalog('{0}_visits.fits'.format(master))
    all_visits = np.load('{0}_visits.npy'.format(master), allow_pickle=True)[0]

    # By filter

    # Exclude DASH
    dash = utils.column_string_operation(tab['product'], 'icxe', 'startswith')
    dash |= utils.column_string_operation(tab['product'], '_icxe',
                                         'count', 'or')

    # Don't exclude DASH
    dash = utils.column_string_operation(tab['product'], 'xxxx', 'startswith')

    groups = {}
    fpstr = {}

    for filt in np.unique(tab['filter']):
        mat = (tab['filter'] == filt) & (~dash)
        groups[filt] = {'filter': filt, 'files': [], 'awspath': [], 'footprints': []}
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

            for k in ['files', 'awspath', 'footprints']:
                groups[filt][k].extend(all_visits[ix][k])

        fp = open('{0}-pointings-{1}.reg'.format(master, filt), 'w')
        fp.write(fpstr[filt])
        fp.close()

        print('{0:6} {1:>3d} {2:>4d} ({3:>4d})'.format(filt, mat.sum(), len(groups[filt]['files']), len(np.unique(groups[filt]['files']))))

    np.save('{0}_filter_groups.npy'.format(master), [groups])

    os.system('aws s3 sync --exclude "*" --include "{0}*" ./ s3://grizli-v1/Mosaics/ --acl public-read'.format(master))

# RGB_PARAMS = {'xsize':4, 'rgb_min':-0.01, 'verbose':True, 'output_dpi': None, 'add_labels':False, 'output_format':'png', 'show_ir':False, 'scl':2, 'suffix':'.rgb', 'mask_empty':False}


RGB_PARAMS = {'xsize': 4,
              'output_dpi': None,
              'rgb_min': -0.01,
              'add_labels': False,
              'output_format': 'png',
              'show_ir': False,
              'scl': 2,
              'suffix': '.rgb',
              'mask_empty': False,
              'tick_interval': 1,
              'pl': 1,  # 1 for f_lambda, 2 for f_nu
              }

# xsize=4, output_dpi=None, HOME_PATH=None, show_ir=False, pl=1, pf=1, scl=1, rgb_scl=[1, 1, 1], ds9=None, force_ir=False, filters=all_filters, add_labels=False, output_format='png', rgb_min=-0.01, xyslice=None, pure_sort=False, verbose=True, force_rgb=None, suffix='.rgb', scale_ab=scale_ab)


def segmentation_figure(label, cat, segfile):
    """
    Make a figure showing a cutout of the segmentation file
    """
    import matplotlib.pyplot as plt
    import numpy as np

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from grizli import utils

    plt.ioff()

    seg = pyfits.open(segfile)
    seg_data = seg[0].data
    seg_wcs = pywcs.WCS(seg[0].header)

    # Randomize seg to get dispersion between neighboring objects
    np.random.seed(hash(label.split('_')[0]) % (10 ** 8))
    rnd_ids = np.append([0], np.argsort(np.random.rand(len(cat)))+1)

    # Make cutout
    th = pyfits.open('{0}.thumb.fits'.format(label), mode='update')
    th_wcs = pywcs.WCS(th[0].header)
    blot_seg = utils.blot_nearest_exact(seg_data, seg_wcs, th_wcs,
                               stepsize=-1, scale_by_pixel_area=False)

    rnd_seg = rnd_ids[np.cast[int](blot_seg)]*1.
    th_ids = np.unique(blot_seg)

    sh = th[0].data.shape
    yp, xp = np.indices(sh)

    thumb_height = 2.
    fig = plt.figure(figsize=[thumb_height*sh[1]/sh[0], thumb_height])
    ax = fig.add_subplot(111)
    rnd_seg[rnd_seg == 0] = np.nan

    ax.imshow(rnd_seg, aspect='equal', cmap='terrain_r',
              vmin=-0.05*len(cat), vmax=1.05*len(cat))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ix = utils.column_values_in_list(cat['number'], th_ids)
    xc, yc = th_wcs.all_world2pix(cat['ra'][ix], cat['dec'][ix], 0)
    xc = np.clip(xc, 0.09*sh[1], 0.91*sh[1])
    yc = np.clip(yc, 0.08*sh[0], 0.92*sh[0])

    for th_id, x_i, y_i in zip(cat['number'][ix], xc, yc):
        if th_id == 0:
            continue

        ax.text(x_i, y_i, '{0:.0f}'.format(th_id), ha='center', va='center', fontsize=8,  color='w')
        ax.text(x_i, y_i, '{0:.0f}'.format(th_id), ha='center', va='center', fontsize=8,  color='k', alpha=0.95)

    ax.set_xlim(0, sh[1]-1)
    ax.set_ylim(0, sh[0]-1)
    ax.set_axis_off()

    fig.tight_layout(pad=0.01)
    fig.savefig('{0}.seg.png'.format(label))
    plt.close(fig)

    # Append to thumbs file
    seg_hdu = pyfits.ImageHDU(data=np.cast[int](blot_seg), name='SEG')
    if 'SEG' in th:
        th.pop('SEG')

    th.append(seg_hdu)
    th.writeto('{0}.thumb.fits'.format(label), overwrite=True,
                 output_verify='fix')
    th.close()


def drizzle_images(label='macs0647-jd1', ra=101.9822125, dec=70.24326667, pixscale=0.1, size=10, wcs=None, pixfrac=0.33, kernel='square', theta=0, half_optical_pixscale=True, filters=['f160w', 'f140w', 'f125w', 'f105w', 'f110w', 'f098m', 'f850lp', 'f814w', 'f775w', 'f606w', 'f475w', 'f555w', 'f600lp', 'f390w', 'f350lp'], skip=None, remove=True, rgb_params=RGB_PARAMS, master='grizli-v1-19.12.04', aws_bucket='s3://grizli/CutoutProducts/', scale_ab=21, thumb_height=2.0, sync_fits=True, subtract_median=True, include_saturated=True, include_ir_psf=False, oversample_psf=False, show_filters=['visb', 'visr', 'y', 'j', 'h'], combine_similar_filters=True, single_output=True, aws_prep_dir=None, make_segmentation_figure=False, get_dict=False, dryrun=False, thumbnail_ext='png', **kwargs):
    """
    label='cp561356'; ra=150.208875; dec=1.850241667; size=40; filters=['f160w','f814w', 'f140w','f125w','f105w','f606w','f475w']

    master: These are sets of large lists of available exposures

        'cosmos': deprecated
        'grizli-cosmos-v2': All imaging covering the COSMOS field
        'candels-july2019': CANDELS fields other than COSMOS
        'grizli-v1': First processing of the Grizli CHArGE dataset
        'grizli-v1-19.12.04': Updated CHArGE fields
            ** this is now a copy from 21.05.20 so that the old lambda
               function can catch it **

        'grizli-v1-21.05.20': ACS fields + new cosmos

    """
    import glob
    import copy
    import os

    import numpy as np

    import astropy.io.fits as pyfits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from drizzlepac.adrizzle import do_driz

    import boto3

    from grizli import prep, utils
    from grizli.pipeline import auto_script

    # Function arguments
    if get_dict:
        frame = inspect.currentframe()
        args = inspect.getargvalues(frame).locals

        pop_args = ['get_dict', 'frame', 'kwargs']
        pop_classes = (np.__class__, do_driz.__class__, SkyCoord.__class__)

        for k in kwargs:
            args[k] = kwargs[k]

        for k in args:
            if isinstance(args[k], pop_classes):
                pop_args.append(k)

        for k in pop_args:
            if k in args:
                args.pop(k)

        return args

    # Boto objects
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')

    if isinstance(ra, str):
        coo = SkyCoord('{0} {1}'.format(ra, dec), unit=(u.hour, u.deg))
        ra, dec = coo.ra.value, coo.dec.value

    if label is None:
        try:
            import mastquery.utils
            label = mastquery.utils.radec_to_targname(ra=ra, dec=dec, round_arcsec=(1/15, 1), targstr='j{rah}{ram}{ras}{sign}{ded}{dem}{des}')
        except:
            label = 'grizli-cutout'

    #master = 'cosmos'
    #master = 'grizli-jan2019'

    if master == 'grizli-jan2019':
        parent = 's3://grizli/MosaicTools/'
        bkt = s3.Bucket('grizli')
    elif master == 'cosmos':
        parent = 's3://grizli-preprocess/CosmosMosaic/'
        bkt = s3.Bucket('grizli-preprocess')
    elif master == 'grizli-cosmos-v2':
        parent = 's3://grizli-cosmos-v2/Mosaics/'
        bkt = s3.Bucket('grizli-cosmos-v2')
    elif master == 'candels-july2019':
        parent = 's3://grizli-v1/Mosaics/'
        bkt = s3.Bucket('grizli-v1')
    elif master == 'grizli-v1-19.12.04':
        parent = 's3://grizli-v1/Mosaics/'
        bkt = s3.Bucket('grizli-v1')
    elif master == 'grizli-v1-19.12.05':
        parent = 's3://grizli-v1/Mosaics/'
        bkt = s3.Bucket('grizli-v1')
    elif master == 'grizli-v1-latest':
        parent = 's3://grizli-v1/Mosaics/'
        bkt = s3.Bucket('grizli-v1')        
    else:
        # Run on local files, e.g., "Prep" directory
        parent = None
        bkt = None
        #remove = False

    # Download summary files from S3
    for ext in ['_visits.fits', '_visits.npy', '_filter_groups.npy'][-1:]:
        newfile = '{0}{1}'.format(master, ext)
        if (not os.path.exists(newfile)) & (parent is not None):

            s3_path = parent.split('/')[-2]
            s3_file = '{0}{1}'.format(master, ext)
            print('{0}{1}'.format(parent, s3_file))
            bkt.download_file(s3_path+'/'+s3_file, s3_file,
                              ExtraArgs={"RequestPayer": "requester"})

            #os.system('aws s3 cp {0}{1}{2} ./'.format(parent, master, ext))

    #tab = utils.read_catalog('{0}_visits.fits'.format(master))
    #all_visits = np.load('{0}_visits.npy'.format(master))[0]
    if parent is not None:
        groups = np.load('{0}_filter_groups.npy'.format(master), allow_pickle=True)[0]
    else:

        if aws_prep_dir is not None:
            spl = aws_prep_dir.replace('s3://', '').split('/')
            prep_bucket = spl[0]
            prep_root = spl[2]

            prep_bkt = s3.Bucket(prep_bucket)

            s3_prep_path = 'Pipeline/{0}/Prep/'.format(prep_root)
            s3_full_path = '{0}/{1}'.format(prep_bucket, s3_prep_path)
            s3_file = '{0}_visits.npy'.format(prep_root)

            # Make output path Prep/../Thumbnails/
            if aws_bucket is not None:
                aws_bucket = ('s3://' +
                              s3_full_path.replace('/Prep/', '/Thumbnails/'))

            print('{0}{1}'.format(s3_prep_path, s3_file))
            if not os.path.exists(s3_file):
                prep_bkt.download_file(os.path.join(s3_prep_path, s3_file),
                            s3_file, ExtraArgs={"RequestPayer": "requester"})

            groups_files = glob.glob('{0}_filter_groups.npy'.format(prep_root))
            visit_query = prep_root+'_'
        else:
            groups_files = glob.glob('*filter_groups.*')
            visit_query = '*'

        # Reformat local visits.npy into a groups file
        if (len(groups_files) == 0):

            visit_file = glob.glob(visit_query+'visits.*')[0]
            visit_root = visit_file.split('_visits')[0]
            visits, groups, info = auto_script.load_visit_info(visit_root, 
                                                               verbose=False)

            #visits, groups, info = np.load(visit_file, allow_pickle=True)
            visit_filters = np.array([v['product'].split('-')[-1]
                                      for v in visits])
                                      
            groups = {}
            sgroups = {}
            
            for filt in np.unique(visit_filters):
                _v = {}
                _v['filter'] = str(filt)
                _v['files'] = []
                _v['footprints'] = []
                _v['awspath'] = []

                ix = np.where(visit_filters == filt)[0]
                for i in ix:
                    _v['files'].extend(visits[i]['files'])
                    _v['footprints'].extend(visits[i]['footprints'])

                Nf = len(_v['files'])
                print('{0:>6}: {1:>3} exposures'.format(filt, Nf))

                if aws_prep_dir is not None:
                    _v['awspath'] = [s3_full_path for file in range(Nf)]
            
                groups[filt] = _v 
                sgroups[str(filt)] = auto_script.visit_dict_to_strings(_v)
                
            #np.save('{0}_filter_groups.npy'.format(visit_root), [groups])
            with open('{0}_filter_groups.yaml'.format(visit_root), 'w') as fp:
                yaml.dump(sgroups, stream=fp, Dumper=yaml.Dumper)
        else:
            print('Use groups file: {0}'.format(groups_files[0]))

            #groups = np.load(groups_files[0], allow_pickle=True)[0]
            with open(groups_files[0]) as fp:
                groups = yaml.load(fp, Loader=yaml.Loader)
                for filt in groups:
                    _v = groups[filt]
                    groups[filt] = auto_script.visit_dict_from_strings(_v)
                    
    #filters = ['f160w','f814w', 'f110w', 'f098m', 'f140w','f125w','f105w','f606w', 'f475w']

    filt_dict = FilterDict()
    filt_dict.meta['label'] = label
    filt_dict.meta['ra'] = ra
    filt_dict.meta['dec'] = dec
    filt_dict.meta['size'] = size
    filt_dict.meta['master'] = master
    filt_dict.meta['parent'] = parent

    if filters is None:
        filters = list(groups.keys())

    has_filts = []
    lower_filters = [f.lower() for f in filters]
    for filt in lower_filters:
        if filt not in groups:
            continue

        visits = [copy.deepcopy(groups[filt])]
        #visits[0]['reference'] = 'CarlosGG/ak03_j1000p0228/Prep/ak03_j1000p0228-f160w_drz_sci.fits'

        visits[0]['product'] = label+'-'+filt

        if wcs is None:
            hdu = utils.make_wcsheader(ra=ra, dec=dec, size=size, pixscale=pixscale, get_hdu=True, theta=theta)

            h = hdu.header
        else:
            h = utils.to_header(wcs)

        if (filt[:2] in ['f0', 'f1', 'g1']) | (not half_optical_pixscale):
            #data = hdu.data
            pass
        else:
            for k in ['NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2']:
                h[k] *= 2

            h['CRPIX1'] -= 0.5
            h['CRPIX2'] -= 0.5

            for k in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
                if k in h:
                    h[k] /= 2

            #data = np.zeros((h['NAXIS2'], h['NAXIS1']), dtype=np.int16)

        #pyfits.PrimaryHDU(header=h, data=data).writeto('ref.fits', overwrite=True, output_verify='fix')
        #visits[0]['reference'] = 'ref.fits'

        print('\n\n###\nMake filter: {0}'.format(filt))

        if include_ir_psf:
            clean_i = False
        else:
            clean_i = remove

        status = utils.drizzle_from_visit(visits[0], h, pixfrac=pixfrac, kernel=kernel, clean=clean_i, include_saturated=include_saturated, skip=skip, dryrun=dryrun)

        if dryrun:
            filt_dict[filt] = status
            continue

        elif status is not None:
            sci, wht, outh, filt_dict[filt], wcs_tab = status

            if subtract_median:
                #med = np.median(sci[sci != 0])
                try:
                    un_data = np.unique(sci[(sci != 0) & np.isfinite(sci)])
                    med = utils.mode_statistic(un_data)
                except:
                    med = 0.

                if not np.isfinite(med):
                    med = 0.

                print('\n\nMedian {0} = {1:.3f}\n\n'.format(filt, med))
                outh['IMGMED'] = (med, 'Median subtracted from the image')
            else:
                med = 0.
                outh['IMGMED'] = (0., 'Median subtracted from the image')

            pyfits.writeto('{0}-{1}_drz_sci.fits'.format(label, filt),
                           data=sci, header=outh, overwrite=True,
                           output_verify='fix')

            pyfits.writeto('{0}-{1}_drz_wht.fits'.format(label, filt),
                           data=wht, header=outh, overwrite=True,
                           output_verify='fix')

            has_filts.append(filt)

            if include_ir_psf:
                from grizli.galfit.psf import DrizzlePSF

                hdu = pyfits.open('{0}-{1}_drz_sci.fits'.format(label, filt),
                                  mode='update')

                flt_files = []  # visits[0]['files']
                for i in range(1, 10000):
                    key = 'FLT{0:05d}'.format(i)
                    if key not in hdu[0].header:
                        break

                    flt_files.append(hdu[0].header[key])

                try:

                    dp = DrizzlePSF(flt_files=flt_files, driz_hdu=hdu[0])
                    
                    if oversample_psf:
                        oN = oversample_psf*2+1
                        cosd = np.cos(dp.driz_wcs.wcs.crval[1]/180*np.pi)
                        dde = 1./(oversample_psf*2)*pixscale/3600
                        dra = dde*cosd
                        sh = sci.shape
                        psfd = np.zeros((oN*sh[0], oN*sh[1]), 
                                        dtype=np.float32)
                        for i in range(oN):
                            for j in range(oN):
                                ra_i = (dp.driz_wcs.wcs.crval[0] +
                                        dra*(i-oversample_psf))
                                de_i = (dp.driz_wcs.wcs.crval[1] - 
                                        dde*(j-oversample_psf))
                                psf_i = dp.get_psf(ra=ra_i, dec=de_i,
                                         filter=filt.upper(),
                                         pixfrac=dp.driz_header['PIXFRAC'],
                                         kernel=dp.driz_header['KERNEL'],
                                         wcs_slice=dp.driz_wcs, 
                                get_extended=filt.lower()[:2] in ['f1','f0'],
                                         verbose=False, get_weight=False)
                                psfd[j::oN,i::oN] += psf_i[1].data
                        
                        psf = pyfits.ImageHDU(data=psfd)
                    else:
                        psf = dp.get_psf(ra=dp.driz_wcs.wcs.crval[0],
                                 dec=dp.driz_wcs.wcs.crval[1],
                                 filter=filt.upper(),
                                 pixfrac=dp.driz_header['PIXFRAC'],
                                 kernel=dp.driz_header['KERNEL'],
                                 wcs_slice=dp.driz_wcs, 
                                 get_extended=filt.lower()[:2] in ['f1','f0'],
                                 verbose=False, get_weight=False)[1]
                    
                    psf.header['OVERSAMP'] = oversample_psf
                    
                    psf.header['EXTNAME'] = 'PSF'
                    #psf[1].header['EXTVER'] = filt
                    hdu.append(psf)
                    hdu.flush()

                except:
                    pass

        if remove:
            os.system('rm *_fl*fits')

    # Dry run, just return dictionary of the found exposure files
    if dryrun:
        return filt_dict

    # Nothing found
    if len(has_filts) == 0:
        return []

    if combine_similar_filters:
        combine_filters(label=label)

    if rgb_params:
        #auto_script.field_rgb(root=label, HOME_PATH=None, filters=has_filts, **rgb_params)
        show_all_thumbnails(label=label, thumb_height=thumb_height, scale_ab=scale_ab, close=True, rgb_params=rgb_params, filters=show_filters, ext=thumbnail_ext)

    if (single_output != 0):
        # Concatenate into a single FITS file
        files = glob.glob('{0}-f*_dr[cz]_sci.fits'.format(label))
        files.sort()

        if combine_similar_filters:
            comb_files = glob.glob('{0}-[a-eg-z]*_dr[cz]_sci.fits'.format(label))
            comb_files.sort()
            files += comb_files

        hdul = None
        for file in files:
            hdu_i = pyfits.open(file)
            hdu_i[0].header['EXTNAME'] = 'SCI'
            if 'NCOMBINE' in hdu_i[0].header:
                if hdu_i[0].header['NCOMBINE'] <= single_output:
                    continue

                filt_i = file.split('-')[-1].split('_dr')[0]
            else:
                filt_i = utils.parse_filter_from_header(hdu_i[0].header)

            for h in hdu_i:
                h.header['EXTVER'] = filt_i
                if hdul is None:
                    hdul = pyfits.HDUList([h])
                else:
                    hdul.append(h)

            print('Add to {0}.thumb.fits: {1}'.format(label, file))

            # Weight
            hdu_i = pyfits.open(file.replace('_sci', '_wht'))
            hdu_i[0].header['EXTNAME'] = 'WHT'
            for h in hdu_i:
                h.header['EXTVER'] = filt_i
                if hdul is None:
                    hdul = pyfits.HDUList([h])
                else:
                    hdul.append(h)

        hdul.writeto('{0}.thumb.fits'.format(label), overwrite=True,
                     output_verify='fix')

        for file in files:
            for f in [file, file.replace('_sci', '_wht')]:
                if os.path.exists(f):
                    print('Remove {0}'.format(f))
                    os.remove(f)

    # Segmentation figure
    thumb_file = '{0}.thumb.fits'.format(label)
    if (make_segmentation_figure) & (os.path.exists(thumb_file)) & (aws_prep_dir is not None):

        print('Make segmentation figure')

        # Fetch segmentation image and catalog
        s3_prep_path = 'Pipeline/{0}/Prep/'.format(prep_root)
        s3_full_path = '{0}/{1}'.format(prep_bucket, s3_prep_path)
        s3_file = '{0}_visits.npy'.format(prep_root)

        has_seg_files = True
        seg_files = ['{0}-ir_seg.fits.gz'.format(prep_root),
                     '{0}_phot.fits'.format(prep_root)]

        for s3_file in seg_files:
            if not os.path.exists(s3_file):
                remote_file = os.path.join(s3_prep_path, s3_file)
                try:
                    print('Fetch {0}'.format(remote_file))
                    prep_bkt.download_file(remote_file, s3_file,
                                   ExtraArgs={"RequestPayer": "requester"})
                except:
                    has_seg_files = False
                    print('Make segmentation figure failed: {0}'.format(remote_file))
                    break

        if has_seg_files:
            s3_cat = utils.read_catalog(seg_files[1])
            segmentation_figure(label, s3_cat, seg_files[0])

    if aws_bucket:
        #aws_bucket = 's3://grizli-cosmos/CutoutProducts/'
        #aws_bucket = 's3://grizli/CutoutProducts/'

        s3 = boto3.resource('s3')
        s3_client = boto3.client('s3')
        bkt = s3.Bucket(aws_bucket.split("/")[2])
        aws_path = '/'.join(aws_bucket.split("/")[3:])

        if sync_fits:
            files = glob.glob('{0}*'.format(label))
        else:
            files = glob.glob('{0}*png'.format(label))

        for file in files:
            print('{0} -> {1}'.format(file, aws_bucket))
            bkt.upload_file(file, '{0}/{1}'.format(aws_path, file).replace('//', '/'), ExtraArgs={'ACL': 'public-read'})

        #os.system('aws s3 sync --exclude "*" --include "{0}*" ./ {1} --acl public-read'.format(label, aws_bucket))

        #os.system("""echo "<pre>" > index.html; aws s3 ls AWSBUCKETX --human-readable | sort -k 1 -k 2 | grep -v index | awk '{printf("%s %s",$1, $2); printf(" %6s %s ", $3, $4); print "<a href="$5">"$5"</a>"}'>> index.html; aws s3 cp index.html AWSBUCKETX --acl public-read""".replace('AWSBUCKETX', aws_bucket))

    return has_filts


def get_cutout_from_aws(label='macs0647-jd1', ra=101.9822125, dec=70.24326667, master='grizli-jan2019', scale_ab=21, thumb_height=2.0, remove=1, aws_bucket="s3://grizli/DropoutThumbnails/", lambda_func='grizliImagingCutout', force=False, **kwargs):
    """
    Get cutout using AWS lambda
    """
    import boto3
    import json

    #func = 'grizliImagingCutout'

    #label = '{0}_{1:05d}'.format(self.cat['root'][ix], self.cat['id'][ix])
    #url = 'https://s3.amazonaws.com/grizli/DropoutThumbnails/{0}.thumb.png'

    session = boto3.Session()
    client = session.client('lambda', region_name='us-east-1')

    event = {
          'label': label,
          "ra": ra,
          "dec": dec,
          "scale_ab": scale_ab,
          "thumb_height": thumb_height,
          "aws_bucket": aws_bucket,
          "remove": remove,
          "master": master,
        }

    for k in kwargs:
        event[k] = kwargs[k]

    bucket_split = aws_bucket.strip("s3://").split('/')
    bucket_name = bucket_split[0]
    bucket_path = '/'.join(bucket_split[1:])

    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bkt = s3.Bucket(bucket_name)

    files = [obj.key for obj in bkt.objects.filter(Prefix='{0}/{1}.thumb.png'.format(bucket_path, label))]

    if (len(files) == 0) | force:
        print('Call lambda: {0}'.format(label))

        print(event)

        response = client.invoke(
            FunctionName=lambda_func,
            InvocationType='Event',
            LogType='Tail',
            Payload=json.dumps(event))
    else:
        response = None
        print('Thumb exists')

    return response


def handler(event, context):
    import os
    import grizli
    print(grizli.__version__)

    os.chdir('/tmp/')
    os.system('rm *')
    os.system('rm -rf matplotlib*')

    print(event)  # ['s3_object_path'], event['verbose'])
    drizzle_images(**event)
    os.system('rm *')


def combine_filters(label='j022708p4901_00273', verbose=True):
    """
    Group nearby filters
    """
    import glob
    import numpy as np
    import astropy.io.fits as pyfits
    from grizli import utils

    filter_queries = {}
    filter_queries['uv'] = '{0}-f[2-3]*sci.fits'.format(label)
    filter_queries['visb'] = '{0}-f[4-5]*sci.fits'.format(label)
    filter_queries['visr'] = '{0}-f[6-8]*sci.fits'.format(label)
    filter_queries['y'] = '{0}-f[01][90][85]*sci.fits'.format(label)
    filter_queries['j'] = '{0}-f1[12][05]*sci.fits'.format(label)
    filter_queries['h'] = '{0}-f1[64]0*sci.fits'.format(label)

    grouped_filters = {}

    for qfilt in filter_queries:

        drz_files = glob.glob(filter_queries[qfilt])
        drz_files.sort()
        grouped_filters[qfilt] = [f.split('_dr')[0].split('-')[-1] for f in drz_files]

        if len(drz_files) > 0:
            drz_files.sort()

            if verbose:
                print('# Combine filters, {0}={1}'.format(qfilt,
                                      '+'.join(drz_files)))

            for i, file in enumerate(drz_files[::-1]):
                drz = pyfits.open(file)
                wht = pyfits.open(file.replace('_sci', '_wht'))
                sci = drz[0].data*1.

                # Subtract background?
                if 'IMGMED' in drz[0].header:
                    sci -= drz[0].header['IMGMED']
                    drz[0].header['IMGMED'] = 0.

                if i == 0:
                    photflam = drz[0].header['PHOTFLAM']

                    num = sci*wht[0].data
                    den = wht[0].data
                    drz_ref = drz
                    drz_ref[0].header['CFILT{0}'.format(i+1)] = utils.parse_filter_from_header(drz[0].header)
                    drz_ref[0].header['NCOMBINE'] = (len(drz_files), 'Number of combined filters')
                else:
                    scl = drz[0].header['PHOTFLAM']/photflam
                    num += sci*scl*(wht[0].data/scl**2)
                    den += wht[0].data/scl**2

                    drz_ref[0].header['CFILT{0}'.format(i+1)] = utils.parse_filter_from_header(drz[0].header)
                    drz_ref[0].header['NDRIZIM'] += drz[0].header['NDRIZIM']

            sci = num/den
            sci[den == 0] = 0
            drz_ref[0].data = sci

            pyfits.writeto('{0}-{1}_drz_sci.fits'.format(label, qfilt),
                           data=sci, header=drz_ref[0].header, overwrite=True,
                           output_verify='fix')

            pyfits.writeto('{0}-{1}_drz_wht.fits'.format(label, qfilt),
                           data=den, header=drz_ref[0].header, overwrite=True,
                           output_verify='fix')

    return grouped_filters


def show_all_thumbnails(label='j022708p4901_00273', filters=['visb', 'visr', 'y', 'j', 'h'], scale_ab=21, close=True, thumb_height=2., rgb_params=RGB_PARAMS, ext='png', xl=0.04, yl=0.98, fs=7):
    """
    Show individual filter and RGB thumbnails
    """
    import glob

    #from PIL import Image

    import numpy as np
    import matplotlib.pyplot as plt

    import astropy.io.fits as pyfits
    from astropy.visualization import make_lupton_rgb
    from grizli.pipeline import auto_script
    from grizli import utils

    all_files = glob.glob('{0}-f*sci.fits'.format(label))
    all_filters = [f.split('_dr')[0].split('-')[-1] for f in all_files]

    ims = {}
    for filter in filters:
        drz_files = glob.glob('{0}-{1}*_dr*sci.fits'.format(label, filter))
        if len(drz_files) > 0:
            im = pyfits.open(drz_files[0])
            ims[filter] = im

    rgb_params['scale_ab'] = scale_ab

    slx, sly, rgb_filts, fig = auto_script.field_rgb(root=label, HOME_PATH=None, **rgb_params)  # xsize=4, output_dpi=None, HOME_PATH=None, show_ir=False, pl=1, pf=1, scl=1, rgb_scl=[1, 1, 1], ds9=None, force_ir=False, filters=all_filters, add_labels=False, output_format='png', rgb_min=-0.01, xyslice=None, pure_sort=False, verbose=True, force_rgb=None, suffix='.rgb', scale_ab=scale_ab)
    if close:
        plt.close()

    #rgb = np.array(Image.open('{0}.rgb.png'.format(label)))
    rgb = plt.imread('{0}.rgb.png'.format(label))

    NX = (len(filters)+1)
    fig = plt.figure(figsize=[thumb_height*NX, thumb_height])
    ax = fig.add_subplot(1, NX, NX)
    ax.imshow(rgb, origin='upper', interpolation='nearest')
    # ax.text(0.05, 0.95, label, ha='left', va='top', transform=ax.transAxes, fontsize=7, color='w', bbox=dict(facecolor='k', edgecolor='None', alpha=0.8))
    # ax.text(0.05, 0.05, ' '.join(rgb_filts), ha='left', va='bottom', transform=ax.transAxes, fontsize=6, color='w', bbox=dict(facecolor='k', edgecolor='None', alpha=0.8))

    for i, filter in enumerate(filters):
        if filter in ims:
            zp_i = utils.calc_header_zeropoint(ims[filter], ext=0)
            scl = 10**(-0.4*(zp_i-5-scale_ab))
            pixscl = utils.get_wcs_pscale(ims[filter][0].header.copy())
            scl *= (0.06/pixscl)**2

            img = ims[filter][0].data*scl

            image = make_lupton_rgb(img, img, img, stretch=0.1, minimum=-0.01)

            ax = fig.add_subplot(1, NX, i+1)
            ax.imshow(255-image, origin='lower', interpolation='nearest')

    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(pad=0.1)

    # Add labels
    #xl, yl = 0.04, 0.98
    for i, filter in enumerate(filters):
        if filter in ims:
            if filter in ['uv', 'visb', 'visr', 'y', 'j', 'h']:
                grouped_filters = []
                h_i = ims[filter][0].header
                for j in range(h_i['NCOMBINE']):
                    grouped_filters.append(h_i['CFILT{0}'.format(j+1)])

                text_label = '+'.join(grouped_filters)
            else:
                text_label = filter

            fig.text((i+xl)/NX, yl, text_label, fontsize=fs,
                     ha='left', va='top', transform=fig.transFigure,
                     bbox=dict(facecolor='w', edgecolor='None', alpha=0.9))

    fig.text((i+1+xl)/NX, yl, label, ha='left', va='top', transform=fig.transFigure, fontsize=fs, color='w', bbox=dict(facecolor='k', edgecolor='None', alpha=0.8))
    fig.text((i+1+0.04)/NX, 1-yl, ' '.join(rgb_filts), ha='left', va='bottom', transform=fig.transFigure, fontsize=fs, color='w', bbox=dict(facecolor='k', edgecolor='None', alpha=0.8))

    fig.savefig('{0}.thumb.{1}'.format(label, ext))
    if close:
        plt.close()

    return fig


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print('Usage: aws_drizzler.py cp561356 150.208875 1.850241667 40 ')
        print(sys.argv)
        exit()

    # print('xxx')
    drizzle_images(label=sys.argv[1], ra=float(sys.argv[2]), dec=float(sys.argv[3]), size=float(sys.argv[4]))
