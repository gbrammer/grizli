"""
Splitting fields into tiles
"""
import os
import glob

import numpy as np
import astropy.wcs as pywcs
from collections import OrderedDict
import yaml

from .. import utils

def make_all_fields():
    """
    Define tile layout
    """
    
    layout = """# field ra dec dx dy
    abell370   39.9750000 -1.5752778   40 40 
    macs0416   64.0333333 -24.0730556  40 40  
    macs1149   177.4000000 22.3972222  40 40  
    abell2744   3.5875000 -30.3966667  40 40
    macs0417   64.3958333 -11.9091667  40 40 
    macs1423   215.9500000 24.0811111  40 40 
    sgas1723    260.895 34.205 20 20
    sgas1226    186.69  21.88  10 10
    dracoii    238.1983333 64.56528 20 20
    whl0137    24.355  -8.457 22 22
    abell1689   197.8765  -1.328 22 22
    cos      150.125 2.2 90 90
    j013804m2156   24.51616  -21.93046 40 40 
    j015020m1006   27.58366  -10.09244 40 40 
    j045412m0302   73.547   -3.013     40 40 
    j091840m8103  139.64290  -81.05227 40 40 
    j134200m2442  205.50969  -24.69773 40 40 
    j152256p2536  230.72345   25.59692 40 40 
    j212928m0741  322.358   -7.69209 40 40 
    gds 53.1592464 -27.7828223 46 46
    egs 214.8288    52.8067333 60 60
    gdn 189.2369029 62.2312839 46 46
    uds 34.40869 -5.16299 30 30
    smacs0723 110.83403 -73.45429 20 20
    j1235 189.025 4.948 20 20
    """
    
    tile_defs = utils.read_catalog(layout)
    
    
def define_tiles(ra=109.3935148, dec=37.74934031, size=(24, 24), tile_size=6, overlap=0.3, field='macs0717', pixscale=0.05, theta=0):
    """
    Tile definition
    """
    
    tile_wcs = OrderedDict()
    
    size_per = tile_size-overlap
    nx = int(np.ceil(size[0]/size_per))
    ny = int(np.ceil(size[1]/size_per))
    
    sx = nx*tile_size-(nx-1)*overlap
    sy = ny*tile_size-(ny-1)*overlap
    px = int(tile_size*60/pixscale)

    # Even number of pixels
    header, parent_wcs = utils.make_wcsheader(ra=ra, dec=dec,
                                              size=(sx*60, sy*60), 
                                              pixscale=pixscale, theta=theta)
    for i in range(nx):
        x0 = int(i*px-i*(overlap/tile_size*px))
        slx = slice(x0, x0+px)
        for j in range(ny):
            y0 = int(j*px-j*(overlap/tile_size*px))
            sly = slice(y0, y0+px)
            
            slice_header = utils.get_wcs_slice_header(parent_wcs, slx, sly)
            slice_wcs = pywcs.WCS(slice_header)
            slice_wcs._header = slice_header
            #tile=(i+1)*10+(j+1)
            tile = '{0:02d}.{1:02d}'.format(i+1, j+1)
            tile_wcs[tile] = slice_wcs
    
    tile_dict = {}
    for k in tile_wcs:
        _h = utils.to_header(tile_wcs[k])
        tile_dict[k] = {'field':field}
        for key in _h:
            tile_dict[k][key] = _h[key]
    
    yml_file = '{0}_{1:02d}mas_tile_wcs.yaml'.format(field, int(pixscale*1000))
    with open(yml_file,'w') as fp:
        yaml.dump(tile_dict, stream=fp)
                                                          
    #np.save(,[tile_wcs])
    reg_file = yml_file.replace('.yaml', '.reg')
    with open(reg_file,'w') as fpr:
        fpr.write('fk5\n')
        for t in tile_wcs:
            fp = tile_wcs[t].calc_footprint()
            pstr = 'polygon('
            pstr += ','.join(['{0:.6f}'.format(i)
                            for i in fp.flatten()])
            pstr += ') # text={{{0}}}\n'.format(t)
            fpr.write(pstr)
    
    return tile_wcs


def make_field_tiles(field='cos', ra=150.125, dec=2.2, rsize=45, tile_npix=2048+256, pscale=0.08, initial_status=90, send_to_database=False, **kwargs):
    """
    Make tiles across COSMOS with a single tangent point
    """
    import numpy as np    
    
    from . import db, tile_mosaic
        
    np.set_printoptions(precision=6)
    
    #field='cos'
    
    # tile_npix = 4096
    # pscale = 0.1
    # 
    # ra, dec = 150.1, 2.29
    # rsize = 48
    # overlap = 0
    # 
    # # New tiles with overlaps
    # ra, dec = 150.125, 2.2
    # rsize = 45
    # pscale = 0.080
    # tile_npix = 2048+256
    
    if 0:
        ra, dec, rsize = 34.40869, -5.16299, 30
        field = 'uds'
        pscale = 0.08
        tile_npix = 2048+256
        
        
    tile_arcmin = tile_npix*pscale/60
    overlap=tile_arcmin/9
    
    print(f'{field}: tile size, arcmin = {tile_arcmin}')
    
    tiles = define_tiles(ra=ra, dec=dec, 
                         size=(rsize*2, rsize*2), 
                         tile_size=tile_arcmin, 
                         overlap=overlap, field=field, 
                         pixscale=pscale, theta=0)
        
    rows = []
    for t in tiles:
        wcs = tiles[t]
        sr = utils.SRegion(wcs.calc_footprint())
                    
        wcsh = utils.to_header(wcs)
        row = [t]
        for k in wcsh:
            row.append(wcsh[k])
        
        try:
            fp = sr.polystr()[0]
        except:
            fp = sr.xy[0].__str__().replace('\n', ',').replace('   ', ',')
            fp = fp.replace(' -',',-')
            fp = fp.replace(' ', '').replace('[','(')
            fp = fp.replace(']', ')').replace(',,',',')

        row.append(fp)     
        
        rows.append(row)
    
    names = ['tile']
    for k in wcsh:
        names.append(k.lower())
    
    names.append('footprint')

    tiles = utils.GTable(rows=rows, names=names)
    tiles['status'] = initial_status
    
    tiles['field'] = field
            
    #db.send_to_database('cosmos_tiles', cos_tile, if_exists='replace')
    if send_to_database:
        db.execute(f"delete from combined_tiles where field = '{field}'")
        db.send_to_database('combined_tiles', tiles, if_exists='append')
    
    return tiles
    
def split_tiles(root='abell2744-080-08.08', ref_tile=(8,8), filters=['visr','f125w','h'], optical=False, suffix='.rgb', xsize=6, zoom_levels=[4,3,2,1], force=False, scl=1, invert=False, verbose=True, rgb_scl=[1,1,1], rgb_min=-0.01):
    """
    Split image into 256 pixel tiles for map display
    """
    import matplotlib.pyplot as plt
    from skimage.io import imsave
    
    from grizli.pipeline import auto_script
    
    nx = (2048+256)
    dpi = int(nx/xsize)
    
    if os.path.exists(f'{root}{suffix}.png') & (~force):
        return True
    
    try:
        _ = auto_script.field_rgb(root=root,
                                  xsize=xsize, filters=filters, 
                                  full_dimensions=2**optical, HOME_PATH=None, 
                                  add_labels=False,
                                  gzext='*', suffix=suffix, 
                                  output_format='png',
                                  scl=scl, invert=invert, 
                                  rgb_scl=rgb_scl, rgb_min=rgb_min)
    except IndexError:
        return False
    
    fig = _[-1]
    
    base = '_'.join(root.split('-')[:-1]).replace('+','_') + '_' + suffix[1:]
    
    tx, ty = np.cast[int](root.split('-')[-1].split('.'))

    for iz, zoom in enumerate(zoom_levels):
        if iz > 0:
            zoom_img = f'{root}{suffix}.{2**iz:d}.png'
            fig.savefig(zoom_img, dpi=dpi/2**iz)
            img = plt.imread(zoom_img)
        else:
            img = plt.imread(f'{root}{suffix}.png')
        
        if verbose:
            print(f'zoom: {zoom} {img.shape}')
        
        img = img[::-1,:,:]
        
        ntile = int(2048/2**(4-zoom)/256)
        left = (tx - ref_tile[0])*ntile
        bot = -(ty - ref_tile[1])*ntile+2*ntile
        # print(zoom, ntile, left, bot)

        #axes[iz].set_xlim(-ntile*0.1, ntile*(1.1)-1)
        #axes[iz].set_ylim(*axes[iz].get_xlim())

        for i in range(ntile):
            xi = left + i
            for j in range(ntile):
                yi = bot - j - 1
                
                slx = slice((i*256), (i+1)*256)
                sly = slice((j*256), (j+1)*256)
                
                tile_file = f'{root}-tiles/{base}/{zoom}/{yi}/{xi}.png'
                if verbose > 1:
                    print(f'  {i} {j} {tile_file}')
                
                dirs = tile_file.split('/')
                for di in range(1,5):
                    dpath = '/'.join(dirs[:di])
                    #print(dpath)
                    if not os.path.exists(dpath):
                        os.mkdir(dpath)
                                     
                imsave(tile_file, img[sly, slx, :][::-1,:,:],
                       plugin='pil', format_str='png')


def make_all_tile_images(root, force=False, ref_tile=(8,8), cleanup=True, zoom_levels=[4,3,2,1], brgb_filts=['visr','visb','uv'], rgb_filts=['visr','j','h'], blue_is_opt=True, make_opt_filters=True, make_ir_filters=True):
    """
    Make tiles for map
    """
    import matplotlib.pyplot as plt
    
    from grizli.pipeline import auto_script
    
    #root = f'{field}-080-08.08'

    files = glob.glob(f'{root}-[hvu]*')
    files += glob.glob(f'{root}*.rgb.png')

    if len(files) == 0:
        auto_script.make_filter_combinations(root, 
                          filter_combinations={'h':['F140W','F160W'], 
                                               'j':['F105W','F110W','F125W'],
                                            'visr':['F850LP','F814W','F775W'],
                                     'visb':['F606W','F555W', 'F606WU'][:-1],
                                'uv':['F438WU','F435W','F435WU', 'F475W'], 
                                'sw':['F070W-CLEAR','F090W-CLEAR',
                                      'F115W-CLEAR','F150W-CLEAR',
                                      'F200W-CLEAR'], 
                                'lw':['F277W-CLEAR','F356W-CLEAR',
                                      'F410M-CLEAR','F444W-CLEAR'],
                            }, 
                                            weight_fnu=False)

    if 'j' in rgb_filts:
        rgb_scl = [1.1, 0.8, 1]
    else:
        rgb_scl = [1,1,1]
        
    split_tiles(root, ref_tile=ref_tile, 
                filters=rgb_filts, zoom_levels=zoom_levels,
                optical=False, suffix='.rgb', xsize=6, scl=1,
                force=force, rgb_scl=rgb_scl)

    plt.close('all')
    
    if len(glob.glob(f'{root}*.brgb.png')) == 0:
        split_tiles(root, ref_tile=ref_tile, 
                    filters=brgb_filts, zoom_levels=zoom_levels,
                    optical=blue_is_opt, suffix='.brgb', xsize=6, scl=2,
                    force=force, rgb_scl=[1., 1.2, 1.4], rgb_min=-0.018)

        plt.close('all')
    
    # JWST SW
    if len(glob.glob(f'{root}*.swrgb.png')) == 0:
        split_tiles(root, ref_tile=ref_tile, 
                    filters=[f.lower() for f in ['F070W-CLEAR','F090W-CLEAR',
                          'F115W-CLEAR','F150W-CLEAR',
                          'F200W-CLEAR']],
                    zoom_levels=zoom_levels,
                    optical=True, suffix='.swrgb', xsize=6, scl=4,
                    force=force, rgb_scl=[1,1,1], rgb_min=-0.018)

        plt.close('all')
    
    # JWST LW
    if len(glob.glob(f'{root}*.lwrgb.png')) == 0:
        split_tiles(root, ref_tile=ref_tile, 
                    filters=[f.lower() for f in ['F277W-CLEAR','F356W-CLEAR',
                          'F360M-CLEAR','F410M-CLEAR','F444W-CLEAR',
                          'F480M-CLEAR']],
                    zoom_levels=zoom_levels,
                    optical=True, suffix='.lwrgb', xsize=6, scl=4,
                    force=force, rgb_scl=[1,1,1], rgb_min=-0.018)

        plt.close('all')
    
    # All NIRCam
    if len(glob.glob(f'{root}*.ncrgb.png')) == 0:
        split_tiles(root, ref_tile=ref_tile, 
                    filters=[f.lower() for f in ['F070W-CLEAR','F090W-CLEAR',
                                                 'F115W-CLEAR','F150W-CLEAR',
                                                 'F200W-CLEAR',
                                                 'F277W-CLEAR','F356W-CLEAR',
                                                'F410M-CLEAR','F444W-CLEAR']],
                    zoom_levels=zoom_levels,
                    optical=True, suffix='.ncrgb', xsize=6, scl=4,
                    force=force, rgb_scl=[1,1,1], rgb_min=-0.018)

        plt.close('all')
    
    if 'dracoii' in root:
        
        split_tiles(root, ref_tile=ref_tile, 
                    filters=[f.lower() for f in ['F090W-CLEAR',
                             'F150W-CLEAR']],
                    zoom_levels=zoom_levels,
                    optical=True, suffix='.swrgb', xsize=6, scl=4,
                    force=force, rgb_scl=[1,1,1], rgb_min=-0.018)

        plt.close('all')

        split_tiles(root, ref_tile=ref_tile, 
                    filters=[f.lower() for f in ['F360M-CLEAR',
                             'F480M-CLEAR']],
                    zoom_levels=zoom_levels,
                    optical=True, suffix='.lwrgb', xsize=6, scl=4,
                    force=force, rgb_scl=[1,1,1], rgb_min=-0.018)

        plt.close('all')

        split_tiles(root, ref_tile=ref_tile, 
                    filters=[f.lower() for f in ['F090W-CLEAR','F150W-CLEAR',
                                                 'F480M-CLEAR',
                                                ]],
                    zoom_levels=zoom_levels,
                    optical=True, suffix='.ncrgb', xsize=6, scl=4,
                    force=force, rgb_scl=[1.4, 0.6, 0.35], rgb_min=-0.018)

        plt.close('all')
        
    if root.startswith('cos'):
        if len(glob.glob(f'{root}*.vi.png')) == 0:
            split_tiles(root, ref_tile=ref_tile, 
                        filters=['f814w','f160w'], zoom_levels=zoom_levels,
                        optical=False, suffix='.vi', xsize=6, scl=0.8,
                        force=force, rgb_scl=[1, 1, 1], rgb_min=-0.018)

            plt.close('all')
        
    # IR
    if make_ir_filters:
        files = []
        for ir_filt in ['f098m','f105w','f110w','f125w','f140w','f160w']:
            files += glob.glob(f'{root}-{ir_filt}*sci.fits*')
        
        # MIRI on 80mas 
        for ir_filt in ['f560w','f770w','f1000w','f1130w',
                        'f1280w','f1500w','f1800w','f2100w','f2550w']:
            files += glob.glob(f'{root}-{ir_filt}*sci.fits*')
        
        files.sort()
        filts = [('-'.join(file.split(f'{root}-')[1:])).split('_')[0]
                 for file in files]
                 
        for filt in filts:
            if os.path.exists(f'{root}.{filt}.png'):
                continue

            split_tiles(root, ref_tile=ref_tile, 
                    filters=[filt], zoom_levels=zoom_levels,
                    optical=False, suffix=f'.{filt}', xsize=6, 
                    force=force, scl=2, invert=True)

            plt.close('all')
    
    if make_opt_filters:
        # Optical, 2X pix
        files = glob.glob(f'{root}-f[2-8]*sci.fits*')
        files += glob.glob(f'{root}-f*clear*sci.fits*')
        files.sort()
        filts = [file.split(f'{root}-')[1].split('_')[0] for file in files]
        for filt in filts:
            if os.path.exists(f'{root}.{filt}.png'):
                continue

            split_tiles(root, ref_tile=ref_tile, 
                filters=[filt], zoom_levels=zoom_levels,
                optical=True, suffix=f'.{filt}', xsize=6, 
                force=force, scl=2, invert=True)

            plt.close('all')
    
    if cleanup:
        files = glob.glob(f'{root}-[vhuj]*fits*')
        files.sort()
        for file in files:
            print(f'rm {file}')
            os.remove(file)


TILE_FILTERS = ['F336W', 'F350LP', 'F390W', 'F435W', 'F438W', 'F475W',
                'F555W', 'F606W', 'F625W', 'F775W', 
                'F814W', 'F850LP', 
                'F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W',
                'F070W-CLEAR','F090W-CLEAR','F115W-CLEAR','F150W-CLEAR', # NRC
                'F200W-CLEAR','F210M-CLEAR','F250M-CLEAR',
                'F150W2-CLEAR','F322W2-CLEAR',
                'F277W-CLEAR','F356W-CLEAR','F444W-CLEAR',
                'F300M-CLEAR','F360M-CLEAR','F410M-CLEAR', 'F480M-CLEAR',
                'CLEAR-F115W','CLEAR-F150W','CLEAR-F200W', # NIS
                'F560W','F770W','F1000W','F1280W','F1500W', # MIR
                'F1800W','F2100W','F2550W',
                ][1:]


def process_tile(field='cos', tile='01.01', filters=TILE_FILTERS, fetch_existing=True, cleanup=True, make_catalog=False, clean_flt=True, pixfrac=0.75, make_tile_images=True):
    """
    """
    import numpy as np
    
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from grizli.aws import visit_processor, db
    import golfir.catalog
    from grizli.pipeline import auto_script
    
    utils.LOGFILE = 'mosaic.log'
    
    print(f'Set status=1 in `combined_tiles` for {field} {tile}')
    db.execute(f"update combined_tiles set status=1 where tile = '{tile}' and field='{field}'")
    
    row = db.SQL(f"select * from combined_tiles where tile = '{tile}' and field='{field}'")
    
    h = pyfits.Header()
    for k in row.colnames:
        if k in ['footprint']:
            continue
        
        h[k.upper()] = row[k][0]
        
    ir_wcs = pywcs.WCS(h)
    
    root = f'{field}-080-{tile}'
    
    if fetch_existing:
        os.system(f"""aws s3 sync s3://grizli-v2/ClusterTiles/{field}/ ./
                          --exclude "*"
                          --include "{root}*_dr?*fits.gz"
                          """.replace('\n', ' '))
        
        files = glob.glob(f'{root}*gz')
        for file in files:
            os.system(f'gunzip --force {file}')
            
    visit_processor.cutout_mosaic(rootname=root, 
                              skip_existing=True,
                              ir_wcs=ir_wcs,
                              filters=filters, 
                              pixfrac=pixfrac,
                              kernel='square', s3output=None, 
                              gzip_output=False, clean_flt=clean_flt)
    
    files = glob.glob(f'{root}*dr*fits')
    if len(files) == 0:
        db.execute(f"update combined_tiles set status=4 where tile = '{tile}'")
        print(f'No images found, set status=4 in `combined_tiles` for {field} {tile}')
        
        return True
    
    if make_catalog:
        
        golfir.catalog.make_charge_detection(root, ext='ir')
    
        phot = auto_script.multiband_catalog(field_root=root) #, **phot_kwargs)
    
        if len(phot) == 0:
            # Empty catalog
            db.execute(f"update combined_tiles set status=10 where tile = '{tile}' AND field = '{field}'")
        
            if cleanup:

                files = glob.glob(f'{root}*')
                files.sort()
                for file in files:
                    print(f'rm {file}')
                    os.remove(file)
        
            return False
        
        for i in [4,5,6]:
            for c in phot.colnames:
                if c.endswith(f'{i}'):
                    phot.remove_column(c)
    
        for c in phot.colnames:
            if c in ['ra','dec','x_world','y_world']:
                continue
            
            if phot[c].dtype == np.float64:
                phot[c] = phot[c].astype(np.float32)
            elif phot[c].dtype == np.int64:
                phot[c] = phot[c].astype(np.int32)
              
        phot['tile'] = tile
        phot['field'] = field
    
        if 'id' in phot.colnames:
            phot.remove_column('id')
    
        for c in ['xmin','xmax','ymin','ymax']:
            if c in phot.colnames:
                phot.rename_column(c, c+'pix')
    
        for c in phot.colnames:
            if '-' in c:
                phot.rename_column(c, c.replace('-','_'))
            
        if 0:
            db.execute('CREATE TABLE combined_tile_phot AS SELECT * FROM cosmos_tile_phot limit 0')
            db.execute('ALTER TABLE combined_tile_phot DROP COLUMN id;')
            db.execute('ALTER TABLE combined_tile_phot ADD COLUMN id SERIAL PRIMARY KEY;')
        
        db.execute(f"DELETE from combined_tile_phot WHERE tile='{tile}' and field='{field}'")
    
        seg = pyfits.open(f'{root}-ir_seg.fits')
    
        ### IDs on edge
        edge = np.unique(seg[0].data[16:19,:])
        edge = np.append(edge, np.unique(seg[0].data[-19:-16,:]))
        edge = np.append(edge, np.unique(seg[0].data[:, 16:19]))
        edge = np.append(edge, np.unique(seg[0].data[:, -19:-16]))
        edge = np.unique(edge)
        phot['edge'] = np.in1d(phot['number'], edge)*1
    
        ### Add missing columns
        cols = db.SQL('select * from combined_tile_phot limit 2').colnames
        for c in phot.colnames:
            if c not in cols:
                print('Add column {0} to `combined_tile_phot` table'.format(c))
                if phot[c].dtype in [np.float64, np.float32]:
                    SQL = "ALTER TABLE combined_tile_phot ADD COLUMN {0} real;".format(c)
                else:
                    SQL = "ALTER TABLE combined_tile_phot ADD COLUMN {0} int;".format(c)
                
                db.execute(SQL)
            
        db.send_to_database('combined_tile_phot', phot, if_exists='append')
    
        if 'id' not in cols:
            # Add unique id index column
            db.execute('ALTER TABLE combined_tile_phot ADD COLUMN id SERIAL PRIMARY KEY;')
    
        # Use db id
        ids = db.SQL(f"SELECT number, id, ra, dec, tile from combined_tile_phot WHERE tile='{tile}' AND field='{field}'")
        idx, dr = ids.match_to_catalog_sky(phot)
        
        phot['id'] = phot['number']
        for c in ['xmin','xmax','ymin','ymax']:
            if c+'pix' in phot.colnames:
                phot.rename_column(c+'pix', c)
    
        golfir.catalog.switch_segments(seg[0].data, phot, ids['id'][idx])
        pyfits.writeto(f'{root}-ir_seg.fits', data=seg[0].data, 
                       header=seg[0].header, overwrite=True)
    
    if make_tile_images:
        ### Make subtiles
        ref_tiles = {'cos': (16,16), 
                     'uds': (11, 10),
                     'abell2744': (8, 8), 
                     'egs': (10,14),
                     'gds': (9,9), 
                     'gdn': (9,9),
                     'sgas1723': (4,4),
                     'sgas1226': (2,2),
                     'smacs0723': (4,4), 
                     'dracoii':(5,4),
                     'egs-v2': (10,14),
                     'j1235':(6,4),
                     'whl0137':(5,5),
                     }
    
        if field in ref_tiles:
            ref_tile = ref_tiles[field]
        else:
            ref_tile = (9, 9)
        
        make_all_tile_images(root, force=False, ref_tile=ref_tile,
                             rgb_filts=['h','j','visr'],
                             brgb_filts=['visr','visb','uv'],
                             blue_is_opt=(field not in ['j013804m2156']), 
                             make_ir_filters=True, 
                             make_opt_filters=True)
    
    print(f'Sync ./{root}-tiles/ >> s3://grizli-v2/ClusterTiles/Map/{field}/')
    
    os.system(f'aws s3 sync ./{root}-tiles/ ' + 
              f' s3://grizli-v2/ClusterTiles/Map/{field}/ ' + 
               '--acl public-read --quiet')
                      
    ### Gzip products
    drz_files = glob.glob(f'{root}-*_dr*fits')
    drz_files += glob.glob(f'{root}*seg.fits')
    drz_files.sort()
        
    for file in drz_files:
        cmd = f'gzip --force {file}'
        print(cmd)
        os.system(cmd)
        
    os.system(f'aws s3 sync ./ s3://grizli-v2/ClusterTiles/{field}/' + 
              f' --exclude "*" --include "{root}*gz"' +
              f' --include "{root}*wcs.csv"' +
              f' --include "{root}*fp.png" --acl public-read')
    
    db.execute(f"update combined_tiles set status=2 where tile = '{tile}' AND field = '{field}'")
    print(f'Set status=2 in `combined_tiles` for {field} {tile}')
    
    if cleanup:
        print(f'rm -rf {root}-tiles')
        os.system(f'rm -rf {root}-tiles')
        
        files = glob.glob(f'{root}*')
        files.sort()
        for file in files:
            print(f'rm {file}')
            os.remove(file)
        



def get_random_tile():
    """
    Find a visit that needs processing
    """
    from grizli.aws import db
    
    all_tiles = db.SQL(f"""SELECT tile, field 
                           FROM combined_tiles
                           WHERE status=0""")
    
    if len(all_tiles) == 0:
        return None, None
    
    tile, field = all_tiles[np.random.randint(0, len(all_tiles))]
    return tile, field


def run_one(own_directory=True, **kwargs):
    """
    Run a single random visit
    """
    import os
    import time
    from grizli.aws import db

    ntile = db.SQL("""SELECT count(status)
                       FROM combined_tiles
                       WHERE status = 0""")['count'][0] 
    
    tile, field = get_random_tile()
    if tile is None:
        with open('/GrizliImaging/tile_finished.txt','w') as fp:
            fp.write(time.ctime() + '\n')
    else:
        print(f'============  Run tile  ==============')
        print(f'{field} {tile}')
        print(f'========= {time.ctime()} ==========')
        
        with open('/GrizliImaging/tile_history.txt','a') as fp:
            fp.write(f'{time.ctime()} {tile}\n')
        
        if own_directory:
            path = f'/GrizliImaging/{field}-{tile}'
            if not os.path.exists(path):
                os.makedirs(path)
                
            os.chdir(path)
            
        #process_visit(tile, clean=clean, sync=sync)
        process_tile(tile=tile, field=field, **kwargs)


def create_mosaic_from_tiles(assoc, filt='ir', clean=True):
    """
    Get tiles overlapping visit footprint
    """
    import glob    
    import astropy.io.fits as pyfits
    import astropy.table
    from grizli.aws import db
    
    olap_tiles = db.SQL(f"""SELECT tile, field
                        FROM combined_tiles t, exposure_files e
                        WHERE e.assoc = '{assoc}'
                        AND polygon(e.footprint) && polygon(t.footprint)
                        GROUP BY tile, field""")
    
    tx = np.array([int(t.split('.')[0]) for t in olap_tiles['tile']])
    ty = np.array([int(t.split('.')[1]) for t in olap_tiles['tile']])
    
    xm = tx.min()
    ym = ty.min()
    
    nx = tx.max()-tx.min()+1
    ny = ty.max()-ty.min()+1
    
    print(olap_tiles)
    
    field = olap_tiles['field'][0]
    for t in olap_tiles['tile']:
        # print(f'Fetch tile s3://grizli-v2/ClusterTiles/{field}/{field}-080-{t}-{filt}*')
        # 
        # os.system(f"aws s3 cp s3://grizli-v2/ClusterTiles/{field}/{field}-080-{t}-{filt}_drz_sci.fits.gz . ")
        # os.system(f"aws s3 cp s3://grizli-v2/ClusterTiles/{field}/{field}-080-{t}-{filt}_drz_wht.fits.gz . ")
        # os.system(f"aws s3 cp s3://grizli-v2/ClusterTiles/{field}/{field}-080-{t}-{filt}_seg.fits.gz . ")
        
        os.system(f"""aws s3 sync s3://grizli-v2/ClusterTiles/{field}/ ./
                          --exclude "*"
                          --include "{field}*{t}-{filt}*_dr?*fits.gz"
                          --include "{field}*{t}-ir_seg.fits.gz"
                          """.replace('\n', ' '))
        
    wcs = db.SQL(f"""SELECT * FROM combined_tiles 
                     WHERE tile = '{xm:02d}.{ym:02d}'
                     AND field = '{field}'""")
    
    h = pyfits.Header()
    for c in wcs.colnames:
        if c in ['footprint']:
            continue
        
        h[c] = wcs[c][0]
    
    xnpix = h['NAXIS1']
    ynpix = h['NAXIS2']
    
    olap = 256
    pad = 32
    
    h['NAXIS1'] = h['NAXIS1']*nx - olap*(nx-1)
    h['NAXIS2'] = h['NAXIS2']*ny - olap*(ny-1)
    
    img_shape = (h['NAXIS2'], h['NAXIS1'])
    sci = np.zeros(img_shape, dtype=np.float32)
    wht = np.zeros(img_shape, dtype=np.float32)
    seg = np.zeros(img_shape, dtype=int)
    
    skip_tiles = []
    
    for tile, txi, tyi in zip(olap_tiles['tile'], tx, ty):
        _file = f'{field}-080-{txi:02d}.{tyi:02d}-{filt}_dr*_sci.fits.gz'
        _files = glob.glob(_file)
        if len(_files) == 0:
            msg = f'*ERROR* {_file} not found'
            utils.log_comment(utils.LOGFILE, msg, verbose=True)
            continue
            
        file = _files[0]
        
        msg = f'Add tile {file} to {assoc} mosaic'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        im = pyfits.open(file)
        if im[0].shape[0] != xnpix:
            print(f'Bad size ({im[0].data.shape})')
            skip_tiles.append(tile)
            continue
            
        olap_dx = (txi-xm)*olap
        olap_dy = (tyi-ym)*olap
        
        slx = slice((txi-xm)*xnpix - olap_dx + pad,
                    (txi-xm+1)*xnpix - olap_dx)
        sly = slice((tyi-ym)*ynpix - olap_dy + pad,
                    (tyi-ym+1)*ynpix - olap_dy)
        
        for k in im[0].header:
            if k not in h:
                h[k] = im[0].header[k]
                
        sci[sly, slx] = im[0].data[pad:, pad:]*1

        im = pyfits.open(file.replace('_sci','_wht'))
        wht[sly, slx] = im[0].data[pad:, pad:]**1

        im = pyfits.open(file.replace('_drz_sci','_seg'))
        seg[sly, slx] = im[0].data[pad:, pad:]**1
    
    _hdu = pyfits.PrimaryHDU(data=sci, header=h)
    _hdu.writeto(f'{assoc}-{filt}_drz_sci.fits', overwrite=True)
    
    _hdu = pyfits.PrimaryHDU(data=wht, header=h)
    _hdu.writeto(f'{assoc}-{filt}_drz_wht.fits', overwrite=True)
    
    _hdu = pyfits.PrimaryHDU(data=seg, header=h)
    _hdu.writeto(f'{assoc}-ir_seg.fits', overwrite=True)
    
    if clean:
        for tile, txi, tyi in zip(olap_tiles['tile'], tx, ty):
            files = glob.glob(f'cos-080-{txi:02d}.{tyi:02d}-{filt}*')
            for file in files:
                msg = f'rm {file}'
                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                os.remove(file)

    #### Catalog
    cols = ['id','thresh', 'npix', 'tnpix',
            'xminpix as xmin', 'xmaxpix as xmax',
            'yminpix as ymin', 'ymaxpix as ymax', 
            'x', 'y', 'x2_image', 'y2_image', 'xy_image',
            'errx2', 'erry2', 'errxy',
            'a_image', 'b_image', 'theta_image',
            'cxx_image', 'cyy_image', 'cxy_image',
            'cflux', 'flux',
            'cpeak', 'peak', 'xcpeak', 'ycpeak', 'xpeak', 'ypeak',
            'flag',
            'x_image', 'y_image',
            #'x_world', 'y_world',
            'ra as x_world', 'dec as y_world', 
            'flux_iso', 'fluxerr_iso', 'area_iso', 'mag_iso', 'kron_radius',
            'kron_rcirc', 'flux_auto', 'fluxerr_auto', 'bkg_auto',
            'flag_auto', 'area_auto', 'flux_radius_flag', 'flux_radius_20',
            'flux_radius', 'flux_radius_90', 'tot_corr', 'mag_auto',
            'magerr_auto', 'flux_aper_0', 'fluxerr_aper_0', 'flag_aper_0',
            'bkg_aper_0', 'mask_aper_0', 'flux_aper_1', 'fluxerr_aper_1',
            'flag_aper_1', 'bkg_aper_1', 'mask_aper_1', 'flux_aper_2',
            'fluxerr_aper_2', 'flag_aper_2', 'bkg_aper_2', 'mask_aper_2',
            'flux_aper_3', 'fluxerr_aper_3', 'flag_aper_3', 'bkg_aper_3',
            'mask_aper_3']
    
    scols = ','.join(cols)
    
    tabs = []
    for tile, txi, tyi in zip(olap_tiles['tile'], tx, ty):
        if tile in skip_tiles:
            continue
            
        _SQL = f"""SELECT {scols} from combined_tile_phot
                   where tile = '{tile}'
                   AND x < 2048 AND y < 2048
                   AND x > {pad} AND y > {pad}"""
                   
        tab = db.SQL(_SQL)
                
        # Pixel offsets
        olap_dx = (txi-xm)*olap
        olap_dy = (tyi-ym)*olap

        dx = (txi-xm)*(2048+olap) - olap_dx
        dy = (tyi-ym)*(2048+olap) - olap_dx
        
        for c in tab.colnames:
            if c in ['xmin', 'xmax', 'x', 'x_image', 'xpeak', 'xcpeak']:
                tab[c] += dx
            elif c in ['ymin', 'ymax', 'y', 'y_image', 'ypeak', 'ycpeak']:
                tab[c] += dy
        
        tabs.append(tab)
        
        msg = f'{assoc}: Query {tile} catalog N={len(tab)} dx={dx} dy={dy}'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
    
    tab = astropy.table.vstack(tabs)
    for c in tab.colnames:
        tab.rename_column(c, c.upper())
    
    # Make it look as expected for grizli model
    if hasattr(tab['MAG_AUTO'], 'filled'):
        tab['MAG_AUTO'] = tab['MAG_AUTO'].filled(99)
        
    tab.rename_column('ID','NUMBER')
    tab['NUMBER'] = tab['NUMBER'].astype(int)
    tab.write(f'{assoc}-ir.cat.fits', overwrite=True)
    
    ### Grism
    if 0:
        os.system(f'aws s3 sync s3://grizli-v2/HST/Pipeline/{assoc}/Prep/ ./ --exclude "*" --include "*flt.fits" --include "*yml"')
    
        # grism_files = glob.glob('iehn*[a-p]_flt.fits')
        # grism_files.sort()
        #     
        # grp = auto_script.grism_prep(field_root=assoc, 
        #                              gris_ref_filters={'G141':['ir']},
        #                              files=grism_files,
        #                              refine_mag_limits=[18,23], 
        #                              PREP_PATH='./')
        # 
        # if len(glob.glob(f'{assoc}*_grism*fits*')) == 0:
        #     grism_files = glob.glob('*GrismFLT.fits')
        #     grism_files.sort()
        # 
        #     catalog = glob.glob(f'{assoc}-*.cat.fits')[0]
        #     try:
        #         seg_file = glob.glob(f'{assoc}-*_seg.fits')[0]
        #     except:
        #         seg_file = None
        # 
        #     grp = multifit.GroupFLT(grism_files=grism_files, direct_files=[], 
        #                             ref_file=None, seg_file=seg_file, 
        #                             catalog=catalog, cpu_count=-1, sci_extn=1, 
        #                             pad=256)
        # 
        #     # Make drizzle model images
        #     grp.drizzle_grism_models(root=assoc, kernel='point', scale=0.15)
        # 
        #     # Free grp object
        #     del(grp)
        # 
        # pline = auto_script.DITHERED_PLINE.copy()
        # args_file = f'{assoc}_fit_args.npy'
        # 
        # if (not os.path.exists(args_file)):
        # 
        #     msg = '# generate_fit_params: ' + args_file
        #     utils.log_comment(utils.LOGFILE, msg, verbose=True, show_date=True)
        # 
        #     pline['pixscale'] = 0.1 #mosaic_args['wcs_params']['pixel_scale']
        #     pline['pixfrac'] = 0.5  #mosaic_args['mosaic_pixfrac']
        #     if pline['pixfrac'] > 0:
        #         pline['kernel'] = 'square'
        #     else:
        #         pline['kernel'] = 'point'
        # 
        #     min_sens = 1.e-4
        #     min_mask = 1.e-4
        # 
        #     fit_trace_shift = True
        # 
        #     args = auto_script.generate_fit_params(field_root=assoc, prior=None, MW_EBV=0.0, pline=pline, fit_only_beams=True, run_fit=True, poly_order=7, fsps=True, min_sens=min_sens, min_mask=min_mask, sys_err=0.03, fcontam=0.2, zr=[0.05, 3.4], save_file=args_file, fit_trace_shift=fit_trace_shift, include_photometry=False, use_phot_obj=False)
        # 
        #     os.system(f'cp {args_file} fit_args.npy')

    
