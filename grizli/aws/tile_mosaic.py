"""
Drizzled mosaics in tiles and subregions

Here the sky is tesselated in 4 degree patches with sizes that are 
increased slightly to be integer multiples of 512 0.1" pixels

Individual subtiles are 256 x 0.1" = 25.6"

"""
from tqdm import tqdm
import numpy as np

from . import db

def define_tile_grid(a=4., phase=0.6):
    """
    Define the tile grid following the PS1 tesselation algorithm from 
    T. Budavari
    https://outerspace.stsci.edu/display/PANSTARRS/PS1+Sky+tessellation+patterns#PS1Skytessellationpatterns-ProjectioncellsintheRINGS.V3tessellation

    Paramters
    ---------
    a : float
        Tile size in degrees
        
    phase : float
       Shift the RA grid by some phase amount to better align tiles with some
       survey fields
    """
    import numpy as np
    from matplotlib import pyplot as plt
    
    from astropy import units as u
    import ligo.skymap.plot

    from astropy.coordinates import SkyCoord
    import astropy.wcs as pywcs
    
    from grizli import utils

    # bottom of first decl row at d=0
    dn = [a/2./180*np.pi]
    
    # lower decl row *centered* on d=0
    dn = [0] #a/2./180*np.pi]
    
    # shift up to align survey fields
    dn = [1./180*np.pi]
    
    theta = np.arctan(a/180*np.pi/2)*2
    dn1 = 1
    
    tn = []
    mn = []
    
    dtheta = 1.e-6
    
    while (dn1 < np.pi/2) & (dn1 > 0):
        dlo = dn[-1] - theta/2
        mni = int(np.floor(2*np.pi*np.cos(dlo)/theta)) + 1
        an = 2*np.pi / mni
        dn1 = np.arctan(np.tan(dn[-1] + theta/2.)*np.cos(an/2)) + theta/2
        ddeg = dn1 / np.pi*180
        niter = 0
        
        while ddeg < np.round(ddeg):
            niter += 1
            _theta = theta + dtheta
            dlo = dn[-1] - _theta/2
            _mni = int(np.floor(2*np.pi*np.cos(dlo)/theta)) + 1
            an = 2*np.pi / mni
            _dn1 = np.arctan(np.tan(dn[-1] + _theta/2.)*np.cos(an/2)) + _theta/2
            ddeg = _dn1 / np.pi*180
            if ddeg < np.round(ddeg):
                theta = _theta
                dn1 = _dn1
                mni = _mni
                
        mn.append(mni)
        tn.append(theta)
        
        if (dn1 < np.pi/2.) & (dn1 > 0):
            print(f'{dn[-1]/np.pi*180:.2f} {mni} {niter}')
            dn.append(dn1)
    
    if a == 2:
        dn[-1] = np.pi/2
        mn[-1] = 1
        tn[-1] = theta*2.
        
    elif (a ==4) & (dn[0] == 1./180*np.pi):
        dn[-1] = np.pi/2
        mn[-1] = 1
        tn[-1] = theta*1.5
        
    else:
        dn.append(np.pi/2)
        mn.append(1)
        tn.append(theta)

    # Double for negative 
    n = len(dn)
    strip = [s for s in range(1, n+1)]
    
    if dn[0] == 0:
        strip = [-s for s in strip[::-1]] + strip[1:]
        dn = [-dni for dni in dn[::-1]] + dn[1:]
        mn = mn[::-1] + mn[1:]
        tn = tn[::-1] + tn[1:]
    else:
        strip = [-s for s in strip[::-1]] + strip
        dn = [-dni for dni in dn[::-1]] + dn
        mn = mn[::-1] + mn
        tn = tn[::-1] + tn
    
    strip = [s for s in range(len(dn))]
    
    # Plot footprints
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    kw = {'projection':'astro hours mollweide'}
    kw = {'projection':'astro globe'}
    kw = {'projection':'astro globe', 'center':'0d +70d'}
    
    kw = dict(projection='astro degrees zoom',
                  center='0h 86d', radius='10 deg')

    #kw = dict(projection='astro degrees zoom',
    #            center='0h 0d', radius='6 deg')
                  
    plt.close('all')
    fig, ax = plt.subplots(1,1,figsize=(8,8), 
                           subplot_kw=kw)

    #plt.title("Aitoff projection of our random data")
    
    ax.grid()
    
    nedge = 10
    step = np.linspace(0, 1, nedge)
    zeros = np.zeros(nedge)
    px = np.hstack([step, zeros+1, step[::-1], zeros])
    py = np.hstack([zeros, step, zeros+1, step[::-1]])
    
    dpix = []
    da = []
    
    names = ['tile', 'strip', 'nx', 
             'crpix1','crpix2','crval1','crval2','npix',
             'r1','d1','r2','d2','r3','d3','r4','d4']
    
    rows = []
    tileid = 0
    
    for j in tqdm(range(len(dn))):
                    
        ai = 2*np.tan(tn[j]/2)*180/np.pi
        npix = ai*3600/0.1
        npixr = int(npix // 512 + 1)*512
        ai = npixr*0.1/3600
        da.append(ai)
        dpix.append(npixr)
        
        ddeg = dn[j]/np.pi*180
        
        h, w = utils.make_wcsheader(ra=0, dec=ddeg, size=ai*3600, 
                                    pixscale=0.1)
        h['CRPIX1'] += 0.5
        h['CRPIX2'] += 0.5
        
        ras = np.linspace(0, 2*np.pi, np.maximum(mn[j]+1, 1))[:-1]
        
        # Shift to avoid straddling ra=0
        if mn[j] > 1:
            ras += ras[1]*phase
        
        col = 'k'
        
        for ir, r in enumerate(ras):
            h['CRVAL1'] = r/np.pi*180
            h['CRVAL2'] = ddeg
            w = pywcs.WCS(h)
            fp = w.calc_footprint()
            
            tileid += 1
            row = [tileid, strip[j], ir+1]
            for k in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2']:
                row.append(h[k])
            
            row.append(npixr)
            row.extend(fp.flatten().astype(np.float32).tolist())
            rows.append([d for d in row])
            
            #fpx, fpy = w.calc_footprint().T
            #fpx = np.append(fpx, fpx[0])
            #fpy = np.append(fpy, fpy[0])
            fpx, fpy = w.all_pix2world(px*npixr, py*npixr, 0)
            
            if ir > 0:
                pl = ax.plot_coord(SkyCoord(fpx, fpy, unit=('deg','deg')), 
                          color=col, alpha=0.5, linewidth=1)#, marker='.')
            else:
                pl = ax.plot_coord(SkyCoord(fpx, fpy, unit=('deg','deg')), 
                                   alpha=0.5, linewidth=1)#, marker='.')
                col = pl[0].get_color()
            
            if (j == 42) & (ir == 1):
                for ik in range(5):
                    for jk in range(5):
                        slx = slice(ik*256, (ik+1)*256)
                        sly = slice(jk*256, (jk+1)*256)
                        
                        wsl = w.slice((sly, slx))
                        fpx, fpy = wsl.all_pix2world(px*256, py*256, 0)
                        pl = ax.plot_coord(SkyCoord(fpx, fpy, 
                                                    unit=('deg','deg')), 
                                  color='k', alpha=0.5, linewidth=1)
    
    tab = utils.GTable(names=names, rows=rows)
    
    return tab
    
    ###### Checking
    df = tab.to_pandas()
    df.to_sql('mosaic_tiles', db._ENGINE, index=False, if_exists='fail', 
              method='multi')
    
    # Look at HST fields
    
    avg_coo = db.SQL("""SELECT parent, count(parent), 
                               avg(ra) as ra, avg(dec) as dec
                        FROM assoc_table where ra > 0
                        GROUP by parent order by count(parent)""")
    
    exp = db.SQL('SELECT assoc, crval1, crval2, footprint FROM exposure_files')
    coo = SkyCoord(exp['crval1'], exp['crval2'], unit=('deg','deg'))
    
    # old
    avg_coo = db.SQL("""SELECT parent, count(parent), 
                               avg(ra) AS ra, avg(dec) as dec 
                        FROM exposure_log
                        WHERE ra > 0 
                          AND awspath not like '%%grizli-cosmos-v2%%'
                        GROUP by parent order by count(parent)""")
                        
    exp = db.SQL("""SELECT parent, ra, dec 
                    FROM exposure_log 
                    WHERE ra > 0 
                      AND awspath NOT LIKE '%%grizli-cosmos-v2%%'""")
                      
    coo = SkyCoord(exp['ra'], exp['dec'], unit=('deg','deg'))
    
    ra, dec = 150.1, 2.2 # cosmos
    ra, dec = 150.1, 2.2 # cosmos
    
    test = avg_coo['count'] > 20
    test = avg_coo['count'] > 150
    
    for k in np.where(test)[0]:
        print(avg_coo['parent'][k])
        ra, dec = avg_coo['ra'][k], avg_coo['dec'][k]
        
        ctab = utils.GTable()
        ctab['ra'] = [ra]
        ctab['dec'] = dec
        idx, dr = ctab.match_to_catalog_sky(tab, other_radec=('crval1','crval2'))
    
        kw = dict(projection='astro degrees zoom',
                      center=f'{ra}d {dec}d', radius='9 deg')
                  
        #plt.close('all')
        fig, ax = plt.subplots(1,1,figsize=(8,8), 
                               subplot_kw=kw)
    
        corners = np.array([np.array(tab[c]) for c in ['r1','d1','r2','d2','r3','d3','r4','d4']])[:,dr < 10*u.deg].T
    
        ax.plot_coord(coo, linestyle='None', marker='.')
    
        for c in corners:
            cc = np.append(c, c[:2])
            ci = SkyCoord(cc[0::2], cc[1::2], unit=('deg','deg'))
            ax.plot_coord(ci, color='k', alpha=0.5, linewidth=1)
        
        ax.grid()
        ax.set_title(avg_coo['parent'][k])


def add_exposure_batch():
    """
    Add a bunch of exposures to the `mosaic_tiles_exposures` table
    """    
    import astropy.table
    import astropy.table
    from tqdm import tqdm
    
    from grizli.aws.tile_mosaic import add_exposure_to_tile_db
    
    filters = db.SQL("""SELECT filter, count(filter) 
                          FROM exposure_files 
                        GROUP BY filter 
                        ORDER BY count(filter)""")
    
    ii = len(filters)-1
    filt = 'F814W'
    
    for ii, filt in enumerate(filters['filter']):
        print(f"{ii} / {len(filters)} {filt} {filters['count'][ii]}")
        
        exp = db.SQL(f"""SELECT eid, assoc, dataset, extension, filter, 
                                sciext, crval1 as ra, crval2 as dec, footprint
                           FROM exposure_files
                           WHERE filter = '{filt}'""")
    
        tiles = db.SQL('select * from mosaic_tiles')
    
        res = [add_exposure_to_tile_db(row=exp[i:i+1], tiles=tiles)
                  for i in tqdm(range(len(exp)))]
        
        for j in range(len(res))[::-1]:
            if res[j] is None:
                print(f"Pop {exp['assoc'][j]} {j}")
                res.pop(j)
                
        db.execute(f"""DELETE from mosaic_tiles_exposures t
                USING exposure_files e
                WHERE t.expid = e.eid
                AND filter = '{filt}'
                """)
        
        N = 100
        for j in tqdm(range(len(res)//N+1)):
            sl = slice(j*N, (j+1)*N)
            #print(j, sl.start, sl.stop)
            
            tab = astropy.table.vstack(res[sl])
             
            df = tab.to_pandas()
            df.to_sql('mosaic_tiles_exposures', db._ENGINE, index=False, 
                      if_exists='append', method='multi')
    
    # Exposure map
    if 0:
        import ligo.skymap.plot
        from matplotlib import pyplot as plt
        from astropy.coordinates import SkyCoord
        from grizli.aws import tile_mosaic
        
        filt = 'F160W'
        
        # CDFS
        ra, dec, rsize, name = 53.14,-27.78, 20, 'gds'
        ra, dec, rsize, name = 189.28, 62.25, 20, 'gdn'
        #ra, dec, rsize, name = 150.0, 2.0, 90, 'cos'
        
        cosd = np.cos(dec/180*np.pi)
        
        res = db.SQL(f"""SELECT tile, subx, suby, subra, subdec, filter, 
                                COUNT(filter) as nexp, 
                                SUM(exptime) as exptime,
                                MIN(expstart) as tmin, 
                                MAX(expstart) as tmax 
                FROM mosaic_tiles_exposures t, exposure_files e
                WHERE t.expid = e.eid
                AND filter = '{filt}'
                AND ABS(subra - {ra})*{cosd} < {rsize/60}
                AND ABS(subdec - {dec}) < {rsize/60}
                GROUP BY tile, subx, suby, subra, subdec, filter
                """)

        kw = dict(projection='astro hours zoom',
                  center=f'{ra}d {dec}d', radius=f'{rsize} arcmin')
        
        s = np.maximum(28*18/rsize, 1)
        
        fig, ax = plt.subplots(1,1,figsize=(6,6), 
                               subplot_kw=kw)
        
        ax.grid()
        
        coo = SkyCoord(res['subra'], res['subdec'], unit=('deg','deg'))
        ax.scatter_coord(coo, c=np.log10(res['exptime']), marker='s', s=s)
        
        for t in np.unique(res['tile']):
            twcs = tile_mosaic.tile_wcs(t)
            coo = SkyCoord(*twcs.calc_footprint().T, unit=('deg','deg'))
            ax.plot_coord(coo, color='r', linewidth=1.2, alpha=0.5)
        
        ax.set_xlabel('R.A.')
        ax.set_ylabel('Dec.')
        ax.set_title(f'{name} - {filt}')
        
        fig.tight_layout(pad=1)
        
    if 0:
        db.execute('ALTER TABLE exposure_files ADD COLUMN eid SERIAL PRIMARY KEY;')
        
        db.execute('GRANT ALL PRIVILEGES ON ALL TABLEs IN SCHEMA public TO db_iam_user')
        db.execute('GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO db_iam_user')
        
        db.execute('ALTER TABLE assoc_table ADD COLUMN aid SERIAL PRIMARY KEY;')
        db.execute('CREATE INDEX on exposure_files (eid)')
        db.execute('CREATE INDEX on mosaic_tiles_exposures (expid)')
        
                    
def add_exposure_to_tile_db(dataset='ibev8xubq', sciext=1, tiles=None, row=None):
    """
    Find subtiles that overlap with an exposure in the `exposure_files`
    table
    """
    import astropy.table
    import astropy.units as u
    import astropy.wcs as pywcs
    
    import numpy as np
    from grizli import utils
    
    if tiles is None:
        tiles = db.SQL('select * from mosaic_tiles')

    if row is None:
        row = db.SQL(f"""SELECT eid, assoc, dataset, extension, filter, 
                                sciext, crval1 as ra, crval2 as dec, footprint 
                           from exposure_files
                           where filter = 'F160W' 
                             and dataset = '{dataset}' 
                             AND sciext={sciext}""")
    
    if len(row) == 0:
        print(f'No exposure data found for dataset={dataset} sciext={sciext}')
        return None
                    
    idx, dr = row.match_to_catalog_sky(tiles, other_radec=('crval1','crval2'))
    
    tix = np.where(dr < 4*np.sqrt(2)*u.deg)[0]
    #tix = np.where(dr < 4*u.deg)[0]
    
    exp_poly = None
    for fp in row['footprint']:
        sr = utils.SRegion(fp)
        for p, s in zip(sr.get_patch(alpha=0.5, color='k'), sr.shapely):
            if exp_poly is None:
                exp_poly = s
            else:
                exp_poly = exp_poly.union(s)
                
            # ax.add_patch(p)
    
    sbuff = utils.SRegion(np.array(exp_poly.buffer(1./60).boundary.xy).T)
    
    tabs = []
    
    for t in tix:
        
        h, w = utils.make_wcsheader(ra=tiles['crval1'][t], 
                                    dec=tiles['crval2'][t],
                                    size=tiles['npix'][t]*0.1, 
                                    pixscale=0.1)
        
        h['CRPIX1'] += 0.5
        h['CRPIX2'] += 0.5
        h['LATPOLE'] = 0.
        
        w = pywcs.WCS(h)
            
        srt = utils.SRegion(w.calc_footprint())
        if not sr.shapely[0].intersects(srt.shapely[0]):
            continue
        
        nsub = tiles['npix'][t]//256
        step = np.arange(nsub)
        px, py = np.meshgrid(step, step)
        px = px.flatten()
        py = py.flatten()
        rd = w.all_pix2world(px*256+128, py*256+128, 0)
        pts = np.array([rd[0], rd[1]]).T
        test = sbuff.path[0].contains_points(pts)
        tw = np.where(test)[0]
        if test.sum() == 0:
            continue
        
        for j, xi, yi in zip(tw, px[tw], py[tw]):
            wsl = w.slice((slice(yi*256, (yi+1)*256), 
                           slice(xi*256, (xi+1)*256)))
            sw = utils.SRegion(wsl.calc_footprint())
            test[j] = sw.shapely[0].intersects(exp_poly)
        
        if test.sum() == 0:
            continue
            
        tmatch = utils.GTable()
        tmatch['tile'] = [tiles['tile'][t]] * test.sum()
        tmatch['subx'] = px[test]
        tmatch['suby'] = py[test]
        tmatch['subra'] = rd[0][test]
        tmatch['subdec'] = rd[1][test]
        tmatch['expid'] = np.unique(row['eid'])[0]
        # tmatch['eassoc'] = np.unique(row['assoc'])[0]
        # tmatch['edataset'] = np.unique(row['dataset'])[0]
        # tmatch['eext'] = np.unique(row['sciext'])[0]
        tmatch['in_mosaic'] = 0
        tabs.append(tmatch)
        
    if len(tabs) > 0:
        tmatch = astropy.table.vstack(tabs)
        return tmatch
    else:
        return None


def tile_wcs(tile):
    """
    Compute tile WCS
    """
    import astropy.wcs as pywcs
    from grizli import utils
    from grizli.aws import db
    
    row = db.SQL(f"""SELECT crval1, crval2, npix
                       FROM mosaic_tiles
                      WHERE tile={tile}""")

    t = 0
    h, w = utils.make_wcsheader(ra=row['crval1'][t], dec=row['crval2'][t],
                                size=row['npix'][t]*0.1, pixscale=0.1)
    
    h['CRPIX1'] += 0.5
    h['CRPIX2'] += 0.5
    h['LATPOLE'] = 0.
    
    wcs = pywcs.WCS(h)
    wcs.pscale = 0.1
    
    return wcs
    
    
def tile_subregion_wcs(tile, subx, suby):
    """
    Compute WCS for a tile subregion
    """
    
    twcs = tile_wcs(tile)
    
    sub_wcs = twcs.slice((slice(suby*256, (suby+1)*256), 
                          slice(subx*256, (subx+1)*256)))
    sub_wcs.pscale = 0.1
    return sub_wcs


#
def get_lambda_client(region_name='us-east-1'):
    """
    Get boto3 client in same region as HST public dataset
    """
    import boto3
    session = boto3.Session()
    client = session.client('lambda', region_name=region_name)
    return client


def send_event_lambda(event, verbose=True, client=None, func='grizli-mosaic_tile'):
    """
    Send a single event to AWS lambda
    """
    import time
    import os
    import yaml

    import numpy as np
    import boto3
    import json

    if client is None:
        client = get_lambda_client(region_name='us-east-1')

    if verbose:
        print('Send event to {0}: {1}'.format(func, event))

    response = client.invoke(FunctionName=func,
                             InvocationType='Event', LogType='Tail',
                             Payload=json.dumps(event))


def count_locked():
    """
    Count number of distinct locked tiles that could still be runing
    on aws lambda
    """
    tiles = db.SQL("""SELECT tile, subx, suby, filter, count(filter)
                      FROM mosaic_tiles_exposures t, exposure_files e
                               WHERE t.expid = e.eid AND tile != 1183
                               AND in_mosaic=9
                               GROUP BY tile, subx, suby, filter
                               ORDER BY filter DESC""")
    
    return len(tiles)


def get_subtile_status(tile=2530, subx=522, suby=461, **kwargs):
    """
    `in_mosaic` status of all entries of a subtile
    """
    resp = db.SQL(f"""SELECT tile, subx, suby, filter, 
                             in_mosaic, count(filter)
                FROM mosaic_tiles_exposures t, exposure_files e
                WHERE t.expid = e.eid
                AND tile={tile} AND subx={subx} AND suby={suby}
                GROUP BY tile, subx, suby, filter, in_mosaic
                ORDER BY (filter, in_mosaic)
                """)
    return resp


def send_all_tiles():
    
    import time
    import os
    from grizli.aws.tile_mosaic import drizzle_tile_subregion
    
    tiles = []
    
    client = get_lambda_client()
    
    nt0 = len(tiles)
    tiles = db.SQL("""SELECT tile, subx, suby, filter, count(filter)
       FROM mosaic_tiles_exposures t, exposure_files e
                WHERE t.expid = e.eid AND in_mosaic = 0 AND tile != 1183
                GROUP BY tile, subx, suby, filter
                ORDER BY filter DESC
                """)
                
    nt1 = len(tiles)
    print(nt1, nt1-nt0)
    
    istart = i = -1
    step = 150 - count_locked()
    
    #for i in range(len(tiles)):
    while i < len(tiles):
        i+=1 
        
        if i-istart == step:
            istart = i
            print(f'{time.ctime()}: Pause for 60s')
            time.sleep(60)
            skip = 150 - count_locked()
            print(f'{time.ctime()}: Run {skip} more')
        
        if tiles['tile'][i] == 1183:
            continue
            
        event = dict(tile=int(tiles['tile'][i]), 
                     subx=int(tiles['subx'][i]),
                     suby=int(tiles['suby'][i]),
                     filter=tiles['filter'][i], 
                     counter=i, 
                     time=time.ctime())
        
        if 1:
            send_event_lambda(event, client=client, func='grizli-mosaic-tile')
        
        else:
            drizzle_tile_subregion(**event, 
                                   s3output=None,
                                   ir_wcs=None, make_figure=False, 
                                   skip_existing=False, verbose=True, 
                                   gzip_output=False)
            
            files='tile.{tile:04d}.{subx:03d}.{suby:03d}.{fx}*fits'
            os.system('rm {files}'.format(fx=event['filter'].lower(), 
                                             **event))
        
        # if (i+1) % 300 == 0:
        #     break
            #time.sleep(10)


def get_tile_status(tile, subx, suby, filter):
    """
    Is a tile "locked" with all exposures set with in_mosaic = 9?
    """

    exp = db.SQL(f"""SELECT dataset, extension, assoc, filter, 
                            exptime, footprint, in_mosaic
                      FROM mosaic_tiles_exposures t, exposure_files e
                      WHERE t.expid = e.eid
                      AND filter='{filter}' AND tile={tile}
                      AND subx={subx} AND suby={suby}""")
    
    if len(exp) == 0:
        status = 'empty'
    elif (exp['in_mosaic'] == 9).sum() == len(exp):
        status = 'locked'
    elif (exp['in_mosaic'] == 1).sum() == len(exp):
        status = 'completed'
    else:
        status = 'go'
    
    return exp, status


def drizzle_tile_subregion(tile=2530, subx=522, suby=461, filter='F160W', s3output=None, ir_wcs=None, make_figure=False, skip_existing=False, verbose=True, gzip_output=False, **kwargs):
    """
    """
    import os
    import astropy.table
    import astropy.units as u
    import astropy.wcs as pywcs
    import astropy.io.fits as pyfits
    
    import numpy as np
    from grizli import utils
    from grizli.aws import visit_processor
    
    exp, status = get_tile_status(tile, subx, suby, filter) 
    
    root = f'tile.{tile:04d}.{subx:03d}.{suby:03d}'
    
    if status in ['empty']:
        if verbose:
            print(f'{root} {filter} ! No exposures found')
            
        return True
    
    elif status in ['locked']:
        print(f'{root} {filter} tile locked')
        return True
    
    elif (status in ['completed']) & (skip_existing):
        print(f'{root} {filter} tile already completed')
        return True
        
    sci_file = f'{root}.{filter.lower()}_drz_sci.fits'
    if skip_existing & os.path.exists(sci_file):
        print(f'Skip file {sci_file}')
        return True
    
    # Lock
    db.execute(f"""UPDATE mosaic_tiles_exposures t
          SET in_mosaic = 9
          FROM exposure_files w
          WHERE t.expid = w.eid
          AND w.filter='{filter}' AND tile={tile}
          AND subx={subx} AND suby={suby}""")
        
    if ir_wcs is None:
        ir_wcs = tile_subregion_wcs(tile, subx, suby)
    
    if verbose:
        print(f'{root} {filter} {len(exp)}')
        
    if s3output is None:
        s3output = f's3://grizli-v2/HST/Pipeline/Tiles/{tile}/'
    
    visit_processor.cutout_mosaic(rootname=root,
                                  product='{rootname}.{f}',
                                  ir_wcs=ir_wcs,
                                  res=exp, 
                                  s3output=s3output, 
                                  make_figure=make_figure,
                                  skip_existing=skip_existing,
                                  gzip_output=gzip_output,
                                  **kwargs)
    
    # Update subtile status
    db.execute(f"""UPDATE mosaic_tiles_exposures t
          SET in_mosaic = 1
          FROM exposure_files w
          WHERE t.expid = w.eid
          AND w.filter='{filter}' AND tile={tile}
          AND subx={subx} AND suby={suby}
           """)


def query_cutout(output='cutout', ra=189.0243001, dec=62.1966953, size=10, filter='F160W'):
    """
    """
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from grizli import utils
    from grizli.aws import tile_mosaic
    
    dd = size/3600
    dr = dd/np.cos(dec/180*np.pi)
    
    bd = 256*0.1/3600
    br = bd/np.cos(dec/180*np.pi)
    
    ss = size/3600/np.cos(dec/180*np.pi)
    
    SQL = f"""SELECT tile, subx, suby, subra, subdec
              FROM mosaic_tiles_exposures t, exposure_files e
              WHERE in_mosaic = 1 AND filter = '{filter.upper()}'
              AND t.expid = e.eid 
              AND polygon ( (
                '((' || subra - {bd} || ', ' || subdec - {br} || '),
                  (' || subra - {bd} || ', ' || subdec + {br} || '),
                  (' || subra + {bd} || ', ' || subdec + {br} || '),
                  (' || subra + {bd} || ', ' || subdec - {br} || '))')::path )
                  ?#
                  polygon (
                   ('(( {ra - dd}  , {dec - dr}), 
                     ( {ra - dd}  , {dec + dr}), 
                     ( {ra + dd}  , {dec + dr}), 
                     ( {ra + dd}  , {dec - dr}))')::path )
              GROUP BY tile, subx, subx"""

    SQL = f"""SELECT tile, subx, suby, subra, subdec
            FROM mosaic_tiles_exposures t, exposure_files e
            WHERE in_mosaic = 1 AND filter = '{filter.upper()}'
            AND t.expid = e.eid 
            AND (
              '((' || subra - {bd} || ', ' || subdec - {br} || '),
                (' || subra - {bd} || ', ' || subdec + {br} || '),
                (' || subra + {bd} || ', ' || subdec + {br} || '),
                (' || subra + {bd} || ', ' || subdec - {br} || '))')::polygon
                && ('(( {ra - dd}  , {dec - dr}), 
                     ( {ra - dd}  , {dec + dr}), 
                     ( {ra + dd}  , {dec + dr}), 
                     ( {ra + dd}  , {dec - dr}))')::polygon
            GROUP BY tile, subx, suby, subra, subdec"""
    #
    cosd = np.cos(dec/180*np.pi)
    rc = size/3600
    rtile = np.sqrt(2)*128*0.1/3600
    
    SQL = f"""SELECT tile, subx, suby, subra, subdec
            FROM mosaic_tiles_exposures t, exposure_files e
            WHERE in_mosaic = 1 AND filter = '{filter.upper()}'
            AND t.expid = e.eid 
            AND ('((' || (subra - {ra})*{cosd} || 
                    ', ' || subdec - {dec} || '),
                    {rtile})')::circle
                && ('((0,0),{rc})')::circle
            GROUP BY tile, subx, suby, subra, subdec"""
    
    res = db.SQL(SQL) 
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.scatter(res['subra'], res['subdec'])
    
    from shapely.geometry.point import Point
    import shapely.affinity
    from descartes import PolygonPatch

    # Let create a circle of radius 1 around center point:
    circ = shapely.geometry.Point((ra, dec)).buffer(rc)
    # Let create the ellipse along x and y:
    ell  = shapely.affinity.scale(circ, 1./cosd, 1.)
    ax.add_patch(PolygonPatch(ell, color='r', alpha=0.5))
    
    for tile, subx, suby in tqdm(zip(res['tile'], res['subx'], res['suby'])):
        tw = tile_mosaic.tile_subregion_wcs(tile, subx, suby)
        sr = utils.SRegion(tw)
        ax.add_patch(sr.get_patch(alpha=0.5)[0])
        
                
def build_mosaic_from_subregions(tile=2530, files=None, filter='f140w'):
    """
    TBD
    """
    from tqdm import tqdm
    import glob
    import numpy as np
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    import os
    
    if 0:
        for filt in ['f105w','f140w','f160w']:
            build_mosaic_from_subregions(tile=2530, files=None, filter=filt)
            
    #tile = 2530
    #filter = 'f140w'
    
    if files is None:
        files = glob.glob(f'*{tile:04d}.*_drz_sci.fits')
        files.sort()
        
    tx = np.array([int(f.split('.')[2]) for f in files])
    ty = np.array([int(f.split('.')[3]) for f in files])
    
    txm, tym = tx.min(), ty.min()
    nx = tx.max() - txm + 1
    ny = ty.max() - tym + 1
    
    if '_drc' in files[0]:
        npix = 512
    else:
        npix = 256
        
    img = np.zeros((ny*npix, nx*npix), dtype=np.float32)
    
    h = None
    for f, xi, yi in tqdm(zip(files, tx, ty)):
                    
        im = pyfits.open(f)
        
        if h is None:
            h = im[0].header
        
        if xi == txm:
            h['CRPIX1'] = im[0].header['CRPIX1']
        
        if yi == tym:
            h['CRPIX2'] = im[0].header['CRPIX2']
        
        if filter in f:
            slx = slice((xi-txm)*npix, (xi-txm+1)*npix)
            sly = slice((yi-tym)*npix, (yi-tym+1)*npix)
            img[sly, slx] += im[0].data

        
    pyfits.writeto(f'mos.{tile}.{filter}_sci.fits', data=img, 
                   header=h, overwrite=True)
    
    os.system(f'aws s3 cp mos.{tile}.{filter}_sci.fits s3://grizli-v2/Scratch/')
    
    
