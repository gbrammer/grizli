"""
Drizzled mosaics in tiles and subregions

Here the sky is tesselated in 4 degree patches with sizes that are 
increased slightly to be integer multiples of 512 0.1" pixels

Individual subtiles are 256 x 0.1" = 25.6"

"""
import os
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
        #
        # Table updates
        if 0:
            db.execute('ALTER TABLE exposure_files ADD COLUMN eid SERIAL PRIMARY KEY;')

            db.execute('GRANT ALL PRIVILEGES ON ALL TABLEs IN SCHEMA public TO db_iam_user')
            db.execute('GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO db_iam_user')
            db.execute('GRANT SELECT ON ALL TABLEs IN SCHEMA public TO readonly')
            db.execute('GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO readonly')
            
            db.execute('ALTER TABLE assoc_table ADD COLUMN aid SERIAL PRIMARY KEY;')
            db.execute('CREATE INDEX on exposure_files (eid)')
            db.execute('CREATE INDEX on exposure_files (eid,filter)')
            db.execute('CREATE INDEX on mosaic_tiles_exposures (expid)')

            db.execute('CREATE INDEX on mosaic_tiles_exposures (tile, subx, suby)')
        
def make_exposure_maps():
    """
    """

    from grizli import utils
    from grizli.aws import tile_mosaic
    from grizli.aws import db
    
    filt = 'F160W'
    
    ra, dec, rsize, name =  53.14628, -27.814, 20, 'gds'
    #ra, dec, rsize, name = 189.22592,  62.24586, 20, 'gdn'
    # ra, dec, rsize, name = 214.95, 52.9, 20, 'egs'
    ra, dec, rsize, name = 150.11322, 2.24068, 48, 'cos'
    ra, dec, rsize, name = 34.34984, -5.18390, 30, 'uds'
    
    ra, dec, rsize, name = 177.40124999999998, 22.39947222222, 12, 'macs1149'
    ra, dec, rsize, name = 157.30641, 26.39197, 12, 'sdss1029'
    ra, dec, rsize, name = 215.93, 24.07, 15, 'macs1423'
    ra, dec, rsize, name = 64.39, -11.91, 15, 'macs0417'
    ra, dec, rsize, name = 3.5301941, -30.3854942, 15, 'abell2744'
    extra = ''
    
    fig, tab = tile_mosaic.exposure_map(ra, dec, rsize, name, 
                                        filt=filt.upper(), s0=18, 
                                        extra=extra)
    fig.tight_layout(pad=0.5)
    fig.savefig('/tmp/map.png')
    fig.tight_layout(pad=0.5)
    fig.savefig('/tmp/map.png')
    
    from grizli.aws import db
    mf = db.SQL("""
    SELECT * from mosfire_extractions natural join mosfire_datemask
    """)
    
    # MF with HST
    # SQL = f"""SELECT m.file, count(m.file)
    #         FROM mosfire_extractions m, exposure_files e
    #         WHERE ('(' || ra_targ || 
    #                 ', ' || dec_targ || ')')::point
    #                 <@ polygon(e.footprint)
    #         GROUP BY m.file"""
    # 
    # res = db.SQL(SQL) 
    
    ###############
    
def find_mosaic_segments(bs=16):
    """
    
    Find "segments" of connected subimages within a tile
    
    bs : bin size relative to 256*0.1" subimages
    
    """
    from scipy.ndimage import label

    from grizli import utils
    from grizli.aws import db
    
    cells = db.SQL(f"""SELECT tile, subx, suby, subra, subdec, filter, 
                              assoc, dataset, exptime
            FROM mosaic_tiles_exposures t, exposure_files e
            WHERE t.expid = e.eid
            """)
    
    cells['segment'] = 0
    
    un = utils.Unique(cells['tile'])
    ns = 0
    
    for t in un.values:
        uni = np.where(un[t])[0]
        
        # img = np.zeros((cells['suby'][uni].max()+1, 
        #                 cells['subx'][uni].max()+1), dtype=bool)
        # img[cells['suby'][uni], cells['subx'][uni]] = True
    
        bs = 16
    
        img = np.zeros((cells['suby'][uni].max()//bs+1, 
                        cells['subx'][uni].max()//bs+1), dtype=bool)
        img[cells['suby'][uni]//bs, cells['subx'][uni]//bs] = True
    
        labeled_array, num_features = label(img)
    
        cells['segment'][uni] = labeled_array[cells['suby'][uni]//bs, 
                                              cells['subx'][uni]//bs] + ns
        
        print(f'tile: {t}  npatch: {num_features}')
        
        ns += num_features

    
    names = ['tile','patch','ra','dec','jname','count','filter',
             'xmin','xmax','ymin','ymax',
             'rmin','rmax','dmin','dmax','status','mtime']
    
    un = utils.Unique(cells['segment'])
    rows = []
    
    import astropy.time
    from tqdm import tqdm
    
    for s in tqdm(un.values):
        uni = np.where(un[s])[0]
        
        unf = utils.Unique(cells['filter'][uni], verbose=False)
        ra = np.mean(cells['subra'][uni])
        dec = np.mean(cells['subdec'][uni])
        jname = utils.radec_to_targname(ra=ra, dec=dec)
        for f in unf.values:
            unfi = cells[uni][unf[f]]
            
            row = [unfi['tile'][0], unfi['segment'][0], 
                   ra, dec, jname, len(unfi), f, 
                   unfi['subx'].min(), unfi['subx'].max(), 
                   unfi['suby'].min(), unfi['suby'].max(), 
                   unfi['subra'].min(), unfi['subra'].max(), 
                   unfi['subdec'].min(), unfi['subdec'].max(),
                   0, astropy.time.Time.now().mjd]
            rows.append(row)
    
    patches = utils.GTable(names=names, rows=rows)
    
    i = 0
    t0 = 0
    
    for i in range(30):
        t = patches['tile'][i]

        os.system(f"""aws s3 sync s3://grizli-mosaic-tiles/Tiles/{t}/ ./  --exclude "*" --include "*{patches['filter'][i].lower()}*sci.fits"  """)
    
        build_mosaic_from_subregions(root=patches['jname'][i], tile=t, 
                                     files=None, 
                                     filter=patches['filter'][i].lower())
        #
        os.system(f'rm tile.{t0:04d}*')
        


def get_axis_center_coord(ax):
    """
    Get sky coords at the center of symap axis
    """
    
    tr = ax.get_transform('world').inverted()
    return tr.transform((np.mean(ax.get_xlim()), np.mean(ax.get_ylim())))


def exposures_in_axis(ax, extra_where=""):
    """
    Query exposure_files
    """
    coo = get_axis_center_coord(ax)
    
    point = f"point '({coo[0]}, {coo[1]})'"
    
    res = db.SQL(f"""SELECT file, filter, assoc from exposure_files 
    WHERE polygon(footprint) @> {point}
    {extra_where}
    ORDER BY assoc
    """)
    return res
    
    
def exposure_map(ra, dec, rsize, name, filt='F160W', s0=16, cmap='viridis', figsize=(6,6), show_tiles=True, res=None, alpha=1., ec='None', vmin=None, vmax=None, extra=''):
    """
    Make an exposure map from a database query
    """
    import ligo.skymap.plot
    from matplotlib import pyplot as plt
    import numpy as np
    
    from astropy.coordinates import SkyCoord
    from grizli.aws import tile_mosaic
    from grizli import utils
    
    cosd = np.cos(dec/180*np.pi)
    
    if res is None:
        res = db.SQL(f"""SELECT tile, subx, suby, subra, subdec, filter, 
                            COUNT(filter) as nexp, 
                            SUM(exptime) as exptime,
                            MIN(expstart) as tmin, 
                            MAX(expstart) as tmax 
            FROM mosaic_tiles_exposures t, exposure_files e
            WHERE t.expid = e.eid
            AND filter = '{filt}'
            {extra}
            AND ABS(subra - {ra})*{cosd} < {rsize/60}
            AND ABS(subdec - {dec}) < {rsize/60}
            GROUP BY tile, subx, suby, subra, subdec, filter
            """)

    kw = dict(projection='astro hours zoom',
              center=f'{ra}d {dec}d', radius=f'{rsize} arcmin')
    
    s = np.maximum(s0*18/rsize, 1)
    
    fig, ax = plt.subplots(1,1,figsize=figsize, 
                           subplot_kw=kw)
    
    ax.grid()
    
    coo = SkyCoord(res['subra'], res['subdec'], unit=('deg','deg'))
    ax.scatter(coo.ra, coo.dec, 
               c=np.log10(res['exptime']),
               marker='s', s=s, 
               cmap=cmap, alpha=alpha, ec=ec, 
               vmin=vmin, vmax=vmax, 
               transform=ax.get_transform('world'))
               
    #ax.scatter_coord(coo, c=np.log10(res['exptime']), marker='s', s=s, 
    #                 cmap=cmap, alpha=alpha, ec=ec, vmin=vmin, vmax=vmax)
        
    if show_tiles:
        for t in np.unique(res['tile']):
            twcs = tile_mosaic.tile_wcs(t)
            coo = SkyCoord(*twcs.calc_footprint().T, unit=('deg','deg'))
            ax.plot_coord(coo, color='r', linewidth=1.2, alpha=0.5)

        # Tile labels
        un = utils.Unique(res['tile'], verbose=False)

        dp = 512*0.1/3600
        rp = dp/cosd
    
        for t in un.values:
            c = (res['subra'][un[t]].min(), res['subdec'][un[t]].min(), 
                 res['subra'][un[t]].max(), res['subdec'][un[t]].max())
            #
            cp = (res['subx'][un[t]].min(), res['suby'][un[t]].min(), 
                 res['subx'][un[t]].max(), res['suby'][un[t]].max())
        
            rc = np.array([c[0]-rp, c[0]-rp, c[2]+rp, c[2]+rp, c[0]-rp])
            dc = np.array([c[1]-dp, c[3]+dp, c[3]+dp, c[1]-dp, c[1]-dp])
        
            label = f'{t:04d}: {cp[0]:03d} - {cp[2]:03d}, {cp[1]:03d} - {cp[3]:03d}'
        
            ax.plot_coord(SkyCoord(rc, dc, unit=('deg','deg')),
                          alpha=0.8, label=label, linewidth=1.2)
    
        ax.legend()
    
    ax.set_xlabel('R.A.')
    ax.set_ylabel('Dec.')
    ax.set_title(f'{name} - {filt}')
    
    #fig.tight_layout(pad=0.8)
    
    return fig, res


TILES = None
TILE_WCS = None

def coords_to_subtile(ra=189.0243001, dec=62.19669, size=0):
    """
    Get tile/subtile associated with sky coordinates
    """
    from grizli import utils
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    global TILES, TILE_WCS
    
    if TILES is None:
        print('Initialize TILES')
        
        if os.path.exists('TILES.csv'):
            TILES = utils.read_catalog('TILES.csv')
        else:
            TILES = db.SQL('select * from mosaic_tiles')
        
        TILES['nsub'] = TILES['npix'] // 256
        TILES['coo'] = SkyCoord(TILES['crval1'], TILES['crval2'], 
                                unit=('deg','deg'))
    
    dr = TILES['coo'].separation(SkyCoord(ra, dec, unit=('deg','deg')))
    in_tile = dr < (2*np.sqrt(2)*u.deg)
    xTILES = TILES[in_tile]
    
    tp = np.array([np.squeeze(tile_wcs(t).all_world2pix([ra], [dec], 
                                                        0)).flatten() 
                   for t in xTILES['tile']]).T
                   
    in_tile = ((tp > 0) & (tp < xTILES['npix'])).sum(axis=0) == 2
    subt = xTILES[in_tile]
    
    subt['fsubx'], subt['fsuby'] = tp[:,in_tile] / 256
    subt['subx'] = subt['fsubx'].astype(int)
    subt['suby'] = subt['fsuby'].astype(int)
    
    ds = size/0.1/256
    subt['xmin'] = np.clip(subt['fsubx'] - ds, 0, subt['npix']).astype(int)
    subt['xmax'] = np.clip(subt['fsubx'] + ds, 0, subt['npix']).astype(int)
    subt['ymin'] = np.clip(subt['fsuby'] - ds, 0, subt['npix']).astype(int)
    subt['ymax'] = np.clip(subt['fsuby'] + ds, 0, subt['npix']).astype(int)
    
    subt['ncut'] = (subt['xmax'] - subt['xmin'] + 1)
    subt['ncut'] *= (subt['ymax'] - subt['ymin'] + 1)
    so = np.argsort(subt['ncut'])
    
    return subt[so]


def cutout_from_coords(output='mos-{tile}-{filter}_{drz}', ra=189.0243001, dec=62.1966953, size=10, filters=['F160W'], theta=0, clean_subtiles=False, send_to_s3=False, make_weight=True, **kwargs): 
    
    subt = coords_to_subtile(ra=ra, dec=dec, size=size)[0]
    
    ll = (subt['xmin'], subt['ymin'])
    ur = (subt['xmax'], subt['ymax'])
    
    resp = []
    for filt in filters:
        ri = build_mosaic_from_subregions(root=output, 
                                     tile=subt['tile'], files=None, 
                                     filter=filt, 
                                     ll=ll, ur=ur,
                                     clean_subtiles=clean_subtiles, 
                                     make_weight=make_weight,
                                     send_to_s3=send_to_s3)
        resp.append(ri)
        
    return resp


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
    
    global TILES
    
    if TILES is None:
        TILES = db.SQL('select * from mosaic_tiles')
        
    if tiles is None:
        tiles = TILES

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
        sra = sr.xy[0][:,0]
        if (sra.min() < 10) & (sra.max() > 350):
            sra[sra > 350] -= 360
            
        sr.xy[0][:,0] = sra
        
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
        wfp = w.calc_footprint()
        wra = wfp[:,0]
        if (wra.min() < 10) & (wra.max() > 350):
            if sr.centroid[0][0] < 10:
                wra[wra > 350] -= 360
            else:
                wra[wra < 10] += 360
                
        wfp[:,0] = wra
        
        srt = utils.SRegion(wfp)
        
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
    
    global TILES
    if TILES is None:
        TILES = db.SQL('select * from mosaic_tiles')
    
    if tile not in TILES['tile']:
        print(f'{tile} not in `mosaic_tiles`')
        return None
        
    row = TILES[TILES['tile'] == tile]
        
    # row = db.SQL(f"""SELECT crval1, crval2, npix
    #                    FROM mosaic_tiles
    #                   WHERE tile={tile}""")

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
                               ORDER BY count(filter) DESC""")
    
    return len(tiles), tiles


def reset_locked():
    """
    Reset in_mosaic 9 > 0 for tiles that may have timed out
    """
    cmd = """UPDATE mosaic_tiles_exposures
             SET in_mosaic = 0 WHERE in_mosaic = 9"""
             
    db.execute(cmd)


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


def reset_tiles_in_assoc(assoc):
    """
    Set in_mosaic status to all subtiles overlapping with an assoc
    """
    res = db.execute(f"""UPDATE mosaic_tiles_exposures
     SET in_mosaic = 0
     FROM (select tile, subx, suby
      from mosaic_tiles_exposures ti, exposure_files e
      where ti.expid = e.eid AND e.assoc = '{assoc}'
      group by tile, subx, suby) subt
     WHERE mosaic_tiles_exposures.tile = subt.tile 
          AND mosaic_tiles_exposures.subx = subt.subx
          AND mosaic_tiles_exposures.suby = subt.suby
    """)


def get_tiles_containing_point(point=(150.24727,2.04512), radius=0.01):
    """
    reset in_mosaic status for subtiles overlapping with a point
    """
    
    circle = f"circle '<({point[0]}, {point[1]}), {radius}>'"
    
    res = db.SQL(f"""SELECT tile, subx, suby, filter, count(filter) as nexp
      from mosaic_tiles_exposures ti, exposure_files e
      where ti.expid = e.eid
      AND polygon(e.footprint) && polygon({circle})    
      group by tile, subx, suby, filter
      """)

    return res


def reset_tiles_containing_point(point=(150.24727,2.04512), radius=0.01):
    """
    reset in_mosaic status for subtiles overlapping with a point
    """
    
    circle = f"circle '<({point[0]}, {point[1]}), {radius}>'"
    
    res = db.execute(f"""UPDATE mosaic_tiles_exposures
     SET in_mosaic = 0
     FROM (select tile, subx, suby
      from mosaic_tiles_exposures ti, exposure_files e
      where ti.expid = e.eid
      AND polygon(e.footprint) && polygon({circle})    
      group by tile, subx, suby) subt
      WHERE mosaic_tiles_exposures.tile = subt.tile 
           AND mosaic_tiles_exposures.subx = subt.subx
           AND mosaic_tiles_exposures.suby = subt.suby      
      """)

    return res


def delete_empty_exposures():
    """
    Delete exposures from tiles where exptime = 0
    """
    SQL = f"""SELECT tile, subx, suby, filter, in_mosaic
                FROM mosaic_tiles_exposures t, exposure_files e
                WHERE t.expid = e.eid
                AND e.exptime < 1.
                """
    res = db.SQL(SQL)
    
    if len(res) > 0:
        SQL = f"""DELETE
                    FROM mosaic_tiles_exposures t
                    USING exposure_files e
                    WHERE t.expid = e.eid
                    AND e.exptime < 1.
                    """
    
        db.execute(SQL)


def send_all_tiles():
    
    import time
    import os
    import numpy as np
    from grizli.aws.tile_mosaic import (drizzle_tile_subregion, reset_locked,
                      get_lambda_client, send_event_lambda, count_locked)
    
    from grizli.aws import db
    from grizli import utils
    
    tiles = []
    
    client = get_lambda_client()
    
    nt0 = len(tiles)
    
    progs = f'AND tile != 1183 AND tile != 1392'
    progs = f'AND tile != 1183'
    progs = ''
    
    tiles = db.SQL(f"""SELECT tile, subx, suby, filter, count(filter)
                FROM mosaic_tiles_exposures t, exposure_files e
                WHERE t.expid = e.eid AND in_mosaic = 0
                {progs}
                AND filter < 'G0'
                GROUP BY tile, subx, suby, filter
                ORDER BY filter ASC
                """)
    
    if 1:
    
        # Skip all optical
        skip = (tiles['filter'] > 'F18') #& (tiles['count'] > 100)
        skip |= (tiles['filter'] < 'F18') & (tiles['count'] > 100)
        timeout = 60
    
        skip = (tiles['filter'] > 'F18') & (tiles['count'] > 100)
        skip |= (tiles['filter'] < 'F18') & (tiles['count'] > 300)
        timeout = 60
        
        tiles = tiles[~skip]
    else:
        # Randomize order for running locally
        ix = np.argsort(np.random.rand(len(tiles)))
        tiles = tiles[ix]
        timeout = 60
        
    nt1 = len(tiles)
    print(nt1, nt0-nt1)
    
    NMAX = len(tiles)

    istart = i = -1
    
    max_locked = 800
    
    step = max_locked - count_locked()[0]
    
    while i < NMAX-1:
        i+=1 
        # if tiles['tile'][i] == 1183:
        #     continue
        
        if i-istart == step:
            istart = i
            print(f'\n ############### \n {time.ctime()}: Pause for {timeout} s  / {step} run previously')
            time.sleep(timeout)
            
            step = np.maximum(max_locked - count_locked()[0], 1)
            print(f'{time.ctime()}: Run {step} more \n ############## \n')
                    
        event = dict(tile=int(tiles['tile'][i]), 
                     subx=int(tiles['subx'][i]),
                     suby=int(tiles['suby'][i]),
                     filter=tiles['filter'][i], 
                     exposure_count=int(tiles['count'][i]),
                     counter=i+2, 
                     time=time.ctime())
        
        if 1:
            send_event_lambda(event, client=client, func='grizli-redshift-fit')
        
        else:
            drizzle_tile_subregion(**event, 
                                   s3output=None,
                                   ir_wcs=None, make_figure=False, 
                                   skip_existing=True, verbose=True, 
                                   gzip_output=False, clean_flt=False)
            
            files='tile.{tile:04d}.{subx:03d}.{suby:03d}.{fx}*fits'
            os.system('rm '+ files.format(fx=event['filter'].lower(), 
                                             **event))
        
        # if (i+1) % 300 == 0:
        #     break
            #time.sleep(10)

    ### Run locally
    tiles = db.SQL(f"""SELECT tile, subx, suby, filter, MAX(in_mosaic) as in_mosaic, count(filter)
                FROM mosaic_tiles_exposures t, exposure_files e
                WHERE t.expid = e.eid AND tile != 1183
                GROUP BY tile, subx, suby, filter
                ORDER BY (tile, filter)
                """)
    
    keep = (tiles['filter'] > 'F199') & (tiles['count'] > 100)
    utils.Unique(tiles['in_mosaic'][keep])
    
    tiles = db.SQL(f"""SELECT tile, subx, suby, filter, count(filter)
                FROM mosaic_tiles_exposures t, exposure_files e
                WHERE t.expid = e.eid AND in_mosaic = 0 AND tile != 1183
                GROUP BY tile, subx, suby, filter
                ORDER BY (tile, filter)
                """)

    skip = (tiles['filter'] > 'F1') & (tiles['count'] < 100)
    skip |= (tiles['filter'] < 'F1') & (tiles['count'] > 300)

    tiles = tiles[~skip]
    
    NMAX = len(tiles)

    istart = i = -1
    
    tile_filt = (0, 0)
    
    while i < NMAX:
        i+=1 
        if tiles['tile'][i] == 1183:
            continue
        
        event = dict(tile=int(tiles['tile'][i]), 
                     subx=int(tiles['subx'][i]),
                     suby=int(tiles['suby'][i]),
                     filter=tiles['filter'][i], 
                     counter=i, 
                     time=time.ctime())
        
        if (event['tile'], event['filter']) != tile_filt:
            print('Next filter, remove flc.fits')
            os.system('rm *flc.fits')
            tile_filt = (event['tile'], event['filter'])
            
        if 0:
            send_event_lambda(event, client=client, func='grizli-mosaic-tile')
        
        else:
            status = drizzle_tile_subregion(**event, 
                                   s3output=None,
                                   ir_wcs=None, make_figure=False, 
                                   clean_flt=False,
                                   skip_existing=True, verbose=True, 
                                   gzip_output=False)
            
            files='tile.{tile:04d}.{subx:03d}.{suby:03d}.{fx}*fits'
            os.system('rm '+ files.format(fx=event['filter'].lower(), 
                                             **event))
    
    
def get_tile_status(tile, subx, suby, filter):
    """
    Is a tile "locked" with all exposures set with in_mosaic = 9?
    """

    exp = db.SQL(f"""SELECT dataset, extension, assoc, filter, 
                            exptime, footprint, in_mosaic, detector
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


def drizzle_tile_subregion(tile=2530, subx=522, suby=461, filter='F160W', s3output=None, ir_wcs=None, make_figure=False, skip_existing=True, verbose=True, gzip_output=False, **kwargs):
    """
    Drizzle a subtile
    
    Parameters
    ----------
    tile, subx, suby : int
        Identifiers of subtile
    
    filter : str
        Filter bandpass
    
    s3output : str
        Output S3 path, defaults to 
        ``s3://grizli-mosaic-tiles/Tiles/{tile}/``
    
    ir_wcs : `~astropy.wcs.WCS`
        Override subtile WCS
    
    skip_existing : bool
        Skip if output already exists or `in_mosaic` is 0 or 9 in the 
        database
    
    gzip_output : bool
        Gzip the drizzle products
    
    kwargs : dict
        Arguments passed through to `grizli.aws.visit_processor.cutout_mosaic`
        
    Returns
    -------
    status : str
        - ``skip completed`` = `tile.subx.suby.filter` has `in_mosaic = 1` 
           in database
        - ``skip locked`` = `tile.subx.suby.filter` has `in_mosaic = 9` 
           in database
        - ``skip empty`` = no exposures found for `tile.subx.suby.filter`
        - ``skip local`` = `tile.subx.suby.filter` found in local directory
        - ``tile.{tile}.{subx}.{suby}.{filter}`` = rootname of created file
        
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
            
        return 'skip empty'
    
    elif status in ['locked']:
        print(f'{root} {filter} tile locked')
        return 'skip locked'
    
    elif (status in ['completed']) & (skip_existing):
        print(f'{root} {filter} tile already completed')
        return 'skip completed'
        
    sci_file = f'{root}.{filter.lower()}_drz_sci.fits'
    if skip_existing & os.path.exists(sci_file):
        print(f'Skip file {sci_file}')
        return 'skip local'
    
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
        s3output = f's3://grizli-mosaic-tiles/Tiles/{tile}/'
    
    try:
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
    except TypeError:
        db.execute(f"""UPDATE mosaic_tiles_exposures t
              SET in_mosaic = 8
              FROM exposure_files w
              WHERE t.expid = w.eid
              AND w.filter='{filter}' AND tile={tile}
              AND subx={subx} AND suby={suby}
               """)
        
    status = '{root}.{f}'
    return status


def query_cutout(output='mos-{tile}-{filter}_{drz}', ra=189.0243001, dec=62.1966953, size=10, filters=['F160W'], theta=0, make_figure=False, make_mosaics=False, all_tiles=True, **kwargs):
    """
    """
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from grizli import utils
    from grizli.aws import tile_mosaic, db
    
    cosd = np.cos(dec/180*np.pi)
    rc = size/3600*np.sqrt(2)
    rtile = np.sqrt(2)*128*0.1/3600
    
    SQL = f"""SELECT tile, subx, suby, subra, subdec, filter
            FROM mosaic_tiles_exposures t, exposure_files e
            WHERE in_mosaic = 1
            AND t.expid = e.eid 
            AND ('((' || (subra - {ra})*{cosd} || 
                    ', ' || subdec - {dec} || '),
                    {rtile})')::circle
                && ('((0,0),{rc})')::circle
            GROUP BY tile, subx, suby, subra, subdec, filter"""
    
    res = db.SQL(SQL) 
    if len(res) == 0:
        msg = f'Nothing found for ({ra:.6f}, {dec:.6f}, {size}")'
        return 'empty nothing found', None
        
    if filters is None:
        filters = np.unique(res['filter']).tolist()
    else:
        keep = utils.column_values_in_list(res['filter'], filters)
        res = res[keep]
    
    if len(res) == 0:
        msg = f'Nothing found for ({ra:.6f}, {dec:.6f}, {size}") {filters}'
        return 'empty filter', None
        
    if make_figure:
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        ax.scatter(res['subra'], res['subdec'], marker='x', alpha=0.)
    
    oh, ow = utils.make_wcsheader(ra=ra, dec=dec, size=size*2, 
                                  pixscale=0.1, theta=theta)
    
    sh = utils.SRegion(ow)
    if make_figure:
        ax.add_patch(sh.get_patch(alpha=0.2, color='r')[0])
    
    res['keep'] = False
                           
    iters = zip(res['tile'], res['subx'], res['suby'])
    
    keys = [res['tile'][i]*1e6+res['subx'][i]*1000+res['suby'][i]
            for i in range(len(res))]
    
    un = utils.Unique(keys, verbose=False)        
    for k in un.values:
        unk = un[k]
        rk = res[unk][0]
        tw = tile_subregion_wcs(rk['tile'], rk['subx'], rk['suby'])
        sr = utils.SRegion(tw)
        isect = sr.intersects(sh.union())
        res['keep'][unk] = isect
        
        if make_figure:
            if isect:
                ec = 'b'
                fc = 'b'
                alpha=0.2
            else:
                ec = '0.8'
                fc = 'None'
                alpha=0.3
                
            ax.add_patch(sr.get_patch(alpha=alpha, fc=fc, ec=ec, zorder=-10)[0])
    
    if res['keep'].sum() == 0:
        msg = f'Nothing found for ({ra:.6f}, {dec:.6f}, {size}") {filters}'
        return 'empty filter', None
    
    res = res[res['keep']]
    
    if make_mosaics:
        make_mosaic_from_table(res, output=output, all_tiles=all_tiles,
                               **kwargs)
        
    return 'ok', res


def make_mosaic_from_table(tab, output='mos-{tile}-{filter}_{drz}', clean_subtiles=False, send_to_s3=False, all_tiles=True, **kwargs):
    """
    """
    from grizli import utils
    un = utils.Unique(tab['tile'], verbose=False)
    if all_tiles:
        tiles = un.values
    else:
        tiles = [un.values[np.argmin(un.count)]]
        
    for t in tiles:
        
        unt = tab[un[t]]
        
        xmi = unt['subx'].min()
        ymi = unt['suby'].min()
        xma = unt['subx'].max()
        yma = unt['suby'].max()
        
        filts = [f.lower() for f in np.unique(unt['filter'])]
        
        ll = (xmi, ymi)
        ur = (xma, yma)
        
        for filt in filts:
            build_mosaic_from_subregions(root=output, 
                                         tile=t, files=None, filter=filt, 
                                         ll=ll, ur=ur,
                                         clean_subtiles=clean_subtiles, 
                                         send_to_s3=send_to_s3)
        
        if 0:
            from grizli.pipeline import auto_script
            ds9 = None
            auto_script.field_rgb(root=f'mos-{t}',  HOME_PATH=None, scl=1, ds9=ds9)


def build_mosaic_from_subregions(root='mos-{tile}-{filter}_{drz}', tile=2530, files=None, filter='f140w', ll=None, ur=None, clean_subtiles=False, send_to_s3=False, make_weight=True):
    """
    TBD
    """
    import os
    from tqdm import tqdm
    import glob
    import numpy as np
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    from grizli import utils
    
    if 0:
        for filt in ['f105w','f140w','f160w']:
            build_mosaic_from_subregions(tile=2530, files=None, filter=filt)
            
    #tile = 2530
    #filter = 'f140w'
    
    if files is None:
        files = glob.glob(f'tile.{tile:04d}.*_dr?_sci.fits')
        files.sort()
        
    if len(files) > 0:
        tx = np.array([int(f.split('.')[2]) for f in files])
        ty = np.array([int(f.split('.')[3]) for f in files])
    
        txm, tym = tx.min(), ty.min()
        
    if ll is None:
        ll = [txm, tym]
    
    if ur is None:
        ur = [tx.max(), ty.max()]
           
    nx = ur[0] - ll[0] + 1
    ny = ur[1] - ll[1] + 1
    
    llw = tile_subregion_wcs(tile, ll[0], ll[1])
    
    MIRI_FILTERS = ['f560w','f770w','f1000w','f1280W',
                    'f1500w','f1800w', 'f2100w','f2550w']
                                         
    if ('clear' in filter):
        npix = 512
        llw = utils.half_pixel_scale(llw)
        drz = 'drc'
    elif (filter in MIRI_FILTERS):
        npix = 256
        drz = 'drz'        
    elif (filter > 'f199') & (filter not in ['g102','g141']):
        npix = 512
        llw = utils.half_pixel_scale(llw)
        drz = 'drc'
    else:
        npix = 256
        drz = 'drz'
        
    img = np.zeros((ny*npix, nx*npix), dtype=np.float32)
    if make_weight:
        imgw = np.zeros_like(img)
        
    llh = utils.to_header(llw)
    llh['NAXIS1'] = nx*npix
    llh['NAXIS2'] = ny*npix
    llh['FILTER'] = filter.upper()
    
    ###### and check difference between opt / ir
    ###### and change drizzle params!
    
    exposures = []
    
    llh['EXPSTART'] = 1e10
    llh['EXPEND'] = 0
    
    im = None
    
    for xi in range(ll[0], ur[0]+1):
        for yi in range(ll[1], ur[1]+1):
            file = f'tile.{tile:04d}.{xi:03d}.{yi:03d}'
            file += f'.{filter}_{drz}_sci.fits'
            
            s3 = f's3://grizli-mosaic-tiles/Tiles/{tile}/'
            db.download_s3_file(s3+file, overwrite=False, verbose=False)
            
            if not os.path.exists(file):
                #print('Skip', file)
                continue
        
            print(file)
            im = pyfits.open(file)
                            
            if llh['EXPSTART'] > im[0].header['EXPSTART']:
                llh['EXPSTART'] = im[0].header['EXPSTART']
            
            if llh['EXPEND'] < im[0].header['EXPEND']:
                llh['EXPEND'] = im[0].header['EXPEND']
                
            for j in range(im[0].header['NDRIZIM']):
                exp = im[0].header[f'FLT{j+1:05d}']
                if exp not in exposures:
                    exposures.append(exp)
                    
            slx = slice((xi-ll[0])*npix, (xi-ll[0]+1)*npix)
            sly = slice((yi-ll[1])*npix, (yi-ll[1]+1)*npix)
            img[sly, slx] += im[0].data
            
            if make_weight:
                wfile = file.replace('_sci','_wht')
                db.download_s3_file(s3+wfile, overwrite=False, verbose=False)
                if os.path.exists(wfile):
                    imw = pyfits.open(wfile)
                    imgw[sly, slx] += imw[0].data                                    
            else:
                wfile = None
                
            if clean_subtiles:
                os.remove(file)
                if wfile is not None:
                    os.remove(wfile)
                    
    llh['NDRIZIM'] = len(exposures)
    for j, exp in enumerate(exposures):
        llh[f'FLT{j+1:05d}'] = exp
    
    if im is not None:    
        for k in im[0].header:
            if (k not in llh) & (k not in ['SIMPLE','BITPIX','DATE-OBS','TIME-OBS']):
                llh[k] = im[0].header[k]
                #print(k, im[0].header[k])    
    
    # Empty
    if 'PHOTFLAM' not in llh:
        llh['PHOTFLAM'] = 0.
        llh['PHOTFNU'] = 0.
        llh['PHOTPLAM'] = 1.
            
    outfile = root.format(tile=tile, filter=filter, drz=drz) + '_sci.fits'
    pyfits.writeto(outfile, data=img, 
                   header=llh, overwrite=True)
    
    if make_weight:
        wfile = root.format(tile=tile, filter=filter, drz=drz) + '_wht.fits'
        pyfits.writeto(wfile, data=imgw, header=llh, overwrite=True)
    else:
        wfile = None
        
    if send_to_s3:
        db.upload_file(outfile, 'grizli-v2', object_name='Scratch/'+outfile)
        if make_weight:
            db.upload_file(wfile, 'grizli-v2', object_name='Scratch/'+wfile)
    
    return outfile, wfile
            
    #os.system(f'aws s3 cp {root}.{tile:04d}-{filter}_drz_sci.fits s3://grizli-v2/Scratch/')
    
    
