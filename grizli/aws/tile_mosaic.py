"""
Drizzled mosaics in tiles and subregions

Here the sky is tesselated in 4 degree patches with sizes that are 
increased slightly to be integer multiples of 512 0.1" pixels

Individual subtiles are 256 x 0.1" = 25.6"

"""
from tqdm import tqdm
import numpy as np

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
    from grizli.aws import db
    
    
    engine = db.get_db_engine()
    
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
    df.to_sql('mosaic_tiles', engine, index=False, if_exists='fail', 
              method='multi')
    
    # Look at HST fields
    
    avg_coo = db.from_sql('select parent, count(parent), avg(ra) as ra, avg(dec) as dec from assoc_table where ra > 0 GROUP by parent order by count(parent)', engine)
    
    exp = db.from_sql('select assoc, crval1, crval2, footprint from exposure_files', engine)
    coo = SkyCoord(exp['crval1'], exp['crval2'], unit=('deg','deg'))
    
    # old
    avg_coo = db.from_sql("select parent, count(parent), avg(ra) as ra, avg(dec) as dec from exposure_log where ra > 0 and awspath not like '%%grizli-cosmos-v2%%' GROUP by parent order by count(parent)", engine)
    exp = db.from_sql("select parent, ra, dec from exposure_log where ra > 0 and awspath not like '%%grizli-cosmos-v2%%'", engine)
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
    from grizli.aws import db
    
    engine = db.get_db_engine()
    
    filters = db.from_sql('select filter, count(filter) from exposure_files group by filter order by count(filter)', engine)
    
    for filt in filters['filter'][-3:]:
        exp = db.from_sql(f"""
                    select eid, assoc, dataset, extension, filter, sciext, 
                           crval1 as ra, crval2 as dec, footprint
                    FROM exposure_files
                    WHERE filter = '{filt}'
                    """, engine)
    
        tiles = db.from_sql('select * from mosaic_tiles', engine)
    
        res = [add_exposure_to_tile_db(row=exp[i:i+1], tiles=tiles, 
               engine=engine)
               for i in tqdm(range(len(exp)))]
        
        for j in range(len(res))[::-1]:
            if res[j] is None:
                res.pop(j)
                
        tab = astropy.table.vstack(res)
        engine.execute(f"""
                DELETE from mosaic_tiles_exposures t
                USING exposure_files e
                WHERE t.expid = e.eid
                AND filter = '{filt}'
                """, engine)
             
        df = tab.to_pandas()
        df.to_sql('mosaic_tiles_exposures', engine, index=False, 
                  if_exists='append', method='multi')
    
    
    # Exposure map
    if 0:
        import ligo.skymap.plot
        from matplotlib import pyplot as plt
        from astropy.coordinates import SkyCoord
        
        filt = 'F105W'
        
        res = db.from_sql(f"""
                SELECT tile, subx, suby, subra, subdec, filter, 
                       COUNT(filter) as nexp, 
                       SUM(exptime) as exptime, MIN(expstart) as tmin, 
                       MAX(expstart) as tmax 
                FROM mosaic_tiles_exposures t, exposure_files e
                WHERE t.expid = e.eid
                AND assoc like 'j12%%' AND tile = 2530
                AND filter = '{filt}'
                GROUP BY tile, subx, suby, subra, subdec, filter
                """, engine)

        kw = {'projection':'astro aitoff'}
        kw = dict(projection='astro degrees zoom',
                      center='150.0d 2.0d', radius='1 deg')

        kw = dict(projection='astro degrees zoom',
                      center='53.2d -27.6d', radius='1 deg')
        
        kw = dict(projection='astro degrees zoom',
                      center='189.2d 62.25d', radius='20 arcmin')
        
        #kw = dict(projection='astro degrees zoom',
        #            center='0h 0d', radius='6 deg')

        plt.close('all')
        fig, ax = plt.subplots(1,1,figsize=(8,8), 
                               subplot_kw=kw)
        
        ax.grid()
        
        coo = SkyCoord(res['subra'], res['subdec'], unit=('deg','deg'))
        ax.scatter_coord(coo, c=np.log10(res['exptime']), marker='s')
        
        for t in np.unique(res['tile']):
            twcs = tile_wcs(t, engine=engine)
            coo = SkyCoord(*twcs.calc_footprint().T, unit=('deg','deg'))
            ax.plot_coord(coo, color='r', linewidth=1.2, alpha=0.5)
            
    if 0:
        engine.execute('ALTER TABLE exposure_files ADD COLUMN eid SERIAL PRIMARY KEY;')
        
        engine.execute('GRANT ALL PRIVILEGES ON ALL TABLEs IN SCHEMA public TO db_iam_user')
        engine.execute('GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO db_iam_user')
        
        engine.execute('ALTER TABLE assoc_table ADD COLUMN aid SERIAL PRIMARY KEY;')
        engine.execute('CREATE INDEX on exposure_files (eid)')
        engine.execute('CREATE INDEX on mosaic_tiles_exposures (expid)')
        
                    
def add_exposure_to_tile_db(dataset='ibev8xubq', sciext=1, tiles=None, row=None, engine=None):
    """
    Find subtiles that overlap with an exposure in the `exposure_files`
    table
    """
    import astropy.table
    import astropy.units as u
    import astropy.wcs as pywcs
    
    import numpy as np
    from grizli import utils
    
    from grizli.aws import db
    if engine is None:
        engine = db.get_db_engine()
        
    if tiles is None:
        tiles = db.from_sql('select * from mosaic_tiles', engine)

    if row is None:
        row = db.from_sql(f"select eid, assoc, dataset, extension, filter, sciext, crval1 as ra, crval2 as dec, footprint from exposure_files where filter = 'F160W' and dataset = '{dataset}' AND sciext={sciext}", engine)
    
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
        
        # engine.execute()
        # df = tmatch.to_pandas()
        # df.to_sql('mosaic_tiles_exposures', engine, index=False, 
        #           if_exists='append', method='multi')
        
        return tmatch
    else:
        return None


def tile_wcs(tile, engine=None):
    """
    Compute tile WCS
    """
    import astropy.wcs as pywcs
    from grizli import utils
    from grizli.aws import db
    
    if engine is None:
        engine = db.get_db_engine()
        
    row = db.from_sql(f"""SELECT crval1, crval2, npix
                          FROM mosaic_tiles
                          WHERE tile={tile}""", 
                      engine)

    t = 0
    h, w = utils.make_wcsheader(ra=row['crval1'][t], 
                                dec=row['crval2'][t],
                                size=row['npix'][t]*0.1, 
                                pixscale=0.1)
    
    h['CRPIX1'] += 0.5
    h['CRPIX2'] += 0.5
    h['LATPOLE'] = 0.
    
    wcs = pywcs.WCS(h)
    wcs.pscale = 0.1
    
    return wcs
    
    
def tile_subregion_wcs(tile, subx, suby, engine=None):
    """
    Compute WCS for a tile subregion
    """
    
    twcs = tile_wcs(tile, engine=engine)
    
    sub_wcs = twcs.slice((slice(suby*256, (suby+1)*256), 
                          slice(subx*256, (subx+1)*256)))
    sub_wcs.pscale = 0.1
    return sub_wcs


def drizzle_tile_subregion(tile, subx, suby, filter='F160W', engine=None, s3output=None, ir_wcs=None, make_figure=False, skip_existing=False, verbose=True, gzip_output=False, **kwargs):
    """
    """
    import astropy.table
    import astropy.units as u
    import astropy.wcs as pywcs
    import astropy.io.fits as pyfits
    
    import numpy as np
    from grizli import utils
    from grizli.aws import visit_processor
    
    from grizli.aws import db
    if engine is None:
        engine = db.get_db_engine()
        
    exp = db.from_sql(f"""
            SELECT dataset, extension, assoc, filter, exptime, footprint
            FROM mosaic_tiles_exposures t, exposure_files e
            WHERE t.expid = e.eid
            AND filter='{filter}' AND tile={tile}
            AND subx={subx} AND suby={suby}
            """, engine)
    
    root = f'tile.{tile:04d}.{subx:03d}.{suby:03d}'
    if len(exp) == 0:
        if verbose:
            print(f'{root} {filter} ! No exposures found')
            
        return True
        
    if ir_wcs is None:
        ir_wcs = tile_subregion_wcs(tile, subx, suby, engine=engine)
    
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
    engine.execute(f"""
          UPDATE mosaic_tiles_exposures t
          SET in_mosaic = 1
          FROM exposure_files w
          WHERE t.expid = w.eid
          AND w.filter='{filter}' AND tile={tile}
          AND subx={subx} AND suby={suby}
           """)


def build_mosaic_from_subregions():
    """
    TBD
    """
    from tqdm import tqdm
    
    tile = 2530
    filter = 'f160w'
    
    files = glob.glob(f'*{tile:04d}.*_drz_sci.fits')
    files.sort()
    tx = np.array([int(f.split('.')[2]) for f in files])
    ty = np.array([int(f.split('.')[3].split('-')[0]) for f in files])
    
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

        
    pyfits.writeto(f'test-{filter}.fits', data=img, header=h, overwrite=True)

    
