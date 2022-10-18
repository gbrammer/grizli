"""
Catalog table tools
"""
import os
import inspect

from collections import OrderedDict
import glob
import traceback

import numpy as np

import astropy.io.fits as pyfits
import astropy.units as u
import astropy.coordinates as coord
from astropy.table import Table

from . import utils

__all__ = ["table_to_radec", 
           "table_to_regions", 
           "randomize_segmentation_labels", 
           "get_ukidss_catalog",
           "get_sdss_catalog",
           "get_twomass_catalog",
           "get_irsa_catalog",
           "get_gaia_radec_at_time",
           "get_gaia_DR2_vizier_columns",
           "get_gaia_DR2_vizier",
           "gaia_dr2_conesearch_query",
           "get_gaia_DR2_catalog",
           "gen_tap_box_query",
           "query_tap_catalog",
           "get_hubble_source_catalog",
           "get_nsc_catalog",
           "get_desdr1_catalog",
           "get_skymapper_catalog",
           "get_vexas_catalog",
           "get_vhs_catalog",
           "get_panstarrs_catalog",
           "get_radec_catalog"]

def table_to_radec(table, output='coords.radec'):
    """Make a "radec" ascii file with ra, dec columns from a table object
    """
    
    if 'X_WORLD' in table.colnames:
        rc, dc = 'X_WORLD', 'Y_WORLD'
    elif 'x_world' in table.colnames:
        rc, dc = 'x_world', 'y_world'        
    elif 'ra' in table.colnames:
        rc, dc = 'ra', 'dec'
    else:
        raise ValueError("Couldn't identify sky coordinates (x_world, ra)")
    
    table[rc].format = '.8f'
    table[dc].format = '.8f'
    
    table[rc, dc].write(output, format='ascii.commented_header',
                        overwrite=True)


def table_to_regions(table, output='ds9.reg', comment=None, header='global color=green width=1', size=0.5, use_ellipse=False, use_world=True, verbose=True, unit_offset_colums=['x'], scale_major=1, extra=None):
    """Make a DS9 region file from a table object
    """
    fp = open(output, 'w')
    fp.write(header+'\n')
    
    
    if ('X_WORLD' in table.colnames) & (use_world):
        xc, yc = 'X_WORLD', 'Y_WORLD'
        is_world = True
    elif ('ra' in table.colnames) & (use_world):
        xc, yc = 'ra', 'dec'
        is_world = True
    elif ('x_world' in table.colnames) & (use_world):
        xc, yc = 'x_world', 'y_world'
        is_world = True
    elif 'x_image' in table.colnames:
        xc, yc = 'x_image','y_image'
        is_world = False
    elif 'x' in table.colnames:
        xc, yc = 'x','y'
        is_world = False
    else:
        raise ValueError("Couldn't identify ra/dec or x/y columns")
    
    xdata = table[xc]*1
    ydata = table[yc]*1
    
    if xc in unit_offset_colums:
        if verbose:
            print(f'Unit offset for columns {xc}, {yc}')
            
        xdata += 1
        ydata += 1
        
    if is_world:
        fp.write('icrs\n')
        sec='"'
    else:
        fp.write('image\n')
        sec=''
        
    # GAIA
    if 'solution_id' in table.colnames:
        e = np.sqrt(table['ra_error']**2+table['dec_error']**2)/1000.
        e = np.maximum(e, 0.1)
    else:
        e = np.ones(len(table))*size
    
    if use_ellipse:
        if is_world:
            if 'a_world' in table.colnames:
                amaj = table['a_world']*3600*scale_major
                amin = table['b_world']*3600*scale_major
                etheta = table['theta_world']/np.pi*180#+90
            else:
                use_ellipse = False
        else:
            if 'a_image' in table.colnames:
                amaj = table['a_image']*scale_major
                amin = table['b_image']*scale_major
                etheta = table['theta_image']/np.pi*180#+90
            else:
                use_ellipse = False    
    
    if verbose:
        print(f'{output}: x = {xc}, y={yc}, ellipse={use_ellipse}')
                            
    if use_ellipse:
        regstr = 'ellipse({x:.7f}, {y:.7f}, {a:.3f}{sec}, {b:.3f}{sec}, {theta:.1f})'
        lines = [regstr.format(x=xdata[i], y=ydata[i], 
                               a=amaj[i], b=amin[i], 
                               sec=sec, theta=etheta[i])
                 for i in range(len(table))] 
        
    else:
        regstr = 'circle({0:.7f}, {1:.7f}, {2:.3f}{3})'
        lines = [regstr.format(xdata[i], ydata[i], e[i], sec)
                 for i in range(len(table))]

    if comment is not None:
        for i in range(len(table)):
            lines[i] += ' # text={{{0}}}'.format(comment[i])
    
    if extra is not None:
        for i, li in enumerate(lines):
            if '#' not in li:
                lines[i] += ' #'
            
            lines[i] += ' '+extra[i]
    
    # newline            
    lines = [l+'\n' for l in lines]
    
    fp.writelines(lines)
    fp.close()


def randomize_segmentation_labels(seg, random_seed=1, fill_value=np.nan):
    """
    Randomize labels on a segmentation image for easier visualization
    
    Parameters
    ----------
    seg : str or array
        Segmentation filename or integer array
    
    random_seed : int
        Random number seed for `numpy.random.seed`
    
    fill_value : scalar
        Value to insert where ``seg == 0``
    
    Returns
    -------
    rand_ids : array
        1D array of the randomized labels with size `max(seg)+1` 
        
    rand_seg : array
        2D rray with randomized labels
        
    """
    
    np.random.seed(random_seed)
    
    if isinstance(seg, str):
        seg_im = pyfits.open(seg)
        seg_data = seg_im[0].data
    else:
        seg_data = seg
        
    imax = seg_data.max()
    
    rand_ids = np.append(fill_value, np.argsort(np.random.rand(imax))+1)
    return rand_ids, rand_ids[seg_data]
    
    
def get_ukidss_catalog(ra=165., dec=34.8, radius=3, database='UKIDSSDR9PLUS',
                       programme_id='LAS'):
    """Query for objects in the UKIDSS catalogs

    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees

    radius : float
        Radius of the query, in arcmin

    Returns
    -------
    table : `~astropy.table.Table`
        Result of the query

    """

    from astroquery.ukidss import Ukidss

    coo = coord.SkyCoord(ra*u.deg, dec*u.deg)

    table = Ukidss.query_region(coo, radius=radius*u.arcmin,
                                database=database, programme_id=programme_id)

    return table


def get_sdss_catalog(ra=165.86, dec=34.829694, radius=3):
    """Query for objects in the SDSS photometric catalog

    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees

    radius : float
        Radius of the query, in arcmin

    Returns
    -------
    table : `~astropy.table.Table`
        Result of the query

    """
    from astroquery.sdss import SDSS

    coo = coord.SkyCoord(ra*u.deg, dec*u.deg)

    fields = ['ra', 'dec', 'raErr', 'decErr', 'petroMag_r', 'petroMagErr_r']
    # print fields
    fields = None

    table = SDSS.query_region(coo, radius=radius*u.arcmin, spectro=False,
                              photoobj_fields=fields)

    return table


def get_twomass_catalog(ra=165.86, dec=34.829694, radius=3, catalog='allwise_p3as_psd'):
    return get_irsa_catalog(ra=ra, dec=dec, radius=radius, catalog='fp_psc', wise=False, twomass=True)


def get_irsa_catalog(ra=165.86, dec=34.829694, tab=None, radius=3, catalog='allwise_p3as_psd', wise=False, twomass=False, ROW_LIMIT=500000, TIMEOUT=3600):
    """Query for objects in the `AllWISE <http://wise2.ipac.caltech.edu/docs/release/allwise/>`_ source catalog

    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees

    radius : float
        Radius of the query, in arcmin

    Returns
    -------
    table : `~astropy.table.Table`
        Result of the query

    """
    from astroquery.irsa import Irsa
    Irsa.ROW_LIMIT = ROW_LIMIT
    Irsa.TIMEOUT = TIMEOUT

    #all_wise = 'wise_allwise_p3as_psd'
    #all_wise = 'allwise_p3as_psd'
    if wise:
        catalog = 'allwise_p3as_psd'
    elif twomass:
        catalog = 'fp_psc'

    coo = coord.SkyCoord(ra*u.deg, dec*u.deg)

    table = Irsa.query_region(coo, catalog=catalog, spatial="Cone",
                              radius=radius*u.arcmin, get_query_payload=False)

    return table


#
def get_gaia_radec_at_time(gaia_tbl, date=2015.5, format='decimalyear'):
    """
    Use `~astropy.coordinates.SkyCoord.apply_space_motion` to compute GAIA positions at a specific observation date

    Parameters
    ----------
    gaia_tbl : `~astropy.table.Table`
        GAIA table query, e.g., provided by `get_gaia_DR2_catalog`.

    date : e.g., float
        Observation date that can be parsed with `~astropy.time.Time`

    format : str
        Date format, see `~astropy.time.Time.FORMATS`.

    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        Projected sky coordinates.

    """
    from astropy.time import Time
    from astropy.coordinates import SkyCoord

    # Distance and radial_velocity are dummy numbers needed
    # to get the space motion correct

    try:
        # Try to use pyia
        import pyia
        g = pyia.GaiaData(gaia_tbl)
        coord = g.get_skycoord(distance=1*u.kpc, frame='icrs',
                               radial_velocity=0.*u.km/u.second)
    except:
        # From table itself
        if 'ref_epoch' in gaia_tbl.colnames:
            ref_epoch = Time(gaia_tbl['ref_epoch'].data,
                             format='decimalyear')
        else:
            ref_epoch = Time(2015.5, format='decimalyear')

        coord = SkyCoord(ra=gaia_tbl['ra'], dec=gaia_tbl['dec'],
                         pm_ra_cosdec=gaia_tbl['pmra'],
                         pm_dec=gaia_tbl['pmdec'],
                         obstime=ref_epoch,
                         frame='icrs',
                         distance=1*u.kpc, radial_velocity=0.*u.km/u.second)

    new_obstime = Time(date, format=format)
    coord_at_time = coord.apply_space_motion(new_obstime=new_obstime)
    return(coord_at_time)


#GAIA_DR2_COLUMNS = None
GAIA_DR2_COLUMNS = ['RA_ICRS','DE_ICRS','Epoch','e_RA_ICRS','e_DE_ICRS',
                    'pmRA','e_pmRA','pmDE','e_pmDE', 
                    'NAL','NAC','Solved','APF','WAL']

def get_gaia_DR2_vizier_columns():
    """
    Get translation of Vizier GAIA DR2 columns.  The subset of desired
    output columns can be specified in the GAIA_DR2_COLUMNS global variable, 
    which, if `None`, will be all available columns.
    """

    file = os.path.join(os.path.dirname(__file__), 'data/gaia_dr2_vizier_columns.txt')
    lines = open(file).readlines()[1:]
    
    gdict = OrderedDict()
    for line in lines:
        viz_col = line.split()[0]
        gaia_col = line.split()[1][1:-1]
        if GAIA_DR2_COLUMNS is not None:
            if viz_col in GAIA_DR2_COLUMNS:
                gdict[viz_col] = gaia_col
        else:
            gdict[viz_col] = gaia_col

    return gdict


def get_gaia_DR2_vizier(ra=165.86, dec=34.829694, radius=3., max=100000,
                    catalog="I/355/gaiadr3", server='vizier.u-strasbg.fr',
                    use_mirror=False, keys=None, mjd=None, clean_mjd=True):
    """
    Query GAIA catalog from Vizier
    
    DR2: ``catalog="I/345/gaia2"``
    EDR3: ``catalog="I/350/gaiaedr3"``
    DR3: ``catalog="I/355/gaiadr3"``
    
    """
    from astroquery.vizier import Vizier
    import astropy.units as u
    import astropy.coordinates as coord

    coo = coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg),
                         frame='icrs')

    gdict = get_gaia_DR2_vizier_columns()
    if keys is None:
        keys = list(gdict.keys())

    try:

        # Hack, Vizier object doesn't seem to allow getting all keys
        # simultaneously (astroquery v0.3.7)
        N = 9
        for i in range(len(keys)//N+1):
            v = Vizier(catalog=catalog, columns=['+_r']+keys[i*N:(i+1)*N])
            v.VIZIER_SERVER = server
            v.ROW_LIMIT = max
            tab = v.query_region(coo, radius="{0}m".format(radius), 
                                 catalog=catalog)[0]
            if i == 0:
                result = tab
            else:
                for k in tab.colnames:
                    #print(i, k)
                    result[k] = tab[k]

        for k in gdict:
            if k in result.colnames:
                result.rename_column(k, gdict[k])
    except:
        utils.log_exception(utils.LOGFILE, traceback)
        return False
    
    if mjd is not None:
        rd = get_gaia_radec_at_time(result, date=mjd, format='mjd')
        result['ra_time'] = rd.ra.deg
        result['dec_time'] = rd.dec.deg
        result.meta['cootime'] = mjd, 'Specified MJD for ra/dec_time'
        if clean_mjd:
            ok = np.isfinite(rd.ra.deg + rd.dec.deg)
            result = result[ok]
            
    return result


def gaia_dr2_conesearch_query(ra=165.86, dec=34.829694, radius=3., max=100000):
    """
    Generate a query string for the TAP servers
    TBD

    Parameters
    ----------
    ra, dec : float
        RA, Dec in decimal degrees

    radius : float
        Search radius, in arc-minutes.

    Returns
    -------
    query : str
        Query string

    """
    query = "SELECT TOP {3} * FROM gaiadr2.gaia_source  WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',{0},{1},{2:.2f}))=1".format(ra, dec, radius/60., max)
    return query


def get_gaia_DR2_catalog(ra=165.86, dec=34.829694, radius=3.,
                         use_mirror=True, max_wait=20,
                         max=100000, output_file='gaia.fits'):
    """Query GAIA DR2 astrometric catalog

    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees

    radius : float
        Radius of the query, in arcmin

    use_mirror : bool
        If True, use the mirror at `gaia.ari.uni-heidelberg.de`.  Otherwise
        use `gea.esac.esa.int`.

    Returns
    -------
    table : `~astropy.table.Table`
        Result of the query

    """
    try:
        import httplib
        from urllib import urlencode
    except:
        # python 3
        import http.client as httplib
        from urllib.parse import urlencode

    # import http.client in Python 3
    # import urllib.parse in Python 3
    import time
    from xml.dom.minidom import parseString

    host = "gea.esac.esa.int"
    port = 80
    pathinfo = "/tap-server/tap/async"

    if use_mirror:
        host = "gaia.ari.uni-heidelberg.de"
        pathinfo = "/tap/async"

    # -------------------------------------
    # Create job

    query = gaia_dr2_conesearch_query(ra=ra, dec=dec, radius=radius, max=max)  # "SELECT TOP 100000 * FROM gaiadr2.gaia_source  WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS',{0},{1},{2:.2f}))=1".format(ra, dec, radius/60.)
    print(query)

    params = urlencode({
        "REQUEST": "doQuery",
        "LANG":    "ADQL",
        "FORMAT":  "fits",
        "PHASE":  "RUN",
        "QUERY":  query
        })

    headers = {
        "Content-type": "application/x-www-form-urlencoded",
        "Accept":       "text/plain"
        }

    connection = httplib.HTTPConnection(host, port)
    connection.request("POST", pathinfo, params, headers)

    # Status
    response = connection.getresponse()
    print("Status: " + str(response.status), "Reason: " + str(response.reason))

    # Server job location (URL)
    location = response.getheader("location")
    print("Location: " + location)

    # Jobid
    jobid = location[location.rfind('/')+1:]
    print("Job id: " + jobid)

    connection.close()

    # -------------------------------------
    # Check job status, wait until finished

    tcount = 0
    while True:
        connection = httplib.HTTPConnection(host, port)
        connection.request("GET", pathinfo+"/"+jobid)
        response = connection.getresponse()
        data = response.read()
        # XML response: parse it to obtain the current status
        dom = parseString(data)

        if use_mirror:
            phaseElement = dom.getElementsByTagName('phase')[0]
        else:
            phaseElement = dom.getElementsByTagName('uws:phase')[0]

        phaseValueElement = phaseElement.firstChild
        phase = phaseValueElement.toxml()
        print("Status: " + phase)

        # Check finished
        if phase == 'COMPLETED':
            break

        #wait and repeat
        time.sleep(0.2)
        tcount += 0.2

        if (phase == 'ERROR') | (tcount > max_wait):
            return False

    # print "Data:"
    # print data

    connection.close()

    # -------------------------------------
    # Get results
    connection = httplib.HTTPConnection(host, port)
    connection.request("GET", pathinfo+"/"+jobid+"/results/result")
    response = connection.getresponse()
    data = response.read()
    outputFileName = output_file + (not use_mirror)*".gz"
    try:
        outputFile = open(outputFileName, "w")
        outputFile.write(data)
    except:
        # Python 3
        outputFile = open(outputFileName, "wb")
        outputFile.write(data)

    outputFile.close()
    connection.close()
    print("Data saved in: " + outputFileName)

    if not use_mirror:
        # ESA archive returns gzipped
        try:
            os.remove(output_file)
        except:
            pass

        os.system('gunzip {output_file}.gz'.format(output_file=output_file))

    table = Table.read(output_file, format='fits')
    return table


def gen_tap_box_query(ra=165.86, dec=34.829694, radius=3., corners=None, max=100000, db='ls_dr7.tractor_primary', columns=['*'], rd_colnames=['ra', 'dec'], wcs_pad=0.5):
    """
    Generate a query string for the NOAO Legacy Survey TAP server

    Parameters
    ----------
    ra, dec : float
        RA, Dec in decimal degrees

    radius : float
        Search radius, in arc-minutes.
    
    corners : 4-tuple, `~astropy.wcs.WCS` or None
        ra_min, ra_max, dec_min, dec_max of a query box to use instead of 
        `radius`.  Or if a `~astropy.wcs.WCS` object, get limits from the 
        `~astropy.wcs.WCS.calc_footprint` method
        
    Returns
    -------
    query : str
        Query string

    """

    rmi = radius/60/2
    cosd = np.cos(dec/180*np.pi)

    if max is not None:
        maxsel = 'TOP {0}'.format(max)
    else:
        maxsel = ''
    
    if corners is not None:
        if hasattr(corners, 'calc_footprint'):
            foot = corners.calc_footprint()
            
            left = foot[:,0].min()
            right = foot[:,0].max()
            bottom = foot[:,1].min()
            top = foot[:,1].max()
            
            dx = (right-left)
            dy = (top-bottom)
            left -= wcs_pad*dx
            right += wcs_pad*dx
            bottom -= wcs_pad*dy
            top += wcs_pad*dy
                        
        elif len(corners) != 4:
            msg = 'corners needs 4 values (ra_min, ra_max, dec_min, dec_max)'
            raise ValueError(msg)    
        else:
            left, right, bottom, top = corners
    else:
        left = ra - rmi / cosd
        right = ra + rmi / cosd
        bottom = dec - rmi
        top = dec + rmi
    
    fmt = dict(rc=rd_colnames[0], dc=rd_colnames[1],
               left=left, right=right,
               top=top, bottom=bottom,
               maxsel=maxsel,
               db=db,
               output_columns=', '.join(columns))
               
    if not np.isfinite(ra+dec):
        query = "SELECT {maxsel} {output_columns} FROM {db} "
    else:
        query = ("SELECT {maxsel} {output_columns} FROM {db} WHERE " +
                 "{rc} > {left} AND {rc} < {right} AND " +
                 "{dc} > {bottom} AND {dc} < {top} ")
        
    return query.format(**fmt)


def query_tap_catalog(ra=165.86, dec=34.829694, radius=3., corners=None, 
                    max_wait=20,
                    db='ls_dr9.tractor', columns=['*'], extra='',
                    rd_colnames=['ra', 'dec'],
                    tap_url='https://datalab.noirlab.edu/tap',
                    max=1000000, clean_xml=True, verbose=True,
                    des=False, gaia=False, nsc=False, vizier=False,
                    skymapper=False,
                    hubble_source_catalog=False, tap_kwargs={}, 
                    **kwargs):
    """Query NOAO Catalog holdings

    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees

    radius : float
        Radius of the query, in arcmin

    corners : 4-tuple, `~astropy.wcs.WCS` or None
        ra_min, ra_max, dec_min, dec_max of a query box to use instead of 
        `radius`.  Or if a `WCS` object, get limits from the 
        `~astropy.wcs.WCS.calc_footprint` method

    db : str
        Parent database (https://datalab.noirlab.edu/query.php).

    columns : list of str
        List of columns to output.  Default ['*'] returns all columns.

    extra : str
        String to add to the end of the positional box query, e.g.,
        'AND mag_auto_i > 16 AND mag_auto_i < 16.5'.

    rd_colnames : str, str
        Column names in `db` corresponding to ra/dec (degrees).

    tap_url : str
        TAP hostname

    des : bool
        Query `des_dr1.main` from NOAO.

    gaia : bool
        Query `gaiadr2.gaia_source` from http://gea.esac.esa.int.

    nsc : bool
        Query the NOAO Source Catalog (Nidever et al. 2018), `nsc_dr1.object`.

    vizier : bool
        Use the VizieR TAP server at  http://tapvizier.u-strasbg.fr/TAPVizieR/tap, see http://tapvizier.u-strasbg.fr/adql/about.html.

    hubble_source_catalog : bool
        Query the Hubble Source Catalog (v3).  If no 'NumImages' criteria is
        found in `extra`, then add an additional requirement:

            >>> extra += 'AND NumImages > 1'

    Returns
    -------
    table : `~astropy.table.Table`
        Result of the query

    """
    from astroquery.utils.tap.core import TapPlus

    # DES DR1
    if des:
        if verbose:
            print('Query DES DR1 from NOAO')

        db = 'des_dr1.main'
        tap_url = 'https://datalab.noirlab.edu/tap'

    # NOAO source catalog, seems to have some junk
    if nsc:
        if verbose:
            print('Query NOAO source catalog')

        db = 'nsc_dr1.object'
        tap_url = 'https://datalab.noirlab.edu/tap'
        extra += ' AND nsc_dr1.object.flags = 0'

    # GAIA DR2
    if gaia:
        if verbose:
            print('Query GAIA DR2 from ESA')

        db = 'gaiadr2.gaia_source'
        tap_url = 'http://gea.esac.esa.int/tap-server/tap'

    # VizieR TAP server
    if vizier:
        if verbose:
            print('Query {0} from VizieR TAP server'.format(db))

        tap_url = 'http://tapvizier.u-strasbg.fr/TAPVizieR/tap'
        rd_colnames = ['RAJ2000', 'DEJ2000']

    if skymapper:
        if verbose:
            print('Query {0} from VizieR TAP server'.format(db))

        tap_url = 'http://tapvizier.u-strasbg.fr/TAPVizieR/tap'
        rd_colnames = ['RAICRS', 'DEICRS']

    if hubble_source_catalog:
        if db is None:
            db = 'dbo.SumPropMagAutoCat'
        elif 'dbo' not in db:
            db = 'dbo.SumPropMagAutoCat'

        tap_url = 'http://vao.stsci.edu/HSCTAP/tapservice.aspx'
        rd_colnames = ['MatchRA', 'MatchDec']
        if 'NumImages' not in extra:
            extra += 'AND NumImages > 1'

    tap = TapPlus(url=tap_url, **tap_kwargs)

    query = gen_tap_box_query(ra=ra, dec=dec, radius=radius, max=max,
                               db=db, columns=columns,
                               rd_colnames=rd_colnames, 
                               corners=corners)

    job = tap.launch_job(query+extra, dump_to_file=True, verbose=verbose)
    try:
        table = job.get_results()
        if clean_xml:
            if hasattr(job, 'outputFile'):
                jobFile = job.outputFile
            else:
                jobFile = job.get_output_file()

            os.remove(jobFile)

        # Provide ra/dec columns
        for c, cc in zip(rd_colnames, ['ra', 'dec']):
            if (c in table.colnames) & (cc not in table.colnames):
                table[cc] = table[c]

        table.meta['TAPURL'] = tap_url, 'TAP URL'
        table.meta['TAPDB'] = db, 'TAP database name'
        table.meta['TAPQUERY'] = query+extra, 'TAP query'
        table.meta['RAQUERY'] = ra, 'Query central RA'
        table.meta['DECQUERY'] = dec, 'Query central Dec'
        table.meta['RQUERY'] = radius, 'Query radius, arcmin'

        if hubble_source_catalog:
            for col in table.colnames:
                if table[col].dtype == 'object':
                    print('Reformat column: {0}'.format(col))
                    strcol = list(table[col])
                    table[col] = strcol

    except:
        if hasattr(job, 'outputFile'):
            jobFile = job.outputFile
        else:
            jobFile = job.get_output_file()

        print('Query failed, check {0} for error messages'.format(jobFile))
        table = None

    return table


# Limit Hubble Source Catalog query to brighter sources in limited bands
HSCv3_FILTER_LIMITS = {'W3_F160W': 23.5,
     'W3_F140W': 23.5,
     'W3_F125W': 23.5,
     'W3_F110W': 23.5,
     'W3_F098M': 23.5,
     'W3_F105W': 23.5,
      'A_F814W': 23.5,
     'W3_F814W': 23.5,
      'A_F606W': 23.5,
     'W3_F606W': 23.5,
     'A_F850LP': 23.5,
    'W3_F850LP': 23.5,
      'A_F775W': 23.5,
     'W3_F775W': 23.5}

HSCv3_COLUMNS = ['MatchRA', 'MatchDec', 'CI', 'CI_Sigma', 
                 'KronRadius', 'KronRadius_Sigma', 'Extinction', 
                 'TargetName', 'NumImages', 'NumFilters', 'NumVisits', 
                 'DSigma']

def get_hubble_source_catalog(ra=0., dec=0., radius=3, corners=None, max=int(1e7), extra=' AND NumImages > 0', kron_max=0.45, dsigma_max=100, clip_singles=10*u.arcsec, verbose=True, columns=HSCv3_COLUMNS, filter_limits=HSCv3_FILTER_LIMITS):
    """
    Query NOAO Source Catalog, which is aligned to GAIA DR1.

    The default `extra` query returns well-detected sources in red bands.

    filter_limits : query on individual HSC filter magnitudes

    """
    import astropy.table
    
    msg = 'Query NOAO Source Catalog ({ra:.5f},{dec:.5f},{radius:.1f}\')'
    print(msg.format(ra=ra, dec=dec, radius=radius))

    if kron_max is not None:
        extra += ' AND KronRadius < {0}'.format(kron_max)
    if dsigma_max is not None:
        extra += ' AND DSigma < {0}'.format(dsigma_max)

    if filter_limits is not None:
        limit_list = ['{0} < {1}'.format(f, filter_limits[f]) 
                      for f in filter_limits]

        filter_selection = ' AND ({0})'.format(' OR '.join(limit_list))
        extra += filter_selection

        columns += [f for f in filter_limits]

        db = 'dbo.SumPropMagAutoCat p join dbo.SumMagAutoCat m on p.MatchID = m.MatchID'
    else:
        db = 'dbo.SumPropMagAutoCat'

    tab = query_tap_catalog(ra=ra, dec=dec, radius=radius, 
                            corners=corners, max=max, 
                            extra=extra, hubble_source_catalog=True, 
                            verbose=verbose, db=db, columns=columns)

    if clip_singles not in [None, False]:
        rr = tab['NumImages'] > 1
        if (rr.sum() > 0) & ((~rr).sum() > 0):
            r0, r1 = tab[rr], tab[~rr]
            idx, dr = utils.GTable(r0).match_to_catalog_sky(r1)
            new = dr > clip_singles
            xtab = astropy.table.vstack([r0, r1[new]])
            if verbose:
                msg = ('HSCv3: Remove {0} NumImages == 1 sources ' +
                       'with tolerance {1}')
                print(msg.format((~new).sum(), clip_singles))

    return tab


def get_nsc_catalog(ra=0., dec=0., radius=3, corners=None, max=100000, extra=' AND (rerr < 0.08 OR ierr < 0.08 OR zerr < 0.08) AND raerr < 0.2 AND decerr < 0.2', verbose=True):
    """
    Query NOAO Source Catalog, which is aligned to GAIA DR1.

    The default `extra` query returns well-detected sources in red bands.

    """
    msg = 'Query NOAO Source Catalog ({ra:.5f},{dec:.5f},{radius:.1f}\')'
    print(msg.format(ra=ra, dec=dec, radius=radius))

    tab = query_tap_catalog(ra=ra, dec=dec, radius=radius, corners=corners, 
                            extra=extra, nsc=True, verbose=verbose, max=max)
    return tab


def get_desdr1_catalog(ra=0., dec=0., radius=3, corners=None, max=100000, extra=' AND (e_rmag < 0.15 OR e_imag < 0.15)', verbose=True):
    """
    Query DES DR1 Catalog from Vizier

    The default `extra` query returns well-detected sources in one or more
    red bands.

    """
    msg = 'Query DES Source Catalog ({ra:.5f},{dec:.5f},{radius:.1f}\')'
    print(msg.format(ra=ra, dec=dec, radius=radius))

    tab = query_tap_catalog(ra=ra, dec=dec, radius=radius, corners=corners,
                            extra=extra, db='"II/357/des_dr1"', 
                            vizier=True, 
                            verbose=verbose, max=max)
    return tab


def get_desdr1_catalog_old(ra=0., dec=0., radius=3, corners=None, max=100000, extra=' AND (magerr_auto_r < 0.15 OR magerr_auto_i < 0.15)', verbose=True):
    """
    Query DES DR1 Catalog.

    The default `extra` query returns well-detected sources in one or more
    red bands.

    """
    msg = 'Query DES Source Catalog ({ra:.5f},{dec:.5f},{radius:.1f}\')'
    print(msg.format(ra=ra, dec=dec, radius=radius))

    tab = query_tap_catalog(ra=ra, dec=dec, radius=radius, corners=corners,
                            extra=extra, des=True, verbose=verbose, max=max)
    return tab


def get_vhs_catalog(ra=0., dec=0., radius=3., corners=None, max_records=500000, verbose=True, extra='', table='vhs_dr4'):
    """
    VEXAS DR2 from vizier

    https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=II/369

    table: 'vexasds', 'vexasps', 'vexassm'

    """
    msg = 'Query VHS DR4 {table} catalog ({ra},{dec},{radius})'
    print(msg.format(table=table, ra=ra, dec=dec, radius=radius))
    tab = query_tap_catalog(ra=ra, dec=dec, radius=radius*2,
                            corners=corners, extra=extra, 
                            vizier=True, 
                            db=f'"II/359/{table}"', 
                            verbose=verbose, max=max_records)
    return tab


def get_vexas_catalog(ra=0., dec=0., radius=3., corners=None, max_records=500000, verbose=True, extra='', table='vexasdes'):
    """
    VEXAS DR2 from vizier
    
    https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=II/369
    
    table: 'vexasds', 'vexasps', 'vexassm'
    
    """
    msg = 'Query VEXAS DR2 {table} catalog ({ra},{dec},{radius})'
    print(msg.format(table=table, ra=ra, dec=dec, radius=radius))
    tab = query_tap_catalog(ra=ra, dec=dec, radius=radius*2,
                            corners=corners, extra=extra, 
                            vizier=True, 
                            db=f'"II/369/{table}"', 
                            verbose=verbose, max=max_records)
    return tab


def get_skymapper_catalog(ra=0., dec=0., radius=3., corners=None, max_records=500000, verbose=True, extra=''):
    """
    Get Skymapper DR1 from Vizier
    """
    msg = 'Query Skymapper DR1 catalog ({ra},{dec},{radius})'
    print(msg.format(ra=ra, dec=dec, radius=radius))
    tab = query_tap_catalog(ra=ra, dec=dec, radius=radius*2,
                            corners=corners, extra=extra, 
                            skymapper=True, db='"II/358/smss"', 
                            verbose=verbose, max=max_records)
    return tab


def get_legacysurveys_catalog(ra=0., dec=0., radius=3., verbose=True, db='ls_dr9.tractor', sn_lim=('r',10), **kwargs):
    """
    Query LegacySurveys TAP catalog
    """
    if verbose:
        msg = 'Query LegacySurveys ({db}) catalog ({ra},{dec},{radius:.2f})'
        print(msg.format(ra=ra, dec=dec, radius=radius, db=db))
    
    if (sn_lim is not None):
        if len(sn_lim) >= 2:
            b = sn_lim[0]
            extra = f' AND flux_{b}*SQRT(flux_ivar_{b}) > {sn_lim[1]}'
            if 'extra' in kwargs:
                kwargs['extra'] += extra
            else:
                kwargs['extra'] = extra
                
    tab = query_tap_catalog(ra=ra, dec=dec, radius=radius, db=db, **kwargs)
    # if (sn_lim is not None) & (len(tab) > 1):
    #     if len(sn_lim) >= 2:
    #         sn = tab[f'flux_{sn_lim[0]}']
    #         sn *= np.sqrt(tab[f'flux_ivar_{sn_lim[0]}'])
    #         
    #         valid = sn > sn_lim[1]
    #         return tab[valid]
            
    return tab


def get_panstarrs_catalog(ra=0., dec=0., radius=3., corners=None, max_records=500000, verbose=True, extra='AND "II/349/ps1".e_imag < 0.2 AND "II/349/ps1".e_RAJ2000 < 0.15 AND "II/349/ps1".e_DEJ2000 < 0.15'):
    """
    Get PS1 from Vizier
    """
    msg = 'Query PanSTARRS catalog ({ra},{dec},{radius})'
    print(msg.format(ra=ra, dec=dec, radius=radius))
    tab = query_tap_catalog(ra=ra, dec=dec, radius=radius*2,
                            corners=corners, extra=extra, 
                            vizier=True, db='"II/349/ps1"', verbose=verbose, 
                            max=max_records)
    return tab


def get_radec_catalog(ra=0., dec=0., radius=3., product='cat', verbose=True, reference_catalogs=['GAIA', 'LS_DR9', 'PS1', 'Hubble', 'NSC', 'SDSS', 'WISE', 'DES'], use_self_catalog=False, **kwargs):
    """Decide what reference astrometric catalog to use

    First search SDSS, then WISE looking for nearby matches.

    Parameters
    ----------
    ra, dec : float
        Center of the query region, decimal degrees

    radius : float
        Radius of the query, in arcmin

    product : str
        Basename of the drizzled product. If a locally-created catalog with
        filename that startswith `product` is found, use that one instead of
        the external (low precision) catalogs so that you're matching
        HST-to-HST astrometry.

    reference_catalogs : list
        Order in which to query reference catalogs.  Options are 'GAIA',
        'PS1' (STScI PanSTARRS), 'SDSS', 'WISE', 'NSC' (NOAO Source Catalog),
        'DES' (Dark Energy Survey DR1), 'Hubble' (Hubble Source Catalog v3), 
        'LS_DR9' (LegacySurveys DR9).

    Returns
    -------
    radec : str
        Filename of the RA/Dec list derived from the parent catalog

    ref_catalog : str, {'SDSS', 'WISE', 'VISIT'}
        Provenance of the `radec` list.

    """
    query_functions = {'SDSS': get_sdss_catalog,
                       'GAIA_TAP': get_gaia_DR2_catalog,
                       'PS1': get_panstarrs_catalog,
                       'WISE': get_irsa_catalog,
                       '2MASS': get_twomass_catalog,
                       'GAIA_Vizier': get_gaia_DR2_vizier,
                       'GAIA': get_gaia_DR2_vizier,
                       'NSC': get_nsc_catalog,
                       'DES': get_desdr1_catalog,
                       'Hubble': get_hubble_source_catalog,
                       'Skymapper': get_skymapper_catalog, 
                       'VEXAS': get_vexas_catalog, 
                       'LS_DR9': get_legacysurveys_catalog}

    # Try queries
    has_catalog = False
    ref_catalog = 'None'
    ref_cat = []

    for ref_src in reference_catalogs:
        try:
            if ref_src == 'GAIA':
                try:
                    ref_cat = query_functions[ref_src](ra=ra, dec=dec,
                                             radius=radius, use_mirror=False)
                except:
                    try:
                        ref_cat = query_functions[ref_src](ra=ra, dec=dec,
                                                 radius=radius)
                    except:
                        ref_cat = False

                # Try GAIA mirror at Heidelberg
                if ref_cat is False:
                    ref_cat = query_functions[ref_src](ra=ra, dec=dec,
                                              radius=radius, use_mirror=True)
            else:
                ref_cat = query_functions[ref_src](ra=ra, dec=dec,
                                                   radius=radius)
            # #
            # ref_cat = query_functions[ref_src](ra=ra, dec=dec,
            #                                    radius=radius)

            valid = np.isfinite(ref_cat['ra']+ref_cat['dec'])
            ref_cat = ref_cat[valid]
            
            if len(ref_cat) < 2:
                raise ValueError

            table_to_regions(ref_cat, output='{0}_{1}.reg'.format(product,
                                                         ref_src.lower()))
            ref_cat['ra', 'dec'].write('{0}_{1}.radec'.format(product,
                                                         ref_src.lower()),
                                    format='ascii.commented_header',
                                    overwrite=True)

            radec = '{0}_{1}.radec'.format(product, ref_src.lower())
            ref_catalog = ref_src
            has_catalog = True
            if len(ref_cat) > 0:
                break
        except:
            print('{0} query failed'.format(ref_src))
            has_catalog = False

    if (ref_src.startswith('GAIA')) & ('date' in kwargs) & has_catalog:
        if kwargs['date'] is not None:
            if 'date_format' in kwargs:
                date_format = kwargs['date_format']
            else:
                date_format = 'mjd'

            gaia_tbl = ref_cat  # utils.GTable.gread('gaia.fits')
            coo = get_gaia_radec_at_time(gaia_tbl, date=kwargs['date'],
                                         format=date_format)

            coo_tbl = utils.GTable()
            coo_tbl['ra'] = coo.ra
            coo_tbl['dec'] = coo.dec

            ok = np.isfinite(coo_tbl['ra']) & np.isfinite(coo_tbl['dec'])

            coo_tbl.meta['date'] = kwargs['date']
            coo_tbl.meta['datefmt'] = date_format

            msg = 'Apply observation ({0},{1}) to GAIA catalog'
            print(msg.format(kwargs['date'], date_format))

            table_to_regions(coo_tbl[ok], output='{0}_{1}.reg'.format(product,
                                                         ref_src.lower()))

            coo_tbl['ra', 'dec'][ok].write('{0}_{1}.radec'.format(product,
                                                         ref_src.lower()),
                                    format='ascii.commented_header',
                                    overwrite=True)

    if not has_catalog:
        return False

    # WISP, check if a catalog already exists for a given rootname and use
    # that if so.
    cat_files = glob.glob('-f1'.join(product.split('-f1')[:-1]) + '-f*.cat*')
    if (len(cat_files) > 0) & (use_self_catalog):
        ref_cat = utils.GTable.gread(cat_files[0])
        root = cat_files[0].split('.cat')[0]
        ref_cat['X_WORLD', 'Y_WORLD'].write('{0}.radec'.format(root),
                                format='ascii.commented_header',
                                overwrite=True)

        radec = '{0}.radec'.format(root)
        ref_catalog = 'VISIT'

    if verbose:
        msg = '{0} - Reference RADEC: {1} [{2}] N={3}'
        print(msg.format(product, radec, ref_catalog, len(ref_cat)))

    return radec, ref_catalog  


