"""
Dumping ground for general utilities
"""
import os
import glob
import inspect
from collections import OrderedDict
import warnings
import itertools
import logging

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.table

import numpy as np

import astropy.units as u

from sregion import SRegion, patch_from_polygon

from . import GRIZLI_PATH

KMS = u.km/u.s
FLAMBDA_CGS = u.erg/u.s/u.cm**2/u.angstrom
FNU_CGS = u.erg/u.s/u.cm**2/u.Hz

# character to skip clearing line on STDOUT printing
NO_NEWLINE = '\x1b[1A\x1b[1M'

# R_V for Galactic extinction
MW_RV = 3.1

MPL_COLORS = {'b': '#1f77b4', 'orange': '#ff7f0e', 'g': '#2ca02c', 'r': '#d62728', 'purple': '#9467bd', 'brown': '#8c564b', 'pink': '#e377c2', 'gray': '#7f7f7f', 'olive': '#bcbd22', 'cyan': '#17becf'}

# sns.color_palette("husl", 8)
SNS_HUSL = {'r': (0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
 'orange': (0.8087954113106306, 0.5634700050056693, 0.19502642696727285),
 'olive': (0.5920891529639701, 0.6418467016378244, 0.1935069134991043),
 'g': (0.19783576093349015, 0.6955516966063037, 0.3995301037444499),
 'sea': (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
 'b': (0.22335772267769388, 0.6565792317435265, 0.8171355503265633),
 'purple': (0.6423044349219739, 0.5497680051256467, 0.9582651433656727),
 'pink': (0.9603888539940703, 0.3814317878772117, 0.8683117650835491)}

GRISM_COLORS = {'G800L': (0.0, 0.4470588235294118, 0.6980392156862745),
      'G102': (0.0, 0.6196078431372549, 0.45098039215686275),
      'G141': (0.8352941176470589, 0.3686274509803922, 0.0),
      'none': (0.8, 0.4745098039215686, 0.6549019607843137),
      'G150': 'k',
      'F277W': (0.0, 0.6196078431372549, 0.45098039215686275),
      'F356W': (0.8352941176470589, 0.3686274509803922, 0.0),
      'F444W': (0.8, 0.4745098039215686, 0.6549019607843137),
      'F410M': (0.0, 0.4470588235294118, 0.6980392156862745),
      'G280': 'purple',
      'F090W': (0.0, 0.4470588235294118, 0.6980392156862745),
      'F115W': (0.0, 0.6196078431372549, 0.45098039215686275),
      'F150W': (0.8352941176470589, 0.3686274509803922, 0.0),
      'F140M': (0.8352941176470589, 0.3686274509803922, 0.0),
      'F158M': (0.8352941176470589, 0.3686274509803922, 0.0),
      'F200W': (0.8, 0.4745098039215686, 0.6549019607843137),
      'F140M': 'orange',
      'BLUE': '#1f77b4',  # Euclid
      'RED': '#d62728',
      'CLEARP': 'b'}

GRISM_MAJOR = {'G102': 0.1, 'G141': 0.1, # WFC3/IR
               'G800L': 0.1,  # ACS/WFC
               'F090W': 0.1, 'F115W': 0.1, 'F150W': 0.1, # NIRISS
               'F140M': 0.1, 'F158M': 0.1, 'F200W': 0.1, 
               'F277W': 0.2, 'F356W': 0.2, 'F444W': 0.2, # NIRCam
               'F410M': 0.2, 
               'BLUE': 0.1, 'RED': 0.1, # Euclid
               'GRISM':0.1, 'G150':0.1  # Roman
               }

GRISM_LIMITS = {'G800L': [0.545, 1.02, 40.],  # ACS/WFC
          'G280': [0.2, 0.4, 14],  # WFC3/UVIS
           'G102': [0.77, 1.18, 23.],  # WFC3/IR
           'G141': [1.06, 1.73, 46.0],
           'GRISM': [0.98, 1.98, 11.],  # WFIRST/Roman
           'G150': [0.98, 1.98, 11.],  
           'F090W': [0.76, 1.04, 45.0],  # NIRISS
           'F115W': [0.97, 1.32, 45.0],
           'F140M': [1.28, 1.52, 45.0],
           'F158M': [1.28, 1.72, 45.0],
           'F150W': [1.28, 1.72, 45.0],
           'F200W': [1.68, 2.30, 45.0],
           'F140M': [1.20, 1.60, 45.0],
           'CLEARP': [0.76, 2.3, 45.0],
           'F277W': [2.5, 3.2, 20.],  # NIRCAM
           'F356W': [3.05, 4.1, 20.],
           'F444W': [3.82, 5.08, 20],
           'F410M': [3.8, 4.38, 20],
           'BLUE': [0.8, 1.2, 10.],  # Euclid
           'RED': [1.1, 1.9, 14.]}

#DEFAULT_LINE_LIST = ['PaB', 'HeI-1083', 'SIII', 'OII-7325', 'ArIII-7138', 'SII', 'Ha+NII', 'OI-6302', 'HeI-5877', 'OIII', 'Hb', 'OIII-4363', 'Hg', 'Hd', 'H8','H9','NeIII-3867', 'OII', 'NeVI-3426', 'NeV-3346', 'MgII','CIV-1549', 'CIII-1908', 'OIII-1663', 'HeII-1640', 'NIII-1750', 'NIV-1487', 'NV-1240', 'Lya']

# Line species for determining individual line fluxes.  See `load_templates`.
DEFAULT_LINE_LIST = ['BrA','BrB','BrG','PfG','PfD',
                     'PaA','PaB','PaG','PaD',
                     'HeI-1083', 'SIII', 'OII-7325', 'ArIII-7138',
                     'SII', 'Ha', 'OI-6302', 'HeI-5877', 'OIII', 'Hb', 
                     'OIII-4363', 'Hg', 'Hd', 'H7', 'H8', 'H9', 'H10', 
                     'NeIII-3867', 'OII', 'NeVI-3426', 'NeV-3346', 'MgII', 
                     'CIV-1549', 'CIII-1906', 'CIII-1908', 'OIII-1663', 
                     'HeII-1640', 'NIII-1750', 'NIV-1487', 'NV-1240', 'Lya']

LSTSQ_RCOND = None

def set_warnings(numpy_level='ignore', astropy_level='ignore'):
    """
    Set global numpy and astropy warnings

    Parameters
    ----------
    numpy_level : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        Numpy error level (see `~numpy.seterr`).

    astropy_level : {'error', 'ignore', 'always', 'default', 'module', 'once'}
        Astropy error level (see `~warnings.simplefilter`).

    """
    from astropy.utils.exceptions import AstropyWarning

    np.seterr(all=numpy_level)
    warnings.simplefilter(astropy_level, category=AstropyWarning)


JWST_TRANSLATE = {'RA_TARG':'TARG_RA', 
                  'DEC_TARG':'TARG_DEC',
                  'EXPTIME':'EFFEXPTM',
                  'PA_V3':'ROLL_REF'}

def get_flt_info(files=[], columns=['FILE', 'FILTER', 'PUPIL', 'INSTRUME', 'DETECTOR', 'TARGNAME', 'DATE-OBS', 'TIME-OBS', 'EXPSTART', 'EXPTIME', 'PA_V3', 'RA_TARG', 'DEC_TARG', 'POSTARG1', 'POSTARG2'], translate=JWST_TRANSLATE, defaults={'PUPIL':'---', 'TARGNAME':'indef','PA_V3':0.0}, jwst_detector=True):
    """Extract header information from a list of FLT files

    Parameters
    ----------
    files : list
        List of exposure filenames.

    Returns
    -------
    tab : `~astropy.table.Table`
        Table containing header keywords

    """
    import astropy.io.fits as pyfits
    from astropy.table import Table

    if not files:
        files = glob.glob('*flt.fits')

    N = len(files)

    data = []
    
    for c in columns[2:]:
        if c not in translate:
            translate[c] = 'xxxxxxxxxxxxxx'
            
    for i in range(N):
        line = [os.path.basename(files[i]).split('.gz')[0]]
        if files[i].endswith('.gz'):
            im = pyfits.open(files[i])
            h = im[0].header.copy()
            im.close()
        else:
            h = pyfits.Header().fromfile(files[i])
        
        if os.path.basename(files[i]).startswith('jw0'):
            with pyfits.open(files[i]) as _im:
                h1 = _im['SCI'].header
                if 'PA_V3' in h1:
                    h['PA_V3'] = h1['PA_V3']
            
        filt = parse_filter_from_header(h, jwst_detector=jwst_detector)
        line.append(filt)
        has_columns = ['FILE', 'FILTER']

        for key in columns[2:]:
            has_columns.append(key)
            if key in h:
                line.append(h[key])
            elif translate[key] in h:
                line.append(h[translate[key]])
            else:
                if key in defaults:
                    line.append(defaults[key])
                else:
                    line.append(np.nan)
                    
                continue

        data.append(line)

    tab = Table(rows=data, names=has_columns)
    
    if 'TARGNAME' in tab.colnames:
        miss = tab['TARGNAME'] == ''
        targs = [t.replace(' ', '-') for t in tab['TARGNAME']]
        
        if miss.sum() > 0:
            for i in np.where(miss)[0]:
                targs[i] = 'indef'
        
        tab['TARGNAME'] = targs
            
    return tab


def radec_to_targname(ra=0, dec=0, round_arcsec=(4, 60), precision=2, targstr='j{rah}{ram}{ras}{sign}{ded}{dem}', header=None):
    """Turn decimal degree coordinates into a string with rounding.

    Parameters
    ----------
    ra, dec : float
        Sky coordinates in decimal degrees

    round_arcsec : (scalar, scalar)
        Round the coordinates to nearest value of `round`, in arcseconds.

    precision : int
        Sub-arcsecond precision, in `~astropy.coordinates.SkyCoord.to_string`.

    targstr : string
        Build `targname` with this parent string.  Arguments
        `rah, ram, ras, rass, sign, ded, dem, des, dess` are computed from the
        (rounded) target coordinates (`ra`, `dec`) and passed to
        `targstr.format`.

    header : `~astropy.io.fits.Header`, None
        Try to get `ra`, `dec` from header keywords, first `CRVAL` and then
        `RA_TARG`, `DEC_TARG`.

    Returns
    -------
    targname : str
        Target string, see the example above.
    
    Examples
    --------
    
    >>> # Test dec: -10d10m10.10s
    >>> dec = -10. - 10./60. - 10.1/3600
    >>> # Test ra: 02h02m02.20s
    >>> cosd = np.cos(dec/180*np.pi)
    >>> ra = 2*15 + 2./60*15 + 2.2/3600.*15
    >>> # Round to nearest arcmin
    >>> from grizli.utils import radec_to_targname
    >>> print(radec_to_targname(ra=ra, dec=dec, round_arcsec=(4,60),
                           targstr='j{rah}{ram}{ras}{sign}{ded}{dem}'))
    j020204m1010 # (rounded to 4 arcsec in RA)
    >>> # Full precision
    >>> targstr = 'j{rah}{ram}{ras}.{rass}{sign}{ded}{dem}{des}.{dess}'
    >>> print(radec_to_targname(ra, dec,round_arcsec=(0.0001, 0.0001),
                                precision=3, targstr=targstr))
    j020202.200m101010.100
    
    """
    import astropy.coordinates
    import astropy.units as u

    import re
    import numpy as np

    if header is not None:
        if 'CRVAL1' in header:
            ra, dec = header['CRVAL1'], header['CRVAL2']
        else:
            if 'RA_TARG' in header:
                ra, dec = header['RA_TARG'], header['DEC_TARG']

    cosd = np.cos(dec/180*np.pi)
    scl = np.array(round_arcsec)/3600*np.array([360/24, 1])

    dec_scl = int(np.round(dec/scl[1]))*scl[1]
    ra_scl = int(np.round(ra/scl[0]))*scl[0]

    coo = astropy.coordinates.SkyCoord(ra=ra_scl*u.deg, dec=dec_scl*u.deg, 
                                       frame='icrs')

    cstr = re.split('[hmsd.]', coo.to_string('hmsdms', precision=precision))
    # targname = ('j{0}{1}'.format(''.join(cstr[0:3]), ''.join(cstr[4:7])))
    # targname = targname.replace(' ', '').replace('+','p').replace('-','m')

    rah, ram, ras, rass = cstr[0:4]
    ded, dem, des, dess = cstr[4:8]
    sign = 'p' if ded[1] == '+' else 'm'

    targname = targstr.format(rah=rah, ram=ram, ras=ras, rass=rass,
                              ded=ded[2:], dem=dem, des=des, dess=dess,
                              sign=sign)

    return targname


def blot_nearest_exact(in_data, in_wcs, out_wcs, verbose=True, stepsize=-1,
                       scale_by_pixel_area=False, wcs_mask=True,
                       fill_value=0):
    """
    Own blot function for blotting exact pixels without rescaling for input
    and output pixel size

    test

    Parameters
    ----------
    in_data : `~numpy.ndarray`
        Input data to blot.

    in_wcs : `~astropy.wcs.WCS`
        Input WCS.  Must have _naxis1, _naxis2 or pixel_shape attributes.

    out_wcs : `~astropy.wcs.WCS`
        Output WCS.  Must have _naxis1, _naxis2 or pixel_shape attributes.

    scale_by_pixel_area : bool
        If True, then scale the output image by the square of the image pixel
        scales (out**2/in**2), i.e., the pixel areas.

    wcs_mask : bool
        Use fast WCS masking.  If False, use ``regions``.

    fill_value : int/float
        Value in `out_data` not covered by `in_data`.

    Returns
    -------
    out_data : `~numpy.ndarray`
        Blotted data.

    """
    from regions import Regions
    from shapely.geometry import Polygon
    import scipy.ndimage as nd
    from drizzlepac import cdriz

    try:
        from .utils_c.interp import pixel_map_c
    except:
        from grizli.utils_c.interp import pixel_map_c
    
    # Shapes, in numpy array convention (y, x)
    if hasattr(in_wcs, 'pixel_shape'):
        in_sh = in_wcs.pixel_shape[::-1]
    elif hasattr(in_wcs, 'array_shape'):
        in_sh = in_wcs.array_shape
    else:
        in_sh = (in_wcs._naxis2, in_wcs._naxis1)

    if hasattr(out_wcs, 'pixel_shape'):
        out_sh = out_wcs.pixel_shape[::-1]
    elif hasattr(out_wcs, 'array_shape'):
        out_sh = out_wcs.array_shape
    else:
        out_sh = (out_wcs._naxis2, out_wcs._naxis1)

    in_px = in_wcs.calc_footprint()
    in_poly = Polygon(in_px).buffer(5./3600.)

    out_px = out_wcs.calc_footprint()
    out_poly = Polygon(out_px).buffer(5./3600)

    olap = in_poly.intersection(out_poly)
    if olap.area == 0:
        if verbose:
            print('No overlap')
        return np.zeros(out_sh)

    # Region mask for speedup
    if np.isclose(olap.area, out_poly.area, 0.01):
        mask = np.ones(out_sh, dtype=bool)
    elif wcs_mask:
        # Use wcs / Path
        from matplotlib.path import Path
        out_xy = out_wcs.all_world2pix(np.array(in_poly.exterior.xy).T, 0)-0.5
        out_xy_path = Path(out_xy)
        yp, xp = np.indices(out_sh)
        pts = np.array([xp.flatten(), yp.flatten()]).T
        mask = out_xy_path.contains_points(pts).reshape(out_sh)
    else:
        olap_poly = np.array(olap.exterior.xy)
        poly_reg = "fk5\npolygon("+','.join(['{0}'.format(p + 1) for p in olap_poly.T.flatten()])+')\n'
        reg = Regions.parse(poly_reg, format='ds9')[0]
        mask = reg.to_mask().to_image(shape=out_sh)

    #yp, xp = np.indices(in_data.shape)
    #xi, yi = xp[mask], yp[mask]
    yo, xo = np.where(mask > 0)

    if stepsize <= 1:
        rd = out_wcs.all_pix2world(xo, yo, 0)
        xf, yf = in_wcs.all_world2pix(rd[0], rd[1], 0)
    else:
        # Seems backwards and doesn't quite agree with above
        blot_wcs = out_wcs
        source_wcs = in_wcs

        if hasattr(blot_wcs, 'pixel_shape'):
            nx, ny = blot_wcs.pixel_shape
        else:
            nx, ny = int(blot_wcs._naxis1), int(blot_wcs._naxis2)

        mapping = cdriz.DefaultWCSMapping(blot_wcs, source_wcs, nx, ny,
                                          stepsize)
        xf, yf = mapping(xo, yo)

    xi, yi = np.cast[int](np.round(xf)), np.cast[int](np.round(yf))

    m2 = (xi >= 0) & (yi >= 0) & (xi < in_sh[1]) & (yi < in_sh[0])
    xi, yi, xf, yf, xo, yo = xi[m2], yi[m2], xf[m2], yf[m2], xo[m2], yo[m2]

    out_data = np.ones(out_sh, dtype=np.float64)*fill_value
    status = pixel_map_c(np.cast[np.float64](in_data), xi, yi, out_data, xo, yo)

    # Fill empty
    func = nd.maximum_filter
    fill = out_data == 0
    filtered = func(out_data, size=5)
    out_data[fill] = filtered[fill]

    if scale_by_pixel_area:
        in_scale = get_wcs_pscale(in_wcs)
        out_scale = get_wcs_pscale(out_wcs)
        out_data *= out_scale**2/in_scale**2

    return out_data.astype(in_data.dtype)


def _slice_ndfilter(data, filter_func, slices, args, size, footprint, kwargs):
    """
    Helper function passing image slices to `scipy.ndimage` filters that is
    pickleable for threading with `multiprocessing`
    
    Parameters
    ----------
    data, filter_func, args, size, footprint : 
        See `multiprocessing_ndfilter`
    
    slices : (slice, slice, slice, slice)
        Array slices for insert a cutout back into a larger parent array
        
    Returns
    -------
    filtered : array-like
        Filtered data
    
    slices : tuple
        `slices` as input
        
    """
    filtered = filter_func(data, *args, 
                           size=size, footprint=footprint,
                           **kwargs)
                           
    return filtered, slices


def multiprocessing_ndfilter(data, filter_func, filter_args=(), size=None, footprint=None, cutout_size=256, n_proc=4, timeout=90, mask=None, verbose=True, **kwargs):
    """
    Cut up a large array and send slices to `scipy.ndimage` filters
    
    Parameters
    ----------
    
    data : array-like
        Main image array
        
    filter_func : function
        Filtering function, e.g., `scipy.ndimage.median_filter`
    
    filter_args : tuple
        Arguments to pass to `filter_func`
    
    size, footprint : int, array-like
        Filter size or footprint, see, e.g., `scipy.ndimage.median_filter`
    
    cutout_size : int
        Size of subimage cutouts
    
    n_proc : int
        Number of `multiprocessing` processes to use
    
    timeout : float
        `multiprocessing` timeout (seconds)
    
    mask : array-like
        Array multiplied to `data` that can zero-out regions to ignore
    
    verbose : bool
        Print status messages
        
    kwargs : dict
        Keyword arguments passed through to `filter_func`
        
    Returns
    -------
    filtered : array-like
        Filtered version of `data`
    
    Examples
    --------

        >>> import time
        >>> import numpy as np
        >>> import scipy.ndimage as nd
        >>> from grizli.utils import multiprocessing_ndfilter
        >>> rnd = np.random.normal(size=(512,512))
        >>> t0 = time.time()
        >>> f_serial = nd.median_filter(rnd, size=10)
        >>> t1 = time.time()
        >>> f_mp = multiprocessing_ndfilter(rnd, nd.median_filter, size=10,
        >>>                                 cutout_size=256, n_proc=4)
        >>> t2 = time.time()
        >>> np.allclose(f_serial, f_mp)
        True
        >>> print(f'  serial: {(t1-t0)*1000:.1f} ms')
        >>> print(f'parallel: {(t2-t1)*1000:.1f} ms')
          serial: 573.9 ms
        parallel: 214.8 ms
                                            
    """
    
    import multiprocessing as mp
    
    try:
        from tqdm import tqdm
    except ImportError:
        verbose = False
        
    sh = data.shape
    
    msg = None
    
    if cutout_size > np.max(sh):
        msg = f'cutout_size={cutout_size} greater than image dimensions, run '
        msg += f'`{filter_func}` directly'
    elif n_proc == 0:
        msg = f'n_proc = 0, run in a single command'
    
    if msg is not None:
        if verbose:
            print(msg)
            
        filtered = filter_func(data, *filter_args, 
                               size=size, footprint=footprint)
        return filtered
    
    # Grid size
    nx = data.shape[1]//cutout_size+1
    ny = data.shape[0]//cutout_size+1

    # Padding
    if footprint is not None:
        fpsh = footprint.shape
        pad = np.max(fpsh)
    elif size is not None:
        pad = size
    else:
        raise ValueError('Either size or footprint must be specified')
        
    if n_proc < 0:
        n_proc = mp.cpu_count()

    n_proc = np.minimum(n_proc, mp.cpu_count())

    pool = mp.Pool(processes=n_proc)
    jobs = []
    
    if mask is not None:
        data_mask = data*mask
    else:
        data_mask = data
        
    # Make image slices
    for i in range(nx):
        xmi = np.maximum(0, i*cutout_size-pad)
        xma = np.minimum(sh[1], (i+1)*cutout_size+pad)
        
        #print(i, xmi, xma)
        if i == 0:
            slx = slice(0, cutout_size)
            x0 = 0
        elif i < nx-1:
            slx = slice(pad, cutout_size + pad)
            x0 = i*cutout_size
        else:
            slx = slice(pad, cutout_size + 1)
            x0 = xmi+pad

        nxs = slx.stop - slx.start
        oslx = slice(x0, x0+nxs)
        
        for j in range(ny):
            ymi = np.maximum(0, j*cutout_size - pad)
            yma = np.minimum(sh[0], (j+1)*cutout_size + pad)
            
            if j == 0:
                sly = slice(0, cutout_size)
                y0 = 0
            elif j < ny-1:
                sly = slice(pad, cutout_size + pad)
                y0 = j*cutout_size
            else:
                sly = slice(pad, cutout_size + 1)
                y0 = ymi+pad
            
            nys = sly.stop - sly.start
            osly = slice(y0, y0+nys)
        
            cut = data_mask[ymi:yma, xmi:xma]
            if cut.max() == 0:
                #print(f'Skip {xmi} {xma} {ymi} {yma}')
                continue
                
            # Make jobs for filtering the image slices
            slices = (osly, oslx, sly, slx)
            _args = (cut, filter_func, slices, 
                     filter_args, size, footprint, kwargs)
            jobs.append(pool.apply_async(_slice_ndfilter, _args))
            
    # Collect results
    pool.close()

    filtered = np.zeros_like(data)
    
    if verbose:
        _iter = tqdm(jobs)
    else:
        _iter = jobs
        
    for res in _iter:
        filtered_i, slices = res.get(timeout=timeout)
        filtered[slices[:2]] += filtered_i[slices[2:]]

    return filtered

def parse_flt_files(files=[], info=None, uniquename=False, use_visit=False,
                    get_footprint=False,
                    translate={'AEGIS-': 'aegis-',
                                 'COSMOS-': 'cosmos-',
                                 'GNGRISM': 'goodsn-',
                                 'GOODS-SOUTH-': 'goodss-',
                                 'UDS-': 'uds-'},
                    visit_split_shift=1.5, max_dt=1e9, 
                    path='../RAW'):
    """Read header information from a list of exposures and parse out groups based on filter/target/orientation.

    Parameters
    ----------
    files : list
        List of exposure filenames.  If not specified, will use ``*flt.fits``.

    info : None or `~astropy.table.Table`
        Output from `~grizli.utils.get_flt_info`.
        
    uniquename : bool
        If True, then split everything by program ID and visit name.  If
        False, then just group by targname/filter/pa_v3.

    use_visit : bool
        For parallel observations with ``targname='ANY'``, use the filename
        up to the visit ID as the target name.  For example:

            >>> flc = 'jbhj64d8q_flc.fits'
            >>> visit_targname = flc[:6]
            >>> print(visit_targname)
            jbhj64

        If False, generate a targname for parallel observations based on the
        pointing coordinates using `radec_to_targname`.  Use this keyword
        for dithered parallels like 3D-HST / GLASS but set to False for
        undithered parallels like WISP.  Should also generally be used with
        ``uniquename=False`` otherwise generates names that are a bit 
        redundant:

            +--------------+---------------------------+
            | `uniquename` | Output Targname           |
            +==============+===========================+
            |     True     | jbhj45-bhj-45-180.0-F814W |
            +--------------+---------------------------+
            |     False    | jbhj45-180.0-F814W        |
            +--------------+---------------------------+

    translate : dict
        Translation dictionary to modify TARGNAME keywords to some other
        value.  Used like:

            >>> targname = 'GOODS-SOUTH-10'
            >>> translate = {'GOODS-SOUTH-': 'goodss-'}
            >>> for k in translate:
            >>>     targname = targname.replace(k, translate[k])
            >>> print(targname)
            goodss-10
    
    visit_split_shift : float
        Separation in ``arcmin`` beyond which exposures in a group are split 
        into separate visits.
    
    path : str
        PATH to search for `flt` files if ``info`` not provided
    
    Returns
    -------
    output_list : dict
        Dictionary split by target/filter/pa_v3. Keys are derived visit
        product names and values are lists of exposure filenames corresponding
        to that set. Keys are generated with the formats like:

            >>> targname = 'macs1149+2223'
            >>> pa_v3 = 32.0
            >>> filter = 'f140w'
            >>> flt_filename = 'ica521naq_flt.fits'
            >>> propstr = flt_filename[1:4]
            >>> visit = flt_filename[4:6]
            >>> # uniquename = False
            >>> print('{0}-{1:05.1f}-{2}'.format(targname, pa_v3, filter))
            macs1149.6+2223-032.0-f140w
            >>> # uniquename = True
            >>> print('{0}-{1:3s}-{2:2s}-{3:05.1f}-{4:s}'.format(targname, propstr, visit, pa_v3, filter))
            macs1149.6+2223-ca5-21-032.0-f140w

    filter_list : dict
        Nested dictionary split by filter and then PA_V3.  This shouldn't
        be used if exposures from completely disjoint pointings are stored
        in the same working directory.
    """

    if info is None:
        if not files:
            files = glob.glob(os.path.join(path), '*flt.fits')

        if len(files) == 0:
            return False

        info = get_flt_info(files)
    else:
        info = info.copy()

    for c in info.colnames:
        if not c.islower():
            info.rename_column(c, c.lower())

    if 'expstart' not in info.colnames:
        info['expstart'] = info['exptime']*0.

    so = np.argsort(info['expstart'])
    info = info[so]

    #pa_v3 = np.round(info['pa_v3']*10)/10 % 360.
    pa_v3 = np.round(np.round(info['pa_v3'], decimals=1)) % 360.

    target_list = []
    for i in range(len(info)):
        # Replace ANY targets with JRhRmRs-DdDmDs
        if info['targname'][i] == 'ANY':
            if use_visit:
                new_targname = info['file'][i][:6]
            else:
                new_targname = 'par-'+radec_to_targname(ra=info['ra_targ'][i],
                                             dec=info['dec_targ'][i])

            target_list.append(new_targname.lower())
        else:
            target_list.append(info['targname'][i])

    target_list = np.array(target_list)
    
    _prog_ids = []
    visits = []
    
    for file in info['file']:
        bfile = os.path.basename(file)
        if bfile.startswith('jw'):
            _prog_ids.append(bfile[2:7])
            visits.append(bfile[7:10])
        else:
            _prog_ids.append(bfile[1:4])
            visits.append(bfile[4:6])
    
    visits = np.array(visits)
    
    info['progIDs'] = _prog_ids

    progIDs = np.unique(info['progIDs'])
    dates = np.array([''.join(date.split('-')[1:])
                      for date in info['date-obs']])

    targets = np.unique(target_list)

    output_list = []  # OrderedDict()
    filter_list = OrderedDict()

    for filter in np.unique(info['filter']):
        filter_list[filter] = OrderedDict()
        angles = np.unique(pa_v3[(info['filter'] == filter)])
        for angle in angles:
            filter_list[filter][angle] = []

    for target in targets:
        # 3D-HST targname translations
        target_use = target
        for key in translate.keys():
            target_use = target_use.replace(key, translate[key])

        # pad i < 10 with zero
        for key in translate.keys():
            if translate[key] in target_use:
                spl = target_use.split('-')
                try:
                    if (int(spl[-1]) < 10) & (len(spl[-1]) == 1):
                        spl[-1] = '{0:02d}'.format(int(spl[-1]))
                        target_use = '-'.join(spl)
                except:
                    pass

        for filter in np.unique(info['filter'][(target_list == target)]):
            angles = np.unique(pa_v3[(info['filter'] == filter) &
                                (target_list == target)])
            for angle in angles:

                exposure_list = []
                exposure_start = []
                product = '{0}-{1:05.1f}-{2}'.format(target_use, angle, filter)
                visit_match = np.unique(visits[(target_list == target) &
                                               (info['filter'] == filter)])

                this_progs = []
                this_visits = []

                for visit in visit_match:
                    ix = (visits == visit) & (target_list == target)
                    ix &= (info['filter'] == filter)
                    
                    # this_progs.append(info['progIDs'][ix][0])
                    # print visit, ix.sum(), np.unique(info['progIDs'][ix])
                    new_progs = list(np.unique(info['progIDs'][ix]))
                    this_visits.extend([visit]*len(new_progs))
                    this_progs.extend(new_progs)

                for visit, prog in zip(this_visits, this_progs):
                    visit_list = []
                    visit_start = []
                    
                    _vstr = '{0}-{1}-{2}-{3:05.1f}-{4}'
                    visit_product = _vstr.format(target_use, prog, visit, 
                                                 angle, filter)

                    use = (target_list == target)
                    use &= (info['filter'] == filter)
                    use &= (visits == visit)
                    use &= (pa_v3 == angle)
                    use &= (info['progIDs'] == prog)
                    
                    if use.sum() == 0:
                        continue

                    for tstart, file in zip(info['expstart'][use],
                                            info['file'][use]):

                        f = file.split('.gz')[0]
                        if f not in exposure_list:
                            visit_list.append(str(f))
                            visit_start.append(tstart)

                    exposure_list = np.append(exposure_list, visit_list)
                    exposure_start.extend(visit_start)

                    filter_list[filter][angle].extend(visit_list)

                    if uniquename:
                        print(visit_product, len(visit_list))
                        so = np.argsort(visit_start)
                        exposure_list = np.array(visit_list)[so]
                        #output_list[visit_product.lower()] = visit_list

                        d = OrderedDict(product=str(visit_product.lower()),
                                        files=list(np.array(visit_list)[so]))
                        output_list.append(d)

                if not uniquename:
                    print(product, len(exposure_list))
                    so = np.argsort(exposure_start)
                    exposure_list = np.array(exposure_list)[so]
                    #output_list[product.lower()] = exposure_list
                    d = OrderedDict(product=str(product.lower()),
                                    files=list(np.array(exposure_list)[so]))
                    output_list.append(d)

    # Split large shifts
    if visit_split_shift > 0:
        split_list = []
        for o in output_list:
            _spl = split_visit(o, path=path, 
                               max_dt=max_dt,
                               visit_split_shift=visit_split_shift)
                               
            split_list.extend(_spl)

        output_list = split_list

    # Get visit footprint from FLT WCS
    if get_footprint:
        from shapely.geometry import Polygon

        N = len(output_list)
        for i in range(N):
            for j in range(len(output_list[i]['files'])):
                flt_file = output_list[i]['files'][j]
                if (not os.path.exists(flt_file)):
                    for gzext in ['', '.gz']:
                        _flt_file = os.path.join(path, flt_file + gzext)
                        if os.path.exists(_flt_file):
                            flt_file = _flt_file
                            break
                    
                flt_j = pyfits.open(flt_file)
                h = flt_j[0].header
                _ext = 0
                if (h['INSTRUME'] == 'WFC3'):
                    _ext = 1
                    if (h['DETECTOR'] == 'IR'):
                        wcs_j = pywcs.WCS(flt_j['SCI', 1])
                    else:
                        wcs_j = pywcs.WCS(flt_j['SCI', 1], fobj=flt_j)
                elif (h['INSTRUME'] == 'WFPC2'):
                    _ext = 1
                    wcs_j = pywcs.WCS(flt_j['SCI', 1])
                else:
                    _ext = 1
                    wcs_j = pywcs.WCS(flt_j['SCI', 1], fobj=flt_j)
                
                if ((wcs_j.pixel_shape is None) & 
                    ('NPIX1' in flt_j['SCI',1].header)):
                    _h = flt_j['SCI',1].header
                    wcs_j.pixel_shape = (_h['NPIX1'], _h['NPIX2'])
                    
                fp_j = Polygon(wcs_j.calc_footprint())
                if j == 0:
                    fp_i = fp_j.buffer(1./3600)
                else:
                    fp_i = fp_i.union(fp_j.buffer(1./3600))
                
                flt_j.close()
                
            output_list[i]['footprint'] = fp_i

    return output_list, filter_list


def split_visit(visit, visit_split_shift=1.5, max_dt=6./24, path='../RAW'):
    """
    Check if files in a visit have large shifts and split them otherwise

    visit : visit dictionary

    visit_split_shift : split if shifts larger than `visit_split_shift` arcmin
    """
    
    ims = []
    for file in visit['files']:
        for gzext in ['', '.gz']:
            _file = os.path.join(path, file) + gzext
            if os.path.exists(_file):
                ims.append(pyfits.open(_file))
                break
    
    #ims = [pyfits.open(os.path.join(path, file)) for file in visit['files']]
    crval1 = np.array([im[1].header['CRVAL1'] for im in ims])
    crval2 = np.array([im[1].header['CRVAL2'] for im in ims])
    expstart = np.array([im[0].header['EXPSTART'] for im in ims])
    dt = np.cast[int]((expstart-expstart[0])/max_dt)
    
    for im in ims:
        im.close()
        
    dx = (crval1 - crval1[0])*60*np.cos(crval2[0]/180*np.pi)
    dy = (crval2 - crval2[0])*60

    dxi = np.cast[int](np.round(dx/visit_split_shift))
    dyi = np.cast[int](np.round(dy/visit_split_shift))
    keys = dxi*100+dyi+1000*dt

    un = np.unique(keys)
    if len(un) == 1:
        return [visit]
    else:
        spl = visit['product'].split('-')
        isJWST = spl[-1].lower().startswith('clear')
        isJWST |= spl[-1].lower() in ['gr150r','gr150c','grismr','grismc']
        if isJWST:
            spl.insert(-2, '')
        else:
            spl.insert(-1, '')
            
        visits = []
        for i in range(len(un)):
            ix = keys == un[i]
            if isJWST:
                spl[-3] = 'abcdefghijklmnopqrsuvwxyz'[i]
            else:
                spl[-2] = 'abcdefghijklmnopqrsuvwxyz'[i]
                
            new_visit = {'files': list(np.array(visit['files'])[ix]),
                         'product': '-'.join(spl)}

            if 'footprints' in visit:
                new_visit['footprints'] = list(np.array(visit['footprints'])[ix])

            visits.append(new_visit)

    return visits


def get_visit_footprints(visits):
    """
    Add `~shapely.geometry.Polygon` 'footprint' attributes to visit dict.

    Parameters
    ----------
    visits : list
        List of visit dictionaries.

    """

    import os

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    from shapely.geometry import Polygon

    N = len(visits)
    for i in range(N):
        for j in range(len(visits[i]['files'])):
            flt_file = visits[i]['files'][j]
            if (not os.path.exists(flt_file)) & os.path.exists('../RAW/'+flt_file):
                flt_file = '../RAW/'+flt_file

            flt_j = pyfits.open(flt_file)
            h = flt_j[0].header
            if (h['INSTRUME'] == 'WFC3') & (h['DETECTOR'] == 'IR'):
                wcs_j = pywcs.WCS(flt_j['SCI', 1])
            else:
                wcs_j = pywcs.WCS(flt_j['SCI', 1], fobj=flt_j)

            fp_j = Polygon(wcs_j.calc_footprint())
            if j == 0:
                fp_i = fp_j
            else:
                fp_i = fp_i.union(fp_j)
            
            flt_j.close()
            
        visits[i]['footprint'] = fp_i

    return visits


def parse_visit_overlaps(visits, buffer=15.):
    """Find overlapping visits/filters to make combined mosaics

    Parameters
    ----------
    visits : list
        Output list of visit information from `~grizli.utils.parse_flt_files`.
        The script looks for files like `visits[i]['product']+'_dr?_sci.fits'`
        to compute the WCS footprint of a visit.  These are produced, e.g., by
        `~grizli.prep.process_direct_grism_visit`.

    buffer : float
        Buffer, in `~astropy.units.arcsec`, to add around visit footprints to
        look for overlaps.

    Returns
    -------
    exposure_groups : list
        List of overlapping visits, with similar format as input `visits`.

    """
    import copy
    from shapely.geometry import Polygon

    N = len(visits)

    exposure_groups = []
    used = np.arange(len(visits)) < 0

    for i in range(N):
        f_i = visits[i]['product'].split('-')[-1]
        if used[i]:
            continue

        if 'footprint' in visits[i]:
            fp_i = visits[i]['footprint'].buffer(buffer/3600.)
        else:
            _products = visits[i]['product']+'_dr?_sci.fits'
            im_i = pyfits.open(glob.glob(_products)[0])
            wcs_i = pywcs.WCS(im_i[0])
            fp_i = Polygon(wcs_i.calc_footprint()).buffer(buffer/3600.)
            im_i.close()
            
        exposure_groups.append(copy.deepcopy(visits[i]))

        for j in range(i+1, N):
            f_j = visits[j]['product'].split('-')[-1]
            if (f_j != f_i) | (used[j]):
                continue

            #
            if 'footprint' in visits[j]:
                fp_j = visits[j]['footprint'].buffer(buffer/3600.)
            else:
                _products = visits[j]['product']+'_dr?_sci.fits'
                im_j = pyfits.open(glob.glob(_products)[0])
                wcs_j = pywcs.WCS(im_j[0])
                fp_j = Polygon(wcs_j.calc_footprint()).buffer(buffer/3600.)
                im_j.close()
            
            olap = fp_i.intersection(fp_j)
            if olap.area > 0:
                used[j] = True
                fp_i = fp_i.union(fp_j)
                exposure_groups[-1]['footprint'] = fp_i
                exposure_groups[-1]['files'].extend(visits[j]['files'])

    for i in range(len(exposure_groups)):
        flt_i = pyfits.open(exposure_groups[i]['files'][0])
        product = flt_i[0].header['TARGNAME'].lower()
        if product == 'any':
            product = 'par-'+radec_to_targname(header=flt_i['SCI', 1].header)

        f_i = exposure_groups[i]['product'].split('-')[-1]
        product += '-'+f_i
        exposure_groups[i]['product'] = product
        flt_i.close()
        
    return exposure_groups


DIRECT_ORDER = {'G102': ['F105W', 'F110W', 'F098M', 'F125W', 'F140W', 'F160W', 'F127M', 'F139M', 'F153M', 'F132N', 'F130N', 'F128N', 'F126N', 'F164N', 'F167N'],
                'G141': ['F140W', 'F160W', 'F125W', 'F105W', 'F110W', 'F098M', 'F127M', 'F139M', 'F153M', 'F132N', 'F130N', 'F128N', 'F126N', 'F164N', 'F167N'],
                'G800L': ['F814W', 'F606W', 'F850LP', 'F775W', 'F435W', 'F105W', 'F110W', 'F098M', 'F125W', 'F140W', 'F160W', 'F127M', 'F139M', 'F153M', 'F132N', 'F130N', 'F128N', 'F126N', 'F164N', 'F167N'],
                'GR150C': ['F115W', 'F150W', 'F200W'], 
                'GR150R': ['F115W', 'F150W', 'F200W']}


def parse_grism_associations(exposure_groups, info,
                             best_direct=DIRECT_ORDER,
                             get_max_overlap=True):
    """Get associated lists of grism and direct exposures

    Parameters
    ----------
    exposure_grups : list
        Output list of overlapping visits from
        `~grizli.utils.parse_visit_overlaps`.

    best_direct : dict
        Dictionary of the preferred direct imaging filters to use with a
        particular grism.

    Returns
    -------
    grism_groups : list
        List of dictionaries with associated 'direct' and 'grism' entries.

    """
    N = len(exposure_groups)

    grism_groups = []
    for i in range(N):
        _espi = exposure_groups[i]['product'].split('-')
        
        if _espi[-2][0] in 'fg':
            pupil_i = _espi[-2]
            f_i = _espi[-1]
            root_i = '-'.join(_espi[:-2])
        else:
            pupil_i = None
            f_i = _espi[-1]
            root_i = '-'.join(_espi[:-1])
            
        if f_i.startswith('g'):
            group = OrderedDict(grism=exposure_groups[i],
                                direct=None)
        else:
            continue
        
        fp_i = exposure_groups[i]['footprint']
        olap_i = 0.
        d_i = f_i
        d_idx = 10
        for j in range(N):
            _espj = exposure_groups[j]['product'].split('-')
            if _espj[-2][0] in 'fg':
                pupil_j = _espj[-2]
                f_j = _espj[-1]
                root_j = '-'.join(_espj[:-2])
            else:
                f_j = _espj[-1]
                root_j = '-'.join(_espj[:-1])
                pupil_j = None
                
            if f_j.startswith('g'):
                continue

            fp_j = exposure_groups[j]['footprint']
            olap = fp_i.intersection(fp_j)

            if (root_j == root_i):

                if pupil_i is not None:
                    if pupil_j == pupil_i:
                        group['direct'] = exposure_groups[j]
                    else:
                        continue
                else:
                    if f_j.upper() not in best_direct[f_i.upper()]:
                        continue
                    if best_direct[f_i.upper()].index(f_j.upper()) < d_idx:
                        d_idx = best_direct[f_i.upper()].index(f_j.upper())
                        group['direct'] = exposure_groups[j]
                        olap_i = olap.area
                        d_i = f_j

        grism_groups.append(group)

    return grism_groups


def get_hst_filter(header, **kwargs):
    """
    Deprecated: use `grizli.utils.parse_filter_from_header`
    """
    
    result = parse_filter_from_header(header, **kwargs)
    return result


def parse_filter_from_header(header, filter_only=False, jwst_detector=False, **kwargs):
    """Get simple filter name out of an HST/JWST image header.

    ACS has two keywords for the two filter wheels, so just return the
    non-CLEAR filter. For example,

        >>> h = astropy.io.fits.Header()
        >>> h['INSTRUME'] = 'ACS'
        >>> h['FILTER1'] = 'CLEAR1L'
        >>> h['FILTER2'] = 'F814W'
        >>> from grizli.utils import parse_filter_from_header
        >>> print(parse_filter_from_header(h))
        F814W
        >>> h['FILTER1'] = 'G800L'
        >>> h['FILTER2'] = 'CLEAR2L'
        >>> print(parse_filter_from_header(h))
        G800L
            
    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Image header with FILTER or FILTER1,FILTER2,...,FILTERN keywords
    
    filter_only : bool
        If true, don't do any special handling with JWST but just return the
        ``FILTER`` keyword itself. Otherwise, for JWST/NIRISS, return 
        ``{PUPIL}-{FILTER}`` and for JWST/NIRCAM, return ``{FILTER}-{PUPIL}``
        
    jwst_detector : bool
        If True, prepend ``DETECTOR`` to output for JWST NIRCam and NIRISS
        to distinguish NIRCam detectors and filter names common between 
        these instruments.
        
    Returns
    -------
    filter : str

    """
    
    if 'INSTRUME' not in header:
        instrume = 'N/A'
    else:
        instrume = header['INSTRUME']
        
    if instrume.strip() == 'ACS':
        for i in [1, 2]:
            filter_i = header['FILTER{0:d}'.format(i)]
            if 'CLEAR' in filter_i:
                continue
            else:
                filter = filter_i

    elif instrume == 'WFPC2':
        filter = header['FILTNAM1']
        
    elif instrume == 'NIRISS':
        if filter_only:
            filter = header['FILTER']
        else:
            filter = '{0}-{1}'.format(header['PUPIL'], header['FILTER'])
        
        if jwst_detector:
            filter = '{0}-{1}'.format(header['DETECTOR'], filter)
            
    elif instrume == 'NIRCAM':
        if filter_only:
            filter = header['FILTER']
        else:
            filter = '{0}-{1}'.format(header['FILTER'], header['PUPIL'])
        if jwst_detector:
            filter = '{0}-{1}'.format(header['DETECTOR'], filter)
            filter = filter.replace('LONG','5')
            
    elif 'FILTER' in header:
        filter = header['FILTER']
        
    else:
        msg = 'Failed to parse FILTER keyword for INSTRUMEnt {0}'
        raise KeyError(msg.format(instrume))

    return filter.upper()


EE_RADII = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8, 1., 1.5, 2.]

def get_filter_obsmode(filter='f160w', acs_chip='wfc1', uvis_chip='uvis2', aper=np.inf, case=str.lower):
    """
    Derive `~pysynphot` obsmode keyword from a filter name, where UVIS filters
    end in 'u'
    """
    if filter.lower()[:2] in ['f0', 'f1', 'g1']:
        inst = 'wfc3,ir'
    else:
        if filter.lower().endswith('u'):
            inst = f'wfc3,{uvis_chip}'
        else:
            inst = f'acs,{acs_chip}'
        
    obsmode = inst + ',' + filter.strip('u').lower()
    if np.isfinite(aper):
        obsmode += f',aper#{aper:4.2f}'
        
    return case(obsmode)
    
def tabulate_encircled_energy(aper_radii=EE_RADII, norm_radius=4.0):

    import pysynphot as S

    from .pipeline import default_params

    # Default spectrum
    sp = S.FlatSpectrum(25, fluxunits='ABMag')

    tab = GTable()
    tab['radius'] = aper_radii*u.arcsec
    tab.meta['RNORM'] = norm_radius, 'Normalization radius, arcsec'

    # IR
    for f in default_params.IR_M_FILTERS+default_params.IR_W_FILTERS:
        obsmode = 'wfc3,ir,'+f.lower()
        
        print(obsmode)
        tab[obsmode] = synphot_encircled_energy(obsmode=obsmode, sp=sp, aper_radii=aper_radii, norm_radius=norm_radius)
        tab.meta['ZP_{0}'.format(obsmode)] = synphot_zeropoint(obsmode=obsmode, radius=norm_radius)

    # Optical.  Wrap in try/except to catch missing filters
    for inst in ['acs,wfc1,', 'wfc3,uvis2,']:
        for f in (default_params.OPT_M_FILTERS + 
                  default_params.OPT_W_FILTERS + 
                  default_params.UV_M_FILTERS + 
                  default_params.UV_W_FILTERS):

            obsmode = inst+f.lower()

            try:
                tab[obsmode] = synphot_encircled_energy(obsmode=obsmode, sp=sp, aper_radii=aper_radii, norm_radius=norm_radius)
                print(obsmode)
                tab.meta['ZP_{0}'.format(obsmode)] = synphot_zeropoint(obsmode=obsmode, radius=norm_radius)
            except:
                # Failed because obsmode not available in synphot
                continue

    tab.meta['PSYNVER'] = S.__version__, 'Pysynphot version'

    tab.write('hst_encircled_energy.fits', overwrite=True)


def synphot_zeropoint(obsmode='wfc3,ir,f160w', radius=4.0):
    """
    Compute synphot for a specific aperture
    """
    import pysynphot as S
    sp = S.FlatSpectrum(25, fluxunits='ABMag')
    
    if np.isfinite(radius):
        bp = S.ObsBandpass(obsmode+',aper#{0:.2f}'.format(radius))
    else:
        bp = S.ObsBandpass(obsmode)

    # bp = S.ObsBandpass(obsmode+',aper#{0:.2f}'.format(radius))
    obs = S.Observation(sp, bp)
    ZP = 25 + 2.5*np.log10(obs.countrate())
    return ZP


def synphot_encircled_energy(obsmode='wfc3,ir,f160w', sp='default', aper_radii=EE_RADII, norm_radius=4.0):
    """
    Compute encircled energy curves with pysynphot
    """
    import pysynphot as S

    if sp == 'default':
        sp = S.FlatSpectrum(25, fluxunits='ABMag')

    # Normalization
    if np.isfinite(norm_radius):
        bp = S.ObsBandpass(obsmode+',aper#{0:.2f}'.format(norm_radius))
    else:
        bp = S.ObsBandpass(obsmode)

    obs = S.Observation(sp, bp)
    norm_counts = obs.countrate()

    counts = np.ones_like(aper_radii)
    for i, r_aper in enumerate(aper_radii):
        #print(obsmode, r_aper)
        bp = S.ObsBandpass(obsmode+',aper#{0:.2f}'.format(r_aper))
        obs = S.Observation(sp, bp)
        counts[i] = obs.countrate()

    return counts / norm_counts


def photfnu_from_photflam(photflam, photplam):
    """
    Compute PHOTFNU from PHOTFLAM+PHOTPLAM, e.g., for ACS/WFC
    """
    ZP = -2.5*np.log10(photflam) - 21.10 - 5*np.log10(photplam) + 18.6921
    photfnu = 10**(-0.4*(ZP-23.9))*1.e-6
    return photfnu


def calc_header_zeropoint(im, ext=0):
    """
    Determine AB zeropoint from image header

    Parameters
    ----------
    im : `~astropy.io.fits.HDUList` or
        Image object or header.

    Returns
    -------
    ZP : float
        AB zeropoint

    """
    from . import model

    scale_exptime = 1.

    if isinstance(im, pyfits.Header):
        header = im
    else:
        if '_dr' in im.filename():
            ext = 0
        elif '_fl' in im.filename():
            if 'DETECTOR' in im[0].header:
                if im[0].header['DETECTOR'] == 'IR':
                    ext = 0
                    bunit = im[1].header['BUNIT']
                else:
                    # ACS / UVIS
                    if ext == 0:
                        ext = 1

                    bunit = im[1].header['BUNIT']

                if bunit == 'ELECTRONS':
                    scale_exptime = im[0].header['EXPTIME']

        header = im[ext].header

    try:
        fi = parse_filter_from_header(im[0].header).upper()
    except:
        fi = None

    # Get AB zeropoint
    if 'APZP' in header:
        ZP = header['ABZP']
    
    elif 'PHOTFNU' in header:
        ZP = -2.5*np.log10(header['PHOTFNU'])+8.90
        ZP += 2.5*np.log10(scale_exptime)
        
    elif 'PHOTFLAM' in header:
        ZP = (-2.5*np.log10(header['PHOTFLAM']) - 21.10 -
              5*np.log10(header['PHOTPLAM']) + 18.6921)

        ZP += 2.5*np.log10(scale_exptime)
    
    elif (fi is not None):
        if fi in model.photflam_list:
            ZP = (-2.5*np.log10(model.photflam_list[fi]) - 21.10 -
                  5*np.log10(model.photplam_list[fi]) + 18.6921)
        else:
            print('Couldn\'t find PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25')
            ZP = 25
    else:
        print('Couldn\'t find FILTER, PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25')
        ZP = 25

    # If zeropoint infinite (e.g., PHOTFLAM = 0), then calculate from synphot
    if not np.isfinite(ZP):
        try:
            import pysynphot as S
            bp = S.ObsBandpass(im[0].header['PHOTMODE'].replace(' ', ','))
            spec = S.FlatSpectrum(0, fluxunits='ABMag')
            obs = S.Observation(spec, bp)
            ZP = 2.5*np.log10(obs.countrate())
        except:
            pass

    return ZP


DEFAULT_PRIMARY_KEYS = ['FILENAME', 'INSTRUME', 'INSTRUME', 'DETECTOR', 'FILTER', 'FILTER1', 'FILTER2', 'EXPSTART', 'DATE-OBS', 'EXPTIME', 'IDCTAB', 'NPOLFILE', 'D2IMFILE', 'PA_V3', 'FGSLOCK', 'GYROMODE', 'PROPOSID']

# For grism
DEFAULT_EXT_KEYS = ['EXTNAME', 'EXTVER', 'MDRIZSKY', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2', 'CTYPE1', 'CTYPE2', 'RADESYS', 'LONPOLE', 'LATPOLE', 'IDCTAB', 'D2IMEXT', 'WCSNAME', 'PHOTMODE', 'ORIENTAT', 'CCDCHIP']


def flt_to_dict(fobj, primary_keys=DEFAULT_PRIMARY_KEYS, extensions=[('SCI', i+1) for i in range(2)], ext_keys=DEFAULT_EXT_KEYS):
    """
    Parse basic elements from a FLT/FLC header to a dictionary

    TBD

    Parameters
    ----------
    fobj : `~astropy.io.fits.HDUList`
        FITS object

    primary_keys : list
        Keywords to extract from the primary extension (0).

    extensions : list
        List of additional extension names / indices.

    ext_keys : list
        Keywords to extract from the extension headers.

    Returns
    -------
    flt_dict : dict

    """
    import astropy.time

    flt_dict = OrderedDict()
    flt_dict['timestamp'] = astropy.time.Time.now().iso
    h0 = fobj[0].header

    # Primary keywords
    for k in primary_keys:
        if k in h0:
            flt_dict[k] = h0[k]

    # Grism keys
    for k in h0:
        if k.startswith('GSKY'):
            flt_dict[k] = h0[k]

    # WCS, etc. keywords from SCI extensions
    flt_dict['extensions'] = OrderedDict()
    count = 0
    for ext in extensions:
        if ext in fobj:
            d_i = OrderedDict()
            h_i = fobj[ext].header
            for k in ext_keys:
                if k in h_i:
                    d_i[k] = h_i[k]

            # Grism keys
            for k in h_i:
                if k.startswith('GSKY'):
                    d_i[k] = h_i[k]

            count += 1
            flt_dict['extensions'][count] = d_i

    return flt_dict


def get_set_bits(value):
    """
    Compute which binary bits are set for an integer
    """

    if hasattr(value, '__iter__'):
        values = value
        single = False
    else:
        values = [value]
        single = True

    result = []

    for v in values:
        try:
            bitstr = np.binary_repr(v)[::-1]
        except:
            result.append([])

        nset = bitstr.count('1')
        setbits = []

        j = -1
        for i in range(nset):
            j = bitstr.index('1', j+1)
            setbits.append(j)

        result.append(setbits)

    if single:
        return result[0]
    else:
        return result


def unset_dq_bits(value, okbits=32+64+512, verbose=False):
    """
    Unset bit flags from a DQ array

    For WFC3/IR, the following DQ bits can usually be unset:

    32, 64: these pixels usually seem OK
       512: blobs not relevant for grism exposures

    Parameters
    ----------
    value : int, `~numpy.ndarray`
        Input DQ value

    okbits : int
        Bits to unset

    verbose : bool
        Print some information

    Returns
    -------
    new_value : int, `~numpy.ndarray`

    """
    bin_bits = np.binary_repr(okbits)
    n = len(bin_bits)
    for i in range(n):
        if bin_bits[-(i+1)] == '1':
            if verbose:
                print(2**i)

            value -= (value & 2**i)

    return value


def detect_with_photutils(sci, err=None, dq=None, seg=None, detect_thresh=2.,
                        npixels=8, grow_seg=5, gauss_fwhm=2., gsize=3,
                        wcs=None, save_detection=False, root='mycat',
                        background=None, gain=None, AB_zeropoint=0.,
                        rename_columns={'xcentroid': 'x_flt',
                                          'ycentroid': 'y_flt',
                                          'ra_icrs_centroid': 'ra',
                                          'dec_icrs_centroid': 'dec'},
                        overwrite=True, verbose=True):
    """
    Use `~photutils` to detect objects and make segmentation map
    
    .. note:: 
        Deprecated in favor of sep catalogs in `~grizli.prep`.
    
    Parameters
    ----------
    sci : `~numpy.ndarray`
        TBD

    err, dq, seg : TBD

    detect_thresh : float
        Detection threshold, in :math:`\sigma`

    grow_seg : int
        Number of pixels to grow around the perimeter of detected objects
        witha  maximum filter

    gauss_fwhm : float
        FWHM of Gaussian convolution kernel that smoothes the detection
        image.

    verbose : bool
        Print logging information to the terminal

    save_detection : bool
        Save the detection images and catalogs

    wcs : `~astropy.wcs.WCS`
        WCS object passed to `photutils.source_properties` used to compute
        sky coordinates of detected objects.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Object catalog with the default parameters.
    """
    import scipy.ndimage as nd

    from photutils import detect_threshold, detect_sources, SegmentationImage
    from photutils import source_properties

    import astropy.io.fits as pyfits
    from astropy.table import Column

    from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
    from astropy.convolution import Gaussian2DKernel

    # DQ masks
    mask = (sci == 0)
    if dq is not None:
        mask |= dq > 0

    # Detection threshold
    if err is None:
        threshold = detect_threshold(sci, snr=detect_thresh, mask=mask)
    else:
        threshold = (detect_thresh * err)*(~mask)
        threshold[mask] = np.median(threshold[~mask])

    if seg is None:
        # Run the source detection and create the segmentation image

        # Gaussian kernel
        sigma = gauss_fwhm * gaussian_fwhm_to_sigma    # FWHM = 2.
        kernel = Gaussian2DKernel(sigma, x_size=gsize, y_size=gsize)
        kernel.normalize()

        if verbose:
            print('{0}: photutils.detect_sources (detect_thresh={1:.1f}, grow_seg={2:d}, gauss_fwhm={3:.1f}, ZP={4:.1f})'.format(root, detect_thresh, grow_seg, gauss_fwhm, AB_zeropoint))

        # Detect sources
        segm = detect_sources(sci*(~mask), threshold, npixels=npixels,
                              filter_kernel=kernel)

        grow = nd.maximum_filter(segm.data, grow_seg)
        seg = np.cast[np.float32](grow)
    else:
        # Use the supplied segmentation image
        segm = SegmentationImage(seg)

    # Source properties catalog
    if verbose:
        print('{0}: photutils.source_properties'.format(root))

    props = source_properties(sci, segm, error=threshold/detect_thresh,
                              mask=mask, background=background, wcs=wcs)

    catalog = props.to_table()

    # Mag columns
    mag = AB_zeropoint - 2.5*np.log10(catalog['source_sum'])
    mag._name = 'mag'
    catalog.add_column(mag)

    try:
        logscale = 2.5/np.log(10)
        mag_err = logscale*catalog['source_sum_err']/catalog['source_sum']
    except:
        mag_err = np.zeros_like(mag)-99

    mag_err._name = 'mag_err'
    catalog.add_column(mag_err)

    # Rename some catalog columns
    for key in rename_columns.keys():
        if key not in catalog.colnames:
            continue

        catalog.rename_column(key, rename_columns[key])
        if verbose:
            print('Rename column: {0} -> {1}'.format(key, rename_columns[key]))

    # Done!
    if verbose:
        print(NO_NEWLINE + ('{0}: photutils.source_properties - {1:d} objects'.format(root, len(catalog))))

    # Save outputs?
    if save_detection:
        seg_file = root + '.detect_seg.fits'
        seg_cat = root + '.detect.cat'
        if verbose:
            print('{0}: save {1}, {2}'.format(root, seg_file, seg_cat))

        if wcs is not None:
            header = wcs.to_header(relax=True)
        else:
            header = None

        pyfits.writeto(seg_file, data=seg, header=header, overwrite=overwrite)

        if os.path.exists(seg_cat) & overwrite:
            os.remove(seg_cat)

        catalog.write(seg_cat, format='ascii.commented_header')

    return catalog, seg


def safe_invert(arr):
    """
    version-safe matrix inversion using np.linalg or np.matrix.I
    """
    try:
        from numpy.linalg import inv
        _inv = inv(arr)
    except:
        _inv = np.matrix(arr).I.A
    
    return _inv
    

def nmad(data):
    """Normalized NMAD=1.4826022 * `~.astropy.stats.median_absolute_deviation`

    """
    import astropy.stats
    return 1.4826022*astropy.stats.median_absolute_deviation(data)


def get_line_wavelengths():
    """Get a dictionary of common emission line wavelengths and line ratios

    Returns
    -------
    line_wavelengths, line_ratios : dict
        Keys are common to both dictionaries and are simple names for lines
        and line complexes.  Values are lists of line wavelengths and line
        ratios.

            >>> from grizli.utils import get_line_wavelengths
            >>> line_wavelengths, line_ratios = get_line_wavelengths()
            >>> print(line_wavelengths['Ha'], line_ratios['Ha'])
            [6564.61] [1.0]
            >>> print(line_wavelengths['OIII'], line_ratios['OIII'])
            [5008.24, 4960.295] [2.98, 1]

        Includes some additional combined line complexes useful for redshift
        fits:

            >>> from grizli.utils import get_line_wavelengths
            >>> line_wavelengths, line_ratios = get_line_wavelengths()
            >>> key = 'Ha+SII+SIII+He'
            >>> print(line_wavelengths[key], '\\n', line_ratios[key])
            [6564.61, 6718.29, 6732.67, 9068.6, 9530.6, 10830.0]
            [1.0, 0.1, 0.1, 0.05, 0.122, 0.04]

    """
    line_wavelengths = OrderedDict()
    line_ratios = OrderedDict()

    # Paschen: https://www.gemini.edu/sciops/instruments/nearir-resources/astronomical-lines/h-lines
    line_wavelengths['BrA'] = [40522.8]
    line_ratios['BrA'] = [1.]
    line_wavelengths['BrB'] = [26258.7]
    line_ratios['BrB'] = [1.]
    line_wavelengths['BrG'] = [21661.178]
    line_ratios['BrG'] = [1.]
    line_wavelengths['PfG'] = [37405.76]
    line_ratios['PfG'] = [1.]
    line_wavelengths['PfD'] = [32969.8]
    line_ratios['PfD'] = [1.]
    line_wavelengths['PaA'] = [18751.0]
    line_ratios['PaA'] = [1.]
    line_wavelengths['PaB'] = [12821.6]
    line_ratios['PaB'] = [1.]
    line_wavelengths['PaG'] = [10941.1]
    line_ratios['PaG'] = [1.]
    line_wavelengths['PaD'] = [10049.0]
    line_ratios['PaD'] = [1.]

    line_wavelengths['Ha'] = [6564.61]
    line_ratios['Ha'] = [1.]
    line_wavelengths['Hb'] = [4862.71]
    line_ratios['Hb'] = [1.]
    line_wavelengths['Hg'] = [4341.692]
    line_ratios['Hg'] = [1.]
    line_wavelengths['Hd'] = [4102.892]
    line_ratios['Hd'] = [1.]

    line_wavelengths['H7'] = [3971.198]
    line_ratios['H7'] = [1.]

    line_wavelengths['H8'] = [3890.166]
    line_ratios['H8'] = [1.]

    line_wavelengths['H9'] = [3836.485]
    line_ratios['H9'] = [1.]

    line_wavelengths['H10'] = [3798.987]
    line_ratios['H10'] = [1.]

    line_wavelengths['H11'] = [3771.70]
    line_ratios['H11'] = [1.]

    line_wavelengths['H12'] = [3751.22]
    line_ratios['H12'] = [1.]

    # Groves et al. 2011, Table 1
    # Osterbrock table 4.4 for H7 to H10
    # line_wavelengths['Balmer 10kK'] = [6564.61, 4862.68, 4341.68, 4101.73]
    # line_ratios['Balmer 10kK'] = [2.86, 1.0, 0.468, 0.259]

    line_wavelengths['Balmer 10kK'] = [6564.61, 4862.68, 4341.68, 4101.73, 3971.198, 3890.166, 3836.485, 3798.987]
    line_ratios['Balmer 10kK'] = [2.86, 1.0, 0.468, 0.259, 0.159, 0.105, 0.0731, 0.0530]

    # Paschen from Osterbrock, e.g., Pa-beta relative to H-gamma
    line_wavelengths['Balmer 10kK'] += line_wavelengths['PaA'] + line_wavelengths['PaB'] + line_wavelengths['PaG'] + line_wavelengths['PaD']
    line_ratios['Balmer 10kK'] += [0.348 * line_ratios['Balmer 10kK'][i] for i in [1, 2, 3, 4]]

    # Osterbrock table 4.4 for H7 to H10
    line_wavelengths['Balmer 10kK + MgII'] = line_wavelengths['Balmer 10kK'] + [2799.117]
    line_ratios['Balmer 10kK + MgII'] = line_ratios['Balmer 10kK'] + [3.]

    # # Paschen from Osterbrock, e.g., Pa-beta relative to H-gamma
    # line_wavelengths['Balmer 10kK + MgII'] += line_wavelengths['PaA'] + line_wavelengths['PaB'] + line_wavelengths['PaG']
    # line_ratios['Balmer 10kK + MgII'] += [0.348 * line_ratios['Balmer 10kK + MgII'][i] for i in [1,2,3]]

    # With Paschen lines & He 10830 from Glikman 2006
    # https://iopscience.iop.org/article/10.1086/500098/pdf
    #line_wavelengths['Balmer 10kK + MgII'] = [6564.61, 4862.68, 4341.68, 4101.73, 3971.198, 2799.117, 12821.6, 10941.1]
    #line_ratios['Balmer 10kK + MgII'] = [2.86, 1.0, 0.468, 0.259, 0.16, 3., 2.86*4.8/100, 2.86*1.95/100]

    # Redden with Calzetti00
    if False:
        from extinction import calzetti00
        Av = 1.0
        Rv = 3.1

        waves = line_wavelengths['Balmer 10kK + MgII']
        ratios = line_ratios['Balmer 10kK + MgII']

        for Av in [0.5, 1.0, 2.0]:
            mred = calzetti00(np.array(waves), Av, Rv)
            fred = 10**(-0.4*mred)

            key = 'Balmer 10kK + MgII Av={0:.1f}'.format(Av)
            line_wavelengths[key] = [w for w in waves]
            line_ratios[key] = [ratios[i]*fred[i] for i in range(len(waves))]

    line_wavelengths['Balmer 10kK + MgII Av=0.5'] = [6564.61, 4862.68, 4341.68, 4101.73, 3971.198, 2799.117, 12821.6, 10941.1]
    line_ratios['Balmer 10kK + MgII Av=0.5'] = [2.009811938798515, 0.5817566641521459, 0.25176970824566913, 0.1338409369665902, 0.08079209880749984, 1.1739297839690317, 0.13092553990513178, 0.05033866127477651]

    line_wavelengths['Balmer 10kK + MgII Av=1.0'] = [6564.61, 4862.68, 4341.68, 4101.73, 3971.198, 2799.117, 12821.6, 10941.1]
    line_ratios['Balmer 10kK + MgII Av=1.0'] = [1.4123580522157504, 0.33844081628543266, 0.13544441450878067, 0.0691636926953466, 0.04079602018575511, 0.4593703792298591, 0.12486521707058751, 0.045436270735820045]

    line_wavelengths['Balmer 10kK + MgII Av=2.0'] = [6564.61, 4862.68, 4341.68, 4101.73, 3971.198, 2799.117, 12821.6, 10941.1]
    line_ratios['Balmer 10kK + MgII Av=2.0'] = [0.6974668768037302, 0.11454218612794999, 0.03919912269578289, 0.018469561340758073, 0.010401970393728362, 0.0703403817712615, 0.11357315292894044, 0.03701729780130422]
    
    ###########
    
    # Reddened with Kriek & Conroy dust, tau_V=0.5
    line_wavelengths['Balmer 10kK t0.5'] = [6564.61, 4862.68, 4341.68, 4101.73]
    line_ratios['Balmer 10kK t0.5'] = [2.86*0.68, 1.0*0.55, 0.468*0.51, 0.259*0.48]

    # Reddened with Kriek & Conroy dust, tau_V=1
    line_wavelengths['Balmer 10kK t1'] = [6564.61, 4862.68, 4341.68, 4101.73]
    line_ratios['Balmer 10kK t1'] = [2.86*0.46, 1.0*0.31, 0.468*0.256, 0.259*0.232]

    line_wavelengths['OIII-4363'] = [4364.436]
    line_ratios['OIII-4363'] = [1.]
    line_wavelengths['OIII'] = [5008.240, 4960.295]
    line_ratios['OIII'] = [2.98, 1]

    # Split doublet, if needed
    line_wavelengths['OIII-4959'] = [4960.295]
    line_ratios['OIII-4959'] = [1]
    line_wavelengths['OIII-5007'] = [5008.240]
    line_ratios['OIII-5007'] = [1]

    line_wavelengths['OII'] = [3727.092, 3729.875]
    line_ratios['OII'] = [1, 1.]

    line_wavelengths['OI-6302'] = [6302.046, 6365.535]
    line_ratios['OI-6302'] = [1, 0.33]
    line_wavelengths['OI-5578'] = [5578.89]
    line_ratios['OI-5578'] = [1]

    # Auroral OII
    # lines roughly taken from https://arxiv.org/pdf/1610.06939.pdf
    line_wavelengths['OII-7325'] = [7322.0, 7332.]
    line_ratios['OII-7325'] = [1.2, 1.]

    # Weak Ar III in SF galaxies
    line_wavelengths['ArIII-7138'] = [7137.77]
    line_ratios['ArIII-7138'] = [1.]

    line_wavelengths['NeIII-3867'] = [3869.87]
    line_ratios['NeIII-3867'] = [1.]
    line_wavelengths['NeIII-3968'] = [3968.59]
    line_ratios['NeIII-3968'] = [1.]
    line_wavelengths['NeV-3346'] = [3343.5]
    line_ratios['NeV-3346'] = [1.]
    line_wavelengths['NeVI-3426'] = [3426.85]
    line_ratios['NeVI-3426'] = [1.]

    line_wavelengths['SIII'] = [9071.1, 9533.2][::-1]
    line_ratios['SIII'] = [1, 2.44][::-1]

    # Split doublet, if needed
    line_wavelengths['SIII-9068'] = [9071.1]
    line_ratios['SIII-9068'] = [1]
    line_wavelengths['SIII-9531'] = [9533.2]
    line_ratios['SIII-9531'] = [1]

    line_wavelengths['SII'] = [6718.29, 6732.67]
    line_ratios['SII'] = [1., 1.]

    line_wavelengths['SII-6717'] = [6718.29]
    line_ratios['SII-6717'] = [1.]
    line_wavelengths['SII-6731'] = [6732.67]
    line_ratios['SII-6731'] = [1.]

    line_wavelengths['SII-4075'] = [4069.75, 4077.5]
    line_ratios['SII-4075'] = [1., 1.]
    line_wavelengths['SII-4070'] = [4069.75]
    line_ratios['SII-4075'] = [1.]
    line_wavelengths['SII-4078'] = [4077.5]
    line_ratios['SII-4078'] = [1.]

    line_wavelengths['HeII-4687'] = [4687.5]
    line_ratios['HeII-4687'] = [1.]
    line_wavelengths['HeII-5412'] = [5412.5]
    line_ratios['HeII-5412'] = [1.]
    line_wavelengths['HeI-5877'] = [5877.249]
    line_ratios['HeI-5877'] = [1.]
    line_wavelengths['HeI-3889'] = [3889.75]
    line_ratios['HeI-3889'] = [1.]
    line_wavelengths['HeI-1083'] = [10832.057, 10833.306]
    line_ratios['HeI-1083'] = [1., 1.]

    # Osterbrock Table 4.5
    # -> N=4
    line_wavelengths['HeI-series'] = [4472.7, 5877.2, 4027.3, 3820.7, 7067.1, 10833.2, 3889.7, 3188.7]
    line_ratios['HeI-series'] = [1., 2.75, 0.474, 0.264, 0.330, 4.42, 2.26, 0.916]

    line_wavelengths['MgII'] = [2799.117]
    line_ratios['MgII'] = [1.]

    line_wavelengths['CIV-1549'] = [1549.480]
    line_ratios['CIV-1549'] = [1.]
    line_wavelengths['CIII-1906'] = [1906.683]
    line_ratios['CIII-1906'] = [1.]
    line_wavelengths['CIII-1908'] = [1908.734]
    line_ratios['CIII-1908'] = [1.]
    line_wavelengths['OIII-1663'] = [1665.85]
    line_ratios['OIII-1663'] = [1.]
    line_wavelengths['HeII-1640'] = [1640.4]
    line_ratios['HeII-1640'] = [1.]

    line_wavelengths['SiIV+OIV-1398'] = [1398.]
    line_ratios['SiIV+OIV-1398'] = [1.]

    # Weak line in LEGA-C spectra
    line_wavelengths['NI-5199'] = [5199.4, 5201.76]
    line_ratios['NI-5199'] = [1., 1.]

    line_wavelengths['NII'] = [6549.86, 6585.27][::-1]
    line_ratios['NII'] = [1.0, 3.0][::-1]

    line_wavelengths['NII-6549'] = [6549.86]
    line_ratios['NII-6549'] = [1.]
    line_wavelengths['NII-6584'] = [6585.27]
    line_ratios['NII-6584'] = [1.]

    line_wavelengths['NIII-1750'] = [1750.]
    line_ratios['NIII-1750'] = [1.]
    line_wavelengths['NIV-1487'] = [1487.]
    line_ratios['NIV-1487'] = [1.]
    line_wavelengths['NV-1240'] = [1240.81]
    line_ratios['NV-1240'] = [1.]

    line_wavelengths['Lya'] = [1215.4]
    line_ratios['Lya'] = [1.]

    line_wavelengths['QSO-UV-lines'] = [line_wavelengths[k][0] for k in ['Lya', 'CIV-1549', 'CIII-1906', 'CIII-1908', 'OIII-1663', 'HeII-1640', 'SiIV+OIV-1398', 'NV-1240', 'NIII-1750']]
    line_ratios['QSO-UV-lines'] = [1., 0.5, 0.1, 0.1, 0.008, 0.09, 0.1, 0.3, 0.05]

    line_wavelengths['QSO-Narrow-lines'] = [line_wavelengths[k][0] for k in ['OII', 'OIII-5007', 'OIII-4959', 'SII-6717', 'SII-6731', 'OI-6302', 'NeIII-3867', 'NeVI-3426', 'NeV-3346']]
    line_ratios['QSO-Narrow-lines'] = [0.2, 1.6, 1.6/2.98, 0.1, 0.1, 0.01, 0.5, 0.2, 0.02]

    # redder lines
    line_wavelengths['QSO-Narrow-lines'] += line_wavelengths['SIII']
    line_ratios['QSO-Narrow-lines'] += [lr*0.05 for lr in line_ratios['SIII']]
    line_wavelengths['QSO-Narrow-lines'] += line_wavelengths['HeI-1083']
    line_ratios['QSO-Narrow-lines'] += [0.2]

    line_wavelengths['Lya+CIV'] = [1215.4, 1549.49]
    line_ratios['Lya+CIV'] = [1., 0.1]

    line_wavelengths['Gal-UV-lines'] = [line_wavelengths[k][0] for k in ['Lya', 'CIV-1549', 'CIII-1906', 'CIII-1908', 'OIII-1663', 'HeII-1640', 'SiIV+OIV-1398', 'NV-1240', 'NIII-1750', 'MgII']]
    line_ratios['Gal-UV-lines'] = [1., 0.15, 0.1, 0.1, 0.008, 0.09, 0.1, 0.05, 0.05, 0.1]

    line_wavelengths['Ha+SII'] = [6564.61, 6718.29, 6732.67]
    line_ratios['Ha+SII'] = [1., 1./10, 1./10]

    line_wavelengths['Ha+SII+SIII+He'] = [6564.61, 6718.29, 6732.67, 9068.6, 9530.6, 10830.]
    line_ratios['Ha+SII+SIII+He'] = [1., 1./10, 1./10, 1./20, 2.44/20, 1./25.]

    line_wavelengths['Ha+NII+SII+SIII+He'] = [6564.61, 6549.86, 6585.27, 6718.29, 6732.67, 9068.6, 9530.6, 10830.]
    line_ratios['Ha+NII+SII+SIII+He'] = [1., 1./(4.*4), 3./(4*4), 1./10, 1./10, 1./20, 2.44/20, 1./25.]

    line_wavelengths['Ha+NII+SII+SIII+He+PaB'] = [6564.61, 6549.86, 6585.27, 6718.29, 6732.67, 9068.6, 9530.6, 10830., 12821]
    line_ratios['Ha+NII+SII+SIII+He+PaB'] = [1., 1./(4.*4), 3./(4*4), 1./10, 1./10, 1./20, 2.44/20, 1./25., 1./10]

    line_wavelengths['Ha+NII+SII+SIII+He+PaB+PaG'] = [6564.61, 6549.86, 6585.27, 6718.29, 6732.67, 9068.6, 9530.6, 10830., 12821, 10941.1]
    line_ratios['Ha+NII+SII+SIII+He+PaB+PaG'] = [1., 1./(4.*4), 3./(4*4), 1./10, 1./10, 1./20, 2.44/20, 1./25., 1./10, 1./10/2.86]

    line_wavelengths['Ha+NII'] = [6564.61, 6549.86, 6585.27]
    n2ha = 1./3  # log NII/Ha ~ -0.6, Kewley 2013
    line_ratios['Ha+NII'] = [1., 1./4.*n2ha, 3./4.*n2ha]

    line_wavelengths['OIII+Hb'] = [5008.240, 4960.295, 4862.68]
    line_ratios['OIII+Hb'] = [2.98, 1, 3.98/6.]

    # Include more balmer lines
    line_wavelengths['OIII+Hb+Hg+Hd'] = line_wavelengths['OIII'] + line_wavelengths['Balmer 10kK'][1:]
    line_ratios['OIII+Hb+Hg+Hd'] = line_ratios['OIII'] + line_ratios['Balmer 10kK'][1:]
    # o3hb = 1./6
    # for i in range(2, len(line_ratios['Balmer 10kK'])-1):
    #         line_ratios['OIII+Hb+Hg+Hd'][i] *= 3.98*o3hb
    # Compute as O3/Hb
    o3hb = 6
    for i in range(2):
        line_ratios['OIII+Hb+Hg+Hd'][i] *= 1./3.98*o3hb

    line_wavelengths['OIII+Hb+Ha'] = [5008.240, 4960.295, 4862.68, 6564.61]
    line_ratios['OIII+Hb+Ha'] = [2.98, 1, 3.98/10., 3.98/10.*2.86]

    line_wavelengths['OIII+Hb+Ha+SII'] = [5008.240, 4960.295, 4862.68, 6564.61, 6718.29, 6732.67]
    line_ratios['OIII+Hb+Ha+SII'] = [2.98, 1, 3.98/10., 3.98/10.*2.86*4, 3.98/10.*2.86/10.*4, 3.98/10.*2.86/10.*4]

    line_wavelengths['OIII+OII'] = [5008.240, 4960.295, 3729.875]
    line_ratios['OIII+OII'] = [2.98, 1, 3.98/4.]

    line_wavelengths['OII+Ne'] = [3729.875, 3869]
    line_ratios['OII+Ne'] = [1, 1./5]

    # Groups of all lines
    line_wavelengths['full'] = [w for w in line_wavelengths['Balmer 10kK']]
    line_ratios['full'] = [w for w in line_ratios['Balmer 10kK']]

    line_wavelengths['full'] += line_wavelengths['NII']
    line_ratios['full'] += [1./5/3.*line_ratios['Balmer 10kK'][1]*r for r in line_ratios['NII']]

    line_wavelengths['full'] += line_wavelengths['SII']
    line_ratios['full'] += [1./3.8/2*line_ratios['Balmer 10kK'][1]*r for r in line_ratios['SII']]

    # Lines from Hagele 2006, low-Z HII galaxies
    # SDSS J002101.03+005248.1
    line_wavelengths['full'] += line_wavelengths['SIII']
    line_ratios['full'] += [401./1000/2.44*line_ratios['Balmer 10kK'][1]*r for r in line_ratios['SIII']]

    # HeI
    line_wavelengths['full'] += line_wavelengths['HeI-series']
    he5877_hb = 127./1000/line_ratios['HeI-series'][1]
    line_ratios['full'] += [he5877_hb*r for r in line_ratios['HeI-series']]

    # NeIII
    line_wavelengths['full'] += line_wavelengths['NeIII-3867']
    line_ratios['full'] += [388./1000 for r in line_ratios['NeIII-3867']]

    line_wavelengths['full'] += line_wavelengths['NeIII-3968']
    line_ratios['full'] += [290./1000 for r in line_ratios['NeIII-3968']]

    # Add UV lines: MgII/Hb = 3
    line_wavelengths['full'] += line_wavelengths['Gal-UV-lines']
    line_ratios['full'] += [r*3/line_ratios['Gal-UV-lines'][-1] for r in line_ratios['Gal-UV-lines']]

    # High O32 - low metallicity
    o32, r23 = 4, 8
    o3_hb = r23/(1+1/o32)

    line_wavelengths['highO32'] = [w for w in line_wavelengths['full']]
    line_ratios['highO32'] = [r for r in line_ratios['full']]

    line_wavelengths['highO32'] += line_wavelengths['OIII']
    line_ratios['highO32'] += [r*o3_hb/3.98 for r in line_ratios['OIII']]

    line_wavelengths['highO32'] += line_wavelengths['OII']
    line_ratios['highO32'] += [r*o3_hb/2/o32 for r in line_ratios['OII']]

    # Low O32 - low metallicity
    o32, r23 = 0.3, 4
    o3_hb = r23/(1+1/o32)

    line_wavelengths['lowO32'] = [w for w in line_wavelengths['full']]
    line_ratios['lowO32'] = [r for r in line_ratios['full']]

    line_wavelengths['lowO32'] += line_wavelengths['OIII']
    line_ratios['lowO32'] += [r*o3_hb/3.98 for r in line_ratios['OIII']]

    line_wavelengths['lowO32'] += line_wavelengths['OII']
    line_ratios['lowO32'] += [r*o3_hb/2/o32 for r in line_ratios['OII']]

    return line_wavelengths, line_ratios


def emission_line_templates():
    """
    Testing FSPS line templates
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from grizli import utils
    import fsps
    sp = fsps.StellarPopulation(imf_type=1, zcontinuous=1)

    sp_params = {}

    sp_params['starburst'] = {'sfh': 4, 'tau': 0.3, 'tage': 0.1,
          'logzsol': -1, 'gas_logz': -1,
          'gas_logu': -2.5}

    sp_params['mature'] = {'sfh': 4, 'tau': 0.2, 'tage': 0.9,
        'logzsol': -0.2, 'gas_logz': -0.2,
        'gas_logu': -2.5}

    line_templates = {}

    for t in sp_params:
        pset = sp_params[t]
        header = 'wave flux\n\n'
        for p in pset:
            header += '{0} = {1}\n'.format(p, pset[p])
            if p == 'tage':
                continue

            print(p, pset[p])
            sp.params[p] = pset[p]

        spec = {}
        for neb in [True, False]:
            sp.params['add_neb_emission'] = neb
            sp.params['add_neb_continuum'] = neb
            wave, spec[neb] = sp.get_spectrum(tage=pset['tage'], peraa=True)
            #plt.plot(wave, spec[neb], alpha=0.5)

        neb_only = spec[True] - spec[False]
        neb_only = neb_only / neb_only.max()
        neb_only = spec[True] / spec[True].max()

        plt.plot(wave, neb_only, label=t, alpha=0.5)

        neb_only[neb_only < 1.e-4] = 0

        np.savetxt('fsps_{0}_lines.txt'.format(t), np.array([wave, neb_only]).T, fmt='%.5e', header=header)

        line_templates[t] = utils.SpectrumTemplate(wave=wave, flux=neb_only, name='fsps_{0}_lines'.format(t))


def pah33(wave_grid):
    """
    Set of 3.3 um PAH lines from Li et al. 2020
    
    Returns
    -------
    pah_templates : list
        List of `~grizli.utils.SpectrumTemplate` templates for three components
        around 3.3 um
    
    """
    pah_templates = {}
    for lc, lw in zip([3.29, 3.40, 3.47], [0.043, 0.031, 0.100]):
        ti = pah_line_template(wave_grid, center_um=lc, fwhm=lw)
        pah_templates[ti.name] = ti
    
    return pah_templates


def pah_line_template(wave_grid, center_um=3.29, fwhm=0.043):
    """
    Make a template for a broad PAH line with a Drude profile
    
    Default parameters in Lai et al. 2020
    https://iopscience.iop.org/article/10.3847/1538-4357/abc002/pdf
    from Tokunaga et al. 1991
    
    Drude equation and normalization from Yamada et al. 2013
    
    Parameters
    ----------
    wave_grid : array-like
        Wavelength grid in angstroms
    
    center_um : float
        Central wavelength in microns
    
    fwhm : float
        Drude profile FWHM in microns
    
    Returns
    -------
    pah_templ : `~grizli.utils.SpectrumTemplate`
        Template with the PAH feature
    
    """
    br = 1.
    gamma_width = fwhm/center_um
    Iv = br*gamma_width**2
    Iv /= ((wave_grid/1.e4/center_um - center_um*1.e4/wave_grid)**2
           + gamma_width**2)

    Inorm = np.pi*2.99e14/2.*br*gamma_width/center_um
    Iv *= 1 / Inorm
    
    # Flambda
    Ilam = Iv * 2.99e18 / (wave_grid)**2
    
    pah_templ = SpectrumTemplate(wave=wave_grid,
                                 flux=Ilam, 
                                 name=f'line PAH-{center_um:.2f}')
    return pah_templ


class SpectrumTemplate(object):
    def __init__(self, wave=None, flux=None, central_wave=None, fwhm=None, velocity=False, fluxunits=FLAMBDA_CGS, waveunits=u.angstrom, name='template', lorentz=False, err=None):
        """Container for template spectra.

        Parameters
        ----------
        wave : array-like
            Wavelength
            In `astropy.units.Angstrom`.

        flux : float array-like
            If float, then the integrated flux of a Gaussian line.  If
            array, then f-lambda flux density.

        central_wave, fwhm : float
            Initialize the template with a Gaussian at this wavelength (in
            `astropy.units.Angstrom`.) that has an integrated flux of `flux`
            and `fwhm` in `astropy.units.Angstrom` or `km/s` for
            `velocity=True`.

        velocity : bool
            `fwhm` is a velocity in `km/s`.

        Attributes
        ----------
        wave, flux : array-like
            Passed from the input parameters or generated/modified later.

        Methods
        -------
        __add__, __mul__ : Addition and multiplication of templates.

        Examples
        --------

            .. plot::
                :include-source:

                import matplotlib.pyplot as plt
                from grizli.utils import SpectrumTemplate

                ha = SpectrumTemplate(central_wave=6563., fwhm=10)
                plt.plot(ha.wave, ha.flux)

                ha_z = ha.zscale(0.1)
                plt.plot(ha_z.wave, ha_z.flux, label='z=0.1')

                plt.legend()
                plt.xlabel(r'$\lambda$')
                plt.xlim(6000, 7500)
                
                plt.show()

        """
        self.wave = wave
        if wave is not None:
            self.wave = np.cast[np.float64](wave)

        self.flux = flux
        if flux is not None:
            self.flux = np.cast[np.float64](flux)

        if err is not None:
            self.err = np.cast[np.float64](err)
        else:
            self.err = None

        self.fwhm = None
        self.velocity = None

        self.fluxunits = fluxunits
        self.waveunits = waveunits
        self.name = name

        if (central_wave is not None) & (fwhm is not None):
            self.fwhm = fwhm
            self.velocity = velocity

            self.wave, self.flux = self.make_gaussian(central_wave, fwhm,
                                                      wave_grid=wave,
                                                      velocity=velocity,
                                                      max_sigma=50,
                                                      lorentz=lorentz)

        self.fnu_units = FNU_CGS
        self.to_fnu()

    @staticmethod
    def make_gaussian(central_wave, fwhm, max_sigma=5, step=0.1,
                      wave_grid=None, velocity=False, clip=1.e-6,
                      lorentz=False):
        """Make Gaussian template

        Parameters
        ----------
        central_wave, fwhm : None or float or array-like
            Central wavelength and FWHM of the desired Gaussian

        velocity : bool
            `fwhm` is a velocity.

        max_sigma, step : float
            Generated wavelength array is

                >>> rms = fwhm/2.35
                >>> xgauss = np.arange(-max_sigma, max_sigma, step)*rms+central_wave

        clip : float
            Clip values where the value of the gaussian function is less than
            `clip` times its maximum (i.e., `1/sqrt(2*pi*sigma**2)`).

        lorentz : bool
            Make a Lorentzian line instead of a Gaussian.

        Returns
        -------
        wave, flux : array-like
            Wavelength and flux of a Gaussian line
        """
        import astropy.constants as const
        from astropy.modeling.models import Lorentz1D

        if hasattr(fwhm, 'unit'):
            rms = fwhm.value/2.35
            velocity = u.physical.get_physical_type(fwhm.unit) == 'speed'
            if velocity:
                rms = central_wave*(fwhm/const.c.to(KMS)).value/2.35
            else:
                rms = fwhm.value/2.35
        else:
            if velocity:
                rms = central_wave*(fwhm/const.c.to(KMS).value)/2.35
            else:
                rms = fwhm/2.35

        if wave_grid is None:
            #print('xxx line', central_wave, max_sigma, rms)

            wave_grid = np.arange(-max_sigma, max_sigma, step)*rms
            wave_grid += central_wave
            wave_grid = np.hstack([91., wave_grid, 1.e8])

        if lorentz:
            if velocity:
                use_fwhm = central_wave*(fwhm/const.c.to(KMS).value)
            else:
                use_fwhm = fwhm

            lmodel = Lorentz1D(amplitude=1, x_0=central_wave, fwhm=use_fwhm)
            line = lmodel(wave_grid)
            line[0:2] = 0
            line[-2:] = 0
            line /= np.trapz(line, wave_grid)
            peak = line.max()
        else:
            # Gaussian
            line = np.exp(-(wave_grid-central_wave)**2/2/rms**2)
            peak = np.sqrt(2*np.pi*rms**2)
            line *= 1./peak  # np.sqrt(2*np.pi*rms**2)

        line[line < 1./peak*clip] = 0

        return wave_grid, line

        #self.wave = xgauss
        #self.flux = gaussian

    def zscale(self, z, scalar=1, apply_igm=True):
        """Redshift the template and multiply by a scalar.

        Parameters
        ----------
        z : float
            Redshift to use.

        scalar : float
            Multiplicative factor.  Additional factor of 1./(1+z) is implicit.

        Returns
        -------
        new_spectrum : `~grizli.utils.SpectrumTemplate`
            Redshifted and scaled spectrum.

        """
        if apply_igm:
            try:
                import eazy.igm
                igm = eazy.igm.Inoue14()
                igmz = igm.full_IGM(z, self.wave*(1+z))
            except:
                igmz = 1.
        else:
            igmz = 1.

        return SpectrumTemplate(wave=self.wave*(1+z),
                                flux=self.flux*scalar/(1+z)*igmz)


    def __add__(self, spectrum):
        """Add two templates together

        The new wavelength array is the union of both input spectra and each
        input spectrum is linearly interpolated to the final grid.

        Parameters
        ----------
        spectrum : `~grizli.utils.SpectrumTemplate`

        Returns
        -------
        new_spectrum : `~grizli.utils.SpectrumTemplate`
        """
        new_wave = np.unique(np.append(self.wave, spectrum.wave))
        new_wave.sort()

        new_flux = np.interp(new_wave, self.wave, self.flux)
        new_flux += np.interp(new_wave, spectrum.wave, spectrum.flux)
        out = SpectrumTemplate(wave=new_wave, flux=new_flux)
        out.fwhm = spectrum.fwhm
        return out


    def __mul__(self, scalar):
        """Multiply spectrum by a scalar value

        Parameters
        ----------
        scalar : float
            Factor to multipy to `self.flux`.

        Returns
        -------
        new_spectrum : `~grizli.utils.SpectrumTemplate`
        """
        out = SpectrumTemplate(wave=self.wave, flux=self.flux*scalar)
        out.fwhm = self.fwhm
        return out


    def to_fnu(self, fnu_units=FNU_CGS):
        """Make fnu version of the template.

        Sets the `flux_fnu` attribute, assuming that the wavelength is given
        in Angstrom and the flux is given in flambda:

            >>> flux_fnu = self.flux * self.wave**2 / 3.e18

        """
        #import astropy.constants as const
        #flux_fnu = self.flux * self.wave**2 / 3.e18
        # flux_fnu = (self.flux*self.fluxunits*(self.wave*self.waveunits)**2/const.c).to(FNU_CGS) #,

        if ((FNU_CGS.__str__() == 'erg / (cm2 Hz s)') &
             (self.fluxunits.__str__() == 'erg / (Angstrom cm2 s)')):
            # Faster
            flux_fnu = self.flux*self.wave**2/2.99792458e18*fnu_units
            if self.err is not None:
                err_fnu = self.err*self.wave**2/2.99792458e18*fnu_units
        else:
            # Use astropy conversion
            flux_fnu = (self.flux*self.fluxunits).to(fnu_units, equivalencies=u.spectral_density(self.wave*self.waveunits))
            if self.err is not None:
                err_fnu = (self.err*self.fluxunits).to(fnu_units, equivalencies=u.spectral_density(self.wave*self.waveunits))

        self.fnu_units = fnu_units
        self.flux_fnu = flux_fnu.value
        if self.err is not None:
            self.err_fnu = err_fnu.value
        else:
            self.err_fnu = None


    def integrate_filter(self, filter, abmag=False, use_wave='filter'):
        """Integrate the template through an `~eazy.FilterDefinition` filter
        object.

        Parameters
        ----------
        filter : `~pysynphot.ObsBandpass`
            Or any object that has `wave` and `throughput` attributes, with
            the former in the same units as the input spectrum.

        abmag : bool
            Return AB magnitude rather than fnu flux

        Returns
        -------
        temp_flux : float

        Examples
        --------
        Compute the WFC3/IR F140W AB magnitude of a pure emission line at the
        5-sigma 3D-HST line detection limit (5e-17 erg/s/cm2):

        >>> import numpy as np
        >>> from grizli.utils import SpectrumTemplate
        >>> from eazy.filters import FilterDefinition
        >>> import pysynphot as S
        >>> line = SpectrumTemplate(central_wave=1.4e4, fwhm=150.,
                                    velocity=True)*5.e-17
        >>> filter = FilterDefinition(bp=S.ObsBandpass('wfc3,ir,f140w'))
        >>> fnu = line.integrate_filter(filter)
        >>> print('AB mag = {0:.3f}'.format(-2.5*np.log10(fnu)-48.6))
        AB mag = 26.619

        """
        INTEGRATOR = np.trapz

        try:
            #import grizli.utils_c
            #interp = grizli.utils_c.interp.interp_conserve_c
            from .utils_c.interp import interp_conserve_c
            interp = interp_conserve_c
        except ImportError:
            interp = np.interp

        #wz = self.wave*(1+z)
        nonzero = filter.throughput > 0
        if (filter.wave[nonzero].min() > self.wave.max()) | (filter.wave[nonzero].max() < self.wave.min()) | (filter.wave[nonzero].min() < self.wave.min()):
            if self.err is None:
                return 0.
            else:
                return 0., 0.

        if use_wave == 'filter':
            # Interpolate to filter wavelengths
            integrate_wave = filter.wave

            integrate_templ = interp(filter.wave.astype(np.float64), 
                                    self.wave, self.flux_fnu, left=0, right=0)

            if self.err is not None:
                templ_ivar = 1./interp(filter.wave.astype(np.float64),
                                       self.wave, self.err_fnu)**2

                templ_ivar[~np.isfinite(templ_ivar)] = 0

                integrate_weight = filter.throughput/filter.wave*templ_ivar/filter.norm
            else:
                integrate_weight = filter.throughput/filter.wave
        else:
            # Interpolate to spectrum wavelengths
            integrate_wave = self.wave
            integrate_templ = self.flux_fnu

            # test = nonzero
            test = np.isfinite(filter.throughput)
            interp_thru = interp(integrate_wave, filter.wave[test],
                                  filter.throughput[test],
                                  left=0, right=0)

            if self.err is not None:
                templ_ivar = 1/self.err_fnu**2
                templ_ivar[~np.isfinite(templ_ivar)] = 0

                integrate_weight = interp_thru/integrate_wave*templ_ivar/filter.norm
            else:
                integrate_weight = interp_thru/integrate_wave  # /templ_err**2

        if hasattr(filter, 'norm') & (self.err is None):
            filter_norm = filter.norm
        else:
            # e.g., pysynphot bandpass
            filter_norm = INTEGRATOR(integrate_weight, integrate_wave)

        # f_nu/lam dlam == f_nu d (ln nu)
        temp_flux = INTEGRATOR(integrate_templ*integrate_weight, integrate_wave) / filter_norm

        if self.err is not None:
            temp_err = 1/np.sqrt(filter_norm)

        if abmag:
            temp_mag = -2.5*np.log10(temp_flux)-48.6
            return temp_mag
        else:
            if self.err is not None:
                return temp_flux, temp_err
            else:
                return temp_flux


def load_templates(fwhm=400, line_complexes=True, stars=False,
                   full_line_list=DEFAULT_LINE_LIST, continuum_list=None,
                   fsps_templates=False, alf_template=False, lorentz=False):
    """Generate a list of templates for fitting to the grism spectra

    The different sets of continuum templates are stored in

        >>> temp_dir = os.path.join(GRIZLI_PATH, 'templates')

    Parameters
    ----------
    fwhm : float
        FWHM of a Gaussian, in km/s, that is convolved with the emission
        line templates.  If too narrow, then can see pixel effects in the
        fits as a function of redshift.

    line_complexes : bool
        Generate line complex templates with fixed flux ratios rather than
        individual lines. This is useful for the redshift fits where there
        would be redshift degeneracies if the line fluxes for individual
        lines were allowed to vary completely freely. See the list of
        available lines and line groups in
        `~grizli.utils.get_line_wavelengths`. Currently,
        `line_complexes=True` generates the following groups:

            Ha+NII+SII+SIII+He
            OIII+Hb
            OII+Ne

    stars : bool
        Get stellar templates rather than galaxies + lines

    full_line_list : None or list
        Full set of lines to try.  The default is set in the global variable
        `~grizli.utils.DEFAULT_LINE_LIST`.

        The full list of implemented lines is in `~grizli.utils.get_line_wavelengths`.

    continuum_list : None or list
        Override the default continuum templates if None.

    fsps_templates : bool
        If True, get the FSPS NMF templates.

    Returns
    -------
    temp_list : dictionary of `~grizli.utils.SpectrumTemplate` objects
        Output template list

    """

    if stars:
        # templates = glob.glob('%s/templates/Pickles_stars/ext/*dat' %(GRIZLI_PATH))
        # templates = []
        # for t in 'obafgkmrw':
        #     templates.extend( glob.glob('%s/templates/Pickles_stars/ext/uk%s*dat' %(os.getenv('THREEDHST'), t)))
        # templates.extend(glob.glob('%s/templates/SPEX/spex-prism-M*txt' %(os.getenv('THREEDHST'))))
        # templates.extend(glob.glob('%s/templates/SPEX/spex-prism-[LT]*txt' %(os.getenv('THREEDHST'))))
        #
        # #templates = glob.glob('/Users/brammer/Downloads/templates/spex*txt')
        # templates = glob.glob('bpgs/*ascii')
        # info = catIO.Table('bpgs/bpgs.info')
        # type = np.array([t[:2] for t in info['type']])
        # templates = []
        # for t in 'OBAFGKM':
        #     test = type == '-%s' %(t)
        #     so = np.argsort(info['type'][test])
        #     templates.extend(info['file'][test][so])
        #
        # temp_list = OrderedDict()
        # for temp in templates:
        #     #data = np.loadtxt('bpgs/'+temp, unpack=True)
        #     data = np.loadtxt(temp, unpack=True)
        #     #data[0] *= 1.e4 # spex
        #     scl = np.interp(5500., data[0], data[1])
        #     name = os.path.basename(temp)
        #     #ix = info['file'] == temp
        #     #name='%5s %s' %(info['type'][ix][0][1:], temp.split('.as')[0])
        #     print(name)
        #     temp_list[name] = utils.SpectrumTemplate(wave=data[0],
        #                                              flux=data[1]/scl)

        # np.save('stars_bpgs.npy', [temp_list])

        # tall = np.load(os.path.join(GRIZLI_PATH,
        #                                  'templates/stars.npy'))[0]
        #
        # return tall
        #
        # temp_list = OrderedDict()
        # for k in tall:
        #     if k.startswith('uk'):
        #         temp_list[k] = tall[k]
        #
        # return temp_list
        #
        # for t in 'MLT':
        #     for k in tall:
        #         if k.startswith('spex-prism-'+t):
        #             temp_list[k] = tall[k]
        #
        # return temp_list

        # return temp_list
        templates = ['M6.5.txt', 'M8.0.txt', 'L1.0.txt', 'L3.5.txt', 'L6.0.txt', 'T2.0.txt', 'T6.0.txt', 'T7.5.txt']
        templates = ['stars/'+t for t in templates]
    else:
        # Intermediate and very old
        # templates = ['templates/EAZY_v1.0_lines/eazy_v1.0_sed3_nolines.dat',
        #              'templates/cvd12_t11_solar_Chabrier.extend.skip10.dat']
        templates = ['eazy_intermediate.dat',
                     'cvd12_t11_solar_Chabrier.dat']

        # Post starburst
        # templates.append('templates/UltraVISTA/eazy_v1.1_sed9.dat')
        templates.append('post_starburst.dat')

        # Very blue continuum
        # templates.append('templates/YoungSB/erb2010_continuum.dat')
        templates.append('erb2010_continuum.dat')

        # Test new templates
        # templates = ['templates/erb2010_continuum.dat',
        # 'templates/fsps/tweak_fsps_temp_kc13_12_006.dat',
        # 'templates/fsps/tweak_fsps_temp_kc13_12_008.dat']

        if fsps_templates:
            #templates = ['templates/fsps/tweak_fsps_temp_kc13_12_0{0:02d}.dat'.format(i+1) for i in range(12)]
            templates = ['fsps/fsps_QSF_12_v3_nolines_0{0:02d}.dat'.format(i+1) for i in range(12)]
            #templates = ['fsps/fsps_QSF_7_v3_nolines_0{0:02d}.dat'.format(i+1) for i in range(7)]

        if alf_template:
            templates.append('alf_SSP.dat')

        if continuum_list is not None:
            templates = continuum_list

    temp_list = OrderedDict()
    for temp in templates:
        data = np.loadtxt(os.path.join(GRIZLI_PATH, 'templates', temp), unpack=True)
        #scl = np.interp(5500., data[0], data[1])
        scl = 1.
        name = temp  # os.path.basename(temp)
        temp_list[name] = SpectrumTemplate(wave=data[0], flux=data[1]/scl,
                                           name=name)

        temp_list[name].name = name

    if stars:
        return temp_list

    # Emission lines:
    line_wavelengths, line_ratios = get_line_wavelengths()

    if line_complexes:
        #line_list = ['Ha+SII', 'OIII+Hb+Ha', 'OII']
        #line_list = ['Ha+SII', 'OIII+Hb', 'OII']
        #line_list = ['Ha+NII+SII+SIII+He+PaB', 'OIII+Hb', 'OII+Ne', 'Lya+CIV']
        #line_list = ['Ha+NII+SII+SIII+He+PaB', 'OIII+Hb+Hg+Hd', 'OII+Ne', 'Lya+CIV']
        line_list = ['Ha+NII+SII+SIII+He+PaB', 'OIII+Hb+Hg+Hd', 'OII+Ne', 'Gal-UV-lines']

    else:
        if full_line_list is None:
            line_list = DEFAULT_LINE_LIST
        else:
            line_list = full_line_list

        #line_list = ['Ha', 'SII']

    # Use FSPS grid for lines
    wave_grid = None
    # if fsps_templates:
    #     wave_grid = data[0]
    # else:
    #     wave_grid = None

    for li in line_list:
        scl = line_ratios[li]/np.sum(line_ratios[li])
        for i in range(len(scl)):
            if ('O32' in li) & (np.abs(line_wavelengths[li][i]-2799) < 2):
                fwhm_i = 2500
                lorentz_i = True
            else:
                fwhm_i = fwhm
                lorentz_i = lorentz

            line_i = SpectrumTemplate(wave=wave_grid,
                                      central_wave=line_wavelengths[li][i],
                                      flux=None, fwhm=fwhm_i, velocity=True,
                                      lorentz=lorentz_i)

            if i == 0:
                line_temp = line_i*scl[i]
            else:
                line_temp = line_temp + line_i*scl[i]

        name = 'line {0}'.format(li)
        line_temp.name = name
        temp_list[name] = line_temp

    return temp_list


def load_beta_templates(wave=np.arange(400, 2.5e4), betas=[-2, -1, 0]):
    """
    Step-function templates with f_lambda ~ (wave/1216.)**beta
    """
    cont_wave = np.arange(400, 2.5e4)
    t0 = {}
    for beta in betas:
        key = 'beta {0}'.format(beta)
        t0[key] = SpectrumTemplate(wave=cont_wave, flux=(cont_wave/1216.)**beta)
    return t0


def load_quasar_templates(broad_fwhm=2500, narrow_fwhm=1200, broad_lines=['HeI-5877', 'MgII', 'Lya', 'CIV-1549', 'CIII-1906', 'CIII-1908', 'OIII-1663', 'HeII-1640', 'SiIV+OIV-1398', 'NIV-1487', 'NV-1240', 'PaB', 'PaG'], narrow_lines=['NIII-1750', 'OII', 'OIII', 'SII', 'OI-6302', 'OIII-4363', 'NeIII-3867', 'NeVI-3426', 'NeV-3346', 'OII-7325', 'ArIII-7138', 'SIII', 'HeI-1083'], include_feii=True, slopes=[-2.8, 0, 2.8], uv_line_complex=True, fixed_narrow_lines=False, t1_only=False, nspline=13, Rspline=30, betas=None, include_reddened_balmer_lines=False):
    """
    Make templates suitable for fitting broad-line quasars
    """

    from collections import OrderedDict
    import scipy.ndimage as nd

    t0 = OrderedDict()
    t1 = OrderedDict()

    broad1 = load_templates(fwhm=broad_fwhm, line_complexes=False, stars=False, full_line_list=['Ha', 'Hb', 'Hg', 'Hd', 'H7', 'H8', 'H9', 'H10'] + broad_lines, continuum_list=[], fsps_templates=False, alf_template=False, lorentz=True)

    narrow1 = load_templates(fwhm=400, line_complexes=False, stars=False, full_line_list=narrow_lines, continuum_list=[], fsps_templates=False, alf_template=False)

    if fixed_narrow_lines:
        if t1_only:
            narrow0 = narrow1
        else:
            narrow0 = load_templates(fwhm=narrow_fwhm, line_complexes=False, stars=False, full_line_list=['QSO-Narrow-lines'], continuum_list=[], fsps_templates=False, alf_template=False)

    else:
        narrow0 = load_templates(fwhm=narrow_fwhm, line_complexes=False, stars=False, full_line_list=narrow_lines, continuum_list=[], fsps_templates=False, alf_template=False)

    if t1_only:
        broad0 = broad1
    else:
        if uv_line_complex:
            full_line_list = ['Balmer 10kK + MgII Av=0.5', 'QSO-UV-lines']
            #broad0 = load_templates(fwhm=broad_fwhm, line_complexes=False, stars=False, full_line_list=['Balmer 10kK + MgII', 'QSO-UV-lines'], continuum_list=[], fsps_templates=False, alf_template=False, lorentz=True)
        else:
            full_line_list = ['Balmer 10kK + MgII Av=0.5']
            #broad0 = load_templates(fwhm=broad_fwhm, line_complexes=False, stars=False, full_line_list=['Balmer 10kK'] + broad_lines, continuum_list=[], fsps_templates=False, alf_template=False, lorentz=True)

        if include_reddened_balmer_lines:
            line_wavelengths, line_ratios = get_line_wavelengths()
            if 'Balmer 10kK + MgII Av=1.0' in line_wavelengths:
                full_line_list += ['Balmer 10kK + MgII']
                full_line_list += ['Balmer 10kK + MgII Av=1.0']
                full_line_list += ['Balmer 10kK + MgII Av=2.0']

        broad0 = load_templates(fwhm=broad_fwhm, line_complexes=False, stars=False, full_line_list=full_line_list, continuum_list=[], fsps_templates=False, alf_template=False, lorentz=True)

        for k in broad0:
            t0[k] = broad0[k]

    for k in broad1:
        t1[k] = broad1[k]

    for k in narrow0:
        t0[k] = narrow0[k]

    for k in narrow1:
        t1[k] = narrow1[k]

    # Fe II
    if include_feii:
        feii_wave, feii_flux = np.loadtxt(os.path.dirname(__file__) + '/data/templates/FeII_VeronCetty2004.txt', unpack=True)

        # smoothing, in units of input velocity resolution
        feii_kern = broad_fwhm/2.3548/75.
        feii_sm = nd.gaussian_filter(feii_flux, feii_kern)
        t0['FeII-VC2004'] = t1['FeII-VC2004'] = SpectrumTemplate(wave=feii_wave, flux=feii_sm, name='FeII-VC2004')

    # Linear continua
    # cont_wave = np.arange(400, 2.5e4)
    # for slope in slopes:
    #     key = 'slope {0}'.format(slope)
    #     t0[key] = t1[key] = SpectrumTemplate(wave=cont_wave, flux=(cont_wave/6563.)**slope)

    if Rspline is not None:
        wspline = np.arange(4200, 2.5e4, 10)
        df_spl = log_zgrid(zr=[wspline[0], wspline[-1]], dz=1./Rspline)
        bsplines = bspline_templates(wspline, df=len(df_spl)+2, log=True,
                                     clip=0.0001)

        for key in bsplines:
            t0[key] = t1[key] = bsplines[key]

    elif nspline > 0:
        # Spline continua
        cont_wave = np.arange(5000, 2.4e4)
        bsplines = bspline_templates(cont_wave, df=nspline, log=True)
        for key in bsplines:
            t0[key] = t1[key] = bsplines[key]

    elif betas is not None:
        btemp = load_beta_templates(wave=np.arange(400, 2.5e4), betas=betas)
        for key in btemp:
            t0[key] = t1[key] = btemp[key]

    else:
        # Observed frame steps
        onedR = -nspline
        wlim = [5000, 18000.0]
        bin_steps, step_templ = step_templates(wlim=wlim, R=onedR,
                                               round=10)
        for key in step_templ:
            t0[key] = t1[key] = step_templ[key]

    # t0['blue'] = t1['blue'] = SpectrumTemplate(wave=cont_wave, flux=(cont_wave/6563.)**-2.8)
    # t0['mid'] = t1['mid'] = SpectrumTemplate(wave=cont_wave, flux=(cont_wave/6563.)**0)
    # t0['red'] = t1['mid'] = SpectrumTemplate(wave=cont_wave, flux=(cont_wave/6563.)**2.8)

    return t0, t1


PHOENIX_LOGG_FULL = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
PHOENIX_LOGG = [4.0, 4.5, 5.0, 5.5]

PHOENIX_TEFF_FULL = [400.0, 420.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 1550.0, 1600.0, 1650.0, 1700.0, 1750.0, 1800.0, 1850.0, 1900.0, 1950.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0, 4300.0, 4400.0, 4500.0, 4600.0, 4700.0, 4800.0, 4900.0, 5000.0]

PHOENIX_TEFF = [400.,  420., 450., 500.,  550., 600.,  650., 700.,  750.,
       800.,  850., 900., 950., 1000., 1050., 1100., 1150., 1200.,
       1300., 1400., 1500., 1600., 1700., 1800., 1900., 2000., 2100.,
       2200., 2300., 2400., 2500., 2600., 2700., 2800., 2900., 3000.,
       3100., 3200., 3300., 3400., 3500., 3600., 3700., 3800., 3900., 4000.,
       4200., 4400., 4600., 4800., 5000., 5500., 5500, 6000., 6500., 7000.]

PHOENIX_ZMET_FULL = [-2.5, -2.0, -1.5, -1.0, -0.5, -0., 0.5]
PHOENIX_ZMET = [-1.0, -0.5, -0.]


def load_phoenix_stars(logg_list=PHOENIX_LOGG, teff_list=PHOENIX_TEFF, zmet_list=PHOENIX_ZMET, add_carbon_star=True, file='bt-settl_t400-7000_g4.5.fits'):
    """
    Load Phoenix stellar templates
    """
    from collections import OrderedDict
    try:
        from urllib.request import urlretrieve
    except:
        from urllib import urlretrieve

    # file='bt-settl_t400-5000_g4.5.fits'
    # file='bt-settl_t400-3500_z0.0.fits'

    try:
        hdu = pyfits.open(os.path.join(GRIZLI_PATH, 'templates/stars/', file))
    except:
        #url = 'https://s3.amazonaws.com/grizli/CONF'
        #url = 'https://erda.ku.dk/vgrid/Gabriel%20Brammer/CONF'
        url = ('https://raw.githubusercontent.com/gbrammer/' +
               'grizli-config/master')

        print('Fetch {0}/{1}'.format(url, file))

        #os.system('wget -O /tmp/{1} {0}/{1}'.format(url, file))
        res = urlretrieve('{0}/{1}'.format(url, file),
                          filename=os.path.join('/tmp', file))

        hdu = pyfits.open(os.path.join('/tmp/', file))

    tab = GTable.gread(hdu[1])
    hdu.close()
    
    tstars = OrderedDict()
    N = tab['flux'].shape[1]
    for i in range(N):
        teff = tab.meta['TEFF{0:03d}'.format(i)]
        logg = tab.meta['LOGG{0:03d}'.format(i)]
        try:
            met = tab.meta['ZMET{0:03d}'.format(i)]
        except:
            met = 0.

        if (logg not in logg_list) | (teff not in teff_list) | (met not in zmet_list):
            #print('Skip {0} {1}'.format(logg, teff))
            continue

        label = 'bt-settl_t{0:05.0f}_g{1:3.1f}_m{2:.1f}'.format(teff, logg, met)

        tstars[label] = SpectrumTemplate(wave=tab['wave'],
                                         flux=tab['flux'][:, i], name=label)

    if add_carbon_star:
        cfile = os.path.join(GRIZLI_PATH, 'templates/stars/carbon_star.txt')
        sp = read_catalog(cfile)
        if add_carbon_star > 1:
            import scipy.ndimage as nd
            cflux = nd.gaussian_filter(sp['flux'], add_carbon_star)
        else:
            cflux = sp['flux']

        tstars['bt-settl_t05000_g0.0_m0.0'] = SpectrumTemplate(wave=sp['wave'], flux=cflux, name='carbon-lancon2002')

    return tstars


def load_sdss_pca_templates(file='spEigenQSO-55732.fits', smooth=3000):
    """
    Load SDSS eigen templates
    """
    from collections import OrderedDict
    import scipy.ndimage as nd

    im = pyfits.open(os.path.join(GRIZLI_PATH, 'templates', file))
    h = im[0].header
    log_wave = np.arange(h['NAXIS1'])*h['COEFF1']+h['COEFF0']
    wave = 10**log_wave

    name = file.split('.fits')[0]

    if smooth > 0:
        dv_in = h['COEFF1']*3.e5
        n = smooth / dv_in
        data = nd.gaussian_filter1d(im[0].data, n, axis=1).astype(np.float64)
        skip = int(n/2.5)
        wave = wave[::skip]
        data = data[:, ::skip]
    else:
        data = im[0].data.astype(np.float64)

    N = h['NAXIS2']
    temp_list = OrderedDict()
    for i in range(N):
        temp_list['{0} {1}'.format(name, i+1)] = SpectrumTemplate(wave=wave, flux=data[i, :])
    
    im.close()
    
    return temp_list


def cheb_templates(wave, order=6, get_matrix=False, log=False, clip=1.e-4, minmax=None):
    """
    Chebyshev polynomial basis functions
    """
    from numpy.polynomial.chebyshev import chebval, chebvander

    if minmax is None:
        mi = wave.min()
        ma = wave.max()
    else:
        mi, ma = np.squeeze(minmax)*1.

    if log:
        xi = np.log(wave)
        mi = np.log(mi)
        ma = np.log(ma)
    else:
        xi = wave*1
    
    x = (xi-mi)*2/(ma-mi)-1

    n_bases = order+1
    
    basis = chebvander(x, order)
    # basis = np.empty((x.shape[0], n_bases), dtype=float)
    # 
    # xr = np.arange(n_bases)
    # for i in range(n_bases):
    #     _c = (xr == i)*1
    #     #print(_c, xr, i)
    #     basis[:,i] = chebval(x, _c)
        
    #for i in range(n_bases):
    out_of_range = (xi < mi) | (xi > ma)
    basis[out_of_range,:] = 0

    if get_matrix:
        return basis

    temp = OrderedDict()
    for i in range(n_bases):
        key = f'cheb o{i}'
        temp[key] = SpectrumTemplate(wave, basis[:,i])
        temp[key].name = key

    return temp
    
def bspline_templates(wave, degree=3, df=6, get_matrix=False, log=False, clip=1.e-4, minmax=None):
    """
    B-spline basis functions, modeled after `~patsy.splines`
    """
    from scipy.interpolate import splev

    order = degree+1
    n_inner_knots = df - order
    inner_knots = np.linspace(0, 1, n_inner_knots + 2)[1:-1]

    norm_knots = np.concatenate(([0, 1] * order,
                                inner_knots))
    norm_knots.sort()

    if log:
        xspl = np.log(wave)
    else:
        xspl = wave*1

    if minmax is None:
        mi = xspl.min()
        ma = xspl.max()
    else:
        mi, ma = minmax

    width = ma-mi
    all_knots = norm_knots*width+mi

    n_bases = len(all_knots) - (degree + 1)
    basis = np.empty((xspl.shape[0], n_bases), dtype=float)

    coefs = np.identity(n_bases)
    basis = splev(xspl, (all_knots, coefs, degree))

    for i in range(n_bases):
        out_of_range = (xspl < mi) | (xspl > ma)
        basis[i][out_of_range] = 0

    wave_peak = np.round(wave[np.argmax(basis, axis=1)])

    maxval = np.max(basis, axis=1)
    for i in range(n_bases):
        basis[i][basis[i] < clip*maxval[i]] = 0

    if get_matrix:
        return np.vstack(basis).T

    temp = OrderedDict()
    for i in range(n_bases):
        key = 'bspl {0} {1:.0f}'.format(i, wave_peak[i])
        temp[key] = SpectrumTemplate(wave, basis[i])
        temp[key].name = key
        temp[key].wave_peak = wave_peak[i]

    temp.knots = all_knots
    temp.degree = degree
    temp.xspl = xspl

    return temp


def eval_bspline_templates(wave, bspl, coefs):
    from scipy.interpolate import splev

    xspl = np.log(wave)
    basis = splev(xspl, (bspl.knots, coefs, bspl.degree))
    return np.array(basis)


def split_spline_template(templ, wavelength_range=[5000, 2.4e4], Rspline=10, log=True):
    """
    Multiply a single template by spline bases to effectively generate a
    spline multiplicative correction that can be fit with linear least
    squares.

    Parameters
    ----------

    templ : `~grizli.utils.SpectrumTemplate`
        Template to split.

    wavelength_range : [float, float]
        Limit the splines to this wavelength range

    Rspline : float
        Effective resolution, R=dlambda/lambda, of the spline correction
        function.

    log : bool
        Log-spaced splines

    Returns
    -------

    stemp : dict

        Dictionary of spline-component templates, with additional attributes:

            wspline = wavelength of the templates / spline correction
            tspline = matrix of the spline corrections
            knots   = peak wavelenghts of each spline component

    """
    from collections import OrderedDict
    from grizli import utils

    if False:
        stars = utils.load_templates(stars=True)
        templ = stars['stars/L1.0.txt']

    wspline = templ.wave

    clip = (wspline > wavelength_range[0]) & (wspline < wavelength_range[1])
    df_spl = len(utils.log_zgrid(zr=wavelength_range, dz=1./Rspline))

    tspline = utils.bspline_templates(wspline[clip], df=df_spl+2, log=log, clip=0.0001, get_matrix=True)

    ix = np.argmax(tspline, axis=0)
    knots = wspline[clip][ix]

    N = tspline.shape[1]
    stemp = OrderedDict()
    for i in range(N):
        name = '{0} {1:.2f}'.format(templ.name, knots[i]/1.e4)
        stemp[name] = utils.SpectrumTemplate(wave=wspline[clip], flux=templ.flux[clip]*tspline[:, i], name=name)
        stemp[name].knot = knots[i]

    stemp.wspline = wspline[clip]
    stemp.tspline = tspline
    stemp.knots = knots

    return stemp


def step_templates(wlim=[5000, 1.8e4], bin_steps=None, R=30, round=10, rest=False, special=None, order=0):
    """
    Step-function templates for easy binning
    """
    if special == 'Dn4000':
        rest = True
        bin_steps = np.hstack([np.arange(850, 3849, 100),
                              [3850, 3950, 4000, 4100],
                              np.arange(4200, 1.7e4, 100)])

    elif special == 'D4000':
        rest = True
        bin_steps = np.hstack([np.arange(850, 3749, 200),
                              [3750, 3950, 4050, 4250],
                              np.arange(4450, 1.7e4, 200)])
    elif special not in ['D4000', 'Dn4000', None]:
        print('step_templates: {0} not recognized (options are \'D4000\', \'Dn4000\', and None)'.format(special))
        return {}

    if bin_steps is None:
        bin_steps = np.round(log_zgrid(wlim, 1./R)/round)*round
    else:
        wlim = [bin_steps[0], bin_steps[-1]]

    ds = np.diff(bin_steps)

    xspec = np.arange(wlim[0]-ds[0], wlim[1]+ds[-1])

    bin_mid = bin_steps[:-1]+ds/2.

    step_templ = {}
    for i in range(len(bin_steps)-1):

        yspec = ((xspec >= bin_steps[i]) & (xspec < bin_steps[i+1]))*1

        for o in range(order+1):
            label = 'step {0:.0f}-{1:.0f} {2}'.format(bin_steps[i], bin_steps[i+1], o)
            if rest:
                label = 'r'+label

            flux = ((xspec-bin_mid[i])/ds[i])**o * (yspec > 0)
            step_templ[label] = SpectrumTemplate(wave=xspec, flux=flux,
                                                 name=label)

    return bin_steps, step_templ


def polynomial_templates(wave, ref_wave=1.e4, order=0, line=False):
    temp = OrderedDict()
    if line:
        for sign in [1, -1]:
            key = 'poly {0}'.format(sign)
            temp[key] = SpectrumTemplate(wave, sign*(wave/ref_wave-1)+1)
            temp[key].name = key

        return temp

    for i in range(order+1):
        key = 'poly {0}'.format(i)
        temp[key] = SpectrumTemplate(wave, (wave/ref_wave-1)**i)
        temp[key].name = key
        temp[key].ref_wave = ref_wave

    return temp


def split_poly_template(templ, ref_wave=1.e4, order=3):
    """
    Multiply a single template by polynomial bases to effectively generate a
    polynomial multiplicative correction that can be fit with linear least
    squares.

    Parameters
    ----------
    templ : `~grizli.utils.SpectrumTemplate`
        Template to split.

    ref_wave : float
       Wavelength where to normalize the polynomials.

    Order : int
        Polynomial order.  Returns order+1 templates.

    Returns
    -------
    ptemp : dict

        Dictionary of polynomial-component templates, with additional
        attributes:

            ref_wave = wavelength where polynomials normalized

    """
    from collections import OrderedDict
    from grizli import utils

    tspline = polynomial_templates(templ.wave, ref_wave=ref_wave,
                                   order=order, line=False)

    ptemp = OrderedDict()

    for i, t in enumerate(tspline):
        name = '{0} poly {1}'.format(templ.name, i)
        ptemp[name] = utils.SpectrumTemplate(wave=templ.wave,
                                 flux=templ.flux*tspline[t].flux,
                                 name=name)
        ptemp[name].ref_wave = ref_wave

    ptemp.ref_wave = ref_wave

    return ptemp


def dot_templates(coeffs, templates, z=0, max_R=5000, apply_igm=True):
    """Compute template sum analogous to `np.dot(coeffs, templates)`.
    """

    if len(coeffs) != len(templates):
        raise ValueError('shapes of coeffs ({0}) and templates ({1}) don\'t match'.format(len(coeffs), len(templates)))

    # for i, te in enumerate(templates):
    #     if i == 0:
    #         tc = templates[te].zscale(z, scalar=coeffs[i])
    #         tl = templates[te].zscale(z, scalar=coeffs[i])
    #     else:
    #         if te.startswith('line'):
    #             tc += templates[te].zscale(z, scalar=0.)
    #         else:
    #             tc += templates[te].zscale(z, scalar=coeffs[i])
    #
    #         tl += templates[te].zscale(z, scalar=coeffs[i])
    wave, flux_arr, is_line = array_templates(templates, max_R=max_R, z=z,
                                              apply_igm=apply_igm)

    # # IGM
    # if apply_igm:
    #     try:
    #         import eazy.igm
    #         IGM = eazy.igm.Inoue14()
    #
    #         lylim = wave < 1250
    #         igmz = np.ones_like(wave)
    #         igmz[lylim] = IGM.full_IGM(z, wave[lylim]*(1+z))
    #     except:
    #         igmz = 1.
    # else:
    #     igmz = 1.
    #
    # is_obsframe = np.array([t.split()[0] in ['bspl', 'step'] for t in templates])
    #
    # flux_arr[~is_obsframe,:] *= igmz
    #
    # # Multiply spline?
    # for i, t in enumerate(templates):
    #     if 'spline' in t:
    #         for j, tj in enumerate(templates):
    #             if is_obsframe[j]:
    #                 print('scale spline: {0} x {1}'.format(tj, t))
    #                 flux_arr[j,:] *= flux_arr[i,:]

    # Continuum
    cont = np.dot(coeffs*(~is_line), flux_arr)
    tc = SpectrumTemplate(wave=wave, flux=cont).zscale(z, apply_igm=False)

    # Full template
    line = np.dot(coeffs, flux_arr)
    tl = SpectrumTemplate(wave=wave, flux=line).zscale(z, apply_igm=False)

    return tc, tl


def array_templates(templates, wave=None, max_R=5000, z=0, apply_igm=False):
    """Return an array version of the templates that have all been interpolated to the same grid.


    Parameters
    ----------
    templates : dictionary of `~grizli.utils.SpectrumTemplate` objects
        Output template list with `NTEMP` templates.

    max_R : float
        Maximum spectral resolution of the regridded templates.

    z : float
        Redshift where to evaluate the templates.  But note that this is only
        used to shift templates produced by `bspline_templates`, which are
        defined in the observed frame.

    Returns
    -------
    wave : `~numpy.ndarray`, dimensions `(NL,)`
        Array containing unique wavelengths.

    flux_arr : `~numpy.ndarray`, dimensions `(NTEMP, NL)`
        Array containing the template fluxes interpolated at `wave`.

    is_line : `~numpy.ndarray`
        Boolean array indicating emission line templates (the key in the
        template dictionary starts with "line ").

    """
    from grizli.utils_c.interp import interp_conserve_c

    if wave is None:
        wstack = []
        for t in templates:
            if t.split()[0] in ['bspl', 'step', 'poly']:
                wstack.append(templates[t].wave/(1+z))
            else:
                wstack.append(templates[t].wave)

        wave = np.unique(np.hstack(wstack))

    clipsum, iter = 1, 0
    while (clipsum > 0) & (iter < 10):
        clip = np.gradient(wave)/wave < 1/max_R
        idx = np.arange(len(wave))[clip]
        wave[idx[::2]] = np.nan
        wave = wave[np.isfinite(wave)]
        iter += 1
        clipsum = clip.sum()
        #print(iter, clipsum)

    NTEMP = len(templates)
    flux_arr = np.zeros((NTEMP, len(wave)))

    for i, t in enumerate(templates):
        if t.split()[0] in ['bspl', 'step', 'poly']:
            flux_arr[i, :] = interp_conserve_c(wave, templates[t].wave/(1+z),
                                          templates[t].flux*(1+z))
        else:
            if hasattr(templates[t], 'flux_flam'):
                # Redshift-dependent eazy-py Template
                flux_arr[i, :] = interp_conserve_c(wave, templates[t].wave,
                                          templates[t].flux_flam(z=z))
            else:
                flux_arr[i, :] = interp_conserve_c(wave, templates[t].wave,
                                          templates[t].flux)

    is_line = np.array([t.startswith('line ') for t in templates])

    # IGM
    if apply_igm:
        try:
            import eazy.igm
            IGM = eazy.igm.Inoue14()

            lylim = wave < 1250
            igmz = np.ones_like(wave)
            igmz[lylim] = IGM.full_IGM(z, wave[lylim]*(1+z))
        except:
            igmz = 1.
    else:
        igmz = 1.

    obsnames = ['bspl', 'step', 'poly']
    is_obsframe = np.array([t.split()[0] in obsnames for t in templates])

    flux_arr[~is_obsframe, :] *= igmz

    # Multiply spline?
    for i, t in enumerate(templates):
        if 'spline' in t:
            for j, tj in enumerate(templates):
                if is_obsframe[j]:
                    ma = flux_arr[j, :].sum()
                    ma = ma if ma > 0 else 1
                    ma = 1

                    flux_arr[j, :] *= flux_arr[i, :]/ma

    return wave, flux_arr, is_line


def compute_equivalent_widths(templates, coeffs, covar, max_R=5000, Ndraw=1000, seed=0, z=0, observed_frame=False):
    """Compute template-fit emission line equivalent widths

    Parameters
    ----------
    templates : dictionary of `~grizli.utils.SpectrumTemplate` objects
        Output template list with `NTEMP` templates.

    coeffs : `~numpy.ndarray`, dimensions (`NTEMP`)
        Fit coefficients

    covar :  `~numpy.ndarray`, dimensions (`NTEMP`, `NTEMP`)
        Covariance matrix

    max_R : float
        Maximum spectral resolution of the regridded templates.

    Ndraw : int
        Number of random draws to extract from the covariance matrix

    seed : positive int
        Random number seed to produce repeatible results. If `None`, then
        use default state.

    z : float
        Redshift where the fit is evaluated

    observed_framme : bool
        If true, then computed EWs are observed frame, otherwise they are
        rest frame at redshift `z`.

    Returns
    -------
    EWdict : dict
        Dictionary of [16, 50, 84th] percentiles of the line EW distributions.

    """

    # Array versions of the templates
    wave, flux_arr, is_line = array_templates(templates, max_R=max_R, z=z)
    keys = np.array(list(templates.keys()))

    EWdict = OrderedDict()
    for key in keys[is_line]:
        EWdict[key] = (0., 0., 0.)

    # Only worry about templates with non-zero coefficients, which should
    # be accounted for in the covariance array (with get_uncertainties=2)
    clip = coeffs != 0
    # No valid lines
    if (is_line & clip).sum() == 0:
        return EWdict

    # Random draws from the covariance matrix
    covar_clip = covar[clip, :][:, clip]
    if seed is not None:
        np.random.seed(seed)

    draws = np.random.multivariate_normal(coeffs[clip], covar_clip, size=Ndraw)

    # Evaluate the continuum fits from the draws
    continuum = np.dot(draws*(~is_line[clip]), flux_arr[clip, :])

    # Compute the emission line EWs
    tidx = np.where(is_line[clip])[0]
    for ix in tidx:
        key = keys[clip][ix]

        # Line template
        line = np.dot(draws[:, ix][:, None], flux_arr[clip, :][ix, :][None, :])

        # Where line template non-zero
        mask = flux_arr[clip, :][ix, :] > 0
        ew_i = np.trapz((line/continuum)[:, mask],
                        wave[mask]*(1+z*observed_frame), axis=1)

        EWdict[key] = np.percentile(ew_i, [16., 50., 84.])

    return EWdict

#####################
# Photometry from Vizier tables


# CFHTLS
CFHTLS_W_VIZIER = 'II/317/cfhtls_w'
CFHTLS_W_BANDS = OrderedDict([('cfht_mega_u', ['umag', 'e_umag']),
                             ('cfht_mega_g', ['gmag', 'e_gmag']),
                             ('cfht_mega_r', ['rmag', 'e_rmag']),
                             ('cfht_mega_i', ['imag', 'e_imag']),
                             ('cfht_mega_z', ['zmag', 'e_zmag'])])

CFHTLS_D_VIZIER = 'II/317/cfhtls_d'
CFHTLS_D_BANDS = OrderedDict([('cfht_mega_u', ['umag', 'e_umag']),
                             ('cfht_mega_g', ['gmag', 'e_gmag']),
                             ('cfht_mega_r', ['rmag', 'e_rmag']),
                             ('cfht_mega_i', ['imag', 'e_imag']),
                             ('cfht_mega_z', ['zmag', 'e_zmag'])])

# SDSS DR12
SDSS_DR12_VIZIER = 'V/147/sdss12'
SDSS_DR12_BANDS = OrderedDict([('SDSS/u', ['umag', 'e_umag']),
                          ('SDSS/g', ['gmag', 'e_gmag']),
                          ('SDSS/r', ['rmag', 'e_rmag']),
                          ('SDSS/i', ['imag', 'e_imag']),
                          ('SDSS/z', ['zmag', 'e_zmag'])])

# PanStarrs
PS1_VIZIER = 'II/349/ps1'
PS1_BANDS = OrderedDict([('PS1.g', ['gKmag', 'e_gKmag']),
                 ('PS1.r', ['rKmag', 'e_rKmag']),
                 ('PS1.i', ['iKmag', 'e_iKmag']),
                 ('PS1.z', ['zKmag', 'e_zKmag']),
                 ('PS1.y', ['yKmag', 'e_yKmag'])])

# KIDS DR3
KIDS_DR3_VIZIER = 'II/347/kids_dr3'
KIDS_DR3_BANDS = OrderedDict([('OCam.sdss.u', ['umag', 'e_umag']),
                          ('OCam.sdss.g', ['gmag', 'e_gmag']),
                          ('OCam.sdss.r', ['rmag', 'e_rmag']),
                          ('OCam.sdss.i', ['imag', 'e_imag'])])

# WISE all-sky
WISE_VIZIER = 'II/328/allwise'
WISE_BANDS = OrderedDict([('WISE/RSR-W1', ['W1mag', 'e_W1mag']),
                          ('WISE/RSR-W2', ['W2mag', 'e_W2mag'])])
# ('WISE/RSR-W3', ['W3mag', 'e_W3mag']),
# ('WISE/RSR-W4', ['W4mag', 'e_W4mag'])])

# VIKING VISTA
VIKING_VIZIER = 'II/343/viking2'
VIKING_BANDS = OrderedDict([('SDSS/z', ['Zpmag', 'e_Zpmag']),
                            ('VISTA/Y',  ['Ypmag',  'e_Ypmag']),
                            ('VISTA/J',  ['Jpmag',  'e_Jpmag']),
                            ('VISTA/H',  ['Hpmag',  'e_Hpmag']),
                            ('VISTA/Ks', ['Kspmag', 'e_Kspmag'])])

# UKIDSS wide surveys
UKIDSS_LAS_VIZIER = 'II/319/las9'
UKIDSS_LAS_BANDS = OrderedDict([('WFCAM_Y', ['Ymag', 'e_Ymag']),
                            ('WFCAM_J', ['Jmag1', 'e_Jmag1']),
                            ('WFCAM_J', ['Jmag2', 'e_Jmag2']),
                            ('WFCAM_H', ['Hmag', 'e_Hmag']),
                            ('WFCAM_K', ['Kmag', 'e_Kmag'])])

UKIDSS_DXS_VIZIER = 'II/319/dxs9'
UKIDSS_DXS_BANDS = OrderedDict([('WFCAM_J', ['Jmag', 'e_Jmag']),
                            ('WFCAM_K', ['Kmag', 'e_Kmag'])])

# GALEX
GALEX_MIS_VIZIER = 'II/312/mis'
GALEX_MIS_BANDS = OrderedDict([('FUV', ['FUV', 'e_FUV']),
                              ('NUV', ['NUV', 'e_NUV'])])

GALEX_AIS_VIZIER = 'II/312/ais'
GALEX_AIS_BANDS = OrderedDict([('FUV', ['FUV', 'e_FUV']),
                              ('NUV', ['NUV', 'e_NUV'])])

# Combined Dict
VIZIER_BANDS = OrderedDict()
VIZIER_BANDS[CFHTLS_W_VIZIER] = CFHTLS_W_BANDS
VIZIER_BANDS[CFHTLS_D_VIZIER] = CFHTLS_D_BANDS
VIZIER_BANDS[SDSS_DR12_VIZIER] = SDSS_DR12_BANDS
VIZIER_BANDS[PS1_VIZIER] = PS1_BANDS
VIZIER_BANDS[KIDS_DR3_VIZIER] = KIDS_DR3_BANDS
VIZIER_BANDS[WISE_VIZIER] = WISE_BANDS
VIZIER_BANDS[VIKING_VIZIER] = VIKING_BANDS
VIZIER_BANDS[UKIDSS_LAS_VIZIER] = UKIDSS_LAS_BANDS
VIZIER_BANDS[UKIDSS_DXS_VIZIER] = UKIDSS_DXS_BANDS
VIZIER_BANDS[GALEX_MIS_VIZIER] = GALEX_MIS_BANDS
VIZIER_BANDS[GALEX_AIS_VIZIER] = GALEX_AIS_BANDS

VIZIER_VEGA = OrderedDict()
VIZIER_VEGA[CFHTLS_W_VIZIER] = False
VIZIER_VEGA[CFHTLS_D_VIZIER] = False
VIZIER_VEGA[SDSS_DR12_VIZIER] = False
VIZIER_VEGA[PS1_VIZIER] = False
VIZIER_VEGA[KIDS_DR3_VIZIER] = False
VIZIER_VEGA[WISE_VIZIER] = True
VIZIER_VEGA[VIKING_VIZIER] = True
VIZIER_VEGA[UKIDSS_LAS_VIZIER] = True
VIZIER_VEGA[UKIDSS_DXS_VIZIER] = True
VIZIER_VEGA[GALEX_MIS_VIZIER] = False
VIZIER_VEGA[GALEX_AIS_VIZIER] = False


def get_Vizier_photometry(ra, dec, templates=None, radius=2, vizier_catalog=PS1_VIZIER, bands=PS1_BANDS, filter_file='/usr/local/share/eazy-photoz/filters/FILTER.RES.latest', MW_EBV=0, convert_vega=False, raw_query=False, verbose=True, timeout=300, rowlimit=50000):
    """
    Fetch photometry from a Vizier catalog

    Requires eazypy/eazy
    """

    from collections import OrderedDict

    import astropy.units as u
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = rowlimit
    Vizier.TIMEOUT = timeout

    #print('xxx', Vizier.ROW_LIMIT, Vizier.TIMEOUT)

    import astropy.coordinates as coord
    import astropy.units as u

    #import pysynphot as S

    from eazy.templates import Template
    from eazy.filters import FilterFile
    from eazy.photoz import TemplateGrid
    from eazy.filters import FilterDefinition

    res = FilterFile(filter_file)

    coo = coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg),
                         frame='icrs')

    columns = ['*']
    #columns = []
    if isinstance(vizier_catalog, list):
        for c in [VIKING_VIZIER]:
            for b in VIZIER_BANDS[c]:
                columns += VIZIER_BANDS[c][b]

        columns = list(np.unique(columns))
        #print("xxx columns", columns)
    else:
        for b in bands:
            columns += bands[b]

    if isinstance(vizier_catalog, list):
        v = Vizier(catalog=VIKING_VIZIER, columns=['+_r']+columns)
    else:
        v = Vizier(catalog=vizier_catalog, columns=['+_r']+columns)

    v.ROW_LIMIT = rowlimit
    v.TIMEOUT = timeout

    #query_catalog = vizier_catalog
    try:
        tabs = v.query_region(coo, radius="{0}s".format(radius),
                          catalog=vizier_catalog)  # [0]

        if raw_query:
            return(tabs)

        tab = tabs[0]

        if False:
            for t in tabs:
                bands = VIZIER_BANDS[t.meta['name']]
                for b in bands:
                    for c in bands[b]:
                        print(t.meta['name'], c, c in t.colnames)  # c = bands[b][0]

        ix = np.argmin(tab['_r'])
        tab = tab[ix]
    except:
        tab = None

        return None

    viz_tables = ', '.join([t.meta['name'] for t in tabs])
    if verbose:
        print('Photometry from vizier catalogs: {0}'.format(viz_tables))

    pivot = []  # OrderedDict()
    flam = []
    eflam = []
    filters = []

    for tab in tabs:

        # Downweight PS1 if have SDSS ?  For now, do nothing
        if (tab.meta['name'] == PS1_VIZIER) & (SDSS_DR12_VIZIER in viz_tables):
            # continue
            err_scale = 1
        else:
            err_scale = 1

        # Only use one CFHT catalog
        if (tab.meta['name'] == CFHTLS_W_VIZIER) & (CFHTLS_D_VIZIER in viz_tables):
            continue

        if (tab.meta['name'] == UKIDSS_LAS_VIZIER):
            flux_scale = 1.33
        else:
            flux_scale = 1.

        convert_vega = VIZIER_VEGA[tab.meta['name']]
        bands = VIZIER_BANDS[tab.meta['name']]

        # if verbose:
        #    print(tab.colnames)

        #filters += [res.filters[res.search(b, verbose=False)[0]] for b in bands]

        to_flam = 10**(-0.4*(48.6))*3.e18  # / pivot(Ang)**2

        for ib, b in enumerate(bands):
            filt = res.filters[res.search(b, verbose=False)[0]]
            filters.append(filt)

            if convert_vega:
                to_ab = filt.ABVega()
            else:
                to_ab = 0.

            fcol, ecol = bands[b]
            pivot.append(filt.pivot())
            flam.append(10**(-0.4*(tab[fcol][0]+to_ab))*to_flam/pivot[-1]**2)
            flam[-1] *= flux_scale
            eflam.append(tab[ecol][0]*np.log(10)/2.5*flam[-1]*err_scale)

    for i in range(len(filters))[::-1]:
        if np.isscalar(flam[i]) & np.isscalar(eflam[i]):
            continue
        else:
            flam.pop(i)
            eflam.pop(i)
            filters.pop(i)
            pivot.pop(i)

    lc = np.array(pivot)  # [pivot[ib] for ib in range(len(bands))]

    if templates is not None:

        eazy_templates = [Template(arrays=(templates[k].wave, templates[k].flux), name=k) for k in templates]

        zgrid = log_zgrid(zr=[0.01, 3.4], dz=0.005)

        tempfilt = TemplateGrid(zgrid, eazy_templates, filters=filters, add_igm=True, galactic_ebv=MW_EBV, Eb=0, n_proc=0, verbose=False)
    else:
        tempfilt = None

    phot = OrderedDict([('flam', np.array(flam)), ('eflam', np.array(eflam)), ('filters', filters), ('tempfilt', tempfilt), ('lc', np.array(lc)), ('source', 'Vizier '+viz_tables)])

    return phot


def generate_tempfilt(templates, filters, zgrid=None, MW_EBV=0):

    from eazy.templates import Template
    from eazy.photoz import TemplateGrid

    # twave, tflux, is_line = array_templates(templates, z=0)
    # eazy_templates = []
    # for i, t in enumerate(templates):
    #     eazy_templates.append(Template(arrays=[twave, np.maximum(twave, 1.e-30)], name=t))

    eazy_templates = [Template(arrays=(templates[k].wave, templates[k].flux), name=k) for k in templates]

    if zgrid is None:
        zgrid = log_zgrid(zr=[0.01, 3.4], dz=0.005)

    tempfilt = TemplateGrid(zgrid, eazy_templates, filters=filters, add_igm=True, galactic_ebv=MW_EBV, Eb=0, n_proc=0, verbose=False)

    return tempfilt


def combine_phot_dict(phots, templates=None, MW_EBV=0):
    """
    Combine photmetry dictionaries
    """
    phot = {}
    phot['flam'] = []
    phot['eflam'] = []
    phot['filters'] = []
    for p in phots:
        phot['flam'] = np.append(phot['flam'], p['flam'])
        phot['eflam'] = np.append(phot['eflam'], p['eflam'])
        phot['filters'].extend(p['filters'])

    if templates is not None:
        phot['tempfilt'] = generate_tempfilt(templates, phot['filters'], MW_EBV=MW_EBV)

    return phot


def get_spectrum_AB_mags(spectrum, bandpasses=[]):
    """
    Integrate a `~pysynphot` spectrum through filter bandpasses

    Parameters
    ----------
    spectrum : type

    bandpasses : list
        List of `pysynphot` bandpass objects, e.g.,

           >>> import pysynphot as S
           >>> bandpasses = [S.ObsBandpass('wfc3,ir,f140w')]


    Returns
    -------
    ab_mags : dict
        Dictionary with keys from `bandpasses` and the integrated magnitudes

    """
    import pysynphot as S
    flat = S.FlatSpectrum(0, fluxunits='ABMag')
    ab_mags = OrderedDict()

    for bp in bandpasses:
        flat_obs = S.Observation(flat, bp)
        spec_obs = S.Observation(spectrum, bp)
        ab_mags[bp.name] = -2.5*np.log10(spec_obs.countrate()/flat_obs.countrate())

    return ab_mags


def log_zgrid(zr=[0.7, 3.4], dz=0.01):
    """Make a logarithmically spaced redshift grid

    Parameters
    ----------
    zr : [float, float]
        Minimum and maximum of the desired grid

    dz : float
        Step size, dz/(1+z)

    Returns
    -------
    zgrid : array-like
        Redshift grid

    """
    zgrid = np.exp(np.arange(np.log(1+zr[0]), np.log(1+zr[1]), dz))-1
    return zgrid

def trapz_dx(x):
    """
    Return trapezoid rule coefficients, useful for numerical integration 
    using a dot product
    
    Parameters
    ----------
    x : array-like
        Independent variable
    
    Returns
    -------
    dx : array_like
        Coefficients for trapezoidal rule integration.
        
    """
    dx = np.zeros_like(x)
    diff = np.diff(x)/2.
    dx[:-1] += diff
    dx[1:] += diff
    return dx


def get_wcs_pscale(wcs, set_attribute=True):
    """Get correct pscale from a `~astropy.wcs.WCS` object

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`

    set_attribute : bool
        Set the `pscale` attribute on `wcs`, along with returning the value.

    Returns
    -------
    pscale : float
        Pixel scale from `wcs.cd`

    """
    from numpy import linalg

    if isinstance(wcs, pyfits.Header):
        wcs = pywcs.WCS(wcs, relax=True)

    if hasattr(wcs.wcs, 'cd'):
        det = linalg.det(wcs.wcs.cd)
    else:
        det = linalg.det(wcs.wcs.pc)

    pscale = np.sqrt(np.abs(det))*3600.
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 
                  'cdelt will be ignored since cd is present', RuntimeWarning)
        
        if hasattr(wcs.wcs, 'cdelt'):
            pscale *= wcs.wcs.cdelt[0]
        
    wcs.pscale = pscale

    return pscale


def transform_wcs(in_wcs, translation=[0., 0.], rotation=0., scale=1.):
    """Update WCS with shift, rotation, & scale

    Parameters
    ----------
    in_wcs: `~astropy.wcs.WCS`
        Input WCS

    translation: [float, float]
        xshift & yshift in pixels

    rotation: float
        CCW rotation (towards East), radians

    scale: float
        Pixel scale factor

    Returns
    -------
    out_wcs: `~astropy.wcs.WCS`
        Modified WCS
    """
    out_wcs = in_wcs.deepcopy()

    #out_wcs.wcs.crpix += np.array(translation)

    # Compute shift for crval, not crpix
    crval = in_wcs.all_pix2world([in_wcs.wcs.crpix-np.array(translation)],
                                     1).flatten()

    # Compute shift at image center
    if hasattr(in_wcs, '_naxis1'):
        refpix = np.array([in_wcs._naxis1/2., in_wcs._naxis2/2.])
    else:
        refpix = np.array(in_wcs._naxis)/2.

    c0 = in_wcs.all_pix2world([refpix], 1).flatten()
    c1 = in_wcs.all_pix2world([refpix-np.array(translation)], 1).flatten()

    out_wcs.wcs.crval += c1-c0

    theta = -rotation
    _mat = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

    try:
        out_wcs.wcs.cd[:2,:2] = np.dot(out_wcs.wcs.cd[:2,:2], _mat)/scale
    except:
        out_wcs.wcs.pc = np.dot(out_wcs.wcs.pc, _mat)/scale

    out_wcs.pscale = get_wcs_pscale(out_wcs)
    #out_wcs.wcs.crpix *= scale
    if hasattr(out_wcs, 'pixel_shape'):
        _naxis1 = int(np.round(out_wcs.pixel_shape[0]*scale))
        _naxis2 = int(np.round(out_wcs.pixel_shape[1]*scale))
        out_wcs._naxis = [_naxis1, _naxis2]
    elif hasattr(out_wcs, '_naxis1'):
        out_wcs._naxis1 = int(np.round(out_wcs._naxis1*scale))
        out_wcs._naxis2 = int(np.round(out_wcs._naxis2*scale))

    return out_wcs


def sip_rot90(input, rot, reverse=False, verbose=False, compare=False):
    """
    Rotate a SIP WCS by increments of 90 degrees using direct transformations
    between x / y coordinates
    
    Parameters
    ----------
    input : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
        Header or WCS
    
    rot : int
        Number of times to rotate the WCS 90 degrees *clockwise*, analogous
        to `numpy.rot90`
    
    reverse : bool
        If `input` is a header and includes a keyword ``ROT90``, then undo 
        the rotation and remove the keyword from the output header
        
    Returns
    -------
    header : `~astropy.io.fits.Header`
        Rotated WCS header
        
    wcs : `~astropy.wcs.WCS`
        Rotated WCS
    
    desc : str
        Description of the transform associated with ``rot``, e.g, 
        ``x=nx-x, y=ny-y`` for ``rot=±2``.
        
    """
    import copy
    import astropy.io.fits
    import astropy.wcs
    import matplotlib.pyplot as plt
    
    if isinstance(input, astropy.io.fits.Header):
        orig = copy.deepcopy(input)
        new = copy.deepcopy(input)
        
        if ('ROT90' in input):
            if reverse:
                rot = -orig['ROT90']
                new.remove('ROT90')
            else:
                new['ROT90'] = orig['ROT90'] + rot
        else:
            new['ROT90'] = rot
    else:
        orig = to_header(input)
        new = to_header(input)

    orig_wcs = pywcs.WCS(orig, relax=True)

    ### CD = [[dra/dx, dra/dy], [dde/dx, dde/dy]]
    ### x = a_i_j * u**i * v**j
    ### y = b_i_j * u**i * v**j
    
    ix = 1
    
    if compare:
        xarr = np.arange(0,2048,64)
        xp, yp = np.meshgrid(xarr, xarr)
        rd = orig_wcs.all_pix2world(xp, yp, ix)

    if rot % 4 == 1:
        # CW 90 deg : x = y, y = (nx - x), u=v, v=-u
        desc = 'x=y, y=nx-x'
        
        new['CRPIX1'] = orig['CRPIX2']
        new['CRPIX2'] = orig['NAXIS1'] - orig['CRPIX1'] + 1

        new['CD1_1'] = orig['CD1_2']
        new['CD1_2'] = -orig['CD1_1']
        new['CD2_1'] = orig['CD2_2']
        new['CD2_2'] = -orig['CD2_1']

        for i in range(new['A_ORDER']+1):
            for j in range(new['B_ORDER']+1):
                Aij  = f'A_{i}_{j}'
                if Aij not in new:
                    continue

                new[f'A_{i}_{j}'] = orig[f'B_{j}_{i}']*(-1)**j
                new[f'B_{i}_{j}'] = orig[f'A_{j}_{i}']*(-1)**j*-1

        new_wcs = astropy.wcs.WCS(new, relax=True)
        
        if compare:
            xr, yr = new_wcs.all_world2pix(*rd, ix)
            xo = yp
            yo = orig['NAXIS1'] - xp

    elif rot % 4 == 3:
        # CW 270 deg : y = x, x = (ny - u), u=-v, v=u
        desc = 'x=ny-y, y=x'

        new['CRPIX1'] = orig['NAXIS2'] - orig['CRPIX2'] + 1
        new['CRPIX2'] = orig['CRPIX1']

        new['CD1_1'] = -orig['CD1_2']
        new['CD1_2'] = orig['CD1_1']
        new['CD2_1'] = -orig['CD2_2']
        new['CD2_2'] = orig['CD2_1']

        for i in range(new['A_ORDER']+1):
            for j in range(new['B_ORDER']+1):
                Aij  = f'A_{i}_{j}'
                if Aij not in new:
                    continue

                new[f'A_{i}_{j}'] = orig[f'B_{j}_{i}']*(-1)**i*-1
                new[f'B_{i}_{j}'] = orig[f'A_{j}_{i}']*(-1)**i

        new_wcs = astropy.wcs.WCS(new, relax=True)
        
        if compare:
            xr, yr = new_wcs.all_world2pix(*rd, ix)
            xo = orig['NAXIS2'] - yp
            yo = xp


    elif rot % 4 == 2:
        # CW 180 deg : x=nx-x, y=ny-y, u=-u, v=-v
        desc = 'x=nx-x, y=ny-y'

        new['CRPIX1'] = orig['NAXIS1'] - orig['CRPIX1'] + 1
        new['CRPIX2'] = orig['NAXIS2'] - orig['CRPIX2'] + 1

        new['CD1_1'] = -orig['CD1_1']
        new['CD1_2'] = -orig['CD1_2']
        new['CD2_1'] = -orig['CD2_1']
        new['CD2_2'] = -orig['CD2_2']

        for i in range(new['A_ORDER']+1):
            for j in range(new['B_ORDER']+1):
                Aij  = f'A_{i}_{j}'
                if Aij not in new:
                    continue

                new[f'A_{i}_{j}'] = orig[f'A_{i}_{j}']*(-1)**j*(-1)**i*-1
                new[f'B_{i}_{j}'] = orig[f'B_{i}_{j}']*(-1)**j*(-1)**i*-1

        new_wcs = astropy.wcs.WCS(new, relax=True)
        
        if compare:
            xr, yr = new_wcs.all_world2pix(*rd, ix)
            xo = orig['NAXIS1'] - xp
            yo = orig['NAXIS2'] - yp
    else:
        # rot=0, do nothing
        desc = 'x=x, y=y'
        new_wcs = orig_wcs
        if compare:
            xo = xp
            yo = yp
            xr, yr = new_wcs.all_world2pix(*rd, ix)
        
    if verbose:
        if compare:
            xrms = nmad(xr-xo)
            yrms = nmad(yr-yo)
            print(f'Rot90: {rot} rms={xrms:.2e} {yrms:.2e}')
            
    if compare:
        fig, axes = plt.subplots(1,2,figsize=(10,5), sharex=True, sharey=True)
        axes[0].scatter(xp, xr-xo)
        axes[0].set_xlabel('dx')
        axes[1].scatter(yp, yr-yo)
        axes[1].set_xlabel('dy')
        for ax in axes:
            ax.grid()
        
        fig.tight_layout(pad=0.5)
    
    return new, new_wcs, desc


def get_wcs_slice_header(wcs, slx, sly):
    """TBD
    """
    #slx, sly = slice(1279, 1445), slice(2665,2813)
    h = wcs.slice((sly, slx)).to_header(relax=True)
    h['NAXIS'] = 2
    h['NAXIS1'] = slx.stop-slx.start
    h['NAXIS2'] = sly.stop-sly.start
    for k in h:
        if k.startswith('PC'):
            h.rename_keyword(k, k.replace('PC', 'CD'))

    return h


def get_common_slices(a_origin, a_shape, b_origin, b_shape):
    """
    Get slices of overlaps between two rectangular grids
    """

    ll = np.min([a_origin, b_origin], axis=0)
    ur = np.max([a_origin+a_shape, b_origin+b_shape], axis=0)

    # other in self
    lls = np.minimum(b_origin - ll, a_shape)
    urs = np.clip(b_origin + b_shape - a_origin, [0, 0], a_shape)

    # self in other
    llo = np.minimum(a_origin - ll, b_shape)
    uro = np.clip(a_origin + a_shape - b_origin, [0, 0], b_shape)

    a_slice = (slice(lls[0], urs[0]), slice(lls[1], urs[1]))
    b_slice = (slice(llo[0], uro[0]), slice(llo[1], uro[1]))
    return a_slice, b_slice


class WCSFootprint(object):
    """
    Helper functions for dealing with WCS footprints
    """

    def __init__(self, wcs, ext=1, label=None):
        if isinstance(wcs, pywcs.WCS):
            self.wcs = wcs.deepcopy()
            if not hasattr(self.wcs, 'pixel_shape'):
                self.wcs.pixel_shape = None

            if self.wcs.pixel_shape is None:
                self.wcs.pixel_shape = [int(p*2) for p in self.wcs.wcs.crpix]
        elif isinstance(wcs, str):
            hdu = pyfits.open(wcs)
            if len(hdu) == 1:
                ext = 0

            self.add_naxis(hdu[ext].header)
            the_wcs = pywcs.WCS(hdu[ext].header, fobj=hdu)
            self.wcs = the_wcs
            hdu.close()
            
        elif isinstance(wcs, pyfits.HDUList):
            if len(wcs) == 1:
                ext = 0
            self.add_naxis(wcs[ext].header)
            the_wcs = pywcs.WCS(wcs[ext].header, fobj=wcs)
            self.wcs = the_wcs
        else:
            print('WCS class not recognized: {0}'.format(wcs.__class__))
            raise ValueError

        self.fp = self.wcs.calc_footprint()
        self.cosdec = np.cos(self.fp[0, 1]/180*np.pi)
        self.label = label
        self.pixel_scale = get_wcs_pscale(self.wcs)

    @property
    def centroid(self):
        return np.mean(self.fp, axis=0)


    @property
    def path(self):
        """
        `~matplotlib.path.Path` object
        """
        import matplotlib.path
        return matplotlib.path.Path(self.fp)


    @property
    def polygon(self):
        """
        `~shapely.geometry.Polygon` object.
        """
        from shapely.geometry import Polygon
        return Polygon(self.fp)


    def get_patch(self, **kwargs):
        """
        `~matplotlib.pach.PathPatch` object
        """
        return patch_from_polygon(self.polygon, **kwargs)


    @property
    def region(self):
        """
        Polygon string in DS9 region format
        """
        return 'polygon({0})'.format(','.join(['{0:.6f}'.format(c) for c in self.fp.flatten()]))


    @staticmethod
    def add_naxis(header):
        """
        If NAXIS keywords not found in an image header, assume the parent
        image dimensions are 2*CRPIX
        """
        for i in [1, 2]:
            if 'NAXIS{0}'.format(i) not in header:
                header['NAXIS{0}'.format(i)] = int(header['CRPIX{0}'.format(i)]*2)


def reproject_faster(input_hdu, output, pad=10, **kwargs):
    """Speed up `reproject` module with array slices of the input image

    Parameters
    ----------
    input_hdu : `~astropy.io.fits.ImageHDU`
        Input image HDU to reproject.

    output : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        Output frame definition.

    pad : int
        Pixel padding on slices cut from the `input_hdu`.

    kwargs : dict
        Arguments passed through to `~reproject.reproject_interp`.  For
        example, `order='nearest-neighbor'`.

    Returns
    -------
    reprojected : `~numpy.ndarray`
        Reprojected data from `input_hdu`.

    footprint : `~numpy.ndarray`
        Footprint of the input array in the output frame.

    Notes
    -----

    `reproject' is an astropy-compatible module that can be installed with
    `pip`.  See https://reproject.readthedocs.io.
    """
    import reproject

    # Output WCS
    if isinstance(output, pywcs.WCS):
        out_wcs = output
    else:
        out_wcs = pywcs.WCS(output, relax=True)

    if 'SIP' in out_wcs.wcs.ctype[0]:
        print('Warning: `reproject` doesn\'t appear to support SIP projection')

    # Compute pixel coordinates of the output frame corners in the input image
    input_wcs = pywcs.WCS(input_hdu.header, relax=True)
    out_fp = out_wcs.calc_footprint()
    input_xy = input_wcs.all_world2pix(out_fp, 0)
    slx = slice(int(input_xy[:, 0].min())-pad, int(input_xy[:, 0].max())+pad)
    sly = slice(int(input_xy[:, 1].min())-pad, int(input_xy[:, 1].max())+pad)

    # Make the cutout
    sub_data = input_hdu.data[sly, slx]
    sub_header = get_wcs_slice_header(input_wcs, slx, sly)
    sub_hdu = pyfits.PrimaryHDU(data=sub_data, header=sub_header)

    # Get the reprojection
    seg_i, fp_i = reproject.reproject_interp(sub_hdu, output, **kwargs)
    return seg_i.astype(sub_data.dtype), fp_i.astype(np.uint8)


def full_spectrum_wcsheader(center_wave=1.4e4, dlam=40, NX=100, spatial_scale=1, NY=10):
    """Make a WCS header for a 2D spectrum

    Parameters
    ----------
    center_wave : float
        Wavelength of the central pixel, in Anstroms

    dlam : float
        Delta-wavelength per (x) pixel

    NX, NY : int
        Number of x & y pixels. Output will have shape `(2*NY, 2*NX)`.

    spatial_scale : float
        Spatial scale of the output, in units of the input pixels

    Returns
    -------
    header : `~astropy.io.fits.Header`
        Output WCS header

    wcs : `~astropy.wcs.WCS`
        Output WCS

    Examples
    --------

        >>> from grizli.utils import make_spectrum_wcsheader
        >>> h, wcs = make_spectrum_wcsheader()
        >>> print(wcs)
        WCS Keywords
        Number of WCS axes: 2
        CTYPE : 'WAVE'  'LINEAR'
        CRVAL : 14000.0  0.0
        CRPIX : 101.0  11.0
        CD1_1 CD1_2  : 40.0  0.0
        CD2_1 CD2_2  : 0.0  1.0
        NAXIS    : 200 20

    """

    h = pyfits.ImageHDU(data=np.zeros((2*NY, 2*NX), dtype=np.float32))

    refh = h.header
    refh['CRPIX1'] = NX+1
    refh['CRPIX2'] = NY+1
    refh['CRVAL1'] = center_wave/1.e4
    refh['CD1_1'] = dlam/1.e4
    refh['CD1_2'] = 0.
    refh['CRVAL2'] = 0.
    refh['CD2_2'] = spatial_scale
    refh['CD2_1'] = 0.
    refh['RADESYS'] = ''

    refh['CTYPE1'] = 'RA---TAN-SIP'
    refh['CUNIT1'] = 'mas'
    refh['CTYPE2'] = 'DEC--TAN-SIP'
    refh['CUNIT2'] = 'mas'

    ref_wcs = pywcs.WCS(refh)
    ref_wcs.pscale = get_wcs_pscale(ref_wcs)

    return refh, ref_wcs


def make_spectrum_wcsheader(center_wave=1.4e4, dlam=40, NX=100, spatial_scale=1, NY=10):
    """Make a WCS header for a 2D spectrum

    Parameters
    ----------
    center_wave : float
        Wavelength of the central pixel, in Anstroms

    dlam : float
        Delta-wavelength per (x) pixel

    NX, NY : int
        Number of x & y pixels. Output will have shape `(2*NY, 2*NX)`.

    spatial_scale : float
        Spatial scale of the output, in units of the input pixels

    Returns
    -------
    header : `~astropy.io.fits.Header`
        Output WCS header

    wcs : `~astropy.wcs.WCS`
        Output WCS

    Examples
    --------

        >>> from grizli.utils import make_spectrum_wcsheader
        >>> h, wcs = make_spectrum_wcsheader()
        >>> print(wcs)
        WCS Keywords
        Number of WCS axes: 2
        CTYPE : 'WAVE'  'LINEAR'
        CRVAL : 14000.0  0.0
        CRPIX : 101.0  11.0
        CD1_1 CD1_2  : 40.0  0.0
        CD2_1 CD2_2  : 0.0  1.0
        NAXIS    : 200 20

    """

    h = pyfits.ImageHDU(data=np.zeros((2*NY, 2*NX), dtype=np.float32))

    refh = h.header
    refh['CRPIX1'] = NX+1
    refh['CRPIX2'] = NY+1
    refh['CRVAL1'] = center_wave
    refh['CD1_1'] = dlam
    refh['CD1_2'] = 0.
    refh['CRVAL2'] = 0.
    refh['CD2_2'] = spatial_scale
    refh['CD2_1'] = 0.
    refh['RADESYS'] = ''

    refh['CTYPE1'] = 'WAVE'
    refh['CTYPE2'] = 'LINEAR'

    ref_wcs = pywcs.WCS(h.header)
    ref_wcs.pscale = np.sqrt(ref_wcs.wcs.cd[0, 0]**2 + ref_wcs.wcs.cd[1, 0]**2)*3600.

    return refh, ref_wcs


def read_gzipped_header(file='test.fits.gz', BLOCK=1024, NMAX=256, nspace=16, strip=False):
    """
    Read primary header from a (potentially large) zipped FITS file

    The script proceeds by reading `NMAX` segments of size `BLOCK` bytes from
    the file and searching for a string `END + ' '*nspace` in the data
    indicating the end of the primary header.

    Parameters
    ----------
    file : str
        Filename of gzipped FITS file

    BLOCK, NMAX, nspace : int
        Parameters for reading bytes from the input file

    strip : bool
        Send output through `strip_header_keys`.


    Returns
    -------
    header : `~astropy.io.fits.Header`
        Header object

    """
    import gzip
    import astropy.io.fits as pyfits

    f = gzip.GzipFile(fileobj=open(file, 'rb'))

    data = b''
    end = b' END'+b' '*nspace

    for i in range(NMAX):
        data_i = f.read(BLOCK)
        if end in data_i:
            break

        data += data_i

    if (i == NMAX-1):
        print('Error: END+{3}*" " not found in first {0}x{1} bytes of {2})'.format(NMAX, BLOCK, file, nspace))
        f.close()
        return {}

    ix = data_i.index(end)
    data += data_i[:ix]+end  # data_i[:ix]

    f.close()
    data_str = data.decode('utf8')
    h = pyfits.Header.fromstring(data_str)

    if strip:
        return strip_header_keys(h, usewcs=True)
    else:
        return h


DRIZZLE_KEYS = ['GEOM', 'DATA', 'DEXP', 'OUDA', 'OUWE', 'OUCO', 'MASK', 'WTSC', 'KERN', 'PIXF', 'COEF', 'OUUN', 'FVAL', 'WKEY', 'SCAL', 'ISCL']


def strip_header_keys(header, comment=True, history=True, drizzle_keys=DRIZZLE_KEYS, usewcs=False, keep_with_wcs=['EXPTIME', 'FILTER', 'TELESCOP', 'INSTRUME', 'DATE-OBS', 'EXPSTART', 'EXPEND']):
    """
    Strip header keywords

    Parameters
    ----------

    comment, history : bool
        Strip 'COMMENT' and 'HISTORY' keywords, respectively.

    drizzle_keys : list
        Strip keys produced by `~drizzlepac.astrodrizzle`.

    usewcs : bool
        Alternatively, just generate a simple WCS-only header from the input
        header.

    keep_with_wcs : list
        Additional keys to try to add to the `usewcs` header.

    Returns
    -------

    header : `~astropy.io.fits.Header`
        Header object.

    """
    import copy
    import astropy.wcs as pywcs

    # Parse WCS and build header
    if usewcs:
        wcs = pywcs.WCS(header)
        h = to_header(wcs)
        for k in keep_with_wcs:
            if k in header:
                if k in header.comments:
                    h[k] = header[k], header.comments[k]
                else:
                    h[k] = header[k]

        if 'FILTER' in keep_with_wcs:
            try:
                h['FILTER'] = (parse_filter_from_header(header),
                               'element selected from filter wheel')
            except:
                pass

        return h

    h = copy.deepcopy(header)
    keys = list(h.keys())
    strip_keys = []
    if comment:
        strip_keys.append('COMMENT')

    if history:
        strip_keys.append('HISTORY')

    for k in keys:
        if k in strip_keys:
            h.remove(k)

        if drizzle_keys:
            if k.startswith('D'):
                if (k[-4:] in drizzle_keys) | k.endswith('VER'):
                    h.remove(k)

    return h


def wcs_from_header(header, relax=True, **kwargs):
    """
    Initialize `~astropy.wcs.WCS` from a `~astropy.io.fits.Header`
    
    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        FITS header with optional ``SIPCRPX1`` and ``SIPCRPX2`` keywords that
        define a separate reference pixel for a SIP header
    
    relax, kwargs : bool, dict
        Keywords passed to `astropy.wcs.WCS`
    
    Returns
    -------
    wcs : `~astropy.wcs.WCS`
        WCS object
        
    """
    wcs = pywcs.WCS(header, relax=relax)
    
    if ('SIPCRPX1' in header) & hasattr(wcs, 'sip'):
        wcs.sip.crpix[0] = header['SIPCRPX1']
        wcs.sip.crpix[1] = header['SIPCRPX2']
    elif ('SIAF_XREF_SCI' in header) & hasattr(wcs, 'sip'):
        wcs.sip.crpix[0] = header['SIAF_XREF_SCI']
        wcs.sip.crpix[1] = header['SIAF_YREF_SCI']
        
    return wcs


def to_header(wcs, add_naxis=True, relax=True, key=None):
    """Modify `astropy.wcs.WCS.to_header` to produce more keywords

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        Input WCS.

    add_naxis : bool
        Add NAXIS keywords from WCS dimensions

    relax : bool
        Passed to `WCS.to_header(relax=)`.

    key : str
        See `~astropy.wcs.WCS.to_header`.

    Returns
    -------
    header : `~astropy.io.fits.Header`
        Output header.

    """
    header = wcs.to_header(relax=relax, key=key)
    if add_naxis:
        if hasattr(wcs, 'pixel_shape'):
            header['NAXIS'] = wcs.naxis
            if wcs.pixel_shape is not None:
                header['NAXIS1'] = wcs.pixel_shape[0]
                header['NAXIS2'] = wcs.pixel_shape[1]

        elif hasattr(wcs, '_naxis1'):
            header['NAXIS'] = wcs.naxis
            header['NAXIS1'] = wcs._naxis1
            header['NAXIS2'] = wcs._naxis2

    for k in header:
        if k.startswith('PC'):
            cd = k.replace('PC', 'CD')
            header.rename_keyword(k, cd)
    
    if hasattr(wcs, 'sip'):
        if hasattr(wcs.sip, 'crpix'):
            header['SIPCRPX1'], header['SIPCRPX2'] = wcs.sip.crpix
            
    return header


def make_wcsheader(ra=40.07293, dec=-1.6137748, size=2, pixscale=0.1, get_hdu=False, theta=0):
    """Make a celestial WCS header

    Parameters
    ----------
    ra, dec : float
        Celestial coordinates in decimal degrees

    size, pixscale : float or 2-list
        Size of the thumbnail, in arcsec, and pixel scale, in arcsec/pixel.
        Output image will have dimensions `(npix,npix)`, where

            >>> npix = size/pixscale

    get_hdu : bool
        Return a `~astropy.io.fits.ImageHDU` rather than header/wcs.

    theta : float
        Position angle of the output thumbnail (degrees)

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        HDU with data filled with zeros if `get_hdu=True`.

    header, wcs : `~astropy.io.fits.Header`, `~astropy.wcs.WCS`
        Header and WCS object if `get_hdu=False`.

    Examples
    --------

        >>> from grizli.utils import make_wcsheader
        >>> h, wcs = make_wcsheader()
        >>> print(wcs)
        WCS Keywords
        Number of WCS axes: 2
        CTYPE : 'RA---TAN'  'DEC--TAN'
        CRVAL : 40.072929999999999  -1.6137748000000001
        CRPIX : 10.0  10.0
        CD1_1 CD1_2  : -2.7777777777777e-05  0.0
        CD2_1 CD2_2  : 0.0  2.7777777777777701e-05
        NAXIS    : 20 20

        >>> from grizli.utils import make_wcsheader
        >>> hdu = make_wcsheader(get_hdu=True)
        >>> print(hdu.data.shape)
        (20, 20)
        >>> print(hdu.header.tostring)
        XTENSION= 'IMAGE   '           / Image extension
        BITPIX  =                  -32 / array data type
        NAXIS   =                    2 / number of array dimensions
        PCOUNT  =                    0 / number of parameters
        GCOUNT  =                    1 / number of groups
        CRPIX1  =                   10
        CRPIX2  =                   10
        CRVAL1  =             40.07293
        CRVAL2  =           -1.6137748
        CD1_1   = -2.7777777777777E-05
        CD1_2   =                  0.0
        CD2_1   =                  0.0
        CD2_2   = 2.77777777777777E-05
        NAXIS1  =                   20
        NAXIS2  =                   20
        CTYPE1  = 'RA---TAN'
        CTYPE2  = 'DEC--TAN'
    """

    if np.isscalar(pixscale):
        cdelt = [pixscale/3600.]*2
    else:
        cdelt = [pixscale[0]/3600., pixscale[1]/3600.]

    if np.isscalar(size):
        npix = np.cast[int](np.round([size/pixscale, size/pixscale]))
    else:
        npix = np.cast[int](np.round([size[0]/pixscale, size[1]/pixscale]))

    hout = pyfits.Header()
    hout['CRPIX1'] = npix[0]/2+1
    hout['CRPIX2'] = npix[1]/2+1
    hout['CRVAL1'] = ra
    hout['CRVAL2'] = dec
    hout['CD1_1'] = -cdelt[0]
    hout['CD1_2'] = hout['CD2_1'] = 0.
    hout['CD2_2'] = cdelt[1]
    hout['NAXIS1'] = npix[0]
    hout['NAXIS2'] = npix[1]
    hout['CTYPE1'] = 'RA---TAN'
    hout['CTYPE2'] = 'DEC--TAN'

    hout['RADESYS'] = 'ICRS'
    hout['EQUINOX'] = 2000
    hout['LATPOLE'] = hout['CRVAL2']
    hout['LONPOLE'] = 180

    hout['PIXASEC'] = pixscale, 'Pixel scale in arcsec'

    wcs_out = pywcs.WCS(hout)

    theta_rad = np.deg2rad(theta)
    mat = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                    [np.sin(theta_rad),  np.cos(theta_rad)]])

    rot_cd = np.dot(mat, wcs_out.wcs.cd)

    for i in [0, 1]:
        for j in [0, 1]:
            hout['CD{0:d}_{1:d}'.format(i+1, j+1)] = rot_cd[i, j]
            wcs_out.wcs.cd[i, j] = rot_cd[i, j]

    cd = wcs_out.wcs.cd
    wcs_out.pscale = get_wcs_pscale(wcs_out)  # np.sqrt((cd[0,:]**2).sum())*3600.

    if get_hdu:
        hdu = pyfits.ImageHDU(header=hout, data=np.zeros((npix[1], npix[0]), dtype=np.float32))
        return hdu
    else:
        return hout, wcs_out


def get_flt_footprint(flt_file, extensions=[1, 2, 3, 4], patch_args=None):
    """
    Compute footprint of all SCI extensions of an HST exposure

    Parameters
    ----------
    extensions : list
        List of extensions to retrieve (can have extras).

    patch_args : dict or None
        If a `dict`, then generate a patch for the footprint passing
        `**patch_args` arguments (e.g., `{'fc':'blue', 'alpha':0.1}`).

    Returns
    -------
    fp / patch : `~shapely.geometry` object or `matplotlib.patch.Patch`
        The footprint or footprint patch.

    """
    from shapely.geometry import Polygon

    im = pyfits.open(flt_file)
    fp = None

    for ext in extensions:
        if ('SCI', ext) not in im:
            continue

        wcs = pywcs.WCS(im['SCI', ext].header, fobj=im)
        p_i = Polygon(wcs.calc_footprint())
        if fp is None:
            fp = p_i
        else:
            fp = fp.union(p_i)
    
    im.close()
    
    if patch_args is not None:
        patch = patch_from_polygon(fp, **patch_args)
        return patch
    else:
        return fp


def make_maximal_wcs(files, pixel_scale=0.1, get_hdu=True, pad=90, verbose=True, theta=0, poly_buffer=1./3600, nsci_extensions=4):
    """
    Compute an ImageHDU with a footprint that contains all of `files`

    Parameters
    ----------
    files : list
        List of HST FITS files (e.g., FLT.) or WCS objects.

    pixel_scale : float
        Pixel scale of output WCS, in `~astropy.units.arcsec`.

    get_hdu : bool
        See below.

    pad : float
        Padding to add to the total image size, in `~astropy.units.arcsec`.

    theta : float
        Position angle, degrees

    nsci_extensions : int
        Number of 'SCI' extensions to try in the exposure files.

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        If `get_hdu` is True.

    -or-

    header, wcs : `~astropy.io.fits.Header`, `~astropy.wcs.WCS`
        If `get_hdu` is False.

    """
    import numpy as np
    from shapely.geometry import Polygon

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    if isinstance(files[0], pywcs.WCS):
        # Already wcs_list
        wcs_list = [(wcs, 'WCS', -1) for wcs in files]
    else:
        wcs_list = []
        for i, file in enumerate(files):
            if not os.path.exists(file):
                continue

            with pyfits.open(file) as im:
                for ext in range(nsci_extensions):
                    if ('SCI', ext+1) not in im:
                        continue

                    wcs = pywcs.WCS(im['SCI', ext+1].header, fobj=im)
                    wcs_list.append((wcs, file, ext))

    group_poly = None
    for i, (wcs, file, chip) in enumerate(wcs_list):
        p_i = Polygon(wcs.calc_footprint())
        if group_poly is None:
            if poly_buffer > 0:
                group_poly = p_i.buffer(1./3600)
            else:
                group_poly = p_i
        else:
            if poly_buffer > 0:
                group_poly = group_poly.union(p_i.buffer(1./3600))
            else:
                group_poly = group_poly.union(p_i)
                 
        x0, y0 = np.cast[float](group_poly.centroid.xy)[:, 0]
        if verbose:
            print('{0:>3d}/{1:>3d}: {2}[SCI,{3}]  {4:>6.2f}'.format(i, len(files), file, chip+1, group_poly.area*3600*np.cos(y0/180*np.pi)))

    px = np.cast[float](group_poly.convex_hull.boundary.xy).T
    #x0, y0 = np.cast[float](group_poly.centroid.xy)[:,0]

    x0 = (px.max(axis=0)+px.min(axis=0))/2.

    cosd = np.array([np.cos(x0[1]/180*np.pi), 1])

    _mat = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

    # Rotated
    pr = ((px-x0)*cosd).dot(_mat)/cosd+x0

    size_arcsec = (pr.max(axis=0)-pr.min(axis=0))*cosd*3600
    sx, sy = size_arcsec

    # sx = (px.max()-px.min())*cosd*3600 # arcsec
    # sy = (py.max()-py.min())*3600 # arcsec

    size = np.maximum(sx+pad, sy+pad)

    if verbose:
        print('\n  Mosaic WCS: ({0:.5f},{1:.5f})  {2:.1f}\'x{3:.1f}\'  {4:.3f}"/pix\n'.format(x0[0], x0[1], (sx+pad)/60., (sy+pad)/60., pixel_scale))

    out = make_wcsheader(ra=x0[0], dec=x0[1], size=(sx+pad*2, sy+pad*2), pixscale=pixel_scale, get_hdu=get_hdu, theta=theta/np.pi*180)

    return out


def half_pixel_scale(wcs):
    """
    Create a new WCS with half the pixel scale of another that can be 
    block-averaged 2x2
    
    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        Input WCS
    
    Returns
    -------
    half_wcs : `~astropy.wcs.WCS`
        New WCS with smaller pixels
    """
    h = to_header(wcs)
    
    for k in ['NAXIS1', 'NAXIS2']: #, 'CRPIX1', 'CRPIX2']:
        h[k] *= 2

    for k in ['CRPIX1', 'CRPIX2']:
        h[k] = h[k]*2 - 0.5
        
    for k in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
        if k in h:
            h[k] /= 2
    
    if 0:
        # Test
        new = pywcs.WCS(h)
        sh = new.pixel_shape
        
        wcorner = wcs.all_world2pix(new.all_pix2world([[-0.5, -0.5], 
                                             [sh[0]-0.5, sh[1]-0.5]], 0),0)
        print('small > large')
        print(', '.join([f'{w:.2f}' for w in wcorner[0]]))
        print(', '.join([f'{w:.2f}' for w in wcorner[1]]), wcs.pixel_shape)
        
        sh = wcs.pixel_shape
        wcorner = new.all_world2pix(wcs.all_pix2world([[-0.5, -0.5], 
                                             [sh[0]-0.5, sh[1]-0.5]], 0),0)
        print('large > small')
        print(', '.join([f'{w:.2f}' for w in wcorner[0]]))
        print(', '.join([f'{w:.2f}' for w in wcorner[1]]), new.pixel_shape)
        
    new_wcs = pywcs.WCS(h, relax=True)
    
    return new_wcs


def header_keys_from_filelist(fits_files, keywords=[], ext=0, colname_case=str.lower):
    """Dump header keywords to a `~astropy.table.Table`

    Parameters
    ----------
    fits_files : list
        List of FITS filenames

    keywords : list or None
        List of header keywords to retrieve.  If `None`, then generate a list
        of *all* keywords from the first file in the list.

    ext : int, tuple
        FITS extension from which to pull the header.  Can be integer or
        tuple, e.g., ('SCI',1) for HST ACS/WFC3 FLT files.

    colname_case : func
        Function to set the case of the output colnames, e.g., `str.lower`,
        `str.upper`, `str.title`.

    Returns
    -------
    tab : `~astropy.table.Table`
        Output table.

    """
    import numpy as np
    import astropy.io.fits as pyfits
    from astropy.table import Table

    # If keywords=None, get full list from first FITS file
    if keywords is None:
        h = pyfits.getheader(fits_files[0], ext)
        keywords = list(np.unique(list(h.keys())))
        keywords.pop(keywords.index(''))
        keywords.pop(keywords.index('HISTORY'))

    # Loop through files
    lines = []
    for file in fits_files:
        line = [file]
        h = pyfits.getheader(file, ext)
        for key in keywords:
            if key in h:
                line.append(h[key])
            else:
                line.append(None)

        lines.append(line)

    # Column names
    table_header = [colname_case(key) for key in ['file']+keywords]

    # Output table
    tab = Table(data=np.array(lines), names=table_header)

    return tab


def parse_s3_url(url='s3://bucket/path/to/file.txt'):
    """
    Parse s3 path string
    
    Parameters
    ----------
    url : str
        Full S3 path, e.g., ``[s3://]{bucket_name}/{s3_object}``
    
    Returns
    -------
    bucket_name : str
        Bucket name
    
    s3_object : str
        Full path of the S3 file object
    
    filename : str
        File name of the object, e.g. ``os.path.basename(s3_object)``
        
    """
    surl = url.strip('s3://')
    spl = surl.split('/')
    if len(spl) < 2:
        print(f"bucket / path not found in {url}")
        return None, None, None
        
    bucket_name = spl[0]
    s3_object = '/'.join(spl[1:])
    filename = os.path.basename(s3_object)
    return bucket_name, s3_object, filename


def fetch_s3_url(url='s3://bucket/path/to/file.txt', file_func=lambda x : os.path.join('./',x), skip_existing=True, verbose=True):
    """
    Fetch file from an S3 bucket
    
    Parameters
    ----------
    url : str
        S3 url of a file to download
    
    file_func : function
        Function applied to the file name extracted from `url`, e.g., to 
        set output directory, rename files, set a prefix, etc.
    
    Returns
    -------
    local_file : str
        Name of local file or `None` if failed to parse `url`
    
    status : int
        Bit flag of results: **1** == file found, **2** = download successful
        
    """
    import traceback
    import boto3
    import botocore.exceptions
    
    s3 = boto3.resource('s3')
    bucket_name, s3_object, filename = parse_s3_url(url=url)
    if bucket_name is None:
        return url, os.path.exists(url)
        
    bkt = s3.Bucket(bucket_name)
    local_file = file_func(filename)
    status = os.path.exists(local_file)*1
    
    if (status > 0) & skip_existing:
        print(f'{local_file} exists, skipping.')
    else:

        try:
            bkt.download_file(s3_object, local_file,
                      ExtraArgs={"RequestPayer": "requester"})
            status += 2
            if verbose:
                print(f'{url} > {local_file}')
                
        except botocore.exceptions.ClientError:
            trace = traceback.format_exc(limit=2)
            msg = trace.split('\n')[-2].split('ClientError: ')[1]
            if verbose:
                print(f'Failed {url}: {msg}')
                
            # Download failed due to a ClientError
            # Forbidden probably means insufficient bucket access privileges
            pass
            
    return local_file, status


def niriss_ghost_mask(im, init_thresh=0.05, init_sigma=3, final_thresh=0.01, final_sigma=3, erosions=0, dilations=9, verbose=True, **kwargs):
    """
    Make a mask for NIRISS imaging ghosts
    
    See also Martel. JWST-STScI-004877 and
    https://github.com/spacetelescope/niriss_ghost
    
    Here
    """
    import scipy.ndimage as nd
    
    if im[0].header['PUPIL'] not in ['F115W','F150W','F200W']:
        return False
    
    if im[0].header['PUPIL'] == 'F115W':
        xgap, ygap = 1156, 927
    elif im[0].header['PUPIL'] == 'F115W':
        xgap, ygap = 1162, 938
    else:
        xgap, ygap = 1156, 944-2
        
    yp, xp = np.indices((2048, 2048))
    
    yg = 2*(ygap-1) - yp
    xg = 2*(xgap-1) - xp

    dx = (xp - xgap)
    dy = (yp - ygap)

    in_img = (xg >= 0) & (xg < 2048)
    in_img &= (yg >= 0) & (yg < 2048)

    in_img &= np.abs(dx) < 400
    in_img &= np.abs(dy) < 400
    
    if 'MDRIZSKY' in im['SCI'].header:
        bkg = im['SCI'].header['MDRIZSKY']
    else:
        bkg = np.nanmedian(im['SCI'].data[im['DQ'].data == 0])
        
    thresh = (im['SCI'].data - bkg)*init_thresh > init_sigma*im['ERR'].data
    thresh &= in_img

    _reflected = np.zeros_like(im['SCI'].data)
    for xpi, ypi, xgi, ygi in zip(xp[thresh], yp[thresh],
                                  xg[thresh], yg[thresh]):
                                  
        _reflected[ygi, xgi] = im['SCI'].data[ypi, xpi] - bkg

    ghost_mask = _reflected*final_thresh > final_sigma*im['ERR'].data
    
    if erosions > 0:
        ghost_mask = nd.binary_erosion(ghost_mask, iterations=erosions)
        
    ghost_mask = nd.binary_dilation(ghost_mask, iterations=dilations)
    
    im[0].header['GHOSTMSK'] = True, 'NIRISS ghost mask applied'
    im[0].header['GHOSTNPX'] = ghost_mask.sum(), 'Pixels in NIRISS ghost mask'
    
    msg = 'NIRISS ghost mask {0} Npix: {1}\n'.format(im[0].header['PUPIL'], 
                                                    ghost_mask.sum())
    log_comment(LOGFILE, msg, verbose=verbose)
    
    return ghost_mask


def get_photom_scale(header):
    """
    Get tabulated scale factor
    
    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Image header
    
    Returns
    -------
    key : str
        Detector + filter key
    
    scale : float
        Scale value.  If `key` not found in the ``data/photom_correction.yml``
        table or if the CTX is newer than that indicated in the correction
        table then return 1.0.
        
    """
    import yaml
    if 'TELESCOP' in header:
        if header['TELESCOP'] not in ['JWST']:
            return header['TELESCOP'], 1.0
    else:
        return None, 1.0
        
    corr_file = os.path.join(os.path.dirname(__file__), 
                             'data/photom_correction.yml')
    
    if not os.path.exists(corr_file):
        return None, 1
    
    with open(corr_file) as fp:
        corr = yaml.load(fp, Loader=yaml.SafeLoader)
    
    print(header['CRDS_CTX'], corr['CRDS_CTX_MAX'], corr_file)
    
    if 'CRDS_CTX' in header:
        if header['CRDS_CTX'] > corr['CRDS_CTX_MAX']:
            return header['CRDS_CTX'], 1.0
            
    key = '{0}-{1}'.format(header['DETECTOR'], header['FILTER'])
    if 'PUPIL' in header:
        key += '-{0}'.format(header['PUPIL'])
    
    if key not in corr:
        return key, 1.0
    
    else:
        return key, 1./corr[key]


def drizzle_from_visit(visit, output, pixfrac=1., kernel='point',
                       clean=True, include_saturated=True, keep_bits=None,
                       dryrun=False, skip=None, extra_wfc3ir_badpix=True,
                       verbose=True,
                       scale_photom=True,
                       niriss_ghost_kwargs={}):
    """
    Make drizzle mosaic from exposures in a visit dictionary
    
    Parameters
    ----------
    visit : dict
        Visit dictionary with 'product' and 'files' keys
    
    output : `~astropy.wcs.WCS`, `~astropy.io.fits.Header`, `~astropy.io.ImageHDU`
        Output frame definition.  Can be a WCS object, header, or FITS HDU
    
    pixfrac : float
        Drizzle `pixfrac`
    
    kernel : str
        Drizzle `kernel` (e.g., 'point', 'square')
    
    clean : bool
        Remove exposure files after adding to the mosaic
    
    include_saturated : bool
        Include pixels with saturated DQ flag
    
    keep_bits : int, None
        Extra DQ bits to keep as valid
    
    dryrun : bool
        If True, don't actually produce the output
    
    skip : int
        Slice skip to drizzle a subset of exposures
    
    extra_wfc3ir_badpix : bool
        Apply extra WFC3/IR bad pix to DQ
    
    verbose : bool
        Some verbose message printing
    
    scale_photom : bool
        For JWST, apply photometry scale corrections from the  
        `grizli/data/photom_correction.yml` table
    
    niriss_ghost_kwargs : dict
        Keyword arguments for `~grizli.utils.niriss_ghost_mask`
    
    Returns
    -------
    outsci : array-like
        SCI array
    
    outwht : array-like
        Inverse variance WHT array
    
    header : `~astropy.io.fits.Header`
        Image header
    
    flist : list
        List of files that were drizzled to the mosaic
        
    wcs_tab : `~astropy.table.Table`
        Table of WCS parameters of individual exposures
    
    """
    from shapely.geometry import Polygon
    import boto3
    from botocore.exceptions import ClientError
    import scipy.ndimage as nd
    
    from .version import __version__ as grizli__version
    
    bucket_name = None
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')

    if isinstance(output, pywcs.WCS):
        outputwcs = output
    elif isinstance(output, pyfits.Header):
        outputwcs = pywcs.WCS(output)
    elif isinstance(output, pyfits.PrimaryHDU) | isinstance(output, pyfits.ImageHDU):
        outputwcs = pywcs.WCS(output.header)
    else:
        return None

    if not hasattr(outputwcs, '_naxis1'):
        outputwcs._naxis1, outputwcs._naxis2 = outputwcs._naxis

    outputwcs.pscale = get_wcs_pscale(outputwcs)

    output_poly = Polygon(outputwcs.calc_footprint())
    count = 0

    ref_photflam = None

    indices = []

    for i in range(len(visit['files'])):
        olap = visit['footprints'][i].intersection(output_poly)
        if olap.area > 0:
            indices.append(i)

    if skip is not None:
        indices = indices[::skip]

    NTOTAL = len(indices)
    
    wcs_rows = []
    wcs_colnames = None
    wcs_keys = {}
    
    bpdata = 0
    
    for i in indices:

        file = visit['files'][i]

        msg = '\n({0:4d}/{1:4d}) Add exposure {2}\n'
        msg = msg.format(count+1, NTOTAL, file)
        log_comment(LOGFILE, msg, verbose=verbose)

        if dryrun:
            continue

        if not os.path.exists(file):
            bucket_i = visit['awspath'][i].split('/')[0]
            if bucket_name != bucket_i:
                bucket_name = bucket_i
                bkt = s3.Bucket(bucket_name)

            s3_path = '/'.join(visit['awspath'][i].split('/')[1:])
            remote_file = os.path.join(s3_path, file)

            print('  (fetch from s3://{0}/{1})'.format(bucket_i, remote_file))

            try:
                bkt.download_file(remote_file, file,
                              ExtraArgs={"RequestPayer": "requester"})
            except ClientError:
                print('  (failed s3://{0}/{1})'.format(bucket_i, remote_file))
                continue

        try:
            flt = pyfits.open(file)
        except OSError:
            print(f'open({file}) failed!')
            continue
            
        sci_list, wht_list, wcs_list = [], [], []
        
        keys = OrderedDict()
        for k in ['EXPTIME', 'TELESCOP', 'FILTER','FILTER1', 'FILTER2', 
                  'PUPIL', 'DETECTOR', 'INSTRUME', 'PHOTFLAM', 'PHOTPLAM', 
                  'PHOTFNU', 'PHOTZPT', 'PHOTBW', 'PHOTMODE', 'EXPSTART', 
                  'EXPEND', 'DATE-OBS', 'TIME-OBS',
                  'UPDA_CTX', 'CRDS_CTX', 'R_DISTOR', 'R_PHOTOM', 'R_FLAT']:
            if k in flt[0].header:
                keys[k] = flt[0].header[k]
        
        if flt[0].header['TELESCOP'] in ['JWST']:
            bits = 4
            include_saturated = False
            
            #bpdata = 0
            _inst = flt[0].header['INSTRUME']
            if (extra_wfc3ir_badpix) & (_inst in ['NIRCAM']):
                _det = flt[0].header['DETECTOR']
                bpfiles = [os.path.join(os.path.dirname(__file__), 
                           f'data/nrc_badpix_230120_{_det}.fits.gz')]
                bpfiles += [os.path.join(os.path.dirname(__file__), 
                           f'data/nrc_lowpix_0916_{_det}.fits.gz')]
                
                for bpfile in bpfiles:
                    if os.path.exists(bpfile):
                        bpdata = pyfits.open(bpfile)[0].data
                        bpdata = nd.binary_dilation(bpdata > 0)*1024
                        msg = f'Use extra badpix in {bpfile}'
                        log_comment(LOGFILE, msg, verbose=verbose)
                        break
            else:
                bpdata = np.zeros(flt['SCI'].data.shape, dtype=int)
                
            # NIRISS ghost mask
            if (_inst in ['NIRISS']) & (niriss_ghost_kwargs is not None):
                if 'verbose' not in niriss_ghost_kwargs:
                    niriss_ghost_kwargs['verbose'] = verbose
                
                _ghost = niriss_ghost_mask(flt, **niriss_ghost_kwargs)
                bpdata |= _ghost*1024
            
            # Negative
            if 'MDRIZSKY' in flt['SCI'].header:
                _low = ((flt['SCI'].data - flt['SCI'].header['MDRIZSKY']) < 
                      -5*flt['ERR'].data)
                msg = f'Extra -5 sigma low pixels: N= {_low.sum()} '
                msg += f' ( {_low.sum()/_low.size*100:.1} %)'
                log_comment(LOGFILE, msg, verbose=verbose)
                bpdata |= _low*1024
            
        elif flt[0].header['DETECTOR'] == 'IR':
            bits = 576
            if extra_wfc3ir_badpix:
                if (i == indices[0]) | (not hasattr(bpdata, 'shape')):
                    bpfile = os.path.join(os.path.dirname(__file__), 
                               'data/wfc3ir_badpix_spars200_22.03.31.fits.gz')
                    bpdata = pyfits.open(bpfile)[0].data
                    
                msg = f'Use extra badpix in {bpfile}'
                log_comment(LOGFILE, msg, verbose=verbose)
        else:
            bits = 64+32
            bpdata = 0
            
        if include_saturated:
            bits |= 256

        if keep_bits is not None:
            bits |= keep_bits
        
        if scale_photom:
            _key, _scale_photom = get_photom_scale(flt[0].header)
        else:
            _scale_photom = 1.0
        
        if 'PHOTFLAM' in keys:
            msg = '  0    PHOTFLAM={0:.2e}, scale={1:.3f}'
            msg = msg.format(keys['PHOTFLAM'], _scale_photom)
            log_comment(LOGFILE, msg, verbose=verbose)
            
            if ref_photflam is None:
                ref_photflam = keys['PHOTFLAM']
                    
        for ext in [1, 2, 3, 4]:
            if ('SCI', ext) in flt:

                h = flt[('SCI', ext)].header
                if 'MDRIZSKY' in h:
                    sky = h['MDRIZSKY']
                else:
                    sky = 0
                
                msg = '  ext (SCI,{0}), sky={1:.3f}'.format(ext, sky)
                log_comment(LOGFILE, msg, verbose=verbose)
                
                if h['BUNIT'] == 'ELECTRONS':
                    to_per_sec = 1./keys['EXPTIME']
                else:
                    to_per_sec = 1.

                phot_scale = to_per_sec * _scale_photom

                if 'PHOTFLAM' in h:
                    if ref_photflam is None:
                        ref_photflam = h['PHOTFLAM']

                    phot_scale = h['PHOTFLAM'] / ref_photflam * _scale_photom
                    
                    if 'PHOTFNU' not in h:
                        h['PHOTFNU'] = (photfnu_from_photflam(h['PHOTFLAM'],
                                                             h['PHOTPLAM']), 
                                        'Inverse sensitivity, Jy/DN')
                                                             
                    msg = '       PHOTFLAM={0:.2e}, scale={1:.3f}'
                    msg = msg.format(h['PHOTFLAM'], phot_scale)
                    log_comment(LOGFILE, msg, verbose=verbose)
                    
                    keys['PHOTFLAM'] = h['PHOTFLAM']
                    for k in ['PHOTFLAM', 'PHOTPLAM', 'PHOTFNU',
                              'PHOTZPT', 'PHOTBW', 'PHOTMODE',
                              'PHOTMJSR', 'PIXAR_SR']:
                        if k in h:
                            keys[k] = h[k]

                    phot_scale *= to_per_sec
                    
                try:
                    wcs_i = pywcs.WCS(header=flt[('SCI', ext)].header, 
                                      fobj=flt)
                    wcs_i.pscale = get_wcs_pscale(wcs_i)
                except KeyError:
                    print(f'Failed to initialize WCS on {file}[SCI,{ext}]')
                    continue
                
                wcsh = to_header(wcs_i)
                row = [file, ext, keys['EXPTIME']]
                
                if wcs_colnames is None:
                    wcs_colnames = ['file','ext','exptime']
                    for k in wcsh:
                        wcs_colnames.append(k.lower())
                        wcs_keys[k.lower()] = wcsh[k]
                        
                for k in wcs_colnames[3:]:
                    ku = k.upper()
                    if ku not in wcsh:
                        print(f'Keyword {ku} not found in WCS header')
                        row.append(wcs_keys[k]*0)
                    else:
                        row.append(wcsh[ku])
                        
                for k in wcsh:
                    if k.lower() not in wcs_colnames:
                        print(f'Extra keyword {ku} found in WCS header')
                
                wcs_rows.append(row)
                
                sci_list.append((flt[('SCI', ext)].data - sky)*phot_scale)

                err = flt[('ERR', ext)].data*phot_scale
                dq = unset_dq_bits(flt[('DQ', ext)].data, bits) | bpdata
                wht = 1/err**2
                wht[(err == 0) | (dq > 0)] = 0

                wht_list.append(wht)

                # wcs_i = HSTWCS(fobj=flt, ext=('SCI',ext), minerr=0.0,
                #                wcskey=' ')
                if not hasattr(wcs_i, 'pixel_shape'):
                    wcs_i.pixel_shape = wcs_i._naxis1, wcs_i._naxis2

                if not hasattr(wcs_i, '_naxis1'):
                    wcs_i._naxis1, wcs_i._naxis2 = wcs_i._naxis

                wcs_list.append(wcs_i)

        if count == 0:
            res = drizzle_array_groups(sci_list, wht_list, wcs_list,
                                     outputwcs=outputwcs,
                                     scale=0.1, kernel=kernel,
                                     pixfrac=pixfrac, calc_wcsmap=False,
                                     verbose=verbose, data=None)

            outsci, outwht, outctx, header, xoutwcs = res
            header['EXPTIME'] = flt[0].header['EXPTIME']
            header['NDRIZIM'] = 1
            header['PIXFRAC'] = pixfrac
            header['KERNEL'] = kernel
            header['OKBITS'] = (bits, "FLT bits treated as valid")
            header['PHOTSCAL'] = _scale_photom, 'Scale factor applied'
            
            header['GRIZLIV'] = grizli__version, 'Grizli code version'
            
            for k in keys:
                header[k] = keys[k]

        else:
            data = outsci, outwht, outctx
            res = drizzle_array_groups(sci_list, wht_list, wcs_list,
                                     outputwcs=outputwcs,
                                     scale=0.1, kernel=kernel,
                                     pixfrac=pixfrac, calc_wcsmap=False,
                                     verbose=verbose, data=data)

            outsci, outwht, outctx = res[:3]
            header['EXPTIME'] += flt[0].header['EXPTIME']
            header['NDRIZIM'] += 1

        count += 1
        header['FLT{0:05d}'.format(count)] = file
        
        flt.close()
        
        #xfiles = glob.glob('*')
        #print('Clean: ', clean, xfiles)
        if clean:
            os.remove(file)

    if 'awspath' in visit:
        awspath = visit['awspath']
    else:
        awspath = ['.' for f in visit['files']]

    if len(awspath) == 1:
        awspath = [awspath[0] for f in visit['files']]
    elif isinstance(awspath, str):
        _awspath = [awspath for f in visit['files']]
        awspath = _awspath

    flist = ['{0}/{1}'.format(awspath, visit['files'][i])
                for i in indices]
    
    if dryrun:
        return flist

    elif count == 0:
        return None

    else:        
        wcs_tab = GTable(names=wcs_colnames, rows=wcs_rows)
        
        outwht *= (wcs_i.pscale/outputwcs.pscale)**4
        return outsci, outwht, header, flist, wcs_tab


def drizzle_array_groups(sci_list, wht_list, wcs_list, outputwcs=None,
                         scale=0.1, kernel='point', pixfrac=1.,
                         calc_wcsmap=False, verbose=True, data=None):
    """Drizzle array data with associated wcs

    Parameters
    ----------
    sci_list, wht_list : list
        List of science and weight `~numpy.ndarray` objects.

    wcs_list : list

    scale : float
        Output pixel scale in arcsec.

    kernel, pixfrac : str, float
        Drizzle parameters

    verbose : bool
        Print status messages

    Returns
    -------
    outsci, outwht, outctx : `~numpy.ndarray`
        Output drizzled science, weight and context images

    header, outputwcs : `~astropy.fits.io.Header`, `~astropy.wcs.WCS`
        Drizzled image header and WCS.

    """
    from drizzlepac import adrizzle
    from drizzlepac import cdriz

    #from stsci.tools import logutil
    #log = logutil.create_logger(__name__)

    # Output header / WCS
    if outputwcs is None:
        #header, outputwcs = compute_output_wcs(wcs_list, pixel_scale=scale)
        header, outputwcs = make_maximal_wcs(wcs_list,
                                             pixel_scale=scale,
                                             verbose=False,
                                             pad=0,
                                             get_hdu=False)
    else:
        header = to_header(outputwcs)

    header['DRIZKERN'] = kernel, "Drizzle kernel"
    header['DRIZPIXF'] = pixfrac, "Drizzle pixfrac"

    if not hasattr(outputwcs, '_naxis1'):
        outputwcs._naxis1, outputwcs._naxis2 = outputwcs._naxis

    # Try to fix deprecated WCS
    for wcs_i in wcs_list:
        if not hasattr(wcs_i, 'pixel_shape'):
            wcs_i.pixel_shape = wcs_i._naxis1, wcs_i._naxis2

        if not hasattr(wcs_i, '_naxis1'):
            wcs_i._naxis1, wcs_i._naxis2 = wcs_i._naxis[:2]

    # Output WCS requires full WCS map?
    if calc_wcsmap < 2:
        ctype = outputwcs.wcs.ctype
        if '-SIP' in ctype[0]:
            print('Output WCS ({0}) requires `calc_wcsmap=2`'.format(ctype))
            calc_wcsmap = 2
        else:
            # Internal WCSMAP not required
            calc_wcsmap = 0

    shape = (header['NAXIS2'], header['NAXIS1'])

    # Output arrays
    if data is not None:
        outsci, outwht, outctx = data
    else:
        outsci = np.zeros(shape, dtype=np.float32)
        outwht = np.zeros(shape, dtype=np.float32)
        outctx = np.zeros(shape, dtype=np.int32)

    # Do drizzle
    N = len(sci_list)
    for i in range(N):
        if verbose:
            #log.info('Drizzle array {0}/{1}'.format(i+1, N))
            msg = 'Drizzle array {0}/{1}'.format(i+1, N)
            log_comment(LOGFILE, msg, verbose=verbose, show_date=True)

        if calc_wcsmap > 1:
            wcsmap = WCSMapAll  # (wcs_list[i], outputwcs)
            #wcsmap = cdriz.DefaultWCSMapping
        else:
            wcsmap = None

        adrizzle.do_driz(sci_list[i].astype(np.float32, copy=False),
                         wcs_list[i],
                         wht_list[i].astype(np.float32, copy=False),
                         outputwcs, outsci, outwht, outctx, 1., 'cps', 1,
                         wcslin_pscale=wcs_list[i].pscale, uniqid=1,
                         pixfrac=pixfrac, kernel=kernel, fillval='0',
                         wcsmap=wcsmap)

    return outsci, outwht, outctx, header, outputwcs


class WCSMapAll:
    """ Sample class to demonstrate how to define a coordinate transformation
    """

    def __init__(self, input, output, origin=0):
        # Verify that we have valid WCS input objects
        import copy
        self.checkWCS(input, 'Input')
        self.checkWCS(output, 'Output')

        self.input = input
        self.output = copy.deepcopy(output)
        #self.output = output

        self.origin = origin
        self.shift = None
        self.rot = None
        self.scale = None

    def checkWCS(self, obj, name):
        try:
            assert isinstance(obj, pywcs.WCS)
        except AssertionError:
            print(name + ' object needs to be an instance or subclass of a PyWCS object.')
            raise

    def forward(self, pixx, pixy):
        """ Transform the input pixx,pixy positions in the input frame
            to pixel positions in the output frame.

            This method gets passed to the drizzle algorithm.
        """
        # This matches WTRAXY results to better than 1e-4 pixels.
        skyx, skyy = self.input.all_pix2world(pixx, pixy, self.origin)
        result = self.output.all_world2pix(skyx, skyy, self.origin)
        return result

    def backward(self, pixx, pixy):
        """ Transform pixx,pixy positions from the output frame back onto their
            original positions in the input frame.
        """
        skyx, skyy = self.output.all_pix2world(pixx, pixy, self.origin)
        result = self.input.all_world2pix(skyx, skyy, self.origin)
        return result

    def get_pix_ratio(self):
        """ Return the ratio of plate scales between the input and output WCS.
            This is used to properly distribute the flux in each pixel in 'tdriz'.
        """
        return self.output.pscale / self.input.pscale

    def xy2rd(self, wcs, pixx, pixy):
        """ Transform input pixel positions into sky positions in the WCS provided.
        """
        return wcs.all_pix2world(pixx, pixy, 1)

    def rd2xy(self, wcs, ra, dec):
        """ Transform input sky positions into pixel positions in the WCS provided.
        """
        return wcs.all_world2pix(ra, dec, 1)


def compute_output_wcs(wcs_list, pixel_scale=0.1, max_size=10000):
    """
    Compute output WCS that contains the full list of input WCS

    Parameters
    ----------
    wcs_list : list
        List of individual `~astropy.wcs.WCS` objects.

    pixel_scale : type
        Pixel scale of the output WCS

    max_size : int
        Maximum size out the output image dimensions

    Returns
    -------
    header : `~astropy.io.fits.Header`
        WCS header

    outputwcs : `~astropy.wcs.WCS`
        Output WCS

    """
    from shapely.geometry import Polygon

    footprint = Polygon(wcs_list[0].calc_footprint())
    for i in range(1, len(wcs_list)):
        fp_i = Polygon(wcs_list[i].calc_footprint())
        footprint = footprint.union(fp_i)

    x, y = footprint.convex_hull.boundary.xy
    x, y = np.array(x), np.array(y)

    # center
    crval = np.array(footprint.centroid.xy).flatten()

    # dimensions in arcsec
    xsize = (x.max()-x.min())*np.cos(crval[1]/180*np.pi)*3600
    ysize = (y.max()-y.min())*3600

    xsize = np.minimum(xsize, max_size*pixel_scale)
    ysize = np.minimum(ysize, max_size*pixel_scale)

    header, outputwcs = make_wcsheader(ra=crval[0], dec=crval[1],
                     size=(xsize, ysize),
                     pixscale=pixel_scale,
                     get_hdu=False,
                     theta=0)

    return header, outputwcs


def symlink_templates(force=False):
    """Symlink templates from module to $GRIZLI/templates as part of the initial setup
    Parameters
    ----------
    force : bool
        Force link files even if they already exist.
    """
    # if 'GRIZLI' not in os.environ:
    #     print('"GRIZLI" environment variable not set!')
    #     return False

    module_path = os.path.dirname(__file__)
    templates_path = os.path.join(module_path, 'data/templates')

    out_path = os.path.join(GRIZLI_PATH, 'templates')

    if (not os.path.exists(out_path)) | force:
        if os.path.exists(out_path):  # (force)
            os.remove(out_path)

            os.symlink(templates_path, out_path)
            print('Symlink: {0} -> {1}'.format(templates_path, out_path))
    else:
        print('Templates directory exists: {0}'.format(out_path))
        print('Use `force=True` to force a new symbolic link.')


def fetch_acs_wcs_files(beams_file, bucket_name='grizli-v1'):
    """
    Fetch wcs files for a given beams.fits files
    """
    from urllib import request
    try:
        import boto3
        HAS_BOTO = True
    except:
        HAS_BOTO = False

    im = pyfits.open(beams_file)
    root = '_'.join(beams_file.split('_')[:-1])

    for i in range(len(im)):
        h = im[i].header
        if 'EXTNAME' not in h:
            continue

        if 'FILTER' not in h:
            continue

        if (h['EXTNAME'] != 'SCI') | (h['FILTER'] not in ['G800L']):
            continue

        ext = {1: 2, 2: 1}[h['CCDCHIP']]

        wcsfile = h['GPARENT'].replace('.fits', '.{0:02d}.wcs.fits'.format(ext))

        # Download the file with S3 or HTTP
        if not os.path.exists(wcsfile):
            print('Fetch {0} from {1}/Pipeline/{2}'.format(wcsfile,
                                                           bucket_name, root))

            if HAS_BOTO:
                s3 = boto3.resource('s3')
                s3_client = boto3.client('s3')
                bkt = s3.Bucket(bucket_name)

                s3_path = 'Pipeline/{0}/Extractions/{1}'.format(root, wcsfile)
                bkt.download_file(s3_path, './{0}'.format(wcsfile),
                                  ExtraArgs={"RequestPayer": "requester"})

            else:
                url = 'https://s3.amazonaws.com/{0}/'.format(bucket_name)
                url += 'Pipeline/{0}/Extractions/{1}'.format(root, wcsfile)

                print('Fetch WCS file: {0}'.format(url))
                req = request.urlretrieve(url, wcsfile)

    im.close()


def fetch_hst_calib(file='iref$uc72113oi_pfl.fits',  ftpdir='https://hst-crds.stsci.edu/unchecked_get/references/hst/', verbose=True, ref_paths={}, remove_corrupt=True):
    """
    TBD
    """
    import os

    ref_dir = file.split('$')[0]
    cimg = file.split('{0}$'.format(ref_dir))[1]
    
    if ref_dir in ref_paths:
        ref_path = ref_paths[ref_dir]
    else:
        ref_path = os.getenv(ref_dir)
        
    iref_file = os.path.join(ref_path, cimg)
    if not os.path.exists(iref_file):
        os.system('curl -o {0} {1}/{2}'.format(iref_file, ftpdir, cimg))
        if 'fits' in iref_file:
            try:
                _im =  pyfits.open(iref_file)
                _im.close()
            except:
                msg = ('Downloaded file {0} appears to be corrupt.\n'
                       'Check that {1}/{2} exists and is a valid file')

                print(msg.format(iref_file, ftpdir, cimg))
                if remove_corrupt:
                    os.remove(iref_file)

                return False
    else:
        if verbose:
            print('{0} exists'.format(iref_file))

    return iref_file


def fetch_hst_calibs(flt_file, ftpdir='https://hst-crds.stsci.edu/unchecked_get/references/hst/', calib_types=['BPIXTAB', 'CCDTAB', 'OSCNTAB', 'CRREJTAB', 'DARKFILE', 'NLINFILE', 'DFLTFILE','PFLTFILE', 'IMPHTTAB', 'IDCTAB', 'NPOLFILE'], verbose=True, ref_paths={}):
    """
    TBD
    Fetch necessary calibration files needed for running calwf3 from STScI FTP

    Old FTP dir: ftp://ftp.stsci.edu/cdbs/iref/"""
    import os

    im = pyfits.open(flt_file)
    if im[0].header['INSTRUME'] == 'ACS':
        ref_dir = 'jref'

    if im[0].header['INSTRUME'] == 'WFC3':
        ref_dir = 'iref'

    if im[0].header['INSTRUME'] == 'WFPC2':
        ref_dir = 'uref'

    if not os.getenv(ref_dir):
        print('No ${0} set!  Put it in ~/.bashrc or ~/.cshrc.'.format(ref_dir))
        return False

    calib_paths = []

    for ctype in calib_types:
        if ctype not in im[0].header:
            continue

        if verbose:
            print('Calib: {0}={1}'.format(ctype, im[0].header[ctype]))

        if im[0].header[ctype] == 'N/A':
            continue

        path = fetch_hst_calib(im[0].header[ctype], ftpdir=ftpdir,
                               verbose=verbose, ref_paths=ref_paths)
        calib_paths.append(path)
    
    im.close()
    
    return calib_paths


def mast_query_from_file_list(files=[], os_open=True):
    """
    Generate a MAST query on datasets in a list.
    """
    if len(files) == 0:
        files = glob.glob('*raw.fits')

    if len(files) == 0:
        print('No `files` specified.')
        return False

    datasets = np.unique([file[:6]+'*' for file in files]).tolist()
    URL = "http://archive.stsci.edu/hst/search.php?action=Search&"
    URL += "sci_data_set_name="+','.join(datasets)
    if os_open:
        os.system('open "{0}"'.format(URL))

    return URL


def fetch_default_calibs(get_acs=False, **kwargs):
    """
    Fetch a set of default HST calibration files 
    """
    paths = {}
    
    for ref_dir in ['iref', 'jref']:
        has_dir = True
        if not os.getenv(ref_dir):    
            has_dir = False        
            # Do directories exist in GRIZLI_PATH?
            if os.path.exists(os.path.join(GRIZLI_PATH, ref_dir)):
                has_dir = True
                paths[ref_dir] = os.path.join(GRIZLI_PATH, ref_dir)
        else:
            paths[ref_dir] = os.getenv(ref_dir)
            
        if not has_dir:
            print("""
No ${0} set!  Make a directory and point to it in ~/.bashrc or ~/.cshrc.
For example,

  $ mkdir $GRIZLI/{0}
  $ export {0}="${GRIZLI}/{0}/" # put this in ~/.bashrc
""".format(ref_dir))

            return False

    # WFC3
    files = ['iref$uc72113oi_pfl.fits',  # F105W Flat
             'iref$uc721143i_pfl.fits',  # F140W flat
             'iref$u4m1335li_pfl.fits',  # G102 flat
             'iref$u4m1335mi_pfl.fits',  # G141 flat
             'iref$w3m18525i_idc.fits',  # IDCTAB distortion table}
             ]
    
    if 'ACS' in kwargs:
        get_acs = kwargs['ACS']
        
    if get_acs:
        files.extend(['jref$n6u12592j_pfl.fits',  # F814 Flat
                      'jref$o841350mj_pfl.fits',  # G800L flat])
                      'jref$v971826jj_npl.fits'])

    for file in files:
        fetch_hst_calib(file, ref_paths=paths)

    badpix = os.path.join(paths['iref'], 'badpix_spars200_Nov9.fits')
    print('Extra WFC3/IR bad pixels: {0}'.format(badpix))
    if not os.path.exists(badpix):
        os.system('curl -o {0}/badpix_spars200_Nov9.fits https://raw.githubusercontent.com/gbrammer/wfc3/master/data/badpix_spars200_Nov9.fits'.format(paths['iref']))

    # Pixel area map
    pam = os.path.join(paths['iref'], 'ir_wfc3_map.fits')
    print('Pixel area map: {0}'.format(pam))
    if not os.path.exists(pam):
        os.system('curl -o {0} https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/data-analysis/pixel-area-maps/_documents/ir_wfc3_map.fits'.format(pam))


def fetch_wfpc2_calib(file='g6q1912hu_r4f.fits', path=os.getenv('uref'), use_mast=False, verbose=True, overwrite=True, skip_existing=True):
    """
    Fetch static WFPC2 calibration file and run `stsci.tools.convertwaiveredfits` on it.

    path : str
        Output path of the reference file (generally should be in $uref).

    use_mast : bool
        If True, try to fetch from "mast.stsci.edu//api/v0/download/file?uri",
        otherwise, fetch from a static directory
        "ssb.stsci.edu/cdbs_open/cdbs/uref_linux/".

    """
    from stsci.tools import convertwaiveredfits

    try:  # Python 3.x
        import http.client as httplib
    except ImportError:  # Python 2.x
        import httplib

    if file.endswith('h'):
        # File like "g6q1912hu.r4h"
        file = file[:-1].replace('.', '_')+'f.fits'

    outPath = os.path.join(path, file)
    if os.path.exists(outPath) & skip_existing:
        print("# fetch_wfpc2_calib: {0} exists".format(outPath))
        return True

    if use_mast:
        server = 'mast.stsci.edu'
        uri = 'mast:HST/product/'+file
        request_path = "/api/v0/download/file?uri="+uri
    else:
        server = 'ssb.stsci.edu'
        request_path = '/cdbs_open/cdbs/uref_linux/'+file

    if verbose:
        print('# fetch_wfpc2_calib: "{0}" to {1}'.format(server+request_path, path))

    conn = httplib.HTTPSConnection(server)

    conn.request("GET", request_path)
    resp = conn.getresponse()
    fileContent = resp.read()
    conn.close()

    # check for file
    if len(fileContent) < 4096:
        print('ERROR: "{0}" failed to download.  Try `use_mast={1}`.'.format(server+request_path, (use_mast is False)))
        status = False
        raise FileNotFoundError
    else:
        print("# fetch_wfpc2_calib: {0} (COMPLETE)".format(outPath))
        status = True

    # save to file
    with open(outPath, 'wb') as FLE:
        FLE.write(fileContent)

    if status:
        # Convert to standard FITS
        try:
            hdu = convertwaiveredfits.convertwaiveredfits(outPath)
            while 'HISTORY' in hdu[0].header:
                hdu[0].header.remove('HISTORY')

            hdu.writeto(outPath.replace('.fits', '_c0h.fits'),
                        overwrite=overwrite, output_verify='fix')
        except:
            return True


def fetch_nircam_skyflats():
    """
    Download skyflat files
    """
    conf_path = os.path.join(GRIZLI_PATH, 'CONF', 'NircamSkyFlat')
    os.system(f'aws s3 sync s3://grizli-v2/NircamSkyflats/ {conf_path} --exclude "*" --include "nrc*fits"')
    
    _files = glob.glob(conf_path+'/*fits')
    _files.sort()
    
    return _files


def fetch_config_files(get_acs=False, get_sky=True, get_stars=True, get_epsf=True, get_jwst=False, get_wfc3=True, **kwargs):
    """
    Config files needed for Grizli
    """
    if 'ACS' in kwargs:
        get_acs = kwargs['ACS']

    cwd = os.getcwd()

    print('Config directory: {0}/CONF'.format(GRIZLI_PATH))

    os.chdir(os.path.join(GRIZLI_PATH, 'CONF'))

    ftpdir = 'ftp://ftp.stsci.edu/cdbs/wfc3_aux/'
    tarfiles = []

    # Config files
    # BASEURL = 'https://s3.amazonaws.com/grizli/CONF/'
    # BASEURL = 'https://erda.ku.dk/vgrid/Gabriel%20Brammer/CONF/'
    BASEURL = ('https://raw.githubusercontent.com/gbrammer/' +
               'grizli-config/master')

    if get_wfc3:
        tarfiles = ['{0}/WFC3.IR.G102.cal.V4.32.tar.gz'.format(ftpdir),
                    '{0}/WFC3.IR.G141.cal.V4.32.tar.gz'.format(ftpdir)]
        tarfiles += [f'{BASEURL}/WFC3.IR.G102.WD.V4.32.tar.gz', 
                    f'{BASEURL}/WFC3.IR.G141.WD.V4.32.tar.gz']

    if get_jwst:
        tarfiles += [f'{BASEURL}/jwst-grism-conf.tar.gz', 
                     f'{BASEURL}/niriss.conf.220725.tar.gz',
                     f'{BASEURL}/nircam-wisp-aug2022.tar.gz']
    
    if get_sky:
        ftpdir = BASEURL
        tarfiles.append('{0}/grism_master_sky_v0.5.tar.gz'.format(ftpdir))

    #gURL = 'http://www.stsci.edu/~brammer/Grizli/Files'
    #gURL = 'https://s3.amazonaws.com/grizli/CONF'
    gURL = BASEURL
    
    tarfiles.append('{0}/WFC3IR_extended_PSF.v1.tar.gz'.format(gURL))

    if get_acs:
        tarfiles += [f'{BASEURL}/ACS.WFC.CHIP1.Stars.conf', 
                    f'{BASEURL}/ACS.WFC.CHIP2.Stars.conf']
        tarfiles.append('{0}/ACS.WFC.sky.tar.gz'.format(gURL))
        tarfiles.append('{0}/ACS_CONFIG.tar.gz'.format(gURL))

    for url in tarfiles:
        file = os.path.basename(url)
        if not os.path.exists(file):
            print('Get {0}'.format(file))
            os.system('curl -o {0} {1}'.format(file, url))

        if '.tar' in file:
            os.system('tar xzvf {0}'.format(file))

    if get_epsf:
        # ePSF files for fitting point sources
        #psf_path = 'http://www.stsci.edu/hst/wfc3/analysis/PSF/psf_downloads/wfc3_ir/'
        #psf_path = 'https://www.stsci.edu/~jayander/STDPSFs/WFC3IR/'
        #psf_root = 'PSFSTD'
        #psf_path = 'https://www.stsci.edu/~jayander/HST1PASS/'
        psf_path = 'https://www.stsci.edu/~jayander/HST1PASS/LIB/'
        psf_path += 'PSFs/STDPSFs/WFC3IR/'
        psf_root = 'STDPSF'
        
        ir_psf_filters = ['F105W', 'F125W', 'F140W', 'F160W']

        # New PSFs
        ir_psf_filters += ['F110W', 'F127M']

        files = ['{0}/{1}_WFC3IR_{2}.fits'.format(psf_path, psf_root, filt)
                 for filt in ir_psf_filters]

        for url in files:
            file = os.path.basename(url).replace('STDPSF', 'PSFSTD')
            if not os.path.exists(file):
                print('Get {0}'.format(file))
                os.system('curl -o {0} {1}'.format(file, url))
            else:
                print('File {0} exists'.format(file))

    if get_stars:
        # Stellar templates
        print('Templates directory: {0}/templates'.format(GRIZLI_PATH))
        os.chdir('{0}/templates'.format(GRIZLI_PATH))

        url = 'https://www.stsci.edu/~brammer/Grizli/Files/'
        files = [url+'stars_pickles.npy', url+'stars_bpgs.npy']

        for url in files:
            file = os.path.basename(url)
            if not os.path.exists(file):
                print('Get {0}'.format(file))
                os.system('curl -o {0} {1}'.format(file, url))
            else:
                print('File {0} exists'.format(file))

        print('ln -s stars_pickles.npy stars.npy')
        os.system('ln -s stars_pickles.npy stars.npy')

    os.chdir(cwd)


class MW_F99(object):
    """
    Wrapper around the `specutils.extinction` / `extinction` modules, which are called differently
    """

    def __init__(self, a_v, r_v=3.1):
        self.a_v = a_v
        self.r_v = r_v

        self.IS_SPECUTILS = False
        self.IS_EXTINCTION = False

        try:
            from specutils.extinction import ExtinctionF99
            self.IS_SPECUTILS = True
            self.F99 = ExtinctionF99(self.a_v, r_v=self.r_v)
        except(ImportError):
            try:
                from extinction import Fitzpatrick99
                self.IS_EXTINCTION = True
                self.F99 = Fitzpatrick99(r_v=self.r_v)

            except(ImportError):
                print("""
Couldn\'t find extinction modules in
`specutils.extinction` or
`extinction.Fitzpatrick99`.

MW extinction not implemented.
""")

        self.status = self.IS_SPECUTILS | self.IS_EXTINCTION

    def __call__(self, wave_input):
        import astropy.units as u

        if isinstance(wave_input, list):
            wave = np.array(wave_input)
        else:
            wave = wave_input

        if self.status is False:
            return np.zeros_like(wave)

        if self.IS_SPECUTILS:
            if hasattr(wave, 'unit'):
                wave_aa = wave
            else:
                wave_aa = wave*u.AA

            return self.F99(wave_aa)

        if self.IS_EXTINCTION:
            if hasattr(wave, 'unit'):
                wave_aa = wave.to(u.AA)
            else:
                wave_aa = wave

            return self.F99(wave_aa, self.a_v, unit='aa')


class EffectivePSF:
    def __init__(self):
        """Tools for handling WFC3/IR Effective PSF

        See documentation at http://www.stsci.edu/hst/wfc3/analysis/PSF.

        PSF files stored in $GRIZLI/CONF/
        """

        self.load_PSF_data()

    def load_PSF_data(self):
        """Load data from PSFSTD files

        Files should be located in ${GRIZLI}/CONF/ directory.
        """
        self.epsf = OrderedDict()

        for filter in ['F105W', 'F110W', 'F125W', 'F140W', 'F160W', 'F127M']:
            file = os.path.join(GRIZLI_PATH, 'CONF',
                                'PSFSTD_WFC3IR_{0}.fits'.format(filter))

            if not os.path.exists(file):
                continue

            with pyfits.open(file) as _im:
                data = _im[0].data.T*1
                data[data < 0] = 0

            self.epsf[filter] = data
        
        # UVIS
        filter_files = glob.glob(os.path.join(GRIZLI_PATH, 'CONF',
                            'PSFSTD_WFC3UV*.fits'))
        filter_files.sort()
        for file in filter_files:
            with pyfits.open(file, ignore_missing_end=True) as _im:
                data = _im[0].data.T*1
                data[data < 0] = 0
                
            filt = '_'.join(file.strip('.fits').split('_')[2:])
            self.epsf[filt+'U'] = data

        # ACS
        filter_files = glob.glob(os.path.join(GRIZLI_PATH, 'CONF',
                            'PSFSTD_ACSWFC*.fits'))
        filter_files.sort()
        for file in filter_files:
            with pyfits.open(file, ignore_missing_end=True) as _im:
                data = _im[0].data.T*1.
                data[data < 0] = 0
                
            filt = '_'.join(file.strip('.fits').split('_')[2:])
            self.epsf[filt] = data
        
        # JWST
        filter_files = glob.glob(os.path.join(GRIZLI_PATH, 'CONF/JWSTePSF',
                            'nircam*.fits'))
        filter_files += glob.glob(os.path.join(GRIZLI_PATH, 'CONF/JWSTePSF',
                            'niriss*.fits'))
        filter_files += glob.glob(os.path.join(GRIZLI_PATH, 'CONF/JWSTePSF',
                            'miri*.fits'))
        filter_files.sort()
        for file in filter_files:
            with pyfits.open(file, ignore_missing_end=True) as _im:
                data = _im[0].data*1 # [::-1,:,:]#[:,::-1,:]
                
                data[data < 0] = 0
                key = '{0}-{1}'.format(_im[0].header['DETECTOR'].upper(),
                                       _im[0].header['FILTER'])
            
                if 'LABEL' in _im[0].header:
                    key += '-' + _im[0].header['LABEL']
                
            self.epsf[key] = data
        
        # Dummy, use F105W ePSF for F098M and F110W
        self.epsf['F098M'] = self.epsf['F105W']
        self.epsf['F128N'] = self.epsf['F125W']
        self.epsf['F130N'] = self.epsf['F125W']
        self.epsf['F132N'] = self.epsf['F125W']
        
        # Dummy filters for IR grisms
        self.epsf['G141'] = self.epsf['F140W']
        self.epsf['G102'] = self.epsf['F105W']
        
        # Extended
        self.extended_epsf = {}
        for filter in ['F105W', 'F125W', 'F140W', 'F160W']:
            file = os.path.join(GRIZLI_PATH, 'CONF',
                                'extended_PSF_{0}.fits'.format(filter))

            if not os.path.exists(file):
                #BASEURL = 'https://erda.ku.dk/vgrid/Gabriel%20Brammer/CONF/'
                BASEURL = ('https://raw.githubusercontent.com/gbrammer/' +
                           'grizli-config/master')

                msg = 'Extended PSF file \'{0}\' not found.'.format(file)
                msg += 'Get the archive from '
                msg += f' {BASEURL}/WFC3IR_extended_PSF.v1.tar.gz'
                msg += ' and unpack in ${GRIZLI}/CONF/'
                raise FileNotFoundError(msg)

            with pyfits.open(file) as _im:
                data = _im[0].data*1
                data[data < 0] = 0

            # Mask center
            NX = data.shape[0]/2-1
            yp, xp = np.indices(data.shape)
            R = np.sqrt((xp-NX)**2+(yp-NX)**2)
            data[R <= 4] = 0.

            self.extended_epsf[filter] = data
            self.extended_N = int(NX)

        self.extended_epsf['F098M'] = self.extended_epsf['F105W']
        self.extended_epsf['F110W'] = self.extended_epsf['F105W']
        self.extended_epsf['F128N'] = self.extended_epsf['F125W']
        self.extended_epsf['F130N'] = self.extended_epsf['F125W']
        self.extended_epsf['F132N'] = self.extended_epsf['F125W']
        self.extended_epsf['G102'] = self.extended_epsf['F105W']
        self.extended_epsf['G141'] = self.extended_epsf['F140W']


    def get_at_position(self, x=507, y=507, filter='F140W', rot90=0):
        """Evaluate ePSF at detector coordinates
        TBD
        """
        epsf = self.epsf[filter]
        
        psf_type = 'HST/Optical'
        
        if filter in ['F098M', 'F110W', 'F105W', 'F125W', 'F140W', 'F160W',
                      'G102','G141','F128N','F130N','F132N']:
            psf_type = 'WFC3/IR'
            
        elif filter.startswith('NRC') | filter.startswith('NIS'):
            # NIRISS, NIRCam 2K
            psf_type = 'JWST/2K'
            
        elif filter.startswith('MIRI'):
            psf_type = 'JWST/MIRI'
        
        self.eval_psf_type = psf_type
        
        if psf_type == 'WFC3/IR':
            #  IR detector
            rx = 1+(np.clip(x, 1, 1013)-0)/507.
            ry = 1+(np.clip(y, 1, 1013)-0)/507.

            # zero index
            rx -= 1
            ry -= 1

            nx = np.clip(int(rx), 0, 2)
            ny = np.clip(int(ry), 0, 2)

            # print x, y, rx, ry, nx, ny

            fx = rx-nx
            fy = ry-ny

            psf_xy = (1-fx)*(1-fy)*epsf[:, :, nx+ny*3]
            psf_xy += fx*(1-fy)*epsf[:, :, (nx+1)+ny*3]
            psf_xy += (1-fx)*fy*epsf[:, :, nx+(ny+1)*3]
            psf_xy += fx*fy*epsf[:, :, (nx+1)+(ny+1)*3]
            
            self.eval_filter = filter
        
        if psf_type == 'JWST/MIRI':
            #  IR detector
            NDET = int(np.sqrt(epsf.shape[2]))
            
            rx = 1+(np.clip(x, 1, 1023)-0)/512.
            ry = 1+(np.clip(y, 1, 1023)-0)/512.

            # zero index
            rx -= 1
            ry -= 1

            # nx = np.clip(int(rx), 0, 2)
            # ny = np.clip(int(ry), 0, 2)
            nx = np.clip(int(rx), 0, NDET-1)
            ny = np.clip(int(ry), 0, NDET-1)
            
            # print x, y, rx, ry, nx, ny

            fx = rx-nx
            fy = ry-ny

            # psf_xy = (1-fx)*(1-fy)*epsf[:, :, nx+ny*3]
            # psf_xy += fx*(1-fy)*epsf[:, :, (nx+1)+ny*3]
            # psf_xy += (1-fx)*fy*epsf[:, :, nx+(ny+1)*3]
            # psf_xy += fx*fy*epsf[:, :, (nx+1)+(ny+1)*3]

            if NDET == 1:
                psf_xy = epsf[:,:,0]
            else:
                psf_xy = (1-fx)*(1-fy)*epsf[:, :, nx+ny*NDET]
                psf_xy += fx*(1-fy)*epsf[:, :, (nx+1)+ny*NDET]
                psf_xy += (1-fx)*fy*epsf[:, :, nx+(ny+1)*NDET]
                psf_xy += fx*fy*epsf[:, :, (nx+1)+(ny+1)*NDET]
            
            # psf_xy = np.rot90(psf_xy.T, 2)
            psf_xy = psf_xy.T
            
            self.eval_filter = filter
        
        if psf_type == 'JWST/2K':
            
            NDET = int(np.sqrt(epsf.shape[2]))
            
            #  IR detector
            rx = 1+(np.clip(x, 1, 2047)-0)/1024.
            ry = 1+(np.clip(y, 1, 2047)-0)/1024.

            # zero index
            rx -= 1
            ry -= 1

            nx = np.clip(int(rx), 0, NDET-1)
            ny = np.clip(int(ry), 0, NDET-1)

            # print x, y, rx, ry, nx, ny

            fx = rx-nx
            fy = ry-ny
            
            if NDET == 1:
                psf_xy = epsf[:,:,0]
            else:
                psf_xy = (1-fx)*(1-fy)*epsf[:, :, nx+ny*NDET]
                psf_xy += fx*(1-fy)*epsf[:, :, (nx+1)+ny*NDET]
                psf_xy += (1-fx)*fy*epsf[:, :, nx+(ny+1)*NDET]
                psf_xy += fx*fy*epsf[:, :, (nx+1)+(ny+1)*NDET]
            
            psf_xy = psf_xy.T
            # psf_xy = np.rot90(psf_xy.T, 2)
            
            self.eval_filter = filter
        
        elif psf_type == 'HST/Optical':

            sh = epsf.shape

            if sh[2] == 90:
                # ACS WFC
                iX, iY = 9, 10  # 9, 10
            else:
                # UVIS
                iX, iY = 7, 8

            rx = 1+(np.clip(x, 1, 4095)-0)/(4096/(iX-1))
            ry = 1+(np.clip(y, 1, 4095)-0)/(4096/(iY-1))

            # zero index
            rx -= 1
            ry -= 1

            nx = np.clip(np.cast[int](rx), 0, iX-1)
            ny = np.clip(np.cast[int](ry), 0, iY-1)

            # print x, y, rx, ry, nx, ny

            fx = rx-nx
            fy = ry-ny

            psf_xy = (1-fx)*(1-fy)*epsf[:, :, nx+ny*iX]
            psf_xy += fx*(1-fy)*epsf[:, :, (nx+1)+ny*iX]
            psf_xy += (1-fx)*fy*epsf[:, :, nx+(ny+1)*iX]
            psf_xy += fx*fy*epsf[:, :, (nx+1)+(ny+1)*iX]

            self.eval_filter = filter
        
        if rot90 != 0:
            self.psf_xy_rot90 = rot90
            psf_xy = np.rot90(psf_xy, rot90)
            
        return psf_xy

    def eval_ePSF(self, psf_xy, dx, dy, rot90=0, extended_data=None):
        """Evaluate PSF at dx,dy coordinates

        TBD
        """
        # So much faster than scipy.interpolate.griddata!
        from scipy.ndimage import map_coordinates

        # ePSF only defined to 12.5 pixels
        if self.eval_psf_type in ['WFC3/IR','HST/Optical']:
            ok = (np.abs(dx) <= 12.5) & (np.abs(dy) <= 12.5)
            coords = np.array([50+4*dx[ok], 50+4*dy[ok]])
        else:
            # JWST are +/- 32 pixels
            sh = psf_xy.shape
            _size = (sh[0]-1)//4
            _x0 = _size*2
            _cen = (_x0-1)//2
            ok = (np.abs(dx) <= _cen) & (np.abs(dy) <= _cen)
            coords = np.array([_x0+4*dx[ok], _x0+4*dy[ok]])

        # Do the interpolation
        interp_map = map_coordinates(psf_xy, coords, order=3)

        # Fill output data
        out = np.zeros_like(dx, dtype=np.float32)
        out[ok] = interp_map

        # Extended PSF
        if extended_data is not None:
            ok = (np.abs(dx) < self.extended_N) 
            ok &= (np.abs(dy) < self.extended_N)
            
            x0 = self.extended_N
            coords = np.array([x0+dy[ok]+0, x0+dx[ok]])
            interp_map = map_coordinates(extended_data, coords, order=0)
            out[ok] += interp_map

        return out

    @staticmethod
    def objective_epsf_center(params, self, psf_xy, sci, ivar, xp, yp, extended_data, ret, ds9):
        """Objective function for fitting ePSFs

        TBD

        params = [normalization, xc, yc, background]
        """
        from numpy.linalg import lstsq

        sh = sci.shape
        y0, x0 = np.array(sh)/2.-1

        dx = xp-params[0]
        dy = yp-params[1]

        ddx = xp  # -x0
        ddy = yp  # -y0

        ddx = ddx/ddx.max()
        ddy = ddy/ddy.max()

        # bkg = params[3] + params[4]*ddx + params[5]*ddy #+ params[6]*ddx*ddy

        psf_offset = self.eval_ePSF(psf_xy, dx, dy, extended_data=extended_data)  # *params[0]

        A = (np.array([psf_offset, np.ones_like(sci), ddx, ddy])*np.sqrt(ivar)).reshape((4, -1))
        scif = (sci*np.sqrt(ivar)).flatten()
        mask = (scif != 0)
        coeffs, _resid, _rank, _s = lstsq(A[:, mask].T, scif[mask], 
                                          rcond=LSTSQ_RCOND)
                                          
        resid = (scif - np.dot(coeffs, A))

        if ds9:
            Ax = (np.array([psf_offset, np.ones_like(sci), ddx, ddy])).reshape((4, -1))
            psf_model = np.dot(coeffs[:1], Ax[:1, :]).reshape(sci.shape)
            bkg = np.dot(coeffs[1:], Ax[1:, :]).reshape(sci.shape)
            ds9.view((sci-psf_model-bkg)*mask.reshape(sci.shape))

        if ret == 'resid':
            return resid
        elif ret == 'lm':
            # masked residuals for LM optimization
            if False:
                print(params, (resid**2).sum(), coeffs[0])

            return resid[resid != 0]
        elif ret == 'model':
            Ax = (np.array([psf_offset, np.ones_like(sci), ddx, ddy])).reshape((4, -1))
            psf_model = np.dot(coeffs[:1], Ax[:1, :]).reshape(sci.shape)
            bkg = np.dot(coeffs[1:], Ax[1:, :]).reshape(sci.shape)
            return psf_model, bkg, Ax, coeffs
        else:
            chi2 = (resid**2).sum()
            #print(params, chi2, coeffs[0])
            return chi2

    @staticmethod
    def objective_epsf(params, self, psf_xy, sci, ivar, xp, yp, extended_data, ret, ds9):
        """Objective function for fitting ePSFs

        TBD

        params = [normalization, xc, yc, background]
        """
        dx = xp-params[1]
        dy = yp-params[2]

        ddx = xp-xp.min()
        ddy = yp-yp.min()

        ddx = ddx/ddx.max()
        ddy = ddy/ddy.max()

        bkg = params[3] + params[4]*ddx + params[5]*ddy  # + params[6]*ddx*ddy

        psf_offset = self.eval_ePSF(psf_xy, dx, dy, extended_data=extended_data)*params[0]

        resid = (sci-psf_offset-bkg)*np.sqrt(ivar)

        if ds9:
            ds9.view(sci-psf_offset-bkg)

        if ret == 'resid':
            return resid
        elif ret == 'lm':
            # masked residuals for LM optimization
            if False:
                print(params, (resid**2).sum())

            return resid[resid != 0]
        elif ret == 'model':
            return psf_offset, bkg, None, None
        else:
            chi2 = (resid**2).sum()
            #print(params, chi2)
            return chi2

    def fit_ePSF(self, sci, center=None, origin=[0, 0], ivar=1, N=7,
                 filter='F140W', tol=1.e-4, guess=None, get_extended=False,
                 method='lm', ds9=None, psf_params=None, only_centering=True, 
                 rot90=0):
        """Fit ePSF to input data
        TBD
        """
        from scipy.optimize import minimize, least_squares

        sh = sci.shape
        if center is None:
            y0, x0 = np.array(sh)/2.-1
        else:
            x0, y0 = center

        xd = x0+origin[1]
        yd = y0+origin[0]

        xc, yc = int(x0), int(y0)

        psf_xy = self.get_at_position(x=xd, y=yd, filter=filter, rot90=rot90)

        yp, xp = np.indices(sh)

        if guess is None:
            if np.isscalar(ivar):
                ix = np.argmax(sci.flatten())
            else:
                ix = np.argmax((sci*(ivar > 0)).flatten())

            xguess = xp.flatten()[ix]
            yguess = yp.flatten()[ix]
        else:
            xguess, yguess = guess

        med_bkg = np.median(sci)

        if only_centering:
            # Only fit for centering and compute normalizations
            guess = [xguess, yguess]
            _objfun = self.objective_epsf_center
        else:
            # Fit for centering and normalization
            guess = [(sci-med_bkg)[yc-N:yc+N, xc-N:xc+N].sum(), xguess, yguess, med_bkg, 0, 0]
            _objfun = self.objective_epsf

        sly = slice(yc-N, yc+N)
        slx = slice(xc-N, xc+N)
        sly = slice(yguess-N, yguess+N)
        slx = slice(xguess-N, xguess+N)

        ivar_mask = np.zeros_like(sci)
        ivar_mask[sly, slx] = 1
        ivar_mask *= ivar

        if get_extended:
            if filter in self.extended_epsf:
                extended_data = self.extended_epsf[filter]
            else:
                extended_data = None
        else:
            extended_data = None

        # Get model
        if psf_params is not None:
            px = psf_params*1
            if len(px) == 2:
                _objfun = self.objective_epsf_center
                px[0] += x0
                px[1] += y0
            else:
                _objfun = self.objective_epsf
                px[1] += x0
                px[2] += y0

            args = (self, psf_xy, sci, ivar_mask, xp, yp, extended_data, 'model', ds9)
            psf_model, bkg, A, coeffs = _objfun(px, *args)
            return psf_model, bkg, A, coeffs

        if method == 'lm':
            args = (self, psf_xy, sci, ivar_mask, xp, yp, extended_data, 'lm', ds9)

            x_scale = 'jac'
            #x_scale = [guess[0], 1, 1, 10, 10, 10]
            #out = least_squares(_objfun, guess, args=args, method='trf', x_scale=x_scale, loss='huber')
            out = least_squares(_objfun, guess, args=args, method='lm', x_scale=x_scale, loss='linear')
            psf_params = out.x*1
        else:
            args = (self, psf_xy, sci, ivar_mask, xp, yp, extended_data, 'chi2', ds9)
            out = minimize(_objfun, guess, args=args, method=method, tol=tol)
            psf_params = out.x*1

        if len(guess) == 2:
            psf_params[0] -= x0
            psf_params[1] -= y0
        else:
            psf_params[1] -= x0
            psf_params[2] -= y0

        # if False:
        # 
        #     psf_fit = epsf.get_ePSF(psf_params, origin=origin,
        #                                filter=filter, shape=sh,
        #                                get_extended=get_extended)
        # 
        #     xargs = (self, psf_xy, sci, ivar_mask, xp, yp, extended_data, 'lm', None)
        #     lm = _objfun(out.x, *xargs)
        #     cargs = (self, psf_xy, sci, ivar_mask, xp, yp, extended_data, 'chi2', None)
        #     chi2 = _objfun(out.x, *cargs)

        return psf_params

        # dx = xp-psf_params[1]
        # dy = yp-psf_params[2]
        # output_psf = self.eval_ePSF(psf_xy, dx, dy)*psf_params[0]
        #
        # return output_psf, psf_params

    def get_ePSF(self, psf_params, sci=None, ivar=1, origin=[0, 0], shape=[20, 20], filter='F140W', get_extended=False, get_background=False, rot90=0):
        """
        Evaluate an Effective PSF
        """
        sh = shape
        y0, x0 = np.array(sh)/2.-1

        xd = x0+origin[1]
        yd = y0+origin[0]

        xc, yc = int(x0), int(y0)

        psf_xy = self.get_at_position(x=xd, y=yd, filter=filter, rot90=rot90)

        yp, xp = np.indices(sh)

        if len(psf_params) == 2:
            _objfun = self.objective_epsf_center
            dx = xp-psf_params[0]-x0
            dy = yp-psf_params[1]-y0
        else:
            _objfun = self.objective_epsf
            dx = xp-psf_params[1]-x0
            dy = yp-psf_params[2]-y0

        if get_extended:
            if filter in self.extended_epsf:
                extended_data = self.extended_epsf[filter]
            else:
                extended_data = None
        else:
            extended_data = None

        if sci is not None:
            ivar_mask = np.ones_like(sci)
            ivar_mask *= ivar
        else:
            sci = np.ones(sh, dtype=float)
            ivar_mask = sci*1

        args = (self, psf_xy, sci, ivar_mask, xp-x0, yp-y0, extended_data, 'model', None)
        output_psf, bkg, _a, _b = _objfun(psf_params, *args)

        #output_psf = self.eval_ePSF(psf_xy, dx, dy, extended_data=extended_data)*psf_params[0]

        if get_background:
            return output_psf, bkg
        else:
            return output_psf


def read_catalog(file, sextractor=False, format=None):
    """
    Wrapper around `~grizli.utils.Gtable.gread`.

    Auto-detects formats 'csv' and 'fits' and defaults to
    'ascii.commented_header'.
    """
    return GTable.gread(file, sextractor=sextractor, format=format)


class GTable(astropy.table.Table):
    """
    Extend `~astropy.table.Table` class with more automatic IO and other
    helper methods.
    """
    @classmethod
    def gread(cls, file, sextractor=False, format=None):
        """Assume `ascii.commented_header` by default

        Parameters
        ----------
        sextractor : bool
            Use `format='ascii.sextractor'`.

        format : None or str
            Override format passed to `~astropy.table.Table.read`.

        Returns
        -------
        tab : `~astropy.table.Table`
            Table object
        """
        import astropy.units as u

        if format is None:
            if sextractor:
                format = 'ascii.sextractor'
            elif isinstance(file, pyfits.BinTableHDU):
                format = 'fits'
            else:
                if file.endswith('.fits'):
                    format = 'fits'
                elif file.endswith('.csv'):
                    format = 'csv'
                elif file.endswith('.vot'):
                    format = 'votable'
                else:
                    format = 'ascii.commented_header'

        #print(file, format)
        tab = cls.read(file, format=format)

        return tab

    def gwrite(self, output, format='ascii.commented_header'):
        """Assume a format for the output table

        Parameters
        ----------
        output : str
            Output filename

        format : str
            Format string passed to `~astropy.table.Table.write`.

        """
        self.write(output, format=format)

    @staticmethod
    def parse_radec_columns(self, rd_pairs=None):
        """Parse column names for RA/Dec and set to `~astropy.units.degree` units if not already set

        Parameters
        ----------
        rd_pairs : `~collections.OrderedDict` or None
            Pairs of {ra:dec} names to search in the column list. If None,
            then uses the following by default.

                >>> rd_pairs = OrderedDict()
                >>> rd_pairs['ra'] = 'dec'
                >>> rd_pairs['ALPHA_J2000'] = 'DELTA_J2000'
                >>> rd_pairs['X_WORLD'] = 'Y_WORLD'

            NB: search is performed in order of ``rd_pairs.keys()`` and stops
            if/when a match is found.

        Returns
        -------
        rd_pair : [str, str]
            Column names associated with RA/Dec.  Returns False if no column
            pairs found based on `rd_pairs`.

        """
        from collections import OrderedDict
        import astropy.units as u

        if rd_pairs is None:
            rd_pairs = OrderedDict()
            rd_pairs['RA'] = 'DEC'
            rd_pairs['ALPHA_J2000'] = 'DELTA_J2000'
            rd_pairs['X_WORLD'] = 'Y_WORLD'
            rd_pairs['ALPHA_SKY'] = 'DELTA_SKY'
            rd_pairs['_RAJ2000'] = '_DEJ2000'

            for k in list(rd_pairs.keys()):
                rd_pairs[k.lower()] = rd_pairs[k].lower()

        rd_pair = None
        for c in rd_pairs:
            if c in self.colnames:
                rd_pair = [c, rd_pairs[c]]
                break

        if rd_pair is None:
            #print('No RA/Dec. columns found in input table.')
            return False

        for c in rd_pair:
            if self[c].unit is None:
                self[c].unit = u.degree

        return rd_pair

    def match_to_catalog_sky(self, other, self_radec=None, other_radec=None, nthneighbor=1, get_2d_offset=False):
        """Compute `~astropy.coordinates.SkyCoord` projected matches between two `GTable` tables.

        Parameters
        ----------
        other : `~astropy.table.Table`, `GTable`, or `list`.
            Other table to match positions from.

        self_radec, other_radec : None or [str, str]
            Column names for RA and Dec.  If None, then try the following
            pairs (in this order):

                >>> rd_pairs = OrderedDict()
                >>> rd_pairs['ra'] = 'dec'
                >>> rd_pairs['ALPHA_J2000'] = 'DELTA_J2000'
                >>> rd_pairs['X_WORLD'] = 'Y_WORLD'

        nthneighbor : int
            See `~astropy.coordinates.SkyCoord.coo.match_to_catalog_sky`.

        Returns
        -------
        idx : int array
            Indices of the matches as in

                >>> matched = self[idx]
                >>> len(matched) == len(other)

        dr : float array
            Projected separation of closest match.

        Examples
        --------

                >>> import astropy.units as u

                >>> ref = GTable.gread('input.cat')
                >>> gaia = GTable.gread('gaia.cat')
                >>> idx, dr = ref.match_to_catalog_sky(gaia)
                >>> close = dr < 1*u.arcsec

                >>> ref_match = ref[idx][close]
                >>> gaia_match = gaia[close]

        """
        from astropy.coordinates import SkyCoord

        if self_radec is None:
            rd = self.parse_radec_columns(self)
        else:
            rd = self.parse_radec_columns(self, rd_pairs={self_radec[0]: self_radec[1]})

        if rd is False:
            print('No RA/Dec. columns found in input table.')
            return False

        self_coo = SkyCoord(ra=self[rd[0]], dec=self[rd[1]], 
                            frame='icrs')

        if isinstance(other, list) | isinstance(other, tuple):
            rd = [slice(0, 1), slice(1, 2)]

        else:
            if other_radec is None:
                rd = self.parse_radec_columns(other)
            else:
                rd = self.parse_radec_columns(other, rd_pairs={other_radec[0]: other_radec[1]})

            if rd is False:
                print('No RA/Dec. columns found in `other` table.')
                return False

        other_coo = SkyCoord(ra=other[rd[0]], dec=other[rd[1]],
                             frame='icrs')

        try:
            idx, d2d, d3d = other_coo.match_to_catalog_sky(self_coo, nthneighbor=nthneighbor)
        except:
            print('Couldn\'t run SkyCoord.match_to_catalog_sky with'
                  'nthneighbor')

            idx, d2d, d3d = other_coo.match_to_catalog_sky(self_coo)
        
        if get_2d_offset:
            cosd = np.cos(self_coo.dec.deg/180*np.pi)
            dra = (other_coo.ra.deg - self_coo.ra.deg[idx])*cosd[idx]
            dde = (other_coo.dec.deg - self_coo.dec.deg[idx])
            return idx, d2d.to(u.arcsec), dra*3600*u.arcsec, dde*3600*u.arcsec
        else:
            return idx, d2d.to(u.arcsec)

    def match_triangles(self, other, self_wcs=None, x_column='X_IMAGE', y_column='Y_IMAGE', mag_column='MAG_AUTO', other_ra='X_WORLD', other_dec='Y_WORLD', pixel_index=1, match_kwargs={}, pad=100, show_diagnostic=False, auto_keep=3, maxKeep=10, auto_limit=3, ba_max=0.99, scale_density=10):
        """

        x_column = 'X_IMAGE'
        y_column = 'Y_IMAGE'
        mag_column = 'MAG_AUTO'
        pixel_index=1
        pad=100

        auto_keep=3
        maxKeep=10
        auto_limit=3
        ba_max = 0.99

        """
        from tristars import match

        if hasattr(other, 'shape'):
            other_radec = other*1.
        else:
            other_radec = np.array([other[other_ra], other[other_dec]]).T

        self_xy = np.array([self[x_column], self[y_column]]).T

        #xy_drz = np.array([cat['X_IMAGE'][ok], cat['Y_IMAGE'][ok]]).T

        if self_wcs is None:
            other_xy = other_radec
            cut = (other_xy[:, 0] > -pad) & (other_xy[:, 0] < self_xy[:, 0].max()+pad) & (other_xy[:, 1] > -pad) & (other_xy[:, 0] < self_xy[:, 1].max()+pad)
            other_xy = other_xy[cut, :]

            xy_center = np.zeros(2)

        else:
            other_xy = self_wcs.all_world2pix(other_radec, pixel_index)
            if hasattr(self_wcs, 'pixel_shape'):
                _naxis1, _naxis2 = self_wcs._naxis
            else:
                _naxis1, _naxis2 = self_wcs._naxis1, self_wcs._naxis2

            cut = (other_xy[:, 0] > -pad) & (other_xy[:, 0] < _naxis1+pad)
            cut &= (other_xy[:, 1] > -pad) & (other_xy[:, 1] < _naxis2+pad)

            other_xy = other_xy[cut, :]
            xy_center = self_wcs.wcs.crpix*1

        if len(other_xy) < 3:
            print('Not enough sources in match catalog')
            return False

        self_xy -= xy_center
        other_xy -= xy_center

        ########
        # Match surface density of drizzled and reference catalogs
        if mag_column is not None:
            icut = np.minimum(len(self)-2, int(scale_density*other_xy.shape[0]))
            self_ix = np.argsort(self[mag_column])[:icut]
        else:
            self_ix = np.arange(self_xy.shape[0])

        self_xy = self_xy[self_ix, :]

        pair_ix = match.match_catalog_tri(self_xy, other_xy, maxKeep=maxKeep, auto_keep=auto_keep, auto_transform=None, auto_limit=auto_limit, size_limit=[5, 1000], ignore_rot=False, ignore_scale=True, ba_max=ba_max)

        if len(pair_ix) == 0:
            print('No matches')
            return False

        tf, dx, rms = match.get_transform(self_xy, other_xy, pair_ix, transform=None, use_ransac=True)

        match_ix = pair_ix*1
        match_ix[:, 0] = self_ix[pair_ix[:, 0]]

        if show_diagnostic:
            fig = match.match_diagnostic_plot(self_xy, other_xy, pair_ix, tf=None, new_figure=True)
            return match_ix, tf, dx, rms, fig
        else:
            return match_ix, tf, dx, rms

    def add_aladdin(self, rd_cols=['ra', 'dec'], fov=0.5, size=(400, 200), default_view="P/DSS2/color"):
        """
        Add AladinLite DIV column to the table

        fov : fov in degrees
        size : size of DIVs (w, h) in pixels (w, h)

        """
        # <!-- include Aladin Lite CSS file in the head section of your page -->
        # <link rel="stylesheet" href="//aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css" />
        #
        # <!-- you can skip the following line if your page already integrates the jQuery library -->
        # <script type="text/javascript" src="//code.jquery.com/jquery-1.12.1.min.js" charset="utf-8"></script>

        ala = ["""    <div id="aladin-lite-div-{i}" style="width:{wsize}px;height:{hsize}px;"></div>
        <script type="text/javascript" src="http://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.js" charset="utf-8"></script>
        <script type="text/javascript">
            var aladin = A.aladin('#aladin-lite-div-{i}', xxxsurvey: "{survey}", fov:{fov}, target: "{ra} {dec}"yyy);
        </script></div>""".format(i=i, ra=row[rd_cols[0]], dec=row[rd_cols[1]], survey=default_view, fov=fov, hsize=size[1], wsize=size[0]).replace('xxx', '{').replace('yyy', '}') for i, row in enumerate(self)]

        self['aladin'] = ala

    def write_sortable_html(self, output, replace_braces=True, localhost=True, max_lines=50, table_id=None, table_class="display compact", css=None, filter_columns=[], buttons=['csv'], toggle=True, use_json=False):
        """Wrapper around `~astropy.table.Table.write(format='jsviewer')`.

        Parameters
        ----------
        output : str
            Output filename.

        replace_braces : bool
            Replace '&lt;' and '&gt;' characters that are converted
            automatically from "<>" by the `~astropy.table.Table.write`
            method. There are parameters for doing this automatically with
            `write(format='html')` but that don't appear to be available with
            `write(format='jsviewer')`.

        localhost : bool
            Use local JS files. Otherwise use files hosted externally.

        filter_columns : list
            Add option to limit min/max values of column data

        buttons : list
            Add buttons for exporting data.  Allowed options are
            'copy', 'csv', 'excel', 'pdf', 'print'.

        toggle : bool
            Add links at top of page for toggling columns on/off

        use_json : bool
            Write the data to a JSON file and strip out of the HTML header.
            Use this for large datasets or if columns include rendered
            images.

        etc : ...
            Additional parameters passed through to `write`.
        """
        #from astropy.table.jsviewer import DEFAULT_CSS
        DEFAULT_CSS = """
body {font-family: sans-serif;}
table.dataTable {width: auto !important; margin: 0 !important;}
.dataTables_filter, .dataTables_paginate {float: left !important; margin-left:1em}
td {font-size: 10pt;}
        """
        if css is not None:
            DEFAULT_CSS += css

        if os.path.exists(output):
            os.remove(output)

        self.write(output, format='jsviewer', css=DEFAULT_CSS,
                            max_lines=max_lines,
                            jskwargs={'use_local_files': localhost},
                            table_id=None, table_class=table_class)

        if replace_braces:
            lines = open(output).readlines()
            if replace_braces:
                for i in range(len(lines)):
                    lines[i] = lines[i].replace('&lt;', '<')
                    lines[i] = lines[i].replace('&gt;', '>')

            fp = open(output, 'w')
            fp.writelines(lines)
            fp.close()

        # Read all lines
        lines = open(output).readlines()

        if 'aladin' in self.colnames:
            # Insert Aladin CSS
            aladin_css = '<link rel="stylesheet" href="https://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css" />\n'

            for il, line in enumerate(lines):
                if '<link href=' in line:
                    break

            lines.insert(il+1, aladin_css)

        # Export buttons
        if buttons:
            # CSS
            css_script = '<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/1.5.1/css/buttons.dataTables.min.css">\n'
            for il, line in enumerate(lines):
                if 'css/jquery.dataTable' in line:
                    break

            lines.insert(il+1, css_script)

            # JS libraries
            js_scripts = """
            <script type="text/javascript" language="javascript" src="https://cdn.datatables.net/buttons/1.5.1/js/dataTables.buttons.min.js"></script>
            <script type="text/javascript" language="javascript" src="https://cdn.datatables.net/buttons/1.5.1/js/buttons.flash.min.js"></script>
            <script type="text/javascript" language="javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>
            <script type="text/javascript" language="javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.32/pdfmake.min.js"></script>
            <script type="text/javascript" language="javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.32/vfs_fonts.js"></script>
            <script type="text/javascript" language="javascript" src="https://cdn.datatables.net/buttons/1.5.1/js/buttons.html5.min.js"></script>
            <script type="text/javascript" language="javascript" src="https://cdn.datatables.net/buttons/1.5.1/js/buttons.print.min.js"></script>
            """

            for il, line in enumerate(lines):
                if 'js/jquery.dataTable' in line:
                    break
            lines.insert(il+2, js_scripts)

            for il, line in enumerate(lines):
                if 'pageLength' in line:
                    break

            button_option = '{spacer}dom: \'Blfrtip\',\n{spacer}buttons: {bstr},\n'.format(spacer=' '*8, bstr=buttons.__repr__())
            lines.insert(il+1, button_option)

        # Range columns
        ic_list = []

        filter_lines = ["<table>\n"]
        descr_pad = ' <span style="display:inline-block; width:10;"></span> '

        for ic, col in enumerate(self.colnames):
            if col in filter_columns:
                found = False
                for i in range(len(lines)):
                    if '<th>{0}'.format(col) in lines[i]:
                        found = True
                        break

                if found:
                    # print(col)
                    ic_list.append(ic)
                    #lines[i] = lines[i].replace(col, '{0} <br> <input type="text" id="{0}_min" name="{0}_min" style="width:30px;"> <input type="text" id="{0}_max" name="{0}_max" style="width:30px;">'.format(col))

                    filter_lines += '<tr> <td> <input type="text" id="{0}_min" name="{0}_min" style="width:40px;"> &#60; </td> <td> {0} </td> <td>  &#60; <input type="text" id="{0}_max" name="{0}_max" style="width:40px;">'.format(col)
                    
                    descr = '\n'
                    if hasattr(self.columns[col], 'description'):
                        if self.columns[col].description is not None:
                            descr = '{0} {1}\n'.format(descr_pad, 
                                            self.columns[col].description)
                                                        
                    filter_lines += descr
                    
        if ic_list:
            # Insert input lines

            for il, line in enumerate(lines):
                if '} );  </script>' in line:
                    break
            
            filter_row = '<tr> <td> <input type="text" id="{0}_min" name="{0}_min" style="width:40px;"> &#60; </td> <td style="align:center;"> <tt>{0}</tt> </td> <td>  &#60; <input type="text" id="{0}_max" name="{0}_max" style="width:40px;">'
            
            filter_rows = []
            for ic in ic_list:
                col = self.colnames[ic]
                row_i = filter_row.format(col)
                descr = '\n'
                if hasattr(self.columns[col], 'description'):
                    if self.columns[col].description is not None:
                        descr = '{0} {1}\n'.format(descr_pad, 
                                        self.columns[col].description)
                                                    
                filter_rows.append(row_i + descr)
                
            filter_input = """

<div style="border:1px solid black; padding:10px; margin:10px">
<b> Filter: </b>
    <table>
      {0}
    </table>
</div>

""".format('\n'.join(filter_rows))

            lines.insert(il+1, filter_input)

            # Javascript push function
            header_lines = ""
            tester = []

            for ic in ic_list:
                header_lines += """
        var min_{0} = parseFloat( $('#{0}_min').val()) || -1e30;
        var max_{0} = parseFloat( $('#{0}_max').val()) ||  1e30;
        var data_{0} = parseFloat( data[{1}] ) || 0;
                """.format(self.colnames[ic], ic)

                tester.append("""( ( isNaN( min_{0} ) && isNaN( max_{0} ) ) || ( isNaN( min_{0} ) && data_{0} <= max_{0} ) || ( min_{0} <= data_{0}  && isNaN( max_{0} ) ) || ( min_{0} <= data_{0}  && data_{0} <= max_{0} ) )""".format(self.colnames[ic]))

            # Javascript filter function
            filter_function = """

//// Parser
// https://stackoverflow.com/questions/19491336/get-url-parameter-jquery-or-how-to-get-query-string-values-in-js @ Reza Baradaran
$.urlParam = function(name){{
    var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);
    if (results==null){{
       return null;
    }}
    else{{
       return decodeURI(results[1]) || 0;
    }}
}}

$.fn.dataTable.ext.search.push(
    function( settings, data, dataIndex ) {{
{0}

        if ( {1} )
        {{
            return true;
        }}
        return false;
    }}
);

//// Update URL with filter parameters
var filter_params = {2};

$.UpdateFilterURL = function () {{
    var i;
    var filter_url = "";
    for (i = 0; i < filter_params.length; i++) {{
        if ($('#'+filter_params[i]+'_min').val() != "") {{
            filter_url += '&'+filter_params[i]+'_min='+
                    $('#'+filter_params[i]+'_min').val();
        }}
        if ($('#'+filter_params[i]+'_max').val() != "") {{
            filter_url += '&'+filter_params[i]+'_max='+
                    $('#'+filter_params[i]+'_max').val();
        }}
    }}

    if (filter_url != "") {{
        var filtered_url = window.location.href.split('?')[0] + '?' + filter_url;
        window.history.pushState('', '', filtered_url);
    }}
}}\n\n""".format(header_lines, "\n && ".join(tester), [self.colnames[ic] for ic in ic_list].__repr__())

            for i, line in enumerate(lines):
                if "$(document).ready(function()" in line:
                    istart = i
                    break

            lines.insert(istart, filter_function)

            # Parse address bar
            lines.insert(istart+2, "\n")
            for ic in ic_list:
                lines.insert(istart+2, "{1}$('#{0}_max').val($.urlParam('{0}_max'));\n".format(self.colnames[ic], ' '*3))
                lines.insert(istart+2, "{1}$('#{0}_min').val($.urlParam('{0}_min'));\n".format(self.colnames[ic], ' '*3))
            lines.insert(istart+2, "\n")

            # Input listener
            listener = """
    // Event listener to the two range filtering inputs to redraw on input
    $('{0}').keyup( function() {{
        table.draw();
        $.UpdateFilterURL();
    }} );
            """.format(', '.join(['#{0}_min, #{0}_max'.format(self.colnames[ic]) for ic in ic_list]))

            for il, line in enumerate(lines):
                if '} );  </script>' in line:
                    break

            lines.insert(il, listener)

            fp = open(output, 'w')
            fp.writelines(lines)
            fp.close()

        if toggle:
            lines = open(output).readlines()

            # Change call to DataTable
            for il, line in enumerate(lines):
                if 'dataTable(' in line:
                    break

            lines[il] = "   var table = {0}\n".format(lines[il].strip().replace('dataTable', 'DataTable'))

            # Add function
            for il, line in enumerate(lines):
                if '} );  </script>' in line:
                    break

            toggler = """
    $('a.toggle-vis').on( 'click', function (e) {
        e.preventDefault();

        // Get the column API object
        var column = table.column( $(this).attr('data-column') );

        // Toggle the visibility
        column.visible( ! column.visible() );
    } );

            """
            lines.insert(il, toggler)

            toggle_div = """
<div style="border:1px solid black; padding:10px; margin:10px">
    <b>Toggle column:</b></br> {0}
</div>

            """.format(' <b>/</b> '.join(['<a class="toggle-vis" data-column="{0}"> <tt>{1}</tt> </a>'.format(ic, col) for ic, col in enumerate(self.colnames)]))

            lines.insert(il+2, toggle_div)

            fp = open(output, 'w')
            fp.writelines(lines)
            fp.close()

        if use_json:
            # Write as json

            # Workaround to get ascii formatting
            #pd = self.to_pandas()
            new = GTable()
            for c in self.colnames:
                new[c] = self[c]

            if 'aladin' in self.colnames:
                pd = GTable(new).to_pandas()
            else:
                new.write('/tmp/table.csv', format='csv', overwrite=True)
                pd = GTable.gread('/tmp/table.csv').to_pandas()

            # Reformat to json
            json_data = '        ' + pd.to_json(orient='values').replace('],[', '\n    ]xxxxxx\n    [\n        ').replace(', ', 'xcommaspacex').replace(',', ',\n        ').replace('xxxxxx', ',').replace('xcommaspacex', ', ')
            json_str = """{{
  "data":
{0}

}}
""".format(json_data.replace('\\""', '"'))

            fp = open(output.replace('.html', '.json'), 'w')
            fp.write(json_str)
            fp.close()

            # Edit HTML file
            lines = open(output).readlines()

            # Add ajax call to DataTable
            for il, line in enumerate(lines):
                if 'pageLength' in line:
                    break

            ajax_call = '{spacer}"ajax": "{json}",\n{spacer}"deferRender": true,\n'.format(spacer=' '*8, json=output.replace('.html', '.json'))
            lines.insert(il+1, ajax_call)

            # Strip out table body
            for ihead, line in enumerate(lines):
                if '</thead>' in line:
                    break

            for itail, line in enumerate(lines[::-1]):
                if '</tr>' in line:
                    break

            fp = open(output, 'w')
            fp.writelines(lines[:ihead+1])
            fp.writelines(lines[-itail:])
            fp.close()


def column_string_operation(col, test, method='contains', logical='or'):
    """
    Analogous to ``str.contains`` but for table column.

    Parameters
    ----------
    col : iterable list of strings
        List of strings to test.  Anything iterable, e.g., list or 
        `~astropy.table.column.Column`.

    test : str, list of strings, None, or slice

        If ``test`` is a string, or list of strings, then run the string
        ``method`` on each entry of ``col`` with ``test`` as the argument or
        each element of the ``test`` list as arguments.

        If ``test`` is None, run `method` on each entry with no arguments, 
        e.g., 'upper'.

        If ``test`` is a ``slice``, return sliced strings for each entry.

    method : str
        String method to apply to each entry of ``col``.  E.g., 'contains',
        'startswith', 'endswith', 'index'.

    logical : ['or','and','not']
        Logical test to use when ``test`` is a list of strings.  For example,
        if you want to test if the column has values that match either
        'value1' or 'value2', then run with

            >>> res = column_to_string_operation(col, ['value1','value2'], method='contains', logical='or')

    Returns
    -------
    result : list
        List of iterated results on the entries of ``col``, e.g., list of 
        ``bool`` or ``string``.

    """
    if isinstance(test, list):
        test_list = test
    else:
        test_list = [test]

    out_test = []

    for i, c_i in enumerate(col):
        if isinstance(test, slice):
            out_test.append(c_i[test])
            continue

        func = getattr(c_i, method)
        if test is None:
            out_test.append(func())
            continue

        list_i = []
        for t_i in test_list:
            try:
                list_i.append(func(t_i))
            except:
                list_i.append(False)

        out_test.append(list_i)

    arr = np.array(out_test)
    sh = arr.shape

    if logical is None:
        return np.squeeze(arr)
    elif logical.upper() == 'AND':
        return np.sum(arr, axis=1) >= sh[1]
    elif logical.upper() == 'NOT':
        return np.sum(arr, axis=1) == 0
    else:  # OR
        return np.sum(arr, axis=1) > 0


def column_values_in_list(col, test_list):
    """Test if column elements "in" an iterable (e.g., a list of strings)

    Parameters
    ----------
    col : `astropy.table.Column` or other iterable
        Group of entries to test

    test_list : iterable
        List of values to search

    Returns
    -------
    test : bool array
        Simple test:
            >>> [c_i in test_list for c_i in col]
    """
    test = np.array([c_i in test_list for c_i in col])
    return test


def fill_between_steps(x, y0, y1, ax=None, *args, **kwargs):
    """
    Make `fill_between` work like linestyle='steps-mid'.
    """
    import matplotlib.pyplot as plt
    
    so = np.argsort(x)
    dx = np.diff(x[so])/2.
    mid = x[so][:-1] + dx
    
    xfull = np.hstack([x[so][0]-dx[0], mid, mid+dx*2/1.e6, x[so][-1]+dx[-1]])
    y0full = np.hstack([y0[0], y0[:-1], y0[1:], y0[-1]])
    y1full = np.hstack([y1[0], y1[:-1], y1[1:], y1[-1]])
    
    # xfull = np.append(np.append(x, mid), mid+np.diff(x[so])/1.e6)
    # y0full = np.append(np.append(y0, y0[:-1]), y0[1:])
    # y1full = np.append(np.append(y1, y1[:-1]), y1[1:])

    so = np.argsort(xfull)
    if ax is None:
        ax = plt.gca()

    ax.fill_between(xfull[so], y0full[so], y1full[so], *args, **kwargs)


def fill_masked_covar(covar, mask):
    """Fill a covariance matrix in a larger array that had masked values

    Parameters
    ----------
    covar : `(M,M)` square `~np.ndarray`
        Masked covariance array.

    mask : bool mask, `N>M`
        The original mask.

    Returns
    -------
    covar_full : `~np.ndarray`
        Full covariance array with dimensions `(N,N)`.

    """
    N = mask.shape[0]
    idx = np.arange(N)[mask]
    covar_full = np.zeros((N, N), dtype=covar.dtype)
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            covar_full[ii, jj] = covar[i, j]

    return covar_full


def log_scale_ds9(im, lexp=1.e12, cmap=[7.97917, 0.8780493], scale=[-0.1, 10]):
    """
    Scale an array like ds9 log scaling
    """
    import numpy as np

    contrast, bias = cmap
    clip = (np.clip(im, scale[0], scale[1])-scale[0])/(scale[1]-scale[0])
    clip_log = np.clip((np.log10(lexp*clip+1)/np.log10(lexp)-bias)*contrast+0.5, 0, 1)

    return clip_log


def mode_statistic(data, percentiles=range(10, 91, 10)):
    """
    Get modal value of a distribution of data following Connor et al. 2017
    https://arxiv.org/pdf/1709.01925.pdf

    Here we fit a spline to the cumulative distribution evaluated at knots
    set to the `percentiles` of the distribution to improve smoothness.
    """
    from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

    so = np.argsort(data)
    order = np.arange(len(data))
    #spl = UnivariateSpline(data[so], order)

    knots = np.percentile(data, percentiles)
    dx = np.diff(knots)
    mask = (data[so] >= knots[0]-dx[0]) & (data[so] <= knots[-1]+dx[-1])
    spl = LSQUnivariateSpline(data[so][mask], order[mask], knots, ext='zeros')

    mask = (data[so] >= knots[0]) & (data[so] <= knots[-1])
    ix = (spl(data[so], nu=1, ext='zeros')*mask).argmax()
    mode = data[so][ix]
    return mode


def make_alf_template():
    """
    Make Alf + FSPS template
    """
    import alf.alf
    import fsps

    ssp = alf.alf.Alf()

    sp = fsps.StellarPopulation(zcontinuous=1)
    sp.params['logzsol'] = 0.2

    # Alf
    m = ssp.get_model(in_place=False, logage=0.96, zh=0.2, mgh=0.2)

    # FSPS
    w, spec = sp.get_spectrum(tage=10**0.96, peraa=True)

    # blue
    blue_norm = spec[w > 3600][0] / m[ssp.wave > 3600][0]
    red_norm = spec[w > 1.7e4][0] / m[ssp.wave > 1.7e4][0]

    templx = np.hstack([w[w < 3600], ssp.wave[(ssp.wave > 3600) & (ssp.wave < 1.7e4)], w[w > 1.7e4]])
    temply = np.hstack([spec[w < 3600]/blue_norm, m[(ssp.wave > 3600) & (ssp.wave < 1.7e4)], spec[w > 1.7e4]/red_norm])

    np.savetxt('alf_SSP.dat', np.array([templx, temply]).T, fmt='%.5e', header='wave flux\nlogage = 0.96\nzh=0.2\nmgh=0.2\nfsps: w < 3600, w > 1.7e4')


def catalog_area(ra=[], dec=[], make_plot=True, NMAX=5000, buff=0.8, verbose=True):
    """Compute the surface area of a list of RA/DEC coordinates

    Parameters
    ----------
    ra, dec : `~numpy.ndarray`
        RA and Dec. coordinates in decimal degrees

    make_plot : bool
        Make a figure.

    NMAX : int
        If the catalog has more then `NMAX` entries, draw `NMAX` random
        samples.

    buff : float
        Buffer in arcmin to add around each catalog point.


    Returns
    -------
    area : float
        Computed catalog area in square arcminutes

    fig : `~matplotlib.figure.Figure`
        Figure object returned if `make_plot==True`.

    """
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon, Point, MultiPolygon, MultiLineString
    from scipy import spatial

    points = np.array([ra, dec])*1.
    center = np.mean(points, axis=1)
    points = (points.T - center)*60.  # arcmin
    points[:, 0] *= np.cos(center[1]/180*np.pi)

    hull = spatial.ConvexHull(points)
    edge = points[hull.vertices, :]

    #pbuff = 1

    if len(ra) > NMAX:
        rnd_idx = np.unique(np.cast[int](np.round(np.random.rand(NMAX)*len(ra))))
    else:
        rnd_idx = np.arange(len(ra))

    poly = Point(points[rnd_idx[0], :]).buffer(buff)
    for i, ix in enumerate(rnd_idx):
        if verbose:
            print(NO_NEWLINE + '{0} {1}'.format(i, ix))

        poly = poly.union(Point(points[ix, :]).buffer(buff))

    # Final (multi)Polygon
    pjoin = poly.buffer(-buff)

    if make_plot:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if isinstance(pjoin, MultiPolygon):
            for p_i in pjoin:
                if isinstance(p_i.boundary, MultiLineString):
                    for s in p_i.boundary:
                        p = s.xy
                        ax.plot(p[0], p[1])
                else:
                    p = p_i.boundary.xy
                    ax.plot(p[0], p[1])
        else:
            p_i = pjoin
            if isinstance(p_i.boundary, MultiLineString):
                for s in p_i.boundary:
                    p = s.xy
                    ax.plot(p[0], p[1])
            else:
                p = p_i.boundary.xy
                ax.plot(p[0], p[1])

        ax.scatter(points[rnd_idx, 0], points[rnd_idx, 1], alpha=0.1, marker='+')

        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_xlabel(r'$\Delta$RA ({0:.5f})'.format(center[0]))
        ax.set_ylabel(r'$\Delta$Dec. ({0:.5f})'.format(center[1]))
        ax.set_title('Total area: {0:.1f} arcmin$^2$'.format(pjoin.area))
        ax.grid()
        fig.tight_layout(pad=0.1)

        return pjoin.area, fig

    else:
        return pjoin.area


def fix_flt_nan(flt_file, bad_bit=4096, verbose=True):
    """
    Fix NaN values in FLT files
    """
    im = pyfits.open(flt_file, mode='update')
    for ext in range(1, 5):
        if ('SCI', ext) in im:
            mask = ~np.isfinite(im['SCI', ext].data)
            if verbose:
                label = 'utils.fix_flt_nan: {0}[SCI,{1}] NaNPixels={2}'
                print(label.format(flt_file, ext, mask.sum()))

            if mask.sum() == 0:
                continue

            im['SCI', ext].data[mask] = 0
            im['DQ', ext].data[mask] |= bad_bit

    im.flush()
    im.close()


def dump_flt_dq(filename, replace=('.fits', '.dq.fits.gz'), verbose=True):
    """Dump FLT/FLC header & DQ extensions to a compact file

    Parameters
    ----------
    filename : str
        FLT/FLC filename.

    replace : (str, str)
        Replace arguments for output filename:

        >>> output_filename = filename.replace(replace[0], replace[1])

    Returns
    -------
    Writes header and compact DQ array to `output_filename`.

    """
    im = pyfits.open(filename)
    hdus = []
    for i in [1, 2, 3, 4]:
        if ('SCI', i) in im:
            header = im['SCI', i].header
            dq = im['DQ', i].data
            nz = np.where(dq > 0)
            dq_data = np.array([nz[0], nz[1], dq[dq > 0]], dtype=np.int16)
            hdu = pyfits.ImageHDU(header=header, data=dq_data)
            hdus.append(hdu)

    output_filename = filename.replace(replace[0], replace[1])

    msg = '# dump_flt_dq: {0} > {1}'.format(filename, output_filename)
    log_comment(LOGFILE, msg, verbose=verbose, show_date=True)

    pyfits.HDUList(hdus).writeto(output_filename, overwrite=True,
                                 output_verify='fix')
    
    im.close()


def apply_flt_dq(filename, replace=('.fits', '.dq.fits.gz'), verbose=True, or_combine=False):
    """
    Read and apply the compact exposure information file

    Parameters
    ----------
    filename : str
        FLT/FLC filename.

    replace : (str, str)
        Replace arguments for output DQ filename:

        >>> output_filename = filename.replace(replace[0], replace[1])

    or_combine : bool
        If True, then apply DQ data in `output_filename` with OR logic.

        If False, then reset the DQ extensions to be exactly those in
        `output_filename`.

    Returns
    -------
    Writes header and compact DQ array to `output_filename`.

    """

    output_filename = filename.replace(replace[0], replace[1])

    if not os.path.exists(output_filename):
        return False

    im = pyfits.open(filename, mode='update')

    msg = '# apply_flt_dq: {1} > {0}'.format(filename, output_filename)
    log_comment(LOGFILE, msg, verbose=verbose, show_date=True)

    dq = pyfits.open(output_filename)
    for ext in [1, 2, 3, 4]:
        if (('SCI', ext) in im) & (('SCI', ext) in dq):
            sh = dq['SCI', ext].data.shape
            if sh[0] == 3:
                i, j, dq_i = dq['SCI', ext].data
            elif sh[1] == 2:
                nz, dq_i = dq['SCI', ext].data
                i, j = np.unravel_index(nz, sh)
            else:
                raise IOError('dq[{0}] shape {1} not recognized'.format(ext, sh))
                
            # Apply DQ
            if or_combine:
                im['DQ', ext].data[i, j] != dq_i
            else:
                im['DQ', ext].data *= 0
                im['DQ', ext].data[i, j] = dq_i

            # Copy header
            has_blotsky = 'BLOTSKY' in im['SCI', ext].header
            for k in dq['SCI', ext].header:
                if k in ['BITPIX', 'NAXIS1', 'NAXIS2', '', 'HISTORY']:
                    continue

                im['SCI', ext].header[k] = dq['SCI', ext].header[k]

            if (not has_blotsky) & ('BLOTSKY' in dq['SCI', ext].header):
                im['SCI', ext].header['BLOTSKY'] = False

    im.flush()
    im.close()


def RGBtoHex(vals, rgbtype=1):
    """Converts RGB values in a variety of formats to Hex values.

    Parameters
    ----------
    vals : tuple
         An RGB/RGBA tuple

    rgb_type : int
        Valid valus are:
         - 1 = Inputs are in the range 0 to 1
         - 256 = Inputs are in the range 0 to 255
 
    Returns
    -------
    hextstr : str
        A hex string in the form '#RRGGBB' or '#RRGGBBAA'

    References
    ----------
    (credit: Rychard @ https://stackoverflow.com/a/48288173)

    """

    msg = "RGB or RGBA inputs to RGBtoHex must have three or four elements!"
    if (len(vals) != 3) & (len(vals) != 4):
        raise Exception(msg)

    if (rgbtype != 1) & (rgbtype != 256):
        raise Exception("rgbtype must be 1 or 256!")

    # Convert from 0-1 RGB/RGBA to 0-255 RGB/RGBA
    if rgbtype == 1:
        vals = [255*x for x in vals[:3]]

    # Ensure values are rounded integers, convert to hex, and concatenate
    return '#' + ''.join(['{:02X}'.format(int(round(x))) for x in vals])


def catalog_mask(cat, ecol='FLUXERR_APER_0', max_err_percentile=90, pad=0.05, pad_is_absolute=False, min_flux_radius=1.):
    """
    Compute a catalog mask for
      1) Objects within `pad` of the edge of the catalog convex hull
      2) Uncertainties < `max_err_percentile`

    """
    test = np.isfinite(cat['FLUX_AUTO'])
    if 'FLUX_RADIUS' in cat.colnames:
        test &= cat['FLUX_RADIUS'] > min_flux_radius

    test &= (cat['THRESH'] > 0) & (cat['THRESH'] < 1e28)

    not_edge = hull_edge_mask(cat['X_IMAGE'], cat['Y_IMAGE'],
                              pad=pad, pad_is_absolute=pad_is_absolute)

    test &= not_edge

    if ecol in cat.colnames:
        valid = np.isfinite(cat[ecol])
        if max_err_percentile < 100:
            test &= cat[ecol] < np.percentile(cat[ecol][(~not_edge) & valid],
                                          max_err_percentile)

    return test


def hull_edge_mask(x, y, pad=100, pad_is_absolute=True, mask=None):
    """
    Compute geometrical edge mask for points within a convex hull

    Parameters
    ----------
    x, y : array
        Coordinates of the catalog

    pad : float
        Buffer padding

    pad_is_absolute : bool
        If True, then the buffer is taken from `pad` (e.g., pixels).  If
        False, then `pad` is treated as a fraction of the linear dimension
        of the catalog (`~sqrt(hull area)`).

    mask : bool array
        Mask to apply to x/y before computing the convex hull

    Returns
    -------
    mask : bool array
        True if points within the buffered hull

    """

    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon, Point

    xy = np.array([x, y]).T

    if mask is None:
        hull = ConvexHull(xy)
    else:
        hull = ConvexHull(xy[mask, :])

    pxy = xy[hull.vertices, :]
    poly = Polygon(pxy)

    if pad_is_absolute:
        buff = -pad
    else:
        # linear dimension ~ sqrt(area)
        buff = -pad*np.sqrt(poly.area)

    pbuff = poly.buffer(buff)
    in_buff = np.array([pbuff.contains(Point([x[i], y[i]])) for i in range(len(x))])

    return in_buff


def convex_hull_wrapper(x, y):
    """
    Generate a convex hull from a list of points
    
    Returns:
    
    pxy : (array, array)
        Tuple of hull vertices
    
    poly : `~shapely.geometry.Polygon`
        Polygon object.
    
    hull : `~scipy.spatial.ConvexHull`
        The hull object.
        
    """
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon, Point

    xy = np.array([x, y]).T
    hull = ConvexHull(xy)
    pxy = xy[hull.vertices, :]
    poly = Polygon(pxy)
    
    return pxy, poly, hull


def hull_area(x, y):
    """
    Return the area of a convex hull of a list of points
    """
    pxy, poly, hull = convex_hull_wrapper(x, y)

    return poly.area
    
    
def remove_text_labels(fig):
    """
    Remove all Text annotations from ``fig.axes``.
    """
    import matplotlib
    
    for ax in fig.axes:
        for child in ax.get_children():
            if isinstance(child, matplotlib.text.Text):
                if child.get_text(): # Don't remove empty labels
                    child.set_visible(False)

    
LOGFILE = '/tmp/grizli.log'


def log_function_arguments(LOGFILE, frame, func='func', verbose=True):
    """
    Log local variables, e.g., parameter arguements to a file

    Parameters
    ----------
    LOGFILE : str or None
        Output file.  If `None`, then force `verbose=True`.

    frame : `~inspect.currentframe()`
        Namespace object.

    func : str
        Function name to use

    verbose : bool
        Print messaage to stdout.

    """
    args = inspect.getargvalues(frame).locals
    args.pop('frame')
    for k in list(args.keys()):
        if hasattr(args[k], '__builtins__'):
            args.pop(k)

    if func is not None:
        logstr = '\n{0}(**{1})\n'
    else:
        logstr = '\n{1}'

    logstr = logstr.format(func, args)
    msg = log_comment(LOGFILE, logstr, verbose=verbose, show_date=True)

    return msg


def ctime_to_iso(mtime, format='%a %b %d %H:%M:%S %Y', strip_decimal=True, verbose=True):
    """
    Convert `time.ctime` strings to ISO dates

    Parameters
    ----------
    mtime : str
        Time string, generally as output from `time.ctime`, e.g., 
        ``'Mon Sep 16 11:23:27 2019'``

    format : str
        `datetime.strptime` format string, codes at 
        https://www.programiz.com/python-programming/datetime/strptime

    strip_decimal : bool
        Strip decimal seconds from end of ISO string

    verbose : bool
        Print a message if conversion fails

    Returns 
    -------
    iso : str
        String in (sortable) ISO format, e.g., ``'2019-09-16 11:23:27.000'``

    """
    from astropy.time import Time
    from datetime import datetime

    # Is already ISO format
    if mtime.count('-') == 2:
        iso = mtime + ''

    else:
        try:
            iso = Time(datetime.strptime(mtime, format), 
                       format='datetime').iso
        except ValueError:
            if verbose:
                print(f'Couldn\'t convert \'{mtime}\' with '
                      f'format \'{format}\'')

            iso = mtime + ''

    if strip_decimal:
        iso = iso.split('.')[0]

    return iso


def nowtime(iso=True):
    """
    Wrapper for `astropy.time.now`

    Parameters
    ----------
    iso : bool
        If True, return time in ISO string, else return Time object

    Returns
    -------
    tnow : str
        See `iso`

    """
    from astropy.time import Time
    tnow = Time.now()
    if iso:
        return tnow.iso
    else:
        return tnow


def log_comment(LOGFILE, comment, verbose=False, show_date=False, mode='a'):
    """
    Log a message to a file, optionally including a date tag
    """
    import time

    if show_date:
        msg = '# ({0})\n'.format(nowtime())
    else:
        msg = ''

    msg += '{0}\n'.format(comment)

    if LOGFILE is not None:
        fp = open(LOGFILE, mode)
        fp.write(msg)
        fp.close()

    if verbose:
        print(msg[:-1])

    return msg


def log_exception(LOGFILE, traceback, verbose=True, mode='a'):
    """
    Log exception information to a file, or print to screen

    Parameters
    ----------
    LOGFILE : str or None
        Output file.  If `None`, then force `verbose=True`.

    traceback : builtin traceback module
        Exception traceback, from global `import traceback`.

    verbose : bool
        Print exception to stdout.

    mode : 'a', 'w'
        File mode on `open(LOGFILE, mode)`, i.e., append or write.

    """
    import time

    trace = traceback.format_exc(limit=2)
    log = '\n########################################## \n'
    log += '# ! Exception ({0})\n'.format(nowtime())
    log += '#\n# !'+'\n# !'.join(trace.split('\n'))
    log += '\n######################################### \n\n'
    if verbose | (LOGFILE is None):
        print(log)

    if LOGFILE is not None:
        fp = open(LOGFILE, mode)
        fp.write(log)
        fp.close()

def simple_LCDM(Om0=0.3, Ode0=0.7, H0=70, Ob0=0.0463, Tcmb0=2.725, name=None):
    """
    Simple LambdaCDM cosmology
    
    Parameters are defined as in `~astropy.cosmology.LambdaCDM`.
    
    """
    from astropy.cosmology import LambdaCDM
    cosmology = LambdaCDM(H0, Om0, Ode0, Tcmb0=Tcmb0, name=name)
    return cosmology


def pixel_polygon_mask(polygon, shape, wcs=None):
    """
    Make a mask points in a 2D array inside a polygon
    
    Parameters
    ----------
    polygon : str, (2,M) array
        Something that `grizli.utils.SRegion` can parse as a polygon
    
    shape : tuple
        2-tuple of image dimensions
    
    wcs : `astropy.wcs.WCS`
        If specified, assume ``polygon`` is sky coordinates and transform to
        image.
    
    Returns
    -------
    mask : array
        ``bool`` array with ``shape`` that is `True` inside `polygon`
    """
    sr = SRegion(polygon, wrap=False)
    
    yp, xp = np.indices(shape)
    pts = np.array([xp.flatten(), yp.flatten()]).T
    
    if wcs is not None:
        pts = wcs.all_pix2world(pts, 0)
        
    mask = np.zeros(shape, dtype=bool).flatten()
    for p in sr.path:
        mask |= p.contains_points(pts)
    
    return mask.reshape(shape)


def make_filter_footprint(filter_size=71, filter_central=0, **kwargs):
    """
    Make a footprint for image filtering
    """
    filter_footprint = np.ones(filter_size, dtype=int)
    
    if filter_central > 0:
        f0 = (filter_size-1)//2
        filter_footprint[f0-filter_central:f0+filter_central] = 0
    
    return filter_footprint


def safe_nanmedian_filter(data, filter_kwargs={}, filter_footprint=None, axis=1, clean=True, cval=0.0):
    """
    Run nanmedian filter on `data`
    
    Parameters
    ----------
    data : array-like
        The 2D data to filter
    
    filter_kwargs : dict
        Arguments to `~grizli.utils.make_filter_footprint` to make a 1D filter
    
    filter_footprint : array-like
        Specify the filter explicitly.  If 1D, then will apply the filter over
        `axis=1` (i.e., `x`) of ``data``
    
    axis : 0 or 1
         Axis over which to apply the filter (``axis=1`` filters on rows)
    
    clean, cval : bool, scalar
        Replace `~numpy.nan` in the output with ``cval``
    
    Returns
    -------
    filter_data : array-like
        Filtered data
    
    filter_name : str
        The type of filter that was applied: `nbutils.nanmedian` if 
        `~grizli.nbutils.nanmedian` was imported successfully and 
        `median_filter` for the fallback to `scip.ndimage.median_filter` 
        otherwise.
    """
    import scipy.ndimage as nd
    
    try:
        from . import nbutils
        _filter_name = 'nbutils.nanmedian'
    except:
        nbutils = None
        _filter_name = 'median_filter'
    
    if filter_footprint is None:
        _filter_footprint = make_filter_footprint(**filter_kwargs)
        if axis == 1:
            _filter_footprint = _filter_footprint[None,:]
        else:
            _filter_footprint = _filter_footprint[:,None]
    else:
        if filter_footprint.ndim == 1:
            if axis == 1:
                _filter_footprint = filter_footprint[None,:]
            else:
                _filter_footprint = filter_footprint[:,None]
        else:
            _filter_footprint = filter_footprint
    
    if nbutils is None:
        filter_data = nd.median_filter(data, footprint=_filter_footprint)
    else:
        filter_data = nd.generic_filter(data, nbutils.nanmedian,
                                        footprint=_filter_footprint)
        if clean:
            filter_data[~np.isfinite(filter_data)] = cval
            
    return filter_data, _filter_name


def argv_to_dict(argv, defaults={}, dot_dict=True):
    """
    Convert a list of (simple) command-line arguments to a dictionary.
    
    Parameters
    ----------
    argv : list of strings
        E.g., ``sys.argv[1:]``.
    
    defaults : dict
        Default dictionary
    
    dot_dict : bool
        If true, then intepret keywords with '.' as nested dictionary keys, 
        e.g., ``--d.key=val`` >> {'d': {'key': 'val'}}
        
    Examples
    --------
    
        # $ myfunc arg1 --p1=1 --l1=1,2,3 --pdict.k1=1 -flag
        >>> argv = 'arg1 --p1=1 --l1=1,2,3 --pdict.k1=1 -flag'.split()
        >>> args, kwargs = argv_to_dict(argv)
        >>> print(args)
        ['arg1']
        >>> print(kwargs)
        {'p1': 1, 'l1': [1, 2, 3], 'pdict': {'k1': 1}, 'flag': True}
        
        # With defaults
        defaults = {'pdict':{'k2':2.0}, 'p2':2.0}
        >>> args, kwargs = argv_to_dict(argv, defaults=defaults)
        >>> print(kwargs)
        {'pdict': {'k2': 2.0, 'k1': 1}, 'p2': 2.0, 'p1': 1, 'l1': [1, 2, 3], 'flag': True}
        
    """
    import copy
    import json
    
    kwargs = copy.deepcopy(defaults)
    args = []
    
    for i, arg in enumerate(argv):        
        if not arg.startswith('-'):
            # Arguments
            try:
                args.append(json.loads(arg))
            except:
                args.append(json.loads(f'"{arg}"'))
                
            continue
            
        spl = arg.strip('--').split('=')
        if len(spl) > 1:
            # Parameter values
            key, val = spl
            val = val.replace('True','true').replace('False','false')
            val = val.replace('None','null')
        else:
            # Parameters, set to true, e.g., -set_flag
            key, val = spl[0], 'true'
            
            # single -
            if key.startswith('-'):
                key = key[1:]
        
        # List values        
        if ',' in val:
            try:
                # Try parsing with JSON
                jval = json.loads(f'[{val}]')
            except:
                # Assume strings
                str_val = ','.join([f'"{v}"' for v in val.split(',')])
                jval = json.loads(f'[{str_val}]')
        else:
            try:
                jval = json.loads(val)
            except:
                # String
                jval = val
        
        # Dict keys, potentially nested        
        if dot_dict & ('.' in key):
            keys = key.split('.')
            d = kwargs
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                
                d = d[k]
                    
            d[keys[-1]] = jval
        else:            
            kwargs[key] = jval
            
    
    return args, kwargs


class Unique(object):
    def __init__(self, array, verbose=True):
        """
        Helper for unique items in an array
        
        Parameters
        ----------
        array : array-like
            Data to parse, generally strings but can be anything that can 
            be parsed by `numpy.unique`


        Attributes
        ----------
        dim : int
            ``size`` of input ``array``
        
        values : list
            Unique elements of ``array``
        
        indices : list
            Integer list length of ``array`` with the indices of ``values``
            for each element
        
        counts : list
            Counts of each element of ``values``
        

        Methods
        -------
        __get__(key)
            Return a `bool` array where entries of ``array`` match the 
            specified ``key``
        
        __iter__
            Iterator over ``values``
        
        """
        if isinstance(array, list):
            self.array = np.array(array)
        else:
            self.array = array
        
        _ = np.unique(self.array, return_counts=True, return_inverse=True)
        self.dim = self.array.size
        self.zeros = np.zeros(self.array.shape, dtype=bool)
        
        self.values = [l for l in _[0]]
        self.indices = _[1]
        self.counts = _[2]
        if verbose:
            self.info(sort_counts=verbose)
    
    @property
    def N(self):
        """
        Number of unique ``values``
        """
        return len(self.values)


    def info(self, sort_counts=-1):
        """
        Print a summary
        """
        print(f'{"N":>4}  {"value":10}')
        print('====  ==========')
        if sort_counts:
            so = np.argsort(self.counts)[::int(sort_counts)]
        else:
            so = np.arange(self.N)
            
        for i in so:
            v, c = self.values[i], self.counts[i]
            print(f'{c:>4}  {v:10}')


    def count(self, key):
        """
        Get occurrences count of a particular ``value``
        """
        if key in self.values:
            ix = self.values.index(key)
            return self.counts[ix]
        else:
            return 0


    def __iter__(self):
        """
        Iterable over `values` attribute
        
        Returns a tuple of the value and the boolean selection array for that 
        value.
        """
        i = 0
        while i < self.N:
            vi = self.values[i]
            yield (vi, self[vi])
            i += 1


    def __getitem__(self, key):
        if key in self.values:
            ix = self.values.index(key)
            test = self.indices == ix
            return test
        else:
            return self.zeros


    # def __iter__(self):
    #     for idx in itertools.count():
    #         try:
    #             yield self.values[idx]
    #         except IndexError:
    #             break

    def __len__(self):
        return self.N


class HubbleXYZ(object):
    def __init__(self, spt_file='', param_dict={}):
        """
        Helper to compute HST geocentric coordinates from orbital parameters
        
        (testing)
        
        Based on http://articles.adsabs.harvard.edu//full/1995ASPC...77..464A/
        """
        if spt_file:
            self.param_dict = self.parse_from_spt(spt_file)
        
        elif param_dict:
            self.param_dict = param_dict
        
        else:
            self.param_dict = {}
        
        self.computed = {}


    @property 
    def _t1985(self):
        """
        Reference time
        """
        import astropy.time
        t0 = astropy.time.Time('1985-01-01T00:00:00Z')
        return t0


    def __call__(self, t_in, **kwargs):
        """
        Convert input time ``t_in`` to seconds since 1/1/85 and ``evaluate``.
        """
        import astropy.time
        if not isinstance(t_in, astropy.time.Time):
            raise ValueError('t_in must be astropy.time.Time object')

        dt = t_in - self._t1985
        xyz = self.evaluate(dt.sec, **kwargs)
        if 'as_table' in kwargs:
            if kwargs['as_table']:
                xyz['time'] = t_in
                
        return xyz


    def __getitem__(self, key):
        return self.param_dict[key]


    def evaluate(self, dt, unit=None, as_table=False):
        """
        Evaluate equations to get positions
        
        Returns
        -------
        x, y, z, r: float
            Coordinates, in km or ``unit``.
            
        """
        
        if not self.param_dict:
            raise ValueError('Orbital parameters not defined in '
                             'self.param_dict')
                    
        p = self.param_dict
        
        t = np.atleast_1d(dt)
        
        # Eq. 1
        bracket = p['M.']*(t-p['tau']) + 0.5*p['M..']*(t-p['tau'])**2
        M = p['M0'] + 2*np.pi*bracket
        
        # Eq. 2
        sinM = np.sin(M)
        cosM = np.cos(M)
        e = p['e']
        nu = M + sinM*(2*e + 3*e**3*cosM**2 - 4./3*e**3*sinM**2 
                       + 5./2*e**2*cosM)
        
        # Eq. 3
        r = p['a(1-e**2)']/(1+e*np.cos(nu))
        # To km
        r /= 1000.
        
        # Eq. 4
        Om = 2*np.pi*(p['Om0'] + p['Om.']*(t-p['tau']))
        
        # Eq. 5
        w = 2*np.pi*(p['w0'] + p['w.']*(t-p['tau']))
        
        self.calc_dict = {'M':M, 'nu':nu, 
                          'a':p['a'], 
                          'i':np.arcsin(p['sini']), 
                          'Om':Om,
                          'w':w}
        
        # Eq. 6
        cosOm = np.cos(Om)
        sinOm = np.sin(Om)
        coswv = np.cos(w+nu)
        sinwv = np.sin(w+nu)
        
        if unit is not None:
            r = (r*u.km).to(unit)
            
        x = r*(cosOm*coswv - p['cosi']*sinOm*sinwv)
        y = r*(sinOm*coswv + p['cosi']*cosOm*sinwv)
        z = r*p['sini']*sinwv
        
        if as_table:
            tab = GTable()
            tab['dt'] = t
            tab['x'] = x
            tab['y'] = y
            tab['z'] = z
            tab['r'] = r
            return tab
        else: 
            return x, y, z, r


    def from_flt(self, flt_file, **kwargs):
        """
        Compute positions at expstart, expmid, expend
        """
        import astropy.time
        
        flt = pyfits.open(flt_file)
        expstart = flt[0].header['EXPSTART']
        expend = flt[0].header['EXPEND']
        expmid = (expstart+expend)/2.
        
        t_in = astropy.time.Time([expstart, expmid, expend], format='mjd')
        flt.close()
        
        return self(t_in, **kwargs)


    def deltat(self, dt):
        """
        Convert a time ``t`` in seconds from 1/1/85 to an ISO time
        """    
        if not hasattr(dt, 'unit'):
            dtsec = dt*u.second
        else:
            dtsec = dt
            
        t = self._t1985 + dtsec
        return t


    def parse_from_spt(self, spt_file):
        """
        Get orbital elements from SPT header
        """
        import astropy.io.fits as pyfits
        import astropy.time
        
        with pyfits.open(spt_file) as _im:
            spt = _im[0].header.copy()
        
        param_dict = {}
        param_dict['tau'] = spt['EPCHTIME']
        param_dict['M0'] = spt['MEANANOM']
        param_dict['M.'] = spt['FDMEANAN']
        param_dict['M..'] = spt['SDMEANAN']
        param_dict['e'] = spt['ECCENTRY']
        param_dict['a(1-e**2)'] = spt['SEMILREC']
        param_dict['a'] = param_dict['a(1-e**2)'] / (1-param_dict['e']**2)
        param_dict['Om0'] = spt['RASCASCN']
        param_dict['Om.'] = spt['RCASCNRV']
        param_dict['w0'] = spt['ARGPERIG']
        param_dict['w.'] = spt['RCARGPER']
        param_dict['cosi'] = spt['COSINCLI']
        param_dict['sini'] = spt['SINEINCL']
        param_dict['Vc'] = spt['CIRVELOC']
        param_dict['timeffec'] = spt['TIMEFFEC']
        param_dict['Torb'] = spt['HSTHORB']*2
        param_dict['tstart'] = spt['OBSSTRTT']
        
        param_dict['tau_time'] = self.deltat(param_dict['tau'])
        param_dict['tstart_time'] = self.deltat(param_dict['tstart'])
                
        return param_dict
    
    @staticmethod
    def xyz_to_lonlat(self, x, y, z, radians=False):
        """
        Compute sublon, sublat, alt from xyz coords with pyproj
        
        xyz must be in meters
        
        """
        import pyproj
        ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=radians)
        return lon, lat, alt


def patch_photutils():
    """
    Patch to fix inconsistency with drizzlepac=3.2.1 and photutils>1.0, where 
    The latter is needed for jwst=1.3.2
    """
    import os

    try:
        import drizzlepac
    except AttributeError:
    
        import photutils
        site_packages = os.path.dirname(photutils.__file__)
        # manual apply patch
        the_file = f'{site_packages}/../drizzlepac/haputils/align_utils.py'
        with open(the_file,'r') as fp:
            lines = fp.readlines()
    
        print(site_packages, len(lines))
        for i, line in enumerate(lines):
            if line.startswith('NoDetectionsWarning'):
                break
    
        if 'hasattr(photutils.findstars' in lines[i+1]:
            print(f'I found the problem on lines {i}-{i+2}: ')
        else:
            msg = """
Lines {0}-{1} in {2} importing photutils were not as expected.  I found 

{3}

but expected 

   NoDetectionsWarning = photutils.findstars.NoDetectionsWarning if \\
                           hasattr(photutils.findstars, 'NoDetectionsWarning') else \\
                           photutils.utils.NoDetectionsWarning


    """.format(i, i+2, the_file, lines[i:i+3])
        
            raise ValueError(msg)
        
        bad = ['   '+lines.pop(i+2-j) for j in range(3)]
        print(''.join(bad[::-1]))

        # Insert the fix
        lines[i] = 'from photutils.utils.exceptions import NoDetectionsWarning\n\n'
        # Rewrite the fie
        with open(f'{site_packages}/../drizzlepac/haputils/align_utils.py','w') as fp:
            fp.writelines(lines)
    
        print(f'Patch applied to {the_file}!')
