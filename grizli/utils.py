"""
Dumping ground for general utilities
"""
import os
import shutil
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
from unik import Unique

from . import GRIZLI_PATH
from .constants import JWST_DQ_FLAGS, KMS, FLAMBDA_CGS, FNU_CGS

# character to skip clearing line on STDOUT printing
NO_NEWLINE = "\x1b[1A\x1b[1M"

# R_V for Galactic extinction
MW_RV = 3.1

MPL_COLORS = {
    "b": "#1f77b4",
    "orange": "#ff7f0e",
    "g": "#2ca02c",
    "r": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

# sns.color_palette("husl", 8)
SNS_HUSL = {
    "r": (0.9677975592919913, 0.44127456009157356, 0.5358103155058701),
    "orange": (0.8087954113106306, 0.5634700050056693, 0.19502642696727285),
    "olive": (0.5920891529639701, 0.6418467016378244, 0.1935069134991043),
    "g": (0.19783576093349015, 0.6955516966063037, 0.3995301037444499),
    "sea": (0.21044753832183283, 0.6773105080456748, 0.6433941168468681),
    "b": (0.22335772267769388, 0.6565792317435265, 0.8171355503265633),
    "purple": (0.6423044349219739, 0.5497680051256467, 0.9582651433656727),
    "pink": (0.9603888539940703, 0.3814317878772117, 0.8683117650835491),
}

# GRISM_COLORS = {
#     "G800L": (0.0, 0.4470588235294118, 0.6980392156862745),
#     "G102": (0.0, 0.6196078431372549, 0.45098039215686275),
#     "G141": (0.8352941176470589, 0.3686274509803922, 0.0),
#     "none": (0.8, 0.4745098039215686, 0.6549019607843137),
#     "G150": "k",
#     "F277W": (0.0, 0.6196078431372549, 0.45098039215686275),
#     "F356W": (0.8352941176470589, 0.3686274509803922, 0.0),
#     "F444W": (0.8, 0.4745098039215686, 0.6549019607843137),
#     "F250M": "lightblue",
#     "F300M": "steelblue",
#     "F335M": "cornflowerblue",
#     "F360M": "royalblue",
#     "F410M": (0.0, 0.4470588235294118, 0.6980392156862745),
#     "F430M": "sandybrown",
#     "F460M": "lightsalmon",
#     "F480M": "coral",
#     "F322W2": "olive",
#     "G280": "purple",
#     "F090W": (0.0, 0.4470588235294118, 0.6980392156862745),
#     "F115W": (0.0, 0.6196078431372549, 0.45098039215686275),
#     "F150W": (0.8352941176470589, 0.3686274509803922, 0.0),
#     "F140M": (0.8352941176470589, 0.3686274509803922, 0.0),
#     "F158M": (0.8352941176470589, 0.3686274509803922, 0.0),
#     "F200W": (0.8, 0.4745098039215686, 0.6549019607843137),
#     "F140M": "orange",
#     "BLUE": "#1f77b4",  # Euclid
#     "RED": "#d62728",
#     "CLEARP": "b",
# }

GRISM_COLORS = {
 'G800L': 'cornflowerblue',
  'G102': 'mediumseagreen',
  'G141': 'chocolate',
  'none': 'pink',
  'G150': 'grey',
 'F277W': 'lightseagreen',
 'F356W': 'sandybrown',
 'F444W': 'violet',
 'F250M': 'lightblue',
 'F300M': 'skyblue',
 'F335M': 'steelblue',
 'F360M': 'cornflowerblue',
 'F410M': 'lightcoral',
 'F430M': 'tomato',
 'F460M': 'indianred',
 'F480M': 'maroon',
'F322W2': 'olive',
  'G280': 'blueviolet',
 'F090W': 'lightsteelblue',
 'F115W': 'forestgreen',
 'F150W': 'goldenrod',
 'F140M': 'peachpuff',
 'F158M': 'lightsalmon',
 'F200W': 'palevioletred',
  'BLUE': 'dodgerblue',
   'RED': 'crimson',
'CLEARP': 'silver',
}


GRISM_MAJOR = {
    "G102": 0.1,
    "G141": 0.1,  # WFC3/IR
    "G800L": 0.1,  # ACS/WFC
    "F090W": 0.1,
    "F115W": 0.1,
    "F150W": 0.1,  # NIRISS
    "F140M": 0.1,
    "F158M": 0.1,
    "F200W": 0.1,
    "F277W": 0.2,
    "F356W": 0.2,
    "F444W": 0.2,  # NIRCam
    "F250M": 0.1,
    "F300M": 0.1,
    "F335M": 0.1,
    "F360M": 0.1,
    "F410M": 0.1,
    "F430M": 0.1,
    "F460M": 0.1,
    "F480M": 0.1,
    "F322W2": 0.1,
    "BLUE": 0.1,
    "RED": 0.1,  # Euclid
    "GRISM": 0.1,
    "G150": 0.1,  # Roman
}

GRISM_LIMITS = {
    "G800L": [0.545, 1.02, 40.0],  # ACS/WFC
    "G280": [0.2, 0.4, 14],  # WFC3/UVIS
    "G102": [0.77, 1.18, 23.0],  # WFC3/IR
    "G141": [1.06, 1.73, 46.0],
    "GRISM": [0.98, 1.98, 11.0],  # WFIRST/Roman
    "G150": [0.98, 1.98, 11.0],
    "F090W": [0.76, 1.04, 45.0],  # NIRISS
    "F115W": [0.97, 1.32, 45.0],
    "F140M": [1.28, 1.52, 45.0],
    "F158M": [1.28, 1.72, 45.0],
    "F150W": [1.28, 1.72, 45.0],
    "F200W": [1.68, 2.30, 45.0],
    "F140M": [1.20, 1.60, 45.0],
    "CLEARP": [0.76, 2.3, 45.0],
    "F277W": [2.5, 3.2, 20.0],  # NIRCAM
    "F356W": [3.05, 4.1, 20.0],
    "F444W": [3.82, 5.08, 20],
    "F250M": [2.4, 2.65, 20],
    "F300M": [2.77, 3.23, 20],
    "F335M": [3.1, 3.6, 20],
    "F360M": [3.4, 3.85, 20],
    "F410M": [3.8, 4.38, 20],
    "F430M": [4.1, 4.45, 20],
    "F460M": [4.5, 4.8, 20],
    "F480M": [4.6, 5.05, 20],
    "F322W2": [2.42, 4.09, 20],
    "BLUE": [0.8, 1.2, 10.0],  # Euclid
    "RED": [1.1, 1.9, 14.0],
}

# DEFAULT_LINE_LIST = ['PaB', 'HeI-1083', 'SIII', 'OII-7325', 'ArIII-7138', 'SII', 'Ha+NII', 'OI-6302', 'HeI-5877', 'OIII', 'Hb', 'OIII-4363', 'Hg', 'Hd', 'H8','H9','NeIII-3867', 'OII', 'NeVI-3426', 'NeV-3346', 'MgII','CIV-1549', 'CIII-1908', 'OIII-1663', 'HeII-1640', 'NIII-1750', 'NIV-1487', 'NV-1240', 'Lya']

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

# Clipping threshold for BKG extensions in drizzle_from_visit
# BKG_CLIP = [scale, percentile_lo, percentile_hi]
# BKG_CLIP = [2, 1, 99]
BKG_CLIP = None


def set_warnings(numpy_level="ignore", astropy_level="ignore"):
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


JWST_TRANSLATE = {
    "RA_TARG": "TARG_RA",
    "DEC_TARG": "TARG_DEC",
    "EXPTIME": "EFFEXPTM",
    "PA_V3": "ROLL_REF",
}


def get_flt_info(
    files=[],
    columns=[
        "FILE",
        "FILTER",
        "PUPIL",
        "INSTRUME",
        "DETECTOR",
        "TARGNAME",
        "DATE-OBS",
        "TIME-OBS",
        "EXPSTART",
        "EXPTIME",
        "PA_V3",
        "RA_TARG",
        "DEC_TARG",
        "POSTARG1",
        "POSTARG2",
    ],
    translate=JWST_TRANSLATE,
    defaults={"PUPIL": "---", "TARGNAME": "indef", "PA_V3": 0.0},
    jwst_detector=True,
):
    """
    Extract header information from a list of FLT files

    Parameters
    ----------
    files : list, optional
        List of exposure filenames. 
        If not provided, 
        it will search for all "*flt.fits" files in the current directory.
    
    columns : list, optional
        List of header keywords to extract from the FITS files. 
        The default columns include:
        - "FILE": Filename
        - "FILTER": Filter used
        - "PUPIL": Pupil element used
        - "INSTRUME": Instrument name
        - "DETECTOR": Detector name
        - "TARGNAME": Target name
        - "DATE-OBS": Observation date
        - "TIME-OBS": Observation time
        - "EXPSTART": Exposure start time
        - "EXPTIME": Exposure time
        - "PA_V3": Position angle of V3 axis
        - "RA_TARG": Right ascension of target
        - "DEC_TARG": Declination of target
        - "POSTARG1": Post-slew offset in axis 1
        - "POSTARG2": Post-slew offset in axis 2

    translate : dict, optional
        Dictionary mapping header keywords to their corresponding FITS keywords. 
        Default is JWST_TRANSLATE.

    defaults : dict, optional
        Dictionary of default values for header keywords 
        that are not present in the FITS files. Default values include:
        - "PUPIL": "---"
        - "TARGNAME": "indef"
        - "PA_V3": 0.0

    jwst_detector : bool, optional
        Flag indicating whether the FITS files are from JWST detectors. 
        Default is True.

    Returns
    -------
    tab : `~astropy.table.Table`
        Table containing header keywords extracted from the FITS files.
    
    """
    
    import astropy.io.fits as pyfits
    from astropy.table import Table

    if not files:
        files = glob.glob("*flt.fits")

    N = len(files)

    data = []

    for c in columns[2:]:
        if c not in translate:
            translate[c] = "xxxxxxxxxxxxxx"

    targprop = []

    for i in range(N):
        line = [os.path.basename(files[i]).split(".gz")[0]]
        if files[i].endswith(".gz"):
            im = pyfits.open(files[i])
            h = im[0].header.copy()
            im.close()
        else:
            h = pyfits.Header().fromfile(files[i])

        if os.path.basename(files[i]).startswith("jw0"):
            with pyfits.open(files[i]) as _im:
                h1 = _im["SCI"].header
                if "PA_V3" in h1:
                    h["PA_V3"] = h1["PA_V3"]

                if "TARGPROP" in h:
                    targprop.append(h["TARGPROP"].lower())
                else:
                    targprop.append("indef")
        else:
            targprop.append("indef")

        filt = parse_filter_from_header(h, jwst_detector=jwst_detector)
        line.append(filt)
        has_columns = ["FILE", "FILTER"]

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

    if "TARGNAME" in tab.colnames:
        miss = tab["TARGNAME"] == ""
        targs = [t.replace(" ", "-") for t in tab["TARGNAME"]]

        if miss.sum() > 0:
            for i in np.where(miss)[0]:
                targs[i] = targprop[i]  #'indef'

        tab["TARGNAME"] = targs

    return tab


def radec_to_targname(
    ra=0,
    dec=0,
    round_arcsec=(4, 60),
    precision=2,
    targstr="j{rah}{ram}{ras}{sign}{ded}{dem}",
    header=None,
):
    """
    Turn decimal degree coordinates into a string with rounding.

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
        if "CRVAL1" in header:
            ra, dec = header["CRVAL1"], header["CRVAL2"]
        else:
            if "RA_TARG" in header:
                ra, dec = header["RA_TARG"], header["DEC_TARG"]

    cosd = np.cos(dec / 180 * np.pi)
    scl = np.array(round_arcsec) / 3600 * np.array([360 / 24, 1])

    dec_scl = int(np.round(dec / scl[1])) * scl[1]
    ra_scl = int(np.round(ra / scl[0])) * scl[0]

    coo = astropy.coordinates.SkyCoord(
        ra=ra_scl * u.deg, dec=dec_scl * u.deg, frame="icrs"
    )

    cstr = re.split("[hmsd.]", coo.to_string("hmsdms", precision=precision))
    # targname = ('j{0}{1}'.format(''.join(cstr[0:3]), ''.join(cstr[4:7])))
    # targname = targname.replace(' ', '').replace('+','p').replace('-','m')

    rah, ram, ras, rass = cstr[0:4]
    ded, dem, des, dess = cstr[4:8]
    sign = "p" if ded[1] == "+" else "m"

    targname = targstr.format(
        rah=rah,
        ram=ram,
        ras=ras,
        rass=rass,
        ded=ded[2:],
        dem=dem,
        des=des,
        dess=dess,
        sign=sign,
    )

    return targname


def blot_nearest_exact(
    in_data,
    in_wcs,
    out_wcs,
    verbose=True,
    stepsize=-1,
    scale_by_pixel_area=False,
    wcs_mask=True,
    fill_value=0,
):
    """
    Own blot function for blotting exact pixels without rescaling for input
    and output pixel size
    
    Parameters
    ----------
    in_data : `~numpy.ndarray`
        Input data to blot.

    in_wcs : `~astropy.wcs.WCS`
        Input WCS.  Must have _naxis1, _naxis2 or pixel_shape attributes.

    out_wcs : `~astropy.wcs.WCS`
        Output WCS.  Must have _naxis1, _naxis2 or pixel_shape attributes.

    verbose : bool, optional
        If True, print information about the overlap. Default is True.

    stepsize : int, optional
        Step size for interpolation. If set to <=1, the function will use the explicit
        wcs mapping ``out_wcs.all_pix2world > in_wcs.all_world2pix``.  If > 1,
        will use
        ``astrodrizzle.DefaultWCSMapping(out_wcs, in_wcs, nx, ny, stepsize)``.

    scale_by_pixel_area : bool
        If True, then scale the output image by the square of the image pixel
        scales (out**2/in**2), i.e., the pixel areas.

    wcs_mask : bool
        Use fast WCS masking.  If False, use ``regions``.

    fill_value : int/float
        Value in ``out_data`` not covered by ``in_data``.

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
        from .utils_numba.interp import pixel_map_c
    except ImportError:
        from grizli.utils_numba.interp import pixel_map_c

    # Shapes, in numpy array convention (y, x)
    if hasattr(in_wcs, "pixel_shape"):
        in_sh = in_wcs.pixel_shape[::-1]
    elif hasattr(in_wcs, "array_shape"):
        in_sh = in_wcs.array_shape
    else:
        in_sh = (in_wcs._naxis2, in_wcs._naxis1)

    if hasattr(out_wcs, "pixel_shape"):
        out_sh = out_wcs.pixel_shape[::-1]
    elif hasattr(out_wcs, "array_shape"):
        out_sh = out_wcs.array_shape
    else:
        out_sh = (out_wcs._naxis2, out_wcs._naxis1)

    in_px = in_wcs.calc_footprint()
    in_poly = Polygon(in_px).buffer(5.0 / 3600.0)

    out_px = out_wcs.calc_footprint()
    out_poly = Polygon(out_px).buffer(5.0 / 3600)

    olap = in_poly.intersection(out_poly)
    if olap.area == 0:
        if verbose:
            print("No overlap")
        return np.zeros(out_sh)

    # Region mask for speedup
    if np.isclose(olap.area, out_poly.area, 0.01):
        mask = np.ones(out_sh, dtype=bool)
    elif wcs_mask:
        # Use wcs / Path
        from matplotlib.path import Path

        out_xy = out_wcs.all_world2pix(np.array(in_poly.exterior.xy).T, 0) - 0.5
        out_xy_path = Path(out_xy)
        yp, xp = np.indices(out_sh)
        pts = np.array([xp.flatten(), yp.flatten()]).T
        mask = out_xy_path.contains_points(pts).reshape(out_sh)
    else:
        olap_poly = np.array(olap.exterior.xy)
        poly_reg = (
            "fk5\npolygon("
            + ",".join(["{0}".format(p + 1) for p in olap_poly.T.flatten()])
            + ")\n"
        )
        reg = Regions.parse(poly_reg, format="ds9")[0]
        mask = reg.to_mask().to_image(shape=out_sh)

    # yp, xp = np.indices(in_data.shape)
    # xi, yi = xp[mask], yp[mask]
    yo, xo = np.where(mask > 0)

    if stepsize <= 1:
        rd = out_wcs.all_pix2world(xo, yo, 0)
        xf, yf = in_wcs.all_world2pix(rd[0], rd[1], 0)
    else:
        # Seems backwards and doesn't quite agree with above
        blot_wcs = out_wcs
        source_wcs = in_wcs

        if hasattr(blot_wcs, "pixel_shape"):
            nx, ny = blot_wcs.pixel_shape
        else:
            nx, ny = int(blot_wcs._naxis1), int(blot_wcs._naxis2)

        mapping = cdriz.DefaultWCSMapping(blot_wcs, source_wcs, nx, ny, stepsize)
        xf, yf = mapping(xo, yo)

    xi, yi = np.asarray(np.round(xf),dtype=int), np.asarray(np.round(yf),dtype=int)

    m2 = (xi >= 0) & (yi >= 0) & (xi < in_sh[1]) & (yi < in_sh[0])
    xi, yi, xf, yf, xo, yo = xi[m2], yi[m2], xf[m2], yf[m2], xo[m2], yo[m2]

    out_data = np.ones(out_sh, dtype=np.float64)*fill_value
    status = pixel_map_c(np.asarray(in_data,dtype=np.float64), xi, yi, out_data, xo, yo)

    # Fill empty
    func = nd.maximum_filter
    fill = out_data == 0
    filtered = func(out_data, size=5)
    out_data[fill] = filtered[fill]

    if scale_by_pixel_area:
        in_scale = get_wcs_pscale(in_wcs)
        out_scale = get_wcs_pscale(out_wcs)
        out_data *= out_scale ** 2 / in_scale ** 2

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
    filtered = filter_func(data, *args, size=size, footprint=footprint, **kwargs)

    return filtered, slices


def multiprocessing_ndfilter(
    data,
    filter_func,
    filter_args=(),
    size=None,
    footprint=None,
    cutout_size=256,
    n_proc=4,
    timeout=90,
    mask=None,
    verbose=True,
    **kwargs,
):
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
        msg = f"cutout_size={cutout_size} greater than image dimensions, run "
        msg += f"`{filter_func}` directly"
    elif n_proc == 0:
        msg = f"n_proc = 0, run in a single command"

    if msg is not None:
        if verbose:
            print(msg)

        filtered = filter_func(data, *filter_args, size=size, footprint=footprint)
        return filtered

    # Grid size
    nx = data.shape[1] // cutout_size + 1
    ny = data.shape[0] // cutout_size + 1

    # Padding
    if footprint is not None:
        fpsh = footprint.shape
        pad = np.max(fpsh)
    elif size is not None:
        pad = size
    else:
        raise ValueError("Either size or footprint must be specified")

    if n_proc < 0:
        n_proc = mp.cpu_count()

    n_proc = np.minimum(n_proc, mp.cpu_count())

    pool = mp.Pool(processes=n_proc)
    jobs = []

    if mask is not None:
        data_mask = data * mask
    else:
        data_mask = data

    # Make image slices
    for i in range(nx):
        xmi = np.maximum(0, i * cutout_size - pad)
        xma = np.minimum(sh[1], (i + 1) * cutout_size + pad)

        # print(i, xmi, xma)
        if i == 0:
            slx = slice(0, cutout_size)
            x0 = 0
        elif i < nx - 1:
            slx = slice(pad, cutout_size + pad)
            x0 = i * cutout_size
        else:
            slx = slice(pad, cutout_size + 1)
            x0 = xmi + pad

        nxs = slx.stop - slx.start
        oslx = slice(x0, x0 + nxs)

        for j in range(ny):
            ymi = np.maximum(0, j * cutout_size - pad)
            yma = np.minimum(sh[0], (j + 1) * cutout_size + pad)

            if j == 0:
                sly = slice(0, cutout_size)
                y0 = 0
            elif j < ny - 1:
                sly = slice(pad, cutout_size + pad)
                y0 = j * cutout_size
            else:
                sly = slice(pad, cutout_size + 1)
                y0 = ymi + pad

            nys = sly.stop - sly.start
            osly = slice(y0, y0 + nys)

            cut = data_mask[ymi:yma, xmi:xma]
            if cut.max() == 0:
                # print(f'Skip {xmi} {xma} {ymi} {yma}')
                continue

            # Make jobs for filtering the image slices
            slices = (osly, oslx, sly, slx)
            _args = (cut, filter_func, slices, filter_args, size, footprint, kwargs)
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


def parse_flt_files(
    files=[],
    info=None,
    uniquename=False,
    use_visit=False,
    get_footprint=False,
    translate={
        "AEGIS-": "aegis-",
        "COSMOS-": "cosmos-",
        "GNGRISM": "goodsn-",
        "GOODS-SOUTH-": "goodss-",
        "UDS-": "uds-",
    },
    visit_split_shift=1.5,
    max_dt=1e9,
    path="../RAW",
):
    """
    Read header information from a list of exposures and parse out groups
     based on filter/target/orientation.

    Parameters
    ----------
    files : list, optional
        List of exposure filenames.  If not specified, will use ``*flt.fits``.

    info : None or `~astropy.table.Table`, optional
        Output from `~grizli.utils.get_flt_info`.

    uniquename : bool, optional
        If True, then split everything by program ID and visit name.  If
        False, then just group by targname/filter/pa_v3.


    use_visit : bool, optional
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
        
    get_footprint : bool, optional
        If True, get the visit footprint from FLT WCS.

    translate : dict, optional
        Translation dictionary to modify TARGNAME keywords to some other
        value.  Used like:

            >>> targname = 'GOODS-SOUTH-10'
            >>> translate = {'GOODS-SOUTH-': 'goodss-'}
            >>> for k in translate:
            >>>     targname = targname.replace(k, translate[k])
            >>> print(targname)
            goodss-10

    visit_split_shift : float, optional
        Separation in ``arcmin`` beyond which exposures in a group are split
        into separate visits.

    max_dt : float, optional
        Maximum time separation between exposures in a visit, in seconds.

    path : str, optional
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
            files = glob.glob(os.path.join(path), "*flt.fits")

        if len(files) == 0:
            return False

        info = get_flt_info(files)
    else:
        info = info.copy()

    for c in info.colnames:
        if not c.islower():
            info.rename_column(c, c.lower())

    if "expstart" not in info.colnames:
        info["expstart"] = info["exptime"] * 0.0

    so = np.argsort(info["expstart"])
    info = info[so]

    # pa_v3 = np.round(info['pa_v3']*10)/10 % 360.
    pa_v3 = np.round(np.round(info["pa_v3"], decimals=1)) % 360.0

    target_list = []
    for i in range(len(info)):
        # Replace ANY targets with JRhRmRs-DdDmDs
        if info["targname"][i] == "ANY":
            if use_visit:
                new_targname = info["file"][i][:6]
            else:
                new_targname = "par-" + radec_to_targname(
                    ra=info["ra_targ"][i], dec=info["dec_targ"][i]
                )

            target_list.append(new_targname.lower())
        else:
            target_list.append(info["targname"][i])

    target_list = np.array(target_list)

    _prog_ids = []
    visits = []

    for file in info["file"]:
        bfile = os.path.basename(file)
        if bfile.startswith("jw"):
            _prog_ids.append(bfile[2:7])
            visits.append(bfile[7:10])
        else:
            _prog_ids.append(bfile[1:4])
            visits.append(bfile[4:6])

    visits = np.array(visits)

    info["progIDs"] = _prog_ids

    progIDs = np.unique(info["progIDs"])
    dates = np.array(["".join(date.split("-")[1:]) for date in info["date-obs"]])

    targets = np.unique(target_list)

    output_list = []  # OrderedDict()
    filter_list = OrderedDict()

    for filter in np.unique(info["filter"]):
        filter_list[filter] = OrderedDict()
        angles = np.unique(pa_v3[(info["filter"] == filter)])
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
                spl = target_use.split("-")
                try:
                    if (int(spl[-1]) < 10) & (len(spl[-1]) == 1):
                        spl[-1] = "{0:02d}".format(int(spl[-1]))
                        target_use = "-".join(spl)
                except:
                    pass

        for filter in np.unique(info["filter"][(target_list == target)]):
            angles = np.unique(
                pa_v3[(info["filter"] == filter) & (target_list == target)]
            )
            for angle in angles:

                exposure_list = []
                exposure_start = []
                product = "{0}-{1:05.1f}-{2}".format(target_use, angle, filter)
                visit_match = np.unique(
                    visits[(target_list == target) & (info["filter"] == filter)]
                )

                this_progs = []
                this_visits = []

                for visit in visit_match:
                    ix = (visits == visit) & (target_list == target)
                    ix &= info["filter"] == filter

                    # this_progs.append(info['progIDs'][ix][0])
                    # print visit, ix.sum(), np.unique(info['progIDs'][ix])
                    new_progs = list(np.unique(info["progIDs"][ix]))
                    this_visits.extend([visit] * len(new_progs))
                    this_progs.extend(new_progs)

                for visit, prog in zip(this_visits, this_progs):
                    visit_list = []
                    visit_start = []

                    _vstr = "{0}-{1}-{2}-{3:05.1f}-{4}"
                    visit_product = _vstr.format(target_use, prog, visit, angle, filter)

                    use = target_list == target
                    use &= info["filter"] == filter
                    use &= visits == visit
                    use &= pa_v3 == angle
                    use &= info["progIDs"] == prog

                    if use.sum() == 0:
                        continue

                    for tstart, file in zip(info["expstart"][use], info["file"][use]):

                        f = file.split(".gz")[0]
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
                        # output_list[visit_product.lower()] = visit_list

                        d = OrderedDict(
                            product=str(visit_product.lower()),
                            files=list(np.array(visit_list)[so]),
                        )
                        output_list.append(d)

                if not uniquename:
                    print(product, len(exposure_list))
                    so = np.argsort(exposure_start)
                    exposure_list = np.array(exposure_list)[so]
                    # output_list[product.lower()] = exposure_list
                    d = OrderedDict(
                        product=str(product.lower()),
                        files=list(np.array(exposure_list)[so]),
                    )
                    output_list.append(d)

    # Split large shifts
    if visit_split_shift > 0:
        split_list = []
        for o in output_list:
            _spl = split_visit(
                o, path=path, max_dt=max_dt, visit_split_shift=visit_split_shift
            )

            split_list.extend(_spl)

        output_list = split_list

    # Get visit footprint from FLT WCS
    if get_footprint:
        from shapely.geometry import Polygon

        N = len(output_list)
        for i in range(N):
            for j in range(len(output_list[i]["files"])):
                flt_file = output_list[i]["files"][j]
                if not os.path.exists(flt_file):
                    for gzext in ["", ".gz"]:
                        _flt_file = os.path.join(path, flt_file + gzext)
                        if os.path.exists(_flt_file):
                            flt_file = _flt_file
                            break

                flt_j = pyfits.open(flt_file)
                h = flt_j[0].header
                _ext = 0
                if h["INSTRUME"] == "WFC3":
                    _ext = 1
                    if h["DETECTOR"] == "IR":
                        wcs_j = pywcs.WCS(flt_j["SCI", 1])
                    else:
                        wcs_j = pywcs.WCS(flt_j["SCI", 1], fobj=flt_j)
                elif h["INSTRUME"] == "WFPC2":
                    _ext = 1
                    wcs_j = pywcs.WCS(flt_j["SCI", 1])
                else:
                    _ext = 1
                    wcs_j = pywcs.WCS(flt_j["SCI", 1], fobj=flt_j)

                if (wcs_j.pixel_shape is None) & ("NPIX1" in flt_j["SCI", 1].header):
                    _h = flt_j["SCI", 1].header
                    wcs_j.pixel_shape = (_h["NPIX1"], _h["NPIX2"])

                fp_j = Polygon(wcs_j.calc_footprint())
                if j == 0:
                    fp_i = fp_j.buffer(1.0 / 3600)
                else:
                    fp_i = fp_i.union(fp_j.buffer(1.0 / 3600))

                flt_j.close()

            output_list[i]["footprint"] = fp_i

    return output_list, filter_list


def split_visit(visit, visit_split_shift=1.5, max_dt=6.0 / 24, path="../RAW"):
    """
    Check if files in a visit have large shifts and split them otherwise

    Parameters
    ----------
    visit : dict
        The visit dictionary containing information about the visit.

    visit_split_shift : float, optional
        The threshold for splitting the visit if shifts are larger than
        ``visit_split_shift`` arcmin. Default is 1.5.

    max_dt : float, optional
        The maximum time difference between exposures in days.
        Default is 6.0 / 24.

    path : str, optional
        The path to the directory containing the visit files.
        Default is "../RAW".

    Returns
    -------
    list of dict
        A list of visit dictionaries, each representing a split visit.
    
    """
    ims = []
    for file in visit["files"]:
        for gzext in ["", ".gz"]:
            _file = os.path.join(path, file) + gzext
            if os.path.exists(_file):
                ims.append(pyfits.open(_file))
                break
    
    #ims = [pyfits.open(os.path.join(path, file)) for file in visit['files']]
    crval1 = np.array([im[1].header['CRVAL1'] for im in ims])
    crval2 = np.array([im[1].header['CRVAL2'] for im in ims])
    expstart = np.array([im[0].header['EXPSTART'] for im in ims])
    dt = np.asarray((expstart-expstart[0])/max_dt,dtype=int)
    
    for im in ims:
        im.close()

    dx = (crval1 - crval1[0]) * 60 * np.cos(crval2[0] / 180 * np.pi)
    dy = (crval2 - crval2[0]) * 60
    

    dxi = np.asarray(np.round(dx/visit_split_shift),dtype=int)
    dyi = np.asarray(np.round(dy/visit_split_shift),dtype=int)
    keys = dxi*100+dyi+1000*dt


    un = np.unique(keys)
    if len(un) == 1:
        return [visit]
    else:
        spl = visit["product"].split("-")
        isJWST = spl[-1].lower().startswith("clear")
        isJWST |= spl[-1].lower() in ["gr150r", "gr150c", "grismr", "grismc"]
        if isJWST:
            spl.insert(-2, "")
        else:
            spl.insert(-1, "")

        visits = []
        for i in range(len(un)):
            ix = keys == un[i]
            if isJWST:
                spl[-3] = "abcdefghijklmnopqrsuvwxyz"[i]
            else:
                spl[-2] = "abcdefghijklmnopqrsuvwxyz"[i]

            new_visit = {
                "files": list(np.array(visit["files"])[ix]),
                "product": "-".join(spl),
            }

            if "footprints" in visit:
                new_visit["footprints"] = list(np.array(visit["footprints"])[ix])

            visits.append(new_visit)

    return visits


def get_visit_footprints(visits):
    """
    Add `~shapely.geometry.Polygon` ``footprint`` attributes to visit dict.

    Parameters
    ----------
    visits : list
        List of visit dictionaries.

    Returns
    -------
    list
        List of visit dictionaries with ``footprint`` attribute added.

    """

    import os

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    from shapely.geometry import Polygon

    N = len(visits)
    for i in range(N):
        for j in range(len(visits[i]["files"])):
            flt_file = visits[i]["files"][j]
            if (not os.path.exists(flt_file)) & os.path.exists("../RAW/" + flt_file):
                flt_file = "../RAW/" + flt_file

            flt_j = pyfits.open(flt_file)
            h = flt_j[0].header
            if (h["INSTRUME"] == "WFC3") & (h["DETECTOR"] == "IR"):
                wcs_j = pywcs.WCS(flt_j["SCI", 1])
            else:
                wcs_j = pywcs.WCS(flt_j["SCI", 1], fobj=flt_j)

            fp_j = Polygon(wcs_j.calc_footprint())
            if j == 0:
                fp_i = fp_j
            else:
                fp_i = fp_i.union(fp_j)

            flt_j.close()

        visits[i]["footprint"] = fp_i

    return visits


def parse_visit_overlaps(visits, buffer=15.0):
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
        f_i = visits[i]["product"].split("-")[-1]
        if used[i]:
            continue

        if "footprint" in visits[i]:
            fp_i = visits[i]["footprint"].buffer(buffer / 3600.0)
        else:
            _products = visits[i]["product"] + "_dr?_sci.fits"
            im_i = pyfits.open(glob.glob(_products)[0])
            wcs_i = pywcs.WCS(im_i[0])
            fp_i = Polygon(wcs_i.calc_footprint()).buffer(buffer / 3600.0)
            im_i.close()

        exposure_groups.append(copy.deepcopy(visits[i]))

        for j in range(i + 1, N):
            f_j = visits[j]["product"].split("-")[-1]
            if (f_j != f_i) | (used[j]):
                continue

            #
            if "footprint" in visits[j]:
                fp_j = visits[j]["footprint"].buffer(buffer / 3600.0)
            else:
                _products = visits[j]["product"] + "_dr?_sci.fits"
                im_j = pyfits.open(glob.glob(_products)[0])
                wcs_j = pywcs.WCS(im_j[0])
                fp_j = Polygon(wcs_j.calc_footprint()).buffer(buffer / 3600.0)
                im_j.close()

            olap = fp_i.intersection(fp_j)
            if olap.area > 0:
                used[j] = True
                fp_i = fp_i.union(fp_j)
                exposure_groups[-1]["footprint"] = fp_i
                exposure_groups[-1]["files"].extend(visits[j]["files"])

    for i in range(len(exposure_groups)):
        flt_i = pyfits.open(exposure_groups[i]["files"][0])
        product = flt_i[0].header["TARGNAME"].lower()
        if product == "any":
            product = "par-" + radec_to_targname(header=flt_i["SCI", 1].header)

        f_i = exposure_groups[i]["product"].split("-")[-1]
        product += "-" + f_i
        exposure_groups[i]["product"] = product
        flt_i.close()

    return exposure_groups


DIRECT_ORDER = {
    "G102": [
        "F105W","F110W","F098M","F125W","F140W","F160W","F127M","F139M","F153M",
        "F132N","F130N","F128N","F126N","F164N","F167N",
    ],
    "G141": [
        "F140W","F160W","F125W","F105W","F110W","F098M","F127M","F139M","F153M",
        "F132N","F130N","F128N","F126N","F164N","F167N",
    ],
    "G800L": [
        "F814W","F606W","F850LP","F775W","F435W","F105W","F110W","F098M","F125W",
        "F140W","F160W","F127M","F139M","F153M","F132N","F130N","F128N","F126N",
        "F164N","F167N",
    ],
    "GR150C": ["F115W", "F150W", "F200W"],
    "GR150R": ["F115W", "F150W", "F200W"],
}


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
    """
    Get simple filter name out of an HST/JWST image header.

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

    if "INSTRUME" not in header:
        instrume = "N/A"
    else:
        instrume = header["INSTRUME"]

    if instrume.strip() == "ACS":
        for i in [1, 2]:
            filter_i = header["FILTER{0:d}".format(i)]
            if "CLEAR" in filter_i:
                continue
            else:
                filter = filter_i

    elif instrume == "WFPC2":
        filter = header["FILTNAM1"]

    elif instrume == "NIRISS":
        if filter_only:
            filter = header["FILTER"]
        else:
            filter = "{0}-{1}".format(header["PUPIL"], header["FILTER"])

        if jwst_detector:
            filter = "{0}-{1}".format(header["DETECTOR"], filter)

    elif instrume == "NIRCAM":
        if filter_only:
            filter = header["FILTER"]
        else:
            filter = "{0}-{1}".format(header["FILTER"], header["PUPIL"])
        if jwst_detector:
            filter = "{0}-{1}".format(header["DETECTOR"], filter)
            filter = filter.replace("LONG", "5")

    elif "FILTER" in header:
        filter = header["FILTER"]

    else:
        msg = "Failed to parse FILTER keyword for INSTRUMEnt {0}"
        raise KeyError(msg.format(instrume))

    return filter.upper()


EE_RADII = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0]


def get_filter_obsmode(
    filter="f160w", acs_chip="wfc1", uvis_chip="uvis2", aper=np.inf, case=str.lower
):
    """
    Derive `~pysynphot` obsmode keyword from a filter name, where UVIS filters
    end in 'u'

    Parameters
    ----------
    filter : str, optional
        The name of the filter. Default is "f160w".
    acs_chip : str, optional
        The ACS chip. Default is "wfc1".
    uvis_chip : str, optional
        The UVIS chip. Default is "uvis2".
    aper : float, optional
        The aperture size. Set to np.inf by default.
    case : function, optional
        The case conversion function. Default is str.lower.

    Returns
    -------
    str
        The `~pysynphot` obsmode keyword derived from the filter name.

    """
    if filter.lower()[:2] in ["f0", "f1", "g1"]:
        inst = "wfc3,ir"
    else:
        if filter.lower().endswith("u"):
            inst = f"wfc3,{uvis_chip}"
        else:
            inst = f"acs,{acs_chip}"

    obsmode = inst + "," + filter.strip("u").lower()
    if np.isfinite(aper):
        obsmode += f",aper#{aper:4.2f}"

    return case(obsmode)


def tabulate_encircled_energy(aper_radii=EE_RADII, norm_radius=4.0):
    """
    Tabulated encircled energy for different aperture radii 
    and normalization radius.

    Parameters
    ----------
    aper_radii : list, optional
        List of aperture radii in arcseconds. 
        Default is [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0].

    norm_radius : float, optional
        Normalization radius in arcseconds. Default is 4.0.

    """

    import pysynphot as S
    from .pipeline import default_params

    # Default spectrum
    sp = S.FlatSpectrum(25, fluxunits="ABMag")

    tab = GTable()
    tab["radius"] = aper_radii * u.arcsec
    tab.meta["RNORM"] = norm_radius, "Normalization radius, arcsec"

    # IR
    for f in default_params.IR_M_FILTERS + default_params.IR_W_FILTERS:
        obsmode = "wfc3,ir," + f.lower()

        print(obsmode)
        tab[obsmode] = synphot_encircled_energy(
            obsmode=obsmode, sp=sp, aper_radii=aper_radii, norm_radius=norm_radius
        )
        tab.meta["ZP_{0}".format(obsmode)] = synphot_zeropoint(
            obsmode=obsmode, radius=norm_radius
        )

    # Optical.  Wrap in try/except to catch missing filters
    for inst in ["acs,wfc1,", "wfc3,uvis2,"]:
        for f in (
            default_params.OPT_M_FILTERS
            + default_params.OPT_W_FILTERS
            + default_params.UV_M_FILTERS
            + default_params.UV_W_FILTERS
        ):

            obsmode = inst + f.lower()

            try:
                tab[obsmode] = synphot_encircled_energy(
                    obsmode=obsmode,
                    sp=sp,
                    aper_radii=aper_radii,
                    norm_radius=norm_radius,
                )
                print(obsmode)
                tab.meta["ZP_{0}".format(obsmode)] = synphot_zeropoint(
                    obsmode=obsmode, radius=norm_radius
                )
            except:
                # Failed because obsmode not available in synphot
                continue

    tab.meta["PSYNVER"] = S.__version__, "Pysynphot version"

    tab.write("hst_encircled_energy.fits", overwrite=True)


def synphot_zeropoint(obsmode="wfc3,ir,f160w", radius=4.0):
    """
    Compute synphot for a specific aperture.

    Parameters
    ----------
    obsmode : str, optional
        The observation mode string. Default is "wfc3,ir,f160w".
    radius : float, optional
        The radius of the aperture in arcseconds. Default is 4.0.

    Returns
    -------
    ZP : float
        The zero point magnitude calculated using synphot.
    
    """
    import pysynphot as S

    sp = S.FlatSpectrum(25, fluxunits="ABMag")

    if np.isfinite(radius):
        bp = S.ObsBandpass(obsmode + ",aper#{0:.2f}".format(radius))
    else:
        bp = S.ObsBandpass(obsmode)

    obs = S.Observation(sp, bp)
    ZP = 25 + 2.5 * np.log10(obs.countrate())
    return ZP


def synphot_encircled_energy(
    obsmode="wfc3,ir,f160w", sp="default", aper_radii=EE_RADII, norm_radius=4.0
):
    """
    Compute encircled energy curves with pysynphot

    Parameters
    ----------
    obsmode : str
        The observation mode string specifying the
          instrument, detector, and filter.

    sp : `pysynphot.spectrum.SourceSpectrum` or None, optional
        The source spectrum to use for the calculation. 
        If None, a flat spectrum with a magnitude of 25 AB mag is used.

    aper_radii : array-like, optional
        The array of aperture radii in arcseconds at which to compute
        the encircled energy. 
        Default is [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0].

    norm_radius : float, optional
        The normalization radius in arcseconds. 
        The encircled energy at this radius will be used to normalize
        the encircled energy curve. If set to np.inf, the normalization 
        will be performed using the full aperture. Default is 4.0.

    Returns
    -------
    counts : array-like
        The array of encircled energy counts normalized to the counts
        at the normalization radius.

    """
    import pysynphot as S

    if sp == "default":
        sp = S.FlatSpectrum(25, fluxunits="ABMag")

    # Normalization
    if np.isfinite(norm_radius):
        bp = S.ObsBandpass(obsmode + ",aper#{0:.2f}".format(norm_radius))
    else:
        bp = S.ObsBandpass(obsmode)

    obs = S.Observation(sp, bp)
    norm_counts = obs.countrate()

    counts = np.ones_like(aper_radii)
    for i, r_aper in enumerate(aper_radii):
        # print(obsmode, r_aper)
        bp = S.ObsBandpass(obsmode + ",aper#{0:.2f}".format(r_aper))
        obs = S.Observation(sp, bp)
        counts[i] = obs.countrate()

    return counts / norm_counts


def photfnu_from_photflam(photflam, photplam):
    """
    Compute PHOTFNU from PHOTFLAM and PHOTPLAM, e.g., for ACS/WFC

    Parameters
    ----------
    photflam : float
        The PHOTFLAM value from the FITS header.

    photplam : float
        The PHOTPLAM value from the FITS header.

    Returns
    -------
    photfnu : float
        The computed PHOTFNU value.

    Examples
    --------
    >>> ZP = -2.5 * np.log10(photflam) - 21.10 - 5 * np.log10(photplam) + 18.6921
    >>> photfnu = 10 ** (-0.4 * (ZP - 23.9)) * 1.0e-6
    
    """
    ZP = -2.5 * np.log10(photflam) - 21.10 - 5 * np.log10(photplam) + 18.6921
    photfnu = 10 ** (-0.4 * (ZP - 23.9)) * 1.0e-6
    return photfnu


def calc_header_zeropoint(im, ext=0):
    """
    Determine AB zeropoint from image header

    Parameters
    ----------
    im : `~astropy.io.fits.HDUList` or
        Image object or header.

    ext : int, optional
        Extension number to use. Default is 0.

    Returns
    -------
    ZP : float
        AB zeropoint
    
    """
    from . import model

    scale_exptime = 1.0

    if isinstance(im, pyfits.Header):
        header = im
    else:
        if "_dr" in im.filename():
            ext = 0
        elif "_fl" in im.filename():
            if "DETECTOR" in im[0].header:
                if im[0].header["DETECTOR"] == "IR":
                    ext = 0
                    bunit = im[1].header["BUNIT"]
                else:
                    # ACS / UVIS
                    if ext == 0:
                        ext = 1

                    bunit = im[1].header["BUNIT"]

                if bunit == "ELECTRONS":
                    scale_exptime = im[0].header["EXPTIME"]

        header = im[ext].header

    try:
        fi = parse_filter_from_header(im[0].header).upper()
    except:
        fi = None

    # Get AB zeropoint
    if "APZP" in header:
        ZP = header["ABZP"]

    elif "PHOTFNU" in header:
        ZP = -2.5 * np.log10(header["PHOTFNU"]) + 8.90
        ZP += 2.5 * np.log10(scale_exptime)

    elif "PHOTFLAM" in header:
        ZP = (
            -2.5 * np.log10(header["PHOTFLAM"])
            - 21.10
            - 5 * np.log10(header["PHOTPLAM"])
            + 18.6921
        )

        ZP += 2.5 * np.log10(scale_exptime)

    elif fi is not None:
        if fi in model.photflam_list:
            ZP = (
                -2.5 * np.log10(model.photflam_list[fi])
                - 21.10
                - 5 * np.log10(model.photplam_list[fi])
                + 18.6921
            )
        else:
            print("Couldn't find PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25")
            ZP = 25
    else:
        print("Couldn't find FILTER, PHOTFNU or PHOTPLAM/PHOTFLAM keywords, use ZP=25")
        ZP = 25

    # If zeropoint infinite (e.g., PHOTFLAM = 0), then calculate from synphot
    if not np.isfinite(ZP):
        try:
            import pysynphot as S

            bp = S.ObsBandpass(im[0].header["PHOTMODE"].replace(" ", ","))
            spec = S.FlatSpectrum(0, fluxunits="ABMag")
            obs = S.Observation(spec, bp)
            ZP = 2.5 * np.log10(obs.countrate())
        except:
            pass

    return ZP


DEFAULT_PRIMARY_KEYS = [
    "FILENAME",
    "INSTRUME",
    "INSTRUME",
    "DETECTOR",
    "FILTER",
    "FILTER1",
    "FILTER2",
    "EXPSTART",
    "DATE-OBS",
    "EXPTIME",
    "IDCTAB",
    "NPOLFILE",
    "D2IMFILE",
    "PA_V3",
    "FGSLOCK",
    "GYROMODE",
    "PROPOSID",
]

# For grism
DEFAULT_EXT_KEYS = [
    "EXTNAME",
    "EXTVER",
    "MDRIZSKY",
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
    "PC1_1",
    "PC1_2",
    "PC2_1",
    "PC2_2",
    "CDELT1",
    "CDELT2",
    "CUNIT1",
    "CUNIT2",
    "CTYPE1",
    "CTYPE2",
    "RADESYS",
    "LONPOLE",
    "LATPOLE",
    "IDCTAB",
    "D2IMEXT",
    "WCSNAME",
    "PHOTMODE",
    "ORIENTAT",
    "CCDCHIP",
]


def flt_to_dict(
    fobj,
    primary_keys=DEFAULT_PRIMARY_KEYS,
    extensions=[("SCI", i + 1) for i in range(2)],
    ext_keys=DEFAULT_EXT_KEYS,
):
    """
    Parse basic elements from a FLT/FLC header to a dictionary

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
    flt_dict["timestamp"] = astropy.time.Time.now().iso
    h0 = fobj[0].header

    # Primary keywords
    for k in primary_keys:
        if k in h0:
            flt_dict[k] = h0[k]

    # Grism keys
    for k in h0:
        if k.startswith("GSKY"):
            flt_dict[k] = h0[k]

    # WCS, etc. keywords from SCI extensions
    flt_dict["extensions"] = OrderedDict()
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
                if k.startswith("GSKY"):
                    d_i[k] = h_i[k]

            count += 1
            flt_dict["extensions"][count] = d_i

    return flt_dict


def mod_dq_bits(value, okbits=32 + 64 + 512, badbits=0, verbose=False):
    """
    Modify bit flags from a DQ array

    For WFC3/IR, the following DQ bits can usually be unset:

    32, 64: these pixels usually seem OK
       512: blobs not relevant for grism exposures

    Parameters
    ----------
    value : int, `~numpy.ndarray`
        Input DQ value

    okbits : int
        Bits to unset

    badbits : int
        Bits to set

    verbose : bool
        Print some information

    Returns
    -------
    new_value : int, `~numpy.ndarray`

    """

    if verbose:
        print(f"Unset bits: {np.binary_repr(okbits)}")
        print(f"Set bits: {np.binary_repr(badbits)}")

    return (value & ~okbits) | badbits



def detect_with_photutils(sci, err=None, dq=None, seg=None, detect_thresh=2.,
                        npixels=8, grow_seg=5, gauss_fwhm=2., gsize=3,
                        wcs=None, save_detection=False, root='mycat',
                        background=None, gain=None, AB_zeropoint=0.,
                        rename_columns={'xcentroid': 'x_flt',
                                          'ycentroid': 'y_flt',
                                          'ra_icrs_centroid': 'ra',
                                          'dec_icrs_centroid': 'dec'},
                        overwrite=True, verbose=True):
    r"""
    Use `~photutils` to detect objects and make segmentation map

    .. note::
        Deprecated in favor of sep catalogs in `~grizli.prep`.

    Parameters
    ----------
    sci : `~numpy.ndarray`

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
    mask = sci == 0
    if dq is not None:
        mask |= dq > 0

    # Detection threshold
    if err is None:
        threshold = detect_threshold(sci, snr=detect_thresh, mask=mask)
    else:
        threshold = (detect_thresh * err) * (~mask)
        threshold[mask] = np.median(threshold[~mask])

    if seg is None:
        # Run the source detection and create the segmentation image

        # Gaussian kernel
        sigma = gauss_fwhm * gaussian_fwhm_to_sigma  # FWHM = 2.
        kernel = Gaussian2DKernel(sigma, x_size=gsize, y_size=gsize)
        kernel.normalize()

        if verbose:
            print(
                "{0}: photutils.detect_sources (detect_thresh={1:.1f}, grow_seg={2:d}, gauss_fwhm={3:.1f}, ZP={4:.1f})".format(
                    root, detect_thresh, grow_seg, gauss_fwhm, AB_zeropoint
                )
            )

        # Detect sources
        segm = detect_sources(
            sci * (~mask), threshold, npixels=npixels, filter_kernel=kernel
        )

        grow = nd.maximum_filter(segm.data, grow_seg)
        seg = np.asarray(grow,dtype=np.float32)
    else:
        # Use the supplied segmentation image
        segm = SegmentationImage(seg)

    # Source properties catalog
    if verbose:
        print("{0}: photutils.source_properties".format(root))

    props = source_properties(
        sci,
        segm,
        error=threshold / detect_thresh,
        mask=mask,
        background=background,
        wcs=wcs,
    )

    catalog = props.to_table()

    # Mag columns
    mag = AB_zeropoint - 2.5 * np.log10(catalog["source_sum"])
    mag._name = "mag"
    catalog.add_column(mag)

    try:
        logscale = 2.5 / np.log(10)
        mag_err = logscale * catalog["source_sum_err"] / catalog["source_sum"]
    except:
        mag_err = np.zeros_like(mag) - 99

    mag_err._name = "mag_err"
    catalog.add_column(mag_err)

    # Rename some catalog columns
    for key in rename_columns.keys():
        if key not in catalog.colnames:
            continue

        catalog.rename_column(key, rename_columns[key])
        if verbose:
            print("Rename column: {0} -> {1}".format(key, rename_columns[key]))

    # Done!
    if verbose:
        print(
            NO_NEWLINE
            + (
                "{0}: photutils.source_properties - {1:d} objects".format(
                    root, len(catalog)
                )
            )
        )

    # Save outputs?
    if save_detection:
        seg_file = root + ".detect_seg.fits"
        seg_cat = root + ".detect.cat"
        if verbose:
            print("{0}: save {1}, {2}".format(root, seg_file, seg_cat))

        if wcs is not None:
            header = wcs.to_header(relax=True)
        else:
            header = None

        pyfits.writeto(seg_file, data=seg, header=header, overwrite=overwrite)

        if os.path.exists(seg_cat) & overwrite:
            os.remove(seg_cat)

        catalog.write(seg_cat, format="ascii.commented_header")

    return catalog, seg


def safe_invert(arr):
    """
    Version-safe matrix inversion using `numpy.linalg` or `numpy.matrix.I`
    
    Parameters
    ----------
    arr : array_like
        The input array to be inverted.
        
    Returns
    -------
    _inv : ndarray
        The inverted array.
    
    """
    try:
        from numpy.linalg import inv
        _inv = inv(arr)
    except:
        _inv = np.matrix(arr).I.A

    return _inv


def nmad(data):
    """
    Normalized NMAD = 1.4826022 * `~.astropy.stats.median_absolute_deviation`

    Parameters
    ----------
    data: array-like
        The input data array.

    Returns
    -------
    nmad: float
        The normalized median absolute deviation of the input data.

    """
    import astropy.stats

    return 1.4826022 * astropy.stats.median_absolute_deviation(data)


def get_line_wavelengths():
    """
    Get a dictionary of common emission line wavelengths and line ratios

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

    # Rh = 0.0010967757
    # k = Rh * (1/n0**2 - 1/n**2)
    # wave = 1./k # Angstroms

    # Pfund n0=5
    line_wavelengths["PfA"] = [74598.8]
    line_ratios["PfA"] = [1.0]
    line_wavelengths["PfB"] = [46537.9]
    line_ratios["PfB"] = [1.0]
    line_wavelengths["PfG"] = [37405.7]
    line_ratios["PfG"] = [1.0]
    line_wavelengths["PfD"] = [32970.0]
    line_ratios["PfD"] = [1.0]
    line_wavelengths["PfE"] = [30392.1]
    line_ratios["PfE"] = [1.0]

    # Brackett n0=4
    line_wavelengths["BrA"] = [40522.8]
    line_ratios["BrA"] = [1.0]
    line_wavelengths["BrB"] = [26258.8]
    line_ratios["BrB"] = [1.0]
    line_wavelengths["BrG"] = [21661.3]
    line_ratios["BrG"] = [1.0]
    line_wavelengths["BrD"] = [19451.0]
    line_ratios["BrD"] = [1.0]
    line_wavelengths["BrE"] = [18179.2]
    line_ratios["BrE"] = [1.0]
    line_wavelengths["BrF"] = [17366.9]
    line_ratios["BrF"] = [1.0]

    # Paschen n0=3
    line_wavelengths["PaA"] = [18756.3]
    line_ratios["PaA"] = [1.0]
    line_wavelengths["PaB"] = [12821.7]
    line_ratios["PaB"] = [1.0]
    line_wavelengths["PaG"] = [10941.2]
    line_ratios["PaG"] = [1.0]
    line_wavelengths["PaD"] = [10052.2]
    line_ratios["PaD"] = [1.0]
    line_wavelengths["Pa8"] = [9548.65]
    line_ratios["Pa8"] = [1.0]
    line_wavelengths["Pa9"] = [9231.60]
    line_ratios["Pa9"] = [1.0]
    line_wavelengths["Pa10"] = [9017.44]
    line_ratios["Pa10"] = [1.0]

    # Balmer n0=2
    line_wavelengths["Ha"] = [6564.697]
    line_ratios["Ha"] = [1.0]
    line_wavelengths["Hb"] = [4862.738]
    line_ratios["Hb"] = [1.0]
    line_wavelengths["Hg"] = [4341.731]
    line_ratios["Hg"] = [1.0]
    line_wavelengths["Hd"] = [4102.936]
    line_ratios["Hd"] = [1.0]

    line_wavelengths["H7"] = [3971.236]
    line_ratios["H7"] = [1.0]
    line_wavelengths["H8"] = [3890.191]
    line_ratios["H8"] = [1.0]
    line_wavelengths["H9"] = [3836.511]
    line_ratios["H9"] = [1.0]
    line_wavelengths["H10"] = [3799.014]
    line_ratios["H10"] = [1.0]
    line_wavelengths["H11"] = [3771.739]
    line_ratios["H11"] = [1.0]
    line_wavelengths["H12"] = [3751.255]
    line_ratios["H12"] = [1.0]

    # Groves et al. 2011, Table 1
    # Osterbrock table 4.4 for H7 to H10
    # line_wavelengths['Balmer 10kK'] = [6564.61, 4862.68, 4341.68, 4101.73]
    # line_ratios['Balmer 10kK'] = [2.86, 1.0, 0.468, 0.259]

    line_wavelengths["Balmer 10kK"] = [
        6564.61,
        4862.68,
        4341.68,
        4101.73,
        3971.198,
        3890.166,
        3836.485,
        3798.987,
    ]
    line_ratios["Balmer 10kK"] = [2.86, 1.0, 0.468, 0.259, 0.159, 0.105, 0.0731, 0.0530]

    # Paschen from Osterbrock, e.g., Pa-beta relative to H-gamma
    line_wavelengths["Balmer 10kK"] += (
        line_wavelengths["PaA"]
        + line_wavelengths["PaB"]
        + line_wavelengths["PaG"]
        + line_wavelengths["PaD"]
    )
    line_ratios["Balmer 10kK"] += [
        0.348 * line_ratios["Balmer 10kK"][i] for i in [1, 2, 3, 4]
    ]

    # Osterbrock table 4.4 for H7 to H10
    line_wavelengths["Balmer 10kK + MgII"] = line_wavelengths["Balmer 10kK"] + [
        2799.117
    ]
    line_ratios["Balmer 10kK + MgII"] = line_ratios["Balmer 10kK"] + [3.0]

    # # Paschen from Osterbrock, e.g., Pa-beta relative to H-gamma
    # line_wavelengths['Balmer 10kK + MgII'] += line_wavelengths['PaA'] + line_wavelengths['PaB'] + line_wavelengths['PaG']
    # line_ratios['Balmer 10kK + MgII'] += [0.348 * line_ratios['Balmer 10kK + MgII'][i] for i in [1,2,3]]

    # With Paschen lines & He 10830 from Glikman 2006
    # https://iopscience.iop.org/article/10.1086/500098/pdf
    # line_wavelengths['Balmer 10kK + MgII'] = [6564.61, 4862.68, 4341.68, 4101.73, 3971.198, 2799.117, 12821.6, 10941.1]
    # line_ratios['Balmer 10kK + MgII'] = [2.86, 1.0, 0.468, 0.259, 0.16, 3., 2.86*4.8/100, 2.86*1.95/100]

    # Redden with Calzetti00
    if False:
        from extinction import calzetti00

        Av = 1.0
        Rv = 3.1

        waves = line_wavelengths["Balmer 10kK + MgII"]
        ratios = line_ratios["Balmer 10kK + MgII"]

        for Av in [0.5, 1.0, 2.0]:
            mred = calzetti00(np.array(waves), Av, Rv)
            fred = 10 ** (-0.4 * mred)

            key = "Balmer 10kK + MgII Av={0:.1f}".format(Av)
            line_wavelengths[key] = [w for w in waves]
            line_ratios[key] = [ratios[i] * fred[i] for i in range(len(waves))]

    line_wavelengths["Balmer 10kK + MgII Av=0.5"] = [
        6564.61,
        4862.68,
        4341.68,
        4101.73,
        3971.198,
        2799.117,
        12821.6,
        10941.1,
    ]
    line_ratios["Balmer 10kK + MgII Av=0.5"] = [
        2.009811938798515,
        0.5817566641521459,
        0.25176970824566913,
        0.1338409369665902,
        0.08079209880749984,
        1.1739297839690317,
        0.13092553990513178,
        0.05033866127477651,
    ]

    line_wavelengths["Balmer 10kK + MgII Av=1.0"] = [
        6564.61,
        4862.68,
        4341.68,
        4101.73,
        3971.198,
        2799.117,
        12821.6,
        10941.1,
    ]
    line_ratios["Balmer 10kK + MgII Av=1.0"] = [
        1.4123580522157504,
        0.33844081628543266,
        0.13544441450878067,
        0.0691636926953466,
        0.04079602018575511,
        0.4593703792298591,
        0.12486521707058751,
        0.045436270735820045,
    ]

    line_wavelengths["Balmer 10kK + MgII Av=2.0"] = [
        6564.61,
        4862.68,
        4341.68,
        4101.73,
        3971.198,
        2799.117,
        12821.6,
        10941.1,
    ]
    line_ratios["Balmer 10kK + MgII Av=2.0"] = [
        0.6974668768037302,
        0.11454218612794999,
        0.03919912269578289,
        0.018469561340758073,
        0.010401970393728362,
        0.0703403817712615,
        0.11357315292894044,
        0.03701729780130422,
    ]

    ###########

    # Reddened with Kriek & Conroy dust, tau_V=0.5
    line_wavelengths["Balmer 10kK t0.5"] = [6564.61, 4862.68, 4341.68, 4101.73]
    line_ratios["Balmer 10kK t0.5"] = [
        2.86 * 0.68,
        1.0 * 0.55,
        0.468 * 0.51,
        0.259 * 0.48,
    ]

    # Reddened with Kriek & Conroy dust, tau_V=1
    line_wavelengths["Balmer 10kK t1"] = [6564.61, 4862.68, 4341.68, 4101.73]
    line_ratios["Balmer 10kK t1"] = [
        2.86 * 0.46,
        1.0 * 0.31,
        0.468 * 0.256,
        0.259 * 0.232,
    ]

    line_wavelengths["OIII-4363"] = [4364.436]
    line_ratios["OIII-4363"] = [1.0]
    line_wavelengths["OIII"] = [5008.240, 4960.295]
    line_ratios["OIII"] = [2.98, 1]

    # Split doublet, if needed
    line_wavelengths["OIII-4959"] = [4960.295]
    line_ratios["OIII-4959"] = [1]
    line_wavelengths["OIII-5007"] = [5008.240]
    line_ratios["OIII-5007"] = [1]

    line_wavelengths["OII"] = [3727.092, 3729.875]
    line_ratios["OII"] = [1, 1.0]

    line_wavelengths["OI-5578"] = [5578.89]
    line_ratios["OI-5578"] = [1]
    line_wavelengths["OI-6302"] = [6302.046, 6365.535]
    line_ratios["OI-6302"] = [1, 0.33]
    line_wavelengths["OI-7776"] = [7776.3]
    line_ratios["OI-7776"] = [1]
    line_wavelengths["OI-8448"] = [8448.7]
    line_ratios["OI-8448"] = [1]
    line_wavelengths["OI-11290"] = [11290.0]
    line_ratios["OI-11290"] = [1]

    # Auroral OII
    # lines roughly taken from https://arxiv.org/pdf/1610.06939.pdf
    line_wavelengths["OII-7325"] = [7321.9, 7332.21]
    line_ratios["OII-7325"] = [1.2, 1.0]

    line_wavelengths["OII-7323"] = [7321.9]
    line_ratios["OII-7323"] = [1.0]
    line_wavelengths["OII-7332"] = [7332.21]
    line_ratios["OII-7332"] = [1.0]

    # Weak Ar III in SF galaxies
    line_wavelengths["ArIII-7138"] = [7137.77]
    line_ratios["ArIII-7138"] = [1.0]
    line_wavelengths["ArIII-7753"] = [7753.19]
    line_ratios["ArIII-7753"] = [1.0]

    line_wavelengths["NeIII-3867"] = [3869.87]
    line_ratios["NeIII-3867"] = [1.0]
    line_wavelengths["NeIII-3968"] = [3968.59]
    line_ratios["NeIII-3968"] = [1.0]
    line_wavelengths["NeV-3346"] = [3343.5]
    line_ratios["NeV-3346"] = [1.0]
    line_wavelengths["NeVI-3426"] = [3426.85]
    line_ratios["NeVI-3426"] = [1.0]

    line_wavelengths["SIII"] = [9071.1, 9533.2][::-1]
    line_ratios["SIII"] = [1, 2.44][::-1]

    # Split doublet, if needed
    line_wavelengths["SIII-9068"] = [9071.1]
    line_ratios["SIII-9068"] = [1]
    line_wavelengths["SIII-9531"] = [9533.2]
    line_ratios["SIII-9531"] = [1]

    line_wavelengths["SIII-6314"] = [6313.81]
    line_ratios["SIII-6314"] = [1.0]

    line_wavelengths["SII"] = [6718.29, 6732.67]
    line_ratios["SII"] = [1.0, 1.0]

    line_wavelengths["SII-6717"] = [6718.29]
    line_ratios["SII-6717"] = [1.0]
    line_wavelengths["SII-6731"] = [6732.67]
    line_ratios["SII-6731"] = [1.0]

    line_wavelengths["SII-4075"] = [4069.75, 4077.5]
    line_ratios["SII-4075"] = [1.0, 1.0]
    line_wavelengths["SII-4070"] = [4069.75]
    line_ratios["SII-4075"] = [1.0]
    line_wavelengths["SII-4078"] = [4077.5]
    line_ratios["SII-4078"] = [1.0]

    line_wavelengths["HeII-4687"] = [4687.5]
    line_ratios["HeII-4687"] = [1.0]
    line_wavelengths["HeII-5412"] = [5412.5]
    line_ratios["HeII-5412"] = [1.0]
    line_wavelengths["HeII-16923"] = [1.69230e4]
    line_ratios["HeII-16923"] = [1.0]
    
    line_wavelengths["HeI-5877"] = [5877.249]
    line_ratios["HeI-5877"] = [1.0]
    line_wavelengths["HeI-3889"] = [3889.75]
    line_ratios["HeI-3889"] = [1.0]
    line_wavelengths["HeI-1083"] = [10832.057, 10833.306]
    line_ratios["HeI-1083"] = [1.0, 1.0]
    line_wavelengths["HeI-3820"] = [3820.7]
    line_ratios["HeI-3820"] = [1.0]
    line_wavelengths["HeI-4027"] = [4027.3]
    line_ratios["HeI-4027"] = [1.0]
    line_wavelengths["HeI-4472"] = [4472.7]
    line_ratios["HeI-4472"] = [1.0]
    line_wavelengths["HeI-6680"] = [6679.995]
    line_ratios["HeI-6680"] = [1.0]
    line_wavelengths["HeI-7065"] = [7067.1]
    line_ratios["HeI-7065"] = [1.0]
    line_wavelengths["HeI-8446"] = [8446.7]
    line_ratios["HeI-8446"] = [1.0]
    
    # From CAFE
    # https://github.com/GOALS-survey/CAFE/blob/master/CAFE/tables/
    
    line_wavelengths["FeII-11128"] = [1.11286e4]
    line_ratios["FeII-11128"] = [1.0]
    line_wavelengths["FeII-12570"] = [1.25702e4]
    line_ratios["FeII-12570"] = [1.0]
    line_wavelengths["FeII-16440"] = [1.64400e4]
    line_ratios["FeII-16440"] = [1.0]
    line_wavelengths["FeII-16877"] = [1.68778e4]
    line_ratios["FeII-16877"] = [1.0]
    line_wavelengths["FeII-17418"] = [1.74188e4]
    line_ratios["FeII-17418"] = [1.0]
    line_wavelengths["FeII-17418"] = [1.74188e4]
    line_ratios["FeII-17418"] = [1.0]
    line_wavelengths["FeII-18362"] = [1.83624e4]
    line_ratios["FeII-18362"] = [1.0]
    
    line_wavelengths["SiVI-19634"] = [1.9634e4]
    line_ratios["SiVI-19634"] = [1.0]
    
    # Ca triplet
    # Wavelengths from https://classic.sdss.org/dr5/algorithms/linestable.php
    # ratio https://iopscience.iop.org/article/10.3847/1538-4365/ad33bc#apjsad33bcf1
    line_wavelengths["CaII-8600"] = [8500.36, 8544.44, 8664.52]
    line_ratios["CaII-8600"] = [0.2, 0.35, 0.3]

    # AGN line?
    # https://academic.oup.com/pasj/article/63/1/L7/1460068#431992120
    line_wavelengths["PII-11886"] = [1.188610e4]
    line_ratios["PII-11886"] = [1.0]
    
    # Osterbrock Table 4.5
    # -> N=4
    line_wavelengths["HeI-series"] = [
        4472.7,
        5877.2,
        4027.3,
        3820.7,
        7067.1,
        10833.2,
        3889.7,
        3188.7,
    ]
    line_ratios["HeI-series"] = [1.0, 2.75, 0.474, 0.264, 0.330, 4.42, 2.26, 0.916]

    # line_wavelengths["MgII"] = [2799.117]
    # line_ratios["MgII"] = [1.0]
    line_wavelengths["MgII"] = [2796.352, 2803.531]
    line_ratios["MgII"] = [1.0, 1.0]

    line_wavelengths["CIV-1549"] = [1549.480]
    line_ratios["CIV-1549"] = [1.0]
    line_wavelengths["CIII-1906"] = [1906.683]
    line_ratios["CIII-1906"] = [1.0]
    line_wavelengths["CIII-1908"] = [1908.734]
    line_ratios["CIII-1908"] = [1.0]
    line_wavelengths["CI-9580"] = [9852.96]  # leave typo for back compatibility
    line_ratios["CI-9580"] = [1.0]
    line_wavelengths["CI-9850"] = [9852.96]
    line_ratios["CI-9850"] = [1.0]

    # Sodium D I lines from Davies 2023
    # https://arxiv.org/abs/2310.17939v2
    line_wavelengths["NaDI"] = [5891.0, 5897.0]
    line_ratios["NaDI"] = [1.0, 1.0]

    # Hutchinson
    # https://iopscience.iop.org/article/10.3847/1538-4357/ab22a2
    line_wavelengths["CIII-1906x"] = [1906.683, 1908.734]
    line_ratios["CIII-1906x"] = [1.5, 1.0]

    line_wavelengths["OIII-1663"] = [1665.85]
    line_ratios["OIII-1663"] = [1.0]
    line_wavelengths["HeII-1640"] = [1640.4]
    line_ratios["HeII-1640"] = [1.0]

    line_wavelengths["SiIV+OIV-1398"] = [1398.0]
    line_ratios["SiIV+OIV-1398"] = [1.0]

    # Weak line in LEGA-C spectra
    line_wavelengths["NI-5199"] = [5199.4, 5201.76]
    line_ratios["NI-5199"] = [1.0, 1.0]

    line_wavelengths["NII"] = [6549.86, 6585.27][::-1]
    line_ratios["NII"] = [1.0, 3.0][::-1]

    line_wavelengths["NII-6549"] = [6549.86]
    line_ratios["NII-6549"] = [1.0]
    line_wavelengths["NII-6584"] = [6585.27]
    line_ratios["NII-6584"] = [1.0]

    line_wavelengths["NIII-1750"] = [1750.0]
    line_ratios["NIII-1750"] = [1.0]
    line_wavelengths["NIV-1487"] = [1487.0]
    line_ratios["NIV-1487"] = [1.0]
    line_wavelengths["NV-1240"] = [1240.81]
    line_ratios["NV-1240"] = [1.0]

    line_wavelengths["Lya"] = [1215.4]
    line_ratios["Lya"] = [1.0]

    line_wavelengths["QSO-UV-lines"] = [
        line_wavelengths[k][0]
        for k in [
            "Lya",
            "CIV-1549",
            "CIII-1906",
            "CIII-1908",
            "OIII-1663",
            "HeII-1640",
            "SiIV+OIV-1398",
            "NV-1240",
            "NIII-1750",
        ]
    ]
    line_ratios["QSO-UV-lines"] = [1.0, 0.5, 0.1, 0.1, 0.008, 0.09, 0.1, 0.3, 0.05]

    line_wavelengths["QSO-Narrow-lines"] = [
        line_wavelengths[k][0]
        for k in [
            "OII",
            "OIII-5007",
            "OIII-4959",
            "SII-6717",
            "SII-6731",
            "OI-6302",
            "NeIII-3867",
            "NeVI-3426",
            "NeV-3346",
        ]
    ]
    line_ratios["QSO-Narrow-lines"] = [
        0.2,
        1.6,
        1.6 / 2.98,
        0.1,
        0.1,
        0.01,
        0.5,
        0.2,
        0.02,
    ]

    # redder lines
    line_wavelengths["QSO-Narrow-lines"] += line_wavelengths["SIII"]
    line_ratios["QSO-Narrow-lines"] += [lr * 0.05 for lr in line_ratios["SIII"]]
    line_wavelengths["QSO-Narrow-lines"] += line_wavelengths["HeI-1083"]
    line_ratios["QSO-Narrow-lines"] += [0.2]

    line_wavelengths["Lya+CIV"] = [1215.4, 1549.49]
    line_ratios["Lya+CIV"] = [1.0, 0.1]

    line_wavelengths["Gal-UV-lines"] = [
        line_wavelengths[k][0]
        for k in [
            "Lya",
            "CIV-1549",
            "CIII-1906",
            "CIII-1908",
            "OIII-1663",
            "HeII-1640",
            "SiIV+OIV-1398",
            "NV-1240",
            "NIII-1750",
            "MgII",
        ]
    ]
    line_ratios["Gal-UV-lines"] = [
        1.0,
        0.15,
        0.1,
        0.1,
        0.008,
        0.09,
        0.1,
        0.05,
        0.05,
        0.1,
    ]

    line_wavelengths["Ha+SII"] = [6564.61, 6718.29, 6732.67]
    line_ratios["Ha+SII"] = [1.0, 1.0 / 10, 1.0 / 10]

    line_wavelengths["Ha+SII+SIII+He"] = [
        6564.61,
        6718.29,
        6732.67,
        9068.6,
        9530.6,
        10830.0,
    ]
    line_ratios["Ha+SII+SIII+He"] = [
        1.0,
        1.0 / 10,
        1.0 / 10,
        1.0 / 20,
        2.44 / 20,
        1.0 / 25.0,
    ]

    line_wavelengths["Ha+NII+SII+SIII+He"] = [
        6564.61,
        6549.86,
        6585.27,
        6718.29,
        6732.67,
        9068.6,
        9530.6,
        10830.0,
    ]
    line_ratios["Ha+NII+SII+SIII+He"] = [
        1.0,
        1.0 / (4.0 * 4),
        3.0 / (4 * 4),
        1.0 / 10,
        1.0 / 10,
        1.0 / 20,
        2.44 / 20,
        1.0 / 25.0,
    ]

    line_wavelengths["Ha+NII+SII+SIII+He+PaB"] = [
        6564.61,
        6549.86,
        6585.27,
        6718.29,
        6732.67,
        9068.6,
        9530.6,
        10830.0,
        12821,
    ]
    line_ratios["Ha+NII+SII+SIII+He+PaB"] = [
        1.0,
        1.0 / (4.0 * 4),
        3.0 / (4 * 4),
        1.0 / 10,
        1.0 / 10,
        1.0 / 20,
        2.44 / 20,
        1.0 / 25.0,
        1.0 / 10,
    ]

    line_wavelengths["Ha+NII+SII+SIII+He+PaB+PaG"] = [
        6564.61,
        6549.86,
        6585.27,
        6718.29,
        6732.67,
        9068.6,
        9530.6,
        10830.0,
        12821,
        10941.1,
    ]
    line_ratios["Ha+NII+SII+SIII+He+PaB+PaG"] = [
        1.0,
        1.0 / (4.0 * 4),
        3.0 / (4 * 4),
        1.0 / 10,
        1.0 / 10,
        1.0 / 20,
        2.44 / 20,
        1.0 / 25.0,
        1.0 / 10,
        1.0 / 10 / 2.86,
    ]

    line_wavelengths["Ha+NII"] = [6564.61, 6549.86, 6585.27]
    n2ha = 1.0 / 3  # log NII/Ha ~ -0.6, Kewley 2013
    line_ratios["Ha+NII"] = [1.0, 1.0 / 4.0 * n2ha, 3.0 / 4.0 * n2ha]

    line_wavelengths["OIII+Hb"] = [5008.240, 4960.295, 4862.68]
    line_ratios["OIII+Hb"] = [2.98, 1, 3.98 / 6.0]

    # Include more balmer lines
    line_wavelengths["OIII+Hb+Hg+Hd"] = (
        line_wavelengths["OIII"] + line_wavelengths["Balmer 10kK"][1:]
    )
    line_ratios["OIII+Hb+Hg+Hd"] = line_ratios["OIII"] + line_ratios["Balmer 10kK"][1:]
    # o3hb = 1./6
    # for i in range(2, len(line_ratios['Balmer 10kK'])-1):
    #         line_ratios['OIII+Hb+Hg+Hd'][i] *= 3.98*o3hb
    # Compute as O3/Hb
    o3hb = 6
    for i in range(2):
        line_ratios["OIII+Hb+Hg+Hd"][i] *= 1.0 / 3.98 * o3hb

    line_wavelengths["OIII+Hb+Ha"] = [5008.240, 4960.295, 4862.68, 6564.61]
    line_ratios["OIII+Hb+Ha"] = [2.98, 1, 3.98 / 10.0, 3.98 / 10.0 * 2.86]

    line_wavelengths["OIII+Hb+Ha+SII"] = [
        5008.240,
        4960.295,
        4862.68,
        6564.61,
        6718.29,
        6732.67,
    ]
    line_ratios["OIII+Hb+Ha+SII"] = [
        2.98,
        1,
        3.98 / 10.0,
        3.98 / 10.0 * 2.86 * 4,
        3.98 / 10.0 * 2.86 / 10.0 * 4,
        3.98 / 10.0 * 2.86 / 10.0 * 4,
    ]

    line_wavelengths["OIII+OII"] = [5008.240, 4960.295, 3729.875]
    line_ratios["OIII+OII"] = [2.98, 1, 3.98 / 4.0]

    line_wavelengths["OII+Ne"] = [3729.875, 3869]
    line_ratios["OII+Ne"] = [1, 1.0 / 5]

    # Groups of all lines
    line_wavelengths["full"] = [w for w in line_wavelengths["Balmer 10kK"]]
    line_ratios["full"] = [w for w in line_ratios["Balmer 10kK"]]

    line_wavelengths["full"] += line_wavelengths["NII"]
    line_ratios["full"] += [
        1.0 / 5 / 3.0 * line_ratios["Balmer 10kK"][1] * r for r in line_ratios["NII"]
    ]

    line_wavelengths["full"] += line_wavelengths["SII"]
    line_ratios["full"] += [
        1.0 / 3.8 / 2 * line_ratios["Balmer 10kK"][1] * r for r in line_ratios["SII"]
    ]

    # Lines from Hagele 2006, low-Z HII galaxies
    # SDSS J002101.03+005248.1
    line_wavelengths["full"] += line_wavelengths["SIII"]
    line_ratios["full"] += [
        401.0 / 1000 / 2.44 * line_ratios["Balmer 10kK"][1] * r
        for r in line_ratios["SIII"]
    ]

    # HeI
    line_wavelengths["full"] += line_wavelengths["HeI-series"]
    he5877_hb = 127.0 / 1000 / line_ratios["HeI-series"][1]
    line_ratios["full"] += [he5877_hb * r for r in line_ratios["HeI-series"]]

    # NeIII
    line_wavelengths["full"] += line_wavelengths["NeIII-3867"]
    line_ratios["full"] += [388.0 / 1000 for r in line_ratios["NeIII-3867"]]

    line_wavelengths["full"] += line_wavelengths["NeIII-3968"]
    line_ratios["full"] += [290.0 / 1000 for r in line_ratios["NeIII-3968"]]

    # Add UV lines: MgII/Hb = 3
    line_wavelengths["full"] += line_wavelengths["Gal-UV-lines"]
    line_ratios["full"] += [
        r * 3 / line_ratios["Gal-UV-lines"][-1] for r in line_ratios["Gal-UV-lines"]
    ]

    # High O32 - low metallicity
    o32, r23 = 4, 8
    o3_hb = r23 / (1 + 1 / o32)

    line_wavelengths["highO32"] = [w for w in line_wavelengths["full"]]
    line_ratios["highO32"] = [r for r in line_ratios["full"]]

    line_wavelengths["highO32"] += line_wavelengths["OIII"]
    line_ratios["highO32"] += [r * o3_hb / 3.98 for r in line_ratios["OIII"]]

    line_wavelengths["highO32"] += line_wavelengths["OII"]
    line_ratios["highO32"] += [r * o3_hb / 2 / o32 for r in line_ratios["OII"]]

    # Low O32 - low metallicity
    o32, r23 = 0.3, 4
    o3_hb = r23 / (1 + 1 / o32)

    line_wavelengths["lowO32"] = [w for w in line_wavelengths["full"]]
    line_ratios["lowO32"] = [r for r in line_ratios["full"]]

    line_wavelengths["lowO32"] += line_wavelengths["OIII"]
    line_ratios["lowO32"] += [r * o3_hb / 3.98 for r in line_ratios["OIII"]]

    line_wavelengths["lowO32"] += line_wavelengths["OII"]
    line_ratios["lowO32"] += [r * o3_hb / 2 / o32 for r in line_ratios["OII"]]

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

    sp_params["starburst"] = {
        "sfh": 4,
        "tau": 0.3,
        "tage": 0.1,
        "logzsol": -1,
        "gas_logz": -1,
        "gas_logu": -2.5,
    }

    sp_params["mature"] = {
        "sfh": 4,
        "tau": 0.2,
        "tage": 0.9,
        "logzsol": -0.2,
        "gas_logz": -0.2,
        "gas_logu": -2.5,
    }

    line_templates = {}

    for t in sp_params:
        pset = sp_params[t]
        header = "wave flux\n\n"
        for p in pset:
            header += "{0} = {1}\n".format(p, pset[p])
            if p == "tage":
                continue

            print(p, pset[p])
            sp.params[p] = pset[p]

        spec = {}
        for neb in [True, False]:
            sp.params["add_neb_emission"] = neb
            sp.params["add_neb_continuum"] = neb
            wave, spec[neb] = sp.get_spectrum(tage=pset["tage"], peraa=True)
            # plt.plot(wave, spec[neb], alpha=0.5)

        neb_only = spec[True] - spec[False]
        neb_only = neb_only / neb_only.max()
        neb_only = spec[True] / spec[True].max()

        plt.plot(wave, neb_only, label=t, alpha=0.5)

        neb_only[neb_only < 1.0e-4] = 0

        np.savetxt(
            "fsps_{0}_lines.txt".format(t),
            np.array([wave, neb_only]).T,
            fmt="%.5e",
            header=header,
        )

        line_templates[t] = utils.SpectrumTemplate(
            wave=wave, flux=neb_only, name="fsps_{0}_lines".format(t)
        )


def pah33(wave_grid):
    """
    Set of 3.3 micron PAH lines from Li et al. 2020

    Parameters
    ----------
    wave_grid : array-like
        Wavelength grid for the templates.

    Returns
    -------
    pah_templates : list
        List of `~grizli.utils.SpectrumTemplate` templates for three components
        around 3.3 microns

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
    br = 1.0
    gamma_width = fwhm / center_um
    Iv = br * gamma_width ** 2
    Iv /= (
        wave_grid / 1.0e4 / center_um - center_um * 1.0e4 / wave_grid
    ) ** 2 + gamma_width ** 2

    Inorm = np.pi * 2.99e14 / 2.0 * br * gamma_width / center_um
    Iv *= 1 / Inorm

    # Flambda
    Ilam = Iv * 2.99e18 / (wave_grid) ** 2

    pah_templ = SpectrumTemplate(
        wave=wave_grid, flux=Ilam, name=f"line PAH-{center_um:.2f}"
    )
    return pah_templ


class SpectrumTemplate(object):

    def __init__(
        self,
        wave=None,
        flux=None,
        central_wave=None,
        fwhm=None,
        velocity=False,
        fluxunits=FLAMBDA_CGS,
        waveunits=u.angstrom,
        name="template",
        lorentz=False,
        err=None,
    ):
        r"""
        Container for template spectra.


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
            ``fwhm`` is a velocity in `km/s`.

        fluxunits : astropy.units.Unit
            Units of the flux. 
            Default is `FLAMBDA_CGS` (1e-17 erg/s/cm^2/Angstrom).

        waveunits : astropy.units.Unit
            Units of the wavelength. Default is Angstrom.

        name : str
            Name of the template. Default is "template".

        lorentz : bool
            Make a Lorentzian line instead of a Gaussian.

        err : float array-like, optional
            Error on the flux.

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
            self.wave = np.asarray(wave,dtype=np.float64)

        self.flux = flux
        if flux is not None:
            self.flux = np.asarray(flux,dtype=np.float64)

        if err is not None:
            self.err = np.asarray(err,dtype=np.float64)
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

            self.wave, self.flux = self.make_gaussian(
                central_wave,
                fwhm,
                wave_grid=wave,
                velocity=velocity,
                max_sigma=50,
                lorentz=lorentz,
            )

        self.fnu_units = FNU_CGS
        self.to_fnu()

    @staticmethod
    def make_gaussian(
        central_wave,
        fwhm,
        max_sigma=5,
        step=0.1,
        wave_grid=None,
        velocity=False,
        clip=1.0e-6,
        lorentz=False,
    ):
        """
        Make Gaussian template

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

        if hasattr(fwhm, "unit"):
            rms = fwhm.value / 2.35
            velocity = u.physical.get_physical_type(fwhm.unit) == "speed"
            if velocity:
                rms = central_wave * (fwhm / const.c.to(KMS)).value / 2.35
            else:
                rms = fwhm.value / 2.35
        else:
            if velocity:
                rms = central_wave * (fwhm / const.c.to(KMS).value) / 2.35
            else:
                rms = fwhm / 2.35

        if wave_grid is None:
            # print('xxx line', central_wave, max_sigma, rms)

            wave_grid = np.arange(-max_sigma, max_sigma, step) * rms
            wave_grid += central_wave
            wave_grid = np.hstack([91.0, wave_grid, 1.0e8])

        if lorentz:
            if velocity:
                use_fwhm = central_wave * (fwhm / const.c.to(KMS).value)
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
            line = np.exp(-((wave_grid - central_wave) ** 2) / 2 / rms ** 2)
            peak = np.sqrt(2 * np.pi * rms ** 2)
            line *= 1.0 / peak  # np.sqrt(2*np.pi*rms**2)

        line[line < 1.0 / peak * clip] = 0

        return wave_grid, line

        # self.wave = xgauss
        # self.flux = gaussian

    def zscale(self, z, scalar=1, apply_igm=True):
        """
        Redshift the template and multiply by a scalar.

        Parameters
        ----------
        z : float
            Redshift to use.

        scalar : float
            Multiplicative factor.  Additional factor of 1/(1+z) is implicit.

        apply_igm : bool
            Apply the intergalactic medium (IGM) attenuation correction.

        Returns
        -------
        new_spectrum : `~grizli.utils.SpectrumTemplate`
            Redshifted and scaled spectrum.

        """
        if apply_igm:
            try:
                import eazy.igm

                igm = eazy.igm.Inoue14()
                igmz = igm.full_IGM(z, self.wave * (1 + z))
            except:
                igmz = 1.0
        else:
            igmz = 1.0

        return SpectrumTemplate(
            wave=self.wave * (1 + z), flux=self.flux * scalar / (1 + z) * igmz
        )

    def __add__(self, spectrum):
        """
        Add two templates together

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
        """
        Multiply spectrum by a scalar value

        Parameters
        ----------
        scalar : float
            Factor to multipy to `self.flux`.

        Returns
        -------
        new_spectrum : `~grizli.utils.SpectrumTemplate`
        """
        out = SpectrumTemplate(wave=self.wave, flux=self.flux * scalar)
        out.fwhm = self.fwhm
        return out

    def to_fnu(self, fnu_units=FNU_CGS):
        """
        Make fnu version of the template.

        Sets the `flux_fnu` attribute, assuming that the wavelength is given
        in Angstrom and the flux is given in flambda:

            >>> flux_fnu = self.flux * self.wave**2 / 3.e18

        """
        # import astropy.constants as const
        # flux_fnu = self.flux * self.wave**2 / 3.e18
        # flux_fnu = (self.flux*self.fluxunits*(self.wave*self.waveunits)**2/const.c).to(FNU_CGS) #,

        if (FNU_CGS.__str__() == "erg / (cm2 Hz s)") & (
            self.fluxunits.__str__() == "erg / (Angstrom cm2 s)"
        ):
            # Faster
            flux_fnu = self.flux * self.wave ** 2 / 2.99792458e18 * fnu_units
            if self.err is not None:
                err_fnu = self.err * self.wave ** 2 / 2.99792458e18 * fnu_units
        else:
            # Use astropy conversion
            flux_fnu = (self.flux * self.fluxunits).to(
                fnu_units, equivalencies=u.spectral_density(self.wave * self.waveunits)
            )
            if self.err is not None:
                err_fnu = (self.err * self.fluxunits).to(
                    fnu_units,
                    equivalencies=u.spectral_density(self.wave * self.waveunits),
                )

        self.fnu_units = fnu_units
        self.flux_fnu = flux_fnu.value
        if self.err is not None:
            self.err_fnu = err_fnu.value
        else:
            self.err_fnu = None

    def integrate_filter(self, filter, abmag=False, use_wave="filter"):
        """
        Integrate the template through an `~eazy.FilterDefinition` filter
        object.

        Parameters
        ----------
        filter : `~pysynphot.ObsBandpass`
            Or any object that has `wave` and `throughput` attributes, with
            the former in the same units as the input spectrum.

        abmag : bool
            Return AB magnitude rather than fnu flux

        use_wave : str, optional
            Determines whether to interpolate the template to the filter 
            wavelengths or the spectrum wavelengths. Default is 'filter'.

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
            from .utils_numba.interp import interp_conserve_c
            interp = interp_conserve_c
        except ImportError:
            interp = np.interp

        # wz = self.wave*(1+z)
        nonzero = filter.throughput > 0
        if (
            (filter.wave[nonzero].min() > self.wave.max())
            | (filter.wave[nonzero].max() < self.wave.min())
            | (filter.wave[nonzero].min() < self.wave.min())
        ):
            if self.err is None:
                return 0.0
            else:
                return 0.0, 0.0

        if use_wave == "filter":
            # Interpolate to filter wavelengths
            integrate_wave = filter.wave

            integrate_templ = interp(
                filter.wave.astype(np.float64),
                self.wave,
                self.flux_fnu,
                left=0,
                right=0,
            )

            if self.err is not None:
                templ_ivar = (
                    1.0
                    / interp(filter.wave.astype(np.float64), self.wave, self.err_fnu)
                    ** 2
                )

                templ_ivar[~np.isfinite(templ_ivar)] = 0

                integrate_weight = (
                    filter.throughput / filter.wave * templ_ivar / filter.norm
                )
            else:
                integrate_weight = filter.throughput / filter.wave
        else:
            # Interpolate to spectrum wavelengths
            integrate_wave = self.wave
            integrate_templ = self.flux_fnu

            # test = nonzero
            test = np.isfinite(filter.throughput)
            interp_thru = interp(
                integrate_wave,
                filter.wave[test],
                filter.throughput[test],
                left=0,
                right=0,
            )

            if self.err is not None:
                templ_ivar = 1 / self.err_fnu ** 2
                templ_ivar[~np.isfinite(templ_ivar)] = 0

                integrate_weight = (
                    interp_thru / integrate_wave * templ_ivar / filter.norm
                )
            else:
                integrate_weight = interp_thru / integrate_wave  # /templ_err**2

        if hasattr(filter, "norm") & (self.err is None):
            filter_norm = filter.norm
        else:
            # e.g., pysynphot bandpass
            filter_norm = INTEGRATOR(integrate_weight, integrate_wave)

        # f_nu/lam dlam == f_nu d (ln nu)
        temp_flux = (
            INTEGRATOR(integrate_templ * integrate_weight, integrate_wave) / filter_norm
        )

        if self.err is not None:
            temp_err = 1 / np.sqrt(filter_norm)

        if abmag:
            temp_mag = -2.5 * np.log10(temp_flux) - 48.6
            return temp_mag
        else:
            if self.err is not None:
                return temp_flux, temp_err
            else:
                return temp_flux

    @property
    def eazy(self):
        """
        Convert to `eazy.template.Template` object
        """
        import eazy.templates

        templ = eazy.templates.Template(arrays=(self.wave, self.flux), name=self.name)
        return templ


def load_templates(
    fwhm=400,
    line_complexes=True,
    stars=False,
    full_line_list=DEFAULT_LINE_LIST,
    continuum_list=None,
    fsps_templates=False,
    alf_template=False,
    lorentz=False,
):
    """
    Generate a list of templates for fitting to the grism spectra

    The different sets of continuum templates are stored in

        >>> temp_dir = os.path.join(GRIZLI_PATH, 'templates')

    Parameters
    ----------
    fwhm : float, optional
        FWHM of a Gaussian, in km/s, that is convolved with the emission
        line templates.  If too narrow, then can see pixel effects in the
        fits as a function of redshift. Default is 400.

    line_complexes : bool, optional
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

    stars : bool, optional
        Get stellar templates rather than galaxies + lines. Default is False.

    full_line_list : None or list, optional
        Full set of lines to try.  The default is set in the global variable
        `~grizli.utils.DEFAULT_LINE_LIST`.

        The full list of implemented lines is in `~grizli.utils.get_line_wavelengths`.

    continuum_list : None or list, optional
        Override the default continuum templates if None.

    fsps_templates : bool, optional
        If True, get the FSPS NMF templates. Default is False.

    alf_template : bool, optional
        If True, include Alf templates. Default is False.

    lorentz : bool, optional
        If True, use Lorentzian line profiles instead of Gaussian. 
        Default is False.

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
        templates = [
            "M6.5.txt",
            "M8.0.txt",
            "L1.0.txt",
            "L3.5.txt",
            "L6.0.txt",
            "T2.0.txt",
            "T6.0.txt",
            "T7.5.txt",
        ]
        templates = ["stars/" + t for t in templates]
    else:
        # Intermediate and very old
        # templates = ['templates/EAZY_v1.0_lines/eazy_v1.0_sed3_nolines.dat',
        #              'templates/cvd12_t11_solar_Chabrier.extend.skip10.dat']
        templates = ["eazy_intermediate.dat", "cvd12_t11_solar_Chabrier.dat"]

        # Post starburst
        # templates.append('templates/UltraVISTA/eazy_v1.1_sed9.dat')
        templates.append("post_starburst.dat")

        # Very blue continuum
        # templates.append('templates/YoungSB/erb2010_continuum.dat')
        templates.append("erb2010_continuum.dat")

        # Test new templates
        # templates = ['templates/erb2010_continuum.dat',
        # 'templates/fsps/tweak_fsps_temp_kc13_12_006.dat',
        # 'templates/fsps/tweak_fsps_temp_kc13_12_008.dat']

        if fsps_templates:
            # templates = ['templates/fsps/tweak_fsps_temp_kc13_12_0{0:02d}.dat'.format(i+1) for i in range(12)]
            templates = [
                "fsps/fsps_QSF_12_v3_nolines_0{0:02d}.dat".format(i + 1)
                for i in range(12)
            ]
            # templates = ['fsps/fsps_QSF_7_v3_nolines_0{0:02d}.dat'.format(i+1) for i in range(7)]

        if alf_template:
            templates.append("alf_SSP.dat")

        if continuum_list is not None:
            templates = continuum_list

    temp_list = OrderedDict()
    for temp in templates:
        data = np.loadtxt(os.path.join(GRIZLI_PATH, "templates", temp), unpack=True)
        # scl = np.interp(5500., data[0], data[1])
        scl = 1.0
        name = temp  # os.path.basename(temp)
        temp_list[name] = SpectrumTemplate(wave=data[0], flux=data[1] / scl, name=name)

        temp_list[name].name = name

    if stars:
        return temp_list

    # Emission lines:
    line_wavelengths, line_ratios = get_line_wavelengths()

    if line_complexes:
        # line_list = ['Ha+SII', 'OIII+Hb+Ha', 'OII']
        # line_list = ['Ha+SII', 'OIII+Hb', 'OII']
        # line_list = ['Ha+NII+SII+SIII+He+PaB', 'OIII+Hb', 'OII+Ne', 'Lya+CIV']
        # line_list = ['Ha+NII+SII+SIII+He+PaB', 'OIII+Hb+Hg+Hd', 'OII+Ne', 'Lya+CIV']
        line_list = [
            "Ha+NII+SII+SIII+He+PaB",
            "OIII+Hb+Hg+Hd",
            "OII+Ne",
            "Gal-UV-lines",
        ]

    else:
        if full_line_list is None:
            line_list = DEFAULT_LINE_LIST
        else:
            line_list = full_line_list

        # line_list = ['Ha', 'SII']

    # Use FSPS grid for lines
    wave_grid = None
    # if fsps_templates:
    #     wave_grid = data[0]
    # else:
    #     wave_grid = None

    for li in line_list:
        scl = line_ratios[li] / np.sum(line_ratios[li])
        for i in range(len(scl)):
            if ("O32" in li) & (np.abs(line_wavelengths[li][i] - 2799) < 2):
                fwhm_i = 2500
                lorentz_i = True
            else:
                fwhm_i = fwhm
                lorentz_i = lorentz

            line_i = SpectrumTemplate(
                wave=wave_grid,
                central_wave=line_wavelengths[li][i],
                flux=None,
                fwhm=fwhm_i,
                velocity=True,
                lorentz=lorentz_i,
            )

            if i == 0:
                line_temp = line_i * scl[i]
            else:
                line_temp = line_temp + line_i * scl[i]

        name = "line {0}".format(li)
        line_temp.name = name
        temp_list[name] = line_temp

    return temp_list


def load_beta_templates(wave=np.arange(400, 2.5e4), betas=[-2, -1, 0]):
    """
    Step-function templates with f_lambda ~ (wave/1216.)**beta
    
    Parameters
    ----------
    wave: array_like
        The wavelength grid.
    beta: float
        The power-law index.
    
    Returns
    -------
    t0: dict
        A dictionary containing the step-function templates.

    """
    t0 = {}
    for beta in betas:
        key = "beta {0}".format(beta)
        t0[key] = SpectrumTemplate(wave=wave, flux=(wave / 1216.0) ** beta)
    return t0


def load_quasar_templates(
    broad_fwhm=2500,
    narrow_fwhm=1200,
    broad_lines=[
        "HeI-5877",
        "MgII",
        "Lya",
        "CIV-1549",
        "CIII-1906",
        "CIII-1908",
        "OIII-1663",
        "HeII-1640",
        "SiIV+OIV-1398",
        "NIV-1487",
        "NV-1240",
        "PaB",
        "PaG",
    ],
    narrow_lines=[
        "NIII-1750",
        "OII",
        "OIII",
        "SII",
        "OI-6302",
        "OIII-4363",
        "NeIII-3867",
        "NeVI-3426",
        "NeV-3346",
        "OII-7325",
        "ArIII-7138",
        "SIII",
        "HeI-1083",
    ],
    include_feii=True,
    slopes=[-2.8, 0, 2.8],
    uv_line_complex=True,
    fixed_narrow_lines=False,
    t1_only=False,
    nspline=13,
    Rspline=30,
    betas=None,
    include_reddened_balmer_lines=False,
):
    """
    Make templates suitable for fitting broad-line quasars
    
    Parameters
    ----------
    broad_fwhm : float, optional
        Full width at half maximum of the broad lines. Default is 2500.

    narrow_fwhm : float, optional
        Full width at half maximum of the narrow lines. Default is 1200.

    broad_lines : list, optional
        List of broad lines to include in the templates.

    narrow_lines : list, optional
        List of narrow lines to include in the templates.

    include_feii : bool, optional
        Whether to include Fe II templates. Default is True.

    slopes : list, optional
        List of slopes for linear continua. Default is [-2.8, 0, 2.8].

    uv_line_complex : bool, optional
        Whether to include UV line complex templates. Default is True.

    fixed_narrow_lines : bool, optional
        Whether to fix the narrow lines. Default is False.

    t1_only : bool, optional
        Whether to only include t1 templates. Default is False.

    nspline : int, optional
        Number of spline continua templates. Default is 13.

    Rspline : int, optional
        Resolution of the spline continua templates. Default is 30.

    betas : list, optional
        List of beta values for beta templates. Default is None.

    include_reddened_balmer_lines : bool, optional
        Whether to include reddened Balmer lines. Default is False.
        
    Returns
    -------
    t0 : OrderedDict
        Dictionary of templates for t0.

    t1 : OrderedDict
        Dictionary of templates for t1.

    """

    from collections import OrderedDict
    import scipy.ndimage as nd

    t0 = OrderedDict()
    t1 = OrderedDict()

    broad1 = load_templates(
        fwhm=broad_fwhm,
        line_complexes=False,
        stars=False,
        full_line_list=["Ha", "Hb", "Hg", "Hd", "H7", "H8", "H9", "H10"] + broad_lines,
        continuum_list=[],
        fsps_templates=False,
        alf_template=False,
        lorentz=True,
    )

    narrow1 = load_templates(
        fwhm=400,
        line_complexes=False,
        stars=False,
        full_line_list=narrow_lines,
        continuum_list=[],
        fsps_templates=False,
        alf_template=False,
    )

    if fixed_narrow_lines:
        if t1_only:
            narrow0 = narrow1
        else:
            narrow0 = load_templates(
                fwhm=narrow_fwhm,
                line_complexes=False,
                stars=False,
                full_line_list=["QSO-Narrow-lines"],
                continuum_list=[],
                fsps_templates=False,
                alf_template=False,
            )

    else:
        narrow0 = load_templates(
            fwhm=narrow_fwhm,
            line_complexes=False,
            stars=False,
            full_line_list=narrow_lines,
            continuum_list=[],
            fsps_templates=False,
            alf_template=False,
        )

    if t1_only:
        broad0 = broad1
    else:
        if uv_line_complex:
            full_line_list = ["Balmer 10kK + MgII Av=0.5", "QSO-UV-lines"]
        else:
            full_line_list = ["Balmer 10kK + MgII Av=0.5"]

        if include_reddened_balmer_lines:
            line_wavelengths, line_ratios = get_line_wavelengths()
            if "Balmer 10kK + MgII Av=1.0" in line_wavelengths:
                full_line_list += ["Balmer 10kK + MgII"]
                full_line_list += ["Balmer 10kK + MgII Av=1.0"]
                full_line_list += ["Balmer 10kK + MgII Av=2.0"]

        broad0 = load_templates(
            fwhm=broad_fwhm,
            line_complexes=False,
            stars=False,
            full_line_list=full_line_list,
            continuum_list=[],
            fsps_templates=False,
            alf_template=False,
            lorentz=True,
        )

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
        feii_wave, feii_flux = np.loadtxt(
            os.path.dirname(__file__) + "/data/templates/FeII_VeronCetty2004.txt",
            unpack=True,
        )

        # smoothing, in units of input velocity resolution
        feii_kern = broad_fwhm / 2.3548 / 75.0
        feii_sm = nd.gaussian_filter(feii_flux, feii_kern)
        t0["FeII-VC2004"] = t1["FeII-VC2004"] = SpectrumTemplate(
            wave=feii_wave, flux=feii_sm, name="FeII-VC2004"
        )

    # Linear continua
    # cont_wave = np.arange(400, 2.5e4)
    # for slope in slopes:
    #     key = 'slope {0}'.format(slope)
    #     t0[key] = t1[key] = SpectrumTemplate(wave=cont_wave, flux=(cont_wave/6563.)**slope)

    if Rspline is not None:
        wspline = np.arange(4200, 2.5e4, 10)
        df_spl = log_zgrid(zr=[wspline[0], wspline[-1]], dz=1.0 / Rspline)
        bsplines = bspline_templates(wspline, df=len(df_spl) + 2, log=True, clip=0.0001)

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
        bin_steps, step_templ = step_templates(wlim=wlim, R=onedR, round=10)
        for key in step_templ:
            t0[key] = t1[key] = step_templ[key]

    # t0['blue'] = t1['blue'] = SpectrumTemplate(wave=cont_wave, flux=(cont_wave/6563.)**-2.8)
    # t0['mid'] = t1['mid'] = SpectrumTemplate(wave=cont_wave, flux=(cont_wave/6563.)**0)
    # t0['red'] = t1['mid'] = SpectrumTemplate(wave=cont_wave, flux=(cont_wave/6563.)**2.8)

    return t0, t1


PHOENIX_LOGG_FULL = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
PHOENIX_LOGG = [4.0, 4.5, 5.0, 5.5]

PHOENIX_TEFF_FULL = [400.0, 420.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0,
                     800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0,
                     1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 1550.0,
                     1600.0, 1650.0, 1700.0, 1750.0, 1800.0, 1850.0, 1900.0, 1950.0,
                     2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0,
                     2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0,
                     3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4100.0, 4200.0, 4300.0,
                     4400.0, 4500.0, 4600.0, 4700.0, 4800.0, 4900.0, 5000.0]

PHOENIX_TEFF = [400.0, 420.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0,
                800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0,
                1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
                2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0,
                2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0,
                3600.0, 3700.0, 3800.0, 3900.0, 4000.0, 4200.0, 4400.0, 4600.0,
                4800.0, 5000.0, 5500.0, 5500, 6000.0, 6500.0, 7000.0]

PHOENIX_ZMET_FULL = [-2.5, -2.0, -1.5, -1.0, -0.5, -0.0, 0.5]
PHOENIX_ZMET = [-1.0, -0.5, -0.0]


def load_phoenix_stars(
    logg_list=PHOENIX_LOGG,
    teff_list=PHOENIX_TEFF,
    zmet_list=PHOENIX_ZMET,
    add_carbon_star=True,
    file="bt-settl_t400-7000_g4.5.fits",
):
    """
    Load Phoenix stellar templates
    
    Parameters
    ----------
    logg_list : list, optional
        List of log(g) values for the templates to load.

    teff_list : list, optional
        List of effective temperature values for the templates to load.

    zmet_list : list, optional
        List of metallicity values for the templates to load.
        
    add_carbon_star : bool, optional
        Whether to include a carbon star template.

    file : str, optional
        Name of the FITS file containing the templates. 
        Default is "bt-settl_t400-7000_g4.5.fits".
    
    Returns
    -------
    tstars : OrderedDict
        Dictionary of SpectrumTemplate objects, with the template names as keys.

    """
    from collections import OrderedDict

    try:
        from urllib.request import urlretrieve
    except:
        from urllib import urlretrieve

    # file='bt-settl_t400-5000_g4.5.fits'
    # file='bt-settl_t400-3500_z0.0.fits'

    try:
        hdu = pyfits.open(os.path.join(GRIZLI_PATH, "templates/stars/", file))
    except:
        # url = 'https://s3.amazonaws.com/grizli/CONF'
        # url = 'https://erda.ku.dk/vgrid/Gabriel%20Brammer/CONF'
        url = "https://raw.githubusercontent.com/gbrammer/" + "grizli-config/master"

        print("Fetch {0}/{1}".format(url, file))

        # os.system('wget -O /tmp/{1} {0}/{1}'.format(url, file))
        res = urlretrieve(
            "{0}/{1}".format(url, file), filename=os.path.join("/tmp", file)
        )

        hdu = pyfits.open(os.path.join("/tmp/", file))

    tab = GTable.gread(hdu[1])
    hdu.close()

    tstars = OrderedDict()
    N = tab["flux"].shape[1]
    for i in range(N):
        teff = tab.meta["TEFF{0:03d}".format(i)]
        logg = tab.meta["LOGG{0:03d}".format(i)]
        try:
            met = tab.meta["ZMET{0:03d}".format(i)]
        except:
            met = 0.0

        if (logg not in logg_list) | (teff not in teff_list) | (met not in zmet_list):
            # print('Skip {0} {1}'.format(logg, teff))
            continue

        label = "bt-settl_t{0:05.0f}_g{1:3.1f}_m{2:.1f}".format(teff, logg, met)

        tstars[label] = SpectrumTemplate(
            wave=tab["wave"], flux=tab["flux"][:, i], name=label
        )

    if add_carbon_star:
        cfile = os.path.join(GRIZLI_PATH, "templates/stars/carbon_star.txt")
        sp = read_catalog(cfile)
        if add_carbon_star > 1:
            import scipy.ndimage as nd

            cflux = nd.gaussian_filter(sp["flux"], add_carbon_star)
        else:
            cflux = sp["flux"]

        tstars["bt-settl_t05000_g0.0_m0.0"] = SpectrumTemplate(
            wave=sp["wave"], flux=cflux, name="carbon-lancon2002"
        )

    return tstars


def load_sdss_pca_templates(file="spEigenQSO-55732.fits", smooth=3000):
    """
    Load SDSS eigen PCA templates
    
    Parameters
    ----------
    file : str, optional
        The name of the FITS file containing the templates. 
        Default is "spEigenQSO-55732.fits".

    smooth : float, optional
        The smoothing parameter for the templates. Default is 3000.
    
    Returns
    -------
    temp_list : OrderedDict
        A dictionary of SpectrumTemplate objects representing the SDSS eigen templates.
    
    """
    from collections import OrderedDict
    import scipy.ndimage as nd

    im = pyfits.open(os.path.join(GRIZLI_PATH, "templates", file))
    h = im[0].header
    log_wave = np.arange(h["NAXIS1"]) * h["COEFF1"] + h["COEFF0"]
    wave = 10 ** log_wave

    name = file.split(".fits")[0]

    if smooth > 0:
        dv_in = h["COEFF1"] * 3.0e5
        n = smooth / dv_in
        data = nd.gaussian_filter1d(im[0].data, n, axis=1).astype(np.float64)
        skip = int(n / 2.5)
        wave = wave[::skip]
        data = data[:, ::skip]
    else:
        data = im[0].data.astype(np.float64)

    N = h["NAXIS2"]
    temp_list = OrderedDict()
    for i in range(N):
        temp_list["{0} {1}".format(name, i + 1)] = SpectrumTemplate(
            wave=wave, flux=data[i, :]
        )

    im.close()

    return temp_list


def cheb_templates(
    wave, order=6, get_matrix=False, log=False, clip=1.0e-4, minmax=None
):
    """
    Chebyshev polynomial basis functions
    
    Parameters
    ----------
    wave : array-like
        The wavelength array.

    order : int
        The order of the Chebyshev polynomial.

    get_matrix : bool, optional
        If True, return array data rather than template objects.
        Default is False.

    log : bool, optional
        If True, use the logarithm of the wavelength array. Default is False.

    clip : float, optional
        The clipping threshold for wavelengths outside the range.
        Default is 1.0e-4.

    minmax : array-like, optional
        The minimum and maximum values of the wavelength range.
        If not provided, the minimum and maximum values of 
        the input wavelength array are used.
        
    Returns
    -------
    templates : OrderedDict
        The Chebyshev polynomial templates.
    
    """
    from numpy.polynomial.chebyshev import chebval, chebvander

    if minmax is None:
        mi = wave.min()
        ma = wave.max()
    else:
        mi, ma = np.squeeze(minmax) * 1.0

    if log:
        xi = np.log(wave)
        mi = np.log(mi)
        ma = np.log(ma)
    else:
        xi = wave * 1

    x = (xi - mi) * 2 / (ma - mi) - 1

    n_bases = order + 1

    basis = chebvander(x, order)

    # for i in range(n_bases):
    out_of_range = (xi < mi) | (xi > ma)
    basis[out_of_range, :] = 0

    if get_matrix:
        return basis

    temp = OrderedDict()
    for i in range(n_bases):
        key = f"cheb o{i}"
        temp[key] = SpectrumTemplate(wave, basis[:, i])
        temp[key].name = key

    return temp


def bspline_templates(
    wave, degree=3, df=6, get_matrix=False, log=False, clip=1.0e-4, minmax=None
):
    """
    B-spline basis functions, modeled after `~patsy.splines`

    Parameters
    ----------
    wave : array-like 
        The wavelength array.
    degree : int 
        The degree of the B-spline basis functions.
    df : int 
        The number of degrees of freedom.
    get_matrix : bool 
        If True, return the basis function matrix.
    log : bool 
        If True, use the logarithm of the wavelength array.
    clip : float 
        The threshold for clipping the basis functions.
    minmax : tuple 
        The minimum and maximum values for the wavelength array.

    Returns
    -------
    basis : array-like 
        The B-spline basis functions.
    temp : OrderedDict 
        The B-spline templates.
    
    """
    from scipy.interpolate import splev

    order = degree + 1
    n_inner_knots = df - order
    inner_knots = np.linspace(0, 1, n_inner_knots + 2)[1:-1]

    norm_knots = np.concatenate(([0, 1] * order, inner_knots))
    norm_knots.sort()

    if log:
        xspl = np.log(wave)
    else:
        xspl = wave * 1

    if minmax is None:
        mi = xspl.min()
        ma = xspl.max()
    else:
        mi, ma = minmax

    width = ma - mi
    all_knots = norm_knots * width + mi

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
        basis[i][basis[i] < clip * maxval[i]] = 0

    if get_matrix:
        return np.vstack(basis).T

    temp = OrderedDict()
    for i in range(n_bases):
        key = "bspl {0} {1:.0f}".format(i, wave_peak[i])
        temp[key] = SpectrumTemplate(wave, basis[i])
        temp[key].name = key
        temp[key].wave_peak = wave_peak[i]

    temp.knots = all_knots
    temp.degree = degree
    temp.xspl = xspl

    return temp


def eval_bspline_templates(wave, bspl, coefs):
    """
    Evaluate B-spline templates at given wavelengths.

    Parameters
    ----------
    wave : array-like 
        The wavelengths at which to evaluate the B-spline templates.
    bspl : scipy.interpolate.BSpline 
        The B-spline object defining the basis functions.
    coefs : array-like 
        The coefficients of the B-spline basis functions.

    Returns
    -------
    array-like: 
        The evaluated B-spline templates at the given wavelengths.
    
    """
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
        templ = stars["stars/L1.0.txt"]

    wspline = templ.wave

    clip = (wspline > wavelength_range[0]) & (wspline < wavelength_range[1])
    df_spl = len(utils.log_zgrid(zr=wavelength_range, dz=1.0 / Rspline))

    tspline = utils.bspline_templates(
        wspline[clip], df=df_spl + 2, log=log, clip=0.0001, get_matrix=True
    )

    ix = np.argmax(tspline, axis=0)
    knots = wspline[clip][ix]

    N = tspline.shape[1]
    stemp = OrderedDict()
    for i in range(N):
        name = "{0} {1:.2f}".format(templ.name, knots[i] / 1.0e4)
        stemp[name] = utils.SpectrumTemplate(
            wave=wspline[clip], flux=templ.flux[clip] * tspline[:, i], name=name
        )
        stemp[name].knot = knots[i]

    stemp.wspline = wspline[clip]
    stemp.tspline = tspline
    stemp.knots = knots

    return stemp


def step_templates(
    wlim=[5000, 1.8e4],
    bin_steps=None,
    R=30,
    round=10,
    rest=False,
    special=None,
    order=0,
):
    """
    Step-function templates for easy binning
    
    Parameters
    ----------    
    wlim : list 
        The wavelength range for the templates.

    bin_steps : ndarray 
        The array of bin steps for the templates.

    R : int 
        The resolution of the templates.

    round : int
        The rounding factor for the bin steps.

    rest : bool 
        Flag indicating whether the templates are in the rest frame.

    special : str 
        Special template type. Options are 'D4000', 'Dn4000', and None.

    order : int 
        The order of the step function templates.
    
    Returns
    -------
    bin_steps : ndarray 
        The array of bin steps for the templates.
    step_templ : dict 
        Dictionary of step function templates.
    
    """
    if special == "Dn4000":
        rest = True
        bin_steps = np.hstack(
            [
                np.arange(850, 3849, 100),
                [3850, 3950, 4000, 4100],
                np.arange(4200, 1.7e4, 100),
            ]
        )

    elif special == "D4000":
        rest = True
        bin_steps = np.hstack(
            [
                np.arange(850, 3749, 200),
                [3750, 3950, 4050, 4250],
                np.arange(4450, 1.7e4, 200),
            ]
        )
    elif special not in ["D4000", "Dn4000", None]:
        print(
            "step_templates: {0} not recognized (options are 'D4000', 'Dn4000', and None)".format(
                special
            )
        )
        return {}

    if bin_steps is None:
        bin_steps = np.round(log_zgrid(wlim, 1.0 / R) / round) * round
    else:
        wlim = [bin_steps[0], bin_steps[-1]]

    ds = np.diff(bin_steps)

    xspec = np.arange(wlim[0] - ds[0], wlim[1] + ds[-1])

    bin_mid = bin_steps[:-1] + ds / 2.0

    step_templ = {}
    for i in range(len(bin_steps) - 1):

        yspec = ((xspec >= bin_steps[i]) & (xspec < bin_steps[i + 1])) * 1

        for o in range(order + 1):
            label = "step {0:.0f}-{1:.0f} {2}".format(bin_steps[i], bin_steps[i + 1], o)
            if rest:
                label = "r" + label

            flux = ((xspec - bin_mid[i]) / ds[i]) ** o * (yspec > 0)
            step_templ[label] = SpectrumTemplate(wave=xspec, flux=flux, name=label)

    return bin_steps, step_templ


def polynomial_templates(wave, ref_wave=1.0e4, order=0, line=False):
    """
    Generate polynomial templates based on the input parameters.

    If `line` is True, the method generates line templates by applying 
    a sign to the polynomial. Otherwise, it generates polynomial templates by 
    raising the wavelength ratio to the power of the polynomial order.

    Each template is stored in the returned dictionary with a key in the format
    "poly {i}", where i is the polynomial order.

    Parameters
    ----------   
    wave : array-like 
        The wavelength array.

    ref_wave : float, optional 
        The reference wavelength. Default is 1.0e4.

    order : int, optional 
        The order of the polynomial. Default is 0.

    line : bool, optional 
        Whether to generate line templates. Default is False.

    Returns
    -------
    temp: OrderedDict 
        A dictionary of SpectrumTemplate objects
        representing the polynomial templates.

    """
    temp = OrderedDict()
    if line:
        for sign in [1, -1]:
            key = "poly {0}".format(sign)
            temp[key] = SpectrumTemplate(wave, sign * (wave / ref_wave - 1) + 1)
            temp[key].name = key

        return temp

    for i in range(order + 1):
        key = "poly {0}".format(i)
        temp[key] = SpectrumTemplate(wave, (wave / ref_wave - 1) ** i)
        temp[key].name = key
        temp[key].ref_wave = ref_wave

    return temp


def split_poly_template(templ, ref_wave=1.0e4, order=3):
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

    tspline = polynomial_templates(
        templ.wave, ref_wave=ref_wave, order=order, line=False
    )

    ptemp = OrderedDict()

    for i, t in enumerate(tspline):
        name = "{0} poly {1}".format(templ.name, i)
        ptemp[name] = utils.SpectrumTemplate(
            wave=templ.wave, flux=templ.flux * tspline[t].flux, name=name
        )
        ptemp[name].ref_wave = ref_wave

    ptemp.ref_wave = ref_wave

    return ptemp


def dot_templates(coeffs, templates, z=0, max_R=5000, apply_igm=True):
    """
    Compute template sum analogous to `np.dot(coeffs, templates)`.

    Parameters
    ----------
    coeffs : array-like
        Coefficients for each template.

    templates : list of `~grizli.utils.SpectrumTemplate`
        List of templates.

    z : float, optional
        Redshift to apply to the templates (default is 0).

    max_R : float, optional
        Maximum spectral resolution to apply to the templates (default is 5000).

    apply_igm : bool, optional
        Apply intergalactic medium (IGM) attenuation to the templates.

    Returns
    -------
    tc : `~grizli.utils.SpectrumTemplate`
        Continuum template.
    tl : `~grizli.utils.SpectrumTemplate`
        Full template (including lines).

    """

    if len(coeffs) != len(templates):
        raise ValueError(
            "shapes of coeffs ({0}) and templates ({1}) don't match".format(
                len(coeffs), len(templates)
            )
        )

    wave, flux_arr, is_line = array_templates(
        templates, max_R=max_R, z=z, apply_igm=apply_igm
    )

    # Continuum
    cont = np.dot(coeffs * (~is_line), flux_arr)
    tc = SpectrumTemplate(wave=wave, flux=cont).zscale(z, apply_igm=False)

    # Full template
    line = np.dot(coeffs, flux_arr)
    tl = SpectrumTemplate(wave=wave, flux=line).zscale(z, apply_igm=False)

    return tc, tl


def array_templates(templates, wave=None, max_R=5000, z=0, apply_igm=False):
    """
    Return an array version of the templates 
    that have all been interpolated to the same grid.


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

    wave : `~numpy.ndarray`, dimensions `(NL,)`, optional
        Array containing unique wavelengths. If not provided, the wavelengths
        will be determined from the templates.

    flux_arr : `~numpy.ndarray`, dimensions `(NTEMP, NL)`, optional
        Array containing the template fluxes interpolated at `wave`. If not
        provided, the fluxes will be computed from the templates.

    is_line : `~numpy.ndarray`, optional
        Boolean array indicating emission line templates (the key in the
        template dictionary starts with "line ").

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
    from grizli.utils_numba.interp import interp_conserve_c

    if wave is None:
        wstack = []
        for t in templates:
            if t.split()[0] in ["bspl", "step", "poly"]:
                wstack.append(templates[t].wave / (1 + z))
            else:
                wstack.append(templates[t].wave)

        wave = np.unique(np.hstack(wstack))

    clipsum, iter = 1, 0
    while (clipsum > 0) & (iter < 10):
        clip = np.gradient(wave) / wave < 1 / max_R
        idx = np.arange(len(wave))[clip]
        wave[idx[::2]] = np.nan
        wave = wave[np.isfinite(wave)]
        iter += 1
        clipsum = clip.sum()
        # print(iter, clipsum)

    NTEMP = len(templates)
    flux_arr = np.zeros((NTEMP, len(wave)))

    for i, t in enumerate(templates):
        if t.split()[0] in ["bspl", "step", "poly"]:
            flux_arr[i, :] = interp_conserve_c(
                wave, templates[t].wave / (1 + z), templates[t].flux * (1 + z)
            )
        else:
            if hasattr(templates[t], "flux_flam"):
                # Redshift-dependent eazy-py Template
                flux_arr[i, :] = interp_conserve_c(
                    wave, templates[t].wave, templates[t].flux_flam(z=z)
                )
            else:
                flux_arr[i, :] = interp_conserve_c(
                    wave, templates[t].wave, templates[t].flux
                )

    is_line = np.array([t.startswith("line ") for t in templates])

    # IGM
    if apply_igm:
        try:
            import eazy.igm

            IGM = eazy.igm.Inoue14()

            lylim = wave < 1250
            igmz = np.ones_like(wave)
            igmz[lylim] = IGM.full_IGM(z, wave[lylim] * (1 + z))
        except:
            igmz = 1.0
    else:
        igmz = 1.0

    obsnames = ["bspl", "step", "poly"]
    is_obsframe = np.array([t.split()[0] in obsnames for t in templates])

    flux_arr[~is_obsframe, :] *= igmz

    # Multiply spline?
    for i, t in enumerate(templates):
        if "spline" in t:
            for j, tj in enumerate(templates):
                if is_obsframe[j]:
                    ma = flux_arr[j, :].sum()
                    ma = ma if ma > 0 else 1
                    ma = 1

                    flux_arr[j, :] *= flux_arr[i, :] / ma

    return wave, flux_arr, is_line


def compute_equivalent_widths(
    templates, coeffs, covar, max_R=5000, Ndraw=1000, seed=0, z=0, observed_frame=False
):
    """
    Compute template-fit emission line equivalent widths

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
        EWdict[key] = (0.0, 0.0, 0.0)

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
    continuum = np.dot(draws * (~is_line[clip]), flux_arr[clip, :])

    # Compute the emission line EWs
    tidx = np.where(is_line[clip])[0]
    for ix in tidx:
        key = keys[clip][ix]

        # Line template
        line = np.dot(draws[:, ix][:, None], flux_arr[clip, :][ix, :][None, :])

        # Where line template non-zero
        mask = flux_arr[clip, :][ix, :] > 0
        ew_i = np.trapz(
            (line / continuum)[:, mask], wave[mask] * (1 + z * observed_frame), axis=1
        )

        EWdict[key] = np.percentile(ew_i, [16.0, 50.0, 84.0])

    return EWdict


#####################
# Photometry from Vizier tables


# CFHTLS
CFHTLS_W_VIZIER = "II/317/cfhtls_w"
CFHTLS_W_BANDS = OrderedDict(
    [
        ("cfht_mega_u", ["umag", "e_umag"]),
        ("cfht_mega_g", ["gmag", "e_gmag"]),
        ("cfht_mega_r", ["rmag", "e_rmag"]),
        ("cfht_mega_i", ["imag", "e_imag"]),
        ("cfht_mega_z", ["zmag", "e_zmag"]),
    ]
)

CFHTLS_D_VIZIER = "II/317/cfhtls_d"
CFHTLS_D_BANDS = OrderedDict(
    [
        ("cfht_mega_u", ["umag", "e_umag"]),
        ("cfht_mega_g", ["gmag", "e_gmag"]),
        ("cfht_mega_r", ["rmag", "e_rmag"]),
        ("cfht_mega_i", ["imag", "e_imag"]),
        ("cfht_mega_z", ["zmag", "e_zmag"]),
    ]
)

# SDSS DR12
SDSS_DR12_VIZIER = "V/147/sdss12"
SDSS_DR12_BANDS = OrderedDict(
    [
        ("SDSS/u", ["umag", "e_umag"]),
        ("SDSS/g", ["gmag", "e_gmag"]),
        ("SDSS/r", ["rmag", "e_rmag"]),
        ("SDSS/i", ["imag", "e_imag"]),
        ("SDSS/z", ["zmag", "e_zmag"]),
    ]
)

# PanStarrs
PS1_VIZIER = "II/349/ps1"
PS1_BANDS = OrderedDict(
    [
        ("PS1.g", ["gKmag", "e_gKmag"]),
        ("PS1.r", ["rKmag", "e_rKmag"]),
        ("PS1.i", ["iKmag", "e_iKmag"]),
        ("PS1.z", ["zKmag", "e_zKmag"]),
        ("PS1.y", ["yKmag", "e_yKmag"]),
    ]
)

# KIDS DR3
KIDS_DR3_VIZIER = "II/347/kids_dr3"
KIDS_DR3_BANDS = OrderedDict(
    [
        ("OCam.sdss.u", ["umag", "e_umag"]),
        ("OCam.sdss.g", ["gmag", "e_gmag"]),
        ("OCam.sdss.r", ["rmag", "e_rmag"]),
        ("OCam.sdss.i", ["imag", "e_imag"]),
    ]
)

# WISE all-sky
WISE_VIZIER = "II/328/allwise"
WISE_BANDS = OrderedDict(
    [("WISE/RSR-W1", ["W1mag", "e_W1mag"]), ("WISE/RSR-W2", ["W2mag", "e_W2mag"])]
)
# ('WISE/RSR-W3', ['W3mag', 'e_W3mag']),
# ('WISE/RSR-W4', ['W4mag', 'e_W4mag'])])

# VIKING VISTA
VIKING_VIZIER = "II/343/viking2"
VIKING_BANDS = OrderedDict(
    [
        ("SDSS/z", ["Zpmag", "e_Zpmag"]),
        ("VISTA/Y", ["Ypmag", "e_Ypmag"]),
        ("VISTA/J", ["Jpmag", "e_Jpmag"]),
        ("VISTA/H", ["Hpmag", "e_Hpmag"]),
        ("VISTA/Ks", ["Kspmag", "e_Kspmag"]),
    ]
)

# UKIDSS wide surveys
UKIDSS_LAS_VIZIER = "II/319/las9"
UKIDSS_LAS_BANDS = OrderedDict(
    [
        ("WFCAM_Y", ["Ymag", "e_Ymag"]),
        ("WFCAM_J", ["Jmag1", "e_Jmag1"]),
        ("WFCAM_J", ["Jmag2", "e_Jmag2"]),
        ("WFCAM_H", ["Hmag", "e_Hmag"]),
        ("WFCAM_K", ["Kmag", "e_Kmag"]),
    ]
)

UKIDSS_DXS_VIZIER = "II/319/dxs9"
UKIDSS_DXS_BANDS = OrderedDict(
    [("WFCAM_J", ["Jmag", "e_Jmag"]), ("WFCAM_K", ["Kmag", "e_Kmag"])]
)

# GALEX
GALEX_MIS_VIZIER = "II/312/mis"
GALEX_MIS_BANDS = OrderedDict([("FUV", ["FUV", "e_FUV"]), ("NUV", ["NUV", "e_NUV"])])

GALEX_AIS_VIZIER = "II/312/ais"
GALEX_AIS_BANDS = OrderedDict([("FUV", ["FUV", "e_FUV"]), ("NUV", ["NUV", "e_NUV"])])

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


def get_Vizier_photometry(
    ra,
    dec,
    templates=None,
    radius=2,
    vizier_catalog=PS1_VIZIER,
    bands=PS1_BANDS,
    filter_file="/usr/local/share/eazy-photoz/filters/FILTER.RES.latest",
    MW_EBV=0,
    convert_vega=False,
    raw_query=False,
    verbose=True,
    timeout=300,
    rowlimit=50000,
):
    """
    Fetch photometry from a Vizier catalog

    Requires eazypy/eazy

    Parameters
    ----------
    ra : float
        Right ascension of the target in degrees.

    dec : float
        Declination of the target in degrees.

    templates : dict, optional
        Dictionary of templates to be used for photometric redshift fitting.

    radius : float, optional
        Search radius around the target position in arcseconds.

    vizier_catalog : str or list, optional
        Name of the Vizier catalog(s) to query or a list of catalog names.

    bands : dict, optional
        Dictionary of band names and corresponding column names in 
        the Vizier catalog.

    filter_file : str, optional
        Path to the filter file.

    MW_EBV : float, optional
        Milky Way E(B-V) reddening value.

    convert_vega : bool, optional
        Flag indicating whether to convert the photometry 
        from Vega to AB magnitude system.

    raw_query : bool, optional
        Flag indicating whether to return the raw query result.

    verbose : bool, optional
        Flag indicating whether to print verbose output.

    timeout : int, optional
        Timeout value for the Vizier query in seconds.

    rowlimit : int, optional
        Maximum number of rows to retrieve from the Vizier catalog.

    Returns
    -------
    phot : OrderedDict
        Dictionary containing the retrieved photometry and related information.

    """

    from collections import OrderedDict

    import astropy.units as u
    from astroquery.vizier import Vizier

    Vizier.ROW_LIMIT = rowlimit
    Vizier.TIMEOUT = timeout

    # print('xxx', Vizier.ROW_LIMIT, Vizier.TIMEOUT)

    import astropy.coordinates as coord
    import astropy.units as u

    # import pysynphot as S

    from eazy.templates import Template
    from eazy.filters import FilterFile
    from eazy.photoz import TemplateGrid
    from eazy.filters import FilterDefinition

    res = FilterFile(filter_file)

    coo = coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")

    columns = ["*"]
    # columns = []
    if isinstance(vizier_catalog, list):
        for c in [VIKING_VIZIER]:
            for b in VIZIER_BANDS[c]:
                columns += VIZIER_BANDS[c][b]

        columns = list(np.unique(columns))
        # print("xxx columns", columns)
    else:
        for b in bands:
            columns += bands[b]

    if isinstance(vizier_catalog, list):
        v = Vizier(catalog=VIKING_VIZIER, columns=["+_r"] + columns)
    else:
        v = Vizier(catalog=vizier_catalog, columns=["+_r"] + columns)

    v.ROW_LIMIT = rowlimit
    v.TIMEOUT = timeout

    # query_catalog = vizier_catalog
    try:
        tabs = v.query_region(
            coo, radius="{0}s".format(radius), catalog=vizier_catalog
        )  # [0]

        if raw_query:
            return tabs

        tab = tabs[0]

        if False:
            for t in tabs:
                bands = VIZIER_BANDS[t.meta["name"]]
                for b in bands:
                    for c in bands[b]:
                        print(t.meta["name"], c, c in t.colnames)  # c = bands[b][0]

        ix = np.argmin(tab["_r"])
        tab = tab[ix]
    except:
        tab = None

        return None

    viz_tables = ", ".join([t.meta["name"] for t in tabs])
    if verbose:
        print("Photometry from vizier catalogs: {0}".format(viz_tables))

    pivot = []  # OrderedDict()
    flam = []
    eflam = []
    filters = []

    for tab in tabs:

        # Downweight PS1 if have SDSS ?  For now, do nothing
        if (tab.meta["name"] == PS1_VIZIER) & (SDSS_DR12_VIZIER in viz_tables):
            # continue
            err_scale = 1
        else:
            err_scale = 1

        # Only use one CFHT catalog
        if (tab.meta["name"] == CFHTLS_W_VIZIER) & (CFHTLS_D_VIZIER in viz_tables):
            continue

        if tab.meta["name"] == UKIDSS_LAS_VIZIER:
            flux_scale = 1.33
        else:
            flux_scale = 1.0

        convert_vega = VIZIER_VEGA[tab.meta["name"]]
        bands = VIZIER_BANDS[tab.meta["name"]]

        # if verbose:
        #    print(tab.colnames)

        # filters += [res.filters[res.search(b, verbose=False)[0]] for b in bands]

        to_flam = 10 ** (-0.4 * (48.6)) * 3.0e18  # / pivot(Ang)**2

        for ib, b in enumerate(bands):
            filt = res.filters[res.search(b, verbose=False)[0]]
            filters.append(filt)

            if convert_vega:
                to_ab = filt.ABVega()
            else:
                to_ab = 0.0

            fcol, ecol = bands[b]
            pivot.append(filt.pivot())
            flam.append(
                10 ** (-0.4 * (tab[fcol][0] + to_ab)) * to_flam / pivot[-1] ** 2
            )
            flam[-1] *= flux_scale
            eflam.append(tab[ecol][0] * np.log(10) / 2.5 * flam[-1] * err_scale)

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

        eazy_templates = [
            Template(arrays=(templates[k].wave, templates[k].flux), name=k)
            for k in templates
        ]

        zgrid = log_zgrid(zr=[0.01, 3.4], dz=0.005)

        tempfilt = TemplateGrid(
            zgrid,
            eazy_templates,
            filters=filters,
            add_igm=True,
            galactic_ebv=MW_EBV,
            Eb=0,
            n_proc=0,
            verbose=False,
        )
    else:
        tempfilt = None

    phot = OrderedDict(
        [
            ("flam", np.array(flam)),
            ("eflam", np.array(eflam)),
            ("filters", filters),
            ("tempfilt", tempfilt),
            ("lc", np.array(lc)),
            ("source", "Vizier " + viz_tables),
        ]
    )

    return phot


def generate_tempfilt(templates, filters, zgrid=None, MW_EBV=0):
    """
    Generate a template grid for photometric redshift fitting.

    Parameters
    ----------
    templates : dict
        Dictionary of templates. Each template should be 
        an instance of `eazy.templates.Template`.

    filters : list
        List of filters to be used for the photometric redshift fitting.

    zgrid : array-like, optional
        Redshift grid. If not provided, a default grid will be used.

    MW_EBV : float, optional
        Milky Way E(B-V) reddening value. Default is 0.

    Returns
    -------
    tempfilt : `eazy.photoz.TemplateGrid`
        Template grid for photometric redshift fitting.
    
    """
    from eazy.templates import Template
    from eazy.photoz import TemplateGrid

    eazy_templates = [
        Template(arrays=(templates[k].wave, templates[k].flux), name=k)
        for k in templates
    ]

    if zgrid is None:
        zgrid = log_zgrid(zr=[0.01, 3.4], dz=0.005)

    tempfilt = TemplateGrid(
        zgrid,
        eazy_templates,
        filters=filters,
        add_igm=True,
        galactic_ebv=MW_EBV,
        Eb=0,
        n_proc=0,
        verbose=False,
    )

    return tempfilt


def combine_phot_dict(phots, templates=None, MW_EBV=0):
    """
    Combine photmetry dictionaries
    
    Parameters
    ----------
    phots : list
        List of photometry dictionaries to combine.

    templates : list, optional
        List of templates to use for generating `tempfilt`.

    MW_EBV : float, optional
        Milky Way E(B-V) reddening value.
    
    Returns
    -------
    dict
        Combined photometry dictionary.
    
    """
    phot = {}
    phot["flam"] = []
    phot["eflam"] = []
    phot["filters"] = []
    for p in phots:
        phot["flam"] = np.append(phot["flam"], p["flam"])
        phot["eflam"] = np.append(phot["eflam"], p["eflam"])
        phot["filters"].extend(p["filters"])

    if templates is not None:
        phot["tempfilt"] = generate_tempfilt(templates, phot["filters"], MW_EBV=MW_EBV)

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

    flat = S.FlatSpectrum(0, fluxunits="ABMag")
    ab_mags = OrderedDict()

    for bp in bandpasses:
        flat_obs = S.Observation(flat, bp)
        spec_obs = S.Observation(spectrum, bp)
        ab_mags[bp.name] = -2.5 * np.log10(spec_obs.countrate() / flat_obs.countrate())

    return ab_mags


def log_zgrid(zr=[0.7, 3.4], dz=0.01):
    """
    Make a logarithmically spaced redshift grid

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
    zgrid = np.exp(np.arange(np.log(1 + zr[0]), np.log(1 + zr[1]), dz)) - 1
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
    diff = np.diff(x) / 2.0
    dx[:-1] += diff
    dx[1:] += diff
    return dx


def get_wcs_pscale(wcs, set_attribute=True):
    """
    Get correct pscale from a `~astropy.wcs.WCS` object

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

    if hasattr(wcs.wcs, "cd"):
        det = linalg.det(wcs.wcs.cd)
    else:
        det = linalg.det(wcs.wcs.pc)

    pscale = np.sqrt(np.abs(det)) * 3600.0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "cdelt will be ignored since cd is present", RuntimeWarning
        )

        if hasattr(wcs.wcs, "cdelt"):
            pscale *= wcs.wcs.cdelt[0]

    wcs.pscale = pscale

    return pscale


def transform_wcs(in_wcs, translation=[0.0, 0.0], rotation=0.0, scale=1.0):
    """
    Update WCS with shift, rotation, & scale

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

    # out_wcs.wcs.crpix += np.array(translation)

    # Compute shift for crval, not crpix
    crval = in_wcs.all_pix2world(
        [in_wcs.wcs.crpix - np.array(translation)], 1
    ).flatten()

    # Compute shift at image center
    if hasattr(in_wcs, "_naxis1"):
        refpix = np.array([in_wcs._naxis1 / 2.0, in_wcs._naxis2 / 2.0])
    else:
        refpix = np.array(in_wcs._naxis) / 2.0

    c0 = in_wcs.all_pix2world([refpix], 1).flatten()
    c1 = in_wcs.all_pix2world([refpix - np.array(translation)], 1).flatten()

    out_wcs.wcs.crval += c1 - c0

    theta = -rotation
    _mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    try:
        out_wcs.wcs.cd[:2, :2] = np.dot(out_wcs.wcs.cd[:2, :2], _mat) / scale
    except:
        out_wcs.wcs.pc = np.dot(out_wcs.wcs.pc, _mat) / scale

    out_wcs.pscale = get_wcs_pscale(out_wcs)
    # out_wcs.wcs.crpix *= scale
    if hasattr(out_wcs, "pixel_shape"):
        _naxis1 = int(np.round(out_wcs.pixel_shape[0] * scale))
        _naxis2 = int(np.round(out_wcs.pixel_shape[1] * scale))
        out_wcs._naxis = [_naxis1, _naxis2]
    elif hasattr(out_wcs, "_naxis1"):
        out_wcs._naxis1 = int(np.round(out_wcs._naxis1 * scale))
        out_wcs._naxis2 = int(np.round(out_wcs._naxis2 * scale))

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

    reverse : bool, optional
        If `input` is a header and includes a keyword ``ROT90``, then undo
        the rotation and remove the keyword from the output header

    verbose : bool, optional
        If True, print the root-mean-square difference between the original
        and rotated coordinates

    compare : bool, optional
        If True, plot the difference between the original and rotated
        coordinates as a function of x and y

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

        if "ROT90" in input:
            if reverse:
                rot = -orig["ROT90"]
                new.remove("ROT90")
            else:
                new["ROT90"] = orig["ROT90"] + rot
        else:
            new["ROT90"] = rot
    else:
        orig = to_header(input)
        new = to_header(input)

    orig_wcs = pywcs.WCS(orig, relax=True)

    ### CD = [[dra/dx, dra/dy], [dde/dx, dde/dy]]
    ### x = a_i_j * u**i * v**j
    ### y = b_i_j * u**i * v**j

    ix = 1

    if compare:
        xarr = np.arange(0, 2048, 64)
        xp, yp = np.meshgrid(xarr, xarr)
        rd = orig_wcs.all_pix2world(xp, yp, ix)

    if rot % 4 == 1:
        # CW 90 deg : x = y, y = (nx - x), u=v, v=-u
        desc = "x=y, y=nx-x"

        new["CRPIX1"] = orig["CRPIX2"]
        new["CRPIX2"] = orig["NAXIS1"] - orig["CRPIX1"] + 1

        new["CD1_1"] = orig["CD1_2"]
        new["CD1_2"] = -orig["CD1_1"]
        new["CD2_1"] = orig["CD2_2"]
        new["CD2_2"] = -orig["CD2_1"]

        for i in range(new["A_ORDER"] + 1):
            for j in range(new["B_ORDER"] + 1):
                Aij = f"A_{i}_{j}"
                if Aij not in new:
                    continue

                new[f"A_{i}_{j}"] = orig[f"B_{j}_{i}"] * (-1) ** j
                new[f"B_{i}_{j}"] = orig[f"A_{j}_{i}"] * (-1) ** j * -1

        new_wcs = astropy.wcs.WCS(new, relax=True)

        if compare:
            xr, yr = new_wcs.all_world2pix(*rd, ix)
            xo = yp
            yo = orig["NAXIS1"] - xp

    elif rot % 4 == 3:
        # CW 270 deg : y = x, x = (ny - u), u=-v, v=u
        desc = "x=ny-y, y=x"

        new["CRPIX1"] = orig["NAXIS2"] - orig["CRPIX2"] + 1
        new["CRPIX2"] = orig["CRPIX1"]

        new["CD1_1"] = -orig["CD1_2"]
        new["CD1_2"] = orig["CD1_1"]
        new["CD2_1"] = -orig["CD2_2"]
        new["CD2_2"] = orig["CD2_1"]

        for i in range(new["A_ORDER"] + 1):
            for j in range(new["B_ORDER"] + 1):
                Aij = f"A_{i}_{j}"
                if Aij not in new:
                    continue

                new[f"A_{i}_{j}"] = orig[f"B_{j}_{i}"] * (-1) ** i * -1
                new[f"B_{i}_{j}"] = orig[f"A_{j}_{i}"] * (-1) ** i

        new_wcs = astropy.wcs.WCS(new, relax=True)

        if compare:
            xr, yr = new_wcs.all_world2pix(*rd, ix)
            xo = orig["NAXIS2"] - yp
            yo = xp

    elif rot % 4 == 2:
        # CW 180 deg : x=nx-x, y=ny-y, u=-u, v=-v
        desc = "x=nx-x, y=ny-y"

        new["CRPIX1"] = orig["NAXIS1"] - orig["CRPIX1"] + 1
        new["CRPIX2"] = orig["NAXIS2"] - orig["CRPIX2"] + 1

        new["CD1_1"] = -orig["CD1_1"]
        new["CD1_2"] = -orig["CD1_2"]
        new["CD2_1"] = -orig["CD2_1"]
        new["CD2_2"] = -orig["CD2_2"]

        for i in range(new["A_ORDER"] + 1):
            for j in range(new["B_ORDER"] + 1):
                Aij = f"A_{i}_{j}"
                if Aij not in new:
                    continue

                new[f"A_{i}_{j}"] = orig[f"A_{i}_{j}"] * (-1) ** j * (-1) ** i * -1
                new[f"B_{i}_{j}"] = orig[f"B_{i}_{j}"] * (-1) ** j * (-1) ** i * -1

        new_wcs = astropy.wcs.WCS(new, relax=True)

        if compare:
            xr, yr = new_wcs.all_world2pix(*rd, ix)
            xo = orig["NAXIS1"] - xp
            yo = orig["NAXIS2"] - yp
    else:
        # rot=0, do nothing
        desc = "x=x, y=y"
        new_wcs = orig_wcs
        if compare:
            xo = xp
            yo = yp
            xr, yr = new_wcs.all_world2pix(*rd, ix)

    if verbose:
        if compare:
            xrms = nmad(xr - xo)
            yrms = nmad(yr - yo)
            print(f"Rot90: {rot} rms={xrms:.2e} {yrms:.2e}")

    if compare:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        axes[0].scatter(xp, xr - xo)
        axes[0].set_xlabel("dx")
        axes[1].scatter(yp, yr - yo)
        axes[1].set_xlabel("dy")
        for ax in axes:
            ax.grid()

        fig.tight_layout(pad=0.5)

    return new, new_wcs, desc


def get_wcs_slice_header(wcs, slx, sly):
    """
    Generate a `~astropy.io.fits.Header` for a sliced WCS object.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        The original WCS object.
    slx : `slice`
        The slice along the x-axis.
    sly : `slice`
        The slice along the y-axis.

    Returns
    -------
    h : `~astropy.io.fits.Header`
        The header for the sliced WCS object.

    """
    h = wcs.slice((sly, slx)).to_header(relax=True)
    h["NAXIS"] = 2
    h["NAXIS1"] = slx.stop - slx.start
    h["NAXIS2"] = sly.stop - sly.start
    for k in h:
        if k.startswith("PC"):
            h.rename_keyword(k, k.replace("PC", "CD"))

    return h


def get_fits_slices(file1, file2):
    """
    Get overlapping slices of FITS files

    Parameters
    ----------
    file1 : str, `~astropy.io.fits.Header`, or `~astropy.io.fits.HDUList`
        First file, header or HDU

    file2 : str, `~astropy.io.fits.Header`, or `~astropy.io.fits.HDUList`
        Second file, header or HDU

    Returns
    -------
    nx, ny : int
        Size of the overlapping region

    valid : bool
        True if there is some overlap

    sl1 : (slice, slice)
        y and x slices of the overlap in ``file1``

    sl2 : (slice, slice)
        y and x slices of the overlap in ``file2``

    """

    # First image
    if isinstance(file1, pyfits.HDUList):
        h1 = file1[0].header

    elif isinstance(file1, str):
        im1 = pyfits.open(file1)
        h1 = im1[0].header

    else:
        h1 = file1

    # Second image
    if isinstance(file2, pyfits.HDUList):
        h2 = file2[0].header

    elif isinstance(file2, str):
        im2 = pyfits.open(file2)
        h2 = im2[0].header

    else:
        h2 = file2

    # origin and shape tuples
    o1 = np.array([-h1["CRPIX2"], -h1["CRPIX1"]]).astype(int)
    sh1 = np.array([h1["NAXIS2"], h1["NAXIS1"]])

    o2 = np.array([-h2["CRPIX2"], -h2["CRPIX1"]]).astype(int)
    sh2 = np.array([h2["NAXIS2"], h2["NAXIS1"]])

    # slices
    sl1, sl2 = get_common_slices(o1, sh1, o2, sh2)

    nx = sl1[1].stop - sl1[1].start
    ny = sl1[0].stop - sl1[0].start
    valid = (nx > 0) & (ny > 0)

    return nx, ny, valid, sl1, sl2


def get_common_slices(a_origin, a_shape, b_origin, b_shape):
    """
    Get slices of overlaps between two rectangular grids
    
    Parameters
    ----------
    a_origin : tuple
        The origin coordinates of grid A.
    a_shape : tuple
        The shape of grid A.
    b_origin : tuple
        The origin coordinates of grid B.
    b_shape : tuple
        The shape of grid B.
    
    Returns
    -------
    a_slice : tuple
        The slices of grid A that overlap with grid B.
    b_slice : tuple
        The slices of grid B that overlap with grid A.
    
    """

    ll = np.min([a_origin, b_origin], axis=0)
    ur = np.max([a_origin + a_shape, b_origin + b_shape], axis=0)

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
            """
            Initialize a WCSObject.

            Parameters
            ----------
            wcs : `pywcs.WCS` or str or `pyfits.HDUList`
                The WCS object or the path to a FITS file or an HDUList object.

            ext : int, optional
                The extension number to use when reading from a FITS file. 
                Default is 1.

            label : str, optional
                A label for the WCS object. Default is None.

            Attributes
            ----------
            wcs : `pywcs.WCS`
                The WCS object.

            fp : numpy.ndarray
                The footprint of the WCS object.

            cosdec : float
                The cosine of the declination of the first point in the footprint.

            label : str or None
                The label for the WCS object.

            pixel_scale : float
                The pixel scale of the WCS object.

            Methods
            -------
            add_naxis(header)
                Add the NAXIS information from the FITS header to the WCS object.

            """

            if isinstance(wcs, pywcs.WCS):
                self.wcs = wcs.deepcopy()
                if not hasattr(self.wcs, "pixel_shape"):
                    self.wcs.pixel_shape = None

                if self.wcs.pixel_shape is None:
                    self.wcs.pixel_shape = [int(p * 2) for p in self.wcs.wcs.crpix]
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
                print("WCS class not recognized: {0}".format(wcs.__class__))
                raise ValueError

            self.fp = self.wcs.calc_footprint()
            self.cosdec = np.cos(self.fp[0, 1] / 180 * np.pi)
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
        return "polygon({0})".format(
            ",".join(["{0:.6f}".format(c) for c in self.fp.flatten()])
        )

    @staticmethod
    def add_naxis(header):
        """
        If NAXIS keywords not found in an image header, assume the parent
        image dimensions are 2*CRPIX

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            FITS header object.
        
        """
        for i in [1, 2]:
            if "NAXIS{0}".format(i) not in header:
                header["NAXIS{0}".format(i)] = int(header["CRPIX{0}".format(i)] * 2)


def reproject_faster(input_hdu, output, pad=10, **kwargs):
    """
    Speed up `reproject` module with array slices of the input image

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

    if "SIP" in out_wcs.wcs.ctype[0]:
        print("Warning: `reproject` doesn't appear to support SIP projection")

    # Compute pixel coordinates of the output frame corners in the input image
    input_wcs = pywcs.WCS(input_hdu.header, relax=True)
    out_fp = out_wcs.calc_footprint()
    input_xy = input_wcs.all_world2pix(out_fp, 0)
    slx = slice(int(input_xy[:, 0].min()) - pad, int(input_xy[:, 0].max()) + pad)
    sly = slice(int(input_xy[:, 1].min()) - pad, int(input_xy[:, 1].max()) + pad)

    # Make the cutout
    sub_data = input_hdu.data[sly, slx]
    sub_header = get_wcs_slice_header(input_wcs, slx, sly)
    sub_hdu = pyfits.PrimaryHDU(data=sub_data, header=sub_header)

    # Get the reprojection
    seg_i, fp_i = reproject.reproject_interp(sub_hdu, output, **kwargs)
    return seg_i.astype(sub_data.dtype), fp_i.astype(np.uint8)


def full_spectrum_wcsheader(center_wave=1.4e4, dlam=40, NX=100, spatial_scale=1, NY=10):
    """
    Make a WCS header for a 2D spectrum

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

    h = pyfits.ImageHDU(data=np.zeros((2 * NY, 2 * NX), dtype=np.float32))

    refh = h.header
    refh["CRPIX1"] = NX + 1
    refh["CRPIX2"] = NY + 1
    refh["CRVAL1"] = center_wave / 1.0e4
    refh["CD1_1"] = dlam / 1.0e4
    refh["CD1_2"] = 0.0
    refh["CRVAL2"] = 0.0
    refh["CD2_2"] = spatial_scale
    refh["CD2_1"] = 0.0
    refh["RADESYS"] = ""

    refh["CTYPE1"] = "RA---TAN-SIP"
    refh["CUNIT1"] = "mas"
    refh["CTYPE2"] = "DEC--TAN-SIP"
    refh["CUNIT2"] = "mas"

    ref_wcs = pywcs.WCS(refh)
    ref_wcs.pscale = get_wcs_pscale(ref_wcs)

    return refh, ref_wcs


def make_spectrum_wcsheader(center_wave=1.4e4, dlam=40, NX=100, spatial_scale=1, NY=10):
    """
    Make a WCS header for a 2D spectrum

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

    h = pyfits.ImageHDU(data=np.zeros((2 * NY, 2 * NX), dtype=np.float32))

    refh = h.header
    refh["CRPIX1"] = NX + 1
    refh["CRPIX2"] = NY + 1
    refh["CRVAL1"] = center_wave
    refh["CD1_1"] = dlam
    refh["CD1_2"] = 0.0
    refh["CRVAL2"] = 0.0
    refh["CD2_2"] = spatial_scale
    refh["CD2_1"] = 0.0
    refh["RADESYS"] = ""

    refh["CTYPE1"] = "WAVE"
    refh["CTYPE2"] = "LINEAR"

    ref_wcs = pywcs.WCS(h.header)
    ref_wcs.pscale = (
        np.sqrt(ref_wcs.wcs.cd[0, 0] ** 2 + ref_wcs.wcs.cd[1, 0] ** 2) * 3600.0
    )

    return refh, ref_wcs


def read_gzipped_header(
    file="test.fits.gz", BLOCK=1024, NMAX=256, nspace=16, strip=False
):
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

    f = gzip.GzipFile(fileobj=open(file, "rb"))

    data = b""
    end = b" END" + b" " * nspace

    for i in range(NMAX):
        data_i = f.read(BLOCK)
        if end in data_i:
            break

        data += data_i

    if i == NMAX - 1:
        print(
            'Error: END+{3}*" " not found in first {0}x{1} bytes of {2})'.format(
                NMAX, BLOCK, file, nspace
            )
        )
        f.close()
        return {}

    ix = data_i.index(end)
    data += data_i[:ix] + end  # data_i[:ix]

    f.close()
    data_str = data.decode("utf8")
    h = pyfits.Header.fromstring(data_str)

    if strip:
        return strip_header_keys(h, usewcs=True)
    else:
        return h


DRIZZLE_KEYS = [
    "GEOM",
    "DATA",
    "DEXP",
    "OUDA",
    "OUWE",
    "OUCO",
    "MASK",
    "WTSC",
    "KERN",
    "PIXF",
    "COEF",
    "OUUN",
    "FVAL",
    "WKEY",
    "SCAL",
    "ISCL",
]


def strip_header_keys(
    header,
    comment=True,
    history=True,
    drizzle_keys=DRIZZLE_KEYS,
    usewcs=False,
    keep_with_wcs=[
        "EXPTIME",
        "FILTER",
        "TELESCOP",
        "INSTRUME",
        "DATE-OBS",
        "EXPSTART",
        "EXPEND",
    ],
):
    """
    Strip header keywords

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Header object to be stripped.


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

        if "FILTER" in keep_with_wcs:
            try:
                h["FILTER"] = (
                    parse_filter_from_header(header),
                    "element selected from filter wheel",
                )
            except:
                pass

        return h

    h = copy.deepcopy(header)
    keys = list(h.keys())
    strip_keys = []
    if comment:
        strip_keys.append("COMMENT")

    if history:
        strip_keys.append("HISTORY")

    for k in keys:
        if k in strip_keys:
            h.remove(k)

        if drizzle_keys:
            if k.startswith("D"):
                if (k[-4:] in drizzle_keys) | k.endswith("VER"):
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

    if ("SIPCRPX1" in header) & hasattr(wcs, "sip"):
        wcs.sip.crpix[0] = header["SIPCRPX1"]
        wcs.sip.crpix[1] = header["SIPCRPX2"]
    elif ("SIAF_XREF_SCI" in header) & hasattr(wcs, "sip"):
        wcs.sip.crpix[0] = header["SIAF_XREF_SCI"]
        wcs.sip.crpix[1] = header["SIAF_YREF_SCI"]

    return wcs


def to_header(wcs, add_naxis=True, relax=True, key=None):
    """
    Modify `astropy.wcs.WCS.to_header` to produce more keywords

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
        if hasattr(wcs, "pixel_shape"):
            header["NAXIS"] = wcs.naxis
            if wcs.pixel_shape is not None:
                header["NAXIS1"] = wcs.pixel_shape[0]
                header["NAXIS2"] = wcs.pixel_shape[1]

        elif hasattr(wcs, "_naxis1"):
            header["NAXIS"] = wcs.naxis
            header["NAXIS1"] = wcs._naxis1
            header["NAXIS2"] = wcs._naxis2

    for k in header:
        if k.startswith("PC"):
            cd = k.replace("PC", "CD")
            header.rename_keyword(k, cd)

    if hasattr(wcs.wcs, "cd"):
        for i in [0, 1]:
            for j in [0, 1]:
                header[f"CD{i+1}_{j+1}"] = wcs.wcs.cd[i][j]

    if hasattr(wcs, "sip"):
        if hasattr(wcs.sip, "crpix"):
            header["SIPCRPX1"], header["SIPCRPX2"] = wcs.sip.crpix

    return header


def make_wcsheader(
    ra=40.07293, dec=-1.6137748, size=2, pixscale=0.1, get_hdu=False, theta=0
):
    """
    Make a celestial WCS header

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
        cdelt = [pixscale / 3600.0] * 2
    else:
        cdelt = [pixscale[0] / 3600.0, pixscale[1] / 3600.0]

    if np.isscalar(size):
        npix = np.asarray(np.round([size/pixscale, size/pixscale]),dtype=int)
    else:
        npix = np.asarray(np.round([size[0]/pixscale, size[1]/pixscale]),dtype=int)

    hout = pyfits.Header()
    hout["CRPIX1"] = (npix[0] - 1) / 2 + 1
    hout["CRPIX2"] = (npix[1] - 1) / 2 + 1
    hout["CRVAL1"] = ra
    hout["CRVAL2"] = dec
    hout["CD1_1"] = -cdelt[0]
    hout["CD1_2"] = hout["CD2_1"] = 0.0
    hout["CD2_2"] = cdelt[1]
    hout["NAXIS1"] = npix[0]
    hout["NAXIS2"] = npix[1]
    hout["CTYPE1"] = "RA---TAN"
    hout["CTYPE2"] = "DEC--TAN"

    hout["RADESYS"] = "ICRS"
    hout["EQUINOX"] = 2000
    hout["LATPOLE"] = hout["CRVAL2"]
    hout["LONPOLE"] = 180

    hout["PIXASEC"] = pixscale, "Pixel scale in arcsec"

    wcs_out = pywcs.WCS(hout)

    theta_rad = np.deg2rad(theta)
    mat = np.array(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ]
    )

    rot_cd = np.dot(mat, wcs_out.wcs.cd)

    for i in [0, 1]:
        for j in [0, 1]:
            hout["CD{0:d}_{1:d}".format(i + 1, j + 1)] = rot_cd[i, j]
            wcs_out.wcs.cd[i, j] = rot_cd[i, j]

    cd = wcs_out.wcs.cd
    wcs_out.pscale = get_wcs_pscale(wcs_out)  # np.sqrt((cd[0,:]**2).sum())*3600.

    if get_hdu:
        hdu = pyfits.ImageHDU(
            header=hout, data=np.zeros((npix[1], npix[0]), dtype=np.float32)
        )
        return hdu
    else:
        return hout, wcs_out


def get_flt_footprint(flt_file, extensions=[1, 2, 3, 4], patch_args=None):
    """
    Compute footprint of all SCI extensions of an HST exposure

    Parameters
    ----------
    flt_file : str
        Path to the FITS file.

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
        if ("SCI", ext) not in im:
            continue

        wcs = pywcs.WCS(im["SCI", ext].header, fobj=im)
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


def make_maximal_wcs(
    files,
    pixel_scale=None,
    get_hdu=True,
    pad=90,
    verbose=True,
    theta=0,
    poly_buffer=1.0 / 3600,
    nsci_extensions=4,
):
    """
    Compute an ImageHDU with a footprint that covers all of ``files``

    Parameters
    ----------
    files : list
        List of HST FITS files (e.g., FLT.) or WCS objects.

    pixel_scale : float, optional
        Pixel scale of output WCS, in `~astropy.units.arcsec`.  If `None`,
        get pixel scale of first file in `files`.

    get_hdu : bool, optional
        If True, return an `~astropy.io.fits.ImageHDU` object. If False, return
        a tuple of `~astropy.io.fits.Header` and `~astropy.wcs.WCS`.

    pad : float, optional
        Padding to add to the total image size, in `~astropy.units.arcsec`.

    theta : float, optional
        Position angle, in degrees.

    poly_buffer : float, optional
        Buffer size to apply to the footprint polygon, in degrees.

    nsci_extensions : int, optional
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

    group_poly = None

    if hasattr(files, "buffer"):
        # Input is a shapely object
        group_poly = files
    elif isinstance(files[0], pywcs.WCS):
        # Already wcs_list
        wcs_list = [(wcs, "WCS", -1) for wcs in files]
    else:
        wcs_list = []
        for i, file in enumerate(files):
            if not os.path.exists(file):
                continue

            with pyfits.open(file) as im:
                for ext in range(nsci_extensions):
                    if ("SCI", ext + 1) not in im:
                        continue

                    wcs = pywcs.WCS(im["SCI", ext + 1].header, fobj=im)
                    wcs_list.append((wcs, file, ext))

    if pixel_scale is None:
        pixel_scale = get_wcs_pscale(wcs_list[0][0])

    if group_poly is None:
        for i, (wcs, file, chip) in enumerate(wcs_list):
            p_i = Polygon(wcs.calc_footprint())
            if group_poly is None:
                if poly_buffer > 0:
                    group_poly = p_i.buffer(1.0 / 3600)
                else:
                    group_poly = p_i
            else:
                if poly_buffer > 0:
                    group_poly = group_poly.union(p_i.buffer(1.0 / 3600))
                else:
                    group_poly = group_poly.union(p_i)

                 
            x0, y0 = np.asarray(group_poly.centroid.xy,dtype=float)[:, 0]
            
            if verbose:
                msg = "{0:>3d}/{1:>3d}: {2}[SCI,{3}]  {4:>6.2f}"
                print(
                    msg.format(
                        i,
                        len(files),
                        file,
                        chip + 1,
                        group_poly.area * 3600 * np.cos(y0 / 180 * np.pi),
                    )
                )


    px = np.asarray(group_poly.convex_hull.boundary.xy,dtype=float).T
    #x0, y0 = np.asarray(group_poly.centroid.xy,dtype=float)[:,0]

    x0 = (px.max(axis=0) + px.min(axis=0)) / 2.0

    cosd = np.array([np.cos(x0[1] / 180 * np.pi), 1])

    _mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Rotated
    pr = ((px - x0) * cosd).dot(_mat) / cosd + x0

    size_arcsec = (pr.max(axis=0) - pr.min(axis=0)) * cosd * 3600
    sx, sy = size_arcsec

    # sx = (px.max()-px.min())*cosd*3600 # arcsec
    # sy = (py.max()-py.min())*3600 # arcsec

    size = np.maximum(sx + pad, sy + pad)

    if verbose:
        msg = "\n  Mosaic WCS: ({0:.5f},{1:.5f}) "
        msg += "{2:.1f}'x{3:.1f}'  {4:.3f}\"/pix\n"
        print(
            msg.format(x0[0], x0[1], (sx + pad) / 60.0, (sy + pad) / 60.0, pixel_scale)
        )

    out = make_wcsheader(
        ra=x0[0],
        dec=x0[1],
        size=(sx + pad * 2, sy + pad * 2),
        pixscale=pixel_scale,
        get_hdu=get_hdu,
        theta=theta / np.pi * 180,
    )

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

    for k in ["NAXIS1", "NAXIS2"]:  # , 'CRPIX1', 'CRPIX2']:
        h[k] *= 2

    for k in ["CRPIX1", "CRPIX2"]:
        h[k] = h[k] * 2 - 0.5

    for k in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]:
        if k in h:
            h[k] /= 2

    if 0:
        # Test
        new = pywcs.WCS(h)
        sh = new.pixel_shape

        wcorner = wcs.all_world2pix(
            new.all_pix2world([[-0.5, -0.5], [sh[0] - 0.5, sh[1] - 0.5]], 0), 0
        )
        print("small > large")
        print(", ".join([f"{w:.2f}" for w in wcorner[0]]))
        print(", ".join([f"{w:.2f}" for w in wcorner[1]]), wcs.pixel_shape)

        sh = wcs.pixel_shape
        wcorner = new.all_world2pix(
            wcs.all_pix2world([[-0.5, -0.5], [sh[0] - 0.5, sh[1] - 0.5]], 0), 0
        )
        print("large > small")
        print(", ".join([f"{w:.2f}" for w in wcorner[0]]))
        print(", ".join([f"{w:.2f}" for w in wcorner[1]]), new.pixel_shape)

    new_wcs = pywcs.WCS(h, relax=True)

    return new_wcs


def header_keys_from_filelist(fits_files, keywords=[], ext=0, colname_case=str.lower):
    """
    Dump header keywords to a `~astropy.table.Table`

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
        keywords.pop(keywords.index(""))
        keywords.pop(keywords.index("HISTORY"))

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
    table_header = [colname_case(key) for key in ["file"] + keywords]

    # Output table
    tab = Table(data=np.array(lines), names=table_header)

    return tab


def parse_s3_url(url="s3://bucket/path/to/file.txt"):
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
    surl = url.strip("s3://")
    spl = surl.split("/")
    if len(spl) < 2:
        print(f"bucket / path not found in {url}")
        return None, None, None

    bucket_name = spl[0]
    s3_object = "/".join(spl[1:])
    filename = os.path.basename(s3_object)
    return bucket_name, s3_object, filename


def fetch_s3_url(
    url="s3://bucket/path/to/file.txt",
    file_func=lambda x: os.path.join("./", x),
    skip_existing=True,
    verbose=True,
):
    """
    Fetch file from an S3 bucket

    Parameters
    ----------
    url : str
        S3 url of a file to download

    file_func : function
        Function applied to the file name extracted from `url`, e.g., to
        set output directory, rename files, set a prefix, etc.

    skip_existing : bool, optional
        If True, skip downloading if the local file already exists.
        Default is True.

    verbose : bool, optional
        If True, print download progress and status messages.
        Default is True.

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

    s3 = boto3.resource("s3")
    bucket_name, s3_object, filename = parse_s3_url(url=url)
    if bucket_name is None:
        return url, os.path.exists(url)

    bkt = s3.Bucket(bucket_name)
    local_file = file_func(filename)
    status = os.path.exists(local_file) * 1

    if (status > 0) & skip_existing:
        print(f"{local_file} exists, skipping.")
    else:

        try:
            bkt.download_file(
                s3_object, local_file, ExtraArgs={"RequestPayer": "requester"}
            )
            status += 2
            if verbose:
                print(f"{url} > {local_file}")

        except botocore.exceptions.ClientError:
            trace = traceback.format_exc(limit=2)
            msg = trace.split("\n")[-2].split("ClientError: ")[1]
            if verbose:
                print(f"Failed {url}: {msg}")

            # Download failed due to a ClientError
            # Forbidden probably means insufficient bucket access privileges
            pass

    return local_file, status


def niriss_ghost_mask(
    im,
    init_thresh=0.05,
    init_sigma=3,
    final_thresh=0.01,
    final_sigma=3,
    erosions=0,
    dilations=9,
    verbose=True,
    **kwargs,
):
    """
    Make a mask for NIRISS imaging ghosts

    See also Martel. JWST-STScI-004877 and
    https://github.com/spacetelescope/niriss_ghost

    Parameters
    ----------
    im : `~astropy.io.fits.HDUList`
        Input image HDUList.

    init_thresh : float, optional
        Initial threshold for detecting ghost pixels. Default is 0.05.

    init_sigma : float, optional
        Initial sigma threshold for detecting ghost pixels. Default is 3.

    final_thresh : float, optional
        Final threshold for detecting ghost pixels. Default is 0.01.

    final_sigma : float, optional
        Final sigma threshold for detecting ghost pixels. Default is 3.

    erosions : int, optional
        Number of binary erosions to apply to the ghost mask. Default is 0.

    dilations : int, optional
        Number of binary dilations to apply to the ghost mask. Default is 9.

    verbose : bool, optional
        If True, print diagnostic messages. Default is True.

    Returns
    -------
    ghost_mask : `~numpy.ndarray`
        Boolean array indicating the positions of the ghost pixels.

    """
    import scipy.ndimage as nd

    if im[0].header["PUPIL"] not in ["F115W", "F150W", "F200W"]:
        return False

    if im[0].header["PUPIL"] == "F115W":
        xgap, ygap = 1156, 927
    elif im[0].header["PUPIL"] == "F115W":
        xgap, ygap = 1162, 938
    else:
        xgap, ygap = 1156, 944 - 2

    yp, xp = np.indices((2048, 2048))

    yg = 2 * (ygap - 1) - yp
    xg = 2 * (xgap - 1) - xp

    dx = xp - xgap
    dy = yp - ygap

    in_img = (xg >= 0) & (xg < 2048)
    in_img &= (yg >= 0) & (yg < 2048)

    in_img &= np.abs(dx) < 400
    in_img &= np.abs(dy) < 400

    if "MDRIZSKY" in im["SCI"].header:
        bkg = im["SCI"].header["MDRIZSKY"]
    else:
        bkg = np.nanmedian(im["SCI"].data[im["DQ"].data == 0])

    thresh = (im["SCI"].data - bkg) * init_thresh > init_sigma * im["ERR"].data
    thresh &= in_img

    _reflected = np.zeros_like(im["SCI"].data)
    for xpi, ypi, xgi, ygi in zip(xp[thresh], yp[thresh], xg[thresh], yg[thresh]):

        _reflected[ygi, xgi] = im["SCI"].data[ypi, xpi] - bkg

    ghost_mask = _reflected * final_thresh > final_sigma * im["ERR"].data

    if erosions > 0:
        ghost_mask = nd.binary_erosion(ghost_mask, iterations=erosions)

    ghost_mask = nd.binary_dilation(ghost_mask, iterations=dilations)

    im[0].header["GHOSTMSK"] = True, "NIRISS ghost mask applied"
    im[0].header["GHOSTNPX"] = ghost_mask.sum(), "Pixels in NIRISS ghost mask"

    msg = "NIRISS ghost mask {0} Npix: {1}\n".format(
        im[0].header["PUPIL"], ghost_mask.sum()
    )
    log_comment(LOGFILE, msg, verbose=verbose)

    return ghost_mask


def get_photom_scale(header, verbose=True):
    """
    Get tabulated scale factor

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Image header

    verbose : bool, optional
        Whether to display verbose output. Default is True.

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

    if "TELESCOP" in header:
        if header["TELESCOP"] not in ["JWST"]:
            msg = f"get_photom_scale: TELESCOP={header['TELESCOP']} is not 'JWST'"
            log_comment(LOGFILE, msg, verbose=verbose)
            return header["TELESCOP"], 1.0
    else:
        return None, 1.0

    corr_file = os.path.join(os.path.dirname(__file__), "data/photom_correction.yml")

    if not os.path.exists(corr_file):
        msg = f"{corr_file} not found."
        log_comment(LOGFILE, msg, verbose=verbose)
        return None, 1

    with open(corr_file) as fp:
        corr = yaml.load(fp, Loader=yaml.SafeLoader)

    if "CRDS_CTX" in header:
        if header["CRDS_CTX"] > corr["CRDS_CTX_MAX"]:
            msg = f"get_photom_scale {corr_file}: {header['CRDS_CTX']} > {corr['CRDS_CTX_MAX']}"
            log_comment(LOGFILE, msg, verbose=verbose)
            return header["CRDS_CTX"], 1.0

    key = "{0}-{1}".format(header["DETECTOR"], header["FILTER"])
    if "PUPIL" in header:
        key += "-{0}".format(header["PUPIL"])

    if key not in corr:
        msg = f"get_photom_scale {corr_file}: {key} not found"
        log_comment(LOGFILE, msg, verbose=verbose)

        return key, 1.0

    else:
        msg = f"get_photom_scale {corr_file}: Scale {key} by {1./corr[key]:.3f}"
        log_comment(LOGFILE, msg, verbose=verbose)

        return key, 1.0 / corr[key]


def jwst_crds_photom_scale(hdul, context="jwst_1293.pmap", scale_internal=True, update=True, verbose=False):
    """
    Scale factors between different JWST CRDS_CONTEXT

    Parameters
    ----------
    hdul : `astropy.io.fits.HDUList`
        Exposure file HDUList, which has header keywords like CRDS_CTX, PHOTMJSR, etc.

    context : str
        Target CRDS context version

    scale_internal : bool
        Include internal correction from ``data/jwst_zeropoints.yml``

    update : bool
        Scale photometry header keywords by the ratio of NEW_PHOTMJSR / OLD_PHOTMJSR

    verbose: bool
        Messaging

    Returns
    -------
    scale : float
        Relative photometry scaling factor NEW_PHOTMJSR / OLD_PHOTMJSR.  Defaults to
        1.0 if not a JWST instrument or if certain necessary header keywords not found

    """
    try:
        from .jwst_utils import get_crds_zeropoint, get_jwst_filter_info
        from .jwst_utils import get_nircam_zeropoint_update
    except ImportError:
        print(
            "jwst_crds_photom_scale: failed to import grizli.jwst_utils.get_crds_zeropoint"
        )
        return 1.0

    if "INSTRUME" not in hdul[0].header:
        return 1.0

    instrument = hdul[0].header["INSTRUME"]

    if instrument not in ["NIRCAM", "MIRI", "NIRISS"]:
        return 1.0

    mode = {"context": context, "verbose": False, "instrument": instrument}
    for k in ["FILTER", "PUPIL", "DETECTOR"]:
        if k in hdul[0].header:
            mode[k.lower()] = hdul[0].header[k]

    ref_ctx, r_photom, ref_photmjsr, ref_pixar_sr = get_crds_zeropoint(**mode)
    if ref_photmjsr is None:
        return 1.0

    if "PHOTMJSR" not in hdul["SCI"].header:
        return 1.0

    if scale_internal:
        key, _mjsr, _scale, _pixar_sr = get_nircam_zeropoint_update(
            header=hdul[0].header,
            verbose=verbose
        )
        if _mjsr is not None:
            ref_photmjsr = _mjsr * _scale

    old_photmjsr = hdul["SCI"].header["PHOTMJSR"]
    scale = ref_photmjsr / old_photmjsr

    msg = f"jwst_crds_photom_scale: {context} photmjsr old, new = "
    msg += f"{old_photmjsr:.3f}  {ref_photmjsr:.3f}  scale = {scale:.3f}"

    if update:

        # Check image units
        if "OBUNIT" in hdul["SCI"].header:
            unit_key = "OBUNIT"
        else:
            unit_key = "BUNIT"

        if hdul["SCI"].header[unit_key].upper() == "MJy/sr".upper():
            # Image was already scaled (cal), so use scale factor
            to_mjysr = scale
        else:
            # Image wasn't scaled, so just use mjsr
            to_mjysr = ref_photmjsr

        # Recalculate PHOTFNU, PHOTFLAM from PIXAR_SR, which could also change
        if ref_pixar_sr is None:
            ref_pixar_sr = hdul["SCI"].header["PIXAR_SR"]

        photfnu = to_mjysr * ref_pixar_sr * 1.0e6

        filter_info = get_jwst_filter_info(hdul[0].header)
        if filter_info is not None:
            plam = filter_info["pivot"] * 1.0e4
        else:
            plam = 5.0e4

        photflam = photfnu * 2.99e-5 / plam ** 2

        for e in [0, "SCI"]:

            hdul[e].header["PIXAR_SR"] = ref_pixar_sr
            hdul[e].header["PHOTFLAM"] = photflam
            hdul[e].header["PHOTFNU"] = photfnu

            for k in ["TO_MJYSR", "PHOTMJSR"]:
                if k in hdul[e].header:
                    hdul[e].header[k] *= scale

            if "ZP" in hdul[e].header:
                hdul[e].header["ZP"] -= 2.5 * np.log10(scale)

            hdul[e].header["CRDS_CTX"] = context
            hdul[e].header["R_PHOTOM"] = os.path.basename(r_photom)

    log_comment(LOGFILE, msg, verbose=verbose)

    return scale


DEFAULT_SNOWBLIND_KWARGS = dict(
    require_prefix="jw",
    max_fraction=0.3,
    new_jump_flag=1024,
    min_radius=4,
    growth_factor=1.5,
    unset_first=True,
)


def jwst_snowblind_mask(
    rate_file,
    require_prefix="jw",
    max_fraction=0.3,
    new_jump_flag=1024,
    min_radius=4,
    growth_factor=1.5,
    unset_first=True,
    verbose=True,
    skip_after_cal_version='1.14',
    **kwargs,
):
    """
    Update JWST DQ mask with `snowblind`.  See
    https://github.com/mpi-astronomy/snowblind.

    Requires ``snowblind > 0.1.2``, which currently is just in the fork at
    https://github.com/gbrammer/snowblind.

    Parameters
    ----------
    rate_file : str
        Filename of a ``rate.fits`` exposure

    require_prefix : str
        Only run if ``rate_file.startswith(require_prefix)``

    max_fraction : float
        Maximum allowed fraction of flagged pixels relative to the total

    new_jump_flag : int
        Integer DQ flag of identified snowballs

    min_radius : int
        Minimum radius of ``JUMP_DET`` flagged groups of pixels

    growth_factor : float
        Scale factor of the DQ mask

    unset_first : bool
        Unset the `new_jump_flag` bit of the DQ array before processing

    verbose : bool
        Whether to print verbose output

    kwargs : dict
        Additional keyword arguments to be passed to `snowblind.SnowblindStep.call()`

    Returns
    -------
    dq : array-like
        Image array with values ``new_jump_flag`` with identified snowballs

    mask_frac : float
        Fraction of masked pixels

    """
    import jwst.datamodels
    from packaging.version import Version

    from . import jwst_utils

    if not os.path.basename(rate_file).startswith(require_prefix):
        return None, None

    try:
        from snowblind import snowblind
        from snowblind import __version__ as snowblind_version
    except ImportError:
        return None, None

    if snowblind_version <= "0.1.2":
        msg = (
            "ImportError: snowblind > 0.1.2 required, get it from the fork at "
            "https://github.com/gbrammer/snowblind if not yet available on the "
            "main repository at https://github.com/mpi-astronomy/snowblind"
        )

        log_comment(LOGFILE, msg, verbose=True)
        return None, None

    step = snowblind.SnowblindStep

    # Do we need to reset header keywords?
    reset_header = False
    with pyfits.open(rate_file) as im:
        reset_header = "OINSTRUM" in im[0].header
        reset_header &= im[0].header["INSTRUME"] == "WFC3"

        if "CAL_VER" in im[0].header:
            im_cal_ver = im[0].header["CAL_VER"]
            if Version(im_cal_ver) >= Version(skip_after_cal_version):
                msg = f"mask_snowballs: {rate_file}  "
                msg += f"{im_cal_ver} > {skip_after_cal_version}, skip"
                log_comment(LOGFILE, msg, verbose=True)
                return np.zeros(im["SCI"].data.shape, dtype=int), 0.0

    if reset_header:
        _ = jwst_utils.set_jwst_to_hst_keywords(rate_file, reset=True, verbose=False)

    with jwst.datamodels.open(rate_file) as dm:
        if unset_first:
            dm.dq -= dm.dq & new_jump_flag

        res = step.call(
            dm,
            save_results=False,
            new_jump_flag=new_jump_flag,
            min_radius=min_radius,
            growth_factor=growth_factor,
            **kwargs,
        )

    if reset_header:
        _ = jwst_utils.set_jwst_to_hst_keywords(rate_file, reset=False, verbose=False)

    _mask_frac = ((res.dq & new_jump_flag) > 0).sum() / res.dq.size

    if _mask_frac > max_fraction:
        msg = f"grizli.utils.jwst_snowblind_mask: {rate_file} problem "
        msg += f" fraction {_mask_frac*100:.2f} > {max_fraction*100:.2f}"
        msg += " turning off..."
        res.dq &= 0
    else:
        msg = f"grizli.utils.jwst_snowblind_mask: {rate_file} {_mask_frac*100:.2f}"
        msg += f" masked with DQ={new_jump_flag}"

    log_comment(LOGFILE, msg, verbose=verbose)

    return (res.dq & new_jump_flag), _mask_frac


def drizzle_from_visit(
    visit,
    output=None,
    pixfrac=1.0,
    kernel="point",
    clean=True,
    include_saturated=True,
    keep_bits=None,
    dryrun=False,
    skip=None,
    extra_wfc3ir_badpix=True,
    verbose=True,
    scale_photom=True,
    internal_nircam_zeropoints=True,
    context="jwst_1293.pmap",
    weight_type="jwst_var",
    rnoise_percentile=99,
    calc_wcsmap=False,
    with_slices=False,
    niriss_ghost_kwargs={},
    use_background_extension=True,
    snowblind_kwargs=None,
    jwst_dq_flags=JWST_DQ_FLAGS,
    nircam_hot_pixel_kwargs={},
    niriss_hot_pixel_kwargs=None, # {'hot_threshold': 7, 'plus_sn_min': 3},
    get_dbmask=True,
    saturated_lookback=1e4,
    write_sat_file=False,
    sat_kwargs={},
    query_persistence_pixels=True,
    **kwargs,
):
    """
    Make drizzle mosaic from exposures in a visit dictionary

    Parameters
    ----------
    visit : dict
        Visit dictionary with 'product' and 'files' keys

    output : `~astropy.wcs.WCS`, `~astropy.io.fits.Header`, `~astropy.io.ImageHDU`
        Output frame definition.  Can be a WCS object, header, or FITS HDU.  If
        None, then generates a WCS with `grizli.utils.make_maximal_wcs`

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

    context : str
        JWST calibration context to use for photometric scaling

    weight_type : 'err', 'median_err', 'time', 'jwst', 'jwst_var', 'median_variance'
        Exposure weighting strategy.

        - The default 'err' strategy uses the full uncertainty array defined in the
          `ERR` image extensions.  The alternative 

        - The 'median_err' strategy uses the median of the `ERR` extension

        - The 'time' strategy weights 'median_err' by the `TIME` extension, if
          available

        - For the 'jwst' strategy, if 'VAR_POISSON' and 'VAR_RNOISE' extensions found,
          weight by VAR_RNOISE + median(VAR_POISSON).  Fall back to 'median_err'
          otherwise.

        - For 'jwst_var', use the *weight* as in ``weight_type='jwst'`` but also
          make a full variance map propagated from the ``ERR`` noise model.

    rnoise_percentile : float
        Percentile defining the upper limit of valid `VAR_RNOISE` values, if that
        extension is found in the exposure files(e.g., for JWST)

    calc_wcsmap : bool
        Calculate and return the WCS map

    get_dbmask : bool
        Get the bad pixel mask from the database

    niriss_ghost_kwargs : dict
        Keyword arguments for `~grizli.utils.niriss_ghost_mask`

    snowblind_kwargs : dict
        Arguments to pass to `~grizli.utils.jwst_snowblind_mask` if `snowblind` hasn't
        already been run on JWST exposures

    jwst_dq_flags : list
        List of JWST flag names to include in the bad pixel mask.  To ignore, set to
        ``None``

    nircam_hot_pixel_kwargs : dict
        Keyword arguments for `grizli.jwst_utils.flag_nircam_hot_pixels`.  Set to
        ``None`` to disable and use the static bad pixel tables.

    niriss_hot_pixel_kwargs : dict
        Keyword arguments for `grizli.jwst_utils.flag_nircam_hot_pixels` when running
        on NIRISS exposures. Set to ``None`` to disable and use the static bad pixel
        tables.

    saturated_lookback : float
        Time, in seconds, to look for saturated pixels in previous exposures
        that can cause persistence.  Skip if ``saturated_lookback <= 0``.

    write_sat_file : bool
        Write persistence saturation tables

    sat_kwargs : dict
        keyword arguments to `~grizli.jwst_utils.get_saturated_pixels`

    query_persistence_pixels : bool
        Also try to query the full saturated pixel history from the DB with
        ``saturated_lookback``

    Returns
    -------
    outsci : array-like
        SCI array

    outwht : array-like
        Inverse variance WHT array

    outvar : array-like
        Optional variance array, if the input weights are not explicitly inverse
        variance

    header : `~astropy.io.fits.Header`
        Image header

    flist : list
        List of files that were drizzled to the mosaic

    wcs_tab : `~astropy.table.Table`
        Table of WCS parameters of individual exposures

    """
    from shapely.geometry import Polygon

    import scipy.ndimage as nd
    from astropy.io.fits import PrimaryHDU, ImageHDU

    from .prep import apply_region_mask_from_db
    from .version import __version__ as grizli__version
    from .jwst_utils import get_jwst_dq_bit, flag_nircam_hot_pixels
    from .jwst_utils import get_saturated_pixel_table, query_persistence

    bucket_name = None

    try:
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.resource("s3")
        s3_client = boto3.client("s3")

    except ImportError:
        s3 = None
        ClientError = None
        s3_client = None

    _valid_weight_type = [
        "err", "median_err", "time", "jwst", "jwst_var", "median_variance",
    ]
    if weight_type not in _valid_weight_type:
        print(f"WARNING: weight_type '{weight_type}' must be 'err', 'median_err', ")
        print(f"         'jwst', 'median_variance', or 'time'; falling back to 'err'.")
        weight_type = "err"

    if isinstance(output, pywcs.WCS):
        outputwcs = output
    elif isinstance(output, pyfits.Header):
        outputwcs = pywcs.WCS(output)
    elif isinstance(output, PrimaryHDU) | isinstance(output, ImageHDU):
        outputwcs = pywcs.WCS(output.header)
    elif output is None:
        _hdu = make_maximal_wcs(
            files=visit["files"], pixel_scale=None, get_hdu=True, verbose=False, pad=4
        )
        outputwcs = pywcs.WCS(_hdu.header)
    else:
        return None

    if not hasattr(outputwcs, "_naxis1"):
        outputwcs._naxis1, outputwcs._naxis2 = outputwcs._naxis

    outputwcs.pscale = get_wcs_pscale(outputwcs)

    output_poly = Polygon(outputwcs.calc_footprint())

    count = 0

    ref_photflam = None

    indices = []
    for i in range(len(visit["files"])):
        if "footprints" in visit:
            if hasattr(visit["footprints"][i], "intersection"):
                olap = visit["footprints"][i].intersection(output_poly)
                if olap.area > 0:
                    indices.append(i)

            elif hasattr(visit["footprints"][i], "__len__"):
                for _fp in visit["footprints"][i]:
                    olap = _fp.intersection(output_poly)
                    if olap.area > 0:
                        indices.append(i)
                        break
            else:
                indices.append(i)
        else:
            indices.append(i)

    if skip is not None:
        indices = indices[::skip]

    NTOTAL = len(indices)

    wcs_rows = []
    wcs_colnames = None
    wcs_keys = {}

    bpdata = 0

    saturated_tables = {}

    for i in indices:

        file = visit["files"][i]

        msg = "\n({0:4d}/{1:4d}) Add exposure {2} "
        msg += "(weight_type='{3}', rnoise_percentile={4})\n"
        msg = msg.format(count + 1, NTOTAL, file, weight_type, rnoise_percentile)
        log_comment(LOGFILE, msg, verbose=verbose)

        if dryrun:
            continue

        if (not os.path.exists(file)) & (s3 is not None):
            bucket_i = visit["awspath"][i].split("/")[0]
            if bucket_name != bucket_i:
                bucket_name = bucket_i
                bkt = s3.Bucket(bucket_name)

            s3_path = "/".join(visit["awspath"][i].split("/")[1:])
            remote_file = os.path.join(s3_path, file)

            print("  (fetch from s3://{0}/{1})".format(bucket_i, remote_file))

            try:
                bkt.download_file(
                    remote_file, file, ExtraArgs={"RequestPayer": "requester"}
                )
            except ClientError:
                print("  (failed s3://{0}/{1})".format(bucket_i, remote_file))
                continue

        try:
            flt = pyfits.open(file)
        except OSError:
            print(f"open({file}) failed!")
            continue

        sci_list, wht_list, wcs_list = [], [], []

        if weight_type == "jwst_var":
            var_list = []
        else:
            var_list = None

        keys = OrderedDict()
        for k in ['EXPTIME', 'TELESCOP', 'FILTER','FILTER1', 'FILTER2', 
                  'PUPIL', 'DETECTOR', 'INSTRUME', 'PHOTFLAM', 'PHOTPLAM', 
                  'PHOTFNU', 'PHOTZPT', 'PHOTBW', 'PHOTMODE', 'EXPSTART', 
                  'EXPEND', 'DATE-OBS', 'TIME-OBS',
                  'UPDA_CTX', 'CRDS_CTX', 'R_DISTOR', 'R_PHOTOM', 'R_FLAT']:
            
            if k in flt[0].header:
                keys[k] = flt[0].header[k]

        bpdata = None
        _nsat = None

        if flt[0].header["TELESCOP"] in ["JWST"]:
            bits = 4
            include_saturated = False

            # bpdata = 0
            _inst = flt[0].header["INSTRUME"]
            _det = flt[0].header['DETECTOR']

            if (extra_wfc3ir_badpix) & (_inst in ['NIRCAM','NIRISS']):
                bpfiles = [os.path.join(os.path.dirname(__file__),
                           f'data/nrc_badpix_251016_{_det}.fits.gz')]
                bpfiles += [os.path.join(os.path.dirname(__file__),
                           f'data/nrc_badpix_240627_{_det}.fits.gz')]
                bpfiles += [os.path.join(os.path.dirname(__file__),
                           f'data/nrc_badpix_240112_{_det}.fits.gz')]
                bpfiles += [os.path.join(os.path.dirname(__file__),
                           f'data/nrc_badpix_231206_{_det}.fits.gz')]
                bpfiles += [os.path.join(os.path.dirname(__file__),
                         f'data/jwst_nircam_newhot_{_det}_extra20231129.fits.gz')]
                bpfiles += [os.path.join(os.path.dirname(__file__),
                           f'data/nrc_badpix_20230710_{_det}.fits.gz')]
                bpfiles += [os.path.join(os.path.dirname(__file__), 
                           f'data/{_det.lower()}_badpix_20241001.fits.gz')] # NIRISS
                bpfiles += [os.path.join(os.path.dirname(__file__), 
                           f'data/{_det.lower()}_badpix_20230710.fits.gz')] # NIRISS
                bpfiles += [os.path.join(os.path.dirname(__file__), 
                           f'data/nrc_badpix_230701_{_det}.fits.gz')]
                bpfiles += [os.path.join(os.path.dirname(__file__),
                           f'data/nrc_badpix_230120_{_det}.fits.gz')]
                bpfiles += [os.path.join(os.path.dirname(__file__), 
                           f'data/nrc_lowpix_0916_{_det}.fits.gz')]

                for bpfile in bpfiles:
                    if os.path.exists(bpfile):
                        bpdata = pyfits.open(bpfile)[0].data
                        if True:
                            bpdata = nd.binary_dilation(bpdata > 0) * 1024
                        else:
                            bpdata = (bpdata > 0) * 1024

                        msg = f"Use extra badpix in {bpfile}"
                        log_comment(LOGFILE, msg, verbose=verbose)
                        break

            if bpdata is None:
                bpdata = np.zeros(flt["SCI"].data.shape, dtype=int)

            # Directly flag hot pixels rather than use mask
            if (_inst in ["NIRCAM"]) & (nircam_hot_pixel_kwargs is not None):
                _sn, dq_flag, _count = flag_nircam_hot_pixels(
                    flt, **nircam_hot_pixel_kwargs
                )
                if (_count > 0) & (_count < 8192):
                    bpdata |= ((dq_flag > 0) * 1024).astype(bpdata.dtype)
                    # extra_wfc3ir_badpix = False
                else:
                    msg = f" flag_nircam_hot_pixels: {_count} out of range"
                    log_comment(LOGFILE, msg, verbose=verbose)

            elif (_inst in ["NIRISS"]) & (niriss_hot_pixel_kwargs is not None):
                _sn, dq_flag, _count = flag_nircam_hot_pixels(
                    flt, **niriss_hot_pixel_kwargs
                )
                if (_count > 0) & (_count < 8192):
                    bpdata |= ((dq_flag > 0) * 1024).astype(bpdata.dtype)
                    # extra_wfc3ir_badpix = False
                else:
                    msg = f" flag_nircam_hot_pixels: {_count} out of range (NIRISS)"
                    log_comment(LOGFILE, msg, verbose=verbose)

            if (snowblind_kwargs is not None) & (_inst in ["NIRCAM", "NIRISS"]):
                # Already processed with snowblind?
                if "SNOWBLND" in flt["SCI"].header:
                    msg = "Already processed with `snowblind`"
                    log_comment(LOGFILE, msg, verbose=verbose)
                else:
                    sdq, sfrac = jwst_snowblind_mask(file, **snowblind_kwargs)
                    if sdq is not None:
                        bpdata |= sdq

            if get_dbmask:
                dbmask = apply_region_mask_from_db(
                    os.path.basename(file), in_place=False, verbose=True
                )
                if dbmask is not None:
                    bpdata |= dbmask * 1

            # NIRISS ghost mask
            if (_inst in ["NIRISS"]) & (niriss_ghost_kwargs is not None):
                if "verbose" not in niriss_ghost_kwargs:
                    niriss_ghost_kwargs["verbose"] = verbose

                _ghost = niriss_ghost_mask(flt, **niriss_ghost_kwargs)
                bpdata |= _ghost * 1024

            # Negative
            if "MDRIZSKY" in flt["SCI"].header:
                _low = (flt["SCI"].data - flt["SCI"].header["MDRIZSKY"]) < -5 * flt[
                    "ERR"
                ].data
                msg = f"Extra -5 sigma low pixels: N= {_low.sum()} "
                msg += f" ( {_low.sum()/_low.size*100:.1} %)"
                log_comment(LOGFILE, msg, verbose=verbose)
                bpdata |= _low * 1024

            # History of saturated pixels for persistence
            if saturated_lookback > 0:

                if _det not in saturated_tables:
                    saturated_tables[_det] = {'expstart':[], 'ij':[]}

                _start = flt[0].header['EXPSTART']
                _df = get_saturated_pixel_table(
                    file=file,
                    output="df",
                    **sat_kwargs,
                )

                _sat_tab = saturated_tables[_det]
                _sat_tab["expstart"].append(_start)
                _sat_tab["ij"].append(_df)

                if write_sat_file:
                    sat_file = file.replace(".fits", ".sat.csv.gz")
                    _df.to_csv(sat_file, index=False)

                _sat_history = np.zeros_like(bpdata, dtype=bool)
                _sat_count = 0
                for _starti, _df in zip(_sat_tab["expstart"], _sat_tab["ij"]):
                    if (
                        (_starti < _start)
                        & ((_start - _starti)*86400 < saturated_lookback)
                    ):
                        _sat_history[_df.i, _df.j] |= True
                        _sat_count += 1

                _nsat = _sat_history.sum()
                bpdata |= _sat_history * 1024
                msg = (
                    f"Found {_nsat} saturated pixels in {_sat_count} "
                    f" previous {_det} exposures within {saturated_lookback:.0f} sec"
                )
                log_comment(LOGFILE, msg, verbose=verbose)

            if query_persistence_pixels & (saturated_lookback > 0):
                try:
                    _pers = query_persistence(
                                file,
                                saturated_lookback=saturated_lookback
                    )
                    if len(_pers) > 0:
                        bpdata[_pers["i"], _pers["j"]] |= 1024
                except:
                    pass

        elif flt[0].header["DETECTOR"] == "IR":
            bits = 576
            if extra_wfc3ir_badpix:
                if (i == indices[0]) | (not hasattr(bpdata, "shape")):
                    bpfile = os.path.join(
                        os.path.dirname(__file__),
                        "data/wfc3ir_badpix_spars200_22.03.31.fits.gz",
                    )
                    bpdata = pyfits.open(bpfile)[0].data

                msg = f"Use extra badpix in {bpfile}"
                log_comment(LOGFILE, msg, verbose=verbose)
        else:
            bits = 64 + 32
            bpdata = 0

        if include_saturated:
            bits |= 256

        if keep_bits is not None:
            bits |= keep_bits

        if scale_photom:
            # Scale to a particular JWST context and update header keywords
            # like PHOTFLAM, PHOTFNU
            _scale_jwst_photom = jwst_crds_photom_scale(
                flt,
                update=True,
                context=context,
                scale_internal=internal_nircam_zeropoints,
                verbose=verbose
            )

            # These might have changed
            for k in ["PHOTFLAM", "PHOTFNU", "PHOTMJSR", "ZP", "R_PHOTOM", "CRDS_CTX"]:
                if k in flt[0].header:
                    keys[k] = flt[0].header[k]

            # Additional scaling
            _key, _scale_photom = get_photom_scale(flt[0].header, verbose=verbose)

        else:
            _scale_photom = 1.0

        if "PHOTFLAM" in keys:
            msg = "  0    PHOTFLAM={0:.2e}, scale={1:.3f}"
            msg = msg.format(keys["PHOTFLAM"], _scale_photom)
            log_comment(LOGFILE, msg, verbose=verbose)

            if ref_photflam is None:
                ref_photflam = keys["PHOTFLAM"]

        median_weight = None

        for ext in [1, 2, 3, 4]:
            if ("SCI", ext) in flt:

                h = flt[("SCI", ext)].header
                if "MDRIZSKY" in h:
                    sky_value = h["MDRIZSKY"]
                else:
                    sky_value = 0

                if (("BKG", ext) in flt) & use_background_extension:
                    has_bkg = True
                    sky = flt["BKG", ext].data + sky_value
                else:
                    has_bkg = False
                    sky = sky_value

                if h["BUNIT"] == "ELECTRONS":
                    to_per_sec = 1.0 / keys["EXPTIME"]
                else:
                    to_per_sec = 1.0

                phot_scale = to_per_sec * _scale_photom

                if "PHOTFLAM" in h:
                    if ref_photflam is None:
                        ref_photflam = h["PHOTFLAM"]

                    phot_scale = h["PHOTFLAM"] / ref_photflam * _scale_photom

                    if "PHOTFNU" not in h:
                        h["PHOTFNU"] = (
                            photfnu_from_photflam(h["PHOTFLAM"], h["PHOTPLAM"]),
                            "Inverse sensitivity, Jy/DN",
                        )

                    msg = "       PHOTFLAM={0:.2e}, scale={1:.3f}"
                    msg = msg.format(h["PHOTFLAM"], phot_scale)
                    log_comment(LOGFILE, msg, verbose=verbose)

                    keys["PHOTFLAM"] = h["PHOTFLAM"]
                    for k in [
                        "PHOTFLAM",
                        "PHOTPLAM",
                        "PHOTFNU",
                        "PHOTZPT",
                        "PHOTBW",
                        "PHOTMODE",
                        "PHOTMJSR",
                        "PIXAR_SR",
                    ]:
                        if k in h:
                            keys[k] = h[k]

                    phot_scale *= to_per_sec

                try:
                    wcs_i = pywcs.WCS(header=flt[("SCI", ext)].header, fobj=flt)
                    wcs_i.pscale = get_wcs_pscale(wcs_i)
                except KeyError:
                    print(f"Failed to initialize WCS on {file}[SCI,{ext}]")
                    continue

                wcsh = to_header(wcs_i)
                row = [file, ext, keys["EXPTIME"]]

                if wcs_colnames is None:
                    wcs_colnames = ["file", "ext", "exptime"]
                    for k in wcsh:
                        wcs_colnames.append(k.lower())
                        wcs_keys[k.lower()] = wcsh[k]

                for k in wcs_colnames[3:]:
                    ku = k.upper()
                    if ku not in wcsh:
                        print(f"Keyword {ku} not found in WCS header")
                        row.append(wcs_keys[k] * 0)
                    else:
                        row.append(wcsh[ku])

                for k in wcsh:
                    if k.lower() not in wcs_colnames:
                        print(f"Extra keyword {ku} found in WCS header")

                wcs_rows.append(row)

                err_data = flt[("ERR", ext)].data * phot_scale

                # JWST: just 1,1024,4096 bits
                if flt[0].header["TELESCOP"] in ["JWST"]:
                    bad_bits = 1 | 1024 | 4096
                    if jwst_dq_flags is not None:
                        bad_bits |= get_jwst_dq_bit(jwst_dq_flags, verbose=verbose)

                    dq = flt[("DQ", ext)].data & bad_bits
                    dq |= bpdata.astype(dq.dtype)

                    # Clipping threshold for BKG extensions, global at top
                    # BKG_CLIP = [scale, percentile_lo, percentile_hi]
                    if has_bkg & (BKG_CLIP is not None):
                        # percentiles
                        bkg_lo, bkg_hi = np.nanpercentile(
                            flt["BKG"].data[dq == 0], BKG_CLIP[1:3]
                        )

                        # make sure lower (upper) limit is negative (positive)
                        clip_lo = -np.abs(bkg_lo)
                        clip_hi = np.abs(bkg_hi)

                        _bad_bkg = flt["BKG"].data < BKG_CLIP[0] * bkg_lo
                        _bad_bkg |= flt["BKG"].data > BKG_CLIP[0] * bkg_hi

                        # OR into dq mask
                        msg = f"Bad bkg pixels: N= {_bad_bkg.sum()} "
                        msg += f" ( {_bad_bkg.sum()/_bad_bkg.size*100:.1} %)"
                        log_comment(LOGFILE, msg, verbose=verbose)
                        dq |= _bad_bkg * 1024

                else:
                    dq = mod_dq_bits(flt[("DQ", ext)].data, okbits=bits) | bpdata

                wht = 1 / err_data ** 2
                _msk = (err_data <= 0) | (dq > 0)
                wht[_msk] = 0

                if weight_type == "jwst_var":
                    _var = err_data**2
                    _var[_msk] = 0
                    var_list.append(_var)

                if weight_type.startswith("jwst"):

                    if (("VAR_RNOISE", ext) in flt) & (rnoise_percentile is not None):
                        _rn_data = flt[("VAR_RNOISE", ext)].data
                        rnoise_value = np.nanpercentile(
                            _rn_data[~_msk], rnoise_percentile
                        )
                        _msk |= _rn_data >= rnoise_value

                    if ("VAR_POISSON", ext) in flt:
                        # Weight by VAR_RNOISE + median(VAR_POISSON)
                        if (~_msk).sum() > 0:
                            _var_data = flt[("VAR_POISSON", ext)].data[~_msk]
                            med_poisson = np.nanmedian(_var_data)
                            var = flt["VAR_RNOISE", ext].data + med_poisson
                            var *= phot_scale ** 2

                            wht = 1.0 / var
                            wht[_msk | (var <= 0)] = 0
                    else:
                        # Fall back to median_err
                        median_weight = np.nanmedian(wht[~_msk])
                        if (not np.isfinite(median_weight)) | ((~_msk).sum() == 0):
                            median_weight = 0

                        wht[~_msk] = median_weight

                median_weight = np.nanmedian(wht[~_msk])
                if (not np.isfinite(median_weight)) | ((~_msk).sum() == 0):
                    median_weight = 0

                msg = f"  ext (SCI,{ext}), sky={sky_value:.3f}"
                msg += f" has_bkg:{has_bkg}"
                msg += f" median_weight:{median_weight:.2e}"

                log_comment(LOGFILE, msg, verbose=verbose)

                # Use median(ERR) as the full image weight,
                # optionally scaling by the TIME array
                if weight_type in ["median_err", "time"]:
                    wht[~_msk] = median_weight

                    if weight_type == "time":
                        if ("TIME", ext) in flt:
                            if flt[("TIME", ext)].data is not None:
                                _time = flt[("TIME", ext)].data * 1
                                tmax = np.nanmax(_time[~_msk])
                                _time /= tmax
                                wht[~_msk] *= _time[~_msk]

                                msg = f"scale weight by (TIME,{ext})/{tmax:.1f}"
                                log_comment(LOGFILE, msg, verbose=verbose)

                wht_list.append(wht)
                sci_i = (flt[("SCI", ext)].data - sky) * phot_scale
                sci_i[wht <= 0] = 0
                sci_list.append(sci_i)

                if not hasattr(wcs_i, "pixel_shape"):
                    wcs_i.pixel_shape = wcs_i._naxis1, wcs_i._naxis2

                if not hasattr(wcs_i, "_naxis1"):
                    wcs_i._naxis1, wcs_i._naxis2 = wcs_i._naxis

                wcs_list.append(wcs_i)

        pscale_ratio = (wcs_i.pscale / outputwcs.pscale)

        if count == 0:
            res = drizzle_array_groups(
                sci_list,
                wht_list,
                wcs_list,
                var_list=var_list,
                median_weight=(weight_type == 'median_variance'),
                outputwcs=outputwcs,
                scale=0.1,
                kernel=kernel,
                pixfrac=pixfrac,
                calc_wcsmap=calc_wcsmap,
                verbose=verbose,
                data=None,
                with_slices=with_slices,
            )

            outsci, outwht, outvar, outctx, header, xoutwcs = res

            header["EXPTIME"] = flt[0].header["EXPTIME"]
            header["NDRIZIM"] = 1
            header["PIXFRAC"] = pixfrac
            header["KERNEL"] = kernel
            header["OKBITS"] = (bits, "FLT bits treated as valid")
            header["PHOTSCAL"] = _scale_photom, "Scale factor applied"

            header["GRIZLIV"] = grizli__version, "Grizli code version"

            header["WHTTYPE"] = weight_type, "Exposure weighting strategy"
            header["RNPERC"] = rnoise_percentile, "VAR_RNOISE clip percentile for JWST"
            header["PSCALER"] = pscale_ratio, "Ratio of input to output pixel scales"

            for k in keys:
                header[k] = keys[k]

        else:

            # outvar = Sum(wht**2 * var) / Sum(wht)**2, so
            # need to accumulate updates to Sum(wht * (wht * var)) / Sum(wht)
            if outvar is None:
                varnum = None
            else:
                varnum = outvar * outwht

            data = outsci, outwht, outctx, varnum

            res = drizzle_array_groups(
                sci_list,
                wht_list,
                wcs_list,
                median_weight=(weight_type == 'median_variance'),
                var_list=var_list,
                outputwcs=outputwcs,
                scale=0.1,
                kernel=kernel,
                pixfrac=pixfrac,
                calc_wcsmap=calc_wcsmap,
                verbose=verbose,
                data=data,
                with_slices=with_slices,
            )

            outsci, outwht, outvar, outctx = res[:4]
            header["EXPTIME"] += flt[0].header["EXPTIME"]
            header["NDRIZIM"] += 1

        count += 1
        header["FLT{0:05d}".format(count)] = file

        if median_weight is not None:
            header["WHT{0:05d}".format(count)] = (
                median_weight,
                f"Median weight of exposure {count}",
            )

        if _nsat is not None:
            header["SAT{0:05d}".format(count)] = (
                _nsat, f"Number of pixels flagged for persistence"
            )

        flt.close()

        # xfiles = glob.glob('*')
        # print('Clean: ', clean, xfiles)
        if clean:
            os.remove(file)

    if "awspath" in visit:
        awspath = visit["awspath"]
    else:
        awspath = ["." for f in visit["files"]]

    if len(awspath) == 1:
        awspath = [awspath[0] for f in visit["files"]]
    elif isinstance(awspath, str):
        _awspath = [awspath for f in visit["files"]]
        awspath = _awspath

    flist = ["{0}/{1}".format(awspath, visit["files"][i]) for i in indices]

    if dryrun:
        return flist

    elif count == 0:
        return None

    else:
        wcs_tab = GTable(names=wcs_colnames, rows=wcs_rows)

        outwht *= pscale_ratio**4 # (wcs_i.pscale / outputwcs.pscale) ** 4
        if outvar is not None:
            # Extra factors of the pixel area ratio in variance, which comes from
            # outvar = varnum / outwht
            outvar *= pscale_ratio**-2

        return outsci, outwht, outvar, outctx, header, flist, wcs_tab


def drizzle_array_groups(
    sci_list,
    wht_list,
    wcs_list,
    var_list=None,
    median_weight=False,
    outputwcs=None,
    scale=0.1,
    kernel="point",
    pixfrac=1.0,
    calc_wcsmap=False,
    verbose=True,
    data=None,
    first_uniqid=1,
    with_slices=False,
    **kwargs,
):
    """
    Drizzle array data with associated wcs

    Parameters
    ----------
    sci_list, wht_list : list
        List of science and weight `~numpy.ndarray` objects.

    wcs_list : list
        List of `~astropy.wcs.WCS` objects for each input array

    var_list : list
        List of separate variance arrays, if distinct from `wht_list`.  The variance
        images are combined as ``Vfinal = Sum(wht_i**2 * var_i) / Sum(wht_i)**2``,
        which reduces to ``Vfinal = 1 / Sum(wht_i)`` for inverse-variance weights
        ``wht_i = 1 / var_i`` typically used with drizzle.

    median_weight : bool
        Use median of ``wht_list`` for weights and ``var_list = [1 / wht_list_i]``,
        e.g., for appropriate Poisson weighting

    scale : float
        Output pixel scale in arcsec.

    kernel, pixfrac : str, float
        Drizzle parameters

    verbose : bool
        Print status messages

    outputwcs : `~astropy.wcs.WCS`, optional
        Output WCS for the drizzled image.

    calc_wcsmap : int, optional
        Flag to indicate whether to calculate the full WCS map. 
        If `calc_wcsmap=0`, the internal WCS map is not required.
        If `calc_wcsmap=1`, the internal WCS map is required.
        If `calc_wcsmap=2`, 
        the internal WCS map is required and the output WCS requires `calc_wcsmap=2`.

    data : tuple, optional
        Tuple containing the previously-drizzled images.  Either
        ``data = outsci, outwht, outctx`` *or*
        ``data = outsci, outwht, outctx, varnum``, where ``varnum = outvar * outwht``

        If not provided, new arrays will be created.

    first_uniqid : int, optional
        First `uniqid` value to use for the drizzle for contex maps

    with_slices : bool
        Compute slices of the overlap with each exposure before drizzling to the output array

    Returns
    -------
    outsci, outwht, outctx : `~numpy.ndarray`
        Output drizzled science, weight, and context images

    outvar : `~numpy.ndarray`, None
        Output variance if ``var_list`` provided or if ``median_weight``

    header : `~astropy.fits.io.Header`
        Drizzled image header.

    outputwcs : `~astropy.wcs.WCS`
        Drizzled image WCS.

    """
    from drizzlepac import adrizzle
    from drizzlepac import cdriz

    # from stsci.tools import logutil
    # log = logutil.create_logger(__name__)

    # Output header / WCS
    if outputwcs is None:
        # header, outputwcs = compute_output_wcs(wcs_list, pixel_scale=scale)
        header, outputwcs = make_maximal_wcs(
            wcs_list, pixel_scale=scale, verbose=False, pad=0, get_hdu=False
        )
    else:
        header = to_header(outputwcs)

    header["DRIZKERN"] = kernel, "Drizzle kernel"
    header["DRIZPIXF"] = pixfrac, "Drizzle pixfrac"

    if not hasattr(outputwcs, "_naxis1"):
        outputwcs._naxis1, outputwcs._naxis2 = outputwcs._naxis

    # Try to fix deprecated WCS
    for wcs_i in wcs_list:
        if not hasattr(wcs_i, "pixel_shape"):
            wcs_i.pixel_shape = wcs_i._naxis1, wcs_i._naxis2

        if not hasattr(wcs_i, "_naxis1"):
            wcs_i._naxis1, wcs_i._naxis2 = wcs_i._naxis[:2]

    # Output WCS requires full WCS map?
    if calc_wcsmap < 2:
        ctype = outputwcs.wcs.ctype
        if "-SIP" in ctype[0]:
            print("Output WCS ({0}) requires `calc_wcsmap=2`".format(ctype))
            calc_wcsmap = 2
        else:
            # Internal WCSMAP not required
            calc_wcsmap = 0

    shape = (header["NAXIS2"], header["NAXIS1"])

    # Use median for weight and propagate variance
    if (median_weight) & (var_list is None):
        var_list = []
        use_weights = []
        for wht_i in wht_list:
            ok_wht = (wht_i > 0) & np.isfinite(wht_i)
            var_i = 1. / wht_i
            var_i[~ok_wht] = 0
            var_list.append(var_i)
            use_weights.append(
                (np.nanmedian(wht_i[ok_wht]) * ok_wht).astype(np.float32)
            )
    else:
        use_weights = wht_list

    # Output arrays
    if data is not None:
        if len(data) == 3:
            outsci, outwht, outctx = data
        else:
            outsci, outwht, outctx, outvar = data
            if outvar is not None:
                _varwht = outwht * 1
                _varctx = outctx * 1

            if (outvar is not None) & (var_list is None):
                msg = (
                    'drizzle_array_groups: WARNING outvar provided in ``data``'
                    ' but var_list not provided'
                )
                log_comment(LOGFILE, msg, verbose=verbose, show_date=True)
    else:
        outsci = np.zeros(shape, dtype=np.float32)
        outwht = np.zeros(shape, dtype=np.float32)
        outctx = np.zeros(shape, dtype=np.int32)
        if var_list is None:
            outvar = None
        else:
            outvar = np.zeros(shape, dtype=np.float32)
            _varwht = np.zeros(shape, dtype=np.float32)
            _varctx = np.zeros(shape, dtype=np.int32)

    needs_var = (outvar is not None) & (var_list is not None)

    # Number of input arrays
    N = len(sci_list)

    # Drizzlepac cannot support >31 input images
    if first_uniqid + N > 31 and verbose:
        msg = "Warning: Too many input images for context map, will wrap around"
        log_comment(LOGFILE, msg, verbose=verbose, show_date=True)

    with_slices &= calc_wcsmap == 0

    for i in range(N):
        # log.info('Drizzle array {0}/{1}'.format(i+1, N))
        msg = "Drizzle array {0}/{1}".format(i + 1, N)

        if calc_wcsmap > 1:
            wcsmap = WCSMapAll  # (wcs_list[i], outputwcs)
            # wcsmap = cdriz.DefaultWCSMapping
        else:
            wcsmap = None

        # if (outwht > 0).sum() > 0:
        #     print(f"xxx owht {np.nanmax(outwht[outwht > 0])}  {(outwht > 0).sum()}")
        #
        if with_slices:
            # lower-left / upper-right corner in the output frame
            sr_i = SRegion(wcs_list[i], pad=1.1)
            xy = np.round(outputwcs.all_world2pix(sr_i.xy[0], 0)).astype(int)

            ll = xy.min(axis=0)
            ur = xy.max(axis=0)

            # Does the exposure not overlap with the target at all?
            if (ur[0] <= 0) | (ur[1] <= 0) | (ll[0] >= shape[1]) | (ll[1] >= shape[0]):
                msg += " slice (skip)"
                log_comment(LOGFILE, msg, verbose=verbose, show_date=True)

                continue

            slx = slice(np.maximum(ll[0], 0), np.minimum(ur[0], shape[1]))
            sly = slice(np.maximum(ll[1], 0), np.minimum(ur[1], shape[0]))

            msg += f"  slice [{sly.start}:{sly.stop}, {slx.start}:{slx.stop}]"

            outputwcs_i = outputwcs.slice((sly, slx))
            outputwcs_i.pscale = get_wcs_pscale(outputwcs_i)

            osci = outsci[sly, slx] * 1
            owht = outwht[sly, slx] * 1
            octx = outctx[sly, slx] * 1

            if outvar is not None:
                ovar = outvar[sly, slx] * 1
                ovarw = _varwht[sly, slx] * 1
                ovarc = _varctx[sly, slx] * 1

        else:
            outputwcs_i = outputwcs

            osci = outsci
            owht = outwht
            octx = outctx

            if outvar is not None:
                ovar = outvar
                ovarw = _varwht
                ovarc = _varctx

        log_comment(LOGFILE, msg, verbose=verbose, show_date=True)

        # if (owht > 0).sum() > 0:
        #     print(f"yyy owht {np.nanmax(owht[owht > 0])}  {(owht > 0).sum()}")
        #
        adrizzle.do_driz(
            sci_list[i].astype(np.float32, copy=False),
            wcs_list[i],
            use_weights[i].astype(np.float32, copy=False),
            outputwcs_i,
            osci,
            owht,
            octx,
            1.0,
            "cps",
            1,
            wcslin_pscale=wcs_list[i].pscale,
            uniqid=((first_uniqid - 1 + i) % 32) + 1,
            pixfrac=pixfrac,
            kernel=kernel,
            fillval="0",
            wcsmap=wcsmap,
        )

        if needs_var:
            adrizzle.do_driz(
                (var_list[i] * use_weights[i]).astype(np.float32, copy=False),
                wcs_list[i],
                use_weights[i].astype(np.float32, copy=False),
                outputwcs_i,
                ovar,
                ovarw,
                ovarc,
                1.0,
                "cps",
                1,
                wcslin_pscale=wcs_list[i].pscale,
                uniqid=1,
                pixfrac=pixfrac,
                kernel=kernel,
                fillval="0",
                wcsmap=wcsmap,
            )

        if with_slices:
            # Put slice back in full image
            outsci[sly, slx] = osci
            outwht[sly, slx] = owht
            outctx[sly, slx] = octx

            if outvar is not None:
                outvar[sly, slx]  =  ovar
                _varwht[sly, slx] =  ovarw
                _varctx[sly, slx] =  ovarc

        #     print(f"yyy owht {np.nanmax(owht[owht > 0])}  {(owht > 0).sum()}")
        #
        # if (outwht > 0).sum() > 0:
        #     print(f"xxx owht {np.nanmax(outwht[outwht > 0])}  {(outwht > 0).sum()}")

    if needs_var:
        # extra factor of Sum(w_i) for var = Sum(w_i**2 * var_i) / Sum(w_i)**2
        outvar /= outwht

    return outsci, outwht, outvar, outctx, header, outputwcs


class WCSMapAll:
    """Sample class to demonstrate how to define a coordinate transformation"""

    def __init__(self, input, output, origin=0):
        """
        Initialize the class.
        Parameters
        ----------
        input : `~grizli.utils.WCSObject`
            Input WCS object.

        output : `~grizli.utils.WCSObject`
            Output WCS object.

        origin : int, optional
            Origin value.

        Attributes
        ----------
        input : `~grizli.utils.WCSObject`
            Input WCS object.

        output : `~grizli.utils.WCSObject`
            Output WCS object.

        origin : int
            Origin value.

        shift : None
            Shift attribute.

        rot : None
            Rot attribute.

        scale : None
            Scale attribute.

        """
        import copy
        self.checkWCS(input, "Input")
        self.checkWCS(output, "Output")
        self.input = input
        self.output = copy.deepcopy(output)
        # self.output = output
        self.origin = 1  # origin
        self.shift = None
        self.rot = None
        self.scale = None

    def checkWCS(self, obj, name):
        """
        Check if the input object is a valid WCS object.
        
        Parameters
        ----------
        obj : `~pywcs.WCS`
            The input object to be checked.
        name : str
            The name of the object.
        
        """
        try:
            assert isinstance(obj, pywcs.WCS)
        except AssertionError:
            print(
                name + " object needs to be an instance or subclass of a PyWCS object."
            )
            raise

    def forward(self, pixx, pixy):
        """
        Transform the input pixx,pixy positions in the input frame
        to pixel positions in the output frame.

        Parameters
        ----------
        pixx : array-like
            The x-coordinates of the input pixel positions.
        pixy : array-like
            The y-coordinates of the input pixel positions.

        Returns
        -------
        result : tuple
            The transformed pixel positions in the output frame.

        """
        # This matches WTRAXY results to better than 1e-4 pixels.
        skyx, skyy = self.input.all_pix2world(pixx, pixy, self.origin)
        result = self.output.all_world2pix(skyx, skyy, self.origin)
        return result

    def backward(self, pixx, pixy):
        """
        Transform pixx,pixy positions from the output frame back onto their
        original positions in the input frame.

        Parameters
        ----------
        pixx : array-like
            The x-coordinates of the output pixel positions.

        pixy : array-like
            The y-coordinates of the output pixel positions.

        Returns
        -------
        result : tuple
            The transformed pixel positions in the input frame.
        
        """
        skyx, skyy = self.output.all_pix2world(pixx, pixy, self.origin)
        result = self.input.all_world2pix(skyx, skyy, self.origin)
        return result

    def get_pix_ratio(self):
        """
        Return the ratio of plate scales between the input and output WCS.
        This is used to properly distribute the flux in each pixel in 'tdriz'.
        """
        return self.output.pscale / self.input.pscale

    def xy2rd(self, wcs, pixx, pixy):
        """
        Transform input pixel positions into sky positions in the WCS provided.

        Parameters
        ----------
        wcs : `~pywcs.WCS`
            The WCS object containing the coordinate transformation.

        pixx : array-like
            The x-coordinates of the input pixel positions.

        pixy : array-like
            The y-coordinates of the input pixel positions.

        Returns
        -------
        ra : array-like
            The right ascension (RA) values in degrees.

        dec : array-like
            The declination (Dec) values in degrees.
        
        """
        return wcs.all_pix2world(pixx, pixy, 1)

    def rd2xy(self, wcs, ra, dec):
        """
        Transform input sky positions into pixel positions in the WCS provided.

        Parameters
        ----------
        wcs : `~pywcs.WCS`
            The WCS object containing the coordinate transformation.

        ra : array-like
            The right ascension (RA) values in degrees.
            
        dec : array-like
            The declination (Dec) values in degrees.

        Returns
        -------
        pixx : array-like
            The x-coordinates of the transformed pixel positions.

        pixy : array-like
            The y-coordinates of the transformed pixel positions.

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
    xsize = (x.max() - x.min()) * np.cos(crval[1] / 180 * np.pi) * 3600
    ysize = (y.max() - y.min()) * 3600

    xsize = np.minimum(xsize, max_size * pixel_scale)
    ysize = np.minimum(ysize, max_size * pixel_scale)

    header, outputwcs = make_wcsheader(
        ra=crval[0],
        dec=crval[1],
        size=(xsize, ysize),
        pixscale=pixel_scale,
        get_hdu=False,
        theta=0,
    )

    return header, outputwcs


def symlink_templates(force=False):
    """
    Symlink templates from module to $GRIZLI/templates
    as part of the initial setup
    
    Parameters
    ----------
    force : bool
        Force link files even if they already exist.
    """
    # if 'GRIZLI' not in os.environ:
    #     print('"GRIZLI" environment variable not set!')
    #     return False

    module_path = os.path.dirname(__file__)
    templates_path = os.path.join(module_path, "data/templates")

    out_path = os.path.join(GRIZLI_PATH, "templates")

    if (not os.path.exists(out_path)) | force:
        if os.path.exists(out_path):  # (force)
            shutil.rmtree(out_path)

            os.symlink(templates_path, out_path)
            print("Symlink: {0} -> {1}".format(templates_path, out_path))
    else:
        print("Templates directory exists: {0}".format(out_path))
        print("Use `force=True` to force a new symbolic link.")


def fetch_acs_wcs_files(beams_file, bucket_name="grizli-v1"):
    """
    Fetch wcs files for a given beams.fits files
    
    Parameters
    ----------
    beams_file : str
        Path to the beams.fits file.

    bucket_name : str, optional
        Name of the S3 bucket to fetch the files from. Default is "grizli-v1".

    """
    from urllib import request

    try:
        import boto3

        HAS_BOTO = True
    except:
        HAS_BOTO = False

    im = pyfits.open(beams_file)
    root = "_".join(beams_file.split("_")[:-1])

    for i in range(len(im)):
        h = im[i].header
        if "EXTNAME" not in h:
            continue

        if "FILTER" not in h:
            continue

        if (h["EXTNAME"] != "SCI") | (h["FILTER"] not in ["G800L"]):
            continue

        ext = {1: 2, 2: 1}[h["CCDCHIP"]]

        wcsfile = h["GPARENT"].replace(".fits", ".{0:02d}.wcs.fits".format(ext))

        # Download the file with S3 or HTTP
        if not os.path.exists(wcsfile):
            print("Fetch {0} from {1}/Pipeline/{2}".format(wcsfile, bucket_name, root))

            if HAS_BOTO:
                s3 = boto3.resource("s3")
                s3_client = boto3.client("s3")
                bkt = s3.Bucket(bucket_name)

                s3_path = "Pipeline/{0}/Extractions/{1}".format(root, wcsfile)
                bkt.download_file(
                    s3_path,
                    "./{0}".format(wcsfile),
                    ExtraArgs={"RequestPayer": "requester"},
                )

            else:
                url = "https://s3.amazonaws.com/{0}/".format(bucket_name)
                url += "Pipeline/{0}/Extractions/{1}".format(root, wcsfile)

                print("Fetch WCS file: {0}".format(url))
                req = request.urlretrieve(url, wcsfile)

    im.close()


def fetch_hst_calib(
    file="iref$uc72113oi_pfl.fits",
    ftpdir="https://hst-crds.stsci.edu/unchecked_get/references/hst/",
    verbose=True,
    ref_paths={},
    remove_corrupt=True,
):
    """
    Fetches the HST calibration file from the given FTP directory or local reference path.

    Parameters
    ----------
    file : str, optional
        The name of the calibration file. Default is "iref$uc72113oi_pfl.fits".

    ftpdir : str, optional
        The FTP directory where the calibration file is located.
        Default is "https://hst-crds.stsci.edu/unchecked_get/references/hst/".

    verbose : bool, optional
        If True, print status messages. Default is True.
        
    ref_paths : dict, optional
        A dictionary of reference paths. Default is an empty dictionary.

    remove_corrupt : bool, optional
        If True, remove the downloaded file if it is corrupt. Default is True.

    Returns
    -------
    str or bool
        The path to the downloaded calibration file if it exists and is valid.
        False if the file is corrupt or cannot be downloaded.

    """
    import os

    ref_dir = file.split("$")[0]
    cimg = file.split("{0}$".format(ref_dir))[1]

    if ref_dir in ref_paths:
        ref_path = ref_paths[ref_dir]
    else:
        ref_path = os.getenv(ref_dir)

    iref_file = os.path.join(ref_path, cimg)
    if not os.path.exists(iref_file):
        os.system("curl -o {0} {1}/{2}".format(iref_file, ftpdir, cimg))
        if "fits" in iref_file:
            try:
                _im = pyfits.open(iref_file)
                _im.close()
            except:
                msg = (
                    "Downloaded file {0} appears to be corrupt.\n"
                    "Check that {1}/{2} exists and is a valid file"
                )

                print(msg.format(iref_file, ftpdir, cimg))
                if remove_corrupt:
                    os.remove(iref_file)

                return False
    else:
        if verbose:
            print("{0} exists".format(iref_file))

    return iref_file


def fetch_hst_calibs(
    flt_file,
    ftpdir="https://hst-crds.stsci.edu/unchecked_get/references/hst/",
    calib_types=[
        "BPIXTAB",
        "CCDTAB",
        "OSCNTAB",
        "CRREJTAB",
        "DARKFILE",
        "NLINFILE",
        "DFLTFILE",
        "PFLTFILE",
        "IMPHTTAB",
        "IDCTAB",
        "NPOLFILE",
    ],
    verbose=True,
    ref_paths={},
):
    """
    Fetch necessary calibration files needed for running calwf3 from STScI FTP

    Parameters
    ----------
    flt_file : str
        Path to the FITS file.

    ftpdir : str, optional
        FTP directory to fetch the calibration files from. 
        Default is "https://hst-crds.stsci.edu/unchecked_get/references/hst/".

    calib_types : list, optional
        List of calibration types to fetch. Default is 
        ["BPIXTAB", "CCDTAB", "OSCNTAB", "CRREJTAB", "DARKFILE", 
        "NLINFILE", "DFLTFILE", "PFLTFILE", "IMPHTTAB", "IDCTAB", "NPOLFILE"].

    verbose : bool, optional
        Whether to print verbose output. Default is True.

    ref_paths : dict, optional
        Dictionary of reference paths. Default is {}.

    Returns
    -------
    calib_paths : list
        List of paths to the fetched calibration files.

    """
    import os

    im = pyfits.open(flt_file)
    if im[0].header["INSTRUME"] == "ACS":
        ref_dir = "jref"

    if im[0].header["INSTRUME"] == "WFC3":
        ref_dir = "iref"

    if im[0].header["INSTRUME"] == "WFPC2":
        ref_dir = "uref"

    if not os.getenv(ref_dir):
        print("No ${0} set!  Put it in ~/.bashrc or ~/.cshrc.".format(ref_dir))
        return False

    calib_paths = []

    for ctype in calib_types:
        if ctype not in im[0].header:
            continue

        if verbose:
            print("Calib: {0}={1}".format(ctype, im[0].header[ctype]))

        if im[0].header[ctype] == "N/A":
            continue

        path = fetch_hst_calib(
            im[0].header[ctype], ftpdir=ftpdir, verbose=verbose, ref_paths=ref_paths
        )
        calib_paths.append(path)

    im.close()

    return calib_paths


def mast_query_from_file_list(files=[], os_open=True):
    """
    Generate a MAST query on datasets in a list.

    Parameters
    ----------
    files : list, optional
        List of filenames to generate the MAST query. If not provided, it will
        search for all "*raw.fits" files in the current directory.

    os_open : bool, optional
        If True, open the MAST query URL in the default web browser. Default is
        True.

    Returns
    -------
    URL : str or False
        The MAST query URL if `os_open` is False. False if no `files` are
        specified or found.

    """
    if len(files) == 0:
        files = glob.glob("*raw.fits")

    if len(files) == 0:
        print("No `files` specified.")
        return False

    datasets = np.unique([file[:6] + "*" for file in files]).tolist()
    URL = "http://archive.stsci.edu/hst/search.php?action=Search&"
    URL += "sci_data_set_name=" + ",".join(datasets)
    if os_open:
        os.system('open "{0}"'.format(URL))

    return URL


def fetch_default_calibs(get_acs=False, **kwargs):
    """
    Fetch a set of default HST calibration files
    
    Parameters
    ----------
    get_acs : bool, optional
        Whether to fetch ACS calibration files. Default is False.

    **kwargs : dict, optional
        Additional keyword arguments.
        
    Returns
    -------
    bool
        True if the calibration files were successfully fetched,
        False otherwise.

    """
    paths = {}

    for ref_dir in ["iref", "jref"]:
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
            print(
                """
No ${0} set!  Make a directory and point to it in ~/.bashrc or ~/.cshrc.
For example,

  $ mkdir $GRIZLI/{0}
  $ export {0}="${GRIZLI}/{0}/" # put this in ~/.bashrc
""".format(
                    ref_dir
                )
            )

            return False

    # WFC3
    files = [
        "iref$uc72113oi_pfl.fits",  # F105W Flat
        "iref$uc721143i_pfl.fits",  # F140W flat
        "iref$u4m1335li_pfl.fits",  # G102 flat
        "iref$u4m1335mi_pfl.fits",  # G141 flat
        "iref$w3m18525i_idc.fits",  # IDCTAB distortion table}
    ]

    if "ACS" in kwargs:
        get_acs = kwargs["ACS"]

    if get_acs:
        files.extend(
            [
                "jref$n6u12592j_pfl.fits",  # F814 Flat
                "jref$o841350mj_pfl.fits",  # G800L flat])
                "jref$v971826jj_npl.fits",
            ]
        )

    for file in files:
        fetch_hst_calib(file, ref_paths=paths)

    badpix = os.path.join(paths["iref"], "badpix_spars200_Nov9.fits")
    print("Extra WFC3/IR bad pixels: {0}".format(badpix))
    if not os.path.exists(badpix):
        os.system(
            "curl -o {0}/badpix_spars200_Nov9.fits https://raw.githubusercontent.com/gbrammer/wfc3/master/data/badpix_spars200_Nov9.fits".format(
                paths["iref"]
            )
        )

    # Pixel area map
    pam = os.path.join(paths["iref"], "ir_wfc3_map.fits")
    print("Pixel area map: {0}".format(pam))
    if not os.path.exists(pam):
        os.system(
            "curl -o {0} https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/data-analysis/pixel-area-maps/_documents/ir_wfc3_map.fits".format(
                pam
            )
        )


def fetch_wfpc2_calib(
    file="g6q1912hu_r4f.fits",
    path=os.getenv("uref"),
    use_mast=False,
    verbose=True,
    overwrite=True,
    skip_existing=True,
):
    """
    Fetch static WFPC2 calibration file and 
    run `stsci.tools.convertwaiveredfits` on it.

    Parameters
    ----------
    file : str, optional
        Name of the calibration file to fetch. Default is "g6q1912hu_r4f.fits".

    path : str, optional
        Output path of the reference file. 
        Default is the value of the "uref" environment variable.

    use_mast : bool, optional
        - If True, try to fetch the calibration file from
          "mast.stsci.edu//api/v0/download/file?uri".

        - If False, fetch from a static directory
          "ssb.stsci.edu/cdbs_open/cdbs/uref_linux/".
        
        Default is False.

    verbose : bool, optional
        If True, print verbose output. Default is True.

    overwrite : bool, optional
        If True, overwrite existing files. Default is True.

    skip_existing : bool, optional
        If True, skip fetching the file if it already exists. Default is True.

    Returns
    -------
    status : bool
        True if the file was successfully fetched and converted, 
        False otherwise.

    """
    from stsci.tools import convertwaiveredfits

    try:  # Python 3.x
        import http.client as httplib
    except ImportError:  # Python 2.x
        import httplib

    if file.endswith("h"):
        # File like "g6q1912hu.r4h"
        file = file[:-1].replace(".", "_") + "f.fits"

    outPath = os.path.join(path, file)
    if os.path.exists(outPath) & skip_existing:
        print("# fetch_wfpc2_calib: {0} exists".format(outPath))
        return True

    if use_mast:
        server = "mast.stsci.edu"
        uri = "mast:HST/product/" + file
        request_path = "/api/v0/download/file?uri=" + uri
    else:
        server = "ssb.stsci.edu"
        request_path = "/cdbs_open/cdbs/uref_linux/" + file

    if verbose:
        print('# fetch_wfpc2_calib: "{0}" to {1}'.format(server + request_path, path))

    conn = httplib.HTTPSConnection(server)

    conn.request("GET", request_path)
    resp = conn.getresponse()
    fileContent = resp.read()
    conn.close()

    # check for file
    if len(fileContent) < 4096:
        print(
            'ERROR: "{0}" failed to download.  Try `use_mast={1}`.'.format(
                server + request_path, (use_mast is False)
            )
        )
        status = False
        raise FileNotFoundError
    else:
        print("# fetch_wfpc2_calib: {0} (COMPLETE)".format(outPath))
        status = True

    # save to file
    with open(outPath, "wb") as FLE:
        FLE.write(fileContent)

    if status:
        # Convert to standard FITS
        try:
            hdu = convertwaiveredfits.convertwaiveredfits(outPath)
            while "HISTORY" in hdu[0].header:
                hdu[0].header.remove("HISTORY")

            hdu.writeto(
                outPath.replace(".fits", "_c0h.fits"),
                overwrite=overwrite,
                output_verify="fix",
            )
        except:
            return True


def fetch_nircam_skyflats(dryrun=False):
    """
    Download skyflat files
    """
    conf_path = os.path.join(GRIZLI_PATH, "CONF", "NircamSkyFlat")

    dry = " --dryrun" if dryrun else ""

    os.system(
        f'aws s3 sync s3://grizli-v2/NircamSkyflats/ {conf_path} --exclude "*" --include "nrc*fits"' + dry
    )

    _files = glob.glob(conf_path + "/*fits")
    _files.sort()

    return _files


def fetch_nircam_wisp_templates(dryrun=False):
    """
    Download STScI wisp templates from s3 mirror
    """
    conf_path = os.path.join(GRIZLI_PATH, "CONF", "NircamWisp")
    
    dry = " --dryrun" if dryrun else ""
    
    os.system(
        f'aws s3 sync s3://grizli-v2/NircamWisp/stsci-v3/ {conf_path} --exclude "*" --include "WISP*fits"' + dry
    )

    _files = glob.glob(conf_path + "/*fits")
    _files.sort()

    return _files


def fetch_config_files(
    get_acs=False,
    get_sky=True,
    get_stars=True,
    get_epsf=True,
    get_jwst=False,
    get_wfc3=True,
    **kwargs,
):
    """
    Config files needed for Grizli
    
    Parameters
    ----------
    get_acs : bool, optional
        Whether to fetch ACS configuration files. Default is False.

    get_sky : bool, optional
        Whether to fetch grism sky files. Default is True.

    get_stars : bool, optional
        Whether to fetch stellar template files. Default is True.

    get_epsf : bool, optional
        Whether to fetch extended PSF files. Default is True.

    get_jwst : bool, optional
        Whether to fetch JWST configuration files. Default is False.

    get_wfc3 : bool, optional
        Whether to fetch WFC3 configuration files. Default is True.

    """
    if "ACS" in kwargs:
        get_acs = kwargs["ACS"]

    cwd = os.getcwd()

    print("Config directory: {0}/CONF".format(GRIZLI_PATH))

    os.chdir(os.path.join(GRIZLI_PATH, "CONF"))

    ftpdir = "ftp://ftp.stsci.edu/cdbs/wfc3_aux/"
    tarfiles = []

    # Config files
    # BASEURL = 'https://s3.amazonaws.com/grizli/CONF/'
    # BASEURL = 'https://erda.ku.dk/vgrid/Gabriel%20Brammer/CONF/'
    BASEURL = "https://raw.githubusercontent.com/gbrammer/" + "grizli-config/master"

    if get_wfc3:
        tarfiles = [
            "{0}/WFC3.IR.G102.cal.V4.32.tar.gz".format(ftpdir),
            "{0}/WFC3.IR.G141.cal.V4.32.tar.gz".format(ftpdir),
        ]
        tarfiles += [
            f"{BASEURL}/WFC3.IR.G102.WD.V4.32.tar.gz",
            f"{BASEURL}/WFC3.IR.G141.WD.V4.32.tar.gz",
        ]

    if get_jwst:
        tarfiles += [
            f"{BASEURL}/jwst-grism-conf.tar.gz",
            f"{BASEURL}/niriss.conf.220725.tar.gz",
            # f"{BASEURL}/nircam-wisp-aug2022.tar.gz",
        ]

    if get_sky:
        ftpdir = BASEURL
        tarfiles.append("{0}/grism_master_sky_v0.5.tar.gz".format(ftpdir))

    # gURL = 'http://www.stsci.edu/~brammer/Grizli/Files'
    # gURL = 'https://s3.amazonaws.com/grizli/CONF'
    gURL = BASEURL

    tarfiles.append("{0}/WFC3IR_extended_PSF.v1.tar.gz".format(gURL))

    if get_acs:
        tarfiles += [
            f"{BASEURL}/ACS.WFC.CHIP1.Stars.conf",
            f"{BASEURL}/ACS.WFC.CHIP2.Stars.conf",
        ]
        tarfiles.append("{0}/ACS.WFC.sky.tar.gz".format(gURL))
        tarfiles.append("{0}/ACS_CONFIG.tar.gz".format(gURL))

    for url in tarfiles:
        file = os.path.basename(url)
        if not os.path.exists(file):
            print("Get {0}".format(file))
            os.system("curl -o {0} {1}".format(file, url))

        if ".tar" in file:
            os.system("tar xzvf {0}".format(file))

    if get_epsf:
        # ePSF files for fitting point sources
        # psf_path = 'http://www.stsci.edu/hst/wfc3/analysis/PSF/psf_downloads/wfc3_ir/'
        # psf_path = 'https://www.stsci.edu/~jayander/STDPSFs/WFC3IR/'
        # psf_root = 'PSFSTD'
        # psf_path = 'https://www.stsci.edu/~jayander/HST1PASS/'

        ### HST1PASS seems to be timing out - december 2023
        psf_path = "https://www.stsci.edu/~jayander/HST1PASS/LIB/"
        psf_path += "PSFs/STDPSFs/WFC3IR"
        psf_root = "STDPSF"

        # mirror of files downloaded some time ago
        psf_path = "https://s3.amazonaws.com/grizli-v2/HST_EPSF"
        psf_root = "PSFSTD"

        ir_psf_filters = ["F105W", "F125W", "F140W", "F160W"]

        # New PSFs
        ir_psf_filters += ["F110W", "F127M"]

        files = [
            "{0}/{1}_WFC3IR_{2}.fits".format(psf_path, psf_root, filt)
            for filt in ir_psf_filters
        ]

        for url in files:
            file = os.path.basename(url).replace("STDPSF", "PSFSTD")
            if not os.path.exists(file):
                print("Get {0}".format(file))
                # os.system('curl -o {0} {1}'.format(file, url))
                with pyfits.open(url, cache=False) as _im:
                    _im.writeto(file, overwrite=True, output_verify="fix")
            else:
                print("File {0} exists".format(file))

    if get_stars:
        # Stellar templates
        print("Templates directory: {0}/templates".format(GRIZLI_PATH))
        os.chdir("{0}/templates".format(GRIZLI_PATH))

        url = "https://www.stsci.edu/~brammer/Grizli/Files/"
        files = [url + "stars_pickles.npy", url + "stars_bpgs.npy"]

        for url in files:
            file = os.path.basename(url)
            if not os.path.exists(file):
                print("Get {0}".format(file))
                os.system("curl -o {0} {1}".format(file, url))
            else:
                print("File {0} exists".format(file))

        print("ln -s stars_pickles.npy stars.npy")
        os.system("ln -s stars_pickles.npy stars.npy")

    os.chdir(cwd)


class MW_F99(object):
    """
    Wrapper around the `specutils.extinction` / `extinction` modules,
    which are called differently
    """

    def __init__(self, a_v, r_v=3.1):
            """
            Initialize the ExtinctionCorrection object.

            Parameters
            ----------
            a_v : float
                The V-band extinction value.

            r_v : float, optional
                The ratio of total to selective extinction (default is 3.1).

            Attributes
            ----------
            a_v : float
                The V-band extinction value.

            r_v : float
                The ratio of total to selective extinction.

            IS_SPECUTILS : bool
                Flag indicating if the specutils.extinction module is available.

            IS_EXTINCTION : bool
                Flag indicating if the extinction.Fitzpatrick99 
                module is available.

            F99 : `~specutils.extinction.ExtinctionF99` 
                    or `~extinction.Fitzpatrick99`
                    The extinction model object.

            status : bool
                Flag indicating if either the specutils.extinction or extinction.
                Fitzpatrick99 module is available.
            
            """
            
            self.a_v = a_v
            self.r_v = r_v

            self.IS_SPECUTILS = False
            self.IS_EXTINCTION = False

            try:
                from specutils.extinction import ExtinctionF99

                self.IS_SPECUTILS = True
                self.F99 = ExtinctionF99(self.a_v, r_v=self.r_v)
            except (ImportError):
                try:
                    from extinction import Fitzpatrick99

                    self.IS_EXTINCTION = True
                    self.F99 = Fitzpatrick99(r_v=self.r_v)

                except (ImportError):
                    print(
                        """
                        Couldn\'t find extinction modules in
                        `specutils.extinction` or
                        `extinction.Fitzpatrick99`.

                        MW extinction not implemented.
                        """
                    )

            self.status = self.IS_SPECUTILS | self.IS_EXTINCTION

    def __call__(self, wave_input):
            """
            Apply the extinction correction to the input wavelength array.

            Parameters
            ----------
            wave_input : array-like
                Input wavelength array.

            Returns
            -------
            array-like
                Extinction correction.

            """
            
            import astropy.units as u

            if isinstance(wave_input, list):
                wave = np.array(wave_input)
            else:
                wave = wave_input

            if self.status is False:
                return np.zeros_like(wave)

            if self.IS_SPECUTILS:
                if hasattr(wave, "unit"):
                    wave_aa = wave
                else:
                    wave_aa = wave * u.AA

                return self.F99(wave_aa)

            if self.IS_EXTINCTION:
                if hasattr(wave, "unit"):
                    wave_aa = wave.to(u.AA)
                else:
                    wave_aa = wave

                return self.F99(wave_aa, self.a_v, unit="aa")


class EffectivePSF:
    def __init__(self, **kwargs):
        """
        Tools for handling WFC3/IR Effective PSF

        See documentation at http://www.stsci.edu/hst/wfc3/analysis/PSF.

        PSF files stored in ``$GRIZLI/CONF/``
        """

        self.load_PSF_data(**kwargs)

    def load_PSF_data(self, jwst_stdpsf=True, **kwargs):
        """
        Load data from PSFSTD files

        Files should be located in ${GRIZLI}/CONF/ directory.
        """
        self.epsf = OrderedDict()

        for filter in ["F105W", "F110W", "F125W", "F140W", "F160W", "F127M"]:
            file = os.path.join(
                GRIZLI_PATH, "CONF", "PSFSTD_WFC3IR_{0}.fits".format(filter)
            )

            if not os.path.exists(file):
                continue

            with pyfits.open(file, ignore_missing_simple=True) as _im:
                data = _im[0].data.T * 1
                data[data < 0] = 0

            self.epsf[filter] = data

        # UVIS
        filter_files = glob.glob(
            os.path.join(GRIZLI_PATH, "CONF", "PSFSTD_WFC3UV*.fits")
        )
        filter_files.sort()
        for file in filter_files:
            with pyfits.open(file, ignore_missing_end=True) as _im:
                data = _im[0].data.T * 1
                data[data < 0] = 0

            filt = "_".join(file.strip(".fits").split("_")[2:])
            self.epsf[filt + "U"] = data

        # ACS
        filter_files = glob.glob(
            os.path.join(GRIZLI_PATH, "CONF", "PSFSTD_ACSWFC*.fits")
        )
        filter_files.sort()
        for file in filter_files:
            with pyfits.open(file, ignore_missing_end=True) as _im:
                data = _im[0].data.T * 1.0
                data[data < 0] = 0

            filt = "_".join(file.strip(".fits").split("_")[2:])
            self.epsf[filt] = data

        # JWST
        filter_files = glob.glob(
            os.path.join(GRIZLI_PATH, "CONF/JWSTePSF", "nircam*.fits")
        )
        filter_files += glob.glob(
            os.path.join(GRIZLI_PATH, "CONF/JWSTePSF", "niriss*.fits")
        )
        filter_files += glob.glob(
            os.path.join(GRIZLI_PATH, "CONF/JWSTePSF", "miri*.fits")
        )
        filter_files.sort()
        for file in filter_files:
            with pyfits.open(file, ignore_missing_end=True) as _im:
                data = _im[0].data * 1  # [::-1,:,:]#[:,::-1,:]

                data[data < 0] = 0
                key = "{0}-{1}".format(
                    _im[0].header["DETECTOR"].upper(), _im[0].header["FILTER"]
                )

                if "LABEL" in _im[0].header:
                    key += "-" + _im[0].header["LABEL"]

            self.epsf[key] = data

        # Dummy, use F105W ePSF for F098M and F110W
        self.epsf["F098M"] = self.epsf["F105W"]
        self.epsf["F128N"] = self.epsf["F125W"]
        self.epsf["F130N"] = self.epsf["F125W"]
        self.epsf["F132N"] = self.epsf["F125W"]

        # Dummy filters for IR grisms
        self.epsf["G141"] = self.epsf["F140W"]
        self.epsf["G102"] = self.epsf["F105W"]

        # Extended
        self.extended_epsf = {}
        for filter in ["F105W", "F125W", "F140W", "F160W"]:
            file = os.path.join(
                GRIZLI_PATH, "CONF", "extended_PSF_{0}.fits".format(filter)
            )

            if not os.path.exists(file):
                # BASEURL = 'https://erda.ku.dk/vgrid/Gabriel%20Brammer/CONF/'
                BASEURL = (
                    "https://raw.githubusercontent.com/gbrammer/"
                    + "grizli-config/master"
                )

                msg = "Extended PSF file '{0}' not found.".format(file)
                msg += "Get the archive from "
                msg += f" {BASEURL}/WFC3IR_extended_PSF.v1.tar.gz"
                msg += " and unpack in ${GRIZLI}/CONF/"
                raise FileNotFoundError(msg)

            with pyfits.open(file) as _im:
                data = _im[0].data * 1
                data[data < 0] = 0

            # Mask center
            NX = data.shape[0] / 2 - 1
            yp, xp = np.indices(data.shape)
            R = np.sqrt((xp - NX) ** 2 + (yp - NX) ** 2)
            data[R <= 4] = 0.0

            self.extended_epsf[filter] = data
            self.extended_N = int(NX)

        self.extended_epsf["F098M"] = self.extended_epsf["F105W"]
        self.extended_epsf["F110W"] = self.extended_epsf["F105W"]
        self.extended_epsf["F128N"] = self.extended_epsf["F125W"]
        self.extended_epsf["F130N"] = self.extended_epsf["F125W"]
        self.extended_epsf["F132N"] = self.extended_epsf["F125W"]
        self.extended_epsf["G102"] = self.extended_epsf["F105W"]
        self.extended_epsf["G141"] = self.extended_epsf["F140W"]

        if jwst_stdpsf:
            self.load_jwst_stdpsf(**kwargs)


    def load_jwst_stdpsf(self, miri_filters=["F560W", "F770W", "F1000W", "F1130W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"], miri_extended=True, nircam_sw_filters=["F200W"], nircam_sw_detectors=["A1","A2","A3","A4","B1","B2","B3","B4"], nircam_lw_filters=["F444W"], nircam_lw_detectors=["AL", "BL"], clip_negative=False, use_astropy_cache=True):
        """
        Load ePSF models from https://www.stsci.edu/~jayander/JWST1PASS/LIB/PSFs/STDPSFs

        Parameters
        ----------
        miri_filters : list
            MIRI filters to load

        miri_extended : bool
            Use extended versions of the MIRI ePSFs

        nircam_sw_filters, nircam_sw_detectors : list, list
            NIRCam SW filters to load

        nircam_lw_filters, nircam_lw_detectors : list, list
            NIRCam LW filters to load

        clip_negative : bool
            Set negative pixels in ePSF models to zero

        use_astropy_cache : bool
            Download the ePSF files with the astropy cache

        """
        from astropy.utils.data import download_file
        try:
            from urllib.request import HTTPError
        except ImportError:
            from urllib3.exceptions import HTTPError

        stdpsf_url = "https://www.stsci.edu/~jayander/JWST1PASS/LIB/PSFs/STDPSFs/"

        if miri_extended:
            miri_path = "MIRI/EXTENDED/STDPSF_MIRI_{filter}_EXTENDED.fits"
        else:
            miri_path = "MIRI/STDPSF_MIRI_{filter}.fits"

        for filter in miri_filters:
            full_url = stdpsf_url + miri_path.format(filter=filter)

            try:
                file_obj = download_file(full_url, cache=use_astropy_cache)
            except HTTPError:
                msg = f"Failed to download ePSF from {full_url}"
                log_comment(LOGFILE, msg, verbose=True, show_date=True)
                continue

            with pyfits.open(file_obj) as im:
                data = np.array([d.T for d in im[0].data]).T * 1.0
                key = os.path.basename(full_url.split('.fits')[0])
                if clip_negative:
                    data[data < 0] = 0

                self.epsf[key] = data

        sw_path = "NIRCam/SWC/{filter}/STDPSF_NRC{detector}_{filter}.fits"

        for filter in nircam_sw_filters:
            for detector in nircam_sw_detectors:
                full_url = stdpsf_url + sw_path.format(filter=filter, detector=detector)

                try:
                    file_obj = download_file(full_url, cache=use_astropy_cache)
                except HTTPError:
                    msg = f"Failed to download ePSF from {full_url}"
                    log_comment(LOGFILE, msg, verbose=True, show_date=True)
                    continue

                with pyfits.open(file_obj) as im:
                    data = np.array([d.T for d in im[0].data]).T * 1.0
                    key = os.path.basename(full_url.split('.fits')[0])
                    if clip_negative:
                        data[data < 0] = 0

                    self.epsf[key] = data

        lw_path = "NIRCam/LWC/STDPSF_NRC{detector}_{filter}.fits"

        for filter in nircam_lw_filters:
            for detector in nircam_lw_detectors:
                full_url = stdpsf_url + lw_path.format(filter=filter, detector=detector)

                try:
                    file_obj = download_file(full_url, cache=use_astropy_cache)
                except HTTPError:
                    msg = f"Failed to download ePSF from {full_url}"
                    log_comment(LOGFILE, msg, verbose=True, show_date=True)
                    continue

                with pyfits.open(file_obj) as im:
                    data = np.array([d.T for d in im[0].data]).T * 1.0
                    key = os.path.basename(full_url.split('.fits')[0])
                    key = key.replace(f"{detector}_", f"{detector}ONG_")
                    if clip_negative:
                        data[data < 0] = 0

                    self.epsf[key] = data


    def get_at_position(self, x=507, y=507, filter="F140W", rot90=0):
        """
        Evaluate ePSF at detector coordinates

        Parameters
        ----------
        x : int
            X pixel coordinate.

        y : int
            Y pixel coordinate.

        filter : str, optional
            Filter name, by default "F140W".

        rot90 : int, optional
            The rotation angle in degrees, by default 0.

        Returns
        -------
        psf_xy : numpy.ndarray
            The evaluated ePSF values at the given coordinates.

        """
        epsf = self.epsf[filter]

        psf_type = "HST/Optical"

        if filter in [
            "F098M",
            "F110W",
            "F105W",
            "F125W",
            "F140W",
            "F160W",
            "G102",
            "G141",
            "F128N",
            "F130N",
            "F132N",
        ]:
            psf_type = "WFC3/IR"

        elif filter.startswith("NRC") | filter.startswith("NIS"):
            # NIRISS, NIRCam 2K
            psf_type = "JWST/2K"

        elif filter.startswith("MIRI"):
            psf_type = "JWST/MIRI"

        elif filter.startswith("STDPSF_MIRI"):
            # Libralato et al. (2024)
            psf_type = "STDPSF_MIRI"

        elif filter.startswith("STDPSF_NRC"):
            psf_type = "STDPSF_NRC"

        self.eval_psf_type = psf_type

        if psf_type == "WFC3/IR":
            #  IR detector
            rx = 1 + (np.clip(x, 1, 1013) - 0) / 507.0
            ry = 1 + (np.clip(y, 1, 1013) - 0) / 507.0

            # zero index
            rx -= 1
            ry -= 1

            nx = np.clip(int(rx), 0, 2)
            ny = np.clip(int(ry), 0, 2)

            # print x, y, rx, ry, nx, ny

            fx = rx - nx
            fy = ry - ny

            psf_xy = (1 - fx) * (1 - fy) * epsf[:, :, nx + ny * 3]
            psf_xy += fx * (1 - fy) * epsf[:, :, (nx + 1) + ny * 3]
            psf_xy += (1 - fx) * fy * epsf[:, :, nx + (ny + 1) * 3]
            psf_xy += fx * fy * epsf[:, :, (nx + 1) + (ny + 1) * 3]

            self.eval_filter = filter

        elif psf_type == "JWST/MIRI":
            #  IR detector
            NDET = int(np.sqrt(epsf.shape[2]))

            rx = 1 + (np.clip(x, 1, 1023) - 0) / 512.0
            ry = 1 + (np.clip(y, 1, 1023) - 0) / 512.0

            # zero index
            rx -= 1
            ry -= 1

            # nx = np.clip(int(rx), 0, 2)
            # ny = np.clip(int(ry), 0, 2)
            nx = np.clip(int(rx), 0, NDET - 1)
            ny = np.clip(int(ry), 0, NDET - 1)

            # print x, y, rx, ry, nx, ny

            fx = rx - nx
            fy = ry - ny

            # psf_xy = (1-fx)*(1-fy)*epsf[:, :, nx+ny*3]
            # psf_xy += fx*(1-fy)*epsf[:, :, (nx+1)+ny*3]
            # psf_xy += (1-fx)*fy*epsf[:, :, nx+(ny+1)*3]
            # psf_xy += fx*fy*epsf[:, :, (nx+1)+(ny+1)*3]

            if NDET == 1:
                psf_xy = epsf[:, :, 0]
            else:
                psf_xy = (1 - fx) * (1 - fy) * epsf[:, :, nx + ny * NDET]
                psf_xy += fx * (1 - fy) * epsf[:, :, (nx + 1) + ny * NDET]
                psf_xy += (1 - fx) * fy * epsf[:, :, nx + (ny + 1) * NDET]
                psf_xy += fx * fy * epsf[:, :, (nx + 1) + (ny + 1) * NDET]

            # psf_xy = np.rot90(psf_xy.T, 2)
            psf_xy = psf_xy.T

            self.eval_filter = filter

        elif psf_type == "STDPSF_MIRI":
            #  IR detector
            NDET = int(np.sqrt(epsf.shape[2]))

            rx = np.interp(x, [1, 358, 1032], [1, 2, 3])
            ry = np.interp(y, [1, 512, 1024], [1, 2, 3])

            # zero index
            rx -= 1
            ry -= 1

            nx = np.clip(int(rx), 0, 2)
            ny = np.clip(int(ry), 0, 2)
            # nx = np.clip(int(rx), 0, NDET - 1)
            # ny = np.clip(int(ry), 0, NDET - 1)

            # print x, y, rx, ry, nx, ny

            fx = rx - nx
            fy = ry - ny

            if NDET == 1:
                psf_xy = epsf[:, :, 0]
            else:
                psf_xy = (1 - fx) * (1 - fy) * epsf[:, :, nx + ny * NDET]
                psf_xy += fx * (1 - fy) * epsf[:, :, (nx + 1) + ny * NDET]
                psf_xy += (1 - fx) * fy * epsf[:, :, nx + (ny + 1) * NDET]
                psf_xy += fx * fy * epsf[:, :, (nx + 1) + (ny + 1) * NDET]

            # psf_xy = np.rot90(psf_xy.T, 2)
            psf_xy = psf_xy.T

            self.eval_filter = filter

        elif psf_type == "STDPSF_NRC":

            NDET = int(np.sqrt(epsf.shape[2]))

            rx = np.interp(x, [0, 512, 1024, 1536, 2048], [1, 2, 3, 4, 5])
            ry = np.interp(y, [0, 512, 1024, 1536, 2048], [1, 2, 3, 4, 5])

            # zero index
            rx -= 1
            ry -= 1

            nx = np.clip(int(rx), 0, 4)
            ny = np.clip(int(ry), 0, 4)
            # nx = np.clip(int(rx), 0, NDET - 1)
            # ny = np.clip(int(ry), 0, NDET - 1)

            # print x, y, rx, ry, nx, ny

            fx = rx - nx
            fy = ry - ny

            if NDET == 1:
                psf_xy = epsf[:, :, 0]
            else:
                psf_xy = (1 - fx) * (1 - fy) * epsf[:, :, nx + ny * NDET]
                psf_xy += fx * (1 - fy) * epsf[:, :, (nx + 1) + ny * NDET]
                psf_xy += (1 - fx) * fy * epsf[:, :, nx + (ny + 1) * NDET]
                psf_xy += fx * fy * epsf[:, :, (nx + 1) + (ny + 1) * NDET]

            # psf_xy = np.rot90(psf_xy.T, 2)
            psf_xy = psf_xy.T

            self.eval_filter = filter

        elif psf_type == "JWST/2K":

            NDET = int(np.sqrt(epsf.shape[2]))

            #  IR detector
            rx = 1 + (np.clip(x, 1, 2047) - 0) / 1024.0
            ry = 1 + (np.clip(y, 1, 2047) - 0) / 1024.0

            # zero index
            rx -= 1
            ry -= 1

            nx = np.clip(int(rx), 0, NDET - 1)
            ny = np.clip(int(ry), 0, NDET - 1)

            # print x, y, rx, ry, nx, ny

            fx = rx - nx
            fy = ry - ny

            if NDET == 1:
                psf_xy = epsf[:, :, 0]
            else:
                psf_xy = (1 - fx) * (1 - fy) * epsf[:, :, nx + ny * NDET]
                psf_xy += fx * (1 - fy) * epsf[:, :, (nx + 1) + ny * NDET]
                psf_xy += (1 - fx) * fy * epsf[:, :, nx + (ny + 1) * NDET]
                psf_xy += fx * fy * epsf[:, :, (nx + 1) + (ny + 1) * NDET]

            psf_xy = psf_xy.T
            # psf_xy = np.rot90(psf_xy.T, 2)

            self.eval_filter = filter

        elif psf_type == "HST/Optical":

            sh = epsf.shape

            if sh[2] == 90:
                # ACS WFC
                iX, iY = 9, 10  # 9, 10
            else:
                # UVIS
                iX, iY = 7, 8

            rx = 1 + (np.clip(x, 1, 4095) - 0) / (4096 / (iX - 1))
            ry = 1 + (np.clip(y, 1, 4095) - 0) / (4096 / (iY - 1))

            # zero index
            rx -= 1
            ry -= 1

            nx = np.clip(np.asarray(rx,dtype=int), 0, iX-1)
            ny = np.clip(np.asarray(ry,dtype=int), 0, iY-1)

            # print x, y, rx, ry, nx, ny

            fx = rx - nx
            fy = ry - ny

            psf_xy = (1 - fx) * (1 - fy) * epsf[:, :, nx + ny * iX]
            psf_xy += fx * (1 - fy) * epsf[:, :, (nx + 1) + ny * iX]
            psf_xy += (1 - fx) * fy * epsf[:, :, nx + (ny + 1) * iX]
            psf_xy += fx * fy * epsf[:, :, (nx + 1) + (ny + 1) * iX]

            self.eval_filter = filter

        if rot90 != 0:
            self.psf_xy_rot90 = rot90
            psf_xy = np.rot90(psf_xy, rot90)

        return psf_xy

    def eval_ePSF(self, psf_xy, dx, dy, extended_data=None):
        """
        Evaluate PSF at dx,dy coordinates
        
        Parameters
        ----------
        psf_xy : numpy.ndarray
            The PSF data.

        dx : numpy.ndarray
            The x-coordinates of the evaluation points.

        dy : numpy.ndarray
            The y-coordinates of the evaluation points.

        rot90 : int, optional
            The rotation angle in degrees, by default 0.

        extended_data : numpy.ndarray, optional
            Extended PSF data, by default None.

        Returns
        -------
        numpy.ndarray
            The evaluated PSF values at the given coordinates.

        """
        # So much faster than scipy.interpolate.griddata!
        from scipy.ndimage import map_coordinates
        # ePSF only defined to 12.5 pixels
        if self.eval_psf_type in ["WFC3/IR", "HST/Optical"]:
            ok = (np.abs(dx) <= 12.5) & (np.abs(dy) <= 12.5)
            coords = np.array([50 + 4 * dx[ok], 50 + 4 * dy[ok]])
        else:
            # JWST are +/- 32 pixels
            sh = psf_xy.shape
            _size = (sh[0] - 1) // 4
            _x0 = _size * 2
            _cen = (_x0 - 1) // 2
            ok = (np.abs(dx) <= _cen) & (np.abs(dy) <= _cen)
            coords = np.array([_x0 + 4 * dx[ok], _x0 + 4 * dy[ok]])
        # Do the interpolation
        interp_map = map_coordinates(psf_xy, coords, order=3)
        # Fill output data
        out = np.zeros_like(dx, dtype=np.float32)
        out[ok] = interp_map
        # Extended PSF
        if extended_data is not None:
            ok = np.abs(dx) < self.extended_N
            ok &= np.abs(dy) < self.extended_N
            x0 = self.extended_N
            coords = np.array([x0 + dy[ok] + 0, x0 + dx[ok]])
            interp_map = map_coordinates(extended_data, coords, order=0)
            out[ok] += interp_map
        return out

    @staticmethod
    def objective_epsf(
        params, self, psf_xy, sci, ivar, xp, yp, extended_data, ret, ds9
    ):
        """
        Objective function for fitting ePSFs

        Parameters
        ----------
        params : list
            List of fitting parameters [normalization, xc, yc, background].
        self : object
            The object instance.
        psf_xy : array-like
            Array of PSF coordinates.
        sci : array-like
            Science image.
        ivar : array-like
            Inverse variance image.
        xp : array-like
            X positions.
        yp : array-like
            Y positions.
        extended_data : bool
            Flag indicating whether extended data is used.
        ret : str
            Return type. Possible values are 'resid', 'lm', 'model', or 'chi2'.
        ds9 : bool
            Flag indicating whether to display the result in DS9.
        Returns
        -------
        resid : array-like
            Residuals.
        lm_resid : array-like
            Masked residuals for LM optimization.
        model : tuple
            Tuple containing the PSF model, background, Ax, and coeffs.
        chi2 : float
            Chi-squared value.
        """
        
        dx = xp - params[1]
        dy = yp - params[2]

        ddx = xp - xp.min()
        ddy = yp - yp.min()

        ddx = ddx / ddx.max()
        ddy = ddy / ddy.max()

        bkg = params[3] + params[4] * ddx + params[5] * ddy  # + params[6]*ddx*ddy

        psf_offset = (
            self.eval_ePSF(psf_xy, dx, dy, extended_data=extended_data) * params[0]
        )

        resid = (sci - psf_offset - bkg) * np.sqrt(ivar)

        if ds9:
            ds9.view(sci - psf_offset - bkg)

        if ret == "resid":
            return resid
        elif ret == "lm":
            # masked residuals for LM optimization
            if False:
                print(params, (resid ** 2).sum())

            return resid[resid != 0]
        elif ret == "model":
            return psf_offset, bkg, None, None
        else:
            chi2 = (resid ** 2).sum()
            # print(params, chi2)
            return chi2

    @staticmethod
    def objective_epsf_center(
        params, self, psf_xy, sci, ivar, xp, yp, extended_data, ret, ds9
    ):
        """
        Objective function for fitting ePSF centers

        Parameters
        ----------
        params : list
            List of fitting parameters [xc, yc].
        self : object
            The object instance.
        psf_xy : array-like
            Array of PSF coordinates.
        sci : array-like
            Science image.
        ivar : array-like
            Inverse variance image.
        xp : array-like
            X positions.
        yp : array-like
            Y positions.
        extended_data : bool
            Flag indicating whether extended data is used.
        ret : str
            Return type. Possible values are 'resid', 'lm', 'model', or 'chi2'.
        ds9 : bool
            Flag indicating whether to display the result in DS9.
        Returns
        -------
        resid : array-like
            Residuals.
        lm_resid : array-like
            Masked residuals for LM optimization.
        model : tuple
            Tuple containing the PSF model, background, Ax, and coeffs.
        chi2 : float
            Chi-squared value.
        """
        from numpy.linalg import lstsq

        dx = xp - params[0]
        dy = yp - params[1]

        ddx = xp
        ddy = yp

        ddx = ddx / ddx.max()
        ddy = ddy / ddy.max()

        psf_offset = (
            self.eval_ePSF(psf_xy, dx, dy, extended_data=extended_data)
        )

        A = (np.array([psf_offset, np.ones_like(sci), ddx, ddy]) * np.sqrt(ivar)).reshape((4, -1))
        scif = (sci * np.sqrt(ivar)).flatten()
        mask = (scif != 0)
        coeffs, _resid, _rank, _s = lstsq(A[:, mask].T, scif[mask], rcond=LSTSQ_RCOND)

        resid = (scif - np.dot(coeffs, A))

        if ds9:
            Ax = (np.array([psf_offset, np.ones_like(sci), ddx, ddy])).reshape((4, -1))
            psf_model = np.dot(coeffs[:1], Ax[:1, :]).reshape(sci.shape)
            bkg = np.dot(coeffs[1:], Ax[1:, :]).reshape(sci.shape)
            ds9.view((sci - psf_model-bkg)*mask.reshape(sci.shape))
        
        if ret == "resid":
            return resid
        elif ret == "lm":
            # masked residuals for LM optimization
            return resid[resid != 0]
        elif ret == "model":
            Ax = (np.array([psf_offset, np.ones_like(sci), ddx, ddy])).reshape((4, -1))
            psf_model = np.dot(coeffs[:1], Ax[:1, :]).reshape(sci.shape)
            bkg = np.dot(coeffs[1:], Ax[1:, :]).reshape(sci.shape)
            return psf_model, bkg, Ax, coeffs
        else:
            chi2 = (resid ** 2).sum()
            return chi2

    def fit_ePSF(
        self,
        sci,
        center=None,
        origin=[0, 0],
        ivar=1,
        N=7,
        filter="F140W",
        tol=1.0e-4,
        guess=None,
        get_extended=False,
        method="lm",
        ds9=None,
        psf_params=None,
        only_centering=True,
        rot90=0,
    ):
        """
        Fit ePSF to input data

        Parameters
        ----------
        sci : numpy.ndarray
            The input data to fit the ePSF to.

        center : tuple, optional
            The center coordinates of the ePSF. If not provided, 
            the center is calculated as the center of the input data.

        origin : list, optional
            The origin coordinates of the ePSF.

        ivar : float or numpy.ndarray, optional
            The inverse variance of the input data. Default is 1.

        N : int, optional
            The size of the ePSF region to fit. Default is 7.

        filter : str, optional
            The filter of the input data. Default is "F140W".

        tol : float, optional
            The tolerance for the fitting algorithm. Default is 1.0e-4.

        guess : tuple, optional
            The initial guess for the ePSF parameters. 
            If not provided, the guess is calculated based on the input data.

        get_extended : bool, optional
            Whether to include extended data in the fitting process. 
            Default is False.

        method : str, optional
            The fitting method to use. Default is "lm".

        ds9 : str, optional
            Optional name of the DS9 instance to display the fitting process. 
            Default is None.

        psf_params : numpy.ndarray, optional
            The parameters of the ePSF model. 
            If provided, the fitting process is skipped 
            and the provided parameters are returned.

        only_centering : bool, optional
            Whether to only fit for centering and compute normalizations. 
            Default is True.

        rot90 : int, optional
            The rotation angle of the ePSF. Default is 0.

        Returns
        -------
        psf_model : numpy.ndarray
            The model of the ePSF.

        bkg : float
            The background level of the ePSF.

        A : numpy.ndarray
            The normalization coefficients of the ePSF.

        coeffs : numpy.ndarray
            The coefficients of the ePSF.

        """
        from scipy.optimize import minimize, least_squares

        sh = sci.shape
        if center is None:
            y0, x0 = np.array(sh) / 2.0 - 1
        else:
            x0, y0 = center

        xd = x0 + origin[1]
        yd = y0 + origin[0]

        xc, yc = int(x0), int(y0)

        psf_xy = self.get_at_position(x=xd, y=yd, filter=filter, rot90=rot90)

        yp, xp = np.indices(sh)

        if guess is None:
            if np.isscalar(ivar):
                ix = np.argmax(sci.flatten())
            else:
                ix = np.argmax((sci * (ivar > 0)).flatten())

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
            guess = [
                (sci - med_bkg)[yc - N : yc + N, xc - N : xc + N].sum(),
                xguess,
                yguess,
                med_bkg,
                0,
                0,
            ]
            _objfun = self.objective_epsf

        sly = slice(yc - N, yc + N)
        slx = slice(xc - N, xc + N)
        sly = slice(yguess - N, yguess + N)
        slx = slice(xguess - N, xguess + N)

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
            px = psf_params * 1
            if len(px) == 2:
                _objfun = self.objective_epsf_center
                px[0] += x0
                px[1] += y0
            else:
                _objfun = self.objective_epsf
                px[1] += x0
                px[2] += y0

            args = (self, psf_xy, sci, ivar_mask, xp, yp, extended_data, "model", ds9)
            psf_model, bkg, A, coeffs = _objfun(px, *args)
            return psf_model, bkg, A, coeffs

        if method == "lm":
            args = (self, psf_xy, sci, ivar_mask, xp, yp, extended_data, "lm", ds9)

            x_scale = "jac"
            # x_scale = [guess[0], 1, 1, 10, 10, 10]
            # out = least_squares(_objfun, guess, args=args, method='trf', x_scale=x_scale, loss='huber')
            out = least_squares(
                _objfun, guess, args=args, method="lm", x_scale=x_scale, loss="linear"
            )
            psf_params = out.x * 1
        else:
            args = (self, psf_xy, sci, ivar_mask, xp, yp, extended_data, "chi2", ds9)
            out = minimize(_objfun, guess, args=args, method=method, tol=tol)
            psf_params = out.x * 1

        if len(guess) == 2:
            psf_params[0] -= x0
            psf_params[1] -= y0
        else:
            psf_params[1] -= x0
            psf_params[2] -= y0

        return psf_params

    def get_ePSF(
        self,
        psf_params,
        sci=None,
        ivar=1,
        origin=[0, 0],
        shape=[20, 20],
        filter="F140W",
        get_extended=False,
        get_background=False,
        rot90=0,
    ):
        """
        Evaluate an Effective PSF

        Parameters
        ----------
        psf_params : list or tuple
            List or tuple of PSF parameters.

        sci : numpy.ndarray, optional
            Science image. Default is None.

        ivar : float, optional
            Inverse variance. Default is 1.

        origin : list, optional
            Origin of the PSF. Default is [0, 0].

        shape : list, optional
            Shape of the PSF. Default is [20, 20].

        filter : str, optional
            Filter name. Default is "F140W".

        get_extended : bool, optional
            Flag to get extended PSF. Default is False.

        get_background : bool, optional
            Flag to get background. Default is False.

        rot90 : int, optional
            Rotation angle. Default is 0.

        Returns
        -------
        output_psf : numpy.ndarray
            Output PSF.

        bkg : numpy.ndarray or None
            Background image if get_background is True, else None.

        """
        sh = shape
        y0, x0 = np.array(sh) / 2.0 - 1

        xd = x0 + origin[1]
        yd = y0 + origin[0]

        xc, yc = int(x0), int(y0)

        psf_xy = self.get_at_position(x=xd, y=yd, filter=filter, rot90=rot90)

        yp, xp = np.indices(sh)

        if len(psf_params) == 2:
            _objfun = self.objective_epsf_center
            dx = xp - psf_params[0] - x0
            dy = yp - psf_params[1] - y0
        else:
            _objfun = self.objective_epsf
            dx = xp - psf_params[1] - x0
            dy = yp - psf_params[2] - y0

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
            ivar_mask = sci * 1

        args = (
            self,
            psf_xy,
            sci,
            ivar_mask,
            xp - x0,
            yp - y0,
            extended_data,
            "model",
            None,
        )
        output_psf, bkg, _a, _b = _objfun(psf_params, *args)

        if get_background:
            return output_psf, bkg
        else:
            return output_psf


def read_catalog(file, sextractor=False, format=None):
    """
    Wrapper around `~grizli.utils.Gtable.gread`.

    Parameters
    ----------
    file : str
        The path to the catalog file.

    sextractor : bool, optional
        If True, assumes the catalog is in SExtractor format.
        Default is False.

    format : str, optional
        The format of the catalog file. Auto-detects formats
        'csv' and 'fits' and defaults to 'ascii.commented_header'.

    Returns
    -------
    table : `~astropy.table.Table`
        The catalog table.

    """
    return GTable.gread(file, sextractor=sextractor, format=format)


class GTable(astropy.table.Table):
    """
    Extend `~astropy.table.Table` class with more automatic IO and other
    helper methods.
    """

    @classmethod
    def gread(cls, file, sextractor=False, format=None):
        """
        Assume `ascii.commented_header` by default

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
                format = "ascii.sextractor"
            elif isinstance(file, pyfits.BinTableHDU):
                format = "fits"
            else:
                if file.endswith(".fits"):
                    format = "fits"
                elif file.endswith(".csv"):
                    format = "csv"
                elif file.endswith(".vot"):
                    format = "votable"
                else:
                    format = "ascii.commented_header"

        # print(file, format)
        tab = cls.read(file, format=format)

        return tab

    def gwrite(self, output, format="ascii.commented_header"):
        """
        Assume a format for the output table

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
        """
        Parse column names for RA/Dec and set to `~astropy.units.degree` units if not already set

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
            rd_pairs["RA"] = "DEC"
            rd_pairs["ALPHA_J2000"] = "DELTA_J2000"
            rd_pairs["X_WORLD"] = "Y_WORLD"
            rd_pairs["ALPHA_SKY"] = "DELTA_SKY"
            rd_pairs["_RAJ2000"] = "_DEJ2000"

            for k in list(rd_pairs.keys()):
                rd_pairs[k.lower()] = rd_pairs[k].lower()

        rd_pair = None
        for c in rd_pairs:
            if c in self.colnames:
                rd_pair = [c, rd_pairs[c]]
                break

        if rd_pair is None:
            # print('No RA/Dec. columns found in input table.')
            return False

        for c in rd_pair:
            if self[c].unit is None:
                self[c].unit = u.degree

        return rd_pair

    def match_to_catalog_sky(
        self,
        other,
        self_radec=None,
        other_radec=None,
        nthneighbor=1,
        get_2d_offset=False,
    ):
        """
        Compute `~astropy.coordinates.SkyCoord` projected matches between
        two `GTable` tables.

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

        get_2d_offset : bool, optional
            If True, compute the 2D offset between the matched coordinates.

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
            print("No RA/Dec. columns found in input table.")
            return False

        self_coo = SkyCoord(ra=self[rd[0]], dec=self[rd[1]], frame="icrs")

        if isinstance(other, list) | isinstance(other, tuple):
            rd = [slice(0, 1), slice(1, 2)]

        else:
            if other_radec is None:
                rd = self.parse_radec_columns(other)
            else:
                rd = self.parse_radec_columns(
                    other, rd_pairs={other_radec[0]: other_radec[1]}
                )

            if rd is False:
                print("No RA/Dec. columns found in `other` table.")
                return False

        other_coo = SkyCoord(ra=other[rd[0]], dec=other[rd[1]], frame="icrs")

        try:
            idx, d2d, d3d = other_coo.match_to_catalog_sky(
                self_coo, nthneighbor=nthneighbor
            )
        except:
            print("Couldn't run SkyCoord.match_to_catalog_sky with" "nthneighbor")

            idx, d2d, d3d = other_coo.match_to_catalog_sky(self_coo)

        if get_2d_offset:
            cosd = np.cos(self_coo.dec.deg / 180 * np.pi)
            dra = (other_coo.ra.deg - self_coo.ra.deg[idx]) * cosd[idx]
            dde = other_coo.dec.deg - self_coo.dec.deg[idx]
            return idx, d2d.to(u.arcsec), dra * 3600 * u.arcsec, dde * 3600 * u.arcsec
        else:
            return idx, d2d.to(u.arcsec)

    def match_triangles(
        self,
        other,
        self_wcs=None,
        x_column="X_IMAGE",
        y_column="Y_IMAGE",
        mag_column="MAG_AUTO",
        other_ra="X_WORLD",
        other_dec="Y_WORLD",
        pixel_index=1,
        match_kwargs={},
        pad=100,
        show_diagnostic=False,
        auto_keep=3,
        maxKeep=10,
        auto_limit=3,
        ba_max=0.99,
        scale_density=10,
    ):
        """
        Match sources between two catalogs using triangles

        Parameters
        ----------
        self : `GTable`
            The first catalog to match.

        other : `~astropy.table.Table` or `GTable` or `list`
            The second catalog to match.

        self_wcs : `~astropy.wcs.WCS`, optional
            The WCS object associated with the first catalog. If provided, the
            positions in `self` will be transformed to pixel coordinates using
            this WCS before matching.

        x_column : str, optional
            The column name in `self` that contains the x-coordinates of the
            sources. Default is "X_IMAGE".

        y_column : str, optional
            The column name in `self` that contains the y-coordinates of the
            sources. Default is "Y_IMAGE".

        mag_column : str, optional
            The column name in `self` that contains the magnitudes of the
            sources. Default is "MAG_AUTO".

        other_ra : str, optional
            The column name in `other` that contains the right ascension
            coordinates of the sources. Default is "X_WORLD".

        other_dec : str, optional
            The column name in `other` that contains the declination
            coordinates of the sources. Default is "Y_WORLD".

        pixel_index : int, optional
            The pixel index convention to use when transforming the positions
            from world coordinates to pixel coordinates. Default is 1.

        match_kwargs : dict, optional
            Additional keyword arguments to pass to the `match_catalog_tri`
            function.

        pad : float, optional
            The padding in pixels to apply to the bounding box of the second
            catalog. Default is 100.

        show_diagnostic : bool, optional
            If True, a diagnostic plot showing the matched sources will be
            created. Default is False.

        auto_keep : int, optional
            The number of matched sources to keep when performing the automatic
            matching. Default is 3.

        maxKeep : int, optional
            The maximum number of matched sources to keep. Default is 10.

        auto_limit : int, optional
            The maximum number of sources to use when performing the automatic
            matching. Default is 3.

        ba_max : float, optional
            The maximum axis ratio allowed when performing the automatic
            matching. Default is 0.99.

        scale_density : float, optional
            The scaling factor to apply to the number of sources in the second
            catalog when matching the surface densities of the two catalogs.
            Default is 10.

        Returns
        -------
        match_ix : `~numpy.ndarray`
            The indices of the matched sources in `self` and `other`.

        tf : `~numpy.ndarray`
            The transformation matrix that maps the positions in `self` to the
            positions in `other`.

        dx : `~numpy.ndarray`
            The residual offsets between the matched sources in `self` and
            `other` after applying the transformation.

        rms : float
            The root-mean-square residual of the matched sources.

        fig : `~matplotlib.figure.Figure`, optional
            The diagnostic plot showing the matched sources. Only returned if
            `show_diagnostic` is True.

        """
        from tristars import match

        if hasattr(other, "shape"):
            other_radec = other * 1.0
        else:
            other_radec = np.array([other[other_ra], other[other_dec]]).T

        self_xy = np.array([self[x_column], self[y_column]]).T

        # xy_drz = np.array([cat['X_IMAGE'][ok], cat['Y_IMAGE'][ok]]).T

        if self_wcs is None:
            other_xy = other_radec
            cut = (
                (other_xy[:, 0] > -pad)
                & (other_xy[:, 0] < self_xy[:, 0].max() + pad)
                & (other_xy[:, 1] > -pad)
                & (other_xy[:, 0] < self_xy[:, 1].max() + pad)
            )
            other_xy = other_xy[cut, :]

            xy_center = np.zeros(2)

        else:
            other_xy = self_wcs.all_world2pix(other_radec, pixel_index)
            if hasattr(self_wcs, "pixel_shape"):
                _naxis1, _naxis2 = self_wcs._naxis
            else:
                _naxis1, _naxis2 = self_wcs._naxis1, self_wcs._naxis2

            cut = (other_xy[:, 0] > -pad) & (other_xy[:, 0] < _naxis1 + pad)
            cut &= (other_xy[:, 1] > -pad) & (other_xy[:, 1] < _naxis2 + pad)

            other_xy = other_xy[cut, :]
            xy_center = self_wcs.wcs.crpix * 1

        if len(other_xy) < 3:
            print("Not enough sources in match catalog")
            return False

        self_xy -= xy_center
        other_xy -= xy_center

        ########
        # Match surface density of drizzled and reference catalogs
        if mag_column is not None:
            icut = np.minimum(len(self) - 2, int(scale_density * other_xy.shape[0]))
            self_ix = np.argsort(self[mag_column])[:icut]
        else:
            self_ix = np.arange(self_xy.shape[0])

        self_xy = self_xy[self_ix, :]

        pair_ix = match.match_catalog_tri(
            self_xy,
            other_xy,
            maxKeep=maxKeep,
            auto_keep=auto_keep,
            auto_transform=None,
            auto_limit=auto_limit,
            size_limit=[5, 1000],
            ignore_rot=False,
            ignore_scale=True,
            ba_max=ba_max,
        )

        if len(pair_ix) == 0:
            print("No matches")
            return False

        tf, dx, rms = match.get_transform(
            self_xy, other_xy, pair_ix, transform=None, use_ransac=True
        )

        match_ix = pair_ix * 1
        match_ix[:, 0] = self_ix[pair_ix[:, 0]]

        if show_diagnostic:
            fig = match.match_diagnostic_plot(
                self_xy, other_xy, pair_ix, tf=None, new_figure=True
            )
            return match_ix, tf, dx, rms, fig
        else:
            return match_ix, tf, dx, rms

    def add_aladdin(
        self,
        rd_cols=["ra", "dec"],
        fov=0.5,
        size=(400, 200),
        default_view="P/DSS2/color",
    ):
        """
        Add AladinLite DIV column to the table

        Parameters
        ----------
        rd_cols : list, optional
            The column names in `self` that contain the right ascension and
            declination coordinates of the sources. Default is ["ra", "dec"].

        fov : float, optional
            The field of view in degrees. Default is 0.5.

        size : tuple, optional
            The size of the DIVs in pixels (width, height). Default is (400, 200).

        default_view : str, optional
            The default view of the AladinLite image. Default is "P/DSS2/color".

        """
        # <!-- include Aladin Lite CSS file in the head section of your page -->
        # <link rel="stylesheet" href="//aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css" />
        #
        # <!-- you can skip the following line if your page already integrates the jQuery library -->
        # <script type="text/javascript" src="//code.jquery.com/jquery-1.12.1.min.js" charset="utf-8"></script>

        ala = [
            """    <div id="aladin-lite-div-{i}" style="width:{wsize}px;height:{hsize}px;"></div>
        <script type="text/javascript" src="http://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.js" charset="utf-8"></script>
        <script type="text/javascript">
            var aladin = A.aladin('#aladin-lite-div-{i}', xxxsurvey: "{survey}", fov:{fov}, target: "{ra} {dec}"yyy);
        </script></div>""".format(
                i=i,
                ra=row[rd_cols[0]],
                dec=row[rd_cols[1]],
                survey=default_view,
                fov=fov,
                hsize=size[1],
                wsize=size[0],
            )
            .replace("xxx", "{")
            .replace("yyy", "}")
            for i, row in enumerate(self)
        ]

        self["aladin"] = ala

    def write_sortable_html(
        self,
        output,
        replace_braces=True,
        localhost=True,
        max_lines=50,
        table_id=None,
        table_class="display compact",
        css=None,
        filter_columns=[],
        add_cone_search=True,
        buttons=["csv"],
        toggle=True,
        use_json=False,
        with_dja_css=False,
        timestamp=True,
    ):
        """
        Wrapper around `~astropy.table.Table.write(format='jsviewer')`.

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

        max_lines : int, optional
            Maximum number of lines to display in the table. Default is 50.

        table_id : str, optional
            ID attribute for the HTML table element.

        table_class : str, optional
            Class attribute for the HTML table element. 
            Default is "display compact".

        css : str, optional
            Additional CSS styles to apply to the table.

        filter_columns : list, optional
            Add option to limit min/max values of column data.

        buttons : list, optional
            Add buttons for exporting data. 
            Allowed options are 'copy', 'csv', 'excel', 'pdf', 'print'.

        toggle : bool, optional
            Add links at top of page for toggling columns on/off.

        use_json : bool, optional
            Write the data to a JSON file and strip out of the HTML header.
            Use this for large datasets or if columns include rendered images.

        with_dja_css : bool, optional
            Include additional CSS styles for Django admin interface.

        timestamp : bool, optional
            Add a timestamp to the output file.

        """
        import time

        # from astropy.table.jsviewer import DEFAULT_CSS
        DEFAULT_CSS = """
body {font-family: sans-serif;}
table.dataTable {width: auto !important; margin: 0 !important;}
.dataTables_filter, .dataTables_paginate {float: left !important; margin-left:1em}
td {font-size: 10pt;}
        """
        if css is not None:
            DEFAULT_CSS += css

        if with_dja_css:
            DEFAULT_CSS += """
select {display: inline; width:100px;}
input[type="search"] {display: inline; width:400px;}
        """

        if os.path.exists(output):
            os.remove(output)

        self.write(
            output,
            format="jsviewer",
            css=DEFAULT_CSS,
            max_lines=max_lines,
            jskwargs={"use_local_files": localhost},
            table_id=None,
            table_class=table_class,
        )

        if replace_braces:
            lines = open(output).readlines()
            if replace_braces:
                for i in range(len(lines)):
                    lines[i] = lines[i].replace("&lt;", "<")
                    lines[i] = lines[i].replace("&gt;", ">")

            fp = open(output, "w")
            fp.writelines(lines)
            fp.close()

        # Read all lines
        lines = open(output).readlines()

        if with_dja_css:
            for i, line in enumerate(lines):
                if "<style>" in line:
                    lines.insert(
                        i,
                        """
    <link rel="stylesheet" href="https://dawn-cph.github.io/dja/assets/css/main.css" />
        """,
                    )
                    break

        if "aladin" in self.colnames:
            # Insert Aladin CSS
            aladin_css = '<link rel="stylesheet" href="https://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.css" />\n'

            for il, line in enumerate(lines):
                if "<link href=" in line:
                    break

            lines.insert(il + 1, aladin_css)

        # Export buttons
        if buttons:
            # CSS
            css_script = '<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/1.5.1/css/buttons.dataTables.min.css">\n'
            for il, line in enumerate(lines):
                if "css/jquery.dataTable" in line:
                    break

            lines.insert(il + 1, css_script)

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
                if "js/jquery.dataTable" in line:
                    break
            lines.insert(il + 2, js_scripts)

            for il, line in enumerate(lines):
                if "pageLength" in line:
                    break

            button_str = "{spacer}dom: 'Blfrtip',\n{spacer}buttons: {bstr},\n"
            button_option = button_str.format(spacer=" " * 8, bstr=buttons.__repr__())
            lines.insert(il + 1, button_option)

        # Range columns
        ic_list = []

        filter_lines = ["<table>\n"]
        descr_pad = ' <span style="display:inline-block; width:10;"></span> '

        filter_input = """
    <tr>
        <td style="width:100px">
            <input type="text" id="{0}_min" name="{0}_min"  
                   style="width:60px;display:inline"> &#60;
        </td> 
        <td style="width:100px"> {0} </td> 
        <td>  &#60; <input type="text" id="{0}_max" name="{0}_max" 
                    style="width:60px;display:inline">
    """

        coord_box = None

        for ic, col in enumerate(self.colnames):
            if col in filter_columns:
                found = False
                for i in range(len(lines)):
                    if "<th>{0}".format(col) in lines[i]:
                        found = True
                        break

                if found:
                    # print(col)
                    ic_list.append(ic)

                    filter_lines += filter_input.format(col)

                    descr = "\n"
                    if hasattr(self.columns[col], "description"):
                        if self.columns[col].description is not None:
                            descr = "{0} {1}\n".format(
                                descr_pad, self.columns[col].description
                            )

                    filter_lines += descr + "  </tr>"

        if ic_list:
            # Insert input lines

            for il, line in enumerate(lines):
                if "} );  </script>" in line:
                    break

            # filter_row = '<tr> <td> <input type="text" id="{0}_min" name="{0}_min" style="width:40px;"> &#60; </td> <td style="align:center;"> <tt>{0}</tt> </td> <td>  &#60; <input type="text" id="{0}_max" name="{0}_max" style="width:40px;">'

            filter_rows = []
            for ic in ic_list:
                col = self.colnames[ic]
                row_i = filter_input.format(col)
                descr = "\n"
                if hasattr(self.columns[col], "description"):
                    if self.columns[col].description is not None:
                        descr = "{0} {1}\n".format(
                            descr_pad, self.columns[col].description
                        )

                filter_rows.append(row_i + descr)

            if ('ra' in filter_columns) & ('dec' in filter_columns) & add_cone_search:
                coord_box = (
                    ' <tr> <td> Cone search </td>'
                    ' <td colspan="2"> <input type="text" id="cone_search" name="cone_search" '
                    ' style="width:175px;display:inline"> </td> </tr>\n'
                )
                filter_rows.insert(0, coord_box)
            else:
                coord_box = None

            filter_input = """

<div style="border:1px solid black; padding:10px; margin:10px">
<b> Filter:</b>
    <table>
      {0}
    </table>
</div>

""".format(
                "\n".join(filter_rows),
            )

            lines.insert(il + 1, filter_input)

            # Javascript push function
            header_lines = ""
            tester = []

            for ic in ic_list:
                header_lines += """
        var min_{0} = parseFloat( $('#{0}_min').val()) || -1e30;
        var max_{0} = parseFloat( $('#{0}_max').val()) ||  1e30;
        var data_{0} = parseFloat( data[{1}] ) || 0;
                """.format(
                    self.colnames[ic], ic
                )

                tester.append(
                    """( ( isNaN( min_{0} ) && isNaN( max_{0} ) ) || ( isNaN( min_{0} ) && data_{0} <= max_{0} ) || ( min_{0} <= data_{0}  && isNaN( max_{0} ) ) || ( min_{0} <= data_{0}  && data_{0} <= max_{0} ) )""".format(
                        self.colnames[ic]
                    )
                )

            # Javascript filter function
            filter_function = r"""

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
    // Search text
    var search_text = $('input[type="search"]').val();
    if ((search_text != "") & (search_text != null)) {{
        filter_url += '&search=' + search_text;
    }}

    var cone_text = $('#cone_search').val();
    if ((cone_text != "") & (cone_text != null)) {{
        filter_url += '&cone_search=' + cone_text;
    }}

    // Table columns
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
    }} else {{
        window.history.pushState('', '', window.location.href.split('?')[0]);
    }}
}}
""".format(
                header_lines,
                "\n && ".join(tester),
                [self.colnames[ic] for ic in ic_list].__repr__(),
            )

            if coord_box is not None:
                filter_function += """
// Parse coordinates for cone search
$.parseCoord = function(rd, hours) {
    // Parse sexagesimal coordinates if ':' found in rd
    if (rd.includes(':')){
        var dms = rd.split(':')
        var deg = dms[0]*1;
        if (deg < 0) {
            var sign = -1;
        } else {
            var sign = 1;
        }
        deg += sign*dms[1]/60. + sign*dms[2]/3600.;
        if (hours > 0) {
            deg *= 360/24.
        }
    } else {
        var deg = rd;
    }
    return deg
}

$.updateConeSearch = function(){
    // Update ra dec filters using cone search
    var coord = $('#cone_search').val().trim();
    var rd = coord.split(',');
    if ((rd.length == 1)) {
        rd = coord.split(' ');
    }

    if ((rd.length > 1)) {
        var ra_deg = parseFloat($.parseCoord(rd[0], 1));
        var dec_deg = parseFloat($.parseCoord(rd[1], 0));
        if ((rd.length < 3)) {
            var cone_deg = 1. / 3600;
        } else {
            var cone_deg = parseFloat(rd[2]) / 3600;
        };

        var cosd = Math.cos(dec_deg / 180 * 3.14159);
        var ra_min_ = ra_deg - cone_deg / cosd;
        var ra_max_ = ra_deg + cone_deg / cosd;
        var dec_min_ = dec_deg - cone_deg;
        var dec_max_ = dec_deg + cone_deg;
        $('#ra_min').val(ra_min_.toFixed(6));
        $('#ra_max').val(ra_max_.toFixed(6));
        $('#dec_min').val(dec_min_.toFixed(6));
        $('#dec_max').val(dec_max_.toFixed(6));
    }
}

"""

            for i, line in enumerate(lines):
                if "$(document).ready(function()" in line:
                    istart = i
                    break

            lines.insert(istart, filter_function)

            # Parse address bar
            lines.insert(istart + 2, "\n")

            if coord_box is not None:
                lines.insert(
                    istart + 2,
                    "   $('#cone_search').val($.urlParam('cone_search'));\n"
                )

            for ic in ic_list:
                lines.insert(
                    istart + 2,
                    "{1}$('#{0}_max').val($.urlParam('{0}_max'));\n".format(
                        self.colnames[ic], " " * 3
                    ),
                )
                lines.insert(
                    istart + 2,
                    "{1}$('#{0}_min').val($.urlParam('{0}_min'));\n".format(
                        self.colnames[ic], " " * 3
                    ),
                )
            lines.insert(istart + 2, "\n")

            # Input listener
            listener = """
    // Initialize search from address bar
    var url_search = $.urlParam('search');
    if ((url_search != "") & (url_search != null)) {{
        table.search(url_search).draw();
    }}

    // Update address bar on search text
    $('input[type="search"]').keyup( function() {{
        $.UpdateFilterURL();
    }} );

    // Listener on cone search box
    $('#cone_search').keyup( function() {{
        $.updateConeSearch();
        table.draw();
        $.UpdateFilterURL();
    }} );

    // Event listener to the two range filtering inputs to redraw on input
    $('{0}').keyup( function() {{
        table.draw();
        $.UpdateFilterURL();
    }} );
            """.format(
                ", ".join(
                    ["#{0}_min, #{0}_max".format(self.colnames[ic]) for ic in ic_list]
                )
            )

            for il, line in enumerate(lines):
                if "dataTable(" in line:
                    lines[il] = "   var table = {0}\n".format(
                        lines[il].strip().replace("dataTable", "DataTable")
                    )
                elif "} );  </script>" in line:
                    break

            lines.insert(il, listener)

        fp = open(output, "w")
        fp.writelines(lines)
        fp.close()

        if toggle:
            lines = open(output).readlines()

            # # Change call to DataTable
            # for il, line in enumerate(lines):
            #     if "dataTable(" in line:
            #         break
            #
            # lines[il] = "   var table = {0}\n".format(
            #     lines[il].strip().replace("dataTable", "DataTable")
            # )

            # Add function
            for il, line in enumerate(lines):
                if "} );  </script>" in line:
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

            """.format(
                " <b>/</b> ".join(
                    [
                        '<a class="toggle-vis" data-column="{0}"> <tt>{1}</tt> </a>'.format(
                            ic, col
                        )
                        for ic, col in enumerate(self.colnames)
                    ]
                )
            )

            lines.insert(il + 2, toggle_div)
        else:
            # Without filter columns
            # Input listener
            listener = """
    // Parse location bar
    $.urlParam = function(name){
        var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(window.location.href);
        if (results==null){
           return null;
        }
        else{
           return decodeURI(results[1]) || 0;
        }
    };

    // Initialize search from address bar
    var url_search = $.urlParam('search');
    if ((url_search != "") & (url_search != null)) {
        table.search(url_search).draw();
    };

    // Listener to update address bar on search text
    $('input[type="search"]').keyup( function() {
        // Search text
        var search_text = $('input[type="search"]').val();
        if ((search_text != "") & (search_text != null)) {
            window.history.pushState(
                '',
                '',
                window.location.href.split('?')[0] + '?search=' + search_text
            );
        } else {
            window.history.pushState('', '', window.location.href.split('?')[0]);
        }
    } );\n"""

            for il, line in enumerate(lines):
                if "dataTable(" in line:
                    lines[il] = "    var table = {0}\n".format(
                        lines[il].strip().replace("dataTable", "DataTable")
                    )
                elif "} );  </script>" in line:
                    lines.insert(il, listener)
                    break

        # Insert timestamp
        if timestamp:
            for i, line in enumerate(lines[::-1]):
                if "</body>" in line:
                    # print('timestamp!',i)
                    lines.insert(
                        -(i + 1),
                        f'<span style="font-size:x-small;"> Created: {time.ctime()} </span>\n',
                    )
                    break

        fp = open(output, "w")
        fp.writelines(lines)
        fp.close()

        if use_json:
            # Write as json

            # Workaround to get ascii formatting
            # pd = self.to_pandas()
            new = GTable()
            for c in self.colnames:
                new[c] = self[c]

            if "aladin" in self.colnames:
                pd = GTable(new).to_pandas()
            else:
                new.write("/tmp/table.csv", format="csv", overwrite=True)
                pd = GTable.gread("/tmp/table.csv").to_pandas()

            # Reformat to json
            json_data = "        " + pd.to_json(orient="values").replace(
                "],[", "\n    ]xxxxxx\n    [\n        "
            ).replace(", ", "xcommaspacex").replace(",", ",\n        ").replace(
                "xxxxxx", ","
            ).replace(
                "xcommaspacex", ", "
            )
            json_str = """{{
  "data":
{0}

}}
""".format(
                json_data.replace('\\""', '"')
            )

            fp = open(output.replace(".html", ".json"), "w")
            fp.write(json_str)
            fp.close()

            # Edit HTML file
            lines = open(output).readlines()

            # Add ajax call to DataTable
            for il, line in enumerate(lines):
                if "pageLength" in line:
                    break

            ajax_call = (
                '{spacer}"ajax": "{json}",\n{spacer}"deferRender": true,\n'.format(
                    spacer=" " * 8, json=output.replace(".html", ".json")
                )
            )
            lines.insert(il + 1, ajax_call)

            # Strip out table body
            for ihead, line in enumerate(lines):
                if "</thead>" in line:
                    break

            for itail, line in enumerate(lines[::-1]):
                if "</tr>" in line:
                    break

            fp = open(output, "w")
            fp.writelines(lines[: ihead + 1])
            fp.writelines(lines[-itail:])
            fp.close()


def column_string_operation(col, test, method="contains", logical="or"):
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
    elif logical.upper() == "AND":
        return np.sum(arr, axis=1) >= sh[1]
    elif logical.upper() == "NOT":
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

    Parameters
    ----------
    x : array_like
        The x-coordinates of the points defining the curve.

    y0 : array_like
        The y-coordinates of the lower curve.

    y1 : array_like
        The y-coordinates of the upper curve.

    ax : `matplotlib.axes.Axes`, optional
        The axes on which to plot the filled region. 
        If not provided, the current axes will be used.

    *args : positional arguments
        Additional positional arguments to be passed to `fill_between`.

    **kwargs : keyword arguments
        Additional keyword arguments to be passed to `fill_between`.

    """
    import matplotlib.pyplot as plt

    so = np.argsort(x)
    dx = np.diff(x[so]) / 2.0
    mid = x[so][:-1] + dx

    xfull = np.hstack([x[so][0] - dx[0], mid, mid + dx * 2 / 1.0e6, x[so][-1] + dx[-1]])
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
    """
    Fill a covariance matrix in a larger array that had masked values

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


def log_scale_ds9(im, lexp=1.0e12, cmap=[7.97917, 0.8780493], scale=[-0.1, 10]):
    """
    Scale an array like ds9 log scaling

    Parameters
    ----------
    im : numpy.ndarray
        Input array to be scaled.

    lexp : float, optional
        Logarithmic scaling factor. Default is 1.0e12.

    cmap : list, optional
        List of contrast and bias values for the colormap. 
        Default is [7.97917, 0.8780493].

    scale : list, optional
        List of minimum and maximum values for scaling the input array. 
        Default is [-0.1, 10].

    Returns
    -------
    numpy.ndarray
        Scaled array using ds9 log scaling.

    """
    import numpy as np

    contrast, bias = cmap
    clip = (np.clip(im, scale[0], scale[1]) - scale[0]) / (scale[1] - scale[0])
    clip_log = np.clip(
        (np.log10(lexp * clip + 1) / np.log10(lexp) - bias) * contrast + 0.5, 0, 1
    )

    return clip_log


def mode_statistic(data, percentiles=range(10, 91, 10)):
    """
    Get modal value of a distribution of data following Connor et al. 2017
    https://arxiv.org/pdf/1709.01925.pdf

    Here we fit a spline to the cumulative distribution evaluated at knots
    set to the `percentiles` of the distribution to improve smoothness.

    Parameters
    ----------
    data : array_like
        The input data array.

    percentiles : array_like, optional
        The percentiles at which to evaluate the cumulative distribution
        function. Default is [10, 50, 90].

    Returns
    -------
    mode : float
        The modal value of the distribution.

    References
    ----------
    Connor, T., et al. 2017, ApJ, 848, 37

    """
    from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

    so = np.argsort(data)
    order = np.arange(len(data))
    # spl = UnivariateSpline(data[so], order)

    knots = np.percentile(data, percentiles)
    dx = np.diff(knots)
    mask = (data[so] >= knots[0] - dx[0]) & (data[so] <= knots[-1] + dx[-1])
    spl = LSQUnivariateSpline(data[so][mask], order[mask], knots, ext="zeros")

    mask = (data[so] >= knots[0]) & (data[so] <= knots[-1])
    ix = (spl(data[so], nu=1, ext="zeros") * mask).argmax()
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
    sp.params["logzsol"] = 0.2

    # Alf
    m = ssp.get_model(in_place=False, logage=0.96, zh=0.2, mgh=0.2)

    # FSPS
    w, spec = sp.get_spectrum(tage=10 ** 0.96, peraa=True)

    # blue
    blue_norm = spec[w > 3600][0] / m[ssp.wave > 3600][0]
    red_norm = spec[w > 1.7e4][0] / m[ssp.wave > 1.7e4][0]

    templx = np.hstack(
        [w[w < 3600], ssp.wave[(ssp.wave > 3600) & (ssp.wave < 1.7e4)], w[w > 1.7e4]]
    )
    temply = np.hstack(
        [
            spec[w < 3600] / blue_norm,
            m[(ssp.wave > 3600) & (ssp.wave < 1.7e4)],
            spec[w > 1.7e4] / red_norm,
        ]
    )

    np.savetxt(
        "alf_SSP.dat",
        np.array([templx, temply]).T,
        fmt="%.5e",
        header="wave flux\nlogage = 0.96\nzh=0.2\nmgh=0.2\nfsps: w < 3600, w > 1.7e4",
    )


def catalog_area(ra=[], dec=[], make_plot=True, NMAX=5000, buff=0.8, verbose=True):
    """
    Compute the surface area of a list of RA/DEC coordinates

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

    verbose : bool
        Print progress information.

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

    points = np.array([ra, dec]) * 1.0
    center = np.mean(points, axis=1)
    points = (points.T - center) * 60.0  # arcmin
    points[:, 0] *= np.cos(center[1] / 180 * np.pi)

    hull = spatial.ConvexHull(points)
    edge = points[hull.vertices, :]

    # pbuff = 1

    if len(ra) > NMAX:

        rnd_idx = np.unique(np.asarray(np.round(np.random.rand(NMAX)*len(ra)),dtype=int))

    else:
        rnd_idx = np.arange(len(ra))

    poly = Point(points[rnd_idx[0], :]).buffer(buff)
    for i, ix in enumerate(rnd_idx):
        if verbose:
            print(NO_NEWLINE + "{0} {1}".format(i, ix))

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

        ax.scatter(points[rnd_idx, 0], points[rnd_idx, 1], alpha=0.1, marker="+")

        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_xlabel(r"$\Delta$RA ({0:.5f})".format(center[0]))
        ax.set_ylabel(r"$\Delta$Dec. ({0:.5f})".format(center[1]))
        ax.set_title("Total area: {0:.1f} arcmin$^2$".format(pjoin.area))
        ax.grid()
        fig.tight_layout(pad=0.1)

        return pjoin.area, fig

    else:
        return pjoin.area


def bounding_polygon(x, y, nsteps=16, use_percentiles=True, closed=True, **kwargs):
    """
    Compute a bounding polygon of min(y), max(y) as a function of ``x``

    Parameters
    ----------
    steps : int
        Number of steps sampling from min(x) to max(x)

    use_percentiles : bool
        If true, use steps in evenly spaced percentiles of x, otherwise use
        evenly spaced steps

    closed : bool
        Close the polygon by duplicating the first point at the last

    Returns
    -------
    px, py : array-like
        Coordinates of the bounding polygon

    """

    if use_percentiles:
        xs = np.nanpercentile(x, np.linspace(0, 100, nsteps+1))
    else:
        xs = np.linspace(x.min(), x.max(), nsteps+1)

    xi = []
    yhi = []
    ylo = []
    for i in range(nsteps):
        sub = (x >= xs[i]) & (x <= xs[i+1])
        xi += [xs[i], xs[i+1]]
        yhi += [y[sub].max()]*2
        ylo += [y[sub].min()]*2

    px = xi + xi[::-1]
    py = yhi + ylo[::-1]
    if closed:
        px.append(xi[0])
        py.append(yhi[0])

    return np.array(px), np.array(py)


def catalog_bounding_polygon(ra, dec, cosd=True, scale=3600., nsteps=128, buffer=(25, -19), simplify=5., as_sregion=True, **kwargs):
    """
    Compute a bounding polygon for a list of catalog positions using the
    intersection of ``bounding_polygon(ra, dec)`` and
    ``bounding_polygon(dec, ra)``.

    Parameters
    ----------
    ra, dec : array-like
        Catalog positions.  These are assumed to be decimal degrees given the
        other parameter defaults, but they can be other values

    cosd : bool
        Rescale dx = (ra - median(ra)) by cos(dec)

    scale : float
        Additional scale factor of relative coordinates.  Default of 3600 scales
        sky coordinates in decimal degrees to offset arcsec

    nsteps : int
        Number of steps for the bounding polygons

    buffer : (float, float)
        Shapely buffers to apply to the intersection of the ``y(x)`` and ``x(y)``
        polygons.

    simplify : float
        Simplify tolerance on the intersection polygon.

        The combination of ``buffer`` and ``simplify`` reduce the complexity of
        the output.   For sky coordinates and ``scale=3600``, ``buffer``
        and ``simplify`` have units of arcsec.

        If ``simplify < buffer[0] + buffer[1]``, then the simplified shape should
        still contain all of the input points.

    as_region, kwargs : bool, **dict
        Return as `sregion.SRegion(**dict)` object

    Returns
    -------
    ro, do : array-like
        Coordinates of the bounding polygon, if ``as_sregion=False``

    olap : `sregion.SRegion`
        Region object if requested

    """
    from shapely.geometry import Polygon

    r0 = np.median(ra)
    d0 = np.median(dec)

    dr = (ra - r0) * scale * np.cos(d0/180*np.pi)**cosd
    dd = (dec - d0) * scale

    px, py = bounding_polygon(dr, dd, nsteps=nsteps, **kwargs)
    pxy = Polygon(np.array([px, py]).T)

    py, px = bounding_polygon(dd, dr, nsteps=nsteps, **kwargs)
    pyx = Polygon(np.array([px, py]).T)

    olap = pxy.intersection(pyx)
    if buffer is not None:
        olap = olap.buffer(buffer[0]).buffer(buffer[1])

    if simplify is not None:
        olap = olap.simplify(simplify)

    ro, do = np.array(olap.boundary.xy)

    ro = ro / scale / np.cos(d0/180*np.pi)**cosd + r0
    do = do / scale + d0

    if as_sregion:
        return SRegion(np.array([ro, do]), **kwargs)
    else:
        return ro, do


def fix_flt_nan(flt_file, bad_bit=4096, verbose=True):
    """
    Fix NaN values in FLT files

    Parameters
    ----------
    flt_file : str
        Path to the FLT file to fix NaN values.

    bad_bit : int, optional
        Bad bit value to be set in the DQ extension for NaN pixels. 
        Default is 4096.

    verbose : bool, optional
        If True, print information about the number of NaN pixels fixed. 
        Default is True.
    
    """
    im = pyfits.open(flt_file, mode="update")
    for ext in range(1, 5):
        if ("SCI", ext) in im:
            mask = ~np.isfinite(im["SCI", ext].data)
            if verbose:
                label = "utils.fix_flt_nan: {0}[SCI,{1}] NaNPixels={2}"
                print(label.format(flt_file, ext, mask.sum()))

            if mask.sum() == 0:
                continue

            im["SCI", ext].data[mask] = 0
            im["DQ", ext].data[mask] |= bad_bit

    im.flush()
    im.close()


def dump_flt_dq(filename, replace=(".fits", ".dq.fits.gz"), verbose=True):
    """
    Dump FLT/FLC header & DQ extensions to a compact file

    Parameters
    ----------
    filename : str
        FLT/FLC filename.

    replace : (str, str)
        Replace arguments for output filename:

        >>> output_filename = filename.replace(replace[0], replace[1])

    verbose : bool, optional
        Whether to display verbose output. Default is True.

    Returns
    -------
    Writes header and compact DQ array to `output_filename`.
    """
    im = pyfits.open(filename)
    hdus = []
    for i in [1, 2, 3, 4]:
        if ("SCI", i) in im:
            header = im["SCI", i].header
            dq = im["DQ", i].data
            nz = np.where(dq > 0)
            dq_data = np.array([nz[0], nz[1], dq[dq > 0]], dtype=np.int16)
            hdu = pyfits.ImageHDU(header=header, data=dq_data)
            hdus.append(hdu)

    output_filename = filename.replace(replace[0], replace[1])

    msg = "# dump_flt_dq: {0} > {1}".format(filename, output_filename)
    log_comment(LOGFILE, msg, verbose=verbose, show_date=True)

    pyfits.HDUList(hdus).writeto(output_filename, overwrite=True, output_verify="fix")

    im.close()


def apply_flt_dq(
    filename, replace=(".fits", ".dq.fits.gz"), verbose=True, or_combine=False
):
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

    verbose : bool, optional
        If True, print verbose output. Default is False.

    Returns
    -------
    Writes header and compact DQ array to `output_filename`.
    """

    output_filename = filename.replace(replace[0], replace[1])

    if not os.path.exists(output_filename):
        return False

    im = pyfits.open(filename, mode="update")

    msg = "# apply_flt_dq: {1} > {0}".format(filename, output_filename)
    log_comment(LOGFILE, msg, verbose=verbose, show_date=True)

    dq = pyfits.open(output_filename)
    for ext in [1, 2, 3, 4]:
        if (("SCI", ext) in im) & (("SCI", ext) in dq):
            sh = dq["SCI", ext].data.shape
            if sh[0] == 3:
                i, j, dq_i = dq["SCI", ext].data
            elif sh[1] == 2:
                nz, dq_i = dq["SCI", ext].data
                i, j = np.unravel_index(nz, sh)
            else:
                raise IOError("dq[{0}] shape {1} not recognized".format(ext, sh))

            # Apply DQ
            if or_combine:
                im["DQ", ext].data[i, j] != dq_i
            else:
                im["DQ", ext].data *= 0
                im["DQ", ext].data[i, j] = dq_i

            # Copy header
            has_blotsky = "BLOTSKY" in im["SCI", ext].header
            for k in dq["SCI", ext].header:
                if k in ["BITPIX", "NAXIS1", "NAXIS2", "", "HISTORY"]:
                    continue

                im["SCI", ext].header[k] = dq["SCI", ext].header[k]

            if (not has_blotsky) & ("BLOTSKY" in dq["SCI", ext].header):
                im["SCI", ext].header["BLOTSKY"] = False

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
        vals = [255 * x for x in vals[:3]]

    # Ensure values are rounded integers, convert to hex, and concatenate
    return "#" + "".join(["{:02X}".format(int(round(x))) for x in vals])


def catalog_mask(
    cat,
    ecol="FLUXERR_APER_0",
    max_err_percentile=90,
    pad=0.05,
    pad_is_absolute=False,
    min_flux_radius=1.0,
    min_nexp=2,
):
    """
    Compute a catalog mask for
      1) Objects within `pad` of the edge of the catalog convex hull
      2) Uncertainties < `max_err_percentile`

    Parameters
    ----------
    cat : `~astropy.table.Table`
        Input catalog.

    ecol : str, optional
        Name of the column containing the flux uncertainties. 
        Default is "FLUXERR_APER_0".

    max_err_percentile : float, optional
        Maximum percentile of flux uncertainties to include in the mask. 
        Default is 90.

    pad : float, optional
        Distance in pixels from the edge of the catalog convex hull
        to consider as the edge. Default is 0.05.

    pad_is_absolute : bool, optional
        If True, `pad` is interpreted as an absolute distance in pixels.
        If False, `pad` is interpreted as a fraction of the catalog size. 
        Default is False.

    min_flux_radius : float, optional
        Minimum value of the flux radius to include in the mask. Default is 1.0.

    min_nexp : int, optional
        Minimum number of exposures to include in the mask. Default is 2.

    Returns
    -------
    mask : `~numpy.ndarray`
        Boolean mask indicating which objects in the catalog pass the criteria.

    """
    test = np.isfinite(cat["FLUX_AUTO"])
    if "FLUX_RADIUS" in cat.colnames:
        test &= cat["FLUX_RADIUS"] > min_flux_radius

    test &= (cat["THRESH"] > 0) & (cat["THRESH"] < 1e28)

    not_edge = hull_edge_mask(
        cat["X_IMAGE"], cat["Y_IMAGE"], pad=pad, pad_is_absolute=pad_is_absolute
    )

    test &= not_edge

    if ecol in cat.colnames:
        valid = np.isfinite(cat[ecol])
        if max_err_percentile < 100:
            test &= cat[ecol] < np.percentile(
                cat[ecol][(~not_edge) & valid], max_err_percentile
            )

    if "NEXP" in cat.colnames:
        if cat["NEXP"].max() >= min_nexp:
            test_nexp = cat["NEXP"] >= min_nexp
            # print('xxx catalog_mask with nexp', len(cat), test_nexp.sum())
            if test_nexp.sum() > 0:
                test &= test_nexp

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
        buff = -pad * np.sqrt(poly.area)

    pbuff = poly.buffer(buff)
    in_buff = np.array([pbuff.contains(Point([x[i], y[i]])) for i in range(len(x))])

    return in_buff


def convex_hull_wrapper(x, y):
    """
    Generate a convex hull from a list of points

    Parameters
    ----------
    x : array-like
        x-coordinates of the points.
    y : array-like
        y-coordinates of the points.

    Returns
    -------
    pxy : tuple
        Tuple of hull vertices.
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
    
    Parameters
    ----------
    x : array-like
        x-coordinates of the points.
    y : array-like
        y-coordinates of the points.
        
    Returns
    -------
    float
        The area of the convex hull.
    
    """
    pxy, poly, hull = convex_hull_wrapper(x, y)

    return poly.area


def remove_text_labels(fig):
    """
    Remove all Text annotations from ``fig.axes``.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        The figure object from which to remove text labels.

    """
    import matplotlib

    for ax in fig.axes:
        for child in ax.get_children():
            if isinstance(child, matplotlib.text.Text):
                if child.get_text():  # Don't remove empty labels
                    child.set_visible(False)


LOGFILE = "/tmp/grizli.log"


def log_function_arguments(LOGFILE, frame, func="func", ignore=[], verbose=True):
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

    ignore : list
        Variable names to ignore

    verbose : bool
        Print messaage to stdout.

    """
    args = inspect.getargvalues(frame).locals
    args.pop("frame")
    for k in list(args.keys()):
        if hasattr(args[k], "__builtins__"):
            args.pop(k)

    for k in ignore:
        if k in args:
            args.pop(k)

    if func is not None:
        logstr = "\n{0}(**{1})\n"
    else:
        logstr = "\n{1}"

    logstr = logstr.format(func, args)
    msg = log_comment(LOGFILE, logstr, verbose=verbose, show_date=True)

    return args


def ctime_to_iso(
    mtime, format="%a %b %d %H:%M:%S %Y", strip_decimal=True, verbose=True
):
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
    if mtime.count("-") == 2:
        iso = mtime + ""

    else:
        try:
            iso = Time(datetime.strptime(mtime, format), format="datetime").iso
        except ValueError:
            if verbose:
                print(f"Couldn't convert '{mtime}' with " f"format '{format}'")

            iso = mtime + ""

    if strip_decimal:
        iso = iso.split(".")[0]

    return iso


def nowtime(iso=False):
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


def figure_timestamp(
    fig,
    x=0.97,
    y=0.02,
    iso=True,
    text_prefix='',
    manual_text=None,
    ha="right",
    va="bottom",
    fontsize=5,
    color="k",
    alpha=1.0,
):
    """
    Add a timestamp to a figure output

    Parameters
    ----------
    fig : `matplotlib` Figure
        Figure object

    x, y : float
        Label position in `fig.transFigure` coordinates (i.e., 0 < x,y < 1)

    iso : bool
        Use ISO-formatted time from `~grizli.utils.ctime_to_iso`, otherwise use
        `time.ctime()`

    text_prefix : str
        String to prepend to date

    manual_text : str
        Force string to use

    ha, va : str
        Horizontal and vertical alignment

    fontsize, color, alpha: int, str, float
        Label properties (in `matplotlib.Figure.text`)

    Returns
    -------
    Adds a timestamp to the `fig` object

    """
    import time

    time_str = time.ctime()

    if iso:
        time_str = ctime_to_iso(time_str, verbose=False)

    if manual_text is not None:
        text = manual_text
    else:
        text = text_prefix + time_str

    fig.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        transform=fig.transFigure,
        color=color,
        alpha=alpha,
    )


def log_comment(LOGFILE, comment, verbose=False, show_date=False, mode="a", **kwargs):
    """
    Log a message to a file, optionally including a date tag

    Parameters
    ----------
    LOGFILE : str
        The path to the log file.

    comment : str
        The message to be logged.

    verbose : bool, optional
        If True, the logged message will also be printed to the console.
        Default is False.

    show_date : bool, optional
        If True, the logged message will include a date tag. Default is False.

    mode : str, optional
        The mode in which the log file will be opened. Default is "a" (append).

    Returns
    -------
    msg : str
        The logged message.

    """
    import time

    if show_date:
        msg = "# ({0})\n".format(nowtime().iso)
    else:
        msg = ""

    msg += "{0}\n".format(comment)

    if LOGFILE is not None:
        fp = open(LOGFILE, mode)
        fp.write(msg)
        fp.close()

    if verbose:
        print(msg[:-1])

    return msg


def log_exception(LOGFILE, traceback, verbose=True, mode="a"):
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
    log = "\n########################################## \n"
    log += "# ! Exception ({0})\n".format(nowtime().iso)
    log += "#\n# !" + "\n# !".join(trace.split("\n"))
    log += "\n######################################### \n\n"
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

    Parameters
    ----------
    Om0 : float, optional
        Density of non-relativistic matter at z=0. Default is 0.3.

    Ode0 : float, optional
        Density of dark energy at z=0. Default is 0.7.

    H0 : float, optional
        Hubble constant at z=0 in km/s/Mpc. Default is 70.

    Ob0 : float, optional
        Density of baryonic matter at z=0. Default is 0.0463.

    Tcmb0 : float, optional
        CMB temperature at z=0 in K. Default is 2.725.

    name : str, optional
        Name of the cosmology. Default is None.

    Returns
    -------
    cosmology : `~astropy.cosmology.LambdaCDM`
        LambdaCDM cosmology object.

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

    Parameters
    ----------
    filter_size : int, optional
        The size of the filter. Default is 71.

    filter_central : int, optional
        The central region of the filter to be set to 0. Default is 0.

    **kwargs
        Additional keyword arguments.

    Returns
    -------
    filter_footprint : numpy.ndarray
        The filter footprint as a numpy array.

    """
    filter_footprint = np.ones(filter_size, dtype=int)

    if filter_central > 0:
        f0 = (filter_size - 1) // 2
        filter_footprint[f0 - filter_central : f0 + filter_central] = 0

    return filter_footprint


def safe_nanmedian_filter(
    data, filter_kwargs={}, filter_footprint=None, axis=1, clean=True, cval=0.0
):
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

        _filter_name = "nbutils.nanmedian"
    except:
        nbutils = None
        _filter_name = "median_filter"

    if filter_footprint is None:
        _filter_footprint = make_filter_footprint(**filter_kwargs)
        if axis == 1:
            _filter_footprint = _filter_footprint[None, :]
        else:
            _filter_footprint = _filter_footprint[:, None]
    else:
        if filter_footprint.ndim == 1:
            if axis == 1:
                _filter_footprint = filter_footprint[None, :]
            else:
                _filter_footprint = filter_footprint[:, None]
        else:
            _filter_footprint = filter_footprint

    if nbutils is None:
        filter_data = nd.median_filter(data, footprint=_filter_footprint)
    else:
        filter_data = nd.generic_filter(
            data, nbutils.nanmedian, footprint=_filter_footprint
        )
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
        if not arg.startswith("-"):
            # Arguments
            try:
                args.append(json.loads(arg))
            except:
                args.append(json.loads(f'"{arg}"'))

            continue

        spl = arg.strip("--").split("=")
        if len(spl) > 1:
            # Parameter values
            key, val = spl
            val = val.replace("True", "true").replace("False", "false")
            val = val.replace("None", "null")
        else:
            # Parameters, set to true, e.g., -set_flag
            key, val = spl[0], "true"

            # single -
            if key.startswith("-"):
                key = key[1:]

        # List values
        if "," in val:
            try:
                # Try parsing with JSON
                jval = json.loads(f"[{val}]")
            except:
                # Assume strings
                str_val = ",".join([f'"{v}"' for v in val.split(",")])
                jval = json.loads(f"[{str_val}]")
        else:
            try:
                jval = json.loads(val)
            except:
                # String
                jval = val

        # Dict keys, potentially nested
        if dot_dict & ("." in key):
            keys = key.split(".")
            d = kwargs
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}

                d = d[k]

            d[keys[-1]] = jval
        else:
            kwargs[key] = jval
            
    return args, kwargs


class HubbleXYZ(object):
    def __init__(self, spt_file="", param_dict={}):
        """
        Initialize the class instance.

        Parameters
        ----------
        spt_file : str 
            Path to the spt file. 
            If provided, the parameters will be parsed from the spt file.

        param_dict : dict
            Dictionary containing the parameter values. 
            If provided, the spt file will be ignored.

        Attributes
        ----------    
        param_dict : dict
            Dictionary containing the parameter values.

        computed : dict 
            Dictionary to store computed values.

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

        t0 = astropy.time.Time("1985-01-01T00:00:00Z")
        return t0

    def __call__(self, t_in, **kwargs):
        """
        Convert input time ``t_in`` to seconds since 1/1/85 and ``evaluate``.

        Parameters
        ----------
        t_in : `~astropy.time.Time`
            Time(s) to convert to seconds since 1/1/85.

        **kwargs : dict, optional
            Additional keyword arguments to pass to the `evaluate` method.

        Returns
        -------
        xyz : dict
            Dictionary containing the x, y, z, and r values.

        """
        import astropy.time

        if not isinstance(t_in, astropy.time.Time):
            raise ValueError("t_in must be astropy.time.Time object")

        dt = t_in - self._t1985
        xyz = self.evaluate(dt.sec, **kwargs)
        if "as_table" in kwargs:
            if kwargs["as_table"]:
                xyz["time"] = t_in

        return xyz

    def __getitem__(self, key):
        return self.param_dict[key]

    def evaluate(self, dt, unit=None, as_table=False):
        """
        Evaluate equations to get positions

        Parameters
        ----------
        dt : float or `~astropy.time.Time`
            Time difference(s) from the reference time in seconds.

        unit : str, optional
            Unit of the output coordinates. Default is None, which returns
            coordinates in kilometers.

        as_table : bool, optional
            If True, return the coordinates as an `~astropy.table.Table`
            object. Default is False.

        Returns
        -------
        x : float or `~astropy.table.Table`
            X-coordinate(s) in the specified unit.

        y : float or `~astropy.table.Table`
            Y-coordinate(s) in the specified unit.

        z : float or `~astropy.table.Table`
            Z-coordinate(s) in the specified unit.

        r : float or `~astropy.table.Table`
            Distance(s) from the origin in the specified unit.

        """

        if not self.param_dict:
            raise ValueError("Orbital parameters not defined in " "self.param_dict")

        p = self.param_dict

        t = np.atleast_1d(dt)

        # Eq. 1
        bracket = p["M."] * (t - p["tau"]) + 0.5 * p["M.."] * (t - p["tau"]) ** 2
        M = p["M0"] + 2 * np.pi * bracket

        # Eq. 2
        sinM = np.sin(M)
        cosM = np.cos(M)
        e = p["e"]
        nu = M + sinM * (
            2 * e
            + 3 * e ** 3 * cosM ** 2
            - 4.0 / 3 * e ** 3 * sinM ** 2
            + 5.0 / 2 * e ** 2 * cosM
        )

        # Eq. 3
        r = p["a(1-e**2)"] / (1 + e * np.cos(nu))
        # To km
        r /= 1000.0

        # Eq. 4
        Om = 2 * np.pi * (p["Om0"] + p["Om."] * (t - p["tau"]))

        # Eq. 5
        w = 2 * np.pi * (p["w0"] + p["w."] * (t - p["tau"]))

        self.calc_dict = {
            "M": M,
            "nu": nu,
            "a": p["a"],
            "i": np.arcsin(p["sini"]),
            "Om": Om,
            "w": w,
        }

        # Eq. 6
        cosOm = np.cos(Om)
        sinOm = np.sin(Om)
        coswv = np.cos(w + nu)
        sinwv = np.sin(w + nu)

        if unit is not None:
            r = (r * u.km).to(unit)

        x = r * (cosOm * coswv - p["cosi"] * sinOm * sinwv)
        y = r * (sinOm * coswv + p["cosi"] * cosOm * sinwv)
        z = r * p["sini"] * sinwv

        if as_table:
            tab = GTable()
            tab["dt"] = t
            tab["x"] = x
            tab["y"] = y
            tab["z"] = z
            tab["r"] = r
            return tab
        else:
            return x, y, z, r

    def from_flt(self, flt_file, **kwargs):
        """
        Compute positions at expstart, expmid, expend.

        Parameters
        ----------
        flt_file : str
            Path to the FLT file.

        **kwargs : dict, optional
            Additional keyword arguments to pass to the `evaluate` method. 

        Returns
        -------
        xyz : dict
            Dictionary containing the x, y, z, and r values.

        """
        import astropy.time

        flt = pyfits.open(flt_file)
        expstart = flt[0].header["EXPSTART"]
        expend = flt[0].header["EXPEND"]
        expmid = (expstart + expend) / 2.0

        t_in = astropy.time.Time([expstart, expmid, expend], format="mjd")
        flt.close()

        return self(t_in, **kwargs)

    def deltat(self, dt):
        """
        Convert a time ``t`` in seconds from 1/1/85 to an ISO time

        Parameters
        ----------
        dt : float
            Time difference from the reference time in seconds.

        Returns
        -------
        iso_time : str
            ISO formatted time string.

        """
        if not hasattr(dt, "unit"):
            dtsec = dt * u.second
        else:
            dtsec = dt

        t = self._t1985 + dtsec
        return t
    
    def parse_from_spt(self, spt_file):
        """
        Get orbital elements from SPT header

        Parameters
        ----------
        spt_file : str
            Path to the SPT file.

        Returns
        -------
        param_dict : dict
            Dictionary containing the orbital parameters parsed 
            from the SPT header.

        """
        import astropy.io.fits as pyfits
        import astropy.time

        with pyfits.open(spt_file) as _im:
            spt = _im[0].header.copy()

        param_dict = {}
        param_dict["tau"] = spt["EPCHTIME"]
        param_dict["M0"] = spt["MEANANOM"]
        param_dict["M."] = spt["FDMEANAN"]
        param_dict["M.."] = spt["SDMEANAN"]
        param_dict["e"] = spt["ECCENTRY"]
        param_dict["a(1-e**2)"] = spt["SEMILREC"]
        param_dict["a"] = param_dict["a(1-e**2)"] / (1 - param_dict["e"] ** 2)
        param_dict["Om0"] = spt["RASCASCN"]
        param_dict["Om."] = spt["RCASCNRV"]
        param_dict["w0"] = spt["ARGPERIG"]
        param_dict["w."] = spt["RCARGPER"]
        param_dict["cosi"] = spt["COSINCLI"]
        param_dict["sini"] = spt["SINEINCL"]
        param_dict["Vc"] = spt["CIRVELOC"]
        param_dict["timeffec"] = spt["TIMEFFEC"]
        param_dict["Torb"] = spt["HSTHORB"] * 2
        param_dict["tstart"] = spt["OBSSTRTT"]

        param_dict["tau_time"] = self.deltat(param_dict["tau"])
        param_dict["tstart_time"] = self.deltat(param_dict["tstart"])

        return param_dict

    @staticmethod
    def xyz_to_lonlat(self, x, y, z, radians=False):
        """
        Compute sublon, sublat, alt from xyz coords with pyproj

        Parameters
        ----------
        x : int
            X-coordinate(s) in meters.

        y : int
            Y-coordinate(s) in meters.

        z : int
            Z-coordinate(s) in meters.

        radians : bool, optional
            If True, the output longitude and latitude will be in radians.
            If False, the output longitude and latitude will be in degrees.
            Default is False.

        Returns
        -------
        lon : float or array-like
            Longitude(s) in degrees or radians, 
            depending on the value of `radians`.

        lat : float or array-like
            Latitude(s) in degrees or radians, 
            depending on the value of `radians`.

        alt : float or array-like
            Altitude(s) in meters.

        """
        import pyproj

        ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
        lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
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
        the_file = f"{site_packages}/../drizzlepac/haputils/align_utils.py"
        with open(the_file, "r") as fp:
            lines = fp.readlines()

        print(site_packages, len(lines))
        for i, line in enumerate(lines):
            if line.startswith("NoDetectionsWarning"):
                break

        if "hasattr(photutils.findstars" in lines[i + 1]:
            print(f"I found the problem on lines {i}-{i+2}: ")
        else:
            msg = """
Lines {0}-{1} in {2} importing photutils were not as expected.  I found 

{3}

but expected 

   NoDetectionsWarning = photutils.findstars.NoDetectionsWarning if \\
                           hasattr(photutils.findstars, 'NoDetectionsWarning') else \\
                           photutils.utils.NoDetectionsWarning


    """.format(
                i, i + 2, the_file, lines[i : i + 3]
            )

            raise ValueError(msg)

        bad = ["   " + lines.pop(i + 2 - j) for j in range(3)]
        print("".join(bad[::-1]))

        # Insert the fix
        lines[i] = "from photutils.utils.exceptions import NoDetectionsWarning\n\n"
        # Rewrite the fie
        with open(f"{site_packages}/../drizzlepac/haputils/align_utils.py", "w") as fp:
            fp.writelines(lines)

        print(f"Patch applied to {the_file}!")

def find_peaks(signal,threshold=0.5,min_dist=1):
    """
    Find peaks in a signal using `scipy.signal.find_peaks`
    Rescales input based on PeakUtils implementation

    Parameters
    ----------
    signal : array-like
        The input signal
    threshold : float
        The relative threshold for peak detection
    min_dist : int
        The minimum distance between peaks
        e.g. a distance of 1 are adjacent peaks
        differs from the PeakUtils implementation where adjacent peaks are seperated by 0
    

    Returns
    -------
    peaks : array-like
        The indices of the peaks in the signal
    """

    # Import required packages
    import scipy.signal

    # Calculate absolute height
    smin = signal.min() # Only calculate this once
    height = threshold*(signal.max()-smin)+smin

    # Find peaks
    peaks,_ = scipy.signal.find_peaks(signal,height=height,distance=min_dist)

    return peaks
