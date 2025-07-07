"""
Extract a NIRCam WFSS spectrum for a position on the sky
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from .. import jwst_utils

jwst_utils.set_quiet_logging()

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.units as u
from astropy.coordinates import SkyCoord

import scipy.ndimage as nd

try:
    from skimage import morphology
except ImportError:
    morphology = None

from .. import utils, grismconf, multifit
from . import db, api_cutout

S3PATH = "s3://grizli-v2/HST/Pipeline/{assoc}/Prep/{dataset}_rate.fits"

THUMB = "https://grizli-cutout.herokuapp.com/thumb?all_filters=False&filters={filter}&ra={ra}&dec={dec}&size={size}&output=fits_weight"


def coordinate_string(ra=189.0706488, dec=62.2089502, precision=2):
    """
    Generate a sexagesimal string from decimal coordinates

    Parameters
    ----------
    ra, dec : float
        Decimal degrees R.A. and declination

    precision: int
        Precision factor for `~astropy.coordinates.SkyCoord.to_string`

    Returns
    -------
    center_coord : `~astropy.coordinates.SkyCoord`
        Coordinate object

    coostr : string
        Sexagesimal string

    """
    center_coord = SkyCoord(f"{ra} {dec}", unit="deg")

    coostr = center_coord.to_string(
        style="hmsdms",
        sep="",
        precision=precision,
    ).replace(" ", "")

    return center_coord, coostr


def thumbnail_cutout(
    ra=189.0706488,
    dec=62.2089502,
    cutout_size=(60 * 0.05 * u.arcsec),
    filter="F444W-CLEAR",
    local=True,
    prefix="cutout",
    skip_existing=True,
    verbose=True,
    **kwargs,
):
    """ """
    from astropy.io import fits
    from astropy.coordinates import SkyCoord

    center_coord, coostr = coordinate_string(ra=ra, dec=dec, precision=2)

    if "," in filter:
        fkey = "ir"
    else:
        fkey = filter.lower()

    cutout_file = f"{prefix}_{coostr}_{fkey}.fits"
    if os.path.exists(cutout_file) & skip_existing:
        msg = f"found thumbnail file: {cutout_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        return cutout_file

    if hasattr(cutout_size, "unit"):
        size = cutout_size.to(u.arcsec).value
    else:
        size = cutout_size

    thumb_url = THUMB.format(ra=ra, dec=dec, size=size, filter=filter.lower())

    if local:
        kws = thumb_url.split("thumb?")[1].split("&")
        kwargs = {}
        for k in kws:
            key, value = k.split("=")
            kwargs[key] = value

        msg = f"api_cutout.get_thumb(**{kwargs}) -> {cutout_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        hdu = api_cutout.get_thumb(**kwargs)
    else:
        msg = f"{thumb_url} -> {cutout_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        hdu = fits.open(thumb_url)

    N = len(hdu) // 2
    if N > 2:
        num = hdu[0].data * 0.0
        den = hdu[0].data * 0
        for i in range(N):
            msg = f"thumbnail: {hdu[i*2+0].header['FILTER']}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            num += hdu[i * 2 + 0].data * hdu[i * 2 + 1].data
            den += hdu[i * 2 + 1].data

        sci = num / den
        sci[den <= 0] = 0
        hdu[0].data = sci
        hdu[1].data = den

        fkey = "ir"
    else:
        fkey = hdu[0].header["FILTER"]

    cutout_file = f"{prefix}_{coostr}_{fkey}.fits"

    hdu[:2].writeto(cutout_file, overwrite=True)
    hdu.close()

    return cutout_file


def s3_cutout(
    ra=189.0706488,
    dec=62.2089502,
    cutout_size=(60 * 0.05 * u.arcsec),
    s3_file="s3://grizli-v2/Scratch/gdn-grizli-v7.4-f444w-clear_drc_sci.fits",
    prefix="cutout",
    ext=0,
    get_hdul=False,
    correct_wcs=False,
    verbose=False,
):
    """ """
    from astrocut import fits_cut
    from astropy.io import fits
    import astrocut.cutouts
    from astropy.coordinates import SkyCoord

    center_coord, coostr = coordinate_string(ra=ra, dec=dec, precision=2)

    fits_options = {}
    fits_options["use_fsspec"] = True
    fits_options["fsspec_kwargs"] = {"default_block_size": 10_000, "anon": True}

    cutout_file = f"{prefix}_{coostr}.fits"
    if os.path.exists(cutout_file) & (not get_hdul):
        return cutout_file

    msg = f"{s3_file} ({ra:.6f},{dec:.6f}) {cutout_size} -> {cutout_file}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    with pyfits.open(s3_file, **fits_options) as hdul:
        cutout = astrocut.cutouts._hducut(
            hdul[ext],
            center_coord,
            (cutout_size, cutout_size),
            correct_wcs=correct_wcs,
            verbose=verbose,
        )

    cutout.header["COOSTR"] = (coostr, "Coordinate identifier")
    hdul = pyfits.PrimaryHDU(data=cutout.data, header=cutout.header)

    if get_hdul:
        return hdul

    hdul.writeto(cutout_file, overwrite=True)

    return cutout_file


def wfss_exposure_footprint(
    header, ypad=32, xpad=8, use_jwst_crds=True, verbose=False, **kwargs
):
    """
    Calculate sky footprint that should disperse onto the detector

    Parameters
    ----------
    header : `astropy.io.fits.Header` or `astropy.table.Row`
        Information for generating an exposure header with astrometry and filter
        keywords or columns

    use_jwst_crds : bool
        Use CRDS trace files

    Returns
    -------
    conf_params : dict
        Parameters used to define the configuration and derived wfss offsets

    """
    import astropy.table
    import grizli.grismconf

    try:
        _ = grizli.grismconf.LoadedConfig
    except:
        grizli.grismconf.LoadedConfig = {}

    if isinstance(header, astropy.table.Row):
        row = {}
        rowd = header
        for c in rowd.colnames:
            if hasattr(rowd[c], "mask"):
                continue
            elif c in ["footprint"]:
                continue
            else:
                row[c] = rowd[c]

        for i in [1, 2]:
            for j in [1, 2]:
                row[f"cd{i}_{j}"] = row[f"cd{i}{j}"]

        # Distortion ?
        row["cd1_2"] *= 4.641 / 3.279
        row["cd2_2"] *= 4.641 / 3.279

        header = pyfits.Header(row)

    wcs = pywcs.WCS(header)
    conf_params = dict(
        instrume=rowd["instrume"],
        grism=rowd["filter"].split("-")[0],
        filter=rowd["pupil"],
        module=rowd["detector"][3],
        use_jwst_crds=use_jwst_crds,
    )
    pkey = " ".join([f"{conf_params[k]}" for k in conf_params])

    if pkey in grizli.grismconf.LoadedConfig:
        conf = grizli.grismconf.LoadedConfig[pkey]
    else:
        conf_file = grizli.grismconf.get_config_filename(**conf_params)
        conf = grizli.grismconf.TransformGrismconf(conf_file)
        conf.get_beams()
        grizli.grismconf.LoadedConfig[pkey] = conf

    # Calculate footprint
    xc = 1024.5
    dy = [xc - ypad, xc + ypad]
    dx = [xc + conf.dxlam["A"][0] - xpad, xc + conf.dxlam["A"][-1] + xpad]
    dxr, dyr = xc - conf.transform.reverse(dx, dy)

    xwfss = np.array([dxr.min(), 2047 + dxr.max(), 2047 + dxr.max(), dxr.min()])
    ywfss = np.array([dyr.min(), dyr.min(), 2047 + dyr.max(), 2047 + dyr.max()])

    gsky = wcs.all_pix2world(np.array([xwfss, ywfss]).T, 0)
    wfss_footprint = utils.SRegion(gsky)

    conf_params["corners"] = gsky
    conf_params["xwfss"] = xwfss
    conf_params["ywfss"] = ywfss
    conf_params["footprint"] = wfss_footprint
    conf_params["orig_wcs"] = wcs
    conf_params["conf"] = conf

    return conf_params


def config_trace_sky(ra, dec, conf_params, xpad=12, ypad=12, **kwargs):
    """
    Estimate sky coordinates of a dispersed trace given exposure WCS
    """
    wcs = conf_params["orig_wcs"]
    conf = conf_params["conf"]

    xpix, ypix = np.squeeze(wcs.all_world2pix([ra], [dec], 0))
    xtr, ytr = np.squeeze(conf.transform.forward(xpix, ypix))
    dy, lam = conf.get_beam_trace(xtr, ytr, dx=conf.dxlam["A"], beam="A")

    xdet, ydet = conf.transform.reverse(xtr + conf.dxlam["A"], ytr + dy)
    tra, tdec = wcs.all_pix2world(xdet, ydet, 0)
    valid = (
        (xdet > -xpad) & (xdet < 2048 + xpad) & (ydet > -ypad) & (ydet < 2048 + ypad)
    )

    return tra, tdec, valid


def trim_wfss_exposure_overlaps(
    ra=189.0706488, dec=62.2089502, exp={}, verbose=True, **kwargs
):
    """
    Trim exposure list to those that should have dispersed spectra
    """
    exp["has_grism"] = False

    for i, row in enumerate(exp):
        config = wfss_exposure_footprint(row, **kwargs)
        wfss_footprint = config["footprint"]
        exp["has_grism"][i] = wfss_footprint.path[0].contains_point((ra, dec))

    msg = (
        f"trim_wfss_exposure_overlaps ({ra:.5f},{dec:.5f}): "
        + f" kept {exp['has_grism'].sum()} / {len(exp)} exposures"
    )
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    return exp[exp["has_grism"]]


def query_exposures(with_api=False, trim_exposures=True, **kwargs):
    """
    Wrapper around query_exposures_api / db

    Parameters
    ----------
    with_api : bool
        Use API search (vs. direct DB query)

    trim_exposures : bool
        Calculate grism spectrum overlaps with
        `grizli.aws.sky_wfss.trim_wfss_exposure_overlaps`

    Returns
    -------
    exp : `~astropy.table.Table`
        Exposure list
    """
    if with_api:
        exp = query_exposures_api(**kwargs)
    else:
        exp = query_exposures_db(**kwargs)

    if len(exp) > 0:
        so = np.argsort(exp["file"])
        exp = exp[so]

    if trim_exposures:
        exp = trim_wfss_exposure_overlaps(exp=exp, **kwargs)

    return exp


def query_exposures_api(
    ra=189.0706488,
    dec=62.2089502,
    radius=120.0,
    filters=["F356W-GRISMR", "F410M-GRISMR", "F444W-GRISMR"],
    verbose=False,
    **kwargs,
):
    """
    Query exposure metadata from API
    """
    if radius > 0:
        coo_str = f"polygon=rect({ra:.6f},{dec:.6f},{radius/60:.2f})"
    else:
        coo_str = f"coords={ra:.6f},{dec:.6f}"

    filt_str = ",".join(filters)

    API_URL = (
        "https://grizli-cutout.herokuapp.com/exposures?"
        + f"{coo_str}&filters={filt_str}&output=csv"
    )

    if verbose:
        print(API_URL)

    exp = utils.read_catalog(API_URL, format="csv")

    return exp


def query_exposures_db(
    ra=189.0706488,
    dec=62.2089502,
    radius=120.0,
    filters=["F356W-GRISMR", "F410M-GRISMR", "F444W-GRISMR"],
    verbose=False,
    **kwargs,
):
    """
    Query exposure metadata from DB
    """
    if radius > 0:
        exposure_query = f"""
        select * from exposure_files 
        where polygon(footprint) && polygon(circle(point({ra},{dec}), {radius/3600}))
        AND filter in ({(','.join(db.quoted_strings(filters))).upper()})
        order by assoc, dataset
        """

    else:
        exposure_query = f"""
    select * from exposure_files 
    where polygon(footprint) @> point({ra},{dec})
    AND filter in ({(','.join(db.quoted_strings(filters))).upper()})
    order by assoc, dataset
    """

    if verbose:
        print(exposure_query)

    exp = db.SQL(exposure_query)

    return exp


def id_from_segmentation(segmentation_image, ra, dec, verbose=True):
    """
    Get ID from segmentation image

    Parameters
    ----------
    segmentation_image : str
        Filename of segmentation image

    ra, dec : float
        Coordinates in decimal degrees

    Returns
    -------
    segment_id : int
        Value of `segmentation_image` at specified coordinates

    """
    with pyfits.open(segmentation_image) as im:
        wcs = pywcs.WCS(im[0].header)
        xp, yp = np.squeeze(wcs.all_world2pix([ra], [dec], 0)).astype(int)
        try:
            segment_id = im[0].data[yp, xp]
            msg = f"{segmentation_image} @ ({ra:.5f},{dec:.5f}) = {segment_id}"
        except IndexError:
            segment_id = 0
            msg = f"({ra:.5f},{dec:.5f}) not in {segmentation_image}"

    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    return segment_id


def simple_zeroth_mask(
    grp, dilation=None, erosion=8, update=True, verbose=True, **kwargs
):
    """
    Compute a simplified mask for zeroth orders shifting the segmentation image

    Parameters
    ----------
    grp : `~grizli.multifit.GroupFLT`
        Exposure group object

    dilation : int, None
        Number of mask dilation iterations

    erosion : int, None
        Number of mask erosion iterations

    update : bool
        Apply mask to DQ, SCI, ERR arrays

    Returns
    -------
    mask : array-like
        Boolean zeroth order mask calculated by shifting the non-zero segmentation mask
        by the expected zeroth order location as defined in the center of the detector

    """
    for flt in grp.FLTs:
        if flt.seg.max() == 0:
            msg = "simple_zeroth_mask: no non-zero segmentation pixels"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            return None
        elif "B" not in flt.conf.beams:
            msg = "zeroth order (B) not found in config"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            return None

        dx = flt.conf.dxlam["B"]
        dy, lam = flt.conf.get_beam_trace(1024, 1024, dx=dx, beam="B")
        N = len(dx)
        dxp = int(dx[N // 2])
        dyp = int(dy[N // 2])

        msg = f"simple_zeroth_mask: dx={dxp} dy={dyp}   erosion={erosion} dilation={dilation}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        mask = flt.seg > 0
        if dilation is not None:
            if morphology is None:
                mask = nd.binary_dilation(mask, iterations=dilation)
            else:
                mask = morphology.isotropic_dilation(mask, dilation)
        if erosion is not None:
            if morphology is None:
                mask = nd.binary_erosion(mask, iterations=erosion)
            else:
                mask = morphology.isotropic_erosion(mask, erosion)

        mask = np.roll(np.roll(mask, dxp, axis=1), dyp, axis=0)

        if update:
            flt.grism.data["DQ"][mask] |= 4096
            flt.grism.data["ERR"][mask] *= 0
            flt.grism.data["SCI"][mask] *= 0

    return mask


def extract_from_coords(
    ra=189.0706488,
    dec=62.2089502,
    grisms=["F356W-GRISMR", "F410M-GRISMR", "F444W-GRISMR"],
    query_with_api=True,
    size=48,
    grp=None,
    clean=False,
    get_cutout=True,
    cutout_filter="F444W-CLEAR",
    segmentation_image=None,
    source_id=0,
    prefix="gdn-grism",
    verbose=True,
    mb_kwargs={},
    thumbnail_size=None,
    filter_kwargs=None,
    savefig=True,
    savefits=True,
    zeroth_mask_kwargs={"erosion": 8},
    use_jwst_crds=False,
    trim_exposures=True,
    pad_exposure=[800, 800],
    **kwargs,
):
    """
    Query and download slitless exposures
    """

    center_coord, coostr = coordinate_string(ra=ra, dec=dec, precision=2)

    group_name = f"{prefix}_{coostr}"
    utils.LOGFILE = f"{group_name}.log.txt"

    exp = query_exposures(
        ra=ra,
        dec=dec,
        filters=grisms,
        with_api=query_with_api,
        verbose=verbose,
        trim_exposures=trim_exposures,
        **kwargs,
    )

    if len(exp) > 0:
        _ = utils.Unique(exp["filter"], verbose=verbose)

    rate_files = []

    msg = f"extract_from_coords: {group_name} {len(exp)} exposures"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    for row in exp:
        rate_file = db.download_s3_file(
            S3PATH.format(**row), overwrite=False, verbose=verbose, **kwargs
        )
        if os.path.exists(rate_file):
            rate_files.append(rate_file)

    if get_cutout:
        if thumbnail_size == None:
            cutout_size = size * 0.04 * u.arcsec
        else:
            cutout_size = thumbnail_size * u.arcsec

        cutout_file = thumbnail_cutout(
            ra=ra,
            dec=dec,
            cutout_size=cutout_size,
            verbose=True,
            prefix=prefix,
            filter=cutout_filter,
            **kwargs,
        )
    else:
        cutout_file = None

    msg = f"extract_from_coords: direct image = {cutout_file}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    if (segmentation_image is not None) & (source_id == 0):
        # Get source from segmentation ID
        source_id = id_from_segmentation(segmentation_image, ra, dec, verbose=verbose)

    if grp is None:
        beams = []
        for j, file in enumerate(rate_files):
            if (verbose & 1) > 0:
                msg = f"{j+1:>2} / {len(rate_files)} GroupFLT {os.path.basename(file)}"
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            grp = multifit.GroupFLT(
                grism_files=[file],
                pad=pad_exposure,
                ref_file=cutout_file,
                verbose=((verbose & 2) > 0),
                cpu_count=-1,
                seg_file=segmentation_image,
                use_jwst_crds=use_jwst_crds,
            )

            if (zeroth_mask_kwargs is not None) & (segmentation_image is not None):
                # print('x before mask', (grp.FLTs[0].grism['DQ'] > 0).sum())
                _ = simple_zeroth_mask(grp, **zeroth_mask_kwargs)
                # print('x  after mask', (grp.FLTs[0].grism['DQ'] > 0).sum())

            if filter_kwargs is not None:
                grp.subtract_median_filter(**filter_kwargs)

            beams += grp.get_beams(
                source_id, center_rd=(ra, dec), size=size, **mb_kwargs
            )

            del grp

    else:
        beams = grp.get_beams(source_id, center_rd=(ra, dec), size=size, **mb_kwargs)

    msg = f"extract_from_coords: {group_name} {len(beams)} beam cutouts"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    mb = multifit.MultiBeam(
        beams,
        group_name=group_name,
        **mb_kwargs,
    )

    # for b in mb.beams:
    #     msk = b.grism['ERR'] > 0
    #     b.grism.data['SCI'] -= np.nanpercentile(b.grism['SCI'][msk], 40)

    _hdu, _fig2d = mb.drizzle_grisms_and_PAs(
        fig_args={
            "mask_segmentation": True,
            "average_only": False,
            "scale_size": 1,
            "cmap": "bone_r",
        }
    )

    _fig1d = mb.oned_figure(
        figsize=(10, 4), units="ujy", bin=2, ylim_percentile=0.2, show_beams=True
    )

    if savefig:
        _fig2d.savefig(f"{group_name}.2d.png")
        _fig1d.savefig(f"{group_name}.1d.png")

    if savefits:
        mb_hdul = mb.write_master_fits(get_trace_table=True, get_hdu=True)
        mb_hdul.writeto(f"{group_name}.beams.fits", overwrite=True)

        oned = mb.oned_spectrum()
        oned_hdu = mb.oned_spectrum_to_hdu()
        oned_hdu.writeto(f"{group_name}.1d.fits", overwrite=True)

    if clean:
        for file in rate_files:
            if os.path.exists(file):
                os.remove(file)

    return mb


def combine_beams_2d(
    mb,
    step=0.5,
    pixfrac=0.75,
    ymax=12.5,
    bkg_percentile=None,
    profile_type="grizli",
    profile_sigma=1.0,
    profile_offset=0.0,
    cont_spline=41,
    zfit_nspline=-1,
    z=None,
    savefig=True,
    verbose=True,
    sys_err=0.03,
    auto_niriss=False,
    zfit_kwargs={},
    ylim=(-3, 10),
    yticks=[0, 3, 5, 7],
    **kwargs,
):
    """
    Create rectified 2D spectra with `msaexp.slit_combine.pseudo_drizzle`

    Parameters
    ----------
    mb : `grizli.multifit.MultiBeam`
        Grizli "beams" object

    step : float
        Step size of spatial axis relative to the native pixels

    pixfrac : float
        Pseudo-drizzle pixfrac value

    ymax : float
        Extent of the cross-dispersion size to plot

    bkg_percentile : float, None
        Percentile of the 2D pixel distribution to use as a pedestal background

    profile_type : ("grizli", "gaussian")
        Extraction profile to use.  With ``grizli``, use the dispersed 2D model
        derived from the direct image cutout.  With ``gaussian``, use a pixel-integrated
        Gaussian with the ``profile_sigma`` and ``profile_offset`` parameters.

    profile_sigma, profile_offset : float, float
        Gaussian profile width and center relative to the expected trace,
        in units of the native pixels

    cont_spline : int
        Degree of the cubic spline used to model the continuum / contamination

    z : float
        Trial redshift used for emission line labels

    Returns
    -------
    b2d : dict
        Derived products with keys of the available grism names.

    """
    import msaexp.utils
    import msaexp.slit_combine
    import msaexp.resample_numba

    mb.compute_model()

    if mb.beams[0].grism.filter.startswith("GR") & auto_niriss:
        if zfit_kwargs is not None:
            if "velocity_sigma" not in zfit_kwargs:
                dv = 800
            else:
                dv = zfit_kwargs["velocity_sigma"]

            zfit_kwargs["velocity_sigma"] = dv
            zfit_kwargs["dz"] = dv / 3.0e5 / 2

        cont_spline = 0
        zfit_nspline = 11

    b2d = {}
    PA_keys = list(mb.PA.keys())
    PA_keys.sort()

    for gr in PA_keys:
        b2d[gr] = {
            "group_name": mb.group_name,
            "sci": [],
            "wht": [],
            "prof": [],
            "bkg": [],
            "idx": [],
            "s1d": [],
            "e1d": [],
            "c1d": [],
            "contam": [],
            "cont_model": [],
            "sensitivity": None,
        }

        for j, pa in enumerate(mb.PA[gr]):
            beams = [mb.beams[i] for i in mb.PA[gr][pa]]
            b2d[gr]["idx"] += mb.PA[gr][pa]

            ydata = []
            ldata = []
            sdata = []
            vdata = []
            mdata = []

            if j == 0:
                b2d[gr]["sensitivity"] = (beams[0].beam.lam, beams[0].beam.sensitivity)

            for beam in beams:
                size = mb.beams[0].sh[0] // 2
                yp, xp = np.indices(beam.sh) * 1.0
                yp += 0.5 - size - beam.beam.ytrace

                lp = xp * 0 + beam.beam.lam / 1.0e4
                ok = np.abs(yp) < ymax
                oktr = np.abs(yp) > 3
                ok &= (beam.grism["DQ"] & (1 + 1024 + 4096)) == 0
                ok &= beam.grism["SCI"] != 0
                ok &= beam.grism["ERR"] > 0
                sok = beam.beam.sensitivity > 0.01 * beam.beam.sensitivity.max()
                ok &= sok

                if bkg_percentile is not None:
                    bkg = np.nanpercentile(beam.grism["SCI"][ok & oktr], bkg_percentile)
                else:
                    bkg = 0.0

                if (bkg > 5 * np.median(beam.grism["ERR"][ok])) & (1):
                    bkg = 0.0

                b2d[gr]["bkg"].append(bkg)

                ydata.append(1 * yp[ok])  # Not sure about the sign
                ldata.append(lp[ok])
                sdata.append(beam.grism["SCI"][ok] - bkg)
                vdata.append(beam.grism["ERR"][ok] ** 2)
                mdata.append(beam.flat_flam.reshape(beam.sh)[ok])

            yb = np.arange(-ymax, ymax + 0.1, step)
            wgrid = np.arange(
                *utils.GRISM_LIMITS[gr][:2], utils.GRISM_LIMITS[gr][2] / 1.0e4
            )
            wb = msaexp.utils.array_to_bin_edges(wgrid)

            b2d[gr]["yb"] = yb
            b2d[gr]["wgrid"] = wgrid
            b2d[gr]["wb"] = wb

            ldata = np.hstack(ldata)
            ydata = np.hstack(ydata)
            sdata = np.hstack(sdata)
            vdata = np.hstack(vdata)
            wdata = 1.0 / vdata
            mdata = np.hstack(mdata)

            if 0:
                msk = np.abs(ydata) > 2
                msk *= (sdata - np.nanmedian(sdata)) < 2 * np.sqrt(vdata)
                cbkg = np.polyfit(ldata[msk], sdata[msk], 1)
                sdata -= np.polyval(cbkg, ldata)

            drz = msaexp.slit_combine.pseudo_drizzle(
                ldata,
                ydata,
                sdata,
                vdata,
                wdata,
                wb,
                yb,
                pixfrac=pixfrac,
            )

            if profile_type == "grizli":
                prof = mdata
                to_ujy = (mdata**0 * u.erg / u.second / u.cm**2 / u.Angstrom).to(
                    u.microJansky, equivalencies=u.spectral_density(ldata * u.micron)
                )
                prof /= to_ujy.value
                if 0:
                    prof /= np.interp(
                        np.hstack(ldata) * 1.0e4, beam.beam.lam, beam.beam.sensitivity
                    )
                prof[~np.isfinite(prof)] = 0.0

            elif profile_type == "gaussian":
                prof = msaexp.resample_numba.pixel_integrated_gaussian_numba(
                    ydata, profile_offset, profile_sigma, dx=1.0
                )

            else:
                b = beams[0]
                # plt.imshow(np.log(b.direct['REF']))

                yprof = np.nansum(b.direct["REF"], axis=1)
                yprof /= yprof.sum()

                import msaexp.resample_numba

                xst = 1.0
                # size = 48
                N = 10

                xarr = np.arange(len(yprof)) * 1.0 - size
                steps = np.linspace(-xst * N, xst * N, N * 2 + 1)

                comps = np.array(
                    [
                        msaexp.resample_numba.pixel_integrated_gaussian_numba(
                            xarr, st, xst, dx=1.0
                        )
                        for st in steps
                    ]
                ).T

                c = np.linalg.lstsq(comps, yprof, rcond=None)

                cprof = np.array(
                    [
                        msaexp.resample_numba.pixel_integrated_gaussian_numba(
                            ydata, st + profile_offset, xst, dx=1.0
                        )
                        for st in steps
                    ]
                ).T
                prof = cprof.dot(c[0] / c[0].sum())

            pdrz = msaexp.slit_combine.pseudo_drizzle(
                ldata,
                ydata,
                prof,
                vdata,
                wdata,
                wb,
                yb,
                pixfrac=0.75,
            )
            p2d = pdrz[0] / pdrz[2]

            sci = drz[0] / drz[2]
            b2d[gr]["sci"].append(sci)
            b2d[gr]["wht"].append(drz[2])
            b2d[gr]["prof"].append(p2d)

    for gr in PA_keys:

        NPA = len(mb.PA[gr])
        fig, axes = plt.subplots(
            1 + NPA * 2,
            1,
            figsize=(8, 1.5 + 2 * NPA),
            sharex=True,
            height_ratios=[1, 1] * NPA + [2],
        )

        # ok = (b2d[gr]["wht"] > 0) & np.isfinite(b2d[gr]["sci"])
        vmax = np.nanpercentile(np.array(b2d[gr]["sci"]), 80)
        avg_wht = np.nanmedian(np.array(b2d[gr]["wht"]))
        vmax = np.maximum(vmax, 3 / np.sqrt(avg_wht))

        yb = b2d[gr]["yb"]
        wgrid = b2d[gr]["wgrid"]
        wb = b2d[gr]["wb"]

        smax = 0.0
        b2d[gr]["wminmax"] = (wgrid.min(), wgrid.max())

        for j, pa in enumerate(mb.PA[gr]):
            sci = b2d[gr]["sci"][j]
            p2d = b2d[gr]["prof"][j]
            wht = b2d[gr]["wht"][j]

            if cont_spline > 0:
                w2 = (wgrid[:, None] * np.ones(sci.shape[0])).T

                okw = wht > 0
                wminmax = (w2[okw].min(), w2[okw].max())
                wminmax = b2d[gr]["wminmax"]

                bspl = utils.bspline_templates(
                    w2.flatten(), df=cont_spline, minmax=wminmax, get_matrix=True
                )
                A = np.vstack(
                    [
                        # np.polynomial.polynomial.polyvander(w2.flatten(), 0).T,
                        p2d.flatten()
                        * bspl.T
                    ]
                )

                test_resid = True
                cont_model = 0.0

                for _iter in range(2):
                    okw = (wht > 0) & test_resid
                    okw &= sci * np.sqrt(wht) > -4
                    bspl = utils.bspline_templates(
                        w2[okw], df=cont_spline, minmax=wminmax, get_matrix=True
                    )
                    Ax = np.vstack(
                        [
                            # np.polynomial.polynomial.polyvander(w2[okw], 0).T,
                            (p2d * np.sqrt(wht))[okw]
                            * bspl.T
                        ]
                    )
                    y = sci[okw] * np.sqrt(wht[okw])
                    cspl = np.linalg.lstsq(Ax.T, y, rcond=None)
                    cont_model = A.T.dot(cspl[0]).reshape(w2.shape)
                    test_resid = (sci - cont_model) * np.sqrt(wht) < 2
                    # test_resid &= nd.binary_erosion(test_resid, iterations=2)

                    # bad_resid = (sci - cont_model) * np.sqrt(wht) > 4
                    # bad_resid &= nd.binary_erosion(bad_resid, iterations=1)
                    # test_resid = ~nd.binary_dilation(bad_resid, iterations=2)

            else:
                cont_model = 0.0

            opt_num = np.nansum((sci - cont_model) * p2d * wht, axis=0)
            opt_den = np.nansum(p2d**2 * wht, axis=0)

            s1d = opt_num / opt_den
            e1d = np.sqrt(1.0 / opt_den)

            if cont_spline > 0:
                A = utils.bspline_templates(
                    wgrid,
                    minmax=b2d[gr]["wminmax"],  # (wgrid.min(), wgrid.max()),
                    df=cont_spline,
                    get_matrix=True,
                ).T

                test_resid = True
                c1d = 0.0

                for _iter in range(2):
                    okw = (opt_den > 0) & test_resid
                    okw &= s1d * np.sqrt(opt_den) > -4
                    bspl = utils.bspline_templates(
                        wgrid[okw],
                        minmax=b2d[gr]["wminmax"],  # (wgrid.min(), wgrid.max()),
                        df=cont_spline,
                        get_matrix=True,
                    )
                    Ax = np.vstack(
                        [
                            # np.polynomial.polynomial.polyvander(w2[okw], 0).T,
                            np.sqrt(opt_den)[okw]
                            * bspl.T
                        ]
                    )
                    y = s1d[okw] * np.sqrt(opt_den[okw])
                    cspl = np.linalg.lstsq(Ax.T, y, rcond=None)
                    c1d = A.T.dot(cspl[0])
                    test_resid = (s1d - c1d) * np.sqrt(opt_den) < 2

                # c1d = cont_model
                s1d -= c1d
            else:
                c1d = np.nansum((sci * 0.0 + cont_model) * p2d * wht, axis=0) / opt_den

            contam = np.nansum((sci * 0.0 + cont_model) * p2d * wht, axis=0) / opt_den

            total_err = np.sqrt(e1d**2 + (sys_err * contam) ** 2)
            # total_err = e1d

            pl = axes[-1].plot(wgrid, s1d / total_err, alpha=0.4, zorder=10)

            axes[-1].plot(
                wgrid, c1d / total_err, alpha=0.2, color=pl[0].get_color(), zorder=1
            )

            axes[-1].fill_between(
                wgrid, e1d**0, -(e1d**0), alpha=0.08, color=pl[0].get_color()
            )

            b2d[gr]["s1d"].append(s1d)
            b2d[gr]["e1d"].append(total_err)
            b2d[gr]["c1d"].append(c1d)
            b2d[gr]["contam"].append(contam)
            b2d[gr]["cont_model"].append(cont_model)

            nmad = 1.48 * np.nanmedian(np.abs(s1d - np.nanmedian(s1d)))
            # smax = np.max([smax, np.nanpercentile(s1d, 80), 3*nmad])
            smax = np.max([smax, np.nanmedian(e1d[(e1d > 0) & (np.isfinite(e1d))])])
            # print(smax)
            # smax = np.maximum(smax, 3*nmad)

            ax = axes[j * 2]
            ax.imshow(
                sci - cont_model * 0,
                aspect="auto",
                extent=(wb[0], wb[-1], yb[0], yb[-1]),
                vmin=-0.1 * vmax,
                vmax=2 * vmax,
                cmap="bone_r",
                origin="lower",
            )
            ax.grid()
            ax.text(
                0.98,
                0.9,
                f"PA: {pa:.0f}",
                ha="right",
                va="top",
                fontsize=8,
                transform=ax.transAxes,
                bbox={"ec": "None", "fc": "w", "alpha": 1.0},
            )

            m2d = s1d * p2d  # + cont_model
            ax = axes[j * 2 + 1]
            ax.imshow(
                m2d,
                aspect="auto",
                extent=(wb[0], wb[-1], yb[0], yb[-1]),
                vmin=-0.1 * vmax,
                vmax=2 * vmax,
                cmap="bone_r",
                origin="lower",
            )
            ax.grid()
            for k in [0, 1]:
                axes[j * 2 + k].set_ylim(-ymax, ymax + 1)
                axes[j * 2 + k].set_yticklabels([])

        scl = 2
        axes[-1].set_ylim(*ylim)
        axes[-1].set_yticks(yticks)
        axes[-1].set_ylabel(r"$\sigma$")
        axes[-1].grid()
        axes[-1].set_xlabel(r"$\lambda$ - " + f"{gr}")

        data = b2d[gr]
        sp = utils.GTable()
        sp["wave"] = data["wgrid"] * u.micron
        wht = 1.0 / np.array(data["e1d"]) ** 2
        sp["flux"] = (
            np.nansum(np.array(data["s1d"]) * wht, axis=0) / np.nansum(wht, axis=0)
        ) * u.microJansky

        sp["contam"] = (
            np.nansum(np.array(data["contam"]) * wht, axis=0) / np.nansum(wht, axis=0)
        ) * u.microJansky

        sp["err"] = 1.0 / np.sqrt(np.nansum(wht, axis=0)) * u.microJansky

        sp["bkg"] = (
            np.nansum(np.array(data["c1d"]) * wht, axis=0) / np.nansum(wht, axis=0)
        ) * u.microJansky

        if data["sensitivity"] is not None:
            sp["sensitivity"] = np.interp(
                sp["wave"] * 1.0e4, *data["sensitivity"], left=0, right=0.0
            )

        sp.meta["GRATING"] = "GRISMR"
        sp.meta["FILTER"] = gr
        sp.meta["INSTRUME"] = "NIRCAM"
        sp.meta["RA"] = mb.ra
        sp.meta["DEC"] = mb.dec
        sp.meta["SPLWMIN"] = b2d[gr]["wminmax"][0]
        sp.meta["SPLWMAX"] = b2d[gr]["wminmax"][1]

        spec_file = f"{mb.group_name}.{gr}.spec.fits".lower()
        msg = f"final spectrum file: {spec_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        sp.write(spec_file, overwrite=True)
        b2d[gr]["spec"] = sp
        b2d[gr]["spec_file"] = spec_file

        axes[-1].plot(wgrid, sp["flux"] / sp["err"], alpha=0.3, color="k", zorder=100)

        b2d[gr]["fig"] = fig

    if zfit_kwargs is not None:
        if zfit_nspline < 0:
            zfit_nspline = cont_spline

        zres = redshift_fit_1d(b2d, nspline=zfit_nspline, **zfit_kwargs)
        z = zres["z"]
    else:
        zres = None

    if z is not None:
        for gr in b2d:
            axes = b2d[gr]["fig"].axes

            for ax in axes:
                yl = ax.get_ylim()
                xl = ax.get_xlim()
                lw, lr = utils.get_line_wavelengths()
                lines = [
                    "OIII-4363",
                    "NeIII-3968",
                    "NeIII-3867",
                    "OII",
                    "OIII",
                    "Hb",
                    "Hg",
                    "Hd",
                    "Ha+NII",
                    "SII",
                    "SIII",
                    "PaB",
                    "PaG",
                    "PaD",
                    "HeI-1083",
                ]
                rest_wave = []
                for li in lines:
                    rest_wave += lw[li]

                ax.vlines(
                    np.array(rest_wave) * (1 + z) / 1.0e4,
                    *yl,
                    color="magenta",
                    linestyle=":",
                    alpha=0.3,
                )
                ax.set_ylim(*yl)
                ax.set_xlim(*xl)

            ax.text(
                0.98,
                0.9,
                f"z = {z:.4f}",
                ha="right",
                va="top",
                fontsize=8,
                color="magenta",
                transform=ax.transAxes,
                bbox={"ec": "None", "fc": "w", "alpha": 1.0},
            )

    for gr in b2d:
        fig = b2d[gr]["fig"]
        fig.tight_layout(pad=0.5)
        if mb.id > 0:
            label = f"{mb.id}\n{mb.group_name}"
        else:
            label = mb.group_name

        utils.figure_timestamp(fig, text_prefix=label + "\n")

        if savefig:
            fig.savefig(f"{mb.group_name}.{gr.lower()}.png")

    return b2d, zres


def redshift_fit_1d(
    b2d,
    zgrid=None,
    dz=0.0003,
    rest_wave=[1.3e4, 4800],
    scale_disp=0.5,
    verbose=True,
    velocity_sigma=None,
    savefig=True,
    nspline=0,
    spline_type="add",
    **kwargs,
):
    """
    Fit redshift with 1D spectra
    """
    from scipy.stats import norm
    from msaexp.spectrum import SpectrumSampler
    import eazy.templates

    templ = eazy.templates.Template("fsps_line_templ.fits")
    null_template = eazy.templates.Template(
        arrays=(templ.wave, np.ones_like(templ.flux_flam()))
    )

    loss_null = 0.0
    specs = {}
    loss = {}
    splines = {}

    for gi in b2d:
        spec = SpectrumSampler(b2d[gi]["spec_file"])
        group_name = b2d[gi]["group_name"]

        if "sensitivity" in spec.spec.colnames:
            sens = spec["sensitivity"] * 1.0e-20

            spec.spec["flux"] /= sens
            spec.spec["err"] /= sens
            spec.spec["full_err"] /= sens
            spec.spec["valid"] &= sens > 0
            spec.valid &= sens > 0

        loss[gi] = norm(loc=0, scale=spec["full_err"][spec.valid])
        if nspline <= 0:
            splines[gi] = None
            null_i = loss[gi].logpdf(spec["flux"][spec.valid]).sum()
        else:
            splines[gi] = spec.bspline_array(
                nspline=nspline,
                minmax=(spec.meta["SPLWMIN"] * 1.0e4, spec.meta["SPLWMAX"] * 1.0e4),
                by_wavelength=True,
                orders=[1],
            )

            tfit = spec.fit_single_template(
                null_template,
                z=0,
                scale_disp=scale_disp,
                velocity_sigma=velocity_sigma,
                spl=splines[gi],
                spline_type=spline_type,
            )
            null_i = loss[gi].logpdf((spec["flux"] - tfit["model"])[spec.valid]).sum()
            # null_i = loss[gi].logpdf(spec["flux"][spec.valid]).sum()

        spec.spec.meta["LOSSNULL"] = (null_i, "Null lnP")
        loss_null += null_i

        specs[gi] = spec

    if zgrid is None:
        zlim = [1000, 0]
        for gi in specs:
            glim = utils.GRISM_LIMITS[gi]
            zgr = [glim[i] * 1.0e4 / rest_wave[i] - 1 for i in [0, 1]]

            zlim[0] = np.minimum(zlim[0], zgr[0])
            zlim[1] = np.maximum(zlim[1], zgr[1])

        zgrid = utils.log_zgrid(zlim, dz)

    msg = f"redshift_fit_1d {group_name} z=[{zgrid.min():.3f}, {zgrid.max():.3f}]"
    msg += f" dz={dz}  nsteps={len(zgrid)}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    lnp = zgrid * 0.0

    if velocity_sigma is None:
        velocity_sigma = dz * 3.0e5

    for j, zi in enumerate(zgrid):
        for gi in specs:
            tfit = specs[gi].fit_single_template(
                templ,
                z=zi,
                scale_disp=scale_disp,
                velocity_sigma=velocity_sigma,
                spl=splines[gi],
                spline_type=spline_type,
            )
            lnp[j] += tfit["lnp"].sum()

    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    ax.plot(zgrid, lnp - loss_null, color="0.4", alpha=0.5)
    ax.grid()

    iz = np.argmax(lnp)
    zbest = zgrid[iz]

    dlnp = np.nanmax(lnp) - loss_null

    msg = f"redshift_fit_1d {group_name} best z={zbest:.5f}  dlnP={dlnp:.1f}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    loss_z = 0.0
    for gi in specs:
        spec = specs[gi]

        tfit = spec.fit_single_template(
            templ,
            z=zbest,
            scale_disp=scale_disp,
            velocity_sigma=velocity_sigma,
            spl=splines[gi],
            spline_type=spline_type,
        )

        b2d[gi]["fig"].axes[-1].plot(
            spec["wave"], tfit["model"] / spec["full_err"], color="magenta", alpha=0.5
        )
        spec.spec["model"] = tfit["model"]
        spec.spec.meta["REDSHIFT"] = (zbest, "Best redshift")

        if (splines[gi] is not None) & (spline_type == "add"):
            spline_continuum = tfit["A"].T[:, 1:].dot(tfit["coeffs"][1:])
            spec.spec["continuum"] = spline_continuum
            b2d[gi]["fig"].axes[-1].plot(
                spec["wave"],
                (spec["flux"] - spline_continuum) / spec["full_err"],
                color="0.5",
                alpha=0.5,
            )

            b2d[gi]["fig"].axes[-1].plot(
                spec["wave"],
                (spec["model"] - spline_continuum) / spec["full_err"],
                color="tomato",
                alpha=0.5,
            )

        loss_i = loss[gi].logpdf((spec["flux"] - tfit["model"])[spec.valid]).sum()

        dlnp_i = loss_i - spec.spec.meta["LOSSNULL"][0]
        spec.spec.meta["DLNP"] = dlnp_i

        spec.spec.meta["LOSSZ"] = (loss_i, "lnP at best redshift")
        loss_z += loss_i

        msg = f"redshift_fit_1d {group_name} {gi} dlnP={dlnp_i:.1f}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        spec.spec.write(b2d[gi]["spec_file"], overwrite=True)

    yl = ax.get_ylim()

    ax.text(
        zbest,
        lnp[iz] - loss_null + 0.02 * (yl[1] - yl[0]),
        f"{zbest:.4f}",
        color="tomato",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    ax.set_ylim(yl[0], yl[0] + 1.05 * (yl[1] - yl[0]))

    ax.set_xlabel("z")
    ax.set_ylabel(r"$\Delta~\log{P(z)}$")
    xl = ax.get_xlim()
    ax.hlines([loss_null * 0], *xl, color="pink", alpha=0.5)
    ax.set_xlim(*xl)

    fig.tight_layout(pad=0.5)

    utils.figure_timestamp(
        fig,
        text_prefix=(group_name + "\n"),
    )

    res = {
        "zgrid": zgrid,
        "lnp": lnp,
        "loss_null": loss_null,
        "z": zbest,
        "fig": fig,
    }

    if savefig:
        fig.savefig(f"{group_name}.lnpz.png")

    return res
