"""
Utilities for handling JWST file/data formats.

Requires https://github.com/spacetelescope/jwst

"""

import os
import inspect
import logging
import traceback

import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from . import utils
from . import GRIZLI_PATH

QUIET_LEVEL = logging.INFO

# CRDS_CONTEXT = 'jwst_0942.pmap' # July 29, 2022 with updated NIRCAM ZPs
# CRDS_CONTEXT = 'jwst_0995.pmap' # 2022-10-06 NRC ZPs and flats
# CRDS_CONTEXT = "jwst_1123.pmap"  # 2023-09-08 NRC specwcs, etc.
CRDS_CONTEXT = "jwst_1293.pmap"  # 2024-09-25

MAX_CTX_FOR_SKYFLATS = "jwst_1130.pmap"

## Some filters are still better with the grizli skyflats
FORCE_SKYFLATS = [
    "F250M",
    "F250M-CLEAR",
    "F300M",
    "F300M-CLEAR",
    "F460M",
    "F460M-CLEAR",
]

# Global variable to control whether or not to try to update
# PP file WCS
DO_PURE_PARALLEL_WCS = True
FIXED_PURE_PARALLEL_WCS_CAL_VER = "1.16"

from .constants import JWST_DQ_FLAGS, PLUS_FOOTPRINT, CORNER_FOOTPRINT


def set_crds_context(fits_file=None, override_environ=False, verbose=True):
    """
    Set CRDS_CONTEXT

    Parameters
    ----------
    fits_file : str
        If provided, try to get CRDS_CONTEXT from header

    override_environ : bool
        Override environment variable if True, otherwise will not change
        the value of an already-set CRDS_CONTEXT environment variable.

    verbose : bool
        Messaging to terminal.

    Returns
    -------
    crds_context : str
        The value of the CRDS_CONTEXT environment variable

    """
    from importlib import reload
    import crds
    import crds.core
    import crds.core.heavy_client

    global CRDS_CONTEXT

    if fits_file is not None:
        with pyfits.open(fits_file) as im:
            if "CRDS_CONTEXT" in im[0].header:
                CRDS_CONTEXT = im[0].header["CRDS_CTX"]

    if os.getenv("CRDS_CONTEXT") is None:
        os.environ["CRDS_CONTEXT"] = CRDS_CONTEXT
    elif override_environ:
        os.environ["CRDS_CONTEXT"] = CRDS_CONTEXT

    msg = f"ENV CRDS_CONTEXT = {os.environ['CRDS_CONTEXT']}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    # Need to reload CRDS modules to catch new CONTEXT
    reload(crds.core)
    reload(crds)
    reload(crds.core.heavy_client)

    return os.environ["CRDS_CONTEXT"]


def crds_reffiles(
    instrument="NIRCAM",
    filter="F444W",
    pupil="GRISMR",
    module="A",
    detector=None,
    exp_type=None,
    date=None,
    reftypes=("photom", "specwcs"),
    header=None,
    context=CRDS_CONTEXT,
    verbose=False,
    **kwargs,
):
    """
    Get WFSS reffiles from CRDS

    Parameters
    ----------
    instrument, filter, pupil, module : str
        Observation mode parameters

    detector, exp_type : str, None
        If not specified, try to set automatically based on the filter / module

    date : `astropy.time.Time`, None
        Observation epoch.  If `None`, use "now".

    reftypes : list
        Reference types to query

    header : `~astropy.io.fits.Header`
        FITS header with keywords that define the mode and supersede the string
        parameters

    context : str
        CRDS_CONTEXT specification

    verbose : bool
        Messaging

    Returns
    -------
    refs : dict
        Result from `crds.getreferences` with keys of ``reftypes`` and values of paths
        to the reference files, which will be downloaded if they're not already found.

    """
    import astropy.time
    import crds
    from . import jwst_utils

    if context is not None:
        jwst_utils.CRDS_CONTEXT = context
        jwst_utils.set_crds_context(verbose=verbose, override_environ=True)

    if header is not None:
        if "INSTRUME" in header:
            instrument = header["INSTRUME"]
        if "FILTER" in header:
            filter = header["FILTER"]
        if "PUPIL" in header:
            pupil = header["PUPIL"]
        if "MODULE" in header:
            module = header["MODULE"]
        if "EXP_TYPE" in header:
            exp_type = header["EXP_TYPE"]

    cpars = {}

    if instrument in ("NIRISS", "NIRCAM", "MIRI"):
        observatory = "jwst"
        if instrument not in ["MIRI"]:
            cpars["meta.instrument.pupil"] = pupil
    else:
        observatory = "hst"

    if instrument == "NIRISS":

        cpars["meta.instrument.detector"] = "NIS"
        if "GR150" in filter:
            cpars["meta.exposure.type"] = "NIS_WFSS"
        else:
            cpars["meta.exposure.type"] = "NIS_IMAGE"

    elif instrument == "NIRCAM":

        cpars["meta.instrument.detector"] = f"NRC{module}LONG"
        cpars["meta.instrument.module"] = module
        cpars["meta.exposure.type"] = exp_type

        if "GRISM" in pupil:
            cpars["meta.exposure.type"] = "NRC_WFSS"
        else:
            cpars["meta.exposure.type"] = "NRC_IMAGE"

    elif instrument == "MIRI":

        cpars["meta.instrument.detector"] = "MIR"
        cpars["meta.exposure.type"] = "MIR_IMAGE"

    if exp_type is not None:
        cpars["meta.exposure.type"] = exp_type

    if detector is not None:
        cpars["meta.instrument.detector"] = detector
        if instrument == "NIRCAM":
            cpars["meta.instrument.channel"] = "LONG" if "LONG" in detector else "SHORT"

    if date is None:
        date = astropy.time.Time.now().iso

    cpars["meta.observation.date"] = date.split()[0]
    cpars["meta.observation.time"] = date.split()[1]

    cpars["meta.instrument.name"] = instrument
    cpars["meta.instrument.filter"] = filter

    refs = crds.getreferences(cpars, reftypes=reftypes, observatory=observatory)

    if verbose:
        msg = f"crds_reffiles: {instrument} {filter} {pupil} {module} ({context})"
        ref_files = " ".join([os.path.basename(refs[k]) for k in refs])
        msg += "\n" + f"crds_reffiles: {ref_files}"
        print(msg)

    return refs


def set_quiet_logging(level=QUIET_LEVEL):
    """
    Silence the verbose logs set by `stpipe`

    Parameters
    ----------
    level : int, optional
        Logging level to be passed to `logging.disable`.

    """
    try:
        import jwst

        logging.disable(level)
    except ImportError:
        pass


def get_jwst_dq_bit(dq_flags=JWST_DQ_FLAGS, verbose=False):
    """
    Get a combined bit from JWST DQ flags

    Parameters
    ----------
    dq_flags : list
        List of flag names

    verbose : bool
        Messaging

    Returns
    -------
    dq_flag : int
        Combined bit flag

    """
    try:
        import jwst.datamodels
    except:
        msg = f"get_jwst_dq_bits: import jwst.datamodels failed"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return 1

    dq_flag = 1
    for _bp in dq_flags:
        dq_flag |= jwst.datamodels.dqflags.pixel[_bp]

    msg = f"get_jwst_dq_bits: {'+'.join(dq_flags)} = {dq_flag}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    return dq_flag


def hdu_to_imagemodel(in_hdu):
    """
    Workaround for initializing a `jwst.datamodels.ImageModel` from a
    normal FITS ImageHDU that could contain HST header keywords and
    unexpected WCS definition.

    TBD

    Parameters
    ----------
    in_hdu : `astropy.io.fits.ImageHDU`

    Returns
    -------
    img : `jwst.datamodels.ImageModel`

    """
    from astropy.io.fits import ImageHDU, HDUList
    from astropy.coordinates import ICRS

    from jwst.datamodels import util
    import gwcs

    set_quiet_logging(QUIET_LEVEL)

    hdu = ImageHDU(data=in_hdu.data, header=in_hdu.header)

    new_header = strip_telescope_header(hdu.header)

    hdu.header = new_header

    # Initialize data model
    img = util.open(HDUList([hdu]))

    # Initialize GWCS
    tform = gwcs.wcs.utils.make_fitswcs_transform(new_header)
    hwcs = gwcs.WCS(
        forward_transform=tform, output_frame=ICRS()
    )  # gwcs.CelestialFrame())
    sh = hdu.data.shape
    hwcs.bounding_box = ((-0.5, sh[0] - 0.5), (-0.5, sh[1] - 0.5))

    # Put gWCS in meta, where blot/drizzle expect to find it
    img.meta.wcs = hwcs

    return img


def change_header_pointing(header, ra_ref=0.0, dec_ref=0.0, pa_v3=0.0):
    """
    Update a FITS header for a new pointing (center + roll).

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Parent header (must contain `V2_REF`, `V3_REF` keywords).

    ra_ref, dec_ref : float
        Pointing center, in decimal degrees, at reference the pixel defined
        in.

    pa_v3 : float
        Position angle of the telescope V3 axis, degrees.

    .. warning::

    Doesn't update PC keywords based on pa_v3, which would rather have to
    be computed from the new `gwcs`.

    """
    from jwst.lib.set_telescope_pointing import compute_local_roll

    set_quiet_logging(QUIET_LEVEL)

    v2_ref = header["V2_REF"]
    v3_ref = header["V3_REF"]

    # Strip units, if any
    args = []
    for v in (pa_v3, ra_ref, dec_ref, v2_ref, v3_ref):
        if hasattr(v, "value"):
            args.append(v.value)
        else:
            args.append(v)

    roll_ref = compute_local_roll(*tuple(args))

    new_header = header.copy()
    new_header["XPA_V3"] = args[0]
    new_header["CRVAL1"] = new_header["RA_REF"] = args[1]
    new_header["CRVAL2"] = new_header["DEC_REF"] = args[2]
    new_header["ROLL_REF"] = roll_ref
    return new_header


def get_jwst_skyflat(header, verbose=True, valid_flat=(0.7, 1.4)):
    """
    Get sky flat for JWST instruments

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        Primary header

    verbose : bool
        Verbose messaging

    valid_flat : (float, float)
        Range of values to define where the flat is valid to avoid corrections
        that are too large

    Returns
    -------
    skyfile : str
        Filename of the sky flat file

    flat_corr : array-like
        The flat correction, equal to the original flat divided by the
        new sky flat, i.e., to take out the former and apply the latter

    dq : array-like
        DQ array with 1024 where flat outside of ``valid_flat`` range

    If no flat file is found, returns ``None`` for all outputs

    """
    filt = utils.parse_filter_from_header(header)

    key = ("{0}-{1}".format(header["detector"], filt)).lower()
    conf_path = os.path.join(GRIZLI_PATH, "CONF", "NircamSkyFlat")
    if "nrcb4" in key:
        skyfile = os.path.join(conf_path, f"{key}_skyflat.fits")
    elif key.startswith("nis-"):
        skyfile = os.path.join(conf_path, f"{key}_skyflat.fits")
    elif key.startswith("mirimage-"):
        key += "-" + header["readpatt"].lower()
        skyfile = os.path.join(conf_path, f"{key}_skyflat.fits")
    else:
        skyfile = os.path.join(conf_path, f"{key}_skyflat_smooth.fits")

    if not os.path.exists(skyfile):
        msg = f"jwst_utils.get_jwst_skyflat: {skyfile} not found"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None, None, None

    with pyfits.open(skyfile) as _im:
        skyflat = _im[0].data * 1

        # flat == 1 are bad
        skyflat[skyflat == 1] = np.nan

    if "R_FLAT" in header:
        oflat = os.path.basename(header["R_FLAT"])
        crds_path = os.getenv("CRDS_PATH")
        crds_path = os.path.join(
            crds_path, "references/jwst", header["instrume"].lower(), oflat
        )

        msg = f"jwst_utils.get_jwst_skyflat: pipeline flat = {crds_path}\n"

        with pyfits.open(crds_path) as oim:
            try:
                flat_corr = oim["SCI"].data / skyflat
            except ValueError:
                msg = f"jwst_utils.get_jwst_skyflat: flat_corr failed"
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
                return None, None, None
    else:
        msg = f"jwst_utils.get_jwst_skyflat: NO pipeline flat\n"
        flat_corr = 1.0 / skyflat

    bad = skyflat < valid_flat[0]
    bad |= skyflat > valid_flat[1]
    bad |= ~np.isfinite(flat_corr)
    flat_corr[bad] = 1

    dq = bad * 1024

    msg += f"jwst_utils.get_jwst_skyflat: new sky flat = {skyfile}\n"
    msg += f"jwst_utils.get_jwst_skyflat: valid_flat={valid_flat}"
    msg += f" nmask={bad.sum()}"

    if "SUBSTRT1" in header:
        if header["SUBSIZE1"] != 2048:
            slx = slice(
                header["SUBSTRT1"] - 1, header["SUBSTRT1"] - 1 + header["SUBSIZE1"]
            )
            sly = slice(
                header["SUBSTRT2"] - 1, header["SUBSTRT2"] - 1 + header["SUBSIZE2"]
            )

            msg += f"\njwst_utils.get_jwst_skyflat: subarray "
            msg += header["APERNAME"]
            msg += f" [{sly.start}:{sly.stop},{slx.start}:{slx.stop}]"

            flat_corr = flat_corr[sly, slx]
            dq = dq[sly, slx]

    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    return skyfile, flat_corr, dq


def check_context_for_skyflats(verbose=True):
    """
    Check that global variables ``CRDS_CONTEXT <= MAX_CTX_FOR_SKYFLATS``

    Returns
    -------
    result : bool

    """
    context = os.getenv("CRDS_CONTEXT")
    if context is None:
        context = CRDS_CONTEXT

    res = context <= MAX_CTX_FOR_SKYFLATS

    msg = f"check_context_for_skyflats: {context} < {MAX_CTX_FOR_SKYFLATS}: {res}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    return res


def img_with_flat(
    input,
    verbose=True,
    overwrite=True,
    apply_photom=True,
    use_skyflats=True,
    mask_dq4_fraction=0.25,
):
    """
    Apply flat-field and photom corrections if nessary

    Parameters
    ----------
    input : str, `~astropy.io.fits.HDUList`
        FITS filename of a JWST image or a previously-opened
        `~astropy.io.fits.HDUList` with SIP wcs information stored in the
        first extension.

    verbose : bool
        Messaging to terminal.

    overwrite : bool
        Overwrite FITS file with updated header keywords.

    apply_photom : bool
        Apply photometric calibration if True and the exposure is not a grism
        or the `apply_photom` parameter is set to False.

    use_skyflats : bool
        Apply sky flat corrections if True and ``CRDS_CONTEXT > MAX_CTX_FOR_SKYFLATS``.

    mask_dq4_fraction : float
        Add an additional check for the fraction of pixels with the DQ=4 bit set.  If
        the fraction is found to be greater than this value, unset them in the DQ
        extension.

    Returns
    -------
    output : `jwst.datamodels.ImageModel`
        Updated data model
    """
    import gc

    import astropy.io.fits as pyfits

    from jwst.datamodels import util
    from jwst.flatfield import FlatFieldStep
    from jwst.gain_scale import GainScaleStep
    from jwst.photom import PhotomStep

    set_quiet_logging(QUIET_LEVEL)

    _ = set_crds_context()

    if not isinstance(input, pyfits.HDUList):
        _hdu = pyfits.open(input)
    else:
        _hdu = input

    skip = False
    if "S_FLAT" in _hdu[0].header:
        if _hdu[0].header["S_FLAT"] == "COMPLETE":
            skip = True

    if "OINSTRUM" not in _hdu[0].header:
        copy_jwst_keywords(_hdu[0].header)

    # if _hdu[0].header['OINSTRUM'] == 'NIRISS':
    #     if _hdu[0].header['OFILTER'].startswith('GR'):
    #         _hdu[0].header['FILTER'] = 'CLEAR'
    #         _hdu[0].header['EXP_TYPE'] = 'NIS_IMAGE'

    # NIRCam grism flats are empty
    # NIRISS has slitless flats that include the mask spots
    if _hdu[0].header["OINSTRUM"] == "NIRCAM":
        if _hdu[0].header["OPUPIL"].startswith("GR"):
            _opup = _hdu[0].header["OPUPIL"]
            msg = f"Set NIRCAM slitless PUPIL {_opup} -> CLEAR for flat"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            _hdu[0].header["PUPIL"] = "CLEAR"
            _hdu[0].header["EXP_TYPE"] = "NRC_IMAGE"
    else:
        # MIRI, NIRISS
        pass

    img = util.open(_hdu)

    if not skip:

        flat_step = FlatFieldStep()
        _flatfile = flat_step.get_reference_file(img, "flat")
        utils.log_comment(
            utils.LOGFILE,
            f"jwst.flatfield.FlatFieldStep: {_flatfile}",
            verbose=verbose,
            show_date=False,
        )

        with_flat = flat_step.process(img)

        # Photom
        if "OPUPIL" in _hdu[0].header:
            _opup = _hdu[0].header["OPUPIL"]
        else:
            _opup = ""

        _ofilt = _hdu[0].header["OFILTER"]
        if _opup.startswith("GR") | _ofilt.startswith("GR") | (not apply_photom):
            output = with_flat
            _photfile = None
        else:
            photom_step = PhotomStep()
            with_phot = photom_step.process(with_flat)
            output = with_phot
            _photfile = photom_step.get_reference_file(img, "photom")
            utils.log_comment(
                utils.LOGFILE,
                f"jwst.flatfield.PhotomStep: {_photfile}",
                verbose=verbose,
                show_date=False,
            )

    else:

        _flatfile = None

        utils.log_comment(
            utils.LOGFILE,
            f"jwst_utils.img_with_flat: Flat already applied",
            verbose=verbose,
            show_date=False,
        )

        output = img

    if isinstance(input, str) & overwrite:
        output.write(input, overwrite=overwrite)
        _hdu.close()

        # Add reference files
        if not skip:
            with pyfits.open(input, mode="update") as _hdu:

                _hdu[0].header["UPDA_CTX"] = (
                    os.environ["CRDS_CONTEXT"],
                    "CRDS_CTX for modified files",
                )

                _hdu[0].header["R_FLAT"] = (os.path.basename(_flatfile), "Applied flat")
                if _photfile is not None:
                    _hdu[0].header["R_PHOTOM"] = (
                        os.path.basename(_photfile),
                        "Applied photom",
                    )

                _hdu.flush()

        _needs_skyflat = check_context_for_skyflats() | (
            _hdu[0].header["OFILTER"] in FORCE_SKYFLATS
        )

        if use_skyflats & _needs_skyflat:
            with pyfits.open(input, mode="update") as _hdu:
                if "FIXFLAT" not in _hdu[0].header:
                    _sky = get_jwst_skyflat(_hdu[0].header)
                    if _sky[0] is not None:
                        if _hdu["SCI"].data.shape == _sky[1].shape:

                            _hdu["SCI"].data *= _sky[1]

                            _skyf = os.path.basename(_sky[0])
                            _hdu[0].header["FIXFLAT"] = (
                                True,
                                "Skyflat correction applied",
                            )
                            _hdu[0].header["FIXFLATF"] = _skyf, "Skyflat file"
                            _dt = _hdu["DQ"].data.dtype
                            _hdu["DQ"].data |= _sky[2].astype(_dt)

                            _hdu.flush()
                else:
                    msg = f"jwst_utils.get_jwst_skyflat: FIXFLAT found"
                    utils.log_comment(
                        utils.LOGFILE, msg, verbose=verbose, show_date=False
                    )
        else:
            # Mask flat
            if _flatfile is not None:
                with pyfits.open(_flatfile) as _flat_im:
                    _flat_dq = _flat_im["DQ"].data * 1
                    _flat_data = _flat_im["SCI"].data * 1

                _bad_flat = _flat_data == 1
                _bad_flat |= _flat_data < 0.6
                _bad_flat |= _flat_data > 1.8

                _flat_dq |= (5 * (_bad_flat)).astype(_flat_dq.dtype)

                with pyfits.open(input, mode="update") as _hdu:
                    _hdu["DQ"].data |= _flat_dq.astype(_hdu["DQ"].data.dtype)
                    _hdu.flush()

        if mask_dq4_fraction is not None:
            with pyfits.open(input, mode="update") as _hdu:

                dq4 = _hdu["DQ"].data & 4
                dq4_frac = (dq4 > 0).sum() / dq4.size

                if dq4_frac > mask_dq4_fraction:

                    msg = f"jwst_utils.img_with_flat: {dq4_frac * 100:.1f}%"
                    msg += f" DQ=4 pixels > {mask_dq4_fraction * 100:.1f}"
                    utils.log_comment(
                        utils.LOGFILE, msg, verbose=verbose, show_date=False
                    )

                    _hdu["DQ"].header["UNSET4"] = True

                    _hdu["DQ"].data -= dq4.astype(_hdu["DQ"].data.dtype)

                    _hdu.flush()

    gc.collect()

    return output


def img_with_wcs(
    input, overwrite=True, fit_sip_header=True, skip_completed=True, verbose=True
):
    """
    Open a JWST exposure and apply the distortion model.

    Parameters
    ----------
    input : object
        Anything `jwst.datamodels.util.open` can accept for initialization.

    overwrite : bool
        Overwrite FITS file

    fit_sip_header : bool
        Run `pipeline_model_wcs_header` to rederive SIP distortion header

    skip_completed : bool
        Skip the `pipeline_model_wcs_header` step if the `GRIZLWCS` keyword is
        already set to True.

    verbose : bool
        Messaging to terminal.

    Returns
    -------
    with_wcs : `jwst.datamodels.ImageModel`
        Image model with full `~gwcs` in `with_wcs.meta.wcs`.

    """
    from packaging.version import Version
    from jwst.datamodels import util
    from jwst.assign_wcs import AssignWcsStep

    # global DO_PURE_PARALLEL_WCS, FIXED_PURE_PARALLEL_WCS_CAL_VER

    set_quiet_logging(QUIET_LEVEL)

    _ = set_crds_context()

    # HDUList -> jwst.datamodels.ImageModel

    # Generate WCS as image
    if not isinstance(input, pyfits.HDUList):
        _hdu = pyfits.open(input)
    else:
        _hdu = input

    if "OINSTRUM" not in _hdu[0].header:
        copy_jwst_keywords(_hdu[0].header)

    if _hdu[0].header["OINSTRUM"] == "NIRISS":
        if _hdu[0].header["OFILTER"].startswith("GR"):
            _hdu[0].header["FILTER"] = "CLEAR"
            _hdu[0].header["EXP_TYPE"] = "NIS_IMAGE"

    elif _hdu[0].header["OINSTRUM"] == "NIRCAM":
        if _hdu[0].header["OPUPIL"].startswith("GR"):
            _hdu[0].header["PUPIL"] = "CLEAR"
            _hdu[0].header["EXP_TYPE"] = "NRC_IMAGE"
    elif _hdu[0].header["OINSTRUM"] == "NIRSPEC":
        if _hdu[0].header["OGRATING"] not in "MIRROR":
            _hdu[0].header["FILTER"] = "F140X"
            _hdu[0].header["GRATING"] = "MIRROR"
            _hdu[0].header["EXP_TYPE"] = "NRS_TACONFIRM"
    else:
        # MIRI
        pass

    img = util.open(_hdu)

    # AssignWcs to pupulate img.meta.wcsinfo
    step = AssignWcsStep()
    _distor_file = step.get_reference_file(img, "distortion")
    utils.log_comment(
        utils.LOGFILE,
        f"jwst.assign_wcs.AssignWcsStep: {_distor_file}",
        verbose=verbose,
        show_date=False,
    )

    with_wcs = step.process(img)

    output = with_wcs

    # Write to a file
    if isinstance(input, str) & overwrite:
        output.write(input, overwrite=overwrite)

        _hdu = pyfits.open(input)

        if "GRIZLWCS" in _hdu[0].header:
            if (_hdu[0].header["GRIZLWCS"]) & (skip_completed):
                fit_sip_header = False

        # wcs = pywcs.WCS(_hdu['SCI'].header, relax=True)
        if fit_sip_header:
            hsip = pipeline_model_wcs_header(
                output,
                set_diff_step=False,
                step=64,
                degrees=[3, 4, 5, 5],
                initial_header=None,
            )

            wcs = pywcs.WCS(hsip, relax=True)
            for k in hsip:
                if k in hsip.comments:
                    _hdu[1].header[k] = hsip[k], hsip.comments[k]
                else:
                    _hdu[1].header[k] = hsip[k]

        else:
            wcs = utils.wcs_from_header(_hdu["SCI"].header, relax=True)

        # Remove WCS inverse keywords
        for _ext in [0, "SCI"]:
            for k in list(_hdu[_ext].header.keys()):
                if k[:3] in ["AP_", "BP_", "PC1", "PC2"]:
                    _hdu[_ext].header.remove(k)

        pscale = utils.get_wcs_pscale(wcs)

        _hdu[1].header["IDCSCALE"] = pscale, "Pixel scale calculated from WCS"
        _hdu[0].header["PIXSCALE"] = pscale, "Pixel scale calculated from WCS"
        _hdu[0].header["GRIZLWCS"] = True, "WCS modified by grizli"

        _hdu[0].header["UPDA_CTX"] = (
            os.environ["CRDS_CONTEXT"],
            "CRDS_CTX for modified files",
        )

        _hdu[0].header["R_DISTOR"] = (
            os.path.basename(_distor_file),
            "Distortion reference file",
        )

        _hdu.writeto(input, overwrite=True)
        _hdu.close()

        if "CAL_VER" in _hdu[0].header:
            _cal_ver = _hdu[0].header["CAL_VER"]
            _needs_fix = Version(_cal_ver) < Version(FIXED_PURE_PARALLEL_WCS_CAL_VER)
        else:
            _needs_fix = True

        if DO_PURE_PARALLEL_WCS & _needs_fix:
            try:
                # Update pointing of pure-parallel exposures
                status = update_pure_parallel_wcs(input, fix_vtype="PARALLEL_PURE")
            except:
                pass

    return output


def convert_cal_to_rate(cal_file, write=True, overwrite=True, verbose=True):
    """
    Undo ``photom`` and ``flat_field`` pipeline steps in a CAL file to make
    it consistent with a Level2 RATE product

    Parameters
    ----------
    cal_file : str
        FITS filename of a cal product

    write : bool
        Write output to ``cal_file.replace("_cal", "_rate")``

    Returns
    -------
    dm : `stdatamodels.jwst.datamodels.image.ImageModel`
        Data model with the inverse photom and flat_field steps applied

    """
    from jwst.photom import PhotomStep
    from jwst.flatfield import FlatFieldStep
    import jwst.datamodels

    OLD_CONTEXT = os.getenv("CRDS_CONTEXT")

    dm = jwst.datamodels.open(cal_file)

    os.environ["CRDS_CONTEXT"] = dm.meta.ref_file.crds.context_used
    set_crds_context()

    msg = f"convert_cal_to_rate: {cal_file}    use {os.environ['CRDS_CONTEXT']}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    cal_step = dm.meta.cal_step.instance

    pipeline_steps = {"photom": PhotomStep(), "flat_field": FlatFieldStep()}

    for step in pipeline_steps:
        if step not in cal_step:
            continue

        if cal_step[step] == "COMPLETE":
            msg = f"convert_cal_to_rate: {cal_file}   undo {step}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            dm = pipeline_steps[step].call(dm, inverse=True)
            dm.meta.cal_step.instance.pop(step)

    if write:
        rate_file = cal_file.replace("_cal", "_rate")
        msg = f"convert_cal_to_rate: {cal_file}  write {rate_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        dm.write(rate_file, overwrite=overwrite)

    # Reset CRDS_CONTEXT
    if OLD_CONTEXT is None:
        os.environ.pop("CRDS_CONTEXT")
    else:
        msg = f"convert_cal_to_rate: {cal_file} reset CRDS_CONTEXT = {OLD_CONTEXT}"
        utils.log_comment(utils.LOGFILE, msg)

        os.environ["CRDS_CONTEXT"] = OLD_CONTEXT

    return dm


def match_gwcs_to_sip(input, step=64, transform=None, verbose=True, overwrite=True):
    """
    Calculate transformation of gwcs to match SIP header, which may have been
    realigned (shift, rotation, scale)

    Parameters
    ----------
    input : str, `~astropy.io.fits.HDUList`
        FITS filename of a JWST image or a previously-opened
        `~astropy.io.fits.HDUList` with SIP wcs information stored in the
        first extension.

    step : int
        Step size of the pixel grid for calculating the tranformation

    transform : `skimage.transform`
        Transform object, e.g., `skimage.transform.SimilarityTransform`
        or `skimage.transform.Euclideanransform`

    verbose : bool
        Verbose messages

    overwrite : bool
        If True and ``input`` is a string, re-write to file

    Returns
    -------
    obj : `jwst.datamodels.image.ImageModel`
        Datamodel with updated WCS object.  The `REF` keywords are updated in
        `img[1].header`.

    Notes
    -----
    The scale factor of transformation is applied by multiplying the
    scale to the last parameters of the `distortion` WCS pipeline.  These
    might not necessarily be scale coefficients for all instrument WCS
    pipelines

    """
    from skimage.transform import SimilarityTransform

    if transform is None:
        transform = SimilarityTransform

    if isinstance(input, str):
        img = pyfits.open(input)
    elif isinstance(input, pyfits.HDUList):
        img = input

    if img[0].header["TELESCOP"] not in ["JWST"]:
        img = set_jwst_to_hst_keywords(img, reset=True)

    obj = img_with_wcs(img)
    # this should be put into `img_with_wcs` with more checks that it's being
    # applied correctly
    if "SCL_REF" in img[1].header:
        tr = obj.meta.wcs.pipeline[0].transform
        for i in range(-8, -2):
            setattr(tr, tr.param_names[i], tr.parameters[i] * img[1].header["SCL_REF"])
    else:
        if hasattr(transform, "scale"):
            img[1].header["SCL_REF"] = (1.0, "Transformation scale factor")

    wcs = pywcs.WCS(img[1].header, relax=True)

    sh = obj.data.shape

    if obj.meta.instrument.name in ["MIRI"]:
        xmin = 300
    else:
        xmin = step

    ymin = step

    xx = np.arange(xmin, sh[1] - 1, step)
    yy = np.arange(ymin, sh[0] - 1, step)

    yp, xp = np.meshgrid(yy, xx)

    rdg = obj.meta.wcs.forward_transform(xp, yp)
    rdw = wcs.all_pix2world(xp, yp, 0)

    Vg = np.array([rdg[0].flatten(), rdg[1].flatten()])
    Vw = np.array([rdw[0].flatten(), rdw[1].flatten()])

    r0 = np.median(Vw, axis=1)
    Vg = (Vg.T - r0).T
    Vw = (Vw.T - r0).T

    cosd = np.cos(r0[1] / 180 * np.pi)
    Vg[0, :] *= cosd
    Vw[0, :] *= cosd

    tf = transform()
    tf.estimate(Vg.T, Vw.T)

    asec = np.array(tf.translation) * np.array([1.0, 1.0]) * 3600
    rot_deg = tf.rotation / np.pi * 180

    Vt = tf(Vg.T).T
    resid = Vt - Vw

    if "PIXSCALE" in img[0].header:
        pscale = img[0].header["PIXSCALE"]
    else:
        pscale = utils.get_wcs_pscale(wcs)

    rms = [utils.nmad(resid[i, :]) * 3600 / pscale for i in [0, 1]]

    if hasattr(tf, "scale"):
        img[1].header["SCL_REF"] *= tf.scale
        _tfscale = tf.scale
    else:
        _tfscale = 1.0

    msg = f"Align to wcs: ({asec[0]:6.3f} {asec[1]:6.3f}) {_tfscale:7.5f}"
    msg += f" {rot_deg:7.5f} ; rms = {rms[0]:6.1e} {rms[1]:6.1e} pix"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    img[1].header["RA_REF"] += tf.translation[0] / cosd
    img[1].header["DEC_REF"] += tf.translation[1]
    img[1].header["ROLL_REF"] -= rot_deg

    obj = img_with_wcs(img)

    # Update scale parameters in transform, but parameters
    # might not be in correct order
    if "SCL_REF" in img[1].header:
        tr = obj.meta.wcs.pipeline[0].transform
        for i in range(-8, -2):
            setattr(tr, tr.param_names[i], tr.parameters[i] * img[1].header["SCL_REF"])

    if overwrite:
        img.writeto(img.filename(), overwrite=True)

    return obj


def get_phot_keywords(input, verbose=True):
    """
    Calculate conversions between JWST ``MJy/sr`` units and PHOTFLAM/PHOTFNU

    Parameters
    ----------
    input : str, `~astropy.io.fits.HDUList`
        FITS filename of a `cal`ibrated JWST image or a previously-opened
        `~astropy.io.fits.HDUList`

    Returns
    -------
    info : dict
        Photometric information

    verbose : bool
        Messaging to terminal.
    """
    import astropy.units as u

    if isinstance(input, str):
        img = pyfits.open(input, mode="update")
    elif isinstance(input, pyfits.HDUList):
        img = input

    # Get tabulated filter info
    filter_info = get_jwst_filter_info(img[0].header)

    # Get pixel area
    if "PIXAR_A2" in img["SCI"].header:
        pscale = np.sqrt(img["SCI"].header["PIXAR_A2"])
    elif "PIXSCALE" in img["SCI"].header:
        pscale = img["SCI"].header["PIXSCALE"]
    else:
        _wcs = pywcs.WCS(img["SCI"].header, relax=True)
        pscale = utils.get_wcs_pscale(_wcs)

    # Check image units
    if "OBUNIT" in img["SCI"].header:
        unit_key = "OBUNIT"
    else:
        unit_key = "BUNIT"

    if img["SCI"].header[unit_key].upper() == "MJy/sr".upper():
        in_unit = u.MJy / u.sr
        to_mjysr = 1.0
    else:
        if filter_info is None:
            in_unit = u.MJy / u.sr
            to_mjysr = -1.0
        else:
            if "photmjsr" in filter_info:
                in_unit = 1.0 * filter_info["photmjsr"] * u.MJy / u.sr
                to_mjysr = filter_info["photmjsr"]
            else:
                in_unit = u.MJy / u.sr
                to_mjysr = 1.0

    # Conversion factor
    pixel_area = (pscale * u.arcsec) ** 2
    tojy = (1 * in_unit).to(u.Jy / pixel_area).value

    # Pivot wavelength
    if filter_info is not None:
        plam = filter_info["pivot"] * 1.0e4
    else:
        plam = 5.0e4

    photflam = tojy * 2.99e-5 / plam**2
    _ZP = -2.5 * np.log10(tojy) + 8.9

    if verbose:
        msg = "# photometry keywords\n"
        msg += f"PHOTFNU = {tojy:.4e}\n"
        msg += f"PHOTPLAM = {plam:.1f}\n"
        msg += f"PHOTFLAM = {photflam:.4e}\n"
        msg += f"ZP = {_ZP:.2f}\n"
        msg += f"TO_MJYSR = {to_mjysr:.3f}\n"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

    # Set header keywords
    for e in [0, "SCI"]:
        img[e].header["PHOTFNU"] = tojy, "Scale factor to Janskys"
        img[e].header["PHOTPLAM"] = (plam, "Bandpass pivot wavelength, A")
        img[e].header["PHOTFLAM"] = (photflam, "Scale to erg/s/cm2/A")
        img[e].header["ZP"] = _ZP, "AB mag zeropoint"
        img[e].header["TO_MJYSR"] = (to_mjysr, "Scale to MJy/sr")

    # Drizzlepac needs ELECTRONS/S
    if "OBUNIT" not in img["SCI"].header:
        img["SCI"].header["OBUNIT"] = (
            img["SCI"].header["BUNIT"],
            "Original image units",
        )

    img["SCI"].header["BUNIT"] = "ELECTRONS/S"

    # Write FITS file if filename provided as input
    if isinstance(input, str):
        img.writeto(input, overwrite=True)
        img.close()

    info = {
        "photfnu": tojy,
        "photplam": plam,
        "photflam": img[0].header["PHOTFLAM"],
        "zp": img[0].header["ZP"],
        "tomjysr": to_mjysr,
    }

    return info


ORIG_KEYS = [
    "TELESCOP",
    "INSTRUME",
    "DETECTOR",
    "FILTER",
    "PUPIL",
    "EXP_TYPE",
    "GRATING",
]


def copy_jwst_keywords(header, orig_keys=ORIG_KEYS, verbose=True):
    """
    Make copies of some header keywords that may need to be modified to
    force the pipeline / astrodrizzle to interpret the images in different
    ways

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        FITS header to modify.

    orig_keys : list
        List of keywords to copy with a prefix of "O" (e.g., "OTELESCOP").

    verbose : bool
        Print status messages.

    """
    for k in orig_keys:
        newk = "O" + k[:7]
        if newk not in header:
            if k in header:
                header[newk] = header[k]
                msg = f"{newk} = {k} {header[k]}"
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose)


def exposure_oneoverf_correction(
    file,
    axis=None,
    thresholds=[5, 4, 3],
    erode_mask=None,
    manual_mask=None,
    nirspec_prism_mask=False,
    dilate_iterations=3,
    deg_pix=64,
    make_plot=True,
    init_model=0,
    in_place=False,
    skip_miri=True,
    force_oneoverf=False,
    verbose=True,
    **kwargs,
):
    """
    1/f correction for individual exposure

    1. Create a "background" mask with `sep`
    2. Identify sources above threshold limit in the background-subtracted
       image
    3. Iterate a row/column correction on threshold-masked images.  A
       chebyshev polynomial is fit to the correction array to try to isolate
       just the high-frequency oscillations.

    Parameters
    ----------
    file : str
        JWST raw image filename

    axis : int
        Axis over which to calculated the correction. If `None`, then defaults
        to ``axis=1`` (rows) for NIRCam and ``axis=1`` (columns) for NIRISS.

    thresholds : list
        List of source identification thresholds

    erode_mask : bool
        Erode the source mask to try to remove individual pixels that satisfy
        the S/N threshold.  If `None`, then set to False if the exposure is a
        NIRISS dispersed image to avoid clipping compact high-order spectra
        from the mask and True otherwise (for NIRISS imaging and NIRCam
        generally).

    manual_mask : array-like, None
        Manually-defined mask with valid pixels set to True.  Should have the same
        dimensions as the exposure data, i.e., (2048, 2048).

    nirspec_prism_mask : bool
        Make an automatic mask for NIRSpec PRISM exposures for axis=1 mask using only
        pixels that won't have PRISM spectra.

    dilate_iterations : int
        Number of `binary_dilation` iterations of the source mask

    deg_pix : int
        Scale in pixels for each degree of the smooth chebyshev polynomial

    make_plot : bool
        Make a diagnostic plot

    init_model : scalar, array-like
        Initial correction model, e.g., for doing both axes

    in_place : bool
        If True, remove the model from the 'SCI' extension of ``file``

    skip_miri : bool
        Don't run on MIRI exposures

    force_oneoverf : bool
        Force the correction even if the `ONEFEXP` keyword is already set

    verbose : bool
        Print status messages

    Returns
    -------
    fig : `~matplotlib.figure.Figure`, None
        Diagnostic figure if `make_plot=True`

    model : array-like
        The row- or column-average correction array
    """
    import numpy as np
    from numpy.polynomial import Chebyshev
    import scipy.ndimage as nd
    import matplotlib.pyplot as plt

    import astropy.io.fits as pyfits
    import sep

    im = pyfits.open(file)
    if (im[0].header["INSTRUME"] in "MIRI") & (skip_miri):
        im.close()

        msg = "exposure_oneoverf_correction: Skip for MIRI"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        return None, 0

    if ("ONEFEXP" in im[0].header) and im[0].header["ONEFEXP"] and (not force_oneoverf):
        im.close()

        msg = "exposure_oneoverf_correction: Skip, already corrected"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        return None, 0

    if axis is None:
        if im[0].header["INSTRUME"] in ("NIRISS", "NIRSPEC"):
            axis = 0
        else:
            axis = 1

    elif axis < 0:
        # Opposite axis
        if im[0].header["INSTRUME"] in ("NIRISS", "NIRSPEC"):
            axis = 1
        else:
            axis = 0

    msg = f"exposure_oneoverf_correction: {file} axis={axis} deg_pix={deg_pix}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    prism_mask = None
    if nirspec_prism_mask & (axis == 1):
        if im[0].header["INSTRUME"] == "NIRSPEC":
            if im[0].header["GRATING"] == "PRISM":
                if im[0].header["EXP_TYPE"] == "NRS_IFU":
                    sflat_file = "sflat_{GRATING}-{FILTER}_{DETECTOR}.fits".format(
                        im[0].header
                    ).lower()
                    if os.path.exists(sflat_file):
                        with pyfits.open(sflat_file) as sflat_:
                            has_sflat = np.isfinite(sflat_[0].data)
                            prism_mask = ~nd.binary_dilation(has_sflat, iterations=4)

                        msg = f"exposure_oneoverf_correction: PRISM mask from {sflat_file}"
                        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

                if prism_mask is None:
                    if im[0].header["DETECTOR"] == "NRS1":
                        empty_slice = (4, 570)
                    else:
                        empty_slice = (1160, 2040)

                    prism_mask = np.zeros(im["SCI"].data.shape, dtype=bool)
                    prism_mask[:, slice(*empty_slice)] = True
                    msg = (
                        f"exposure_oneoverf_correction: PRISM empty mask {empty_slice}"
                    )
                    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    if im[0].header["INSTRUME"] in ("NIRSPEC"):
        erode_mask = False

    if erode_mask is None:
        if im[0].header["FILTER"].startswith("GR150"):
            erode_mask = False
        elif im[0].header["PUPIL"].startswith("GRISM"):
            erode_mask = False
        else:
            erode_mask = True

    dq = utils.mod_dq_bits(im["DQ"].data, okbits=4)
    dqmask = dq == 0
    mask = dqmask

    err = im["ERR"].data
    dqmask &= (err > 0) & np.isfinite(err)

    sci = im["SCI"].data.astype(np.float32) - init_model
    if deg_pix == 0:
        bw = sci.shape[0] // 64
    else:
        bw = deg_pix

    bkg = sep.Background(sci, mask=~dqmask, bw=bw, bh=bw)
    back = bkg.back()

    sn_mask = (sci - back) / err > thresholds[0]
    if erode_mask:
        sn_mask = nd.binary_erosion(sn_mask)

    sn_mask = nd.binary_dilation(sn_mask, iterations=dilate_iterations)

    mask = dqmask & ~sn_mask
    if prism_mask is not None:
        mask = dqmask & prism_mask

    if manual_mask is not None:
        msg = f"exposure_oneoverf_correction: manual_mask {manual_mask.sum()}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        mask &= manual_mask

    cheb = 0

    if make_plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    else:
        fig = None

    for _iter, thresh in enumerate(thresholds):

        if deg_pix == 0:
            sci = im["SCI"].data * 1.0 - init_model
        else:
            sci = im["SCI"].data * 1.0 - back - init_model

        sci[~mask] = np.nan
        med = np.nanmedian(sci, axis=axis)

        if axis == 0:
            model = np.zeros_like(sci) + (med - cheb)
        else:
            model = (np.zeros_like(sci) + (med - cheb)).T

        sn_mask = ((sci - model) / err > thresh) & dqmask
        if erode_mask:
            sn_mask = nd.binary_erosion(sn_mask)

        sn_mask = nd.binary_dilation(sn_mask, iterations=dilate_iterations)
        mask = dqmask & ~sn_mask
        mask &= (sci - model) / err > -thresh

        if prism_mask is not None:
            mask = dqmask & prism_mask

        if manual_mask is not None:
            mask &= manual_mask

        if make_plot:
            ax.plot(med, alpha=0.5)

    nx = med.size
    xarr = np.linspace(-1, 1, nx)
    ok = np.isfinite(med)

    if deg_pix == 0:
        # Don't remove anything from the median profile
        cheb = 0.0
        deg = -1

    elif deg_pix >= nx:
        # Remove constant component
        cheb = np.nanmedian(med)
        deg = 0

    else:
        # Remove smooth component
        deg = nx // deg_pix

        for _iter in range(3):
            cfit = Chebyshev.fit(xarr[ok], med[ok], deg=deg)
            cheb = cfit(xarr)
            ok = np.isfinite(med) & (np.abs(med - cheb) < 0.05)

        if make_plot:
            ax.plot(np.arange(nx)[ok], cheb[ok], color="r")

    if axis == 0:
        model = np.zeros_like(sci) + (med - cheb)
    else:
        model = (np.zeros_like(sci) + (med - cheb)).T

    if make_plot:
        ax.set_title(f"{file} axis={axis}")
        ax.grid()

        fig.tight_layout(pad=0)

    im.close()

    if in_place:
        msg = f"exposure_oneoverf_correction: {file} apply to file"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        with pyfits.open(file, mode="update") as im:
            im[0].header["ONEFEXP"] = True, "Exposure 1/f correction applied"
            im[0].header["ONEFAXIS"] = axis, "Axis for 1/f correction"
            im[0].header["ONEFDEG"] = deg, "Degree of smooth component"
            im[0].header["ONEFNPIX"] = deg_pix, "Pixels per smooth degree"

            model[~np.isfinite(model)] = 0

            im["SCI"].data -= model
            im.flush()

        if make_plot:
            fig.savefig(file.split(".fits")[0] + f"_onef_axis{axis}.png")
            plt.close("all")

    return fig, model


def initialize_jwst_image(
    filename,
    verbose=True,
    max_dq_bit=14,
    orig_keys=ORIG_KEYS,
    oneoverf_correction=True,
    oneoverf_kwargs={"make_plot": False},
    use_skyflats=True,
    nircam_edge=8,
):
    """
    Make copies of some header keywords to make the headers look like
    and HST instrument

    1) Apply gain correction [*NOT PERFORMED*]
    2) Clip DQ bits
    3) Copy header keywords
    4) Apply exposure-level 1/f correction
    5) Apply flat field if necessary
    6) Initalize WCS

    Parameters
    ----------
    filename : str
        Filename of the JWST exposure.

    verbose : bool
        Messaging to terminal.

    max_dq_bit : int
        Maximum DQ bit to allow in the clipped DQ array.

    orig_keys : list
        List of keywords to copy with a prefix of "O" (e.g., "OTELESCOP").

    oneoverf_correction : bool
        Apply 1/f correction to the exposure if True.

    oneoverf_kwargs : dict
        Keyword arguments for `exposure_oneoverf_correction`.

    use_skyflats : bool
        Apply skyflat correction if True.

    nircam_edge : int
        Number of pixels to trim from the edges of NIRCam exposures.

    Returns
    -------
    status : bool
        True if finished successfully

    """
    frame = inspect.currentframe()
    utils.log_function_arguments(
        utils.LOGFILE, frame, "jwst_utils.initialize_jwst_image"
    )

    import gc

    import astropy.io.fits as pyfits
    import scipy.ndimage as nd

    from jwst.flatfield import FlatFieldStep
    from jwst.gain_scale import GainScaleStep

    set_quiet_logging(QUIET_LEVEL)

    _ = set_crds_context()

    img = pyfits.open(filename)

    if "OTELESCO" in img[0].header:
        tel = img[0].header["OTELESCO"]
    elif "TELESCOP" in img[0].header:
        tel = img[0].header["TELESCOP"]
    else:
        tel = None

    if tel not in ["JWST"]:
        msg = f'TELESCOP keyword ({tel}) not "JWST"'
        # utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        raise ValueError(msg)

    # if img['SCI'].header['BUNIT'].upper() == 'DN/S':
    #     gain_file = GainScaleStep().get_reference_file(img, 'gain')
    #
    #     with pyfits.open(gain_file) as gain_im:
    #         gain_median = np.median(gain_im[1].data)
    #
    #     img[0].header['GAINFILE'] = gain_file
    #     img[0].header['GAINCORR'] = True, 'Manual gain correction applied'
    #     img[0].header['GAINVAL'] = gain_median, 'Gain value applied'
    #
    #     msg = f'GAINVAL = {gain_median:.2f}\n'
    #     msg += f'GAINFILE = {gain_file}'
    #     utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    #
    #     img['SCI'].data *= gain_median
    #     img['SCI'].header['BUNIT'] = 'ELECTRONS/S'
    #     img['ERR'].data *= gain_median
    #     img['ERR'].header['BUNIT'] = 'ELECTRONS/S'
    #
    #     for k in ['VAR_POISSON','VAR_RNOISE','VAR_FLAT']:
    #         if k in img:
    #             img[k].data *= gain_median**2

    copy_jwst_keywords(img[0].header, orig_keys=orig_keys, verbose=verbose)
    img[0].header["PA_V3"] = img[1].header["PA_V3"]

    if "ENGQLPTG" in img[0].header:
        if img[0].header["ENGQLPTG"] == "CALCULATED_TRACK_TR_202111":
            msg = f"ENGQLPTG = CALCULATED_TR_202105"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            img[0].header["ENGQLPTG"] = "CALCULATED_TR_202105"

    if "PATTTYPE" in img[0].header:
        if img[0].header["PATTTYPE"].endswith("WITH-NIRCAM"):
            patt = img[0].header["PATTTYPE"]
            new_patt = patt.replace("NIRCAM", "NIRCam")
            msg = f"PATTTYPE {patt} > {new_patt}"
            img[0].header["PATTTYPE"] = new_patt

    for k in ["TARGET", "TARGNAME"]:
        if k in img[0].header:
            targ = img[0].header[k].replace(" ", "-")
            targ = targ.replace(";", "-")
            msg = f"{k} > {targ} (no spaces)"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            img[0].header[k] = targ

    # Get flat field ref file
    _flatfile = FlatFieldStep().get_reference_file(img, "flat")
    img[0].header["PFLTFILE"] = os.path.basename(_flatfile)
    msg = f"PFLTFILE = {_flatfile}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    # Clip DQ keywords
    img[0].header["MAXDQBIT"] = max_dq_bit, "Max DQ bit allowed"
    msg = f"Clip MAXDQBIT = {max_dq_bit}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    dq = np.zeros_like(img["DQ"].data)
    dq[img["DQ"].data >= 2 ** (max_dq_bit + 1)] = 2**max_dq_bit
    dqm = img["DQ"].data > 0

    for bit in range(max_dq_bit + 1):
        dq[dqm] |= img["DQ"].data[dqm] & 2**bit

    dq[img["DQ"].data < 0] = 2**bit

    if img[0].header["OINSTRUM"] == "MIRI":
        for b in [2, 4]:
            dq4 = (dq & b > 0).sum()
            if dq4 / dq.size > 0.4:
                msg = f"Unset MIRI DQ bit={b}"
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
                dq -= dq & b
                # dq -= dq & 2

        # msg = f'Mask left side of MIRI'
        # utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        # dq[:,:302] |= 1024

        # Dilate MIRI mask
        msg = f"initialize_jwst_image: Dilate MIRI window mask"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        edge = nd.binary_dilation(((dq & 2**9) > 0), iterations=6)
        dq[edge] |= 1024

    elif img[0].header["OINSTRUM"] == "NIRCAM":
        _det = img[0].header["DETECTOR"]

        bpfiles = [
            os.path.join(
                os.path.dirname(__file__), f"data/nrc_badpix_240112_{_det}.fits.gz"
            )
        ]
        bpfiles += [
            os.path.join(
                os.path.dirname(__file__), f"data/nrc_badpix_231206_{_det}.fits.gz"
            )
        ]
        bpfiles += [
            os.path.join(
                os.path.dirname(__file__), f"data/nrc_badpix_20230710_{_det}.fits.gz"
            )
        ]
        bpfiles += [
            os.path.join(
                os.path.dirname(__file__), f"data/nrc_badpix_230120_{_det}.fits.gz"
            )
        ]
        bpfiles += [
            os.path.join(
                os.path.dirname(__file__), f"data/nrc_lowpix_0916_{_det}.fits.gz"
            )
        ]

        for bpfile in bpfiles:
            if os.path.exists(bpfile) & False:
                bpdata = pyfits.open(bpfile)[0].data
                bpdata = nd.binary_dilation(bpdata > 0) * 1024
                if dq.shape == bpdata.shape:
                    msg = f"initialize_jwst_image: Use extra badpix in {bpfile}"
                    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
                    dq |= bpdata.astype(dq.dtype)

                break

        # if _det in ['NRCALONG','NRCBLONG']:
        if True:
            msg = f"initialize_jwst_image: Mask outer ring of {nircam_edge} pixels"
            msg += f" for {_det}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            dq[:nircam_edge, :] |= 1024
            dq[-nircam_edge:, :] |= 1024
            dq[:, :nircam_edge] |= 1024
            dq[:, -nircam_edge:] |= 1024

    img["DQ"].data = dq

    img[0].header["EXPTIME"] = img[0].header["EFFEXPTM"] * 1

    img[1].header["NGOODPIX"] = (dq == 0).sum()
    img[1].header["EXPNAME"] = img[0].header["EXPOSURE"]
    img[1].header["MEANDARK"] = 0.0

    for _ext in [0, "SCI"]:
        for k in list(img[_ext].header.keys()):
            if k[:3] in ["AP_", "BP_"]:
                img[_ext].header.remove(k)

    # AstroDrizzle needs a time extension, which can be empty
    # but with a PIXVALUE keyword.
    # The header below is designed after WFC3/IR

    img.writeto(filename, overwrite=True)
    img.close()

    _nircam_grism = False

    ### Flat-field
    # Flat-field first?

    needs_flat = True

    if oneoverf_correction:
        if "deg_pix" in oneoverf_kwargs:
            if oneoverf_kwargs["deg_pix"] == 2048:
                # Do flat field now for aggressive 1/f correction, since the pixel-level
                # 1/f correction takes out structure that should be flat-fielded
                _ = img_with_flat(filename, overwrite=True, use_skyflats=use_skyflats)
                needs_flat = False

        # NIRCam grism
        if img[0].header["OINSTRUM"] == "NIRCAM":
            if "GRISM" in img[0].header["OPUPIL"]:
                _nircam_grism = True

        if not _nircam_grism:

            try:
                _ = exposure_oneoverf_correction(
                    filename, in_place=True, **oneoverf_kwargs
                )
            except ValueError:
                # Should only fail for test data
                utils.log_exception(utils.LOGFILE, traceback)
                msg = f"exposure_oneoverf_correction: failed for {filename}"
                utils.log_comment(utils.LOGFILE, msg)

                pass

            if "other_axis" in oneoverf_kwargs:
                if oneoverf_kwargs["other_axis"]:
                    try:
                        _ = exposure_oneoverf_correction(
                            filename, in_place=True, axis=-1, **oneoverf_kwargs
                        )
                    except ValueError:
                        # Should only fail for test data
                        utils.log_exception(utils.LOGFILE, traceback)
                        msg = f"exposure_oneoverf_correction: axis=-1 failed for {filename}"
                        utils.log_comment(utils.LOGFILE, msg)

                        pass

    ### Flat-field
    if needs_flat:
        _ = img_with_flat(filename, overwrite=True, use_skyflats=use_skyflats)

    # Now do "1/f" correction to subtract NIRCam grism sky
    if _nircam_grism:
        if img[0].header["OPUPIL"] == "GRISMR":
            _disp_axis = 0
        else:
            _disp_axis = 1

        msg = f"exposure_oneoverf_correction: NIRCam grism sky {filename}"
        utils.log_comment(utils.LOGFILE, msg)

        # Subtract average along dispersion axis
        exposure_oneoverf_correction(
            filename,
            in_place=True,
            erode_mask=False,
            thresholds=[4, 3],
            axis=_disp_axis,
            dilate_iterations=5,
            deg_pix=0,
        )

    _ = img_with_wcs(filename, overwrite=True)

    get_phot_keywords(filename)

    # Add TIME extension
    if "TIME" not in img:
        img = pyfits.open(filename)

        time = pyfits.ImageHDU(data=img["SCI", 1].data)
        # np.ones_like(img['SCI',1].data)*img[0].header['EXPTIME'])
        time.data = None
        time.header["EXTNAME"] = "TIME"
        time.header["EXTVER"] = 1
        time.header["PIXVALUE"] = img[0].header["EXPTIME"] * 1.0
        time.header["BUNIT"] = "SECONDS"
        time.header["NPIX1"] = img["SCI"].header["NAXIS1"] * 1
        time.header["NPIX2"] = img["SCI"].header["NAXIS2"] * 1
        time.header["INHERIT"] = True

        img.append(time)
        img.writeto(filename, overwrite=True)
        img.close()

    gc.collect()
    return True


# # for NIRISS images; NIRCam,MIRI TBD
# # band: [photflam, photfnu, pivot_wave]
# NIS_PHOT_KEYS = {'F090W': [1.098934e-20, 2.985416e-31, 0.9025],
#                  'F115W': [6.291060e-21, 2.773018e-31, 1.1495],
#                  'F140M': [9.856255e-21, 6.481079e-31, 1.4040],
#                  'F150W': [4.198384e-21, 3.123540e-31, 1.4935],
#                  'F158M': [7.273483e-21, 6.072128e-31, 1.5820],
#                  'F200W': [2.173398e-21, 2.879494e-31, 1.9930],
#                  'F277W': [1.109150e-21, 2.827052e-31, 2.7643],
#                  'F356W': [6.200034e-22, 2.669862e-31, 3.5930],
#                  'F380M': [2.654520e-21, 1.295626e-30, 3.8252],
#                  'F430M': [2.636528e-21, 1.613895e-30, 4.2838],
#                  'F444W': [4.510426e-22, 2.949531e-31, 4.4277],
#                  'F480M': [1.879639e-21, 1.453752e-30, 4.8152]}
#


def set_jwst_to_hst_keywords(
    input, reset=False, verbose=True, orig_keys=ORIG_KEYS, oneoverf_correction=True
):
    """
    Make primary header look like an HST instrument

    Parameters
    ----------
    input : str, `~astropy.io.fits.HDUList`
        Filename or FITS HDUList object to modify.

    reset : bool
        Reset original JWST keywords to their original values.

    verbose : bool
        Messaging to terminal.

    orig_keys : list
        List of keywords to copy with a prefix of "O" (e.g., "OTELESCOP").

    oneoverf_correction : bool
        Apply 1/f correction to the exposure if True.

    Returns
    -------
    img : `~astropy.io.fits.HDUList`
        Modified FITS HDUList object.

    """
    frame = inspect.currentframe()
    utils.log_function_arguments(
        utils.LOGFILE, frame, "jwst_utils.set_jwst_to_hst_keywords"
    )

    import astropy.io.fits as pyfits

    if isinstance(input, str):
        img = pyfits.open(input)
    else:
        img = input

    HST_KEYS = {"TELESCOP": "HST", "INSTRUME": "WFC3", "DETECTOR": "IR"}

    if "OTELESCO" not in img[0].header:
        _status = initialize_jwst_image(
            input, oneoverf_correction=oneoverf_correction, verbose=verbose
        )

        # Reopen
        if isinstance(input, str):
            img = pyfits.open(input, mode="update")
        else:
            img = input

    if reset:
        for k in orig_keys:
            newk = "O" + k[:7]
            if newk in img[0].header:
                img[0].header[k] = img[0].header[newk]
                msg = f"Reset: {k} > {img[0].header[newk]} ({newk})"
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
    else:
        for k in HST_KEYS:
            img[0].header[k] = HST_KEYS[k]
            msg = f"  Set: {k} > {HST_KEYS[k]}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    # for x in 'ABCD':
    #     if 'GAINVAL' in img[0].header:
    #         gain = img[0].header['GAINVAL']
    #     else:
    #         gain = 1.0
    #
    #     img[0].header[f'ATODGN{x}'] = gain
    #     img[0].header[f'READNSE{x}'] = 12.9

    # TIME keyword seems to get corrupted?
    if "TIME" in img:
        img["TIME"].header["PIXVALUE"] = img[0].header["EXPTIME"]

    if isinstance(input, str):
        img.writeto(input, overwrite=True)

    return img


def strip_telescope_header(header, simplify_wcs=True):
    """
    Strip non-JWST keywords that confuse `jwst.datamodels.util.open`.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Input FITS header.

    simplify_wcs : bool
        Simplify the WCS header to just the CD matrix.

    Returns
    -------
    new_header : `~astropy.io.fits.Header`
        Modified FITS header.

    """
    new_header = header.copy()

    if "TELESCOP" in new_header:
        if new_header["TELESCOP"] != "JWST":
            keys = ["TELESCOP", "FILTER", "DETECTOR", "INSTRUME"]
            for key in keys:
                if key in header:
                    new_header.remove(key)

    if simplify_wcs:
        # Make simple WCS header
        orig_wcs = pywcs.WCS(new_header)
        new_header = orig_wcs.to_header()

        new_header["EXTNAME"] = "SCI"
        new_header["RADESYS"] = "ICRS"
        new_header["CDELT1"] = -new_header["PC1_1"]
        new_header["CDELT2"] = new_header["PC2_2"]
        new_header["PC1_1"] = -1
        new_header["PC2_2"] = 1

    return new_header


def wcs_from_datamodel(datamodel, **kwargs):
    """
    Initialize `~astropy.wcs.WCS` object from `wcsinfo` parameters, accounting
    for the aperture reference position that sets the tangent point

    Parameters
    ----------
    datamodel : `jwst.datamodels.image.ImageModel`

    kwargs : dict
        Keyword arguments passed to `~grizli.utils.wcs_from_header`

    Returns
    -------
    wcs : `~astropy.wcs.WCS`
    """
    header = pyfits.Header(datamodel.meta.wcsinfo.instance)
    header["NAXIS"] = 2

    sh = datamodel.data.shape

    header["NAXIS1"] = sh[1]
    header["NAXIS2"] = sh[0]

    # header['SIPCRPX1'] = header['siaf_xref_sci']
    # header['SIPCRPX2'] = header['siaf_yref_sci']

    wcs = utils.wcs_from_header(header, **kwargs)

    return wcs


LSQ_ARGS = dict(
    jac="2-point",
    bounds=(-np.inf, np.inf),
    method="trf",
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-12,
    x_scale=1.0,
    loss="soft_l1",
    f_scale=1000.0,
    diff_step=1.0e-6,
    tr_solver=None,
    tr_options={},
    jac_sparsity=None,
    max_nfev=100,
    verbose=0,
    kwargs={},
)


def model_wcs_header(
    datamodel,
    get_sip=True,
    degree=4,
    fit_crval=True,
    fit_rot=True,
    fit_scale=True,
    step=32,
    crpix=None,
    lsq_args=LSQ_ARGS,
    get_guess=True,
    set_diff_step=True,
    initial_header=None,
    fast_coeffs=True,
    uvxy=None,
    **kwargs,
):
    """
    Make a header with a better SIP WCS derived from the JWST `gwcs` object

    Parameters
    ----------
    datamodel : `jwst.datamodels.ImageModel`
        Image model with full `~gwcs` in `with_wcs.meta.wcs`.

    get_sip : bool
        If True, fit a `astropy.modeling.models.SIP` distortion model to the
        image WCS.

    degree : int
        Degree of the SIP polynomial model.

    fit_crval, fit_rot, fit_scale : bool
        Fit the CRVAL, rotation, and scale of the SIP model.

    step : int
        For fitting the SIP model, generate a grid of detector pixels every
        `step` pixels in both axes for passing through
        `datamodel.meta.wcs.forward_transform`.

    crpix : (float, float)
        Refernce pixel.  If `None` set to the array center.

    lsq_args : dict
        Arguments for `scipy.optimize.least_squares`.

    get_guess : bool
        Get initial guess for SIP coefficients from the `datamodel.meta.wcs`.

    set_diff_step : bool
        Set `lsq_args['diff_step']` to the pixel scale.

    initial_header : None or `~astropy.io.fits.Header`
        Initial header to use as a guess for the SIP fit.

    fast_coeffs : bool
        Use a fast method to compute the SIP coefficients.

    uvxy : None or (array, array, array, array)
        Manually specify detector and target coordinates for positions to use
        for the SIP fit, where ``uvxy = (x, y, ra, dec)`` and the detector
        coordinates are zero-index.  If not specified, make a grid with ``step``

    Returns
    -------
    header : '~astropy.io.fits.Header`
        Header with simple WCS definition: CD rotation but no distortion.

    """
    from astropy.io.fits import Header
    from scipy.optimize import least_squares
    import jwst.datamodels

    set_quiet_logging()

    datamodel = jwst.datamodels.open(datamodel)
    sh = datamodel.data.shape

    if "order" in kwargs:
        msg = "WARNING: Keyword `order` has been renamed to `degree`"
        print(msg)
        degree = kwargs["order"]

    if crpix is None:
        crpix = np.array(sh)[::-1] / 2.0 + 0.5
    else:
        crpix = np.array(crpix)

    crp0 = crpix - 1

    crval = datamodel.meta.wcs.forward_transform(crp0[0], crp0[1])
    cdx = datamodel.meta.wcs.forward_transform(crp0[0] + 1, crp0[1])
    cdy = datamodel.meta.wcs.forward_transform(crp0[0], crp0[1] + 1)

    # use utils.to_header in grizli to replace the below (from datamodel.wcs)
    header = Header()
    header["RADESYS"] = "ICRS"
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"

    header["CUNIT1"] = header["CUNIT2"] = "deg"

    header["CRPIX1"] = crpix[0]
    header["CRPIX2"] = crpix[1]

    header["CRVAL1"] = crval[0]
    header["CRVAL2"] = crval[1]

    cosd = np.cos(crval[1] / 180 * np.pi)

    header["CD1_1"] = (cdx[0] - crval[0]) * cosd
    header["CD1_2"] = (cdy[0] - crval[0]) * cosd

    header["CD2_1"] = cdx[1] - crval[1]
    header["CD2_2"] = cdy[1] - crval[1]

    cd = np.array(
        [[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]]
    )

    if not get_sip:
        return header

    # Fit a SIP header to the gwcs transformed coordinates
    if datamodel.meta.instrument.name in ["MIRI"]:
        xmin = 300
        ymin = step
    else:
        xmin = step
        ymin = step

    if uvxy is None:
        u, v = np.meshgrid(
            np.arange(xmin, sh[1] - 1, step), np.arange(ymin, sh[0] - 1, step)
        )
        x, y = datamodel.meta.wcs.forward_transform(u, v)
    else:
        u, v, x, y = uvxy

    a_names = []
    b_names = []

    a_rand = []
    b_rand = []

    sip_step = []

    for i in range(degree + 1):
        for j in range(degree + 1):
            ext = "{0}_{1}".format(i, j)
            if (i + j) > degree:
                continue

            if ext in ["0_0", "0_1", "1_0"]:
                continue

            a_names.append("A_" + ext)
            b_names.append("B_" + ext)
            sip_step.append(1.0e-3 ** (i + j))

    Npre = fit_rot * 1 + fit_scale * 1 + fit_crval * 2
    Nparam = Npre + len(a_names) + len(b_names)

    p0 = np.zeros(Nparam)
    # p0[:4] += cd.flatten()

    sip_step = np.array(sip_step)
    # p0[4:len(a_names)+4] = np.random.normal(size=len(a_names))*sip_step
    # p0[4+len(a_names):] = np.random.normal(size=len(a_names))*sip_step

    if set_diff_step:
        diff_step = np.hstack([np.ones(Npre) * 0.01 / 3600, sip_step, sip_step])
        lsq_args["diff_step"] = diff_step
        print("xxx", len(p0), len(diff_step))

    if datamodel.meta.instrument.name == "NIRISS":
        a0 = {
            "A_0_2": 3.8521180058449584e-08,
            "A_0_3": -1.2910469982047994e-11,
            "A_0_4": 3.642187826984494e-15,
            "A_1_1": -8.156851592950884e-08,
            "A_1_2": -1.2336474525621777e-10,
            "A_1_3": 1.1169942988845159e-13,
            "A_2_0": 3.5236920263776116e-07,
            "A_2_1": -9.622992486408194e-11,
            "A_2_2": -2.1150777639693208e-14,
            "A_3_0": -3.517117816321703e-11,
            "A_3_1": 1.252016786545716e-13,
            "A_4_0": -2.5596007366022595e-14,
        }
        b0 = {
            "B_0_2": -6.478494215243917e-08,
            "B_0_3": -4.2460992201562465e-10,
            "B_0_4": 2.501714355762585e-13,
            "B_1_1": 4.127407304584838e-07,
            "B_1_2": -2.774351986369079e-11,
            "B_1_3": 3.4947161649623674e-15,
            "B_2_0": -7.509503977158588e-07,
            "B_2_1": -2.1263593068617203e-10,
            "B_2_2": 1.3621493497144034e-13,
            "B_3_0": -2.099145095489808e-11,
            "B_3_1": -1.613481283521298e-14,
            "B_4_0": 2.38606562938391e-14,
        }

        for i, k in enumerate(a_names):
            if k in a0:
                p0[Npre + i] = a0[k]

        for i, k in enumerate(b_names):
            if k in b0:
                p0[Npre + len(b_names) + i] = b0[k]

    elif get_guess:

        if datamodel.meta.instrument.name in ["MIRI"]:
            xmin = 300
        else:
            xmin = step

        h = datamodel.meta.wcs.to_fits_sip(
            degree=degree,
            crpix=crpix,
            bounding_box=((xmin, sh[1] - step), (step, sh[0] - step)),
        )

        cd = np.array([[h["CD1_1"], h["CD1_2"]], [h["CD2_1"], h["CD2_2"]]])
        # p0[:Npre] = cd.flatten()*1

        a0 = {}
        b0 = {}

        for k in h:
            if k.startswith("A_") & ("ORDER" not in k):
                a0[k] = h[k]
            elif k.startswith("B_") & ("ORDER" not in k):
                b0[k] = h[k]

        for i, k in enumerate(a_names):
            if k in a0:
                p0[Npre + i] = a0[k]

        for i, k in enumerate(b_names):
            if k in b0:
                p0[Npre + len(b_names) + i] = b0[k]

    elif initial_header is not None:
        h = initial_header
        cd = np.array([[h["CD1_1"], h["CD1_2"]], [h["CD2_1"], h["CD2_2"]]])
        # p0[:Npre] = cd.flatten()*1

        a0 = {}
        b0 = {}

        for k in h:
            if k.startswith("A_") & ("ORDER" not in k):
                a0[k] = h[k]
            elif k.startswith("B_") & ("ORDER" not in k):
                b0[k] = h[k]

        for i, k in enumerate(a_names):
            if k in a0:
                p0[Npre + i] = a0[k]

        for i, k in enumerate(b_names):
            if k in b0:
                p0[Npre + len(b_names) + i] = b0[k]

    # args = (u.flatten(), v.flatten(), x.flatten(), y.flatten(), crpix, a_names, b_names, cd, 0)
    fit_type = fit_rot * 1 + fit_scale * 2 + fit_crval * 4

    args = (
        u.flatten(),
        v.flatten(),
        x.flatten(),
        y.flatten(),
        crval,
        crpix,
        a_names,
        b_names,
        cd,
        fit_type,
        0,
    )

    # Fit the SIP coeffs
    if fast_coeffs:
        _func = _objective_lstsq_sip
        p0 = p0[:Npre]
        # lsq_args['diff_step'] = 0.01

    else:
        _func = _objective_sip
        # lsq_args['diff_step'] = 1.e-6

    fit = least_squares(_func, p0, args=args, **lsq_args)

    # Get the results
    args = (
        u.flatten(),
        v.flatten(),
        x.flatten(),
        y.flatten(),
        crval,
        crpix,
        a_names,
        b_names,
        cd,
        fit_type,
        1,
    )

    _ = _func(fit.x, *args)
    pp, cd_fit, crval_fit, a_coeff, b_coeff, ra_nmad, dec_nmad = _

    header["CRVAL1"] = crval_fit[0]
    header["CRVAL2"] = crval_fit[1]

    # Put in the header
    for i in range(2):
        for j in range(2):
            header["CD{0}_{1}".format(i + 1, j + 1)] = cd_fit[i, j]

    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"

    header["NAXIS"] = 2
    sh = datamodel.data.shape
    header["NAXIS1"] = sh[1]
    header["NAXIS2"] = sh[0]

    header["A_ORDER"] = degree
    for k in a_coeff:
        header[k] = a_coeff[k]

    header["B_ORDER"] = degree
    for k in b_coeff:
        header[k] = b_coeff[k]

    header["PIXSCALE"] = utils.get_wcs_pscale(header), "Derived pixel scale"

    header["SIP_ROT"] = pp[0], "SIP fit CD rotation, deg"
    header["SIP_SCL"] = pp[1], "SIP fit CD scale"
    header["SIP_DRA"] = pp[2][0], "SIP fit CRVAL1 offset, arcsec"
    header["SIP_DDE"] = pp[2][1], "SIP fit CRVAL1 offset, arcsec"

    header["SIPSTATU"] = fit.status, "least_squares result status"
    header["SIPCOST"] = fit.cost, "least_squares result cost"
    header["SIPNFEV"] = fit.nfev, "least_squares result nfev"
    header["SIPNJEV"] = fit.njev, "least_squares result njev"
    header["SIPOPTIM"] = fit.optimality, "least_squares result optimality"
    header["SIPSUCSS"] = fit.success, "least_squares result success"
    header["SIPFAST"] = fast_coeffs, "SIP fit with fast least squares"

    if fast_coeffs:
        _xnmad = ra_nmad
        _ynmad = dec_nmad
    else:
        _xnmad = ra_nmad / header["PIXSCALE"]
        _ynmad = dec_nmad / header["PIXSCALE"]

    header["SIPRAMAD"] = _xnmad, "RA NMAD, pix"
    header["SIPDEMAD"] = _ynmad, "Dec NMAD, pix"

    header["CRDS_CTX"] = (
        datamodel.meta.ref_file.crds.context_used,
        "CRDS context file",
    )

    bn = os.path.basename
    try:
        header["DISTFILE"] = (
            bn(datamodel.meta.ref_file.distortion.name),
            "Distortion reference file",
        )
    except TypeError:
        header["DISTFILE"] = "N/A", "Distortion reference file (failed)"

    try:
        header["FOFFFILE"] = (
            bn(datamodel.meta.ref_file.filteroffset.name),
            "Filter offset reference file",
        )
    except TypeError:
        header["FOFFFILE"] = "N/A", "Filter offset reference file (failed)"

    return header


def pipeline_model_wcs_header(
    datamodel,
    step=64,
    degrees=[3, 4, 5, 5],
    lsq_args=LSQ_ARGS,
    crpix=None,
    verbose=True,
    initial_header=None,
    max_rms=1.0e-4,
    set_diff_step=False,
    get_guess=False,
    fast_coeffs=True,
    uvxy=None,
):
    """
    Iterative pipeline to refine the SIP headers

    Parameters
    ----------
    datamodel : `jwst.datamodels.ImageModel`
        Input image model.

    step : int
        Step size for fitting the SIP model.

    degrees : list
        List of SIP polynomial degrees to try.

    lsq_args : dict
        Arguments for `scipy.optimize.least_squares`.

    crpix : None or (float, float)
        Reference pixel.  If `None` set to the array center.

    verbose : bool
        Messaging to terminal.

    initial_header : None or `~astropy.io.fits.Header`
        Initial header to use as a guess for the SIP fit.

    max_rms : float
        Maximum RMS for the SIP fit.

    set_diff_step : bool
        Set `lsq_args['diff_step']` to the pixel scale.

    get_guess : bool
        Get initial guess for SIP coefficients from the `datamodel.meta.wcs`.

    fast_coeffs : bool
        Use a fast method to compute the SIP coefficients.

    uvxy : None or (array, array, array, array)
        Manually specify detector and target coordinates for positions to use
        for the SIP fit, where ``uvxy = (x, y, ra, dec)`` and the detector
        coordinates are zero-index.  If not specified, make a grid with ``step``

    Returns
    -------
    header : '~astropy.io.fits.Header`
        Header with SIP WCS

    """

    frame = inspect.currentframe()
    utils.log_function_arguments(
        utils.LOGFILE, frame, "jwst_utils.pipeline_model_wcs_header"
    )

    filter_offset = None

    if datamodel.meta.instrument.name in ["MIRI"]:
        crpix = [516.5, 512.5]
        get_guess = True
        degrees = [4]

    if crpix is None:
        meta = datamodel.meta.wcsinfo
        if meta.siaf_xref_sci is None:
            crpix = [meta.crpix1 * 1, meta.crpix2 * 1]
        else:
            crpix = [meta.siaf_xref_sci * 1, meta.siaf_yref_sci * 1]

        # Get filter offset
        tr = datamodel.meta.wcs.pipeline[0].transform
        if hasattr(tr, "offset_0"):
            # not crpix itself
            if np.abs(tr.offset_0) < 100:
                filter_offset = [tr.offset_0.value, tr.offset_1.value]

                crpix[0] -= filter_offset[0]
                crpix[1] -= filter_offset[1]

    ndeg = len(degrees)

    for i, deg in enumerate(degrees):
        if i == 0:
            h = initial_header

        h = model_wcs_header(
            datamodel,
            step=step,
            lsq_args=lsq_args,
            initial_header=h,
            degree=deg,
            get_sip=True,
            get_guess=get_guess,
            crpix=crpix,
            set_diff_step=set_diff_step,
            fast_coeffs=fast_coeffs,
            uvxy=uvxy,
        )

        xrms = h["SIPRAMAD"]
        yrms = h["SIPDEMAD"]
        msg = f"Fit SIP degree={deg} rms= {xrms:.2e}, {yrms:.2e} pix"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        if xrms < max_rms:
            break

        if (i + 1 < ndeg) & (deg == 5):
            ## add a delta so that fits are updated
            if h["A_5_0"] == 0:
                h["A_5_0"] = 1.0e-16

            if h["B_0_5"] == 0:
                h["B_0_5"] = 1.0e-16

    if filter_offset is not None:
        h["SIPFXOFF"] = filter_offset[0], "Filter offset, x pix"
        h["SIPFYOFF"] = filter_offset[1], "Filter offset, y pix"

    return h


def _objective_sip(
    params, u, v, ra, dec, crval, crpix, a_names, b_names, cd, fit_type, ret
):
    """
    Objective function for fitting SIP coefficients

    Parameters
    ----------
    params : list
        List of SIP coefficients.

    u, v, ra, dec : array
        Grid of detector and sky coordinates.

    crval, crpix : array
        Reference pixel

    a_names, b_names : list
        List of SIP coefficient names.

    cd : array
        CD matrix.

    fit_type : int
        Bitmask for fitting rotation, scale, and CRVAL offsets.

    ret : int
        Return behavior

    Returns
    -------
    if ret == 1:
        pp, cd_i, crval_i, a_coeff, b_coeff, ra_nmad, dec_nmad : values derived from the input ``params``
    else:
        dr : array-like
            Residuals for fit optimization

    """
    from astropy.modeling import models, fitting

    # u, v, x, y, crpix, a_names, b_names, cd = data

    # cdx = params[0:4].reshape((2, 2))
    # fit_type = fit_rot*1 + fit_scale*2 + fit_crval*4
    i0 = 0
    if (fit_type & 1) > 0:
        rotation = params[0]
        i0 += 1
    else:
        rotation = 0

    if (fit_type & 2) > 0:
        scale = 10 ** params[i0]
        i0 += 1
    else:
        scale = 1.0

    theta = -rotation
    _mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    cd_i = np.dot(cd, _mat) / scale

    if (fit_type & 4) > 0:
        crval_offset = params[i0 : i0 + 2]
        i0 += 2
    else:
        crval_offset = np.zeros(2)

    crval_i = crval + crval_offset

    a_params = params[i0 : i0 + len(a_names)]
    b_params = params[i0 + len(a_names) :]

    a_coeff = {}
    for i in range(len(a_names)):
        a_coeff[a_names[i]] = a_params[i]

    b_coeff = {}
    for i in range(len(b_names)):
        b_coeff[b_names[i]] = b_params[i]

    # Build header
    _h = pyfits.Header()
    for i in [0, 1]:
        for j in [0, 1]:
            _h[f"CD{i+1}_{j+1}"] = cd_i[i, j]

    _h["CRPIX1"] = crpix[0]
    _h["CRPIX2"] = crpix[1]
    _h["CRVAL1"] = crval_i[0]
    _h["CRVAL2"] = crval_i[1]

    _h["A_ORDER"] = 5
    for k in a_coeff:
        _h[k] = a_coeff[k]

    _h["B_ORDER"] = 5
    for k in b_coeff:
        _h[k] = b_coeff[k]

    _h["RADESYS"] = "ICRS    "
    _h["CTYPE1"] = "RA---TAN-SIP"
    _h["CTYPE2"] = "DEC--TAN-SIP"
    _h["CUNIT1"] = "deg     "
    _h["CUNIT2"] = "deg     "

    # _w = pywcs.WCS(_h)
    _w = utils.wcs_from_header(_h, relax=True)

    ro, do = _w.all_pix2world(u, v, 0)

    cosd = np.cos(ro / 180 * np.pi)
    if ret == 1:
        ra_nmad = utils.nmad((ra - ro) * cosd * 3600)
        dec_nmad = utils.nmad((dec - do) * 3600)

        pp = (rotation / np.pi * 180, scale, crval_offset * 3600)
        print("xxx", params, ra_nmad, dec_nmad, pp)

        return pp, cd_i, crval_i, a_coeff, b_coeff, ra_nmad, dec_nmad

    # print(params, np.abs(dr).max())
    dr = np.append((ra - ro) * cosd, dec - do) * 3600.0

    return dr


def _objective_lstsq_sip(
    params, u, v, ra, dec, crval, crpix, a_names, b_names, cd, fit_type, ret
):
    """
    Objective function for fitting SIP header

    Parameters
    ----------
    params : list
        List of SIP coefficients.

    u, v, ra, dec : array
        Detector and sky coordinates.

    crval, crpix : array
        Reference pixel and sky coordinates.

    a_names, b_names : list
        List of SIP coefficient names.

    cd : array
        CD matrix.

    fit_type : int
        Bitmask for fitting rotation, scale, and CRVAL offsets.

    ret : int
        Return status.

    Returns
    -------
    if ret == 1:
        pp, cd_i, crval_i, a_coeff, b_coeff, ra_nmad, dec_nmad : values derived from input ``params``
    else:
        dr : array-like
            Residuals for fit optimization

    """
    from astropy.modeling.fitting import LinearLSQFitter
    from astropy.modeling.polynomial import Polynomial2D

    # u, v, x, y, crpix, a_names, b_names, cd = data

    # cdx = params[0:4].reshape((2, 2))
    # fit_type = fit_rot*1 + fit_scale*2 + fit_crval*4
    i0 = 0
    if (fit_type & 1) > 0:
        rotation = params[0]
        i0 += 1
    else:
        rotation = 0

    if (fit_type & 2) > 0:
        scale = 10 ** params[i0]
        i0 += 1
    else:
        scale = 1.0

    theta = -rotation
    _mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    cd_i = np.dot(cd, _mat) / scale

    if (fit_type & 4) > 0:
        crval_offset = params[i0 : i0 + 2]
        i0 += 2
    else:
        crval_offset = np.zeros(2)

    crval_i = crval + crval_offset

    # Build header
    _h = pyfits.Header()
    for i in [0, 1]:
        for j in [0, 1]:
            _h[f"CD{i+1}_{j+1}"] = cd_i[i, j]

    _h["CRPIX1"] = crpix[0]
    _h["CRPIX2"] = crpix[1]
    _h["CRVAL1"] = crval_i[0]
    _h["CRVAL2"] = crval_i[1]

    _h["RADESYS"] = "ICRS    "
    _h["CTYPE1"] = "RA---TAN-SIP"
    _h["CTYPE2"] = "DEC--TAN-SIP"
    _h["CUNIT1"] = "deg     "
    _h["CUNIT2"] = "deg     "

    a_params = params[i0 : i0 + len(a_names)]
    b_params = params[i0 + len(a_names) :]

    a_coeff = {}
    for i in range(len(a_names)):
        a_coeff[a_names[i]] = 0.0  # a_params[i]

    b_coeff = {}
    for i in range(len(b_names)):
        b_coeff[b_names[i]] = 0.0  # b_params[i]

    _h["A_ORDER"] = 5
    _h["B_ORDER"] = 5

    # Zero SIP coeffs
    for k in a_coeff:
        _h[k] = 0.0

    for k in b_coeff:
        _h[k] = 0.0

    # _w = pywcs.WCS(_h)
    _w = utils.wcs_from_header(_h, relax=True)

    # Calculate pixel offsets in empty SIP WCS
    up, vp = _w.all_world2pix(ra, dec, 0)
    uv = np.array([u.flatten(), v.flatten()]).T
    uvi = uv - (crpix - 1)

    uvp = np.array([up.flatten(), vp.flatten()]).T
    fg = uvp - uv

    poly = Polynomial2D(degree=5)
    for p in poly.fixed:
        key = "A_" + p[1:]
        if (key not in a_names) | (p in ["c0_0", "c0_1", "c1_0"]):
            poly.fixed[p] = True

    fitter = LinearLSQFitter()
    afit = fitter(poly, *uvi.T, fg[:, 0])
    bfit = fitter(poly, *uvi.T, fg[:, 1])
    fgm = np.array([afit(*uvi.T), bfit(*uvi.T)]).T

    a_coeff = {}
    b_coeff = {}

    for p, _a, _b in zip(afit.param_names, afit.parameters, bfit.parameters):
        key = "A_" + p[1:]
        if key in a_names:
            a_coeff[key] = _a
            b_coeff[key.replace("A_", "B_")] = _b

    if ret == 1:
        dx = fg - fgm
        x_nmad = utils.nmad(dx[:, 0])
        y_nmad = utils.nmad(dx[:, 1])

        pp = (rotation / np.pi * 180, scale, crval_offset * 3600)

        return pp, cd_i, crval_i, a_coeff, b_coeff, x_nmad, y_nmad

    # Residual is in x,y pixels
    dr = (fgm - fg).flatten()

    # print(params, (dr**2).sum())

    return dr


def _xobjective_sip(params, u, v, x, y, crpix, a_names, b_names, ret):
    """
    Objective function for fitting SIP coefficients

    Parameters
    ----------
    params : list
        List of SIP coefficients.

    u, v, x, y : array-like
        Detector and sky coordinates.

    crpix : array-like
        Reference pixel

    a_names, b_names : list
        List of SIP coefficient names.

    ret : int
        Return status.

    Returns
    -------
    if ret == 1:
        cdx : (2,2) array
            CD matrix

        a_coeff : array-like
            SIP "A" coefficients

        b_coeff : array-like
            SIP "B" coefficients
    else:
        dr : array-like
            Residuals for fit optimization

    """
    from astropy.modeling import models, fitting

    # u, v, x, y, crpix, a_names, b_names, cd = data

    cdx = params[0:4].reshape((2, 2))
    a_params = params[4 : 4 + len(a_names)]
    b_params = params[4 + len(a_names) :]

    a_coeff = {}
    for i in range(len(a_names)):
        a_coeff[a_names[i]] = a_params[i]

    b_coeff = {}
    for i in range(len(b_names)):
        b_coeff[b_names[i]] = b_params[i]

    if ret == 1:
        return cdx, a_coeff, b_coeff

    off = 1

    sip = models.SIP(
        crpix=crpix - off, a_order=4, b_order=4, a_coeff=a_coeff, b_coeff=b_coeff
    )

    fuv, guv = sip(u, v)
    xo, yo = np.dot(cdx, np.array([u + fuv - crpix[0], v + guv - crpix[1]]))
    dr = np.append(x - xo, y - yo) * 3600.0 / 0.065

    return dr


def compare_gwcs_sip(
    file,
    save=False,
    step=32,
    use_gwcs_func=False,
    func_kwargs={"degree": 5, "crpix": None, "max_pix_error": 0.01},
):
    """
    Make a figure comparing the `gwcs` and SIP WCS of a JWST exposure with
    the round trip transformation ``pixel -> gwcs_RaDec -> sip_pixel``

    Parameters
    ----------
    file : str
        Filename, e.g., ``jw...._cal.fits``

    save : bool
        Save the figure to ``file.replace('.fits', '.sip.png')``

    step : int
        Step size for the test pixel grid

    use_gwcs_func : bool
        Use the `gwcs` forward transform to generate the SIP header rather
        than the `astropy.wcs` SIP header.

    func_kwargs : dict
        Keyword arguments for the `gwcs` SIP header generation.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure object

    """
    import matplotlib.pyplot as plt

    im = pyfits.open(file)

    obj = img_with_wcs(im)

    if use_gwcs_func:
        if "npoints" not in func_kwargs:
            func_kwargs["npoints"] = step

        h = obj.meta.wcs.to_fits_sip(**func_kwargs)
        wcs = pywcs.WCS(h, relax=True)
    else:
        wcs = pywcs.WCS(im["SCI"].header, relax=True)

    sh = im["SCI"].data.shape

    xarr = np.arange(0, sh[0], step)

    # Round-trip of pixel > gwcs_RaDec > sip_pixel
    u, v = np.meshgrid(xarr, xarr)
    rd = obj.meta.wcs.forward_transform(u, v)
    up, vp = wcs.all_world2pix(*rd, 0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].scatter(
        u, up - u, alpha=0.5, label=r"$\Delta x$" + f" rms={utils.nmad(up-u):.1e}"
    )
    axes[0].scatter(
        v, vp - v, alpha=0.5, label=r"$\Delta y$" + f" rms={utils.nmad(vp-v):.1e}"
    )

    axes[0].legend(loc="lower center")
    axes[0].grid()
    axes[0].set_xlabel("pixel")
    axes[0].set_ylabel(r"$\Delta$ pixel")

    axes[0].text(
        0.5, 0.98, file, ha="center", va="top", transform=axes[0].transAxes, fontsize=6
    )

    label = f"{im[0].header['INSTRUME']} {utils.parse_filter_from_header(im[0].header)}"
    axes[0].text(
        0.5, 0.94, label, ha="center", va="top", transform=axes[0].transAxes, fontsize=8
    )

    scl = sh[1] / axes[0].get_ylim()[1] * 4
    axes[0].set_xticks(np.arange(0, sh[1] + 1, 512))

    axes[1].quiver(u, v, up - u, vp - v, alpha=0.4, units="x", scale=step / scl)

    axes[1].set_xticks(np.arange(0, sh[1] + 1, 512))
    axes[1].set_yticks(np.arange(0, sh[0] + 1, 512))
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].grid()

    fig.tight_layout(pad=0.5)

    if save:
        figfile = file.replace(".fits", ".sip.png").split(".gz")[0]
        print(figfile)

        fig.savefig(figfile)

    return fig


def load_jwst_filter_info():
    """
    Load the filter info in `grizli/data/jwst_bp_info.yml`

    Returns
    -------
    bp : dict
        Full filter information dictionary for JWST instruments
    """
    import yaml

    path = os.path.join(os.path.dirname(__file__), "data", "jwst_bp_info.yml")

    with open(path) as fp:
        bp = yaml.load(fp, yaml.SafeLoader)

    return bp


def get_jwst_filter_info(header):
    """
    Retrieve filter info from tabulated file for INSTRUME/FILTER/PUPIL
    combination in a primary FITS header

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Primary header with INSTRUME, FILTER, [PUPIL] keywords

    Returns
    -------
    info : dict
        Filter information

    """
    # from grizli.jwst_utils import __file__
    import yaml

    if "INSTRUME" not in header:
        print(f"Keyword INSTRUME not in header")
        return None

    bp = load_jwst_filter_info()

    inst = header["INSTRUME"]
    if inst not in bp:
        print(f"INSTRUME={inst} not in jwst_bp_info.yml table")
        return None

    info = None

    for k in ["PUPIL", "FILTER"]:
        if k in header:
            if header[k] in bp[inst]:
                info = bp[inst][header[k]]
                info["name"] = header[k]
                info["keyword"] = k
                info["meta"] = bp["meta"]
                break

    return info


def calc_jwst_filter_info(context="jwst_1130.pmap"):
    """
    Calculate JWST filter properties from tabulated `eazy` filter file and
    photom reference files

    Calculated attributes:

        - ``pivot`` = Filter pivot wavelength, microns
        - ``rectwidth`` = Filter rectangular width, microns
        - ``ebv0.1`` = Milky way extinction, mag for E(B-V) = 0.1, Rv=3.1
        - ``ab_vega`` = AB - Vega mag conversion
        - ``eazy_fnumber`` = Filter number in the `eazy` filter file
        - ``photmjsr`` = Photometric conversion if ref files available

    Parameters
    ----------
    context : str
        CRDS context file to use for reference files. Default is `jwst_1130.pmap`

    Returns
    -------
    bp : dict
        Filter information dictionary for JWST instruments with the above
        attributes for each filter in each instrument configuration.

    """
    import yaml
    import glob
    import eazy.filters
    import astropy.time

    from . import grismconf

    res = eazy.filters.FilterFile(path=None)

    bp = {
        "meta": {
            "created": astropy.time.Time.now().iso.split()[0],
            "crds_context": context,
            "wave_unit": "micron",
            "description": {
                "pivot": "Filter pivot wavelength",
                "rectwith": "Filter rectangular width",
                "ebv0.1": "Milky Way extinction, mag for E(B-V)=0.1, Rv=3.1",
                "ab_vega": "AB - Vega mag conversion",
                "eazy_fnumber": "Filter number in the eazy filter file",
                "photmjsr": "Photometric conversion if ref files available",
                "photfile": "Photom reference file",
            },
        }
    }

    detectors = {
        "NIRCAM": {
            ("F200W", "CLEAR"): [
                "NRCA1",
                "NRCA2",
                "NRCA3",
                "NRCA4",
                "NRCA1",
                "NRCA2",
                "NRCA3",
                "NRCA4",
            ],
            ("F444W", "CLEAR"): ["NRCALONG", "NRCBLONG"],
        },
        "NIRISS": {("CLEAR", "F200W"): ["NIS"]},
        "MIRI": {("F770W", None): ["MIRIMAGE"]},
    }

    exp_type = {
        "NIRCAM": "NRC_IMAGE",
        "NIRISS": "NIS_IMAGE",
        "MIRI": "MIRI_IMAGE",
    }

    for inst in ["NIRCAM", "NIRISS", "MIRI"]:
        bp[inst] = {}
        fn = res.search(f"jwst_{inst}", verbose=False)
        print(f"\n{inst}\n=====")

        if context is not None:
            phot_files = []
            for key in detectors[inst]:
                for d in detectors[inst][key]:
                    kws = dict(
                        instrument=inst,
                        filter=key[0],
                        pupil=key[1],
                        detector=d,
                        reftypes=("photom",),
                        exp_type=exp_type[inst],
                    )
                    refs = crds_reffiles(**kws)
                    phot_files.append(refs["photom"])
        else:
            # photometry calib
            phot_files = glob.glob(
                f"{os.getenv('CRDS_PATH')}/references/"
                + f"jwst/{inst.lower()}/*photom*fits"
            )

        if len(phot_files) > 0:
            phot_files.sort()
            phot_files = phot_files[::-1]
            phots = [utils.read_catalog(file) for file in phot_files]
        else:
            phots = None

        for j in fn:
            fi = res[j + 1]
            key = fi.name.split()[0].split("_")[-1].upper()
            if inst == "MIRI":
                key = key.replace("F0", "F")
                if key.endswith("C"):
                    continue

            bp[inst][key] = {
                "pivot": float(fi.pivot / 1.0e4),
                "ab_vega": float(fi.ABVega),
                "ebv0.1": float(fi.extinction_correction(EBV=0.1)),
                "rectwidth": float(fi.rectwidth / 1.0e4),
                "eazy_fnumber": int(j + 1),
            }

            if phots is not None:
                ix = None
                if inst == "MIRI":
                    for k, phot in enumerate(phots):
                        ix = phot["filter"] == key
                        if ix.sum() > 0:
                            break

                elif inst == "NIRISS":
                    for k, phot in enumerate(phots):
                        try:
                            filt = [f.strip() for f in phot["filter"]]
                            pupil = [f.strip() for f in phot["pupil"]]
                            phot["filter"] = filt
                            phot["pupil"] = pupil
                        except:
                            continue

                        ix = phot["filter"] == key
                        ix |= phot["pupil"] == key
                        if ix.sum() > 0:
                            break

                elif inst == "NIRCAM":
                    for k, phot in enumerate(phots):
                        if key in phot["pupil"]:
                            ix = phot["pupil"] == key
                        else:
                            ix = phot["filter"] == key
                            ix &= phot["pupil"] == "CLEAR"

                        if ix.sum() > 0:
                            break

                if ix is not None:
                    _d = bp[inst][key]
                    if ix.sum() > 0:
                        _d["photmjsr"] = float(phot["photmjsr"][ix][0])
                        _d["photfile"] = os.path.basename(phot_files[k])
                    else:
                        _d["photmjsr"] = None
                        _d["photfile"] = None

                        print(f"{inst} {key} not found in {phot_files}")
            else:
                _d = bp[inst][key]
                _d["photmjsr"] = None
                _d["photfile"] = None

            print(f"{key} {_d['pivot']:.3f} {_d['photmjsr']}")

    with open("jwst_bp_info.yml", "w") as fp:
        yaml.dump(bp, fp)

    return bp


def get_crds_zeropoint(
    instrument="NIRCAM",
    detector="NRCALONG",
    filter="F444W",
    pupil="CLEAR",
    date=None,
    context="jwst_0989.pmap",
    verbose=False,
    **kwargs,
):
    """
    Get ``photmjsr`` photometric zeropoint for a partiular JWST instrument imaging
    mode

    To-do: add MIRI time-dependence

    Parameters
    ----------
    instrument : str
        ``NIRCAM, NIRISS, MIRI``

    detector : str
        Different detectors for NIRCAM.  Set to ``NIS`` and ``MIRIMAGE`` for
        NIRISS and MIRI, respectively.

    filter, pupil : str
        Bandpass filters.  ``pupil=None`` for MIRI.

    date : `astropy.time.Time`
        Optional observation date

    context : str
        CRDS_CTX context to use

    verbose : bool
        Messaging

    Returns
    -------
    context : str
        Copy of the input ``context``

    ref_file : str
        Path to the ``photom`` reference file

    mjsr : float
        Photometric zeropoint ``mjsr`` in units of MJy / Sr

    pixar_sr : float
        Pixel area in Sr from primary header of ``photom`` file

    """
    from crds import CrdsLookupError

    from . import grismconf

    if instrument == "NIRCAM":
        module = detector[3]
        exp_type = "NRC_IMAGE"
    elif instrument == "NIRISS":
        exp_type = "NIS_IMAGE"
        detector = "NIS"
        module = None
    elif instrument == "MIRI":
        exp_type = "MIR_IMAGE"
        detector = "MIRIMAGE"
        pupil = None
        module = None
    else:
        return context, None, None

    mode = dict(
        instrument=instrument,
        filter=filter,
        pupil=pupil,
        module=module,
        date=date,
        exp_type=exp_type,
        detector=detector,
        context=context,
    )

    try:
        refs = grismconf.crds_reffiles(
            reftypes=("photom",),
            header=None,
            verbose=verbose,
            **mode,
        )
    except CrdsLookupError:
        return None, None

    # Pixel area
    pixar_sr = None

    with pyfits.open(refs["photom"]) as im:
        if "PIXAR_SR" in im[0].header:
            pixar_sr = im[0].header["PIXAR_SR"]

    ph = utils.read_catalog(refs["photom"])
    ph["fstr"] = [f.strip() for f in ph["filter"]]

    if instrument in ["NIRISS", "NIRCAM"]:
        ph["pstr"] = [p.strip() for p in ph["pupil"]]
        row = (ph["fstr"] == filter) & (ph["pstr"] == pupil)
        if row.sum() == 0:
            print(f"get_crds_zeropoint: {mode} not found in {refs['photom']}")
            mjsr = None
        else:
            mjsr = ph["photmjsr"][row][0]
    else:
        row = ph["fstr"] == filter
        if row.sum() == 0:
            print(f"get_crds_zeropoint: {mode} not found in {refs['photom']}")
            mjsr = None
        else:
            mjsr = ph["photmjsr"][row][0]

    if verbose:
        if mjsr is not None:
            print(f"crds_reffiles: photmjsr = {mjsr:.4f}  pixar_sr = {pixar_sr:.3e}")
        else:
            print(f"crds_reffiles: photmjsr = {mjsr}  pixar_sr = {pixar_sr:.3e}")

    return context, refs["photom"], mjsr, pixar_sr


def get_nircam_zeropoint_update(
    detector="NRCALONG",
    filter="F444W",
    pupil="CLEAR",
    header=None,
    verbose=False,
    **kwargs,
):
    """
    Get latest correction factors for NIRCAM zeropoints

    Parameters
    ----------
    detector : str
        Detector name

    filter, pupil : str
        Element name in filter and pupil wheels

    verbose : bool
        Messaging

    Returns
    -------
    key : str
        String key for the detector + filter combination

    mjysr : float, None
        Zerpoint from CRDS, None if ``key`` not found

    scale : float, None
        Scale factor to multiply to ``mjysr``

    pixar_sr : float
        Pixel area in steradians

    """
    import yaml

    zeropoint_file = os.path.join(
        os.path.dirname(__file__), "data", "jwst_zeropoints.yml"
    )

    if header is not None:
        if "DETECTOR" in header:
            detector = header["DETECTOR"]

        if "OFILTER" in header:
            filter = header["OFILTER"]
        elif "FILTER" in header:
            filter = header["FILTER"]

        if "OPUPIL" in header:
            pupil = header["OPUPIL"]
        elif "PUPIL" in header:
            pupil = header["PUPIL"]
        else:
            pupil = None

    if pupil is not None:
        key = f"{detector}-{filter}-{pupil}".upper()
    else:
        key = f"{detector}-{filter}".upper()

    if not os.path.exists(zeropoint_file):
        msg = "get_nircam_zeropoint_update: "
        msg += f"{zeropoint_file} not found"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return key, None, None, None

    with open(zeropoint_file) as _fp:
        zp_data = yaml.load(_fp, Loader=yaml.Loader)

    if key not in zp_data:
        msg = "get_nircam_zeropoint_update: "
        msg += f"{key} not found in {zeropoint_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return key, None, None, None

    _mjsr, _scale, _pixar_sr = zp_data[key]

    msg = "get_nircam_zeropoint_update: "
    msg += f"{key} mjsr={_mjsr:.4f} scale={_scale:.4f} pixar_sr={_pixar_sr:.2e}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    return (key, _mjsr, _scale, _pixar_sr)


def query_pure_parallel_wcs(
    assoc, pad_hours=1.0, verbose=True, products=["1b", "2", "2a", "2b"]
):
    """
    Query the archive for the *prime* exposures associated with pure-parallel
    observations, since the header WCS for the latter aren't correct

    ToDo: NIRSpec MSA cal files are for extractions

    Parameters
    ----------
    assoc : str
        Association name in the grizli database `assoc_table` table

    pad_hours : float
        Time padding for the MAST query, hours

    verbose : bool
        Messaging

    products : list, None
        List of archive ``productLevel`` values to include in the MAST query

    Returns
    -------
    prime : `~astropy.table.Table`
        Matched table of computed Prime exposures

    res : `~astropy.table.Table`
        Full MAST query table

    times : `~astropy.table.Table`
        Exposure query from the grizli database

    """
    import astropy.units as u
    from mastquery import jwst

    from .aws import db

    times = db.SQL(
        f"""select "dataURL", t_min, t_max, instrument_name,
    proposal_id, filter
    from assoc_table where assoc_name = '{assoc}'
    order by t_min
    """
    )

    trange = [
        times["t_min"].min() - pad_hours / 24.0,
        times["t_min"].max() + pad_hours / 24.0,
    ]

    msg = "jwst_utils.get_pure_parallel_wcs: "
    msg += f"Found {len(times)} exposures for {assoc}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    msg = "jwst_utils.get_pure_parallel_wcs: "
    msg += f"expstart = [{trange[0]:.3f}, {trange[1]:.3f}]"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    filters = []
    filters += jwst.make_query_filter("expstart", range=trange)

    if products is not None:
        filters += jwst.make_query_filter("productLevel", values=products)

    inst = times["instrument_name"][0]
    inst_keys = {"NIRISS": "NIS", "NIRCAM": "NRC", "MIRI": "MIR", "NIRSPEC": "NRS"}
    key = inst_keys[inst]
    instruments = []
    for k in ["NRC", "NIS", "NRS", "MIR"]:
        if k != key:
            instruments.append(k)

    # try:
    res = jwst.query_all_jwst(
        recent_days=None,
        filters=filters,
        columns="*",
        instruments=instruments,
        fix=False,
    )
    # except KeyError:
    #     # Some missing wcs
    #     filters += jwst.make_query_filter('cd1_1',
    #                     range=[-1,1])
    #
    #     res = jwst.query_all_jwst(recent_days=None, filters=filters, columns='*',
    #                           instruments=instruments,
    #                           fix=True)

    if "t_min" not in res.colnames:
        res["t_min"] = res["expstart"]
        res["t_max"] = res["expend"]
        res["targname"] = res["targprop"]
        res["proposal_id"] = res["program"]
        res["instrument_name"] = res["instrume"]
        res["dataURL"] = res["dataURI"]
        # ok = np.array([len(s) > 5 for s in res['s_region']])
        if hasattr(res["s_region"], "mask"):
            res = res[~res["s_region"].mask]

        jwst.set_footprint_centroids(res)

    so = np.argsort(res["expstart"])
    res = res[so]

    msg = "jwst_utils.get_pure_parallel_wcs: "
    msg += f"Found {len(res)} MAST entries for "
    msg += f"expstart = [{trange[0]:.3f}, {trange[1]:.3f}]"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    if len(res) <= len(times):
        msg = "jwst_utils.get_pure_parallel_wcs: "
        msg += f"Didn't find prime exposures for {assoc}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        return None, None, None

    res["par_dt"] = 0.0
    res["par_dt"].description = "Time offset to parallel exposure"
    res["par_dt"].unit = u.second
    res["par_dt"].format = ".1f"
    res["t_min"].format = ".3f"

    res["par_file"] = res["dataURL"]

    res["all_dt"] = 1e4

    for j, t in enumerate(times):

        test = res["instrument_name"] != t["instrument_name"]

        delta_time = t["t_min"] - res["expstart"][test]

        res["all_dt"][test] = np.minimum(
            res["all_dt"][test], np.abs(delta_time) * 86400
        )

    # Group by detector
    mat = res["all_dt"] < res["effexptm"]
    und = utils.Unique(res[mat]["detector"], verbose=False)

    ind = und.counts == len(times)
    if ind.sum() > 0:
        det = und.values[np.where(ind)[0][0]]
        print(f"Use detector {det}")
        res = res[(res["detector"] == det) & mat]

    rowix = []
    for j, t in enumerate(times):

        test = res["instrument_name"] != t["instrument_name"]

        delta_time = t["t_min"] - res["expstart"][test]

        res["all_dt"][test] = np.minimum(
            res["all_dt"][test], np.abs(delta_time) * 86400
        )

        ix = np.argmin(np.abs(delta_time))

        rowix.append(np.where(test)[0][ix])

        res["par_dt"][rowix[-1]] = delta_time[ix] * 86400
        res["par_file"][rowix[-1]] = os.path.basename(t["dataURL"])

    prime = res[rowix]
    prime["assoc_name"] = assoc

    # polygon strings
    if "s_region" in prime.colnames:
        poly = []
        for s in prime["s_region"]:
            sr = utils.SRegion(s)
            poly += sr.polystr()

        prime["footprint"] = poly

    return prime, res, times


def query_pure_parallel_wcs_to_database(
    programs=[1571, 3383, 4681, 2514, 3990], output="/tmp/jwst_pure_parallels.html"
):
    """
    Run pure parallel wcs query for all associations from PP proposals

    Parameters
    ----------
    programs : list
        List of program IDs to query.
        - 1571: PASSAGE
        - 2514: PANORAMIC
        - 3383 & 4681: OutThere
        - 3990: Morishita+Mason

    output : str
        Output HTML table (default: /tmp/jwst_pure_parallels.html)

    """
    from .aws import db

    pstr = ",".join([f"'{p}'" for p in programs])
    pp = db.SQL(
        f"""select assoc_name, max(proposal_id) as proposal_id, 
count(assoc_name), max(filter) as filter
from assoc_table where proposal_id in ({pstr})
group by assoc_name order by max(t_min)
"""
    )

    columns = [
        "assoc_name",
        "filename",
        "apername",
        "ra",
        "dec",
        "gs_v3_pa",
        "par_dt",
        "par_file",
        "t_min",
        "t_max",
        "proposal_id",
        "instrument_name",
        "targname",
        "footprint",
    ]

    if 0:
        # Initialize table
        prime, res, times = query_pure_parallel_wcs(pp["assoc_name"][0])
        db.send_to_database(
            "pure_parallel_exposures", prime[columns], index=False, if_exists="append"
        )

        db.execute(f"delete from pure_parallel_exposures where True")

        db.execute("CREATE INDEX on pure_parallel_exposures (assoc_name, par_file)")
        db.execute("CREATE INDEX on pure_parallel_exposures (assoc_name)")

    # Add all
    exist = db.SQL(
        """select assoc_name, count(assoc_name),
    min(proposal_id) as primary_program, min(filename) as primary_file,
    max(targname) as targname,
    min(apername) as apername,
    max(apername) as max_apername,
    max(substr(par_file, 4,4)) as purepar_program,
    min(par_file) as purepar_file,
    min(t_min) as t_min, max(t_max) as t_max, 
    min(par_dt) as min_dt, max(par_dt) as max_dt
    from pure_parallel_exposures group by assoc_name order by min(t_min)
    """
    )

    for i, assoc in enumerate(pp["assoc_name"]):
        if assoc in exist["assoc_name"]:
            print(f"Skip: {assoc}")
            continue

        try:
            prime, res, times = query_pure_parallel_wcs(assoc)
        except ValueError:
            print(f"Failed: {assoc}")
            continue

        if 0:
            db.execute(
                f"delete from pure_parallel_exposures where assoc_name = '{assoc}'"
            )

        db.send_to_database(
            "pure_parallel_exposures", prime[columns], index=False, if_exists="append"
        )

    # Redo query
    exist = db.SQL(
        """select assoc_name, count(assoc_name),
    min(proposal_id) as primary_program, min(filename) as primary_file,
    max(targname) as targname,
    min(apername) as apername,
    max(apername) as max_apername,
    max(substr(par_file, 4,4)) as purepar_program,
    min(par_file) as purepar_file,
    min(t_min) as t_min, max(t_max) as t_max,
    min(par_dt) as min_dt, max(par_dt) as max_dt
    from pure_parallel_exposures group by assoc_name order by min(t_min)
    """
    )

    desc = {
        "t_min": "Visit start, mjd",
        "t_max": "Visit end, mjd",
        "count": "Exposure count",
        "primary_program": "Primary program ID",
        "purepar_program": "Parallel program ID",
        "min_dt": "Minimum dt between primary and par exposures, sec",
        "max_dt": "Maximum dt between primary and par exposures, sec",
    }

    for k in desc:
        exist[k].description = desc[k]

    exist["t_min"].format = ".3f"
    exist["t_max"].format = ".3f"
    exist["min_dt"].format = ".1f"
    exist["max_dt"].format = ".1f"

    exist.write_sortable_html(
        output,
        use_json=False,
        localhost=False,
        max_lines=100000,
        filter_columns=list(desc.keys()),
    )


def update_pure_parallel_wcs(file, fix_vtype="PARALLEL_PURE", verbose=True):
    """
    Update pointing information of pure parallel exposures using the pointing
    information of the prime exposures from the MAST database and `pysiaf`

    1. Find the FGS log from a MAST query that is closest in ``EXPSTART`` to ``file``
    2. Use the ``ra_v1, dec_v1, pa_v3`` values of the FGS log to set the pointing
       attitude with `pysiaf`
    3. Compute the sky position of the ``CRPIX`` reference pixel of ``file`` with
       `pysiaf` and put that position in the ``CRVAL`` keywords

    Parameters
    ----------
    file : str
        Filename of a pure-parallel exposure (rate.fits)

    fix_vtype : str
        Run if ``file[0].header['VISITYPE'] == fix_vtype``

    verbose : bool
        Status messaging

    Returns
    -------
    status : None, True
        Returns None if some problem is found

    """
    from scipy.optimize import minimize
    import pysiaf
    from pysiaf.utils import rotations

    import mastquery.jwst
    from .aws import db

    if not os.path.exists(file):
        msg = "jwst_utils.update_pure_parallel_wcs: "
        msg += f" {file} not found"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None

    with pyfits.open(file) as im:
        h0 = im[0].header.copy()
        h1 = im[1].header.copy()
        if "VISITYPE" not in im[0].header:
            msg = "jwst_utils.update_pure_parallel_wcs: "
            msg += f" VISITYPE not found in header {file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            return None

    # Is this a PARALLEL_PURE exposure?
    vtype = h0["VISITYPE"]

    if vtype != fix_vtype:
        msg = "jwst_utils.update_pure_parallel_wcs: "
        msg += f" VISITYPE ({vtype}) != {fix_vtype}, skip"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None

    crval_init = h1["CRVAL1"], h1["CRVAL2"]

    # Get correct pointing from FGS logs
    dt = 0.01
    gs = mastquery.jwst.query_guidestar_log(
        mjd=(h0["EXPSTART"] - dt, h0["EXPEND"] + dt),
        program=None,
        exp_type=["FGS_FINEGUIDE"],
    )

    keep = gs["expstart"] < h0["EXPSTART"]
    keep &= gs["expend"] > h0["EXPEND"]

    if keep.sum() == 0:
        msg = f"jwst_utils.update_pure_parallel_wcs: par_file='{file}'"
        msg += " couldn't find corresponding exposure in FGS logs"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None

    gs = gs[keep][0]
    pos = (gs["ra_v1"], gs["dec_v1"], gs["pa_v3"])
    att = rotations.attitude(0.0, 0.0, *pos)

    # And apply the pointing to the parallel aperture and reference pixel
    par_aper = pysiaf.Siaf(h0["INSTRUME"])[h0["APERNAME"]]
    par_aper.set_attitude_matrix(att)

    crpix = h1["CRPIX1"], h1["CRPIX2"]
    crpix_init = par_aper.sky_to_sci(*crval_init)

    crval_fix = par_aper.sci_to_sky(*crpix)

    msg = f"jwst_utils.update_pure_parallel_wcs: {file}"
    msg += "\n" + f"jwst_utils.update_pure_parallel_wcs: FGS {gs['fileName']} "
    msg += "\n" + f"jwst_utils.update_pure_parallel_wcs: original crval "
    msg += f"{crval_init[0]:.6f} {crval_init[1]:.6f}"
    msg += "\n" + f"jwst_utils.update_pure_parallel_wcs:      new crval "
    msg += f"{crval_fix[0]:.6f} {crval_fix[1]:.6f}"
    msg += "\n" + f"jwst_utils.update_pure_parallel_wcs:           dpix "
    msg += f"{crpix[0] - crpix_init[0]:6.3f} {crpix[1] - crpix_init[1]:6.3f}"

    _ = utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    with pyfits.open(file, mode="update") as im:
        im[1].header["CRVAL1"] = crval_fix[0]
        im[1].header["CRVAL2"] = crval_fix[1]
        im[1].header["PUREPWCS"] = True, "WCS updated from PP query"
        im[1].header["PUREPEXP"] = gs["fileName"], "FGS log file"

        im.flush()

    return True


def update_pure_parallel_wcs_old(
    file,
    fix_vtype="PARALLEL_PURE",
    recenter_footprint=True,
    verbose=True,
    fit_kwargs={"method": "powell", "tol": 1.0e-5},
    good_threshold=1.0,
):
    """
    Update pointing information of pure parallel exposures using the pointing
    information of the prime exposures from the MAST database and `pysiaf`

    *Deprecated*: use `grizli.jwst_utils.update_pure_parallel_wcs`

    1. Find the prime exposure from the MAST query that is closest in ``EXPSTART``
       to ``file``
    2. Use the ``apername, ra, dec`` values of the prime exposure from the MAST query
       and ``PA_V3`` from the ``file`` header to set the the pointing attitude with
       `pysiaf`
    3. Compute the sky position of the ``CRPIX`` reference pixel of ``file`` with
       `pysiaf` and put that position in the ``CRVAL`` keywords

    Parameters
    ----------
    file : str
        Filename of a pure-parallel exposure (rate.fits)

    fix_vtype : str
        Run if ``file[0].header['VISITYPE'] == fix_vtype``

    recenter_footprint : bool
        Recenter the footprint of the parallel exposure to match the prime exposure.

    verbose : bool
        Status messaging

    fit_kwargs : dict
        Arguments to pass to `scipy.optimize.minimize`.

    good_threshold : float
        Threshold for the fit to be considered "good" and update the WCS keywords.

    Returns
    -------
    status : None, True
        Returns None if some problem is found

    """
    from scipy.optimize import minimize
    import pysiaf
    from pysiaf.utils import rotations

    import mastquery.jwst
    from .aws import db

    if not os.path.exists(file):
        msg = "jwst_utils.update_pure_parallel_wcs: "
        msg += f" {file} not found"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None

    with pyfits.open(file) as im:
        h0 = im[0].header.copy()
        h1 = im[1].header.copy()
        if "VISITYPE" not in im[0].header:
            msg = "jwst_utils.update_pure_parallel_wcs: "
            msg += f" VISITYPE not found in header {file}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            return None

    # Is this a PARALLEL_PURE exposure?
    vtype = h0["VISITYPE"]

    if vtype != fix_vtype:
        msg = "jwst_utils.update_pure_parallel_wcs: "
        msg += f" VISITYPE ({vtype}) != {fix_vtype}, skip"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None

    # Find a match in the db
    try:
        _api = "https://grizli-cutout.herokuapp.com/pure_parallel?file={0}"
        prime = utils.read_catalog(_api.format(os.path.basename(file)), format="csv")
    except:
        try:
            prime = db.SQL(
                f"""select * from pure_parallel_exposures
            where par_file = '{os.path.basename(file)}'
            AND apername != 'NRS_FULL_MSA'
            """
            )
        except:
            msg = "jwst_utils.update_pure_parallel_wcs: db query failed"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            return None

    if len(prime) == 0:
        msg = f"jwst_utils.update_pure_parallel_wcs: par_file='{file}'"
        msg += " not found in db.pure_parallel_exposures"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
        return None

    crval_init = h1["CRVAL1"], h1["CRVAL2"]

    # Get correct pointing from FGS logs
    dt = 0.01
    gs = mastquery.jwst.query_guidestar_log(
        mjd=(h0["EXPSTART"] - dt, h0["EXPEND"] + dt),
        program=None,
        exp_type=["FGS_FINEGUIDE"],
    )

    keep = gs["expstart"] < h0["EXPSTART"]
    keep &= gs["expend"] > h0["EXPEND"]
    gs = gs[keep]

    # OK, we have a row, now compute the pysiaf pointing for the prime
    row = prime[0]

    # pa_v3 = h1['PA_V3']
    # pa_v3 = h1['ROLL_REF']
    pa_v3 = row["gs_v3_pa"]

    pos = np.array([row["ra"], row["dec"], pa_v3])

    prime_aper = pysiaf.Siaf(row["instrument_name"])[row["apername"]]

    if 0:
        pos = (h1["RA_V1"], h1["DEC_V1"], h1["PA_V3"])
        att = rotations.attitude(0.0, 0.0, *pos)

    if recenter_footprint:
        xy = utils.SRegion(row["footprint"]).xy[0].T

        if recenter_footprint > 1:
            x0 = pos * 1.0  # (row['ra'], row['dec'], row['gs_v3_pa'])
        else:
            x0 = pos[:2]

        _fit = minimize(
            objfun_pysiaf_pointing, x0, args=(prime_aper, xy, pa_v3, 0), **fit_kwargs
        )

        if recenter_footprint > 1:
            dPA = _fit.x[2] - pa_v3
        else:
            dPA = 0.0

        att = objfun_pysiaf_pointing(_fit.x, prime_aper, xy, pa_v3, 1)
        prime_aper.set_attitude_matrix(att)

        tv2, tv3 = prime_aper.sky_to_tel(row["ra"], row["dec"])

        msg = f"jwst_utils.update_pure_parallel_wcs: {prime_aper.AperName} offset"
        msg += f" v2,v3 = {tv2 - prime_aper.V2Ref:6.3f}, {tv3 - prime_aper.V3Ref:6.3f}"
        msg += f"  dPA = {dPA:.2f} "
        msg += f"(dx**2 = {_fit.fun:.2e}, nfev = {_fit.nfev})"

        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        if _fit.fun > good_threshold:
            att = rotations.attitude(prime_aper.V2Ref, prime_aper.V3Ref, *pos)
            prime_aper.set_attitude_matrix(att)

    else:
        att = rotations.attitude(prime_aper.V2Ref, prime_aper.V3Ref, *pos)
        prime_aper.set_attitude_matrix(att)

    # And apply the pointing to the parallel aperture and reference pixel
    par_aper = pysiaf.Siaf(h0["INSTRUME"])[h0["APERNAME"]]
    if 0:
        par_pos = (h1["RA_REF"], h1["DEC_REF"], pa_v3)  # h1['ROLL_REF'])
        apos = rotations.attitude(par_aper.V2Ref, par_aper.V3Ref, *par_pos)
        par_aper.set_attitude_matrix(apos)

    par_aper.set_attitude_matrix(att)

    crpix = h1["CRPIX1"], h1["CRPIX2"]
    crpix_init = par_aper.sky_to_sci(*crval_init)

    crval_fix = par_aper.sci_to_sky(*crpix)

    msg = f"jwst_utils.update_pure_parallel_wcs: {file}"
    msg += "\n" + f"jwst_utils.update_pure_parallel_wcs: prime {row['filename']} "
    msg += f"{row['apername']}"
    msg += "\n" + f"jwst_utils.update_pure_parallel_wcs: original crval "
    msg += f"{crval_init[0]:.6f} {crval_init[1]:.6f}"
    msg += "\n" + f"jwst_utils.update_pure_parallel_wcs:      new crval "
    msg += f"{crval_fix[0]:.6f} {crval_fix[1]:.6f}"
    msg += "\n" + f"jwst_utils.update_pure_parallel_wcs:           dpix "
    msg += f"{crpix[0] - crpix_init[0]:6.3f} {crpix[1] - crpix_init[1]:6.3f}"

    _ = utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    with pyfits.open(file, mode="update") as im:
        im[1].header["CRVAL1"] = crval_fix[0]
        im[1].header["CRVAL2"] = crval_fix[1]
        im[1].header["PUREPWCS"] = True, "WCS updated from PP query"
        im[1].header["PUREPEXP"] = row["filename"], "Prime exposure file"

        im.flush()

    return True


def objfun_pysiaf_pointing(theta, ap, xy, pa, ret):
    """
    Objective function for fitting a `pysiaf` attitude based on a MAST database
    footprint

    Parameters
    ----------
    theta : (float, float, float)
        ``ra``, ``dec`` and ``pa_v3`` at the aperture reference position

    ap : `pysiaf.Aperture`
        Aperture

    xy : array-like, (2,4)
        Footprint from the MAST query, e.g., ``xy = SRegion(footprint).xy[0].T``.

    pa : float
        Position angle of the prime exposure.

    ret : int
        Return behavior

    Returns
    -------
    if ret == 1:
        att : array-like
            Attitude matrix derived from the inputs
    else:
        resid : float
            Sum of squared differences ``ap.corners - xy`` for optimization

    """
    from pysiaf.utils import rotations

    if len(theta) == 3:
        pos = theta
    else:
        pos = [theta[0], theta[1], pa]

    att = rotations.attitude(ap.V2Ref, ap.V3Ref, *pos)
    ap.set_attitude_matrix(att)

    if ret == 1:
        return att

    tsky = np.array(ap.sky_to_tel(*xy))
    so = np.argsort(tsky[0, :])

    try:
        corners = np.array(ap.corners("tel"))
    except:
        corners = np.array(ap.corners("tel", rederive=False))

    cso = np.argsort(corners[0, :])
    diff = tsky[:, so] - corners[:, cso]

    # print(theta, (diff**2).sum())

    return (diff**2).sum()


def compute_siaf_pa_offset(
    c1, c2, c2_pa=202.9918, swap_coordinates=True, verbose=False
):
    """
    Eq. 10 from Bonaventura et al. for the small PA offset based on the median catalog position

    Seems to have a sign error relative to what APT calculates internally, used if `swap_coordinates=True`

    Parameters
    ----------
    c1 : array-like
        Catalog position (ra, dec).

    c2 : array-like
        APT position (ra, dec).

    c2_pa : float
        APT position angle (deg).

    swap_coordinates : bool
        Swap the coordinates to agree with APT.

    verbose : bool
        Print messaging to the terminal.

    Returns
    -------
    new_pa : float
        New position angle.

    dphi : float
        Delta position angle.

    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if not hasattr(c1, "ra"):
        cat_coord = SkyCoord(*c1, unit="deg")
    else:
        cat_coord = c1

    ac = cat_coord.ra.deg / 180 * np.pi
    dc = cat_coord.dec.deg / 180 * np.pi

    if not hasattr(c2, "ra"):
        apt_coord = SkyCoord(*c2, unit="deg")
    else:
        apt_coord = c2

    dx = apt_coord.spherical_offsets_to(cat_coord)

    ap = apt_coord.ra.deg / 180 * np.pi
    dp = apt_coord.dec.deg / 180 * np.pi

    # Needs to swap to agree with APT
    if swap_coordinates:
        cx, cy = ac * 1, dc * 1
        ac, dc = ap, dp
        ap, dp = cx, cy

    num = np.sin(ap - ac) * (np.sin(dc) + np.sin(dp))
    den = np.cos(dc) * np.cos(dp) + np.cos(ap - ac) * (1 + np.sin(dc) * np.sin(dp))

    dphi = np.arctan(num / den) / np.pi * 180

    if verbose:
        print(
            f"Catalog offset: {dx[0].to(u.arcsec):5.2f} {dx[1].to(u.arcsec):5.2f} new APA: {c2_pa + dphi:.5f}"
        )

    return c2_pa + dphi, dphi


def get_miri_photmjsr(
    file=None,
    filter="F770W",
    subarray="FULL",
    mjd=60153.23,
    photom_file="jwst_miri_photom_0201.fits",
    verbose=True,
):
    """
    Get time-dependent MIRI photometry values

    Parameters
    ----------
    file : str
        Image filename

    filter : str
        MIRI filter name

    subarray : str
        Detector subarray used

    mjd : float
        Observation epoch

    photom_file : str
        CRDS ``photom`` reference file name

    verbose : bool
        messaging

    Returns
    -------
    photmjsr : float
        Photometric scaling

    photom_corr : float
        Time-dependent correction for the filter and observation epoch

    """
    import astropy.io.fits as pyfits
    import jwst.datamodels

    if file is not None:
        with pyfits.open(file) as im:
            h = im[0].header
            filter = h["FILTER"]
            mjd = h["EXPSTART"]
            try:
                subarray = h["SUBARRAY"]
            except KeyError:
                pass

    try:
        from jwst.photom.miri_imager import time_corr_photom
    except ImportError:
        # msg = 'Failed to import `jwst.photom.miri_imager` to include time-dependence'
        # utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

        time_corr_photom = time_corr_photom_copy

    PATH = os.path.join(os.getenv("CRDS_PATH"), "references", "jwst", "miri")
    local_file = os.path.join(PATH, photom_file)
    remote_file = "https://jwst-crds.stsci.edu/unchecked_get/references/jwst/"
    remote_file += "jwst_miri_photom_0201.fits"

    use_path = local_file if os.path.exists(local_file) else remote_file

    with jwst.datamodels.open(use_path) as ref:

        test = ref.phot_table["filter"] == filter
        test &= ref.phot_table["subarray"] == subarray
        if test.sum() == 0:
            msg = f"Row not found in {photom_file} for {filter} / {subarray}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            return np.nan, None

        row = np.where(test)[0][0]
        photmjsr = ref.phot_table["photmjsr"][row]

        try:
            photom_corr = time_corr_photom(ref.timecoeff[row], mjd)
        except:
            photom_corr = 0.0

    return (photmjsr, photom_corr)


def time_corr_photom_copy(param, t):
    """
    Short Summary
    --------------
    Time dependent PHOTOM function.

    The model parameters are amplitude, tau, t0. t0 is the reference day
    from which the time-dependent parameters were derived. This function will return
    a correction to apply to the PHOTOM value at a given MJD.

    N.B.: copied from [jwst.photom.miri_imager](https://github.com/spacetelescope/jwst/blob/master/jwst/photom/miri_imager.py#L9)

    Parameters
    ----------
    param : numpy array
        Set of parameters for the PHOTOM value
    t : int
        Modified Julian Day (MJD) of the observation

    Returns
    -------
    corr: float
        The time-dependent correction to the photmjsr term.
    """

    amplitude, tau, t0 = param["amplitude"], param["tau"], param["t0"]
    corr = amplitude * np.exp(-(t - t0) / tau)

    return corr


def get_saturated_pixels(
    file="jw02561001002_06101_00001_nrca3_rate.fits",
    dq_array=None,
    saturated_flag="SATURATED",
    erode_dilate=(2, 5),
    rc_flag="RC",
    rc_iterations=2,
    **kwargs,
):
    """
    Get list of saturated pixels, e.g., for use in persistence masking

    Parameters
    ----------
    file : str
        Exposure filename with "DQ" extension

    dq_array : array-like

    saturated_flag : str
        Flag name in `jwst.datamodels.mask.pixel` to treat as "saturated"

    rc_flag : str
        Flag name in `jwst.datamodels.mask.pixel` for the "RC" pixels to exclude

    rc_iterations : int
        If > 0, make a mask of pixels flagged with the "RC" bit, dilate it and
        exclude them from the saturated list

    Returns
    -------
    flagged : array-like
        Boolean mask of flagged pixels
    """
    import scipy.ndimage as nd
    from skimage import morphology
    from jwst.datamodels.mask import pixel

    if dq_array is None:
        with pyfits.open(file) as im:
            dq_array = im["DQ"].data * 1

    flagged = (dq_array & pixel[saturated_flag]) > 0

    if rc_iterations > 0:
        rc = (dq_array & pixel[rc_flag]) > 0
        rc = nd.binary_dilation(rc, iterations=rc_iterations)
        flagged &= ~rc

    if erode_dilate is not None:
        extra = nd.binary_erosion(flagged, iterations=erode_dilate[0])
        extra = morphology.isotropic_dilation(extra, erode_dilate[1])
        flagged |= extra

    return flagged


def get_nirspec_persistence_mask(
    file="jw01180136001_11101_00002_nrs1_rate.fits",
    ok_bits=4,
    rnoise_threshold=10,
    closing_iterations=2,
    erosion_iterations=4,
    dilation_iterations=1,
    verbose=True,
    **kwargs,
):
    """
    Make a mask for NIRSpec pixels likely to cause persistence.

    Parameters
    ----------
    file : str
        Filename of a NIRSPec exposure file with a minimum of ``'DQ'`` and
        ``'VAR_RNOISE'`` extensions.

    ok_bits : int
        DQ bits to ignore when making the mask

    rnoise_threshold : float
        Threshold for masking pixels relative to ``median(VAR_RNOISE)``

    closing_iterations : int
        Number of `~scipy.ndimage.binary_closing` iterations on initial mask
        ``initial = valid_dq & VAR_RNOISE > rnoise_threshold * median_rnoise``

    erosion_iterations : int
        Number of `~scipy.ndimage.binary_erosion` iterations

    dilation_iterations : int
        Number of `~skimage.morphology.isotropic_dilation` iterations

    Returns
    -------
    flagged : bool array
        Final mask defined by

        .. code-block:: python
            :dedent:

            initial = valid_dq & (VAR_RNOISE > rnoise_threshold * median_rnoise)
            closed = scipy.ndimage.binary_closing(initial, closing_iterations)
            eroded = scipy.ndimage.binary_erosion(closed, erosion_iterations)
            flagged = skimage.morphology.isotropic_dilation(eroded, dilation_iterations)

    """
    import scipy.ndimage as nd
    import skimage.morphology

    with pyfits.open(file) as im:
        # DQ mask
        valid_dq = utils.mod_dq_bits(im["DQ"].data, ok_bits) == 0

        # Median RNOISE
        med_rnoise = np.nanmedian(im["VAR_RNOISE"].data[valid_dq])

        mask = nd.binary_closing(
            valid_dq & (im["VAR_RNOISE"].data > rnoise_threshold * med_rnoise),
            iterations=closing_iterations,
        )

    eroded = nd.binary_erosion(mask, iterations=erosion_iterations)

    flagged = skimage.morphology.isotropic_dilation(eroded, dilation_iterations)

    msg = f"get_nirspec_persistence_mask: {file} N={flagged.sum()}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    return flagged


def get_saturated_pixel_table(output="table", use_nirspec="auto", **kwargs):
    """
    Get table of pixel indices from `~grizli.jwst_utils.get_saturated_pixels`

    Parameters
    ----------
    output : ["array", "table", "df", "file"]
        Output type

    use_nirspec : bool, 'auto'
        - ``True`` and ``file`` keyword provided: use
          `~grizli.jwst_utils.get_nirspec_persistence_mask`
        - ``'auto'`` and ``'_nrs[12]'`` in ``kwargs[file]``: use
          `~grizli.jwst_utils.get_nirspec_persistence_mask`
        - else use `~grizli.jwst_utils.get_saturated_pixels`

    kwargs : dict
        Keyword args passed to `~grizli.jwst_utils.get_saturated_pixels` or
        `~grizli.jwst_utils.get_nirspec_persistence_mask`.

    Returns
    -------
    tab : (array, array), `~grizli.utils.GTable`, `pandas.DataFrame`
      - ``output="array"``: (i, j) array indices
      - ``output="table"``: Table with ``i`` and ``j`` columns
      - ``output="df"``: `pandas.DataFrame` with ``i`` and ``j`` columns

    """

    if ("file" in kwargs) & (use_nirspec in ["auto"]):
        use_nirspec = ("_nrs1" in kwargs["file"]) | ("_nrs2" in kwargs["file"])
    elif "file" not in kwargs:
        use_nirspec = False

    if use_nirspec:
        flagged = get_nirspec_persistence_mask(**kwargs)
    else:
        flagged = get_saturated_pixels(**kwargs)

    i, j = np.unravel_index(np.where(flagged.flatten())[0], flagged.shape)
    if output == "array":
        return (i, j)

    tab = utils.GTable()
    if "file" in kwargs:
        tab.meta["file"] = kwargs["file"]

    tab["i"] = i.astype(np.int32)
    tab["j"] = j.astype(np.int32)

    if output == "table":
        return tab

    elif output == "file":

        if "file" in kwargs:
            output_file = kwargs["file"].replace(".fits", ".sat.csv.gz")
            output_file = output_file.replace(".gz.gz", ".gz")
        else:
            output_file = "sat.csv.gz"

        df = tab.to_pandas()
        df.to_csv(output_file, index=False)

        return output_file, df

    else:
        return tab.to_pandas()


def query_persistence(flt_file, saturated_lookback=1.0e4, verbose=True):
    """
    Query ``exposure_saturated`` table for possible persistence
    """
    from .aws import db

    froot = os.path.basename(flt_file).split("_rate.fits")[0]

    # Get exposure start
    t0 = db.SQL(
        f"""
        SELECT expstart, detector from exposure_files
        WHERE file = '{froot}'
    """
    )

    # Bad pixels from other exposures within dt interval
    tstart = t0["expstart"][0] - saturated_lookback / 86400

    bpix_command = f"""
        SELECT i,j FROM exposure_files NATURAL JOIN exposure_saturated
        WHERE expstart > {tstart} AND expstart < {t0['expstart'][0]}
              AND detector = '{t0['detector'][0]}'
        GROUP BY i,j
    """
    res = db.SQL(bpix_command)

    # Single merged query is factors slower....
    if 0:
        res = db.SQL(
            f"""SELECT i, j
        FROM exposure_files e1, (exposure_files NATURAL JOIN exposure_saturated) e2
        WHERE
            e1.file = '{froot}'
            AND e2.expstart > e1.expstart - {saturated_lookback/86400}
            AND e2.expstart < e1.expstart
            AND e1.detector = e2.detector
        GROUP BY i,j
        """
        )

    msg = "query_persistence: "
    msg += f"Found {len(res)} flagged pixels for {flt_file} in `exposure_saturated`"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    return res


def flag_nirspec_hot_pixels(
    data="jw02073008001_03101_00002_nrs2_rate.fits",
    rnoise_percentile=90,
    rnoise_threshold=16,
    hot_filter_sn_max=-3,
    corner_sn_max=-2,
    jwst_dq_flags=JWST_DQ_FLAGS,
    dilate_footprint=PLUS_FOOTPRINT,
    **kwargs,
):
    """
    Flag NIRSpec MOS hot pixels

    Parameters
    ----------
    data : str, `~astropy.io.fits.HDUList`
        NIRSpec image filename or open HDU object with SCI, ERR, DQ extensions.

    rnoise_percentile : float
        Percentile of rnoise array for the absolute threshold

    rnoise_threshold : float
        The absolute ``hot_threshold`` is
        ``percentile(ERR_RNOISE, rnoise_percentile) * rnoise_threshold``

    max_filter_size, hot_filter_sn_max, corner_sn_max, jwst_dq_flags : int, float, float
        See `~grizli.jwst_utils.flag_nircam_hot_pixels`

    dilate_footprint : array-like
        Footprint for binary dilation on the dq mask

    Returns
    -------
    sn : array-like
        S/N array derived from ``file``

    dq : array-like, int
        Flagged pixels where
        ``hot = jwst.datamodels.mask.pixel["HOT"]`` and
        ``plus = jwst.datamodels.mask.pixel["WARM"]``

    count : int
        Number of flagged pixels

    """
    import scipy.ndimage as nd
    from jwst.datamodels.mask import pixel as pixel_codes

    if isinstance(data, str):
        is_open = True
        rate = pyfits.open(data)
    else:
        rate = data
        is_open = False

    bits = get_jwst_dq_bit(jwst_dq_flags)

    mask = (rate["DQ"].data & bits > 0) | (rate["ERR"].data <= 0)
    mask |= (rate["SCI"].data < -3 * rate["ERR"].data) | (
        ~np.isfinite(rate["SCI"].data)
    )

    pval = np.nanpercentile(np.sqrt(rate["VAR_RNOISE"].data[mask]), rnoise_percentile)
    hot_threshold = pval * rnoise_threshold

    sn, dq_flag, count = flag_nircam_hot_pixels(
        data=rate,
        err_extension="DATA",
        hot_threshold=hot_threshold,
        hot_filter_sn_max=hot_filter_sn_max,
        plus_sn_min=hot_threshold,
        corner_sn_max=corner_sn_max,
        jwst_dq_flags=jwst_dq_flags,
        **kwargs,
    )

    if dilate_footprint is not None:
        for flag in ["HOT", "WARM"]:
            dq_flag |= (
                nd.binary_dilation(
                    dq_flag & pixel_codes[flag] > 0, structure=dilate_footprint
                )
                * pixel_codes[flag]
            )

    if is_open:
        rate.close()

    return sn, dq_flag, count


def flag_nircam_hot_pixels(
    data="jw01837039001_02201_00001_nrcblong_rate.fits",
    err_extension="ERR",
    hot_threshold=7,
    max_filter_size=3,
    hot_filter_sn_max=5,
    plus_sn_min=4,
    corner_sn_max=3,
    jwst_dq_flags=JWST_DQ_FLAGS,
    verbose=True,
    **kwargs,
):
    """
    Flag isolated hot pixels and "plusses" around known bad pixels

    Parameters
    ----------
    data : str, `~astropy.io.fits.HDUList`
        NIRCam image filename or open HDU

    hot_threshold : float
        S/N threshold for central hot pixel

    max_filter_size : int
        Size of the local maximum filter where the central pixel is zeroed out

    hot_filter_sn_max : float
        Maximum allowed S/N of the local maximum excluding the central pixel

    plus_sn_min : float
        Minimum S/N of the pixels in a "plus" around known bad pixels

    corner_sn_max : float
        Maximum S/N of the corners around known bad pixels

    jwst_dq_flags : list
        List of JWST flag names

    verbose : bool
        Messaging

    Returns
    -------
    sn : array-like
        S/N array derived from ``file``

    dq : array-like, int
        Flagged pixels where ``hot = HOT`` and ``plus = WARM``

    count : int
        Number of flagged pixels

    Examples
    --------

        .. plot::
            :include-source:

            import numpy as np
            import matplotlib.pyplot as plt
            import astropy.io.fits as pyfits

            from grizli.jwst_utils import flag_nircam_hot_pixels

            signal = np.zeros((48,48), dtype=np.float32)

            # hot
            signal[16,16] = 10

            # plus
            for off in [-1,1]:
                signal[32+off, 32] = 10
                signal[32, 32+off] = 7

            err = np.ones_like(signal)
            np.random.seed(1)
            noise = np.random.normal(size=signal.shape)*err

            dq = np.zeros(signal.shape, dtype=int)
            dq[32,32] = 2048 # HOT

            header = pyfits.Header()
            header['MDRIZSKY'] = 0.

            hdul = pyfits.HDUList([
                pyfits.ImageHDU(data=signal+noise, name='SCI', header=header),
                pyfits.ImageHDU(data=err, name='ERR'),
                pyfits.ImageHDU(data=dq, name='DQ'),
            ])

            sn, dq_flag, count = flag_nircam_hot_pixels(hdul)

            fig, axes = plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True)

            axes[0].imshow(signal + noise, vmin=-2, vmax=9, cmap='gray')
            axes[0].set_xlabel('Simulated data')
            axes[1].imshow(dq_flag, cmap='magma')
            axes[1].set_xlabel('Flagged pixels')

            for ax in axes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            fig.tight_layout(pad=1)

            plt.show()

    """
    import scipy.ndimage as nd
    from jwst.datamodels.mask import pixel as pixel_codes

    if isinstance(data, str):
        is_open = True
        rate = pyfits.open(data)
    else:
        rate = data
        is_open = False

    bits = get_jwst_dq_bit(jwst_dq_flags)

    mask = (rate["DQ"].data & bits > 0) | (rate["ERR"].data <= 0)
    mask |= (rate["SCI"].data < -3 * rate["ERR"].data) | (
        ~np.isfinite(rate["SCI"].data)
    )

    if "MDRIZSKY" in rate["SCI"].header:
        bkg = rate["SCI"].header["MDRIZSKY"]
    else:
        bkg = np.nanmedian(rate["SCI"].data[~mask])

    indat = rate["SCI"].data - bkg

    if "BKG" in rate:
        indat -= rate["BKG"].data

    indat[mask] = 0.0
    if err_extension == "ERR":
        sn = indat / rate["ERR"].data
    elif err_extension == "VAR_RNOISE":
        sn = indat / np.sqrt(rate["VAR_RNOISE"].data)
    else:
        sn = indat * 1.0

    sn[mask] = 0

    ##########
    # Isolated hot pixels
    footprint = np.ones((max_filter_size, max_filter_size), dtype=bool)
    footprint[(max_filter_size - 1) // 2, (max_filter_size - 1) // 2] = False

    snmax = nd.maximum_filter(sn, footprint=footprint)

    hi = sn > hot_threshold
    if hot_filter_sn_max < 0:
        hot = hi & (snmax < sn * -1 / hot_filter_sn_max)
    else:
        hot = hi & (snmax < hot_filter_sn_max)

    ###########
    # Plus mask
    sn_up = sn * 1
    sn_up[mask] = 1000

    dplus = nd.minimum_filter(sn, footprint=PLUS_FOOTPRINT)

    dcorner = nd.maximum_filter(sn, footprint=CORNER_FOOTPRINT)

    if corner_sn_max < 0:
        plusses = (dplus > plus_sn_min) & (dcorner < dplus * -1 / corner_sn_max)
    else:
        plusses = (dplus > plus_sn_min) & (dcorner < corner_sn_max)

    plusses &= (rate["DQ"].data & bits > 0) | hot

    plus_mask = nd.binary_dilation(plusses, structure=PLUS_FOOTPRINT)

    dq = (hot * pixel_codes["HOT"]) | (plus_mask * pixel_codes["WARM"])

    msg = f"flag_nircam_hot_pixels : hot={hot.sum()} plus={plus_mask.sum()}"
    utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

    if is_open:
        rate.close()

    return sn, dq, (dq > 0).sum()


def mast_exposure_attitude(
    filename="jw06640052001_0310h_00001_nrs1_rate.fits",
    row=None,
    gs_pa_offset=-0.1124,
    verbose=True,
    **kwargs,
):
    """
    Generate JWST spacecraft attitude matrix relevant for a particular exposure from
    MAST query columns

    Parameters
    ----------
    filename : str
        JWST exposure filename

    gs_pa_offset : float
        Empirical offset added to ``gs_v3_pa`` from the MAST query to match
        ``ROLL_REF`` in the science headers, since ``ROLL_REF`` doesn't seem to be
        available directly from queries to the MAST db.  The default value was
        derived from an MSA exposure.

    Returns
    -------
    att : (3, 3) array-like
        Attitude matrix for use with `pysiaf`

    """
    from mastquery.jwst import make_query_filter, query_jwst
    import pysiaf
    from pysiaf import rotations

    file_split = filename.split("_")
    filters = make_query_filter("fileSetName", values=["_".join(file_split[:3])])

    instrument_short = file_split[3][:3].upper()
    instrument = {
        "NRS": "NIRSPEC",
        "NRC": "NIRCAM",
        "MIR": "MIRI",
        "NIS": "NIRISS",
    }[instrument_short]

    if row is None:
        mast = query_jwst(
            instrument=instrument_short,
            filters=filters,
            columns="*",
            rates_and_cals=True,
            extensions=["rate", "cal"],
        )

        if len(mast) == 0:
            msg = f"No MAST exposures found for {filename}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            return None

        row = mast[0]

    siaf = pysiaf.siaf.Siaf(instrument)

    ap = siaf[row["apername"]]

    roll = row["gs_v3_pa"] + gs_pa_offset
    idl_v2, idl_v3 = ap.idl_to_tel(row["xoffset"], row["yoffset"])
    att = rotations.attitude(idl_v2, idl_v3, row["targ_ra"], row["targ_dec"], roll)

    return att


MAST_APERTURES = {
    "NIRSPEC": [
        "NRS_S200A1_SLIT",
        "NRS_S200A2_SLIT",
        "NRS_S400A1_SLIT",
        "NRS_S1600A1_SLIT",
        "NRS_S200B1_SLIT",
        "NRS_FULL_IFU",
        "NRS_VIGNETTED_MSA1",
        "NRS_VIGNETTED_MSA2",
        "NRS_VIGNETTED_MSA3",
        "NRS_VIGNETTED_MSA4",
    ],
    "NIRCAM": [
        "NRCA1_FULL",
        "NRCA2_FULL",
        "NRCA3_FULL",
        "NRCA4_FULL",
        "NRCA5_FULL",
        "NRCB1_FULL",
        "NRCB2_FULL",
        "NRCB3_FULL",
        "NRCB4_FULL",
        "NRCB5_FULL",
    ],
    "MIRI": ["MIRIM_ILLUM", "MIRIM_TALRS", "MIRIM_SLIT"],
    "NIRISS": ["NIS_CEN"],
}


def mast_exposure_apertures(
    filename="jw06640052001_0310h_00001_nrs1_rate.fits",
    mast_apertures=MAST_APERTURES,
    attitude=None,
    output="list",
    siaf_frame="sky",
    **kwargs,
):
    """
    Generate `pysiaf` apertures associated with a particular telescope pointing

    Parameters
    ----------
    filename : str
        JWST exposure filename

    mast_apertures : dict
        List of apertures to generate with keys of the instrument name

    attitude : array-like
        `pysiaf` attitude matrix.  If not provided, try to generate from ``filename``
        and other inputs

    output : string
        Output type:
          - ``list``: list of `pysiaf.Aperture` objects
          - ``sregion``: list of `sregion.SRegion` objects
          - ``reg``: list of DS9 region strings

    Returns
    -------
    result : list
        List of aperture information based on ``output``
    """
    import pysiaf

    # Get attitude matrix
    if attitude is None:
        attitude = mast_exposure_attitude(filename=filename, **kwargs)
        if attitude is None:
            return None

    apertures = []
    for instrument in mast_apertures:
        siaf = pysiaf.Siaf(instrument)
        for aper_name in mast_apertures[instrument]:
            ap = siaf[aper_name]
            ap.set_attitude_matrix(attitude)
            apertures.append(ap)

    if output == "list":
        return apertures
    elif output in ["sregion", "reg"]:
        sregions = []
        for ap in apertures:
            try:
                sr = utils.SRegion(np.array(ap.corners(siaf_frame)), wrap=False)
            except TypeError:
                sr = utils.SRegion(
                    np.array(ap.corners(siaf_frame, rederive=False)), wrap=False
                )
            sregions.append(sr)

        for sr, ap in zip(sregions, apertures):
            sr.label = ap.AperName

        if output == "sregion":
            return sregions
        else:
            regs = [sr.region[0] for sr in sregions]
            return regs
    else:
        msg = "mast_exposure_apertures: output must be 'list', 'sregion', 'reg'"
        raise ValueError(msg)
