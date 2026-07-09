"""Run JWST outlier detection on Grizli-prepped rate files."""

import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits

from grizli import jwst_utils
from grizli import utils

SCI = "SCI"
ERR = "ERR"
DQ = "DQ"

JWST_OUTLIER_BIT = 16
GRIZLI_OUTLIER_BIT = 4096
#pre-make a mask to unset DQ=16 to save linespace...
JWST_OUTLIER_CLEAR_MASK = np.bitwise_not(np.uint32(JWST_OUTLIER_BIT))

#grizli modifies key names so shold check for both
ORIGINAL_KEYS = {
    "TELESCOP": "OTELESCO",
    "INSTRUME": "OINSTRUM",
    "DETECTOR": "ODETECTO",
    "EXP_TYPE": "OEXP_TYP",
}


def log(msg, verbose=True):
    """
    Write a JWST outlier message to the Grizli log.

    Parameters
    ----------
    msg : object
        Message to write to the log.

    verbose : bool
        Print the message if ``True``.

    Returns
    -------
    None
    """
    utils.log_comment(
        utils.LOGFILE,
        "# jwst_rate_outliers: " + str(msg),
        verbose=verbose,
        show_date=True,
    )


def pathfix(path):
    """
    Return an absolute ``Path`` with user expansion applied.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path.

    Returns
    -------
    path : pathlib.Path
        Absolute path.
    """
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    return path


def hval(header, key, default=""):
    """
    Get a FITS header value, checking Grizli original-key aliases first.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header to query.

    key : str
        Header keyword.

    default : object
        Default value returned if ``key`` is not found.

    Returns
    -------
    value : object
        Header value.
    """
    key = key.upper()
    original_key = ORIGINAL_KEYS.get(key, "O" + key[:7])
    fallback = header.get(key, default)
    value = header.get(original_key, fallback)

    return value


def file_info(path):
    """
    Read grouping metadata from a rate file.

    Parameters
    ----------
    path : str or pathlib.Path
        Input FITS rate file.

    Returns
    -------
    info : dict
        Dictionary with instrument, detector, filter, pupil, exposure type,
        and science-array shape.
    """
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        info = {
            "instrument": str(hval(header, "INSTRUME")).strip().upper(),
            "detector": str(hval(header, "DETECTOR")).strip().upper(),
            "filter": str(hval(header, "FILTER")).strip().upper(),
            "pupil": str(hval(header, "PUPIL")).strip().upper(),
            "exp_type": str(
                hval(header, "EXP_TYPE", "NRC_IMAGE")
            ).strip().upper(),
            "shape": tuple(hdul[SCI].data.shape),
        }

    return info


def get_mdrizsky(path):
    """
    Get the ``MDRIZSKY`` value from the science extension.

    Parameters
    ----------
    path : str or pathlib.Path
        Input FITS rate file.

    Returns
    -------
    mdrizsky : float
        Sky level recorded in the ``SCI`` header.
    """
    with fits.open(path, memmap=False) as hdul:
        header = hdul[SCI].header
        mdrizsky = float(header["MDRIZSKY"])

    return mdrizsky


def rate_to_cal_path(path):
    """
    Convert a ``*_rate.fits`` filename to the temporary ``*_cal.fits`` name.

    Parameters
    ----------
    path : str or pathlib.Path
        Input rate filename.

    Returns
    -------
    cal_path : pathlib.Path
        Temporary cal filename.
    """
    path = Path(path)
    cal_path = path.with_name(path.name.replace("_rate.fits", "_cal.fits"))

    return cal_path


def cleanup_files(prep_dir=".", verbose=True):
    """
    Delete JWST outlier intermediate files from a prep directory.

    Parameters
    ----------
    prep_dir : str or pathlib.Path
        Directory to clean.

    verbose : bool
        Print log messages if ``True``.

    Returns
    -------
    removed : int
        Number of files removed.
    """
    prep_dir = pathfix(prep_dir)
    removed = 0

    for pattern in ("*blot.fits", "*median.fits"):
        for path in prep_dir.glob(pattern):
            if path.is_file():
                log(f"removing intermediate file {path.name}", verbose)
                path.unlink()
                removed += 1

    log(
        f"removed {removed} JWST outlier intermediate files from {prep_dir}",
        verbose,
    )

    return removed


def make_model(path, index):
    """
    Make an ImageModel from a Grizli-prepped rate file.

    Parameters
    ----------
    path : str or pathlib.Path
        Input FITS rate file.

    index : int
        One-indexed dither number used to set ``meta.group_id``.

    Returns
    -------
    model : jwst.datamodels.ImageModel
        Data model passed to ``OutlierDetectionStep``.

    old_dq : ndarray
        Copy of the input DQ array before clearing JWST outlier bit 16.
    """
    path = Path(path)

    # Do not rewrite the Grizli-prepped rate file here.  At this stage we
    # only want to make new DQ flags and write them explicitly below.
    model = jwst_utils.match_gwcs_to_sip(
        str(path),
        overwrite=False,
    )

    # Name the files for the JWST pipeline.
    model.meta.filename = path.name
    model.meta.group_id = f"dither{index:03d}"

    # Keep the input DQ array before letting the JWST step add DQ=16.
    old_dq = model.dq.copy()

    # Clear any existing JWST OUTLIER bit before the step runs.  The step can
    # set bit 16 again, and this module maps that mask to Grizli DQ=4096.
    model.dq[...] = (
        np.asarray(model.dq, dtype=np.uint32) & JWST_OUTLIER_CLEAR_MASK
    ).astype(model.dq.dtype, copy=False)

    # The science extension should still have sky level in it.  Set the
    # background metadata from the prior AstroDrizzle MDRIZSKY value.
    model.meta.background.subtracted = False
    model.meta.background.level = get_mdrizsky(path)

    return model, old_dq


def run_jwst_outliers(models, **kwargs):
    """
    Run JWST ``OutlierDetectionStep`` on a list of data models.

    Parameters
    ----------
    models : list
        JWST data models made from Grizli-prepped rate files.

    **kwargs : dict
        Keyword arguments passed directly to ``OutlierDetectionStep``.

    Returns
    -------
    outlier_dq_arrays : list
        Updated DQ arrays from the outlier step, converted to ``uint32``.
    """
    from jwst.datamodels import ModelContainer
    from jwst.outlier_detection.outlier_detection_step import (
        OutlierDetectionStep,
    )

    container = ModelContainer(open_models=False)
    for model in models:
        container.append(model)

    step = OutlierDetectionStep(**kwargs)
    result = step.process(container)

    # make list of output dq arrays that have the new bits set.
    try:
        # jwst=1.13
        outlier_dq_arrays = [ 
            np.asarray(result[i].dq, dtype=np.uint32).copy()
            for i in range(len(models))
        ]
    except TypeError:
        # jwst=1.16
        # datamodel orig. dq attr should be updated too?
        outlier_dq_arrays = [ 
            np.asarray(m.dq, dtype=np.uint32).copy()
            for m in models
        ]

    return outlier_dq_arrays


def run_rate_file_group_outlier_dq(
    rate_files,
    driz_cr_snr="8.0 5.0",
    driz_cr_scale="2.5 0.7",
    min_files=2,
    verbose=True,
    jwst_outliers_kwargs=None,
):
    """
    Run JWST outlier detection for one group of rate files.

    Parameters
    ----------
    rate_files : list
        Rate files to process together.

    driz_cr_snr : str
        Default ``snr`` value passed to ``OutlierDetectionStep``.

    driz_cr_scale : str
        Default ``scale`` value passed to ``OutlierDetectionStep``.

    min_files : int
        Minimum number of files needed to run the outlier step.

    verbose : bool
        Print log messages if ``True``.

    jwst_outliers_kwargs : dict or None
        Extra keyword arguments passed to ``OutlierDetectionStep``.  Explicit
        ``snr`` and ``scale`` values override ``driz_cr_snr`` and
        ``driz_cr_scale``.

    Returns
    -------
    status : int
        ``1`` if the group was processed and ``0`` if it was skipped.
    """
    if jwst_outliers_kwargs is None:
        step_kwargs = {}
    else:
        step_kwargs = dict(jwst_outliers_kwargs)

    if "snr" not in step_kwargs:
        step_kwargs["snr"] = driz_cr_snr

    if "scale" not in step_kwargs:
        step_kwargs["scale"] = driz_cr_scale

    paths = [pathfix(rate_file) for rate_file in rate_files]

    if len(paths) < min_files:
        log("only %d file(s); skipping" % len(paths), verbose)
        return 0

    log("running JWST outlier DQ on %d rate files." % len(paths), verbose)

    cal_paths = [rate_to_cal_path(path) for path in paths]

    models = []
    old_dq = []
    renamed = []

    try:
        # Rename *_rate.fits to *_cal.fits so the JWST pipeline 
        # doesnt delete the *rate files when done w. O.R. step ...
        for rate_path, cal_path in zip(paths, cal_paths):
            log(f"rename {rate_path.name} -> {cal_path.name}", verbose)
            rate_path.rename(cal_path)
            renamed.append((rate_path, cal_path))

        # Run outlier detection on the temporary *_cal.fits files.
        # Loop over each temporary *_cal.fits file.
        for i, cal_path in enumerate(cal_paths, 1):
            # Make the JWST data model and copy the starting DQ array.
            model, dq = make_model(cal_path, i)

            # Add this model to the list passed to OutlierDetectionStep.
            models.append(model)

            # Add the starting DQ array to the list used for comparisons.
            old_dq.append(dq)

        # Run OutlierDetectionStep and return one updated DQ array per model.
        result_dq_arrays = run_jwst_outliers(models, **step_kwargs)

        # Loop over each file and map JWST DQ=16 to Grizli DQ=4096.
        for i, cal_path in enumerate(cal_paths):
            # Select the updated DQ array for this file.
            result_dq = result_dq_arrays[i]

            # Find pixels where OutlierDetectionStep set JWST DQ=16.
            # get boolean where bit was set
            step_outliers = (result_dq & JWST_OUTLIER_BIT) > 0

            # Find pixels that already had JWST DQ=16 before the step.
            # returns boolean where bit was set
            old_step_outliers = (old_dq[i] & JWST_OUTLIER_BIT) > 0

            # Keep every pixel flagged by either the old or new DQ=16 mask.
            # this gives true where bit=16 was set (before or after)
            grizli_outliers = step_outliers | old_step_outliers

            # Find pixels that already had Grizli DQ=4096 before the step.
            old_grizli_outliers = (old_dq[i] & GRIZLI_OUTLIER_BIT) > 0

            # Find pixels that are newly getting Grizli DQ=4096 here.
            # grab ones that are new for accounting in log
            new_grizli_outliers = grizli_outliers & ~old_grizli_outliers

            with fits.open(cal_path, mode="update", memmap=False) as hdul:
                #pull out DQ array that is in rate file...
                dq_data = np.asarray(hdul[DQ].data, dtype=np.uint32)
                # clear any existing DQ=16 bits in file...
                dq_data = dq_data & JWST_OUTLIER_CLEAR_MASK

                # set grizli bit where bit=16 was set (before or after)
                dq_data |= (
                    grizli_outliers.astype(np.uint32) * GRIZLI_OUTLIER_BIT
                )
                # update dq array in rate file
                hdul[DQ].data[...] = dq_data.astype(
                    hdul[DQ].data.dtype,
                    copy=False,
                )
                # save
                hdul.flush()

            log(
                "updated %s: mapped DQ=%d to DQ=%d; %d new pixels."
                % (
                    paths[i].name,
                    JWST_OUTLIER_BIT,
                    GRIZLI_OUTLIER_BIT,
                    int(np.count_nonzero(new_grizli_outliers)),
                ),
                verbose,
            )

    except Exception:
        log("JWST outlier DQ failed; traceback follows.", verbose)
        # throw error msg to grizli log if needed...
        utils.log_exception(utils.LOGFILE, traceback)
        raise

    finally:
        # Close all models before renaming files back.
        for model in models:
            model.close()

        # Rename files back at the end.
        for rate_path, cal_path in reversed(renamed):
            log(f"rename back {cal_path.name} -> {rate_path.name}", verbose)
            cal_path.rename(rate_path)

    return 1


def run_nircam_rate_outliers(
    visit,
    driz_cr_snr="8.0 5.0",
    driz_cr_scale="2.5 0.7",
    min_files=2,
    group_by=("detector", "filter", "pupil", "shape"),
    clean=True,
    verbose=True,
    jwst_outliers_kwargs=None,
):
    """
    Run JWST outlier detection for grouped files in a visit.

    Parameters
    ----------
    visit : dict
        Visit dictionary with a ``files`` entry containing rate filenames.

    driz_cr_snr : str
        Default ``snr`` value passed to ``OutlierDetectionStep``.

    driz_cr_scale : str
        Default ``scale`` value passed to ``OutlierDetectionStep``.

    min_files : int
        Minimum number of files needed in a group to run the outlier step.

    group_by : tuple
        Metadata keys used to group files before running the outlier step.

    clean : bool
        Delete JWST intermediate files after processing if ``True``.

    verbose : bool
        Print log messages if ``True``.

    jwst_outliers_kwargs : dict or None
        Extra keyword arguments passed to ``OutlierDetectionStep``.

    Returns
    -------
    status : int
        ``1`` after processing all groups.
    """
    prep_dir = pathfix(visit["files"][0]).parent

    groups = defaultdict(list)
    for filename in visit["files"]:
        path = pathfix(filename)
        info = file_info(path)
        key = tuple(info[item] for item in group_by)
        groups[key].append(path)

    for key in sorted(groups, key=str):
        log("group %s: %d files" % (key, len(groups[key])), verbose)

        run_rate_file_group_outlier_dq(
            groups[key],
            driz_cr_snr=driz_cr_snr,
            driz_cr_scale=driz_cr_scale,
            min_files=min_files,
            verbose=verbose,
            jwst_outliers_kwargs=jwst_outliers_kwargs,
        )

    if clean:
        cleanup_files(prep_dir, verbose=verbose)

    return 1
