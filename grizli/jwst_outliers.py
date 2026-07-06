import numpy as np
import copy
import traceback
from collections import defaultdict
from pathlib import Path
import os
import shutil
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import fits
from astropy.modeling import custom_model, models as amodels
from astropy.wcs import WCS as FitsWCS
from gwcs import coordinate_frames as cf
from gwcs import wcs as gwcs_wcs
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
from grizli import utils
from grizli import jwst_utils

SCI = "SCI"
ERR = "ERR"
DQ = "DQ"
ORIGINAL_KEYS = {"TELESCOP": "OTELESCO", "INSTRUME": "OINSTRUM", "DETECTOR": "ODETECTO", "EXP_TYPE": "OEXP_TYP"}

def log(msg, verbose=True):
    utils.log_comment(utils.LOGFILE, "# jwst_rate_outliers: " + str(msg), verbose=verbose, show_date=True)

def pathfix(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path

def hval(header, key, default=""):
    key = key.upper()
    return header.get(ORIGINAL_KEYS.get(key, "O" + key[:7]), header.get(key, default))

def file_info(path):
    with fits.open(path, memmap=False) as hdul:
        h = hdul[0].header
        return {
            "instrument": str(hval(h, "INSTRUME")).strip().upper(),
            "detector": str(hval(h, "DETECTOR")).strip().upper(),
            "filter": str(hval(h, "FILTER")).strip().upper(),
            "pupil": str(hval(h, "PUPIL")).strip().upper(),
            "exp_type": str(hval(h, "EXP_TYPE", "NRC_IMAGE")).strip().upper(),
            "shape": tuple(hdul[SCI].data.shape),
        }
    
def get_mdrizsky(path):
    with fits.open(path, memmap=False) as hdul:
        h = hdul["SCI"].header
        return float(h["MDRIZSKY"])

def rate_to_cal_path(path):
    path = Path(path)
    return path.with_name(path.name.replace("_rate.fits", "_cal.fits"))

def cleanup_files(prep_dir=".", verbose=True): #delete *blot.fits and *median.fits files which seem to not be getting removed...
    prep_dir = pathfix(prep_dir)
    removed = 0
    for pattern in ("*blot.fits", "*median.fits"):
        for path in prep_dir.glob(pattern):
            if path.is_file():
                log(f"removing intermediate file {path.name}", verbose)
                path.unlink() #.unlink() is same as delete 
                removed += 1

    log(f"removed {removed} JWST outlier intermediate files from {prep_dir}", verbose)

    return removed

def make_model(path, index):
    """
    Make ImageModel from grizli-prepped rate file w/ dither naming /bkg stuff for outlier rejection. 
    """
    path = Path(path)

    model = jwst_utils.match_gwcs_to_sip(
        str(path),
        overwrite=False)   # do not rewrite the Grizli-prepped rate file here, we only want to make new DQ flags at this stage and flush those

    # name the files for jwst-pipeline (still needs this?)
    model.meta.filename = path.name
    model.meta.group_id = f"dither{index:03d}"

    # bkg stuff - not sure is used/matters but set it just in case. 
    model.meta.background.subtracted = False #sci extension still has sky-level in it I think
    model.meta.background.level = get_mdrizsky(path) #use mdrizsky value from prior astrodrizzle run, shoudl be in SCI header

    return model, model.dq.copy()

def run_jwst_outliers(models, driz_cr_snr, driz_cr_scale):
    from jwst.datamodels import ModelContainer
    from jwst.outlier_detection.outlier_detection_step import OutlierDetectionStep

    container = ModelContainer(open_models=False) #make "container" object
    for m in models:
        container.append(m) #add the DataModels made from the grizli-processed rate files into the container.

    result = OutlierDetectionStep( #run the rejection step w/ all defualt params except for the snr and scale param
        snr=driz_cr_snr,
        scale=driz_cr_scale, 
        save_intermediate_results=False,       
        in_memory=True).process(container)
    
    output_list_of_updated_dq_arrays = [ #make list of output dq arrays that have the new bits set. 
        np.asarray(result[i].dq, dtype=np.uint32).copy()
        for i in range(len(models))
    ]

    return output_list_of_updated_dq_arrays

def run_rate_file_group_outlier_dq(
    rate_files,
    driz_cr_snr="8.0 5.0",
    driz_cr_scale="2.5 0.7",
    min_files=2,
    verbose=True):
    
    paths = [pathfix(f) for f in rate_files]

    if len(paths) < min_files:
        log("only %d file(s); skipping" % len(paths), verbose)
        return 0

    log("running JWST outlier DQ on %d rate files." % len(paths), verbose)

    cal_paths = [rate_to_cal_path(p) for p in paths]

    models = []
    old_dq = []
    renamed = []

    try:
        #Rename *_rate.fits -> *_cal.fits (gets around weird bug where jwst pipeline deletes rate files at the end?)
        for rate_path, cal_path in zip(paths, cal_paths):
            log(f"rename {rate_path.name} -> {cal_path.name}", verbose)
            rate_path.rename(cal_path)
            renamed.append((rate_path, cal_path))

        # Run outlier detection on the temporary "_cal".fits files...
        for i, cal_path in enumerate(cal_paths, 1):
            model, dq = make_model(cal_path, i)

            model.meta.filename = cal_path.name
            model.meta.group_id = f"dither{i:03d}"

            models.append(model)
            old_dq.append(dq)

        result_dq_arrays = run_jwst_outliers(
            models,
            driz_cr_snr,
            driz_cr_scale,
        )

        #write new dq bits... (might already be done but just to make sure)
        for i, cal_path in enumerate(cal_paths):
            result_dq = result_dq_arrays[i]
            added = result_dq & ~old_dq[i]

            with fits.open(cal_path, mode="update", memmap=False) as hdul:
                new_dq = np.asarray(hdul[DQ].data, dtype=np.uint32) | added #use "or" operator to keep old bits too...
                hdul[DQ].data[...] = new_dq.astype(
                    hdul[DQ].data.dtype,
                    copy=False,
                )
                hdul.flush() #save

            log("updated %s: added %d new bits to DQ-array." % (paths[i].name, int(np.count_nonzero(added != 0))), verbose)

    finally:
        # Close sll the models before renaming files back...
        for model in models:
            model.close()

        #Rename back at end
        for rate_path, cal_path in reversed(renamed):
                cal_path.rename(rate_path)

    return 1

def run_nircam_rate_outliers(
    visit,
    driz_cr_snr="8.0 5.0",
    driz_cr_scale="2.5 0.7",
    min_files=2,
    group_by=("detector", "filter", "pupil", "shape"),
    clean=True,
    verbose=True): 

    prep_dir = pathfix(visit["files"][0]).parent

    groups = defaultdict(list)
    for filename in visit["files"]:
        path = pathfix(filename)
        info = file_info(path)
        key = tuple(info[k] for k in group_by)
        groups[key].append(path)

    for key in sorted(groups, key=str):
        log("group %s: %d files" % (key, len(groups[key])), verbose)

        _ = run_rate_file_group_outlier_dq( #run the main-processing function
            groups[key],
            driz_cr_snr=driz_cr_snr,
            driz_cr_scale=driz_cr_scale,
            min_files=min_files,
            verbose=verbose)
        
    if clean:
        cleanup_files(prep_dir, verbose=verbose) #delete stuff we dont need

    return 1
