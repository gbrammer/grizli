"""
Query the NASA Horizons Small Body Identification API for Solar System objects
that could pass through JWST images

See `grizli.horizons.demo`:

.. image:: ../_static/j100028p0218_cosmos-1-f1800w_00157_small_bodies.png

"""
import glob
import os

import numpy as np
import urllib.request
import json
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, Angle
import astropy.time
import astropy.units as u

from . import utils
from .aws import db

## Spacecraft codes:
## BODY_CODE is for calculating the time-specific geocentric VECTORS of the
## spacecraft and MPC is the code used when observing "from" the spacecraft
JWST_BODY_CODE = "2021-130A"
JWST_MPC = "500@-170"

HST_BODY_CODE = "1990-037B"
HST_MPC = "500@-48"

SPACECRAFT_BODY_CODE = JWST_BODY_CODE
SPACECRAFT_MPC = JWST_MPC

def query_horizons_small_bodies(assoc="j100028p0218_cosmos-1-f1800w_00157", coords=None, mjd_range=None, prefix="user", with_db=True, sb_ident_params="&two-pass=true&suppress-first-pass=true", query_arcmin=15, ephem_steps=256, make_plot=True, plot_all_tracks=False):
    """
    Query the Horizons Small Body Identification tool at the epoch and pointing
    location of the exposures in a grizli/dja association.

    The script first queries Horizons to "observe" the spacecraft from the Earth
    geocenter to get its time-dependent position, then queries the sb_ident API.

    The spacecraft is specified in the ``SPACECRAFT_BODY_CODE`` and
    ``SPACECRAFT_MPC`` global variables, where the first is used to retrieve
    the spacecraft geocentric position and velocity vectors and the latter is
    used to generate the SB ephemerides as observed by the spacecraft. The
    defaults are set for JWST: ``JWST_MPC = "500@-170"`` and ``JWST_BODY_CODE =
    "2021-130A"``.

    .. note::
        Note that the sb_ident query can take several minutes to complete.  It
        also seems to timeout if more than ~2 queries are sent simultaneously.

    Parameters
    ----------
    assoc : str
        Association name

    coords : (float, float), None
        Coordinates to query.  Both ``coords`` and ``mjd_range`` must be
        specified for manual input

    mjd_range : [float, float]
        Begginning and end MJD times to query.  The spacecraft location is
        computed at ``mean(mjd_range)``.

    prefix : str
        File prefix for output when ``coords`` and ``mjd_range`` provided

    with_db : bool
        Get assoc data with a DB query, otherwise use the API at
        https://grizli-cutout.herokuapp.com/assoc_json?name={assoc}

    sb_ident_params : str
        Parameters for the horizons sb_ident API

    query_arcmin : float
        Radius of the query from the pointing center.  Probably set this
        a bit larger than the expected instrumental FoV.

    ephem_steps : int
        Number of steps requested in the body ephemeris.  If ``ephem_steps < 0``
        interpret as a step size in seconds and calculate from
        ``nsteps = (mjd_range[1] - mjd_range[0]) * 86400 / abs(ephem_steps)``.

    make_plot : bool
        Make a simple diagnostic plot

    plot_all_tracks : bool
        Plot all nearby tracks, or only those that intersect with the exposure
        footprints

    Returns
    -------
    fig : `matplotlib.Figure`
        Footprint + tracks plot
    
    fephem : dict
        Dictionary of SB ephemerides
    
    keep : list
        Boolean list of elements in ``fephem`` that intersect with the exposure
        footprints

    """

    if (coords is None) | (mjd_range is None):

        if with_db:
            assoc_data = db.SQL(
                f"select * from assoc_table where assoc_name = '{assoc}'"
            )
        else:
            API_URL = (
                f"https://grizli-cutout.herokuapp.com/assoc_json?name={assoc}"
            )

            with urllib.request.urlopen(API_URL) as url:
                data = json.loads(url.read().decode())
            for k in data:
                data[k] = [data[k][i] for i in data[k]]

            assoc_data = utils.GTable(data)

        print(f"{assoc}  N={len(assoc_data)}")

        ra = assoc_data["ra"].mean()
        dec = assoc_data["dec"].mean()

        mjd_range = [assoc_data["t_min"].min(), assoc_data["t_max"].max()]

        prefix = assoc

        if assoc_data["instrument_name"][0].upper() in [
            "NIRCAM", "MIRI", "NIRISS", "NIRSPEC"
        ]:
            BODY_CODE = JWST_BODY_CODE
            MPC_CODE = JWST_MPC
        else:
            BODY_CODE = HST_BODY_CODE
            MPC_CODE = HST_MPC

    else:
        BODY_CODE = SPACECRAFT_BODY_CODE
        MPC_CODE = SPACECRAFT_MPC

        ra, dec = coords
        make_plot = False

    if ephem_steps < 0:
        nsteps = int(
            np.maximum(
                4,
                (mjd_range[1] - mjd_range[0]) * 86400 / np.abs(ephem_steps)
            )
        )
    else:
        nsteps = ephem_steps

    epoch = astropy.time.Time(np.mean(mjd_range), format="mjd")  # .iso

    coo = SkyCoord(ra, dec, unit="deg")
    coo_str = (
        coo.to_string(style="hmsdms", precision=2, sep=":")
        .replace("-", "M")
        .replace(":", "-")
    )

    coo_str = coo_str.replace("+", "")

    clean_str = coo.to_string(
        style="hmsdms", precision=2, sep=":"
    ).replace(":", " ")

    sb_json = f"{prefix}_sb.json"

    if os.path.exists(sb_json):
        print(f"Load from {sb_json}.  Epoch: {epoch.iso}  Coords: {clean_str}")
        with open(sb_json) as fp:
            response = json.load(fp)

    else:

        ########
        # Get JWST geocentric coords from horizon
        print(f"Get JWST xyz position at {epoch.iso}")

        pos_url = (
            f"https://ssd.jpl.nasa.gov/api/horizons.api?format=json"
            f"&COMMAND='{BODY_CODE}'"
            "&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='VECTORS'"
            "&OUT_UNITS='KM-S'"
            f"&CENTER='500'"
            f"&START_TIME='{epoch.iso}'"
            f"&STOP_TIME='{(epoch+3 * u.minute).iso}'&STEP_SIZE='1%20m'"
        ).replace(" ", "%20")

        with urllib.request.urlopen(pos_url) as url:
            response = json.loads(url.read().decode())

        with open(f"{prefix}_jwst_xyz.json", "w") as fp:
            json.dump(response, fp)

        rows = response["result"].split("\n")
        for i, row in enumerate(rows):
            if "X =" in row:
                break

        xyz = rows[i].replace("=", " ").split()[1::2]
        v_xyz = rows[i + 1].replace("=", " ").split()[1::2]

        #########
        # Query for small bodies
        radius = query_arcmin / 60

        url = (
            f"https://ssd-api.jpl.nasa.gov/sb_ident.api?"  # sb-kind=a"
            + "xobs="
            + ",".join([f"{float(x):.16f}" for x in xyz])
            + ","
            + ",".join([f"{float(x):.16f}" for x in v_xyz])
            # "&two-pass=true&suppress-first-pass=true"  # &req-elem=false"
            + sb_ident_params
            + f"&obs-time={epoch.iso.replace(' ', '_').split('.')[0]}"
            + f"&fov-ra-center={coo_str.split()[0]}"
            + f"&fov-ra-hwidth={radius:.3f}"
            + f"&fov-dec-center={coo_str.split()[1]}"
            + f"&fov-dec-hwidth={radius:.3f}"
        )

        # else:
        print(f"Query sb_ident: {clean_str}  at  {epoch.iso}")
        print(url)

        with urllib.request.urlopen(url) as url:
            response = json.loads(url.read().decode())

        with open(sb_json, "w") as fp:
            json.dump(response, fp)

    #########
    # Get ephemerides
    start_time = astropy.time.Time(mjd_range[0], format="mjd")
    stop_time = astropy.time.Time(mjd_range[1], format="mjd")

    ephem = {}
    fephem = {}

    if "n_second_pass" in response:
        print(f"N = {response['n_second_pass']} objects found")

        for row in response["data_second_pass"]:
            body_name = row[0]
            if "(" in body_name:
                key = body_name.split("(")[1][:-1]
            else:
                key = body_name

            key_str = body_name.replace(" ", "_")

            track_file = f"{prefix}_sb_{key_str}.csv"
            if os.path.exists(track_file):
                print(f"Read {track_file}")
                fephem[body_name] = utils.read_catalog(track_file)
                continue

            print(body_name)

            pos_url = (
                f"https://ssd.jpl.nasa.gov/api/horizons.api?format=json"
                f"&COMMAND='{key.replace('/','%2F')}'"
                "&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='OBSERVER'"
                "&OUT_UNITS='KM-S'"
                "&ANG_FORMAT='DEG'&CAL_FORMAT=JD"
                f"&CENTER='{MPC_CODE}'"
                f"&START_TIME='{start_time.iso}'"
                f"&STOP_TIME='{stop_time.iso}'&STEP_SIZE='{nsteps}'"
            ).replace(" ", "%20")

            print(pos_url)

            with urllib.request.urlopen(pos_url) as url:
                ephem[body_name] = json.loads(url.read().decode())

            eph = ephem[body_name]["result"]
            ind = []
            first = last = None

            for i, row in enumerate(eph.split("\n")):
                if "$SOE" in row:
                    first = i + 1
                elif "$EOE" in row:
                    last = i

            if (first is None) | (last is None):
                # Problem with ephemeris
                print(" ephemeris problem!")
                continue

            data = []
            for row in eph.split("\n")[first:last]:
                spl = row.split()
                mjd = astropy.time.Time(float(spl[0]), format="jd").mjd
                data.append([mjd, float(spl[1]), float(spl[2])])

            fephem[body_name] = astropy.table.Table(
                rows=data, names=["mjd", "ra", "dec"]
            )

            fephem[body_name].meta["name"] = body_name

            fephem[body_name].write(track_file, overwrite=True)

    #############
    # Make a figure

    if make_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter([ra], [dec], alpha=0.0)

        footprints = []
        for row in assoc_data:
            sr = utils.SRegion(row["footprint"])
            sr.add_patch_to_axis(ax, fc="tomato", alpha=0.1, ec="tomato")
            footprints.append(sr)

        keep = []
        for key in fephem:
            coo = np.array([fephem[key]["ra"], fephem[key]["dec"]]).T
            in_fp = False
            for sr in footprints:
                in_fp |= sr.path[0].contains_points(coo).sum() > 0

            if in_fp | (plot_all_tracks):
                keep.append(key)
                ax.plot(
                    fephem[key]["ra"], fephem[key]["dec"], label=key, alpha=0.5
                )

        leg = ax.legend()
        leg.set_title(assoc)
        ax.set_xlim(*ax.get_xlim()[::-1])
        ax.set_aspect(1.0 / np.cos(dec / 180 * np.pi))

        fig.savefig(f"{prefix}_sb.png")

    else:
        fig = None
        keep = None

    return fig, fephem, keep


def get_sb_mask(
    assoc,
    mask_size={"MIRI": '3"', "NIRCAM": '0.5"'},
    region_colors={"MIRI": "pink", "NIRCAM": "lightblue"},
    parallels=True,
    t_pad=90.0,
):
    """
    Calculate an image mask based on the small body ephemerides

    Parameters
    ----------
    assoc : str
        Association name, already processed with
        `~grizli.horizons.query_horizons_small_bodies`

    mask_size : dict
        Mask size along the ephemeris track

    region_colors : dict
        Colors applied to regions in the DS9 output

    parallels : bool
        Extend query to additional associations with the same ``obs_id`` as the
        exposures in ``assoc``

    t_pad : float
        Time in seconds to pad to the beginning and end of the exposure metadata
        when calculating the ephemeris overlap

    Returns
    -------
    assoc_data : table
        Exposure metadata from the db query

    mask : table
        Table of polygon masks, if overlaps found
    """
    import time

    files = glob.glob(f"{assoc}*_sb_*).csv")
    files.sort()
    if len(files) == 0:
        print(f"#  {assoc}  {0}  {len(files)} !")
        pass

    assoc_data = db.SQL(f"""
    SELECT dataset, extension, sciext, assoc, instrume,
           filter, footprint, expstart, exptime
    FROM exposure_files WHERE assoc = '{assoc}'
    ORDER BY expstart
    """)

    if parallels:
        # pad = parallel_pad / 60 / 24.0  # minutes -> days
        #
        # t_min = assoc_data["expstart"].min() - pad
        # t_max = (assoc_data["expstart"] + assoc_data["exptime"] / 86400.0).min() + pad
        #
        # assoc_data = db.SQL(
        #     f"""
        # select dataset, extension, sciext, assoc, instrume, filter, footprint, expstart, exptime from exposure_files
        # where expstart < {t_max} AND (expstart + exptime / 86400) > {t_min}
        #     """
        # )
        obset = np.unique([d.split("_")[0] for d in assoc_data["dataset"]])
        obset_query = []
        for o in obset:
            obset_query.append(f"dataset LIKE '{o}%%'")

        assoc_data = db.SQL(f"""
        SELECT dataset, extension, sciext, assoc, instrume,
               filter, footprint, expstart, exptime
        FROM exposure_files
        where {' OR '.join(obset_query)}
        ORDER BY expstart
        """)

        print(
            f"Get parallels: {' '.join(obset.tolist())}, "
            + f" {len(np.unique(assoc_data['assoc']))} associations"
        )

    footprints = [utils.SRegion(fp) for fp in assoc_data["footprint"]]

    tabs = [utils.read_catalog(file) for file in files]
    keys = [file.split("_sb_")[1].split(".csv")[0] for file in files]

    coo = [np.array([tab["ra"], tab["dec"]]).T for tab in tabs]

    mask_rows = []

    masks = []

    print(f"#  {assoc}  {len(assoc_data)}  {len(files)}")

    SDAY = 86400
    CIRCLE = "CIRCLE({0},{1},{2})"

    for i, sr in enumerate(footprints):
        path = sr.path[0]
        row = assoc_data[i]

        for j, tab in enumerate(tabs):
            time_range = tab["mjd"] > (row["expstart"] - t_pad / SDAY)
            time_range &= tab["mjd"] < (
                row["expstart"] + (row["exptime"] + t_pad) / SDAY
            )
            in_foot = path.contains_points(coo[j][time_range, :])
            if in_foot.sum() > 0:
                # print(keys[j])
                in_coords = coo[j][time_range, :][in_foot, :]
                print(
                    f"   {len(mask_rows)+1}  {row['dataset']}  "
                    + f"{keys[j]}   {in_foot.sum()}"
                )

                for k, ck in enumerate(in_coords):
                    srk = utils.SRegion(
                        CIRCLE.format(*ck, mask_size[row["instrume"]]),
                        ncircle=12,
                    )

                    if k == 0:
                        reg = srk.shapely[0]
                    else:
                        reg = reg.union(srk.shapely[0])

                row_j = dict(row)
                row_j["key"] = keys[j]
                row_j["mask"] = utils.SRegion(reg).polystr(precision=5)[0]

                mask_rows.append(row_j)

    if len(mask_rows) > 0:
        mask = utils.GTable(rows=mask_rows)
        mask.write(f"{assoc}_sb_mask.csv", overwrite=True)
        labels = []
        with open(f"{assoc}_sb_mask.reg", "w") as fp:
            fp.write("icrs\n")
            for k, row in enumerate(mask):
                sr = utils.SRegion(row["mask"])
                color = region_colors[row["instrume"]]
                if row["key"] not in labels:
                    label = "text={{{dataset} {key}}}".format(**row)
                else:
                    label = ""  # " {dataset} {key} ".format(**row)

                fp.write(sr.region[0] + f"# color={color} {label}\n")

                labels.append(row["key"])
    else:
        with open(f"{assoc}_sb_mask.empty", "w") as fp:
            fp.write(time.ctime())

        mask = None

    return assoc_data, mask


def run_all(
    coords=(150.0763324, 2.3321910),
    filters=["F560W", "F770W", "F1000W", "F1280W", "F1500W", "F1800W", "F2100W"],
    instruments=["MIRI"],
    assoc_name=None,
    order_by="RANDOM()",
    query_only=False,
    **kwargs,
):
    """
    Run `grizli.horizons.query_horizons_small_bodies` on all associations
    found in list from a query to the ``assoc_table`` db table

    Parameters
    ----------
    coords, filters, instruments : list
        (ra, dec) coordinates and lists of filters and instruments for the
        ``assoc_table`` query

    assoc_name : None, str
        Explicit single association to process

    query_only : bool
        Just run the association query, don't run the SB Ident tool

    Returns
    -------
    None : outputs are region and table files
    """
    if assoc_name is not None:
        query_selection = f"assoc = '{assoc_name}'"
    else:
        query_selection = f"""instrume in ({','.join(db.quoted_strings(instruments))})
    AND filter in ({','.join(db.quoted_strings(filters))})
    AND (
        polygon(circle(point({coords[0]}, {coords[1]}), 0.5))
        && polygon(footprint)
    )"""

    QUERY = f"""
    SELECT assoc as assoc_name, filter FROM exposure_files
    WHERE
    {query_selection}
    GROUP BY assoc, filter
    ORDER BY {order_by}
    """

    print(QUERY)

    miri_assoc = db.SQL(QUERY)

    print(len(miri_assoc), " associations")

    if query_only:
        return miri_assoc

    for i, assoc in enumerate(miri_assoc["assoc_name"]):
        print(f"{i} {assoc}")
        sb_json = f"{assoc}_sb.json"
        if os.path.exists(sb_json):
            print(f"  found {sb_json}")
            continue

        plt.close("all")
        res = query_horizons_small_bodies(assoc)
        print("\n")


def demo():
    """
    Generate a demonstration figure
    """
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    from grizli import horizons, utils

    assoc = "j100028p0218_cosmos-1-f1800w_00157"

    sb = horizons.query_horizons_small_bodies(
        assoc=assoc,
        coords=None,
        mjd_range=None,
        prefix="user",
        with_db=True,
        sb_ident_params="&two-pass=true&suppress-first-pass=true",
        query_arcmin=15,
        ephem_steps=256,
        make_plot=False,
        plot_all_tracks=False
    )

    msk = horizons.get_sb_mask(
        assoc,
        mask_size={"MIRI": '3"', "NIRCAM": '0.5"'},
        region_colors={"MIRI": "pink", "NIRCAM": "lightblue"},
        parallels=True,
        t_pad=90.0,
    )

    with_sb = np.isin(msk[0]['dataset'], msk[1]['dataset'])
    row = msk[0][with_sb][0]
    
    assoc_mosaic = (
    "https://s3.amazonaws.com/grizli-v2/assoc_mosaic/v7.0/"
        + "{assoc}-{filter}_drc_sci.fits.gz".format(**row)
    ).lower()
    
    if row['instrume'] == 'MIRI':
        assoc_mosaic = assoc_mosaic.replace('_drc', '_drz')
        
    with pyfits.open(assoc_mosaic) as im:
        data = im[0].data * 1
        wcs = pywcs.WCS(im[0].header)

    rms = utils.nmad(data[(data != 0) & np.isfinite(data)])

    sh = data.shape

    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * sh[0] / sh[1]))

    ax.imshow(
        data / rms * np.nan**(data == 0),
        vmin=-2, vmax=10,
        cmap='bone_r',
    )

    ax.axis('off')

    colors = {}

    for row in msk[1]:
        sr = utils.SRegion(row['mask'])
        xy = wcs.all_world2pix(*sr.xy[0].T, 0)
        key = row['key']
        if key in colors:
            label = None
            color = colors[key]
        else:
            label = key.replace('_', ' ')
            color = None

        pl = ax.plot(*xy, color=color, label=label)
        if key not in colors:
            colors[key] = pl[0].get_color()

    leg = ax.legend(loc='upper right', fontsize=9)
    ax.text(
        0.05, 0.98, assoc, ha='left', va='top',
        transform=ax.transAxes, fontsize=10
    )

    fig.tight_layout(pad=1)

    fig.savefig(f"{assoc}_small_bodies.png")

if __name__ == "__main__":

    import sys
    import yaml

    kwargs = {}

    for arg in sys.argv:
        if "=" in arg:
            key = arg.split("=")[0].replace("-", "")
            val = arg.split("=")[1]
            if val.title() in ["True", "False"]:
                kwargs[key] = val.title() in ["True"]
            elif key == "coords":
                kwargs["coords"] = np.array(val.split(",")).astype(float)
            elif key in ["filters", "instruments"]:
                kwargs[key] = val.upper().split(",")
            elif key in ["assoc_name", "order_by"]:
                kwargs[key] = val

    print("kwargs\n======\n" + yaml.dump(kwargs).strip())

    run_all(**kwargs)
