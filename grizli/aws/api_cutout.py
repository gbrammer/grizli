"""
Code from API for making PNG and FITS cutouts
"""

def parse_coords(coo):
    """
    Parse `?coords={ra},{dec}` input
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    cc = coo.strip()
    if ',' in cc:
        cc = cc.split(',')
    else:
        cc = cc.split(' ')
    
    if ':' in cc[0]:
        crd = SkyCoord(' '.join(cc), unit=('hour','deg'))
        ra, dec = crd.ra.degree, crd.dec.degree
    else:
        ra, dec = float(cc[0]), float(cc[1])
    
    return ra, dec

def get_hashroot():
    """
    Generate a random hash for temporary filenames
    """
    import secrets

    hash_key = secrets.token_urlsafe(16)[:8]
    return hash_key.lower().replace("-", "x")


VALID_FILTERS = [
    "g141",
    "f127m",
    "f139m",
    "f336w",
    "f140w",
    "f660n",
    "f814w",
    "g102",
    "f105w",
    "f160w",
    "f200lp",
    "f475w",
    "f775w",
    "f110w",
    "f153m",
    "f606w",
    "f438w",
    "f435w",
    "f125w",
    "f350lp",
    "f850lp",
    "f098m",
    "f658n",
    "f845m",
    "f390w",
    "f275w",
    "f763m",
    "f555w",
    "f070w-clear",
    "f090w-clear",
    "clear-f090w",
    "f115w-clear",
    "clear-f115w",
    "f150w-clear",
    "clear-f150w",
    "f200w-clear",
    "clear-f200w",
    "f277w-clear",
    "f356w-clear",
    "f410m-clear",
    "f444w-clear",
    "f150w2-clear",
    "f322w2-clear",
    "f150w2-f162m",
    "f140m-clear",
    "f182m-clear",
    "f210m-clear",
    "f187n-clear",
    "f150w2-f164n",
    "f250m-clear",
    "f300m-clear",
    "f335m-clear",
    "f360m-clear",
    "f430m-clear",
    "f460m-clear",
    "f480m-clear",
    "f560w",
    "f770w",
    "f1000w",
    "f1280w",
    "f090wn-clear",
    "f115wn-clear",
    "f150wn-clear",
    "f200wn-clear",
    "f140mn-clear",
    "f158mn-clear",
    "clearp-f277w",
    "clearp-f356w",
    "clearp-f444w",
    "clearp-f430m",
    "clearp-f480m",
    "f356w-grismr",
    "f444w-grismr",
    "f444w-f405n",
    "f444w-f466n",
    "f444w-f470n",
    "f1500w",
    "f1800w",
    "f2100w",
]


def get_thumb(output="png", remove_tiles=True, verbose=False, **kws):
    """
    Get thumbnail cutout.  This is the same function that is used behind the web-based
    cutout API, so all of the parameters are implemented here.
    """
    import os
    import glob

    import numpy as np
    from skimage.io import imsave, imread

    import matplotlib.pyplot as plt

    if False:
        plt.rcParams['backend'] = 'Agg'
        plt.ioff()

    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs

    from grizli.aws import tile_mosaic, db
    from grizli.aws.tile_mosaic import cutout_from_coords, coords_to_subtile
    from grizli.pipeline import auto_script
    from grizli import utils

    CUTOUT_PREFIX = "cutout"
    
    kwargs = dict(
        ra=189.0243001,
        dec=62.1966953,
        size=10,
        scl=1,
        scale_ab=None,
        rgb_min=-0.01,
        invert=False,
        rgb_scl=[1, 1, 1],
        pl=0,
        pf=1,
        all_filters=False,
        pad=2,
        nirspec=False,
        nrs_source="magenta",
        nrs_other="pink",
        msa_other="lightblue",
        nrs_lw=1,
        nrs_alpha=0.5,
        dpi_scale=1,
    )

    for k in kwargs:
        if k in kws:
            if hasattr(kws[k], "lower"):
                if kws[k].lower() in ["false", "true"]:
                    kwargs[k] = kws[k].lower() == "true"
                elif "," in kws[k]:
                    kwargs[k] = np.array(kws[k].split(",")).astype(float).tolist()
                else:
                    try:
                        kwargs[k] = float(kws[k])
                    except ValueError:
                        kwargs[k] = kws[k]
            else:
                try:
                    kwargs[k] = float(kws[k])
                except ValueError:
                    kwargs[k] = kws[k]

    for k in ("coords", "coord"):
        if k in kws:
            coo = kws[k]
            kwargs["ra"], kwargs["dec"] = parse_coords(coo)
            break

    if "filters" in kws:
        fk = kws["filters"]
        kwargs["filters"] = []
        for f in fk.split(","):
            fsp = f.strip().lower()
            if fsp in VALID_FILTERS:
                kwargs["filters"].append(fsp)
    else:
        kwargs["filters"] = ["f814w", "f125w", "f160w"]

    kwargs["pad"] = int(kwargs["pad"])

    if "asinh" in kws:
        if kws["asinh"]:
            xkws = dict(  # filters=['f444w-clear','f277w-clear','f115w-clear'],
                rgb_scl=[2.0, 0.8, 1.0],
                norm_kwargs={
                    "stretch": "asinh",
                    "min_cut": -0.01,
                    "max_cut": 1.0,
                    "clip": True,
                    "asinh_a": 0.03,
                },
                pl=1.5,
                pf=1,
            )

            for k in xkws:
                if k not in kws:
                    kwargs[k] = xkws[k]
        else:
            kwargs["norm_kwargs"] = None
    else:
        kwargs["norm_kwargs"] = None

    if "slit" in kws:
        slit = kws["slit"]
        slit = np.array(slit.split(",")).astype(float)
        # print('Slit: ', slit)
        if len(slit) != 5:
            slit = None
    else:
        slit = None

    utils.log_comment(utils.LOGFILE, f"args: {kwargs}", verbose=verbose)
    
    hash = get_hashroot()
    base = f"{CUTOUT_PREFIX}-{hash}"
    resp = cutout_from_coords(
        output=base + "-{tile}-{filter}_{drz}",
        make_weight=(output == "fits_weight"),
        clean_subtiles=False,
        **kwargs,
    )

    utils.log_comment(utils.LOGFILE, f"cutout_from_coords: {resp}", verbose=verbose)
    
    rfix = resp[0][0].replace("f444w-f466n", "f466n-clear")
    rfix = rfix.replace("f150w2-f162m", "f162m-clear")
    rfix = rfix.replace("f150w2-f164n", "f164n-clear")
    rfix = rfix.replace("-clear", "clear").replace("clearp-", "clear")
    root = "-".join(rfix.split("-")[:-1])

    nresp = len(resp)
    fix_resp = []
    for r in resp:
        rfix = r[0].replace("f444w-f466n", "f466n-clear")
        rfix = rfix.replace("f150w2-f162m", "f162m-clear")
        rfix = rfix.replace("f150w2-f164n", "f164n-clear")
        if r[0] != rfix:
            utils.log_comment(
                utils.LOGFILE,
                f"Rename: {r[0]} to {rfix}",
                verbose=verbose
            )
            os.system(f"mv {r[0]} {rfix}")
            
            fix_resp.append((rfix, r[1]))
            if "fits" in rfix:
                with pyfits.open(rfix, mode="update") as im:
                    fi = im[0].header["FILTER"]
                    if "CLEAR" not in fi:
                        im[0].header["FILTER"] = fi.split("-")[1] + "-CLEAR"
                    im.flush()
        elif "clearp" in r[0]:
            rfix = r[0].replace("clearp-", "").replace("_dr", "-clearp_dr")
            
            utils.log_comment(
                utils.LOGFILE,
                f"Rename: {r[0]} to {rfix}",
                verbose=verbose
            )
            os.system(f"mv {r[0]} {rfix}")
            
            fix_resp.append((rfix, r[1]))
            if "fits" in rfix:
                with pyfits.open(rfix, mode="update") as im:
                    fi = im[0].header["FILTER"]
                    if fi.startswith("CLEARP"):
                        fi = fi[7:] + "-CLEARP"
                        im[0].header["FILTER"] = fi
                    im.flush()

        else:
            fix_resp.append(r)

    resp = fix_resp
    for j, f in enumerate(kwargs["filters"]):
        if f.startswith("clearp-"):
            msg = f"Rename filter {kwargs['filters'][j]} to {f.replace('clearp-','')}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)

            kwargs["filters"][j] = f.replace("clearp-", "") + "-clearp"
        else:
            ffix = f.lower().replace("f444w-f466n", "f466n-clear")
            ffix = ffix.replace("f150w2-f162m", "f162m-clear")
            ffix = ffix.replace("f150w2-f164n", "f164n-clear")

            if ffix != f.lower():
                msg = f"Rename filter {kwargs['filters'][j]} to {ffix}"
                utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
                kwargs["filters"][j] = ffix

    if remove_tiles:
        _tile = resp[0][0].split("-")[2]
        tile_files = glob.glob(f"tile.*{_tile}.*fits")
        tile_files.sort()
        for _file in tile_files:
            utils.log_comment(utils.LOGFILE, f"rm {_file}", verbose=verbose)
            os.remove(_file)

    # COSMOS ACS tiles off by one pixel as of Apr. 2023
    for ext in ["sci", "wht"]:
        _file = f"{base}-1392-f814w_drc_{ext}.fits"
        if os.path.exists(_file):
            utils.log_comment(utils.LOGFILE, "Roll ACS by one pixel", verbose=verbose)
            
            with pyfits.open(_file, mode="update") as im:
                im[0].data = np.roll(np.roll(im[0].data, 1, axis=0), 1, axis=1)
                im.flush()

    if output.startswith("fits"):
        in_hdu = [None]
        hdul = None
        xpf = None

        for r in resp:
            for e in r:
                if e not in in_hdu:
                    im = pyfits.open(e)
                    if im[0].data.max() == 0:
                        continue

                    if hdul is None:
                        fobj = pyfits.PrimaryHDU
                    else:
                        fobj = pyfits.ImageHDU

                    in_hdu.append(e)
                    im[0].header["EXTNAME"] = im[0].header["FILTER"]
                    if "_wht" in e:
                        im[0].header["EXTVER"] = "WHT"
                    else:
                        im[0].header["EXTVER"] = "SCI"

                    # Cutout slices
                    sh = im[0].data.shape

                    imw = pywcs.WCS(im[0].header)
                    pscl = np.abs(im[0].header["CD1_1"] * 3600)

                    if xpf is None:
                        xp0 = np.squeeze(
                            imw.all_world2pix([kwargs["ra"]], [kwargs["dec"]], 0)
                        )
                        xpf = xp0 * 1.0
                        if pscl < 0.06:
                            xpf /= 2.0  # (np.array(xpf))/2.

                        # xp = np.round(xpf+1).flatten().astype(int) * 2
                        xp = np.round(xpf * 2).astype(int)

                    nn = int(np.floor(kwargs["size"] / 0.05))

                    if pscl > 0.06:
                        slx = slice((xp[0] - nn) // 2, (xp[0] + nn) // 2)
                        sly = slice((xp[1] - nn) // 2, (xp[1] + nn) // 2)
                    else:
                        slx = slice((xp[0] - nn), (xp[0] + nn))
                        sly = slice((xp[1] - nn), (xp[1] + nn))

                    slh = utils.get_wcs_slice_header(imw, slx, sly)
                    hdui = fobj(data=im[0].data[sly, slx], header=im[0].header)

                    for k in slh:
                        hdui.header[k] = slh[k]

                    if hdul is None:
                        hdul = pyfits.HDUList([hdui])
                    else:
                        hdul.append(hdui)

        if hdul is None:
            msg = "No cutout found for ra={0}, dec={1}, filters={2}"
            return msg.format(kwargs["ra"], kwargs["dec"], kwargs["filters"])
            # return find_overlapping_exposures(request)

        retfile = root + ".fits"

        for r in resp:
            for e in r:
                if e is not None:
                    utils.log_comment(utils.LOGFILE, f"rm {e}", verbose=verbose)
                    os.remove(e)

        return hdul

    # remove empty
    keep = []
    smax = 0
    pscl = 0

    for r in resp:
        for e in r:
            if e is not None:
                im = pyfits.open(e)
                if im[0].data.max() == 0:
                    os.remove(e)
                else:
                    sh = im[-0].data.shape
                    if sh[0] > smax:
                        smax = sh[0]
                        pscl = np.abs(im[0].header["CD1_1"] * 3600)

                    keep.append(e)

    if len(keep) == 0:
        msg = "No cutout found for ra={0}, dec={1}, filters={2}"
        return msg.format(kwargs["ra"], kwargs["dec"], kwargs["filters"])

    rgb_kws = dict(
        root=root,
        HOME_PATH=None,
        scl=kwargs["scl"],
        add_labels=False,
        output_dpi=72,
        get_rgb_array=True,
        show_ir=False,
        invert=kwargs["invert"],
        filters=kwargs["filters"],
        scale_ab=kwargs["scale_ab"],
        rgb_min=kwargs["rgb_min"],
        norm_kwargs=kwargs["norm_kwargs"],
        rgb_scl=kwargs["rgb_scl"],
        pl=kwargs["pl"],
        pf=kwargs["pf"],
        xsize=6,
        tick_interval=25.6,
    )

    _rgb = auto_script.field_rgb(**rgb_kws)

    if _rgb.shape[0] == smax // 2:
        rgb = np.zeros((_rgb.shape[0] * 2, _rgb.shape[1] * 2, 3), dtype=_rgb.dtype)
        for i in [0, 1]:
            for j in [0, 1]:
                rgb[i::2, j::2, :] = _rgb
    else:
        rgb = _rgb

    rsh = rgb.shape
    # Center on target position
    im = pyfits.open(keep[-1])
    sh = im[0].data.shape

    imw = pywcs.WCS(im[0].header)
    xpf = np.array(imw.all_world2pix([kwargs["ra"]], [kwargs["dec"]], 0))
    xp = np.round(xpf).flatten().astype(int)
    nn = int(np.floor(kwargs["size"] / pscl * sh[0] / smax))

    slx = slice((xp[0] - nn) * rsh[0] // sh[0], (xp[0] + nn) * rsh[0] // sh[0] + 1)
    sly = slice((xp[1] - nn) * rsh[0] // sh[0], (xp[1] + nn) * rsh[0] // sh[0] + 1)
    # print(xp, nn, pscl, smax, rgb.shape, slx, sly)

    pfile = f"{root}.rgb.png"

    # Draw slit?
    if slit is not None:

        dpi = nn * smax // sh[0]
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.axis("off")
        ax.imshow(rgb, origin="lower")
        dpi *= kwargs["dpi_scale"]

        pscale = pscl * sh[0] / smax  # utils.get_wcs_pscale(imw)
        sx = np.array([-0.5, -0.5, 0.5, 0.5, -0.5]) * slit[3] / pscale
        sy = np.array([-0.5, 0.5, 0.5, -0.5, -0.5]) * slit[4] / pscale
        theta = -slit[2] / 180 * np.pi
        _mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        srot = np.array([sx, sy]).T.dot(_mat)
        srot += np.squeeze(imw.all_world2pix([slit[0]], [slit[1]], 0))
        ax.plot(*srot.T, color="w", alpha=0.8, lw=1)

        ax.set_xlim(slx.start, slx.stop - 1)
        ax.set_ylim(sly.start, sly.stop - 1)
        fig.tight_layout(pad=0)
        fig.savefig(pfile, dpi=dpi)
        plt.close("all")

    elif kwargs["nirspec"] in [True]:

        if "metafile" in kws:
            metafile = kws["metafile"]
        else:
            metafile = None

        patt_num = kws["nrs_dither"] if "nrs_dither" in kws else None
        pattern_match = "" if patt_num is None else f" AND patt_num in ({patt_num})"

        cosd = np.cos(kwargs["dec"] / 180 * np.pi)

        slits = db.SQL(
            f"""select is_source, footprint, msametfl from nirspec_slits
        WHERE valid = True {pattern_match}
        AND polygon(circle(point({kwargs['ra']},{kwargs['dec']}),{kwargs['size']/3600*1.5/cosd})) @> point(ra,dec)
        """
        )

        utils.log_comment(utils.LOGFILE, f"! Slits: {len(slits)}", verbose=verbose)

        dpi = nn * smax // sh[0]
        dpi *= kwargs["dpi_scale"]

        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.axis("off")
        ax.imshow(rgb, origin="lower")

        pscale = pscl * sh[0] / smax  # utils.get_wcs_pscale(imw)

        if len(slits) > 0:
            for row in slits:
                sr = utils.SRegion(row["footprint"])
                xsl, ysl = np.squeeze(imw.all_world2pix(*sr.xy[0].T, 0))
                xsl = np.append(xsl, xsl[0])
                ysl = np.append(ysl, ysl[0])
                if row["is_source"]:
                    ec = kwargs["nrs_source"]
                else:
                    ec = kwargs["nrs_other"]

                ls = "-"
                zorder = 101

                if metafile is not None:
                    if not row["msametfl"].startswith(metafile):
                        ec = kwargs["msa_other"]
                        # ls = '--'
                        zorder = 100

                ax.plot(
                    xsl,
                    ysl,
                    c=ec,
                    alpha=kwargs["nrs_alpha"],
                    lw=kwargs["nrs_lw"],
                    ls=ls,
                    zorder=zorder,
                )

        ax.set_xlim(slx.start, slx.stop - 1)
        ax.set_ylim(sly.start, sly.stop - 1)

        fig.tight_layout(pad=0)
        fig.savefig(pfile, dpi=dpi)
        plt.close("all")

    else:
        # Show all filters?
        rsub = rgb[sly, slx, :]
        if kwargs["all_filters"]:
            # Show individual filter cutouts
            ish = rsub.shape
            Nf = len(keep)
            out = np.zeros(
                (ish[0], (ish[1]) * (Nf + 1) + kwargs["pad"] * Nf, ish[2]),
                dtype=rsub.dtype,
            )
            Nx = ish[1]
            out[:, -Nx:, :] += rsub

            for ix, e in enumerate(keep):
                with pyfits.open(e) as im:
                    filters_i = [im[0].header["FILTER"].lower()]

                _ximg = auto_script.field_rgb(
                    root=root,
                    HOME_PATH=None,
                    scl=kwargs["scl"],
                    add_labels=False,
                    output_dpi=72,
                    get_rgb_array=True,
                    show_ir=False,
                    invert=True,
                    filters=filters_i,
                    scale_ab=kwargs["scale_ab"],
                    rgb_min=kwargs["rgb_min"],
                    norm_kwargs=None,
                    rgb_scl=kwargs["rgb_scl"],
                    pl=kwargs["pl"],
                    pf=kwargs["pf"],
                    xsize=6,
                    tick_interval=25.6,
                )

                ish = _ximg.shape
                if ish[0] == rsh[0] * 2:
                    # print("double", ish, rsh)
                    _img = np.zeros(rsh)
                    for i in [0, 1]:
                        for j in [0, 1]:
                            _img += _ximg[i::2, j::2, :]

                    _img = (_img / 4.0).astype(_ximg.dtype)

                elif ish[0] == rsh[0] // 2:
                    # print("half", ish, rsh)
                    _img = np.zeros(rgb.shape, dtype=_ximg.dtype)
                    for i in [0, 1]:
                        for j in [0, 1]:
                            _img[i::2, j::2, :] = _ximg
                else:
                    _img = _ximg

                utils.log_comment(
                    utils.LOGFILE,
                    f"all_filters: {filters_i}",
                    verbose=verbose
                )
                
                _img = _img[sly, slx, :]

                if _img.dtype != rgb.dtype:
                    _img = _img * 1.0 / _img.max()

                ileft = ix * (Nx + kwargs["pad"])
                out[:, ileft : ileft + Nx, :] += _img

            rsub = out

        imsave(
            pfile,
            np.clip(np.round(rsub[::-1, :, :] * 256), 0, 255).astype(np.uint8),
        )

    # retimg = send_file(pfile, mimetype='image/png')
    retimg = imread(pfile)

    # Cleanup
    utils.log_comment(utils.LOGFILE, f"rm {pfile}", verbose=verbose)
    
    os.remove(pfile)
    for f in keep:
        if os.path.exists(f):
            utils.log_comment(utils.LOGFILE, f"rm {f}", verbose=verbose)
            os.remove(f)

    return retimg
