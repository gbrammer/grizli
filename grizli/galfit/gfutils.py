def link_seg_products(root='j021737-051344', filter='f140w'):
    """Symlink catalog products from the automatically-generated "ir" products to filenames for a given filter.

    Parameters
    ----------
    root : str
        Rootname of the mosaic images.

    filter : str
        Filter name.

    """
    import os
    os.system('ln -s {0}-ir_seg.fits {0}-{1}_seg.fits'.format(root, filter))
    os.system('ln -s {0}-ir.cat {0}-{1}.cat'.format(root, filter))


def galfit_model_image(root='j021737-051344', ids=[738], filter='f140w', ds9=None, ds9_frame=None, comps=None, galfit_exec='galfit', seg_threshold=0.001):
    """
    Run galfit on a list of IDs and generate model-subtracted images

    Parameters
    ----------
    root : str
        Rootname of the mosaic images.  Will look for images with filenames:

        >>> sci = '{0}-{1}_drz_sci.fits'.format(root, filter)
        >>> seg = '{0}-{1}_seg.fits'.format(root, filter)
        >>> cat = '{0}-{1}.cat'.format(root, filter)

    ids : list
        List of IDs in the segmentation image / catalog to fit.

    filter : str
        Filter name.

    ds9 : `~grizli.utils.DS9`
        DS9 instance for plotting the galfit results interactively.

    ds9_frame : type
        DS9 frame to load the display

    comps : list
        List of model component names.  If none, defaults to a two-component
        model of a Sersic plus an exponential disk.  The Sersic component is
        initialized with `n=4`:

        >>> from grizli.galfit.galfit import GalfitSersic, GalfitExpdisk
        >>> comps = [GalfitSersic(n=4), galfit.GalfitExpdisk()]

    galfit_exec : str
        Path to the galfit executable.

    seg_threshold : float
        Threshold in DN (i.e., e-/s) to use for making the model segmentation
        polygons.

    Returns
    -------
    Creates the following images:

    >>> # Model-subtracted image
    >>> sub = '{0}-{1}_galfit_clean.fits'.format(root, filter)

    >>> # Model image
    >>> model = pyfits.writeto('{0}-{1}_galfit.fits'.format(root, filter)

    >>> # Segmentation image where galfit models > `seg_threshold`
    >>> gf_seg = '{0}-{1}_galfit_seg.fits'.format(root, filter)

    >>> # Segmentation image with the fitted object IDs removed
    >>> orig_seg = '{0}-{1}_galfit_orig_seg.fits'.format(root, filter)

    """
    import astropy.io.fits as pyfits
    from grizli.galfit import galfit
    from grizli.galfit.galfit import GalfitSersic, GalfitExpdisk

    if False:
        # Testing
        #reload(galfit)
        filter = 'f140w'

        root = 'sdssj1723+3411'
        ids = [449, 371, 286, 679, 393, 576, 225]

        root = 'sdssj2340+2947'
        ids = [486, 463]  # , 439]

    gf = galfit.Galfitter(root=root, filter=filter, galfit_exec=galfit_exec)  # '../Galfit/galfit')

    #comps = [GalfitSersic(n=4)]
    if comps is None:
        comps = [GalfitSersic(n=4), galfit.GalfitExpdisk()]
        #comps = [GalfitSersic(n=4, dev=True), galfit.GalfitExpdisk()]
        #comps = [GalfitSersic(n=4, dev=True), GalfitSersic(n=4)]

    # Fit galfit components
    for id in ids:
        res = gf.fit_object(id=id, radec=(None, None), size=30, get_mosaic=True, components=comps, recenter=True, gaussian_guess=False)

        if ds9:
            im = pyfits.open('{0}-{1}_galfit_{2:05d}.fits'.format(root, filter, id))
            if ds9_frame is not None:
                ds9.frame(ds9_frame)

            im = pyfits.open('{0}-{1}_galfit_{2:05d}.fits'.format(root, filter, id))
            ds9.view(gf.sci[0].data - im[0].data/gf.sci[0].header['EXPTIME'], header=gf.sci[0].header)

    ###############
    # Combine components into a single image and make new segmentation images
    full_model = gf.sci[0].data*0
    full_seg = gf.seg[0].data*0
    orig_seg = gf.seg[0].data*1

    for id in ids:
        im = pyfits.open('{0}-{1}_galfit_{2:05d}.fits'.format(root, filter, id))
        full_model += im[0].data/gf.sci[0].header['EXPTIME']
        full_seg[(im[0].data/gf.sci[0].header['EXPTIME'] > seg_threshold) & (full_seg == 0)] = id
        orig_seg[(orig_seg == id)] = 0

        if ds9:
            if ds9_frame is not None:
                ds9.frame(ds9_frame)

            ds9.view(gf.sci[0].data - full_model)

    # Model-subtracted image
    pyfits.writeto('{0}-{1}_galfit_clean.fits'.format(root, filter), data=gf.sci[0].data-full_model, header=gf.sci[0].header, clobber=True)

    # Model image
    pyfits.writeto('{0}-{1}_galfit.fits'.format(root, filter), data=full_model, header=gf.sci[0].header, clobber=True)

    # Segmentation image where galfit models > `seg_threshold`
    pyfits.writeto('{0}-{1}_galfit_seg.fits'.format(root, filter), data=full_seg, header=gf.seg[0].header, clobber=True)

    # Segmentation image with the fitted object IDs removed
    pyfits.writeto('{0}-{1}_galfit_orig_seg.fits'.format(root, filter), data=orig_seg, header=gf.seg[0].header, clobber=True)
