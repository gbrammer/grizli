Installation instructions

Requirements
------------

**Python modules:**
astropy
scipy
matplotlib
astroquery
drizzlepac
sklearn
skimage
sewpy
peakutils
(optional: pyds9)


`Grizli` has been developed to work in the `astroconda <http://astroconda.readthedocs.io/en/latest/>`__
Python environment, which provides most of the required modules listed here, including general utilities like `numpy`, `scipy`, and `matplotlib`, as well as astronomy tools like `astropy` and specific software for dealing with space-telescope data (`stsci.tools`, `drizzlepac`, etc.).  It has been tested to run in astroconda with Python versions ``2.7.12`` and ``3.5.2``.

There are a few additional required modules not provided with `astroconda`, summarized here.  `/usr/local/share/python` is a good place to download and compile Python modules not provided automatically with `astroconda`:

    .. code:: bash

        cd /usr/local/share/python # or some other location, even /tmp/

`peakutils <http://pythonhosted.org/PeakUtils/>`__ - detecting peaks in 1D data

    .. code:: bash

        pip install peakutils

`sewpy <https://github.com/megalut/sewpy>`__ - Astropy-compatible wrapper for `SExtractor <http://www.astromatic.net/software/sextractor>`__.  This is necessary for the `grizli.prep` image pre-processing and also requires that you have a working version of SExtractor installed (i.e., `sex`), which can be its own can of worms.

    .. code:: bash

        git clone https://github.com/megalut/sewpy.git
        # For Python 3, get the fork below
        # git clone https://github.com/gbrammer/sewpy.git
        cd sewpy
        python setup.py install

`astroquery <https://astroquery.readthedocs.io>`__ - astropy affiliated package for querying astronomical databases.  This is only necessary if you want to use the tools in `grizli.prep` for astrometric alignment to the SDSS and/or WISE source catalogs.

    .. code:: bash

        pip install astroquery
        # or with anaconda
        conda install -c astropy astroquery


Build ``grizli``
----------------
``grizli`` - The main code repository.  There is an old version of `grizli` available to `pip`, but for now the code should be downloaded directly from the GitHub repository until the versioning and tagging is straightened out:

    .. code:: bash

        git clone https://github.com/gbrammer/grizli.git
        cd grizli
        python setup.py install

`Grizli` requires a few environmental variables to be set that point to directory location of configuration files.  The "`export`" lines below can be put into the ~/.bashrc setup file so that they're set automatically.

    .. code:: bash
        
        export GRIZLI="${HOME}/grizli" # or anywhere else
        export iref="${GRIZLI}/iref/"  # for WFC3 calibration files
        export jref="${GRIZLI}/jref/"  # for ACS calibration files
        
        # Make the directories, assuming they don't already exist
        mkdir $GRIZLI
        mkdir $GRIZLI/CONF      # needed for grism configuration files
        mkdir $GRIZLI/templates # for redshift fits
        
        mkdir $iref
        mkdir $jref

There are configuration and reference files not provided with the code repository that must be downloaded.  Helper scripts are provided to download files that are currently hard-coded:
    
    .. code:: python
    
        >>> import grizli
        >>> # set ACS=True below to get files necessary for G800L processing
        >>> grizli.utils.fetch_default_calibs(ACS=False) # to iref/iref
        >>> grizli.utils.fetch_config_files()            # to $GRIZLI/CONF
    
The grism redshift fits require galaxy SED templates that are provided with the repository but that need to be in a specific directory, `$GRIZLI/templates`.  This is done so that users can modify/add templates in that directory without touching the files in the repository itself.  For default processing they can by symlinked from the repository:

    .. code:: bash
        
        # Get installed location of grizli
        dist=`python -c "import grizli; import os; print(os.path.dirname(grizli.__file__))"`
        
        cd $GRIZLI/templates                # created above
        ln -s ${dist}/data/templates/* ./
        



