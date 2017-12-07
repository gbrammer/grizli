Installation instructions

Dependencies
------------

`Grizli` has been developed to work in the `astroconda
<http://astroconda.readthedocs.io/en/latest/>`__ Python environment, which
provides most of the required modules listed here, including general utilities
like `numpy`, `scipy`, and `matplotlib`, as well as astronomy tools like
`astropy` and specific software for dealing with space-telescope data
(`stsci.tools`, `drizzlepac`, etc.). It has been tested to run in astroconda
with Python versions ``2.7.12`` and ``3.5.2``.

There are a few additional required modules not provided with `astroconda`,
summarized here. `/usr/local/share/python` is a good place to download and
compile Python modules not provided automatically with `astroconda`:

    .. code:: bash

        cd /usr/local/share/python # or some other location, even /tmp/

`scikit-learn <http://scikit-learn.org/>`__ - Machine learning tools. This is
provided with a full anaconda/astroconda distribution but may not be supplied
with a "`miniconda <http://conda.pydata.org/miniconda.html>`__" distribution.

    .. code:: python
    
        >>> # test if it's installed
        >>> import sklearn

    .. code:: bash
    
        pip install scikit-learn
        # or with anaconda
        conda install scikit-learn
        
`peakutils <http://pythonhosted.org/PeakUtils/>`__ - detecting peaks in 1D data

    .. code:: bash

        pip install peakutils

`sewpy <https://github.com/megalut/sewpy>`__ - Astropy-compatible wrapper for
`SExtractor <http://www.astromatic.net/software/sextractor>`__. This is
necessary for the `grizli.prep` image pre-processing and also requires that
you have a working version of SExtractor installed (i.e., `sex`), which can be
its own can of worms.

    .. code:: bash

        git clone https://github.com/megalut/sewpy.git
        # For Python 3, get the fork below
        # git clone https://github.com/gbrammer/sewpy.git
        cd sewpy
        python setup.py install

`astroquery <https://astroquery.readthedocs.io>`__ - astropy affiliated
package for querying astronomical databases. This is only necessary if you
want to use the tools in `grizli.prep` for astrometric alignment to the SDSS
and/or WISE source catalogs.

    .. code:: bash

        pip install astroquery
        # or with anaconda
        conda install -c astropy astroquery

`lacosmicx <https://github.com/cmccully/lacosmicx>`__ - Fast Python
implementation of Pieter van Dokkum's `L.A.Cosmic
<http://www.astro.yale.edu/dokkum/lacosmic/>`__ (`ref
<http://adsabs.harvard.edu/abs/2001PASP..113.1420V>`__) software for
identifying cosmic rays in single images. The image preparation wrapper
scripts in `grizli.prep` run `lacosmicx` if a supplied list of direct or grism
images contains only a single file.

    .. code:: bash

        git clone https://github.com/cmccully/lacosmicx.git
        cd lacosmicx
        python setup.py install

`shapely <http://toblerity.org/shapely/manual.html>`__ - Tools for handling
geometry calculations, e.g., overlapping polygons. Currently only used by
`~grizli.utils.parse_visit_overlaps`. Installation used to be tricky to
compile the required associated `GEOS <http://trac.osgeo.org/geos/>`_ library,
but now appears to be trivial under conda:

    .. code:: bash

        conda install shapely

`hsaquery <https://github.com/gbrammer/esa-hsaquery>`__ - Python tools for 
querying exposure-level data in the 
`ESA Hubble Science Archive <http://archives.esac.esa.int/ehst/#search>`__.:

    .. code:: bash

        pip install hsaquery
        
                
Build ``grizli``
----------------
``grizli`` - The main code repository. There is an old version of `grizli`
available to `pip`, but for now the code should be downloaded directly from
the GitHub repository until the versioning and tagging is straightened out:

    .. code:: bash

        git clone https://github.com/gbrammer/grizli.git

        cd grizli
        python setup.py install

`Grizli` requires a few environmental variables to be set that point to
directory location of configuration files. The "`export`" lines below can be
put into the *~/.bashrc* or *~/.bash_profile* setup files so that the system
variables are set automatically when you start a new terminal/shell session.

    .. code:: bash
        
        # Put these lines in ~/.bashrc
        export GRIZLI="${HOME}/grizli" # or anywhere else
        export iref="${GRIZLI}/iref/"  # for WFC3 calibration files
        export jref="${GRIZLI}/jref/"  # for ACS calibration files
        
        # Make the directories, assuming they don't already exist
        mkdir $GRIZLI
        mkdir $GRIZLI/CONF      # needed for grism configuration files
        mkdir $GRIZLI/templates # for redshift fits
        
        mkdir $iref
        mkdir $jref

There are configuration and reference files not provided with the code
repository that must be downloaded. Helper scripts are provided to download
files that are currently hard-coded:
    
    .. code:: python
    
        >>> import grizli
        >>> # set ACS=True below to get files necessary for G800L processing
        >>> grizli.utils.fetch_default_calibs(ACS=False) # to iref/iref
        >>> grizli.utils.fetch_config_files()            # to $GRIZLI/CONF
    
The grism redshift fits require galaxy SED templates that are provided with
the repository but that need to be in a specific directory,
`$GRIZLI/templates`. This is done so that users can modify/add templates in
that directory without touching the files in the repository itself. For
default processing they can by symlinked from the repository:

    .. code:: bash
        
        # Get installed location of grizli
        dist=`python -c "import grizli; import os; print(os.path.dirname(grizli.__file__))"`
        
        cd $GRIZLI/templates                # created above
        ln -s ${dist}/data/templates/* ./
        



