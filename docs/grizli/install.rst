Installation instructions

Python Environment
------------------

`Grizli` has been developed within a `miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_ Python environment. Module
dependencies, including general utilities like `numpy`, `scipy`, and
`matplotlib` and astronomy tools like `astropy` and specific software for
dealing with space-telescope data (`stsci.tools`, `drizzlepac`, etc.) are
installed using the provided ``requirements.txt`` file (see below). Most
development is done in a ``python 3.9`` environment on a MacBook Pro running
Mojave 10.14.6.  The basic build is tested in (Linux) python ``3.7``,
``3.8`` and ``3.9`` with the `GitHub actions <https://github.com/gbrammer/grizli/actions>`_ continuous integration (CI)
tools, but the current test suite does not yet test all of the full
functionality of the code.

Preferred installation with conda/pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The environment can be installed with ``pip`` and the `requirements.txt <https://github.com/gbrammer/grizli/blob/master/requirements.txt>`_ file, which was added in 2021 to enable the github "actions" CI testing environment (migrated from `travis-ci.org <https://travis-ci.org>`_).  The instructions below assume you have `conda` installed, e.g., with `miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_. 

.. code:: bash
    
    # Generate a conda environment named "grizli39" or anything else
    # This will just provide the base python distribution (incl. pip)
    conda create -n grizli39 python=3.9 pip numpy cython
            
    # Activate the environment.  This needs to be done each time you 
    # start a new terminal, or put it in ~/.bashrc
    conda activate grizli39

    # or some other location, even /tmp/
    cd /usr/local/share/python 

    # Fetch the grizli repo
    git clone https://github.com/gbrammer/grizli.git
    cd grizli
        
    # Compile and install the grizli module.  Only needs to be done
    # once or after updating the repository (e.g., with `git pull`).
    # "--editable" builds the cython extensions needed for pytest
    pip install --editable . -r requirements.txt
    
    # One last dependency that doesn't install with pip and is needed
    # for the WFC3/IR pipeline calwf3
    conda install hstcal
    
    # Run basic tests with pytest *after downloading config files as below*
    pip install pytest
    pytest
    
If you are planning to run simultaneous fits to grism spectra plus photometry using the `eazy-py <https://github.com/gbrammer/eazy-py>`_ connection, install `eazy-py` from the repository to ensure that you get its dependencies.

.. code:: bash

    cd /usr/local/share/python # location from above
    conda activate grizli39 # or whatever was chosen above
    
    # Fetch the eazy-py repo
    git clone --recurse-submodules https://github.com/gbrammer/eazy-py.git
    cd eazy-py
    
    # Only needs to be done once or after updating the repository.
    pip install . -r requirements.txt

    # Run basic tests with pytest
    # (pysynphot failure is not critical)
    pytest
    
Once you've built the code, proceed to `Set up directories and fetch config files`_.

Installation with conda and `environment.yml`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: 

    As of May 2021 the conda/pip installation above is favored since the CI
    tests were migrated from travis to GitHub actions, which are run on each
    push to the repository.

An `environment.yml <https://github.com/gbrammer/grizli/blob/master/environment.yml>`__ file is included with the `~grizli` distribution to 
provide an automatic way of installing the required dependencies, getting
them from `conda`, `pip`, and directly from `github` as necessary.  To use 
this file, do the following

.. code:: bash

    cd /usr/local/share/python # or some other location, even /tmp/

    # Fetch the grizli repo
    git clone https://github.com/gbrammer/grizli.git
    cd grizli
    
    # Generate a conda environment named "grizli39" or anything else
    # (Note environment_min.yml renamed to environment.yml for >0.10.0)
    conda env create -f environment.yml -n grizli39
            
    # Activate the environment.  This needs to be done each time you 
    # start a new terminal, or put it in ~/.bashrc
    conda activate grizli39
    
    # Compile and install the grizli module.  Only needs to be done
    # once or after updating the repository.
    python setup.py install 

Manual installation of dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a few additional modules that `grizli` may use but that aren't explicitly listed in the `requirements.txt <https://github.com/gbrammer/grizli/blob/master/requirements.txt>`_ file.   

**Amazon Web Services** - If you're running the full *HST* reduction pipeline with `grizli`, the code can automatically pull FITS files from the public AWS S3 bucket mirror of the archive. This requires the AWS command line tools and the `boto3` module:

    .. code:: bash

        # Put your AWS credentials, etc. in ~/.aws 
        pip install awscli
        pip install boto3    

`lacosmicx <https://github.com/cmccully/lacosmicx>`__ - Fast Python
implementation of Pieter van Dokkum's `L.A.Cosmic
<http://www.astro.yale.edu/dokkum/lacosmic/>`__ (`2001PASP..113.1420V
<http://adsabs.harvard.edu/abs/2001PASP..113.1420V>`__) software for
identifying cosmic rays in single images. The image preparation wrapper
scripts in `grizli.prep` run `lacosmicx` if a supplied list of direct or grism
images contains only a single file.

    .. code:: bash

        git clone https://github.com/cmccully/lacosmicx.git
        cd lacosmicx
        python setup.py install

.. note::
    
    The `lacosmicx` dependency was removed from `environment.yml` file
    2019.12.31 because it was breaking on OSX Mojave 10.14.6 with a
    compilation error like `unsupported option '-fopenmp'`. The workaround
    below with the Homebrew version of `gcc` may work after verifying the
    correct path to the `gcc-8` executable:
    
    .. code:: bash
        
        brew install gcc
        CC=/usr/local/Cellar/gcc/10.2.0/bin/gcc-10 pip install git+https://github.com/cmccully/lacosmicx.git


Set up directories and fetch config files
-----------------------------------------
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
    
        >>> import grizli.utils
        >>> # HST calibs to iref/iref
        >>> # set get_acs=True below to get files necessary for G800L processing
        >>> grizli.utils.fetch_default_calibs(get_acs=False)
        >>> # config files to $GRIZLI/CONF
        >>> # set get_jwst=True to get config files for jwst processing
        >>> grizli.utils.fetch_config_files(get_acs=False, get_jwst=False)
    
The grism redshift fits require galaxy SED templates that are provided with
the repository but that need to be in a specific directory,
`$GRIZLI/templates`. This is done so that users can modify/add templates in
that directory without touching the files in the repository itself. For
default processing they can by symlinked from the repository:

    .. code:: python

        >>> import grizli.utils
        >>> grizli.utils.symlink_templates(force=False)
        >>> # Set force=True to symlink files even if they already exist in 
        >>> # $GRIZLI/templates/.




