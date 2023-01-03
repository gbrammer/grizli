Installation
~~~~~~~~~~~~~~

The easiest way to install the latest ``grizli`` release into a :ref:`fresh virtualenv or conda environment <environment>` is:

.. code-block:: bash

   pip install grizli

If you are installing ``grizli`` for the first time, make sure to also set up :ref:`directories and download 
reference files <directories>`. If you will be working with HST data, the following will install all 
necessary libraries:

.. code-block:: bash

   pip install "grizli[hst]"
   conda install hstcal

If you will be working with JWST data, the following is the recommended installation process:

.. code-block:: bash

   pip install "grizli[jwst]"


More detailed instructions are available :ref:`below <installation>`.

Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^

`Grizli` has been developed within a `miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_ Python environment. Module
dependencies, including general utilities like `numpy`, `scipy`, `matplotlib`, 
astronomy tools like `astropy` and specific software for dealing with space-telescope
data (`stsci.tools`, `drizzlepac`, etc.) are all installed using the standard 
``pip install`` method (see :ref:`below <installation>`). Most development is done in a ``python 3.9``
environment on a MacBook Pro running Mojave 10.14.6.  The basic build is tested in
(Linux) python ``3.7``, ``3.8`` and ``3.9`` with the `GitHub actions
<https://github.com/gbrammer/grizli/actions>`_ continuous integration (CI) tools, but
the current test suite does not yet test all of the full functionality of the code.

.. _environment:

Setting up a Local Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend that ``grizli`` be installed in a new virtual environment to minimize conflicts
with dependencies for other packages. Possible virtual environment managers are ``conda``, ``venv``, ``pyenv``, etc.
Here we give example with setting up a ``conda`` environment. 

- Generate a ``conda`` environment named ``grizli39`` (or anything else you prefer).
  This will just provide the base Python distribution (along with ``pip``):

.. code-block:: bash

    conda create --name grizli39 python=3.9

- Activate the environment. If you chose a name other than ``grizli39``,
  substitute that below:

.. code-block:: bash

    conda activate grizli39

.. note::

   The activation needs to be done *each time* you start a new terminal. Alternatively,
   if you want it automatically done for every new terminal, you need to put the above
   command in your ``~/.bashrc`` file.

.. _installation:

Preferred installation with pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The latest release of ``grizli`` can be installed with ``pip``:

.. code-block:: bash
  
  pip install grizli
  
There are five available options for installing dependencies: ``hst``, ``jwst``, ``aws``, 
``test`` and ``docs``. These can be installed as follows:

.. code-block:: bash

  pip install "grizli[hst]"

or 

.. code-block:: bash

  pip install "grizli[jwst]"

or

.. code-block:: bash

  pip install "grizli[jwst,test]"

To minimize conflict of dependencies, install only the ones that you need. 

.. _additional:

Additional dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

``pip`` will install all needed dependencies.  If you will be working with
HST data, you will also need the ``hstcal`` library which is only available 
via ``conda``:

.. code-block:: bash

  conda install hstcal
        
``eazy-py``
###########


If you are planning to run simultaneous fits to grism spectra plus photometry using the
`eazy-py <https://github.com/gbrammer/eazy-py>`_ connection, install ``eazy-py`` 
to ensure that you get *its* dependencies and templates.

.. code-block:: bash

    git clone https://github.com/gbrammer/eazy-py.git
    cd eazy-py
    pip install .

- Download the templates (in a Python interpreter):

.. code-block:: python

    import eazy
    eazy.fetch_eazy_photoz()


- Optional: Run basic tests with ``pytest``. Note that the ``pysynphot`` failure is not critical:

.. code-block:: bash

    pytest
        
.. _directories:

Set up directories and fetch additional files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``grizli`` requires a several environmental variables to be set that point to the
directory location of configuration files. The ``export`` lines below can be put into
the ``~/.bashrc`` or ``~/.bash_profile`` setup files so that the system variables are
set automatically when you start a new terminal/shell session.

.. code-block:: bash

    export GRIZLI="${HOME}/grizli" # or anywhere else
    export iref="${GRIZLI}/iref/"  # for WFC3 calibration files
    export jref="${GRIZLI}/jref/"  # for ACS calibration files

- Create these directories, assuming that they do not already exist:

.. code-block:: bash

    mkdir $GRIZLI
    mkdir $GRIZLI/CONF      # needed for grism configuration files
    mkdir $GRIZLI/templates # for redshift fits
    mkdir $iref
    mkdir $jref

- Download the calibration and configuration files not provided with the code
  repository. Helper scripts are provided to download files that are currently
  hard-coded. HST calibrations will be downloaded to the ``$iref`` and ``$jref``
  directories. Set ``get_acs=True`` to get files necessary for G800L processing:

.. code-block:: python

    import grizli.utils
    grizli.utils.fetch_default_calibs(get_acs=False)

Configuration files will be downliaded to the ``$GRIZLI/CONF`` directory. Set 
``get_jwst=True`` to get config files for JWST processing:

.. code-block:: python

    grizli.utils.fetch_config_files(get_acs=False, get_jwst=False)

- The grism redshift fits require galaxy SED templates that are provided with the
  repository but that need to be in a specific directory, ``$GRIZLI/templates``. This is
  done so that users can modify/add templates in that directory without touching the
  files in the repository itself. For default processing they can by symlinked from the
  repository. Set ``force=True`` to symlink files even if they already exist in 
  ``$GRIZLI/templates/``:

.. code-block:: python

    import grizli.utils
    grizli.utils.symlink_templates(force=True)

- Run basic tests with `pytest`:

.. code-block:: bash

    pip install ".[test]"
    pytest

Installing ``grizli`` from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need to install ``grizli` form a specific branch or need an editable version 
of the library, you can do this directly from the repository.

- Create a dedicated environment. See instructions :ref:`above <environment>`.
- Change into a directory where the ``grizli`` repo will live. 
- Fetch the ``grizli`` repo and change into the newly cloned directory:

.. code-block:: bash

    git clone https://github.com/gbrammer/grizli.git
    cd grizli

- If you are installing from a branch, checkout the branch.
- Compile and install the ``grizli`` module. This only needs to be done once (on initial
  ``clone``), or after updating the repository (e.g., after a ``git pull``).

.. code-block:: bash

   pip install -e .
   
The ``-e`` flag stands for ``editable``. Or to install the optional dependencies:

.. code-block:: bash

   pip install -e ".[jwst,test]"


See :ref:`above <additional>` for the additional dependencies that need to be installed.

Using HST Files Staged on AWS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``grizli`` can automatically pull FITS files from the public AWS S3 bucket mirror of the
*HST* archive, which can be useful when running the full *HST* reduction pipeline. This
requires that the AWS command line tools and the ``boto3`` and ``awscli`` modules be installed:

.. code-block:: bash

    # Put your AWS credentials, etc. in ~/.aws 
    pip install grizli ".[aws]"
