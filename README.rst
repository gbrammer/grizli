Grizli - *Grism redshift & line* analysis software for space-based slitless spectroscopy
========================================================================================

What is ``grizli``?
~~~~~~~~~~~~~~~~~~~

This early release of ``grizli`` is intended to demonstrate and
demystify some general techniques for manipulating *HST* slitless
spectroscopic observations, providing software kernels to address
questions such as

    "How does the WFC3/IR G141 grism disperse the spectrum of a
    star/galaxy at pixel position ``(x,y)`` in my F140W direct image?".

Much of the background related to this question in the context of the
currently available software tools was discussed in a document by
`Brammer, Pirzkal and Ryan
(2014) <https://github.com/WFC3Grism/CodeDescription/>`__. Along with a
detailed description of the format of the configuration files originally
developed for the aXe software, we provided a compact `Python
script <https://github.com/WFC3Grism/CodeDescription/blob/master/axe_disperse.py>`__
to address exactly the question above and strip away all of the many
layers of bookkeeping, file-IO, etc. in existing analysis pipelines such
as aXe (`Kummel et al.
2009 <http://adsabs.harvard.edu/abs/2009PASP..121...59K>`__) and
"THREEDHST" (`Brammer et al.
2012 <http://adsabs.harvard.edu/abs/2012ApJS..200...13B>`__, `Momcheva
et al. 2015 <http://adsabs.harvard.edu/abs/2015arXiv151002106M>`__). In
fact, that relatively simple script serves as the low-level kernel for
the way ``grizli`` computes the grism dispersion.

Eventually, ``grizli`` is intended to encourage general users to move
away from simple "data reduction" (e.g., extracting a 1D spectrum of a
single object akin to standard slit spectroscopy) and toward enabling
more quantitative and comprehensive **modeling** and fitting of slitless
spectroscopic observations, which typically involve overlapping spectra
of hundreds or thousands of objects in exposures taken with one or more
separate grisms and at multiple dispersion position angles. The products
of this type of analysis will be complete and uniform characterization
of the spectral properties (e.g., continuum shape, redshifts, line
fluxes) of all objects in a given exposure taken in the slitless
spectroscopic mode.

Installation
~~~~~~~~~~~~

``grizli`` has the following Python dependencies, with the software
versions on the development machine (MacBook Pro 2013, Mac0S 10.10.5) as
indicated:

-  Numpy (1.10.4)
-  Scipy (0.15.1) / (scipy.ndimage, 2.0)
-  (Matplotlib, 1.5.1) - generally needed for making plots, though not
   much plotting yet in ``grizli``
-  cython (0.20.1) - Needed for compiling the C-accelerated functions in
   utils\_c
-  `astropy <http://www.astropy.org/>`__ (1.1.1)
-  `scikit-learn <http://scikit-learn.org/stable/install.html>`__
   (0.14.1) - For the linear least squares solver in the emonstration
   line-fitting code.
-  `stwcs <http://stsdas.stsci.edu/stsci_python_epydoc/stwcs/index.html>`__
   (1.2.3.dev49124) - STScI packages for *HST*-specific WCS definitions.
   This may be deprecated by
   `astropy.wcs <http://docs.astropy.org/en/stable/wcs/>`__.
-  `drizzlepac <http://drizzlepac.stsci.edu/>`__ (2.1.3.dev49124) -
   STScI drizzle/blot library
-  `(photutils) <https://photutils.readthedocs.org/en/latest/>`__
   (0.2.1) - necessary if you want to create photometric catalogs and
   segmentation images directly within python
-  `(pysynphot) <http://pysynphot.readthedocs.org/en/latest/>`__
   (0.9.8.2.dev) - Not required, but can be useful for dealing with
   *HST* filters and sensitivies, as well as handling unit conversions
   for synthetic spectra

The easiest way to satisfy all of these dependencies is to work within
the STScI/Gemini "Ureka" Python distribution:
http://ssb.stsci.edu/ureka/. The current development release of
``Grizli`` has been tested in Python 2.7 within the "SSBX" Ureka
distribution, available here: http://ssb.stsci.edu/ssb\_software.shtml.

**Installation update (Apr 29, 2016):** Grizli has been tested within the newly-released `"astroconda" <http://astroconda.readthedocs.io/en/latest/>`__ environment, which should now provide the easiest way to obtain and maintain all of the external dependencies.

.. code:: bash

    ### If you have the Github SSH key enabled
    git clone git@github.com:gbrammer/grizli.git

    ### Otherwise just use https
    git clone https://github.com/gbrammer/grizli.git

    ### Build and install
    cd grizli
    python setup.py build
    python setup.py install

``grizli`` requires an environment variable (``$GRIZLI``) set to point
to a working directory where the grism trace configuration and any
additional setup files will live. For example, with the BASH shell in
``${HOME}/.bashrc``:

.. code:: bash

    export GRIZLI="${HOME}/Grizli"

``grizli`` uses the aXe configuration files to define spectral traces
and sensitivies. For *HST* WFC3/IR grism spectroscopy, these files can
be downloaded here:
http://www.stsci.edu/hst/wfc3/analysis/grism\_obs/wfc3-grism-resources.html.
**Put the downloaded configuration files in ``${GRIZLI}/CONF``**, e.g.,

.. code:: bash

    mkdir $GRIZLI/CONF
    cd $GRIZLI/CONF
    wget http://www.stsci.edu/ftp/cdbs/wfc3_aux/WFC3.IR.G102.cal.V4.3.tar.gz
    tar xzvf WFC3.IR.G102.cal.V4.3.tar.gz
    wget http://www.stsci.edu/ftp/cdbs/wfc3_aux/WFC3.IR.G141.cal.V4.3.tar.gz
    tar xzvf WFC3.IR.G141.cal.V4.3.tar.gz

Demo: working with WFC3/IR grism exposures (FLT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out the `Grizly
Demo <https://github.com/gbrammer/grizli/blob/master/docs/Grizli%20Demo.ipynb>`__
IPython notebook or
`demo.py <https://github.com/gbrammer/grizli/blob/master/docs/demo.py>`__.
