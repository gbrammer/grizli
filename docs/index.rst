.. raw:: html

    <style media="screen" type="text/css">
      h1 {display:none;}
    </style>

*********
Grizli
*********

.. raw:: html

   <img src="_static/grizli_logo.png" onerror="this.src='_static/grizli_logo.png'; this.onerror=null;" width="495"/>
   
*Grism redshift & line* analysis software for space-based slitless spectroscopy
========================================================================================

The `grizli` source code is maintained in a repository at https://github.com/gbrammer/grizli/.  Please report any issues `there <https://github.com/gbrammer/grizli/issues>`__.

What is ``grizli``?
~~~~~~~~~~~~~~~~~~~

This early release of `grizli` is intended to demonstrate and
demystify some general techniques for manipulating *HST* slitless
spectroscopic observations, providing software kernels to address
questions such as

    "How does the WFC3/IR G141 grism disperse the spectrum of a
    star/galaxy at pixel position `(x,y)` in my F140W direct image?".

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
the way `grizli` computes the grism dispersion.

Eventually, `grizli` is intended to encourage and enable general users to move
away from simple "data reduction" (e.g., extracting a 1D spectrum of a
single object akin to standard slit spectroscopy) and toward
more quantitative and comprehensive **modeling** and fitting of slitless
spectroscopic observations, which typically involve overlapping spectra
of hundreds or thousands of objects in exposures taken with one or more
separate grisms and at multiple dispersion position angles. The products
of this type of analysis will be complete and uniform characterization
of the spectral properties (e.g., continuum shape, redshifts, line
fluxes) of all objects in a given exposure taken in the slitless
spectroscopic mode.

.. warning::
    Grizli is under active development and significant changes may be 
    made to the algorithms and code structure.  Check the `commit history <https://github.com/gbrammer/grizli/commits/master>`__ and pull the changes from the repository to make sure you're running with up-to-date bug
    fixes and feature implementation.
    
  
Installation
~~~~~~~~~~~~
      
.. toctree::
   :maxdepth: 2

   grizli/install.rst

Working Examples
~~~~~~~~~~~~~~~~~~~~~~
In lieu of detailed documentation for all of the `grizli` processing beyond information extracted from the docstrings, the following IPython/jupyter notebooks demonstrate various aspects of the code functionality.  They render statically in the GitHub pages or can be run locally after cloning and installing the software repository.

- `Grizly Demo <https://github.com/gbrammer/grizli/blob/master/examples/Grizli%20Demo.ipynb>`__: Basic interaction with WFC3/IR slitless exposures.

- `Basic-Sim <https://github.com/gbrammer/grizli/blob/master/examples/Basic-Sim.ipynb>`__ **(5.5.16)**: Basic simulations based on single WFC3/IR grism and direct exposures

- `multimission-simulation <https://github.com/gbrammer/grizli/blob/master/examples/multimission-simulation.ipynb>`__ **(5.11.16)**: 
  
  1. Demonstration of more advanced simulation techniques using deep image mosaics and external catalogs/segmentation images as reference.
  2. Provide a comparison between dispersed spectra from WFC3/G141, *JWST*/NIRISS and *WFIRST*.

- `WFC3IR_Reduction <https://github.com/gbrammer/grizli/blob/master/examples/WFC3IR_Reduction.ipynb>`__ **(9.6.16)**: End-to-end processing of WFC3/IR data.

  1. Pre-processing of files downloaded from MAST (astrometric alignment & background subtraction)
  2. Field contamination modeling
  3. Spectral extractions
  4. Redshift & emission line fits (multiple grisms)

- `NIRISS-simulation <https://github.com/gbrammer/grizli/blob/master/examples/NIRISS-simulation.ipynb>`__ **(11.11.16)**: Simulation and analysis of JWST/NIRISS observations

  1. Simulate NIRISS spectra in three blocking filters and two orients offset by 90 degrees.
  2. Simulation field taken from the Hubble WFC3/IR Ultra-Deep Field, with the WFC3 F140W image as the morphological reference and photo-z templates taken as the spectral models. 
  3. Extract spectra and fit redshifts and emission lines from the combined six exposures.

- `NewSpectrumFits <https://github.com/gbrammer/grizli/blob/master/examples/NewSpectrumFits.ipynb>`__ **(09.05.17)**: New fitting tools

  1. Unify the fitting tools between the stacked and exposure-level 2D cutouts ("beams").

- `Fit-Optimization <https://github.com/gbrammer/grizli/blob/master/examples/Fit-Optimization.ipynb>`__ **(09.14.17)**: Custom fitting

  1. Demonstrate some of the workings behind the fitting wrapper scripts by defining custom model functions with parameters to optimize.

API
~~~

.. toctree::
   :maxdepth: 2

   grizli/prep.rst
   grizli/index.rst
   changelog