
.. image:: examples/grizli_logo.png

.. image:: https://github.com/gbrammer/grizli/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/gbrammer/grizli/actions

.. image:: https://badge.fury.io/py/grizli.svg
    :target: https://badge.fury.io/py/grizli
    
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1146904.svg
   :target: https://doi.org/10.5281/zenodo.1146904

.. image:: https://readthedocs.org/projects/grizli/badge/?version=latest
   :target: https://grizli.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
   
*Grism redshift & line* analysis software for space-based slitless spectroscopy
========================================================================================

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

Installation & Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Installation instructions and documentation (in progress) can be found at http://grizli.readthedocs.io.

Working Examples
~~~~~~~~~~~~~~~~~~~~~~

The following are IPython/jupyter notebooks demonstrating various aspects of the code functionality.  They render statically in the GitHub pages or can be run locally after cloning and installing the software repository.

- `Grizli-Pipeline <https://github.com/gbrammer/grizli-notebooks/blob/main/Grizli-Pipeline.ipynb>`__ : End-to-end processing of WFC3/IR data.

  1. Query the MAST archive and automatically download files
  2. Image pre-processing  (astrometric alignment & background subtraction)
  3. Field contamination modeling
  4. Spectral extractions
  5. Redshift & emission line fits (multiple grisms)

- `Fit-with-Photometry <https://github.com/gbrammer/grizli-notebooks/blob/main/Fit-with-Photometry.ipynb>`__ : Demonstrate simultaneous fitting with grism spectra + ancillary photometry

- `NewSpectrumFits <https://github.com/gbrammer/grizli-notebooks/blob/main/NewSpectrumFits.ipynb>`__: Demonstration of the lower-level fitting tools

  1. Unify the fitting tools between the stacked and exposure-level 2D cutouts ("beams").
 
- `Fit-Optimization <https://github.com/gbrammer/grizli-notebooks/blob/main/Fit-Optimization.ipynb>`__ **(09.14.17)**: Custom fitting (hasn't been tested recently)

  1. Demonstrate some of the workings behind the fitting wrapper scripts by defining custom model functions with parameters to optimize.

The notebooks below are deprecated and haven't been tested against the master branch since perhaps late 2017.

- `Grizli Demo <https://github.com/gbrammer/grizli-notebooks/blob/main/Grizli%20Demo.ipynb>`__: Simple interaction with WFC3/IR spectra

- `Basic-Sim <https://github.com/gbrammer/grizli-notebooks/blob/main/Basic-Sim.ipynb>`__ **(5.5.16)**: Basic simulations based on single WFC3/IR grism and direct exposures

- `multimission-simulation <https://github.com/gbrammer/grizli-notebooks/blob/main/multimission-simulation.ipynb>`__ **(5.11.16)**: 
  
  1. Demonstration of more advanced simulation techniques using deep image mosaics and external catalogs/segmentation images as reference.
  2. Provide a comparison between dispersed spectra from WFC3/G141, *JWST*/NIRISS and *WFIRST*.

- `WFC3IR_Reduction <https://github.com/gbrammer/grizli-notebooks/blob/main/WFC3IR_Reduction.ipynb>`__ **(9.6.16)**: End-to-end processing of WFC3/IR data.

  1. Pre-processing of files downloaded from MAST (astrometric alignment & background subtraction)
  2. Field contamination modeling
  3. Spectral extractions
  4. Redshift & emission line fits (multiple grisms)
  
- `NIRISS-simulation <https://github.com/gbrammer/grizli-notebooks/blob/main/NIRISS-simulation.ipynb>`__ **(11.11.16)**: Simulation and analysis of JWST/NIRISS observations

  1. Simulate NIRISS spectra in three blocking filters and two orients offset by 90 degrees.
  2. Simulation field taken from the Hubble WFC3/IR Ultra-Deep Field, with the WFC3 F140W image as the morphological reference and photo-z templates taken as the spectral models. 
  3. Extract spectra and fit redshifts and emission lines from the combined six exposures.


