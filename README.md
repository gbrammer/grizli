# Grizli - *Grism redshift & line* analysis software for space-based slitless spectroscopy

### What is `grizli`?

This early release of `grizli` is intended to demonstrate and demystify some general techniques for manipulating *HST* slitless spectroscopic observations, providing software kernels to address questions such as 

> "How does the G141 grism disperse the spectrum of a star/galaxy at pixel position `(x,y)` in my F140W direct image?".  

Much of the background related to this question in the context of the currently available software tools was discussed in a document by Brammer, Pirzkal and Ryan (2014), available [here](https://github.com/WFC3Grism/CodeDescription/).  Along with a detailed description of the format of the configuration files originally developed for the aXe software, we provided a compact [Python script](https://github.com/WFC3Grism/CodeDescription/blob/master/axe_disperse.py) to address exactly the question above and strip away all of the many layers of bookkeeping, file-IO, etc. in existing analysis pipelines such as aXe (Kummel et al. 2009) and "THREEDHST" (Brammer et al. 2012, Momcheva et al. 2015).  In fact, the relatively simple script serves as the fundamental kernel for the way `grizli` computes the grism dispersions.  

Eventually, `grizli` is intended to move away from simple "data reduction" (e.g.., extracting a 1D spectrum of a single object akin to standard slit spectroscopy) and toward enabling more quantitative and comprehensive **modeling** and fitting of slitless spectroscopic observations, which typically involve overlapping spectra of hundreds or thousands of objects in exposures taken with one or more separate grisms and at multiple dispersion position angles.  The products of this type of analysis will be complete and uniform characterization of the spectral properties (e.g., continuum shape, redshifts, line fluxes) of all objects in a given exposure taken in the slitless spectroscopic mode. 
 
### Installation and download

`grizli` has the following Python dependencies:
    
* Numpy
* Scipy (scipy.ndimage)
* (Matplotlib) - generally needed for making plots, though not much plotting yet in `grizli`
* cython - Needed for compiling the C-accelerated functions in utils_c
* [astropy](http://www.astropy.org/)
* [scikit-learn](http://scikit-learn.org/stable/install.html) - For the linear least squares solver in the emonstration line-fitting code.
* [stwcs](http://stsdas.stsci.edu/stsci_python_epydoc/stwcs/index.html) - STScI packages for *HST*-specific WCS definitions.  This may be deprecated by [astropy.wcs](http://docs.astropy.org/en/stable/wcs/).
* [drizzlepac](http://drizzlepac.stsci.edu/) - STScI drizzle/blot library
* ([photutils](https://photutils.readthedocs.org/en/latest/)) - necessary if you want to create photometric catalogs and segmentation images directly within python
* ([pysynphot](http://pysynphot.readthedocs.org/en/latest/)) - Not required, but can be useful for dealing with *HST* filters and sensitivies, as well as handling unit conversions for synthetic spectra
     
The easiest way to satisfy all of these dependencies is to work within the STScI/Gemini "Ureka" Python distribution: http://ssb.stsci.edu/ureka/.  The current development release of `Grizli` has been tested in Python 2.7 within the "SSBX" Ureka distribution, available here: http://ssb.stsci.edu/ssb_software.shtml.
 
```bash
### If you have the Github SSH key enabled
git clone git@github.com:gbrammer/grizli.git

### Otherwise just use https
git clone https://github.com:gbrammer/grizli.git

### Build and install
cd grizli
python setup.py build
python setup.py install
```

`grizli` requires a system variable (`$GRIZLI`) set to point to a working directory where the grism trace configuration and any additional setup files will live.  For example, with the BASH shell in `${HOME}/.bashrc`:

```bash
export GRIZLI="${HOME}/Grizli"
```

`grizli` uses the aXe configuration files to define spectral traces and sensitivies.  For *HST* WFC3/IR grism spectroscopy, these files can be downloaded here: http://www.stsci.edu/hst/wfc3/analysis/grism_obs/wfc3-grism-resources.html.  **Put the downloaded configuration files in `${GRIZLI}/CONF`**, e.g., 

```bash
mkdir $GRIZLI/CONF
cd $GRIZLI/CONF
wget http://www.stsci.edu/ftp/cdbs/wfc3_aux/WFC3.IR.G102.cal.V4.3.tar.gz
tar xzvf WFC3.IR.G102.cal.V4.3.tar.gz
wget http://www.stsci.edu/ftp/cdbs/wfc3_aux/WFC3.IR.G141.cal.V4.3.tar.gz
tar xzvf WFC3.IR.G141.cal.V4.3.tar.gz
```

### Demo: working with WFC3/IR grism exposures (FLT) 

