"""
Model grism spectra in individual FLTs
"""
import os
import glob

from collections import OrderedDict
import copy
import traceback

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

import astropy.io.fits as pyfits
from astropy.table import Table
import astropy.wcs as pywcs
import astropy.units as u

#import stwcs

# Helper functions from a document written by Pirzkal, Brammer & Ryan
from . import grismconf
from . import utils
# from .utils_c import disperse
# from .utils_c import interp
from . import GRIZLI_PATH

# Would prefer 'nearest' but that occasionally segment faults out
SEGMENTATION_INTERP = 'nearest'

# Factors for converting HST countrates to Flamba flux densities
photflam_list = {'F098M': 6.0501324882418389e-20,
            'F105W': 3.038658152508547e-20,
            'F110W': 1.5274130068787271e-20,
            'F125W': 2.2483414275260141e-20,
            'F140W': 1.4737154005353565e-20,
            'F160W': 1.9275637653833683e-20,
            'F435W': 3.1871480286278679e-19,
            'F606W': 7.8933594352047833e-20,
            'F775W': 1.0088466875014488e-19,
            'F814W': 7.0767633156044843e-20,
            'VISTAH': 1.9275637653833683e-20*0.95,
            'GRISM': 1.e-20,
            'G150': 1.e-20,
            'G800L': 1.,
            'G280':  1., 
            'F444W': 1.e-20, 
            'F115W': 1., 
            'F150W': 1.,
            'F200W': 1.}

# Filter pivot wavelengths
photplam_list = {'F098M': 9864.722728110915,
            'F105W': 10551.046906405772,
            'F110W': 11534.45855553774,
            'F125W': 12486.059785775655,
            'F140W': 13922.907350356367,
            'F160W': 15369.175708965562,
            'F435W': 4328.256914042873,
            'F606W': 5921.658489236346,
            'F775W': 7693.297933335407,
            'F814W': 8058.784799323767,
            'VISTAH': 1.6433e+04,
            'GRISM': 1.6e4, # WFIRST/Roman
            'G150': 1.46e4, # WFIRST/Roman
            'G800L': 7.4737026e3,
            'G280': 3651., 
            'F070W': 7.043e+03, # NIRCam
            'F090W': 9.023e+03,
            'F115W': 1.150e+04, # NIRISS
            'F150W': 1.493e+04, # NIRISS
            'F200W': 1.993e+04, # NIRISS
            'F150W2': 1.658e+04,
            'F140M': 1.405e+04,
            'F158M': 1.582e+04, # NIRISS
            'F162M': 1.627e+04,
            'F182M': 1.845e+04,
            'F210M': 2.096e+04,
            'F164N': 1.645e+04,
            'F187N': 1.874e+04,
            'F212N': 2.121e+04,
            'F277W': 2.758e+04,
            'F356W': 3.568e+04,
            'F444W': 4.404e+04,
            'F322W2': 3.232e+04,
            'F250M': 2.503e+04,
            'F300M': 2.987e+04,
            'F335M': 3.362e+04,
            'F360M': 3.624e+04,
            'F380M': 3.825e+04, # NIRISS
            'F410M': 4.082e+04,
            'F430M': 4.280e+04,
            'F460M': 4.626e+04,
            'F480M': 4.816e+04,
            'F323N': 3.237e+04,
            'F405N': 4.052e+04,
            'F466N': 4.654e+04,
            'F470N': 4.708e+04}

# character to skip clearing line on STDOUT printing
#no_newline = '\x1b[1A\x1b[1M'

# Demo for computing photflam and photplam with pysynphot
if False:
    import pysynphot as S
    n = 1.e-20
    spec = S.FlatSpectrum(n, fluxunits='flam')
    photflam_list = {}
    photplam_list = {}
    for filter in ['F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W', 'G102', 'G141']:
        bp = S.ObsBandpass('wfc3,ir,{0}'.format(filter.lower()))
        photplam_list[filter] = bp.pivot()
        obs = S.Observation(spec, bp)
        photflam_list[filter] = n/obs.countrate()

    for filter in ['F435W', 'F606W', 'F775W', 'F814W']:
        bp = S.ObsBandpass('acs,wfc1,{0}'.format(filter.lower()))
        photplam_list[filter] = bp.pivot()
        obs = S.Observation(spec, bp)
        photflam_list[filter] = n/obs.countrate()


class GrismDisperser(object):
    def __init__(self, id=0, direct=None,
                       segmentation=None, origin=[500, 500],
                       xcenter=0., ycenter=0., pad=(0,0), grow=1, beam='A',
                       conf=['WFC3', 'F140W', 'G141'], scale=1.,
                       fwcpos=None, MW_EBV=0., yoffset=0, xoffset=None):
        """Object for computing dispersed model spectra

        Parameters
        ----------
        id : int
            Only consider pixels in the segmentation image with value `id`.
            Default of zero to match the default empty segmentation image.

        direct : `~numpy.ndarray`
            Direct image cutout in f_lambda units (i.e., e-/s times PHOTFLAM).
            Default is a trivial zeros array.

        segmentation : `~numpy.ndarray` (float32) or None
            Segmentation image.  If None, create a zeros array with the same
            shape as `direct`.

        origin : [int, int]
            `origin` defines the lower left pixel index (y,x) of the `direct`
            cutout from a larger detector-frame image

        xcenter, ycenter : float, float
            Sub-pixel centering of the exact center of the object, relative
            to the center of the thumbnail.  Needed for getting exact
            wavelength grid correct for the extracted 2D spectra.

        pad : int, int
            Offset between origin = [0,0] and the true lower left pixel of the
            detector frame.  This can be nonzero for cases where one creates
            a direct image that extends beyond the boundaries of the nominal
            detector frame to model spectra at the edges.

        grow : int >= 1
            Interlacing factor.

        beam : str
            Spectral order to compute.  Must be defined in `self.conf.beams`

        conf : [str, str, str] or `grismconf.aXeConf` object.
            Pre-loaded aXe-format configuration file object or if list of
            strings determine the appropriate configuration filename with
            `grismconf.get_config_filename` and load it.

        scale : float
            Multiplicative factor to apply to the modeled spectrum from
            `compute_model`.

        fwcpos : float
            Rotation position of the NIRISS filter wheel

        MW_EBV : float
            Galactic extinction

        yoffset : float
            Cross-dispersion offset to apply to the trace
        
        xoffset : float
            Dispersion offset to apply to the trace
            
        Attributes
        ----------
        sh : 2-tuple
            shape of the direct array

        sh_beam : 2-tuple
            computed shape of the 2D spectrum

        seg : `~numpy.array`
            segmentation array

        lam : `~numpy.array`
            wavelength along the trace

        ytrace : `~numpy.array`
            y pixel center of the trace.  Has same dimensions as sh_beam[1].

        sensitivity : `~numpy.array`
            conversion factor from native e/s to f_lambda flux densities

        lam_beam, ytrace_beam, sensitivity_beam : `~numpy.array`
            Versions of the above attributes defined for just the specific
            pixels of the pixel beam, not the full 2D extraction.

        modelf, model : `~numpy.array`, `~numpy.ndarray`
            2D model spectrum.  `model` is linked to `modelf` with "reshape",
            the later which is a flattened 1D array where the fast
            calculations are actually performed.

        model : `~numpy.ndarray`
            2D model spectrum linked to `modelf` with reshape.

        slx_parent, sly_parent : slice
            slices defined relative to `origin` to match the location of the
            computed 2D spectrum.

        total_flux : float
            Total f_lambda flux in the thumbail within the segmentation
            region.
        """

        self.id = id

        # lower left pixel of the `direct` array in native detector
        # coordinates
        self.origin = origin
        if isinstance(pad, int):
            self.pad = [pad, pad]
        else:
            self.pad = pad
            
        self.grow = grow

        # Galactic extinction
        self.MW_EBV = MW_EBV
        self.init_galactic_extinction(self.MW_EBV)

        self.fwcpos = fwcpos
        self.scale = scale

        # Direct image
        if direct is None:
            direct = np.zeros((20, 20), dtype=np.float32)

        self.direct = direct
        self.sh = self.direct.shape
        if self.direct.dtype is not np.float32:
            self.direct = np.cast[np.float32](self.direct)

        # Segmentation image, defaults to all zeros
        if segmentation is None:
            #self.seg = np.zeros_like(self.direct, dtype=np.float32)
            empty = np.zeros_like(self.direct, dtype=np.float32)
            self.set_segmentation(empty)
        else:
            self.set_segmentation(segmentation.astype(np.float32))

        # Initialize attributes
        self.spectrum_1d = None
        self.is_cgs = False

        self.xc = self.sh[1]/2+self.origin[1]
        self.yc = self.sh[0]/2+self.origin[0]

        # Sub-pixel centering of the exact center of the object, relative
        # to the center of the thumbnail
        self.xcenter = xcenter
        self.ycenter = ycenter

        self.beam = beam

        # Config file
        if isinstance(conf, list):
            conf_f = grismconf.get_config_filename(*conf)
            self.conf = grismconf.load_grism_config(conf_f)
        else:
            self.conf = conf

        # Get Pixel area map (xxx need to add test for WFC3)
        self.PAM_value = self.get_PAM_value(verbose=False)

        self.process_config()

        self.yoffset = yoffset
        
        if xoffset is not None:
            self.xoffset = xoffset
            
        if (yoffset != 0) | (xoffset is not None):
            #print('yoffset!', yoffset)
            self.add_ytrace_offset(yoffset)


    def set_segmentation(self, seg_array):
        """
        Set Segmentation array and `total_flux`.
        """
        self.seg = seg_array*1
        self.seg_ids = list(np.unique(self.seg))
        try:
            self.total_flux = self.direct[self.seg == self.id].sum()
            if self.total_flux == 0:
                self.total_flux = 1
        except:
            self.total_flux = 1.

    def init_galactic_extinction(self, MW_EBV=0., R_V=utils.MW_RV):
        """
        Initialize Fitzpatrick 99 Galactic extinction

        Parameters
        ----------
        MW_EBV : float
            Local E(B-V)

        R_V : float
            Relation between specific and total extinction,
            ``a_v = r_v * ebv``.

        Returns
        -------
        Sets `self.MW_F99` attribute, which is a callable function that
        returns the extinction for a supplied array of wavelengths.

        If MW_EBV <= 0, then sets `self.MW_F99 = None`.

        """
        self.MW_F99 = None
        if MW_EBV > 0:
            self.MW_F99 = utils.MW_F99(MW_EBV*R_V, r_v=R_V)

    def process_config(self):
        """Process grism config file

        Parameters
        ----------
        none

        Returns
        -------
        Sets attributes that define how the dispersion is computed.  See the
        attributes list for `~grizli.model.GrismDisperser`.
        """
        from .utils_c import interp

        # Get dispersion parameters at the reference position
        self.dx = self.conf.dxlam[self.beam]  # + xcenter #-xoffset
        if self.grow > 1:
            self.dx = np.arange(self.dx[0]*self.grow, self.dx[-1]*self.grow)

        xoffset = 0.

        if ('G14' in self.conf.conf_file) & (self.beam == 'A'):
            xoffset = -0.5  # necessary for WFC3/IR G141, v4.32

        # xoffset = 0. # suggested by ACS
        # xoffset = -2.5 # test

        self.xoffset = xoffset
        self.ytrace_beam, self.lam_beam = self.conf.get_beam_trace(
                            x=(self.xc+self.xcenter-self.pad[1])/self.grow,
                            y=(self.yc+self.ycenter-self.pad[0])/self.grow,
                        dx=(self.dx+self.xcenter*0+self.xoffset)/self.grow,
                            beam=self.beam, fwcpos=self.fwcpos)

        self.ytrace_beam *= self.grow

        # Integer trace
        # Add/subtract 20 for handling int of small negative numbers
        dyc = np.cast[int](self.ytrace_beam+20)-20+1

        # Account for pixel centering of the trace
        self.yfrac_beam = self.ytrace_beam - np.floor(self.ytrace_beam)

        # Interpolate the sensitivity curve on the wavelength grid.
        ysens = self.lam_beam*0
        so = np.argsort(self.lam_beam)

        conf_sens = self.conf.sens[self.beam]
        if self.MW_F99 is not None:
            MWext = 10**(-0.4*(self.MW_F99(conf_sens['WAVELENGTH']*u.AA)))
        else:
            MWext = 1.

        ysens[so] = interp.interp_conserve_c(self.lam_beam[so],
                                             conf_sens['WAVELENGTH'],
                                             conf_sens['SENSITIVITY']*MWext,
                                             integrate=1, left=0, right=0)
        self.lam_sort = so

        # Needs term of delta wavelength per pixel for flux densities
        # dl = np.abs(np.append(self.lam_beam[1] - self.lam_beam[0],
        #                     np.diff(self.lam_beam)))
        # ysens *= dl#*1.e-17
        self.sensitivity_beam = ysens

        # Initialize the model arrays
        self.NX = len(self.dx)
        self.sh_beam = (self.sh[0], self.sh[1]+self.NX)

        self.modelf = np.zeros(np.product(self.sh_beam), dtype=np.float32)
        self.model = self.modelf.reshape(self.sh_beam)
        self.idx = np.arange(self.modelf.size,
                             dtype=np.int64).reshape(self.sh_beam)

        # Indices of the trace in the flattened array
        self.x0 = np.array(self.sh, dtype=np.int64) // 2
        self.x0 -= 1  # zero index!

        self.dxpix = self.dx - self.dx[0] + self.x0[1]  # + 1
        try:
            self.flat_index = self.idx[dyc + self.x0[0], self.dxpix]
        except IndexError:
            #print('Index Error', id, dyc.dtype, self.dxpix.dtype, self.x0[0], self.xc, self.yc, self.beam, self.ytrace_beam.max(), self.ytrace_beam.min())
            raise IndexError

        # Trace, wavelength, sensitivity across entire 2D array
        self.dxfull = np.arange(self.sh_beam[1], dtype=int)
        self.dxfull += self.dx[0]-self.x0[1]

        # self.ytrace, self.lam = self.conf.get_beam_trace(x=self.xc,
        #                  y=self.yc, dx=self.dxfull, beam=self.beam)

        self.ytrace, self.lam = self.conf.get_beam_trace(
                                x=(self.xc+self.xcenter-self.pad[1])/self.grow,
                                y=(self.yc+self.ycenter-self.pad[0])/self.grow,
                            dx=(self.dxfull+self.xcenter+xoffset)/self.grow,
                                beam=self.beam, fwcpos=self.fwcpos)

        self.ytrace *= self.grow

        ysens = self.lam*0
        so = np.argsort(self.lam)
        ysens[so] = interp.interp_conserve_c(self.lam[so],
                                             conf_sens['WAVELENGTH'],
                                             conf_sens['SENSITIVITY']*MWext,
                                             integrate=1, left=0, right=0)

        # dl = np.abs(np.append(self.lam[1] - self.lam[0],
        #                       np.diff(self.lam)))
        # ysens *= dl#*1.e-17
        self.sensitivity = ysens

        # Slices of the parent array based on the origin parameter
        self.slx_parent = slice(self.origin[1] + self.dxfull[0] + self.x0[1],
                            self.origin[1] + self.dxfull[-1] + self.x0[1]+1)

        self.sly_parent = slice(self.origin[0], self.origin[0] + self.sh[0])

        # print 'XXX wavelength: %s %s %s' %(self.lam[-5:], self.lam_beam[-5:], dl[-5:])

    def add_ytrace_offset(self, yoffset):
        """Add an offset in Y to the spectral trace

        Parameters
        ----------
        yoffset : float
            Y-offset to apply

        """
        from .utils_c.interp import interp_conserve_c
        
        self.ytrace_beam, self.lam_beam = self.conf.get_beam_trace(
                            x=(self.xc+self.xcenter-self.pad[1])/self.grow,
                            y=(self.yc+self.ycenter-self.pad[0])/self.grow,
                     dx=(self.dx+self.xcenter*0+self.xoffset)/self.grow,
                            beam=self.beam, fwcpos=self.fwcpos)

        self.ytrace_beam *= self.grow
        self.yoffset = yoffset

        self.ytrace_beam += yoffset

        # Integer trace
        # Add/subtract 20 for handling int of small negative numbers
        dyc = np.cast[int](self.ytrace_beam+20)-20+1

        # Account for pixel centering of the trace
        self.yfrac_beam = self.ytrace_beam - np.floor(self.ytrace_beam)

        try:
            self.flat_index = self.idx[dyc + self.x0[0], self.dxpix]
        except IndexError:
            # print 'Index Error', id, self.x0[0], self.xc, self.yc, self.beam, self.ytrace_beam.max(), self.ytrace_beam.min()
            raise IndexError

        # Trace, wavelength, sensitivity across entire 2D array
        self.ytrace, self.lam = self.conf.get_beam_trace(
                            x=(self.xc+self.xcenter-self.pad[1])/self.grow,
                            y=(self.yc+self.ycenter-self.pad[0])/self.grow,
                   dx=(self.dxfull+self.xcenter+self.xoffset)/self.grow,
                            beam=self.beam, fwcpos=self.fwcpos)

        self.ytrace *= self.grow
        self.ytrace += yoffset
        
        # Reset sensitivity
        ysens = self.lam_beam*0
        so = np.argsort(self.lam_beam)

        conf_sens = self.conf.sens[self.beam]
        if self.MW_F99 is not None:
            MWext = 10**(-0.4*(self.MW_F99(conf_sens['WAVELENGTH']*u.AA)))
        else:
            MWext = 1.

        ysens[so] = interp_conserve_c(self.lam_beam[so],
                                             conf_sens['WAVELENGTH'],
                                             conf_sens['SENSITIVITY']*MWext,
                                             integrate=1, left=0, right=0)
        self.lam_sort = so
        self.sensitivity_beam = ysens

        # Full array
        ysens = self.lam*0
        so = np.argsort(self.lam)
        ysens[so] = interp_conserve_c(self.lam[so],
                                             conf_sens['WAVELENGTH'],
                                             conf_sens['SENSITIVITY']*MWext,
                                             integrate=1, left=0, right=0)

        self.sensitivity = ysens


    def compute_model(self, id=None, thumb=None, spectrum_1d=None,
                      in_place=True, modelf=None, scale=None, is_cgs=False,
                      apply_sensitivity=True, reset=True):
        """Compute a model 2D grism spectrum

        Parameters
        ----------
        id : int
            Only consider pixels in the segmentation image (`self.seg`) with
            values equal to `id`.

        thumb : `~numpy.ndarray` with shape = `self.sh` or None
            Optional direct image.  If `None` then use `self.direct`.

        spectrum_1d : [`~numpy.array`, `~numpy.array`] or None
            Optional 1D template [wave, flux] to use for the 2D grism model.
            If `None`, then implicitly assumes flat f_lambda spectrum.

        in_place : bool
            If True, put the 2D model in `self.model` and `self.modelf`,
            otherwise put the output in a clean array or preformed `modelf`.

        modelf : `~numpy.array` with shape = `self.sh_beam`
            Preformed (flat) array to which the 2D model is added, if
            `in_place` is False.

        scale : float or None
           Multiplicative factor to apply to the modeled spectrum.

        is_cgs : bool
            Units of `spectrum_1d` fluxes are f_lambda cgs.

        Returns
        -------
        model : `~numpy.ndarray`
            If `in_place` is False, returns the 2D model spectrum.  Otherwise
            the result is stored in `self.model` and `self.modelf`.
        """
        from .utils_c import disperse
        from .utils_c import interp

        if id is None:
            id = self.id
            total_flux = self.total_flux
        else:
            self.id = id
            total_flux = self.direct[self.seg == id].sum()

        # Template (1D) spectrum interpolated onto the wavelength grid
        if in_place:
            self.spectrum_1d = spectrum_1d

        if scale is None:
            scale = self.scale
        else:
            self.scale = scale

        if spectrum_1d is not None:
            xspec, yspec = spectrum_1d
            scale_spec = self.sensitivity_beam*0.
            int_func = interp.interp_conserve_c
            scale_spec[self.lam_sort] = int_func(self.lam_beam[self.lam_sort],
                                                xspec, yspec)*scale
        else:
            scale_spec = scale

        self.is_cgs = is_cgs
        if is_cgs:
            scale_spec /= total_flux

        # Output data, fastest is to compute in place but doesn't zero-out
        # previous result
        if in_place:
            self.modelf *= (1-reset)
            modelf = self.modelf
        else:
            if modelf is None:
                modelf = self.modelf*(1-reset)

        # Optionally use a different direct image
        if thumb is None:
            thumb = self.direct
        else:
            if thumb.shape != self.sh:
                print("""
Error: `thumb` must have the same dimensions as the direct image! ({0:d},{1:d})
                """.format(self.sh[0], self.sh[1]))
                return False

        # Now compute the dispersed spectrum using the C helper
        if apply_sensitivity:
            sens_curve = self.sensitivity_beam
        else:
            sens_curve = 1.

        nonz = (sens_curve*scale_spec) != 0

        if (nonz.sum() > 0) & (id in self.seg_ids):
            status = disperse.disperse_grism_object(thumb, self.seg,
                                 np.float32(id),
                                 self.flat_index[nonz],
                                 self.yfrac_beam[nonz].astype(np.float64),
                            (sens_curve*scale_spec)[nonz].astype(np.float64),
                                 modelf, 
                                 self.x0,
                                 np.array(self.sh, dtype=np.int64),
                                 self.x0,
                                 np.array(self.sh_beam, dtype=np.int64))

        #print('yyy PAM')
        modelf /= self.PAM_value  # = self.get_PAM_value()

        if not in_place:
            return modelf
        else:
            self.model = modelf.reshape(self.sh_beam)
            return True

    def init_optimal_profile(self, seg_ids=None):
        """Initilize optimal extraction profile
        """
        if seg_ids is None:
            ids = [self.id]
        else:
            ids = seg_ids

        for i, id in enumerate(ids):
            if hasattr(self, 'psf_params'):
                m_i = self.compute_model_psf(id=id, in_place=False)
            else:
                m_i = self.compute_model(id=id, in_place=False)

            #print('Add {0} to optimal profile'.format(id))

            if i == 0:
                m = m_i
            else:
                m += m_i

        m = m.reshape(self.sh_beam)
        m[m < 0] = 0
        self.optimal_profile = m/m.sum(axis=0)

    def optimal_extract(self, data, bin=0, ivar=1., weight=1.):
        """`Horne (1986) <http://adsabs.harvard.edu/abs/1986PASP...98..609H>`_ optimally-weighted 1D extraction

        Parameters
        ----------
        data : `~numpy.ndarray` with shape `self.sh_beam`
            2D data to extract

        bin : int, optional
            Simple boxcar averaging of the output 1D spectrum

        ivar : float or `~numpy.ndarray` with shape `self.sh_beam`
            Inverse variance array or scalar float that multiplies the
            optimal weights

        weight : TBD

        Returns
        -------
        wave, opt_flux, opt_rms : `~numpy.array`
            `wave` is the wavelength of 1D array
            `opt_flux` is the optimally-weighted 1D extraction
            `opt_rms` is the weighted uncertainty of the 1D extraction

            All are optionally binned in wavelength if `bin` > 1.
        """
        import scipy.ndimage as nd

        if not hasattr(self, 'optimal_profile'):
            self.init_optimal_profile()

        if data.shape != self.sh_beam:
            print("""
`data` ({0},{1}) must have the same shape as the data array ({2},{3})
            """.format(data.shape[0], data.shape[1], self.sh_beam[0],
                  self.sh_beam[1]))
            return False

        if not isinstance(ivar, float):
            if ivar.shape != self.sh_beam:
                print("""
`ivar` ({0},{1}) must have the same shape as the data array ({2},{3})
                """.format(ivar.shape[0], ivar.shape[1], self.sh_beam[0],
                      self.sh_beam[1]))
                return False

        num = self.optimal_profile*data*ivar*weight
        den = self.optimal_profile**2*ivar*weight
        opt_flux = num.sum(axis=0)/den.sum(axis=0)
        opt_var = 1./den.sum(axis=0)

        if bin > 1:
            kern = np.ones(bin, dtype=float)/bin
            opt_flux = nd.convolve(opt_flux, kern)[bin // 2::bin]
            opt_var = nd.convolve(opt_var, kern**2)[bin // 2::bin]
            wave = self.lam[bin // 2::bin]
        else:
            wave = self.lam

        opt_rms = np.sqrt(opt_var)
        opt_rms[opt_var == 0] = 0

        return wave, opt_flux, opt_rms

    def trace_extract(self, data, r=0, bin=0, ivar=1., dy0=0):
        """Aperture extraction along the trace

        Parameters
        ----------
        data : array-like
            Data array with dimenions equivalent to those of `self.model`

        r : int
            Radius of of the aperture to extract, in pixels.  The extraction
            will be performed from `-r` to `+r` pixels below and above the
            central pixel of the trace.

        bin : int, optional
            Simple boxcar averaging of the output 1D spectrum

        ivar : float or `~numpy.ndarray` with shape `self.sh_beam`
            Inverse variance array or scalar float that multiplies the
            optimal weights

        dy0 : float
            Central pixel to extract, relative to the central pixel of
            the trace

        Returns
        -------
        wave, opt_flux, opt_rms : `~numpy.array`

        `wave` is the wavelength of 1D array
        `opt_flux` is the 1D aperture extraction
        `opt_rms` is the uncertainty of the 1D extraction, derived from
                  the sum of the pixel variances within the aperture

        All are optionally binned in wavelength if `bin` > 1.
        """
        dy = np.cast[int](np.round(self.ytrace+dy0))
        aper = np.zeros_like(self.model)
        y0 = self.sh_beam[0] // 2
        for d in range(-r, r+1):
            for i in range(self.sh_beam[1]):
                aper[y0+d+dy[i]-1, i] = 1

        var = 1./ivar
        if not np.isscalar(ivar):
            var[ivar == 0] = 0

        opt_flux = np.sum(data*aper, axis=0)
        opt_var = np.sum(var*aper, axis=0)

        if bin > 1:
            kern = np.ones(bin, dtype=float)/bin
            opt_flux = nd.convolve(opt_flux, kern)[bin // 2::bin]
            opt_var = nd.convolve(opt_var, kern**2)[bin // 2::bin]
            wave = self.lam[bin // 2::bin]
        else:
            wave = self.lam

        opt_rms = np.sqrt(opt_var)

        return wave, opt_flux, opt_rms

    def contained_in_full_array(self, full_array):
        """Check if subimage slice is fully contained within larger array
        """
        sh = full_array.shape
        if (self.sly_parent.start < 0) | (self.slx_parent.start < 0):
            return False

        if (self.sly_parent.stop >= sh[0]) | (self.slx_parent.stop >= sh[1]):
            return False

        return True

    def add_to_full_image(self, data, full_array):
        """Add spectrum cutout back to the full array

        `data` is *added* to `full_array` in place, so, for example, to
        subtract `self.model` from the full array, call the function with

            >>> self.add_to_full_image(-self.model, full_array)

        Parameters
        ----------
        data : `~numpy.ndarray` shape `self.sh_beam` (e.g., `self.model`)
            Spectrum cutout

        full_array : `~numpy.ndarray`
            Full detector array, where the lower left pixel of `data` is given
            by `origin`.

        """

        if self.contained_in_full_array(full_array):
            full_array[self.sly_parent, self.slx_parent] += data
        else:
            sh = full_array.shape

            xpix = np.arange(self.sh_beam[1])
            xpix += self.origin[1] + self.dxfull[0] + self.x0[1]

            ypix = np.arange(self.sh_beam[0])
            ypix += self.origin[0]

            okx = (xpix >= 0) & (xpix < sh[1])
            oky = (ypix >= 0) & (ypix < sh[1])

            if (okx.sum() == 0) | (oky.sum() == 0):
                return False

            sly = slice(ypix[oky].min(), ypix[oky].max()+1)
            slx = slice(xpix[okx].min(), xpix[okx].max()+1)
            full_array[sly, slx] += data[oky, :][:, okx]

        # print sly, self.sly_parent, slx, self.slx_parent
        return True

    def cutout_from_full_image(self, full_array):
        """Get beam-sized cutout from a full image

        Parameters
        ----------
        full_array : `~numpy.ndarray`
            Array of the size of the parent array from which the cutout was
            extracted.  If possible, the function first tries the slices with

                >>> sub = full_array[self.sly_parent, self.slx_parent]

            and then computes smaller slices for cases where the beam spectrum
            falls off the edge of the parent array.

        Returns
        -------
        cutout : `~numpy.ndarray`
            Array with dimensions of `self.model`.

        """
        # print self.sly_parent, self.slx_parent, full_array.shape

        if self.contained_in_full_array(full_array):
            data = full_array[self.sly_parent, self.slx_parent]
        else:
            sh = full_array.shape
            ###
            xpix = np.arange(self.sh_beam[1])
            xpix += self.origin[1] + self.dxfull[0] + self.x0[1]

            ypix = np.arange(self.sh_beam[0])
            ypix += self.origin[0]

            okx = (xpix >= 0) & (xpix < sh[1])
            oky = (ypix >= 0) & (ypix < sh[1])

            if (okx.sum() == 0) | (oky.sum() == 0):
                return False

            sly = slice(ypix[oky].min(), ypix[oky].max()+1)
            slx = slice(xpix[okx].min(), xpix[okx].max()+1)

            data = self.model*0.
            data[oky, :][:, okx] += full_array[sly, slx]

        return data

    def twod_axis_labels(self, wscale=1.e4, limits=None, mpl_axis=None):
        """Set 2D wavelength (x) axis labels based on spectral parameters

        Parameters
        ----------
        wscale : float
            Scale factor to divide from the wavelength units.  The default
            value of 1.e4 results in wavelength ticks in microns.

        limits : None, list = `[x0, x1, dx]`
            Will automatically use the whole wavelength range defined by the
            spectrum. To change, specify `limits = [x0, x1, dx]` to
            interpolate `self.beam.lam_beam` between x0*wscale and x1*wscale.

        mpl_axis : `matplotlib.axes._axes.Axes`
            Plotting axis to place the labels, e.g.,

            >>> fig = plt.figure()
            >>> mpl_axis = fig.add_subplot(111)

        Returns
        -------
        Nothing if `mpl_axis` is supplied, else pixels and wavelengths of the
        tick marks.
        """
        xarr = np.arange(len(self.lam))
        if limits:
            xlam = np.arange(limits[0], limits[1], limits[2])
            xpix = np.interp(xlam, self.lam/wscale, xarr)
        else:
            xlam = np.unique(np.cast[int](self.lam / 1.e4*10)/10.)
            xpix = np.interp(xlam, self.lam/wscale, xarr)

        if mpl_axis is None:
            return xpix, xlam
        else:
            mpl_axis.set_xticks(xpix)
            mpl_axis.set_xticklabels(xlam)

    def twod_xlim(self, x0, x1=None, wscale=1.e4, mpl_axis=None):
        """Set wavelength (x) axis limits on a 2D spectrum

        Parameters
        ----------
        x0 : float or list/tuple of floats
            minimum or (min,max) of the plot limits

        x1 : float or None
            max of the plot limits if x0 is a float

        wscale : float
            Scale factor to divide from the wavelength units.  The default
            value of 1.e4 results in wavelength ticks in microns.

        mpl_axis : `matplotlib.axes._axes.Axes`
            Plotting axis to place the labels.

        Returns
        -------
        Nothing if `mpl_axis` is supplied else pixels the desired wavelength
        limits.
        """
        if isinstance(x0, list) | isinstance(x0, tuple):
            x0, x1 = x0[0], x0[1]

        xarr = np.arange(len(self.lam))
        xpix = np.interp([x0, x1], self.lam/wscale, xarr)

        if mpl_axis:
            mpl_axis.set_xlim(xpix)
        else:
            return xpix

    def x_init_epsf(self, flat_sensitivity=False, psf_params=None, psf_filter='F140W', yoff=0.0, skip=0.5, get_extended=False, seg_mask=True):
        """Initialize ePSF fitting for point sources
        TBD
        """
        import scipy.sparse
        import scipy.ndimage

        #print('SKIP: {0}'.format(skip))

        EPSF = utils.EffectivePSF()
        if psf_params is None:
            self.psf_params = [self.total_flux, 0., 0.]
        else:
            self.psf_params = psf_params

        if self.psf_params[0] is None:
            self.psf_params[0] = self.total_flux  # /photflam_list[psf_filter]

        origin = np.array(self.origin) - np.array(self.pad)

        self.psf_yoff = yoff
        self.psf_filter = psf_filter

        self.psf = EPSF.get_ePSF(self.psf_params, sci=self.psf_sci,
                                 ivar=self.psf_ivar, origin=origin,
                                 shape=self.sh, filter=psf_filter,
                                 get_extended=get_extended)

        # self.psf_params[0] /= self.psf.sum()
        # self.psf /= self.psf.sum()

        # Center in detector coords
        y0, x0 = np.array(self.sh)/2.-1
        if len(self.psf_params) == 2:
            xd = x0+self.psf_params[0] + origin[1]
            yd = y0+self.psf_params[1] + origin[0]
        else:
            xd = x0+self.psf_params[1] + origin[1]
            yd = y0+self.psf_params[2] + origin[0]

        # Get wavelength array
        psf_xy_lam = []
        psf_ext_lam = []

        for i, filter in enumerate(['F105W', 'F125W', 'F160W']):
            psf_xy_lam.append(EPSF.get_at_position(x=xd, y=yd, filter=filter))
            psf_ext_lam.append(EPSF.extended_epsf[filter])

        filt_ix = np.arange(3)
        filt_lam = np.array([1.0551, 1.2486, 1.5369])*1.e4

        yp_beam, xp_beam = np.indices(self.sh_beam)
        xarr = np.arange(0, self.lam_beam.shape[0], skip)
        xarr = xarr[xarr <= self.lam_beam.shape[0]-1]
        xbeam = np.arange(self.lam_beam.shape[0])*1.

        #xbeam += 1.

        # yoff = 0 #-0.15
        psf_model = self.model*0.
        A_psf = []
        lam_psf = []

        if len(self.psf_params) == 2:
            lam_offset = self.psf_params[0]  # self.sh[1]/2 - self.psf_params[1] - 1
        else:
            lam_offset = self.psf_params[1]  # self.sh[1]/2 - self.psf_params[1] - 1

        self.lam_offset = lam_offset

        for xi in xarr:
            yi = np.interp(xi, xbeam, self.ytrace_beam)
            li = np.interp(xi, xbeam, self.lam_beam)

            if len(self.psf_params) == 2:
                dx = xp_beam-self.psf_params[0]-xi-x0
                dy = yp_beam-self.psf_params[1]-yi+yoff-y0
            else:
                dx = xp_beam-self.psf_params[1]-xi-x0
                dy = yp_beam-self.psf_params[2]-yi+yoff-y0

            # wavelength-dependent
            ii = np.interp(li, filt_lam, filt_ix, left=-1, right=10)
            if ii == -1:
                psf_xy_i = psf_xy_lam[0]*1
                psf_ext_i = psf_ext_lam[0]*1
            elif ii == 10:
                psf_xy_i = psf_xy_lam[2]*1
                psf_ext_i = psf_ext_lam[2]*1
            else:
                ni = int(ii)
                f = 1-(li-filt_lam[ni])/(filt_lam[ni+1]-filt_lam[ni])
                psf_xy_i = f*psf_xy_lam[ni] + (1-f)*psf_xy_lam[ni+1]
                psf_ext_i = f*psf_ext_lam[ni] + (1-f)*psf_ext_lam[ni+1]

            if not get_extended:
                psf_ext_i = None

            psf = EPSF.eval_ePSF(psf_xy_i, dx, dy, extended_data=psf_ext_i)
            if len(self.psf_params) > 2:
                psf *= self.psf_params[0]

            #print(xi, psf.sum())

            if seg_mask:
                segm = nd.maximum_filter((self.seg == self.id)*1., size=7)
                #yps, xps = np.indices(self.sh)
                seg_i = nd.map_coordinates(segm, np.array([dx+x0, dy+y0]), order=1, mode='constant', cval=0.0, prefilter=True) > 0
            else:
                seg_i = 1

            A_psf.append((psf*seg_i).flatten())
            lam_psf.append(li)

        # Sensitivity
        self.lam_psf = np.array(lam_psf)

        #photflam = photflam_list[psf_filter]
        photflam = 1

        if flat_sensitivity:
            psf_sensitivity = np.abs(np.gradient(self.lam_psf))*photflam
        else:
            sens = self.conf.sens[self.beam]
            # so = np.argsort(self.lam_psf)
            # s_i = interp.interp_conserve_c(self.lam_psf[so], sens['WAVELENGTH'], sens['SENSITIVITY'], integrate=1)
            # psf_sensitivity = s_i*0.
            # psf_sensitivity[so] = s_i

            if self.MW_F99 is not None:
                MWext = 10**(-0.4*(self.MW_F99(sens['WAVELENGTH']*u.AA)))
            else:
                MWext = 1.

            psf_sensitivity = self.get_psf_sensitivity(sens['WAVELENGTH'], sens['SENSITIVITY']*MWext)

        self.psf_sensitivity = psf_sensitivity
        self.A_psf = scipy.sparse.csr_matrix(np.array(A_psf).T)
        # self.init_extended_epsf()

        self.PAM_value = self.get_PAM_value()
        self.psf_scale_to_data = 1.
        self.psf_renorm = 1.

        self.renormalize_epsf_model()

        self.init_optimal_profile()

    def get_psf_sensitivity(self, wave, sensitivity):
        """
        Integrate the sensitivity curve to the wavelengths for the
        PSF model
        """
        from .utils_c import interp

        so = np.argsort(self.lam_psf)
        s_i = interp.interp_conserve_c(self.lam_psf[so], wave, sensitivity, integrate=1)
        psf_sensitivity = s_i*0.
        psf_sensitivity[so] = s_i
        return psf_sensitivity

    def renormalize_epsf_model(self, spectrum_1d=None, verbose=False):
        """
        Ensure normalization correct
        """
        from .utils_c import interp

        if not hasattr(self, 'A_psf'):
            print('ePSF not initialized')
            return False

        if spectrum_1d is None:
            dl = 0.1
            flat_x = np.arange(self.lam.min()-10, self.lam.max()+10, dl)
            flat_y = flat_x*0.+1.e-17
            spectrum_1d = [flat_x, flat_y]

        tab = self.conf.sens[self.beam]
        if self.MW_F99 is not None:
            MWext = 10**(-0.4*(self.MW_F99(tab['WAVELENGTH']*u.AA)))
        else:
            MWext = 1.

        sens_i = interp.interp_conserve_c(spectrum_1d[0], tab['WAVELENGTH'], tab['SENSITIVITY']*MWext, integrate=1, left=0, right=0)
        total_sens = np.trapz(spectrum_1d[1]*sens_i/np.gradient(spectrum_1d[0]), spectrum_1d[0])

        m = self.compute_model_psf(spectrum_1d=spectrum_1d, is_cgs=True, in_place=False).reshape(self.sh_beam)
        #m2 = self.compute_model(spectrum_1d=[flat_x, flat_y], is_cgs=True, in_place=False).reshape(self.sh_beam)

        renorm = total_sens / m.sum()
        self.psf_renorm = renorm

        # Scale model to data, depends on Pixel Area Map and PSF normalization
        scale_to_data = self.PAM_value  # * (self.psf_params[0]/0.975)
        self.psf_scale_to_data = scale_to_data
        renorm /= scale_to_data  # renorm PSF

        if verbose:
            print('Renorm ePSF model: {0:0.3f}'.format(renorm))

        self.A_psf *= renorm

    def get_PAM_value(self, verbose=False):
        """
        Apply Pixel Area Map correction to WFC3 effective PSF model

        http://www.stsci.edu/hst/wfc3/pam/pixel_area_maps
        """
        confp = self.conf.conf_dict
        if ('INSTRUMENT' in confp) & ('CAMERA' in confp):
            instr = '{0}-{1}'.format(confp['INSTRUMENT'], confp['CAMERA'])
            if instr != 'WFC3-IR':
                return 1
        else:
            return 1

        try:
            with pyfits.open(os.getenv('iref')+'ir_wfc3_map.fits') as pam:
                pam_data = pam[1].data
                
            pam_value = pam_data[int(self.yc-self.pad[0]),
                                 int(self.xc-self.pad[1])]
            pam.close()
        except:
            pam_value = 1

        if verbose:
            msg = 'PAM correction at x={0}, y={1}: {2:.3f}'
            print(msg.format(self.xc-self.pad[1],
                             self.yc-self.pad[0],
                             pam_value))

        return pam_value

    def init_extended_epsf(self):
        """
        Hacky code for adding extended component of the EPSFs
        """
        ext_file = os.path.join(GRIZLI_PATH, 'CONF',
                            'ePSF_extended_splines.npy')

        if not os.path.exists(ext_file):
            return False

        bg_splines = np.load(ext_file, allow_pickle=True)[0]
        spline_waves = np.array(list(bg_splines.keys()))
        spline_waves.sort()
        spl_ix = np.arange(len(spline_waves))

        yarr = np.arange(self.sh_beam[0]) - self.sh_beam[0]/2.+1
        dy = self.psf_params[2]

        spl_data = self.model * 0.
        for i in range(self.sh_beam[1]):
            dy_i = dy + self.ytrace[i]
            x_i = np.interp(self.lam[i], spline_waves, spl_ix)
            if (x_i == 0) | (x_i == len(bg_splines)-1):
                spl_data[:, i] = bg_splines[spline_waves[int(x_i)]](yarr-dy_i)
            else:
                f = x_i-int(x_i)
                sp = bg_splines[spline_waves[int(x_i)]](yarr-dy_i)*(1-f)
                sp += bg_splines[spline_waves[int(x_i)+1]](yarr-dy_i)*f

                spl_data[:, i] = sp

        self.ext_psf_data = np.maximum(spl_data, 0)

    def compute_model_psf(self, id=None, spectrum_1d=None, in_place=True, is_cgs=False, apply_sensitivity=True):
        """
        Compute model with PSF morphology template
        """
        from .utils_c import interp

        if spectrum_1d is None:
            #modelf = np.array(self.A_psf.sum(axis=1)).flatten()
            #model = model.reshape(self.sh_beam)
            coeffs = np.ones(self.A_psf.shape[1])
            if not is_cgs:
                coeffs *= self.total_flux
        else:
            dx = np.diff(self.lam_psf)[0]
            if dx < 0:
                coeffs = interp.interp_conserve_c(self.lam_psf[::-1],
                                                  spectrum_1d[0],
                                                  spectrum_1d[1])[::-1]
            else:
                coeffs = interp.interp_conserve_c(self.lam_psf,
                                                  spectrum_1d[0],
                                                  spectrum_1d[1])

            if not is_cgs:
                coeffs *= self.total_flux

        modelf = self.A_psf.dot(coeffs*self.psf_sensitivity).astype(np.float32)
        model = modelf.reshape(self.sh_beam)

        # if hasattr(self, 'ext_psf_data'):
        #     model += self.ext_psf_data*model.sum(axis=0)
        #     modelf = model.flatten()
        #     model = modelf.reshape(self.sh_beam)

        if in_place:

            self.spectrum_1d = spectrum_1d
            self.is_cgs = is_cgs

            self.modelf = modelf  # .flatten()
            self.model = model
            #self.modelf = self.model.flatten()
            return True
        else:
            return modelf  # .flatten()


class ImageData(object):
    """Container for image data with WCS, etc."""

    def __init__(self, sci=None, err=None, dq=None,
                 header=None, wcs=None, photflam=1., photplam=1.,
                 origin=[0, 0], pad=(0,0), process_jwst_header=True,
                 instrument='WFC3', filter='G141', pupil=None, module=None, 
                 hdulist=None,
                 sci_extn=1, fwcpos=None):
        """
        Parameters
        ----------
        sci : `~numpy.ndarray`
            Science data

        err, dq : `~numpy.ndarray` or None
            Uncertainty and DQ data.  Defaults to zero if None

        header : `~astropy.io.fits.Header`
            Associated header with `data` that contains WCS information

        wcs : `~astropy.wcs.WCS` or None
            WCS solution to use.  If `None` will derive from the `header`.

        photflam : float
            Multiplicative conversion factor to scale `data` to set units
            to f_lambda flux density.  If data is grism spectra, then use
            photflam=1

        origin : [int, int]
            Origin of lower left pixel in detector coordinates
        
        pad : int,int
            Padding to apply to the image dimensions in numpy axis order
        
        process_jwst_header : bool
            If the image is detected as coming from JWST NIRISS or NIRCAM, 
            generate the necessary header WCS keywords
        
        instrument : str
            Instrument where the image came from
        
        filter : str
            Filter from the image header.  For WFC3 and NIRISS this is the 
            dispersing element

        pupil : str
            Pupil from the image header (JWST instruments).  For NIRISS this 
            is the blocking filter and for NIRCAM this is the dispersing 
            element
        
        module : str
            Instrument module for NIRCAM ('A' or 'B')
            
        hdulist : `~astropy.io.fits.HDUList`, optional
            If specified, read `sci`, `err`, `dq` from the HDU list from a
            FITS file, e.g., WFC3 FLT.

        sci_extn : int
            Science EXTNAME to read from the HDUList, for example,
            `sci` = hdulist['SCI',`sci_extn`].
        
        fwcpos : float
            Filter wheel encoder position (NIRISS)
            
        Attributes
        ----------
        parent_file : str
            Filename of the parent from which the data were extracted

        data : dict
            Dictionary to store pixel data, with keys 'SCI', 'DQ', and 'ERR'.
            If a reference image has been supplied and processed, will also
            have an entry 'REF'.  The data arrays can also be addressed with
            the `__getitem__` method, i.e.,

                >>> self = ImageData(...)
                >>> print np.median(self['SCI'])

        pad : int, int
            Additional padding around the nominal image dimensions in 
            numpy array order

        wcs : `~astropy.wcs.WCS`
            WCS of the data array

        header : `~astropy.io.fits.Header`
            FITS header

        filter, instrument, photflam, photplam, APZP : str, float
            Parameters taken from the header

        ref_file, ref_photlam, ref_photplam, ref_filter : str, float
            Corresponding parameters for the reference image, if necessary.

        """
        import copy
        
        med_filter = None
        bkg_array = None
        
        # Easy way, get everything from an image HDU list
        if isinstance(hdulist, pyfits.HDUList):

            if ('REF', sci_extn) in hdulist:
                ref_h = hdulist['REF', sci_extn].header
                ref_data = hdulist['REF', sci_extn].data/ref_h['PHOTFLAM']
                ref_data = np.cast[np.float32](ref_data)

                ref_file = ref_h['REF_FILE']
                ref_photflam = 1.
                ref_photplam = ref_h['PHOTPLAM']
                #ref_filter = ref_h['FILTER']
                ref_filter = utils.parse_filter_from_header(ref_h)
                
            else:
                ref_data = None

            if ('SCI', sci_extn) in hdulist:
                sci = np.cast[np.float32](hdulist['SCI', sci_extn].data)
                err = np.cast[np.float32](hdulist['ERR', sci_extn].data)
                dq = np.cast[np.int16](hdulist['DQ', sci_extn].data)
                
                if ('MED',sci_extn) in hdulist:
                    mkey = ('MED',sci_extn)
                    med_filter = np.cast[np.float32](hdulist[mkey].data)
                    
                if ('BKG',sci_extn) in hdulist:
                    mkey = ('BKG',sci_extn)
                    bkg_array = np.cast[np.float32](hdulist[mkey].data)

                base_extn = ('SCI', sci_extn)

            else:
                if ref_data is None:
                    raise KeyError('No SCI or REF extensions found')

                # Doesn't have SCI, get from ref
                sci = err = ref_data*0.+1
                dq = np.zeros(sci.shape, dtype=np.int16)
                base_extn = ('REF', sci_extn)

            if 'ORIGINX' in hdulist[base_extn].header:
                h0 = hdulist[base_extn].header
                origin = [h0['ORIGINY'], h0['ORIGINX']]
            else:
                origin = [0, 0]

            self.sci_extn = sci_extn
            header = hdulist[base_extn].header.copy()

            if 'PARENT' in header:
                self.parent_file = header['PARENT']
            else:
                self.parent_file = hdulist.filename()

            if 'CPDIS1' in header:
                if 'Lookup' in header['CPDIS1']:
                    self.wcs_is_lookup = True
                else:
                    self.wcs_is_lookup = False
            else:
                self.wcs_is_lookup = False

            status = False
            for ext in [base_extn, 0]:
                h = hdulist[ext].header
                if 'INSTRUME' in h:
                    status = True
                    break

            if not status:
                msg = ('Couldn\'t find \'INSTRUME\' keyword in the headers' +
                       ' of extensions 0 or (SCI,{0:d})'.format(sci_extn))
                raise KeyError(msg)

            instrument = h['INSTRUME']
            filter = utils.parse_filter_from_header(h, filter_only=True)

            if 'PUPIL' in h:
                pupil = h['PUPIL']
            
            if 'MODULE' in h:
                module = h['MODULE']
            else:
                module = None
                
            if 'PHOTPLAM' in h:
                photplam = h['PHOTPLAM']
            elif filter in photplam_list:
                photplam = photplam_list[filter]
            else:
                photplam = 1

            if 'PHOTFLAM' in h:
                photflam = h['PHOTFLAM']
            
            elif filter in photflam_list:
                photflam = photflam_list[filter]
            
            elif 'PHOTUJA2' in header:
                # JWST calibrated products
                per_pix = header['PIXAR_SR']
                if header['BUNIT'].strip() == 'MJy/sr':
                    photfnu = per_pix*1e6
                else:
                    photfnu = 1./(header['PHOTMJSR']*1.e6)*per_pix

                photflam = photfnu/1.e23*3.e18/photplam**2
                
            else:
                photflam = 1.
                
                
            # For NIRISS
            if 'FWCPOS' in h:
                fwcpos = h['FWCPOS']

            self.mdrizsky = 0.
            if 'MDRIZSKY' in header:
                #sci -= header['MDRIZSKY']
                self.mdrizsky = header['MDRIZSKY']

            # ACS bunit
            #self.exptime = 1.
            if 'EXPTIME' in hdulist[0].header:
                self.exptime = hdulist[0].header['EXPTIME']
            else:
                self.exptime = hdulist[0].header['EFFEXPTM']

            # if 'BUNIT' in header:
            #     if header['BUNIT'] == 'ELECTRONS':
            #         self.exptime = hdulist[0].header['EXPTIME']
            #         # sci /= self.exptime
            #         # err /= self.exptime

            sci = (sci-self.mdrizsky)

            if 'BUNIT' in header:
                if header['BUNIT'] == 'ELECTRONS':
                    sci /= self.exptime
                    err /= self.exptime

            if filter.startswith('G'):
                photflam = 1

            if (instrument == 'NIRCAM') & (pupil is not None):
                if pupil.startswith('G'):
                    photflam = 1

            if 'PAD' in header:
                pad = [header['PAD'], header['PAD']]
            elif ('PADX' in header) & ('PADY' in header):
                pad = [header['PADY'], header['PADX']]
            else:
                pad = [0,0]
            
            self.grow = 1
            if 'GROW' in header:
                self.grow = header['GROW']

        else:
            if sci is None:
                sci = np.zeros((1014, 1014))

            self.parent_file = 'Unknown'
            self.sci_extn = None
            self.grow = 1
            ref_data = None

            if 'EXPTIME' in header:
                self.exptime = header['EXPTIME']
            else:
                self.exptime = 1.

            if 'MDRIZSKY' in header:
                self.mdrizsky = header['MDRIZSKY']
            else:
                self.mdrizsky = 0.

            if 'CPDIS1' in header:
                if 'Lookup' in header['CPDIS1']:
                    self.wcs_is_lookup = True
                else:
                    self.wcs_is_lookup = False
            else:
                self.wcs_is_lookup = False

        self.is_slice = False

        # Array parameters
        if isinstance(pad, int):
            self.pad = [pad, pad]
        else:
            self.pad = pad
            
        self.origin = origin
        self.fwcpos = fwcpos  # NIRISS
        self.MW_EBV = 0.

        self.data = OrderedDict()
        self.data['SCI'] = sci*photflam

        self.sh = np.array(self.data['SCI'].shape)

        # Header-like parameters
        self.filter = filter
        self.pupil = pupil
        
        if (instrument == 'NIRCAM'):
            # Fallback if module not specified
            if module is None:
                if 'MODULE' not in header:
                    self.module = 'A'
                else:
                    self.module = header['MODULE']
            else:
                self.module = module
        else:
            self.module = module
        
        self.instrument = instrument
        self.header = header
        if 'ISCUTOUT' in self.header:
            self.is_slice = self.header['ISCUTOUT']

        self.header['EXPTIME'] = self.exptime

        self.photflam = photflam
        self.photplam = photplam
        self.ABZP = (0*np.log10(self.photflam) - 21.10 -
                      5*np.log10(self.photplam) + 18.6921)
        self.thumb_extension = 'SCI'

        if err is None:
            self.data['ERR'] = np.zeros_like(self.data['SCI'])
        else:
            self.data['ERR'] = err*photflam
            if self.data['ERR'].shape != tuple(self.sh):
                raise ValueError('err and sci arrays have different shapes!')

        if dq is None:
            self.data['DQ'] = np.zeros_like(self.data['SCI'], dtype=np.int16)
        else:
            self.data['DQ'] = dq
            if self.data['DQ'].shape != tuple(self.sh):
                raise ValueError('err and dq arrays have different shapes!')

        if ref_data is None:
            self.data['REF'] = None
            self.ref_file = None
            self.ref_photflam = None
            self.ref_photplam = None
            self.ref_filter = None
        else:
            self.data['REF'] = ref_data
            self.ref_file = ref_file
            self.ref_photflam = ref_photflam
            self.ref_photplam = ref_photplam
            self.ref_filter = ref_filter
        
        if med_filter is not None:
            self.data['MED'] = med_filter

        if bkg_array is not None:
            self.data['BKG'] = bkg_array
            
        self.wcs = None

        # if (instrument in ['NIRISS', 'NIRCAM']) & (~self.is_slice):
        #     if process_jwst_header:
        #         self.update_jwst_wcsheader(hdulist)

        if self.header is not None:
            if wcs is None:
                self.get_wcs()
            else:
                self.wcs = wcs.copy()
                if not hasattr(self.wcs, 'pixel_shape'):
                    self.wcs.pixel_shape = self.wcs._naxis1, self.wcs._naxis2

        else:
            self.header = pyfits.Header()

        # Detector chip
        if 'CCDCHIP' in self.header:
            self.ccdchip = self.header['CCDCHIP']
        else:
            self.ccdchip = 1

        # Galactic extinction
        if 'MW_EBV' in self.header:
            self.MW_EBV = self.header['MW_EBV']
        else:
            self.MW_EBV = 0.

    def unset_dq(self):
        """Flip OK data quality bits using utils.unset_dq_bits

        OK bits are defined as

            >>> okbits_instrument = {'WFC3': 32+64+512, # blob OK
                                     'NIRISS': 1+2+4,
                                     'NIRCAM': 1+2+4,
                                     'WFIRST': 0,
                                     'WFI': 0}
        """

        okbits_instrument = {'WFC3': 32+64+512,  # blob OK
                             'NIRISS': 1+2+4, #+4096+4100+18432+18436+1024+16384+1,
                             'NIRCAM': 1+2+4,
                             'WFIRST': 0, 
                             'WFI': 0}

        if self.instrument not in okbits_instrument:
            okbits = 1
        else:
            okbits = okbits_instrument[self.instrument]

        self.data['DQ'] = utils.unset_dq_bits(self.data['DQ'], okbits=okbits)


    def flag_negative(self, sigma=-3):
        """Flag negative data values with dq=4

        Parameters
        ----------
        sigma : float
            Threshold for setting bad data

        Returns
        -------
        n_negative : int
            Number of flagged negative pixels

        If `self.data['ERR']` is zeros, do nothing.
        """
        if self.data['ERR'].max() == 0:
            return 0

        bad = self.data['SCI'] < sigma*self.data['ERR']
        self.data['DQ'][bad] |= 4
        return bad.sum()

    def update_jwst_wcsheader(self, hdulist, force=False):
        """
        For now generate an approximate SIP header for NIRISS/NIRCam
        
        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            FITS HDU list
        
        force : bool
            
        
        """
        import jwst
        from . import jwst_utils as _jwst

        datamodel = _jwst.img_with_wcs(hdulist)
        if (jwst.__version__ < '1.3.2') | force:
            # Need to compute own transformed header
            sip_header = _jwst.model_wcs_header(datamodel, get_sip=True)
        else:
            sip_header = utils.to_header(datamodel.get_fits_wcs())
            
        for k in sip_header:
            self.header[k] = sip_header[k]

        # Remove PC
        for i in [1, 2]:
            for j in [1, 2]:
                k = 'PC{0}_{1}'.format(i, j)
                if k in self.header:
                    self.header.remove(k)

    def get_wcs(self, pc2cd=False):
        """Get WCS from header"""
        import numpy.linalg
        import stwcs

        if self.wcs_is_lookup:

            if 'CCDCHIP' in self.header:
                ext = {1: 2, 2: 1}[self.header['CCDCHIP']]
            else:
                ext = self.header['EXTVER']

            if os.path.exists(self.parent_file):
                with pyfits.open(self.parent_file) as fobj:
                    wcs = stwcs.wcsutil.hstwcs.HSTWCS(fobj=fobj, 
                                                      ext=('SCI', ext))
                if np.max(self.pad) > 0:
                    wcs = self.add_padding_to_wcs(wcs, pad=self.pad)

            else:
                # Get WCS from a stripped wcs.fits file (from self.save_wcs)
                # already padded.
                wcsfile = self.parent_file.replace('.fits', 
                                              '.{0:02d}.wcs.fits'.format(ext))
                
                with pyfits.open(wcsfile) as fobj:
                    fh = fobj[0].header
                    if fh['NAXIS'] == 0:
                        fh['NAXIS'] = 2
                        fh['NAXIS1'] = int(fh['CRPIX1']*2)
                        fh['NAXIS2'] = int(fh['CRPIX2']*2)

                    wcs = stwcs.wcsutil.hstwcs.HSTWCS(fobj=fobj, ext=0)

            # Object is a cutout
            if self.is_slice:
                slx = slice(self.origin[1], self.origin[1]+self.sh[1])
                sly = slice(self.origin[0], self.origin[0]+self.sh[0])

                wcs = self.get_slice_wcs(wcs, slx=slx, sly=sly)

        else:
            fobj = None
            wcs = pywcs.WCS(self.header, relax=True, fobj=fobj)

        if not hasattr(wcs, 'pscale'):
            wcs.pscale = utils.get_wcs_pscale(wcs)

        self.wcs = wcs
        if not hasattr(self.wcs, 'pixel_shape'):
            self.wcs.pixel_shape = self.wcs._naxis1, self.wcs._naxis2

    @staticmethod
    def add_padding_to_wcs(wcs_in, pad=(64,256)):
        """Pad the appropriate WCS keywords
        
        Parameters
        ----------
        wcs_in : `~astropy.wcs.WCS`
            Input WCS
        
        pad : int, int
            Number of pixels to pad, in array order (axis2, axis1)
        
        Returns
        -------
        wcs_out : `~astropy.wcs.WCS`
            Padded WCS
        
        """
        wcs = wcs_in.deepcopy()

        is_new = True
        for attr in ['naxis1', '_naxis1']:
            if hasattr(wcs, attr):
                is_new = False
                value = wcs.__getattribute__(attr)
                if value is not None:
                    wcs.__setattr__(attr, value+2*pad[1])

        for attr in ['naxis2', '_naxis2']:
            if hasattr(wcs, attr):
                is_new = False
                value = wcs.__getattribute__(attr)
                if value is not None:
                    wcs.__setattr__(attr, value+2*pad[0])

        # Handle changing astropy.wcs.WCS attributes
        if is_new:
            #for i in range(len(wcs._naxis)):
            #    wcs._naxis[i] += 2*pad
            wcs._naxis[0] += 2*pad[1]
            wcs._naxis[1] += 2*pad[0]
            
            wcs.naxis1, wcs.naxis2 = wcs._naxis
        else:
            wcs.naxis1 = wcs._naxis1
            wcs.naxis2 = wcs._naxis2

        wcs.wcs.crpix[0] += pad[1]
        wcs.wcs.crpix[1] += pad[0]

        # Pad CRPIX for SIP
        for wcs_ext in [wcs.sip]:
            if wcs_ext is not None:
                wcs_ext.crpix[0] += pad[1]
                wcs_ext.crpix[1] += pad[0]

        # Pad CRVAL for Lookup Table, if necessary (e.g., ACS)
        for wcs_ext in [wcs.cpdis1, wcs.cpdis2, wcs.det2im1, wcs.det2im2]:
            if wcs_ext is not None:
                wcs_ext.crval[0] += pad[1]
                wcs_ext.crval[1] += pad[0]

        return wcs

    def add_padding(self, pad=(64,256)):
        """Pad the data array and update WCS keywords"""
        
        if isinstance(pad, int):
            _pad = [pad, pad]
        else:
            _pad = pad
        
        # Update data array
        new_sh = np.array([s for s in self.sh])
        new_sh[0] += 2*pad[0]
        new_sh[1] += 2*pad[1]
        
        for key in ['SCI', 'ERR', 'DQ', 'REF']:
            if key not in self.data:
                continue
            else:
                if self.data[key] is None:
                    continue

            data = self.data[key]
            new_data = np.zeros(new_sh, dtype=data.dtype)
            new_data[pad[0]:-pad[0], pad[1]:-pad[1]] += data
            self.data[key] = new_data

        self.sh = new_sh
        
        for i in range(2):
            self.pad[i] += _pad[i]

        # Padded image dimensions
        self.header['NAXIS1'] += 2*_pad[1]
        self.header['NAXIS2'] += 2*_pad[0]

        self.header['CRPIX1'] += _pad[1]
        self.header['CRPIX2'] += _pad[0]

        # Add padding to WCS
        self.wcs = self.add_padding_to_wcs(self.wcs, pad=_pad)
        
        if not hasattr(self.wcs, 'pixel_shape'):
            self.wcs.pixel_shape = self.wcs._naxis1, self.wcs._naxis2

    def shrink_large_hdu(self, hdu=None, extra=100, verbose=False):
        """Shrink large image mosaic to speed up blotting

        Parameters
        ----------
        hdu : `~astropy.io.fits.ImageHDU`
            Input reference HDU

        extra : int
            Extra border to put around `self.data` WCS to ensure the reference
            image is large enough to encompass the distorted image

        Returns
        -------
        new_hdu : `~astropy.io.fits.ImageHDU`
            Image clipped to encompass `self.data['SCI']` + margin of `extra`
            pixels.

        Make a cutout of the larger reference image around the desired FLT
        image to make blotting faster for large reference images.
        """
        ref_wcs = pywcs.WCS(hdu.header)

        # Borders of the flt frame
        naxis = [self.header['NAXIS1'], self.header['NAXIS2']]
        xflt = [-extra, naxis[0]+extra, naxis[0]+extra, -extra]
        yflt = [-extra, -extra, naxis[1]+extra, naxis[1]+extra]

        raflt, deflt = self.wcs.all_pix2world(xflt, yflt, 0)
        xref, yref = np.cast[int](ref_wcs.all_world2pix(raflt, deflt, 0))
        ref_naxis = [hdu.header['NAXIS1'], hdu.header['NAXIS2']]

        # Slices of the reference image
        xmi = np.maximum(0, xref.min())
        xma = np.minimum(ref_naxis[0], xref.max())
        slx = slice(xmi, xma)

        ymi = np.maximum(0, yref.min())
        yma = np.minimum(ref_naxis[1], yref.max())
        sly = slice(ymi, yma)

        if ((xref.min() < 0) | (yref.min() < 0) |
             (xref.max() > ref_naxis[0]) | (yref.max() > ref_naxis[1])):
            if verbose:
                msg = 'Image cutout: x={0}, y={1} [Out of range]'
                print(msg.format(slx, sly))
            return hdu
        else:
            if verbose:
                print('Image cutout: x={0}, y={1}'.format(slx, sly))

        # Sliced subimage
        slice_wcs = ref_wcs.slice((sly, slx))
        slice_header = hdu.header.copy()
        hwcs = slice_wcs.to_header(relax=True)

        for k in hwcs.keys():
            if not k.startswith('PC'):
                slice_header[k] = hwcs[k]

        slice_data = hdu.data[sly, slx]*1
        new_hdu = pyfits.ImageHDU(data=slice_data, header=slice_header)

        return new_hdu

    def expand_hdu(self, hdu=None, verbose=True):
        """TBD
        """
        ref_wcs = pywcs.WCS(hdu.header)

        # Borders of the flt frame
        naxis = [self.header['NAXIS1'], self.header['NAXIS2']]
        xflt = [-self.pad[1], naxis[0]+self.pad[1],
                naxis[0]+self.pad[1], -self.pad[1]]
        yflt = [-self.pad[0], -self.pad[0],
                naxis[1]+self.pad[0], naxis[1]+self.pad[0]]

        raflt, deflt = self.wcs.all_pix2world(xflt, yflt, 0)
        xref, yref = np.cast[int](ref_wcs.all_world2pix(raflt, deflt, 0))
        ref_naxis = [hdu.header['NAXIS1'], hdu.header['NAXIS2']]

        pad_min = np.minimum(xref.min(), yref.min())
        pad_max = np.maximum((xref-ref_naxis[0]).max(), 
                             (yref-ref_naxis[1]).max())

        if (pad_min > 0) & (pad_max < 0):
            # do nothing
            return hdu

        pad = np.maximum(np.abs(pad_min), pad_max) + 64
        if verbose:
            msg = '{0} / Pad ref HDU with {1:d} pixels'
            print(msg.format(self.parent_file, pad))

        # Update data array
        sh = hdu.data.shape
        new_sh = np.array(sh) + 2*pad

        new_data = np.zeros(new_sh, dtype=hdu.data.dtype)
        new_data[pad:-pad, pad:-pad] += hdu.data

        header = hdu.header.copy()

        # Padded image dimensions
        header['NAXIS1'] += 2*pad
        header['NAXIS2'] += 2*pad

        # Add padding to WCS
        header['CRPIX1'] += pad
        header['CRPIX2'] += pad

        new_hdu = pyfits.ImageHDU(data=new_data, header=header)
        return new_hdu


    def blot_from_hdu(self, hdu=None, segmentation=False, grow=3,
                      interp='nearest'):
        """Blot a rectified reference image to detector frame

        Parameters
        ----------
        hdu : `~astropy.io.fits.ImageHDU`
            HDU of the reference image

        segmentation : bool, False
            If True, treat the reference image as a segmentation image and
            preserve the integer values in the blotting.
            
            If specified as number > 1, then use `~grizli.utils.blot_nearest_exact` 
            rather than a hacky pixel area ratio method to blot integer 
            segmentation maps.
            
        grow : int, default=3
            Number of pixels to dilate the segmentation regions

        interp : str,
            Form of interpolation to use when blotting float image pixels.
            Valid options: {'nearest', 'linear', 'poly3', 'poly5' (default), 'spline3', 'sinc'}

        Returns
        -------
        blotted : `np.ndarray`
            Blotted array with the same shape and WCS as `self.data['SCI']`.
        """

        import astropy.wcs
        from drizzlepac import astrodrizzle

        #ref = pyfits.open(refimage)
        if hdu.data.dtype.type != np.float32:
            #hdu.data = np.cast[np.float32](hdu.data)
            refdata = np.cast[np.float32](hdu.data)
        else:
            refdata = hdu.data

        if 'ORIENTAT' in hdu.header.keys():
            hdu.header.remove('ORIENTAT')

        if segmentation:
            seg_ones = np.cast[np.float32](refdata > 0)-1

        ref_wcs = pywcs.WCS(hdu.header, relax=True)
        flt_wcs = self.wcs.copy()

        # Fix some wcs attributes that might not be set correctly
        for wcs in [ref_wcs, flt_wcs]:

            if hasattr(wcs, '_naxis1'):
                wcs.naxis1 = wcs._naxis1
                wcs.naxis2 = wcs._naxis2
            else:
                wcs._naxis1, wcs._naxis2 = wcs._naxis

            if (not hasattr(wcs.wcs, 'cd')) & hasattr(wcs.wcs, 'pc'):
                wcs.wcs.cd = wcs.wcs.pc

            if hasattr(wcs, 'idcscale'):
                if wcs.idcscale is None:
                    wcs.idcscale = np.mean(np.sqrt(np.sum(wcs.wcs.cd**2, axis=0))*3600.)  # np.sqrt(np.sum(wcs.wcs.cd[0,:]**2))*3600.
            else:
                #wcs.idcscale = np.sqrt(np.sum(wcs.wcs.cd[0,:]**2))*3600.
                wcs.idcscale = np.mean(np.sqrt(np.sum(wcs.wcs.cd**2, axis=0))*3600.)  # np.sqrt(np.sum(wcs.wcs.cd[0,:]**2))*3600.

            wcs.pscale = utils.get_wcs_pscale(wcs)

        if segmentation:
            # Handle segmentation images a bit differently to preserve
            # integers.
            # +1 here is a hack for some memory issues
            
            if segmentation*1 == 1:
                seg_interp = 'nearest'
            
                blotted_ones = astrodrizzle.ablot.do_blot(seg_ones+1, ref_wcs,
                                    flt_wcs, 1, coeffs=True,
                                    interp=seg_interp,
                                    sinscl=1.0, stepsize=10, wcsmap=None)
            
                blotted_seg = astrodrizzle.ablot.do_blot(refdata*1., ref_wcs,
                                    flt_wcs, 1, coeffs=True,
                                    interp=seg_interp,
                                    sinscl=1.0, stepsize=10, wcsmap=None)
            
                blotted_ones[blotted_ones == 0] = 1
            
                #pixel_ratio = (flt_wcs.idcscale / ref_wcs.idcscale)**2
                #in_seg = np.abs(blotted_ones - pixel_ratio) < 1.e-2
            
                ratio = np.round(blotted_seg/blotted_ones)
                seg = nd.maximum_filter(ratio, size=grow, 
                                        mode='constant', cval=0)
                ratio[ratio == 0] = seg[ratio == 0]
                blotted = ratio
            else:
                blotted = utils.blot_nearest_exact(refdata, ref_wcs, flt_wcs, 
                                               verbose=True, stepsize=-1,
                                               scale_by_pixel_area=False, 
                                               wcs_mask=True,
                                               fill_value=0)
        else:
            # Floating point data
            blotted = astrodrizzle.ablot.do_blot(refdata, ref_wcs, flt_wcs, 1,
                                coeffs=True, interp=interp, sinscl=1.0,
                                stepsize=10, wcsmap=None)

        return blotted

    @staticmethod
    def get_slice_wcs(wcs, slx=slice(480, 520), sly=slice(480, 520)):
        """Get slice of a WCS including higher orders like SIP and DET2IM

        The normal `~astropy.wcs.wcs.WCS` `slice` method doesn't apply the
        slice to all of the necessary keywords.  For example, SIP WCS also
        has a `CRPIX` reference pixel that needs to be offset along with
        the main `CRPIX`.

        Parameters
        ----------
        slx, sly : slice
            Slices in x and y dimensions to extract

        """
        NX = slx.stop - slx.start
        NY = sly.stop - sly.start

        slice_wcs = wcs.slice((sly, slx))
        if hasattr(slice_wcs, '_naxis1'):
            slice_wcs.naxis1 = slice_wcs._naxis1 = NX
            slice_wcs.naxis2 = slice_wcs._naxis2 = NY
        else:
            slice_wcs._naxis = [NX, NY]
            slice_wcs._naxis1, slice_wcs._naxis2 = NX, NY

        if hasattr(slice_wcs, 'sip'):
            if slice_wcs.sip is not None:
                for c in [0, 1]:
                    slice_wcs.sip.crpix[c] = slice_wcs.wcs.crpix[c]

        ACS_CRPIX = [4096/2, 2048/2]  # ACS
        dx_crpix = slice_wcs.wcs.crpix[0] - ACS_CRPIX[0]
        dy_crpix = slice_wcs.wcs.crpix[1] - ACS_CRPIX[1]
        for ext in ['cpdis1', 'cpdis2', 'det2im1', 'det2im2']:
            if hasattr(slice_wcs, ext):
                wcs_ext = slice_wcs.__getattribute__(ext)
                if wcs_ext is not None:
                    wcs_ext.crval[0] += dx_crpix
                    wcs_ext.crval[1] += dy_crpix
                    slice_wcs.__setattr__(ext, wcs_ext)

        return slice_wcs

    def get_slice(self, slx=slice(480, 520), sly=slice(480, 520),
                 get_slice_header=True):
        """Return cutout version of the `ImageData` object

        Parameters
        ----------
        slx, sly : slice
            Slices in x and y dimensions to extract

        get_slice_header : bool
            Compute the full header of the slice.  This takes a bit of time
            and isn't necessary in all cases so can be omitted if only the
            sliced data are of interest and the header isn't needed.

        Returns
        -------
        slice_obj : `ImageData`
            New `ImageData` object of the sliced subregion

        """

        origin = [sly.start, slx.start]
        NX = slx.stop - slx.start
        NY = sly.stop - sly.start

        # Test dimensions
        if (origin[0] < 0) | (origin[0]+NY > self.sh[0]):
            raise ValueError('Out of range in y')

        if (origin[1] < 0) | (origin[1]+NX > self.sh[1]):
            raise ValueError('Out of range in x')

        # Sliced subimage
        # sly = slice(origin[0], origin[0]+N)
        # slx = slice(origin[1], origin[1]+N)

        slice_origin = [self.origin[i] + origin[i] for i in range(2)]

        slice_wcs = self.get_slice_wcs(self.wcs, slx=slx, sly=sly)
        # slice_wcs = self.wcs.slice((sly, slx))
        #slice_wcs.naxis1 = slice_wcs._naxis1 = NX
        #slice_wcs.naxis2 = slice_wcs._naxis2 = NY

        # Getting the full header can be slow as there appears to
        # be substantial overhead with header.copy() and wcs.to_header()
        if get_slice_header:
            slice_header = self.header.copy()
            slice_header['NAXIS1'] = NX
            slice_header['NAXIS2'] = NY

            # Sliced WCS keywords
            hwcs = slice_wcs.to_header(relax=True)
            for k in hwcs:
                if not k.startswith('PC'):
                    slice_header[k] = hwcs[k]
                else:
                    cd = k.replace('PC', 'CD')
                    slice_header[cd] = hwcs[k]
        else:
            slice_header = pyfits.Header()

        # Generate new object
        if (self.data['REF'] is not None) & (self.data['SCI'] is None):
            _sci = _err = _dq = None
        else:
            _sci = self.data['SCI'][sly, slx]/self.photflam
            _err = self.data['ERR'][sly, slx]/self.photflam
            _dq = self.data['DQ'][sly, slx]*1
            
        slice_obj = ImageData(sci=_sci, err=_err, dq=_dq,
                              header=slice_header, wcs=slice_wcs,
                              photflam=self.photflam, photplam=self.photplam,
                              origin=slice_origin, instrument=self.instrument,
                              filter=self.filter, pupil=self.pupil, 
                              module=self.module,
                              process_jwst_header=False)

        slice_obj.ref_photflam = self.ref_photflam
        slice_obj.ref_photplam = self.ref_photplam
        slice_obj.ref_filter = self.ref_filter

        slice_obj.mdrizsky = self.mdrizsky
        slice_obj.exptime = self.exptime

        slice_obj.ABZP = self.ABZP
        slice_obj.thumb_extension = self.thumb_extension

        if self.data['REF'] is not None:
            slice_obj.data['REF'] = self.data['REF'][sly, slx]*1
        else:
            slice_obj.data['REF'] = None
        
        if 'MED' in self.data:
            slice_obj.data['MED'] = self.data['MED'][sly, slx]*1

        if 'BKG' in self.data:
            slice_obj.data['BKG'] = self.data['BKG'][sly, slx]*1
        
        slice_obj.grow = self.grow
        slice_obj.pad = self.pad
        slice_obj.parent_file = self.parent_file
        slice_obj.ref_file = self.ref_file
        slice_obj.sci_extn = self.sci_extn
        slice_obj.is_slice = True

        # if hasattr(slice_obj.wcs, 'sip'):
        #     if slice_obj.wcs.sip is not None:
        #         for c in [0,1]:
        #             slice_obj.wcs.sip.crpix[c] = slice_obj.wcs.wcs.crpix[c]
        #
        # ACS_CRPIX = [4096/2,2048/2] # ACS
        # dx_crpix = slice_obj.wcs.wcs.crpix[0] - ACS_CRPIX[0]
        # dy_crpix = slice_obj.wcs.wcs.crpix[1] - ACS_CRPIX[1]
        # for ext in ['cpdis1','cpdis2','det2im1','det2im2']:
        #     if hasattr(slice_obj.wcs, ext):
        #         wcs_ext = slice_obj.wcs.__getattribute__(ext)
        #         if wcs_ext is not None:
        #             wcs_ext.crval[0] += dx_crpix
        #             wcs_ext.crval[1] += dy_crpix
        #             slice_obj.wcs.__setattr__(ext, wcs_ext)

        return slice_obj  # , slx, sly

    def get_HDUList(self, extver=1):
        """Convert attributes and data arrays to a `~astropy.io.fits.HDUList`

        Parameters
        ----------
        extver : int, float, str
            value to use for the 'EXTVER' header keyword.  For example, with
            extver=1, the science extension can be addressed with the index
            `HDU['SCI',1]`.

        returns : `~astropy.io.fits.HDUList`
            HDUList with header keywords copied from `self.header` along with
            keywords for additional attributes. Will have `ImageHDU`
            extensions 'SCI', 'ERR', and 'DQ', as well as 'REF' if a reference
            file had been supplied.
        """
        h = self.header.copy()
        h['EXTVER'] = extver  # self.filter #extver
        h['FILTER'] = self.filter, 'element selected from filter wheel'
        h['PUPIL'] = self.pupil, 'element selected from pupil wheel'
        h['INSTRUME'] = (self.instrument,
                         'identifier for instrument used to acquire data')
        if self.module is not None:
            h['MODULE'] = self.module, 'Instrument module'

        h['PHOTFLAM'] = (self.photflam,
                         'inverse sensitivity, ergs/cm2/Ang/electron')

        h['PHOTPLAM'] = self.photplam, 'Pivot wavelength (Angstroms)'
        h['PARENT'] = self.parent_file, 'Parent filename'
        h['SCI_EXTN'] = self.sci_extn, 'EXTNAME of the science data'
        h['ISCUTOUT'] = self.is_slice, 'Arrays are sliced from larger image'
        h['ORIGINX'] = self.origin[1], 'Origin from parent image, x'
        h['ORIGINY'] = self.origin[0], 'Origin from parent image, y'
        
        if isinstance(self.pad, int):
            _pad = (self.pad, self.pad)
        else:
            _pad = self.pad
            
        h['PADX'] = (_pad[1], 'Image padding used axis1')
        h['PADY'] = (_pad[0], 'Image padding used axis2')

        hdu = []

        exptime_corr = 1.
        if 'BUNIT' in self.header:
            if self.header['BUNIT'] == 'ELECTRONS':
                exptime_corr = self.exptime

        # Put back into original units
        sci_data = self['SCI']*exptime_corr + self.mdrizsky
        err_data = self['ERR']*exptime_corr

        hdu.append(pyfits.ImageHDU(data=sci_data, header=h,
                                   name='SCI'))
        hdu.append(pyfits.ImageHDU(data=err_data, header=h,
                                   name='ERR'))
        hdu.append(pyfits.ImageHDU(data=self.data['DQ'], header=h, name='DQ'))
        
        if 'MED' in self.data:
            hdu.append(pyfits.ImageHDU(data=self.data['MED'],
                                       header=h, name='MED'))
        
        if 'BKG' in self.data:
            hdu.append(pyfits.ImageHDU(data=self.data['BKG'],
                                       header=h, name='BKG'))

        if self.data['REF'] is not None:
            h['PHOTFLAM'] = self.ref_photflam
            h['PHOTPLAM'] = self.ref_photplam
            h['FILTER'] = self.ref_filter
            h['REF_FILE'] = self.ref_file

            hdu.append(pyfits.ImageHDU(data=self.data['REF'], header=h,
                                       name='REF'))

        hdul = pyfits.HDUList(hdu)

        return hdul


    def __getitem__(self, ext):
        if self.data[ext] is None:
            return None

        if ext == 'REF':
            return self.data['REF']/self.ref_photflam
        elif ext == 'DQ':
            return self.data['DQ']
        else:
            return self.data[ext]/self.photflam


    def get_common_slices(self, other, verify_parent=True):
        """
        Get slices of overlaps between two `ImageData` objects
        """
        if verify_parent:
            if self.parent_file != other.parent_file:
                msg = ('Parent expodures don\'t match!\n' +
                       '   self: {0}\n'.format(self.parent_file) +
                       '  other: {0}\n'.format(other.parent_file))
                raise IOError(msg)

        ll = np.min([self.origin, other.origin], axis=0)
        ur = np.max([self.origin+self.sh, other.origin+other.sh], axis=0)

        # other in self
        lls = np.minimum(other.origin - ll, self.sh)
        urs = np.clip(other.origin + self.sh - self.origin, [0, 0], self.sh)

        # self in other
        llo = np.minimum(self.origin - ll, other.sh)
        uro = np.clip(self.origin + other.sh - other.origin, [0, 0], other.sh)

        self_slice = (slice(lls[0], urs[0]), slice(lls[1], urs[1]))
        other_slice = (slice(llo[0], uro[0]), slice(llo[1], uro[1]))
        return self_slice, other_slice


class GrismFLT(object):
    """Scripts for modeling of individual grism FLT images"""

    def __init__(self, grism_file='', sci_extn=1, direct_file='',
                 pad=(64,256), ref_file=None, ref_ext=0, seg_file=None,
                 shrink_segimage=True, force_grism='G141', verbose=True,
                 process_jwst_header=True):
        """Read FLT files and, optionally, reference/segmentation images.

        Parameters
        ----------
        grism_file : str
            Grism image (optional).
            Empty string or filename of a FITS file that must contain
            extensions ('SCI', `sci_extn`), ('ERR', `sci_extn`), and
            ('DQ', `sci_extn`).  For example, a WFC3/IR "FLT" FITS file.

        sci_extn : int
            EXTNAME of the file to consider.  For WFC3/IR this can only be
            1.  For ACS and WFC3/UVIS, this can be 1 or 2 to specify the two
            chips.

        direct_file : str
            Direct image (optional).
            Empty string or filename of a FITS file that must contain
            extensions ('SCI', `sci_extn`), ('ERR', `sci_extn`), and
            ('DQ', `sci_extn`).  For example, a WFC3/IR "FLT" FITS file.

        pad : int, int
            Padding to add around the periphery of the images to allow
            modeling of dispersed spectra for objects that could otherwise
            fall off of the direct image itself.  Modeling them requires an
            external reference image (`ref_file`) that covers an area larger
            than the individual direct image itself (e.g., a mosaic of a
            survey field).

            For WFC3/IR spectra, the first order spectra reach 248 and 195
            pixels for G102 and G141, respectively, and `pad` could be set
            accordingly if the reference image is large enough.

        ref_file : str or `~astropy.io.fits.ImageHDU`/`~astropy.io.fits.PrimaryHDU`
            Image mosaic to use as the reference image in place of the direct
            image itself.  For example, this could be the deeper image
            drizzled from all direct images taken within a single visit or it
            could be a much deeper/wider image taken separately in perhaps
            even a different filter.

            .. note::
                Assumes that the WCS are aligned between `grism_file`,
                `direct_file` and `ref_file`!

        ref_ext : int
            FITS extension to use if `ref_file` is a filename string.

        seg_file : str or `~astropy.io.fits.ImageHDU`/`~astropy.io.fits.PrimaryHDU`
            Segmentation image mosaic to associate pixels with discrete
            objects.  This would typically be generated from a rectified
            image like `ref_file`, though here it is not required that
            `ref_file` and `seg_file` have the same image dimensions but
            rather just that the WCS are aligned between them.

        shrink_segimage : bool
            Try to make a smaller cutout of the reference images to speed
            up blotting and array copying.  This is most helpful for very
            large input mosaics.

        force_grism : str
            Use this grism in "simulation mode" where only `direct_file` is
            specified.

        verbose : bool
            Print status messages to the terminal.

        Attributes
        ----------
        grism, direct : `ImageData`
            Grism and direct image data and parameters

        conf : `~grizli.grismconf.aXeConf`
            Grism configuration object.

        seg : array-like
            Segmentation image array.

        model : array-like
            Model of the grism exposure with the same dimensions as the
            full detector array.

        object_dispersers : dict
            Container for storing information about what objects have been
            added to the model of the grism exposure

        catalog : `~astropy.table.Table`
            Associated photometric catalog.  Not required.

        """
        import stwcs.wcsutil

        # Read files
        self.grism_file = grism_file
        _GRISM_OPEN = False
        if os.path.exists(grism_file):
            grism_im = pyfits.open(grism_file)
            _GRISM_OPEN = True

            if grism_im[0].header['INSTRUME'] == 'ACS':
                wcs = stwcs.wcsutil.HSTWCS(grism_im, ext=('SCI', sci_extn))
            else:
                wcs = None

            self.grism = ImageData(hdulist=grism_im, sci_extn=sci_extn,
                                   wcs=wcs,
                                   process_jwst_header=process_jwst_header)
        else:
            if (grism_file is None) | (grism_file == ''):
                self.grism = None
            else:
                print('\nFile not found: {0}!\n'.format(grism_file))
                raise IOError

        self.direct_file = direct_file
        _DIRECT_OPEN = False
        if os.path.exists(direct_file):
            direct_im = pyfits.open(direct_file)
            _DIRECT_OPEN = True
            if direct_im[0].header['INSTRUME'] == 'ACS':
                wcs = stwcs.wcsutil.HSTWCS(direct_im, ext=('SCI', sci_extn))
            else:
                wcs = None

            self.direct = ImageData(hdulist=direct_im, sci_extn=sci_extn,
                                    wcs=wcs,
                                    process_jwst_header=process_jwst_header)
        else:
            if (direct_file is None) | (direct_file == ''):
                self.direct = None
            else:
                print('\nFile not found: {0}!\n'.format(direct_file))
                raise IOError

        ### Simulation mode, no grism exposure
        if isinstance(pad, int):
            self.pad = [pad, pad]
        else:
            self.pad = pad
        
        if self.grism is not None:
            if np.max(self.grism.pad) > 0:
                self.pad = self.grism.pad
        
        if (self.grism is None) & (self.direct is not None):
            self.grism = ImageData(hdulist=direct_im, sci_extn=sci_extn)
            self.grism_file = self.direct_file
            self.grism.filter = force_grism

        # Grism exposure only, assumes will get reference from ref_file
        if (self.direct is None) & (self.grism is not None):
            self.direct = ImageData(hdulist=grism_im, sci_extn=sci_extn)
            self.direct_file = self.grism_file

        # Add padding
        if self.direct is not None:
            if np.max(self.pad) > 0:
                self.direct.add_padding(self.pad)

            self.direct.unset_dq()
            nbad = self.direct.flag_negative(sigma=-3)
            self.direct.data['SCI'] *= (self.direct.data['DQ'] == 0)
            self.direct.data['SCI'] *= (self.direct.data['ERR'] > 0)

        if self.grism is not None:
            if np.max(self.pad) > 0:
                self.grism.add_padding(self.pad)
                self.pad = self.grism.pad

            self.grism.unset_dq()
            nbad = self.grism.flag_negative(sigma=-3)
            self.grism.data['SCI'] *= (self.grism.data['DQ'] == 0)
            self.grism.data['SCI'] *= (self.grism.data['ERR'] > 0)

        # Load data from saved model files, if available
        # if os.path.exists('%s_model.fits' %(self.grism_file)):
        #     pass

        # Holder for the full grism model array
        self.model = np.zeros_like(self.direct.data['SCI'])

        # Grism configuration
        
        if self.grism.instrument in ['NIRCAM', 'NIRISS']:
            direct_filter = self.grism.pupil
        elif 'DFILTER' in self.grism.header:
            direct_filter = self.grism.header['DFILTER']
        else:
            direct_filter = self.direct.filter
        
        conf_args = dict(instrume=self.grism.instrument, 
                         filter=direct_filter, 
                         grism=self.grism.filter,
                         module=self.grism.module,
                         chip=self.grism.ccdchip)
        
        self.conf_file = grismconf.get_config_filename(**conf_args)
        self.conf = grismconf.load_grism_config(self.conf_file)

        self.object_dispersers = OrderedDict()

        # Blot reference image
        self.process_ref_file(ref_file, ref_ext=ref_ext,
                              shrink_segimage=shrink_segimage,
                              verbose=verbose)

        # Blot segmentation image
        self.process_seg_file(seg_file, shrink_segimage=shrink_segimage,
                              verbose=verbose)

        # End things
        self.get_dispersion_PA()

        self.catalog = None
        self.catalog_file = None

        self.is_rotated = False
        self.has_edge_mask = False
        
        # Cleanup
        if _GRISM_OPEN:
            grism_im.close()
        
        if _DIRECT_OPEN:
            direct_im.close()


    def process_ref_file(self, ref_file, ref_ext=0, shrink_segimage=True,
                         verbose=True):
        """Read and blot a reference image

        Parameters
        ----------
        ref_file : str or `~astropy.fits.io.ImageHDU` / `~astropy.fits.io.PrimaryHDU`
            Filename or `astropy.io.fits` Image HDU of the reference image.

        shrink_segimage : bool
            Try to make a smaller cutout of the reference image to speed
            up blotting and array copying.  This is most helpful for very
            large input mosaics.

        verbose : bool
            Print some status information to the terminal

        Returns
        -------
        status : bool
            False if `ref_file` is None.  True if completes successfully.

        The blotted reference image is stored in the array attribute
        `self.direct.data['REF']`.

        The `ref_filter` attribute is determined from the image header and the
        `ref_photflam` scaling is taken either from the header if possible, or
        the global `photflam` variable defined at the top of this file.
        """
        if ref_file is None:
            return False

        if (isinstance(ref_file, pyfits.ImageHDU) |
             isinstance(ref_file, pyfits.PrimaryHDU)):
            self.ref_file = ref_file.fileinfo()['file'].name
            ref_str = ''
            ref_hdu = ref_file

            _IS_OPEN = False
        else:
            self.ref_file = ref_file
            ref_str = '{0}[0]'.format(self.ref_file)
            
            _IS_OPEN = True
            ref_im = pyfits.open(ref_file, load_lazy_hdus=False)
            ref_hdu = ref_im[ref_ext]
        
        refh = ref_hdu.header
            
        if shrink_segimage:
            ref_hdu = self.direct.shrink_large_hdu(ref_hdu,
                                                   extra=np.max(self.pad),
                                                   verbose=True)

        if verbose:
            msg = '{0} / blot reference {1}'
            print(msg.format(self.direct_file, ref_str))

        blotted_ref = self.grism.blot_from_hdu(hdu=ref_hdu,
                                      segmentation=False, interp='poly5')

        header_values = {}
        self.direct.ref_filter = utils.parse_filter_from_header(refh)
        self.direct.ref_file = ref_str

        key_list = {'PHOTFLAM': photflam_list, 'PHOTPLAM': photplam_list}
        for key in ['PHOTFLAM', 'PHOTPLAM']:
            if key in refh:
                try:
                    header_values[key] = ref_hdu.header[key]*1.
                except TypeError:
                    msg = 'Problem processing header keyword {0}: ** {1} **'
                    print(msg.format(key, ref_hdu.header[key]))
                    raise TypeError
            else:
                filt = self.direct.ref_filter
                if filt in key_list[key]:
                    header_values[key] = key_list[key][filt]
                else:
                    msg = 'Filter "{0}" not found in {1} tabulated list'
                    print(msg.format(filt, key))
                    raise IndexError

        # Found keywords
        self.direct.ref_photflam = header_values['PHOTFLAM']
        self.direct.ref_photplam = header_values['PHOTPLAM']

        # TBD: compute something like a cross-correlation offset
        # between blotted reference and the direct image itself
        self.direct.data['REF'] = np.cast[np.float32](blotted_ref)
        # print self.direct.data['REF'].shape, self.direct.ref_photflam

        self.direct.data['REF'] *= self.direct.ref_photflam

        # Fill empty pixels in the reference image from the SCI image,
        # but don't do it if direct['SCI'] is just a copy from the grism
        # if not self.direct.filter.startswith('G'):
        #     empty = self.direct.data['REF'] == 0
        #     self.direct.data['REF'][empty] += self.direct['SCI'][empty]

        # self.direct.data['ERR'] *= 0.
        # self.direct.data['DQ'] *= 0
        self.direct.ABZP = (0*np.log10(self.direct.ref_photflam) - 21.10 -
                      5*np.log10(self.direct.ref_photplam) + 18.6921)

        self.direct.thumb_extension = 'REF'
        
        if _IS_OPEN:
            ref_im.close()
            
        # refh['FILTER'].upper()
        return True

    def process_seg_file(self, seg_file, shrink_segimage=True, verbose=True):
        """Read and blot a rectified segmentation image

        Parameters
        ----------
        seg_file : str or `~astropy.fits.io.ImageHDU` / `~astropy.fits.io.PrimaryHDU`
            Filename or `astropy.io.fits` Image HDU of the segmentation image.

        shrink_segimage : bool
            Try to make a smaller cutout of the segmentation image to speed
            up blotting and array copying.  This is most helpful for very
            large input mosaics.

        verbose : bool
            Print some status information to the terminal

        Returns
        -------
        The blotted segmentation image is stored in the attribute `GrismFLT.seg`.

        """
        if seg_file is not None:
            if (isinstance(seg_file, pyfits.ImageHDU) |
                 isinstance(seg_file, pyfits.PrimaryHDU)):
                self.seg_file = ''
                seg_str = ''
                seg_hdu = seg_file
                segh = seg_hdu.header
                _IS_OPEN = False
            else:
                self.seg_file = seg_file
                seg_str = '{0}[0]'.format(self.seg_file)
                seg_im = pyfits.open(seg_file)
                seg_hdu = seg_im[0]
                _IS_OPEN = True

            if shrink_segimage:
                seg_hdu = self.direct.shrink_large_hdu(seg_hdu,
                                                       extra=np.max(self.pad),
                                                       verbose=True)

                # Make sure image big enough
                seg_hdu = self.direct.expand_hdu(seg_hdu)

            if verbose:
                msg = '{0} / blot segmentation {1}'
                print(msg.format(self.direct_file, seg_str))

            blotted_seg = self.grism.blot_from_hdu(hdu=seg_hdu,
                                          segmentation=True, grow=3,
                                          interp='poly5')
            self.seg = blotted_seg
            
            if _IS_OPEN:
                seg_im.close()
                
        else:
            self.seg = np.zeros(self.direct.sh, dtype=np.float32)

    def get_dispersion_PA(self, decimals=0):
        """Compute exact PA of the dispersion axis, including tilt of the
        trace and the FLT WCS

        Parameters
        ----------
        decimals : int or None
            Number of decimal places to round to, passed to `~numpy.round`.
            If None, then don't round.

        Returns
        -------
        dispersion_PA : float
            PA (angle East of North) of the dispersion axis.
        """
        from astropy.coordinates import Angle
        import astropy.units as u

        # extra tilt of the 1st order grism spectra
        if 'BEAMA' in self.conf.conf_dict:
            x0 = self.conf.conf_dict['BEAMA']
        else:
            x0 = np.array([10,30])
        
        dy_trace, lam_trace = self.conf.get_beam_trace(x=507, y=507, dx=x0,
                                                       beam='A')

        extra = np.arctan2(dy_trace[1]-dy_trace[0], x0[1]-x0[0])/np.pi*180

        # Distorted WCS
        crpix = self.direct.wcs.wcs.crpix
        xref = [crpix[0], crpix[0]+1]
        yref = [crpix[1], crpix[1]]
        r, d = self.direct.wcs.all_pix2world(xref, yref, 1)
        pa = Angle((extra +
                    np.arctan2(np.diff(r)*np.cos(d[0]/180*np.pi),
                               np.diff(d))[0]/np.pi*180)*u.deg)

        dispersion_PA = pa.wrap_at(360*u.deg).value
        if decimals is not None:
            dispersion_PA = np.round(dispersion_PA, decimals=decimals)

        self.dispersion_PA = dispersion_PA
        return float(dispersion_PA)

    def compute_model_orders(self,
                             id=0,
                             x=None,
                             y=None,
                             size=10,
                             mag=-1,
                             spectrum_1d=None,
                             is_cgs=False,
                             compute_size=False,
                             max_size=None,
                             min_size=26,
                             store=True,
                             in_place=True,
                             get_beams=None,
                             psf_params=None,
                             verbose=True):
        """Compute dispersed spectrum for a given object id

        Parameters
        ----------
        id : int
            Object ID number to match in the segmentation image

        x, y : float
            Center of the cutout to extract

        size : int
            Radius of the cutout to extract.  The cutout is equivalent to

            >>> xc, yc = int(x), int(y)
            >>> thumb = self.direct.data['SCI'][yc-size:yc+size, xc-size:xc+size]

        mag : float
            Specified object magnitude, which will be compared to the
            "MMAG_EXTRACT_[BEAM]" parameters in `self.conf` to decide if the
            object is bright enough to compute the higher spectral orders.
            Default of -1 means compute all orders listed in `self.conf.beams`

        spectrum_1d : None or [`~numpy.array`, `~numpy.array`]
            Template 1D spectrum to convolve with the grism disperser.  If
            None, assumes trivial spectrum flat in f_lambda flux densities.
            Otherwise, the template is taken to be

            >>> wavelength, flux = spectrum_1d

        is_cgs : bool
            Flux units of `spectrum_1d[1]` are cgs f_lambda flux densities,
            rather than normalized in the detection band.

        compute_size : bool
            Ignore `x`, `y`, and `size` and compute the extent of the
            segmentation polygon directly using
            `utils_c.disperse.compute_segmentation_limits`.

        max_size : int or None
            Enforce a maximum size of the cutout when using `compute_size`.

        store : bool
            If True, then store the computed beams in the OrderedDict
            `self.object_dispersers[id]`.

            If many objects are computed, this can be memory intensive. To
            save memory, set to False and then the function just stores the
            input template spectrum (`spectrum_1d`) and the beams will have
            to be recomputed if necessary.

        in_place : bool
            If True, add the computed spectral orders into `self.model`.
            Otherwise, make a clean array with only the orders of the given
            object.
        
        get_beams : list or None
            Spectral orders to retrieve with names as defined in the 
            configuration files, e.g., ['A'] generally for the +1st order of 
            HST grisms.  If `None`, then get all orders listed in the 
            `beams` attribute of the `~grizli.grismconf.aXeConf`
            configuration object.
        
        psf_params : list
            Optional parameters for generating an `~grizli.utils.EffectivePSF`
            object for the spatial morphology.
            
        Returns
        -------
        output : bool or `numpy.array`
            If `in_place` is True, return status of True if everything goes
            OK. The computed spectral orders are stored in place in
            `self.model`.

            Returns False if the specified `id` is not found in the
            segmentation array independent of `in_place`.

            If `in_place` is False, return a full array including the model
            for the single object.
        """
        from .utils_c import disperse

        if id in self.object_dispersers:
            object_in_model = True
            beams = self.object_dispersers[id]

            out = self.object_dispersers[id]

            # Handle pre 0.3.0-7 formats
            if len(out) == 3:
                old_cgs, old_spectrum_1d, beams = out
            else:
                old_cgs, old_spectrum_1d = out
                beams = None

        else:
            object_in_model = False
            beams = None

        if self.direct.data['REF'] is None:
            ext = 'SCI'
        else:
            ext = 'REF'

        # set up the beams to extract
        if get_beams is None:
            beam_names = self.conf.beams
        else:
            beam_names = get_beams

        # Did we initialize the PSF model this call?
        INIT_PSF_NOW = False

        # Do we need to compute the dispersed beams?
        if beams is None:
            # Use catalog
            xcat = ycat = None
            if self.catalog is not None:
                ix = self.catalog['id'] == id
                if ix.sum() == 0:
                    if verbose:
                        print(f'ID {id} not found in segmentation image')
                    return False
                
                if hasattr(self.catalog['x_flt'][ix][0], 'unit'):
                    xcat = self.catalog['x_flt'][ix][0].value - 1
                    ycat = self.catalog['y_flt'][ix][0].value - 1
                else:
                    xcat = self.catalog['x_flt'][ix][0] - 1
                    ycat = self.catalog['y_flt'][ix][0] - 1

                # print '!!! X, Y: ', xcat, ycat, self.direct.origin, size

                # use x, y if defined
                if x is not None:
                    xcat = x
                if y is not None:
                    ycat = y

            if (compute_size) | (x is None) | (y is None) | (size is None):
                # Get the array indices of the segmentation region
                out = disperse.compute_segmentation_limits(self.seg, id,
                                         self.direct.data[ext],
                                         self.direct.sh)

                ymin, ymax, y, xmin, xmax, x, area, segm_flux = out
                if (area == 0) | ~np.isfinite(x) | ~np.isfinite(y):
                    if verbose:
                        print('ID {0:d} not found in segmentation image'.format(id))
                    return False

                # Object won't disperse spectrum onto the grism image
                if ((ymax < self.pad[0]-5) |
                     (ymin > self.direct.sh[0]-self.pad[0]+5) |
                     (ymin == 0) |
                     (ymax == self.direct.sh[0]) |
                     (xmin == 0) |
                     (xmax == self.direct.sh[1])):
                    return True

                if compute_size:
                    try:
                        size = int(np.ceil(np.max([x-xmin, xmax-x,
                                                   y-ymin, ymax-y])))
                    except ValueError:
                        return False

                    size += 4

                    # Enforce minimum size
                    # size = np.maximum(size, 16)
                    size = np.maximum(size, min_size)
                    
                    # To do: enforce a larger minimum cutout size for grisms 
                    # that need it, e.g., UVIS/G280L
                    
                    # maximum size
                    if max_size is not None:
                        size = np.min([size, max_size])

                    # Avoid problems at the array edges
                    size = np.min([size, int(x)-2, int(y)-2])

                    if (size < 4):
                        return True

            # Thumbnails
            # print '!! X, Y: ', x, y, self.direct.origin, size

            if xcat is not None:
                xc, yc = int(np.round(xcat))+1, int(np.round(ycat))+1
                xcenter = (xcat-(xc-1))
                ycenter = (ycat-(yc-1))
            else:
                xc, yc = int(np.round(x))+1, int(np.round(y))+1
                xcenter = (x-(xc-1))
                ycenter = (y-(yc-1))

            origin = [yc-size + self.direct.origin[0],
                      xc-size + self.direct.origin[1]]

            thumb = self.direct.data[ext][yc-size:yc+size, xc-size:xc+size]
            seg_thumb = self.seg[yc-size:yc+size, xc-size:xc+size]

            # Test that the id is actually in the thumbnail
            test = disperse.compute_segmentation_limits(seg_thumb, id, thumb,
                                                        np.array(thumb.shape))
            if test[-2] == 0:
                if verbose:
                    print(f'ID {id} not found in segmentation image')
                return False

            # # Get precomputed dispersers
            # beams, old_spectrum_1d, old_cgs = None, None, False
            # if object_in_model:
            #     out = self.object_dispersers[id]
            #
            #     # Handle pre 0.3.0-7 formats
            #     if len(out) == 3:
            #         old_cgs, old_spectrum_1d, old_beams = out
            #     else:
            #         old_cgs, old_spectrum_1d = out
            #         old_beams = None
            #
            #     # Pull out just the requested beams
            #     if old_beams is not None:
            #         beams = OrderedDict()
            #         for b in beam_names:
            #             beams[b] = old_beams[b]
            #
            # if beams is None:

            # Compute spectral orders ("beams")
            beams = OrderedDict()

            for b in beam_names:
                # Only compute order if bright enough
                if mag > self.conf.conf_dict['MMAG_EXTRACT_{0}'.format(b)]:
                    continue

                try:
                    beam = GrismDisperser(id=id,
                                          direct=thumb, 
                                          segmentation=seg_thumb, 
                                          xcenter=xcenter,
                                          ycenter=ycenter,
                                          origin=origin,
                                          pad=self.pad,
                                          grow=self.grism.grow,
                                          beam=b,
                                          conf=self.conf,
                                          fwcpos=self.grism.fwcpos,
                                          MW_EBV=self.grism.MW_EBV)
                except: 
                    utils.log_exception(utils.LOGFILE, traceback)
                    
                    continue

                # Set PSF model if necessary
                if psf_params is not None:
                    store = True
                    INIT_PSF_NOW = True
                    if self.direct.ref_filter is None:
                        psf_filter = self.direct.filter
                    else:
                        psf_filter = self.direct.ref_filter

                    beam.x_init_epsf(flat_sensitivity=False, 
                                     psf_params=psf_params, 
                                     psf_filter=psf_filter, yoff=0.)

                beams[b] = beam

        # Compute old model
        if object_in_model:
            for b in beams:
                beam = beams[b]
                if hasattr(beam, 'psf') & (not INIT_PSF_NOW):
                    store = True
                    beam.compute_model_psf(spectrum_1d=old_spectrum_1d,
                                       is_cgs=old_cgs)
                else:
                    beam.compute_model(spectrum_1d=old_spectrum_1d,
                                       is_cgs=old_cgs)

        if get_beams:
            out_beams = OrderedDict()
            for b in beam_names:
                out_beams[b] = beams[b]
            return out_beams

        if in_place:
            # Update the internal model attribute
            output = self.model

            if store:
                # Save the computed beams
                self.object_dispersers[id] = is_cgs, spectrum_1d, beams
            else:
                # Just save the model spectrum (or empty spectrum)
                self.object_dispersers[id] = is_cgs, spectrum_1d, None
        else:
            # Create a fresh array
            output = np.zeros_like(self.model)

        # if in_place:
        #     ### Update the internal model attribute
        #     output = self.model
        # else:
        #     ### Create a fresh array
        #     output = np.zeros_like(self.model)

        # Set PSF model if necessary
        if psf_params is not None:
            if self.direct.ref_filter is None:
                psf_filter = self.direct.filter
            else:
                psf_filter = self.direct.ref_filter

        # Loop through orders and add to the full model array, in-place or
        # a separate image
        for b in beams:
            beam = beams[b]
            # Subtract previously-added model
            if object_in_model & in_place:
                beam.add_to_full_image(-beam.model, output)

            # Update PSF params
            # if psf_params is not None:
            #     skip_init_psf = False
            #     if hasattr(beam, 'psf_params'):
            #         skip_init_psf |= np.product(np.isclose(beam.psf_params, psf_params)) > 0
            #
            #     if not skip_init_psf:
            #         beam.x_init_epsf(flat_sensitivity=False, psf_params=psf_params, psf_filter=psf_filter, yoff=0.06)

            # Compute model
            if hasattr(beam, 'psf'):
                beam.compute_model_psf(spectrum_1d=spectrum_1d, is_cgs=is_cgs)
            else:
                beam.compute_model(spectrum_1d=spectrum_1d, is_cgs=is_cgs)

            # Add in new model
            beam.add_to_full_image(beam.model, output)

        if in_place:
            return True
        else:
            return beams, output


    def compute_full_model(self, ids=None, mags=None, mag_limit=22, store=True, verbose=False, size=10, min_size=26, compute_size=True):
        """Compute flat-spectrum model for multiple objects.

        Parameters
        ----------
        ids : None, list, or `~numpy.array`
            id numbers to compute in the model.  If None then take all ids
            from unique values in `self.seg`.

        mags : None, float, or list / `~numpy.array`
            magnitudes corresponding to list if `ids`.  If None, then compute
            magnitudes based on the flux in segmentation regions and
            zeropoints determined from PHOTFLAM and PHOTPLAM.
        
        size, compute_size : int, bool
            Sizes of individual cutouts, see 
            `~grizli.model.GrismFLT.compute_model_orders`.
            
        Returns
        -------
        Updated model stored in `self.model` attribute.
        """
        try:
            from tqdm import tqdm
            has_tqdm = True
        except:
            has_tqdm = False
            print('(`pip install tqdm` for a better verbose iterator)')
            
        from .utils_c import disperse

        if ids is None:
            ids = np.unique(self.seg)[1:]

        # If `mags` array not specified, compute magnitudes within
        # segmentation regions.
        if mags is None:
            if verbose:
                print('Compute IDs/mags')

            mags = np.zeros(len(ids))
            for i, id in enumerate(ids):
                out = disperse.compute_segmentation_limits(self.seg, id,
                                self.direct.data[self.direct.thumb_extension],
                                     self.direct.sh)

                ymin, ymax, y, xmin, xmax, x, area, segm_flux = out
                mags[i] = self.direct.ABZP - 2.5*np.log10(segm_flux)

            ix = mags < mag_limit
            ids = ids[ix]
            mags = mags[ix]

        else:
            if np.isscalar(mags):
                mags = [mags for i in range(len(ids))]
            else:
                if len(ids) != len(mags):
                    raise ValueError('`ids` and `mags` lists different sizes')

        # Now compute the full model
        if verbose & has_tqdm:
            iterator = tqdm(zip(ids, mags))
        else:
            iterator = zip(ids, mags)
            
        for id_i, mag_i in iterator:
            self.compute_model_orders(id=id_i, compute_size=compute_size,
                                      mag=mag_i, size=size,
                                      in_place=True, store=store, 
                                      min_size=min_size)


    def smooth_mask(self, gaussian_width=4, threshold=2.5):
        """Compute a mask where smoothed residuals greater than some value

        Perhaps useful for flagging contaminated pixels that aren't in the
        model, such as high orders dispersed from objects that fall off of the
        direct image, but this hasn't yet been extensively tested.

        Parameters
        ----------
        gaussian_width : float
            Width of the Gaussian filter used with 
            `~scipy.ndimage.gaussian_filter`.

        threshold : float
            Threshold, in sigma, above which to flag residuals.

        Returns
        -------
        Nothing, but pixels are masked in `self.grism.data['SCI']`.
        """
        import scipy.ndimage as nd

        mask = self.grism['SCI'] != 0
        resid = (self.grism['SCI'] - self.model)*mask
        sm = nd.gaussian_filter(np.abs(resid), gaussian_width)
        resid_mask = (np.abs(sm) > threshold*self.grism['ERR'])
        self.grism.data['SCI'][resid_mask] = 0

    def blot_catalog(self, input_catalog, columns=['id', 'ra', 'dec'],
                     sextractor=False, ds9=None):
        """Compute detector-frame coordinates of sky positions in a catalog.

        Parameters
        ----------
        input_catalog : `~astropy.table.Table`
            Full catalog with sky coordinates.  Can be SExtractor or other.

        columns : [str,str,str]
            List of columns that specify the object id, R.A. and Decl.  For
            catalogs created with SExtractor this might be
            ['NUMBER', 'X_WORLD', 'Y_WORLD'].

            Detector coordinates will be computed with
            `self.direct.wcs.all_world2pix` with `origin=1`.

        ds9 : `~grizli.ds9.DS9`, optional
            If provided, load circular regions at the derived detector
            coordinates.

        Returns
        -------
        catalog : `~astropy.table.Table`
            New catalog with columns 'x_flt' and 'y_flt' of the detector
            coordinates.  Also will copy the `columns` names to columns with
            names 'id','ra', and 'dec' if necessary, e.g., for SExtractor
            catalogs.

        """
        from astropy.table import Column

        if sextractor:
            columns = ['NUMBER', 'X_WORLD', 'Y_WORLD']

        # Detector coordinates.  N.B.: 1 indexed!
        xy = self.direct.wcs.all_world2pix(input_catalog[columns[1]],
                                           input_catalog[columns[2]], 1,
                                           tolerance=-4,
                                           quiet=True)

        # Objects with positions within the image
        sh = self.direct.sh
        keep = ((xy[0] > 0) & (xy[0] < sh[1]) &
                (xy[1] > (self.pad[0]-5)) & (xy[1] < (sh[0]-self.pad[0]+5)))

        catalog = input_catalog[keep]

        # Remove columns if they exist
        for col in ['x_flt', 'y_flt']:
            if col in catalog.colnames:
                catalog.remove_column(col)

        # Columns with detector coordinates
        catalog.add_column(Column(name='x_flt', data=xy[0][keep]))
        catalog.add_column(Column(name='y_flt', data=xy[1][keep]))

        # Copy standardized column names if necessary
        if ('id' not in catalog.colnames):
            catalog.add_column(Column(name='id', data=catalog[columns[0]]))

        if ('ra' not in catalog.colnames):
            catalog.add_column(Column(name='ra', data=catalog[columns[1]]))

        if ('dec' not in catalog.colnames):
            catalog.add_column(Column(name='dec', data=catalog[columns[2]]))

        # Show positions in ds9
        if ds9:
            for i in range(len(catalog)):
                x_flt, y_flt = catalog['x_flt'][i], catalog['y_flt'][i]
                reg = 'circle {0:f} {1:f} 5\n'.format(x_flt, y_flt)
                ds9.set('regions', reg)

        return catalog

    def photutils_detection(self, use_seg=False, data_ext='SCI',
                            detect_thresh=2., grow_seg=5, gauss_fwhm=2.,
                            verbose=True, save_detection=False, ZP=None):
        """Use photutils to detect objects and make segmentation map

        Parameters
        ----------
        detect_thresh : float
            Detection threshold, in sigma

        grow_seg : int
            Number of pixels to grow around the perimeter of detected objects
            witha  maximum filter

        gauss_fwhm : float
            FWHM of Gaussian convolution kernel that smoothes the detection
            image.

        verbose : bool
            Print logging information to the terminal

        save_detection : bool
            Save the detection images and catalogs

        ZP : float or None
            AB magnitude zeropoint of the science array.  If `None` then, try
            to compute based on PHOTFLAM and PHOTPLAM values and use zero if
            that fails.

        Returns
        -------
        status : bool
            True if completed successfully.  False if `data_ext=='REF'` but
            no reference image found.

        Stores an astropy.table.Table object to `self.catalog` and a
        segmentation array to `self.seg`.

        """
        if ZP is None:
            if ((self.direct.filter in photflam_list.keys()) &
                 (self.direct.filter in photplam_list.keys())):
                # ABMAG_ZEROPOINT from
                # http://www.stsci.edu/hst/wfc3/phot_zp_lbn
                ZP = (-2.5*np.log10(photflam_list[self.direct.filter]) -
                       21.10 - 5*np.log10(photplam_list[self.direct.filter]) +
                       18.6921)
            else:
                ZP = 0.

        if use_seg:
            seg = self.seg
        else:
            seg = None

        if self.direct.data['ERR'].max() != 0.:
            err = self.direct.data['ERR']/self.direct.photflam
        else:
            err = None

        if (data_ext == 'REF'):
            if (self.direct.data['REF'] is not None):
                err = None
            else:
                print('No reference data found for `self.direct.data[\'REF\']`')
                return False

        go_detect = utils.detect_with_photutils
        cat, seg = go_detect(self.direct.data[data_ext]/self.direct.photflam,
                             err=err, dq=self.direct.data['DQ'], seg=seg,
                             detect_thresh=detect_thresh, npixels=8,
                             grow_seg=grow_seg, gauss_fwhm=gauss_fwhm,
                             gsize=3, wcs=self.direct.wcs,
                             save_detection=save_detection,
                             root=self.direct_file.split('.fits')[0],
                             background=None, gain=None, AB_zeropoint=ZP,
                             overwrite=True, verbose=verbose)

        self.catalog = cat
        self.catalog_file = '<photutils>'

        self.seg = seg

        return True

    def load_photutils_detection(self, seg_file=None, seg_cat=None,
                                 catalog_format='ascii.commented_header'):
        """
        Load segmentation image and catalog, either from photutils
        or SExtractor.

        If SExtractor, use `catalog_format='ascii.sextractor'`.

        """
        root = self.direct_file.split('.fits')[0]

        if seg_file is None:
            seg_file = root + '.detect_seg.fits'

        if not os.path.exists(seg_file):
            print('Segmentation image {0} not found'.format(seg_file))
            return False
        
        with pyfits.open(seg_file) as seg_im: 
            self.seg = seg_im[0].data.astype(np.float32)

        if seg_cat is None:
            seg_cat = root + '.detect.cat'

        if not os.path.exists(seg_cat):
            print('Segmentation catalog {0} not found'.format(seg_cat))
            return False

        self.catalog = Table.read(seg_cat, format=catalog_format)
        self.catalog_file = seg_cat

    def save_model(self, overwrite=True, verbose=True):
        """Save model properties to FITS file
        """
        try:
            import cPickle as pickle
        except:
            # Python 3
            import pickle

        root = self.grism_file.split('_flt.fits')[0].split('_rate.fits')[0]
        root = root.split('_elec.fits')[0]
        
        if isinstance(self.pad, int):
            _pad = (self.pad, self.pad)
        else:
            _pad = self.pad
        
        h = pyfits.Header()
        h['GFILE'] = (self.grism_file, 'Grism exposure name')
        h['GFILTER'] = (self.grism.filter, 'Grism spectral element')
        h['INSTRUME'] = (self.grism.instrument, 'Instrument of grism file')
        h['PADX'] = (_pad[1], 'Image padding used axis1')
        h['PADY'] = (_pad[0], 'Image padding used axis2')
        h['DFILE'] = (self.direct_file, 'Direct exposure name')
        h['DFILTER'] = (self.direct.filter, 'Grism spectral element')
        h['REF_FILE'] = (self.ref_file, 'Reference image')
        h['SEG_FILE'] = (self.seg_file, 'Segmentation image')
        h['CONFFILE'] = (self.conf_file, 'Configuration file')
        h['DISP_PA'] = (self.dispersion_PA, 'Dispersion position angle')

        h0 = pyfits.PrimaryHDU(header=h)
        model = pyfits.ImageHDU(data=self.model, header=self.grism.header,
                                name='MODEL')

        seg = pyfits.ImageHDU(data=self.seg, header=self.grism.header,
                              name='SEG')

        hdu = pyfits.HDUList([h0, model, seg])

        if 'REF' in self.direct.data:
            ref_header = self.grism.header.copy()
            ref_header['FILTER'] = self.direct.ref_filter
            ref_header['PARENT'] = self.ref_file
            ref_header['PHOTFLAM'] = self.direct.ref_photflam
            ref_header['PHOTPLAM'] = self.direct.ref_photplam

            ref = pyfits.ImageHDU(data=self.direct['REF'],
                                  header=ref_header, name='REFERENCE')

            hdu.append(ref)

        hdu.writeto('{0}_model.fits'.format(root), overwrite=overwrite,
                    output_verify='fix')

        fp = open('{0}_model.pkl'.format(root), 'wb')
        pickle.dump(self.object_dispersers, fp)
        fp.close()

        if verbose:
            print('Saved {0}_model.fits and {0}_model.pkl'.format(root))

    def save_full_pickle(self, verbose=True):
        """Save entire `GrismFLT` object to a pickle
        """
        try:
            import cPickle as pickle
        except:
            # Python 3
            import pickle

        root = self.grism_file.split('_flt.fits')[0].split('_cmb.fits')[0]
        root = root.split('_flc.fits')[0].split('_rate.fits')[0]
        root = root.split('_elec.fits')[0]
        
        if root == self.grism_file:
            # unexpected extension, so just insert before '.fits'
            root = self.grism_file.split('.fits')[0]
            
        hdu = pyfits.HDUList([pyfits.PrimaryHDU()])
        
        # Remove dummy extensions if REF found
        skip_direct_extensions  = []
        if 'REF' in self.direct.data:
            if self.direct.data['REF'] is not None:
                skip_direct_extensions = ['SCI','ERR','DQ']
                
        for key in self.direct.data.keys():
            if key in skip_direct_extensions:
                hdu.append(pyfits.ImageHDU(data=None,
                                       header=self.direct.header,
                                       name='D'+key))
            else:
                hdu.append(pyfits.ImageHDU(data=self.direct.data[key],
                                       header=self.direct.header,
                                       name='D'+key))

        for key in self.grism.data.keys():
            hdu.append(pyfits.ImageHDU(data=self.grism.data[key],
                                       header=self.grism.header,
                                       name='G'+key))

        hdu.append(pyfits.ImageHDU(data=self.seg,
                                   header=self.grism.header,
                                   name='SEG'))

        hdu.append(pyfits.ImageHDU(data=self.model,
                                   header=self.grism.header,
                                   name='MODEL'))

        hdu.writeto('{0}.{1:02d}.GrismFLT.fits'.format(root, self.grism.sci_extn), overwrite=True, output_verify='fix')

        # zero out large data objects
        self.direct.data = self.grism.data = self.seg = self.model = None

        fp = open('{0}.{1:02d}.GrismFLT.pkl'.format(root, 
                                                self.grism.sci_extn), 'wb')
        pickle.dump(self, fp)
        fp.close()

        self.save_wcs(overwrite=True, verbose=False)

    def save_wcs(self, overwrite=True, verbose=True):
        """TBD
        """
        if self.direct.parent_file == self.grism.parent_file:
            base_list = [self.grism]
        else:
            base_list = [self.direct, self.grism]

        for base in base_list:
            hwcs = base.wcs.to_fits(relax=True)
            hwcs[0].header['PADX'] = base.pad[1]
            hwcs[0].header['PADY'] = base.pad[0]

            if 'CCDCHIP' in base.header:
                ext = {1: 2, 2: 1}[base.header['CCDCHIP']]
            else:
                ext = base.header['EXTVER']

            wcsfile = base.parent_file.replace('.fits', f'.{ext:02d}.wcs.fits')

            try:
                hwcs.writeto(wcsfile, overwrite=overwrite)
            except:
                hwcs.writeto(wcsfile, clobber=overwrite)

            if verbose:
                print(wcsfile)

    def load_from_fits(self, save_file):
        """Load saved data from a FITS file

        Parameters
        ----------
        save_file : str
            Filename of the saved output

        Returns
        -------
        True if completed successfully
        """
        fits = pyfits.open(save_file)
        self.seg = fits['SEG'].data*1
        self.model = fits['MODEL'].data*1
        self.direct.data = OrderedDict()
        self.grism.data = OrderedDict()

        for ext in range(1, len(fits)):
            key = fits[ext].header['EXTNAME'][1:]

            if fits[ext].header['EXTNAME'].startswith('D'):
                if fits[ext].data is None:
                    self.direct.data[key] = None
                else:
                    self.direct.data[key] = fits[ext].data*1
                    
            elif fits[ext].header['EXTNAME'].startswith('G'):
                if fits[ext].data is None:
                    self.grism.data[key] = None
                else:
                    self.grism.data[key] = fits[ext].data*1
            else:
                pass
        
        fits.close()
        del(fits)

        return True

    def transform_JWST_WFSS(self, verbose=True):
        """
        Rotate data & wcs so that spectra are increasing to +x
        
        # ToDo - do this correctly for the SIP WCS / CRPIX keywords
        
        """

        if self.grism.instrument not in ['NIRCAM', 'NIRISS']:
            return True

        if self.grism.instrument == 'NIRISS':
            if self.grism.filter == 'GR150C':
                rot = 2
            else:
                rot = -1

        elif self.grism.instrument in ['NIRCAM', 'NIRCAMA']:
            if self.grism.module == 'A':
                #  Module A
                if self.grism.pupil == 'GRISMC':
                    rot = 1
                else:
                    # Do nothing, A+GRISMR disperses to +x
                    return True
            else:
                # Module B
                if self.grism.pupil == 'GRISMC':
                    rot = 1
                else:
                    rot = 2
                
        elif self.grism.instrument == 'NIRCAMB':
            if self.grism.pupil == 'GRISMC':
                rot = 1
            else:
                rot = 2

        if self.is_rotated:
            rot *= -1
        
        self.is_rotated = not self.is_rotated
        if verbose:
            print('Transform JWST WFSS: flip={0}'.format(self.is_rotated))

        # Compute new CRPIX coordinates
        # center = np.array(self.grism.sh)/2.+0.5
        # crpix = self.grism.wcs.wcs.crpix
        # 
        # rad = np.deg2rad(-90*rot)
        # mat = np.zeros((2, 2))
        # mat[0, :] = np.array([np.cos(rad), -np.sin(rad)])
        # mat[1, :] = np.array([np.sin(rad), np.cos(rad)])
        # 
        # crpix_new = np.dot(mat, crpix-center)+center
        
        # Full rotated SIP header
        orig_header = utils.to_header(self.grism.wcs)
        hrot, wrot, desc = utils.sip_rot90(orig_header, rot)
        
        for obj in [self.grism, self.direct]:
            
            for k in hrot:
                obj.header[k] = hrot[k]
                
            # obj.header['CRPIX1'] = crpix_new[0]
            # obj.header['CRPIX2'] = crpix_new[1]
            # 
            # # Get rotated CD
            # out_wcs = utils.transform_wcs(obj.wcs, translation=[0., 0.], rotation=rad, scale=1.)
            # new_cd = out_wcs.wcs.cd
            # 
            # for i in range(2):
            #     for j in range(2):
            #         obj.header['CD{0}_{1}'.format(i+1, j+1)] = new_cd[i, j]

            # Update wcs
            obj.get_wcs()
            if obj.wcs.wcs.has_pc():
                obj.get_wcs()

            # Rotate data
            for k in obj.data.keys():
                if obj.data[k] is not None:
                    obj.data[k] = np.rot90(obj.data[k], rot)

        # Rotate segmentation image
        self.seg = np.rot90(self.seg, rot)
        self.model = np.rot90(self.model, rot)

        #print('xx Rotate images {0}'.format(rot))

        if self.catalog is not None:
            #print('xx Rotate catalog {0}'.format(rot))
            self.catalog = self.blot_catalog(self.catalog,
                          sextractor=('X_WORLD' in self.catalog.colnames))
    
    
    def apply_POM(self, warn_if_too_small=True, verbose=True):
        """
        Apply pickoff mask to segmentation map to control sources that are dispersed onto the detector
        """
        if not self.grism.instrument.startswith('NIRCAM'):
            print('POM only defined for NIRCam')
            return True
        
        pom_path = os.path.join(GRIZLI_PATH,
            f'CONF/GRISM_NIRCAM/V*/NIRCAM_LW_POM_Mod{self.grism.module}.fits')
        
        pom_files = glob.glob(pom_path)
        
        if len(pom_files) == 0:
            print(f'Couldn\'t find POM reference files {pom_path}')
            return False
        
        pom_files.sort()
        pom_file = pom_files[-1]
        
        if verbose:
            print(f'NIRCam: apply POM geometry from {pom_file}')
            
        pom = pyfits.open(pom_file)[-1]
        pomh = pom.header
        
        if self.grism.pupil.lower() == 'grismc':
            _warn = self.pad[0] < 790
            _padix = 0
        elif self.grism.pupil.lower() == 'grismr':
            _warn = self.pad[1] < 790
            _padix = 1
        else:
            _warn = False
            
        if _warn & warn_if_too_small:
            print(f'Warning: `pad[{_padix}]` should be > 790 for '
                  f'NIRCam/{self.grism.pupil} to catch '
                  'all out-of-field sources within the POM coverage.')
                  
        # Slice geometry
        a_origin = np.array([-self.pad[0], -self.pad[1]])
        a_shape = np.array(self.grism.sh)

        b_origin = np.array([-pomh['NOMYSTRT'], -pomh['NOMXSTRT']])
        b_shape = np.array(pom.data.shape)

        self_sl, pom_sl = utils.get_common_slices(a_origin, a_shape, 
                                                b_origin, b_shape)
        
        pom_data = self.seg*0
        pom_data[self_sl] += pom.data[pom_sl]
        self.pom_data = pom_data
        self.seg *= (pom_data > 0)
        return True
        
    def mask_mosaic_edges(self, sky_poly=None, verbose=True, force=False, err_scale=10, dq_mask=False, dq_value=1024, resid_sn=7):
        """
        Mask edges of exposures that might not have modeled spectra
        """
        from regions import Regions
        import scipy.ndimage as nd

        if (self.has_edge_mask) & (force is False):
            return True

        if sky_poly is None:
            return True

        xy_image = self.grism.wcs.all_world2pix(np.array(sky_poly.boundary.xy).T, 0)

        # Calculate edge for mask
        #xedge = 100
        x0 = 0
        y0 = (self.grism.sh[0] - 2*self.pad[0])/2
        dx = np.arange(500)
        tr_y, tr_lam = self.conf.get_beam_trace(x0, y0, dx=dx, beam='A')
        tr_sens = np.interp(tr_lam, self.conf.sens['A']['WAVELENGTH'],
                                   self.conf.sens['A']['SENSITIVITY'],
                                   left=0, right=0)

        xedge = dx[tr_sens > tr_sens.max()*0.05].max()

        xy_image[:, 0] += xedge

        xy_str = 'image;polygon('+','.join(['{0:.1f}'.format(p + 1) for p in xy_image.flatten()])+')'
        reg = Regions.parse(xy_str, format='ds9')[0]
        mask = reg.to_mask().to_image(shape=self.grism.sh).astype(bool)

        # Only mask large residuals
        if resid_sn > 0:
            resid_mask = (self.grism['SCI'] - self.model) > resid_sn*self.grism['ERR']
            resid_mask = nd.binary_dilation(resid_mask, iterations=3)
            mask &= resid_mask

        if dq_mask:
            self.grism.data['DQ'] |= dq_value*mask
            if verbose:
                print('# mask mosaic edges: {0} ({1}, {2} pix) DQ={3:.0f}'.format(self.grism.parent_file, self.grism.filter, xedge, dq_value))
        else:
            self.grism.data['ERR'][mask] *= err_scale
            if verbose:
                print('# mask mosaic edges: {0} ({1}, {2} pix) err_scale={3:.1f}'.format(self.grism.parent_file, self.grism.filter, xedge, err_scale))

        self.has_edge_mask = True


    def get_trace_region_from_sky(self, ra, dec, width=2):
        """
        Make a region file for the trace in pixel coordinates given sky position
        TBD
        """
        return None
        
    def old_make_edge_mask(self, scale=3, force=False):
        """Make a mask for the edge of the grism FoV that isn't covered by the direct image

        Parameters
        ----------
        scale : float
            Scale factor to multiply to the mask before it's applied to the
            `self.grism.data['ERR']` array.

        force : bool
             Force apply the mask even if `self.has_edge_mask` is set
             indicating that the function has already been run.

        Returns
        -------
        Nothing, updates `self.grism.data['ERR']` in place.
        Sets `self.has_edge_mask = True`.

        """
        import scipy.ndimage as nd

        if (self.has_edge_mask) & (force is False):
            return True

        kern = (np.arange(self.conf.conf_dict['BEAMA'][1]) > self.conf.conf_dict['BEAMA'][0])*1.
        kern /= kern.sum()

        if self.direct['REF'] is not None:
            mask = self.direct['REF'] == 0
        else:
            mask = self.direct['SCI'] == 0

        full_mask = nd.convolve(mask*1., kern.reshape((1, -1)),
                                origin=(0, -kern.size//2+20))

        self.grism.data['ERR'] *= np.exp(full_mask*scale)

        self.has_edge_mask = True


class BeamCutout(object):
    def __init__(self, flt=None, beam=None, conf=None,
                 get_slice_header=True, fits_file=None, scale=1.,
                 contam_sn_mask=[10, 3], min_mask=0.01, min_sens=0.08,
                 mask_resid=True, isJWST=False):
        """Cutout spectral object from the full frame.

        Parameters
        ----------
        flt : `GrismFLT`
            Parent FLT frame.

        beam : `GrismDisperser`
            Object and spectral order to consider

        conf : `.grismconf.aXeConf`
            Pre-computed configuration file.  If not specified will regenerate
            based on header parameters, which might be necessary for
            multiprocessing parallelization and pickling.

        get_slice_header : bool
            TBD

        fits_file : None or str
            Optional FITS file containing the beam information, rather than
            reading directly from a `GrismFLT` object with the `flt` and
            `beam` paremters.  Load with `load_fits`.

        contam_sn_mask : TBD

        min_mask : float
            Minimum factor relative to the maximum pixel value of the flat
            f-lambda model where the 2D cutout data are considered good.

        min_sens : float
            Minimum sensitivity relative to the maximum for a given grism
            above which pixels are included in the fit.

        Attributes
        ----------
        grism, direct : `ImageData` (sliced)
            Cutouts of the grism and direct images.

        beam : `GrismDisperser`
            High-level tools for computing dispersed models of the object

        mask : array-like (bool)
            Basic mask where `grism` DQ > 0 | ERR == 0 | SCI == 0.

        fit_mask, DoF : array-like, int
            Additional mask, DoF is `fit_mask.sum()` representing the
            effective degrees of freedom for chi-squared.

        ivar : array-like
            Inverse variance array, taken from `grism` 1/ERR^2

        model, modelf : array-like
            2D and flattened versions of the object model array

        contam : array-like
            Contamination model

        scif : array_like
            Flattened version of `grism['SCI'] - contam`.

        flat_flam : array-like
            Flattened version of the flat-flambda object model

        poly_order : int
            Order of the polynomial model
        """
        self.background = 0.
        self.module = None
        
        if fits_file is not None:
            self.load_fits(fits_file, conf)
        else:
            self.init_from_input(flt, beam, conf, get_slice_header)

        self.beam.scale = scale
        
        self._parse_params = {'contam_sn_mask':contam_sn_mask, 
                              'min_mask':min_mask,
                              'min_sens':min_sens,
                              'mask_resid':mask_resid}
                               
        # self.contam_sn_mask = contam_sn_mask
        # self.min_mask = min_mask
        # self.min_sens = min_sens
        # self.mask_resid = mask_resid

        self._parse_from_data(isJWST=isJWST, **self._parse_params)


    def _parse_from_data(self, contam_sn_mask=[10, 3], min_mask=0.01,
                         seg_ids=None, min_sens=0.08, mask_resid=True, isJWST=False):
        """
        See parameter description for `~grizli.model.BeamCutout`.
        """
        # bad pixels or problems with uncertainties
        self.mask = ((self.grism.data['DQ'] > 0) |
                     (self.grism.data['ERR'] == 0) |
                     (self.grism.data['SCI'] == 0))
        
        self.var = self.grism.data['ERR']**2
        self.var[self.mask] = 1.e30
        self.ivar = 1/self.var
        self.ivar[self.mask] = 0

        self.thumbs = {}

        #self.compute_model = self.beam.compute_model
        #self.model = self.beam.model
        self.modelf = self.beam.modelf  # .flatten()
        self.model = self.beam.modelf.reshape(self.beam.sh_beam)
        
        # Attributes
        self.size = self.modelf.size
        self.wave = self.beam.lam
        self.sh = self.beam.sh_beam

        # Initialize for fits
        if seg_ids is None:
            self.flat_flam = self.compute_model(in_place=False, is_cgs=True)
        else:
            for i, sid in enumerate(seg_ids):
                flat_i = self.compute_model(id=sid, in_place=False,
                                            is_cgs=True)
                if i == 0:
                    self.flat_flam = flat_i
                else:
                    self.flat_flam += flat_i

        # OK data where the 2D model has non-zero flux
        
        self.fit_mask = (~self.mask.flatten()) & (self.ivar.flatten() != 0)
        self.fit_mask &= (self.flat_flam > min_mask*self.flat_flam.max())
        #self.fit_mask &= (self.flat_flam > 3*self.contam.flatten())

        # Apply minimum sensitivity mask
        self.sens_mask = 1.
        if min_sens > 0:
            flux_min_sens = (self.beam.sensitivity <
                             min_sens*self.beam.sensitivity.max())*1.

            if flux_min_sens.sum() > 0:
                test_spec = [self.beam.lam, flux_min_sens]
                if seg_ids is None:
                    flat_sens = self.compute_model(in_place=False,
                                                   is_cgs=True,
                                  spectrum_1d=test_spec)
                else:
                    for i, sid in enumerate(seg_ids):
                        f_i = self.compute_model(id=sid, in_place=False,
                                           is_cgs=True, spectrum_1d=test_spec)
                        if i == 0:
                            flat_sens = f_i
                        else:
                            flat_sens += f_i

                # self.sens_mask = flat_sens == 0
                # Make mask along columns
                is_masked = (flat_sens.reshape(self.sh) > 0).sum(axis=0)
                self.sens_mask = (np.dot(np.ones((self.sh[0], 1)), is_masked[None, :]) == 0).flatten()
                self.fit_mask &= self.sens_mask

        # Flat versions of sci/ivar arrays
        self.scif = (self.grism.data['SCI'] - self.contam).flatten()
        self.ivarf = self.ivar.flatten()
        self.wavef = np.dot(np.ones((self.sh[0], 1)), self.wave[None, :]).flatten()

        # Mask large residuals where throughput is low
        if mask_resid:
            resid = np.abs(self.scif - self.flat_flam)*np.sqrt(self.ivarf)
            bad_resid = (self.flat_flam < 0.05*self.flat_flam.max())
            bad_resid &= (resid > 5)
            self.bad_resid = bad_resid
            self.fit_mask *= ~bad_resid
        else:
            self.bad_resid = np.zeros_like(self.fit_mask)

        # Mask very contaminated
        contam_mask = ((self.contam*np.sqrt(self.ivar) > contam_sn_mask[0]) &
                      (self.model*np.sqrt(self.ivar) < contam_sn_mask[1]))
        #self.fit_mask *= ~contam_mask.flatten()
        self.contam_mask = ~nd.maximum_filter(contam_mask, size=5).flatten()
        self.poly_order = None
        # self.init_poly_coeffs(poly_order=1)


    def init_from_input(self, flt, beam, conf=None, get_slice_header=True):
        """Initialize from data objects

        Parameters
        ----------
        flt : `GrismFLT`
            Parent FLT frame.

        beam : `GrismDisperser`
            Object and spectral order to consider

        conf : `.grismconf.aXeConf`
            Pre-computed configuration file.  If not specified will regenerate
            based on header parameters, which might be necessary for
            multiprocessing parallelization and pickling.

        get_slice_header : bool
            Get full header of the sliced data.  Costs some overhead so can
            be skipped if full header information isn't required.

        Returns
        -------
        Loads attributes to `self`.
        """
        self.id = beam.id
        if conf is None:
            conf = grismconf.load_grism_config(flt.conf_file)

        self.beam = GrismDisperser(id=beam.id, direct=beam.direct*1,
                           segmentation=beam.seg*1, origin=beam.origin,
                           pad=beam.pad, grow=beam.grow,
                           beam=beam.beam, conf=conf, xcenter=beam.xcenter,
                           ycenter=beam.ycenter, fwcpos=flt.grism.fwcpos,
                           MW_EBV=flt.grism.MW_EBV)

        if hasattr(beam, 'psf_params'):
            self.beam.x_init_epsf(psf_params=beam.psf_params, psf_filter=beam.psf_filter, yoff=beam.psf_yoff)

        if beam.spectrum_1d is None:
            self.compute_model()  # spectrum_1d=beam.spectrum_1d)
        else:
            self.compute_model(spectrum_1d=beam.spectrum_1d,
                                    is_cgs=beam.is_cgs)

        slx_thumb = slice(self.beam.origin[1],
                          self.beam.origin[1]+self.beam.sh[1])

        sly_thumb = slice(self.beam.origin[0],
                          self.beam.origin[0]+self.beam.sh[0])

        self.direct = flt.direct.get_slice(slx_thumb, sly_thumb,
                                           get_slice_header=get_slice_header)
        self.grism = flt.grism.get_slice(self.beam.slx_parent,
                                         self.beam.sly_parent,
                                         get_slice_header=get_slice_header)

        self.contam = flt.model[self.beam.sly_parent, self.beam.slx_parent]*1
        if self.beam.id in flt.object_dispersers:
            self.contam -= self.beam.model


    def load_fits(self, file, conf=None, direct_extn=1, grism_extn=2):
        """Initialize from FITS file

        Parameters
        ----------
        file : str
            FITS file to read (as output from `write_fits`).

        Returns
        -------
        Loads attributes to `self`.
        """
        if isinstance(file, str):
            hdu = pyfits.open(file)
            file_is_open = True
        else:
            file_is_open = False
            hdu = file

        self.direct = ImageData(hdulist=hdu, sci_extn=direct_extn)
        self.grism = ImageData(hdulist=hdu, sci_extn=grism_extn)

        self.contam = hdu['CONTAM'].data*1
        try:
            self.modelf = hdu['MODEL'].data.flatten().astype(np.float32)*1
        except:
            self.modelf = self.grism['SCI'].flatten().astype(np.float32)*0.

        if ('REF', 1) in hdu:
            direct = hdu['REF', 1].data*1
        else:
            direct = hdu['SCI', 1].data*1

        h0 = hdu[0].header

        # if 'DFILTER' in self.grism.header:
        #     direct_filter = self.grism.header['DFILTER']
        # else:
        #     direct_filter = self.direct.filter
        # #
        if 'DFILTER' in self.grism.header:
            direct_filter = self.grism.header['DFILTER']
        if self.grism.instrument in ['NIRCAM', 'NIRISS']:
            direct_filter = self.grism.pupil
        else:
            direct_filter = self.direct.filter

        if conf is None:
            conf_args = dict(instrume=self.grism.instrument, 
                             filter=direct_filter, 
                             grism=self.grism.filter,
                             module=self.grism.module,
                             chip=self.grism.ccdchip)
            
            self.conf_file = grismconf.get_config_filename(**conf_args)
            conf = grismconf.load_grism_config(self.conf_file)

        if 'GROW' in self.grism.header:
            grow = self.grism.header['GROW']
        else:
            grow = 1

        if 'MW_EBV' in h0:
            self.grism.MW_EBV = h0['MW_EBV']
        else:
            self.grism.MW_EBV = 0

        self.grism.fwcpos = h0['FWCPOS']
        if (self.grism.fwcpos == 0) | (self.grism.fwcpos == ''):
            self.grism.fwcpos = None

        if 'TYOFFSET' in h0:
            yoffset = h0['TYOFFSET']
        else:
            yoffset = 0.
        
        if 'TXOFFSET' in h0:
            xoffset = h0['TXOFFSET']
        else:
            xoffset = None
            
        if ('PADX' in h0) & ('PADY' in h0):
            _pad = [h0['PADY'], h0['PADX']]
        elif ('PAD' in h0):
            _pad = [h0['PAD'], h0['PAD']]
        
        self.beam = GrismDisperser(id=h0['ID'], direct=direct,
                                   segmentation=hdu['SEG'].data*1,
                                   origin=self.direct.origin,
                                   pad=_pad,
                                   grow=grow, beam=h0['BEAM'],
                                   xcenter=h0['XCENTER'],
                                   ycenter=h0['YCENTER'],
                                   conf=conf, fwcpos=self.grism.fwcpos,
                                   MW_EBV=self.grism.MW_EBV,
                                   yoffset=yoffset, xoffset=xoffset)

        self.grism.parent_file = h0['GPARENT']
        self.direct.parent_file = h0['DPARENT']
        self.id = h0['ID']
        self.modelf = self.beam.modelf
        
        # Cleanup
        if file_is_open:
            hdu.close()
            
    @property
    def trace_table(self):
        """
        Table of trace parameters.  Trace is unit-indexed.
        """
        dtype = np.float32

        tab = utils.GTable()
        tab.meta['CONFFILE'] = os.path.basename(self.beam.conf.conf_file)

        tab['wavelength'] = np.cast[dtype](self.beam.lam*u.Angstrom)
        tab['trace'] = np.cast[dtype](self.beam.ytrace + self.beam.sh_beam[0]/2 - self.beam.ycenter)

        sens_units = u.erg/u.second/u.cm**2/u.Angstrom/(u.electron/u.second)
        tab['sensitivity'] = np.cast[dtype](self.beam.sensitivity*sens_units)

        return tab

    def write_fits(self, root='beam_', overwrite=True, strip=False, include_model=True, get_hdu=False, get_trace_table=True):
        """Write attributes and data to FITS file

        Parameters
        ----------
        root : str
            Output filename will be

               '{root}_{self.id}.{self.grism.filter}.{self.beam}.fits'

            with `self.id` zero-padded with 5 digits.

        overwrite : bool
            Overwrite existing file.

        strip : bool
            Strip out extensions that aren't totally necessary for
            regenerating the `ImageData` object.  That is, strip out the
            direct image `SCI`, `ERR`, and `DQ` extensions if `REF` is
            defined.  Also strip out `MODEL`.

        get_hdu : bool
            Return `~astropy.io.fits.HDUList` rather than writing a file.

        Returns
        -------
        hdu : `~astropy.io.fits.HDUList`
            If `get_hdu` is True

        outfile : str
            If `get_hdu` is False, return the output filename.

        """
        h0 = pyfits.Header()
        h0['ID'] = self.beam.id, 'Object ID'
        h0['PADX'] = self.beam.pad[1], 'Padding of input image axis1'
        h0['PADY'] = self.beam.pad[0], 'Padding of input image axis2'
        h0['BEAM'] = self.beam.beam, 'Grism order ("beam")'
        h0['XCENTER'] = (self.beam.xcenter,
                         'Offset of centroid wrt thumb center')
        h0['YCENTER'] = (self.beam.ycenter,
                         'Offset of centroid wrt thumb center')

        if hasattr(self.beam, 'yoffset'):
            h0['TYOFFSET'] = (self.beam.yoffset,
                         'Cross dispersion offset of the trace')
        
        if hasattr(self.beam, 'xoffset'):
            h0['TXOFFSET'] = (self.beam.xoffset,
                         'Dispersion offset of the trace')
        
        h0['GPARENT'] = (self.grism.parent_file,
                         'Parent grism file')

        h0['DPARENT'] = (self.direct.parent_file,
                         'Parent direct file')

        h0['FWCPOS'] = (self.grism.fwcpos,
                         'Filter wheel position (NIRISS)')

        h0['MW_EBV'] = (self.grism.MW_EBV,
                         'Milky Way exctinction E(B-V)')

        hdu = pyfits.HDUList([pyfits.PrimaryHDU(header=h0)])
        hdu.extend(self.direct.get_HDUList(extver=1))
        hdu.append(pyfits.ImageHDU(data=np.cast[np.int32](self.beam.seg),
                                   header=hdu[-1].header, name='SEG'))

        # 2D grism spectra
        grism_hdu = self.grism.get_HDUList(extver=2)

        #######
        # 2D Spectroscopic WCS
        hdu2d, wcs2d = self.get_2d_wcs()

        # Get available 'WCSNAME'+key
        for key in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if 'WCSNAME{0}'.format(key) not in self.grism.header:
                break
            else:
                wcsname = self.grism.header['WCSNAME{0}'.format(key)]
                if wcsname == 'BeamLinear2D':
                    break

        h2d = wcs2d.to_header(key=key)
        for ext in grism_hdu:
            for k in h2d:
                ext.header[k] = h2d[k], h2d.comments[k]
        ####

        hdu.extend(grism_hdu)
        hdu.append(pyfits.ImageHDU(data=self.contam, header=hdu[-1].header,
                                   name='CONTAM'))

        if include_model:
            hdu.append(pyfits.ImageHDU(data=np.cast[np.float32](self.model),
                                       header=hdu[-1].header, name='MODEL'))

        if get_trace_table:
            trace_hdu = pyfits.table_to_hdu(self.trace_table)
            trace_hdu.header['EXTNAME'] = 'TRACE'
            trace_hdu.header['EXTVER'] = 2
            hdu.append(trace_hdu)

        if strip:
            # Blotted reference is attached, don't need individual direct
            # arrays.
            if self.direct['REF'] is not None:
                for ext in [('SCI', 1), ('ERR', 1), ('DQ', 1)]:
                    if ext in hdu:
                        ix = hdu.index_of(ext)
                        p = hdu.pop(ix)

            # This can be regenerated
            # if strip & 2:
            #     ix = hdu.index_of('MODEL')
            #     p = hdu.pop(ix)

            # Put Primary keywords in first extension
            SKIP_KEYS = ['EXTEND', 'SIMPLE']
            for key in h0:
                if key not in SKIP_KEYS:
                    hdu[1].header[key] = (h0[key], h0.comments[key])
                    hdu['SCI', 2].header[key] = (h0[key], h0.comments[key])

        if get_hdu:
            return hdu

        outfile = '{0}_{1:05d}.{2}.{3}.fits'.format(root, self.beam.id,
                                         self.grism.filter.lower(),
                                         self.beam.beam)

        hdu.writeto(outfile, overwrite=overwrite)

        return outfile

    def compute_model(self, use_psf=True, **kwargs):
        """Link to `self.beam.compute_model`

        `self.beam` is a `GrismDisperser` object.
        """
        if use_psf & hasattr(self.beam, 'psf'):
            result = self.beam.compute_model_psf(**kwargs)
        else:
            result = self.beam.compute_model(**kwargs)

        reset_inplace = True
        if 'in_place' in kwargs:
            reset_inplace = kwargs['in_place']

        if reset_inplace:
            self.modelf = self.beam.modelf  # .flatten()
            self.model = self.beam.modelf.reshape(self.beam.sh_beam)

        return result

    def get_wavelength_wcs(self, wavelength=1.3e4):
        """Compute *celestial* WCS of the 2D spectrum array for a specified central wavelength

        This essentially recenters the celestial SIP WCS such that the
        desired wavelength was at the object position as observed in the
        direct image (which has associated geometric distortions etc).

        Parameters
        ----------
        wavelength : float
            Central wavelength to use for derived WCS.

        Returns
        -------
        header : `~astropy.io.fits.Header`
            FITS header

        wcs : `~astropy.wcs.WCS`
            Derived celestial WCS

        """
        wcs = self.grism.wcs.deepcopy()

        xarr = np.arange(self.beam.lam_beam.shape[0])

        # Trace properties at desired wavelength
        dx = np.interp(wavelength, self.beam.lam_beam, xarr)
        dy = np.interp(wavelength, self.beam.lam_beam, self.beam.ytrace_beam)

        dl = np.interp(wavelength, self.beam.lam_beam[1:],
                                   np.diff(self.beam.lam_beam))

        ysens = np.interp(wavelength, self.beam.lam_beam,
                          self.beam.sensitivity_beam)

        # Update CRPIX
        dc = 0  # python array center to WCS pixel center

        for wcs_ext in [wcs.sip, wcs.wcs]:
            if wcs_ext is None:
                continue
            else:
                cr = wcs_ext.crpix

            cr[0] += dx + self.beam.x0[1] + self.beam.dxfull[0] + dc
            cr[1] += dy + dc

        for wcs_ext in [wcs.cpdis1, wcs.cpdis2, wcs.det2im1, wcs.det2im2]:
            if wcs_ext is None:
                continue
            else:
                cr = wcs_ext.crval

            cr[0] += dx + self.beam.sh[0]/2 + self.beam.dxfull[0] + dc
            cr[1] += dy + dc

        # Make SIP CRPIX match CRPIX
        # if wcs.sip is not None:
        #     for i in [0,1]:
        #         wcs.sip.crpix[i] = wcs.wcs.crpix[i]

        for wcs_ext in [wcs.sip]:
            if wcs_ext is not None:
                for i in [0, 1]:
                    wcs_ext.crpix[i] = wcs.wcs.crpix[i]

        # WCS header
        header = wcs.to_header(relax=True)
        for key in header:
            if key.startswith('PC'):
                header.rename_keyword(key, key.replace('PC', 'CD'))

        header['LONPOLE'] = 180.
        header['RADESYS'] = 'ICRS'
        header['LTV1'] = (0.0, 'offset in X to subsection start')
        header['LTV2'] = (0.0, 'offset in Y to subsection start')
        header['LTM1_1'] = (1.0, 'reciprocal of sampling rate in X')
        header['LTM2_2'] = (1.0, 'reciprocal of sampling rate in X')
        header['INVSENS'] = (ysens, 'inverse sensitivity, 10**-17 erg/s/cm2')
        header['DLDP'] = (dl, 'delta wavelength per pixel')

        return header, wcs

    def get_2d_wcs(self, data=None, key=None):
        """Get simplified WCS of the 2D spectrum

        Parameters
        ----------
        data : array-like
            Put this data in the output HDU rather than empty zeros

        key : None
            Key for WCS extension, passed to `~astropy.wcs.WCS.to_header`.

        Returns
        -------
        hdu : `~astropy.io.fits.ImageHDU`
            Image HDU with header and data properties.

        wcs : `~astropy.wcs.WCS`
            WCS appropriate for the 2D spectrum with spatial (y) and spectral
            (x) axes.

            .. note::
                Assumes linear dispersion and trace functions!

        """
        h = pyfits.Header()

        h['WCSNAME'] = 'BeamLinear2D'

        h['CRPIX1'] = self.beam.sh_beam[0]/2 - self.beam.xcenter
        h['CRPIX2'] = self.beam.sh_beam[0]/2 - self.beam.ycenter

        # Wavelength, A
        h['CNAME1'] = 'Wave-Angstrom'
        h['CTYPE1'] = 'WAVE'
        #h['CUNIT1'] = 'Angstrom'
        h['CRVAL1'] = self.beam.lam_beam[0]
        h['CD1_1'] = self.beam.lam_beam[1] - self.beam.lam_beam[0]
        h['CD1_2'] = 0.

        # Linear trace
        h['CNAME2'] = 'Trace'
        h['CTYPE2'] = 'LINEAR'
        h['CRVAL2'] = -1*self.beam.ytrace_beam[0]
        h['CD2_2'] = 1.
        h['CD2_1'] = -(self.beam.ytrace_beam[1] - self.beam.ytrace_beam[0])

        if data is None:
            data = np.zeros(self.beam.sh_beam, dtype=np.float32)

        hdu = pyfits.ImageHDU(data=data, header=h)
        wcs = pywcs.WCS(hdu.header)

        #wcs.pscale = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[1,0]**2)*3600.
        wcs.pscale = utils.get_wcs_pscale(wcs)

        return hdu, wcs

    def full_2d_wcs(self, data=None):
        """Get trace WCS of the 2D spectrum

        Parameters
        ----------
        data : array-like
            Put this data in the output HDU rather than empty zeros

        Returns
        -------
        hdu : `~astropy.io.fits.ImageHDU`
            Image HDU with header and data properties.

        wcs : `~astropy.wcs.WCS`
            WCS appropriate for the 2D spectrum with spatial (y) and spectral
            (x) axes.

            .. note::
                Assumes linear dispersion and trace functions!

        """
        h = pyfits.Header()
        h['CRPIX1'] = self.beam.sh_beam[0]/2 - self.beam.xcenter
        h['CRPIX2'] = self.beam.sh_beam[0]/2 - self.beam.ycenter
        h['CRVAL1'] = self.beam.lam_beam[0]/1.e4
        h['CD1_1'] = (self.beam.lam_beam[1] - self.beam.lam_beam[0])/1.e4
        h['CD1_2'] = 0.

        h['CRVAL2'] = -1*self.beam.ytrace_beam[0]
        h['CD2_2'] = 1.
        h['CD2_1'] = -(self.beam.ytrace_beam[1] - self.beam.ytrace_beam[0])

        h['CTYPE1'] = 'RA---TAN-SIP'
        h['CUNIT1'] = 'mas'
        h['CTYPE2'] = 'DEC--TAN-SIP'
        h['CUNIT2'] = 'mas'

        #wcs_header = grizli.utils.to_header(self.grism.wcs)

        x = np.arange(len(self.beam.lam_beam))
        c = np.polyfit(x, self.beam.lam_beam/1.e4, 2)
        #c = np.polyfit((self.beam.lam_beam-self.beam.lam_beam[0])/1.e4, x/h['CD1_1'], 2)

        ct = np.polyfit(x, self.beam.ytrace_beam, 2)

        h['A_ORDER'] = 2
        h['B_ORDER'] = 2

        h['A_0_2'] = 0.
        h['A_1_2'] = 0.
        h['A_2_2'] = 0.
        h['A_2_1'] = 0.
        h['A_2_0'] = c[0]  # /c[1]
        h['CD1_1'] = c[1]

        h['B_0_2'] = 0.
        h['B_1_2'] = 0.
        h['B_2_2'] = 0.
        h['B_2_1'] = 0.
        if ct[1] != 0:
            h['B_2_0'] = ct[0]  # /ct[1]
        else:
            h['B_2_0'] = 0

        #h['B_2_0'] = 0

        if data is None:
            data = np.zeros(self.beam.sh_beam, dtype=np.float32)

        hdu = pyfits.ImageHDU(data=data, header=h)
        wcs = pywcs.WCS(hdu.header)

        # xf = x + h['CRPIX1']-1
        # coo = np.array([xf, xf*0])
        # tr = wcs.all_pix2world(coo.T, 0)

        #wcs.pscale = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[1,0]**2)*3600.
        wcs.pscale = utils.get_wcs_pscale(wcs)

        return hdu, wcs

    def get_sky_coords(self):
        """Get WCS coordinates of the center of the direct image

        Returns
        -------
        ra, dec : float
            Center coordinates of the beam thumbnail in decimal degrees
        """
        pix_center = np.array([self.beam.sh][::-1])/2.
        pix_center -= np.array([self.beam.xcenter, self.beam.ycenter])
        if self.direct.wcs.sip is not None:
            for i in range(2):
                self.direct.wcs.sip.crpix[i] = self.direct.wcs.wcs.crpix[i]

        ra, dec = self.direct.wcs.all_pix2world(pix_center, 1)[0]
        return ra, dec

    def get_dispersion_PA(self, decimals=0):
        """Compute exact PA of the dispersion axis, including tilt of the
        trace and the FLT WCS

        Parameters
        ----------
        decimals : int or None
            Number of decimal places to round to, passed to `~numpy.round`.
            If None, then don't round.

        Returns
        -------
        dispersion_PA : float
            PA (angle East of North) of the dispersion axis.
        """
        from astropy.coordinates import Angle
        import astropy.units as u

        # extra tilt of the 1st order grism spectra
        if 'BEAMA' in self.beam.conf.conf_dict:
            x0 = self.beam.conf.conf_dict['BEAMA']
        else:
            x0 = np.array([10,30])
            
        dy_trace, lam_trace = self.beam.conf.get_beam_trace(x=507, y=507,
                                                         dx=x0, beam='A')

        extra = np.arctan2(dy_trace[1]-dy_trace[0], x0[1]-x0[0])/np.pi*180

        # Distorted WCS
        crpix = self.direct.wcs.wcs.crpix
        xref = [crpix[0], crpix[0]+1]
        yref = [crpix[1], crpix[1]]
        r, d = self.direct.wcs.all_pix2world(xref, yref, 1)
        pa = Angle((extra +
                    np.arctan2(np.diff(r)*np.cos(d[0]/180*np.pi),
                               np.diff(d))[0]/np.pi*180)*u.deg)

        dispersion_PA = pa.wrap_at(360*u.deg).value
        if decimals is not None:
            dispersion_PA = np.round(dispersion_PA, decimals=decimals)

        return float(dispersion_PA)

    def init_epsf(self, center=None, tol=1.e-3, yoff=0., skip=1., flat_sensitivity=False, psf_params=None, N=4, get_extended=False, only_centering=True):
        """Initialize ePSF fitting for point sources
        TBD
        """
        import scipy.sparse

        EPSF = utils.EffectivePSF()
        ivar = 1/self.direct['ERR']**2
        ivar[~np.isfinite(ivar)] = 0
        ivar[self.direct['DQ'] > 0] = 0

        ivar[self.beam.seg != self.id] = 0

        if ivar.max() == 0:
            ivar = ivar+1.

        origin = np.array(self.direct.origin) - np.array(self.direct.pad)
        if psf_params is None:
            self.beam.psf_ivar = ivar*1
            self.beam.psf_sci = self.direct['SCI']*1
            self.psf_params = EPSF.fit_ePSF(self.direct['SCI'],
                                                  ivar=ivar,
                                                  center=center, tol=tol,
                                                  N=N, origin=origin,
                                                  filter=self.direct.filter,
                                                  get_extended=get_extended,
                                                only_centering=only_centering)
        else:
            self.beam.psf_ivar = ivar*1
            self.beam.psf_sci = self.direct['SCI']*1
            self.psf_params = psf_params

        self.beam.x_init_epsf(flat_sensitivity=False, psf_params=self.psf_params, psf_filter=self.direct.filter, yoff=yoff, skip=skip, get_extended=get_extended)

        self._parse_from_data(**self._parse_params)

        return None

        # self.psf = EPSF.get_ePSF(self.psf_params, origin=origin, shape=self.beam.sh, filter=self.direct.filter)
        #
        # self.psf_resid = self.direct['SCI'] - self.psf
        #
        # y0, x0 = np.array(self.beam.sh)/2.-1
        #
        # # Center in detector coords
        # xd = self.psf_params[1] + self.direct.origin[1] - self.direct.pad + x0
        # yd = self.psf_params[2] + self.direct.origin[0] - self.direct.pad + y0
        #
        # # Get wavelength array
        # psf_xy_lam = []
        # for i, filter in enumerate(['F105W', 'F125W', 'F160W']):
        #     psf_xy_lam.append(EPSF.get_at_position(x=xd, y=yd, filter=filter))
        #
        # filt_ix = np.arange(3)
        # filt_lam = np.array([1.0551, 1.2486, 1.5369])*1.e4
        #
        # yp_beam, xp_beam = np.indices(self.beam.sh_beam)
        # #skip = 1
        # xarr = np.arange(0,self.beam.lam_beam.shape[0], skip)
        # xarr = xarr[xarr <= self.beam.lam_beam.shape[0]-1]
        # xbeam = np.arange(self.beam.lam_beam.shape[0])*1.
        #
        # #yoff = 0 #-0.15
        # psf_model = self.model*0.
        # A_psf = []
        # lam_psf = []
        #
        # lam_offset = self.beam.sh[1]/2 - self.psf_params[1] - 1
        # self.lam_offset = lam_offset
        #
        # for xi in xarr:
        #     yi = np.interp(xi, xbeam, self.beam.ytrace_beam)
        #     li = np.interp(xi, xbeam, self.beam.lam_beam)
        #     dx = xp_beam-self.psf_params[1]-xi-x0
        #     dy = yp_beam-self.psf_params[2]-yi+yoff-y0
        #
        #     # wavelength-dependent
        #     ii = np.interp(li, filt_lam, filt_ix, left=-1, right=10)
        #     if ii == -1:
        #         psf_xy_i = psf_xy_lam[0]*1
        #     elif ii == 10:
        #         psf_xy_i = psf_xy_lam[2]*1
        #     else:
        #         ni = int(ii)
        #         f = 1-(li-filt_lam[ni])/(filt_lam[ni+1]-filt_lam[ni])
        #         psf_xy_i = f*psf_xy_lam[ni] + (1-f)*psf_xy_lam[ni+1]
        #
        #     psf = EPSF.eval_ePSF(psf_xy_i, dx, dy)*self.psf_params[0]
        #
        #     A_psf.append(psf.flatten())
        #     lam_psf.append(li)
        #
        # # Sensitivity
        # self.lam_psf = np.array(lam_psf)
        # if flat_sensitivity:
        #     s_i_scale = np.abs(np.gradient(self.lam_psf))*self.direct.photflam
        # else:
        #     sens = self.beam.conf.sens[self.beam.beam]
        #     so = np.argsort(self.lam_psf)
        #     s_i = interp.interp_conserve_c(self.lam_psf[so], sens['WAVELENGTH'], sens['SENSITIVITY'])*np.gradient(self.lam_psf[so])*self.direct.photflam
        #     s_i_scale = s_i*0.
        #     s_i_scale[so] = s_i
        #
        # self.A_psf = scipy.sparse.csr_matrix(np.array(A_psf).T*s_i_scale)

    # def xcompute_model_psf(self, id=None, spectrum_1d=None, in_place=True, is_cgs=True):
    #     if spectrum_1d is None:
    #         model = np.array(self.A_psf.sum(axis=1))
    #         model = model.reshape(self.beam.sh_beam)
    #     else:
    #         dx = np.diff(self.lam_psf)[0]
    #         if dx < 0:
    #             coeffs = interp.interp_conserve_c(self.lam_psf[::-1],
    #                                               spectrum_1d[0],
    #                                               spectrum_1d[1])[::-1]
    #         else:
    #             coeffs = interp.interp_conserve_c(self.lam_psf,
    #                                               spectrum_1d[0],
    #                                               spectrum_1d[1])
    #
    #
    #         model = self.A_psf.dot(coeffs).reshape(self.beam.sh_beam)
    #
    #     if in_place:
    #         self.model = model
    #         self.beam.model = self.model
    #         return True
    #     else:
    #         return model.flatten()

    # Below here will be cut out after verifying that the demos
    # can be run with the new fitting tools
    def init_poly_coeffs(self, poly_order=1, fit_background=True):
        """Initialize arrays for polynomial fits to the spectrum

        Provides capabilities of fitting n-order polynomials to observed
        spectra rather than galaxy/stellar templates.

        Parameters
        ----------
        poly_order : int
            Order of the polynomial

        fit_background : bool
            Compute additional arrays for allowing the background to be fit
            along with the polynomial coefficients.

        Returns
        -------
        Polynomial parameters stored in attributes `y_poly`, `n_poly`, ...

        """
        # Already done?
        if poly_order == self.poly_order:
            return None

        self.poly_order = poly_order

        # Model: (a_0 x**0 + ... a_i x**i)*continuum + line
        yp, xp = np.indices(self.beam.sh_beam)
        NX = self.beam.sh_beam[1]
        self.xpf = (xp.flatten() - NX/2.)
        self.xpf /= (NX/2.)

        # Polynomial continuum arrays
        if fit_background:
            self.n_bg = 1
            self.A_poly = [self.flat_flam*0+1]
            self.A_poly.extend([self.xpf**order*self.flat_flam
                                for order in range(poly_order+1)])
        else:
            self.n_bg = 0
            self.A_poly = [self.xpf**order*self.flat_flam
                                for order in range(poly_order+1)]

        # Array for generating polynomial "template"
        x = (np.arange(NX) - NX/2.) / (NX/2.)
        self.y_poly = np.array([x**order for order in range(poly_order+1)])
        self.n_poly = self.y_poly.shape[0]
        self.n_simp = self.n_poly + self.n_bg

        self.DoF = self.fit_mask.sum()

    # def load_templates(self, fwhm=400, line_complexes=True):
    #     """TBD
    #
    #     ***
    #         These below will probably be cut since they're all now implemented
    #         in more detail in multifit.py.  Need to update demos before
    #         taking them out completely.
    #     ***
    #
    #     """
    #     # templates = ['templates/EAZY_v1.0_lines/eazy_v1.0_sed1_nolines.dat',
    #     # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed2_nolines.dat',
    #     # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed3_nolines.dat',
    #     # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed4_nolines.dat',
    #     # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed5_nolines.dat',
    #     # 'templates/EAZY_v1.0_lines/eazy_v1.0_sed6_nolines.dat',
    #     # 'templates/cvd12_t11_solar_Chabrier.extend.dat',
    #     # 'templates/dobos11/bc03_pr_ch_z02_ltau07.0_age09.2_av2.5.dat']
    #
    #     templates = ['templates/EAZY_v1.0_lines/eazy_v1.0_sed3_nolines.dat',
    #                  'templates/cvd12_t11_solar_Chabrier.extend.dat']
    #
    #     temp_list = OrderedDict()
    #     for temp in templates:
    #         data = np.loadtxt(GRIZLI_PATH + '/' + temp, unpack=True)
    #         scl = np.interp(5500., data[0], data[1])
    #         name = os.path.basename(temp)
    #         temp_list[name] = utils.SpectrumTemplate(wave=data[0],
    #                                                          flux=data[1]/scl)
    #         #plt.plot(temp_list[-1].wave, temp_list[-1].flux, label=temp, alpha=0.5)
    #
    #     line_wavelengths = {} ; line_ratios = {}
    #     line_wavelengths['Ha'] = [6564.61]; line_ratios['Ha'] = [1.]
    #     line_wavelengths['Hb'] = [4862.68]; line_ratios['Hb'] = [1.]
    #     line_wavelengths['Hg'] = [4341.68]; line_ratios['Hg'] = [1.]
    #     line_wavelengths['Hd'] = [4102.892]; line_ratios['Hd'] = [1.]
    #     line_wavelengths['OIIIx'] = [4364.436]; line_ratios['OIIIx'] = [1.]
    #     line_wavelengths['OIII'] = [5008.240, 4960.295]; line_ratios['OIII'] = [2.98, 1]
    #     line_wavelengths['OIII+Hb'] = [5008.240, 4960.295, 4862.68]; line_ratios['OIII+Hb'] = [2.98, 1, 3.98/8.]
    #
    #     line_wavelengths['OIII+Hb+Ha'] = [5008.240, 4960.295, 4862.68, 6564.61]; line_ratios['OIII+Hb+Ha'] = [2.98, 1, 3.98/10., 3.98/10.*2.86]
    #
    #     line_wavelengths['OIII+Hb+Ha+SII'] = [5008.240, 4960.295, 4862.68, 6564.61, 6718.29, 6732.67]
    #     line_ratios['OIII+Hb+Ha+SII'] = [2.98, 1, 3.98/10., 3.98/10.*2.86*4, 3.98/10.*2.86/10.*4, 3.98/10.*2.86/10.*4]
    #
    #     line_wavelengths['OII'] = [3729.875]; line_ratios['OII'] = [1]
    #     line_wavelengths['OI'] = [6302.046]; line_ratios['OI'] = [1]
    #
    #     line_wavelengths['Ha+SII'] = [6564.61, 6718.29, 6732.67]; line_ratios['Ha+SII'] = [1., 1./10, 1./10]
    #     line_wavelengths['SII'] = [6718.29, 6732.67]; line_ratios['SII'] = [1., 1.]
    #
    #     if line_complexes:
    #         #line_list = ['Ha+SII', 'OIII+Hb+Ha', 'OII']
    #         line_list = ['Ha+SII', 'OIII+Hb', 'OII']
    #     else:
    #         line_list = ['Ha', 'SII', 'OIII', 'Hb', 'OII']
    #         #line_list = ['Ha', 'SII']
    #
    #     for line in line_list:
    #         scl = line_ratios[line]/np.sum(line_ratios[line])
    #         for i in range(len(scl)):
    #             line_i = utils.SpectrumTemplate(wave=line_wavelengths[line][i],
    #                                       flux=None, fwhm=fwhm, velocity=True)
    #
    #             if i == 0:
    #                 line_temp = line_i*scl[i]
    #             else:
    #                 line_temp = line_temp + line_i*scl[i]
    #
    #         temp_list['line {0}'.format(line)] = line_temp
    #
    #     return temp_list
    #
    # def fit_at_z(self, z=0., templates={}, fitter='lstsq', poly_order=3):
    #     """TBD
    #     """
    #     import copy
    #
    #     import sklearn.linear_model
    #     import numpy.linalg
    #
    #     self.init_poly_coeffs(poly_order=poly_order)
    #
    #     NTEMP = len(self.A_poly)
    #     A_list = copy.copy(self.A_poly)
    #     ok_temp = np.ones(NTEMP+len(templates), dtype=bool)
    #
    #     for i, key in enumerate(templates.keys()):
    #         NTEMP += 1
    #         temp = templates[key].zscale(z, 1.)
    #         spectrum_1d = [temp.wave, temp.flux]
    #
    #         if ((temp.wave[0] > self.beam.lam_beam[-1]) |
    #             (temp.wave[-1] < self.beam.lam_beam[0])):
    #
    #             A_list.append(self.flat_flam*1)
    #             ok_temp[NTEMP-1] = False
    #             #print 'skip TEMP: %d, %s' %(i, key)
    #             continue
    #         else:
    #             pass
    #             #print 'TEMP: %d' %(i)
    #
    #         temp_model = self.compute_model(spectrum_1d=spectrum_1d,
    #                                         in_place=False)
    #
    #         ### Test that model spectrum has non-zero pixel values
    #         #print 'TEMP: %d, %.3f' %(i, temp_model[self.fit_mask].max()/temp_model.max())
    #         if temp_model[self.fit_mask].max()/temp_model.max() < 0.2:
    #             #print 'skipx TEMP: %d, %s' %(i, key)
    #             ok_temp[NTEMP-1] = False
    #
    #         A_list.append(temp_model)
    #
    #     A = np.vstack(A_list).T
    #     out_coeffs = np.zeros(NTEMP)
    #
    #     ### LSTSQ coefficients
    #     if fitter == 'lstsq':
    #         out = numpy.linalg.lstsq(A[self.fit_mask, :][:, ok_temp],
    #                                  self.scif[self.fit_mask])
    #         lstsq_coeff, residuals, rank, s = out
    #         coeffs = lstsq_coeff
    #     else:
    #         clf = sklearn.linear_model.LinearRegression()
    #         status = clf.fit(A[self.fit_mask, :][:, ok_temp],
    #                          self.scif[self.fit_mask])
    #         coeffs = clf.coef_
    #
    #     out_coeffs[ok_temp] = coeffs
    #     model = np.dot(A, out_coeffs)
    #     model_2d = model.reshape(self.beam.sh_beam)
    #
    #     chi2 = np.sum(((self.scif - model)**2*self.ivarf)[self.fit_mask])
    #
    #     return A, out_coeffs, chi2, model_2d
    #
    # def fit_redshift(self, prior=None, poly_order=1, fwhm=500,
    #                  make_figure=True, zr=None, dz=None, verbose=True):
    #     """TBD
    #     """
    #     # if False:
    #     #     reload(grizlidev.utils); utils = grizlidev.utils
    #     #     reload(grizlidev.utils_c); reload(grizlidev.model);
    #     #     reload(grizlidev.grismconf); reload(grizlidev.utils); reload(grizlidev.multifit); reload(grizlidev); reload(grizli)
    #     #
    #     #     beams = []
    #     #     if id in flt.object_dispersers:
    #     #         b = flt.object_dispersers[id]['A']
    #     #         beam = grizli.model.BeamCutout(flt, b, conf=flt.conf)
    #     #         #print beam.grism.pad, beam.beam.grow
    #     #         beams.append(beam)
    #     #     else:
    #     #         print flt.grism.parent_file, 'ID %d not found' %(id)
    #     #
    #     #     #plt.imshow(beam.beam.direct*(beam.beam.seg == id), interpolation='Nearest', origin='lower', cmap='viridis_r')
    #     #     self = beam
    #     #
    #     #     #poly_order = 3
    #
    #     if self.grism.filter == 'G102':
    #         if zr is None:
    #             zr = [0.78e4/6563.-1, 1.2e4/5007.-1]
    #         if dz is None:
    #             dz = [0.001, 0.0005]
    #
    #     if self.grism.filter == 'G141':
    #         if zr is None:
    #             zr = [1.1e4/6563.-1, 1.65e4/5007.-1]
    #         if dz is None:
    #             dz = [0.003, 0.0005]
    #
    #     zgrid = utils.log_zgrid(zr, dz=dz[0])
    #     NZ = len(zgrid)
    #
    #     templates = self.load_templates(fwhm=fwhm)
    #     NTEMP = len(templates)
    #
    #     out = self.fit_at_z(z=0., templates=templates, fitter='lstsq',
    #                         poly_order=poly_order)
    #
    #     A, coeffs, chi2, model_2d = out
    #
    #     chi2 = np.zeros(NZ)
    #     coeffs = np.zeros((NZ, coeffs.shape[0]))
    #
    #     for i in range(NZ):
    #         out = self.fit_at_z(z=zgrid[i], templates=templates,
    #                             fitter='lstsq', poly_order=poly_order)
    #
    #         A, coeffs[i,:], chi2[i], model_2d = out
    #         if verbose:
    #             print(utils.NO_NEWLINE + '{0:.4f} {1:9.1f}'.format(zgrid[i], chi2[i]))
    #
    #     # peaks
    #     import peakutils
    #     chi2nu = (chi2.min()-chi2)/self.DoF
    #     indexes = peakutils.indexes((chi2nu+0.01)*(chi2nu > -0.004), thres=0.003, min_dist=20)
    #     num_peaks = len(indexes)
    #     # plt.plot(zgrid, (chi2-chi2.min())/ self.DoF)
    #     # plt.scatter(zgrid[indexes], (chi2-chi2.min())[indexes]/ self.DoF, color='r')
    #
    #
    #     ### zoom
    #     if ((chi2.max()-chi2.min())/self.DoF > 0.01) & (num_peaks < 5):
    #         threshold = 0.01
    #     else:
    #         threshold = 0.001
    #
    #     zgrid_zoom = utils.zoom_zgrid(zgrid, chi2/self.DoF, threshold=threshold, factor=10)
    #     NZOOM = len(zgrid_zoom)
    #
    #     chi2_zoom = np.zeros(NZOOM)
    #     coeffs_zoom = np.zeros((NZOOM, coeffs.shape[1]))
    #
    #     for i in range(NZOOM):
    #         out = self.fit_at_z(z=zgrid_zoom[i], templates=templates,
    #                             fitter='lstsq', poly_order=poly_order)
    #
    #         A, coeffs_zoom[i,:], chi2_zoom[i], model_2d = out
    #         if verbose:
    #             print(utils.NO_NEWLINE + '- {0:.4f} {1:9.1f}'.format(zgrid_zoom[i], chi2_zoom[i]))
    #
    #     zgrid = np.append(zgrid, zgrid_zoom)
    #     chi2 = np.append(chi2, chi2_zoom)
    #     coeffs = np.append(coeffs, coeffs_zoom, axis=0)
    #
    #     so = np.argsort(zgrid)
    #     zgrid = zgrid[so]
    #     chi2 = chi2[so]
    #     coeffs=coeffs[so,:]
    #
    #     ### Best redshift
    #     templates = self.load_templates(line_complexes=False, fwhm=fwhm)
    #     zbest = zgrid[np.argmin(chi2)]
    #     out = self.fit_at_z(z=zbest, templates=templates,
    #                         fitter='lstsq', poly_order=poly_order)
    #
    #     A, coeffs_full, chi2_best, model_full = out
    #
    #     ## Continuum fit
    #     mask = np.isfinite(coeffs_full)
    #     for i, key in enumerate(templates.keys()):
    #         if key.startswith('line'):
    #             mask[self.n_simp+i] = False
    #
    #     model_continuum = np.dot(A, coeffs_full*mask)
    #     model_continuum = model_continuum.reshape(self.beam.sh_beam)
    #
    #     ### 1D spectrum
    #     model1d = utils.SpectrumTemplate(wave=self.beam.lam,
    #                     flux=np.dot(self.y_poly.T,
    #                           coeffs_full[self.n_bg:self.n_poly+self.n_bg]))
    #
    #     cont1d = model1d*1
    #
    #     line_flux = OrderedDict()
    #     for i, key in enumerate(templates.keys()):
    #         temp_i = templates[key].zscale(zbest, coeffs_full[self.n_simp+i])
    #         model1d += temp_i
    #         if not key.startswith('line'):
    #             cont1d += temp_i
    #         else:
    #             line_flux[key.split()[1]] = (coeffs_full[self.n_simp+i] * 1.)
    #                                          #self.beam.total_flux/1.e-17)
    #
    #
    #     fit_data = OrderedDict()
    #     fit_data['poly_order'] = poly_order
    #     fit_data['fwhm'] = fwhm
    #     fit_data['zbest'] = zbest
    #     fit_data['zgrid'] = zgrid
    #     fit_data['A'] = A
    #     fit_data['coeffs'] = coeffs
    #     fit_data['chi2'] = chi2
    #     fit_data['model_full'] = model_full
    #     fit_data['coeffs_full'] = coeffs_full
    #     fit_data['line_flux'] = line_flux
    #     #fit_data['templates_full'] = templates
    #     fit_data['model_cont'] = model_continuum
    #     fit_data['model1d'] = model1d
    #     fit_data['cont1d'] = cont1d
    #
    #     fig = None
    #     if make_figure:
    #         fig = self.show_redshift_fit(fit_data)
    #         #fig.savefig('fit.pdf')
    #
    #     return fit_data, fig

    def show_redshift_fit(self, fit_data):
        """Make a plot based on results from `simple_line_fit`.

        Parameters
        ----------
        fit_data : dict
            returned data from `simple_line_fit`.  I.e.,

            >>> fit_outputs = BeamCutout.simple_line_fit()
            >>> fig = BeamCutout.show_simple_fit_results(fit_outputs)

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object that can be optionally written to a hardcopy file.
        """
        import matplotlib.gridspec

        #zgrid, A, coeffs, chi2, model_best, model_continuum, model1d = fit_outputs

        # Full figure
        fig = plt.figure(figsize=(12, 5))
        #fig = plt.Figure(figsize=(8,4))

        # 1D plots
        gsb = matplotlib.gridspec.GridSpec(3, 1)

        xspec, yspec, yerr = self.beam.optimal_extract(self.grism.data['SCI']
                                                        - self.contam,
                                                        ivar=self.ivar)

        flat_model = self.flat_flam.reshape(self.beam.sh_beam)
        xspecm, yspecm, yerrm = self.beam.optimal_extract(flat_model)

        out = self.beam.optimal_extract(fit_data['model_full'])
        xspecl, yspecl, yerrl = out

        ax = fig.add_subplot(gsb[-2:, :])
        ax.errorbar(xspec/1.e4, yspec, yerr, linestyle='None', marker='o',
                    markersize=3, color='black', alpha=0.5,
                    label='Data (id={0:d})'.format(self.beam.id))

        ax.plot(xspecm/1.e4, yspecm, color='red', linewidth=2, alpha=0.8,
                label=r'Flat $f_\lambda$ ({0})'.format(self.direct.filter))

        zbest = fit_data['zgrid'][np.argmin(fit_data['chi2'])]
        ax.plot(xspecl/1.e4, yspecl, color='orange', linewidth=2, alpha=0.8,
                label='Template (z={0:.4f})'.format(zbest))

        ax.legend(fontsize=8, loc='lower center', scatterpoints=1)

        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel('flux (e-/s)')

        if self.grism.filter == 'G102':
            xlim = [0.7, 1.25]

        if self.grism.filter == 'G141':
            xlim = [1., 1.8]

        xt = np.arange(xlim[0], xlim[1], 0.1)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_xticks(xt)

        ax = fig.add_subplot(gsb[-3, :])
        ax.plot(fit_data['zgrid'], fit_data['chi2']/self.DoF)
        for d in [1, 4, 9]:
            ax.plot(fit_data['zgrid'],
                    fit_data['chi2']*0+(fit_data['chi2'].min()+d)/self.DoF,
                    color='{0:.1f}'.format(d/20.))

        # ax.set_xticklabels([])
        ax.set_ylabel(r'$\chi^2/(\nu={0:d})$'.format(self.DoF))
        ax.set_xlabel('z')
        ax.set_xlim(fit_data['zgrid'][0], fit_data['zgrid'][-1])

        # axt = ax.twiny()
        # axt.set_xlim(np.array(ax.get_xlim())*1.e4/6563.-1)
        # axt.set_xlabel(r'$z_\mathrm{H\alpha}$')

        # 2D spectra
        gst = matplotlib.gridspec.GridSpec(4, 1)
        if 'viridis_r' in plt.colormaps():
            cmap = 'viridis_r'
        else:
            cmap = 'cubehelix_r'

        ax = fig.add_subplot(gst[0, :])
        ax.imshow(self.grism.data['SCI'], vmin=-0.05, vmax=0.2, cmap=cmap,
                  interpolation='Nearest', origin='lower', aspect='auto')
        ax.set_ylabel('Observed')

        ax = fig.add_subplot(gst[1, :])
        mask2d = self.fit_mask.reshape(self.beam.sh_beam)
        ax.imshow((self.grism.data['SCI'] - self.contam)*mask2d,
                  vmin=-0.05, vmax=0.2, cmap=cmap,
                  interpolation='Nearest', origin='lower', aspect='auto')
        ax.set_ylabel('Masked')

        ax = fig.add_subplot(gst[2, :])
        ax.imshow(fit_data['model_full']+self.contam, vmin=-0.05, vmax=0.2,
                  cmap=cmap, interpolation='Nearest', origin='lower',
                  aspect='auto')

        ax.set_ylabel('Model')

        ax = fig.add_subplot(gst[3, :])
        ax.imshow(self.grism.data['SCI']-fit_data['model_full']-self.contam,
                  vmin=-0.05, vmax=0.2, cmap=cmap, interpolation='Nearest',
                  origin='lower', aspect='auto')
        ax.set_ylabel('Resid.')

        for ax in fig.axes[-4:]:
            self.beam.twod_axis_labels(wscale=1.e4,
                                       limits=[xlim[0], xlim[1], 0.1],
                                       mpl_axis=ax)
            self.beam.twod_xlim(xlim, wscale=1.e4, mpl_axis=ax)
            ax.set_yticklabels([])

        ax.set_xlabel(r'$\lambda$')

        for ax in fig.axes[-4:-1]:
            ax.set_xticklabels([])

        gsb.tight_layout(fig, pad=0.1, h_pad=0.01, rect=(0, 0, 0.5, 1))
        gst.tight_layout(fig, pad=0.1, h_pad=0.01, rect=(0.5, 0.01, 1, 0.98))

        return fig

    def simple_line_fit(self, fwhm=48., grid=[1.12e4, 1.65e4, 1, 4],
                        fitter='lstsq', poly_order=3):
        """Function to fit a Gaussian emission line and a polynomial continuum

        Parameters
        ----------
        fwhm : float
            FWHM of the emission line

        grid : list `[l0, l1, dl, skip]`
            The base wavelength array will be generated like

                >>> wave = np.arange(l0, l1, dl)

            and lines will be generated every `skip` wavelength grid points:

                >>> line_centers = wave[::skip]

        fitter : str, 'lstsq' or 'sklearn'
            Least-squares fitting function for determining template
            normalization coefficients.

        order : int (>= 0)
            Polynomial order to use for the continuum

        Returns
        -------
        line_centers : length N `~numpy.array`
            emission line center positions

        coeffs : (N, M) `~numpy.ndarray` where `M = (poly_order+1+1)`
            Normalization coefficients for the continuum and emission line
            templates.

        chi2 : `~numpy.array`
            Chi-squared evaluated at each line_centers[i]

        ok_data : `~numpy.ndarray`
            Boolean mask of pixels used for the Chi-squared calculation.
            Consists of non-masked DQ pixels, non-zero ERR pixels and pixels
            where `self.model > 0.03*self.model.max()` for the flat-spectrum
            model.


        best_model : `~numpy.ndarray`
            2D array with best-fit continuum + line model

        best_model_cont : `~numpy.ndarray`
            2D array with Best-fit continuum-only model.

        best_line_center : float
            wavelength where chi2 is minimized.

        best_line_flux : float
            Emission line flux where chi2 is minimized
        """
        # Test fit
        import sklearn.linear_model
        import numpy.linalg
        clf = sklearn.linear_model.LinearRegression()

        # Continuum
        self.compute_model()
        self.model = self.modelf.reshape(self.beam.sh_beam)

        # OK data where the 2D model has non-zero flux
        ok_data = (~self.mask.flatten()) & (self.ivar.flatten() != 0)
        ok_data &= (self.modelf > 0.03*self.modelf.max())

        # Flat versions of sci/ivar arrays
        scif = (self.grism.data['SCI'] - self.contam).flatten()
        ivarf = self.ivar.flatten()

        # Model: (a_0 x**0 + ... a_i x**i)*continuum + line
        yp, xp = np.indices(self.beam.sh_beam)
        xpf = (xp.flatten() - self.beam.sh_beam[1]/2.)
        xpf /= (self.beam.sh_beam[1]/2)

        # Polynomial continuum arrays
        A_list = [xpf**order*self.modelf for order in range(poly_order+1)]

        # Extra element for the computed line model
        A_list.append(self.modelf*1)
        A = np.vstack(A_list).T

        # Normalized Gaussians on a grid
        waves = np.arange(grid[0], grid[1], grid[2])
        line_centers = waves[grid[3] // 2::grid[3]]

        rms = fwhm/2.35
        gaussian_lines = np.exp(-(line_centers[:, None]-waves)**2/2/rms**2)
        gaussian_lines /= np.sqrt(2*np.pi*rms**2)

        N = len(line_centers)
        coeffs = np.zeros((N, A.shape[1]))
        chi2 = np.zeros(N)
        chi2min = 1e30

        # Loop through line models and fit for template coefficients
        # Compute chi-squared.
        for i in range(N):
            self.compute_model(spectrum_1d=[waves, gaussian_lines[i, :]])

            A[:, -1] = self.model.flatten()
            if fitter == 'lstsq':
                out = np.linalg.lstsq(A[ok_data, :], scif[ok_data], 
                                      rcond=utils.LSTSQ_RCOND)
                lstsq_coeff, residuals, rank, s = out
                coeffs[i, :] += lstsq_coeff
                model = np.dot(A, lstsq_coeff)
            else:
                status = clf.fit(A[ok_data, :], scif[ok_data])
                coeffs[i, :] = clf.coef_
                model = np.dot(A, clf.coef_)

            chi2[i] = np.sum(((scif-model)**2*ivarf)[ok_data])

            if chi2[i] < chi2min:
                chi2min = chi2[i]

        # print chi2
        ix = np.argmin(chi2)
        self.compute_model(spectrum_1d=[waves, gaussian_lines[ix, :]])
        A[:, -1] = self.model.flatten()
        best_coeffs = coeffs[ix, :]*1
        best_model = np.dot(A, best_coeffs).reshape(self.beam.sh_beam)

        # Continuum
        best_coeffs_cont = best_coeffs*1
        best_coeffs_cont[-1] = 0.
        best_model_cont = np.dot(A, best_coeffs_cont)
        best_model_cont = best_model_cont.reshape(self.beam.sh_beam)

        best_line_center = line_centers[ix]
        best_line_flux = coeffs[ix, -1]*self.beam.total_flux/1.e-17

        return (line_centers, coeffs, chi2, ok_data,
                best_model, best_model_cont,
                best_line_center, best_line_flux)

    def show_simple_fit_results(self, fit_outputs):
        """Make a plot based on results from `simple_line_fit`.

        Parameters
        ----------
        fit_outputs : tuple
            returned data from `simple_line_fit`.  I.e.,

            >>> fit_outputs = BeamCutout.simple_line_fit()
            >>> fig = BeamCutout.show_simple_fit_results(fit_outputs)

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Figure object that can be optionally written to a hardcopy file.
        """
        import matplotlib.gridspec

        line_centers, coeffs, chi2, ok_data, best_model, best_model_cont, best_line_center, best_line_flux = fit_outputs

        # Full figure
        fig = plt.figure(figsize=(10, 5))
        #fig = plt.Figure(figsize=(8,4))

        # 1D plots
        gsb = matplotlib.gridspec.GridSpec(3, 1)

        xspec, yspec, yerr = self.beam.optimal_extract(self.grism.data['SCI']
                                                        - self.contam,
                                                        ivar=self.ivar)

        flat_model = self.compute_model(in_place=False)
        flat_model = flat_model.reshape(self.beam.sh_beam)
        xspecm, yspecm, yerrm = self.beam.optimal_extract(flat_model)

        xspecl, yspecl, yerrl = self.beam.optimal_extract(best_model)

        ax = fig.add_subplot(gsb[-2:, :])
        ax.errorbar(xspec/1.e4, yspec, yerr, linestyle='None', marker='o',
                    markersize=3, color='black', alpha=0.5,
                    label='Data (id={0:d})'.format(self.beam.id))

        ax.plot(xspecm/1.e4, yspecm, color='red', linewidth=2, alpha=0.8,
                label=r'Flat $f_\lambda$ ({0})'.format(self.direct.filter))

        ax.plot(xspecl/1.e4, yspecl, color='orange', linewidth=2, alpha=0.8,
                label='Cont+line ({0:.4f}, {1:.2e})'.format(best_line_center/1.e4, best_line_flux*1.e-17))

        ax.legend(fontsize=8, loc='lower center', scatterpoints=1)

        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel('flux (e-/s)')

        ax = fig.add_subplot(gsb[-3, :])
        ax.plot(line_centers/1.e4, chi2/ok_data.sum())
        ax.set_xticklabels([])
        ax.set_ylabel(r'$\chi^2/(\nu={0:d})$'.format(ok_data.sum()))

        if self.grism.filter == 'G102':
            xlim = [0.7, 1.25]

        if self.grism.filter == 'G141':
            xlim = [1., 1.8]

        xt = np.arange(xlim[0], xlim[1], 0.1)
        for ax in fig.axes:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_xticks(xt)

        axt = ax.twiny()
        axt.set_xlim(np.array(ax.get_xlim())*1.e4/6563.-1)
        axt.set_xlabel(r'$z_\mathrm{H\alpha}$')

        # 2D spectra
        gst = matplotlib.gridspec.GridSpec(3, 1)
        if 'viridis_r' in plt.colormaps():
            cmap = 'viridis_r'
        else:
            cmap = 'cubehelix_r'

        ax = fig.add_subplot(gst[0, :])
        ax.imshow(self.grism.data['SCI'], vmin=-0.05, vmax=0.2, cmap=cmap,
                  interpolation='Nearest', origin='lower', aspect='auto')
        ax.set_ylabel('Observed')

        ax = fig.add_subplot(gst[1, :])
        ax.imshow(best_model+self.contam, vmin=-0.05, vmax=0.2, cmap=cmap,
                  interpolation='Nearest', origin='lower', aspect='auto')
        ax.set_ylabel('Model')

        ax = fig.add_subplot(gst[2, :])
        ax.imshow(self.grism.data['SCI']-best_model-self.contam, vmin=-0.05,
                  vmax=0.2, cmap=cmap, interpolation='Nearest',
                  origin='lower', aspect='auto')
        ax.set_ylabel('Resid.')

        for ax in fig.axes[-3:]:
            self.beam.twod_axis_labels(wscale=1.e4,
                                       limits=[xlim[0], xlim[1], 0.1],
                                       mpl_axis=ax)
            self.beam.twod_xlim(xlim, wscale=1.e4, mpl_axis=ax)
            ax.set_yticklabels([])

        ax.set_xlabel(r'$\lambda$')

        for ax in fig.axes[-3:-1]:
            ax.set_xticklabels([])

        gsb.tight_layout(fig, pad=0.1, h_pad=0.01, rect=(0, 0, 0.5, 1))
        gst.tight_layout(fig, pad=0.1, h_pad=0.01, rect=(0.5, 0.1, 1, 0.9))

        return fig
