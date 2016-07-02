"""
Model grism spectra in individual FLTs   
"""
import os
import collections

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

import astropy.io.fits as pyfits
from astropy.table import Table
import astropy.wcs as pywcs

#import stwcs

### Helper functions from a document written by Pirzkal, Brammer & Ryan 
from . import grismconf
from . import utils
from .utils_c import disperse
from .utils_c import interp

### Factors for converting HST countrates to Flamba flux densities
photflam = {'F098M': 6.0501324882418389e-20, 
            'F105W': 3.038658152508547e-20, 
            'F110W': 1.5274130068787271e-20, 
            'F125W': 2.2483414275260141e-20, 
            'F140W': 1.4737154005353565e-20, 
            'F160W': 1.9275637653833683e-20, 
            'F435W': 3.1871480286278679e-19, 
            'F606W': 7.8933594352047833e-20, 
            'F775W': 1.0088466875014488e-19, 
            'F814W': 7.0767633156044843e-20, 
            'VISTAH':1.9275637653833683e-20*0.95,
            'GRISM': 1.e-20}
 
### Filter pivot wavelengths
photplam = {'F098M': 9864.722728110915, 
            'F105W': 10551.046906405772, 
            'F110W': 11534.45855553774, 
            'F125W': 12486.059785775655, 
            'F140W': 13922.907350356367, 
            'F160W': 15369.175708965562,
            'F435W': 4328.256914042873, 
            'F606W': 5921.658489236346,
            'F775W': 7693.297933335407,
            'F814W': 8058.784799323767,
            'VISTAH':1.6433e+04,
            'GRISM': 1.6e4}

no_newline = '\x1b[1A\x1b[1M' # character to skip clearing line on STDOUT printing

### Demo for computing photflam and photplam with pysynphot
if False:
    import pysynphot as S
    n = 1.e-20
    spec = S.FlatSpectrum(n, fluxunits='flam')
    photflam = {}
    photplam = {}
    for filter in ['F098M', 'F105W', 'F110W', 'F125W', 'F140W', 'F160W']:
        bp = S.ObsBandpass('wfc3,ir,%s' %(filter.lower()))
        photplam[filter] = bp.pivot()
        obs = S.Observation(spec, bp)
        photflam[filter] = n/obs.countrate()
        
    for filter in ['F435W', 'F606W', 'F775W', 'F814W']:
        bp = S.ObsBandpass('acs,wfc1,%s' %(filter.lower()))
        photplam[filter] = bp.pivot()
        obs = S.Observation(spec, bp)
        photflam[filter] = n/obs.countrate()

class GrismDisperser(object):
    def __init__(self, id=0, direct=np.zeros((20,20), dtype=np.float), 
                       segmentation=None, origin=[500, 500], pad=0, beam='A',
                       conf=['WFC3','F140W', 'G141']):
        """Object for computing dispersed model spectra
        
        Parameters
        ----------
        id: int
            Only consider pixels in the segmentation image with value `id`.  
            Default of zero to match the default empty segmentation image.
        
        direct: ndarray
            Direct image cutout in f_lambda units (i.e., e-/s times PHOTFLAM).
            Default is a trivial zeros array.
        
        segmentation: ndarray(float32) or None
            Segmentation image.  If None, create a zeros array with the same 
            shape as `direct`.
        
        origin: [int,int]
            `origin` defines the lower left pixel index (y,x) of the `direct` 
            cutout from a larger detector-frame image
        
        pad: int
            Offset between origin = [0,0] and the true lower left pixel of the 
            detector frame.  This can be nonzero for cases where one creates
            a direct image that extends beyond the boundaries of the nominal
            detector frame to model spectra at the edges.
        
        beam: str
            Spectral order to compute.  Must be defined in `self.conf.beams`
        
        conf: [str,str,str] or `grismconf.aXeConf` object.
            Pre-loaded aXe-format configuration file object or if list of 
            strings determine the appropriate configuration filename with 
            `grismconf.get_config_filename` and load it.
            
        Useful Attributes
        -----------------
        sh: 2-tuple
            shape of the direct array
        
        sh_beam: 2-tuple
            computed shape of the 2D spectrum
        
        seg: ndarray
            segmentation array
        
        lam: array
            wavelength along the trace

        ytrace: array
            y pixel center of the trace.  Has same dimensions as sh_beam[1]. 
        
        sensitivity: array
            conversion factor from native e/s to f_lambda flux densities
          
        modelf, model: array, ndarray
            2D model spectrum.  `model` is linked to `modelf` with "reshape", 
            the later which is a flattened 1D array where the fast 
            calculations are actually performed.
        
        model: ndarray
            2D model spectrum linked to `modelf` with reshape.
        
        slx_parent, sly_parent: slices
            slices defined relative to `origin` to match the location of the 
            computed 2D spectrum.
        """
        
        self.id = id
        
        ### lower left pixel of the `direct` array in native detector
        ### coordinates
        self.origin = origin
        self.pad = pad
        
        ### Direct image
        self.direct = direct
        self.sh = self.direct.shape
        if self.direct.dtype is not np.float:
            self.direct = np.cast[np.float](self.direct)
        
        ### Segmentation image, defaults to all zeros
        if segmentation is None:
            self.seg = np.zeros_like(self.direct, dtype=np.float32)
        else:
            self.seg = segmentation
            if self.seg.dtype is not np.float32:
                self.seg = np.cast[np.float32](self.seg)
                
        if isinstance(conf, list):
            conf_f = grismconf.get_config_filename(conf[0], conf[1], conf[2])
            self.conf = grismconf.load_grism_config(conf_f)
        else:
            self.conf = conf
        
        ### Initialize attributes
        self.xc = self.sh[1]/2+self.origin[1]
        self.yc = self.sh[0]/2+self.origin[0]
        self.beam = beam
        
        ### Get dispersion parameters at the reference position
        self.dx = self.conf.dxlam[self.beam] #+xcenter-xoff
        self.ytrace_beam, self.lam_beam = self.conf.get_beam_trace(
                                                         x=self.xc-self.pad,
                                                         y=self.yc-self.pad,
                                                         dx=self.dx,
                                                         beam=self.beam)
                
        ### Integer trace
        # 20 for handling int of small negative numbers    
        dyc = np.cast[int](self.ytrace_beam+20)-20+1 
        
        ### Account for pixel centering of the trace
        self.yfrac_beam = self.ytrace_beam - np.floor(self.ytrace_beam)
        
        ### Interpolate the sensitivity curve on the wavelength grid. 
        ysens = self.lam_beam*0
        so = np.argsort(self.lam_beam)
        ysens[so] = interp.interp_conserve_c(self.lam_beam[so],
                                 self.conf.sens[self.beam]['WAVELENGTH'], 
                                 self.conf.sens[self.beam]['SENSITIVITY'])
        self.lam_sort = so
        
        ### Needs term of delta wavelength per pixel for flux densities
        dl = np.abs(np.append(self.lam_beam[1] - self.lam_beam[0],
                              np.diff(self.lam_beam)))
        ysens *= dl*1.e-17
        self.sensitivity_beam = ysens
        
        ### Initialize the model arrays
        self.NX = len(self.dx)
        self.sh_beam = (self.sh[0], self.sh[1]+self.NX)
        
        self.modelf = np.zeros(np.product(self.sh_beam))
        self.model = self.modelf.reshape(self.sh_beam)
        self.idx = np.arange(self.modelf.size).reshape(self.sh_beam)
        
        ## Indices of the trace in the flattened array 
        self.x0 = np.array(self.sh)/2
        self.dxpix = self.dx - self.dx[0] + self.x0[1] #+ 1
        self.flat_index = self.idx[dyc + self.x0[0], self.dxpix]
        
        ###### Trace, wavelength, sensitivity across entire 2D array
        self.dxfull = np.arange(self.sh_beam[1], dtype=int) 
        self.dxfull += self.dx[0]-self.x0[1]
        
        self.ytrace, self.lam = self.conf.get_beam_trace(x=self.xc,
                         y=self.yc, dx=self.dxfull, beam=self.beam)
        
        ysens = self.lam*0
        so = np.argsort(self.lam)
        ysens[so] = interp.interp_conserve_c(self.lam[so],
                                 self.conf.sens[self.beam]['WAVELENGTH'], 
                                 self.conf.sens[self.beam]['SENSITIVITY'])
        
        self.sensitivity = ysens
        
        ## Slices of the parent array based on the origin parameter
        self.slx_parent = slice(self.origin[1] + self.dxfull[0] + self.x0[1],
                            self.origin[1] + self.dxfull[-1] + self.x0[1]+1)
        
        self.sly_parent = slice(self.origin[0], self.origin[0] + self.sh[0])
        
        self.spectrum_1d = [None, None]
        
    def compute_model(self, id=None, thumb=None, spectrum_1d=None,
                      in_place=True, outdata=None):
        """Compute a model 2D grism spectrum

        Parameters
        ----------
        id: int
            Only consider pixels in the segmentation image (`self.seg`) with 
            values equal to `id`.
        
        thumb: array with shape = `self.sh` or None
            Optional direct image.  If `None` then use `self.direct`.
        
        spectrum_1d: [array, array] or None
            Optional 1D template [wave, flux] to use for the 2D grism model.
            If `None`, then implicitly assumes flat f_lambda spectrum.
                
        in_place: bool
            If True, put the 2D model in `self.model` and `self.modelf`, 
            otherwise put the output in a clean array or preformed `outdata`. 
            
        outdata: array with shape = `self.sh_beam`
            Preformed array to which the 2D model is added, if `in_place` is
            False.
            
        Returns
        -------
        model: ndarray
            If `in_place` is False, returns the 2D model spectrum.  Otherwise
            the result is stored in `self.model` and `self.modelf`.
        """
        
        if id is None:
            id = self.id
        else:
            self.id = id
            
        ### Template (1D) spectrum interpolated onto the wavelength grid
        if spectrum_1d is not None:
            self.spectrum_1d = spectrum_1d
            xspec, yspec = spectrum_1d
            scale_spec = self.sensitivity_beam*0.
            int_func = interp.interp_conserve_c
            scale_spec[self.lam_sort] = int_func(self.lam_beam[self.lam_sort],
                                                xspec, yspec)
        else:
            scale_spec = 1.
            self.spectrum_1d = [None, None]
            
        ### Output data, fastest is to compute in place but doesn't zero-out
        ### previous result                    
        if in_place:
            self.modelf *= 0
            outdata = self.modelf
        else:
            if outdata is None:
                outdata = self.modelf*0
                
        ### Optionally use a different direct image
        if thumb is None:
            thumb = self.direct
        else:
            if thumb.shape != self.sh:
                print """
Error: `thumb` must have the same dimensions as the direct image! (%d,%d)      
                """ %(self.sh[0], self.sh[1])
                return False

        ### Now compute the dispersed spectrum using the C helper
        status = disperse.disperse_grism_object(thumb, self.seg, id,
                                 self.flat_index, self.yfrac_beam,
                                 self.sensitivity_beam*scale_spec,
                                 outdata, self.x0, np.array(self.sh),
                                 self.x0, np.array(self.sh_beam))

        if not in_place:
            return outdata
        else:
            return True
    
    def optimal_extract(self, data, bin=0, ivar=1.):        
        """Horne (1986) optimally-weighted 1D extraction
        
        Parameters
        ----------
        data: ndarray with shape == `self.sh_beam`
            2D data to extract
        
        bin: int, optional
            Simple boxcar averaging of the output 1D spectrum
        
        ivar: float or ndarray with shape == `self.sh_beam`
            Inverse variance array or scalar float that multiplies the 
            optimal weights
            
        Returns
        -------
        wave, opt_flux, opt_rms: array-like
            `wave` is the wavelength of 1D array
            `opt_flux` is the optimally-weighted 1D extraction
            `opt_rms` is the weighted uncertainty of the 1D extraction
            
            All are optionally binned in wavelength if `bin` > 1.
        """
        import scipy.ndimage as nd
               
        if not hasattr(self, 'optimal_profile'):
            m = self.compute_model(id=self.id, in_place=False)
            m = m.reshape(self.sh_beam)
            m[m < 0] = 0
            self.optimal_profile = m/m.sum(axis=0)
        
        if data.shape != self.sh_beam:
            print """
`data` (%d,%d) must have the same shape as the data array (%d,%d)
            """ %(data.shape[0], data.shape[1], self.sh_beam[0], 
                  self.sh_beam[1])
            return False

        if not isinstance(ivar, float):
            if ivar.shape != self.sh_beam:
                print """
`ivar` (%d,%d) must have the same shape as the data array (%d,%d)
                """ %(ivar.shape[0], ivar.shape[1], self.sh_beam[0], 
                      self.sh_beam[1])
                return False
                
        num = self.optimal_profile*data*ivar
        den = self.optimal_profile**2*ivar
        opt_flux = num.sum(axis=0)/den.sum(axis=0)
        opt_var = 1./den.sum(axis=0)
                
        if bin > 0:
            kern = np.ones(bin, dtype=float)/bin
            opt_flux = nd.convolve(opt, kern)[bin/2::bin]
            opt_var = nd.convolve(opt_var, kern**2)[bin/2::bin]
            wave = self.lam[bin/2::bin]
        else:
            wave = self.lam
                
        opt_rms = np.sqrt(opt_var)
        opt_rms[opt_var == 0] = 0
        
        return wave, opt_flux, opt_rms
    
    def add_to_full_image(self, data, full_array):
        """Add spectrum cutout back to the full array
        
        Parameters
        ----------
        data: ndarray shape `self.sh_beam` (e.g., self.model)
            Spectrum cutout
        
        full_array: ndarray
            Full detector array, where the lower left pixel of `data` is given
            by `origin`.
        
        `data` is *added* to `full_array` in place, so, for example, to 
        subtract `self.model` from the full array, call the function with 
        
        >>> self.add_to_full_image(-self.model, full_array)
         
        """
        
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
        full_array[sly, slx] += data[oky,:][:,okx]
        return True

class ImageData(object):
    """Container for image data with WCS, etc."""
    def __init__(self, sci=np.zeros((1014,1014)), err=None, dq=None,
                 header=None, photflam=1., origin=[0,0], 
                 instrument='WFC3', filter='G141', hdulist=None, sci_ext=1):
        """
        Parameters
        ----------
        data: ndarray
            Data for direct image
        
        header: astropy.io.fits.Header
            Associated header with `data` that contains WCS information
        
        photflam: float
            Multiplicative conversion factor to scale `data` to set units 
            to f_lambda flux density.  If data is grism spectra, then use
            photflam=1
        
        origin: [int,int]
            Origin of lower left pixel in detector coordinates
        """
        
        ### Easy way, get everything from an image HDU list
        if isinstance(hdulist, pyfits.HDUList):
            sci = np.cast[float](hdulist['SCI',sci_ext].data)
            err = np.cast[float](hdulist['ERR',sci_ext].data)
            dq = np.cast[int](hdulist['DQ',sci_ext].data)
            origin = [0,0]
            
            header = hdulist['SCI',sci_ext].header.copy()
            
            h0 = hdulist[0].header
            photflam = hdulist[0].header['PHOTFLAM']
            instrument = hdulist[0].header['INSTRUME']
            filter = utils.get_hst_filter(h0)
            
        self.sci_data = sci*photflam
        self.photflam = photflam
        self.sh = np.array(self.sci_data.shape)
        self.pad = 0
        self.origin = origin
        
        if err is None:
            self.err_data = np.zeros_like(self.sci_data)
        else:
            self.err_data = err*photflam
            if self.err_data.shape != tuple(self.sh):
                raise ValueError ('err and sci arrays have different shapes!')
        
        if dq is None:
            self.dq_data = np.zeros_like(self.sci_data, dtype=int)
        else:
            self.dq_data = dq
            if self.dq_data.shape != tuple(self.sh):
                raise ValueError ('err and dq arrays have different shapes!')
        
        ### Header-like parameters
        self.filter = filter
        self.instrument = instrument
        self.header = header
        
        self.wcs = None
        if self.header is not None:
            self.get_wcs()
        else:
            self.header = pyfits.Header()
            
    def unset_dq(self):
        """Flip OK data quality bits using utils.unset_dq_bits
        
        OK bits are defined as 
        
            okbits_instrument = {'WFC3': 32+64+512, # blob OK
                                 'NIRISS': 0,
                                 'WFIRST': 0,}
        """
                             
        okbits_instrument = {'WFC3': 32+64+512, # blob OK
                             'NIRISS': 0,
                             'WFIRST': 0,}
        
        if self.instrument not in okbits_instrument:
            okbits = 1
        else:
            okbits = okbits_instrument[self.instrument]
                
        self.dq_data = utils.unset_dq_bits(self.dq_data, okbits=okbits)
    
    def flag_negative(self, sigma=-3):
        """Flag negative data values with dq=4
        
        Parameters
        ----------
        sigma: float
            Threshold for setting bad data
        
        Returns
        -------
        n_negative: int
            Number of flagged negative pixels
            
        If `self.err_data` is zeros, do nothing.
        """
        if self.err_data.max() == 0:
            return 0
        
        bad = self.sci_data < sigma*self.err_data
        self.dq_data[bad] |= 4
        return bad.sum()
        
            
    def get_wcs(self):
        """Get WCS from header"""
        self.wcs = pywcs.WCS(self.header)
        if not hasattr(self.wcs, 'pscale'):
            self.wcs.pscale = np.sqrt(self.wcs.wcs.cd[0,0]**2 +
                                      self.wcs.wcs.cd[1,0]**2)*3600.
    
    def add_padding(self, pad=0):
        """Pad the data array and update WCS keywords"""
        
        ### Update data array
        new_sh = self.sh + 2*pad
        for attr in ['sci_data', 'err_data', 'dq_data']:
            #for data in [self.sci_data, self.err_data, self.dq_data]:
            data = self.__getattribute__(attr)
            new_data = np.zeros(new_sh, dtype=data.dtype)
            new_data[pad:-pad, pad:-pad] += data
            self.__setattr__(attr, new_data)
            
        # new_sci = np.zeros(new_sh, dtype=self.sci.dtype)
        # new_sci[pad:-pad,pad:-pad] = self.sci
        # self.sci = new_sci
        
        self.sh = new_sh
        self.pad += pad
        
        ### Padded image dimensions
        self.header['NAXIS1'] += 2*pad
        self.header['NAXIS2'] += 2*pad
        self.wcs.naxis1 = self.wcs._naxis1 = self.header['NAXIS1']
        self.wcs.naxis2 = self.wcs._naxis2 = self.header['NAXIS2']

        ### Add padding to WCS
        self.header['CRPIX1'] += pad
        self.header['CRPIX2'] += pad        
        self.wcs.wcs.crpix[0] += pad
        self.wcs.wcs.crpix[1] += pad
        if self.wcs.sip is not None:
            self.wcs.sip.crpix[0] += pad
            self.wcs.sip.crpix[1] += pad           
    
    def shrink_large_hdu(self, hdu=None, extra=100, verbose=False):
        """Shrink large image mosaic to speed up blotting
        
        Parameters
        ----------
        hdu: astropy.io.ImageHDU
            Input reference HDU
        
        extra: int
            Extra border to put around `self.sci_data` to ensure the reference
            image is large enough to encompass the distorted image
            
        Returns
        -------
        new_hdu: astropy.io.ImageHDU
            Image clipped to encompass `self.sci_data` + margin of `extra`
            pixels.
        
        Make a cutout of the larger reference image around the desired FLT
        image to make blotting faster for large reference images.
        """
        ref_wcs = pywcs.WCS(hdu.header)
        
        ### Borders of the flt frame
        naxis = [self.header['NAXIS1'], self.header['NAXIS2']]
        xflt = [-extra, naxis[0]+extra, naxis[0]+extra, -extra]
        yflt = [-extra, -extra, naxis[1]+extra, naxis[1]+extra]
                
        raflt, deflt = self.wcs.all_pix2world(xflt, yflt, 0)
        xref, yref = np.cast[int](ref_wcs.all_world2pix(raflt, deflt, 0))
        ref_naxis = [hdu.header['NAXIS1'], hdu.header['NAXIS2']]
        
        ### Slices of the reference image
        xmi = np.maximum(0, xref.min())
        xma = np.minimum(ref_naxis[0], xref.max())
        slx = slice(xmi, xma)
        
        ymi = np.maximum(0, yref.min())
        yma = np.minimum(ref_naxis[1], yref.max())
        sly = slice(ymi, yma)
        
        if ((xref.min() < 0) | (yref.min() < 0) | 
            (xref.max() > ref_naxis[0]) | (yref.max() > ref_naxis[1])):
            if verbose:
                print ('Segmentation cutout: x=%s, y=%s [Out of range]' 
                    %(slx, sly))
            return hdu
        else:
            if verbose:
                print 'Segmentation cutout: x=%s, y=%s' %(slx, sly)
        
        ### Sliced subimage
        slice_wcs = ref_wcs.slice((sly, slx))
        slice_header = hdu.header.copy()
        hwcs = slice_wcs.to_header()

        for k in hwcs.keys():
           if not k.startswith('PC'):
               slice_header[k] = hwcs[k]
                
        slice_data = hdu.data[sly, slx]*1
        new_hdu = pyfits.ImageHDU(data=slice_data, header=slice_header)
        
        return new_hdu
                    
    def blot_from_hdu(self, hdu=None, segmentation=False, grow=3, 
                      interp='poly5'):
        """Blot a rectified reference image to detector frame
        
        Parameters
        ----------
        hdu: astropy.io.ImageHDU
            HDU of the reference image
        
        segmentation: bool, False
            If True, treat the reference image as a segmentation image and 
            preserve the integer values in the blotting.
        
        grow: int, default=3
            Number of pixels to dilate the segmentation regions
            
        interp: str
            Form of interpolation to use when blotting float image pixels. 
            Valid options::
            "nearest", "linear", "poly3", "poly5" (default), "spline3", "sinc"
                    
        Returns
        -------
        blotted: ndarray
            Blotted array with the same shape and WCS as `self.sci_data`.
        """
        
        import astropy.wcs
        from drizzlepac import astrodrizzle
        
        #ref = pyfits.open(refimage)
        if hdu.data.dtype != np.float32:
            hdu.data = np.cast[np.float32](hdu.data)
        
        refdata = hdu.data
        if 'ORIENTAT' in hdu.header.keys():
            hdu.header.remove('ORIENTAT')
            
        if segmentation:
            seg_ones = np.cast[np.float32](refdata > 0)-1
        
        ref_wcs = pywcs.WCS(hdu.header)        
        flt_wcs = self.wcs.copy()
        
        ### Fix some wcs attributes that might not be set correctly
        for wcs in [ref_wcs, flt_wcs]:
            if (not hasattr(wcs.wcs, 'cd')) & hasattr(wcs.wcs, 'pc'):
                wcs.wcs.cd = wcs.wcs.pc
                
            if hasattr(wcs, 'idcscale'):
                if wcs.idcscale is None:
                    wcs.idcscale = np.sqrt(np.sum(wcs.wcs.cd[0,:]**2))*3600.
            else:
                wcs.idcscale = np.sqrt(np.sum(wcs.wcs.cd[0,:]**2))*3600.
            
            wcs.pscale = np.sqrt(wcs.wcs.cd[0,0]**2 +
                                 wcs.wcs.cd[1,0]**2)*3600.
        
        if segmentation:
            ### Handle segmentation images a bit differently to preserve
            ### integers.
            ### +1 here is a hack for some memory issues
            blotted_seg = astrodrizzle.ablot.do_blot(refdata+0, ref_wcs,
                                flt_wcs, 1, coeffs=True, interp='nearest',
                                sinscl=1.0, stepsize=1, wcsmap=None)
            
            blotted_ones = astrodrizzle.ablot.do_blot(seg_ones+1, ref_wcs,
                                flt_wcs, 1, coeffs=True, interp='nearest',
                                sinscl=1.0, stepsize=1, wcsmap=None)
            
            blotted_ones[blotted_ones == 0] = 1
            ratio = np.round(blotted_seg/blotted_ones)
            seg = nd.maximum_filter(ratio, size=grow, mode='constant', cval=0)
            ratio[ratio == 0] = seg[ratio == 0]
            blotted = ratio
            
        else:
            ### Floating point data
            blotted = astrodrizzle.ablot.do_blot(refdata, ref_wcs, flt_wcs, 1,
                                coeffs=True, interp=interp, sinscl=1.0,
                                stepsize=10, wcsmap=None)
        
        return blotted
    
    def get_slice(self, origin=[500, 500], N=20):
        """Return cutout version of `self`
        
        Parameters
        ----------
        origin: [int, int]
            Lower left pixel of the slice to cut out
        
        N: int
            Size of the (square) cutout, in pixels
        
        Returns
        -------
        slice_obj: ImageData
            New `ImageData` object of the subregion
        
        slx, sly: 
            Slice parameters of the parent array used to cut out the data
        """
        
        ### Test dimensions
        if (origin[0]-N < 0) | (origin[0]+N > self.sh[0]):
            raise ValueError ('Out of range in y')
        
        if (origin[1]-N < 0) | (origin[1]+N > self.sh[1]):
            raise ValueError ('Out of range in x')
        
        ### Sliced subimage
        sly = slice(origin[0], origin[0]+N)
        slx = slice(origin[1], origin[1]+N)
        
        slice_origin = [self.origin[i] + origin[i] for i in range(2)]
        
        slice_wcs = self.wcs.slice((sly, slx))
        slice_wcs.naxis1 = slice_wcs._naxis1 = N
        slice_wcs.naxis2 = slice_wcs._naxis2 = N
        slice_header = self.header.copy()
        slice_header['NAXIS1'] = slice_header['NAXIS2'] = N
        
        ### Sliced WCS keywords
        hwcs = slice_wcs.to_header()
        for k in hwcs.keys():
            if not k.startswith('PC'):
                slice_header[k] = hwcs[k]
        
        ### Generate new object        
        slice_obj = ImageData(sci=self.sci_data[sly, slx]/self.photflam, 
                        err=self.err_data[sly, slx]/self.photflam, 
                        dq=self.dq_data[sly, slx], 
                        header=slice_header, photflam=self.photflam,
                        origin=slice_origin, instrument=self.instrument,
                        filter=self.filter)
        
        slice_obj.pad = slice_obj.pad             
        
        return slice_obj, slx, sly
        
class GrismFLT(object):
    """
    Scripts for simple modeling of individual grism FLT images
    
    tbd: 
        o helper functions for extracting 2D spectra
        o lots of book-keeping for handling SExtractor objects & catalogs
        ...
        
    """
    def __init__(self, flt_file='ico205lwq_flt.fits', sci_ext=('SCI',1),
                 direct_image=None, refimage=None, segimage=None, refext=0,
                 verbose=True, pad=100, shrink_segimage=True,
                 force_grism=None):
        
        ### Read the FLT FITS File
        self.flt_file = flt_file
        ### Simulation mode
        if (flt_file is None) & (direct_image is not None):
            self.flt_file = direct_image
        
        self.sci_ext = sci_ext
        
        #self.wcs = pywcs.WCS(self.im['SCI',1].header)
        #self.im = pyfits.open(self.flt_file)
        self.pad = pad
                    
        self.read_flt()
        self.flt_wcs = pywcs.WCS(self.im[tuple(sci_ext)].header)
        
        ### Padded image dimensions
        self.flt_wcs.naxis1 = self.im_header['NAXIS1']+2*self.pad
        self.flt_wcs.naxis2 = self.im_header['NAXIS2']+2*self.pad
        self.flt_wcs._naxis1 = self.flt_wcs.naxis1
        self.flt_wcs._naxis2 = self.flt_wcs.naxis2
        
        ### Add padding to WCS
        self.flt_wcs.wcs.crpix[0] += self.pad
        self.flt_wcs.wcs.crpix[1] += self.pad
        
        if self.flt_wcs.sip is not None:
            self.flt_wcs.sip.crpix[0] += self.pad
            self.flt_wcs.sip.crpix[1] += self.pad           
                
        self.refimage = refimage
        self.refimage_im = None
        
        self.segimage = segimage
        self.segimage_im = None
        self.seg = np.zeros(self.im_data['SCI'].shape, dtype=np.float32)
        
        if direct_image is not None:
            ### Case where FLT is a grism exposure and FLT direct 
            ### image provided
            if verbose:
                print '%s / Blot reference image: %s' %(self.flt_file,
                                                        refimage)
            
            self.refimage = direct_image
            self.refimage_im = pyfits.open(self.refimage)
            self.filter = self.get_filter(self.refimage_im[refext].header)
                                        
            self.photflam = photflam[self.filter]
            self.flam = self.refimage_im[self.sci_ext].data*self.photflam
            
            ### Bad DQ bits
            dq = self.unset_dq_bits(dq=self.refimage_im['DQ'].data,
                                    okbits=32+64)            
            self.dmask = dq == 0
            
        if refimage is not None:
            ### Case where FLT is a grism exposure and reference direct 
            ### image provided
            if verbose:
                print '%s / Blot reference image: %s' %(self.flt_file,
                                                        refimage)
            
            self.refimage_im = pyfits.open(self.refimage)
            self.filter = self.get_filter(self.refimage_im[refext].header)
                                        
            self.flam = self.get_blotted_reference(self.refimage_im,
                                                   segmentation=False)
            self.photflam = photflam[self.filter]
            self.flam *= self.photflam
            self.dmask = np.ones(self.flam.shape, dtype=bool)
        
        if segimage is not None:
            if verbose:
                print '%s / Blot segmentation image: %s' %(self.flt_file,
                                                           segimage)
            
            self.segimage_im = pyfits.open(self.segimage)
            if shrink_segimage:
                self.shrink_segimage_to_flt()
                
            self.process_segimage()
        
        self.pivot = photplam[self.filter]
                        
        # This needed for the C dispersing function
        self.clip = np.cast[np.double](self.flam*self.dmask)
        
        ### Read the configuration file.  
        ## xx generalize this to get the grism information from the FLT header
        #self.grism = self.im_header0['FILTER'].upper()
        self.grism = force_grism
        if self.grism is None:
            self.grism = self.get_filter(self.im_header0)
        
        self.instrume = self.im_header0['INSTRUME']
        
        self.conf_file = grismconf.get_config_filename(self.instrume,
                                                      self.filter, self.grism)
        
        self.conf = grismconf.load_grism_config(self.conf_file)
                
        #### Get dispersion PA, requires conf for tilt of 1st order
        self.get_dispersion_PA()
                
        ### full_model is a flattened version of the FLT image
        self.modelf = np.zeros(self.sh_pad[0]*self.sh_pad[1])
        self.model = self.modelf.reshape(self.sh_pad)
        self.idx = np.arange(self.modelf.size).reshape(self.sh_pad)
                    
    def read_flt(self):
        """
        Read 'SCI', 'ERR', and 'DQ' extensions of `self.flt_file`, 
        add padding if necessary.
        
        Store result in `self.im_data`.
        """
        
        self.im = pyfits.open(self.flt_file)
        self.im_header0 = self.im[0].header.copy()
        self.im_header = self.im[tuple(self.sci_ext)].header.copy()
        
        self.sh_flt = list(self.im[tuple(self.sci_ext)].data.shape)
        self.sh_pad = [x+2*self.pad for x in self.sh_flt]
        
        slx = slice(self.pad, self.pad+self.sh_flt[1])
        sly = slice(self.pad, self.pad+self.sh_flt[0])
        
        self.im_data = {}
        for ext in ['SCI', 'ERR', 'DQ']:
            iext = (ext, self.sci_ext[1])
            self.im_data[ext] = np.zeros(self.sh_pad,
                                         dtype=self.im[iext].data.dtype)
                                         
            self.im_data[ext][sly, slx] = self.im[iext].data*1
        
        self.im_data['DQ'] = self.unset_dq_bits(dq=self.im_data['DQ'],
                                                okbits=32+64+512)
        
        ### Negative pixels
        neg_pixels = self.im_data['SCI'] < -3*self.im_data['ERR']
        self.im_data['DQ'][neg_pixels] |= 1024
        self.im_data['SCI'][self.im_data['DQ'] > 0] = 0
        
        self.im_data_sci_background = False
    
    def get_filter(self, header):
        """
        Get simple filter name out of an HST image header.  
        
        ACS has two keywords for the two filter wheels, so just return the 
        non-CLEAR filter.
        """
        if header['INSTRUME'].strip() == 'ACS':
            for i in [1,2]:
                filter = header['FILTER%d' %(i)]
                if 'CLEAR' in filter:
                    continue
                else:
                    filter = acsfilt
        else:
            filter = header['FILTER'].upper()
        
        return filter
                
    def clean_for_mp(self):
        """
        zero out io.fits objects to make suitable for multiprocessing
        parallelization
        """
        self.im = None
        self.refimage_im = None
        self.segimage_im = None
    
    def re_init(self):
        """
        Open io.fits objects again
        """
        self.im = pyfits.open(self.flt_file)
        if self.refimage:
            self.refimage_im = pyfits.open(self.refimage)
        if self.segimage:
            self.segimage_im = pyfits.open(self.segimage)
    
    def save_generated_data(self, verbose=True):
        """
        Save flam, seg, and modelf arrays to an HDF5 file
        """
        #for self in g.FLTs:
        import h5py
        h5file = self.flt_file.replace('flt.fits','flt.model.hdf5')
        h5f = h5py.File(h5file, 'w')
        
        flam = h5f.create_dataset('flam', data=self.flam)
        flam.attrs['refimage'] = self.refimage
        flam.attrs['pad'] = self.pad
        flam.attrs['filter'] = self.filter
        flam.attrs['photflam'] = self.photflam
        flam.attrs['pivot'] = self.pivot
        
        seg = h5f.create_dataset('seg', data=self.seg, compression='gzip')
        seg.attrs['segimage'] = self.segimage
        seg.attrs['pad'] = self.pad
        
        model = h5f.create_dataset('modelf', data=self.modelf,
                                   compression='gzip')
        
        h5f.close()
    
        if verbose:
            print 'Save data to %s' %(h5file)
            
    def load_generated_data(self, verbose=True):
        """
        Load flam, seg, and modelf arrays from an HDF5 file
        """
        import h5py
        h5file = self.flt_file.replace('flt.fits','flt.model.hdf5')
        if not os.path.exists(h5file):
            return False
        
        if verbose:
            print 'Load data from %s' %(h5file)
        
        h5f = h5py.File(h5file, 'r')
        if flam in h5f:
            flam = h5f['flam']
            if flam.attrs['refimage'] != self.refimage:
                print ("`refimage` doesn't match!  saved=%s, new=%s"
                       %(flam.attrs['refimage'], self.refimage))
            else:
                self.flam = np.array(flam)
                for attr in ['pad', 'filter', 'photflam', 'pivot']:
                    self.__setattr__(attr, flam.attrs[attr])
                
            
        if 'seg' in h5f:
            seg = h5f['seg']
            if flam.attrs['segimage'] != self.segimage:
                print ("`segimage` doesn't match!  saved=%s, new=%s"
                       %(flam.attrs['segimage'], self.segimage))
            else:
                self.seg = np.array(seg)
                for attr in ['pad']:
                    self.__setattr__(attr, flam.attrs[attr])
        
        if 'modelf' in h5f:
            self.modelf = np.array(h5f['modelf'])
            self.model = self.modelf.reshape(self.sh_pad)
            
    def get_dispersion_PA(self):
        """
        Compute exact PA of the dispersion axis, including tilt of the 
        trace and the FLT WCS
        """
        from astropy.coordinates import Angle
        import astropy.units as u
                    
        ### extra tilt of the 1st order grism spectra
        x0 =  self.conf.conf['BEAMA']
        dy_trace, lam_trace = self.conf.get_beam_trace(x=507, y=507, dx=x0,
                                                       beam='A')
        
        extra = np.arctan2(dy_trace[1]-dy_trace[0], x0[1]-x0[0])/np.pi*180
                
        ### Distorted WCS
        crpix = self.flt_wcs.wcs.crpix
        xref = [crpix[0], crpix[0]+1]
        yref = [crpix[1], crpix[1]]
        r, d = self.all_pix2world(xref, yref)
        pa =  Angle((extra + 
                     np.arctan2(np.diff(r), np.diff(d))[0]/np.pi*180)*u.deg)
        
        self.dispersion_PA = pa.wrap_at(360*u.deg).value
        
    def unset_dq_bits(self, dq=None, okbits=32+64+512, verbose=False):
        """
        Unset bit flags from a (WFC3/IR) DQ array
        
        32, 64: these pixels usually seem OK
           512: blobs not relevant for grism exposures
        """
        bin_bits = np.binary_repr(okbits)
        n = len(bin_bits)
        for i in range(n):
            if bin_bits[-(i+1)] == '1':
                if verbose:
                    print 2**i
                
                dq -= (dq & 2**i)
        
        return dq
        
    def all_world2pix(self, ra, dec, idx=1, tolerance=1.e-4):
        """
        Handle awkward pywcs.all_world2pix for scalar arguments
        """
        if np.isscalar(ra):
            x, y = self.flt_wcs.all_world2pix([ra], [dec], idx,
                                 tolerance=tolerance, maxiter=100, quiet=True)
            return x[0], y[0]
        else:
            return self.flt_wcs.all_world2pix(ra, dec, idx,
                                 tolerance=tolerance, maxiter=100, quiet=True)
    
    def all_pix2world(self, x, y, idx=1):
        """
        Handle awkward pywcs.all_world2pix for scalar arguments
        """
        if np.isscalar(x):
            ra, dec = self.flt_wcs.all_pix2world([x], [y], idx)
            return ra[0], dec[0]
        else:
            return self.flt_wcs.all_pix2world(x, y, idx)
    
    def blot_catalog(self, catalog_table, ra='ra', dec='dec',
                     sextractor=False):
        """
        Make x_flt and y_flt columns of detector coordinates in `self.catalog` 
        using the image WCS and the sky coordinates in the `ra` and `dec`
        columns.
        """
        from astropy.table import Column
        
        if sextractor:
            ra, dec = 'X_WORLD', 'Y_WORLD'
            if ra.lower() in catalog_table.colnames:
                ra, dec = ra.lower(), dec.lower()
        
        
        tolerance=-4
        xy = None
        ## Was having problems with `wcs not converging` with some image
        ## headers, so was experimenting between astropy.wcs and stwcs.HSTWCS.  
        ## Problem was probably rather the header itself, so this can likely
        ## be removed and simplified
        for wcs, wcsname in zip([self.flt_wcs, 
                                 pywcs.WCS(self.im_header, relax=True)], 
                                 ['astropy.wcs', 'HSTWCS']):
            if xy is not None:
                break
            for i in range(4):    
                try:
                    xy = wcs.all_world2pix(catalog_table[ra],
                                           catalog_table[dec], 1,
                                           tolerance=np.log10(tolerance+i),
                                           quiet=True)
                    break
                except:
                    print ('%s / all_world2pix failed to ' %(wcsname) + 
                           'converge at tolerance = %d' %(tolerance+i))
                
        sh = self.im_data['SCI'].shape
        keep = ((xy[0] > 0) & 
                (xy[0] < sh[1]) & 
                (xy[1] > (self.pad-5)) & 
                (xy[1] < (self.pad+self.sh_flt[0]+5)))
                
        self.catalog = catalog_table[keep]
        
        for col in ['x_flt', 'y_flt']:
            if col in self.catalog.colnames:
                self.catalog.remove_column(col)
                
        self.catalog.add_column(Column(name='x_flt', data=xy[0][keep]))
        self.catalog.add_column(Column(name='y_flt', data=xy[1][keep]))
        
        if sextractor:
            self.catalog.rename_column('X_WORLD', 'ra')
            self.catalog.rename_column('Y_WORLD', 'dec')
            self.catalog.rename_column('NUMBER', 'id')
            
        return self.catalog
        
        if False:
            ### Compute full model
            self.modelf*=0
            for mag_lim, beams in zip([[10,24], [24,28]], ['ABCDEF', 'A']): 
                ok = ((self.catalog['MAG_AUTO'] > mag_lim[0]) &
                      (self.catalog['MAG_AUTO'] < mag_lim[1]))
                      
                so = np.argsort(self.catalog['MAG_AUTO'][ok])
                for i in range(ok.sum()):
                    ix = so[i]
                    print '%d id=%d mag=%.2f' %(i+1,
                                          self.catalog['NUMBER'][ok][ix],
                                          self.catalog['MAG_AUTO'][ok][ix])
                                          
                    for beam in beams:
                        self.compute_model(id=self.catalog['NUMBER'][ok][ix],
                                           x=self.catalog['x_flt'][ok][ix],
                                           y=self.catalog['y_flt'][ok][ix],
                                           beam=beam,  sh=[60, 60],
                                           verbose=True, in_place=True)
        
        outm = self.compute_model(id=self.catalog['NUMBER'][ok][ix],
                                  x=self.catalog['x_flt'][ok][ix],
                                  y=self.catalog['y_flt'][ok][ix], 
                                  beam='A', sh=[60, 60], verbose=True,
                                  in_place=False)
        
    def show_catalog_positions(self, ds9=None):
        if not ds9:
            return False
        
        n = len(self.catalog)
        for i in range(n):
            try:
                ds9.set_region('circle %f %f 2' %(self.catalog['x_flt'][i],
                                              self.catalog['y_flt'][i]))
            except:
                ds9.set('regions', 'circle %f %f 5\n' %(
                                              self.catalog['x_flt'][i],
                                              self.catalog['y_flt'][i]))
                                              
        if False:
            ok = catalog_table['MAG_AUTO'] < 23
    
    def shrink_segimage_to_flt(self):
        """
        Make a cutout of the larger reference image around the desired FLT
        image to make blotting faster for large reference images.
        """
        
        im = self.segimage_im
        ext = 0
        
        #ref_wcs = stwcs.wcsutil.HSTWCS(im, ext=ext)
        ref_wcs = pywcs.WCS(im[ext].header)
        
        naxis = self.im_header['NAXIS1'], self.im_header['NAXIS2']
        xflt = [-self.pad*2, naxis[0]+self.pad*2, 
                naxis[0]+self.pad*2, -self.pad*2]
                
        yflt = [-self.pad*2, -self.pad*2, 
                naxis[1]+self.pad*2, naxis[1]+self.pad*2]
                
        raflt, deflt = self.flt_wcs.all_pix2world(xflt, yflt, 0)
        xref, yref = np.cast[int](ref_wcs.all_world2pix(raflt, deflt, 0))
        ref_naxis = im[ext].header['NAXIS1'], im[ext].header['NAXIS2']
        
        xmi = np.maximum(0, xref.min())
        xma = np.minimum(ref_naxis[0], xref.max())
        slx = slice(xmi, xma)
        
        ymi = np.maximum(0, yref.min())
        yma = np.minimum(ref_naxis[1], yref.max())
        sly = slice(ymi, yma)
        
        if ((xref.min() < 0) | (yref.min() < 0) | 
            (xref.max() > ref_naxis[0]) | (yref.max() > ref_naxis[1])):
            
            print ('%s / Segmentation cutout: x=%s, y=%s [Out of range]'
                    %(self.flt_file, slx, sly))
            return False
        else:
            print '%s / Segmentation cutout: x=%s, y=%s' %(self.flt_file, 
                                                           slx, sly)
            
        slice_wcs = ref_wcs.slice((sly, slx))
        slice_header = im[ext].header.copy()
        hwcs = slice_wcs.to_header()
        #slice_header = hwcs
        for k in hwcs.keys():
           if not k.startswith('PC'):
               slice_header[k] = hwcs[k]
                
        slice_data = im[ext].data[sly, slx]*1
        
        hdul = pyfits.ImageHDU(data=slice_data, header=slice_header)
        self.segimage_im = pyfits.HDUList(hdul)
        
    def process_segimage(self):
        """
        Blot the segmentation image
        """
        self.seg = self.get_blotted_reference(self.segimage_im,
                                              segmentation=True)
        self.seg_ids = None
        
    def get_segment_coords(self, id=1):
        """
        Get centroid of a given ID segment
        
        ToDo: use photutils to do this
        """
        yp, xp = np.indices(self.seg.shape)
        id_mask = self.seg == id
        norm = np.sum(self.flam[id_mask])
        xi = np.sum((xp*self.flam)[id_mask])/norm
        yi = np.sum((yp*self.flam)[id_mask])/norm
        return xi+1, yi+1, id_mask
    
    def photutils_detection(self, detect_thresh=2, grow_seg=5, gauss_fwhm=2.,
                            compute_beams=None, verbose=True,
                            save_detection=False, wcs=None):
        """
        Use photutils to detect objects and make segmentation map
        
        ToDo: abstract to general script with data/thresholds 
              and just feed in image arrays
        """
        import scipy.ndimage as nd
        
        from photutils import detect_threshold, detect_sources
        from photutils import source_properties, properties_table
        from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
        from astropy.convolution import Gaussian2DKernel
        
        ### Detection threshold
        if '_flt' in self.refimage:
            threshold = (detect_thresh * self.refimage_im['ERR'].data)
        else:
            threshold = detect_threshold(self.clip, snr=detect_thresh)
        
        ### Gaussian kernel
        sigma = gauss_fwhm * gaussian_fwhm_to_sigma    # FWHM = 2.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        
        if verbose:
            print '%s: photutils.detect_sources (detect_thresh=%.1f, grow_seg=%d, gauss_fwhm=%.1f)' %(self.refimage, detect_thresh, grow_seg, gauss_fwhm)
            
        ### Detect sources
        segm = detect_sources(self.clip/self.photflam, threshold, 
                              npixels=5, filter_kernel=kernel)   
        grow = nd.maximum_filter(segm.array, grow_seg)
        self.seg = np.cast[np.float32](grow)
        
        ### Source properties catalog
        if verbose:
            print  '%s: photutils.source_properties' %(self.refimage)
        
        props = source_properties(self.clip/self.photflam, segm, wcs=wcs)
        self.catalog = properties_table(props)
        
        if verbose:
            print no_newline + ('%s: photutils.source_properties - %d objects'
                                 %(self.refimage, len(self.catalog)))
        
        #### Save outputs?
        if save_detection:
            seg_file = self.refimage.replace('.fits', '.detect_seg.fits')
            seg_cat = self.refimage.replace('.fits', '.detect.cat')
            if verbose:
                print '%s: save %s, %s' %(self.refimage, seg_file, seg_cat)
            
            pyfits.writeto(seg_file, data=self.seg, 
                           header=self.refimage_im[self.sci_ext].header,
                           clobber=True)
                
            if os.path.exists(seg_cat):
                os.remove(seg_cat)
            
            self.catalog.write(seg_cat, format='ascii.commented_header')
        
        #### Compute grism model for the detected segments 
        if compute_beams is not None:
            self.compute_full_model(compute_beams, mask=None, verbose=verbose)
                        
    def load_photutils_detection(self, seg_file=None, seg_cat=None, 
                                 catalog_format='ascii.commented_header'):
        """
        Load segmentation image and catalog, either from photutils 
        or SExtractor.  
        
        If SExtractor, use `catalog_format='ascii.sextractor'`.
        
        """
        if seg_file is None:
            seg_file = self.refimage.replace('.fits', '.detect_seg.fits')
        
        if not os.path.exists(seg_file):
            print 'Segmentation image %s not found' %(segfile)
            return False
        
        self.seg = np.cast[np.float32](pyfits.open(seg_file)[0].data)
        
        if seg_cat is None:
            seg_cat = self.refimage.replace('.fits', '.detect.cat')
        
        if not os.path.exists(seg_cat):
            print 'Segmentation catalog %s not found' %(seg_cat)
            return False
        
        self.catalog = Table.read(seg_cat, format=catalog_format)
                
    def compute_full_model(self, compute_beams=['A','B'], mask=None,
                           verbose=True, sh=[20,20]):
        """
        Compute full grism model for objects in the catalog masked with 
        the `mask` boolean array
        
        `sh` is the cutout size used to model each object
        
        """
        if mask is not None:
            cat_mask = self.catalog[mask]
        else:
            cat_mask = self.catalog
        
        if 'xcentroid' in self.catalog.colnames:
            xcol = 'xcentroid'
            ycol = 'ycentroid'
        else:
            xcol = 'x_flt'
            ycol = 'y_flt'
                
        for i in range(len(cat_mask)):
            line = cat_mask[i]
            for beam in compute_beams:
                if verbose:
                    print no_newline + ('%s: compute_model - id=%4d, beam=%s'
                                         %(self.refimage, line['id'], beam))
                
                self.compute_model(id=line['id'], x=line[xcol], y=line[ycol],
                                   sh=sh, beam=beam)
            
    def align_bright_objects(self, flux_limit=4.e-18, xspec=None, yspec=None,
                             ds9=None, max_shift=10,
                             cutout_dimensions=[14,14]):
        """
        Try aligning grism spectra based on the traces of bright objects
        """
        if self.seg_ids is None:
            self.seg_ids = np.cast[int](np.unique(self.seg)[1:])
            self.seg_flux = collections.OrderedDict()
            for id in self.seg_ids:
                self.seg_flux[id] = 0
            
            self.seg_mask = self.seg > 0
            for s, f in zip(self.seg[self.seg_mask],
                            self.flam[self.seg_mask]):
                #print s
                self.seg_flux[int(s)] += f
        
            self.seg_flux = np.array(self.seg_flux.values())
        
            ids = self.seg_ids[self.seg_flux > flux_limit]
            seg_dx = self.seg_ids*0.
            seg_dy = self.seg_ids*0.
        
        ccx0 = None
        for id in ids:
            xi, yi, id_mask = self.get_segment_coords(id)
            if ((xi > 1014-201) | (yi < cutout_dimensions[0]) | 
                (yi > 1014-20)  | (xi < cutout_dimensions[1])):
                continue
                
            beam = BeamCutout(x=xi, y=yi, cutout_dimensions=cutout_dimensions,
                              beam='A', conf=self.conf, GrismFLT=self) #direct_flam=self.flam, grism_flt=self.im)
            ix = self.seg_ids == id
            
            dx, dy, ccx, ccy = beam.align_spectrum(xspec=xspec, yspec=yspec)
            
            ### Try eazy template
            # sed = eazy.getEazySED(id-1, MAIN_OUTPUT_FILE='goodsn_3dhst.v4.1.dusty', OUTPUT_DIRECTORY='/Users/brammer/3DHST/Spectra/Release/v4.1/EazyRerun/OUTPUT/', CACHE_FILE='Same', scale_flambda=True, verbose=False, individual_templates=False)
            # dx, dy, ccx, ccy = beam.align_spectrum(xspec=sed[0], yspec=sed[1]/self.seg_flux[ix])
            plt.plot(ccx/ccx.max(), color='b', alpha=0.3)
            plt.plot(ccy/ccy.max(), color='g', alpha=0.3)
            
            if ccx0 is None:
                ccx0 = ccx*1.
                ccy0 = ccy*1.
                ncc = 0.
            else:
                ccx0 += ccx
                ccy0 += ccy
                ncc += 1
                
            if (np.abs(dx) > max_shift) | (np.abs(dy) > max_shift):
                print ('[bad] ID:%d (%7.1f,%7.1f), offset=%7.2f %7.2f' 
                       %(id, xi, yi, dx, dy))
                #break
                continue
            
            seg_dx[ix] = dx
            seg_dy[ix] = dy
            print ('ID:%d (%7.1f,%7.1f), offset=%7.2f %7.2f' 
                    %(id, xi, yi, dx, dy))
            
            if ds9:
                beam.init_dispersion(xoff=0, yoff=0)
                beam.compute_model(beam.thumb, xspec=xspec, yspec=yspec)
                m0 = beam.model*1    
                beam.init_dispersion(xoff=-dx, yoff=-dy)
                beam.compute_model(beam.thumb, xspec=xspec, yspec=yspec)
                m1 = beam.model*1
                ds9.view(beam.cutout_sci-m1)
        
        ok = seg_dx != 0
        xsh, x_rms = np.mean(seg_dx[ok]), np.std(seg_dx[ok])
        ysh, y_rms = np.mean(seg_dy[ok]), np.std(seg_dy[ok])
        print ('dx = %7.3f (%7.3f), dy = %7.3f (%7.3f)' 
                %(xsh, x_rms, ysh, y_rms))
                
        return xsh, ysh, x_rms, y_rms
    
    def update_wcs_with_shift(self, xsh=0.0, ysh=0.0, drizzle_ref=True):
        """
        Update WCS
        """
        import drizzlepac.updatehdr
                
        if hasattr(self, 'xsh'):
            self.xsh += xsh
            self.ysh += ysh
        else:
            self.xsh = xsh
            self.ysh = ysh
            
        h = self.im_header
        rd = self.all_pix2world([h['CRPIX1'], h['CRPIX1']+xsh], 
                                [h['CRPIX2'], h['CRPIX2']+ysh])
        h['CRVAL1'] = rd[0][1]
        h['CRVAL2'] = rd[1][1]
        self.im[tuple(self.sci_ext)].header = h
        #self.flt_wcs = stwcs.wcsutil.HSTWCS(self.im, ext=tuple(self.sci_ext))
        self.flt_wcs = pywcs.WCS(self.im[self.sci_ext].header)
        
        self.flt_wcs.naxis1 = self.im[sci_ext].header['NAXIS1']+2*self.pad
        self.flt_wcs.naxis2 = self.im[sci_ext].header['NAXIS2']+2*self.pad
        self.flt_wcs.wcs.crpix[0] += self.pad
        self.flt_wcs.wcs.crpix[1] += self.pad
        
        if self.flt_wcs.sip is not None:
            self.flt_wcs.sip.crpix[0] += self.pad
            self.flt_wcs.sip.crpix[1] += self.pad           
        
        if drizzle_ref:
            if (self.refimage) & (self.refimage_im is None):
                self.refimage_im = pyfits.open(self.refimage)
                 
            if self.refimage_im:
                self.flam = self.get_blotted_reference(self.refimage_im,
                                                       segmentation=False)
                self.flam *= photflam[self.filter]
                self.clip = np.cast[np.double](self.flam*self.dmask)
            
            if (self.segimage) & (self.segimage_im is None):
                self.segimage_im = pyfits.open(self.segimage)
            
            if self.segimage_im:
                self.process_segimage()
    
    def get_blotted_reference(self, refimage=None, segmentation=False, refext=0):
        """
        Use AstroDrizzle to blot reference / segmentation images to the FLT
        frame
        """
        #import stwcs
        import astropy.wcs
        from drizzlepac import astrodrizzle
        
        #ref = pyfits.open(refimage)
        if refimage[refext].data.dtype != np.float32:
            refimage[refext].data = np.cast[np.float32](refimage[refext].data)
        
        refdata = refimage[refext].data
        if 'ORIENTAT' in refimage[refext].header.keys():
            refimage[refext].header.remove('ORIENTAT')
            
        if segmentation:
            ## todo: allow getting these from cached outputs for 
            ##       cases of very large mosaics            
            seg_ones = np.cast[np.float32](refdata > 0)-1
        
        # refdata = np.ones(refdata.shape, dtype=np.float32)
        # seg_ones = refdata
        
        #ref_wcs = astropy.wcs.WCS(refimage[refext].header)
        #ref_wcs = stwcs.wcsutil.HSTWCS(refimage, ext=refext)
        ref_wcs = pywcs.WCS(refimage[refext].header)
        
        #flt_wcs = stwcs.wcsutil.HSTWCS(self.im, ext=('SCI',1))
        flt_wcs = self.flt_wcs
        
        for wcs in [ref_wcs, flt_wcs]:
            if (not hasattr(wcs.wcs, 'cd')) & hasattr(wcs.wcs, 'pc'):
                wcs.wcs.cd = wcs.wcs.pc
                
            if hasattr(wcs, 'idcscale'):
                if wcs.idcscale is None:
                    wcs.idcscale = np.sqrt(np.sum(wcs.wcs.cd[0,:]**2))*3600.
            else:
                wcs.idcscale = np.sqrt(np.sum(wcs.wcs.cd[0,:]**2))*3600.
            
            wcs.pscale = np.sqrt(wcs.wcs.cd[0,0]**2 +
                                 wcs.wcs.cd[1,0]**2)*3600.
            
            #print 'IDCSCALE: %.3f' %(wcs.idcscale)
            
        #print refimage.filename(), ref_wcs.idcscale, ref_wcs.wcs.cd, flt_wcs.idcscale, ref_wcs.orientat
            
        if segmentation:
            #print '\nSEGMENTATION\n\n',(seg_ones+1).dtype, refdata.dtype, ref_wcs, flt_wcs
            ### +1 here is a hack for some memory issues
            blotted_seg = astrodrizzle.ablot.do_blot(refdata+0, ref_wcs,
                                flt_wcs, 1, coeffs=True, interp='nearest',
                                sinscl=1.0, stepsize=1, wcsmap=None)
            
            blotted_ones = astrodrizzle.ablot.do_blot(seg_ones+1, ref_wcs,
                                flt_wcs, 1, coeffs=True, interp='nearest',
                                sinscl=1.0, stepsize=1, wcsmap=None)
            
            blotted_ones[blotted_ones == 0] = 1
            ratio = np.round(blotted_seg/blotted_ones)
            grow = nd.maximum_filter(ratio, size=3, mode='constant', cval=0)
            ratio[ratio == 0] = grow[ratio == 0]
            blotted = ratio
            
        else:
            #print '\nREFDATA\n\n', refdata.dtype, ref_wcs, flt_wcs
            blotted = astrodrizzle.ablot.do_blot(refdata, ref_wcs, flt_wcs, 1, coeffs=True, interp='poly5', sinscl=1.0, stepsize=10, wcsmap=None)
        
        return blotted
        
    def compute_model(self, id=0, x=588.28, y=40.54, sh=[10,10], 
                      xspec=None, yspec=None, beam='A', verbose=False,
                      in_place=True, outdata=None):
        """
        Compute a model spectrum, so simple!
        
        Compute a model in a box of size `sh` around pixels `x` and `y` 
        in the direct image.
        
        Only consider pixels in the segmentation image with value = `id`.
        
        If xspec / yspec = None, the default assumes flat flambda spectra
        
        If `in place`, update the model in `self.model` and `self.modelf`, 
        otherwise put the output in a clean array.  This latter might be slow
        if the overhead of computing a large image array is high.
        """
        xc, yc = int(x), int(y)
        xcenter = x - xc
        
        ### Get dispersion parameters at the reference position
        dy, lam = self.conf.get_beam_trace(x=x-self.pad, y=y-self.pad,
                                           dx=self.conf.dxlam[beam]+xcenter,
                                           beam=beam)
        
        ### Integer trace
        # 20 for handling int of small negative numbers    
        dyc = np.cast[int](dy+20)-20+1 
        
        ### Account for pixel centering of the trace
        yfrac = dy-np.floor(dy)
        
        ### Interpolate the sensitivity curve on the wavelength grid. 
        ysens = lam*0
        so = np.argsort(lam)
        ysens[so] = interp.interp_conserve_c(lam[so],
                                 self.conf.sens[beam]['WAVELENGTH'], 
                                 self.conf.sens[beam]['SENSITIVITY'])
        
        ### Needs term of delta wavelength per pixel for flux densities
        # ! here assumes linear dispersion
        ysens *= np.abs(lam[1]-lam[0])*1.e-17
        
        if xspec is not None:
            yspec_int = ysens*0.
            yspec_int[so] = interp.interp_conserve_c(lam[so], xspec, yspec)
            ysens *= yspec_int
                    
        x0 = np.array([yc, xc])
        slx = self.conf.dxlam[beam]+xc
        ok = (slx < self.sh_pad[1]) & (slx > 0)
        
        if in_place:
            #self.modelf *= 0
            outdata = self.modelf
        else:
            if outdata is None:
                outdata = self.modelf*0
        
        ### This is an array of indices for the spectral trace
        try:
            idxl = self.idx[dyc[ok]+yc,slx[ok]]
        except:
            if verbose:
                print ('Dispersed trace falls off the image: x=%.2f y=%.2f'
                        %(x, y))
            
            return False
            
        ### Loop over pixels in the direct FLT and add them into a final 2D
        ### spectrum (in the full (flattened) FLT frame)
        ## adds into the output array, initializing full array to zero 
        ## could be very slow
        status = disperse.disperse_grism_object(self.clip, self.seg, id, idxl,
                                                yfrac[ok], ysens[ok], outdata,
                                                x0, np.array(self.clip.shape),
                                                np.array(sh),
                                                np.array(self.sh_pad))
                
        if not in_place:
            return outdata
        else:
            return True
    
    def fit_background(self, degree=3, sn_limit=0.1, pfit=None, apply=True,
                       verbose=True, ds9=None):
        """
        Fit a 2D polynomial background model to the grism exposure, only
        condidering pixels where

          self.model < sn_limit * self.im_data['ERR']
          
        """
        from astropy.modeling import models, fitting
        
        yp, xp = np.indices(self.sh_pad)
        xp  = (xp - self.sh_pad[1]/2.)/(self.sh_flt[1]/2)
        yp  = (yp - self.sh_pad[0]/2.)/(self.sh_flt[0]/2)
        
        if pfit is None:
            mask = ((self.im_data['DQ'] == 0) &
                    (self.model/self.im_data['ERR'] < sn_limit) &
                    (self.im_data['SCI'] != 0) & 
                    (self.im_data['SCI'] > -4*self.im_data['ERR']) &
                    (self.im_data['SCI'] < 6*self.im_data['ERR']) & 
                    (self.im_data['ERR'] < 1000))
                      
            poly = models.Polynomial2D(degree=degree)
            fit = fitting.LinearLSQFitter()
            pfit = fit(poly, xp[mask], yp[mask], self.im_data['SCI'][mask])
            pout = pfit(xp, yp)
            
            if ds9:
                ds9.view((self.im_data['SCI']-pout)*mask)
        else:
            pout = pfit(xp, yp)
            
        if apply:
            if self.pad > 0:
                slx = slice(self.pad, -self.pad)
                sly = slice(self.pad, -self.pad)

            else:
                slx = slice(0, self.sh_flt[1])
                sly = slice(0, self.sh_flt[0])
                
            self.im_data['SCI'][sly, slx] -= pout[sly, slx]
            self.im_data_sci_background = True
        
        if verbose:
            print ('fit_background, %s: p0_0=%7.4f' 
                   %(self.flt_file, pfit.parameters[0]))
                
        self.fit_background_result = pfit #(pfit, xp, yp)
            
class BeamCutout(object):
    """
    Cutout 2D spectrum from the full frame
    """
    def __init__(self, x=588.28, y=40.54, id=0, conf=None,
                 cutout_dimensions=[10,10], beam='A', GrismFLT=None):   
                
        self.beam = beam
        self.x, self.y = x, y
        self.xc, self.yc = int(x), int(y)
        self.id = id
        
        self.xcenter = self.x-self.xc
        
        if GrismFLT is not None:
            self.pad = GrismFLT.pad
        else:
            self.pad = 0
            
        self.dx = conf.dxlam[beam]
        
        self.cutout_dimensions = cutout_dimensions
        self.shd = np.array((2*cutout_dimensions[0], 2*cutout_dimensions[1]))
        self.lld = [self.yc-cutout_dimensions[0],
                    self.xc-cutout_dimensions[1]]
        self.shg = np.array((2*cutout_dimensions[0], 
                             2*cutout_dimensions[1] + conf.nx[beam]))
        self.llg = [self.yc-cutout_dimensions[0],
                    self.xc-cutout_dimensions[1]+self.dx[0]]
        
        self.x_index = np.arange(self.shg[1])
        self.y_index = np.arange(self.shg[0])

        self.modelf = np.zeros(self.shg, dtype=np.double).flatten()
        self.model = self.modelf.reshape(self.shg)

        self.conf = conf
        self.beam = beam
        self.init_dispersion()
        
        self.thumb = None
        self.cutout_sci = None    
        self.shc = None
        self.cutout_seg = np.zeros(self.shg, dtype=np.float32)
        
        self.wave = ((np.arange(self.shg[1]) + 1 - self.cutout_dimensions[1])
                      *(self.lam[1]-self.lam[0]) + self.lam[0])
        self.contam = 0
        
        if GrismFLT is not None:
            self.thumb = self.get_flam_thumb(GrismFLT.flam)*1
            self.cutout_sci = self.get_cutout(GrismFLT.im_data['SCI'])*1
            self.cutout_dq = self.get_cutout(GrismFLT.im_data['DQ'])*1
            self.cutout_err = self.get_cutout(GrismFLT.im_data['ERR'])*1
            self.shc = self.cutout_sci.shape
            
            self.cutout_seg = self.get_flam_thumb(GrismFLT.seg,
                                                  dtype=np.float32)
            self.total_flux = np.sum(self.thumb[self.cutout_seg == self.id])
            self.clean_thumb()
            
            self.grism = GrismFLT.grism
            self.dispersion_PA = GrismFLT.dispersion_PA
            self.filter = GrismFLT.filter
            self.photflam = GrismFLT.photflam
            self.pivot = GrismFLT.pivot
            
            self.compute_ivar(mask=True)
            
        # if direct_flam is not None:
        #     self.thumb = self.get_flam_thumb(direct_flam)
        # 
        # if grism_flt is not None:
        #     self.cutout_sci = self.get_cutout(grism_flt['SCI'].data)*1
        #     self.cutout_dq = self.get_cutout(grism_flt['DQ'].data)*1
        #     self.cutout_err = self.get_cutout(grism_flt['ERR'].data)*1
        #     self.shc = self.cutout_sci.shape
            #beam.cutout[beam.cutout_dq > 0] = 0
        
        #if segm_flt is not None:
         #   self.cutout_seg = self.get_cutout(segm_flt)*1
    
    def clean_thumb(self):
        """
        zero out negative pixels in self.thumb
        """
        self.thumb[self.thumb < 0] = 0
        self.total_flux = np.sum(self.thumb[self.cutout_seg == self.id])
    
    def compute_ivar(self, mask=True):
        self.ivar = np.cast[np.float32](1/(self.cutout_err**2))
        self.ivar[(self.cutout_err == 0)] = 0.
        if mask:
            self.ivar[(self.cutout_dq > 0)] = 0
            
    def init_dispersion(self, xoff=0, yoff=0):
        """
        Allow for providing offsets to the dispersion function
        """
        
        dx = self.conf.dxlam[self.beam]+self.xcenter-xoff
        self.dy, self.lam = self.conf.get_beam_trace(x=self.x-xoff-self.pad,
                                                     y=self.y+yoff-self.pad, 
                                                     dx=dx, beam=self.beam)
        
        self.dy += yoff
        
        # 20 for handling int of small negative numbers
        self.dyc = np.cast[int](self.dy+20)+-20+1
        self.yfrac = self.dy-np.floor(self.dy)
        
        dl = np.abs(self.lam[1]-self.lam[0])
        self.ysens = interp.interp_conserve_c(self.lam,
                                     self.conf.sens[self.beam]['WAVELENGTH'],
                                     self.conf.sens[self.beam]['SENSITIVITY'])
        self.ysens *= dl/1.e17
        
        self.idxl = np.arange(np.product(self.shg)).reshape(self.shg)[self.dyc+self.cutout_dimensions[0], self.dx-self.dx[0]+self.cutout_dimensions[1]]
        
    def get_flam_thumb(self, flam_full, xoff=0, yoff=0, dtype=np.double):
        dim = self.cutout_dimensions
        return np.cast[dtype](flam_full[self.yc+yoff-dim[0]:self.yc+yoff+dim[0], self.xc+xoff-dim[1]:self.xc+xoff+dim[1]])
    
    def twod_axis_labels(self, wscale=1.e4, limits=None, mpl_axis=None):
        """
        Set x axis *tick labels* on a 2D spectrum to wavelength units
        
        Defaults to a wavelength scale of microns with wscale=1.e4
        
        Will automatically use the whole wavelength range defined by the spectrum.  To change,
        specify `limits = [x0, x1, dx]` to interpolate self.wave between x0*wscale and x1*wscale.
        """
        xarr = np.arange(len(self.wave))
        if limits:
            xlam = np.arange(limits[0], limits[1], limits[2])
            xpix = np.interp(xlam, self.wave/wscale, xarr)
        else:
            xlam = np.unique(np.cast[int](self.wave / 1.e4*10)/10.)
            xpix = np.interp(xlam, self.wave/wscale, xarr)
        
        if mpl_axis is None:
            pass
            #return xpix, xlam
        else:
            mpl_axis.set_xticks(xpix)
            mpl_axis.set_xticklabels(xlam)
    
    def twod_xlim(self, x0, x1=None, wscale=1.e4, mpl_axis=None):
        """
        Set x axis *limits* on a 2D spectrum to wavelength units
        
        defaults to a scale of microns with wscale=1.e4
        
        """
        if isinstance(x0, list):
            x0, x1 = x0[0], x0[1]
        
        xarr = np.arange(len(self.wave))
        xpix = np.interp([x0,x1], self.wave/wscale, xarr)
        
        if mpl_axis:
            mpl_axis.set_xlim(xpix)
        else:
            return xpix
            
    def compute_model(self, flam_thumb, id=0, yspec=None, xspec=None, in_place=True):
        
        x0 = np.array([self.cutout_dimensions[0], self.cutout_dimensions[0]])
        sh_thumb = np.array((self.shd[0]/2, self.shd[1]/2))  
        if in_place:
            self.modelf *= 0
            out = self.modelf
        else:
            out = self.modelf*0
            
        ynorm=1
        if xspec is not self.lam:
            if yspec is not None:
                ynorm = interp.interp_conserve_c(self.lam, xspec, yspec)
        else:
            ynorm = yspec
            
        status = disperse.disperse_grism_object(flam_thumb, self.cutout_seg, id, self.idxl, self.yfrac, self.ysens*ynorm, out, x0, self.shd, sh_thumb, self.shg)
        
        if not in_place:
            return out
            
    def get_slices(self):
        sly = slice(self.llg[0], self.llg[0]+self.shg[0])
        slx = slice(self.llg[1], self.llg[1]+self.shg[1])
        return sly, slx
        
    def get_cutout(self, data):
        sly, slx = self.get_slices()
        return data[sly, slx]
        
    def make_wcs_header(self, data=None):
        #import stwcs
        h = pyfits.Header()
        h['CRPIX1'] = self.cutout_dimensions[1]#+0.5
        h['CRPIX2'] = self.cutout_dimensions[0]#+0.5
        h['CRVAL1'] = self.lam[0]        
        h['CD1_1'] = self.lam[1]-self.lam[0]
        h['CD1_2'] = 0.
        
        h['CRVAL2'] = -self.dy[0]
        h['CD2_2'] = 1.
        h['CD2_1'] = -(self.dy[1]-self.dy[0])
        
        h['CTYPE1'] = 'WAVE'
        h['CTYPE2'] = 'LINEAR'
        
        if data is None:
            np.zeros(self.shg)
        
        data = hdul = pyfits.HDUList([pyfits.ImageHDU(data=data, header=h)])
        #wcs = stwcs.wcsutil.HSTWCS(hdul, ext=0)
        wcs = pywcs.WCS(hdul[0].header)
        
        wcs.pscale = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[1,0]**2)*3600.
        
        return hdul[0], wcs
        
    def align_spectrum(self, xspec=None, yspec=None):
        """
        Try to compute alignment of the reference image using cross correlation
        """
        from astropy.modeling import models, fitting
        
        clean_cutout = self.cutout_sci*1.
        clean_cutout[self.cutout_dq > 0] = 0
        #max = np.percentile(clean_cutout[clean_cutout != 0], clip_percentile)
        #clean_cutout[(clean_cutout > max) | (clean_cutout < -3*self.cutout_err)] = 0
        clean_cutout[(clean_cutout < -3*self.cutout_err) | ~np.isfinite(self.cutout_err)] = 0.
        
        self.compute_model(self.thumb, xspec=xspec, yspec=yspec)
        
        ### Cross correlation
        cc = nd.correlate(self.model/self.model.sum(), clean_cutout/clean_cutout.sum())
        
        sh = cc.shape
        shx = sh[1]/2.; shy = sh[0]/2.

        yp, xp = np.indices(cc.shape)
        shx = sh[1]/2; shy = sh[0]/2
        xp = (xp-shx); yp = (yp-shy)

        cc[:,:shx-shy] = 0
        cc[:,shx+shy:] = 0
        ccy = cc.sum(axis=1)
        ccx = cc.sum(axis=0)
        
        #fit = fitting.LevMarLSQFitter()
        #mod = models.Polynomial1D(degree=6) #(1, 0, 1)
        fit = fitting.LinearLSQFitter()

        ix = np.argmax(ccx)
        p2 = models.Polynomial1D(degree=2)
        px = fit(p2, xp[0, ix-1:ix+2], ccx[ix-1:ix+2]/ccx.max())
        dx = -px.parameters[1]/(2*px.parameters[2])

        iy = np.argmax(ccy)
        py = fit(p2, yp[iy-1:iy+2, 0], ccy[iy-1:iy+2]/ccy.max())
        dy = -py.parameters[1]/(2*py.parameters[2])
        
        return dx, dy, ccx, ccy
        
    def optimal_extract(self, data, bin=0):        
        import scipy.ndimage as nd
                
        if not hasattr(self, 'opt_profile'):
            m = self.compute_model(self.thumb, id=self.id, in_place=False).reshape(self.shg)
            m[m < 0] = 0
            self.opt_profile = m/m.sum(axis=0)
            
        num = self.opt_profile*data*self.ivar.reshape(self.shg)
        den = self.opt_profile**2*self.ivar.reshape(self.shg)
        opt = num.sum(axis=0)/den.sum(axis=0)
        opt_var = 1./den.sum(axis=0)
        
        wave = self.wave
        
        if bin > 0:
            kern = np.ones(bin, dtype=float)/bin
            opt = nd.convolve(opt, kern)[bin/2::bin]
            opt_var = nd.convolve(opt_var, kern**2)[bin/2::bin]
            wave = self.wave[bin/2::bin]
            
        opt_rms = np.sqrt(opt_var)
        opt_rms[opt_var == 0] = 0
        
        return wave, opt, opt_rms
    
    def simple_line_fit(self, fwhm=5., grid=[1.12e4, 1.65e4, 1, 20]):
        """
        Demo: fit continuum and an emission line over a wavelength grid
        """
        import sklearn.linear_model
        clf = sklearn.linear_model.LinearRegression()
                
        ### Continuum
        self.compute_model(self.thumb, id=self.id)
        ### OK data
        ok = (self.ivar.flatten() != 0) & (self.modelf > 0.03*self.modelf.max())
        
        scif = (self.cutout_sci - self.contam).flatten()
        ivarf = self.ivar.flatten()
        
        ### Model: (ax + b)*continuum + line
        yp, xp = np.indices(self.shg)
        xpf = (xp.flatten() - self.shg[1]/2.)/(self.shg[1]/2)
        
        xpf = ((self.wave[:,None]*np.ones(self.shg[0]) - self.pivot)/1000.).T.flatten()
        A = np.vstack([xpf*self.modelf*1, self.modelf*1, self.modelf*1]).T
        
        ### Fit lines
        wave_array = np.arange(grid[0], grid[1], grid[2])
        line_centers = wave_array[grid[3]/2::grid[3]]
        
        rms = fwhm/2.35
        gaussian_lines = 1/np.sqrt(2*np.pi*rms**2)*np.exp(-(line_centers[:,None]-wave_array)**2/2/rms**2)
        
        N = len(line_centers)
        coeffs = np.zeros((N, 3))
        chi2 = np.zeros(N)
        chi2min = 1e30
        
        for i in range(N):
            self.compute_model(self.thumb, id=self.id, xspec=wave_array, yspec=gaussian_lines[i,:])
            A[:,2] = self.modelf
            status = clf.fit(A[ok,:], scif[ok])
            coeffs[i,:] = clf.coef_
            
            model = np.dot(A, clf.coef_)
            chi2[i] = np.sum(((scif-model)**2*ivarf)[ok])
            
            if chi2[i] < chi2min:
                print no_newline + '%d, wave=%.1f, chi2=%.1f, line_flux=%.1f' %(i, line_centers[i], chi2[i], coeffs[i,2]*self.total_flux/1.e-17) 
                chi2min = chi2[i]
                
        ### Best    
        ix = np.argmin(chi2)
        self.compute_model(self.thumb, id=self.id, xspec=wave_array, yspec=gaussian_lines[ix,:])
        A[:,2] = self.modelf
        model = np.dot(A, coeffs[ix,:])
        
        return line_centers, coeffs, chi2, ok, model.reshape(self.shg), line_centers[ix], coeffs[ix,2]*self.total_flux/1.e-17

        