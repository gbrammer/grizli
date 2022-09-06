"""
Generate PSF at an arbitrary position in a drizzled image using the WFC3/IR
effective PSFs.
"""
import os
from collections import OrderedDict

import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

#from research.dash.epsf import DrizzlePSF

DEMO_LIST = ['ib6o23rtq_flt.fits', 'ib6o23rwq_flt.fits', 'ib6o23rzq_flt.fits', 'ib6o23s2q_flt.fits']
DEMO_IMAGE = 'wfc3-ersii-g01-b6o-23-119.0-f140w_drz_sci.fits'

try:
    from .. import utils, model
except:
    from grizli import utils, model


class DrizzlePSF(object):
    def __init__(self, flt_files=DEMO_LIST, info=None, driz_image=DEMO_IMAGE, driz_hdu=None, beams=None, full_flt_weight=True):
        """
        Object for making drizzled PSFs

        Parameters
        ----------
        flt_files : list
            List of FLT files that were used to create the drizzled image.

        driz_image : str
            Filename of the drizzled image.

        """
        if info is None:
            if beams is not None:
                info = self._get_wcs_from_beams(beams)
            else:
                if flt_files is None:
                    info = self._get_wcs_from_hdrtab(driz_image)
                    if info is None:
                        info = self._get_wcs_from_csv(driz_image)
                else:
                    info = self._get_flt_wcs(flt_files, 
                                             full_flt_weight=full_flt_weight)

        self.flt_keys, self.wcs, self.footprint = info
        self.flt_files = list(np.unique([key[0] for key in self.flt_keys]))

        self.ePSF = utils.EffectivePSF()

        if driz_hdu is None:
            self.driz_image = driz_image
            self.driz_header = pyfits.getheader(driz_image)
            self.driz_wcs = pywcs.WCS(self.driz_header)
            self.driz_pscale = utils.get_wcs_pscale(self.driz_wcs)
        else:
            self.driz_image = driz_image
            self.driz_header = driz_hdu.header
            self.driz_wcs = pywcs.WCS(self.driz_header)
            self.driz_pscale = utils.get_wcs_pscale(self.driz_wcs)

        self.driz_wcs.pscale = self.driz_pscale


    @staticmethod
    def _get_flt_wcs(flt_files, full_flt_weight=True):
        """
        Get WCS info from FLT FITS files
        
        Parameters
        ----------
        flt_files : list
            List of filenames
        
        full_flt_weight : bool
            If True, set `expweight` from 'ERR' extension of the FITS files, 
            else weight by EXPTIME keyword
        """
        from shapely.geometry import Polygon, Point

        from grizli import utils
        wcs = OrderedDict()
        footprint = OrderedDict()

        flt_keys = []
        for file in flt_files:
            flt_j = pyfits.open(file)
            for ext in range(1, 5):
                if ('SCI', ext) in flt_j:
                    key = file, ext
                    wcs[key] = pywcs.WCS(flt_j['SCI', ext], relax=True,
                                          fobj=flt_j)

                    wcs[key].pscale = utils.get_wcs_pscale(wcs[key])

                    if (('ERR',ext) in flt_j) & full_flt_weight:
                        wht = (1./flt_j['ERR',ext].data**2).astype(np.float32)
                        wht[~np.isfinite(wht)] = 0
                        wcs[key].expweight = np.pad(wht, 32)
                    else:
                        wcs[key].expweight = flt_j[0].header['EXPTIME']
                        
                    footprint[key] = Polygon(wcs[key].calc_footprint())
                    flt_keys.append(key)

        return flt_keys, wcs, footprint


    @staticmethod
    def _get_wcs_from_beams(beams):
        """
        TBD
        """
        from shapely.geometry import Polygon, Point

        from grizli import utils
        wcs = OrderedDict()
        footprint = OrderedDict()

        flt_keys = []
        for beam in beams:
            #flt_j = pyfits.open(file)
            file = beam.direct.parent_file
            ext = beam.direct.sci_extn

            key = file, ext
            wcs[key] = beam.direct.wcs.copy()

            wcs[key].pscale = utils.get_wcs_pscale(wcs[key])

            wcs[key].expweight = beam.grism.exptime

            footprint[key] = Polygon(wcs[key].calc_footprint())
            flt_keys.append(key)

        return flt_keys, wcs, footprint


    @staticmethod
    def _get_wcs_from_csv(drz_file):
        """
        Read the attached CSV file that contains exposure WCS info
        """
        import glob

        csv_file = drz_file.split('_drz_sci')[0].split('_drc_sci')[0]
        csv_file += '_wcs.csv'
        
        if  not os.path.exists(csv_file):
            print(f'No WCS CSV file {csv_file} found for {drz_file}')
            return None
        else:
            print(f'Get exposure WCS from {csv_file}')
            
        csv = utils.read_catalog(csv_file)
        flt_keys = []
        wcs = {}
        footprint = {}
        
        for row in csv:
            key = row['file'], row['ext']
            _h = pyfits.Header()
            for c in row.colnames:
                _h[c] = row[c]
            
            flt_keys.append(key)
            wcs[key] = pywcs.WCS(_h, relax=True)
            
            wcs[key].pscale = utils.get_wcs_pscale(wcs[key])
            
            if 'EXPTIME' in _h:
                wcs[key].expweight = _h['EXPTIME']
            else:
                wcs[key].expweight = 1
            
            sr = utils.SRegion(wcs[key])
            footprint[key] = sr.shapely[0]
            
        return flt_keys, wcs, footprint


    @staticmethod
    def _get_wcs_from_hdrtab(drz_file):
        """
        Read tabulated exposure WCS info from the HDRTAB
        extension of an AstroDrizzle output file
        """
        from shapely.geometry import Polygon, Point
        
        drz = pyfits.open(drz_file)
        if 'HDRTAB' not in drz:
            print('No HDRTAB extension found in {0}'.format(drz_file))
            return None

        hdr = utils.GTable(drz['HDRTAB'].data)
        wcs = OrderedDict()
        footprint = OrderedDict()

        flt_keys = []
        N = len(hdr)

        if 'CCDCHIP' in hdr.colnames:
            ext_key = 'CCDCHIP'
        else:
            ext_key = 'EXTNAME'

        for i in range(N):
            h = pyfits.Header()
            for c in hdr.colnames:
                try:
                    h[c] = hdr[c][i]
                except:
                    h[c] = 1

            key = (h['ROOTNAME'], h[ext_key])
            flt_keys.append(key)
            wcs[key] = pywcs.WCS(h, relax=True)
            wcs[key].pscale = utils.get_wcs_pscale(wcs[key])

            if 'EXPTIME' in h:
                wcs[key].expweight = h['EXPTIME']
            else:
                wcs[key].expweight = 1

            footprint[key] = Polygon(wcs[key].calc_footprint())

        return flt_keys, wcs, footprint

    def get_driz_cutout(self, ra=53.06967306, dec=-27.72333015, size=15, get_cutout=False, N=None, odd=True):
        """
        TBD
        """
        xy = self.driz_wcs.all_world2pix(np.array([[ra, dec]]), 0)[0]
        xyp = np.cast[int](np.round(xy))
        if N is None:
            N = int(np.round(size*self.wcs[self.flt_keys[0]].pscale/self.driz_pscale))

        slx = slice(xyp[0]-N, xyp[0]+N+odd)
        sly = slice(xyp[1]-N, xyp[1]+N+odd)

        wcs_slice = model.ImageData.get_slice_wcs(self.driz_wcs, slx, sly)

        wcs_slice.pscale = utils.get_wcs_pscale(wcs_slice)

        # outsci = np.zeros((2*N,2*N), dtype=np.float32)
        # outwht = np.zeros((2*N,2*N), dtype=np.float32)
        # outctx = np.zeros((2*N,2*N), dtype=np.int32)
        if get_cutout > 1:
            os.system("getfits -o sub.fits {0} {1} {2} {3} {3}".format(self.driz_image, xyp[0], xyp[1], 2*N))
            hdu = pyfits.open('sub.fits')
            return slx, sly, hdu
        elif get_cutout == 1:
            im = pyfits.open(self.driz_image)
            data = im[0].data[sly, slx]*1
            header = utils.to_header(wcs_slice, relax=True)
            hdu = pyfits.PrimaryHDU(data=data, header=header)
            return slx, sly, pyfits.HDUList([hdu])

        return slx, sly, wcs_slice

    @staticmethod
    def _get_empty_driz(wcs):
        if hasattr(wcs, 'pixel_shape'):
            sh = wcs.pixel_shape[::-1]
        else:
            if (not hasattr(wcs, '_naxis1')) & hasattr(wcs, '_naxis'):
                wcs._naxis1, wcs._naxis2 = wcs._naxis

            sh = (wcs._naxis2, wcs._naxis1)

        outsci = np.zeros(sh, dtype=np.float32)
        outwht = np.zeros(sh, dtype=np.float32)
        outctx = np.zeros(sh, dtype=np.int32)
        return outsci, outwht, outctx

    def go(self, ra=53.06967306, dec=-27.72333015):
        """
        Testing
        """
        import scipy.optimize
        wcs, footprint = None, None
        
        self = DrizzlePSF(info=(wcs, footprint), driz_image='cosmos-full-v1.2.8-f160w_drz_sci.fits')

        slx, sly, wcs_slice = self.get_driz_cutout(ra=ra, dec=dec)
        xx, yy, drz_cutout = self.get_driz_cutout(ra=ra, dec=dec, get_cutout=True)

        psf = self.get_psf(ra=ra, dec=dec, filter='F160W', wcs_slice=wcs_slice)

        init = (0, 0, drz_cutout[0].data.sum())
        chi2 = self.objfun(init, self, ra, dec, wcs_slice, filter, drz_cutout)

        out = scipy.optimize.minimize(self.objfun, init, args=(self, ra, dec, wcs_slice, filter, drz_cutout), method='Powell', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1.e-3, callback=None, options=None)

        psf = self.get_psf(ra=ra+out.x[0]/3600., dec=dec+out.x[1]/3600., filter=filter, wcs_slice=wcs_slice, verbose=False)

    @staticmethod
    def objfun_center(params, self, ra, dec, wcs_slice, filter, _Abg, sci, wht, ret):
        xoff, yoff = params[:2]
        
        psf_offset = self.get_psf(ra=ra+xoff/3600./10., 
                                  dec=dec+yoff/3600./10.,
                                  filter=filter, wcs_slice=wcs_slice, verbose=False)

        _A = np.vstack([_Abg, psf_offset[1].data.flatten()])
        _Ax = _A*np.sqrt(wht.flatten())
        _y = (sci*np.sqrt(wht)).flatten()

        _res = np.linalg.lstsq(_Ax.T, _y, rcond=utils.LSTSQ_RCOND)

        model = _A.T.dot(_res[0]).reshape(sci.shape)
        chi2 = ((sci-model)**2*wht).sum()
        print(params, chi2)
        if ret == 1:
            return params, _res, model, chi2
        else:
            return chi2
    
    
    @staticmethod
    def xobjfun_center(params, self, ra, dec, _Abg, sci, wht, ret):
        """
        Objective function for fitting the PSF model to a drizzled cutout
        """
        xoff, yoff = params[:2]
        
        filt = self.driz_header['FILTER']
        
        psf_offset = self.get_psf(ra=ra+xoff/3600./10., 
                                  dec=dec+yoff/3600./10.,
                                  filter=filt.upper(),
                                  pixfrac=self.driz_header['PIXFRAC'],
                                  kernel=self.driz_header['KERNEL'],
                                  wcs_slice=self.driz_wcs, 
                                 get_extended=filt.lower()[:2] in ['f1','f0'],
                                   verbose=False)

        if _Abg is not None:
            _A = np.vstack([_Abg, psf_offset[1].data.flatten()])
        else:
            _A = np.atleast_2d(psf_offset[1].data.flatten())
            
        _Ax = _A*np.sqrt(wht.flatten())
        _y = (sci*np.sqrt(wht)).flatten()

        _res = np.linalg.lstsq(_Ax.T, _y, rcond=utils.LSTSQ_RCOND)

        model = _A.T.dot(_res[0]).reshape(sci.shape)
        #chi2 = ((sci-model)**2*wht).sum()
        chi2 = _res[1][0]
        print(params, chi2)
        if ret == 1:
            return params, _res, model, chi2
        else:
            return chi2


    @staticmethod
    def objfun(params, self, ra, dec, wcs_slice, filter, drz_cutout):
        xoff, yoff = params[:2]
        psf = self.get_psf(ra=ra+xoff/3600., dec=dec+yoff/3600., filter=filter, wcs_slice=wcs_slice, verbose=False)
        chi2 = ((psf[1].data*params[2] - drz_cutout[0].data)**2).sum()
        print(params, chi2)
        return chi2

    def get_psf(self, ra=53.06967306, dec=-27.72333015, filter='F140W', pixfrac=0.1, kernel='point', verbose=True, wcs_slice=None, get_extended=True, get_weight=False, ds9=None, npix=13, renormalize=True):
        from drizzlepac import adrizzle
        from shapely.geometry import Polygon, Point

        pix = np.arange(-npix, npix+1)

        #wcs_slice = self.get_driz_cutout(ra=ra, dec=dec)
        if wcs_slice is None:
            wcs_slice = self.driz_wcs.copy()

        outsci, outwht, outctx = self._get_empty_driz(wcs_slice)

        count = 1
        for key in self.flt_keys:
            if self.footprint[key].contains(Point(ra, dec)):
                file, ext = key
                if verbose:
                    print('{0}[SCI,{1}]'.format(file, ext))

                xy = self.wcs[key].all_world2pix(np.array([[ra, dec]]), 0)[0]
                xyp = np.cast[int](xy) #np.round(xy))  # +1
                dx = xy[0]-int(xy[0])#-0.5
                dy = xy[1]-int(xy[1])#-0.5

                if ext == 2:
                    # UVIS
                    # print('UVIS1!')
                    chip_offset = 2051
                else:
                    chip_offset = 0

                psf_xy = self.ePSF.get_at_position(xy[0], xy[1]+chip_offset,
                                                   filter=filter)
                yp, xp = np.meshgrid(pix-dy, pix-dx, sparse=False, 
                                     indexing='ij')
                if get_extended:
                    if filter in self.ePSF.extended_epsf:
                        extended_data = self.ePSF.extended_epsf[filter]
                    else:
                        extended_data = None
                else:
                    extended_data = None

                psf = self.ePSF.eval_ePSF(psf_xy, xp, yp, 
                                          extended_data=extended_data)

                # if get_weight:
                #     fltim = pyfits.open(file)
                #     flt_weight = fltim[0].header['EXPTIME']
                # else:
                #     flt_weight = 1
                flt_weight = self.wcs[key].expweight

                N = npix
                slx = slice(xyp[0]-N, xyp[0]+N+1)
                sly = slice(xyp[1]-N, xyp[1]+N+1)
                
                if hasattr(flt_weight, 'ndim'):
                    if flt_weight.ndim == 2:
                        wslx = slice(xyp[0]-N+32, xyp[0]+N+1+32)
                        wsly = slice(xyp[1]-N+32, xyp[1]+N+1+32)
                        flt_weight = self.wcs[key].expweight[wsly, wslx]
                    
                psf_wcs = model.ImageData.get_slice_wcs(self.wcs[key], 
                                                        slx, sly)

                psf_wcs.pscale = utils.get_wcs_pscale(self.wcs[key])

                try:
                    adrizzle.do_driz(psf, psf_wcs, psf*0+flt_weight,
                                 wcs_slice,
                                 outsci, outwht, outctx, 1., 'cps', 1,
                                 wcslin_pscale=1., uniqid=1,
                                 pixfrac=pixfrac, kernel=kernel, fillval=0,
                                 stepsize=10, wcsmap=None)
                except:
                    psf_wcs._naxis1, psf_wcs._naxis2 = psf_wcs._naxis

                    adrizzle.do_driz(psf, psf_wcs, psf*0+flt_weight,
                                 wcs_slice,
                                 outsci, outwht, outctx, 1., 'cps', 1,
                                 wcslin_pscale=1., uniqid=1,
                                 pixfrac=pixfrac, kernel=kernel, fillval=0,
                                 stepsize=10, wcsmap=None)

                if ds9 is not None:
                    count += 1
                    hdu = pyfits.HDUList([pyfits.PrimaryHDU(), pyfits.ImageHDU(data=psf*100, header=utils.to_header(psf_wcs))])
                    ds9.set('frame {0}'.format(count+1))
                    ds9.set_pyfits(hdu)

        #ss = 1000000/2
        if renormalize:
            ss = 1./outsci.sum()*psf.sum()
        else:
            ss = 1.
            
        hdu = pyfits.HDUList([pyfits.PrimaryHDU(), pyfits.ImageHDU(data=outsci*ss, header=utils.to_header(wcs_slice))])
       
        if ds9 is not None:
            ds9.set('frame 2')
            ds9.set_pyfits(hdu)

        return hdu
