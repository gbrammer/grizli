"""General utilities"""
import os
import glob
import collections

import numpy as np

# character to skip clearing line on STDOUT printing
no_newline = '\x1b[1A\x1b[1M' 

def get_flt_info(files=[]):
    from astropy.io.fits import Header
    from astropy.table import Table
    
    if not files:
        files=glob.glob('*flt.fits')
    
    N = len(files)
    columns = ['FILE', 'FILTER', 'TARGNAME', 'DATE-OBS', 'TIME-OBS', 'EXPSTART', 'EXPTIME', 'PA_V3', 'RA_TARG', 'DEC_TARG', 'POSTARG1', 'POSTARG2']
    data = []
    head = Header()
    for i in range(N):
        line = [files[i]]
        h = Header().fromfile(files[i])
        filt = get_hst_filter(h)
        line.append(filt)
        for key in columns[2:]:
            line.append(h[key])
        
        data.append(line)
    
    tab = Table(rows=data, names=columns)
    return tab

def radec_to_targname(ra=0, dec=0, header=None):
    """TBD
    """
    import astropy.coordinates 
    import astropy.units as u
    import re
    
    if header is not None:
        if 'CRVAL1' in header:
            ra, dec = header['CRVAL1'], header['CRVAL2']
        else:
            if 'RA_TARG' in header:
                ra, dec = header['RA_TARG'], header['DEC_TARG']
    
    coo = astropy.coordinates.SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    
    cstr = re.split('[hmsd.]', coo.to_string('hmsdms'))
    targname = ('J%s%s' %(''.join(cstr[0:3]), ''.join(cstr[4:7])))
    targname = targname.replace(' ', '')
    
    return targname
    
def parse_flt_files(files=[], info=None, uniquename=False, 
                    translate = {'AEGIS-':'aegis-', 
                                 'COSMOS-':'cosmos-', 
                                 'GNGRISM':'goodsn-', 
                                 'GOODS-SOUTH-':'goodss-', 
                                 'UDS-':'uds-'}):
    """TBD
    Read a files.info file and make ASN files for each visit/filter[/date].
    """    
    
    if info is None:
        if not files:
            files=glob.glob('*flt.fits')
    
        if len(files) == 0:
            return False
        
        info = get_flt_info(files)
    else:
        info = info.copy()
        
    for c in info.colnames:
        if not c.islower(): 
            info.rename_column(c, c.lower())

    if 'expstart' not in info.colnames:
        info['expstart'] = info['exptime']*0.

    so = np.argsort(info['expstart'])
    info = info[so]

    #pa_v3 = np.round(info['pa_v3']*10)/10 % 360.
    pa_v3 = np.round(info['pa_v3']) % 360.
    
    target_list = []
    for i in range(len(info)):
        #### Replace ANY targets with JRhRmRs-DdDmDs
        if info['targname'][i] == 'ANY':            
            new_targname = 'par-'+radec_to_targname(ra=info['ra_targ'][i],
                                             dec=info['dec_targ'][i])
                                              
            target_list.append(new_targname.lower())
        else:
            target_list.append(info['targname'][i])
    
    target_list = np.array(target_list)

    info['progIDs'] = [file[1:4] for file in info['file']]

    progIDs = np.unique(info['progIDs'])
    visits = np.array([os.path.basename(file)[4:6] for file in info['file']])
    dates = np.array([''.join(date.split('-')[1:]) for date in info['date-obs']])
    
    targets = np.unique(target_list)
    
    output_list = collections.OrderedDict()
    filter_list = collections.OrderedDict()
    
    for filter in np.unique(info['filter']):
        filter_list[filter] = collections.OrderedDict()
        
        angles = np.unique(pa_v3[(info['filter'] == filter)]) 
        for angle in angles:
            filter_list[filter][angle] = []
            
    for target in targets:
        #### 3D-HST targname translations
        target_use = target
        for key in translate.keys():
            target_use = target_use.replace(key, translate[key])
            
        ## pad i < 10 with zero
        for key in translate.keys():
            if translate[key] in target_use:
                spl = target_use.split('-')
                try:
                    if (int(spl[-1]) < 10) & (len(spl[-1]) == 1):
                        spl[-1] = '%02d' %(int(spl[-1]))
                        target_use = '-'.join(spl)
                except:
                    pass

        for filter in np.unique(info['filter'][(target_list == target)]):
            angles = np.unique(pa_v3[(info['filter'] == filter) & 
                              (target_list == target)])
                                          
            for angle in angles:
                exposure_list = []
                exposure_start = []
                product='%s-%05.1f-%s' %(target_use, angle, filter)             

                visit_match = np.unique(visits[(target_list == target) &
                                               (info['filter'] == filter)])
                
                this_progs = []
                this_visits = []
                
                for visit in visit_match:
                    ix = (visits == visit) & (target_list == target) & (info['filter'] == filter)
                    #this_progs.append(info['progIDs'][ix][0])
                    #print visit, ix.sum(), np.unique(info['progIDs'][ix])
                    new_progs = list(np.unique(info['progIDs'][ix]))
                    this_visits.extend([visit]*len(new_progs))
                    this_progs.extend(new_progs)
                    
                for visit, prog in zip(this_visits, this_progs):
                    visit_list = []
                    visit_start = []
                    visit_product='%s-%s-%s-%05.1f-%s' %(target_use, prog, 
                                                 visit, angle, filter)             
                                            
                    use = ((target_list == target) & 
                           (info['filter'] == filter) & 
                           (visits == visit) & (pa_v3 == angle) &
                           (info['progIDs'] == prog))
                           
                    if use.sum() == 0:
                        continue

                    for tstart, file in zip(info['expstart'][use],
                                            info['file'][use]):
                                            
                        f = file.split('.gz')[0]
                        if f not in exposure_list:
                            visit_list.append(f)
                            visit_start.append(tstart)
                    
                    exposure_list = np.append(exposure_list, visit_list)
                    exposure_start.extend(visit_start)
                    
                    filter_list[filter][angle].extend(visit_list)
                    
                    if uniquename:
                        print visit_product, len(visit_list)
                        so = np.argsort(visit_start)
                        exposure_list = np.array(visit_list)[so]
                        output_list[visit_product.lower()] = visit_list
                    
                if not uniquename:
                    print product, len(exposure_list)
                    so = np.argsort(exposure_start)
                    exposure_list = np.array(exposure_list)[so]
                    output_list[product.lower()] = exposure_list
    
    return output_list, filter_list
    
def get_hst_filter(header):
    """Get simple filter name out of an HST image header.  
    
    Parameters
    ----------
    header: astropy.io.fits.Header
        Image header with FILTER or FILTER1,FILTER2,...,FILTERN keywords
    
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
        filter = header['FILTER']
    
    return filter.upper()
    
def unset_dq_bits(value, okbits=32+64+512, verbose=False):
    """
    Unset bit flags from a DQ array
    
    32, 64: these pixels usually seem OK
       512: blobs not relevant for grism exposures
    """
    bin_bits = np.binary_repr(okbits)
    n = len(bin_bits)
    for i in range(n):
        if bin_bits[-(i+1)] == '1':
            if verbose:
                print 2**i
            
            value -= (value & 2**i)
    
    return value

def detect_with_photutils(sci, err=None, dq=None, seg=None, detect_thresh=2.,
                        npixels=8, grow_seg=5, gauss_fwhm=2., gsize=3, 
                        wcs=None, save_detection=False, root='mycat',
                        background=None, gain=None, AB_zeropoint=0., 
                        rename_columns = {'xcentroid': 'x_flt',
                                          'ycentroid': 'y_flt',
                                          'ra_icrs_centroid': 'ra',
                                          'dec_icrs_centroid': 'dec'},
                        clobber=True, verbose=True):
    """Use photutils to detect objects and make segmentation map
    
    Parameters
    ----------
    detect_thresh: float
        Detection threshold, in sigma
    
    grow_seg: int
        Number of pixels to grow around the perimeter of detected objects
        witha  maximum filter
    
    gauss_fwhm: float
        FWHM of Gaussian convolution kernel that smoothes the detection
        image.
    
    verbose: bool
        Print logging information to the terminal
    
    save_detection: bool
        Save the detection images and catalogs
    
    wcs: astropy.wcs.WCS
        WCS object passed to `photutils.source_properties` used to compute
        sky coordinates of detected objects.
    
    ToDo: abstract to general script with data/thresholds 
          and just feed in image arrays
          
    Returns
    ---------
    catalog: astropy.table.Table
        Object catalog with the default parametersobject to `catalog`.
    """
    import scipy.ndimage as nd
    
    from photutils import detect_threshold, detect_sources, SegmentationImage
    from photutils import source_properties, properties_table
    
    import astropy.io.fits as pyfits
    from astropy.table import Column
    
    from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
    from astropy.convolution import Gaussian2DKernel
    
    ### DQ masks
    mask = (sci == 0)
    if dq is not None:
        mask |= dq > 0
    
    ### Detection threshold
    if err is None:
        threshold = detect_threshold(sci, snr=detect_thresh, mask=mask)
    else:
        threshold = (detect_thresh * err)*(~mask)
        threshold[mask] = np.median(threshold[~mask])
    
    if seg is None:
        ####### Run the source detection and create the segmentation image
                        
        ### Gaussian kernel
        sigma = gauss_fwhm * gaussian_fwhm_to_sigma    # FWHM = 2.
        kernel = Gaussian2DKernel(sigma, x_size=gsize, y_size=gsize)
        kernel.normalize()
    
        if verbose:
            print '%s: photutils.detect_sources (detect_thresh=%.1f, grow_seg=%d, gauss_fwhm=%.1f, ZP=%.1f)' %(root, detect_thresh, grow_seg, gauss_fwhm, AB_zeropoint)
        
        ### Detect sources
        segm = detect_sources(sci*(~mask), threshold, npixels=npixels,
                              filter_kernel=kernel)   
                          
        grow = nd.maximum_filter(segm.array, grow_seg)
        seg = np.cast[np.float32](grow)
    else:
        ######## Use the supplied segmentation image
        segm = SegmentationImage(seg)
        
    ### Source properties catalog
    if verbose:
        print  '%s: photutils.source_properties' %(root)
    
    props = source_properties(sci, segm, error=threshold/detect_thresh,
                              mask=mask, effective_gain=gain,
                              background=background, wcs=wcs)
                              
    catalog = properties_table(props)
    
    ### Mag columns
    mag = AB_zeropoint - 2.5*np.log10(catalog['source_sum'])
    mag._name = 'mag'
    catalog.add_column(mag)
      
    try:
        logscale = 2.5/np.log(10)
        mag_err = logscale*catalog['source_sum_err']/catalog['source_sum']
    except:
        mag_err = np.zeros_like(mag)-99
    
    mag_err._name = 'mag_err'
    catalog.add_column(mag_err)
    
    ### Rename some catalog columns    
    for key in rename_columns.keys():
        if key not in catalog.colnames:
            continue
        
        catalog.rename_column(key, rename_columns[key])
        if verbose:
            print 'Rename column: %s -> %s' %(key, rename_columns[key])
    
    ### Done!
    if verbose:
        print no_newline + ('%s: photutils.source_properties - %d objects'
                             %(root, len(catalog)))
    
    #### Save outputs?
    if save_detection:
        seg_file = root + '.detect_seg.fits'
        seg_cat  = root + '.detect.cat'
        if verbose:
            print '%s: save %s, %s' %(root, seg_file, seg_cat)
        
        if wcs is not None:
            header = wcs.to_header(relax=True)
        else:
            header=None
            
        pyfits.writeto(seg_file, data=seg, header=header, clobber=clobber)
            
        if os.path.exists(seg_cat) & clobber:
            os.remove(seg_cat)
        
        catalog.write(seg_cat, format='ascii.commented_header')
    
    return catalog, seg
    
#
def nmad(data):
    """TBD
    """
    import astropy.stats
    return 1.48*astropy.stats.median_absolute_deviation(data)

def get_line_wavelengths():
    """TBD
    """
    line_wavelengths = {} ; line_ratios = {}
    line_wavelengths['Ha'] = [6564.61]
    line_ratios['Ha'] = [1.]
    line_wavelengths['Hb'] = [4862.68]
    line_ratios['Hb'] = [1.]
    line_wavelengths['Hg'] = [4341.68]
    line_ratios['Hg'] = [1.]
    line_wavelengths['Hd'] = [4102.892]
    line_ratios['Hd'] = [1.]
    line_wavelengths['OIIIx'] = [4364.436]
    line_ratios['OIIIx'] = [1.]
    line_wavelengths['OIII'] = [5008.240, 4960.295]
    line_ratios['OIII'] = [2.98, 1]
    line_wavelengths['OIII+Hb'] = [5008.240, 4960.295, 4862.68]
    line_ratios['OIII+Hb'] = [2.98, 1, 3.98/8.]
    
    line_wavelengths['OIII+Hb+Ha'] = [5008.240, 4960.295, 4862.68, 6564.61]
    line_ratios['OIII+Hb+Ha'] = [2.98, 1, 3.98/10., 3.98/10.*2.86]

    line_wavelengths['OIII+Hb+Ha+SII'] = [5008.240, 4960.295, 4862.68, 6564.61, 6718.29, 6732.67]
    line_ratios['OIII+Hb+Ha+SII'] = [2.98, 1, 3.98/10., 3.98/10.*2.86*4, 3.98/10.*2.86/10.*4, 3.98/10.*2.86/10.*4]

    line_wavelengths['OIII+OII'] = [5008.240, 4960.295, 3729.875]
    line_ratios['OIII+OII'] = [2.98, 1, 3.98/4.]

    line_wavelengths['OII'] = [3729.875]
    line_ratios['OII'] = [1]
    line_wavelengths['OI'] = [6302.046]
    line_ratios['OI'] = [1]

    line_wavelengths['NeIII'] = [3869]
    line_ratios['NeIII'] = [1.]
    line_wavelengths['NeV'] = [3346.8]
    line_ratios['NeV'] = [1.]
    line_wavelengths['NeVI'] = [3426.85]
    line_ratios['NeVI'] = [1.]
    line_wavelengths['SIII'] = [9068.6, 9530.6]
    line_ratios['SIII'] = [1, 2.44]
    line_wavelengths['HeII'] = [4687.5]
    line_ratios['HeII'] = [1.]
    line_wavelengths['HeI'] = [5877.2]
    line_ratios['HeI'] = [1.]
    line_wavelengths['HeIb'] = [3889.5]
    line_ratios['HeIb'] = [1.]
    
    line_wavelengths['MgII'] = [2799.117]
    line_ratios['MgII'] = [1.]
    line_wavelengths['CIV'] = [1549.480]
    line_ratios['CIV'] = [1.]
    line_wavelengths['Lya'] = [1215.4]
    line_ratios['Lya'] = [1.]

    line_wavelengths['Ha+SII'] = [6564.61, 6718.29, 6732.67]
    line_ratios['Ha+SII'] = [1., 1./10, 1./10]
    
    line_wavelengths['SII'] = [6718.29, 6732.67]
    line_ratios['SII'] = [1., 1.]   
    
    return line_wavelengths, line_ratios 
    
class SpectrumTemplate(object):
    """TBD
    """
    def __init__(self, wave=None, flux=None, fwhm=None, velocity=False):
        """TBD
        """
        self.wave = wave
        self.flux = flux
        
        ### Gaussian
        if (wave is not None) & (fwhm is not None):
            rms = fwhm/2.35
            if velocity:
                rms *= wave/3.e5
                
            xgauss = np.arange(-5,5.01,0.1)*rms+wave
            gaussian = np.exp(-(xgauss-wave)**2/2/rms**2)
            gaussian /= np.sqrt(2*np.pi*rms**2)
            
            self.wave = xgauss
            self.flux = gaussian
    
    def __add__(self, spectrum):
        """TBD
        """
        new_wave = np.unique(np.append(self.wave, spectrum.wave))
        new_wave.sort()
        
        new_flux = np.interp(new_wave, self.wave, self.flux)
        new_flux += np.interp(new_wave, spectrum.wave, spectrum.flux)
        return SpectrumTemplate(wave=new_wave, flux=new_flux)
    
    def __mul__(self, scalar):
        return SpectrumTemplate(wave=self.wave, flux=self.flux*scalar)
        
    def zscale(self, z, scalar=1):
        return SpectrumTemplate(wave=self.wave*(1+z),
                                flux=self.flux*scalar/(1+z))

def log_zgrid(zr=[0.7,3.4], dz=0.01):
    """TBD
    """
    zgrid = np.exp(np.arange(np.log(1+zr[0]), np.log(1+zr[1]), dz))-1
    return zgrid

def zoom_zgrid(zgrid, chi2nu, threshold=0.01, factor=10, grow=7):
    """TBD
    """
    import scipy.ndimage as nd
    
    mask = (chi2nu-chi2nu.min()) < threshold
    if grow > 1:
        mask_grow = nd.maximum_filter(mask*1, size=grow)
        mask = mask_grow > 0
        
    if mask.sum() == 0:
        return []
    
    idx = np.arange(zgrid.shape[0])
    out_grid = []
    for i in idx[mask]:
        if i == idx[-1]:
            continue
            
        out_grid = np.append(out_grid, np.linspace(zgrid[i], zgrid[i+1], factor+2)[1:-1])
    
    return out_grid

def get_wcs_pscale(wcs):
    """Get correct pscale from `astropy.wcs.WCS` object
    
    Parameters
    ----------
    wcs : `astropy.wcs.WCS`
        
    Returns
    -------
    pscale : float
        Pixel scale from `wcs.cd`
        
    """
    from numpy import linalg
    det = linalg.det(wcs.wcs.cd)
    pscale = np.sqrt(np.abs(det))*3600.
    return pscale
    
def make_spectrum_wcsheader(center_wave=1.4e4, dlam=40, NX=100, spatial_scale=1, NY=10):
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    
    h = pyfits.ImageHDU(data=np.zeros((2*NY, 2*NX), dtype=np.float32))
    
    refh = h.header
    refh['CRPIX1'] = NX+1
    refh['CRPIX2'] = NY+1
    refh['CRVAL1'] = center_wave
    refh['CD1_1'] = dlam
    refh['CD1_2'] = 0.
    refh['CRVAL2'] = 0.
    refh['CD2_2'] = spatial_scale
    refh['CD2_1'] = 0.
    refh['RADESYS'] = ''
    
    refh['CTYPE1'] = 'WAVE'
    refh['CTYPE2'] = 'LINEAR'
    
    ref_wcs = pywcs.WCS(h.header)
    ref_wcs.pscale = np.sqrt(ref_wcs.wcs.cd[0,0]**2 + ref_wcs.wcs.cd[1,0]**2)*3600.
    
    return refh, ref_wcs

def make_wcsheader(ra=40.07293, dec=-1.6137748, size=2, pixscale=0.1, get_hdu=False, theta=0):
    """TBD
    
    """
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    
    cdelt = pixscale/3600.
    if isinstance(size, list):
        npix = np.cast[int]([size[0]/pixscale, size[1]/pixscale])
    else:
        npix = np.cast[int]([size/pixscale, size/pixscale])
        
    hout = pyfits.Header()
    hout['CRPIX1'] = npix[0]/2
    hout['CRPIX2'] = npix[1]/2
    hout['CRVAL1'] = ra
    hout['CRVAL2'] = dec
    hout['CD1_1'] = -cdelt
    hout['CD1_2'] = hout['CD2_1'] = 0.
    hout['CD2_2'] = cdelt
    hout['NAXIS1'] = npix[0]
    hout['NAXIS2'] = npix[1]
    hout['CTYPE1'] = 'RA---TAN'
    hout['CTYPE2'] = 'DEC--TAN'
    
    wcs_out = pywcs.WCS(hout)
    
    theta_rad = np.deg2rad(theta)
    mat = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], 
                    [np.sin(theta_rad),  np.cos(theta_rad)]])

    rot_cd = np.dot(mat, wcs_out.wcs.cd)
    
    for i in [0,1]:
        for j in [0,1]:
            hout['CD%d_%d' %(i+1, j+1)] = rot_cd[i,j]
            wcs_out.wcs.cd[i,j] = rot_cd[i,j]
                
    cd = wcs_out.wcs.cd
    wcs_out.pscale = np.sqrt((cd[0,:]**2).sum())*3600.
        
    if get_hdu:
        hdu = pyfits.ImageHDU(header=hout, data=np.zeros((npix[1], npix[0]), dtype=np.float32))
        return hdu
    else:
        return hout, wcs_out

    
    