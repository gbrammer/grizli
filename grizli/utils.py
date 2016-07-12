"""General utilities"""
import os
import numpy as np

# character to skip clearing line on STDOUT printing
no_newline = '\x1b[1A\x1b[1M' 

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
            print '%s: photutils.detect_sources (detect_thresh=%.1f, grow_seg=%d, gauss_fwhm=%.1f)' %(root, detect_thresh, grow_seg, gauss_fwhm)
        
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
            header = wcs.to_header()
        else:
            header=None
            
        pyfits.writeto(seg_file, data=seg, header=header, clobber=clobber)
            
        if os.path.exists(seg_cat) & clobber:
            os.remove(seg_cat)
        
        catalog.write(seg_cat, format='ascii.commented_header')
    
    return catalog, seg
    
