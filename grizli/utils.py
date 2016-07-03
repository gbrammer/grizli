"""General utilities"""
import numpy as np

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
