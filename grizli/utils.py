"""General utilities"""
import os
import glob
from collections import OrderedDict

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.table

import numpy as np

# character to skip clearing line on STDOUT printing
no_newline = '\x1b[1A\x1b[1M' 

def get_flt_info(files=[]):
    """Extract header information from a list of FLT files
    
    Parameters
    -----------
    files : list
        List of exposure filenames.
        
    Returns
    --------
    tab : `~astropy.table.Table`
        Table containing header keywords
        
    """
    import astropy.io.fits as pyfits
    from astropy.table import Table
    
    if not files:
        files=glob.glob('*flt.fits')
    
    N = len(files)
    columns = ['FILE', 'FILTER', 'TARGNAME', 'DATE-OBS', 'TIME-OBS', 'EXPSTART', 'EXPTIME', 'PA_V3', 'RA_TARG', 'DEC_TARG', 'POSTARG1', 'POSTARG2']
    data = []

    for i in range(N):
        line = [os.path.basename(files[i]).split('.gz')[0]]
        if files[i].endswith('.gz'):
            im = pyfits.open(files[i])
            h = im[0].header
        else:
            h = pyfits.Header().fromfile(files[i])
        
        filt = get_hst_filter(h)
        line.append(filt)
        has_columns = ['FILE', 'FILTER']
        for key in columns[2:]:
            if key in h:
                line.append(h[key])
                has_columns.append(key)
            else:
                continue
                
        data.append(line)
    
    tab = Table(rows=data, names=has_columns)
    return tab

def radec_to_targname(ra=0, dec=0, header=None):
    """Turn decimal degree coordinates into a string
    
    Example:

        >>> from grizli.utils import radec_to_targname
        >>> print(radec_to_targname(ra=10., dec=-10.))
        j004000-100000
    
    Parameters
    -----------
    ra, dec : float
        Sky coordinates in decimal degrees
    
    header : `~astropy.io.fits.Header` or None
        Optional FITS header with CRVAL or RA/DEC_TARG keywords.  If 
        specified, read `ra`/`dec` from CRVAL1/CRVAL2 or RA_TARG/DEC_TARG
        keywords, whichever are available
    
    Returns
    --------
    targname : str
        Target name like jHHMMSS[+-]DDMMSS.
    
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
    
    cstr = re.split('[hmsd.]', coo.to_string('hmsdms', precision=2))
    targname = ('j{0}{1}'.format(''.join(cstr[0:3]), ''.join(cstr[4:7])))
    targname = targname.replace(' ', '')
    
    return targname
    
def parse_flt_files(files=[], info=None, uniquename=False, use_visit=False,
                    get_footprint = False, 
                    translate = {'AEGIS-':'aegis-', 
                                 'COSMOS-':'cosmos-', 
                                 'GNGRISM':'goodsn-', 
                                 'GOODS-SOUTH-':'goodss-', 
                                 'UDS-':'uds-'}):
    """Read header information from a list of exposures and parse out groups based on filter/target/orientation.
    
    Parameters
    -----------
    files : list
        List of exposure filenames.  If not specified, use *flt.fits.
        
    info : None or output from `~grizli.utils.get_flt_info`.
    
    uniquename : bool
        If True, then split everything by program ID and visit name.  If 
        False, then just group by targname/filter/pa_v3.
    
    use_visit : bool
        For parallel observations with `targname='ANY'`, use the filename 
        up to the visit ID as the target name.  For example:
        
            >>> flc = 'jbhj64d8q_flc.fits'
            >>> visit_targname = flc[:6]
            >>> print(visit_targname)
            jbhj64
        
        If False, generate a targname for parallel observations based on the
        pointing coordinates using `radec_to_targname`.  Use this keyword
        for dithered parallels like 3D-HST / GLASS but set to False for
        undithered parallels like WISP.  Should also generally be used with
        `uniquename=False` otherwise generates names that are a bit redundant:
            
            +--------------+---------------------------+
            | `uniquename` | Output Targname           |
            +==============+===========================+
            |     True     | jbhj45-bhj-45-180.0-F814W |
            +--------------+---------------------------+
            |     False    | jbhj45-180.0-F814W        |
            +--------------+---------------------------+
            
    translate : dict
        Translation dictionary to modify TARGNAME keywords to some other 
        value.  Used like:
        
            >>> targname = 'GOODS-SOUTH-10'
            >>> translate = {'GOODS-SOUTH-': 'goodss-'}
            >>> for k in translate:
            >>>     targname = targname.replace(k, translate[k])
            >>> print(targname)
            goodss-10
        
    Returns
    --------
    output_list : dict
        Dictionary split by target/filter/pa_v3. Keys are derived visit
        product names and values are lists of exposure filenames corresponding
        to that set. Keys are generated with the formats like:
            
            >>> targname = 'macs1149+2223'
            >>> pa_v3 = 32.0
            >>> filter = 'f140w'
            >>> flt_filename = 'ica521naq_flt.fits'
            >>> propstr = flt_filename[1:4]
            >>> visit = flt_filename[4:6]
            >>> # uniquename = False
            >>> print('{0}-{1:05.1f}-{2}'.format(targname, pa_v3, filter))
            macs1149.6+2223-032.0-f140w
            >>> # uniquename = True
            >>> print('{0}-{1:3s}-{2:2s}-{3:05.1f}-{4:s}'.format(targname, propstr, visit, pa_v3, filter))
            macs1149.6+2223-ca5-21-032.0-f140w
        
    filter_list : dict
        Nested dictionary split by filter and then PA_V3.  This shouldn't  
        be used if exposures from completely disjoint pointings are stored
        in the same working directory.
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
            if use_visit:
                new_targname=info['file'][i][:6]
            else:
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
    
    output_list = [] #OrderedDict()
    filter_list = OrderedDict()
    
    for filter in np.unique(info['filter']):
        filter_list[filter] = OrderedDict()
        
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
                        spl[-1] = '{0:02d}'.format(int(spl[-1]))
                        target_use = '-'.join(spl)
                except:
                    pass

        for filter in np.unique(info['filter'][(target_list == target)]):
            angles = np.unique(pa_v3[(info['filter'] == filter) & 
                              (target_list == target)])
                                          
            for angle in angles:
                exposure_list = []
                exposure_start = []
                product='{0}-{1:05.1f}-{2}'.format(target_use, angle, filter)             

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
                    visit_product = '{0}-{1}-{2}-{3:05.1f}-{4}'.format(target_use, prog, visit, angle, filter)             
                                            
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
                            visit_list.append(str(f))
                            visit_start.append(tstart)
                    
                    exposure_list = np.append(exposure_list, visit_list)
                    exposure_start.extend(visit_start)
                    
                    filter_list[filter][angle].extend(visit_list)
                    
                    if uniquename:
                        print(visit_product, len(visit_list))
                        so = np.argsort(visit_start)
                        exposure_list = np.array(visit_list)[so]
                        #output_list[visit_product.lower()] = visit_list
                        
                        d = OrderedDict(product=str(visit_product.lower()),
                                        files=list(np.array(visit_list)[so]))
                        output_list.append(d)
                        
                if not uniquename:
                    print(product, len(exposure_list))
                    so = np.argsort(exposure_start)
                    exposure_list = np.array(exposure_list)[so]
                    #output_list[product.lower()] = exposure_list
                    d = OrderedDict(product=str(product.lower()),
                                    files=list(np.array(exposure_list)[so]))
                    output_list.append(d)
    
    ### Get visit footprint from FLT WCS
    if get_footprint:
        from shapely.geometry import Polygon
        
        N = len(output_list)
        for i in range(N):
            for j in range(len(output_list[i]['files'])):
                flt_file = output_list[i]['files'][j]
                if (not os.path.exists(flt_file)) & os.path.exists('../RAW/'+flt_file):
                    flt_file = '../RAW/'+flt_file
                    
                flt_j = pyfits.open(flt_file)
                h = flt_j[0].header
                if (h['INSTRUME'] == 'WFC3') & (h['DETECTOR'] == 'IR'):
                    wcs_j = pywcs.WCS(flt_j['SCI',1])
                else:
                    wcs_j = pywcs.WCS(flt_j['SCI',1], fobj=flt_j)
                    
                fp_j = Polygon(wcs_j.calc_footprint())
                if j == 0:
                    fp_i = fp_j
                else:
                    fp_i = fp_i.union(fp_j)
            
            output_list[i]['footprint'] = fp_i
            
    return output_list, filter_list

def parse_visit_overlaps(visits, buffer=15.):
    """Find overlapping visits/filters to make combined mosaics
    
    Parameters
    ----------
    visits : list
        Output list of visit information from `~grizli.utils.parse_flt_files`.
        The script looks for files like `visits[i]['product']+'_dr?_sci.fits'` 
        to compute the WCS footprint of a visit.  These are produced, e.g., by 
        `~grizli.prep.process_direct_grism_visit`.
        
    buffer : float
        Buffer, in `~astropy.units.arcsec`, to add around visit footprints to 
        look for overlaps.
    
    Returns
    -------
    exposure_groups : list
        List of overlapping visits, with similar format as input `visits`.
        
    """
    import copy
    from shapely.geometry import Polygon
    
    N = len(visits)

    exposure_groups = []
    used = np.arange(len(visits)) < 0
    
    for i in range(N):
        f_i = visits[i]['product'].split('-')[-1]
        if used[i]:
            continue
        
        im_i = pyfits.open(glob.glob(visits[i]['product']+'_dr?_sci.fits')[0])
        wcs_i = pywcs.WCS(im_i[0])
        fp_i = Polygon(wcs_i.calc_footprint()).buffer(buffer/3600.)
        
        exposure_groups.append(copy.deepcopy(visits[i]))
        
        for j in range(i+1, N):
            f_j = visits[j]['product'].split('-')[-1]
            if (f_j != f_i) | (used[j]):
                continue
                
            im_j = pyfits.open(glob.glob(visits[j]['product']+'_dr?_sci.fits')[0])
            wcs_j = pywcs.WCS(im_j[0])
            fp_j = Polygon(wcs_j.calc_footprint()).buffer(buffer/3600.)
            
            olap = fp_i.intersection(fp_j)
            if olap.area > 0:
                used[j] = True
                fp_i = fp_i.union(fp_j)
                exposure_groups[-1]['footprint'] = fp_i
                exposure_groups[-1]['files'].extend(visits[j]['files'])
                
    for i in range(len(exposure_groups)):
        flt_i = pyfits.open(exposure_groups[i]['files'][0])
        product = flt_i[0].header['TARGNAME'].lower()        
        if product == 'any':
            product = 'par-'+radec_to_targname(header=flt_i['SCI',1].header)
        
        f_i = exposure_groups[i]['product'].split('-')[-1]
        product += '-'+f_i
        exposure_groups[i]['product'] = product
    
    return exposure_groups
    
def parse_grism_associations(exposure_groups, 
                             best_direct={'G102':'F105W', 'G141':'F140W'},
                             get_max_overlap=True):
    """Get associated lists of grism and direct exposures
    
    Parameters
    ----------
    exposure_grups : list
        Output list of overlapping visits from
        `~grizli.utils.parse_visit_overlaps`.
    
    best_direct : dict
        Dictionary of the preferred direct imaging filters to use with a 
        particular grism.
    
    Returns
    -------
    grism_groups : list
        List of dictionaries with associated 'direct' and 'grism' entries.
        
    """
    N = len(exposure_groups)
    
    grism_groups = []
    for i in range(N):
        f_i = exposure_groups[i]['product'].split('-')[-1]
        root_i = exposure_groups[i]['product'].split('-'+f_i)[0]
        
        if f_i.startswith('g'):
            group = OrderedDict(grism=exposure_groups[i], 
                                direct=None)
        else:
            continue
        
        fp_i = exposure_groups[i]['footprint']
        olap_i = 0.
        d_i = f_i
        
        #print('\nx\n')
        for j in range(N):
            f_j = exposure_groups[j]['product'].split('-')[-1]
            if f_j.startswith('g'):
                continue
                 
            fp_j = exposure_groups[j]['footprint']
            olap = fp_i.intersection(fp_j)
            root_j = exposure_groups[j]['product'].split('-'+f_j)[0]

            #print(root_j, root_i, root_j == root_i)
            if (root_j == root_i):
                if (group['direct'] is not None):
                    pass
                    if (group['direct']['product'].startswith(root_i)) & (d_i.upper() == best_direct[f_i.upper()]):
                        continue
                    
                group['direct'] = exposure_groups[j]
                olap_i = olap.area
                d_i = f_j
                #print(0,group['grism']['product'], group['direct']['product'])
            #     continue
                
            #print(exposure_groups[i]['product'], exposure_groups[j]['product'], olap.area*3600.)
            
            #print(exposure_groups[j]['product'], olap_i, olap.area)
            if olap.area > 0:
                if group['direct'] is None:
                    group['direct'] = exposure_groups[j]
                    olap_i = olap.area
                    d_i = f_j
                    #print(1,group['grism']['product'], group['direct']['product'])
                else:
                    #if (f_j.upper() == best_direct[f_i.upper()]):
                    if get_max_overlap:
                        if olap.area < olap_i:
                            continue
                        
                        if d_i.upper() == best_direct[f_i.upper()]:
                            continue
                                
                    group['direct'] = exposure_groups[j]
                    #print(exposure_groups[j]['product'])
                    olap_i = olap.area
                    d_i = f_j
                    #print(2,group['grism']['product'], group['direct']['product'])
                    
        grism_groups.append(group)
    
    return grism_groups
            
def get_hst_filter(header):
    """Get simple filter name out of an HST image header.  
    
    ACS has two keywords for the two filter wheels, so just return the 
    non-CLEAR filter. For example, 
    
        >>> h = astropy.io.fits.Header()
        >>> h['INSTRUME'] = 'ACS'
        >>> h['FILTER1'] = 'CLEAR1L'
        >>> h['FILTER2'] = 'F814W'
        >>> from grizli.utils import get_hst_filter
        >>> print(get_hst_filter(h))
        F814W
        >>> h['FILTER1'] = 'G800L'
        >>> h['FILTER2'] = 'CLEAR2L'
        >>> print(get_hst_filter(h))
        G800L
    
    Parameters
    -----------
    header : `~astropy.io.fits.Header`
        Image header with FILTER or FILTER1,FILTER2,...,FILTERN keywords
    
    Returns
    --------
    filter : str
            
    """
    if header['INSTRUME'].strip() == 'ACS':
        for i in [1,2]:
            filter_i = header['FILTER{0:d}'.format(i)]
            if 'CLEAR' in filter_i:
                continue
            else:
                filter = filter_i
    elif header['INSTRUME'] == 'WFPC2':
        filter = header['FILTNAM1']
    else:
        filter = header['FILTER']
    
    return filter.upper()
    
def unset_dq_bits(value, okbits=32+64+512, verbose=False):
    """
    Unset bit flags from a DQ array
    
    For WFC3/IR, the following DQ bits can usually be unset: 
    
    32, 64: these pixels usually seem OK
       512: blobs not relevant for grism exposures
    
    Parameters
    ----------
    value : int, `~numpy.ndarray`
        Input DQ value
    
    okbits : int
        Bits to unset
        
    verbose : bool
        Print some information
        
    Returns
    -------
    new_value : int, `~numpy.ndarray`
    
    """
    bin_bits = np.binary_repr(okbits)
    n = len(bin_bits)
    for i in range(n):
        if bin_bits[-(i+1)] == '1':
            if verbose:
                print(2**i)
            
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
    """Use `photutils <https://photutils.readthedocs.io/>`__ to detect objects and make segmentation map
    
    Parameters
    ----------
    sci : `~numpy.ndarray`
        TBD
    
    err, dq, seg : TBD
    
    detect_thresh : float
        Detection threshold, in :math:`\sigma`
    
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
    
    wcs : `~astropy.wcs.WCS`
        WCS object passed to `photutils.source_properties` used to compute
        sky coordinates of detected objects.
    
    Returns
    ---------
    catalog : `~astropy.table.Table`
        Object catalog with the default parameters.
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
            print('{0}: photutils.detect_sources (detect_thresh={1:.1f}, grow_seg={2:d}, gauss_fwhm={3:.1f}, ZP={4:.1f})'.format(root, detect_thresh, grow_seg, gauss_fwhm, AB_zeropoint))
        
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
        print('{0}: photutils.source_properties'.format(root))
    
    props = source_properties(sci, segm, error=threshold/detect_thresh,
                              mask=mask, background=background, wcs=wcs)
                              
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
            print('Rename column: {0} -> {1}'.format(key, rename_columns[key]))
    
    ### Done!
    if verbose:
        print(no_newline + ('{0}: photutils.source_properties - {1:d} objects'.format(root, len(catalog))))
    
    #### Save outputs?
    if save_detection:
        seg_file = root + '.detect_seg.fits'
        seg_cat  = root + '.detect.cat'
        if verbose:
            print('{0}: save {1}, {2}'.format(root, seg_file, seg_cat))
        
        if wcs is not None:
            header = wcs.to_header(relax=True)
        else:
            header=None
            
        pyfits.writeto(seg_file, data=seg, header=header, clobber=clobber)
            
        if os.path.exists(seg_cat) & clobber:
            os.remove(seg_cat)
        
        catalog.write(seg_cat, format='ascii.commented_header')
    
    return catalog, seg
    
def nmad(data):
    """Normalized NMAD=1.48 * `~.astropy.stats.median_absolute_deviation`
    
    """
    import astropy.stats
    return 1.48*astropy.stats.median_absolute_deviation(data)

def get_line_wavelengths():
    """Get a dictionary of common emission line wavelengths and line ratios
    
    Returns
    -------
    line_wavelengths, line_ratios : dict
        Keys are common to both dictionaries and are simple names for lines
        and line complexes.  Values are lists of line wavelengths and line 
        ratios.
        
            >>> from grizli.utils import get_line_wavelengths
            >>> line_wavelengths, line_ratios = get_line_wavelengths()
            >>> print(line_wavelengths['Ha'], line_ratios['Ha'])
            [6564.61] [1.0]
            >>> print(line_wavelengths['OIII'], line_ratios['OIII'])
            [5008.24, 4960.295] [2.98, 1]
        
        Includes some additional combined line complexes useful for redshift
        fits:
        
            >>> from grizli.utils import get_line_wavelengths
            >>> line_wavelengths, line_ratios = get_line_wavelengths()
            >>> key = 'Ha+SII+SIII+He'
            >>> print(line_wavelengths[key], '\\n', line_ratios[key])
            [6564.61, 6718.29, 6732.67, 9068.6, 9530.6, 10830.0]
            [1.0, 0.1, 0.1, 0.05, 0.122, 0.04]
        
    """
    line_wavelengths = OrderedDict() ; line_ratios = OrderedDict()
    line_wavelengths['Ha'] = [6564.61]
    line_ratios['Ha'] = [1.]
    line_wavelengths['Hb'] = [4862.68]
    line_ratios['Hb'] = [1.]
    line_wavelengths['Hg'] = [4341.68]
    line_ratios['Hg'] = [1.]
    line_wavelengths['Hd'] = [4102.892]
    line_ratios['Hd'] = [1.]
    line_wavelengths['OIII-4363'] = [4364.436]
    line_ratios['OIII-4363'] = [1.]
    line_wavelengths['OIII'] = [5008.240, 4960.295]
    line_ratios['OIII'] = [2.98, 1]
    line_wavelengths['OIII+Hb'] = [5008.240, 4960.295, 4862.68]
    line_ratios['OIII+Hb'] = [2.98, 1, 3.98/6.]
    
    line_wavelengths['OIII+Hb+Ha'] = [5008.240, 4960.295, 4862.68, 6564.61]
    line_ratios['OIII+Hb+Ha'] = [2.98, 1, 3.98/10., 3.98/10.*2.86]

    line_wavelengths['OIII+Hb+Ha+SII'] = [5008.240, 4960.295, 4862.68, 6564.61, 6718.29, 6732.67]
    line_ratios['OIII+Hb+Ha+SII'] = [2.98, 1, 3.98/10., 3.98/10.*2.86*4, 3.98/10.*2.86/10.*4, 3.98/10.*2.86/10.*4]

    line_wavelengths['OIII+OII'] = [5008.240, 4960.295, 3729.875]
    line_ratios['OIII+OII'] = [2.98, 1, 3.98/4.]

    line_wavelengths['OII'] = [3729.875]
    line_ratios['OII'] = [1]
    
    line_wavelengths['OII+Ne'] = [3729.875, 3869]
    line_ratios['OII+Ne'] = [1, 1./5]
    
    line_wavelengths['OI-6302'] = [6302.046, 6363.67]
    line_ratios['OI-6302'] = [1, 0.33]

    line_wavelengths['NeIII'] = [3869]
    line_ratios['NeIII'] = [1.]
    line_wavelengths['NeV'] = [3346.8]
    line_ratios['NeV'] = [1.]
    line_wavelengths['NeVI'] = [3426.85]
    line_ratios['NeVI'] = [1.]
    line_wavelengths['SIII'] = [9068.6, 9530.6][::-1]
    line_ratios['SIII'] = [1, 2.44][::-1]
    line_wavelengths['HeII'] = [4687.5]
    line_ratios['HeII'] = [1.]
    line_wavelengths['HeI-5877'] = [5877.2]
    line_ratios['HeI-5877'] = [1.]
    line_wavelengths['HeI-3889'] = [3889.5]
    line_ratios['HeI-3889'] = [1.]
    
    line_wavelengths['MgII'] = [2799.117]
    line_ratios['MgII'] = [1.]
    
    line_wavelengths['CIV'] = [1549.480]
    line_ratios['CIV'] = [1.]
    line_wavelengths['CIII]'] = [1908.]
    line_ratios['CIII]'] = [1.]
    line_wavelengths['OIII]'] = [1663.]
    line_ratios['OIII]'] = [1.]
    line_wavelengths['HeII-1640'] = [1640.]
    line_ratios['HeII-1640'] = [1.]
    line_wavelengths['NIII]'] = [1750.]
    line_ratios['NIII]'] = [1.]
    line_wavelengths['NIV'] = [1487.]
    line_ratios['NIV'] = [1.]
    line_wavelengths['NV'] = [1240.]
    line_ratios['NV'] = [1.]

    line_wavelengths['Lya'] = [1215.4]
    line_ratios['Lya'] = [1.]
    
    line_wavelengths['Lya+CIV'] = [1215.4, 1549.49]
    line_ratios['Lya+CIV'] = [1., 0.1]
    
    line_wavelengths['Ha+SII'] = [6564.61, 6718.29, 6732.67]
    line_ratios['Ha+SII'] = [1., 1./10, 1./10]
    line_wavelengths['Ha+SII+SIII+He'] = [6564.61, 6718.29, 6732.67, 9068.6, 9530.6, 10830.]
    line_ratios['Ha+SII+SIII+He'] = [1., 1./10, 1./10, 1./20, 2.44/20, 1./25.]

    line_wavelengths['Ha+NII+SII+SIII+He'] = [6564.61, 6549.86, 6585.27, 6718.29, 6732.67, 9068.6, 9530.6, 10830.]
    line_ratios['Ha+NII+SII+SIII+He'] = [1., 1./(4.*4), 3./(4*4), 1./10, 1./10, 1./20, 2.44/20, 1./25.]
    
    line_wavelengths['NII'] = [6549.86, 6585.27]
    line_ratios['NII'] = [1., 3]
    
    line_wavelengths['SII'] = [6718.29, 6732.67]
    line_ratios['SII'] = [1., 1.]   
    
    return line_wavelengths, line_ratios 
    
class SpectrumTemplate(object):
    def __init__(self, wave=None, flux=None, fwhm=None, velocity=False):
        """Container for template spectra.   
                
        Parameters
        ----------
        wave, fwhm : None or float or array-like
            If both are float, then initialize with a Gaussian.  
            In `astropy.units.Angstrom`.
            
        flux : None or array-like
            Flux array (f-lambda flux density)
            
        velocity : bool
            `fwhm` is a velocity.
            
        Attributes
        ----------
        wave, flux : array-like
            Passed from the input parameters or generated/modified later.
        
        Methods
        -------
        __add__, __mul__ : Addition and multiplication of templates.
        
        Examples
        --------
        
            .. plot::
                :include-source:

                import matplotlib.pyplot as plt
                from grizli.utils import SpectrumTemplate
                
                ha = SpectrumTemplate(wave=6563., fwhm=10)
                plt.plot(ha.wave, ha.flux)
                
                ha_z = ha.zscale(0.1)
                plt.plot(ha_z.wave, ha_z.flux, label='z=0.1')
                
                plt.legend()
                plt.xlabel(r'$\lambda$')
                
                plt.show()
            
        """
        self.wave = wave
        self.flux = flux

        if (wave is not None) & (fwhm is not None):
            self.make_gaussian(wave, fwhm, velocity=velocity)
            
    def make_gaussian(self, wave, fwhm, max_sigma=5, step=0.1, 
                      velocity=False):
        """Make Gaussian template
        
        Parameters
        ----------
        wave, fwhm : None or float or array-like
            Central wavelength and FWHM of the desired Gaussian
            
        velocity : bool
            `fwhm` is a velocity.
        
        max_sigma, step : float
            Generated wavelength array is
                
                >>> rms = fwhm/2.35
                >>> xgauss = np.arange(-max_sigma, max_sigma, step)*rms+wave

        Returns
        -------
        Stores `wave`, `flux` attributes.        
        """
        rms = fwhm/2.35
        if velocity:
            rms *= wave/3.e5
            
        xgauss = np.arange(-max_sigma, max_sigma, step)*rms+wave
        gaussian = np.exp(-(xgauss-wave)**2/2/rms**2)
        gaussian /= np.sqrt(2*np.pi*rms**2)
        
        self.wave = xgauss
        self.flux = gaussian

    def zscale(self, z, scalar=1):
        """Redshift the template and multiply by a scalar.
        
        Parameters
        ----------
        z : float
            Redshift to use.
        
        scalar : float
            Multiplicative factor.  Additional factor of 1./(1+z) is implicit.
        
        Returns
        -------
        new_spectrum : `~grizli.utils.SpectrumTemplate`   
            Redshifted and scaled spectrum.
            
        """
        try:
            import eazy.igm
            igm = eazy.igm.Inoue14()
            igmz = igm.full_IGM(z, self.wave*(1+z))
        except:
            igmz = 1.
            
        return SpectrumTemplate(wave=self.wave*(1+z),
                                flux=self.flux*scalar/(1+z)*igmz)
    
    def __add__(self, spectrum):
        """Add two templates together
        
        The new wavelength array is the union of both input spectra and each
        input spectrum is linearly interpolated to the final grid.
        
        Parameters
        ----------
        spectrum : `~grizli.utils.SpectrumTemplate`
        
        Returns
        -------
        new_spectrum : `~grizli.utils.SpectrumTemplate`
        """
        new_wave = np.unique(np.append(self.wave, spectrum.wave))
        new_wave.sort()
        
        new_flux = np.interp(new_wave, self.wave, self.flux)
        new_flux += np.interp(new_wave, spectrum.wave, spectrum.flux)
        return SpectrumTemplate(wave=new_wave, flux=new_flux)
    
    def __mul__(self, scalar):
        """Multiply spectrum by a scalar value
        
        Parameters
        ----------
        scalar : float
            Factor to multipy to `self.flux`.
        
        Returns
        -------
        new_spectrum : `~grizli.utils.SpectrumTemplate`    
        """
        return SpectrumTemplate(wave=self.wave, flux=self.flux*scalar)
        
def log_zgrid(zr=[0.7,3.4], dz=0.01):
    """Make a logarithmically spaced redshift grid
    
    Parameters
    ----------
    zr : [float, float]
        Minimum and maximum of the desired grid
    
    dz : float
        Step size, dz/(1+z)
    
    Returns
    -------
    zgrid : array-like
        Redshift grid
    
    """
    zgrid = np.exp(np.arange(np.log(1+zr[0]), np.log(1+zr[1]), dz))-1
    return zgrid

### Deprecated
# def zoom_zgrid(zgrid, chi2nu, threshold=0.01, factor=10, grow=7):
#     """TBD
#     """
#     import scipy.ndimage as nd
#     
#     mask = (chi2nu-chi2nu.min()) < threshold
#     if grow > 1:
#         mask_grow = nd.maximum_filter(mask*1, size=grow)
#         mask = mask_grow > 0
#         
#     if mask.sum() == 0:
#         return []
#     
#     idx = np.arange(zgrid.shape[0])
#     out_grid = []
#     for i in idx[mask]:
#         if i == idx[-1]:
#             continue
#             
#         out_grid = np.append(out_grid, np.linspace(zgrid[i], zgrid[i+1], factor+2)[1:-1])
#     
#     return out_grid

def get_wcs_pscale(wcs):
    """Get correct pscale from a `~astropy.wcs.WCS` object
    
    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        
    Returns
    -------
    pscale : float
        Pixel scale from `wcs.cd`
        
    """
    from numpy import linalg
    det = linalg.det(wcs.wcs.cd)
    pscale = np.sqrt(np.abs(det))*3600.
    return pscale
    
def transform_wcs(in_wcs, translation=[0.,0.], rotation=0., scale=1.):
    """Update WCS with shift, rotation, & scale
    
    Paramters
    ---------
    in_wcs: `~astropy.wcs.WCS`
        Input WCS
        
    translation: [float, float]
        xshift & yshift in pixels
    
    rotation: float
        CCW rotation (towards East), radians
    
    scale: float
        Pixel scale factor
    
    Returns
    -------
    out_wcs: `~astropy.wcs.WCS`
        Modified WCS
    """
    out_wcs = in_wcs.deepcopy()
    out_wcs.wcs.crpix += np.array(translation)
    theta = -rotation
    _mat = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    
    out_wcs.wcs.cd = np.dot(out_wcs.wcs.cd, _mat)/scale
    out_wcs.pscale = get_wcs_pscale(out_wcs)
    out_wcs.wcs.crpix *= scale
    if hasattr(out_wcs, '_naxis1'):
        out_wcs._naxis1 = int(np.round(out_wcs._naxis1*scale))
        out_wcs._naxis2 = int(np.round(out_wcs._naxis2*scale))
        
    return out_wcs
    
def get_wcs_slice_header(wcs, slx, sly):
    """TBD
    """
    #slx, sly = slice(1279, 1445), slice(2665,2813)
    h = wcs.slice((sly, slx)).to_header(relax=True)
    h['NAXIS'] = 2
    h['NAXIS1'] = slx.stop-slx.start
    h['NAXIS2'] = sly.stop-sly.start
    for k in h:
        if k.startswith('PC'):
            h.rename_keyword(k, k.replace('PC', 'CD'))
    
    return h
    
def reproject_faster(input_hdu, output, pad=10, **kwargs):
    """Speed up `reproject` module with array slices of the input image
    
    Parameters
    ----------
    input_hdu : `~astropy.io.fits.ImageHDU`
        Input image HDU to reproject.  
        
    output : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        Output frame definition.
    
    pad : int
        Pixel padding on slices cut from the `input_hdu`.
        
    kwargs : dict
        Arguments passed through to `~reproject.reproject_interp`.  For 
        example, `order='nearest-neighbor'`.
    
    Returns
    -------
    reprojected : `~numpy.ndarray`
        Reprojected data from `input_hdu`.
        
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output frame.
    
    .. note::
    
    `reproject' is an astropy-compatible module that can be installed with 
    `pip`.  See https://reproject.readthedocs.io.
     
    """
    import reproject
    
    # Output WCS
    if isinstance(output, pywcs.WCS):
        out_wcs = output
    else:
        out_wcs = pywcs.WCS(output, relax=True)
    
    if 'SIP' in out_wcs.wcs.ctype[0]:
        print('Warning: `reproject` doesn\'t appear to support SIP projection')
        
    # Compute pixel coordinates of the output frame corners in the input image
    input_wcs = pywcs.WCS(input_hdu.header, relax=True)
    out_fp = out_wcs.calc_footprint()
    input_xy = input_wcs.all_world2pix(out_fp, 0)
    slx = slice(int(input_xy[:,0].min())-pad, int(input_xy[:,0].max())+pad)
    sly = slice(int(input_xy[:,1].min())-pad, int(input_xy[:,1].max())+pad)
    
    # Make the cutout
    sub_data = input_hdu.data[sly, slx]
    sub_header = get_wcs_slice_header(input_wcs, slx, sly)
    sub_hdu = pyfits.PrimaryHDU(data=sub_data, header=sub_header)
    
    # Get the reprojection
    seg_i, fp_i = reproject.reproject_interp(sub_hdu, output, **kwargs)
    return seg_i.astype(sub_data.dtype), fp_i.astype(np.uint8)
     
def make_spectrum_wcsheader(center_wave=1.4e4, dlam=40, NX=100, spatial_scale=1, NY=10):
    """Make a WCS header for a 2D spectrum
    
    Parameters
    ----------
    center_wave : float
        Wavelength of the central pixel, in Anstroms
        
    dlam : float
        Delta-wavelength per (x) pixel
        
    NX, NY : int
        Number of x & y pixels. Output will have shape `(2*NY, 2*NX)`.
        
    spatial_scale : float
        Spatial scale of the output, in units of the input pixels
    
    Returns
    -------
    header : `~astropy.io.fits.Header`
        Output WCS header
    
    wcs : `~astropy.wcs.WCS`
        Output WCS
    
    Examples
    --------
        
        >>> from grizli.utils import make_spectrum_wcsheader
        >>> h, wcs = make_spectrum_wcsheader()
        >>> print(wcs)
        WCS Keywords
        Number of WCS axes: 2
        CTYPE : 'WAVE'  'LINEAR'  
        CRVAL : 14000.0  0.0  
        CRPIX : 101.0  11.0  
        CD1_1 CD1_2  : 40.0  0.0  
        CD2_1 CD2_2  : 0.0  1.0  
        NAXIS    : 200 20

    """
    
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

def to_header(wcs, relax=True):
    """Modify `astropy.wcs.WCS.to_header` to produce more keywords
    
    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        Input WCS.
    
    relax : bool
        Passed to `WCS.to_header(relax=)`.
        
    Returns
    -------
    header : `~astropy.io.fits.Header`
        Output header.
        
    """
    header = wcs.to_header(relax=relax)
    if hasattr(wcs, '_naxis1'):
        header['NAXIS'] = wcs.naxis
        header['NAXIS1'] = wcs._naxis1
        header['NAXIS2'] = wcs._naxis2
    
    for k in header:
        if k.startswith('PC'):
            cd = k.replace('PC','CD')
            header.rename_keyword(k, cd)
    
    return header
    
def make_wcsheader(ra=40.07293, dec=-1.6137748, size=2, pixscale=0.1, get_hdu=False, theta=0):
    """Make a celestial WCS header
        
    Parameters
    ----------
    ra, dec : float
        Celestial coordinates in decimal degrees
        
    size, pixscale : float or 2-list
        Size of the thumbnail, in arcsec, and pixel scale, in arcsec/pixel.
        Output image will have dimensions `(npix,npix)`, where
            
            >>> npix = size/pixscale
            
    get_hdu : bool
        Return a `~astropy.io.fits.ImageHDU` rather than header/wcs.
        
    theta : float
        Position angle of the output thumbnail
    
    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU` 
        HDU with data filled with zeros if `get_hdu=True`.
    
    header, wcs : `~astropy.io.fits.Header`, `~astropy.wcs.WCS`
        Header and WCS object if `get_hdu=False`.

    Examples
    --------
    
        >>> from grizli.utils import make_wcsheader
        >>> h, wcs = make_wcsheader()
        >>> print(wcs)
        WCS Keywords
        Number of WCS axes: 2
        CTYPE : 'RA---TAN'  'DEC--TAN'  
        CRVAL : 40.072929999999999  -1.6137748000000001  
        CRPIX : 10.0  10.0  
        CD1_1 CD1_2  : -2.7777777777777e-05  0.0  
        CD2_1 CD2_2  : 0.0  2.7777777777777701e-05  
        NAXIS    : 20 20
        
        >>> from grizli.utils import make_wcsheader
        >>> hdu = make_wcsheader(get_hdu=True)
        >>> print(hdu.data.shape)
        (20, 20)
        >>> print(hdu.header.tostring)
        XTENSION= 'IMAGE   '           / Image extension                                
        BITPIX  =                  -32 / array data type                                
        NAXIS   =                    2 / number of array dimensions                     
        PCOUNT  =                    0 / number of parameters                           
        GCOUNT  =                    1 / number of groups                               
        CRPIX1  =                   10                                                  
        CRPIX2  =                   10                                                  
        CRVAL1  =             40.07293                                                  
        CRVAL2  =           -1.6137748                                                  
        CD1_1   = -2.7777777777777E-05                                                  
        CD1_2   =                  0.0                                                  
        CD2_1   =                  0.0                                                  
        CD2_2   = 2.77777777777777E-05                                                  
        NAXIS1  =                   20                                                  
        NAXIS2  =                   20                                                  
        CTYPE1  = 'RA---TAN'                                                            
        CTYPE2  = 'DEC--TAN'
    """
    
    if np.isscalar(pixscale):
        cdelt = [pixscale/3600.]*2
    else:
        cdelt = [pixscale[0]/3600., pixscale[1]/3600.]
        
    if np.isscalar(size):
        npix = np.cast[int]([size/pixscale, size/pixscale])
    else:
        npix = np.cast[int]([size[0]/pixscale, size[1]/pixscale])
        
    hout = pyfits.Header()
    hout['CRPIX1'] = npix[0]/2
    hout['CRPIX2'] = npix[1]/2
    hout['CRVAL1'] = ra
    hout['CRVAL2'] = dec
    hout['CD1_1'] = -cdelt[0]
    hout['CD1_2'] = hout['CD2_1'] = 0.
    hout['CD2_2'] = cdelt[1]
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
            hout['CD{0:d}_{1:d}'.format(i+1, j+1)] = rot_cd[i,j]
            wcs_out.wcs.cd[i,j] = rot_cd[i,j]
                
    cd = wcs_out.wcs.cd
    wcs_out.pscale = get_wcs_pscale(wcs_out) #np.sqrt((cd[0,:]**2).sum())*3600.
        
    if get_hdu:
        hdu = pyfits.ImageHDU(header=hout, data=np.zeros((npix[1], npix[0]), dtype=np.float32))
        return hdu
    else:
        return hout, wcs_out
    
def fetch_hst_calib(file='iref$uc72113oi_pfl.fits',  ftpdir='https://hst-crds.stsci.edu/unchecked_get/references/hst/', verbose=True):
    """
    TBD
    """
    import os
    
    ref_dir = file.split('$')[0]
    cimg = file.split('{0}$'.format(ref_dir))[1]
    iref_file = os.path.join(os.getenv(ref_dir), cimg)
    if not os.path.exists(iref_file):
        os.system('curl -o {0} {1}/{2}'.format(iref_file, ftpdir, cimg))
    else:
        if verbose:
            print('{0} exists'.format(iref_file))
        
def fetch_hst_calibs(flt_file, ftpdir='https://hst-crds.stsci.edu/unchecked_get/references/hst/', calib_types=['BPIXTAB', 'CCDTAB', 'OSCNTAB', 'CRREJTAB', 'DARKFILE', 'NLINFILE', 'PFLTFILE', 'IMPHTTAB', 'IDCTAB', 'NPOLFILE'], verbose=True):
    """
    TBD
    Fetch necessary calibration files needed for running calwf3 from STScI FTP
    
    Old FTP dir: ftp://ftp.stsci.edu/cdbs/iref/"""
    import os
            
    im = pyfits.open(flt_file)
    if im[0].header['INSTRUME'] == 'ACS':
        ref_dir = 'jref'
    
    if im[0].header['INSTRUME'] == 'WFC3':
        ref_dir = 'iref'
    
    if not os.getenv(ref_dir):
        print('No ${0} set!  Put it in ~/.bashrc or ~/.cshrc.'.format(ref_dir))
        return False
    
    for ctype in calib_types:
        if ctype not in im[0].header:
            continue
            
        if verbose:
            print('Calib: {0}={1}'.format(ctype, im[0].header[ctype]))
        
        if im[0].header[ctype] == 'N/A':
            continue
        
        fetch_hst_calib(im[0].header[ctype], ftpdir=ftpdir, verbose=verbose)
            
    return True
    
def fetch_default_calibs(ACS=False):
    
    for ref_dir in ['iref','jref']:
        if not os.getenv(ref_dir):
            print("""
No ${0} set!  Make a directory and point to it in ~/.bashrc or ~/.cshrc.
For example,

  $ mkdir $GRIZLI/{0}
  $ export {0}="${GRIZLI}/{0}/" # put this in ~/.bashrc
""".format(ref_dir))

            return False
        
    ### WFC3
    files = ['iref$uc72113oi_pfl.fits', #F105W Flat
             'iref$uc721143i_pfl.fits', #F140W flat
             'iref$u4m1335li_pfl.fits', #G102 flat
             'iref$u4m1335mi_pfl.fits', #G141 flat
             'iref$w3m18525i_idc.fits', #IDCTAB distortion table}
             ]
    
    if ACS:
        files.extend(['jref$n6u12592j_pfl.fits',#F814 Flat
                      'jref$o841350mj_pfl.fits', #G800L flat])
                      ])
    
    for file in files:
        fetch_hst_calib(file)
    
    badpix = '{0}/badpix_spars200_Nov9.fits'.format(os.getenv('iref'))
    print('Extra WFC3/IR bad pixels: {0}'.format(badpix))
    if not os.path.exists(badpix):
        os.system('curl -o {0}/badpix_spars200_Nov9.fits https://raw.githubusercontent.com/gbrammer/wfc3/master/data/badpix_spars200_Nov9.fits'.format(os.getenv('iref')))
    
def fetch_config_files(ACS=False):
    """
    Config files needed for Grizli
    """
    cwd = os.getcwd()
    
    print('Config directory: {0}/CONF'.format(os.getenv('GRIZLI')))
    
    os.chdir('{0}/CONF'.format(os.getenv('GRIZLI')))
    
    tarfiles = ['ftp://ftp.stsci.edu/cdbs/wfc3_aux/WFC3.IR.G102.cal.V4.32.tar.gz',
 'ftp://ftp.stsci.edu/cdbs/wfc3_aux/WFC3.IR.G141.cal.V4.32.tar.gz',
 'ftp://ftp.stsci.edu/cdbs/wfc3_aux/grism_master_sky_v0.5.tar.gz']
    
    if ACS:
        tarfiles.append('http://www.stsci.edu/~brammer/Grizli/Files/' + 
                        'ACS.WFC.sky.tar.gz')

        tarfiles.append('http://www.stsci.edu/~brammer/Grizli/Files/' + 
                        'ACS_CONFIG.tar.gz')
                        
    for url in tarfiles:
        file=os.path.basename(url)
        if not os.path.exists(file):
            print('Get {0}'.format(file))
            os.system('curl -o {0} {1}'.format(file, url))
        
        os.system('tar xzvf {0}'.format(file))
    
    # ePSF files for fitting point sources
    files = ['http://www.stsci.edu/hst/wfc3/analysis/PSF/psf_downloads/wfc3_ir/PSFSTD_WFC3IR_{0}.fits'.format(filter) for filter in ['F105W', 'F125W', 'F140W', 'F160W']]
    for url in files:
        file=os.path.basename(url)
        if not os.path.exists(file):
            print('Get {0}'.format(file))
            os.system('curl -o {0} {1}'.format(file, url))
        else:
            print('File {0} exists'.format(file))
    
    # Stellar templates
    print('Templates directory: {0}/templates'.format(os.getenv('GRIZLI')))
    os.chdir('{0}/templates'.format(os.getenv('GRIZLI')))
    
    files = ['http://www.stsci.edu/~brammer/Grizli/Files/stars_pickles.npy',
             'http://www.stsci.edu/~brammer/Grizli/Files/stars_bpgs.npy']
            
    for url in files:
        file=os.path.basename(url)
        if not os.path.exists(file):
            print('Get {0}'.format(file))
            os.system('curl -o {0} {1}'.format(file, url))
        else:
            print('File {0} exists'.format(file))
    
    print('ln -s stars_pickles.npy stars.npy')
    os.system('ln -s stars_pickles.npy stars.npy')
    
    os.chdir(cwd)
      
class EffectivePSF(object):
    def __init__(self):
        """Tools for handling WFC3/IR Effective PSF

        See documentation at http://www.stsci.edu/hst/wfc3/analysis/PSF.
        
        PSF files stored in $GRIZLI/CONF/
        
        Attributes
        ----------
        
        Methods
        -------
        
        """
        
        self.load_PSF_data()
        
    def load_PSF_data(self):
        """Load data from PSFSTD files
        
        Files should be located in ${GRIZLI}/CONF/ directory.
        """
        self.epsf = {}
        for filter in ['F105W', 'F125W', 'F140W', 'F160W']:
            file = os.path.join(os.getenv('GRIZLI'), 'CONF',
                                'PSFSTD_WFC3IR_{0}.fits'.format(filter))
            
            data = pyfits.open(file)[0].data.T
            data[data < 0] = 0 
            
            self.epsf[filter] = data
        
    def get_at_position(self, x=507, y=507, filter='F140W'):
        """Evaluate ePSF at detector coordinates
        TBD
        """
        epsf = self.epsf[filter]

        rx = 1+(x-0)/507.
        ry = 1+(y-0)/507.
                
        # zero index
        rx -= 1
        ry -= 1 

        nx = np.clip(int(rx), 0, 2)
        ny = np.clip(int(ry), 0, 2)

        # print x, y, rx, ry, nx, ny

        fx = rx-nx
        fy = ry-ny

        psf_xy = (1-fx)*(1-fy)*epsf[:, :, nx+ny*3]
        psf_xy += fx*(1-fy)*epsf[:, :, (nx+1)+ny*3]
        psf_xy += (1-fx)*fy*epsf[:, :, nx+(ny+1)*3]
        psf_xy += fx*fy*epsf[:, :, (nx+1)+(ny+1)*3]

        return psf_xy
    
    def eval_ePSF(self, psf_xy, dx, dy):
        """Evaluate PSF at dx,dy coordinates
        
        TBD
        """
        # So much faster than scipy.interpolate.griddata!
        from scipy.ndimage.interpolation import map_coordinates
        
        # ePSF only defined to 12.5 pixels
        ok = (np.abs(dx) < 12.5) & (np.abs(dy) < 12.5)
        coords = np.array([50+4*dx[ok], 50+4*dy[ok]])
        
        # Do the interpolation
        interp_map = map_coordinates(psf_xy, coords, order=3)
        
        # Fill output data
        out = np.zeros_like(dx, dtype=np.float32)
        out[ok] = interp_map
        return out
    
    @staticmethod
    def objective_epsf(params, self, psf_xy, sci, ivar, xp, yp):
        """Objective function for fitting ePSFs
        
        TBD
        
        params = [normalization, xc, yc, background]
        """
        dx = xp-params[1]
        dy = yp-params[2]

        ddx = xp-xp.min()
        ddy = yp-yp.min()

        psf_offset = self.eval_ePSF(psf_xy, dx, dy)*params[0] + params[3] + params[4]*ddx + params[5]*ddy + params[6]*ddx*ddy
        
        chi2 = np.sum((sci-psf_offset)**2*ivar)
        #print params, chi2
        return chi2
    
    def fit_ePSF(self, sci, center=None, origin=[0,0], ivar=1, N=7, 
                 filter='F140W', tol=1.e-4):
        """Fit ePSF to input data
        TBD
        """
        from scipy.optimize import minimize
        
        sh = sci.shape
        if center is None:
            y0, x0 = np.array(sh)/2.
        else:
            x0, y0 = center
        
        xd = x0+origin[1]
        yd = y0+origin[0]
        
        xc, yc = int(x0), int(y0)
        
        psf_xy = self.get_at_position(x=xd, y=yd, filter=filter)
        
        yp, xp = np.indices(sh)
        args = (self, psf_xy, sci[yc-N:yc+N, xc-N:xc+N], ivar[yc-N:yc+N, xc-N:xc+N], xp[yc-N:yc+N, xc-N:xc+N], yp[yc-N:yc+N, xc-N:xc+N])
        guess = [sci[yc-N:yc+N, xc-N:xc+N].sum()/psf_xy.sum(), x0, y0, 0, 0, 0, 0]
        
        out = minimize(self.objective_epsf, guess, args=args, method='Powell',
                       tol=tol)
        
        params = out.x
        dx = xp-params[1]
        dy = yp-params[2]
        output_psf = self.eval_ePSF(psf_xy, dx, dy)*params[0]
        
        return output_psf, params
    
class GTable(astropy.table.Table):
    """
    Extend `~astropy.table.Table` class with more automatic IO and other
    helper methods.
    """ 
    @classmethod
    def gread(cls, file, sextractor=False, format=None):
        """Assume `ascii.commented_header` by default
        
        Parameters
        ----------
        sextractor : bool
            Use `format='ascii.sextractor'`.
        
        format : None or str
            Override format passed to `~astropy.table.Table.read`.
        
        Returns
        -------
        tab : `~astropy.table.Table`
            Table object
        """
        import astropy.units as u
        
        if format is None:
            if sextractor:
                format = 'ascii.sextractor'
            else:
                format = 'ascii.commented_header'
        
        #print(file, format)            
        tab = cls.read(file, format=format)
        
        return tab
    
    def gwrite(self, output, format='ascii.commented_header'):
        """Assume a format for the output table
        
        Parameters
        ----------
        output : str
            Output filename
            
        format : str
            Format string passed to `~astropy.table.Table.write`.
        
        """
        self.write(output, format=format)
    
    @staticmethod
    def parse_radec_columns(self, rd_pairs=None):
        """Parse column names for RA/Dec and set to `~astropy.units.degree` units if not already set
        
        Parameters
        ----------
        rd_pairs : `~collections.OrderedDict` or None
            Pairs of {ra:dec} names to search in the column list. If None,
            then uses the following by default. 
            
                >>> rd_pairs = OrderedDict()
                >>> rd_pairs['ra'] = 'dec'
                >>> rd_pairs['ALPHA_J2000'] = 'DELTA_J2000'
                >>> rd_pairs['X_WORLD'] = 'Y_WORLD'
            
            NB: search is performed in order of ``rd_pairs.keys()`` and stops
            if/when a match is found.
            
        Returns
        -------
        rd_pair : [str, str]
            Column names associated with RA/Dec.  Returns False if no column
            pairs found based on `rd_pairs`.
            
        """
        from collections import OrderedDict
        import astropy.units as u
        
        if rd_pairs is None:
            rd_pairs = OrderedDict()
            rd_pairs['ra'] = 'dec'
            rd_pairs['ALPHA_J2000'] = 'DELTA_J2000'
            rd_pairs['X_WORLD'] = 'Y_WORLD'
            rd_pairs['ALPHA_SKY'] = 'DELTA_SKY'
               
        rd_pair = None 
        for c in rd_pairs:
            if c.upper() in [col.upper() for col in self.colnames]:
                rd_pair = [c, rd_pairs[c]]
                break
        
        if rd_pair is None:
            #print('No RA/Dec. columns found in input table.')
            return False
        
        for c in rd_pair:
            if self[c].unit is None:
                self[c].unit = u.degree
        
        return rd_pair
       
    def match_to_catalog_sky(self, other, self_radec=None, other_radec=None):
        """Compute `~astropy.coordinates.SkyCoord` projected matches between two `GTable` tables.
        
        Parameters
        ----------
        other : `~astropy.table.Table` or `GTable`
            Other table to match positions from.
        
        self_radec, other_radec : None or [str, str]
            Column names for RA and Dec.  If None, then try the following
            pairs (in this order): 
            
                >>> rd_pairs = OrderedDict()
                >>> rd_pairs['ra'] = 'dec'
                >>> rd_pairs['ALPHA_J2000'] = 'DELTA_J2000'
                >>> rd_pairs['X_WORLD'] = 'Y_WORLD'
        
        Returns
        -------
        idx : int array
            Indices of the matches as in 
            
                >>> matched = self[idx]
                >>> len(matched) == len(other)
        
        dr : float array
            Projected separation of closest match.
            
        Example
        -------
                
                >>> import astropy.units as u

                >>> ref = GTable.gread('input.cat')
                >>> gaia = GTable.gread('gaia.cat')
                >>> idx, dr = ref.match_to_catalog_sky(gaia)
                >>> close = dr < 1*u.arcsec

                >>> ref_match = ref[idx][close]
                >>> gaia_match = gaia[close]
        
        """
        from astropy.coordinates import SkyCoord
        
        if self_radec is None:
            rd = self.parse_radec_columns(self)
        else:
            rd = self.parse_radec_columns(self, rd_pairs={self_radec[0]:self_radec[1]})
            
        if rd is False:
            print('No RA/Dec. columns found in input table.')
            return False
            
        self_coo = SkyCoord(ra=self[rd[0]], dec=self[rd[1]])

        if other_radec is None:
            rd = self.parse_radec_columns(other)
        else:
            rd = self.parse_radec_columns(other, rd_pairs={other_radec[0]:other_radec[1]})

        if rd is False:
            print('No RA/Dec. columns found in `other` table.')
            return False
            
        other_coo = SkyCoord(ra=other[rd[0]], dec=other[rd[1]])
                     
        idx, d2d, d3d = other_coo.match_to_catalog_sky(self_coo)
        return idx, d2d
            
    def write_sortable_html(self, output, replace_braces=True, localhost=True, max_lines=50, table_id=None, table_class="display compact", css=None):
        """Wrapper around `~astropy.table.Table.write(format='jsviewer')`.
        
        Parameters
        ----------
        output : str
            Output filename.
            
        replace_braces : bool
            Replace '&lt;' and '&gt;' characters that are converted 
            automatically from "<>" by the `~astropy.table.Table.write`
            method. There are parameters for doing this automatically with 
            `write(format='html')` but that don't appear to be available with 
            `write(format='jsviewer')`.
            
        localhost : bool
            Use local JS files. Otherwise use files hosted externally.
            
        etc : ...
            Additional parameters passed through to `write`.
        """
        #from astropy.table.jsviewer import DEFAULT_CSS
        DEFAULT_CSS = """
body {font-family: sans-serif;}
table.dataTable {width: auto !important; margin: 0 !important;}
.dataTables_filter, .dataTables_paginate {float: left !important; margin-left:1em}
td {font-size: 10pt;}
        """
        if css is not None:
            DEFAULT_CSS += css

        self.write(output, format='jsviewer', css=DEFAULT_CSS,
                            max_lines=max_lines,
                            jskwargs={'use_local_files':localhost},
                            table_id=None, table_class=table_class)

        if replace_braces:
            lines = open(output).readlines()
            if replace_braces:
                for i in range(len(lines)):
                    lines[i] = lines[i].replace('&lt;', '<')
                    lines[i] = lines[i].replace('&gt;', '>')

            fp = open(output, 'w')
            fp.writelines(lines)
            fp.close()
    
def column_values_in_list(col, test_list):
    """Test if column elements "in" an iterable (e.g., a list of strings)
    
    Parameters
    ----------
    col : `astropy.table.Column` or other iterable
        Group of entries to test
        
    test_list : iterable
        List of values to search 
    
    Returns
    -------
    test : bool array
        Simple test:
            >>> [c_i in test_list for c_i in col]
    """
    test = np.array([c_i in test_list for c_i in col])
    return test
    
def fill_between_steps(x, y0, y1, ax=None, *args, **kwargs):
    """
    Make `fill_between` work like linestyle='steps-mid'.
    """
    so = np.argsort(x)
    mid = x[so][:-1] + np.diff(x[so])/2.
    xfull = np.append(np.append(x, mid), mid+np.diff(x[so])/1.e6)
    y0full = np.append(np.append(y0, y0[:-1]), y0[1:])
    y1full = np.append(np.append(y1, y1[:-1]), y1[1:])
    
    so = np.argsort(xfull)
    if ax is None:
        ax = plt.gca()
    
    ax.fill_between(xfull[so], y0full[so], y1full[so], *args, **kwargs)
    