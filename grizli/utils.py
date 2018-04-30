"""General utilities"""
import os
import glob
from collections import OrderedDict

import warnings

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.table

import numpy as np

import astropy.units as u

KMS = u.km/u.s
FLAMBDA_CGS = u.erg/u.s/u.cm**2/u.angstrom
FNU_CGS = u.erg/u.s/u.cm**2/u.Hz

# character to skip clearing line on STDOUT printing
NO_NEWLINE = '\x1b[1A\x1b[1M' 

# R_V for Galactic extinction
MW_RV = 3.1

GRISM_COLORS = {'G800L':(0.0, 0.4470588235294118, 0.6980392156862745),
      'G102':(0.0, 0.6196078431372549, 0.45098039215686275),
      'G141':(0.8352941176470589, 0.3686274509803922, 0.0),
      'none':(0.8, 0.4745098039215686, 0.6549019607843137),
      'GRISM':'k',
      'F277W':(0.0, 0.6196078431372549, 0.45098039215686275),
      'F356W':(0.8352941176470589, 0.3686274509803922, 0.0),
      'F444W':(0.8, 0.4745098039215686, 0.6549019607843137),
      'F410M':(0.0, 0.4470588235294118, 0.6980392156862745),
      'G280':'purple',
      'F090W':(0.0, 0.4470588235294118, 0.6980392156862745),
      'F115W':(0.0, 0.6196078431372549, 0.45098039215686275),
      'F150W':(0.8352941176470589, 0.3686274509803922, 0.0),
      'F140M':(0.8352941176470589, 0.3686274509803922, 0.0),
      'F158M':(0.8352941176470589, 0.3686274509803922, 0.0),
      'F200W':(0.8, 0.4745098039215686, 0.6549019607843137),
      'F140M':'orange',
      'BLUE':'#1f77b4', # Euclid
      'RED':'#d62728',
      'CLEARP':'b'}

GRISM_MAJOR = {'G102':0.1, 'G141':0.1, 'G800L':0.1, 'F090W':0.1, 'F115W':0.1, 'F150W':0.1, 'F140M':0.1, 'F158M':0.1, 'F200W':0.1, 'F277W':0.2, 'F356W':0.2, 'F444W':0.2, 'F410M':0.2, 'BLUE':0.1, 'RED':0.1}

GRISM_LIMITS = {'G800L':[0.545, 1.02, 40.], # ACS/WFC
          'G280':[0.2,0.4, 14], # WFC3/UVIS
           'G102':[0.77, 1.18, 23.], # WFC3/IR
           'G141':[1.06, 1.73, 46.0],
           'GRISM':[0.98, 1.98, 11.], # WFIRST
           'F090W':[0.76,1.04, 45.0], # NIRISS
           'F115W':[0.97,1.32, 45.0],
           'F140M':[1.28,1.52, 45.0],
           'F158M':[1.28,1.72, 45.0],
           'F150W':[1.28,1.72, 45.0],
           'F200W':[1.68,2.30, 45.0],
           'F140M':[1.20,1.60, 45.0],
           'CLEARP':[0.76, 2.3,45.0],
           'F277W':[2.5, 3.2, 20.], # NIRCAM
           'F356W':[3.05, 4.1, 20.],
           'F444W':[3.75, 5.05, 20],
           'F410M':[3.7, 4.5, 20],
           'BLUE':[0.8, 1.2, 10.], # Euclid
           'RED':[1.1, 1.9, 14.]}

DEFAULT_LINE_LIST = ['PaB', 'HeI-1083', 'SIII', 'SII', 'Ha', 'OI-6302', 'OIII', 'Hb', 'OIII-4363', 'Hg', 'Hd', 'NeIII-3867', 'OII', 'NeVI-3426', 'NeV-3346', 'MgII','CIV-1549', 'CIII-1908', 'OIII-1663', 'HeII-1640', 'NIII-1750', 'NIV-1487', 'NV-1240', 'Lya']

def set_warnings(numpy_level='ignore', astropy_level='ignore'):
    """
    Set global numpy and astropy warnings
    
    Parameters
    ----------
    numpy_level : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        Numpy error level (see `~numpy.seterr`).
        
    astropy_level : {'error', 'ignore', 'always', 'default', 'module', 'once'}
        Astropy error level (see `~warnings.simplefilter`).
    
    """
    from astropy.utils.exceptions import AstropyWarning
    
    np.seterr(all=numpy_level)
    warnings.simplefilter(astropy_level, category=AstropyWarning)
    
def get_flt_info(files=[], columns=['FILE', 'FILTER', 'INSTRUME', 'DETECTOR', 'TARGNAME', 'DATE-OBS', 'TIME-OBS', 'EXPSTART', 'EXPTIME', 'PA_V3', 'RA_TARG', 'DEC_TARG', 'POSTARG1', 'POSTARG2']):
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

def get_visit_footprints(visits):
    """
    Add `~shapely.geometry.Polygon` 'footprint' attributes to visit dict.
        
    Parameters
    ----------
    visits : list
        List of visit dictionaries.
    
    """
    
    import os
    
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    
    from shapely.geometry import Polygon
    
    N = len(visits)
    for i in range(N):
        for j in range(len(visits[i]['files'])):
            flt_file = visits[i]['files'][j]
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
        
        visits[i]['footprint'] = fp_i
    
    return visits
    
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

DIRECT_ORDER = {'G102': ['F105W', 'F110W', 'F098M', 'F125W', 'F140W', 'F160W', 'F127M', 'F139M', 'F153M', 'F132N', 'F130N', 'F128N', 'F126N', 'F164N', 'F167N'],
                'G141': ['F140W', 'F160W', 'F125W', 'F105W', 'F110W', 'F098M', 'F127M', 'F139M', 'F153M', 'F132N', 'F130N', 'F128N', 'F126N', 'F164N', 'F167N'],
                'G800L': ['F814W', 'F606W', 'F850LP', 'F435W', 'F105W', 'F110W', 'F098M', 'F125W', 'F140W', 'F160W', 'F127M', 'F139M', 'F153M', 'F132N', 'F130N', 'F128N', 'F126N', 'F164N', 'F167N']}
                
def parse_grism_associations(exposure_groups, 
                             best_direct=DIRECT_ORDER,
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
        d_idx = 10
        for j in range(N):
            f_j = exposure_groups[j]['product'].split('-')[-1]
            if f_j.startswith('g'):
                continue
                 
            fp_j = exposure_groups[j]['footprint']
            olap = fp_i.intersection(fp_j)
            root_j = exposure_groups[j]['product'].split('-'+f_j)[0]

            #print(root_j, root_i, root_j == root_i)
            if (root_j == root_i):
                # if (group['direct'] is not None):
                #     pass
                #     if (group['direct']['product'].startswith(root_i)) & (d_i.upper() == best_direct[f_i.upper()]):
                #         continue
                
                if f_j.upper() not in best_direct[f_i.upper()]:
                    #print(f_j.upper())
                    continue
                    
                if best_direct[f_i.upper()].index(f_j.upper()) < d_idx:
                    d_idx = best_direct[f_i.upper()].index(f_j.upper())
                    group['direct'] = exposure_groups[j]
                    olap_i = olap.area
                    d_i = f_j
                #print(0,group['grism']['product'], group['direct']['product'])
            #     continue
                
            #print(exposure_groups[i]['product'], exposure_groups[j]['product'], olap.area*3600.)
            
            # #print(exposure_groups[j]['product'], olap_i, olap.area)
            # if olap.area > 0:
            #     if group['direct'] is None:
            #         group['direct'] = exposure_groups[j]
            #         olap_i = olap.area
            #         d_i = f_j
            #         #print(1,group['grism']['product'], group['direct']['product'])
            #     else:
            #         #if (f_j.upper() == best_direct[f_i.upper()]):
            #         if get_max_overlap:
            #             if olap.area < olap_i:
            #                 continue
            #             
            #             if d_i.upper() == best_direct[f_i.upper()]:
            #                 continue
            #                     
            #         group['direct'] = exposure_groups[j]
            #         #print(exposure_groups[j]['product'])
            #         olap_i = olap.area
            #         d_i = f_j
            #         #print(2,group['grism']['product'], group['direct']['product'])
                    
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
    if 'FILTER' in header:
        return header['FILTER'].upper()
        
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
        raise KeyError ('Filter keyword not found for instrument {0}'.format(header['INSTRUME']))
    
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
        print(NO_NEWLINE + ('{0}: photutils.source_properties - {1:d} objects'.format(root, len(catalog))))
    
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
    
    line_wavelengths['PaB'] = [12821]
    line_ratios['PaB'] = [1.]
    line_wavelengths['Ha'] = [6564.61]
    line_ratios['Ha'] = [1.]
    line_wavelengths['Hb'] = [4862.68]
    line_ratios['Hb'] = [1.]
    line_wavelengths['Hg'] = [4341.68]
    line_ratios['Hg'] = [1.]
    line_wavelengths['Hd'] = [4102.892]
    line_ratios['Hd'] = [1.]

    # Groves et al. 2011, Table 1
    line_wavelengths['Balmer 10kK'] = [6564.61, 4862.68, 4341.68, 4101.73]
    line_ratios['Balmer 10kK'] = [2.86, 1.0, 0.468, 0.259]
    
    # Reddened with Kriek & Conroy dust, tau_V=0.5
    line_wavelengths['Balmer 10kK t0.5'] = [6564.61, 4862.68, 4341.68, 4101.73]
    line_ratios['Balmer 10kK t0.5'] = [2.86*0.68, 1.0*0.55, 0.468*0.51, 0.259*0.48]
    
    # Reddened with Kriek & Conroy dust, tau_V=1
    line_wavelengths['Balmer 10kK t1'] = [6564.61, 4862.68, 4341.68, 4101.73]
    line_ratios['Balmer 10kK t1'] = [2.86*0.46, 1.0*0.31, 0.468*0.256, 0.259*0.232]
    
    line_wavelengths['OIII-4363'] = [4364.436]
    line_ratios['OIII-4363'] = [1.]
    line_wavelengths['OIII'] = [5008.240, 4960.295]
    line_ratios['OIII'] = [2.98, 1]
    
    # Split doublet, if needed
    line_wavelengths['OIII-4959'] = [4960.295]
    line_ratios['OIII-4959'] = [1]
    line_wavelengths['OIII-5007'] = [5008.240]
    line_ratios['OIII-5007'] = [1]
    
    line_wavelengths['OII'] = [3727.092, 3729.875]
    line_ratios['OII'] = [1, 1.] 
    
    line_wavelengths['OI-6302'] = [6302.046, 6363.67]
    line_ratios['OI-6302'] = [1, 0.33]
    line_wavelengths['OI-5578'] = [5578.6]
    line_ratios['OI-5578'] = [1]

    line_wavelengths['NeIII-3867'] = [3867.5]
    line_ratios['NeIII-3867'] = [1.]
    line_wavelengths['NeV-3346'] = [3346.8]
    line_ratios['NeV-3346'] = [1.]
    line_wavelengths['NeVI-3426'] = [3426.85]
    line_ratios['NeVI-3426'] = [1.]
    
    line_wavelengths['SIII'] = [9068.6, 9530.6][::-1]
    line_ratios['SIII'] = [1, 2.44][::-1]
    
    # Split doublet, if needed
    line_wavelengths['SIII-9068'] = [9068.6]
    line_ratios['SIII-9068'] = [1]
    line_wavelengths['SIII-9531'] = [9530.6]
    line_ratios['SIII-9531'] = [1]
    
    line_wavelengths['SII'] = [6718.29, 6732.67]
    line_ratios['SII'] = [1., 1.]   
    
    line_wavelengths['HeII-4687'] = [4687.5]
    line_ratios['HeII-4697'] = [1.]
    line_wavelengths['HeII-5412'] = [5412.5]
    line_ratios['HeII-5410'] = [1.]
    line_wavelengths['HeI-5877'] = [5877.2]
    line_ratios['HeI-5877'] = [1.]
    line_wavelengths['HeI-3889'] = [3889.5]
    line_ratios['HeI-3889'] = [1.]
    line_wavelengths['HeI-1083'] = [10830.]
    line_ratios['HeI-1083'] = [1.]
    
    line_wavelengths['MgII'] = [2799.117]
    line_ratios['MgII'] = [1.]
    
    line_wavelengths['CIV-1549'] = [1549.480]
    line_ratios['CIV-1549'] = [1.]
    line_wavelengths['CIII-1908'] = [1908.734]
    line_ratios['CIII-1908'] = [1.]
    line_wavelengths['OIII-1663'] = [1665.85]
    line_ratios['OIII-1663'] = [1.]
    line_wavelengths['HeII-1640'] = [1640.4]
    line_ratios['HeII-1640'] = [1.]
    
    line_wavelengths['NII'] = [6549.86, 6585.27]
    line_ratios['NII'] = [1., 3]
    line_wavelengths['NIII-1750'] = [1750.]
    line_ratios['NIII-1750'] = [1.]
    line_wavelengths['NIV-1487'] = [1487.]
    line_ratios['NIV-1487'] = [1.]
    line_wavelengths['NV-1240'] = [1240.81]
    line_ratios['NV-1240'] = [1.]

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
    
    line_wavelengths['OIII+Hb'] = [5008.240, 4960.295, 4862.68]
    line_ratios['OIII+Hb'] = [2.98, 1, 3.98/6.]
    
    line_wavelengths['OIII+Hb+Ha'] = [5008.240, 4960.295, 4862.68, 6564.61]
    line_ratios['OIII+Hb+Ha'] = [2.98, 1, 3.98/10., 3.98/10.*2.86]

    line_wavelengths['OIII+Hb+Ha+SII'] = [5008.240, 4960.295, 4862.68, 6564.61, 6718.29, 6732.67]
    line_ratios['OIII+Hb+Ha+SII'] = [2.98, 1, 3.98/10., 3.98/10.*2.86*4, 3.98/10.*2.86/10.*4, 3.98/10.*2.86/10.*4]

    line_wavelengths['OIII+OII'] = [5008.240, 4960.295, 3729.875]
    line_ratios['OIII+OII'] = [2.98, 1, 3.98/4.]
    
    line_wavelengths['OII+Ne'] = [3729.875, 3869]
    line_ratios['OII+Ne'] = [1, 1./5]
    
    return line_wavelengths, line_ratios 
    
class SpectrumTemplate(object):
    def __init__(self, wave=None, flux=None, central_wave=None, fwhm=None, velocity=False, fluxunits=FLAMBDA_CGS, waveunits=u.angstrom, name='', lorentz=False):
        """Container for template spectra.   
                
        Parameters
        ----------
        wave : array-like
            Wavelength 
            In `astropy.units.Angstrom`.
            
        flux : float array-like
            If float, then the integrated flux of a Gaussian line.  If 
            array, then f-lambda flux density.
        
        central_wave, fwhm : float
            Initialize the template with a Gaussian at this wavelength (in
            `astropy.units.Angstrom`.) that has an integrated flux of `flux` 
            and `fwhm` in `astropy.units.Angstrom` or `km/s` for 
            `velocity=True`.
            
        velocity : bool
            `fwhm` is a velocity in `km/s`.
            
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
        if wave is not None:
            self.wave = np.cast[np.float](wave)
            
        self.flux = flux
        if flux is not None:
            self.flux = np.cast[np.float](flux)

        self.fwhm = None
        self.velocity = None
        
        self.fluxunits = fluxunits
        self.waveunits = waveunits
        self.name = name
        
        if (central_wave is not None) & (fwhm is not None):
            self.fwhm = fwhm
            self.velocity = velocity
            
            self.wave, self.flux = self.make_gaussian(central_wave, fwhm,
                                                      wave_grid=wave,
                                                      velocity=velocity,
                                                      max_sigma=10,
                                                      lorentz=lorentz)
        
        self.fnu_units = FNU_CGS
        self.to_fnu()
        
    @staticmethod 
    def make_gaussian(central_wave, fwhm, max_sigma=5, step=0.1,
                      wave_grid=None, velocity=False, clip=1.e-5,
                      lorentz=False):
        """Make Gaussian template
        
        Parameters
        ----------
        central_wave, fwhm : None or float or array-like
            Central wavelength and FWHM of the desired Gaussian
            
        velocity : bool
            `fwhm` is a velocity.
        
        max_sigma, step : float
            Generated wavelength array is
                
                >>> rms = fwhm/2.35
                >>> xgauss = np.arange(-max_sigma, max_sigma, step)*rms+central_wave
        
        clip : float
            Clip values where the value of the gaussian function is less than 
            `clip` times its maximum (i.e., `1/sqrt(2*pi*sigma**2)`).
        
        lorentz : book
            Make a Lorentzian line instead of a Gaussian.
            
        Returns
        -------
        wave, flux : array-like
            Wavelength and flux of a Gaussian line
        """
        import astropy.constants as const
                
        if hasattr(fwhm, 'unit'):
            rms = fwhm.value/2.35
            velocity = u.physical.get_physical_type(fwhm.unit) == 'speed'
            if velocity:
                rms = central_wave*(fwhm/const.c.to(KMS)).value/2.35
            else:
                rms = fwhm.value/2.35
        else:
            if velocity:
                rms = central_wave*(fwhm/const.c.to(KMS).value)/2.35
            else:
                rms = fwhm/2.35
                
        if wave_grid is None:
            wave_grid = np.arange(-max_sigma, max_sigma, step)*rms
            wave_grid += central_wave
            wave_grid = np.hstack([91., wave_grid, 1.e8])
            
        if lorentz:
            from astropy.modeling.models import Lorentz1D
            if velocity:
                use_fwhm = central_wave*(fwhm/const.c.to(KMS).value)
            else:
                use_fwhm = fwhm
                
            lmodel = Lorentz1D(amplitude=1, x_0=central_wave, fwhm=use_fwhm)
            line = lmodel(wave_grid)
            line[0:2] = 0
            line[-2:] = 0
            line /= np.trapz(line, wave_grid)
            peak = line.max()
        else:
            # Gaussian
            line = np.exp(-(wave_grid-central_wave)**2/2/rms**2)
            peak = np.sqrt(2*np.pi*rms**2)
            line /= np.sqrt(2*np.pi*rms**2)
            
        line[line < peak*clip] = 0
        
        return wave_grid, line
        
        #self.wave = xgauss
        #self.flux = gaussian

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
        out = SpectrumTemplate(wave=new_wave, flux=new_flux)
        out.fwhm = spectrum.fwhm
        return out
    
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
        out = SpectrumTemplate(wave=self.wave, flux=self.flux*scalar)
        out.fwhm = self.fwhm
        return out
    
    def to_fnu(self, fnu_units=FNU_CGS):
        """Make fnu version of the template.
        
        Sets the `flux_fnu` attribute, assuming that the wavelength is given 
        in Angstrom and the flux is given in flambda:
        
            >>> flux_fnu = self.flux * self.wave**2 / 3.e18
            
        """
        #import astropy.constants as const
        #flux_fnu = self.flux * self.wave**2 / 3.e18
        #flux_fnu = (self.flux*self.fluxunits*(self.wave*self.waveunits)**2/const.c).to(FNU_CGS) #,     
        
        if ((FNU_CGS.__str__() == 'erg / (cm2 Hz s)') & 
            (self.fluxunits.__str__() == 'erg / (Angstrom cm2 s)')):
            # Faster
            flux_fnu = self.flux*self.wave**2/2.99792458e18*fnu_units
        else:
            # Use astropy conversion
            flux_fnu = (self.flux*self.fluxunits).to(fnu_units, equivalencies=u.spectral_density(self.wave*self.waveunits))
        
        self.fnu_units = fnu_units
        self.flux_fnu = flux_fnu.value
        
    def integrate_filter(self, filter, abmag=False):
        """Integrate the template through an `~eazy.FilterDefinition` filter
        object.
        
        Parameters
        ----------
        filter : `~pysynphot.ObsBandpass`
            Or any object that has `wave` and `throughput` attributes, with 
            the former in the same units as the input spectrum.
        
        abmag : bool
            Return AB magnitude rather than fnu flux
        
        Returns
        -------
        temp_flux : float
            
        Examples
        --------
        
        Compute the WFC3/IR F140W AB magnitude of a pure emission line at the 
        5-sigma 3D-HST line detection limit (5e-17 erg/s/cm2):
        
            >>> from grizli.utils import SpectrumTemplate
            >>> from eazy.filters import FilterDefinition
            >>> import pysynphot as S 

            >>> line = SpectrumTemplate(central_wave=1.4e4, fwhm=150.,
                                        velocity=True)*5.e-17
                                        
            >>> filter = FilterDefinition(bp=S.ObsBandpass('wfc3,ir,f140w'))
            
            >>> fnu = line.integrate_filter(filter)
            
            >>> print('AB mag = {0:.3f}'.format(-2.5*np.log10(fnu)-48.6))
            AB mag = 26.619
            
        """
        INTEGRATOR = np.trapz
        
        try:
            import grizli.utils_c
            interp = grizli.utils_c.interp.interp_conserve_c
        except ImportError:
            interp = np.interp
        
        #wz = self.wave*(1+z)
        if (filter.wave.min() > self.wave.max()) | (filter.wave.max() < self.wave.min()) | (filter.wave.min() < self.wave.min()):
            return 0.
                
        templ_filter = interp(filter.wave.astype(np.float), self.wave,
                              self.flux_fnu)
        
        if hasattr(filter, 'norm'):
            filter_norm = filter.norm
        else:
            # e.g., pysynphot bandpass
            filter_norm = INTEGRATOR(filter.throughput/filter.wave,
                                     filter.wave)
            
        # f_nu/lam dlam == f_nu d (ln nu)    
        temp_flux = INTEGRATOR(filter.throughput*templ_filter/filter.wave, filter.wave) / filter_norm
        
        if abmag:
            return -2.5*np.log10(temp_flux)-48.6
        else:
            return temp_flux

def load_templates(fwhm=400, line_complexes=True, stars=False,
                   full_line_list=None, continuum_list=None,
                   fsps_templates=False, alf_template=False, lorentz=False):
    """Generate a list of templates for fitting to the grism spectra
    
    The different sets of continuum templates are stored in 
    
        >>> temp_dir = os.path.join(os.getenv('GRIZLI'), 'templates')
        
    Parameters
    ----------
    fwhm : float
        FWHM of a Gaussian, in km/s, that is convolved with the emission
        line templates.  If too narrow, then can see pixel effects in the 
        fits as a function of redshift.
    
    line_complexes : bool
        Generate line complex templates with fixed flux ratios rather than
        individual lines. This is useful for the redshift fits where there
        would be redshift degeneracies if the line fluxes for individual
        lines were allowed to vary completely freely. See the list of
        available lines and line groups in
        `~grizli.utils.get_line_wavelengths`. Currently,
        `line_complexes=True` generates the following groups:
        
            Ha+NII+SII+SIII+He
            OIII+Hb
            OII+Ne
        
    stars : bool
        Get stellar templates rather than galaxies + lines
        
    full_line_list : None or list
        Full set of lines to try.  The default is set in the global variable
        `~grizli.utils.DEFAULT_LINE_LIST`, which is currently
        
            >>> full_line_list = ['PaB', 'HeI-1083', 'SIII', 'SII', 'Ha',
                                  'OI-6302', 'OIII', 'Hb', 'OIII-4363', 'Hg',
                                  'Hd', 'NeIII', 'OII', 'NeVI', 'NeV', 
                                  'MgII','CIV-1549', 'CIII-1908', 'OIII-1663', 
                                  'HeII-1640', 'NIII-1750', 'NIV-1487', 
                                  'NV-1240', 'Lya']
        
        The full list of implemented lines is in `~grizli.utils.get_line_wavelengths`.
    
    continuum_list : None or list
        Override the default continuum templates if None.
    
    fsps_templates : bool
        If True, get the FSPS NMF templates.
        
    Returns
    -------
    temp_list : dictionary of `~grizli.utils.SpectrumTemplate` objects
        Output template list
    
    """
    
    if stars:
        # templates = glob.glob('%s/templates/Pickles_stars/ext/*dat' %(os.getenv('GRIZLI')))
        # templates = []
        # for t in 'obafgkmrw':
        #     templates.extend( glob.glob('%s/templates/Pickles_stars/ext/uk%s*dat' %(os.getenv('THREEDHST'), t)))
        # templates.extend(glob.glob('%s/templates/SPEX/spex-prism-M*txt' %(os.getenv('THREEDHST'))))
        # templates.extend(glob.glob('%s/templates/SPEX/spex-prism-[LT]*txt' %(os.getenv('THREEDHST'))))
        # 
        # #templates = glob.glob('/Users/brammer/Downloads/templates/spex*txt')
        # templates = glob.glob('bpgs/*ascii')
        # info = catIO.Table('bpgs/bpgs.info')
        # type = np.array([t[:2] for t in info['type']])
        # templates = []
        # for t in 'OBAFGKM':
        #     test = type == '-%s' %(t)
        #     so = np.argsort(info['type'][test])
        #     templates.extend(info['file'][test][so])
        #             
        # temp_list = OrderedDict()
        # for temp in templates:
        #     #data = np.loadtxt('bpgs/'+temp, unpack=True)
        #     data = np.loadtxt(temp, unpack=True)
        #     #data[0] *= 1.e4 # spex
        #     scl = np.interp(5500., data[0], data[1])
        #     name = os.path.basename(temp)
        #     #ix = info['file'] == temp
        #     #name='%5s %s' %(info['type'][ix][0][1:], temp.split('.as')[0])
        #     print(name)
        #     temp_list[name] = utils.SpectrumTemplate(wave=data[0],
        #                                              flux=data[1]/scl)
        
        # np.save('stars_bpgs.npy', [temp_list])
        
        
        # tall = np.load(os.path.join(os.getenv('GRIZLI'), 
        #                                  'templates/stars.npy'))[0]
        # 
        # return tall
        # 
        # temp_list = OrderedDict()
        # for k in tall:
        #     if k.startswith('uk'):
        #         temp_list[k] = tall[k]
        # 
        # return temp_list
        # 
        # for t in 'MLT':
        #     for k in tall:
        #         if k.startswith('spex-prism-'+t):
        #             temp_list[k] = tall[k]
        #             
        # return temp_list
        
        #return temp_list
        templates = ['M6.5.txt', 'M8.0.txt', 'L1.0.txt', 'L3.5.txt', 'L6.0.txt', 'T2.0.txt', 'T6.0.txt', 'T7.5.txt']
        templates = ['stars/'+t for t in templates]
    else:
        ## Intermediate and very old
        # templates = ['templates/EAZY_v1.0_lines/eazy_v1.0_sed3_nolines.dat',  
        #              'templates/cvd12_t11_solar_Chabrier.extend.skip10.dat']     
        templates = ['eazy_intermediate.dat', 
                     'cvd12_t11_solar_Chabrier.dat']
                 
        ## Post starburst
        #templates.append('templates/UltraVISTA/eazy_v1.1_sed9.dat')
        templates.append('post_starburst.dat')
    
        ## Very blue continuum
        #templates.append('templates/YoungSB/erb2010_continuum.dat')
        templates.append('erb2010_continuum.dat')
    
        ### Test new templates
        # templates = ['templates/erb2010_continuum.dat',
        # 'templates/fsps/tweak_fsps_temp_kc13_12_006.dat',
        # 'templates/fsps/tweak_fsps_temp_kc13_12_008.dat']
    
        if fsps_templates:
            #templates = ['templates/fsps/tweak_fsps_temp_kc13_12_0{0:02d}.dat'.format(i+1) for i in range(12)]
            templates = ['fsps/fsps_QSF_12_v3_nolines_0{0:02d}.dat'.format(i+1) for i in range(12)]
            #templates = ['fsps/fsps_QSF_7_v3_nolines_0{0:02d}.dat'.format(i+1) for i in range(7)]
            
        
        if alf_template:
            templates.append('alf_SSP.dat')
            
        if continuum_list is not None:
            templates = continuum_list
        
    temp_list = OrderedDict()
    for temp in templates:
        data = np.loadtxt(os.path.join(os.getenv('GRIZLI'), 'templates', temp), unpack=True)
        #scl = np.interp(5500., data[0], data[1])
        scl = 1.
        name = temp #os.path.basename(temp)
        temp_list[name] = SpectrumTemplate(wave=data[0], flux=data[1]/scl,
                                           name=name)
        
        temp_list[name].name = name
    
    if stars:
        return temp_list
        
    ### Emission lines:
    line_wavelengths, line_ratios = get_line_wavelengths()
     
    if line_complexes:
        #line_list = ['Ha+SII', 'OIII+Hb+Ha', 'OII']
        #line_list = ['Ha+SII', 'OIII+Hb', 'OII']
        line_list = ['Ha+NII+SII+SIII+He', 'OIII+Hb', 'OII+Ne', 'Lya+CIV']
    else:
        if full_line_list is None:
            line_list = DEFAULT_LINE_LIST
        else:
            line_list = full_line_list
            
        #line_list = ['Ha', 'SII']
    
    # Use FSPS grid for lines
    wave_grid = None
    # if fsps_templates:
    #     wave_grid = data[0]
    # else:
    #     wave_grid = None   
    
    for li in line_list:
        scl = line_ratios[li]/np.sum(line_ratios[li])
        for i in range(len(scl)):
            line_i = SpectrumTemplate(wave=wave_grid, 
                                      central_wave=line_wavelengths[li][i], 
                                      flux=None, fwhm=fwhm, velocity=True,
                                      lorentz=lorentz)
                                      
            if i == 0:
                line_temp = line_i*scl[i]
            else:
                line_temp = line_temp + line_i*scl[i]
        
        name = 'line {0}'.format(li)
        line_temp.name = name
        temp_list[name] = line_temp
                                 
    return temp_list    

def load_sdss_pca_templates(file='spEigenQSO-55732.fits', smooth=3000):
    """
    Load SDSS eigen templates
    """
    from collections import OrderedDict
    import scipy.ndimage as nd
    
    im = pyfits.open(os.path.join(os.getenv('GRIZLI'), 'templates', file))
    h = im[0].header
    log_wave = np.arange(h['NAXIS1'])*h['COEFF1']+h['COEFF0']
    wave = 10**log_wave
    
    name = file.split('.fits')[0]
    
    if smooth > 0:
        dv_in = h['COEFF1']*3.e5
        n = smooth / dv_in
        data = nd.gaussian_filter1d(im[0].data, n, axis=1).astype(np.float64)
        skip = int(n/2.5)
        wave = wave[::skip]
        data = data[:,::skip]
    else:
        data = im[0].data.astype(np.float64)
        
    N = h['NAXIS2']
    temp_list = OrderedDict()
    for i in range(N):
        temp_list['{0} {1}'.format(name, i+1)] = SpectrumTemplate(wave=wave, flux=data[i,:])
    
    return temp_list
    
def polynomial_templates(wave, order=0, line=False):
    temp = OrderedDict()  
    if line:
        for sign in [1,-1]:
            key = 'poly {0}'.format(sign)
            temp[key] = SpectrumTemplate(wave, sign*(wave/1.e4-1)+1)
            temp[key].name = key
            
        return temp
        
    for i in range(order+1):
        key = 'poly {0}'.format(i)
        temp[key] = SpectrumTemplate(wave, (wave/1.e4-1)**i)
        temp[key].name = key
    
    return temp
        
def dot_templates(coeffs, templates, z=0, max_R=5000):
    """Compute template sum analogous to `np.dot(coeffs, templates)`.
    """  
    
    if len(coeffs) != len(templates):
        raise ValueError ('shapes of coeffs ({0}) and templates ({1}) don\'t match'.format(len(coeffs), len(templates)))
          
    # for i, te in enumerate(templates):
    #     if i == 0:
    #         tc = templates[te].zscale(z, scalar=coeffs[i])
    #         tl = templates[te].zscale(z, scalar=coeffs[i])
    #     else:
    #         if te.startswith('line'):
    #             tc += templates[te].zscale(z, scalar=0.)
    #         else:
    #             tc += templates[te].zscale(z, scalar=coeffs[i])
    #            
    #         tl += templates[te].zscale(z, scalar=coeffs[i])
    wave, flux_arr, is_line = array_templates(templates, max_R=max_R)
    
    # Continuum
    cont = np.dot(coeffs*(~is_line), flux_arr)
    tc = SpectrumTemplate(wave=wave, flux=cont).zscale(z)
    
    # Full template
    line = np.dot(coeffs, flux_arr)
    tl = SpectrumTemplate(wave=wave, flux=line).zscale(z)
    
    return tc, tl
    
def array_templates(templates, max_R=5000):
    """Return an array version of the templates that have all been interpolated to the same grid.
    
    
    Parameters
    ----------
    templates : dictionary of `~grizli.utils.SpectrumTemplate` objects
        Output template list with `NTEMP` templates.  
    
    max_R : float
        Maximum spectral resolution of the regridded templates.
        
    Returns
    -------
    wave : `~numpy.ndarray`, dimensions `(NL,)`
        Array containing unique wavelengths.
        
    flux_arr : `~numpy.ndarray`, dimensions `(NTEMP, NL)`
        Array containing the template fluxes interpolated at `wave`.
    
    is_line : `~numpy.ndarray`
        Boolean array indicating emission line templates (the key in the 
        template dictionary starts with "line ").
        
    """
    from grizli.utils_c.interp import interp_conserve_c
        
    wave = np.unique(np.hstack([templates[t].wave for t in templates]))
    clipsum, iter = 1, 0
    while (clipsum > 0) & (iter < 10):
        clip = np.gradient(wave)/wave < 1/max_R
        idx = np.arange(len(wave))[clip]
        wave[idx[::2]] = np.nan
        wave = wave[np.isfinite(wave)]
        iter += 1
        clipsum = clip.sum()
        #print(iter, clipsum)
        
    NTEMP = len(templates)
    flux_arr = np.zeros((NTEMP, len(wave)))
    
    for i, t in enumerate(templates):
        flux_arr[i,:] = interp_conserve_c(wave, templates[t].wave,
                                          templates[t].flux)
    
    is_line = np.array([t.startswith('line ') for t in templates])
    
    return wave, flux_arr, is_line
    
def compute_equivalent_widths(templates, coeffs, covar, max_R=5000, Ndraw=1000, seed=0):
    """Compute template-fit emission line equivalent widths
    
    Parameters
    ----------
    templates : dictionary of `~grizli.utils.SpectrumTemplate` objects
        Output template list with `NTEMP` templates.  
    
    coeffs : `~numpy.ndarray`, dimensions (`NTEMP`)
        Fit coefficients
        
    covar :  `~numpy.ndarray`, dimensions (`NTEMP`, `NTEMP`)
        Covariance matrix
        
    max_R : float
        Maximum spectral resolution of the regridded templates.

    Ndraw : int
        Number of random draws to extract from the covariance matrix
    
    seed : positive int
        Random number seed to produce repeatible results. If `None`, then 
        use default state.
        
    Returns
    -------
    EWdict : dict
        Dictionary of [16, 50, 84th] percentiles of the line EW distributions.
    
    """
    
    if False:
        # Testing
        templates = t1
        chi2, coeffsx, coeffs_errx, covarx = mb.xfit_at_z(z=tfit['z'], templates=templates, fitter='nnls', fit_background=True, get_uncertainties=2)
        covar = covarx[mb.N:,mb.N:]
        coeffs = coeffsx[mb.N:]
    
    # Array versions of the templates
    wave, flux_arr, is_line = array_templates(templates, max_R=max_R)
    keys = np.array(list(templates.keys()))
    
    EWdict = OrderedDict()
    for key in keys[is_line]:
        EWdict[key] = (0., 0., 0.)
        
    # Only worry about templates with non-zero coefficients, which should
    # be accounted for in the covariance array (with get_uncertainties=2)
    clip = coeffs != 0
    # No valid lines
    if (is_line & clip).sum() == 0:
        return EWdict
    
    # Random draws from the covariance matrix
    covar_clip = covar[clip,:][:,clip]
    if seed is not None:
        np.random.seed(seed)
    
    draws = np.random.multivariate_normal(coeffs[clip], covar_clip, size=Ndraw)
    
    # Evaluate the continuum fits from the draws             
    continuum = np.dot(draws*(~is_line[clip]), flux_arr[clip,:])
    
    # Compute the emission line EWs
    tidx = np.where(is_line[clip])[0]
    for ix in tidx:
        key = keys[clip][ix]

        # Line template
        line = np.dot(draws[:,ix][:,None], flux_arr[clip,:][ix,:][None,:])
        
        # Where line template non-zero
        mask = flux_arr[clip,:][ix,:] > 0
        ew_i = np.trapz((line/continuum)[:,mask], wave[mask], axis=1)
        
        EWdict[key] = np.percentile(ew_i, [16., 50., 84.])
    
    return EWdict

def get_spectrum_AB_mags(spectrum, bandpasses=[]):
    """
    Integrate a `~pysynphot` spectrum through filter bandpasses
    
    Parameters
    ----------
    spectrum : type

    bandpasses : list
        List of `pysynphot` bandpass objects, e.g., 
        
           >>> import pysynphot as S
           >>> bandpasses = [S.ObsBandpass('wfc3,ir,f140w')]
    
    
    Returns
    -------
    ab_mags : dict
        Dictionary with keys from `bandpasses` and the integrated magnitudes
        
    """
    import pysynphot as S
    flat = S.FlatSpectrum(0, fluxunits='ABMag')
    ab_mags = OrderedDict()
    
    for bp in bandpasses:
        flat_obs = S.Observation(flat, bp)
        spec_obs = S.Observation(spectrum, bp)
        ab_mags[bp.name] = -2.5*np.log10(spec_obs.countrate()/flat_obs.countrate())
    
    return ab_mags
    
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

def full_spectrum_wcsheader(center_wave=1.4e4, dlam=40, NX=100, spatial_scale=1, NY=10):
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
    refh['CRVAL1'] = center_wave/1.e4
    refh['CD1_1'] = dlam/1.e4
    refh['CD1_2'] = 0.
    refh['CRVAL2'] = 0.
    refh['CD2_2'] = spatial_scale
    refh['CD2_1'] = 0.
    refh['RADESYS'] = ''
    
    refh['CTYPE1'] = 'RA---TAN-SIP'
    refh['CUNIT1'] = 'mas'
    refh['CTYPE2'] = 'DEC--TAN-SIP'
    refh['CUNIT2'] = 'mas'
    
    ref_wcs = pywcs.WCS(refh)    
    ref_wcs.pscale = get_wcs_pscale(ref_wcs)
    
    return refh, ref_wcs
    
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

def header_keys_from_filelist(fits_files, keywords=[], ext=0, colname_case=str.lower):
    """Dump header keywords to a `~astropy.table.Table`
    
    Parameters
    ----------
    fits_files : list
        List of FITS filenames
    
    keywords : list or None
        List of header keywords to retrieve.  If `None`, then generate a list
        of *all* keywords from the first file in the list.
        
    ext : int, tuple
        FITS extension from which to pull the header.  Can be integer or 
        tuple, e.g., ('SCI',1) for HST ACS/WFC3 FLT files. 
    
    colname_case : func
        Function to set the case of the output colnames, e.g., `str.lower`, 
        `str.upper`, `str.title`.
        
    Returns
    -------
    tab : `~astropy.table.Table`
        Output table.
    
    """
    import numpy as np
    import astropy.io.fits as pyfits
    from astropy.table import Table
    
    # If keywords=None, get full list from first FITS file
    if keywords is None:
        h = pyfits.getheader(fits_files[0], ext)
        keywords = list(np.unique(list(h.keys())))
        keywords.pop(keywords.index(''))
        keywords.pop(keywords.index('HISTORY'))
    
    # Loop through files
    lines = []
    for file in fits_files:
        line = [file]
        h = pyfits.getheader(file, ext)
        for key in keywords:
            if key in h:
                line.append(h[key])
            else:
                line.append(None)
        
        lines.append(line)
    
    # Column names
    table_header = [colname_case(key) for key in ['file']+keywords]
    
    # Output table
    tab = Table(data=np.array(lines), names=table_header)
    
    return tab        
def drizzle_array_groups(sci_list, wht_list, wcs_list, scale=0.1, kernel='point', pixfrac=1., verbose=True):
    """Drizzle array data with associated wcs
    
    Parameters
    ----------
    sci_list, wht_list : list
        List of science and weight `~numpy.ndarray` objects.
        
    wcs_list : list
    
    scale : float
        Output pixel scale in arcsec.
    
    kernel, pixfrac : str, float
        Drizzle parameters
    
    verbose : bool
        Print status messages
        
    Returns
    -------
    outsci, outwht, outctx : `~numpy.ndarray`
        Output drizzled science, weight and context images
        
    header, outputwcs : `~astropy.fits.io.Header`, `~astropy.wcs.WCS`
        Drizzled image header and WCS.
    
    """
    from drizzlepac.astrodrizzle import adrizzle
    from stsci.tools import logutil
    log = logutil.create_logger(__name__)
    
    # Output header / WCS    
    header, outputwcs = compute_output_wcs(wcs_list, pixel_scale=scale)
    shape = (header['NAXIS2'], header['NAXIS1'])
    
    # Output arrays
    outsci = np.zeros(shape, dtype=np.float32)
    outwht = np.zeros(shape, dtype=np.float32)
    outctx = np.zeros(shape, dtype=np.int32)
    
    # Do drizzle
    N = len(sci_list)
    for i in range(N):
        if verbose:
            log.info('Drizzle array {0}/{1}'.format(i+1, N))
            
        adrizzle.do_driz(sci_list[i].astype(np.float32, copy=False), 
                         wcs_list[i], 
                         wht_list[i].astype(np.float32, copy=False),
                         outputwcs, outsci, outwht, outctx, 1., 'cps', 1,
                         wcslin_pscale=wcs_list[i].pscale, uniqid=1, 
                         pixfrac=pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=None)
        
    return outsci, outwht, outctx, header, outputwcs
    
def compute_output_wcs(wcs_list, pixel_scale=0.1, max_size=10000):
    """
    Compute output WCS that contains the full list of input WCS
    
    Parameters
    ----------
    wcs_list : list
        List of individual `~astropy.wcs.WCS` objects.
        
    pixel_scale : type
        Pixel scale of the output WCS
    
    max_size : int
        Maximum size out the output image dimensions
        
    Returns
    -------
    header : `~astropy.io.fits.Header`
        WCS header
        
    outputwcs : `~astropy.wcs.WCS`
        Output WCS
    
    """
    from shapely.geometry import Polygon
            
    footprint = Polygon(wcs_list[0].calc_footprint())
    for i in range(1, len(wcs_list)):
        fp_i = Polygon(wcs_list[i].calc_footprint())
        footprint = footprint.union(fp_i)
    
    x, y = footprint.convex_hull.boundary.xy
    x, y = np.array(x), np.array(y)

    # center
    crval = np.array(footprint.centroid.xy).flatten()
    
    # dimensions in arcsec
    xsize = (x.max()-x.min())*np.cos(crval[1]/180*np.pi)*3600
    ysize = (y.max()-y.min())*3600

    xsize = np.minimum(xsize, max_size*pixel_scale)
    ysize = np.minimum(ysize, max_size*pixel_scale)
    
    header, outputwcs = make_wcsheader(ra=crval[0], dec=crval[1], size=(xsize, ysize), pixscale=pixel_scale, get_hdu=False, theta=0)
    
    return header, outputwcs

def symlink_templates(force=False):
    """Symlink templates from module to $GRIZLI/templates as part of the initial setup
    
    Parameters
    ----------
    force : bool
        Force link files even if they already exist.
    """
    if 'GRIZLI' not in os.environ:
        print('"GRIZLI" environment variable not set!')
        return False
        
    module_path = os.path.dirname(__file__)
    out_path = os.path.join(os.environ['GRIZLI'], 'templates')
    
    files = glob.glob(os.path.join(module_path, 'data/templates/*'))
    #print(files)
    for file in files:
        filename = os.path.basename(file)
        out_file = os.path.join(out_path, filename)
        #print(filename, out_file)
        if (not os.path.exists(out_file)) | force:
            if os.path.exists(out_file): # (force)
                os.remove(out_file)
                
            os.symlink(file, out_file)
            print('Symlink: {0} -> {1}'.format(file, out_path))
        else:
            print('File exists: {0}'.format(out_file))
            
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
    
    badpix = os.path.join(os.getenv('iref'), 'badpix_spars200_Nov9.fits')
    print('Extra WFC3/IR bad pixels: {0}'.format(badpix))
    if not os.path.exists(badpix):
        os.system('curl -o {0}/badpix_spars200_Nov9.fits https://raw.githubusercontent.com/gbrammer/wfc3/master/data/badpix_spars200_Nov9.fits'.format(os.getenv('iref')))
    
    # Pixel area map
    pam = os.path.join(os.getenv('iref'), 'ir_wfc3_map.fits')
    print('Pixel area map: {0}'.format(pam))
    if not os.path.exists(pam):
        os.system('curl -o {0} http://www.stsci.edu/hst/wfc3/pam/ir_wfc3_map.fits'.format(pam))
    
def fetch_config_files(ACS=False):
    """
    Config files needed for Grizli
    """
    cwd = os.getcwd()
    
    print('Config directory: {0}/CONF'.format(os.getenv('GRIZLI')))
    
    os.chdir(os.path.join(os.getenv('GRIZLI'), 'CONF'))
    
    ftpdir = 'ftp://ftp.stsci.edu/cdbs/wfc3_aux/'
    tarfiles = ['{0}/WFC3.IR.G102.cal.V4.32.tar.gz'.format(ftpdir),
                '{0}/WFC3.IR.G141.cal.V4.32.tar.gz'.format(ftpdir),
                '{0}/grism_master_sky_v0.5.tar.gz'.format(ftpdir)]
    
    gURL = 'http://www.stsci.edu/~brammer/Grizli/Files'
    tarfiles.append('{0}/WFC3IR_extended_PSF.v1.tar.gz'.format(gURL))
    
    if ACS:
        tarfiles.append('{0}/ACS.WFC.sky.tar.gz'.format(gURL))
        tarfiles.append('{0}/ACS_CONFIG.tar.gz'.format(gURL))
                        
    for url in tarfiles:
        file=os.path.basename(url)
        if not os.path.exists(file):
            print('Get {0}'.format(file))
            os.system('curl -o {0} {1}'.format(file, url))
        
        os.system('tar xzvf {0}'.format(file))
    
    # ePSF files for fitting point sources
    psf_path = 'http://www.stsci.edu/hst/wfc3/analysis/PSF/psf_downloads/wfc3_ir/'
    files = ['{0}/PSFSTD_WFC3IR_{1}.fits'.format(psf_path, filt) 
             for filt in ['F105W', 'F125W', 'F140W', 'F160W']]
             
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

class MW_F99(object):
    """
    Wrapper around the `specutils.extinction` / `extinction` modules, which are called differently
    """
    def __init__(self, a_v, r_v=3.1):
        self.a_v = a_v
        self.r_v = r_v
        
        self.IS_SPECUTILS = False
        self.IS_EXTINCTION = False
        
        try:
            from specutils.extinction import ExtinctionF99
            self.IS_SPECUTILS = True
            self.F99 = ExtinctionF99(self.a_v, r_v=self.r_v)
        except(ImportError):
            try:
                from extinction import Fitzpatrick99
                self.IS_EXTINCTION = True
                self.F99 = Fitzpatrick99(r_v=self.r_v)
                
            except(ImportError):
                print("""
Couldn\'t find extinction modules in 
`specutils.extinction` or 
`extinction.Fitzpatrick99`.

MW extinction not implemented.
""")

        self.status = self.IS_SPECUTILS | self.IS_EXTINCTION
        
    def __call__(self, wave_input):
        import astropy.units as u
        
        if isinstance(wave_input, list):
            wave = np.array(wave_input)
        else:
            wave = wave_input
            
        if self.status is False:
            return np.zeros_like(wave)
             
        if self.IS_SPECUTILS:
            if hasattr(wave, 'unit'):
                wave_aa = wave
            else:
                wave_aa = wave*u.AA
            
            return self.F99(wave_aa)
        
        if self.IS_EXTINCTION:
            if hasattr(wave, 'unit'):
                wave_aa = wave.to(u.AA)
            else:
                wave_aa = wave
                
            return self.F99(wave_aa, self.a_v, unit='aa')
            
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
        
        # Dummy, use F105W ePSF for F098M and F110W
        self.epsf['F098M'] = self.epsf['F105W']
        self.epsf['F110W'] = self.epsf['F105W']
        
        # Extended
        self.extended_epsf = {}
        for filter in ['F105W', 'F125W', 'F140W', 'F160W']:
            file = os.path.join(os.getenv('GRIZLI'), 'CONF',
                                'extended_PSF_{0}.fits'.format(filter))
            
            if not os.path.exists(file):
                msg = 'Extended PSF file \'{0}\' not found.'.format(file)
                msg += '\n                   Get the archive from http://www.stsci.edu/~brammer/Grizli/Files/WFC3IR_extended_PSF.v1.tar.gz'
                msg += '\n                   and unpack in ${GRIZLI}/CONF/' 
                raise FileNotFoundError(msg)
                
            data = pyfits.open(file)[0].data#.T
            data[data < 0] = 0 
            
            # Mask center
            NX = data.shape[0]/2-1
            yp, xp = np.indices(data.shape)
            R = np.sqrt((xp-NX)**2+(yp-NX)**2)
            data[R <= 4] = 0.
            
            self.extended_epsf[filter] = data
            self.extended_N = int(NX)
            
        self.extended_epsf['F098M'] = self.extended_epsf['F105W']
        self.extended_epsf['F110W'] = self.extended_epsf['F105W']
        
    def get_at_position(self, x=507, y=507, filter='F140W'):
        """Evaluate ePSF at detector coordinates
        TBD
        """
        epsf = self.epsf[filter]
        
        rx = 1+(np.clip(x,1,1013)-0)/507.
        ry = 1+(np.clip(y,1,1013)-0)/507.
                
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
        self.eval_filter = filter
        
        return psf_xy
    
    def eval_ePSF(self, psf_xy, dx, dy, extended_data=None):
        """Evaluate PSF at dx,dy coordinates
        
        TBD
        """
        # So much faster than scipy.interpolate.griddata!
        from scipy.ndimage.interpolation import map_coordinates
        
        # ePSF only defined to 12.5 pixels
        ok = (np.abs(dx) <= 12.5) & (np.abs(dy) <= 12.5)
        coords = np.array([50+4*dx[ok], 50+4*dy[ok]])
        
        # Do the interpolation
        interp_map = map_coordinates(psf_xy, coords, order=3)
        
        # Fill output data
        out = np.zeros_like(dx, dtype=np.float32)
        out[ok] = interp_map
        
        # Extended PSF
        if extended_data is not None:
            ok = (np.abs(dx) < self.extended_N) & (np.abs(dy) < self.extended_N)
            x0 = self.extended_N
            coords = np.array([x0+dy[ok]+0, x0+dx[ok]])
            interp_map = map_coordinates(extended_data, coords, order=0)
            out[ok] += interp_map
            
        return out
    
    @staticmethod
    def objective_epsf(params, self, psf_xy, sci, ivar, xp, yp, extended_data, ret):
        """Objective function for fitting ePSFs
        
        TBD
        
        params = [normalization, xc, yc, background]
        """
        dx = xp-params[1]
        dy = yp-params[2]

        ddx = xp-xp.min()
        ddy = yp-yp.min()

        ddx = ddx/ddx.max()
        ddy = ddy/ddy.max()
        
        psf_offset = self.eval_ePSF(psf_xy, dx, dy, extended_data=extended_data)*params[0] + params[3] + params[4]*ddx + params[5]*ddy + params[6]*ddx*ddy
        
        if ret == 'resid':
            return (sci-psf_offset)*np.sqrt(ivar)
            
        chi2 = np.sum((sci-psf_offset)**2*(ivar))
        #print(params, chi2)
        return chi2
    
    def fit_ePSF(self, sci, center=None, origin=[0,0], ivar=1, N=7, 
                 filter='F140W', tol=1.e-4, guess=None, get_extended=False):
        """Fit ePSF to input data
        TBD
        """
        from scipy.optimize import minimize
        
        sh = sci.shape
        if center is None:
            y0, x0 = np.array(sh)/2.-1
        else:
            x0, y0 = center
        
        xd = x0+origin[1]
        yd = y0+origin[0]
        
        xc, yc = int(x0), int(y0)
        
        psf_xy = self.get_at_position(x=xd, y=yd, filter=filter)
        
        yp, xp = np.indices(sh)
                    
        if guess is None:
            if np.isscalar(ivar):
                ix = np.argmax(sci.flatten())
            else:
                ix = np.argmax((sci*(ivar > 0)).flatten())
        
            xguess = xp.flatten()[ix]
            yguess = yp.flatten()[ix]
        else:
            xguess, yguess = guess
            
        guess = [sci[yc-N:yc+N, xc-N:xc+N].sum()/psf_xy.sum(), xguess, yguess, 0, 0, 0, 0]
        sly = slice(yc-N, yc+N); slx = slice(xc-N, xc+N)
        sly = slice(yguess-N, yguess+N); slx = slice(xguess-N, xguess+N)
        
        if get_extended:
            extended_data = self.extended_epsf[filter]
        else:
            extended_data = None
        
        args = (self, psf_xy, sci[sly, slx], ivar[sly, slx], xp[sly, slx], yp[sly, slx], extended_data, 'chi2')
        
        out = minimize(self.objective_epsf, guess, args=args, method='Powell', tol=tol)
        
        psf_params = out.x
        psf_params[1] -= x0
        psf_params[2] -= y0
        
        return psf_params
        
        # dx = xp-psf_params[1]
        # dy = yp-psf_params[2]
        # output_psf = self.eval_ePSF(psf_xy, dx, dy)*psf_params[0]
        # 
        # return output_psf, psf_params
    
    def get_ePSF(self, psf_params, origin=[0,0], shape=[20,20], filter='F140W', get_extended=False):
        """
        Evaluate an Effective PSF
        """
        sh = shape
        y0, x0 = np.array(sh)/2.-1
        
        xd = x0+origin[1]
        yd = y0+origin[0]
        
        xc, yc = int(x0), int(y0)
        
        psf_xy = self.get_at_position(x=xd, y=yd, filter=filter)
        
        yp, xp = np.indices(sh)
        
        dx = xp-psf_params[1]-x0
        dy = yp-psf_params[2]-y0
        
        if get_extended:
            extended_data = self.extended_epsf[filter]
        else:
            extended_data = None
            
        output_psf = self.eval_ePSF(psf_xy, dx, dy, extended_data=extended_data)*psf_params[0]
        
        return output_psf
        
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
            elif isinstance(file, pyfits.BinTableHDU):
                format = 'fits'
            else:            
                if file.endswith('.fits'):
                    format='fits'
                elif file.endswith('.csv'):
                    format = 'csv'
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
            rd_pairs['RA'] = 'DEC'
            rd_pairs['ALPHA_J2000'] = 'DELTA_J2000'
            rd_pairs['X_WORLD'] = 'Y_WORLD'
            rd_pairs['ALPHA_SKY'] = 'DELTA_SKY'
            
            for k in list(rd_pairs.keys()):
                rd_pairs[k.lower()] = rd_pairs[k].lower()
                
        rd_pair = None 
        for c in rd_pairs:
            if c in self.colnames:
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

        if isinstance(other, list) | isinstance(other, tuple):
            rd = [slice(0,1),slice(1,2)]
            
        else:
            if other_radec is None:
                rd = self.parse_radec_columns(other)
            else:
                rd = self.parse_radec_columns(other, rd_pairs={other_radec[0]:other_radec[1]})

            if rd is False:
                print('No RA/Dec. columns found in `other` table.')
                return False
            
        other_coo = SkyCoord(ra=other[rd[0]], dec=other[rd[1]])
                     
        idx, d2d, d3d = other_coo.match_to_catalog_sky(self_coo)
        return idx, d2d.to(u.arcsec)
            
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
    
        if os.path.exists(output):
            os.remove(output)
            
        self.write(output, format='jsviewer', css=DEFAULT_CSS,
                            max_lines=max_lines,
                            jskwargs={'use_local_files':localhost},
                            table_id=None, table_class=table_class)#,
                            #overwrite=True)

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

def fill_masked_covar(covar, mask):
    """Fill a covariance matrix in a larger array that had masked values
    
    Parameters
    ----------
    covar : `(M,M)` square `~np.ndarray`
        Masked covariance array.
        
    mask : bool mask, `N>M`
        The original mask.
    
    Returns
    -------
    covar_full : `~np.ndarray`
        Full covariance array with dimensions `(N,N)`.
    
    """
    N = mask.shape[0]
    idx = np.arange(N)[mask]
    covar_full = np.zeros((N,N), dtype=covar.dtype)
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            covar_full[ii,jj] = covar[i,j]
    
    return covar_full
    
def log_scale_ds9(im, lexp=1.e12, cmap=[7.97917, 0.8780493], scale=[-0.1,10]):
    """
    Scale an array like ds9 log scaling
    """
    import numpy as np
    
    contrast, bias = cmap
    clip = (np.clip(im, scale[0], scale[1])-scale[0])/(scale[1]-scale[0])
    clip_log = np.clip((np.log10(lexp*clip+1)/np.log10(lexp)-bias)*contrast+0.5, 0, 1)
    
    return clip_log

def mode_statistic(data, percentiles=range(10,91,10)):
    """
    Get modal value of a distribution of data following Connor et al. 2017
    https://arxiv.org/pdf/1709.01925.pdf
    
    Here we fit a spline to the cumulative distribution evaluated at knots 
    set to the `percentiles` of the distribution to improve smoothness.
    """
    from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
    
    so = np.argsort(data)
    order = np.arange(len(data))
    #spl = UnivariateSpline(data[so], order)
    
    knots = np.percentile(data, percentiles)
    dx = np.diff(knots)
    mask = (data[so] >= knots[0]-dx[0]) & (data[so] <= knots[-1]+dx[-1])
    spl = LSQUnivariateSpline(data[so][mask], order[mask], knots, ext='zeros')
    
    mask = (data[so] >= knots[0]) & (data[so] <= knots[-1])
    ix = (spl(data[so], nu=1, ext='zeros')*mask).argmax()
    mode = data[so][ix]
    return mode
    
def make_alf_template():
    """
    Make Alf + FSPS template
    """
    import alf.alf
    import fsps
    
    ssp = alf.alf.Alf()
    
    sp = fsps.StellarPopulation(zcontinuous=1)
    sp.params['logzsol'] = 0.2

    # Alf
    m = ssp.get_model(in_place=False, logage=0.96, zh=0.2, mgh=0.2)
    
    # FSPS
    w, spec = sp.get_spectrum(tage=10**0.96, peraa=True)
    
    # blue
    blue_norm = spec[w > 3600][0] / m[ssp.wave > 3600][0]
    red_norm = spec[w > 1.7e4][0] / m[ssp.wave > 1.7e4][0]
    
    templx = np.hstack([w[w < 3600], ssp.wave[(ssp.wave > 3600) & (ssp.wave < 1.7e4)], w[w > 1.7e4]])
    temply = np.hstack([spec[w < 3600]/blue_norm, m[(ssp.wave > 3600) & (ssp.wave < 1.7e4)], spec[w > 1.7e4]/red_norm])
    
    np.savetxt('alf_SSP.dat', np.array([templx, temply]).T, fmt='%.5e', header='wave flux\nlogage = 0.96\nzh=0.2\nmgh=0.2\nfsps: w < 3600, w > 1.7e4')
    