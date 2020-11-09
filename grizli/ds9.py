"""
Extensions of the `~pyds9.DS9` object
"""
import os

# Extend pyds9.DS9 class
try:
    # Avoid choking if pyds9 not available
    import pyds9
    ds9_obj = pyds9.DS9
except:
    print('NB: Couldn\'t `import pyds9`.  The `grizli.ds9.DS9` object will be broken')
    ds9_obj = object

def show_open_targets():
    """
    Summarize open ds9 targets
    """
    import pyds9
    import numpy as np
    
    targets = pyds9.ds9_targets()
    print('i         target frame    ra       dec       x     y ')
    for i, target in enumerate(targets):
        targ = target.split()[-1]
        ds9 = DS9(target=targ)
        ra, dec = np.cast[float](ds9.get('pan fk5').split())
        xi, yi = np.cast[float](ds9.get('pan image').split())
        
        frame = ds9.get('frame')
        print(f'{i} {targ:14} {frame:>5} {ra:8.4f} {dec:8.4f} ({xi:5.0f} {yi:5.0f})')
        
        
def get_ds9(i=-1):
    """
    Get ds9 xpa target
    
    Parameters
    ----------
    i : int
        Index of target in `~pyds9.ds9_targets`  list.
    
    Returns
    -------
    obj : `~grizli.ds9.DS9`
        DS9 object.
    """
    import pyds9
    targets = pyds9.ds9_targets()
    #print(targets)
    obj = DS9(target=targets[i].split()[-1])
    return obj
    
class DS9(ds9_obj):
    """
    Extend `~pyds9.DS9` object with convenience classes
    """

    def view(self, img, header=None, simplify_wcs=True):
        """
        From pysao
        """
        import astropy.wcs as pywcs
        from .utils import to_header
        
        if hasattr(img, 'header'):
            # FITS HDU
            self.set_np2arr(img.data)
            if not header:
                header = img.header
        else:
            self.set_np2arr(img)
        
        if header:
            if simplify_wcs:
                wcs = pywcs.WCS(header)
                simple_header = to_header(wcs)
                self.set("wcs replace", simple_header.tostring())
            else:
                self.set("wcs replace", header.tostring())
                
    def frame(self, id):
        self.set(f'frame {id}')

    def scale(self, z1, z2):
        self.set(f'scale limits {z1} {z2}')

    def set_defaults(self, match='image', verbose=False):
        """
        Match frame, set log scale
        """
        commands = f"""
        xpaset -p ds9 scale log
        xpaset -p ds9 scale limits -0.1 10
        xpaset -p ds9 cmap value 3.02222 0.647552
        xpaset -p ds9 match frame {match}
        xpaset -p ds9 frame lock {match}
        xpaset -p ds9 match colorbar
        xpaset -p ds9 lock colorbar
        xpaset -p ds9 match scale"""

        for c in commands.split('\n'):
            if 'xpaset' in c:
                try:
                    self.set(' '.join(c.split()[3:]))
                except:
                    print('xpa command failed: ', c)
                if verbose:
                    print(c)
    
    
    def match(self, match='image'):
        commands = """
        xpaset -p ds9 match frames %s
        xpaset -p ds9 frame lock %s
        """ % (match, match)

        for c in commands.split('\n'):
            if 'xpaset' in c:
                self.set(' '.join(c.split()[3:]))
    
    
    def cds_query(self, radius=1.):
        """
        Open browswer with CDS catalog query around central position
        """
        rd = self.get('pan fk5').strip()
        rdst = rd.replace('+', '%2B').replace('-', '%2D').replace(' ', '+')
        url = (f'"http://vizier.u-strasbg.fr/viz-bin/VizieR?'
               f'-c={rdst}&-c.rs={radius:.1f}"')
               
        os.system(f'open {url}')
    
    
    def eso_query(self, radius=1., dp_types=['CUBE','IMAGE'], extra=''):
        """
        Open browser with ESO archive query around central position.
        
        ``radius`` in arcmin.
        """
        ra, dec = self.get('pan fk5').strip().split()
        
        dp_type = ','.join(dp_types)
        
        url = (f'"https://archive.eso.org/scienceportal/home?'
                f'pos={ra},{dec}&r={radius/60.}&dp_type={dp_type}{extra}"')
                        
        os.system(f'open {url}')
    
    
    def mast_query(self, instruments=['WFC3','ACS','WFPC2'], max=1000):
        """
        Open browser with MAST archive query around central position
        """
        ra, dec = self.get('pan fk5').strip().split()
        if len(instruments) > 0:
            instr='&sci_instrume='+','.join(instruments)
        else:
            instr = ''
            
        url = (f'"https://archive.stsci.edu/hst/search.php?RA={ra}&DEC={dec}'
               f'&sci_aec=S{instr}&max_records={max}&outputformat=HTML_Table'
                '&action=Search"')
                
        os.system(f'open {url}')
    
    
    def alma_query(self, mirror="almascience.eso.org", extra=''):
        """
        Open browser with ALMA archive query around central position
        """
        ra, dec = self.get('pan fk5').strip().split()
    
        url = (f"https://{mirror}/asax/?result_view=observation"
               f"&raDec={ra}%20{dec}{extra}")
        os.system(f'open "{url}"')
    
    
    def show_legacysurvey(self, layer='dr8', zoom=14):
        """
        Open browser with legacysurvey.org panner around central position
        """
        ra, dec = self.get('pan fk5').strip().split()
        url = (f'"http://legacysurvey.org/viewer?ra={ra}&dec={dec}'
               f'&layer={layer}&zoom={zoom}"')
                
        os.system(f'open {url}')
        
        