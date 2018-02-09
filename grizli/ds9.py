"""
Extensions of the `~pyds9.DS9` object
"""

# Extend pyds9.DS9 class
try:
    # Avoid choking if pyds9 not available
    import pyds9
    ds9_obj = pyds9.DS9
except:
    print('NB: Couldn\'t `import pyds9`.  The `grizli.ds9.DS9` object will be broken')
    ds9_obj = object

class DS9(ds9_obj):
    """
    Extend `~pyds9.DS9` object with convenience classes
    """
    def view(self, img, header=None):
        """
        From pysao
        """
        if hasattr(img, 'header'):
            ### FITS HDU
            self.set_np2arr(img.data)
            if not header:
                header = img.header
            self.set("wcs replace", header.tostring())
        else:
            self.set_np2arr(img)
            if header:
                self.set("wcs replace", header.tostring())

    def frame(self, id):
        self.set('frame %d' %(id))
    
    def scale(self, z1, z2):
        self.set('scale limits %f %f' %(z1, z2))
        
    def set_defaults(self, match='image', verbose=False):
        """
        Match frame, set log scale
        """
        commands = """
        xpaset -p ds9 scale log
        xpaset -p ds9 scale limits -0.1 10
        xpaset -p ds9 cmap value 3.02222 0.647552
        xpaset -p ds9 match frames %s 
        xpaset -p ds9 frame lock %s
        xpaset -p ds9 match colorbars 
        xpaset -p ds9 lock colorbar
        xpaset -p ds9 match scales""" %(match, match)
                
        for c in commands.split('\n'):
            if 'xpaset' in c:
                self.set(' '.join(c.split()[3:]))
                if verbose:
                    print (c)
    
    def match(self, match='image'):
        commands = """
        xpaset -p ds9 match frames %s 
        xpaset -p ds9 frame lock %s
        """ %(match, match)
        
        for c in commands.split('\n'):
            if 'xpaset' in c:
                self.set(' '.join(c.split()[3:]))