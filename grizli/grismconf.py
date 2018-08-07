"""
Demonstrate aXe trace polynomials.
  
Initial code taken from `(Brammer, Pirzkal, & Ryan 2014) <https://github.com/WFC3Grism/CodeDescription>`_, which contains a detailed 
explanation how the grism configuration parameters and coefficients are defined and evaluated.
"""
import os
import numpy as np

from . import GRIZLI_PATH

class aXeConf():
    def __init__(self, conf_file='WFC3.IR.G141.V2.5.conf'):
        """Read an aXe-compatible configuration file
        
        Parameters
        ----------
        conf_file: str
            Filename of the configuration file to read
        
        """
        if conf_file is not None:
            self.conf = self.read_conf_file(conf_file)
            self.conf_file = conf_file
            self.count_beam_orders()
            
            ## Global XOFF/YOFF offsets
            if 'XOFF' in self.conf.keys():
                self.xoff = np.float(conf['XOFF'])
            else:
                self.xoff = 0.

            if 'YOFF' in self.conf.keys():
                self.yoff = np.float(conf['YOFF'])
            else:
                self.yoff = 0.
            
    def read_conf_file(self, conf_file='WFC3.IR.G141.V2.5.conf'):
        """Read an aXe config file, convert floats and arrays
        
        Parameters
        ----------
        conf_file: str
            Filename of the configuration file to read.
        
        Parameters are stored in an OrderedDict in `self.conf`.
        """
        from collections import OrderedDict
    
        conf = OrderedDict()
        lines = open(conf_file).readlines()
        for line in lines:
            ## empty / commented lines
            if (line.startswith('#')) | (line.strip() == '') | ('"' in line):
                continue
        
            ## split the line, taking out ; and # comments
            spl = line.split(';')[0].split('#')[0].split()
            param = spl[0]
            if len(spl) > 2: 
                value = np.cast[float](spl[1:])
            else:
                try:
                    value = float(spl[1])
                except:
                    value = spl[1]

            conf[param] = value
    
        return conf
    
    def count_beam_orders(self):
        """Get the maximum polynomial order in DYDX or DLDP for each beam
        """
        self.orders = {}
        for beam in ['A','B','C','D','E','F','G','H','I','J']:
            order = 0
            while 'DYDX_{0:s}_{1:d}'.format(beam, order) in self.conf.keys():
                order += 1
            
            while 'DLDP_{0:s}_{1:d}'.format(beam, order) in self.conf.keys():
                order += 1
            
            self.orders[beam] = order-1

    def get_beams(self):
        """Get beam parameters and read sensitivity curves
        """
        import os
        from collections import OrderedDict
        from astropy.table import Table, Column
        
        self.dxlam = OrderedDict()
        self.nx = OrderedDict()
        self.sens = OrderedDict()
        self.beams = []
        
        for beam in self.orders:
            if self.orders[beam] > 0:
                self.beams.append(beam)
                self.dxlam[beam] = np.arange(self.conf['BEAM{0}'.format(beam)].min(), self.conf['BEAM{0}'.format(beam)].max(), dtype=int)
                self.nx[beam] = int(self.dxlam[beam].max()-self.dxlam[beam].min())+1
                self.sens[beam] = Table.read('{0}/{1}'.format(os.path.dirname(self.conf_file), self.conf['SENSITIVITY_{0}'.format(beam)]))
                #self.sens[beam].wave = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                #self.sens[beam].sens = np.cast[np.double](self.sens[beam]['SENSITIVITY'])
                
                ### Need doubles for interpolating functions
                for col in self.sens[beam].colnames:
                    data = np.cast[np.double](self.sens[beam][col])
                    self.sens[beam].remove_column(col)
                    self.sens[beam].add_column(Column(data=data, name=col))
                
                ### Scale BEAM F
                if (beam == 'F') & ('G141' in self.conf_file): 
                    self.sens[beam]['SENSITIVITY'] *= 0.35
                
                
                if (beam == 'B') & ('G141' in self.conf_file): 
                    if self.conf['SENSITIVITY_B'] == 'WFC3.IR.G141.0th.sens.1.fits':
                        self.sens[beam]['SENSITIVITY'] *= 2
                    
                # wave = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                # sens = np.cast[np.double](self.sens[beam]['SENSITIVITY']
                # self.sens[beam]['WAVELENGTH'] = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                # self.sens[beam]['SENSITIVITY'] = )
        
        self.beams.sort()
        
    def field_dependent(self, xi, yi, coeffs):
        """aXe field-dependent coefficients
        
        See the `aXe manual <http://axe.stsci.edu/axe/manual/html/node7.html#SECTION00721200000000000000>`_ for a description of how the field-dependent coefficients are specified.
        
        Parameters
        ----------
        xi, yi : float or array-like
            Coordinate to evaluate the field dependent coefficients, where
            `xi = x-REFX` and `yi = y-REFY`.
        
        coeffs : array-like
            Field-dependency coefficients
        
        Returns
        -------
        a : float or array-like
            Evaluated field-dependent coefficients
            
        """
        ## number of coefficients for a given polynomial order
        ## 1:1, 2:3, 3:6, 4:10, order:order*(order+1)/2
        if isinstance(coeffs, float):
            order = 1
        else:
            order = int(-1+np.sqrt(1+8*len(coeffs))) // 2
    
        ## Build polynomial terms array
        ## $a = a_0+a_1x_i+a_2y_i+a_3x_i^2+a_4x_iy_i+a_5yi^2+$ ...
        xy = []
        for p in range(order):
            for px in range(p+1):
                #print 'x**%d y**%d' %(p-px, px)
                xy.append(xi**(p-px)*yi**(px))
    
        ## Evaluate the polynomial, allowing for N-dimensional inputs
        a = np.sum((np.array(xy).T*coeffs).T, axis=0)
    
        return a
    
    def evaluate_dp(self, dx, dydx):
        """Evalate arc length along the trace given trace polynomial coefficients
        
        Parameters
        ----------
        dx : array-like
            x pixel to evaluate
        
        dydx : array-like
            Coefficients of the trace polynomial
        
        Returns
        -------
        dp : array-like
            Arc length along the trace at position `dx`.
            
        For `dydx` polynomial orders 0, 1 or 2, integrate analytically.  
        Higher orders must be integrated numerically.
        
        **Constant:** 
            .. math:: dp = dx

        **Linear:** 
            .. math:: dp = \sqrt{1+\mathrm{DYDX}[1]}\cdot dx
        
        **Quadratic:** 
            .. math:: u = \mathrm{DYDX}[1] + 2\ \mathrm{DYDX}[2]\cdot dx
            
            .. math:: dp = (u \sqrt{1+u^2} + \mathrm{arcsinh}\ u) / (4\cdot \mathrm{DYDX}[2])
        
        """
        ## dp is the arc length along the trace
        ## $\lambda = dldp_0 + dldp_1 dp + dldp_2 dp^2$ ...
        
        poly_order = len(dydx)-1
        if (poly_order == 2):
            if dydx[2] == 0:
                poly_order = 1
                
        if poly_order == 0:   ## dy=0
            dp = dx                      
        elif poly_order == 1: ## constant dy/dx
            dp = np.sqrt(1+dydx[1]**2)*(dx)
        elif poly_order == 2: ## quadratic trace
            u0 = dydx[1]+2*dydx[2]*(0)
            dp0 = (u0*np.sqrt(1+u0**2)+np.arcsinh(u0))/(4*dydx[2])
            u = dydx[1]+2*dydx[2]*(dx)
            dp = (u*np.sqrt(1+u**2)+np.arcsinh(u))/(4*dydx[2])-dp0
        else:
            ## high order shape, numerical integration along trace
            ## (this can be slow)
            xmin = np.minimum((dx).min(), 0)
            xmax = np.maximum((dx).max(), 0)
            xfull = np.arange(xmin, xmax)
            dyfull = 0
            for i in range(1, poly_order):
                dyfull += i*dydx[i]*(xfull-0.5)**(i-1)
            
            ## Integrate from 0 to dx / -dx
            dpfull = xfull*0.
            lt0 = xfull < 0
            if lt0.sum() > 1:
                dpfull[lt0] = np.cumsum(np.sqrt(1+dyfull[lt0][::-1]**2))[::-1]
                dpfull[lt0] *= -1
                
            #
            gt0 = xfull > 0
            if gt0.sum() > 0:
                dpfull[gt0] = np.cumsum(np.sqrt(1+dyfull[gt0]**2))
              
            dp = np.interp(dx, xfull, dpfull)
            if dp[-1] == dp[-2]:
                dp[-1] = dp[-2]+np.diff(dp)[-2]
                
        return dp
        
    def get_beam_trace(self, x=507, y=507, dx=0., beam='A', fwcpos=None):
        """Get an aXe beam trace for an input reference pixel and list of output x pixels `dx`
        
        Parameters
        ----------
        x, y : float or array-like
            Evaluate trace definition at detector coordinates `x` and `y`.
            
        dx : float or array-like
            Offset in x pixels from `(x,y)` where to compute trace offset and 
            effective wavelength
            
        beam : str
            Beam name (i.e., spectral order) to compute.  By aXe convention, 
            `beam='A'` is the first order, 'B' is the zeroth order and 
            additional beams are the higher positive and negative orders.
            
        fwcpos : None or float
            For NIRISS, specify the filter wheel position to compute the 
            trace rotation
        
        Returns
        -------
        dy : float or array-like
            Center of the trace in y pixels offset from `(x,y)` evaluated at
            `dx`.
            
        lam : float or array-like
            Effective wavelength along the trace evaluated at `dx`.
            
        """
        NORDER = self.orders[beam]+1
        
        xi, yi = x-self.xoff, y-self.yoff
        xoff_beam = self.field_dependent(xi, yi, self.conf['XOFF_{0}'.format(beam)])
        yoff_beam = self.field_dependent(xi, yi, self.conf['YOFF_{0}'.format(beam)])
    
        ## y offset of trace (DYDX)
        dydx = np.zeros(NORDER) #0 #+1.e-80
        for i in range(NORDER):
            if 'DYDX_{0:s}_{1:d}'.format(beam, i) in self.conf.keys():
                coeffs = self.conf['DYDX_{0:s}_{1:d}'.format(beam, i)]
                dydx[i] = self.field_dependent(xi, yi, coeffs)
            
        # $dy = dydx_0+dydx_1 dx+dydx_2 dx^2+$ ...

        dy = yoff_beam
        for i in range(NORDER):
            dy += dydx[i]*(dx-xoff_beam)**i
        
        ## wavelength solution    
        dldp = np.zeros(NORDER)
        for i in range(NORDER):
            if 'DLDP_{0:s}_{1:d}'.format(beam, i) in self.conf.keys():
                coeffs = self.conf['DLDP_{0:s}_{1:d}'.format(beam, i)]
                dldp[i] = self.field_dependent(xi, yi, coeffs)
        
        dp = self.evaluate_dp(dx-xoff_beam, dydx)
        # ## dp is the arc length along the trace
        # ## $\lambda = dldp_0 + dldp_1 dp + dldp_2 dp^2$ ...
        # if self.conf['DYDX_ORDER_%s' %(beam)] == 0:   ## dy=0
        #     dp = dx-xoff_beam                      
        # elif self.conf['DYDX_ORDER_%s' %(beam)] == 1: ## constant dy/dx
        #     dp = np.sqrt(1+dydx[1]**2)*(dx-xoff_beam)
        # elif self.conf['DYDX_ORDER_%s' %(beam)] == 2: ## quadratic trace
        #     u0 = dydx[1]+2*dydx[2]*(0)
        #     dp0 = (u0*np.sqrt(1+u0**2)+np.arcsinh(u0))/(4*dydx[2])
        #     u = dydx[1]+2*dydx[2]*(dx-xoff_beam)
        #     dp = (u*np.sqrt(1+u**2)+np.arcsinh(u))/(4*dydx[2])-dp0
        # else:
        #     ## high order shape, numerical integration along trace
        #     ## (this can be slow)
        #     xmin = np.minimum((dx-xoff_beam).min(), 0)
        #     xmax = np.maximum((dx-xoff_beam).max(), 0)
        #     xfull = np.arange(xmin, xmax)
        #     dyfull = 0
        #     for i in range(1, NORDER):
        #         dyfull += i*dydx[i]*(xfull-0.5)**(i-1)
        #     
        #     ## Integrate from 0 to dx / -dx
        #     dpfull = xfull*0.
        #     lt0 = xfull <= 0
        #     if lt0.sum() > 1:
        #         dpfull[lt0] = np.cumsum(np.sqrt(1+dyfull[lt0][::-1]**2))[::-1]
        #         dpfull[lt0] *= -1
        #     #
        #     gt0 = xfull >= 0
        #     if gt0.sum() > 0:
        #         dpfull[gt0] = np.cumsum(np.sqrt(1+dyfull[gt0]**2))
        #       
        #     dp = np.interp(dx-xoff_beam, xfull, dpfull)
        
        ## Evaluate dldp    
        lam = dp*0.
        for i in range(NORDER):
            lam += dldp[i]*dp**i
        
        ### NIRISS rotation?
        if fwcpos is not None:
            if 'FWCPOS_REF' not in self.conf.keys():
                print('Parameter fwcpos={0} supplied but no FWCPOS_REF in {1:s}'.format(fwcpos, self.conf_file))
                return dy, lam
            
            order = self.conf['DYDX_ORDER_{0}'.format(beam)]
            if order != 2:
                print('ORDER={0:d} not supported for NIRISS rotation'.format(order))
                return dy, lam
                
            theta = (fwcpos - self.conf['FWCPOS_REF'])/180*np.pi*1
            theta *= -1 # DMS rotation
            #print('DMS')
            
            if theta == 0:
                return dy, lam
                
            ### For the convention of swapping/inverting axes for GR150C
            # if 'GR150C' in self.conf_file:
            #     theta = -theta
            
            ### If theta is small, use a small angle approximation.  
            ### Otherwise, 1./tan(theta) blows up and results in numerical 
            ### noise.
            xp = (dx-xoff_beam)/np.cos(theta)
            if (1-np.cos(theta) < 5.e-8):
                #print('Approximate!', xoff_beam, np.tan(theta))
                dy = dy + (dx-xoff_beam)*np.tan(theta)
                delta = 0.
                #print('Approx')
            else:
                ### Full transformed trace coordinates
                c = dydx
                #print('Not approx')
                
                beta = c[1]+2*c[2]*xp-1/np.tan(theta)
                chi = c[0]+c[1]*xp+c[2]*xp**2
                if theta < 0:
                    psi = (-beta+np.sqrt(beta**2-4*c[2]*chi))
                    psi *= 1./2/c[2]/np.tan(theta)
                    delta = psi*np.tan(theta)
                    dy = dx*np.tan(theta) + psi/np.cos(theta)
                else:
                    psi = (-beta-np.sqrt(beta**2-4*c[2]*chi))
                    psi *= 1./2/c[2]/np.tan(theta)
                    delta = psi*np.tan(theta)
                    dy = dx*np.tan(theta) + psi/np.cos(theta)
            
            ### Evaluate wavelength at 'prime position along the trace
            dp = self.evaluate_dp(xp+delta, dydx)
            
            lam = dp*0.
            for i in range(NORDER):
                lam += dldp[i]*dp**i
                    
        return dy, lam
        
    def show_beams(self, beams=['E','D','C','B','A']):
        """
        Make a demo plot of the beams of a given configuration file
        """
        import matplotlib.pyplot as plt
        
        x0, x1 = 507, 507
        dx = np.arange(-800,1200)

        if 'WFC3.UV' in self.conf_file:
            x0, x1 = 2073, 250
            dx = np.arange(-1200,1200)
        if 'G800L' in self.conf_file:
            x0, x1 = 2124, 1024
            dx = np.arange(-1200,1200)
            
        s=200 # marker size
        fig = plt.figure(figsize=[10,3])
        plt.scatter(0,0,marker='s', s=s, color='black', edgecolor='0.8',
                    label='Direct')
        
        for beam in beams:
            if 'XOFF_{0}'.format(beam) not in self.conf.keys():
                continue
            
            xoff = self.field_dependent(x0, x1, self.conf['XOFF_{0}'.format(beam)])
            dy, lam = self.get_beam_trace(x0, x1, dx=dx, beam=beam)
            xlim = self.conf['BEAM{0}'.format(beam)]
            ok = (dx >= xlim[0]) & (dx <= xlim[1])
            plt.scatter(dx[ok]+xoff, dy[ok], c=lam[ok]/1.e4, marker='s', s=s,
                        alpha=0.5, edgecolor='None')
            plt.text(np.median(dx[ok]), np.median(dy[ok])+1, beam,
                     ha='center', va='center', fontsize=14)
            print('Beam {0}, lambda=({1:.1f} - {2:.1f})'.format(beam, lam[ok].min(), lam[ok].max()))
            
        plt.grid()
        plt.xlabel(r'$\Delta x$')
        plt.ylabel(r'$\Delta y$')

        cb = plt.colorbar(pad=0.01, fraction=0.05)    
        cb.set_label(r'$\lambda\,(\mu\mathrm{m})$')
        plt.title(self.conf_file)
        plt.tight_layout()
        plt.savefig('{0}.pdf'.format(self.conf_file))    

def get_config_filename(instrume='WFC3', filter='F140W',
                        grism='G141', chip=1):
    """Generate a config filename based on the instrument, filter & grism combination. 
    
    Config files assumed to be found the directory specified by the `$GRIZLI` 
    environment variable, i.e., `${GRIZLI}/CONF`.
        
    Parameters
    ----------
    instrume : {'ACS', 'WFC3', 'NIRISS', 'NIRCam', 'WFIRST'}
        Instrument used
        
    filter : str
        Direct image filter.  This is only used for WFC3/IR, where the grism
        configuration files have been determined for each direct+grism 
        combination separately based on the filter wedge offsets of the 
        filters.
        
    grism : str
        Grism name.  Valid combinations are the following:
            
            ACS : G800L (assumed)
            WFC3 : G102, G141
            NIRISS : GR150R, GR150C
            NIRCam : F322W2, F356W, F430M, F444W, F460M
            WFIRST : (basic assumptions about the WFI grism)
            
    chip : int
        For ACS/WFC and UVIS, specifies the chip to use.  Note that this 
        is switched with respect to the header EXTNAME extensions:
        
            EXTVER = 1 is extension 1 / (SCI,1) of the flt/flc files but 
            corresponds to CCDCHIP = 2 and the ACS.WFC3.CHIP2 config files.

            and 
            
            EXTVER = 2 is extension 4 / (SCI,2) of the flt/flc files but 
            corresponds to CCDCHIP = 1 and the ACS.WFC3.CHIP1 config files.
    
    Returns
    -------
    conf_file : str
        String path of the configuration file.
            
    """   
    if instrume == 'ACS':
        conf_file = os.path.join(GRIZLI_PATH, 
                    'CONF/ACS.WFC.CHIP{0:d}.Cycle13.5.conf'.format(chip))
                           
    if instrume == 'WFC3':
        if grism == 'G280':
            conf_file = os.path.join(GRIZLI_PATH, 'CONF/G280/',
         'WFC3.UVIS.G280.cal/WFC3.UVIS.G280.CHIP{0:d}.V2.0.conf'.format(chip))
        
            return conf_file
            
        conf_file = os.path.join(GRIZLI_PATH, 
                                 'CONF/{0}.{1}.V4.32.conf'.format(grism, filter))
        
        ## When direct + grism combination not found for WFC3 assume F140W
        if not os.path.exists(conf_file):
            conf_file = os.path.join(GRIZLI_PATH,
                                 'CONF/{0}.{1}.V4.32.conf'.format(grism, 'F140W'))
              
    if instrume == 'NIRISS':
        conf_file = os.path.join(GRIZLI_PATH,
                                 'CONF/{0}.{1}.conf'.format(grism, filter))
        if not os.path.exists(conf_file):
            print('CONF/{0}.{1}.conf'.format(grism, filter))
            conf_file = os.path.join(GRIZLI_PATH,
                                 'CONF/NIRISS.{0}.conf'.format(filter))
        
    # if instrume == 'NIRCam':
    #     conf_file = os.path.join(GRIZLI_PATH,
    #         'CONF/aXeSIM_NC_2016May/CONF/NIRCam_LWAR_{0}.conf'.format(grism))
    if instrume == 'NIRCAM':
        conf_file = os.path.join(GRIZLI_PATH,
                                 'CONF/NIRCam.A.{0}.{1}.conf'.format(filter, grism))
        
    if instrume == 'WFIRST':
        conf_file = os.path.join(GRIZLI_PATH, 'CONF/WFIRST.conf')

    if instrume == 'SYN':
        conf_file = os.path.join(GRIZLI_PATH, 'CONF/syn.conf')
    
    # Euclid NISP, config files @ 
    # http://www.astrodeep.eu/euclid-spectroscopic-simulations/
    
    if instrume == 'NISP':
        if grism == 'BLUE':
            conf_file = os.path.join(GRIZLI_PATH, 'CONF/Euclid.Gblue.0.conf')
        else:
            conf_file = os.path.join(GRIZLI_PATH, 'CONF/Euclid.Gred.0.conf')
            
    return conf_file
        
def load_grism_config(conf_file):
    """Load parameters from an aXe configuration file
    
    Parameters
    ----------
    conf_file : str
        Filename of the configuration file
    
    Returns
    -------
    conf : `~grizli.grismconf.aXeConf`
        Configuration file object.  Runs `conf.get_beams()` to read the 
        sensitivity curves.
    """
    conf = aXeConf(conf_file)
    conf.get_beams()
    return conf
