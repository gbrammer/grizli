"""
Demonstrate aXe trace polynomials.

  v1.0 - October 14, 2014  (G. Brammer, N. Pirzkal, R. Ryan) 

"""
import os
import numpy as np

class aXeConf():
    def __init__(self, conf_file='WFC3.IR.G141.V2.5.conf'):
        
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
        """
        Read an aXe config file, convert floats and arrays
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
        """
        Get the maximum polynomial order in DYDX or DLDP for each beam
        """
        self.orders = {}
        for beam in ['A','B','C','D','E','F','G','H','I','J']:
            order = 0
            while 'DYDX_%s_%d' %(beam, order) in self.conf.keys():
                order += 1
            
            while 'DLDP_%s_%d' %(beam, order) in self.conf.keys():
                order += 1
            
            self.orders[beam] = order-1

    def get_beams(self):
        """
        Get beam parameters and sensitivity curves
        """
        import os
        import collections
        from astropy.table import Table, Column
        
        self.dxlam = collections.OrderedDict()
        self.nx = collections.OrderedDict()
        self.sens = collections.OrderedDict()
        self.beams = []
        
        for beam in self.orders:
            if self.orders[beam] > 0:
                self.beams.append(beam)
                self.dxlam[beam] = np.arange(self.conf['BEAM%s' %(beam)].min(), self.conf['BEAM%s' %(beam)].max(), dtype=int)
                self.nx[beam] = int(self.dxlam[beam].max()-self.dxlam[beam].min())+1
                self.sens[beam] = Table.read('%s/%s' %(os.path.dirname(self.conf_file), self.conf['SENSITIVITY_%s' %(beam)]))
                #self.sens[beam].wave = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                #self.sens[beam].sens = np.cast[np.double](self.sens[beam]['SENSITIVITY'])
                
                ### Need doubles for interpolating functions
                for col in self.sens[beam].colnames:
                    data = np.cast[np.double](self.sens[beam][col])
                    self.sens[beam].remove_column(col)
                    self.sens[beam].add_column(Column(data=data, name=col))
                
                ### Scale BEAM F
                if (beam == 'F') & ('G141' in self.conf_file): # & ('gbb' in self.conf_file):
                    self.sens[beam]['SENSITIVITY'] *= 0.35
                    
                # wave = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                # sens = np.cast[np.double](self.sens[beam]['SENSITIVITY']
                # self.sens[beam]['WAVELENGTH'] = np.cast[np.double](self.sens[beam]['WAVELENGTH'])
                # self.sens[beam]['SENSITIVITY'] = )
    
                
    def field_dependent(self, xi, yi, coeffs):
        """
        aXe field-dependent coefficients
        """
        ## number of coefficients for a given polynomial order
        ## 1:1, 2:3, 3:6, 4:10, order:order*(order+1)/2
        if isinstance(coeffs, float):
            order = 1
        else:
            order = int(-1+np.sqrt(1+8*len(coeffs)))/2
    
        ## Build polynomial terms array
        ## $a = a_0+a_1x_i+a_2y_i+a_3x_i^2+a_4x_iy_i+a_5yi^2+$ ...
        xy = []
        for p in range(order):
            for px in range(p+1):
                xy.append(xi**(p-px)*yi**(px))
    
        ## Evaluate the polynomial, allowing for N-dimensional inputs
        a = np.sum((np.array(xy).T*coeffs).T, axis=0)
    
        return a
    
    def get_beam_trace(self, x=507, y=507, dx=0., beam='A'):
        """
        Get an aXe beam trace for an input reference pixel and 
        list of output x pixels dx
        """
        NORDER = self.orders[beam]+1
        
        xi, yi = x-self.xoff, y-self.yoff
        xoff_beam = self.field_dependent(xi, yi, self.conf['XOFF_%s' %(beam)])
        yoff_beam = self.field_dependent(xi, yi, self.conf['YOFF_%s' %(beam)])
    
        ## y offset of trace (DYDX)
        dydx = np.zeros(NORDER) #0 #+1.e-80
        for i in range(NORDER):
            if 'DYDX_%s_%d' %(beam, i) in self.conf.keys():
                coeffs = self.conf['DYDX_%s_%d' %(beam, i)]
                dydx[i] = self.field_dependent(xi, yi, coeffs)
            
        # $dy = dydx_0+dydx_1 dx+dydx_2 dx^2+$ ...
        dy = yoff_beam
        for i in range(NORDER):
            dy += dydx[i]*(dx-xoff_beam)**i
        
        ## wavelength solution    
        dldp = np.zeros(NORDER)
        for i in range(NORDER):
            if 'DLDP_%s_%d' %(beam, i) in self.conf.keys():
                coeffs = self.conf['DLDP_%s_%d' %(beam, i)]
                dldp[i] = self.field_dependent(xi, yi, coeffs)
        
        ## dp is the arc length along the trace
        ## $\lambda = dldp_0 + dldp_1 dp + dldp_2 dp^2$ ...
        if self.conf['DYDX_ORDER_%s' %(beam)] == 0:   ## dy=0
            dp = dx-xoff_beam                      
        elif self.conf['DYDX_ORDER_%s' %(beam)] == 1: ## constant dy/dx
            dp = np.sqrt(1+dydx[1]**2)*(dx-xoff_beam)
        elif self.conf['DYDX_ORDER_%s' %(beam)] == 2: ## quadratic trace
            u0 = dydx[1]+2*dydx[2]*(0)
            dp0 = (u0*np.sqrt(1+u0**2)+np.arcsinh(u0))/(4*dydx[2])
            u = dydx[1]+2*dydx[2]*(dx-xoff_beam)
            dp = (u*np.sqrt(1+u**2)+np.arcsinh(u))/(4*dydx[2])-dp0
        else:
            ## high order shape, numerical integration along trace
            ## (this can be slow)
            xmin = np.minimum((dx-xoff_beam).min(), 0)
            xmax = np.maximum((dx-xoff_beam).max(), 0)
            xfull = np.arange(xmin, xmax)
            dyfull = 0
            for i in range(1, NORDER):
                dyfull += i*dydx[i]*(xfull-0.5)**(i-1)
            
            ## Integrate from 0 to dx / -dx
            dpfull = xfull*0.
            lt0 = xfull <= 0
            if lt0.sum() > 1:
                dpfull[lt0] = np.cumsum(np.sqrt(1+dyfull[lt0][::-1]**2))[::-1]
                dpfull[lt0] *= -1
            #
            gt0 = xfull >= 0
            if gt0.sum() > 0:
                dpfull[gt0] = np.cumsum(np.sqrt(1+dyfull[gt0]**2))
              
            dp = np.interp(dx-xoff_beam, xfull, dpfull)
        
        ## Evaluate dldp    
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
            if 'XOFF_%s' %(beam) not in self.conf.keys():
                continue
            
            xoff = self.field_dependent(x0, x1, self.conf['XOFF_%s' %(beam)])
            dy, lam = self.get_beam_trace(x0, x1, dx=dx, beam=beam)
            xlim = self.conf['BEAM%s' %(beam)]
            ok = (dx >= xlim[0]) & (dx <= xlim[1])
            plt.scatter(dx[ok]+xoff, dy[ok], c=lam[ok]/1.e4, marker='s', s=s,
                        alpha=0.5, edgecolor='None')
            plt.text(np.median(dx[ok]), np.median(dy[ok])+1, beam,
                     ha='center', va='center', fontsize=14)
            print 'Beam %s, lambda=(%.1f - %.1f)' %(beam, lam[ok].min(),
                                                    lam[ok].max())
            
        plt.grid()
        plt.xlabel(r'$\Delta x$')
        plt.ylabel(r'$\Delta y$')

        cb = plt.colorbar(pad=0.01, fraction=0.05)    
        cb.set_label(r'$\lambda\,(\mu\mathrm{m})$')
        plt.title(self.conf_file)
        plt.tight_layout()
        plt.savefig('%s.pdf' %(self.conf_file))    

def get_config_filename(instrume='WFC3', filter='F140W',
                        grism='G141'):
    """
    Generate a config filename based on the instrument, filter & grism
    combination. 
    
    Config files assumed to be found in $GRIZLI environment variable
    """   
    if instrume == 'WFC3':
        conf_file = os.path.join(os.getenv('GRIZLI'), 
                                 'CONF/%s.%s.V4.3.conf' %(grism, filter))
        
        ## When direct + grism combination not found for WFC3 assume F140W
        if not os.path.exists(conf_file):
            conf_file = os.path.join(os.getenv('GRIZLI'),
                                 'CONF/%s.%s.V4.3.conf' %(grism, 'F140W'))
              
    if instrume == 'NIRISS':
        conf_file = os.path.join(os.getenv('GRIZLI'),
                                 'CONF/NIRISS.%s.conf' %(grism))
    
    if instrume == 'NIRCam':
        conf_file = os.path.join(os.getenv('GRIZLI'),
            'CONF/aXeSIM_NC_2016May/CONF/NIRCam_LWAR_%s.conf' %(grism))
    
    if instrume == 'WFIRST':
        conf_file = os.path.join(os.getenv('GRIZLI'), 'CONF/WFIRST.conf')
    
    return conf_file
        
def load_grism_config(conf_file):
    """
    Load parameters from an aXe configuration file
    """
    conf = aXeConf(conf_file)
    conf.get_beams()
    return conf
