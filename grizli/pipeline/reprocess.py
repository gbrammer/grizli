"""
Reprocessing scripts for variable WFC3/IR backgrounds
"""
from .. import utils

def reprocess_wfc3ir(parallel=False, cpu_count=0):
    """
    Run reprocessing script to flatten IR backgrounds
    
    """
    if parallel:
        # Might require light backend to avoid segfaults in multiprocessing
        import matplotlib as mpl
        mpl.rcParams['backend'] = 'agg'

    import glob
    import os
        
    # https://github.com/gbrammer/wfc3
    try:
        from reprocess_wfc3 import reprocess_wfc3
    except:
        try:
            from mywfc3 import reprocess_wfc3
        except:
            print("""
    Couldn\'t `import reprocess_wfc3`.  
    Get it from https://github.com/gbrammer/reprocess_wfc3 """)
            return False
    
    # Fetch calibs in serial
    print('\ngrizli.pipeline.reprocess: Fetch calibrations...\n')
    files=glob.glob('*raw.fits')
    for file in files:
        reprocess_wfc3.fetch_calibs(file, ftpdir='https://hst-crds.stsci.edu/unchecked_get/references/hst/', verbose=False)    
        
    # Make ramp diagnostic images    
    if parallel:
        files=glob.glob('*raw.fits')
        reprocess_wfc3.show_ramps_parallel(files, cpu_count=cpu_count)
    
        # Reprocess all raw files
        files=glob.glob('*raw.fits')
        reprocess_wfc3.reprocess_parallel(files, cpu_count=cpu_count)
    else:
        files=glob.glob('*raw.fits')
        for file in files:
            if not os.path.exists(file.replace('raw.fits','ramp.png')):
                reprocess_wfc3.show_MultiAccum_reads(raw=file, stats_region=[[300,700], [300,700]])
        
        
        for file in files:
            if not os.path.exists(file.replace('raw.fits','flt.fits')):
                reprocess_wfc3.make_IMA_FLT(raw=file, stats_region=[[300,700], [300,700]])
        
def inspect(root='grizli', force=False):
    """
    Run the GUI inspection tool on the `ramp.png` images to flag problematic
    reads with high backgrounds and/or satellite trails.
    
    Click the right mouse button to flag a given read and go to the next 
    object with the 'n' key.  
    
    Type 'q' when done.
    
    Parameters
    ----------
    root : str
        Rootname for the output inspection file:
        
            >>> root = 'grizli'
            >>> file = '{0}_inspect.fits'.format(root)
    
    Returns
    -------
    Nothing returned.  Makes the inspection file and runs the reprocessing.
    
    
    .. note:: If the script fails puking lots of Tk-related messages, be sure
              to run this script iin a fresh python session *before* importing
              `~matplotlib`.

    """
    import os
    import glob
    
    import matplotlib
    matplotlib.use("TkAgg") ### This needs to be run first!
    
    #import mywfc3.reprocess_wfc3
    from reprocess_wfc3 import reprocess_wfc3
    
    import astropy.io.fits as pyfits
    import numpy as np
    
    files = glob.glob('*ramp.png')
    files.sort()

    # Run the GUI, 'q' to quit
    try:
        import mywfc3.inspect
        if os.path.exists('{0}_inspect.fits'.format(root)):
            if force:
                x = mywfc3.inspect.ImageClassifier(images=files,
                                           logfile='{0}_inspect'.format(root))
        else:
            x = mywfc3.inspect.ImageClassifier(images=files,
                                       logfile='{0}_inspect'.format(root))                
    except:
        pass
    
    if not os.path.exists('{0}_inspect.fits'.format(root)):
        return True
        
    #im = pyfits.open('inspect_raw.info.fits')
    im = pyfits.open('{0}_inspect.fits'.format(root))
    tab = im[1].data

    fl = im['FLAGGED'].data
    is_flagged = fl.sum(axis=1) > 0

    sat_files = [file.replace('ramp.png', 'flt.fits') for file in tab['images'][is_flagged]]
    
    read_idx = np.arange(14, dtype=int)+1
    idx = np.arange(fl.shape[0])
    
    for i in idx[is_flagged]:
        
        pop_reads = list(read_idx[fl[i,:] > 0])
                
        raw = tab['images'][i].replace('_ramp.png', '_raw.fits')
        
        ramp_file = tab['images'][i].replace('_ramp.png', '_ramp.dat')
        #sn_pop = mywfc3.inspect.check_background_SN(ramp_file=ramp_file, show=False)
        #pop_reads = np.cast[int](np.unique(np.hstack((pop_reads, sn_pop))))
        #pop_reads = list(pop_reads)
        
        flt = raw.replace('_raw','_flt')
        if os.path.exists(flt):
            flt_im = pyfits.open(flt)
            if 'NPOP' in flt_im[0].header:
                if flt_im[0].header['NPOP'] > 0:
                    print('Skip %s' %(flt))
                    continue
        
        print('Process %s %s' %(raw, pop_reads))
        
        reprocess_wfc3.make_IMA_FLT(raw=raw, pop_reads=pop_reads, flatten_ramp=True)
    
    # Remove "killed"
    kill_files = [file.replace('ramp.png', 'flt.fits') for file in tab['images'][tab['kill'] > 0]]
    for file in kill_files:
        if os.path.exists(file):
            os.remove(file)
        
    
    return True

def make_masks(files=None, inspect='grizli_inspect.fits', ext=1):
    """
    Make satellite trail masks
    """
    import os
    import astropy.io.fits as pyfits
    
    try:
        from .. import utils
        from ..ds9 import DS9
    except:
        from grizli import utils
        from grizli.ds9 import DS9
        
    if files is None:
        insp = utils.read_catalog(inspect)
        flag = (insp['satellite'] > 0) | (insp['earth'] > 0)
        files=[f.replace('_ramp.png','_flt.fits') for f in insp['images'][flag]]
    
    ds9 = DS9()
    for file in files:
        im = pyfits.open(file)
        med = np.median(im['SCI',ext].data)
        ds9.view(im['SCI',ext].data-med)
        reg_file = file.replace('_flt.fits','.*.mask.reg').replace('_flc.fits','.*.mask.reg').replace('_c0m.fits','.*.mask.reg').replace('*', '{0:02d}'.format(ext))
        if os.path.exists(reg_file):
            ds9.set('regions file '+reg_file)
        
        x = input(file+': draw region (x to skip, q to abort): ')
        if x in ['q']:
            print('Abort.')
            #continue
            return False
        elif x in ['x']:
            print('Skip {0}.'.format(file))
            continue
        
        ds9.set('regions save '+reg_file)
