"""
Reprocessing scripts for variable WFC3/IR backgrounds
"""

def reprocess_wfc3ir(parallel=False):

    import matplotlib as mpl
    mpl.rcParams['backend'] = 'agg'

    import glob
    import os

    # https://github.com/gbrammer/wfc3
    try:
        from mywfc3 import reprocess_wfc3
    except:
        try:
            from reprocess_wfc3 import reprocess_wfc3
        except:
            print("""
    Couldn\'t `import mywfc3.reprocess_wfc3`.  
    Get it from https://github.com/gbrammer/wfc3 """)
            return False
        
    # Make ramp diagnostic images    
    if parallel:
        files=glob.glob('*raw.fits')
        reprocess_wfc3.show_ramps_parallel(files, cpu_count=4)
    
        # Reprocess all raw files
        files=glob.glob('*raw.fits')
        reprocess_wfc3.reprocess_parallel(files, cpu_count=4)
    else:
        files=glob.glob('*raw.fits')
        for file in files:
            if not os.path.exists(file.replace('raw.fits','ramp.png')):
                reprocess_wfc3.show_MultiAccum_reads(raw=file, stats_region=[[300,700], [300,700]])
        
        
        for file in files:
            if not os.path.exists(file.replace('raw.fits','flt.fits')):
                reprocess_wfc3.make_IMA_FLT(raw=file, stats_region=[[300,700], [300,700]])
        
def inspect(root='grizli'):
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
              to run this script in a fresh python session *before* importing
              `~matplotlib`.

    """
    import os
    import glob
    
    import matplotlib
    matplotlib.use("TkAgg") ### This needs to be run first!
    
    import mywfc3.inspect
    import mywfc3.reprocess_wfc3
    
    import astropy.io.fits as pyfits
    import numpy as np
    
    files = glob.glob('*ramp.png')
    files.sort()

    # Run the GUI, 'q' to quit
    x = mywfc3.inspect.ImageClassifier(images=files, logfile='{0}_inspect'.format(root))

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
        sn_pop = mywfc3.inspect.check_background_SN(ramp_file=ramp_file, show=False)
        pop_reads = np.cast[int](np.unique(np.hstack((pop_reads, sn_pop))))
        pop_reads = list(pop_reads)
        
        flt = raw.replace('_raw','_flt')
        if os.path.exists(flt):
            flt_im = pyfits.open(flt)
            if 'NPOP' in flt_im[0].header:
                if flt_im[0].header['NPOP'] > 0:
                    print('Skip %s' %(flt))
                    continue
        
        print('Process %s %s' %(raw, pop_reads))
        
        mywfc3.reprocess_wfc3.make_IMA_FLT(raw=raw, pop_reads=pop_reads, flatten_ramp=True)
    
    return True