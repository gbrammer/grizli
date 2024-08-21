"""
Recalibrate JWST exposures with `snowblind` with the procedure as described there
"""
import os
import inspect
import warnings
import numpy as np

if os.path.exists('/GrizliImaging'):
    HOME = '/GrizliImaging'
else:
    HOME = os.getcwd()


def test():
    from grizli import prep
    
    file = 'jw01727039001_02101_00003_nrca2_rate.fits'
    file = 'jw02079004002_09201_00001_nrca2_rate.fits'
    file = 'jw01283004001_02201_00009_nrca1_rate.fits'
    
    do_recalibrate(rate_file='../RAW/' + file,
                   cores='all',
                   min_radius=3,
                   after_jumps=20,
                   context='jwst_1180.pmap',
                  )
    
    prep.fresh_flt_file(file, 
                        oneoverf_kwargs={'deg_pix':2048, 'dilate_iterations':3,
                                                               'thresholds':[5,4,3],
                                                               'other_axis':True})


def get_random_reprocess_file():
    """
    Get a table row of a file to be processed
    
    Returns
    -------
    res : None, table row
        Returns a single row from `reprocess_rates` with ``status == 0``, or None if 
        nothing found
    
    """
    from grizli.aws import db
    
    rows = db.SQL(f"""SELECT *
                           FROM reprocess_rates
                           WHERE status = 0 order by random() limit 1
                           """)
    
    if len(rows) == 0:
        return None
    else:
        return rows


def run_one(verbose=True, remove_result=True):
    """
    Run a single random file from `reprocess_rates` with ``status == 0``
    """
    import os
    import time
    import boto3
    
    from grizli.aws import db
    from grizli import utils
    
    row = get_random_reprocess_file()
    
    if row is None:
        with open(os.path.join(HOME, 'reprocess_finished.txt'),'w') as fp:
            fp.write(time.ctime() + '\n')
        
        return None
        
    print(f'============  Reprocess exposure  ==============')
    print(f"{row['rate_file'][0]}")
    print(f'========= {time.ctime()} ==========')
    
    with open(os.path.join(HOME, 'reprocess_history.txt'),'a') as fp:
        fp.write(f"{time.ctime()} {row['rate_file'][0]}\n")
    
    # Update db
    db.execute(f"""update reprocess_rates
               set status = 1, ctime={time.time()}
               where
               rate_file = '{row['rate_file'][0]}'
               AND prefix = '{row['prefix'][0]}'
               AND bucket = '{row['bucket'][0]}';
               """)
    
    #################
    # Run it
    #################
    try:
        status = do_recalibrate(**row[0])
    except:
        row['status'] = 5
        
        # Update db
        db.execute(f"""update reprocess_rates
                   set status = 3, ctime={time.time()}
                   where
                   rate_file = '{row['rate_file'][0]}'
                   AND prefix = '{row['prefix'][0]}'
                   AND bucket = '{row['bucket'][0]}';
                   """)
        
        return row
    
    ###############
    # Copy output
    ###############
    if status & os.path.exists(row['rate_file'][0]):
        
        s3 = boto3.resource('s3')
        bkt = s3.Bucket(row['bucket'][0])
        
        local_file = row['rate_file'][0]
        s3_prefix = os.path.join(row['prefix'][0], row['rate_file'][0])
        
        msg = f"Send {local_file} > s3://{row['bucket'][0]}/{s3_prefix}"
        utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
                    
        bkt.upload_file(local_file, s3_prefix, ExtraArgs={'ACL': row['acl'][0]},
                        Callback=None, Config=None)
                    
        row['status'] = 2
        row['ctime'] = time.time()
        
        # Update db
        db.execute(f"""update reprocess_rates
                   set status = 2, ctime={time.time()}
                   where
                   rate_file = '{row['rate_file'][0]}'
                   AND prefix = '{row['prefix'][0]}'
                   AND bucket = '{row['bucket'][0]}';
                   """)
        
        # Remove result
        if remove_result:
            msg = f"rm {row['rate_file'][0]}"
            utils.log_comment(utils.LOGFILE, msg, verbose=verbose)
            
            os.remove(row['rate_file'][0])
            
    else:
        ###############
        # Problem
        ###############
        row['status'] = 3
        
        # Update db
        db.execute(f"""update reprocess_rates
                   set status = 3, ctime={time.time()}
                   where
                   rate_file = '{row['rate_file'][0]}'
                   AND prefix = '{row['prefix'][0]}'
                   AND bucket = '{row['bucket'][0]}';
                   """)
    return row


def do_recalibrate(rate_file='jw06541001001_03101_00001_nrs1_rate.fits', cores='half', min_radius=3, after_jumps=20, context=None, clean=True, skip_snowblind_after='1.14', **kwargs):
    """
    Recalibrate JWST uncal exposures with `snowblind` snowball masking
    
    Parameters
    ----------
    rate_file : str
        Filename of the output countrate exposure, must end in ``rate.fits``.
        The uncalibrated file will be fetched from MAST with 
        ``rate_file.replace('_rate.fits', '_uncal.fits')``
    
    cores : str
        Number of cores to use for the JWST pipeline steps, e.g., ``1``, ``quarter``,
        ``half``, ``all``
    
    min_radius, after_jumps : int, int
        `snowblind` parameters
    
    context : str
        Optional explicit CRDS context to use
    
    clean : bool
        Remove intermediate files
    
    Returns
    -------
    status : bool
        True if completed successfully.  False if ``rate_file`` doesn't have the 
        expected ``rate.fits`` suffix.  The ``rate_file`` product will be left in the 
        working directory.
    """
    from packaging.version import Version
    from snowblind import SnowblindStep
    import jwst
    from jwst.pipeline import Detector1Pipeline
    from jwst.step import RampFitStep
    from jwst.step import GainScaleStep
    
    import mastquery.utils

    try:
        from .. import jwst_utils
        from .. import utils
    except ImportError:
        from grizli import jwst_utils
        from grizli import utils
    
    frame = inspect.currentframe()
    _LOGFILE = utils.LOGFILE
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    utils.LOGFILE = rate_file.split('.fits')[0] + '.log'
    utils.log_function_arguments(utils.LOGFILE, frame, 'do_recalibrate',
                                 ignore=['_LOGFILE'])
    
    if not rate_file.endswith('_rate.fits'):
        print(f'{rate_file} does not end with _rate.fits')
        return False
        
    if context is not None:
        jwst_utils.CRDS_CONTEXT = context
        jwst_utils.set_crds_context(verbose=False, override_environ=True)
    
    output_path, base_file = os.path.split(rate_file)
    mastquery.utils.download_from_mast([base_file.replace('_rate.fits','_uncal.fits')])
    
    # try:
    #     ncores = int(cores)
    # except ValueError:
    #     ncores = cores
    ncores = cores
    
    steps = {
        "jump": {
            "save_results": True,
            "maximum_cores": ncores
        },
        "ramp_fit": {
            "skip": True,
        },
        "gain_scale": {
            "skip": True,
        },
    }

    Detector1Pipeline.call(base_file.replace('_rate', '_uncal'), steps=steps)

    if Version(jwst.__version__) < Version(skip_snowblind_after):
        SnowblindStep.call(base_file.replace('_rate','_jump'),
                       min_radius=int(min_radius),
                       after_jumps=int(after_jumps),
                       save_results=True,
                       suffix="snowblind")
        next_exten = '_snowblind'
    else:
        msg = f'jwst {jwst.__version__} > {skip_snowblind_after}, skip snowblind'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        
        next_exten = '_jump'
        
    rate, rateints = RampFitStep.call(base_file.replace('_rate', next_exten),
                                      maximum_cores=ncores)

    rate = GainScaleStep.call(rate)
    
    # Mask out nans and remove DQ=4
    bad = ~np.isfinite(rate.data)
    rate.dq[bad] |= 1
    rate.data[bad] = 0
    rate.dq -= (rate.dq & 4)

    rate.save(rate_file)
    
    if clean:
        for ext in ['_jump', '_snowblind', '_uncal', '_trapsfilled']:
            if os.path.exists(base_file.replace('_rate', ext)):
                print(f"Remove {base_file.replace('_rate', ext)}")
                os.remove(base_file.replace('_rate', ext))
            
    utils.LOGFILE = _LOGFILE
    
    return True
