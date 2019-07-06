import os
import numpy as np
from .. import prep, utils

# Filters
IR_M_FILTERS = ['F098M', 'F127M', 'F139M', 'F153M']
IR_W_FILTERS = ['F105W', 'F110W', 'F125W', 'F140W', 'F160W']
IR_GRISMS = ['G102', 'G141']

OPT_M_FILTERS = ['F410M', 'F467M', 'F547M', 'F550M', 'F621M', 'F689M', 'F763M', 'F845M']
OPT_W_FILTERS = ['F200LP','F350LP', 'F435W', 'F438W', 'F439W', 'F450W', 'F475W', 'F475X', 'F555W', 'F569W', 'F600LP', 'F606W', 'F622W', 'F625W', 'F675W', 'F702W', 'F775W', 'F791W', 'F814W', 'F850LP']
OPT_GRISMS = ['G800L']
UV_GRISMS = ['G280']

VALID_FILTERS = OPT_M_FILTERS + OPT_W_FILTERS + OPT_GRISMS #+ UV_GRISMS 
VALID_FILTERS += IR_M_FILTERS + IR_W_FILTERS + IR_GRISMS

GRIS_REF_FILTERS = {'G141': ['F140W', 'F160W', 'F125W', 'F105W', 'F110W', 
                             'F098M', 'F127M', 'F139M', 'F153M', 'F132N', 
                             'F130N', 'F128N', 'F126N', 'F164N', 'F167N'],
                    'G102': ['F105W', 'F098M', 'F110W', 'F125W', 'F140W', 
                             'F160W', 'F127M', 'F139M', 'F153M', 'F132N', 
                             'F130N', 'F128N', 'F126N', 'F164N', 'F167N'],
                    'G800L': ['F814W', 'F850LP', 'F606W', 'F435W', 'F775W',
                              'F555W', 'opt']}

def test_aws_availability():
    """
    Test if aws s3 is available
    """
    s3_status = os.system('aws s3 ls s3://stpubdata --request-payer requester > /tmp/aws.x')
    if s3_status == 0:
        s3_sync='cp'  # As of late October 2018, 's3 sync' not working with 'stpubdata'
    else:
        s3_sync=False # Fetch from ESA archive
    
    return s3_sync

def write_params_to_yml(kwargs, output_file='grizli.auto_script.yml', verbose=True):
    """
    Write grizli parameters to a file
    """
    import time
    import yaml
    import grizli
    
    # Make copies of some parameters that can't be dumped with yaml
    try:
        phot_apertures = kwargs['multiband_catalog_args']['phot_apertures']
        kwargs['multiband_catalog_args']['phot_apertures'] = None 
    except:
        phot_apertures = None
    
    try:    
        filter_kernel = kwargs['multiband_catalog_args']['detection_params']['filter_kernel'] 
        kwargs['multiband_catalog_args']['detection_params']['filter_kernel'] = None
    except:
        filter_kernel = None
     
    # Write the file   
    fp = open(output_file,'w')  
    fp.write('# {0}\n'.format(time.ctime()))
    fp.write('# Grizli version = {0}\n'.format(grizli.__version__))
      
    for k in kwargs: 
        
        try: 
            # Write copies of dicts
            d = {k:kwargs[k].copy()} 
        except: 
            # Write single variables
            d = {k:kwargs[k]} 
        
        # flow_style = False to get the correct dict/variable formatting
        yaml.dump(d, stream=fp, default_flow_style=False) 
    
    fp.close()
    
    # Revert the things we changed above
    if phot_apertures is not None:
        kwargs['multiband_catalog_args']['phot_apertures'] = phot_apertures 
    
    if filter_kernel is not None:
        kwargs['multiband_catalog_args']['detection_params']['filter_kernel'] = filter_kernel
    
    if verbose:
        print('\n# Write parameters to {0}\n'.format(output_file))
        
def get_yml_parameters(local_file=None, copy_defaults=False, verbose=True, skip_unknown_parameters=True):
    """
    Read default parameters from the YAML file in `grizli/data`
    
    Returns:
    
    kwargs : dict
        Parameter dictionary (with nested sub dictionaries).
        
    """
    import yaml
    import shutil
    
    default_yml = 'auto_script_defaults.yml'
    
    path = os.path.join(os.path.dirname(__file__), '..', 'data', default_yml)
    
    if copy_defaults:
        if local_file is None:
            local_file = default_yml
            
        shutil.copy(path, local_file)
        if verbose:
            print('Copied default parameter file to {0}'.format(local_file))
            
        return False
        
    fp = open(path)
    kwargs = yaml.load(fp)
    fp.close()
    
    if local_file is not None:
        fp = open(local_file)
        local_args = yaml.load(fp)
        fp.close()
        
        for k in local_args:
            if (k not in kwargs) and skip_unknown_parameters:
                print('Skipping user keyword {0}'.format(k))
                continue
            
            # Dictionaries
            if isinstance(local_args[k], dict):
                for subk in local_args[k]:
                    if (subk not in kwargs[k]) and skip_unknown_parameters:
                        print('Skipping user keyword {0}.{1}'.format(k, subk))
                        continue
                    
                    kwargs[k][subk] = local_args[k][subk]
            else:
                kwargs[k] = local_args[k]
    
    if kwargs['fetch_files_args']['s3_sync']:
        kwargs['fetch_files_args']['s3_sync'] = test_aws_availability()
        
    # catalog defaults
    dd = kwargs['multiband_catalog_args']
    if dd['detection_params'] is None:
        dd['detection_params'] =  prep.SEP_DETECT_PARAMS 
    
    if dd['phot_apertures'] is None:
        dd['phot_apertures'] = prep.SEXTRACTOR_PHOT_APERTURES_ARCSEC
            
    return kwargs

def compare_args():
    from grizli.pipeline import default_params
    y = default_params.get_yml_parameters(copy_defaults=False)   
    
    d = default_params.get_args_dict()
    
    for k in y:
        if k not in d:
            print(k)
        else:
            if y[k] != d[k]:
                print('!!', k)
                print(y[k])
                print(d[k])
    pass
    
    