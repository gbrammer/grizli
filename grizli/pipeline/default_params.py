"""
Parameter handling with YAML
"""

import os
import traceback
import yaml

import numpy as np
from .. import prep, utils

# ACS/WFC & WFC3 UVIS + IR filters
IR_N_FILTERS = ['F126N', 'F128N', 'F130N', 'F132N', 'F164N', 'F167N']
IR_M_FILTERS = ['F098M', 'F127M', 'F139M', 'F153M']
IR_W_FILTERS = ['F105W', 'F110W', 'F125W', 'F140W', 'F160W']

NIRISS_FILTERS = ['F090W', 'F115W', 'F150W', 'F200W', 
                    'F277W', 'F356W', 'F444W',
                    'F140M', 'F158M', 'F380M', 'F430M', 'F480M']
NIRCAM_LW_FILTERS = ['F322W2', 'F277W', 'F356W', 'F444W', 
                    'F250M', 'F300M', 'F335M', 'F360M', 'F410M', 'F430M', 'F460M', 'F480M',
                    'F323N', 'F405N', 'F466N', 'F470N']
NIRCAM_SW_FILTERS = ['F150W2', 'F070W', 'F090W', 'F115W', 'F150W', 'F200W', 
                        'F140M', 'F162M', 'F182M', 'F210M', 
                        'F164N', 'F187N', 'F212N']
MIRI_FILTERS = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 
                'F1500W', 'F1800W', 'F2100W', 'F2550W']
JW_FILTERS = np.unique(NIRISS_FILTERS + NIRCAM_LW_FILTERS + NIRCAM_SW_FILTERS + MIRI_FILTERS).tolist()

OPT_N_FILTERS = ['F469N', 'F487N', 'F502N']
OPT_N_FILTERS += ['FQ436N', 'FQ437N', 'FQ492N', 'FQ508N', 'FQ575N']
OPT_N_FILTERS += ['F631N', 'F645N', 'F656N', 'F657N', 'F658N', 
                    'F660N', 'F665N', 'F673N', 'F680N']
OPT_N_FILTERS += ['FQ619N', 'FQ634N', 'FQ672N', 'FQ674N', 'FQ727N', 'FQ750N', 
                    'FQ889N', 'FQ906N', 'FQ924N', 'FQ937N', 'F892N', 'F953N']

OPT_M_FILTERS = ['F410M', 'F467M', 'F547M', 'F550M', 'F621M', 'F689M', 'F763M', 'F845M']
OPT_W_FILTERS = ['F200LP', 'F350LP', 'F435W', 'F438W', 'F439W', 'F450W', 'F475W', 'F475X', 
                    'F555W', 'F569W', 'F600LP', 'F606W', 'F622W', 'F625W', 'F675W', 'F702W', 
                    'F775W', 'F791W', 'F814W', 'F850LP']

UV_N_FILTERS = ['F280N', 'F343N', 'F373N', 'F395N']
UV_N_FILTERS += ['FQ232N', 'FQ243N', 'FQ378N']
UV_M_FILTERS = ['F390M']
UV_W_FILTERS = ['F275W', 'F336W', 'F390W']

IR_GRISMS = ['G102', 'G141']
OPT_GRISMS = ['G800L']
UV_GRISMS = ['G280']

JW_GRISMS = ['GR150C', 'GR150R', 'GRISMR', 'GRISMC']

VALID_FILTERS = OPT_M_FILTERS + OPT_W_FILTERS + OPT_GRISMS  # + UV_GRISMS
VALID_FILTERS += IR_M_FILTERS + IR_W_FILTERS + IR_GRISMS
VALID_FILTERS += JW_FILTERS + JW_GRISMS

ALL_IMAGING_FILTERS = UV_N_FILTERS + UV_M_FILTERS + UV_W_FILTERS
ALL_IMAGING_FILTERS += OPT_N_FILTERS + OPT_M_FILTERS + OPT_W_FILTERS
ALL_IMAGING_FILTERS += IR_N_FILTERS + IR_M_FILTERS + IR_W_FILTERS

GRIS_REF_FILTERS = {'G141': ['F140W', 'F160W', 'F125W', 'F105W', 'F110W',
                             'F098M', 'F127M', 'F139M', 'F153M', 'F132N',
                             'F130N', 'F128N', 'F126N', 'F164N', 'F167N'],
                    'G102': ['F105W', 'F098M', 'F110W', 'F125W', 'F140W',
                             'F160W', 'F127M', 'F139M', 'F153M', 'F132N',
                             'F130N', 'F128N', 'F126N', 'F164N', 'F167N'],
                    'G800L': ['F814W', 'F850LP', 'F606W', 'F435W', 'F775W',
                              'F555W', 'opt'], 
                    'GR150C': ['F115W', 'F150W', 'F200W', 
                               'CLEAR-F115W', 'CLEAR-F150W', 'CLEAR-F200W'], 
                    'GR150R': ['F115W', 'F150W', 'F200W',
                               'CLEAR-F115W', 'CLEAR-F150W', 'CLEAR-F200W'],
                    'GRISMR': ['F277W-CLEAR','F356W-CLEAR','F410M-CLEAR',
                               'F444W-CLEAR','F277W','F356W','F410M','F444W'],
                    'GRISMC': ['F277W-CLEAR','F356W-CLEAR','F410M-CLEAR',
                               'F444W-CLEAR','F277W','F356W','F410M','F444W']
                    }


def test_aws_availability():
    """
    Test if aws s3 is available
    """
    s3_status = os.system('aws s3 ls s3://stpubdata --request-payer requester > /tmp/aws.x')
    if s3_status == 0:
        s3_sync = 'cp'  # As of late October 2018, 's3 sync' not working with 'stpubdata'
    else:
        s3_sync = False  # Fetch from ESA archive

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
    fp = open(output_file, 'w')
    fp.write('# {0}\n'.format(time.ctime()))
    fp.write('# Grizli version = {0}\n'.format(grizli.__version__))

    for k in kwargs:

        try:
            # Write copies of dicts
            d = {k: kwargs[k].copy()}
        except:
            # Write single variables
            d = {k: kwargs[k]}

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


def safe_yaml_loader(yamlfile, loaders=[yaml.FullLoader, yaml.Loader, None]):
    """
    Try different YAML loaders
    """
    args = None
    for loader in loaders:
        with open(yamlfile) as fp:
            try:
                args = yaml.load(fp, Loader=loader)
                break
                print(loader)
            except:
                pass

    if args is None:
        raise IOError(f'Failed to load {yamlfile} with {loaders}.')

    return args


def get_yml_parameters(local_file=None, copy_defaults=False, verbose=True, skip_unknown_parameters=True):
    """
    Read default parameters from the YAML file in `grizli/data`

    Returns:

    kwargs : dict
        Parameter dictionary (with nested sub dictionaries).

    """
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

    kwargs = safe_yaml_loader(path)

    if local_file is not None:

        local_args = safe_yaml_loader(local_file)

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
        dd['detection_params'] = prep.SEP_DETECT_PARAMS

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
