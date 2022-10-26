import unittest

import os
import glob

import numpy as np

from grizli import utils
from grizli.pipeline import auto_script

class UtilsTester(unittest.TestCase):
    
    def test_parse_wfc3_visits(self):
        """
        """
        
        # These are copies of the WFC3-ERSII demo FLT files but where 
        # the data extensions have been set to dtype=uint8 so that they 
        # gzip up small
        
        path = os.path.dirname(utils.__file__)        
        if not os.path.exists('wfc3-parse-test'):
            os.system(f'tar xzvf {path}/tests/data/wfc3-parse-test.tar.gz')
            
        os.chdir('wfc3-parse-test/Prep/')
        
        files = glob.glob('../RAW/*flt.fits')
        files.sort()
        info = utils.get_flt_info(files)
        
        assert(len(info) == len(files))
        
        _visits, _filters = utils.parse_flt_files(info=info, 
                                      uniquename=True, get_footprint=True, 
                                      use_visit=True, max_dt=0.5, 
                                      visit_split_shift=2)
        
        assert(len(_visits) == 4)
        
        _groups = utils.parse_grism_associations(_visits, info)
        assert(len(_groups) == 2)
        
        ### From auto_script
        kwargs = auto_script.get_yml_parameters(local_file='my_params.yml', 
                                                copy_defaults=False)
        
        root = 'test'
        
        for bval in [True, False]:
            kwargs['parse_visits_args']['combine_same_pa'] = bval
            _ = auto_script.parse_visits(field_root=root,
                                         **kwargs['parse_visits_args'])

            visits, groups, info = _
        
            assert(hasattr(info, 'colnames'))
            assert(len(info) == 16)
        
            assert(len(visits) == 4)
        
        assert(os.path.exists(f'{root}_visits.yaml'))
        
        __ = auto_script.load_visits_yaml(f'{root}_visits.yaml')
        _visits, _groups, _info = __
        
        for v in [visits[0], _visits[0]]:
            for k in ['product','files','footprint']:
                assert(k in v)
        
            assert(v['product'].startswith('wfc3-ers'))
            assert(len(v['files']) == 4)
        
        # Grism groups
        for g in [groups, _groups]:
            assert(len(g) == 2)
            
            gi = g[0]
            for k in ['grism', 'direct']:
                assert(k in gi)
            
            gr = gi['grism'] 
            assert(gr['product'] == 'wfc3-ersii-g01-b6o-21-119.0-g102')
            assert(len(gr['files']) == 4)
            
        os.chdir('../../')
        if os.path.exists('wfc3-parse-test'):
            os.system('rm -rf wfc3-parse-test')
            