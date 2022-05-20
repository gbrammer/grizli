"""
Tests for JWST imaging and spectra, including 
"""
import os
import glob
import unittest

from .. import utils, multifit, GRIZLI_PATH
from ..pipeline import auto_script


class JWSTFittingTools(unittest.TestCase):
    
    def test_config(self):
        """
        Fetch config files if CONF not found
        """
        conf_path = os.path.join(GRIZLI_PATH, 'CONF')
        if not os.path.exists(conf_path): # if CONF dir doesn't already exist, create it
            os.mkdir(conf_path)
        
        if not os.path.exists(os.path.join(conf_path,
                              'GR150C.F115W.conf')):
            print(f'Download config and calib files to {conf_path}')
            #utils.fetch_default_calibs(ACS=False)
            utils.fetch_config_files(get_epsf=True, get_jwst=True)
            files = glob.glob(f'{conf_path}/*')
            print('Files: ', '\n'.join(files))

        assert(os.path.exists(os.path.join(conf_path,
                              'GR150C.F115W.conf')))
        return True
    
    def test_multibeam(self):
        """
        Can we initialize a multibeam file?
        """
        path = os.path.dirname(utils.__file__)
        print(path)
        beams_file = path + '/tests/data/niriss_jw01324001001_test.beams.fits'
        mb = multifit.MultiBeam(beams_file, group_name='jw01324001001',
                                MW_EBV=-1, fcontam=0.1, sys_err=0.03)
    
        assert(mb.N == 2)
        assert('F115W' in mb.PA)
        
        _ = mb.compute_model()
        
        spec = mb.oned_spectrum()
    
    #def test_parse_visits(self):
    #    """
    #    """
    #    
    #    # These are copies of the WFC3-ERSII demo FLT files but where 
    #    # the data extensions have been set to dtype=uint8 so that they 
    #    # gzip up small
    #    
    #    path = os.path.dirname(utils.__file__)        
    #    if not os.path.exists('niriss-parse-test'):
    #        os.system(f'tar xzvf {path}/tests/data/niriss-parse-test.tar.gz')
    #        
    #    os.chdir('jwst-parse-test/Prep/')
    #    
    #    files = glob.glob('../RAW/*rate.fits')
    #    files.sort()
    #    info = utils.get_flt_info(files)
    #    
    #    assert(len(info) == len(files))
    #    
    #    _visits, _filters = utils.parse_flt_files(info=info, 
    #                                  uniquename=True, get_footprint=True, 
    #                                  use_visit=True, max_dt=0.5, 
    #                                  visit_split_shift=2)
    #    
    #    assert(len(_visits) == 4)
    #    
    #    _groups = utils.parse_grism_associations(_visits)
    #    assert(len(_groups) == 2)
    #    
    #    ### From auto_script
    #    kwargs = auto_script.get_yml_parameters(local_file='my_params.yml', 
    #                                            copy_defaults=False)
    #    
    #    root = 'test'
    #    
    #    for bval in [True, False]:
    #        kwargs['parse_visits_args']['combine_same_pa'] = bval
    #        _ = auto_script.parse_visits(field_root=root,
    #                                     **kwargs['parse_visits_args'])
#
    #        visits, groups, info = _
    #    
    #        assert(hasattr(info, 'colnames'))
    #        assert(len(info) == 16)
    #    
    #        assert(len(visits) == 4)
    #    
    #    assert(os.path.exists(f'{root}_visits.yaml'))
    #    
    #    __ = auto_script.load_visits_yaml(f'{root}_visits.yaml')
    #    _visits, _groups, _info = __
    #    
    #    for v in [visits[0], _visits[0]]:
    #        for k in ['product','files','footprint']:
    #            assert(k in v)
    #    
    #        assert(v['product'].startswith('wfc3-ers'))
    #        assert(len(v['files']) == 4)
    #    
    #    # Grism groups
    #    for g in [groups, _groups]:
    #        assert(len(g) == 2)
    #        
    #        gi = g[0]
    #        for k in ['grism', 'direct']:
    #            assert(k in gi)
    #        
    #        gr = gi['grism'] 
    #        assert(gr['product'] == 'wfc3-ersii-g01-b6o-21-119.0-g102')
    #        assert(len(gr['files']) == 4)
    #        
    #    os.chdir('../../')
    #    if os.path.exists('wfc3-parse-test'):
    #        os.system('rm -rf wfc3-parse-test')
        
    